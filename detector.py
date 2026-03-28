"""
detector.py — Core ML inference engine for TruthLens
Uses pre-trained models from Hugging Face — no training needed, no GPU required.

Models used:
  Text  : facebook/bart-large-mnli  (Zero-Shot Classification)
  Image : Extracted text via pytesseract → same text pipeline
          + CLIP-based image credibility heuristics
"""

import re
import math
import pytesseract
from PIL import Image
import numpy as np

# ── Lazy model loader (cached after first load) ──────────────────────────────
_text_classifier = None
_clip_model = None
_clip_processor = None

def _get_text_classifier():
    global _text_classifier
    if _text_classifier is None:
        from transformers import pipeline
        _text_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU always — no GPU needed
        )
    return _text_classifier


def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


# ── Linguistic red-flag analysis ─────────────────────────────────────────────
CLICKBAIT_PATTERNS = [
    r"\b(shocking|unbelievable|you won\'t believe|secret|exposed|they don\'t want you to know)\b",
    r"\b(miracle|cure|100%|guaranteed|proven|scientifically proven)\b",
    r"(!{2,}|\?{2,})",          # Multiple ! or ?
    r"\b(BREAKING|URGENT|ALERT)\b",
    r"\b(share before|deleted|censored|banned)\b",
    r"[A-Z]{4,}",               # Excessive caps words
]

CREDIBILITY_SIGNALS = [
    r"\b(according to|study by|research shows|published in|cited by)\b",
    r"\b(university|institute|journal|report|official)\b",
    r"\b(percent|data|statistics|survey|sample size)\b",
]

def _linguistic_score(text: str) -> dict:
    """Returns a dict of red-flag and credibility signal counts."""
    text_lower = text.lower()
    red_flags = sum(
        len(re.findall(p, text_lower, re.IGNORECASE))
        for p in CLICKBAIT_PATTERNS
    )
    cred_signals = sum(
        len(re.findall(p, text_lower, re.IGNORECASE))
        for p in CREDIBILITY_SIGNALS
    )
    word_count = len(text.split())
    # Normalise
    red_flag_rate = min(red_flags / max(word_count / 20, 1), 1.0)
    cred_rate = min(cred_signals / max(word_count / 30, 1), 1.0)
    return {
        "red_flags": red_flags,
        "cred_signals": cred_signals,
        "red_flag_rate": red_flag_rate,
        "cred_rate": cred_rate,
    }


# ── Main text analysis ────────────────────────────────────────────────────────
def analyze_text(text: str) -> dict:
    """
    Classify text as Real / Fake using:
    1. Zero-shot NLI classification (BART-large-MNLI)
    2. Linguistic red-flag analysis
    Combined with weighted fusion.
    """
    text = text.strip()
    if len(text) < 20:
        raise ValueError("Text too short — please provide at least a sentence.")

    classifier = _get_text_classifier()

    # NLI zero-shot
    labels = ["real news", "fake news", "misinformation", "credible reporting"]
    result = classifier(text[:1024], candidate_labels=labels, multi_label=False)

    label_map = {
        "real news": "REAL",
        "credible reporting": "REAL",
        "fake news": "FAKE",
        "misinformation": "FAKE",
    }
    nli_scores = {"REAL": 0.0, "FAKE": 0.0}
    for lbl, score in zip(result["labels"], result["scores"]):
        nli_scores[label_map[lbl]] += score

    # Linguistic
    ling = _linguistic_score(text)
    # High red-flag rate → push fake; high cred rate → push real
    ling_fake_push = ling["red_flag_rate"] * 0.25
    ling_real_push = ling["cred_rate"] * 0.15

    raw_fake = nli_scores["FAKE"] + ling_fake_push - ling_real_push
    raw_real = nli_scores["REAL"] + ling_real_push - ling_fake_push

    # Softmax-normalise
    total = math.exp(raw_fake) + math.exp(raw_real)
    fake_conf = math.exp(raw_fake) / total
    real_conf = math.exp(raw_real) / total

    verdict = "FAKE" if fake_conf > real_conf else "REAL"

    # Build explanation bullets
    bullets = _build_text_bullets(text, ling, nli_scores)

    return {
        "verdict": verdict,
        "fake_confidence": round(fake_conf * 100, 1),
        "real_confidence": round(real_conf * 100, 1),
        "mode": "text",
        "bullets": bullets,
        "linguistic": ling,
        "source_text": text[:300] + ("..." if len(text) > 300 else ""),
    }


def _build_text_bullets(text, ling, nli_scores):
    bullets = []
    if ling["red_flags"] > 0:
        bullets.append(f"⚠️ Found {ling['red_flags']} sensationalist language pattern(s) — common in misinformation")
    if ling["cred_signals"] > 0:
        bullets.append(f"✅ Found {ling['cred_signals']} credibility signal(s) — citations, data references, or institutions mentioned")
    if nli_scores["FAKE"] > 0.55:
        bullets.append("🤖 NLI model strongly associates content with fake/misleading news patterns")
    elif nli_scores["REAL"] > 0.55:
        bullets.append("🤖 NLI model associates content with credible, factual reporting patterns")
    else:
        bullets.append("🤖 NLI model found mixed signals — content is borderline")
    wc = len(text.split())
    if wc < 30:
        bullets.append("📏 Very short content — harder to classify with high confidence")
    elif wc > 200:
        bullets.append("📏 Detailed article length — more features extracted for analysis")
    return bullets


# ── Image analysis ────────────────────────────────────────────────────────────
def analyze_image(pil_image: Image.Image) -> dict:
    """
    Analyse a news image by:
    1. OCR — extract any text in the image → run text pipeline
    2. CLIP — score image against real/fake news visual prompts
    3. Image metadata heuristics (compression artifacts, extreme contrast)
    """
    # Step 1: OCR
    ocr_text = ""
    try:
        ocr_text = pytesseract.image_to_string(pil_image).strip()
    except Exception:
        ocr_text = ""

    # Step 2: CLIP visual scoring
    clip_result = _clip_image_score(pil_image)

    # Step 3: Image quality heuristics
    img_heuristics = _image_heuristics(pil_image)

    # Combine
    if ocr_text and len(ocr_text.split()) >= 5:
        # We have usable OCR text — blend text + image signals
        text_result = analyze_text(ocr_text)
        text_fake = text_result["fake_confidence"] / 100
        text_real = text_result["real_confidence"] / 100

        # Weighted blend: 50% text, 35% CLIP, 15% heuristics
        combined_fake = (0.50 * text_fake) + (0.35 * clip_result["fake"]) + (0.15 * img_heuristics["fake_score"])
        combined_real = 1 - combined_fake

        verdict = "FAKE" if combined_fake > 0.5 else "REAL"
        bullets = _build_image_bullets(ocr_text, clip_result, img_heuristics, has_text=True)
        bullets = text_result["bullets"][:2] + bullets
    else:
        # No OCR — image only
        combined_fake = (0.65 * clip_result["fake"]) + (0.35 * img_heuristics["fake_score"])
        combined_real = 1 - combined_fake
        verdict = "FAKE" if combined_fake > 0.5 else "REAL"
        bullets = _build_image_bullets("", clip_result, img_heuristics, has_text=False)

    return {
        "verdict": verdict,
        "fake_confidence": round(combined_fake * 100, 1),
        "real_confidence": round(combined_real * 100, 1),
        "mode": "image",
        "bullets": bullets,
        "ocr_text": ocr_text[:200] + ("..." if len(ocr_text) > 200 else "") if ocr_text else None,
        "clip_scores": clip_result,
        "img_heuristics": img_heuristics,
    }


def _clip_image_score(pil_image: Image.Image) -> dict:
    """Use CLIP to score image against real/fake news prompts."""
    try:
        import torch
        model, processor = _get_clip()
        prompts = [
            "a real news photograph",
            "a manipulated or fake news image",
            "credible journalism photo",
            "misleading viral social media image",
        ]
        inputs = processor(text=prompts, images=pil_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).tolist()

        fake_score = (probs[1] + probs[3]) / 2
        real_score = (probs[0] + probs[2]) / 2
        return {"fake": fake_score, "real": real_score, "probs": probs}
    except Exception:
        # Fallback neutral
        return {"fake": 0.5, "real": 0.5, "probs": [0.25, 0.25, 0.25, 0.25]}


def _image_heuristics(pil_image: Image.Image) -> dict:
    """Simple image quality / manipulation heuristics."""
    arr = np.array(pil_image.convert("RGB"), dtype=float)
    # Saturation variance — over-saturated images common in fake news
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    sat = (arr.max(axis=2) - arr.min(axis=2)) / (arr.max(axis=2) + 1e-6)
    mean_sat = sat.mean()
    # Noise — high noise can indicate manipulation
    noise = np.std(arr)
    # Extreme contrast
    contrast = arr.std()

    fake_score = 0.5
    flags = []
    if mean_sat > 0.6:
        fake_score += 0.12
        flags.append("high colour saturation (over-processed)")
    if noise > 80:
        fake_score += 0.08
        flags.append("high image noise (possible manipulation)")
    if contrast < 20:
        fake_score += 0.05
        flags.append("very low contrast (possible screenshot artifact)")

    fake_score = min(max(fake_score, 0), 1)
    return {"fake_score": fake_score, "flags": flags, "mean_sat": float(mean_sat)}


def _build_image_bullets(ocr_text, clip_result, img_heuristics, has_text):
    bullets = []
    if has_text:
        bullets.append(f"📷 Text extracted from image via OCR ({len(ocr_text.split())} words) — used for language analysis")
    else:
        bullets.append("📷 No readable text found in image — relying on visual analysis only")

    clip_fake = clip_result["fake"]
    if clip_fake > 0.55:
        bullets.append("🤖 CLIP model associates visual style with misleading/viral imagery")
    elif clip_fake < 0.45:
        bullets.append("🤖 CLIP model associates visual style with credible news photography")
    else:
        bullets.append("🤖 CLIP model finds mixed visual signals")

    for flag in img_heuristics["flags"]:
        bullets.append(f"⚠️ Image quality flag: {flag}")

    if not img_heuristics["flags"]:
        bullets.append("✅ No obvious image manipulation artifacts detected")

    return bullets


# ── Multi-modal fusion ────────────────────────────────────────────────────────
def analyze_multimodal(text: str | None, pil_image: Image.Image | None) -> dict:
    """
    True multi-modal fusion: analyse text and image independently,
    then combine with learned-weight late fusion.
    """
    results = {}

    if text and len(text.strip()) > 10:
        results["text"] = analyze_text(text)
    if pil_image is not None:
        results["image"] = analyze_image(pil_image)

    if not results:
        raise ValueError("Please provide at least one of: text or image.")

    if len(results) == 1:
        # Only one modality — return as-is
        key = list(results.keys())[0]
        r = results[key]
        r["mode"] = "multi"
        return r

    # Late fusion — weighted average
    # Text is stronger signal; image reinforces
    t_fake = results["text"]["fake_confidence"] / 100
    i_fake = results["image"]["fake_confidence"] / 100

    # Agreement bonus: if both agree strongly, boost confidence
    agreement = 1 - abs(t_fake - i_fake)
    weight_text = 0.60
    weight_image = 0.40

    fused_fake = (weight_text * t_fake) + (weight_image * i_fake)
    # Apply agreement bonus (max ±5%)
    if t_fake > 0.5 and i_fake > 0.5:
        fused_fake = min(fused_fake + 0.05 * agreement, 0.98)
    elif t_fake < 0.5 and i_fake < 0.5:
        fused_fake = max(fused_fake - 0.05 * agreement, 0.02)

    fused_real = 1 - fused_fake
    verdict = "FAKE" if fused_fake > 0.5 else "REAL"

    # Merge bullets
    bullets = []
    bullets.append(f"⚡ Multi-modal fusion: Text ({results['text']['fake_confidence']}% fake) + Image ({results['image']['fake_confidence']}% fake)")
    if abs(t_fake - i_fake) < 0.15:
        bullets.append("✅ Both modalities agree — higher confidence in verdict")
    else:
        bullets.append("⚠️ Text and image signals diverge — verdict based on weighted fusion")
    bullets += results["text"]["bullets"][:2]
    bullets += results["image"]["bullets"][:2]

    return {
        "verdict": verdict,
        "fake_confidence": round(fused_fake * 100, 1),
        "real_confidence": round(fused_real * 100, 1),
        "mode": "multi",
        "bullets": bullets,
        "text_result": results.get("text"),
        "image_result": results.get("image"),
    }
