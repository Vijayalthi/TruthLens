# 🚀 TruthLens — Complete Deployment Guide

This guide walks you through putting your project on **GitHub** and deploying it live on **Hugging Face Spaces** so your guide and interviewers can open it in any browser.

---

## STEP 1 — Set Up GitHub Repository

1. Go to [github.com](https://github.com) and sign in (create account if needed)
2. Click the **"+"** icon → **"New repository"**
3. Name it: `truthlens-fake-news-detector`
4. Set to **Public**
5. Do NOT check "Add README" (we already have one)
6. Click **"Create repository"**

### Upload files to GitHub

Option A — **Using GitHub website** (easiest):
- Click **"uploading an existing file"** on the new repo page
- Drag and drop ALL the files from this project folder
- Make sure to include the `.streamlit/` folder too
- Write commit message: `Initial commit — TruthLens Fake News Detector`
- Click **"Commit changes"**

Option B — **Using Git CLI**:
```bash
cd fake-news-detector/
git init
git add .
git commit -m "Initial commit — TruthLens Fake News Detector"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/truthlens-fake-news-detector.git
git push -u origin main
```

---

## STEP 2 — Deploy on Hugging Face Spaces

1. Go to [huggingface.co](https://huggingface.co) and create a free account
2. Click your profile icon → **"New Space"**
3. Fill in:
   - **Space name**: `truthlens-fake-news-detector`
   - **License**: MIT
   - **SDK**: Streamlit  ← IMPORTANT
   - **Hardware**: CPU Basic (Free)
4. Click **"Create Space"**

### Connect to your GitHub repo

5. In your new Space, go to **"Files"** tab
6. You can either:
   - **Upload files directly** (drag and drop same files from GitHub), OR
   - Use the HF Git URL shown on the page to push from CLI:
     ```bash
     git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/truthlens-fake-news-detector
     git push hf main
     ```

7. Hugging Face will automatically detect `app.py` and start building!

### Wait for build (~5-8 minutes first time)
- You'll see a green **"Running"** badge when it's live
- Your app URL will be: `https://huggingface.co/spaces/YOUR_HF_USERNAME/truthlens-fake-news-detector`

---

## STEP 3 — Share with your guide

Send Dr. A Kavitha this URL — she can open it in any browser, no installation needed.

For interviews, say:
> "You can view the live demo at [your HF URL] — it's deployed on Hugging Face Spaces and the source code is at [your GitHub URL]"

---

## Running locally (for development)

```bash
# 1. Install Python 3.10+
# 2. Install tesseract (for OCR):
#    Windows: https://github.com/UB-Mannheim/tesseract/wiki
#    Mac:     brew install tesseract
#    Linux:   sudo apt install tesseract-ocr

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

App opens at: http://localhost:8501

---

## Project Structure

```
truthlens-fake-news-detector/
├── app.py              ← Main Streamlit app (entry point)
├── detector.py         ← ML inference engine (BART + CLIP + OCR)
├── ui_components.py    ← All UI rendering functions
├── style.css           ← Custom CSS theme
├── requirements.txt    ← Python packages
├── packages.txt        ← System packages (tesseract) for HF Spaces
├── README.md           ← Project description (shown on HF)
├── DEPLOYMENT.md       ← This file
└── .streamlit/
    └── config.toml     ← Streamlit theme config
```

---

## How the models work (for viva)

| Component | Model | Purpose |
|-----------|-------|---------|
| Text classifier | `facebook/bart-large-mnli` | Zero-shot NLI — classifies text as real/fake without any training |
| Vision model | `openai/clip-vit-base-patch32` | Scores images against real vs fake news visual prompts |
| OCR | Tesseract | Extracts text from images (WhatsApp forwards, screenshots) |
| Fusion | Weighted late fusion | Combines text (60%) + image (40%) signals |

### Why zero-shot instead of training?
Training a custom model on 10k+ examples requires:
- GPU (expensive, crashes Colab)
- Perfect dataset (class imbalance causes "always fake" bug)
- Weeks of tuning

Zero-shot classification uses models already trained on 100M+ examples from Meta and OpenAI. This is how **production AI systems at companies like Google and Meta work** — they don't retrain from scratch; they adapt pre-trained models.

---

## Troubleshooting

**"Module not found"** → Run `pip install -r requirements.txt`

**"Tesseract not found"** → Install tesseract for your OS (see local setup above)

**App loads but model is slow first time** → Models download on first use (~1-2 GB). Subsequent runs are instant.

**HF Space stuck on "Building"** → Check the build logs tab for errors. Usually a package version issue.
