---
title: TruthLens — Fake News Detector
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
license: mit
short_description: Multi-modal fake news detection using BART + CLIP + OCR
---

# TruthLens — Fake News Detector


A multi-modal fake news detection system that combines:
- 🧠 **BART-large-MNLI** for zero-shot text classification
- 👁️ **CLIP ViT** for image-based credibility scoring
- 🔤 **OCR (Tesseract)** for extracting text from news images
- ⚡ **Late fusion** combining all signals into a final verdict


