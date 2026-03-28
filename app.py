import streamlit as st
import time
from PIL import Image
import io

# Page config - MUST be first
st.set_page_config(
    page_title="TruthLens — Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load custom CSS
with open("style.css", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from detector import analyze_text, analyze_image, analyze_multimodal
from ui_components import (
    render_header, render_result_card, render_confidence_chart,
    render_analysis_breakdown, render_footer, render_how_it_works
)

# ── Header ──────────────────────────────────────────────────────────────────
render_header()

# ── Mode Selector ────────────────────────────────────────────────────────────
st.markdown("""
<div class="mode-label">SELECT ANALYSIS MODE</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    text_btn = st.button("📝  Text Analysis", key="text_mode", use_container_width=True)
with col2:
    image_btn = st.button("🖼️  Image Analysis", key="image_mode", use_container_width=True)
with col3:
    multi_btn = st.button("⚡  Multi-Modal", key="multi_mode", use_container_width=True)

# Track selected mode in session state
if "mode" not in st.session_state:
    st.session_state.mode = "text"

if text_btn:
    st.session_state.mode = "text"
    st.session_state.result = None
if image_btn:
    st.session_state.mode = "image"
    st.session_state.result = None
if multi_btn:
    st.session_state.mode = "multi"
    st.session_state.result = None

mode = st.session_state.mode

# Active mode indicator
mode_names = {"text": "📝 Text Analysis", "image": "🖼️ Image Analysis", "multi": "⚡ Multi-Modal Fusion"}
st.markdown(f"""
<div class="active-mode-badge">{mode_names[mode]}</div>
""", unsafe_allow_html=True)

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# ── Input Panel ──────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    news_text = None
    uploaded_image = None
    pil_image = None

    if mode == "text":
        st.markdown('<div class="input-label">PASTE YOUR NEWS ARTICLE OR HEADLINE</div>', unsafe_allow_html=True)
        news_text = st.text_area(
            label="news_input",
            placeholder="Paste the full news article, headline, or any suspicious content here...\n\nExample: 'Scientists discover that drinking coffee reverses aging by 20 years, new study claims...'",
            height=260,
            label_visibility="collapsed",
            key="text_input"
        )
        char_count = len(news_text) if news_text else 0
        st.markdown(f'<div class="char-count">{char_count} characters · {len(news_text.split()) if news_text else 0} words</div>', unsafe_allow_html=True)

    elif mode == "image":
        st.markdown('<div class="input-label">UPLOAD AN IMAGE OF THE NEWS</div>', unsafe_allow_html=True)
        st.markdown('<div class="input-sublabel">Upload a screenshot, photo of a headline, WhatsApp forward, or any news image</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            key="image_input"
        )
        if uploaded_image:
            pil_image = Image.open(uploaded_image)
            st.image(pil_image, caption="Uploaded image", use_column_width=True)

    elif mode == "multi":
        st.markdown('<div class="input-label">PASTE TEXT</div>', unsafe_allow_html=True)
        news_text = st.text_area(
            label="multi_text",
            placeholder="Paste the article or caption text here...",
            height=160,
            label_visibility="collapsed",
            key="multi_text_input"
        )
        st.markdown('<div class="input-label" style="margin-top:12px">UPLOAD ASSOCIATED IMAGE</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "Upload image",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
            key="multi_image_input"
        )
        if uploaded_image:
            pil_image = Image.open(uploaded_image)
            st.image(pil_image, caption="Uploaded image", use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analyze button
    analyze_disabled = False
    if mode == "text" and not news_text:
        analyze_disabled = True
    elif mode == "image" and not uploaded_image:
        analyze_disabled = True
    elif mode == "multi" and not news_text and not uploaded_image:
        analyze_disabled = True

    if st.button("🔍  ANALYZE NOW", key="analyze_btn", use_container_width=True, disabled=analyze_disabled):
        with st.spinner(""):
            st.markdown('<div class="analyzing-msg">🧠 Running AI analysis...</div>', unsafe_allow_html=True)
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.015)
                progress.progress(i + 1)
            progress.empty()

            try:
                if mode == "text":
                    result = analyze_text(news_text)
                elif mode == "image":
                    result = analyze_image(pil_image)
                elif mode == "multi":
                    result = analyze_multimodal(news_text, pil_image)

                st.session_state.result = result
                st.session_state.mode_used = mode
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.result = None

# ── Results Panel ─────────────────────────────────────────────────────────────
with right_col:
    if "result" in st.session_state and st.session_state.result:
        result = st.session_state.result
        render_result_card(result)
        st.markdown("<br>", unsafe_allow_html=True)
        render_confidence_chart(result)
        st.markdown("<br>", unsafe_allow_html=True)
        render_analysis_breakdown(result)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <div class="empty-title">Results appear here</div>
            <div class="empty-sub">Choose a mode, provide input,<br>and click Analyze Now</div>
        </div>
        """, unsafe_allow_html=True)

# ── How It Works ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
render_how_it_works()
render_footer()
