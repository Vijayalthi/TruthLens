"""
ui_components.py — All visual rendering components for TruthLens
"""
import streamlit as st
import plotly.graph_objects as go


def render_header():
    st.markdown("""
    <div class="header-wrap">
        <div class="header-badge">AI-POWERED · MULTI-MODAL</div>
        <h1 class="header-title">Truth<span class="accent">Lens</span></h1>
        <p class="header-sub">Advanced fake news detection using transformer models & computer vision.<br>
        Analyze text, images, or both together for a fused AI verdict.</p>
    </div>
    """, unsafe_allow_html=True)


def render_result_card(result: dict):
    verdict = result["verdict"]
    fake_conf = result["fake_confidence"]
    real_conf = result["real_confidence"]
    mode = result.get("mode", "text")

    if verdict == "FAKE":
        verdict_class = "verdict-fake"
        verdict_icon = "🚨"
        verdict_label = "LIKELY FAKE"
        conf_pct = fake_conf
    else:
        verdict_class = "verdict-real"
        verdict_icon = "✅"
        verdict_label = "LIKELY REAL"
        conf_pct = real_conf

    mode_labels = {"text": "Text", "image": "Image", "multi": "Multi-Modal"}

    st.markdown(f"""
    <div class="result-card {verdict_class}">
        <div class="result-mode-tag">{mode_labels.get(mode, mode).upper()} ANALYSIS</div>
        <div class="result-icon">{verdict_icon}</div>
        <div class="result-verdict-label">{verdict_label}</div>
        <div class="result-confidence">{conf_pct}% confidence</div>
        <div class="result-bar-wrap">
            <div class="result-bar" style="width:{conf_pct}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # OCR preview
    if result.get("ocr_text"):
        st.markdown(f"""
        <div class="ocr-preview">
            <span class="ocr-label">OCR EXTRACTED TEXT</span>
            <span class="ocr-text">"{result['ocr_text']}"</span>
        </div>
        """, unsafe_allow_html=True)


def render_confidence_chart(result: dict):
    fake_conf = result["fake_confidence"]
    real_conf = result["real_confidence"]
    verdict = result["verdict"]

    pull_fake = 0.08 if verdict == "FAKE" else 0
    pull_real = 0.08 if verdict == "REAL" else 0

    fig = go.Figure(data=[go.Pie(
        labels=["Fake", "Real"],
        values=[fake_conf, real_conf],
        hole=0.55,
        pull=[pull_fake, pull_real],
        marker=dict(
            colors=["#ff4757", "#2ed573"],
            line=dict(color="#0d0d0d", width=3)
        ),
        textfont=dict(size=13, color="white", family="Space Mono"),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value}% confidence<extra></extra>",
        direction="clockwise",
    )])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=240,
        annotations=[dict(
            text=f"<b>{fake_conf if verdict == 'FAKE' else real_conf}%</b>",
            x=0.5, y=0.5,
            font=dict(size=22, color="white", family="Space Mono"),
            showarrow=False
        )]
    )

    st.markdown('<div class="chart-label">CONFIDENCE DISTRIBUTION</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_analysis_breakdown(result: dict):
    bullets = result.get("bullets", [])
    if not bullets:
        return
    st.markdown('<div class="breakdown-label">ANALYSIS BREAKDOWN</div>', unsafe_allow_html=True)
    items = "".join([f'<div class="breakdown-item">{b}</div>' for b in bullets])
    st.markdown(f'<div class="breakdown-wrap">{items}</div>', unsafe_allow_html=True)

    # Multi-modal individual scores
    if result.get("mode") == "multi":
        if result.get("text_result") and result.get("image_result"):
            t = result["text_result"]
            i = result["image_result"]
            st.markdown(f"""
            <div class="modality-scores">
                <div class="mod-score-item">
                    <div class="mod-label">📝 TEXT SIGNAL</div>
                    <div class="mod-fake" style="color:{'#ff4757' if t['verdict']=='FAKE' else '#2ed573'}">
                        {t['verdict']} · {t['fake_confidence'] if t['verdict']=='FAKE' else t['real_confidence']}%
                    </div>
                </div>
                <div class="mod-divider">+</div>
                <div class="mod-score-item">
                    <div class="mod-label">🖼️ IMAGE SIGNAL</div>
                    <div class="mod-fake" style="color:{'#ff4757' if i['verdict']=='FAKE' else '#2ed573'}">
                        {i['verdict']} · {i['fake_confidence'] if i['verdict']=='FAKE' else i['real_confidence']}%
                    </div>
                </div>
                <div class="mod-divider">→</div>
                <div class="mod-score-item">
                    <div class="mod-label">⚡ FUSED</div>
                    <div class="mod-fake" style="color:{'#ff4757' if result['verdict']=='FAKE' else '#2ed573'}">
                        {result['verdict']} · {result['fake_confidence'] if result['verdict']=='FAKE' else result['real_confidence']}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_how_it_works():
    st.markdown("""
    <div class="hiw-section">
        <div class="hiw-title">HOW IT WORKS</div>
        <div class="hiw-grid">
            <div class="hiw-card">
                <div class="hiw-num">01</div>
                <div class="hiw-icon">🧠</div>
                <div class="hiw-name">NLI Classification</div>
                <div class="hiw-desc">BART-large-MNLI model runs zero-shot classification — no training needed. Classifies text intent as real or fake.</div>
            </div>
            <div class="hiw-card">
                <div class="hiw-num">02</div>
                <div class="hiw-icon">🔤</div>
                <div class="hiw-name">Linguistic Analysis</div>
                <div class="hiw-desc">Pattern matching detects sensationalist language, clickbait triggers, and credibility signals like citations and data references.</div>
            </div>
            <div class="hiw-card">
                <div class="hiw-num">03</div>
                <div class="hiw-icon">👁️</div>
                <div class="hiw-name">CLIP Vision Model</div>
                <div class="hiw-desc">OpenAI's CLIP scores images against real vs. fake news visual prompts. OCR extracts text from image screenshots.</div>
            </div>
            <div class="hiw-card">
                <div class="hiw-num">04</div>
                <div class="hiw-icon">⚡</div>
                <div class="hiw-name">Multi-Modal Fusion</div>
                <div class="hiw-desc">Late fusion combines text (60%) and image (40%) signals. Agreement between modalities boosts confidence in verdict.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_footer():
    st.markdown("""
    <div class="footer">
        <div class="footer-title">TruthLens</div>
    </div>
    """, unsafe_allow_html=True)
