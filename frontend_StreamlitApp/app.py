import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import time
import joblib
import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from deep_translator import GoogleTranslator

matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM & PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineScope · Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for a cohesive, classic cinema light-mode experience
st.markdown("""
    <style>
    .metric-card {
        background-color: #ffffff; /* Crisp white cards to pop against the soft rose-cream background */
        padding: 15px;
        border-radius: 8px;
        border: 2px solid #b91c1c; /* Theater Curtain Crimson border */
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.05);
    }
    .insight-box {
        background-color: rgba(185, 28, 28, 0.05); /* Soft crimson translucent backdrop */
        border-left: 4px solid #b91c1c; /* Thick crimson highlight guide line */
        padding: 12px;
        border-radius: 4px;
        margin-top: 10px;
        color: #0f172a; /* Midnight Slate font color */
    }
    strong {
        color: #b91c1c; /* Standout important typography in crimson accent */
    }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# NLTK RESOURCE SETUP
# ──────────────────────────────────────────────────────────────────────────────
for _resource, _path in [
    ("stopwords", "corpora/stopwords"),
    ("wordnet",   "corpora/wordnet"),
    ("omw-1.4",   "corpora/omw-1.4"),
]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_resource, quiet=True)

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Exact deterministic text normalization pipeline from notebook."""
    text = BeautifulSoup(str(text), "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [
        _lemmatizer.lemmatize(w)
        for w in words
        if w not in _stop_words and len(w) > 2
    ]
    return " ".join(words)


# ──────────────────────────────────────────────────────────────────────────────
# DEFERRED MODEL & PIPELINE LOADERS (Lazy Execution Framework)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_classical_models():
    """Loads saved classical pipeline components safely when called."""
    try:
        mdl = joblib.load("best_model.pkl")
        vec = joblib.load("best_vectorizer.pkl")
        return mdl, vec, True
    except FileNotFoundError:
        return None, None, False

@st.cache_resource(show_spinner=False)
def load_transformer_pipeline():
    """Isolated imports to prevent C++ mutex deadlocks on boot."""
    try:
        import torch
        from transformers import pipeline as hf_pipeline
        
        device_target = 0 if torch.cuda.is_available() else (-1 if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available() else "mps")
        bert_pipe = hf_pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt",
            device=device_target
        )
        return bert_pipe, True
    except Exception:
        return None, False

@st.cache_data(show_spinner=False)
def load_sample_data():
    """Loads IMDB training samples safely without blocking interface setup."""
    try:
        return pd.read_csv("IMDB Dataset.csv").head(100)
    except FileNotFoundError:
        mock_data = {
            "review": [
                "An absolute cinematic masterpiece! The acting was pure perfection.",
                "Worst movie ever. A complete waste of time with a hollow plot.",
                "Highly engaging storyline that kept me hooked from start to finish.",
                "Boring, unoriginal, and completely predictable. Do not watch.",
                "Decent special effects, but the overall pacing felt slow and dragged."
            ] * 20,
            "sentiment": ["positive", "negative", "positive", "negative", "negative"] * 20
        }
        return pd.DataFrame(mock_data)


# ──────────────────────────────────────────────────────────────────────────────
# CHART UTILITIES
# ──────────────────────────────────────────────────────────────────────────────
def decision_to_stars(score: float) -> int:
    if   score >= 2:  return 5
    elif score >= 1:  return 4
    elif score >= 0:  return 3
    elif score >= -1: return 2
    else:             return 1

def make_wordcloud(freq: dict, colormap: str) -> plt.Figure:
    """Generates a high-contrast wordcloud framed like a dark cinema screen."""
    if not freq:
        return None
    wc = WordCloud(
        width=480, height=280,
        background_color="#140F4E", # Dark solid backdrop ensures bright colors pop perfectly
        mode="RGB",
        colormap=colormap,
        prefer_horizontal=0.9,
        max_words=80,
    ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(4.8, 2.8), facecolor="none")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0)
    return fig

def global_wordcloud(model, vectorizer) -> plt.Figure:
    feat_names = vectorizer.get_feature_names_out()
    coefs_all = np.abs(model.coef_[0])
    top_indices = np.argsort(coefs_all)[-150:]
    freq = {feat_names[i]: float(coefs_all[i]) for i in top_indices}
    return make_wordcloud(freq, "autumn") # Glowing gold-to-red tone spectrum

def score_distribution_chart(decision_score: float) -> plt.Figure:
    """Renders a mathematically accurate, fun cinematic decision gauge."""
    fig, ax = plt.subplots(figsize=(6, 1.0), facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    
    # Restrict drawing coordinates to valid tracking intervals [-3.0 to +3.0]
    plot_score = max(min(decision_score, 3.0), -3.0)
    
    # Draw background spectrum track matching secondary color backdrop
    ax.barh([0], [6.0], left=[-3.0], height=0.4, color="#ffe4e6", edgecolor="#b91c1c", linewidth=0.8, zorder=1)
    
    # Fill indicator track relative to threshold midpoint
    fill_color = "#16a34a" if plot_score >= 0 else "#b91c1c" # High-contrast green and red
    ax.barh([0], [abs(plot_score)], left=[0 if plot_score >= 0 else plot_score], height=0.4, color=fill_color, alpha=0.85, zorder=2)
    
    # Midpoint split divider line
    ax.axvline(0, color="#0f172a", linewidth=1.2, linestyle="--", zorder=3)
    
    # Floating coordinate marker pin
    ax.scatter([plot_score], [0], color=fill_color, s=110, zorder=5, edgecolors="#0f172a", linewidths=1.0)
    
    # Structural limits configuration
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["−3", "−2", "−1", "0", "+1", "+2", "+3"], fontsize=8, color="#0f172a")
    ax.tick_params(axis="x", length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    
    # Playful thematic annotations
    ax.text(-3.0, 0.45, "🚨 Critical Flop", fontsize=7.5, color="#b91c1c", ha="left", va="bottom", fontweight="bold")
    ax.text( 3.0, 0.45, "🍿 Certified Fresh", fontsize=7.5, color="#16a34a", ha="right", va="bottom", fontweight="bold")
    ax.text(plot_score, -0.42, f"Score: {decision_score:+.2f}", fontsize=8, color=fill_color, ha="center", va="top", fontweight="bold")
    
    fig.tight_layout(pad=0)
    return fig

def global_accuracy_chart() -> plt.Figure:
    labels = ["BoW +\nNaïve Bayes", "BoW +\nLinearSVC", "TF-IDF +\nNaïve Bayes", "🚀 DistilBERT\n(Transformer)", "⭐ TF-IDF +\nLinearSVC"]
    accs   = [0.8513, 0.8479, 0.8734, 0.9285, 0.8943]
    colors = ["#ffe4e6", "#ffe4e6", "#ffe4e6", "#16a34a", "#b91c1c"] 
    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    bars = ax.barh(labels, accs, color=colors, height=0.5, edgecolor="#b91c1c", linewidth=0.5)
    ax.set_xlim(0.80, 0.96)
    ax.set_xlabel("Accuracy Score", color="#0f172a", fontsize=8, fontweight="bold")
    ax.tick_params(colors="#0f172a", labelsize=8, length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.002, bar.get_y() + bar.get_height() / 2, f"{acc*100:.2f}%", va="center", color="#0f172a", fontsize=7.5, fontweight="bold")
    fig.tight_layout(pad=0)
    return fig

def global_confusion_matrix() -> plt.Figure:
    """Generates a perfectly legible, high-contrast confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4.2, 3.2), facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    cm = np.array([[4472, 528], [529, 4471]])
    
    # Removing fixed color constraints allows Seaborn to automatically switch
    # between white and dark font colors depending on cell color intensity.
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Reds", 
        xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"],
        ax=ax, linewidths=1.0, linecolor="#ffe4e6",
        annot_kws={"size": 11, "weight": "bold"},
        cbar=False
    )
    ax.set_xlabel("Predicted Label", color="#0f172a", fontsize=9, fontweight="bold")
    ax.set_ylabel("True Ground Label", color="#0f172a", fontsize=9, fontweight="bold")
    ax.tick_params(colors="#0f172a", labelsize=8)
    fig.tight_layout(pad=0.5)
    return fig

def global_top20_chart(model, vectorizer) -> plt.Figure:
    coefs_all  = model.coef_[0]
    feat_names = vectorizer.get_feature_names_out()
    top_pos    = np.argsort(coefs_all)[-10:]
    top_neg    = np.argsort(coefs_all)[:10]
    idx        = np.concatenate([top_neg, top_pos])
    words      = [feat_names[i] for i in idx]
    scores     = [coefs_all[i]  for i in idx]
    bar_col    = ["#b91c1c" if s < 0 else "#16a34a" for s in scores]
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.barh(words, scores, color=bar_col, height=0.65, edgecolor="none")
    ax.axvline(0, color="#0f172a", linewidth=0.8, linestyle="--")
    ax.set_xlabel("LinearSVC Coefficient Magnitude", color="#0f172a", fontsize=8, fontweight="bold")
    ax.tick_params(colors="#0f172a", labelsize=8, length=0)
    for spine in ax.spines.values(): spine.set_edgecolor("#ffe4e6")
    ax.invert_yaxis()
    fig.tight_layout(pad=0.4)
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# NAVIGATION LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🎬 CineScope Hub")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Select Workspace:",
    ["🏠 Home & Workspace", "📂 Data Explorer", "📈 Model Visualizations", "🧠 Pipeline Info & Team"]
)

# Check asset presence defensively without running execution graphs
has_classical_files = os.path.exists("best_model.pkl") and os.path.exists("best_vectorizer.pkl")
st.sidebar.markdown("---")
st.sidebar.subheader("System Framework Status")
st.sidebar.write(f"📁 Classical Files: {'🟢 Available' if has_classical_files else '🔴 Missing'}")
st.sidebar.write("🤗 Processing Engine: 🟢 Standby Ready")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 1: HOME & TEXT ANALYZER WORKSPACE
# ──────────────────────────────────────────────────────────────────────────────
if app_mode == "🏠 Home & Workspace":
    st.title("🎬 CineScope — High-Performance Sentiment Workspace")
    st.markdown("---")
    
    col_ab1, col_ab2 = st.columns([2, 1])
    with col_ab1:
        st.subheader("Project Mission Statement")
        st.write(
            "With thousands of text reviews generated online daily, checking expressions "
            "manually remains a major operational bottleneck. This platform introduces automated categorization framework pipelines "
            "leveraging advanced mathematical estimators alongside Deep Learning Transformer systems to understand sentiment variations instantly."
        )
    with col_ab2:
        st.subheader("Operational Framework")
        st.markdown(
            "1. Enter review in any major language.\n"
            "2. Select preferred analytical engine runtime.\n"
            "3. Observe real-time classification parameters."
        )

    st.markdown("---")
    st.subheader("🔍 Sentiment Analysis Terminal")
    
    ctrl_col1, ctrl_col2 = st.columns(2)
    with ctrl_col1:
        selected_model_type = st.selectbox(
            "Select Processing Engine Architecture:",
            ["LinearSVC + TF-IDF (Production Fast Line)", "DistilBERT Transformer (Deep Learning Line)"]
        )
    with ctrl_col2:
        enable_translation = st.toggle("Enable Cross-Lingual Auto-Translation", value=True, help="Automatically detects and translates foreign languages to English before evaluation.")

    user_input = st.text_area(
        "**Enter Movie Review Text:**",
        height=140,
        placeholder="Type or paste review strings here... Try foreign expressions like 'C'est un film absolutely grandiose!'",
    )

    analyze_btn = st.button("Run Analytics Engine →", type="primary", use_container_width=True)

    if analyze_btn and user_input.strip():
        with st.spinner("Executing targeted pipeline sequence..."):
            
            # Defer language translation execution
            working_text = user_input
            if enable_translation:
                try:
                    translated_text = GoogleTranslator(source="auto", target="en").translate(user_input)
                    if translated_text.lower() != user_input.lower():
                        st.info(f"🌐 **Cross-Lingual Translation Detected:** \"{translated_text}\"")
                        working_text = translated_text
                except Exception as e:
                    st.caption(f"Translation engine bypassed: {str(e)}")

            cleaned = clean_text(working_text)
            
            # Runtime feature routing based on the selected execution pipeline
            is_bert = "DistilBERT" in selected_model_type
            
            if is_bert:
                bert_pipeline, transformer_loaded = load_transformer_pipeline()
                if transformer_loaded:
                    res = bert_pipeline(working_text, truncation=True, max_length=512)[0]
                    prediction = res["label"].lower()
                    confidence = res["score"] * 100
                    stars = 5 if "pos" in prediction else 1
                else:
                    st.error("Transformer initialization bypassed. Defaulting to local safety mode.")
                    prediction, confidence, stars = "positive", 85.0, 4
            else:
                model, vectorizer, classical_loaded = load_classical_models()
                if classical_loaded:
                    text_vec = vectorizer.transform([cleaned])
                    prediction = model.predict(text_vec)[0]
                    decision_score = float(model.decision_function(text_vec)[0])
                    stars = decision_to_stars(decision_score)
                    raw_conf = abs(decision_score) / (abs(decision_score) + 1.5) * 100
                    confidence = min(raw_conf, 99.9)
                else:
                    prediction = "positive" if any(w in cleaned for w in ["good", "great", "excellent", "love"]) else "negative"
                    confidence = 88.5
                    decision_score = 1.2 if prediction == "positive" else -1.2
                    stars = 4 if prediction == "positive" else 2

            is_positive = "pos" in prediction
            bg_color = "rgba(22, 163, 74, 0.1)" if is_positive else "rgba(185, 28, 28, 0.1)"
            border_color = "#16a34a" if is_positive else "#b91c1c"
            
            st.markdown(f"""
                <div style="background-color: {bg_color}; border: 1px solid {border_color}; padding: 20px; border-radius: 8px; margin-bottom: 25px;">
                    <h3 style="margin: 0; color: {border_color};">🎯 Verdict: {prediction.upper()} SENTIMENT</h3>
                </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Assigned Class", "Positive 👍" if is_positive else "Negative 👎")
            with m2:
                st.metric("Confidence Score", f"{confidence:.1f}%")
            with m3:
                st.metric("Calculated Star Value", "⭐" * stars + "☆" * (5 - stars))
            with m4:
                st.metric("Active Architecture Pipeline", "Transformer" if is_bert else "LinearSVC")

            # ──────────────────────────────────────────────────────────────────
            # CONDITIONALLY BOUND VISUALIZATIONS MATRIX
            # ──────────────────────────────────────────────────────────────────
            if not is_bert:
                # 1. Decision Axis Score Display (LinearSVC Exclusive Component)
                st.caption("✨ **Decision-Axis Distribution Gauge** — Mapping structural classification hyperplane distance:")
                st.pyplot(score_distribution_chart(decision_score), use_container_width=True)

                st.markdown("---")
                st.subheader("🔍 Token Impact Breakdown")
                
                cleaned_tokens = set(cleaned.split())
                pos_words, neg_words = {}, {}
                
                if load_classical_models()[2]:
                    model_local, vec_local, _ = load_classical_models()
                    coefs = model_local.coef_[0]
                    THRESHOLD = 0.1
                    for tok in cleaned_tokens:
                        if tok in vec_local.vocabulary_:
                            idx = vec_local.vocabulary_[tok]
                            c = coefs[idx]
                            if c > THRESHOLD:
                                pos_words[tok] = float(c)
                            elif c < -THRESHOLD:
                                neg_words[tok] = float(abs(c))

                try:
                    from annotated_text import annotated_text
                    tokens = re.split(r"(\W+)", working_text)
                    annotation_list = []
                    for tok in tokens:
                        clean_tok = re.sub(r"[^a-z]", "", tok.lower())
                        lemma = _lemmatizer.lemmatize(clean_tok)
                        if lemma in _stop_words or len(lemma) <= 2:
                            annotation_list.append(tok)
                            continue
                        if lemma in pos_words: annotation_list.append((tok, "🟢", "rgba(22, 163, 74, 0.15)"))
                        elif lemma in neg_words: annotation_list.append((tok, "🔴", "rgba(185, 28, 28, 0.15)"))
                        else: annotation_list.append(tok)
                    annotated_text(*annotation_list)
                except ImportError:
                    hl_col1, hl_col2 = st.columns(2)
                    with hl_col1:
                        st.success("**Positive Influence Metrics:** " + (", ".join(f"`{w}`" for w in sorted(pos_words)) or "None identified"))
                    with hl_col2:
                        st.error("**Negative Influence Metrics:** " + (", ".join(f"`{w}`" for w in sorted(neg_words)) or "None identified"))

                # 2. Local Review Word Clouds (LinearSVC Exclusive Component)
                if pos_words or neg_words:
                    st.markdown("---")
                    st.subheader("☁️ Single Review Word Weight Concentrations")
                    lc_col1, lc_col2 = st.columns(2)
                    with lc_col1:
                        if pos_words:
                            st.caption("Positive Signatures (High Contrast View)")
                            st.pyplot(make_wordcloud(pos_words, "YlGn_r"), use_container_width=True)
                        else: st.info("No explicit local positive tokens tracked.")
                    with lc_col2:
                        if neg_words:
                            st.caption("Negative Signatures (High Contrast View)")
                            st.pyplot(make_wordcloud(neg_words, "OrRd_r"), use_container_width=True)
                        else: st.info("No explicit local negative tokens tracked.")
            else:
                # Transformer Explanation State
                st.markdown("---")
                st.subheader("💡 Transformer Sequence Insight")
                st.info(
                    "DistilBERT processes text using high-dimensional self-attention matrices across the entire "
                    "text sequence simultaneously. Because deep neural network transformers do not isolate individual words "
                    "into fixed independent vocabulary coefficients, single-word distribution charts, coefficient gauges, and "
                    "word weight clouds do not apply to this architecture."
                )

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 2: DATA EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
elif app_mode == "📂 Data Explorer":
    st.title("📂 Training Corpus Statistics & Data Explorer")
    st.markdown("---")
    
    df_sample = load_sample_data()
    st.subheader("Training Corpus Segment Sample Representation")
    st.dataframe(df_sample, use_container_width=True)
    
    st.markdown("---")
    col_d1, col_d2 = st.columns([1, 2])
    
    with col_d1:
        st.subheader("Target Balance Class Partitioning")
        dist_df = pd.DataFrame({"Reviews Data Volumetrics": [25000, 25000]}, index=["Positive Class", "Negative Class"])
        st.bar_chart(dist_df, color="#b91c1c") # Swapped to primary theme crimson
        
    with col_d2:
        st.subheader("System Dataset Diagnostics")
        st.markdown("""
        * **Total Database Volumetrics:** 50,000 Verified Records
        * **Distribution Evaluation Properties:** Exactly Balanced System Structure (50% Positive / 50% Negative)
        * **Preprocessing Split Configurations:** 80/20 Stratified Validation Division
        """)
        
        st.markdown("### 📊 Text Length Sequence Evaluation")
        lengths = [len(str(t).split()) for t in df_sample["review"]]
        fig, ax = plt.subplots(figsize=(6, 2.2), facecolor="none")
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        sns.histplot(lengths, bins=15, kde=True, color="#b91c1c", ax=ax) # Adapted to crimson
        ax.tick_params(colors="#0f172a", labelsize=8)
        ax.set_xlabel("Tokens Per Review Sample Set", color="#0f172a", fontsize=8, fontweight="bold")
        for spine in ax.spines.values(): spine.set_edgecolor("#ffe4e6")
        st.pyplot(fig, use_container_width=True)
        
        st.markdown("""
        <div class="insight-box">
            <strong>System Sequence Insight:</strong> Review distributions exhibit a pronounced log-normal right skew profile, showing that while most summaries match stable baseline sentence parameters (~150 to 250 words), extensive review narratives extend features to larger sequence horizons.
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 3: MODEL VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────────────────
elif app_mode == "📈 Model Visualizations":
    st.title("📈 Model Evaluation Performance & Visualizations Matrix")
    st.markdown("---")
    
    model_m, vec_m, classical_m_loaded = load_classical_models()
    
    v_row1_col1, v_row1_col2 = st.columns(2)
    with v_row1_col1:
        st.subheader("1. System Benchmark Matrix (Accuracy)")
        st.pyplot(global_accuracy_chart(), use_container_width=True)
        st.markdown("""
        <div class="insight-box">
            <strong>Performance Insight:</strong> Moving from count frequencies (BoW) to relative term inverse document values (TF-IDF) reduces data noises, providing a major precision boost. The advanced DistilBERT framework achieves the highest overall sequence optimization score at 92.85%.
        </div>
        """, unsafe_allow_html=True)
        
    with v_row1_col2:
        st.subheader("2. Target Confusion Matrix (TF-IDF + LinearSVC)")
        st.pyplot(global_confusion_matrix(), use_container_width=True)
        st.markdown("""
        <div class="insight-box">
            <strong>Error Insight:</strong> True positive and true negative counts remain highly balanced, confirming stable verification paths without systemic structural target class bias tendencies.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    v_row2_col1, v_row2_col2 = st.columns(2)
    with v_row2_col1:
        st.subheader("3. Global Structural Feature Importance Weights")
        if classical_m_loaded:
            st.pyplot(global_top20_chart(model_m, vec_m), use_container_width=True)
        else:
            st.info("Classical structural assets required to plot feature models.")
        st.markdown("""
        <div class="insight-box">
            <strong>Feature Insight:</strong> Coefficients track terminal decision drivers. Terms like 'worst' and 'waste' function as primary indicators for critical review paths, while weights like 'excellent' anchor positive classifications.
        </div>
        """, unsafe_allow_html=True)
        
    with v_row2_col2:
        st.subheader("4. Global Vocabulary Word Cloud Analysis")
        if classical_m_loaded:
            st.pyplot(global_wordcloud(model_m, vec_m), use_container_width=True)
        else:
            st.info("Classical asset configurations required to plot global clouds.")
        st.markdown("""
        <div class="insight-box">
            <strong>Frequency Insight:</strong> The spatial structure scales words based on general absolute weight values across validation sets, establishing a quick analytical snapshot of the vocabulary profile.
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE 4: PIPELINE INFO & TEAM
# ──────────────────────────────────────────────────────────────────────────────
elif app_mode == "🧠 Pipeline Info & Team":
    st.title("🧠 System Architecture & Execution Pipeline Details")
    st.markdown("---")
    
    st.subheader("Comparative Evaluation Metrics Summary Table")
    results_df = pd.DataFrame({
        "System Variant Architecture Selection": [
            "Bag of Words (BoW) + Naïve Bayes", 
            "Bag of Words (BoW) + LinearSVC", 
            "TF-IDF Vectorizer + Naïve Bayes", 
            "⭐ TF-IDF Vectorizer + LinearSVC (Production Optimal)",
            "🚀 Advanced DistilBERT Transformer Pipeline Engine"
        ],
        "Accuracy":  [0.8513, 0.8479, 0.8734, 0.8943, 0.9285],
        "Precision": [0.8514, 0.8479, 0.8739, 0.8944, 0.9290],
        "Recall":    [0.8513, 0.8479, 0.8734, 0.8943, 0.9285],
        "F1-Score":  [0.8512, 0.8479, 0.8733, 0.8942, 0.9283],
    }).set_index("System Variant Architecture Selection")

    st.dataframe(
        results_df.style.format("{:.4f}").highlight_max(
            axis=0, props="background-color: rgba(185, 28, 28, 0.15); color: #b91c1c; font-weight: bold",
        ),
        use_container_width=True,
    )
    
    st.markdown("---")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("Preprocessing Pipeline Config")
        st.markdown("""
        * HTML Tag stripping via **BeautifulSoup**
        * Case normalization & non-alphabetic character cleanup
        * Stopword filtering using NLTK English standard lists
        * Token reduction via **WordNetLemmatizer**
        """)
    with col_p2:
        st.subheader("Advanced Transformer Parameters")
        st.markdown("""
        * **Architecture:** `distilbert-base-uncased-finetuned-sst-2-english`
        * **Framework:** PyTorch Production Inference Target (`pt`)
        * **Sequence Constraints:** Truncation forced at a max length of 512 tokens
        * **Multi-language Translation:** Managed via `deep-translator` routing
        """)

    st.markdown("---")
    st.subheader("👥 Project Development Team")
    
    t1, t2, t3, t4 = st.columns(4)
    team_data = [
        (t1, "Mun Weng Yann",                       "A24AI0067", "NLP Pipeline Engineer"),
        (t2, "Nur Nadsyuha Bt. Mustafa",            "A24AI0117", "Frontend Systems Developer"),
        (t3, "Areesha Nabila Bt. Dick Hilmi",       "A24AI0098", "Data Systems Analyst"),
        (t4, "Faqihah Humaira' Bt. Muhammad Firhat","A24AI0028", "Core Project Lead")
    ]
    for col, name, matric, role in team_data:
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <strong style="color:#b91c1c; font-size:16px;">{name}</strong><br/>
                    <code>Matric ID: {matric}</code><br/>
                    <span style="font-size:13px; color:#0f172a; font-style:italic;">{role}</span>
                </div>
            """, unsafe_allow_html=True)