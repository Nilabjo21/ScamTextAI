"""
ScamText AI - Streamlit Application
User-facing interface for scam detection and classification
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import FeatureEngineer
from src.model_training import BaselineModel
from src.prediction import ScamDetector
from src.pipeline import ScamTextAIPipeline


# Page configuration
st.set_page_config(
    page_title="ScamText AI",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 94% 0%, rgba(49, 94, 251, 0.07), transparent 34rem),
            #f4f6fb;
    }
    [data-testid="stHeader"] {
        background: rgba(244, 246, 251, 0.88);
        backdrop-filter: blur(12px);
    }
    [data-testid="stSidebar"] {
        background: #111d33;
        border-right: 1px solid #26344d;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        color: #f4f7ff;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #aebcd2;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2b3a54;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4.5rem;
    }
    h1, h2, h3 {
        color: #17233c;
        letter-spacing: -0.025em;
    }
    h2 {
        margin-top: 0.5rem;
    }
    [data-testid="stCaptionContainer"] {
        color: #64728a;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5eaf3;
        border-radius: 14px;
        padding: 1rem 1.15rem;
        box-shadow: 0 3px 14px rgba(25, 43, 77, 0.035);
    }
    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
        min-height: 2.85rem;
        transition: transform 120ms ease, box-shadow 120ms ease;
    }
    div.stButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
        box-shadow: 0 5px 14px rgba(49, 94, 251, 0.2);
    }
    div.stButton > button:hover,
    div.stDownloadButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px);
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        justify-content: flex-start;
        border: 1px solid #34445f;
        background: #1b2942;
        color: #d8e1f2;
        text-align: left;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        border-color: #91b1ff;
        background: #263b60;
        color: #ffffff;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button[kind="primary"] {
        border-color: #315efb;
        background: #315efb;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(49, 94, 251, 0.25);
    }
    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e4e9f2;
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 8px 24px rgba(25, 43, 77, 0.045);
    }
    [data-testid="stTextArea"] textarea,
    [data-testid="stFileUploader"] section {
        border-radius: 10px;
    }
    [data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 0.55rem;
    }
    [data-testid="stMain"] [data-testid="stRadio"] > div[role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #e4e9f2;
        border-radius: 999px;
        padding: 0.45rem 0.95rem;
    }
    .scam-hero {
        position: relative;
        overflow: hidden;
        margin: 0.35rem 0 1.8rem;
        padding: 2rem 2.2rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 22px;
        color: #ffffff;
        background: linear-gradient(112deg, #111e38 0%, #1d3763 58%, #315efb 120%);
        box-shadow: 0 16px 34px rgba(27, 51, 91, 0.16);
    }
    .scam-hero::after {
        position: absolute;
        top: -130px;
        right: -45px;
        width: 310px;
        height: 310px;
        border: 1px solid rgba(255, 255, 255, 0.11);
        border-radius: 50%;
        box-shadow: 0 0 0 34px rgba(255, 255, 255, 0.035), 0 0 0 70px rgba(255, 255, 255, 0.025);
        content: "";
        pointer-events: none;
    }
    .scam-hero-brand {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 1.2rem;
    }
    .scam-hero-name {
        color: #ffffff;
        font-size: 1.06rem;
        font-weight: 700;
        letter-spacing: 0.01em;
    }
    .scam-hero-kicker {
        margin: 0 0 0.45rem;
        color: #b9cdf7;
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .scam-hero-title {
        max-width: 710px;
        margin: 0;
        color: #ffffff;
        font-size: clamp(1.75rem, 3vw, 2.45rem);
        font-weight: 700;
        letter-spacing: -0.045em;
        line-height: 1.15;
    }
    .scam-hero-copy {
        max-width: 650px;
        margin: 0.7rem 0 0;
        color: #d4def0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .scam-hero-tag {
        position: absolute;
        right: 1.6rem;
        bottom: 1.5rem;
        color: #c4d3f1;
        font-size: 0.72rem;
        font-weight: 650;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .scam-info-card {
        height: 100%;
        padding: 1.25rem 1.35rem;
        border: 1px solid #e3e9f3;
        border-radius: 16px;
        background: linear-gradient(150deg, #ffffff 0%, #f8faff 100%);
        box-shadow: 0 8px 24px rgba(25, 43, 77, 0.045);
    }
    .scam-info-eyebrow {
        margin: 0 0 0.4rem;
        color: #466ad3;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }
    .scam-info-title {
        margin: 0 0 0.75rem;
        color: #17233c;
        font-size: 1.1rem;
        font-weight: 700;
    }
    .scam-info-copy {
        margin: 0;
        color: #58677f;
        font-size: 0.92rem;
        line-height: 1.65;
    }
    .scam-section-label {
        margin: 1.4rem 0 0.25rem;
        color: #263653;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }
    #scamtext-analysis-results {
        scroll-margin-top: 1.5rem;
    }
    @media (max-width: 720px) {
        .scam-hero {
            padding: 1.5rem;
        }
        .scam-hero-tag {
            position: static;
            display: inline-block;
            margin-top: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_detector():
    """Load the trained baseline and its fitted feature mapping, if available."""
    artifact_path = PROJECT_ROOT / "models" / "scamtext_detector.joblib"
    if not artifact_path.is_file():
        return None
    artifact = joblib.load(artifact_path)
    model = BaselineModel("logistic")
    model.model = artifact["model"]
    model.threshold = float(artifact["threshold"])
    feature_engineer = FeatureEngineer()
    feature_engineer.tfidf = artifact["vectorizer"]
    return ScamDetector(model, feature_engineer)


def apply_result_background(is_scam):
    """Tint the main app canvas with a subtle result-specific color."""
    background = "#ffe5e1" if is_scam else "#e2f2e5"
    header_background = "#fff0ed" if is_scam else "#eff8f0"
    st.markdown(
        f"""
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {{
            background-color: {background} !important;
            background-image: none !important;
        }}
        [data-testid="stHeader"] {{
            background-color: {header_background} !important;
        }}
        [data-testid="stMainBlockContainer"] {{
            background-color: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def reset_result_background():
    """Use a neutral canvas outside the message result page."""
    st.markdown(
        """
        <style>
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: #ffffff !important;
            background-image: none !important;
        }
        [data-testid="stHeader"] {
            background: #ffffff !important;
        }
        [data-testid="stMainBlockContainer"] {
            background-color: transparent !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    pages = [
        ("Detector", "Check a message"),
        ("Analytics", "Model & data"),
        ("About", "About the project"),
        ("Train model", "Train model"),
    ]
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "Detector"

    with st.sidebar:
        st.markdown(
            '<div class="scam-hero-brand"><div class="scam-hero-name">ScamText AI</div></div>',
            unsafe_allow_html=True,
        )
        st.caption("MESSAGE SCREENING · NLP")
        st.divider()
        for page_name, label in pages:
            if st.button(
                label,
                key=f"navigation_{page_name}",
                type="primary" if st.session_state["selected_page"] == page_name else "secondary",
                use_container_width=True,
            ):
                st.session_state["selected_page"] = page_name
                st.rerun()
        st.divider()
        st.caption("A screening aid, not a guarantee.")
        st.caption("Verify suspicious requests through an official channel.")

    page = st.session_state["selected_page"]
    review = st.session_state.get("single_message_review")
    if page != "Detector":
        reset_result_background()
    elif review:
        apply_result_background(review["prediction"]["is_scam"])

    st.markdown(
        """
        <section class="scam-hero">
          <div class="scam-hero-brand">
            <div class="scam-hero-name">ScamText AI</div>
          </div>
          <p class="scam-hero-kicker">NLP-powered message review</p>
          <h1 class="scam-hero-title">A clearer first look at suspicious messages.</h1>
          <p class="scam-hero-copy">
            Review a message with a trained text classifier, then inspect the signals
            behind the result. Designed to inform a decision—not make it for you.
          </p>
          <span class="scam-hero-tag">Research prototype</span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if page == "Detector":
        detector_page()
    elif page == "Analytics":
        analytics_page()
    elif page == "About":
        about_page()
    elif page == "Train model":
        train_page()


def detector_page():
    st.header("Check a message")
    st.write("Paste a message for a spam-like assessment, or review a CSV in one batch.")
    input_method = st.radio(
        "Choose how to analyze",
        ["Single message", "Batch CSV"],
        horizontal=True,
    )

    if input_method == "Single message":
        input_col, guidance_col = st.columns([1.55, 1], gap="large")
        with input_col:
            st.markdown('<p class="scam-section-label">Message to review</p>', unsafe_allow_html=True)
            with st.form("single_message_form"):
                message = st.text_area(
                    "Message text",
                    placeholder="Paste the SMS or message text here…",
                    height=190,
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button(
                    "Review message",
                    type="primary",
                    use_container_width=True,
                    icon=":material/search:",
                )
        with guidance_col:
            st.markdown(
                """
                <div class="scam-info-card">
                  <p class="scam-info-eyebrow">What to expect</p>
                  <h3 class="scam-info-title">Evidence, not a verdict</h3>
                  <p class="scam-info-copy">
                    The classifier looks for patterns learned from labeled messages.
                    You’ll see its spam score alongside separate text signals and
                    model evidence.
                  </p>
                  <p class="scam-section-label">Use good judgment</p>
                  <p class="scam-info-copy">
                    Never use a score alone to decide whether to share money,
                    passwords, or personal details. Verify requests with the
                    organization using its official contact information.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if submitted:
            st.session_state.pop("single_message_review", None)
            if message.strip():
                analyze_single_message(message)
            else:
                st.warning("Enter a message before submitting.")
        if st.session_state.get("single_message_review"):
            render_single_message_result()
    else:
        upload_col, guidance_col = st.columns([1.55, 1], gap="large")
        with upload_col:
            st.markdown('<p class="scam-section-label">Batch file</p>', unsafe_allow_html=True)
            with st.container(border=True):
                st.write("Upload a CSV containing a `message` column.")
                uploaded_file = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
                process_batch = st.button(
                    "Review CSV",
                    type="primary",
                    use_container_width=True,
                    icon=":material/upload_file:",
                )
        with guidance_col:
            st.markdown(
                """
                <div class="scam-info-card">
                  <p class="scam-info-eyebrow">Batch review</p>
                  <h3 class="scam-info-title">Keep your data tidy</h3>
                  <p class="scam-info-copy">
                    Include one message per row in a column named <strong>message</strong>.
                    Empty entries are skipped. Results can be reviewed in the app
                    and exported as a CSV.
                  </p>
                  <p class="scam-section-label">Privacy reminder</p>
                  <p class="scam-info-copy">
                    Only upload messages you are permitted to process. Remove
                    personal information that is not needed for your review.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if uploaded_file and process_batch:
            analyze_batch(uploaded_file)


def analyze_single_message(message):
    detector = load_detector()
    if detector is None:
        st.warning("No trained model is available. Run the training pipeline from the Train model page.")
        return
    try:
        prediction = detector.predict(message)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.session_state["single_message_review"] = {
        "message": message,
        "prediction": prediction,
        "threshold": detector.model.threshold,
    }
    apply_result_background(prediction["is_scam"])
    st.session_state["scroll_to_results"] = True


def render_single_message_result():
    review = st.session_state["single_message_review"]
    prediction = review["prediction"]
    st.markdown('<div id="scamtext-analysis-results"></div>', unsafe_allow_html=True)
    st.divider()
    st.subheader("Assessment")
    if prediction["is_scam"]:
        st.error("Flagged as spam-like", icon="⚠️")
    else:
        st.success("Not flagged by the spam classifier", icon="✅")
    score_col, confidence_col = st.columns(2)
    score_col.metric("Model spam score", f"{prediction['scam_probability']:.1%}")
    confidence_col.metric("Decision threshold", f"{review['threshold']:.1%}")
    st.caption("Scores are not calibrated risk estimates. A message not flagged may still be fraudulent.")

    st.subheader("Review signals")
    if prediction['indicators']:
        indicator_rows = [
            {"Signal type": category.replace("_", " ").title(), "Observed text or pattern": ", ".join(values)}
            for category, values in prediction["indicators"].items()
        ]
        st.dataframe(pd.DataFrame(indicator_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No configured lexical warning signals were detected. This does not establish that the message is safe.")

    if prediction["top_features"]:
        with st.expander("Model evidence and normalized text"):
            evidence = pd.DataFrame(prediction["top_features"])
            evidence = evidence.rename(columns={"feature": "Text feature", "contribution": "Contribution"})
            evidence["Contribution"] = evidence["Contribution"].map(lambda value: f"{value:.4f}")
            st.dataframe(evidence, hide_index=True, use_container_width=True)
            st.caption("Feature contributions show influential terms for this linear model; they are not causal explanations.")
            st.text(prediction["cleaned_message"])
    else:
        with st.expander("Normalized text"):
            st.text(prediction["cleaned_message"])

    if st.session_state.pop("scroll_to_results", False):
        st.html(
            """
            <script>
            requestAnimationFrame(() => {
                const target = document.getElementById("scamtext-analysis-results");
                if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
            });
            </script>
            """,
            unsafe_allow_javascript=True,
        )


def analyze_batch(uploaded_file):
    st.subheader("Batch Analysis Results")

    try:
        df = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read CSV: {exc}")
        return
    if "message" not in df.columns:
        st.error("CSV must contain a 'message' column")
        return

    detector = load_detector()
    if detector is None:
        st.warning("No trained model is available. Run the training pipeline first.")
        return
    messages = df["message"].fillna("").astype(str).tolist()
    if not messages:
        st.warning("The uploaded CSV contains no messages.")
        return
    progress_bar = st.progress(0)
    results = []
    for idx, message in enumerate(messages, start=1):
        if not message.strip():
            results.append({
                "message": message,
                "prediction": "NOT ANALYZED",
                "confidence": None,
                "scam_probability": None,
                "indicators": "",
            })
        else:
            prediction = detector.predict(message)
            results.append({
                "message": message,
                "prediction": "SCAM" if prediction["is_scam"] else "NOT FLAGGED",
                "confidence": prediction["confidence"],
                "scam_probability": prediction["scam_probability"],
                "indicators": "; ".join(
                    f"{category}: {', '.join(values)}"
                    for category, values in prediction["indicators"].items()
                ),
            })
        progress_bar.progress(idx / len(messages))

    results_df = pd.DataFrame(results)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Messages", len(results_df))
    with col2:
        st.metric("Flagged as scam", (results_df["prediction"] == "SCAM").sum())
    with col3:
        st.metric("Not flagged", (results_df["prediction"] == "NOT FLAGGED").sum())
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    st.download_button(
        label="📥 Download Results as CSV",
        data=results_df.to_csv(index=False),
        file_name="scam_detection_results.csv",
        mime="text/csv",
    )


def analytics_page():
    st.header("Model and data")
    output_dir = PROJECT_ROOT / "outputs"
    comparison_path = output_dir / "model_comparison.csv"
    stats_path = output_dir / "dataset_statistics.json"
    if stats_path.is_file():
        import json
        with stats_path.open(encoding="utf-8") as stats_file:
            stats = json.load(stats_file)
        ham_col, spam_col, source_col = st.columns(3)
        ham_col.metric("Unique legitimate messages", f"{stats['classes']['legitimate']:,}")
        spam_col.metric("Unique spam-like messages", f"{stats['classes']['spam_or_scam']:,}")
        source_col.metric("Unique total", f"{stats['samples']:,}")
        st.subheader("Dataset sources")
        source_rows = [
            {"Source": source.replace("_", " "), "Unique messages": count}
            for source, count in stats["sources"].items()
        ]
        st.dataframe(pd.DataFrame(source_rows), hide_index=True, use_container_width=True)
        skipped = stats.get("skipped_source_rows", {})
        if skipped:
            skipped_rows = [
                {"File": filename, "Rows excluded": counts["rows"]}
                for filename, counts in skipped.items()
            ]
            st.caption("Malformed or empty rows were excluded rather than assigned labels.")
            st.dataframe(pd.DataFrame(skipped_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Dataset statistics will appear after a successful training run.")

    if comparison_path.is_file():
        st.subheader("Held-out test performance")
        st.caption("Results come from the latest pipeline run; the test split is not used to select the decision threshold.")
        comparison = pd.read_csv(comparison_path, index_col=0)
        st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
    else:
        st.info("Measured model results will appear here after training.")


def about_page():
    st.header("About this project")
    st.write(
        "ScamText AI is an academic prototype that screens message text for patterns "
        "associated with spam. It combines a trained binary classifier with separate, "
        "inspectable text signals."
    )
    with st.expander("How predictions are produced", expanded=True):
        st.write(
            "Text is normalized and represented with word-level TF-IDF features. "
            "The current interface uses the trained Logistic Regression model. "
            "The model reports spam-like versus legitimate; it does not predict a "
            "specific scam category."
        )
    with st.expander("Responsible use"):
        st.write(
            "A model score and a missing warning signal cannot establish that a "
            "message is safe. Do not share sensitive message contents unnecessarily. "
            "Confirm urgent payment, account, or identity requests through a trusted "
            "official channel."
        )
    st.caption("ScamText AI · An Explainable Text Classifier Prototype")


def train_page():
    st.header("Train and evaluate")
    st.write("Run the local pipeline to validate the data, fit both baselines, and refresh held-out evaluation results.")
    include_distilbert = st.checkbox(
        "Also fine-tune DistilBERT (downloads a pretrained model and may take considerably longer)",
        value=False,
    )

    if st.button("Start training", type="primary", use_container_width=True):
        pipeline = ScamTextAIPipeline(
            data_dir=PROJECT_ROOT / "data",
            output_dir=PROJECT_ROOT / "outputs",
            models_dir=PROJECT_ROOT / "models",
        )
        try:
            with st.spinner("Loading verified labeled datasets, training, and evaluating..."):
                _, comparison = pipeline.run_pipeline(include_distilbert=include_distilbert)
            load_detector.clear()
            st.success("Training and held-out evaluation completed.")
            st.dataframe(comparison, use_container_width=True)
        except (RuntimeError, ValueError, OSError, ImportError) as exc:
            st.error(f"Training failed: {exc}")


if __name__ == "__main__":
    main()
