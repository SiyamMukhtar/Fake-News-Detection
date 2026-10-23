
from pathlib import Path

import joblib
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

# ---------- Load saved ML artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }
    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #14213d, #244a73);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero p {
        color: #e4edf7;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("📰 NewsGuard")
    st.caption("Machine Learning Demo")
    st.divider()

    st.subheader("About this project")
    st.write(
        "This app uses TF-IDF text features and a "
        "Logistic Regression model to classify news text."
    )

    st.subheader("Model information")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Text features:** TF-IDF")
    st.write("**Grouped test accuracy:** 98.03%")

    st.divider()
    st.caption("Developed as a Machine Learning portfolio project.")

# ---------- Main dashboard ----------
st.markdown(
    """
    <div class="hero">
        <h1>Fake News Detection</h1>
        <p>Explore how a machine learning model classifies news text.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Check that these files exist: "
        "models/fake_news_model.pkl and "
        "models/tfidf_vectorizer.pkl"
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load model files: {exc}")
    st.stop()

st.subheader("Analyze a news article")
st.write("Paste a headline or article below, or load an example.")

example_col1, example_col2 = st.columns(2)

example_fake = (
    "BREAKING: A viral post claims a secret device can "
    "provide unlimited free electricity to every home."
)
example_neutral = (
    "The education department announced a meeting with "
    "school administrators to discuss the upcoming academic year."
)

if "news_input" not in st.session_state:
    st.session_state.news_input = ""

with example_col1:
    if st.button("Load Example 1", use_container_width=True):
        st.session_state.news_input = example_fake

with example_col2:
    if st.button("Load Example 2", use_container_width=True):
        st.session_state.news_input = example_neutral

news_text = st.text_area(
    "News text",
    key="news_input",
    height=220,
    placeholder="Paste the news headline or full article here...",
)

if st.button("🔍 Analyze News", type="primary", use_container_width=True):
    if not news_text.strip():
        st.warning("Please enter some news text first.")
    else:
        try:
            features = vectorizer.transform([news_text])
            prediction = model.predict(features)[0]
            label = str(prediction).upper()

            st.divider()
            st.subheader("Analysis result")

            result_col, confidence_col = st.columns(2)

            with result_col:
                if label == "FAKE":
                    st.error("Model prediction: FAKE")
                elif label == "REAL":
                    st.success("Model prediction: REAL")
                else:
                    st.info(f"Model prediction: {label}")

            with confidence_col:
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features)[0]
                    classes = [str(c).upper() for c in model.classes_]
                    confidence = max(probabilities) * 100

                    st.metric(
                        "Model confidence",
                        f"{confidence:.2f}%",
                    )

                    with st.expander("View class scores"):
                        for class_name, probability in zip(
                            classes, probabilities
                        ):
                            st.write(
                                f"{class_name}: "
                                f"{probability * 100:.2f}%"
                            )
                else:
                    st.info("This model does not provide probability scores.")

            st.info(
                "These results reflect patterns learned from the training "
                "dataset. They do not verify whether the article is true."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.subheader("Important limitation")
st.write(
    "This is a text classification demonstration, not a fact-checking "
    "service. Predictions can be wrong, especially for unfamiliar topics, "
    "short text, satire, or news written differently from the training data."
)

st.caption(
    "Fake News Detection | Python • Streamlit • Scikit-learn"
)
