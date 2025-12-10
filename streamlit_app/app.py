import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# PATHS
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = ROOT / "data" / "cleaned_youtube_ad_revenue.csv"
PREPROCESSOR = ROOT / "models" / "preprocessor.joblib"
BEST_MODEL = ROOT / "models" / "best_model.joblib"
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="YouTube Ad Revenue Predictor", layout="wide")

# -----------------------------
# LOAD MODEL + PREPROCESSOR
# -----------------------------
@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR)

@st.cache_resource
def load_model():
    return joblib.load(BEST_MODEL)

preprocessor = load_preprocessor()
model = load_model()

st.title("🎬 YouTube Ad Revenue Predictor (Content Monetization Modeler)")
st.write("Predict ad revenue using ML + interpret using SHAP 🤖📈")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🧮 Single Prediction", "📂 Batch Prediction", "🧠 Model Explainability"])

# -----------------------------
# TAB 1 — EDA
# -----------------------------
with tab1:
    st.header("📊 Exploratory Data Analysis Outputs")

    png_files = list(OUTPUTS.glob("*.png"))
    if len(png_files) == 0:
        st.info("No EDA images found in outputs/. Run your EDA script first.")
    else:
        for img in png_files:
            st.subheader(img.name)
            st.image(str(img), use_column_width=True)

# -----------------------------
# TAB 2 — SINGLE PREDICTION
# -----------------------------
with tab2:
    st.header("🧮 Predict Revenue for a Single Video")

    with st.form("single_input"):
        views = st.number_input("Views", value=10000)
        likes = st.number_input("Likes", value=1000)
        comments = st.number_input("Comments", value=200)
        watch_time = st.number_input("Watch Time (minutes)", value=9500.0)
        video_length = st.number_input("Video Length (minutes)", value=50.0)
        subscribers = st.number_input("Subscribers", value=500000)

        category = st.selectbox("Category", ["Education", "Entertainment", "Music", "Sports", "News", "Other"])
        device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet", "TV"])
        country = st.text_input("Country Code (e.g., IN, US, CA)", value="IN")

        submitted = st.form_submit_button("Predict Revenue")

    if submitted:
        raw_df = pd.DataFrame([{
            "video_id": "dummy",
            "date": None,
            "views": views,
            "likes": likes,
            "comments": comments,
            "watch_time_minutes": watch_time,
            "video_length_minutes": video_length,
            "subscribers": subscribers,
            "category": category,
            "device": device,
            "country": country
        }])

        st.subheader("🔎 Raw Input")
        st.dataframe(raw_df)

        # transform + predict
        X = preprocessor.transform(raw_df)
        pred = model.predict(X)[0]

        st.success(f"💰 **Predicted Revenue: ${pred:.2f} USD**")

        # SHAP explanation
        st.subheader("🧠 SHAP Explanation")
        try:
            clean_df = pd.read_csv(DATA_CLEAN)
            bg = clean_df.drop(columns=["ad_revenue_usd"]).sample(300, random_state=0)
            explainer = shap.Explainer(model, bg)
            shap_values = explainer(X)

            st.write("Feature contributions (top factors):")
            fig, ax = plt.subplots(figsize=(8,5))
            shap.plots.bar(shap_values, max_display=10, show=False)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")

# -----------------------------
# TAB 3 — BATCH PREDICTION
# -----------------------------
with tab3:
    st.header("📂 Batch Prediction (Upload CSV)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.write("First rows of uploaded file:")
        st.dataframe(df_raw.head())

        X = preprocessor.transform(df_raw)
        preds = model.predict(X)

        df_raw["predicted_revenue"] = preds

        st.subheader("Predictions")
        st.dataframe(df_raw.head())

        csv_bytes = df_raw.to_csv(index=False).encode()
        st.download_button("Download Predictions CSV", csv_bytes, file_name="predicted_revenue.csv")

# -----------------------------
# TAB 4 — MODEL EXPLAINABILITY
# -----------------------------
with tab4:
    st.header("🧠 Global Model Explainability")

    shap_summary_path = OUTPUTS / "shap_summary.png"
    if shap_summary_path.exists():
        st.subheader("Global SHAP Summary")
        st.image(str(shap_summary_path), use_column_width=True)
    else:
        st.info("Run Step 5 SHAP script to generate explainability visuals.")

    st.subheader("Download Trained Model (.joblib)")
    with open(BEST_MODEL, "rb") as f:
        st.download_button("Download best_model.joblib", f, file_name="best_model.joblib")
