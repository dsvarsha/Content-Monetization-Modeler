"""
Step5 - SHAP explainability (global + per-sample)
Run from project root:
python scripts/step5_shap.py

Outputs:
 - outputs/shap_summary.png
 - outputs/shap_sample_waterfall.png
 - outputs/shap_values_sample.csv
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = ROOT / "data" / "cleaned_youtube_ad_revenue.csv"
MODELS_DIR = ROOT / "models"
BEST_MODEL = MODELS_DIR / "best_model.joblib"
PREPROC = MODELS_DIR / "preprocessor.joblib"  # may exist
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

print("Loading cleaned data:", DATA_CLEAN)
df = pd.read_csv(DATA_CLEAN)
print("Shape:", df.shape)

# Features / target
target_col = "ad_revenue_usd"
if target_col not in df.columns:
    raise KeyError("Cleaned dataset missing target column")

X = df.drop(columns=[target_col])
y = df[target_col].values

print("Loading model:", BEST_MODEL)
model = joblib.load(BEST_MODEL)  # sklearn model saved in Step 4

# Build SHAP Explainer
print("Preparing SHAP explainer (using a small background sample)...")
# Use a small background sample (500 rows or less) for speed / memory
bg = X.sample(n=min(500, len(X)), random_state=42)
explainer = shap.Explainer(model, bg, feature_perturbation="interventional")  # model-agnostic, but efficient for linear

# Global explanation (summary)
print("Computing SHAP values for a sample for global importance...")
shap_vals = explainer(X.sample(n=min(2000, len(X)), random_state=1))  # subset for speed
plt.figure(figsize=(10,6))
shap.plots.bar(shap_vals, max_display=20, show=False)  # bar plot for global importance
plt.title("SHAP Feature Importance (global)")
plt.tight_layout()
plt.savefig(OUT / "shap_summary.png", dpi=200)
plt.close()
print("Saved:", OUT / "shap_summary.png")

# Per-sample explanation (take one sample from test-like distribution)
sample_idx = int(len(X) * 0.1)  # deterministic pick ~10% into dataset
x_sample = X.iloc[[sample_idx]]
sv = explainer(x_sample)  # shap values for single sample

# Save sample SHAP values as CSV
sv_df = pd.DataFrame({
    "feature": sv.feature_names,
    "shap_value": sv.values[0],
    "feature_value": x_sample.values[0]
})
sv_df = sv_df.sort_values(by="shap_value", key=lambda s: s.abs(), ascending=False)
sv_df.to_csv(OUT / "shap_values_sample.csv", index=False)
print("Saved:", OUT / "shap_values_sample.csv")

# Waterfall / bar for sample
plt.figure(figsize=(10,6))
# shap.plots.waterfall(sv[0], show=False)  # interactive in notebooks; use bar for compatibility
shap.plots.bar(sv, max_display=20, show=False)
plt.title("SHAP contributions for sample (absolute ranking)")
plt.tight_layout()
plt.savefig(OUT / "shap_sample_bar.png", dpi=200)
plt.close()
print("Saved:", OUT / "shap_sample_bar.png")

# Also attempt a waterfall as an image (if supported)
try:
    fig = shap.plots.waterfall(sv[0], show=False)
    fig = plt.gcf()
    fig.set_size_inches(10,6)
    fig.tight_layout()
    plt.savefig(OUT / "shap_sample_waterfall.png", dpi=200)
    plt.close()
    print("Saved:", OUT / "shap_sample_waterfall.png")
except Exception as e:
    print("Waterfall plot failed to save (this is fine on some environments):", e)

print("SHAP step complete. Check outputs/shap_summary.png and outputs/shap_sample_bar.png (and waterfall if available).")
