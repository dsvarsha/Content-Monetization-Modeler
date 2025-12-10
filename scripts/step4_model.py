"""
Step 4 - Baseline model training & evaluation
Run from project root:
python scripts/step4_model.py
Outputs:
 - models/best_model.joblib
 - models/model_results.csv
 - (prints metrics)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_CLEAN = ROOT / "data" / "cleaned_youtube_ad_revenue.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
RESULTS_CSV = MODELS_DIR / "model_results.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------- LOAD ----------
print("Loading cleaned data:", DATA_CLEAN)
if not DATA_CLEAN.exists():
    raise FileNotFoundError(f"Cleaned data not found. Run step3_preprocessing.py first. Expected at {DATA_CLEAN}")
df = pd.read_csv(DATA_CLEAN)
print("Shape:", df.shape)

# ---------- PREPARE X, y ----------
target_col = "ad_revenue_usd"
if target_col not in df.columns:
    raise KeyError(f"Target column {target_col} not found in cleaned data")

X = df.drop(columns=[target_col])
y = df[target_col].values

# If target was log-transformed earlier, note it here; script assumes raw target.
print("Features shape:", X.shape, "Target shape:", y.shape)

# ---------- TRAIN/TEST SPLIT ----------
# We'll stratify by target quantile buckets to keep distribution similar
y_q = pd.qcut(y, q=10, duplicates="drop")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_q
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------- MODELS SETUP ----------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=5000),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
}

results = []

# helper metrics
def rmse(a,b): return np.sqrt(mean_squared_error(a,b))

# ---------- TRAIN & EVALUATE ----------
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, model in models.items():
    print(f"\nTraining and cross-validating: {name}")
    # quick CV on training set (R2)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
    print(f"  CV R2 (5-fold) mean: {cv_scores.mean():.4f} std: {cv_scores.std():.4f}")
    # fit on train
    model.fit(X_train, y_train)
    # predict on test
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rm = rmse(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"  Test R2: {r2:.4f} | RMSE: {rm:.4f} | MAE: {mae:.4f}")
    results.append({"model": name, "cv_r2_mean": cv_scores.mean(), "cv_r2_std": cv_scores.std(),
                    "test_r2": r2, "test_rmse": rm, "test_mae": mae})
    # save model object temporarily
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")

# ---------- SELECT BEST MODEL ----------
res_df = pd.DataFrame(results).sort_values(by="test_r2", ascending=False).reset_index(drop=True)
res_df.to_csv(RESULTS_CSV, index=False)
print("\nModel results saved to:", RESULTS_CSV)
print(res_df)

best_name = res_df.loc[0, "model"]
best_model_path = MODELS_DIR / f"{best_name}.joblib"
# copy best model to best_model.joblib
best_model = joblib.load(best_model_path)
joblib.dump(best_model, BEST_MODEL_PATH)
print(f"\nBest model: {best_name} saved to {BEST_MODEL_PATH}")

# ---------- QUICK PREDICTION EXAMPLE ----------
sample_X = X_test.iloc[:5]
sample_pred = best_model.predict(sample_X)
print("\nSample predictions (first 5 test rows):")
print(sample_pred)

print("\nStep 4 complete — models trained and best model saved.")
print("Next: Step 5 - explainability (SHAP) or Step 6 - Streamlit app. Which do you want next?")
