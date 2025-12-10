"""
Advanced preprocessing for Content Monetization Modeler
Saves:
 - data/cleaned_youtube_ad_revenue.csv
 - models/preprocessor.joblib

Run from project root:
python scripts/step3_preprocessing.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# ---------- CONFIG ----------
ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_IN = ROOT / "data" / "youtube_ad_revenue_dataset.csv"
DATA_OUT = ROOT / "data" / "cleaned_youtube_ad_revenue.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
PIPE_PATH = MODEL_DIR / "preprocessor.joblib"

# Toggle these as you like
APPLY_LOG_FEATURES = True      # log1p on selected features
APPLY_LOG_TARGET = False       # log1p on target (ad_revenue_usd). If True, stored target will be log1p(y)
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99
RANDOM_STATE = 42

# Features expected (safe-check)
EXPECTED_NUMERIC = [
    "views", "likes", "comments", "watch_time_minutes",
    "video_length_minutes", "subscribers", "ad_revenue_usd"
]

# Categorical treatment
ONEHOT_CATEGORICAL = ["device", "category"]  # low-cardinality -> one-hot
FREQ_ENCODE = ["country"]                    # high-cardinality -> frequency encoding

# ---------- HELPERS ----------
def winsorize_series(s, lower=0.01, upper=0.99):
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

class PandasSelector(BaseEstimator, TransformerMixin):
    """Select subset of DataFrame columns and return DataFrame (not numpy array)"""
    def __init__(self, cols):
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.cols]

def safe_divide(numer, denom):
    denom = np.where(denom==0, np.nan, denom)
    return numer / denom

# ---------- LOAD ----------
print("Loading:", DATA_IN)
if not DATA_IN.exists():
    raise FileNotFoundError(f"Missing dataset. Place CSV at: {DATA_IN}")

df = pd.read_csv(DATA_IN, low_memory=False)
print("Initial shape:", df.shape)

# ---------- CLEAN & DUPLICATE ----------

# Drop exact duplicates
dupe_count = df.duplicated().sum()
if dupe_count > 0:
    print(f"Dropping {dupe_count} exact duplicate rows")
    df = df.drop_duplicates().reset_index(drop=True)

# ---------- PARSE DATE (if present) ----------
if "date" in df.columns:
    # try parsing; keep original if not parseable
    try:
        df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
        # create useful temporal features (optional)
        df["upload_day"] = df["date_parsed"].dt.day
        df["upload_month"] = df["date_parsed"].dt.month
        df["upload_weekday"] = df["date_parsed"].dt.weekday
    except Exception as e:
        print("Date parsing failed:", e)

# ---------- MISSING FLAGS (create before imputation) ----------
for col in ["likes", "comments", "watch_time_minutes"]:
    if col in df.columns:
        df[f"{col}_missing_flag"] = df[col].isna().astype(int)

# subscribers missing flag (if ever missing)
if "subscribers" in df.columns:
    df["subscribers_missing_flag"] = df["subscribers"].isna().astype(int)

# ---------- WINSORIZE numeric columns (cap extreme outliers) ----------
numeric_cols = [c for c in EXPECTED_NUMERIC if c in df.columns]
for col in numeric_cols:
    # only perform winsorize if column has numeric dtype
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = winsorize_series(df[col], lower=WINSOR_LOWER, upper=WINSOR_UPPER)

# ---------- FEATURE ENGINEERING ----------
# engagement_rate = (likes + comments) / views
if set(["likes","comments","views"]).issubset(df.columns):
    df["engagement_rate"] = (df["likes"].fillna(0) + df["comments"].fillna(0)) / df["views"].replace({0:np.nan})
    # fill any infinite/nan with 0
    df["engagement_rate"] = df["engagement_rate"].replace([np.inf, -np.inf], np.nan).fillna(0)

# watch_time_per_view
if set(["watch_time_minutes","views"]).issubset(df.columns):
    df["watch_time_per_view"] = safe_divide(df["watch_time_minutes"], df["views"])
    df["watch_time_per_view"] = df["watch_time_per_view"].replace([np.inf, -np.inf], np.nan)

# length_ratio = watch_time_minutes / video_length_minutes
if set(["watch_time_minutes","video_length_minutes"]).issubset(df.columns):
    df["length_ratio"] = safe_divide(df["watch_time_minutes"], df["video_length_minutes"])
    df["length_ratio"] = df["length_ratio"].replace([np.inf, -np.inf], np.nan)

# subscribers_missing_flag already created above. If subscribers has nulls, impute later.

# ---------- FREQUENCY ENCODING for high-cardinality categoricals ----------
for col in FREQ_ENCODE:
    if col in df.columns:
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq).fillna(0)

# ---------- Decide which numeric features to log-transform ----------
log_features = []
if APPLY_LOG_FEATURES:
    for c in ["views", "watch_time_minutes", "subscribers"]:
        if c in df.columns:
            log_features.append(c)

# target log
if APPLY_LOG_TARGET and "ad_revenue_usd" in df.columns:
    df["ad_revenue_usd"] = np.log1p(df["ad_revenue_usd"])

# ---------- PREPARE COLUMNS FOR PIPELINE ----------
# Numeric features for model (exclude target)
numeric_for_model = [
    "views", "likes", "comments", "watch_time_minutes", "video_length_minutes",
    "subscribers", "engagement_rate", "watch_time_per_view", "length_ratio"
]
numeric_for_model = [c for c in numeric_for_model if c in df.columns]

# include any missing flags and freq encoded cols
flag_cols = [c for c in df.columns if c.endswith("_missing_flag")]
freq_cols = [f"{c}_freq" for c in FREQ_ENCODE if f"{c}_freq" in df.columns]

numeric_final = sorted(set(numeric_for_model + flag_cols + freq_cols))

# One-hot categories
onehot_cols = [c for c in ONEHOT_CATEGORICAL if c in df.columns]

print("Numeric features used in pipeline:", numeric_final)
print("One-hot categorical features:", onehot_cols)

# ---------- PIPELINE SETUP ----------
# numeric transformer: impute -> (optional log transform via FunctionTransformer) -> scale
numeric_transform_steps = [
    ("imputer", SimpleImputer(strategy="median"))
]

if APPLY_LOG_FEATURES:
    # apply log1p via FunctionTransformer
    numeric_transform_steps.append(("log1p", FunctionTransformer(np.log1p, validate=False)))

numeric_transform_steps.append(("scaler", StandardScaler()))

numeric_transformer = Pipeline(numeric_transform_steps)

# categorical transformer: impute (constant) -> onehot
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

])

# build ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_final),
        ("cat", categorical_transformer, onehot_cols)
    ],
    remainder="drop"  # drop other columns
)

# Fit transformer on dataframe (drop target)
X_fit = df.copy()
if "ad_revenue_usd" in X_fit.columns:
    X_fit = X_fit.drop(columns=["ad_revenue_usd"])

print("Fitting preprocessing pipeline...")
preprocessor.fit(X_fit)

# Save pipeline
joblib.dump(preprocessor, PIPE_PATH)
print("Saved preprocessor to:", PIPE_PATH)

# ---------- TRANSFORM & SAVE CLEANED CSV ----------
print("Transforming dataset...")
X_trans = preprocessor.transform(X_fit)

# Extract feature names (for numeric + onehot)
num_features_after = numeric_final
onehot_feature_names = []
if onehot_cols:
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    ohe_cols = ohe.get_feature_names_out(onehot_cols).tolist()
    onehot_feature_names = list(ohe_cols)

all_feature_names = num_features_after + onehot_feature_names

X_df = pd.DataFrame(X_trans, columns=all_feature_names, index=df.index)

# attach target if present
if "ad_revenue_usd" in df.columns:
    X_df["ad_revenue_usd"] = df["ad_revenue_usd"].values

# Save cleaned CSV
X_df.to_csv(DATA_OUT, index=False)
print("Saved cleaned dataset to:", DATA_OUT)
print("Final shape:", X_df.shape)

print("Preprocessing complete. Pipeline & cleaned data ready for modeling.")
