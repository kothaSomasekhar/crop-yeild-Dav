"""
train.py – Preprocessing, model training, and evaluation for Indian crop yield prediction.

WHY this file exists:
  We need a reproducible offline training step so the Streamlit app can simply
  load pre-trained models instead of re-training every time a user opens the page.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────────
RAW_CSV   = os.path.join("data", "crops.csv")
CLEAN_CSV = os.path.join("data", "clean.csv")
METRICS_CSV = os.path.join("data", "metrics.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1 – Loading raw data")
print("=" * 60)
df_raw = pd.read_csv(RAW_CSV)
print(f"Raw shape: {df_raw.shape}")
print(f"Columns  : {list(df_raw.columns)}")
print(f"Missing  :\n{df_raw.isnull().sum()}\n")

# WHY sample: reduce dataset size for faster training while maintaining representativeness
df = df_raw.sample(10000, random_state=42).reset_index(drop=True)
print(f"Sampled to {len(df):,} rows from {len(df_raw):,} raw rows")

# ─────────────────────────────────────────────
# 1. Basic wrangling
# ─────────────────────────────────────────────
print("STEP 2 – Cleaning & wrangling")

df = df_raw.copy()

# WHY strip whitespace: raw data often has trailing spaces in string columns
# that silently create duplicate categories.
for col in df.select_dtypes("object").columns:
    df[col] = df[col].str.strip()

# WHY title-case: ensures "kharif", "Kharif", "KHARIF" all become "Kharif"
df["State_Name"]    = df["State_Name"].str.title()
df["District_Name"] = df["District_Name"].str.title()
df["Season"]        = df["Season"].str.strip().str.title()
df["Crop"]          = df["Crop"].str.strip().str.title()

# WHY convert Crop_Year to int: it might have been read as float in some versions
df["Crop_Year"] = df["Crop_Year"].astype(int)

# WHY remove duplicates: duplicate rows skew model training and inflate counts
before_dedup = len(df)
df = df.drop_duplicates()
print(f"  Removed {before_dedup - len(df)} duplicates")

# WHY drop rows where Area is 0 or NaN: Yield = Production/Area would be
# infinite or undefined; these rows are not usable.
df = df[df["Area"] > 0].copy()

# WHY drop rows where Production is missing: target computation needs Production.
df = df.dropna(subset=["Production"])

# WHY drop Coconut rows: Coconut production is measured in nuts, not tonnes,
# making it incomparable with other crops.
before_coconut = len(df)
df = df[df["Crop"] != "Coconut"].copy()
print(f"  Removed {before_coconut - len(df)} Coconut rows")

# WHY create Yield: the dataset has no explicit Yield column; yield is the
# standard agronomic productivity metric (tonnes per hectare).
df["Yield"] = df["Production"] / df["Area"]

# WHY drop Yield <= 0: nonsensical values after division
df = df[df["Yield"] > 0].copy()

print(f"  After cleaning: {df.shape[0]} rows")

# ─────────────────────────────────────────────
# 2. Train/test split
# ─────────────────────────────────────────────
FEATURES_CAT = ["State_Name", "Crop", "Season"]
FEATURES_NUM = ["Area", "Crop_Year"]
TARGET       = "Yield"

X = df[FEATURES_CAT + FEATURES_NUM]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"\nSTEP 3 – Train/test split: {len(X_train):,} train | {len(X_test):,} test")

# ─────────────────────────────────────────────
# 3. Save clean CSV
# ─────────────────────────────────────────────
df.to_csv(CLEAN_CSV, index=False)
print(f"\n  Clean data saved -> {CLEAN_CSV}  ({len(df):,} rows)")

# ─────────────────────────────────────────────
# 6. Preprocessing pipeline
# ─────────────────────────────────────────────
# WHY a Pipeline: bundling preprocessing inside the pipeline prevents data
# leakage – the encoder/scaler only learns from training data and is then
# applied (not re-fitted) on the test set.

# WHY OneHotEncoder with handle_unknown='ignore': at prediction time the app
# might pass a state/crop the encoder has never seen; 'ignore' makes it output
# all-zeros for that category instead of crashing.
cat_transformer = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# WHY StandardScaler: Linear Regression is sensitive to feature magnitude;
# scaling makes gradient-based optimisation faster and numerically stable.
num_transformer = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cat", cat_transformer, FEATURES_CAT),
    ("num", num_transformer, FEATURES_NUM),
])

# ─────────────────────────────────────────────
# 7. Train models on log1p(Yield)
# ─────────────────────────────────────────────
# WHY log1p: yield is right-skewed; log transformation makes it more normal,
# which improves linear model performance. expm1 converts back to original scale.
print("\nSTEP 6 – Training models on log1p(Yield)")

y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

models = {
    "Linear Regression": Pipeline([
        ("prep", preprocessor),
        ("model", LinearRegression())
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocessor),
        # WHY n_estimators=100, max_depth=20, min_samples_leaf=5: as requested.
        ("model", RandomForestRegressor(
            n_estimators=100, max_depth=20, min_samples_leaf=5, n_jobs=-1, random_state=42
        ))
    ]),
}

metrics_rows = []

for name, pipe in models.items():
    print(f"\n  Training: {name}")
    
    # Print hyperparameters for Random Forest
    if "Random Forest" in name:
        rf = pipe.named_steps["model"]
        print(f"    Hyperparameters: n_estimators={rf.n_estimators}, "
              f"min_samples_leaf={rf.min_samples_leaf}, "
              f"random_state={rf.random_state}, n_jobs={rf.n_jobs}")
        print(f"    Training rows used: {len(X_train):,}")
    
    pipe.fit(X_train, y_train_log)
    
    # Predictions on log scale
    y_train_pred_log = pipe.predict(X_train)
    y_test_pred_log = pipe.predict(X_test)
    
    # Convert back to original scale
    y_train_pred = np.expm1(y_train_pred_log)
    y_test_pred = np.expm1(y_test_pred_log)
    
    # Metrics on original scale
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae  = mean_absolute_error(y_test, y_test_pred)
    
    # WHY 5-fold CV: gives a robust generalisation estimate with reasonable computational cost.
    cv_scores = cross_val_score(pipe, X_train, y_train_log, cv=5, scoring="r2", n_jobs=-1)
    cv_r2_mean = cv_scores.mean()
    cv_r2_std  = cv_scores.std()
    
    print(f"    Train R2 : {train_r2:.4f}")
    print(f"    Test R2  : {test_r2:.4f}")
    print(f"    RMSE     : {rmse:.4f}")
    print(f"    MAE      : {mae:.4f}")
    print(f"    CV R2    : {cv_r2_mean:.4f} +/- {cv_r2_std:.4f}")
    
    metrics_rows.append({
        "Model":      name,
        "Train_R2":   round(train_r2, 4),
        "Test_R2":    round(test_r2, 4),
        "RMSE":       round(rmse, 4),
        "MAE":        round(mae, 4),
        "CV_R2_Mean": round(cv_r2_mean, 4),
        "CV_R2_Std":  round(cv_r2_std, 4),
    })
    
    # Save each full pipeline (preprocessor + model)
    safe_name = name.lower().replace(" ", "_")
    joblib.dump(pipe, os.path.join(MODEL_DIR, f"{safe_name}.pkl"))
    print(f"    Saved -> models/{safe_name}.pkl")
    
    # Store predictions for the last model for app use (on original scale)
    joblib.dump((y_test.values, y_test_pred), os.path.join(MODEL_DIR, f"{safe_name}_preds.pkl"))

# ─────────────────────────────────────────────
# 7. Save preprocessor separately (for feature names in app)
# ─────────────────────────────────────────────
# WHY save the preprocessor alone: the Streamlit app needs feature names
# for the Random Forest feature-importance chart.
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

# ─────────────────────────────────────────────
# 8. Save metadata the app needs at runtime
# ─────────────────────────────────────────────
# WHY save these lists: the app's prediction form needs to populate dropdowns
# with the exact values the model was trained on.
meta = {
    "states":       sorted(df["State_Name"].unique().tolist()),
    "crops":        sorted(df["Crop"].unique().tolist()),
    "seasons":      sorted(df["Season"].unique().tolist()),
    "year_min":     int(df["Crop_Year"].min()),
    "year_max":     int(df["Crop_Year"].max()),
    "features_cat": FEATURES_CAT,
    "features_num": FEATURES_NUM,
    "log_transform": True,  # Flag to indicate log transform was used
}
joblib.dump(meta, os.path.join(MODEL_DIR, "meta.pkl"))
print("\n  Metadata saved -> models/meta.pkl")

# ─────────────────────────────────────────────
# 9. Save metrics CSV
# ─────────────────────────────────────────────
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"\n  Metrics saved -> {METRICS_CSV}")
print("\n" + metrics_df.to_string(index=False))

# ─────────────────────────────────────────────
# 10. Explain why Linear Regression won (if it did)
# ─────────────────────────────────────────────
lr_row = metrics_df[metrics_df["Model"] == "Linear Regression"].iloc[0]
rf_row = metrics_df[metrics_df["Model"] == "Random Forest"].iloc[0]

if lr_row["Test_R2"] > rf_row["Test_R2"]:
    print("\n" + "=" * 60)
    print("Linear Regression won. Why?")
    print("=" * 60)
    print("1. Log transformation makes the target distribution more normal,")
    print("   which benefits linear models more than tree-based models.")
    print("2. With no IQR capping, the relationship between features and log-yield")
    print("   is approximately linear, especially after one-hot encoding.")
    print("3. Random Forest with max_depth=20 and min_samples_leaf=5 may be")
    print("   underfitting complex interactions in the high-dimensional feature space.")
    print("=" * 60)

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
