"""
QuarterCast modeling pipeline.

Trains a Ridge regression baseline and a LightGBM regressor to predict SUE
(Standardized Unexpected Earnings), using a time-aware train/val/test split
(no random shuffling — we split by fiscal year to avoid lookahead leakage).

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import joblib

FEATURES_PATH = "model_features.csv"
MODEL_OUTPUT_PATH = "../output/lgbm_model.pkl"
RIDGE_OUTPUT_PATH = ".." \
"" \
"/output/ridge_model.pkl"

NUMERIC_FEATURES = [
    "revenue_yoy_growth", "net_income_yoy_growth", "revenue_qoq_growth",
    "net_margin", "net_margin_yoy_change",
    "sue_lag1", "sue_lag2",
    "net_margin_sector_z", "revenue_growth_sector_z",
]
CATEGORICAL_FEATURES = ["sector"]
TARGET = "sue"

TRAIN_YEARS = range(2010, 2020)   # 2010-2019
VAL_YEARS = range(2020, 2022)     # 2020-2021 (includes COVID)
TEST_YEARS = range(2022, 2026)    # 2022-2025


def load_and_split():
    df = pd.read_csv(FEATURES_PATH, parse_dates=["end"])
    df = df[df["fiscal_year"].between(2010, 2025)]  # drop thin 2009, partial 2026

    train = df[df["fiscal_year"].isin(TRAIN_YEARS)]
    val = df[df["fiscal_year"].isin(VAL_YEARS)]
    test = df[df["fiscal_year"].isin(TEST_YEARS)]

    print(f"Train: {len(train)} rows ({train['fiscal_year'].min()}-{train['fiscal_year'].max()})")
    print(f"Val:   {len(val)} rows ({val['fiscal_year'].min()}-{val['fiscal_year'].max()})")
    print(f"Test:  {len(test)} rows ({test['fiscal_year'].min()}-{test['fiscal_year'].max()})")

    return train, val, test


def directional_accuracy(y_true, y_pred):
    """Did we get the sign right (beat vs miss), regardless of magnitude error?"""
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    dir_acc = directional_accuracy(y_true, y_pred)
    print(f"  [{name}] RMSE={rmse:.4f}  MAE={mae:.4f}  Directional Acc={dir_acc:.3f}")
    return {"rmse": rmse, "mae": mae, "directional_accuracy": dir_acc}


def train_ridge(train, val):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    # Ridge can't handle NaN — impute numeric features with column median
    X_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    X_val = val[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in NUMERIC_FEATURES:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_val[col] = X_val[col].fillna(median)

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train, train[TARGET])

    print("\nRidge baseline:")
    evaluate("train", train[TARGET], pipe.predict(X_train))
    evaluate("val", val[TARGET], pipe.predict(X_val))

    return pipe


def train_lightgbm(train, val):
    # LightGBM handles NaN and categoricals natively — no imputation/encoding needed
    X_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    X_val = val[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=20,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, train[TARGET],
        eval_set=[(X_val, val[TARGET])],
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    print("\nLightGBM:")
    evaluate("train", train[TARGET], model.predict(X_train))
    evaluate("val", val[TARGET], model.predict(X_val))

    return model


def main():
    train, val, test = load_and_split()

    ridge_pipe = train_ridge(train, val)
    lgbm_model = train_lightgbm(train, val)

    # Final test-set evaluation (only look at this once both models are finalized)
    X_test_ridge = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in NUMERIC_FEATURES:
        X_test_ridge[col] = X_test_ridge[col].fillna(train[col].median())

    X_test_lgbm = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X_test_lgbm[col] = X_test_lgbm[col].astype("category")

    print("\n=== FINAL TEST SET RESULTS (2022-2025) ===")
    ridge_test_metrics = evaluate("Ridge - test", test[TARGET], ridge_pipe.predict(X_test_ridge))
    lgbm_test_metrics = evaluate("LightGBM - test", test[TARGET], lgbm_model.predict(X_test_lgbm))

    # Sanity check: a naive "always predict 0" baseline (since SUE is
    # standardized, predicting the mean/zero is the trivial benchmark to beat)
    naive_pred = np.zeros(len(test))
    print()
    evaluate("Naive (always 0)", test[TARGET], naive_pred)

    joblib.dump(ridge_pipe, RIDGE_OUTPUT_PATH)
    joblib.dump(lgbm_model, MODEL_OUTPUT_PATH)
    print(f"\nModels saved to {RIDGE_OUTPUT_PATH} and {MODEL_OUTPUT_PATH}")

    return train, val, test, ridge_pipe, lgbm_model


if __name__ == "__main__":
    main()
