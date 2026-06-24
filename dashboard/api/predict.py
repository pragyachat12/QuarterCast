"""
Vercel Python serverless function (BaseHTTPRequestHandler convention).

GET /api/predict?ticker=AAPL&end=2023-09-30

Uses a scikit-learn HistGradientBoostingRegressor (not LightGBM) for the
live deployment specifically, since LightGBM's compiled binary requires
libgomp, which isn't available in Vercel's Python runtime. The research
pipeline (scripts/train_model.py) still uses LightGBM — this is a
deployment-only swap, with near-identical performance (RMSE 0.68 vs 0.68,
~88% directional accuracy vs ~89%).
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "_data", "hgb_model_for_dashboard.pkl")
DATA_PATH = os.path.join(BASE_DIR, "_data", "model_features.csv")

NUMERIC_FEATURES = [
    "revenue_yoy_growth", "net_income_yoy_growth", "revenue_qoq_growth",
    "net_margin", "net_margin_yoy_change",
    "sue_lag1", "sue_lag2",
    "net_margin_sector_z", "revenue_growth_sector_z",
]
CATEGORICAL_FEATURES = ["sector"]
ALL_FEATURES = NUMERIC_FEATURES + ["sector_enc"]

_model = None
_encoder = None
_df = None
_explainer = None


def _load():
    global _model, _encoder, _df, _explainer
    if _model is None:
        bundle = joblib.load(MODEL_PATH)
        _model = bundle["model"]
        _encoder = bundle["encoder"]
    if _df is None:
        df = pd.read_csv(DATA_PATH, parse_dates=["end"])
        _df = df
    if _explainer is None:
        # Model-agnostic explainer (works for sklearn HistGradientBoosting,
        # unlike TreeExplainer which is LightGBM/XGBoost-specific)
        background = _df[NUMERIC_FEATURES].median().to_frame().T
        background["sector_enc"] = 0
        _explainer = shap.Explainer(_model.predict, background)
    return _model, _encoder, _df, _explainer


def _make_X(row):
    X = row[NUMERIC_FEATURES].copy()
    X["sector_enc"] = _encoder.transform(row[CATEGORICAL_FEATURES])
    return X


def _predict_one(ticker, end):
    model, encoder, df, explainer = _load()

    rows = df[df["ticker"] == ticker]
    if end:
        rows = rows[rows["end"].astype(str) == end]
    else:
        rows = rows.sort_values("end").tail(1)

    if rows.empty:
        return None

    row = rows.iloc[[0]]
    X = _make_X(row)

    predicted = float(model.predict(X)[0])
    actual_raw = row["sue"].iloc[0]
    actual = float(actual_raw) if pd.notna(actual_raw) else None

    shap_result = explainer(X)
    shap_values = shap_result.values[0]
    base_value = float(np.array(shap_result.base_values).flatten()[0])

    contributions = []
    for i, feat in enumerate(ALL_FEATURES):
        if feat == "sector_enc":
            raw_val = str(row["sector"].iloc[0])
            label = "sector"
        else:
            raw_val = row[feat].iloc[0]
            if hasattr(raw_val, "item"):
                raw_val = raw_val.item()
            if isinstance(raw_val, float) and np.isnan(raw_val):
                raw_val = None
            label = feat
        contributions.append({
            "feature": label,
            "value": round(float(shap_values[i]), 4),
            "raw_value": raw_val,
        })
    contributions.sort(key=lambda c: abs(c["value"]), reverse=True)

    return {
        "ticker": ticker,
        "end": str(row["end"].iloc[0].date()),
        "sector": str(row["sector"].iloc[0]),
        "actual_sue": round(actual, 4) if actual is not None else None,
        "predicted_sue": round(predicted, 4),
        "base_value": round(base_value, 4),
        "contributions": contributions,
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        ticker = params.get("ticker", [""])[0].upper()
        end = params.get("end", [None])[0]

        if not ticker:
            self._send(400, {"error": "Missing required 'ticker' query param"})
            return

        try:
            result = _predict_one(ticker, end)
        except Exception as e:
            self._send(500, {"error": str(e)})
            return

        if result is None:
            self._send(404, {"error": f"No data found for ticker={ticker} end={end}"})
            return

        self._send(200, result)

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())
