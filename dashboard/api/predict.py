"""
Vercel Python serverless function (BaseHTTPRequestHandler convention).

GET /api/predict?ticker=AAPL&end=2023-09-30

Uses a scikit-learn HistGradientBoostingRegressor for live predictions.
Feature attribution is computed via a self-contained ablation method
(perturb each feature to the training median one at a time, measure the
shift in prediction) rather than the `shap` library, since `shap` could
not be reliably installed in Vercel's Python serverless runtime (likely a
C-extension build/timeout issue). This is a standard, simpler alternative
to SHAP — sometimes called occlusion-based or ablation-based attribution —
and is noted explicitly here as a deployment-environment workaround. The
research pipeline (notebooks/SHAP analysis) still uses the full `shap`
library locally, where there's no such constraint.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
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

_model = None
_encoder = None
_df = None
_medians = None


def _load():
    global _model, _encoder, _df, _medians
    if _model is None:
        bundle = joblib.load(MODEL_PATH)
        _model = bundle["model"]
        _encoder = bundle["encoder"]
    if _df is None:
        _df = pd.read_csv(DATA_PATH, parse_dates=["end"])
    if _medians is None:
        _medians = _df[NUMERIC_FEATURES].median()
    return _model, _encoder, _df, _medians


def _make_X(row):
    X = row[NUMERIC_FEATURES].copy()
    X["sector_enc"] = _encoder.transform(row[CATEGORICAL_FEATURES])
    return X


def _ablation_contributions(model, X_row, medians):
    """
    For each numeric feature, replace it with the training-set median and
    measure how much the prediction shifts. The shift (full_pred - ablated_pred)
    is attributed to that feature's deviation from "typical".
    Sector is handled separately as a single categorical contribution by
    comparing to a neutral/most-common sector encoding.
    """
    full_pred = float(model.predict(X_row)[0])
    contributions = []

    for feat in NUMERIC_FEATURES:
        X_ablated = X_row.copy()
        X_ablated[feat] = medians[feat]
        ablated_pred = float(model.predict(X_ablated)[0])
        contributions.append({
            "feature": feat,
            "value": round(full_pred - ablated_pred, 4),
            "raw_value": None,  # filled in by caller
        })

    # Sector: ablate by setting to the most common sector code (mode-like neutral)
    X_ablated_sector = X_row.copy()
    X_ablated_sector["sector_enc"] = 0  # arbitrary neutral baseline
    ablated_sector_pred = float(model.predict(X_ablated_sector)[0])
    contributions.append({
        "feature": "sector",
        "value": round(full_pred - ablated_sector_pred, 4),
        "raw_value": None,
    })

    return full_pred, contributions


def _predict_one(ticker, end):
    model, encoder, df, medians = _load()

    rows = df[df["ticker"] == ticker]
    if end:
        rows = rows[rows["end"].astype(str) == end]
    else:
        rows = rows.sort_values("end").tail(1)

    if rows.empty:
        return None

    row = rows.iloc[[0]]
    X = _make_X(row)

    predicted, contributions = _ablation_contributions(model, X, medians)

    actual_raw = row["sue"].iloc[0]
    actual = float(actual_raw) if pd.notna(actual_raw) else None

    # Fill in raw values for display
    for c in contributions:
        if c["feature"] == "sector":
            c["raw_value"] = str(row["sector"].iloc[0])
        else:
            raw_val = row[c["feature"]].iloc[0]
            if hasattr(raw_val, "item"):
                raw_val = raw_val.item()
            if isinstance(raw_val, float) and np.isnan(raw_val):
                raw_val = None
            c["raw_value"] = raw_val

    contributions.sort(key=lambda c: abs(c["value"]), reverse=True)

    return {
        "ticker": ticker,
        "end": str(row["end"].iloc[0].date()),
        "sector": str(row["sector"].iloc[0]),
        "actual_sue": round(actual, 4) if actual is not None else None,
        "predicted_sue": round(predicted, 4),
        "base_value": round(float(model.predict(
            pd.DataFrame([medians.to_dict() | {"sector_enc": 0}])
        )[0]), 4),
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
