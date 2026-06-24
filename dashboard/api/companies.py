"""
GET /api/companies

Returns the list of available tickers (+ sector, latest quarter) so the
dashboard can populate its company selector without bundling the full
dataset into the frontend bundle.
"""

import os
import json
import pandas as pd
from http.server import BaseHTTPRequestHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "_data", "model_features.csv")

_cache = None


def _load_list():
    global _cache
    if _cache is None:
        df = pd.read_csv(DATA_PATH, parse_dates=["end"])
        latest = df.sort_values("end").groupby("ticker").tail(1)
        _cache = [
            {
                "ticker": row["ticker"],
                "sector": row["sector"],
                "latest_end": str(row["end"].date()),
            }
            for _, row in latest.sort_values(["sector", "ticker"]).iterrows()
        ]
    return _cache


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = _load_list()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())
