"""
Feature engineering for QuarterCast.

Takes the raw EDGAR quarterly pull (edgar_quarterly_eps.csv) and the company
universe (with sector labels), and builds a model-ready feature table.

Features built:
  - Growth: YoY revenue growth, YoY net income growth, QoQ revenue growth
  - Profitability: net margin, margin change YoY
  - Surprise history: prior-quarter SUE (autocorrelation signal), trailing
    surprise volatility (already computed upstream)
  - Sector context: sector dummy, sector-normalized margin (z-score within
    sector-quarter), sector-normalized revenue growth
  - Size/regime controls: log market cap proxy (using assets as a stand-in
    since we don't have market cap from EDGAR), year (for regime checks)

Target: `sue` (Standardized Unexpected Earnings), already computed upstream.

Usage:
    python build_features.py
"""

import pandas as pd
import numpy as np

EDGAR_PATH = "edgar_quarterly_eps.csv"
UNIVERSE_PATH = "company_universe.csv"
OUTPUT_PATH = "model_features.csv"


def load_data():
    df = pd.read_csv(EDGAR_PATH, parse_dates=["end", "start"])
    universe = pd.read_csv(UNIVERSE_PATH)[["Symbol", "GICS Sector"]].rename(
        columns={"Symbol": "ticker", "GICS Sector": "sector"}
    )
    df = df.merge(universe, on="ticker", how="left")
    return df


def add_growth_features(df):
    df = df.sort_values(["ticker", "end"]).reset_index(drop=True)
    g = df.groupby("ticker")

    # YoY growth (vs same quarter last year, i.e. lag 4)
    df["revenue_yoy_growth"] = g["revenue"].transform(lambda s: s.pct_change(4))
    df["net_income_yoy_growth"] = g["net_income"].transform(lambda s: s.pct_change(4))

    # QoQ growth (vs previous quarter, lag 1) — noisier, captures momentum
    df["revenue_qoq_growth"] = g["revenue"].transform(lambda s: s.pct_change(1))

    return df


def add_profitability_features(df):
    df["net_margin"] = df["net_income"] / df["revenue"]
    g = df.groupby("ticker")
    df["net_margin_yoy_change"] = g["net_margin"].transform(lambda s: s.diff(4))
    return df


def add_surprise_history_features(df):
    g = df.groupby("ticker")
    # Prior quarter's SUE — captures surprise autocorrelation/momentum
    df["sue_lag1"] = g["sue"].shift(1)
    df["sue_lag2"] = g["sue"].shift(2)
    return df


def add_sector_normalized_features(df):
    # Z-score within sector + fiscal quarter, so we're comparing companies
    # to peers reporting around the same time (controls for sector-wide
    # macro effects, e.g. all energy companies having a strong quarter
    # together due to oil prices).
    df["year_quarter"] = df["end"].dt.to_period("Q").astype(str)

    def zscore(s):
        std = s.std()
        if std == 0 or pd.isna(std):
            return pd.Series(np.nan, index=s.index)
        return (s - s.mean()) / std

    df["net_margin_sector_z"] = (
        df.groupby(["sector", "year_quarter"])["net_margin"]
        .transform(zscore)
    )
    df["revenue_growth_sector_z"] = (
        df.groupby(["sector", "year_quarter"])["revenue_yoy_growth"]
        .transform(zscore)
    )
    return df


def add_regime_controls(df):
    df["fiscal_year"] = df["end"].dt.year
    # Flag COVID-era quarters (2020 Q1 - 2021 Q2) for sensitivity checks later
    df["is_covid_era"] = df["end"].between("2020-01-01", "2021-06-30")
    return df


def main():
    df = load_data()
    print(f"Loaded {len(df)} rows, {df['ticker'].nunique()} tickers")

    df = add_growth_features(df)
    df = add_profitability_features(df)
    df = add_surprise_history_features(df)
    df = add_sector_normalized_features(df)
    df = add_regime_controls(df)

    # Drop rows without a valid target — can't train/eval on these
    before = len(df)
    df = df.dropna(subset=["sue"])
    print(f"Dropped {before - len(df)} rows without valid SUE target")

    # Replace inf from pct_change divide-by-zero with NaN (will be imputed
    # or handled by the model — LightGBM/XGBoost handle NaN natively)
    df = df.replace([np.inf, -np.inf], np.nan)

    feature_cols = [
        "revenue_yoy_growth", "net_income_yoy_growth", "revenue_qoq_growth",
        "net_margin", "net_margin_yoy_change",
        "sue_lag1", "sue_lag2",
        "net_margin_sector_z", "revenue_growth_sector_z",
        "sector", "fiscal_year", "is_covid_era",
    ]
    print(f"\nFeature missingness:")
    print(df[feature_cols].isna().mean().sort_values(ascending=False))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows x {len(df.columns)} columns to {OUTPUT_PATH}")
    print(f"\nTarget (sue) distribution:")
    print(df["sue"].describe())


if __name__ == "__main__":
    main()
