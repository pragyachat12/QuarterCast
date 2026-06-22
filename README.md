# QuarterCast — Earnings Surprise Forecasting with Explainable ML

Predicting the magnitude of quarterly earnings surprises across 100+ S&P 500
companies (multi-sector), using fundamentals data from SEC EDGAR, gradient
boosting regression, and SHAP for interpretability.

## Motivation

Most earnings-surprise research relies on analyst consensus estimates, which
are proprietary and not freely available. This project uses a **Standardized
Unexpected Earnings (SUE)** approach instead — a well-established methodology
(Bernard & Thomas, 1989) that proxies "expected" earnings using a seasonal
random walk (same quarter, prior year) rather than analyst forecasts. This
keeps the entire pipeline reproducible from free public data (SEC EDGAR)
while still capturing genuine surprise — deviations from a company's own
seasonal earnings pattern.

## Research question

> Can fundamentals-based features predict the *magnitude* of a company's
> earnings surprise, and do the predictive drivers differ systematically
> across sectors?

## Method overview

1. **Universe**: 112 S&P 500 companies, sector-stratified (proportional to
   GICS sector representation) to avoid bias toward any one industry.
2. **Target**: SUE — surprise standardized by each company's trailing 2-year
   surprise volatility, computed from EDGAR XBRL data.
3. **Features**: revenue growth, margin trends, leverage ratios, prior
   surprise autocorrelation, sector indicators (sector-normalized where
   appropriate).
4. **Models**: Ridge regression baseline → LightGBM/XGBoost regressor.
5. **Evaluation**: time-aware train/val/test split (no random shuffling —
   split by fiscal period to avoid lookahead leakage). RMSE/MAE plus
   directional accuracy.
6. **Interpretability**: SHAP values — global feature importance and
   per-sector comparison, plus local explanations for notable predictions.

## Repo structure

```
data/         raw + processed datasets (EDGAR pulls, engineered features)
scripts/      data fetching, feature engineering, modeling pipeline
notebooks/    exploratory analysis, SHAP visualization
output/       trained models, prediction outputs, metrics
figures/      SHAP plots, model diagnostics, dashboard screenshots
```

## Known limitations

- **SUE is a proxy, not literal consensus surprise.** Analyst estimate data
  is proprietary; this is the standard academic workaround and is named
  explicitly here rather than presented as ground truth.
- **Survivorship bias**: universe drawn from current S&P 500 constituents,
  so delisted/failed companies are excluded.
- **Sector sample sizes vary** (e.g. Energy and Real Estate have fewer
  constituents); sector-level SHAP patterns for small sectors should be
  read with appropriate caution.

## Status

🚧 In progress — data pipeline and modeling stage.

## Author

Pragya Chaturvedi — built as part of independent research toward graduate
study applications (Fall 2027), extending prior work on fairness-aware ML
(COMPAS) and fundamentals data pipelines (SEC EDGAR / Yahoo Finance).
