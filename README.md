# QuarterCast — Earnings Surprise Forecasting with Explainable ML

Predicting the magnitude of quarterly earnings surprises across 110 S&P 500
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

1. **Universe**: 110 S&P 500 companies (112 sampled, 2 excluded — banks
   without standard revenue reporting), sector-stratified proportional to
   GICS sector representation to avoid bias toward any one industry.
2. **Target**: SUE — surprise standardized by each company's trailing 2-year
   surprise volatility, computed from EDGAR XBRL data (2009–2026). The
   standardization denominator is floored at its 5th percentile to prevent
   division-by-near-zero artifacts for companies with unusually stable
   surprise histories, and the result is winsorized at ±5.
3. **Features**: YoY/QoQ revenue growth, net margin and margin change,
   prior-quarter surprise (autocorrelation signal), sector-normalized
   z-scores for margin and growth, sector indicator.
4. **Models**: Ridge regression baseline → LightGBM regressor, trained with
   a **time-aware split** — train 2010–2019, validate 2020–2021 (includes
   COVID as a stress test), test 2022–2025. No random shuffling, to avoid
   lookahead leakage.
5. **Evaluation**: RMSE, MAE, and directional accuracy (did we get
   beat-vs-miss right, independent of magnitude error).
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
- **Revenue reporting tags vary across companies and time** (ASC 606
  adoption ~2018 changed XBRL tagging industry-wide); the fetch script
  falls back across multiple known tags, but 2 companies (banks, which
  report interest income rather than standard revenue) are excluded.
- **Survivorship bias**: universe drawn from current S&P 500 constituents,
  so delisted/failed companies are excluded.
- **Sector sample sizes vary** (e.g. Energy and Real Estate have fewer
  constituents); sector-level SHAP patterns for small sectors should be
  read with appropriate caution.

## Status

✅ Data pipeline, feature engineering, and modeling complete.
🚧 Next: SHAP interpretability analysis (global + per-sector + local
explanations).
