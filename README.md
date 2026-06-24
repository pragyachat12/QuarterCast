# QuarterCast — predicting earnings surprises, and explaining why

This project forecasts the magnitude of quarterly earnings surprises across 110 S&P 500 companies, using nothing but public SEC filings — and explains exactly what drove each prediction, rather than treating the model as a black box.

## Motivation

Most earnings-surprise research relies on analyst consensus estimates — i.e. "what Wall Street expected." That data is proprietary and expensive, which puts it out of reach for an independent project.

Instead, this uses **Standardized Unexpected Earnings (SUE)**, a well-established methodology from Bernard & Thomas (1989). The idea: proxy "expected" earnings using a seasonal random walk — a company's own results from the same quarter, one year prior — rather than relying on analyst forecasts. Most companies have fairly predictable seasonal patterns, so comparing a quarter to its year-ago counterpart is a reasonable, defensible stand-in for "did this beat expectations." It also keeps the entire pipeline reproducible from free public data (SEC EDGAR), with no paywalled dependencies.

## Research question

Can fundamentals-based features predict the *magnitude* of a company's earnings surprise — and do the drivers of that prediction differ systematically across sectors?

## Method overview

1. **Universe**: 110 S&P 500 companies (sampled 112, excluded 2 — banks report interest income rather than standard revenue, which broke the pipeline's assumptions). Sampled proportionally across all 11 GICS sectors to avoid skewing toward any one industry.
2. **Target**: SUE — surprise standardized by each company's trailing 2-year surprise volatility. The denominator is floored at its 5th percentile to prevent a handful of unusually stable companies from producing artificially extreme values, and the result is winsorized at ±5.
3. **Features**: YoY/QoQ revenue growth, net margin and margin change, prior-quarter surprise (which turns out to carry real autocorrelation signal), and sector-normalized z-scores for margin and growth.
4. **Models**: a Ridge regression baseline, then a LightGBM regressor, trained with a strict time-aware split — train on 2010–2019, validate on 2020–2021 (deliberately including COVID as a stress test), test on 2022–2025. No random shuffling, to avoid lookahead leakage.
5. **Evaluation**: RMSE and MAE for magnitude error, plus directional accuracy — did the model get beat-vs-miss right, independent of how close the magnitude was.
6. **Interpretability**: SHAP values for global feature importance, a per-sector comparison, and local explanations for individual predictions.

## Repo structure

```
data/         raw + processed datasets (EDGAR pulls, engineered features)
scripts/      data fetching, feature engineering, modeling pipeline
notebooks/    exploratory analysis, SHAP visualization
output/       trained models, prediction outputs, metrics
figures/      SHAP plots, model diagnostics, dashboard screenshots
```

## Known limitations

- **SUE is a proxy, not literal consensus surprise.** Analyst estimate data is proprietary; this is a standard academic workaround, named explicitly here rather than presented as ground truth.
- **Revenue reporting tags are inconsistent across companies and time** (a 2018 accounting rule change shifted how revenue gets tagged industry-wide). The fetch script falls back across several known tags, but 2 companies — both banks — still fell through.
- **Survivorship bias**: the universe is drawn from current S&P 500 constituents, so delisted or failed companies aren't represented.
- **Sector sample sizes vary** — Energy and Real Estate have noticeably fewer constituents, so sector-level patterns there should be read with appropriate caution.

## Status

Data pipeline, feature engineering, and modeling complete.
Next up: the full SHAP interpretability pass 
