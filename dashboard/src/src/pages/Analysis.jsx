const GLOBAL_IMPORTANCE = [
  { feature: 'Margin change (YoY)', value: 0.564 },
  { feature: 'Net income growth (YoY)', value: 0.245 },
  { feature: 'Prior quarter surprise', value: 0.068 },
  { feature: 'Net margin', value: 0.040 },
  { feature: 'Sector', value: 0.030 },
  { feature: 'Surprise, 2 quarters ago', value: 0.028 },
  { feature: 'Revenue growth (YoY)', value: 0.019 },
  { feature: 'Revenue growth (QoQ)', value: 0.014 },
  { feature: 'Growth vs sector peers', value: 0.008 },
  { feature: 'Margin vs sector peers', value: -0.002 },
]

const SECTOR_PERFORMANCE = [
  { sector: 'Consumer Discretionary', n: 119, rmse: 0.386, dirAcc: 0.950 },
  { sector: 'Materials', n: 74, rmse: 0.609, dirAcc: 0.946 },
  { sector: 'Consumer Staples', n: 84, rmse: 0.551, dirAcc: 0.929 },
  { sector: 'Health Care', n: 164, rmse: 0.528, dirAcc: 0.915 },
  { sector: 'Industrials', n: 205, rmse: 0.768, dirAcc: 0.893 },
  { sector: 'Financials', n: 186, rmse: 0.636, dirAcc: 0.887 },
  { sector: 'Information Technology', n: 183, rmse: 0.813, dirAcc: 0.874 },
  { sector: 'Energy', n: 50, rmse: 0.835, dirAcc: 0.840 },
  { sector: 'Utilities', n: 65, rmse: 0.648, dirAcc: 0.831 },
  { sector: 'Communication Services', n: 48, rmse: 0.869, dirAcc: 0.792 },
  { sector: 'Real Estate', n: 72, rmse: 0.846, dirAcc: 0.708 },
]

const MODEL_COMPARISON = [
  { model: 'Naive (always 0)', rmse: 1.087, mae: 0.820, dirAcc: 0.008 },
  { model: 'Ridge (baseline)', rmse: 0.909, mae: 0.702, dirAcc: 0.686 },
  { model: 'Gradient boosting', rmse: 0.676, mae: 0.494, dirAcc: 0.891 },
]

function Bar({ value, max, positive }) {
  const widthPct = (Math.abs(value) / max) * 100
  return (
    <div className="analysis-bar-track">
      <div
        className={`analysis-bar ${positive ? 'bar-pos' : 'bar-neg'}`}
        style={{ width: `${widthPct}%` }}
      />
    </div>
  )
}

export default function Analysis() {
  const maxImportance = Math.max(...GLOBAL_IMPORTANCE.map(d => Math.abs(d.value)))
  const maxDirAcc = 1

  return (
    <main className="main page-content">
      <section className="hero hero-compact">
        <h1>What the model actually learned</h1>
        <p className="hero-sub">
          Results on held-out 2022–2025 data, never seen during training.
        </p>
      </section>

      <section className="content-block">
        <h2>Model comparison</h2>
        <p className="section-note">
          Directional accuracy = did the model get beat-vs-miss right, independent of
          magnitude error.
        </p>
        <div className="comparison-table">
          <div className="comparison-row comparison-header">
            <span>Model</span><span>RMSE</span><span>MAE</span><span>Directional Acc.</span>
          </div>
          {MODEL_COMPARISON.map(m => (
            <div className="comparison-row" key={m.model}>
              <span>{m.model}</span>
              <span className="mono">{m.rmse.toFixed(3)}</span>
              <span className="mono">{m.mae.toFixed(3)}</span>
              <span className="mono">{(m.dirAcc * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </section>

      <section className="content-block">
        <h2>Global feature importance</h2>
        <p className="section-note">
          Permutation importance on the test set — how much model error increases
          when a feature's values are shuffled. Margin change and net income growth
          dominate; sector-relative features contribute least.
        </p>
        <div className="ledger">
          {GLOBAL_IMPORTANCE.map(d => (
            <div className="ledger-row analysis-row" key={d.feature}>
              <span className="ledger-feature">{d.feature}</span>
              <Bar value={d.value} max={maxImportance} positive={d.value >= 0} />
              <span className="ledger-raw">{d.value.toFixed(3)}</span>
            </div>
          ))}
        </div>
      </section>

      <section className="content-block">
        <h2>Performance by sector</h2>
        <p className="section-note">
          The cross-sector angle: prediction quality varies meaningfully by
          industry. Consumer Discretionary and Materials predict cleanly;
          Real Estate and Communication Services are harder — plausibly because
          their earnings are driven by factors (lease accounting, content cycles)
          not well captured by these fundamentals.
        </p>
        <div className="ledger">
          {SECTOR_PERFORMANCE.map(s => (
            <div className="ledger-row analysis-row" key={s.sector}>
              <span className="ledger-feature">{s.sector} <span className="n-tag">n={s.n}</span></span>
              <Bar value={s.dirAcc} max={maxDirAcc} positive={true} />
              <span className="ledger-raw">{(s.dirAcc * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </section>
    </main>
  )
}
