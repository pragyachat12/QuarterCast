import { useState, useEffect, useMemo } from 'react'

const FEATURE_LABELS = {
  net_income_yoy_growth: 'Net income growth (YoY)',
  revenue_yoy_growth: 'Revenue growth (YoY)',
  revenue_qoq_growth: 'Revenue growth (QoQ)',
  net_margin: 'Net margin',
  net_margin_yoy_change: 'Margin change (YoY)',
  sue_lag1: 'Prior quarter surprise',
  sue_lag2: 'Surprise, 2 quarters ago',
  net_margin_sector_z: 'Margin vs sector peers',
  revenue_growth_sector_z: 'Growth vs sector peers',
  sector: 'Sector',
}

// Features that are true percentage growth/margin rates
const PERCENT_FEATURES = new Set([
  'revenue_yoy_growth', 'net_income_yoy_growth', 'revenue_qoq_growth',
  'net_margin', 'net_margin_yoy_change',
])

// Features that are already-standardized scores (SUE, z-scores) — show as
// plain decimals, never as a percentage
const SCORE_FEATURES = new Set([
  'sue_lag1', 'sue_lag2', 'net_margin_sector_z', 'revenue_growth_sector_z',
])

function formatRaw(feature, raw) {
  if (raw === null || raw === undefined) return '—'
  if (feature === 'sector') return raw
  if (typeof raw !== 'number') return raw
  if (PERCENT_FEATURES.has(feature)) return (raw * 100).toFixed(1) + '%'
  if (SCORE_FEATURES.has(feature)) return raw.toFixed(3)
  return raw.toFixed(3)
}

export default function Dashboard() {
  const [companies, setCompanies] = useState([])
  const [selected, setSelected] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sectorFilter, setSectorFilter] = useState('All')

  useEffect(() => {
    fetch('/api/companies')
      .then(r => r.json())
      .then(data => {
        setCompanies(data)
        if (data.length) setSelected(data[0])
      })
      .catch(() => setError('Could not load company list.'))
  }, [])

  useEffect(() => {
    if (!selected) return
    setLoading(true)
    setError(null)
    fetch(`/api/predict?ticker=${selected.ticker}&end=${selected.latest_end}`)
      .then(r => {
        if (!r.ok) throw new Error('Prediction failed')
        return r.json()
      })
      .then(setPrediction)
      .catch(() => setError('Could not load prediction for this company.'))
      .finally(() => setLoading(false))
  }, [selected])

  const sectors = useMemo(() => {
    const s = new Set(companies.map(c => c.sector))
    return ['All', ...Array.from(s).sort()]
  }, [companies])

  const filteredCompanies = useMemo(() => {
    if (sectorFilter === 'All') return companies
    return companies.filter(c => c.sector === sectorFilter)
  }, [companies, sectorFilter])

  const maxAbsValue = useMemo(() => {
    if (!prediction) return 1
    return Math.max(...prediction.contributions.map(c => Math.abs(c.value)), 0.01)
  }, [prediction])

  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="sidebar-filter">
          <select value={sectorFilter} onChange={e => setSectorFilter(e.target.value)}>
            {sectors.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className="company-list">
          {filteredCompanies.map(c => (
            <button
              key={c.ticker}
              className={`company-row ${selected?.ticker === c.ticker ? 'active' : ''}`}
              onClick={() => setSelected(c)}
            >
              <span className="ticker">{c.ticker}</span>
              <span className="sector-tag">{c.sector}</span>
            </button>
          ))}
          {companies.length === 0 && !error && (
            <div className="empty-state">Loading companies…</div>
          )}
        </div>
      </aside>

      <main className="main">
        {error && <div className="error-banner">{error}</div>}

        {loading && <div className="empty-state">Running model + computing feature contributions…</div>}

        {!loading && prediction && (
          <>
            <div className="prediction-summary">
              <div className="summary-block">
                <span className="summary-label">{prediction.ticker} · {prediction.sector}</span>
                <span className="summary-quarter">Quarter ending {prediction.end}</span>
              </div>
              <div className="summary-numbers">
                <div className="num-card">
                  <span className="num-label">Predicted SUE</span>
                  <span className={`num-value ${prediction.predicted_sue >= 0 ? 'pos' : 'neg'}`}>
                    {prediction.predicted_sue >= 0 ? '+' : ''}{prediction.predicted_sue.toFixed(2)}
                  </span>
                </div>
                <div className="num-card">
                  <span className="num-label">Actual SUE</span>
                  <span className={`num-value ${
                    prediction.actual_sue === null ? '' : prediction.actual_sue >= 0 ? 'pos' : 'neg'
                  }`}>
                    {prediction.actual_sue === null ? '—' :
                      `${prediction.actual_sue >= 0 ? '+' : ''}${prediction.actual_sue.toFixed(2)}`}
                  </span>
                </div>
                <div className="num-card">
                  <span className="num-label">Model baseline</span>
                  <span className="num-value muted">{prediction.base_value.toFixed(2)}</span>
                </div>
              </div>
            </div>

            <div className="ledger">
              <div className="ledger-header">
                <span>Feature</span>
                <span className="ledger-header-center">Contribution to prediction</span>
                <span className="ledger-header-right">Value</span>
              </div>
              {prediction.contributions.map(c => {
                const widthPct = (Math.abs(c.value) / maxAbsValue) * 50
                const isPositive = c.value >= 0
                return (
                  <div className="ledger-row" key={c.feature}>
                    <span className="ledger-feature">{FEATURE_LABELS[c.feature] || c.feature}</span>
                    <div className="ledger-bar-track">
                      <div className="ledger-zero-line" />
                      <div
                        className={`ledger-bar ${isPositive ? 'bar-pos' : 'bar-neg'}`}
                        style={{
                          width: `${widthPct}%`,
                          [isPositive ? 'left' : 'right']: '50%',
                        }}
                      />
                    </div>
                    <span className="ledger-raw">{formatRaw(c.feature, c.raw_value)}</span>
                  </div>
                )
              })}
            </div>
            <p className="footnote">
              Feature contributions computed live via ablation (each feature is set
              to its typical value, and the resulting shift in prediction is
              attributed to it). Bars extending right push the surprise estimate up
              (toward a beat); bars extending left push it down (toward a miss).
            </p>
          </>
        )}
      </main>
    </div>
  )
}
