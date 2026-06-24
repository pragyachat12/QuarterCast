import { useState, useEffect, useMemo } from 'react'
import './App.css'

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

function formatRaw(feature, raw) {
  if (raw === null || raw === undefined) return '—'
  if (feature === 'sector') return raw
  if (typeof raw === 'number') {
    if (feature.includes('growth') || feature.includes('margin') && feature !== 'net_margin_sector_z') {
      return (raw * 100).toFixed(1) + '%'
    }
    return raw.toFixed(3)
  }
  return raw
}

export default function App() {
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

  const maxAbsShap = useMemo(() => {
    if (!prediction) return 1
    return Math.max(...prediction.contributions.map(c => Math.abs(c.value)), 0.01)
  }, [prediction])

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <span className="logo">QuarterCast</span>
          <span className="header-sub">Live earnings-surprise predictions, explained</span>
        </div>
        <div className="header-right">
          <span className="legend-item"><span className="dot dot-green" /> beat</span>
          <span className="legend-item"><span className="dot dot-coral" /> miss</span>
        </div>
      </header>

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

          {loading && <div className="empty-state">Running model + computing SHAP values…</div>}

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
                  const widthPct = (Math.abs(c.value) / maxAbsShap) * 50
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
                SHAP values computed live for this prediction. Bars extending right push the
                surprise estimate up (toward a beat); bars extending left push it down (toward a miss).
              </p>
            </>
          )}
        </main>
      </div>
    </div>
  )
}
