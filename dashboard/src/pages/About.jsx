export default function About() {
  return (
    <main className="main page-content">
      <section className="hero">
        <h1>Predicting earnings surprises, and explaining why</h1>
        <p className="hero-sub">
          QuarterCast forecasts the magnitude of quarterly earnings surprises across
          110 S&amp;P 500 companies using only public financial filings — and shows
          exactly which fundamentals drove each prediction.
        </p>
      </section>

      <section className="content-block">
        <h2>The problem</h2>
        <p>
          Most earnings-surprise research compares a company's actual results to
          Wall Street analyst consensus estimates. That estimate data is proprietary
          and expensive — out of reach for an independent or academic project.
        </p>
      </section>

      <section className="content-block">
        <h2>The approach</h2>
        <p>
          Instead, QuarterCast uses <strong>Standardized Unexpected Earnings (SUE)</strong> —
          a well-established methodology (Bernard &amp; Thomas, 1989) that proxies
          "expected" earnings using a seasonal random walk: a company's own results
          from the same quarter, one year prior. The surprise is then standardized
          by that company's trailing volatility, so a $0.10 surprise means something
          different for a stable utility than for a volatile growth stock.
        </p>
        <p>
          This keeps the entire pipeline reproducible from free public data
          (SEC EDGAR XBRL filings) while still capturing genuine deviations from a
          company's seasonal earnings pattern.
        </p>
      </section>

      <section className="content-block">
        <h2>The model</h2>
        <p>
          A gradient-boosting regression model is trained on fundamentals —
          revenue growth, margin trends, prior-surprise history, and how a company
          compares to its sector peers — using a strict time-aware split (train on
          2010–2019, validate on 2020–2021 including the COVID shock, test on
          2022–2025) to avoid lookahead leakage.
        </p>
      </section>

      <section className="content-block">
        <h2>The explanation layer</h2>
        <p>
          Every live prediction on the <a href="/dashboard">Dashboard</a> comes with
          a feature-by-feature breakdown of what pushed the estimate up or down,
          computed via ablation: each feature is set to its typical (median) value
          one at a time, and the resulting shift in prediction is attributed to that
          feature. It's a simpler, dependency-light cousin of SHAP — same goal,
          no black box.
        </p>
      </section>

      <section className="content-block limitations">
        <h2>Known limitations</h2>
        <ul>
          <li>
            SUE is a proxy for surprise, not literal consensus surprise — named
            explicitly here rather than presented as ground truth.
          </li>
          <li>
            Universe is drawn from current S&amp;P 500 constituents, so delisted or
            failed companies are excluded (survivorship bias).
          </li>
          <li>
            Sector sample sizes vary — Energy and Real Estate have fewer
            constituents, so sector-level patterns there carry more uncertainty.
          </li>
        </ul>
      </section>
    </main>
  )
}
