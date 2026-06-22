"""
Run this LOCALLY (SEC EDGAR is blocked in Claude's sandbox).

Pulls quarterly EPS (diluted) history for every company in company_universe.csv
from SEC EDGAR's companyfacts API, then computes the SUE-based earnings
surprise target:

    Expected EPS_t  = Actual EPS_{t-4}   (same quarter, prior year)
    Surprise_t      = Actual EPS_t - Expected EPS_t
    SUE_t           = Surprise_t / std(Surprise_{t-1...t-8})   (rolling 2yr std)

Also pulls a handful of fundamentals features (Revenue, NetIncome, Assets,
Liabilities) for the same periods so we can build features later.

IMPORTANT: SEC EDGAR requires a descriptive User-Agent with your contact info,
or it will block you. Edit USER_AGENT below before running.

Usage:
    pip install requests pandas tqdm
    python fetch_edgar_data.py
"""

import requests
import pandas as pd
import time
import json
import os
from tqdm import tqdm

# ---- EDIT THIS before running ----
USER_AGENT = "username user@mail.example.com"
# -----------------------------------

HEADERS = {"User-Agent": USER_AGENT}
UNIVERSE_PATH = "data/company_universe.csv"
OUTPUT_PATH = "output/edgar_quarterly_eps.csv"
CACHE_DIR = "edgar_cache"  # raw JSON cache so re-runs don't re-hit EDGAR

# EDGAR uses 10-digit zero-padded CIK
def pad_cik(cik):
    return str(cik).zfill(10)

def fetch_company_facts(cik, ticker):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{pad_cik(cik)}.json"
    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"  [{ticker}] HTTP {resp.status_code} — skipping")
        return None

    data = resp.json()
    with open(cache_path, "w") as f:
        json.dump(data, f)
    time.sleep(0.15)  # stay under SEC's 10 req/sec limit
    return data

def extract_quarterly_concept(facts, concept, unit="USD"):
    """
    Pull a quarterly (10-Q derived) time series for a given XBRL concept,
    e.g. 'EarningsPerShareDiluted' or 'Revenues'.
    Returns list of dicts: {end_date, start_date, val, fy, fp, form}
    """
    try:
        node = facts["facts"]["us-gaap"][concept]["units"][unit]
    except KeyError:
        return []

    rows = []
    for item in node:
        # Only keep quarterly figures (3-month periods), from 10-Q/10-K
        if item.get("form") not in ("10-Q", "10-K"):
            continue
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        days = (end_dt - start_dt).days
        if not (75 <= days <= 100):  # roughly one quarter
            continue
        rows.append({
            "start": start, "end": end, "val": item.get("val"),
            "fy": item.get("fy"), "fp": item.get("fp"), "form": item.get("form")
        })
    return rows

def process_ticker(row):
    ticker, cik = row["Symbol"], row["CIK"]
    facts = fetch_company_facts(cik, ticker)
    if facts is None:
        return pd.DataFrame()

    eps_rows = extract_quarterly_concept(facts, "EarningsPerShareDiluted", unit="USD/shares")

    # Revenue reporting tag varies by company and changed industry-wide around
    # 2018 (ASC 606 adoption). Try tags in priority order and merge results,
    # since a single company may use different tags in different periods.
    REVENUE_TAGS = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
    ]
    rev_rows = []
    seen_ends = set()
    for tag in REVENUE_TAGS:
        tag_rows = extract_quarterly_concept(facts, tag, unit="USD")
        for r in tag_rows:
            if r["end"] not in seen_ends:
                rev_rows.append(r)
                seen_ends.add(r["end"])

    ni_rows = extract_quarterly_concept(facts, "NetIncomeLoss", unit="USD")
    assets_rows = extract_quarterly_concept(facts, "Assets", unit="USD")  # instant, handle separately
    liab_rows = extract_quarterly_concept(facts, "Liabilities", unit="USD")

    if not eps_rows:
        return pd.DataFrame()

    df = pd.DataFrame(eps_rows).rename(columns={"val": "eps"})
    df["end"] = pd.to_datetime(df["end"])
    df = df.drop_duplicates(subset="end").sort_values("end")

    # Merge revenue and net income on end date (best-effort, may have gaps)
    for extra_rows, colname in [(rev_rows, "revenue"), (ni_rows, "net_income")]:
        if extra_rows:
            extra_df = pd.DataFrame(extra_rows)[["end", "val"]].rename(columns={"val": colname})
            extra_df["end"] = pd.to_datetime(extra_df["end"])
            extra_df = extra_df.drop_duplicates(subset="end")
            df = df.merge(extra_df, on="end", how="left")
        else:
            df[colname] = None

    df["ticker"] = ticker
    df["cik"] = cik
    return df[["ticker", "cik", "end", "start", "eps", "revenue", "net_income", "fy", "fp", "form"]]

def main():
    universe = pd.read_csv(UNIVERSE_PATH)
    all_dfs = []

    print(f"Fetching EDGAR data for {len(universe)} companies...")
    for _, row in tqdm(universe.iterrows(), total=len(universe)):
        df = process_ticker(row)
        if not df.empty:
            all_dfs.append(df)

    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values(["ticker", "end"])
    print(f"\nPulled {len(full)} quarterly rows across {full['ticker'].nunique()} companies")

    # ---- Compute SUE-based surprise target ----
    full["eps_lag4"] = full.groupby("ticker")["eps"].shift(4)  # same quarter, prior year
    full["surprise"] = full["eps"] - full["eps_lag4"]

    # Rolling std of surprise over trailing 8 quarters (2 years), per ticker
    full["surprise_std"] = (
        full.groupby("ticker")["surprise"]
        .transform(lambda s: s.rolling(window=8, min_periods=4).std())
    )
    full["sue"] = full["surprise"] / full["surprise_std"]

    full.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Rows with valid SUE target: {full['sue'].notna().sum()} / {len(full)}")
    print(f"Date range: {full['end'].min()} to {full['end'].max()}")

if __name__ == "__main__":
    main()
