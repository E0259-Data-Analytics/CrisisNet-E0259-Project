#!/usr/bin/env python3
"""Download SEC EDGAR companyfacts needed for Altman Z-score coverage."""

import json
import time
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "crisisnet-data"
COMPANY_LIST = DATA_ROOT / "data" / "company_list.csv"
OUT_DIR = DATA_ROOT / "Module_A" / "sec_xbrl"
FACTS_DIR = OUT_DIR / "company_facts"
SUBMISSIONS_DIR = OUT_DIR / "submissions"
HEADERS = {"User-Agent": "CrisisNet-IISc research@iisc.ac.in"}


def main():
    companies = pd.read_csv(COMPANY_LIST)
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Fetching SEC ticker mapping...")
    cik_data = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=HEADERS,
        timeout=30,
    ).json()
    ticker_to_cik = {
        entry["ticker"].upper(): str(entry["cik_str"]).zfill(10)
        for entry in cik_data.values()
    }

    if "cik" in companies.columns:
        for _, row in companies.iterrows():
            ticker = str(row["ticker"]).upper()
            if ticker in ticker_to_cik or pd.isna(row.get("cik")):
                continue
            try:
                ticker_to_cik[ticker] = str(int(float(row["cik"]))).zfill(10)
            except Exception:
                pass

    pd.DataFrame([
        {"ticker": t, "cik": ticker_to_cik.get(str(t).upper(), "NOT_FOUND")}
        for t in companies["ticker"]
    ]).to_csv(OUT_DIR / "ticker_cik_mapping.csv", index=False)

    print("[2/3] Downloading companyfacts...")
    success = 0
    for ticker in companies["ticker"]:
        ticker = str(ticker).upper()
        cik = ticker_to_cik.get(ticker)
        if not cik:
            print(f"  {ticker}: no CIK")
            continue
        try:
            url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            with open(FACTS_DIR / f"{ticker}_facts.json", "w") as f:
                json.dump(r.json(), f)
            success += 1
            time.sleep(0.15)
        except Exception as exc:
            print(f"  {ticker}: failed companyfacts ({exc})")
            time.sleep(0.5)

    print("[3/3] Downloading submissions...")
    for ticker in companies["ticker"]:
        ticker = str(ticker).upper()
        cik = ticker_to_cik.get(ticker)
        if not cik:
            continue
        try:
            url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            with open(SUBMISSIONS_DIR / f"{ticker}_submissions.json", "w") as f:
                json.dump(r.json(), f)
            time.sleep(0.15)
        except Exception as exc:
            print(f"  {ticker}: failed submissions ({exc})")
            time.sleep(0.5)

    print(f"Downloaded companyfacts for {success}/{len(companies)} tickers")


if __name__ == "__main__":
    main()
