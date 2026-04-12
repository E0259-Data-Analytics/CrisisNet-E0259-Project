from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent


def match_ticker(entity: str, names: list[str], tickers: list[str]) -> str | None:
    if not isinstance(entity, str) or not entity.strip():
        return None
    best = difflib.get_close_matches(entity, names, n=1, cutoff=0.5)
    if best:
        idx = names.index(best[0])
        return tickers[idx]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Map Chapter 11 labels to tickers via fuzzy match.")
    parser.add_argument("--labels", type=str, required=True, help="CSV with entity_name and file_date.")
    parser.add_argument(
        "--companies",
        type=str,
        default=str(REPO_ROOT / "crisisnet-data" / "data" / "company_list.csv"),
        help="Company list with ticker and company_name.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output parquet with ticker + distress_start.")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    companies = pd.read_csv(args.companies)
    if "company_name" not in companies.columns or "ticker" not in companies.columns:
        raise ValueError("company_list.csv must contain company_name and ticker columns")
    if "entity_name" not in labels.columns:
        raise ValueError("labels file must contain entity_name column")

    names = companies["company_name"].astype(str).tolist()
    tickers = companies["ticker"].astype(str).tolist()
    labels["ticker"] = labels["entity_name"].apply(lambda x: match_ticker(x, names, tickers))
    labels = labels.dropna(subset=["ticker"])

    if "file_date" in labels.columns:
        labels["distress_start"] = pd.to_datetime(labels["file_date"], errors="coerce")
    elif "date" in labels.columns:
        labels["distress_start"] = pd.to_datetime(labels["date"], errors="coerce")
    else:
        raise ValueError("labels file must contain file_date or date column")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels[["ticker", "distress_start"]].to_parquet(out_path, index=False)
    print(f"Wrote {len(labels)} mapped labels to {out_path}")


if __name__ == "__main__":
    main()
