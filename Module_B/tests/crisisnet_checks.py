import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import numpy as np

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
DATA_ROOT = REPO_ROOT / "crisisnet-data"
LOG_PATH = MODULE_ROOT / "tests" / "analysis_log.txt"
JSONL_PATH = MODULE_ROOT / "tests" / "analysis_log.jsonl"


@dataclass
class LogRecord:
    metric: str
    value: object
    details: str = ""


def _write_log(records: List[LogRecord]) -> None:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n")
        for r in records:
            f.write(f"- {r.metric}: {r.value}"
                    + (f" | {r.details}" if r.details else "")
                    + "\n")
        f.write("\n")

    with JSONL_PATH.open("a", encoding="utf-8") as f:
        for r in records:
            payload = {
                "timestamp_utc": timestamp,
                "metric": r.metric,
                "value": r.value,
                "details": r.details,
            }
            f.write(json.dumps(payload, default=str) + "\n")


def _date_range(df: pd.DataFrame) -> Tuple[str, str]:
    idx = df.index
    return idx.min().strftime("%Y-%m-%d"), idx.max().strftime("%Y-%m-%d")


def summarize_splits() -> List[LogRecord]:
    records: List[LogRecord] = []

    for split in ["train", "validation", "test"]:
        prices_path = DATA_ROOT / "splits" / "stock_prices" / f"{split}.parquet"
        macro_path = DATA_ROOT / "splits" / "fred_macro" / f"{split}.parquet"

        prices = pd.read_parquet(prices_path)
        close = prices.xs("Close", axis=1, level=1)
        pr_min, pr_max = _date_range(prices)
        missing_close = close.isna().mean().mean()
        records.append(LogRecord(
            metric=f"prices_{split}_shape",
            value=str(prices.shape),
            details=f"dates {pr_min} to {pr_max}; tickers {close.shape[1]}; mean missing close {missing_close:.2%}"
        ))

        macro = pd.read_parquet(macro_path)
        mc_min, mc_max = _date_range(macro)
        missing_macro = macro.isna().mean().mean()
        records.append(LogRecord(
            metric=f"macro_{split}_shape",
            value=str(macro.shape),
            details=f"dates {mc_min} to {mc_max}; series {macro.shape[1]}; mean missing {missing_macro:.2%}"
        ))

    return records


def summarize_labels() -> List[LogRecord]:
    records: List[LogRecord] = []
    for split in ["train", "validation", "test"]:
        dd_path = DATA_ROOT / "splits" / "labels" / "distress_drawdowns" / f"{split}.parquet"
        ed_path = DATA_ROOT / "splits" / "labels" / "energy_defaults" / f"{split}.parquet"

        dd = pd.read_parquet(dd_path)
        ed = pd.read_parquet(ed_path)

        records.append(LogRecord(
            metric=f"distress_drawdowns_{split}_count",
            value=int(dd.shape[0]),
            details=f"unique tickers {dd['ticker'].nunique()}"
        ))
        records.append(LogRecord(
            metric=f"energy_defaults_{split}_count",
            value=int(ed.shape[0]),
            details=f"unique tickers {ed['ticker'].nunique()}"
        ))

    return records


def label_coverage() -> List[LogRecord]:
    records: List[LogRecord] = []
    prices = pd.read_parquet(DATA_ROOT / "splits" / "stock_prices" / "train.parquet")
    tickers = set(prices.columns.get_level_values(0))

    dd = pd.read_parquet(DATA_ROOT / "splits" / "labels" / "distress_drawdowns" / "train.parquet")
    ed = pd.read_parquet(DATA_ROOT / "splits" / "labels" / "energy_defaults" / "train.parquet")

    dd_tickers = {t for t in dd["ticker"].unique() if isinstance(t, str)}
    ed_tickers = {t for t in ed["ticker"].unique() if isinstance(t, str)}
    dd_missing = sorted(dd_tickers - tickers)
    ed_missing = sorted(ed_tickers - tickers)

    records.append(LogRecord(
        metric="label_coverage_distress_drawdowns_train",
        value=f"missing {len(dd_missing)}",
        details=", ".join(dd_missing) if dd_missing else "all present"
    ))
    records.append(LogRecord(
        metric="label_coverage_energy_defaults_train",
        value=f"missing {len(ed_missing)}",
        details=", ".join(ed_missing) if ed_missing else "all present"
    ))
    return records


def summarize_module2() -> List[LogRecord]:
    records: List[LogRecord] = []
    tenk_dir = DATA_ROOT / "Module_2" / "10k_extracted" / "10-K"
    tenk_files = list(tenk_dir.glob("*.json"))
    records.append(LogRecord(
        metric="module2_10k_extracted_files",
        value=len(tenk_files),
        details=f"path {tenk_dir}"
    ))

    meta_path = DATA_ROOT / "Module_2" / "filings_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "filing_date" in meta.columns:
            meta["filing_date"] = pd.to_datetime(meta["filing_date"], errors="coerce")
            mmin = meta["filing_date"].min()
            mmax = meta["filing_date"].max()
            dr = f"{mmin.date() if pd.notna(mmin) else 'NA'} to {mmax.date() if pd.notna(mmax) else 'NA'}"
        else:
            dr = "unknown"
        records.append(LogRecord(
            metric="module2_filings_metadata_rows",
            value=int(meta.shape[0]),
            details=f"cols {meta.shape[1]}; filing_date range {dr}"
        ))

    transcripts = DATA_ROOT / "Module_2" / "transcripts" / "huggingface" / "train.jsonl"
    if transcripts.exists():
        size_gb = transcripts.stat().st_size / (1024**3)
        records.append(LogRecord(
            metric="module2_transcripts_size_gb",
            value=f"{size_gb:.2f}",
            details=str(transcripts)
        ))
    return records


def summarize_module3() -> List[LogRecord]:
    records: List[LogRecord] = []
    edges_path = DATA_ROOT / "Module_3" / "edges_template.csv"
    disclosures_path = DATA_ROOT / "Module_3" / "customer_disclosures_raw.csv"

    if edges_path.exists():
        edges = pd.read_csv(edges_path)
        records.append(LogRecord(
            metric="module3_edges_count",
            value=int(edges.shape[0]),
            details=f"cols {edges.shape[1]}"
        ))
    if disclosures_path.exists():
        disc = pd.read_csv(disclosures_path)
        records.append(LogRecord(
            metric="module3_disclosures_count",
            value=int(disc.shape[0]),
            details=f"cols {disc.shape[1]}"
        ))
    return records


def _auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    # Mann–Whitney U statistic based AUC
    values = np.concatenate([pos, neg])
    ranks = pd.Series(values).rank().to_numpy()
    n_pos = len(pos)
    n_neg = len(neg)
    rank_sum_pos = ranks[:n_pos].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg)) if n_pos * n_neg else float("nan")


def simple_signal_test(random_seed: int = 42, negatives_per_event: int = 5) -> List[LogRecord]:
    records: List[LogRecord] = []
    prices = pd.read_parquet(DATA_ROOT / "splits" / "stock_prices" / "train.parquet")
    close = prices.xs("Close", axis=1, level=1).sort_index()
    log_ret = np.log(close).diff()
    vol30 = log_ret.rolling(30).std() * math.sqrt(252)

    rolling_max = close.rolling(30).max()
    dd = (close / rolling_max) - 1
    max_dd_30 = dd.rolling(30).min()

    distress = pd.read_parquet(DATA_ROOT / "splits" / "labels" / "distress_drawdowns" / "train.parquet")
    distress["distress_start"] = pd.to_datetime(distress["distress_start"], errors="coerce")
    distress["distress_end"] = pd.to_datetime(distress["distress_end"], errors="coerce")

    rng = random.Random(random_seed)

    pos_vol = []
    pos_dd = []
    neg_vol = []
    neg_dd = []

    for _, row in distress.iterrows():
        ticker = row["ticker"]
        start = row["distress_start"]
        end = row["distress_end"]
        if ticker not in close.columns:
            continue
        if pd.isna(start) or start not in close.index:
            continue
        v = vol30.at[start, ticker]
        d = max_dd_30.at[start, ticker]
        if pd.isna(v) or pd.isna(d):
            continue
        pos_vol.append(float(v))
        pos_dd.append(float(d))

        # Build negative pool: dates with valid features and outside distress window
        valid = vol30[ticker].notna() & max_dd_30[ticker].notna()
        if pd.notna(end):
            valid = valid & ~((valid.index >= start) & (valid.index <= end))
        candidates = valid[valid].index.tolist()
        if not candidates:
            continue
        k = min(negatives_per_event, len(candidates))
        sampled = rng.sample(candidates, k) if len(candidates) >= k else [rng.choice(candidates) for _ in range(k)]
        for dt in sampled:
            neg_vol.append(float(vol30.at[dt, ticker]))
            neg_dd.append(float(max_dd_30.at[dt, ticker]))

    pos_vol = np.array(pos_vol)
    pos_dd = np.array(pos_dd)
    neg_vol = np.array(neg_vol)
    neg_dd = np.array(neg_dd)

    def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) < 2 or len(b) < 2:
            return float("nan")
        s1 = a.var(ddof=1)
        s2 = b.var(ddof=1)
        pooled = math.sqrt(((len(a)-1)*s1 + (len(b)-1)*s2) / (len(a)+len(b)-2))
        return float((a.mean() - b.mean()) / pooled) if pooled else float("nan")

    records.append(LogRecord(
        metric="signal_test_pos_samples",
        value=int(len(pos_vol)),
        details=f"neg samples {len(neg_vol)}"
    ))

    if len(pos_vol) and len(neg_vol):
        auc_vol = _auc_rank(pos_vol, neg_vol)
        auc_dd = _auc_rank(-pos_dd, -neg_dd)  # more negative drawdown => higher risk
        records.append(LogRecord(
            metric="signal_vol30_auc",
            value=f"{auc_vol:.3f}",
            details=f"pos mean {pos_vol.mean():.4f}, neg mean {neg_vol.mean():.4f}, d {cohen_d(pos_vol, neg_vol):.3f}"
        ))
        records.append(LogRecord(
            metric="signal_maxdd30_auc",
            value=f"{auc_dd:.3f}",
            details=f"pos mean {pos_dd.mean():.4f}, neg mean {neg_dd.mean():.4f}, d {cohen_d(pos_dd, neg_dd):.3f}"
        ))
    return records


def nlp_risk_baseline(lookahead_days: int = 365) -> List[LogRecord]:
    records: List[LogRecord] = []

    tenk_dir = DATA_ROOT / "Module_2" / "10k_extracted" / "10-K"
    meta_path = DATA_ROOT / "Module_2" / "filings_metadata.csv"
    map_path = DATA_ROOT / "Module_A" / "sec_xbrl" / "ticker_cik_mapping.csv"
    labels_path = DATA_ROOT / "splits" / "labels" / "distress_drawdowns" / "train.parquet"

    if not (tenk_dir.exists() and meta_path.exists() and map_path.exists() and labels_path.exists()):
        return records

    meta = pd.read_csv(meta_path)
    meta["CIK"] = pd.to_numeric(meta["CIK"], errors="coerce")
    meta["filing_date"] = pd.to_datetime(meta["Filing Date"], errors="coerce")

    cik_map = pd.read_csv(map_path)
    cik_map["cik"] = pd.to_numeric(cik_map["cik"], errors="coerce")
    cik_to_ticker = {int(c): t for t, c in zip(cik_map["ticker"], cik_map["cik"]) if pd.notna(c)}

    distress = pd.read_parquet(labels_path)
    distress["distress_start"] = pd.to_datetime(distress["distress_start"], errors="coerce")
    distress["distress_end"] = pd.to_datetime(distress["distress_end"], errors="coerce")

    keyword_list = [
        "bankrupt", "bankruptcy", "liquidity", "default", "covenant",
        "distress", "going concern", "impairment", "restructuring",
        "material weakness", "going-concern", "delist", "chapter 11",
        "credit downgrade", "downgrade", "solvency", "insolvency"
    ]
    neg_words = [
        "adverse", "decline", "deteriorate", "loss", "losses", "liability",
        "litigation", "risk", "uncertain", "uncertainty", "negative"
    ]

    rows = []
    for fp in tenk_dir.glob("*.json"):
        name = fp.stem
        parts = name.split("_")
        if len(parts) < 3:
            continue
        cik = pd.to_numeric(parts[0], errors="coerce")
        year = pd.to_numeric(parts[2], errors="coerce")
        if pd.isna(cik) or pd.isna(year):
            continue
        cik = int(cik)
        year = int(year)
        ticker = cik_to_ticker.get(cik)
        if not ticker:
            continue

        meta_subset = meta[(meta["CIK"] == cik) & (meta["filing_date"].dt.year == year)]
        filing_date = meta_subset["filing_date"].min()
        if pd.isna(filing_date):
            continue

        try:
            payload = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        text = payload.get("item_1A") or payload.get("item_1a") or ""
        if not isinstance(text, str):
            continue
        lowered = text.lower()
        words = lowered.split()
        word_count = len(words)
        if word_count == 0:
            continue

        kw_hits = sum(lowered.count(k) for k in keyword_list)
        neg_hits = sum(lowered.count(k) for k in neg_words)

        label = 0
        dd = distress[distress["ticker"] == ticker]
        if not dd.empty:
            window_end = filing_date + pd.Timedelta(days=lookahead_days)
            if ((dd["distress_start"] >= filing_date) & (dd["distress_start"] <= window_end)).any():
                label = 1

        rows.append({
            "ticker": ticker,
            "cik": cik,
            "year": year,
            "filing_date": filing_date,
            "word_count": word_count,
            "keyword_rate": kw_hits / word_count,
            "neg_rate": neg_hits / word_count,
            "label": label,
        })

    if not rows:
        return records

    df = pd.DataFrame(rows)
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]

    records.append(LogRecord(
        metric="nlp_baseline_samples",
        value=int(df.shape[0]),
        details=f"positives {int(pos.shape[0])}, negatives {int(neg.shape[0])}"
    ))

    if len(pos) > 0 and len(neg) > 0:
        auc_kw = _auc_rank(pos["keyword_rate"].to_numpy(), neg["keyword_rate"].to_numpy())
        auc_neg = _auc_rank(pos["neg_rate"].to_numpy(), neg["neg_rate"].to_numpy())
        records.append(LogRecord(
            metric="nlp_keyword_rate_auc",
            value=f"{auc_kw:.3f}",
            details=f"pos mean {pos['keyword_rate'].mean():.6f}, neg mean {neg['keyword_rate'].mean():.6f}"
        ))
        records.append(LogRecord(
            metric="nlp_neg_rate_auc",
            value=f"{auc_neg:.3f}",
            details=f"pos mean {pos['neg_rate'].mean():.6f}, neg mean {neg['neg_rate'].mean():.6f}"
        ))

    return records


def walk_forward_validation() -> List[LogRecord]:
    records: List[LogRecord] = []
    prices = pd.read_parquet(DATA_ROOT / "splits" / "stock_prices" / "train.parquet")
    close = prices.xs("Close", axis=1, level=1).sort_index()
    log_ret = np.log(close).diff()
    vol30 = log_ret.rolling(30).std() * math.sqrt(252)
    rolling_max = close.rolling(30).max()
    dd30 = (close / rolling_max) - 1
    maxdd30 = dd30.rolling(30).min()

    distress = pd.read_parquet(DATA_ROOT / "splits" / "labels" / "distress_drawdowns" / "train.parquet")
    distress["distress_start"] = pd.to_datetime(distress["distress_start"], errors="coerce")
    distress["distress_end"] = pd.to_datetime(distress["distress_end"], errors="coerce")

    # Build daily labels per ticker for distress windows
    labels = pd.DataFrame(False, index=close.index, columns=close.columns)
    for _, row in distress.iterrows():
        t = row["ticker"]
        s = row["distress_start"]
        e = row["distress_end"]
        if t not in labels.columns or pd.isna(s) or pd.isna(e):
            continue
        mask = (labels.index >= s) & (labels.index <= e)
        labels.loc[mask, t] = True

    # Risk score: z(vol30) + z(-maxdd30) within each ticker
    z_vol = (vol30 - vol30.mean()) / vol30.std(ddof=0)
    z_dd = (-maxdd30 - (-maxdd30).mean()) / (-maxdd30).std(ddof=0)
    score = z_vol + z_dd

    folds = [
        ("2015-01-02", "2018-12-31", "2019-01-01", "2019-12-31"),
        ("2015-01-02", "2019-12-31", "2020-01-01", "2020-12-31"),
        ("2015-01-02", "2020-12-31", "2021-01-01", "2021-12-31"),
    ]

    for i, (tr_start, tr_end, va_start, va_end) in enumerate(folds, start=1):
        train_mask = (score.index >= tr_start) & (score.index <= tr_end)
        val_mask = (score.index >= va_start) & (score.index <= va_end)

        s_val = score[val_mask].stack()
        y_val = labels[val_mask].stack()

        df_val = pd.DataFrame({"score": s_val, "label": y_val}).dropna()
        pos = df_val[df_val["label"]]["score"]
        neg = df_val[~df_val["label"]]["score"]

        if len(pos) == 0 or len(neg) == 0:
            records.append(LogRecord(
                metric=f"walk_forward_fold_{i}_auc",
                value="nan",
                details="insufficient positives or negatives"
            ))
            continue

        auc = _auc_rank(pos.to_numpy(), neg.to_numpy())
        records.append(LogRecord(
            metric=f"walk_forward_fold_{i}_auc",
            value=f"{auc:.3f}",
            details=f"val positives {len(pos)}, negatives {len(neg)}; val period {va_start} to {va_end}"
        ))

    return records


def run_all() -> None:
    all_records: List[LogRecord] = []
    all_records += summarize_splits()
    all_records += summarize_labels()
    all_records += label_coverage()
    all_records += summarize_module2()
    all_records += summarize_module3()
    all_records += simple_signal_test()
    all_records += nlp_risk_baseline()
    all_records += walk_forward_validation()
    _write_log(all_records)


if __name__ == "__main__":
    run_all()
