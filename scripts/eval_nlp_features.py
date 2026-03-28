from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_ROOT = Path("crisisnet-data")


def _auc_rank(pos: np.ndarray, neg: np.ndarray) -> float:
    values = np.concatenate([pos, neg])
    ranks = pd.Series(values).rank().to_numpy()
    n_pos = len(pos)
    n_neg = len(neg)
    rank_sum_pos = ranks[:n_pos].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg)) if n_pos * n_neg else float("nan")


def quarter_start_date(q: str) -> pd.Timestamp:
    year = int(q[:4])
    quarter = int(q[-1])
    month = 1 + (quarter - 1) * 3
    return pd.Timestamp(year=year, month=month, day=1)


def pick_score_column(df: pd.DataFrame, score_col: Optional[str]) -> str:
    if score_col:
        if score_col not in df.columns:
            raise ValueError(f"score column not found: {score_col}")
        return score_col
    for col in ["tenk_score_4q_mean", "calls_score_4q_mean", "tenk_score", "calls_score"]:
        if col in df.columns:
            return col
    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    if not topic_cols:
        raise ValueError("no suitable score column found in features")
    return topic_cols[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NLP features vs distress labels.")
    parser.add_argument(
        "--features",
        type=str,
        default=str(DATA_ROOT / "Module_2" / "features" / "X_nlp.parquet"),
    )
    parser.add_argument("--score-col", type=str, default=None)
    parser.add_argument("--lookahead-days", type=int, default=365)
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    score_col = pick_score_column(df, args.score_col)

    labels_path = DATA_ROOT / "splits" / "labels" / "distress_drawdowns" / "train.parquet"
    distress = pd.read_parquet(labels_path)
    distress["distress_start"] = pd.to_datetime(distress["distress_start"], errors="coerce")

    df = df.dropna(subset=[score_col, "ticker", "quarter"]).copy()
    df["quarter_start"] = df["quarter"].map(quarter_start_date)

    labels = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        q_start = row["quarter_start"]
        window_end = q_start + pd.Timedelta(days=args.lookahead_days)
        dd = distress[distress["ticker"] == ticker]
        label = 0
        if not dd.empty:
            if ((dd["distress_start"] >= q_start) & (dd["distress_start"] <= window_end)).any():
                label = 1
        labels.append(label)

    df["label"] = labels
    pos = df[df["label"] == 1][score_col].to_numpy()
    neg = df[df["label"] == 0][score_col].to_numpy()

    if len(pos) == 0 or len(neg) == 0:
        print("Insufficient positives or negatives for AUC.")
        return

    auc = _auc_rank(pos, neg)
    print(f"Score column: {score_col}")
    print(f"Samples: {len(df)} | Positives: {len(pos)} | Negatives: {len(neg)}")
    print(f"AUC: {auc:.3f}")


if __name__ == "__main__":
    main()
