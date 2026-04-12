from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

DATA_ROOT = REPO_ROOT / "crisisnet-data"


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


def load_labels(path: Optional[str]) -> pd.DataFrame:
    if path is None:
        labels_path = DATA_ROOT / "splits" / "labels" / "distress_drawdowns" / "train.parquet"
        df = pd.read_parquet(labels_path)
    else:
        p = Path(path)
        if p.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
    if "distress_start" not in df.columns:
        if "file_date" in df.columns:
            df["distress_start"] = df["file_date"]
        elif "date" in df.columns:
            df["distress_start"] = df["date"]
    if "ticker" not in df.columns:
        raise ValueError("labels file must contain a 'ticker' column")
    df["distress_start"] = pd.to_datetime(df["distress_start"], errors="coerce")
    return df


def compute_auc(
    df: pd.DataFrame,
    distress: pd.DataFrame,
    score_col: str,
    lookahead_days: int,
) -> Tuple[float, int, int, int]:
    df = df.dropna(subset=[score_col, "ticker", "quarter"]).copy()
    df["quarter_start"] = df["quarter"].map(quarter_start_date)

    labels = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        q_start = row["quarter_start"]
        window_end = q_start + pd.Timedelta(days=lookahead_days)
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
        return float("nan"), len(df), len(pos), len(neg)
    auc = _auc_rank(pos, neg)
    return auc, len(df), len(pos), len(neg)


def compute_labels(
    df: pd.DataFrame,
    distress: pd.DataFrame,
    score_col: str,
    lookahead_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    df = df.dropna(subset=[score_col, "ticker", "quarter"]).copy()
    df["quarter_start"] = df["quarter"].map(quarter_start_date)

    labels = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        q_start = row["quarter_start"]
        window_end = q_start + pd.Timedelta(days=lookahead_days)
        dd = distress[distress["ticker"] == ticker]
        label = 0
        if not dd.empty:
            if ((dd["distress_start"] >= q_start) & (dd["distress_start"] <= window_end)).any():
                label = 1
        labels.append(label)
    return df[score_col].to_numpy(), np.array(labels, dtype=int)


def fbeta_from_counts(tp: int, fp: int, fn: int, beta: float = 2.0) -> float:
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


def best_fbeta(scores: np.ndarray, labels: np.ndarray, beta: float = 2.0) -> Tuple[float, float]:
    if scores.size == 0:
        return float("nan"), float("nan")
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    best_score = -1.0
    best_thresh = thresholds[0]
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        f = fbeta_from_counts(tp, fp, fn, beta=beta)
        if f > best_score:
            best_score = f
            best_thresh = float(t)
    return best_score, best_thresh


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NLP features vs distress labels.")
    parser.add_argument(
        "--features",
        type=str,
        default=str(ROOT / "results" / "X_nlp.parquet"),
    )
    parser.add_argument("--score-col", type=str, default=None)
    parser.add_argument("--lookahead-days", type=int, default=365)
    parser.add_argument("--labels", type=str, default=None, help="Optional labels file (parquet/csv).")
    parser.add_argument("--all-cols", action="store_true", help="Evaluate all numeric columns.")
    parser.add_argument(
        "--metric",
        type=str,
        default="f2",
        choices=["f2", "auc"],
        help="Primary metric to report.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    distress = load_labels(args.labels)

    if args.all_cols:
        drop_cols = {"ticker", "quarter", "label"}
        score_cols = [
            c for c in df.columns
            if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        for col in score_cols:
            scores, labels = compute_labels(df, distress, col, args.lookahead_days)
            n = len(labels)
            n_pos = int(labels.sum())
            n_neg = int(n - n_pos)
            if args.metric == "auc":
                auc, _, _, _ = compute_auc(df, distress, col, args.lookahead_days)
                if np.isnan(auc):
                    print(f"{col:45s} AUC=nan (pos={n_pos}, neg={n_neg}, n={n})")
                else:
                    print(f"{col:45s} AUC={auc:.3f} (pos={n_pos}, neg={n_neg}, n={n})")
            else:
                f2, thresh = best_fbeta(scores, labels, beta=2.0)
                if np.isnan(f2):
                    print(f"{col:45s} F2=nan (pos={n_pos}, neg={n_neg}, n={n})")
                else:
                    print(f"{col:45s} F2={f2:.3f} (thr={thresh:.4f}, pos={n_pos}, neg={n_neg}, n={n})")
        return

    score_col = pick_score_column(df, args.score_col)
    print(f"Score column: {score_col}")
    scores, labels = compute_labels(df, distress, score_col, args.lookahead_days)
    n = len(labels)
    n_pos = int(labels.sum())
    n_neg = int(n - n_pos)
    print(f"Samples: {n} | Positives: {n_pos} | Negatives: {n_neg}")
    if args.metric == "auc":
        auc, _, _, _ = compute_auc(df, distress, score_col, args.lookahead_days)
        if np.isnan(auc):
            print("Insufficient positives or negatives for AUC.")
            return
        print(f"AUC: {auc:.3f}")
    else:
        f2, thresh = best_fbeta(scores, labels, beta=2.0)
        if np.isnan(f2):
            print("Insufficient positives or negatives for F2.")
            return
        print(f"F2: {f2:.3f} (best threshold={thresh:.4f})")


if __name__ == "__main__":
    main()
