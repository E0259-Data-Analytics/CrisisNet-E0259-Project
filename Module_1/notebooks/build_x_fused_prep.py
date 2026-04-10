"""
CrisisNet Module A — Alignment Script (A1)
==========================================
Post-hoc preparation of X_ts.parquet for Module D fusion.

This script does NOT re-train any models. It reads the existing
X_ts.parquet, adds the quarter string column required for merge
alignment with X_nlp and X_graph, and re-exports.

Usage:
    cd Module_1/notebooks
    python build_x_fused_prep.py

Output:
    Module_1/results/X_ts.parquet  (updated in-place with 'quarter' column)
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT   = Path(__file__).resolve().parent.parent   # Module_1/
REPO_ROOT      = PROJECT_ROOT.parent                      # CrisisNet-E0259-Project/
CRISISNET_DATA = REPO_ROOT / "crisisnet-data"

X_TS_PATH = PROJECT_ROOT / "results" / "X_ts.parquet"

# ── A1: Add/fix quarter string column ────────────────────────────────
# X_ts uses Timestamp index (e.g. 2020-06-30). Modules B and C use
# string quarters ('2020Q2'). The fusion merge requires a common key.
# Always re-derive from Date — guards against the pipeline bug where
# 'quarter' was in feature_cols and got coerced to NaN→0 by to_numeric.
X_ts = pd.read_parquet(X_TS_PATH).reset_index()
X_ts['quarter'] = pd.to_datetime(X_ts['Date']).apply(
    lambda d: f"{d.year}Q{(d.month - 1) // 3 + 1}"
)

# ── A3: Replace inline distress_label with unified canonical labels ────
# Drop existing label (inline-computed) and merge the authoritative one
# from label_unified.parquet so all modules share the same binary signal.
labels = pd.read_parquet(CRISISNET_DATA / "data" / "label_unified.parquet")
X_ts = X_ts.drop(columns=['distress_label'], errors='ignore')
X_ts = X_ts.merge(
    labels[['ticker', 'quarter', 'distress_label']],
    on=['ticker', 'quarter'],
    how='left'
)
X_ts['distress_label'] = X_ts['distress_label'].fillna(0).astype(int)

X_ts.to_parquet(X_TS_PATH, index=False)
print(f"X_ts updated: {X_ts.shape[0]} rows × {X_ts.shape[1]} columns")
print(f"  'quarter' column added  ✓")
print(f"  'distress_label' merged from label_unified.parquet  ✓")
print(f"  Saved to: {X_TS_PATH}")
