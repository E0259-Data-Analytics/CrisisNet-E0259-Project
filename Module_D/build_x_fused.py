"""
CrisisNet Module D — Step 1: Feature Alignment & Fusion
=========================================================
Merges X_ts (Module A), X_graph (Module C), and optionally X_nlp (Module B)
into a single X_fused.parquet ready for LightGBM training.

Module B integration (single-line change):
    Currently X_nlp is skipped if the file is absent.
    To wire in Module B, uncomment the X_NLP_PATH line below — that is the
    only change needed once Module B produces X_nlp_finbert.parquet.

Usage:
    cd CrisisNet-E0259-Project
    python Module_D/build_x_fused.py

Output:
    Module_D/X_fused.parquet   (~1500 rows x ~155 features)
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parent.parent
MODULE_D    = REPO_ROOT / "Module_D"

X_TS_PATH   = REPO_ROOT / "Module_1" / "results"  / "X_ts.parquet"
X_GRAPH_PATH= REPO_ROOT / "Module_C" / "results"  / "exports" / "X_graph.parquet"
LABELS_PATH = REPO_ROOT / "crisisnet-data" / "data" / "label_unified.parquet"

# ── Module B single-line toggle ────────────────────────────────────────────────
# Uncomment the line below to activate Module B NLP features:
# X_NLP_PATH = REPO_ROOT / "Module_B" / "results" / "X_nlp_finbert.parquet"
X_NLP_PATH  = None   # Module B not yet merged — remove this line when B is ready

OUT_PATH    = MODULE_D / "X_fused.parquet"

# ── Columns never treated as features ─────────────────────────────────────────
SKIP = {'ticker', 'quarter', 'Date', 'distress_label', 'year'}

# ── 1. Load X_ts ──────────────────────────────────────────────────────────────
print("[1/5] Loading X_ts (Module A)…")
X_ts = pd.read_parquet(X_TS_PATH).reset_index(drop=True)

# Re-derive quarter from Date — guards against any lingering coercion bug
X_ts['quarter'] = pd.to_datetime(X_ts['Date']).apply(
    lambda d: f"{d.year}Q{(d.month - 1) // 3 + 1}"
)
print(f"      X_ts:   {X_ts.shape}  |  tickers: {X_ts['ticker'].nunique()}"
      f"  |  quarters: {X_ts['quarter'].nunique()}")

# ── 2. Load X_graph ────────────────────────────────────────────────────────────
print("[2/5] Loading X_graph (Module C)…")
X_graph_raw = pd.read_parquet(X_GRAPH_PATH)

# Columns to skip when prefixing (metadata / merge keys)
g_skip = {'ticker', 'quarter', 'year', 'name', 'subsector', 'defaulted'}
X_graph = X_graph_raw.rename(
    columns={c: f"graph_{c}" for c in X_graph_raw.columns if c not in g_skip}
)
# Drop non-numeric metadata columns
X_graph = X_graph.drop(
    columns=[c for c in ['name', 'subsector', 'defaulted', 'year']
             if c in X_graph.columns],
    errors='ignore'
)
print(f"      X_graph: {X_graph.shape}  |  tickers: {X_graph['ticker'].nunique()}"
      f"  |  quarters: {X_graph['quarter'].nunique()}")

# ── 3. Load X_nlp (Module B — optional) ───────────────────────────────────────
X_nlp = None
if X_NLP_PATH is not None:
    print("[3/5] Loading X_nlp (Module B)…")
    X_nlp_raw = pd.read_parquet(X_NLP_PATH)
    X_nlp = X_nlp_raw.rename(
        columns={c: f"nlp_{c}" for c in X_nlp_raw.columns if c not in SKIP}
    )
    print(f"      X_nlp:  {X_nlp.shape}  |  tickers: {X_nlp['ticker'].nunique()}")
else:
    print("[3/5] X_nlp skipped — Module B not yet integrated (expected)")

# ── 4. Merge on (ticker, quarter) ─────────────────────────────────────────────
print("[4/5] Merging feature matrices…")

# Start from X_ts time-series features
ts_cols = [c for c in X_ts.columns if c not in SKIP]
X_fused = X_ts[['ticker', 'quarter', 'Date'] + ts_cols].copy()

# Left-join graph features
X_fused = X_fused.merge(X_graph, on=['ticker', 'quarter'], how='left')

# Left-join NLP features (Module B — will be active after single-line change)
if X_nlp is not None:
    X_fused = X_fused.merge(X_nlp, on=['ticker', 'quarter'], how='left')

# Attach canonical labels (overwrite any existing distress_label)
labels = pd.read_parquet(LABELS_PATH)[['ticker', 'quarter', 'distress_label']]
X_fused = X_fused.drop(columns=['distress_label'], errors='ignore')
X_fused = X_fused.merge(labels, on=['ticker', 'quarter'], how='left')
X_fused['distress_label'] = X_fused['distress_label'].fillna(0).astype(int)

# Derive year integer for temporal splits
X_fused['year'] = pd.to_datetime(X_fused['Date']).dt.year

# ── 5. Handle missing values ───────────────────────────────────────────────────
feat_cols = [c for c in X_fused.columns
             if c not in {'ticker', 'quarter', 'Date', 'distress_label', 'year'}]
X_fused[feat_cols] = X_fused[feat_cols].fillna(0)
X_fused['distress_label'] = X_fused['distress_label'].fillna(0).astype(int)

# ── 6. Save ────────────────────────────────────────────────────────────────────
# Coerce any mixed-type object columns (e.g. graph_louvain_community_label)
# to string so pyarrow can serialise them.
for col in X_fused.select_dtypes(include='object').columns:
    if col not in {'ticker', 'quarter', 'Date'}:
        X_fused[col] = X_fused[col].astype(str)

X_fused.to_parquet(OUT_PATH, index=False)
print(f"[5/5] Saved X_fused → {OUT_PATH}")
print(f"      Shape: {X_fused.shape}")
print(f"      Tickers: {X_fused['ticker'].nunique()}")
print(f"      Quarters: {X_fused['quarter'].nunique()}")
print(f"      Distress events: {X_fused['distress_label'].sum()}")
print(f"      Feature columns: {len(feat_cols)}")
if X_nlp is not None:
    nlp_cols  = [c for c in feat_cols if c.startswith('nlp_')]
    print(f"      NLP features included: {len(nlp_cols)}")
else:
    print("      NLP features: NOT included (Module B pending)")
