"""
CrisisNet Module D — Step 1: Feature Alignment, Fusion & Engineering
=====================================================================
Merges X_ts (Module A), X_graph (Module C), and optionally X_nlp (Module B)
into X_fused.parquet, then engineers recall-boosting temporal features.

Root cause of poor recall (diagnosed):
  1. Zero-fill of missing financial ratios makes distressed companies look neutral.
  2. Model only sees point-in-time snapshots — no deterioration trajectory.
  3. No sector-wide stress features — misses 2019Q3 sector-wide oil crash.

Fixes applied (Section 5):
  F1. Forward-fill financial ratios within ticker before zero-fill.
  F2. Lag features (1Q, 2Q, 4Q) for top risk metrics.
  F3. Delta features (quarter-over-quarter change).
  F4. 4-quarter rolling trend slope (OLS) — captures sustained deterioration.
  F5. Sector-relative features — company vs peer-group median.
  F6. Sector distress contagion rate — fraction of sector peers in distress.
  F7. Macro momentum — oil/spread changes over 4 quarters.

Module B integration (single-line change):
    Uncomment X_NLP_PATH below once Module B produces X_nlp_finbert.parquet.

Usage:
    python Module_D/build_x_fused.py
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
MODULE_D     = REPO_ROOT / "Module_D"

X_TS_PATH    = REPO_ROOT / "Module_1" / "results"  / "X_ts.parquet"
X_GRAPH_PATH = REPO_ROOT / "Module_C" / "results"  / "exports" / "X_graph.parquet"
LABELS_PATH  = REPO_ROOT / "crisisnet-data" / "data" / "label_unified.parquet"
COMPANY_PATH = REPO_ROOT / "crisisnet-data" / "data" / "company_list.csv"

# Module B single-line toggle — uncomment to activate NLP features:
# X_NLP_PATH = REPO_ROOT / "Module_B" / "results" / "X_nlp_finbert.parquet"
X_NLP_PATH   = None

OUT_PATH     = MODULE_D / "X_fused.parquet"
SKIP         = {'ticker', 'quarter', 'Date', 'distress_label', 'year'}

# ── Key financial ratios — forward-fill instead of zero-fill ──────────────────
# These are balance-sheet / market-derived metrics. When a company stops
# reporting or data is unavailable, zero-filling makes them look "average".
# Forward-fill (carry last known value) is strictly more conservative.
FFILL_COLS = [
    'altman_z', 'X1_wc_ta', 'X2_re_ta', 'X3_ebit_ta', 'X4_mcap_tl', 'X5_rev_ta',
    'merton_dd', 'merton_pd', 'asset_volatility', 'leverage_ratio',
    'debt_to_equity', 'interest_coverage', 'current_ratio', 'debt_to_assets',
    'free_cashflow', 'fcf_to_debt',
]

# ── Features to engineer lags / deltas for (top SHAP + financial ratios) ──────
LAG_COLS = [
    'max_drawdown_6m', 'volatility_30d', 'vol_60d_last', 'vol_60d_mean',
    'altman_z', 'merton_dd', 'close_price', 'momentum_60d',
    'hy_oas', 'bbb_spread', 'ted_spread', 'oil_wti',
    'return_zscore_30d_mean', 'price_sma200_ratio_mean',
    'debt_to_equity', 'current_ratio', 'interest_coverage',
]

# ── 1. Load X_ts ──────────────────────────────────────────────────────────────
print("[1/6] Loading X_ts (Module A)…")
X_ts = pd.read_parquet(X_TS_PATH).reset_index(drop=True)
X_ts['quarter'] = pd.to_datetime(X_ts['Date']).apply(
    lambda d: f"{d.year}Q{(d.month - 1) // 3 + 1}"
)
print(f"      X_ts: {X_ts.shape}  |  tickers: {X_ts['ticker'].nunique()}")

# ── 2. Load X_graph ────────────────────────────────────────────────────────────
print("[2/6] Loading X_graph (Module C)…")
X_graph_raw = pd.read_parquet(X_GRAPH_PATH)
g_skip  = {'ticker', 'quarter', 'year', 'name', 'subsector', 'defaulted'}
X_graph = X_graph_raw.rename(
    columns={c: f"graph_{c}" for c in X_graph_raw.columns if c not in g_skip}
)
X_graph = X_graph.drop(
    columns=[c for c in ['name', 'subsector', 'defaulted', 'year'] if c in X_graph.columns],
    errors='ignore'
)
# Keep the raw subsector for sector-relative features
subsectors = X_graph_raw[['ticker', 'subsector']].drop_duplicates() \
             if 'subsector' in X_graph_raw.columns else None
print(f"      X_graph: {X_graph.shape}")

# ── 3. Load company subsector mapping ─────────────────────────────────────────
if subsectors is None and COMPANY_PATH.exists():
    comp = pd.read_csv(COMPANY_PATH)
    if 'subsector' in comp.columns:
        subsectors = comp[['ticker', 'subsector']].drop_duplicates()

# ── 4. Load X_nlp (optional) ──────────────────────────────────────────────────
X_nlp = None
if X_NLP_PATH is not None:
    print("[3/6] Loading X_nlp (Module B)…")
    X_nlp_raw = pd.read_parquet(X_NLP_PATH)
    X_nlp = X_nlp_raw.rename(
        columns={c: f"nlp_{c}" for c in X_nlp_raw.columns if c not in SKIP}
    )
    print(f"      X_nlp: {X_nlp.shape}")
else:
    print("[3/6] X_nlp skipped — Module B pending")

# ── 5. Merge ───────────────────────────────────────────────────────────────────
print("[4/6] Merging modules…")
ts_cols  = [c for c in X_ts.columns if c not in SKIP]
X_fused  = X_ts[['ticker', 'quarter', 'Date'] + ts_cols].copy()
X_fused  = X_fused.merge(X_graph, on=['ticker', 'quarter'], how='left')
if X_nlp is not None:
    X_fused = X_fused.merge(X_nlp, on=['ticker', 'quarter'], how='left')

labels   = pd.read_parquet(LABELS_PATH)[['ticker', 'quarter', 'distress_label']]
X_fused  = X_fused.drop(columns=['distress_label'], errors='ignore')
X_fused  = X_fused.merge(labels, on=['ticker', 'quarter'], how='left')
X_fused['distress_label'] = X_fused['distress_label'].fillna(0).astype(int)
X_fused['year'] = pd.to_datetime(X_fused['Date']).dt.year

# ── 6. Feature engineering for recall ─────────────────────────────────────────
print("[5/6] Engineering recall-boosting features…")

# Sort for time-ordered operations
X_fused = X_fused.sort_values(['ticker', 'quarter']).reset_index(drop=True)

# F1. Forward-fill financial ratios within ticker (don't zero-fill missing data)
for col in FFILL_COLS:
    if col in X_fused.columns:
        X_fused[col] = X_fused.groupby('ticker')[col].transform(
            lambda s: s.replace(0, np.nan).ffill()
        )

# F2 + F3. Lag features & deltas for key risk metrics
lag_cols_present = [c for c in LAG_COLS if c in X_fused.columns]
for col in lag_cols_present:
    for lag in [1, 2, 4]:
        lagged = X_fused.groupby('ticker')[col].shift(lag)
        X_fused[f'{col}_lag{lag}q'] = lagged
    # Quarter-over-quarter delta (direction of change is the key signal)
    X_fused[f'{col}_delta1q'] = X_fused[col] - X_fused[f'{col}_lag1q']
    # Year-over-year delta (structural deterioration signal)
    X_fused[f'{col}_delta4q'] = X_fused[col] - X_fused[f'{col}_lag4q']

print(f"      Lag/delta features added: {len(lag_cols_present) * 5} new columns")

# F4. 4-quarter rolling OLS trend slope (sustained deterioration)
def _rolling_slope(s, window=4):
    """Positive slope = improving; negative = deteriorating."""
    out = np.full(len(s), np.nan)
    x   = np.arange(window, dtype=float)
    x   -= x.mean()
    for i in range(window - 1, len(s)):
        y = s.iloc[i - window + 1: i + 1].values.astype(float)
        if np.isnan(y).any():
            continue
        out[i] = np.dot(x, y - y.mean()) / max(np.dot(x, x), 1e-9)
    return pd.Series(out, index=s.index)

slope_cols = ['altman_z', 'merton_dd', 'max_drawdown_6m',
              'volatility_30d', 'hy_oas', 'close_price']
slope_cols = [c for c in slope_cols if c in X_fused.columns]
for col in slope_cols:
    X_fused[f'{col}_trend4q'] = X_fused.groupby('ticker')[col].transform(_rolling_slope)

print(f"      Trend-slope features added: {len(slope_cols)}")

# F5. Sector-relative features (company vs peer-group median per quarter)
if subsectors is not None:
    X_fused = X_fused.merge(subsectors, on='ticker', how='left')
    sector_rel_cols = ['max_drawdown_6m', 'volatility_30d', 'altman_z',
                       'merton_dd', 'close_price']
    sector_rel_cols = [c for c in sector_rel_cols if c in X_fused.columns]
    for col in sector_rel_cols:
        sector_med = X_fused.groupby(['subsector', 'quarter'])[col].transform('median')
        X_fused[f'{col}_vs_sector'] = X_fused[col] - sector_med   # negative = worse than peers
    X_fused = X_fused.drop(columns=['subsector'], errors='ignore')
    print(f"      Sector-relative features added: {len(sector_rel_cols)}")

# F6. Sector distress contagion rate (lagged 1Q to avoid leakage)
# "How many of my sector peers were in distress last quarter?"
if subsectors is not None:
    tmp = X_fused.merge(subsectors, on='ticker', how='left')
    sect_dist = tmp.groupby(['subsector', 'quarter'])['distress_label'].mean().reset_index()
    sect_dist.columns = ['subsector', 'quarter', 'sector_distress_rate']
    tmp = tmp.merge(sect_dist, on=['subsector', 'quarter'], how='left')
    # Lag by 1Q to prevent leakage (we only know last quarter's sector state)
    tmp['sector_distress_rate_lag1q'] = tmp.groupby('ticker')['sector_distress_rate'].shift(1)
    X_fused['sector_distress_rate_lag1q'] = tmp['sector_distress_rate_lag1q'].values
    print("      Sector distress contagion rate (lagged 1Q) added")

# F7. Macro momentum — 4-quarter oil / spread changes (captures crisis onset)
macro_momentum_cols = ['oil_wti', 'hy_oas', 'vix_mean', 'ted_spread', 'baa_spread']
macro_momentum_cols = [c for c in macro_momentum_cols if c in X_fused.columns]
for col in macro_momentum_cols:
    # These are shared across tickers per quarter — use any ticker's series
    lag4 = X_fused.groupby('ticker')[col].shift(4)
    X_fused[f'{col}_momentum4q'] = X_fused[col] - lag4

print(f"      Macro-momentum features added: {len(macro_momentum_cols)}")

# ── 7. Final missing-value handling ───────────────────────────────────────────
# Zero-fill everything that remains (lag/delta NaN from window start is fine as 0)
feat_cols = [c for c in X_fused.columns if c not in SKIP]
numeric_feats = [c for c in feat_cols if pd.api.types.is_numeric_dtype(X_fused[c])]
X_fused[numeric_feats] = X_fused[numeric_feats].fillna(0)
X_fused['distress_label'] = X_fused['distress_label'].fillna(0).astype(int)

# ── 8. Save ────────────────────────────────────────────────────────────────────
# Coerce any remaining mixed-type object columns
for col in X_fused.select_dtypes(include='object').columns:
    if col not in {'ticker', 'quarter', 'Date'}:
        X_fused[col] = X_fused[col].astype(str)

X_fused.to_parquet(OUT_PATH, index=False)
print(f"[6/6] Saved X_fused → {OUT_PATH}")
print(f"      Shape:            {X_fused.shape}")
print(f"      Tickers:          {X_fused['ticker'].nunique()}")
print(f"      Quarters:         {X_fused['quarter'].nunique()}")
print(f"      Distress events:  {X_fused['distress_label'].sum()}")
print(f"      Total features:   {len(numeric_feats)}")
if X_nlp is not None:
    print(f"      NLP features:    {sum(1 for c in numeric_feats if c.startswith('nlp_'))}")
else:
    print("      NLP features:     NOT included (Module B pending)")
