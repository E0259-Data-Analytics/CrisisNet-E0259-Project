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
import json
import warnings
import pandas as pd
import numpy as np
from pandas.errors import PerformanceWarning

warnings.filterwarnings('ignore', category=PerformanceWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent
MODULE_D     = REPO_ROOT / "Module_D"

X_TS_PATH    = REPO_ROOT / "Module_A" / "results"  / "X_ts.parquet"
X_GRAPH_PATH = REPO_ROOT / "Module_C" / "results"  / "exports" / "X_graph.parquet"
LABELS_PATH  = REPO_ROOT / "crisisnet-data" / "data" / "label_unified.parquet"
COMPANY_PATH = REPO_ROOT / "crisisnet-data" / "data" / "company_list.csv"

# CHANGED: Use FinBERT features (richer — includes going_concern_flag,
# distress_phrase_count, distress_phrase_rate, covenant_flag, topic_kl_shift,
# readability features).  X_nlp_selected had 22 basic features; X_nlp_finbert has 44.
X_NLP_PATH = REPO_ROOT / "Module_B" / "results" / "X_nlp_finbert.parquet"

OUT_PATH     = MODULE_D / "X_fused.parquet"
SKIP         = {'ticker', 'quarter', 'Date', 'distress_label', 'year'}
FINANCIALS_DIRS = [
    REPO_ROOT / "crisisnet-data" / "Module_A" / "market_data" / "financials",
    REPO_ROOT / "crisisnet-data" / "Module_1" / "market_data" / "financials",
]
SEC_FACTS_DIRS = [
    REPO_ROOT / "crisisnet-data" / "Module_A" / "sec_xbrl" / "company_facts",
    REPO_ROOT / "crisisnet-data" / "Module_1" / "sec_xbrl" / "company_facts",
]

# ── Key financial ratios — forward-fill instead of zero-fill ──────────────────
# These are balance-sheet / market-derived metrics. When a company stops
# reporting or data is unavailable, zero-filling makes them look "average".
# Forward-fill (carry last known value) is strictly more conservative.
FFILL_COLS = [
    'altman_z', 'X1_wc_ta', 'X2_re_ta', 'X3_ebit_ta', 'X4_mcap_tl', 'X5_rev_ta',
    'merton_dd', 'merton_pd', 'asset_volatility', 'leverage_ratio',
    'debt_to_equity', 'interest_coverage', 'current_ratio', 'debt_to_assets',
    'free_cashflow', 'fcf_to_debt',
    # Post-Altman formula baselines — forward-fill within ticker (same logic as Altman)
    'ohlson_score', 'ohlson_pd',       # Ohlson O-Score (1980)
    'zmijewski_score', 'zmijewski_pd', # Zmijewski Score (1984)
    'ni_ta', 'tl_ta', 'ohlson_size', 'ohlson_oeneg',
    'ohlson_cfo_tl', 'ohlson_intwo', 'ohlson_chin',
]

# ── Features to engineer lags / deltas for (top SHAP + financial ratios) ──────
LAG_COLS = [
    'max_drawdown_6m', 'volatility_30d', 'vol_60d_last', 'vol_60d_mean',
    'altman_z', 'merton_dd', 'close_price', 'momentum_60d',
    'ohlson_pd', 'zmijewski_pd', 'merton_pd',   # post-Altman formula models
    'hy_oas', 'bbb_spread', 'ted_spread', 'oil_wti',
    'return_zscore_30d_mean', 'price_sma200_ratio_mean',
    'debt_to_equity', 'current_ratio', 'interest_coverage',
    # NLP temporal signals — delta4q captures YoY tone deterioration
    'nlp_tenk_score', 'nlp_tenk_score_4q_mean',
    # FinBERT-specific distress signals
    'nlp_distress_phrase_rate', 'nlp_topic_kl_shift', 'nlp_readability_fog_approx',
]


def _load_statement(fin_dir, ticker, stmt):
    path = fin_dir / f"{ticker}_{stmt}.csv"
    if not path.exists():
        return None
    if stmt == 'info':
        try:
            raw = pd.read_csv(path)
            return {
                c: pd.to_numeric(raw[c].iloc[0], errors='coerce')
                for c in ['marketCap'] if c in raw.columns
            }
        except Exception:
            return None
    try:
        df = pd.read_csv(path, index_col=0).T
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()].sort_index()
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception:
        return None


def _quarter_key(date_like):
    d = pd.to_datetime(date_like, errors='coerce')
    if pd.isna(d):
        return None
    return f"{d.year}Q{((d.month - 1) // 3) + 1}"


def _pick_latest_by_quarter(rows):
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df['filed'] = pd.to_datetime(df['filed'], errors='coerce')
    df = df.sort_values(['quarter', 'score', 'filed'])
    return df.groupby('quarter')['val'].last().to_dict()


def _sec_instant_series(us_gaap, tag_candidates, unit='USD'):
    rows = []
    for tag in tag_candidates:
        for e in us_gaap.get(tag, {}).get('units', {}).get(unit, []):
            if 'val' not in e or 'end' not in e:
                continue
            q = _quarter_key(e.get('end'))
            if q is None:
                continue
            frame = str(e.get('frame') or '')
            frame_match = frame == f"CY{q[:4]}Q{q[-1]}I"
            rows.append({
                'quarter': q,
                'val': pd.to_numeric(e.get('val'), errors='coerce'),
                'filed': e.get('filed'),
                'score': 2 if frame_match else 1,
            })
        if rows:
            break
    return _pick_latest_by_quarter(rows)


def _sec_duration_series(us_gaap, tag_candidates, unit='USD'):
    direct_rows = []
    annual_rows = []
    for tag in tag_candidates:
        for e in us_gaap.get(tag, {}).get('units', {}).get(unit, []):
            if 'val' not in e or not e.get('start') or not e.get('end'):
                continue
            start = pd.to_datetime(e.get('start'), errors='coerce')
            end = pd.to_datetime(e.get('end'), errors='coerce')
            if pd.isna(start) or pd.isna(end):
                continue
            q = _quarter_key(end)
            if q is None:
                continue
            days = (end - start).days + 1
            frame = str(e.get('frame') or '')
            val = pd.to_numeric(e.get('val'), errors='coerce')
            if pd.isna(val):
                continue
            if 60 <= days <= 120 and not frame.endswith('I'):
                direct_rows.append({
                    'quarter': q, 'val': val, 'filed': e.get('filed'),
                    'score': 2 if frame == f"CY{q[:4]}Q{q[-1]}" else 1,
                })
            elif 300 <= days <= 380 and end.month == 12:
                annual_rows.append({
                    'quarter': q, 'year': end.year, 'val': val,
                    'filed': e.get('filed'), 'score': 2 if frame == f"CY{end.year}" else 1,
                })
        if direct_rows or annual_rows:
            break

    out = _pick_latest_by_quarter(direct_rows)
    annual = _pick_latest_by_quarter(annual_rows)
    for q4, annual_val in annual.items():
        if not q4.endswith('Q4') or q4 in out:
            continue
        year = q4[:4]
        prior = [out.get(f"{year}Q{i}") for i in [1, 2, 3]]
        if all(v is not None and not pd.isna(v) for v in prior):
            out[q4] = annual_val - sum(prior)
    return out


def _build_altman_from_sec(X_base):
    """Compute original Altman 1968 components from SEC companyfacts XBRL."""
    facts_dir = next((p for p in SEC_FACTS_DIRS if p.exists()), None)
    if facts_dir is None:
        return None

    tags = {
        'assets': ['Assets'],
        'current_assets': ['AssetsCurrent'],
        'current_liabilities': ['LiabilitiesCurrent'],
        'liabilities': ['Liabilities'],
        'equity': [
            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
            'StockholdersEquity',
            'PartnersCapital',
        ],
        'retained_earnings': ['RetainedEarningsAccumulatedDeficit'],
        'shares': ['CommonStockSharesOutstanding', 'EntityCommonStockSharesOutstanding'],
        'revenue': [
            'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax',
            'SalesRevenueNet',
        ],
        'ebit': [
            # Priority-ordered: try each tag and stop at the first with data.
            # SEC filers use different line-item names across years and form types.
            'OperatingIncomeLoss',
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
            # Additional tags frequently used by U.S. Energy sector filers:
            'IncomeLossBeforeIncomeTaxExpenseBenefit',
            'IncomeLossFromOperationsBeforeIncomeTaxes',
            'ProfitLoss',                              # fallback for partnerships/MLPs
        ],
    }

    records = []
    for ticker, g in X_base.groupby('ticker'):
        facts_path = facts_dir / f"{ticker}_facts.json"
        if not facts_path.exists():
            continue
        try:
            us_gaap = json.load(open(facts_path)).get('facts', {}).get('us-gaap', {})
        except Exception:
            continue
        series = {
            'assets': _sec_instant_series(us_gaap, tags['assets']),
            'current_assets': _sec_instant_series(us_gaap, tags['current_assets']),
            'current_liabilities': _sec_instant_series(us_gaap, tags['current_liabilities']),
            'liabilities': _sec_instant_series(us_gaap, tags['liabilities']),
            'equity': _sec_instant_series(us_gaap, tags['equity']),
            'retained_earnings': _sec_instant_series(us_gaap, tags['retained_earnings']),
            'shares': _sec_instant_series(us_gaap, tags['shares'], unit='shares'),
            'revenue': _sec_duration_series(us_gaap, tags['revenue']),
            'ebit': _sec_duration_series(us_gaap, tags['ebit']),
        }

        for _, row in g[['ticker', 'quarter', 'close_price']].iterrows():
            q = row['quarter']
            ta = series['assets'].get(q, np.nan)
            ca = series['current_assets'].get(q, np.nan)
            cl = series['current_liabilities'].get(q, np.nan)
            tl = series['liabilities'].get(q, np.nan)
            eq = series['equity'].get(q, np.nan)
            if pd.isna(tl) and not pd.isna(ta) and not pd.isna(eq):
                tl = ta - eq
            wc = ca - cl if not pd.isna(ca) and not pd.isna(cl) else np.nan
            re_val = series['retained_earnings'].get(q, np.nan)
            rev = series['revenue'].get(q, np.nan)
            ebit = series['ebit'].get(q, np.nan)
            shares = series['shares'].get(q, np.nan)
            mcap = row['close_price'] * shares if not pd.isna(row['close_price']) and not pd.isna(shares) else np.nan

            X1 = wc / ta if not pd.isna(ta) and ta != 0 and not pd.isna(wc) else np.nan
            X2 = re_val / ta if not pd.isna(ta) and ta != 0 and not pd.isna(re_val) else np.nan
            X3 = ebit / ta if not pd.isna(ta) and ta != 0 and not pd.isna(ebit) else np.nan
            X4 = mcap / tl if not pd.isna(tl) and tl != 0 and not pd.isna(mcap) else np.nan
            X5 = rev / ta if not pd.isna(ta) and ta != 0 and not pd.isna(rev) else np.nan
            z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5 \
                if all(not pd.isna(x) for x in [X1, X2, X3, X4, X5]) else np.nan
            records.append({
                'ticker': ticker, 'quarter': q,
                'altman_z': z, 'X1_wc_ta': X1, 'X2_re_ta': X2,
                'X3_ebit_ta': X3, 'X4_mcap_tl': X4, 'X5_rev_ta': X5,
            })

    if not records:
        return None
    return pd.DataFrame(records).drop_duplicates(['ticker', 'quarter'], keep='last')


def _build_original_altman_features(X_base):
    """Recompute Altman 1968 components from SEC XBRL or raw statements."""
    sec_df = _build_altman_from_sec(X_base)
    if sec_df is not None and sec_df['altman_z'].notna().sum() > 0:
        return sec_df

    fin_dir = next((p for p in FINANCIALS_DIRS if p.exists()), None)
    if fin_dir is None:
        return None

    records = []
    for ticker, g in X_base.groupby('ticker'):
        bs = _load_statement(fin_dir, ticker, 'balance_sheet')
        inc = _load_statement(fin_dir, ticker, 'income')
        info = _load_statement(fin_dir, ticker, 'info') or {}
        if not isinstance(bs, pd.DataFrame) or not isinstance(inc, pd.DataFrame):
            continue

        for _, row in g[['ticker', 'quarter', 'Date', 'close_price']].iterrows():
            qdate = pd.to_datetime(row['Date'])
            bs_dates = bs.index[bs.index <= qdate + pd.Timedelta(days=45)]
            inc_dates = inc.index[inc.index <= qdate + pd.Timedelta(days=45)]
            if len(bs_dates) == 0 or len(inc_dates) == 0:
                continue

            bs_row = bs.loc[bs_dates[-1]]
            inc_row = inc.loc[inc_dates[-1]]
            ta = bs_row.get('Total Assets', np.nan)
            tl = bs_row.get('Total Liabilities Net Minority Interest',
                            bs_row.get('Total Liabilities', np.nan))
            ca = bs_row.get('Current Assets', np.nan)
            cl = bs_row.get('Current Liabilities', np.nan)
            wc = bs_row.get('Working Capital', np.nan)
            if pd.isna(wc) and not pd.isna(ca) and not pd.isna(cl):
                wc = ca - cl
            re_val = bs_row.get('Retained Earnings', np.nan)
            rev = inc_row.get('Total Revenue', np.nan)
            # EBIT fallback chain — same as module1_pipeline.py
            ebit = inc_row.get('EBIT', np.nan)
            if pd.isna(ebit):
                ebit = inc_row.get('Operating Income', np.nan)
            if pd.isna(ebit):
                _ni = inc_row.get('Net Income', np.nan)
                _ie = inc_row.get('Interest Expense', np.nan)
                _tx = inc_row.get('Tax Provision',
                                  inc_row.get('Income Tax Expense', np.nan))
                if not any(pd.isna(v) for v in [_ni, _ie, _tx]):
                    ebit = _ni + abs(_ie) + abs(_tx)

            # Shares fallback chain — same as module1_pipeline.py
            mcap = info.get('marketCap', np.nan)
            _cp  = row.get('close_price', np.nan)
            if not pd.isna(_cp):
                for _sf in ['Ordinary Shares Number', 'Share Issued',
                            'Common Stock Shares Outstanding', 'Common Stock']:
                    _sh = bs_row.get(_sf, np.nan)
                    if not pd.isna(_sh) and _sh > 1000:
                        mcap = _cp * _sh
                        break

            X1 = wc / ta if not pd.isna(ta) and ta != 0 and not pd.isna(wc) else np.nan
            X2 = re_val / ta if not pd.isna(ta) and ta != 0 and not pd.isna(re_val) else np.nan
            X3 = ebit / ta if not pd.isna(ta) and ta != 0 and not pd.isna(ebit) else np.nan
            X4 = mcap / tl if not pd.isna(tl) and tl != 0 and not pd.isna(mcap) else np.nan
            X5 = rev / ta if not pd.isna(ta) and ta != 0 and not pd.isna(rev) else np.nan
            z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5 \
                if all(not pd.isna(x) for x in [X1, X2, X3, X4, X5]) else np.nan
            records.append({
                'ticker': ticker, 'quarter': row['quarter'],
                'altman_z': z, 'X1_wc_ta': X1, 'X2_re_ta': X2,
                'X3_ebit_ta': X3, 'X4_mcap_tl': X4, 'X5_rev_ta': X5,
            })

    if not records:
        return None
    return pd.DataFrame(records).drop_duplicates(['ticker', 'quarter'], keep='last')

# ── 1. Load X_ts ──────────────────────────────────────────────────────────────
print("[1/6] Loading X_ts (Module A)…")
# X_ts.parquet is saved with (ticker, Date) as a multi-index from module1_pipeline.
# reset_index() (NOT drop=True) is required to promote those index levels back
# to regular columns so the downstream .apply() / merge calls can access them.
X_ts = pd.read_parquet(X_TS_PATH).reset_index()
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
if X_NLP_PATH is not None and X_NLP_PATH.exists():
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

# Generate label_unified.parquet inline if the file is absent from the repo.
# Required by: build_x_fused.py (this file) for the merge at step 4.
# Source files: Labels/energy_defaults_curated.csv + Labels/distress_from_drawdowns.csv
def _generate_label_unified(out_path: Path, data_root: Path):
    """Merge hard defaults and soft drawdown labels into a unified parquet."""
    defaults_path  = data_root / "Labels" / "energy_defaults_curated.csv"
    drawdowns_path = data_root / "Labels" / "distress_from_drawdowns.csv"
    company_path   = data_root / "data"   / "company_list.csv"

    if not defaults_path.exists() or not company_path.exists():
        print("      WARNING: source label CSVs not found; label_unified will be all-zero")
        return

    companies = pd.read_csv(company_path)
    tickers   = companies["ticker"].tolist()

    quarters = [f"{y}Q{q}" for y in range(2015, 2026) for q in range(1, 5)]
    idx   = pd.MultiIndex.from_product([tickers, quarters], names=["ticker", "quarter"])
    df_lb = pd.DataFrame({"distress_label": 0}, index=idx).reset_index()

    def _q(d):
        d = pd.to_datetime(d, errors="coerce")
        return None if pd.isna(d) else f"{d.year}Q{(d.month-1)//3+1}"

    def _mark(ticker, event_date, lead_q=4):
        eq = _q(event_date)
        if eq is None:
            return
        y, q = int(eq[:4]), int(eq[-1])
        for offset in range(lead_q + 1):
            q2, y2 = q - offset, y
            while q2 < 1:
                q2 += 4; y2 -= 1
            mask = (df_lb["ticker"] == ticker) & (df_lb["quarter"] == f"{y2}Q{q2}")
            df_lb.loc[mask, "distress_label"] = 1

    defaults = pd.read_csv(defaults_path)
    for _, row in defaults.iterrows():
        t = str(row.get("ticker", "")).strip().upper()
        if t in tickers:
            _mark(t, str(row.get("event_date", row.get("date", ""))), lead_q=4)

    if drawdowns_path.exists():
        ddowns = pd.read_csv(drawdowns_path)
        tcol = next((c for c in ddowns.columns if "ticker" in c.lower()), None)
        dcol = next((c for c in ddowns.columns if "date" in c.lower() or "start" in c.lower()), None)
        if tcol and dcol:
            for _, row in ddowns.iterrows():
                t = str(row[tcol]).strip().upper()
                if t in tickers:
                    _mark(t, str(row[dcol]), lead_q=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_lb.to_parquet(out_path, index=False)
    pos = int(df_lb["distress_label"].sum())
    print(f"      Generated label_unified.parquet: {len(df_lb)} rows, "
          f"{pos} distress ({pos/len(df_lb):.1%} prevalence) → {out_path}")


if not LABELS_PATH.exists():
    print("      label_unified.parquet not found — generating inline from source CSVs…")
    _generate_label_unified(LABELS_PATH, LABELS_PATH.parent.parent)

labels   = pd.read_parquet(LABELS_PATH)[['ticker', 'quarter', 'distress_label']]
X_fused  = X_fused.drop(columns=['distress_label'], errors='ignore')
X_fused  = X_fused.merge(labels, on=['ticker', 'quarter'], how='left')
X_fused['distress_label'] = X_fused['distress_label'].fillna(0).astype(int)
X_fused['year'] = pd.to_datetime(X_fused['Date']).dt.year

# Restore original Altman 1968 score/components if Module A exported zeroed
# financial ratios because the renamed data path was unavailable.
altman_cols = ['altman_z', 'X1_wc_ta', 'X2_re_ta', 'X3_ebit_ta', 'X4_mcap_tl', 'X5_rev_ta']
if all(c in X_fused.columns for c in altman_cols):
    altman_nonzero = int((X_fused['altman_z'].fillna(0) != 0).sum())
    if altman_nonzero == 0:
        print("      Altman Z is zeroed in X_ts; recomputing original score from SEC XBRL/company financials…")
        altman_df = _build_original_altman_features(X_fused)
        if altman_df is not None:
            X_fused = X_fused.merge(
                altman_df, on=['ticker', 'quarter'], how='left', suffixes=('', '_orig')
            )
            for c in altman_cols:
                orig = f'{c}_orig'
                if orig in X_fused.columns:
                    X_fused[c] = X_fused[orig].combine_first(X_fused[c])
            X_fused = X_fused.drop(columns=[f'{c}_orig' for c in altman_cols], errors='ignore')
            restored = int((X_fused['altman_z'].fillna(0) != 0).sum())
            print(f"      Original Altman Z restored for {restored} rows")
        else:
            print("      WARNING: SEC XBRL/company financials not found; Altman Z remains zeroed")

# ── 6. Feature engineering for recall ─────────────────────────────────────────
print("[5/6] Engineering recall-boosting features…")

# Sort for time-ordered operations
X_fused = X_fused.sort_values(['ticker', 'quarter']).reset_index(drop=True)

# F0. Forward-fill NLP features within ticker (annual Q1 10-K signals → Q2/Q3/Q4)
# Root cause of NLP not helping: 82% of rows had zero for all NLP features because
# X_nlp_selected is Q1-only (annual 10-K filing quarter).  Without ffill, Q2/Q3/Q4
# rows got zero-filled and the model never saw the NLP signal for those quarters.
# After ffill, Q1 values propagate to Q2→Q3→Q4 until the next year's Q1 filing.
if X_nlp is not None:
    nlp_feat_cols = [c for c in X_fused.columns if c.startswith('nlp_')]
    if nlp_feat_cols:
        X_fused[nlp_feat_cols] = X_fused.groupby('ticker')[nlp_feat_cols].transform('ffill')
        still_zero = (X_fused[nlp_feat_cols] == 0).all(axis=1).sum()
        print(f"      NLP forward-fill: {len(nlp_feat_cols)} cols "
              f"({still_zero} rows still zero — pre-first-filing quarters)")

    # F0b. NLP × Market interaction features
    # Distress language combined with high market volatility = stronger signal
    if 'nlp_distress_phrase_rate' in X_fused.columns and 'volatility_30d' in X_fused.columns:
        X_fused['nlp_distress_x_vol'] = (
            X_fused['nlp_distress_phrase_rate'] * X_fused['volatility_30d']
        )
        print("      NLP × volatility interaction feature added")
    # Negative sentiment + wide credit spreads = compounding distress signal
    if 'nlp_tenk_score' in X_fused.columns and 'hy_oas' in X_fused.columns:
        X_fused['nlp_neg_x_hy_oas'] = (
            (1 - X_fused['nlp_tenk_score'].clip(0, 1)) * X_fused['hy_oas']
        )
        print("      NLP × HY OAS interaction feature added")

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
    nlp_feature_names = [c for c in X_fused.columns if c.startswith('nlp_') or c in ('nlp_distress_x_vol', 'nlp_neg_x_hy_oas')]
    print(f"      NLP features (FinBERT): {len(nlp_feature_names)} columns")
    print(f"        Base: {sum(1 for c in nlp_feature_names if not any(x in c for x in ['lag','delta','trend','vs_sector','momentum']))}")
    print(f"        Lag/delta: {sum(1 for c in nlp_feature_names if any(x in c for x in ['lag','delta','trend']))}")
    print(f"        Interaction: {sum(1 for c in nlp_feature_names if 'nlp_distress_x_vol' in c or 'nlp_neg_x_hy_oas' in c)}")
else:
    print("      NLP features:     NOT included (Module B pending)")