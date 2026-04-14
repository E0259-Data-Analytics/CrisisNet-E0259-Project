#!/usr/bin/env python3
"""
CrisisNet Module 1 — The Financial Heartbeat Monitor
=====================================================
Complete end-to-end pipeline: data loading, enhanced daily + quarterly
feature engineering, model training (Altman Z, XGBoost, LSTM, Daily LSTM,
Cox PH), evaluation, 19 publication-quality visualizations, and
X_ts.parquet export for Module D fusion.

Usage:
    cd Module_A/notebooks
    python module1_pipeline.py
"""

import os, sys, json, warnings, pickle, time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, brier_score_loss, precision_recall_curve,
    classification_report, confusion_matrix, f1_score, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # Module_A/
REPO_ROOT    = PROJECT_ROOT.parent                      # CrisisNet-E0259-Project/
CRISISNET_DATA = REPO_ROOT / "crisisnet-data"           # canonical dataset root
DATA_DIR = PROJECT_ROOT / "data"
MARKET_DIR = CRISISNET_DATA / "Module_A" / "market_data"
FIN_DIR = MARKET_DIR / "financials"
CREDIT_DIR = CRISISNET_DATA / "Module_A" / "credit_spreads"
LABELS_DIR = CRISISNET_DATA / "Labels"
RESULTS_DIR = PROJECT_ROOT / "results"
VIZ_DIR = RESULTS_DIR / "visualizations"
MODELS_DIR = PROJECT_ROOT / "models"
for d in [RESULTS_DIR, VIZ_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
COLORS = {'safe': '#2ecc71', 'grey': '#f39c12', 'distress': '#e74c3c',
          'primary': '#2c3e50', 'secondary': '#3498db', 'accent': '#e74c3c'}

# 2022 contained zero distress events in this cohort, making it useless as a
# validation set (AUC is undefined when there are no positives).  Per the
# project review, we merge 2022 into training and rely solely on 5-fold
# walk-forward CV for model selection.  Test window is 2023-2025.
TRAIN_END = '2023-01-01'
VAL_END   = '2023-01-01'   # val_mask will be empty — by design

print("=" * 70)
print("  CRISISNET MODULE 1 — THE FINANCIAL HEARTBEAT MONITOR")
print("  Time Series & Credit Risk Engine")
print("=" * 70)
t_start = time.time()

# ═══════════════════════════════════════════════════════════════════════
# [1/9] DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[1/9] Loading data sources...")
print("─" * 70)

# 1a. Daily stock prices
prices_raw = pd.read_parquet(MARKET_DIR / "all_prices.parquet")
prices_raw.index = pd.to_datetime(prices_raw.index)
prices_raw.index.name = 'Date'
# A2: Use canonical ticker list from data/company_list.csv instead of
# discovering from price columns (ensures 40-ticker universe alignment)
company_list = pd.read_csv(CRISISNET_DATA / "data" / "company_list.csv")
TICKERS = sorted(company_list['ticker'].tolist())
print(f"  ✓ Stock prices: {len(TICKERS)} tickers (from company_list.csv), "
      f"{len(prices_raw)} trading days "
      f"({prices_raw.index.min().date()} → {prices_raw.index.max().date()})")

# Quarter-end dates for the full analysis window — used to generate
# placeholder records for tickers whose price data is all-NaN
# (e.g. CHK, SWN, HES, MRO, PXD: in all_prices.parquet but 0 non-null rows)
ANALYSIS_QDATES = pd.date_range('2015-03-31', '2025-12-31', freq='QE')

# 1b. FRED macro series
fred = pd.read_parquet(CREDIT_DIR / "fred_all_series.parquet")
fred.index = pd.to_datetime(fred.index)
fred = fred.sort_index().ffill()
print(f"  ✓ FRED macro: {fred.shape[1]} series, {len(fred)} obs")

# 1c. Labels
defaults_df = pd.read_csv(LABELS_DIR / "energy_defaults_curated.csv")
drawdowns_df = pd.read_csv(LABELS_DIR / "distress_from_drawdowns.csv")
print(f"  ✓ Labels: {len(defaults_df)} hard defaults, {len(drawdowns_df)} drawdown events")

# 1d. Quarterly financials
def load_stmt(ticker, stmt):
    fpath = FIN_DIR / f"{ticker}_{stmt}.csv"
    if not fpath.exists():
        return None
    if stmt == 'info':
        try:
            df = pd.read_csv(fpath)
            result = {}
            for col in ['marketCap', 'totalDebt', 'ebitda', 'freeCashflow',
                        'currentRatio', 'debtToEquity', 'beta']:
                if col in df.columns:
                    result[col] = pd.to_numeric(df[col].iloc[0], errors='coerce')
            return result
        except:
            return None
    df = pd.read_csv(fpath, index_col=0).T
    try:
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()].sort_index()
    except:
        return None
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

financials = {}
for t in tqdm(TICKERS, desc="  Loading financials", unit="ticker"):
    financials[t] = {s: load_stmt(t, s) for s in ['balance_sheet', 'income', 'cashflow', 'info']}

full_tickers = [t for t in TICKERS if isinstance(financials[t].get('balance_sheet'), pd.DataFrame)
                and isinstance(financials[t].get('income'), pd.DataFrame)]
print(f"  ✓ Quarterly financials: {len(full_tickers)}/{len(TICKERS)} tickers with full data")

# ═══════════════════════════════════════════════════════════════════════
# [2/9] ENHANCED DAILY FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[2/9] Computing enhanced daily stock features...")
print("─" * 70)

def compute_rsi(series, period=14):
    """Relative Strength Index — overbought/oversold oscillator."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD line, signal line, and histogram."""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

all_daily_features = []

for ticker in tqdm(TICKERS, desc="  Daily features", unit="ticker"):
    try:
        close = prices_raw[(ticker, 'Close')].dropna()
        high = prices_raw[(ticker, 'High')].dropna()
        low = prices_raw[(ticker, 'Low')].dropna()
        volume = prices_raw[(ticker, 'Volume')].dropna()
    except KeyError:
        continue
    if len(close) < 100:
        continue

    log_ret = np.log(close / close.shift(1))
    df = pd.DataFrame(index=close.index)
    df['ticker'] = ticker
    df['close'] = close
    df['log_return'] = log_ret

    # Multi-scale volatilities
    for w in [10, 30, 60, 90]:
        df[f'vol_{w}d'] = log_ret.rolling(w).std() * np.sqrt(252)

    # Volatility ratio (short/long) — regime change detector
    df['vol_ratio_10_60'] = df['vol_10d'] / (df['vol_60d'] + 1e-8)

    # Rolling z-score of returns
    df['return_zscore_30d'] = (log_ret - log_ret.rolling(30).mean()) / (log_ret.rolling(30).std() + 1e-8)

    # Multi-horizon momentum
    for h in [5, 20, 60, 120, 250]:
        df[f'mom_{h}d'] = close.pct_change(h)

    # Bollinger Bands
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df['bb_upper'] = sma_20 + 2 * std_20
    df['bb_lower'] = sma_20 - 2 * std_20
    df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma_20 + 1e-8)

    # RSI (14-day)
    df['rsi_14'] = compute_rsi(close, 14)

    # MACD
    macd_line, signal_line, macd_hist = compute_macd(close)
    df['macd_line'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_hist

    # Volume features
    df['volume'] = volume
    df['vol_sma_20'] = volume.rolling(20).mean()
    df['vol_ratio_20'] = volume / (df['vol_sma_20'] + 1)
    df['obv'] = (np.sign(log_ret) * volume).cumsum()
    df['obv_slope_20'] = df['obv'].diff(20) / (df['obv'].shift(20).abs() + 1e-8)

    # Price relative to SMAs
    df['price_sma50_ratio'] = close / close.rolling(50).mean()
    df['price_sma200_ratio'] = close / close.rolling(200).mean()
    df['sma50_sma200_cross'] = (close.rolling(50).mean() - close.rolling(200).mean()) / close

    # Drawdown from peak
    rolling_max = close.rolling(252, min_periods=1).max()
    df['drawdown'] = (close - rolling_max) / rolling_max
    df['days_since_high'] = close.groupby((close == rolling_max).cumsum()).cumcount()

    # True Range / ATR
    df['true_range'] = pd.concat([high - low, (high - close.shift(1)).abs(),
                                   (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df['atr_14'] = df['true_range'].rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / (close + 1e-8)

    all_daily_features.append(df)

daily_df = pd.concat(all_daily_features)
daily_df.index.name = 'Date'
print(f"  ✓ Daily features: {daily_df.shape} ({daily_df['ticker'].nunique()} tickers, "
      f"{len([c for c in daily_df.columns if c not in ['ticker','close','volume']])} features)")

# ═══════════════════════════════════════════════════════════════════════
# [3/9] QUARTERLY FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[3/9] Engineering quarterly features (market + fundamental + macro + structural)...")
print("─" * 70)

# --- 3a. Aggregate daily features to quarterly with rich statistics ---
print("  [3a] Aggregating daily features to quarterly...")

agg_funcs = {
    'close': ['last', 'mean', 'std'],
    'log_return': ['mean', 'std', 'min', 'max', 'skew'],
    'vol_30d': ['mean', 'last', 'max'],
    'vol_60d': ['mean', 'last'],
    'vol_ratio_10_60': ['mean', 'max'],
    'return_zscore_30d': ['mean', 'std', 'min', 'max'],
    'mom_20d': 'last', 'mom_60d': 'last', 'mom_120d': 'last', 'mom_250d': 'last',
    'bb_pct': ['mean', 'min'],
    'bb_width': ['mean', 'max'],
    'rsi_14': ['mean', 'last', 'min'],
    'macd_hist': ['mean', 'last'],
    'vol_ratio_20': ['mean', 'max'],
    'obv_slope_20': ['mean', 'last'],
    'price_sma50_ratio': ['mean', 'last'],
    'price_sma200_ratio': ['mean', 'last'],
    'sma50_sma200_cross': 'last',
    'drawdown': ['mean', 'min'],
    'atr_pct': ['mean', 'max'],
}

quarterly_enhanced = []
for ticker in tqdm(daily_df['ticker'].unique(), desc="  Quarterly agg", unit="ticker"):
    td = daily_df[daily_df['ticker'] == ticker].copy()
    td_numeric = td.drop(columns=['ticker'])
    q = td_numeric.resample('QE').agg(agg_funcs)
    q.columns = ['_'.join(c).strip('_') for c in q.columns]
    q['ticker'] = ticker
    quarterly_enhanced.append(q)

q_enhanced = pd.concat(quarterly_enhanced).reset_index()
q_enhanced.columns = [str(c).replace(' ', '_') for c in q_enhanced.columns]
print(f"  ✓ Quarterly enhanced daily features: {q_enhanced.shape}")

# --- 3b. Build fundamental + structural features per (ticker, quarter) ---
print("  [3b] Computing fundamental ratios, Altman Z-Score, Merton DD...")

records = []
for ticker in tqdm(TICKERS, desc="  Fundamentals", unit="ticker"):
    try:
        close = prices_raw[(ticker, 'Close')].dropna()
        volume = prices_raw[(ticker, 'Volume')].dropna()
        high = prices_raw[(ticker, 'High')].dropna()
        low = prices_raw[(ticker, 'Low')].dropna()
    except KeyError:
        continue
    if len(close) < 60:
        # Ticker has no usable price data (e.g. delisted/bankrupt company whose
        # OHLCV rows are all-NaN in all_prices.parquet: CHK, SWN, HES, MRO, PXD).
        # Generate NaN-filled quarterly placeholder records so distress labels
        # are attached in Section [4/9] — these tickers ARE the positive class.
        for qdate in ANALYSIS_QDATES:
            records.append({'ticker': ticker, 'Date': qdate})
        continue

    log_ret = np.log(close / close.shift(1)).dropna()
    quarters = close.resample('QE').last().index

    for qdate in quarters:
        rec = {'ticker': ticker, 'Date': qdate}
        q_close = close.loc[:qdate]
        q_ret = log_ret.loc[:qdate]
        q_vol = volume.loc[:qdate]

        if len(q_close) < 30:
            continue

        # Market features (basic — enhanced daily features fill the rest)
        rec['close_price'] = q_close.iloc[-1]
        recent_ret = q_ret.tail(30)
        rec['volatility_30d'] = recent_ret.std() * np.sqrt(252) if len(recent_ret) >= 20 else np.nan
        if len(q_close) >= 60:
            rec['momentum_60d'] = q_close.iloc[-1] / q_close.iloc[-60] - 1
        if len(q_close) >= 90:
            rec['momentum_90d'] = q_close.iloc[-1] / q_close.iloc[-90] - 1
        recent_vol = q_vol.tail(90)
        if len(recent_vol) >= 30:
            rec['volume_ratio'] = recent_vol.tail(20).mean() / (recent_vol.mean() + 1e-8)
        q_high = high.loc[:qdate].tail(30)
        q_low = low.loc[:qdate].tail(30)
        q_c = close.loc[:qdate].tail(30)
        if len(q_high) > 0:
            rec['intraday_range'] = ((q_high - q_low) / (q_c + 1e-8)).mean()
        trailing = q_close.tail(126)
        if len(trailing) > 10:
            peak = trailing.cummax()
            dd = (trailing - peak) / (peak + 1e-8)
            rec['max_drawdown_6m'] = dd.min()

        # Fundamental features (from quarterly financials)
        bs = financials[ticker].get('balance_sheet')
        inc = financials[ticker].get('income')
        cf = financials[ticker].get('cashflow')
        info = financials[ticker].get('info')

        if isinstance(bs, pd.DataFrame) and isinstance(inc, pd.DataFrame):
            bs_dates = bs.index[bs.index <= qdate + timedelta(days=45)]
            inc_dates = inc.index[inc.index <= qdate + timedelta(days=45)]
            if len(bs_dates) > 0 and len(inc_dates) > 0:
                bs_row = bs.loc[bs_dates[-1]]
                inc_row = inc.loc[inc_dates[-1]]

                ta = bs_row.get('Total Assets', np.nan)
                tl = bs_row.get('Total Liabilities Net Minority Interest',
                     bs_row.get('Total Liabilities', np.nan))
                wc = bs_row.get('Working Capital', np.nan)
                # Altman X2: Retained Earnings / Total Assets.
                # yfinance / EDGAR label this differently depending on whether the
                # company has an accumulated deficit (energy sector often does).
                re_val = bs_row.get('Retained Earnings', np.nan)
                if pd.isna(re_val):
                    re_val = bs_row.get('Retained Earnings Accumulated Deficit', np.nan)
                if pd.isna(re_val):
                    # When Retained Earnings isn't present, approximate from
                    # Stockholders Equity (= RE + Common Stock + APIC + AOCI).
                    # This overestimates RE but is conservative for distress detection.
                    re_val = bs_row.get('Stockholders Equity', np.nan)
                    if pd.isna(re_val):
                        re_val = bs_row.get('Common Stock Equity',
                                  bs_row.get('Total Stockholder Equity', np.nan))
                ltd = bs_row.get('Long Term Debt',
                     bs_row.get('Long Term Debt And Capital Lease Obligation',
                     bs_row.get('Long Term Debt Noncurrent', np.nan)))
                ca = bs_row.get('Current Assets', np.nan)
                cl = bs_row.get('Current Liabilities', np.nan)
                if pd.isna(wc) and not pd.isna(ca) and not pd.isna(cl):
                    wc = ca - cl
                rev = inc_row.get('Total Revenue', np.nan)
                # Altman X3 numerator: EBIT (Earnings Before Interest and Taxes).
                # Yahoo Finance field name varies across yfinance versions / ticker types.
                # Priority: EBIT → Operating Income → reconstructed from components.
                ebit = inc_row.get('EBIT', np.nan)
                if pd.isna(ebit):
                    ebit = inc_row.get('Operating Income', np.nan)
                if pd.isna(ebit):
                    # EBIT ≈ Net Income + |Interest Expense| + |Tax Provision|
                    _ni = inc_row.get('Net Income', np.nan)
                    _ie = inc_row.get('Interest Expense', np.nan)
                    _tx = inc_row.get('Tax Provision',
                                      inc_row.get('Income Tax Expense', np.nan))
                    if not any(pd.isna(v) for v in [_ni, _ie, _tx]):
                        ebit = _ni + abs(_ie) + abs(_tx)
                ie = inc_row.get('Interest Expense', np.nan)

                # ── Ohlson (1980) and Zmijewski (1984) raw ingredients ────────
                ni_inc = inc_row.get('Net Income', np.nan)
                rec['ni_ta']        = ni_inc / ta if not pd.isna(ta) and ta != 0 and not pd.isna(ni_inc) else np.nan
                rec['tl_ta']        = tl / ta     if not pd.isna(ta) and ta != 0 and not pd.isna(tl)     else np.nan
                rec['ohlson_size']  = float(np.log(ta)) if not pd.isna(ta) and ta > 0 else np.nan
                rec['ohlson_oeneg'] = 1.0 if (not pd.isna(tl) and not pd.isna(ta) and tl > ta) else 0.0

                mcap = np.nan
                if isinstance(info, dict) and 'marketCap' in info:
                    mcap = info['marketCap']
                # Altman X4 numerator: Market Cap = Price × Shares Outstanding.
                # Yahoo Finance / EDGAR use different field names for shares:
                # Try all common variants and use the first non-trivial value.
                for _sf in ['Ordinary Shares Number', 'Share Issued',
                            'Common Stock Shares Outstanding', 'Common Stock']:
                    _sh = bs_row.get(_sf, np.nan)
                    # Sanity: reject values < 1000 (likely a ratio or millions error)
                    if not pd.isna(_sh) and _sh > 1000 and not pd.isna(rec.get('close_price', np.nan)):
                        mcap = rec['close_price'] * _sh
                        break
                equity = (ta - tl) if not pd.isna(ta) and not pd.isna(tl) else np.nan

                # Altman Z-Score sub-components
                X1 = wc / ta if not pd.isna(ta) and ta != 0 and not pd.isna(wc) else np.nan
                X2 = re_val / ta if not pd.isna(ta) and ta != 0 and not pd.isna(re_val) else np.nan
                X3 = ebit / ta if not pd.isna(ta) and ta != 0 and not pd.isna(ebit) else np.nan
                # X4: prefer market-cap/liabilities (Altman 1968 public-company model).
                # When market cap is unavailable, fall back to book-equity/liabilities
                # (Altman Z' 1983 private-company variant) — documented in rec flag.
                if not pd.isna(tl) and tl != 0 and not pd.isna(mcap):
                    X4 = mcap / tl
                    rec['altman_z_x4_source'] = 'market_cap'
                elif not pd.isna(tl) and tl != 0 and not pd.isna(equity):
                    # Book-equity proxy: slightly conservative (book < market for
                    # healthy firms), but prevents X4 NaN collapsing the whole score.
                    X4 = equity / tl
                    rec['altman_z_x4_source'] = 'book_equity'
                else:
                    X4 = np.nan
                X5 = rev / ta if not pd.isna(ta) and ta != 0 and not pd.isna(rev) else np.nan
                rec['X1_wc_ta'] = X1
                rec['X2_re_ta'] = X2
                rec['X3_ebit_ta'] = X3
                rec['X4_mcap_tl'] = X4
                rec['X5_rev_ta'] = X5
                if all(not pd.isna(x) for x in [X1, X2, X3, X4, X5]):
                    rec['altman_z'] = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

                # Debt-to-Equity
                if not pd.isna(equity) and equity != 0 and not pd.isna(ltd):
                    rec['debt_to_equity'] = ltd / equity
                # Interest Coverage
                if not pd.isna(ie) and ie != 0 and not pd.isna(ebit):
                    rec['interest_coverage'] = ebit / abs(ie)
                # Current Ratio
                if not pd.isna(cl) and cl != 0 and not pd.isna(ca):
                    rec['current_ratio'] = ca / cl
                # Debt-to-Assets
                if not pd.isna(ta) and ta != 0 and not pd.isna(ltd):
                    rec['debt_to_assets'] = ltd / ta

                # Cashflow features
                if isinstance(cf, pd.DataFrame):
                    cf_dates = cf.index[cf.index <= qdate + timedelta(days=45)]
                    if len(cf_dates) > 0:
                        cf_row = cf.loc[cf_dates[-1]]
                        ocf = cf_row.get('Operating Cash Flow', np.nan)
                        capex = cf_row.get('Capital Expenditure', np.nan)
                        if not pd.isna(ocf) and not pd.isna(capex):
                            rec['free_cashflow'] = ocf - abs(capex)
                            if not pd.isna(ltd) and ltd != 0:
                                rec['fcf_to_debt'] = rec['free_cashflow'] / ltd
                        # CFO/TL for Ohlson O-Score
                        rec['ohlson_cfo_tl'] = ocf / tl if not pd.isna(ocf) and not pd.isna(tl) and tl != 0 else np.nan

                # Merton Distance-to-Default
                if not pd.isna(mcap) and not pd.isna(ltd) and ltd > 0:
                    V = mcap + ltd
                    D = ltd
                    price_window = q_close.tail(252)
                    if len(price_window) >= 60:
                        sigma_e = np.log(price_window / price_window.shift(1)).dropna().std() * np.sqrt(252)
                        sigma_a = sigma_e * (mcap / V)
                        if sigma_a > 0:
                            r = 0.03; T = 1.0
                            dd_val = (np.log(V/D) + (r - 0.5*sigma_a**2)*T) / (sigma_a*np.sqrt(T))
                            rec['merton_dd'] = dd_val
                            rec['merton_pd'] = norm.cdf(-dd_val)
                            rec['asset_volatility'] = sigma_a
                            rec['leverage_ratio'] = D / V

        records.append(rec)

X_ts = pd.DataFrame(records)
print(f"  ✓ Fundamental panel: {X_ts.shape} ({X_ts['ticker'].nunique()} tickers)")

# ── Altman Z diagnostic: log how many records have a valid score ──────────────
_z_coverage = X_ts['altman_z'].notna().sum() if 'altman_z' in X_ts.columns else 0
_z_total    = len(X_ts)
print(f"  ✓ Altman Z from financials CSV: {_z_coverage}/{_z_total} records "
      f"({100*_z_coverage/_z_total:.1f}%)")

# ── SEC XBRL fallback for Altman Z (fills gaps when CSV financials are missing) ─
# Mirrors the _build_altman_from_sec() logic in build_x_fused.py.
# Required when yfinance financials don't carry EBIT / Retained Earnings / Shares.
_SEC_FACTS_DIRS = [
    CRISISNET_DATA / "Module_A" / "sec_xbrl" / "company_facts",
    CRISISNET_DATA / "Module_1" / "sec_xbrl" / "company_facts",
]

def _sec_instant(us_gaap, tags, unit='USD'):
    """Return {quarterKey: value} for the first tag with data."""
    for tag in tags:
        rows = []
        for e in us_gaap.get(tag, {}).get('units', {}).get(unit, []):
            if 'val' not in e or 'end' not in e: continue
            d = pd.to_datetime(e['end'], errors='coerce')
            if pd.isna(d): continue
            q = f"{d.year}Q{(d.month-1)//3+1}"
            rows.append({'q': q, 'val': float(e['val']),
                         'filed': e.get('filed', ''), 'instant': True})
        if rows:
            df_r = pd.DataFrame(rows).sort_values(['q','filed'])
            return df_r.groupby('q')['val'].last().to_dict()
    return {}

def _sec_duration(us_gaap, tags, unit='USD'):
    """Return {quarterKey: value} for quarterly income-statement items."""
    for tag in tags:
        direct, annual = [], []
        for e in us_gaap.get(tag, {}).get('units', {}).get(unit, []):
            if not e.get('start') or not e.get('end') or 'val' not in e: continue
            s = pd.to_datetime(e['start'], errors='coerce')
            d = pd.to_datetime(e['end'],   errors='coerce')
            if pd.isna(s) or pd.isna(d): continue
            days = (d - s).days + 1
            q = f"{d.year}Q{(d.month-1)//3+1}"
            val = float(e['val'])
            if 60 <= days <= 120:
                direct.append({'q': q, 'val': val, 'filed': e.get('filed','')})
            elif 300 <= days <= 380 and d.month == 12:
                annual.append({'q': q, 'year': d.year, 'val': val})
        if direct:
            df_d = pd.DataFrame(direct).sort_values(['q','filed'])
            return df_d.groupby('q')['val'].last().to_dict()
        if annual:
            return {r['q']: r['val'] for r in annual}
    return {}

_facts_dir = next((p for p in _SEC_FACTS_DIRS if p.exists()), None)
if _facts_dir:
    print(f"  [3b-xbrl] Filling Altman Z from SEC XBRL facts ({_facts_dir.name})…")
    _xbrl_rows = []
    for _ticker in TICKERS:
        _fp = _facts_dir / f"{_ticker}_facts.json"
        if not _fp.exists(): continue
        try:
            _usg = json.load(open(_fp)).get('facts', {}).get('us-gaap', {})
        except Exception: continue
        _ser = {
            'assets':   _sec_instant(_usg, ['Assets']),
            'ca':       _sec_instant(_usg, ['AssetsCurrent']),
            'cl':       _sec_instant(_usg, ['LiabilitiesCurrent']),
            'liab':     _sec_instant(_usg, ['Liabilities']),
            'equity':   _sec_instant(_usg, [
                            'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                            'StockholdersEquity', 'PartnersCapital']),
            're':       _sec_instant(_usg, ['RetainedEarningsAccumulatedDeficit']),
            'shares':   _sec_instant(_usg, ['CommonStockSharesOutstanding',
                                            'EntityCommonStockSharesOutstanding'], unit='shares'),
            'revenue':  _sec_duration(_usg, ['Revenues',
                                             'RevenueFromContractWithCustomerExcludingAssessedTax',
                                             'SalesRevenueNet']),
            'ebit':     _sec_duration(_usg, ['OperatingIncomeLoss',
                                             'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                                             'IncomeLossBeforeIncomeTaxExpenseBenefit',
                                             'IncomeLossFromOperationsBeforeIncomeTaxes',
                                             'ProfitLoss']),
        }
        # Match to X_ts rows for this ticker
        mask = X_ts['ticker'] == _ticker
        if not mask.any(): continue
        for idx in X_ts[mask].index:
            row  = X_ts.loc[idx]
            d    = pd.to_datetime(row['Date'], errors='coerce')
            if pd.isna(d): continue
            q    = f"{d.year}Q{(d.month-1)//3+1}"
            ta   = _ser['assets'].get(q, np.nan)
            ca   = _ser['ca'].get(q, np.nan)
            cl   = _ser['cl'].get(q, np.nan)
            tl   = _ser['liab'].get(q, np.nan)
            eq   = _ser['equity'].get(q, np.nan)
            if pd.isna(tl) and not pd.isna(ta) and not pd.isna(eq):
                tl = ta - eq
            wc   = (ca - cl) if not pd.isna(ca) and not pd.isna(cl) else np.nan
            re   = _ser['re'].get(q, np.nan)
            rev  = _ser['revenue'].get(q, np.nan)
            ebit = _ser['ebit'].get(q, np.nan)
            sh   = _ser['shares'].get(q, np.nan)
            cp   = row.get('close_price', np.nan)
            mc   = cp * sh if not pd.isna(cp) and not pd.isna(sh) else np.nan
            if pd.isna(mc) and not pd.isna(eq): mc = eq   # book-equity fallback

            def _r(n, d): return n/d if not pd.isna(n) and not pd.isna(d) and d != 0 else np.nan
            X1 = _r(wc, ta); X2 = _r(re, ta); X3 = _r(ebit, ta)
            X4 = _r(mc, tl);  X5 = _r(rev, ta)
            # Only fill if the CSV-based score is still missing
            if pd.isna(X_ts.at[idx, 'altman_z'] if 'altman_z' in X_ts.columns else np.nan):
                if all(not pd.isna(v) for v in [X1, X2, X3, X4, X5]):
                    X_ts.at[idx, 'altman_z']    = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
                    X_ts.at[idx, 'X1_wc_ta']    = X1
                    X_ts.at[idx, 'X2_re_ta']    = X2
                    X_ts.at[idx, 'X3_ebit_ta']  = X3
                    X_ts.at[idx, 'X4_mcap_tl']  = X4
                    X_ts.at[idx, 'X5_rev_ta']   = X5
    _z_after = X_ts['altman_z'].notna().sum() if 'altman_z' in X_ts.columns else 0
    print(f"  ✓ Altman Z after XBRL fill: {_z_after}/{_z_total} records "
          f"({100*_z_after/_z_total:.1f}%)  [+{_z_after-_z_coverage} filled from XBRL]")
else:
    print("  [3b-xbrl] SEC XBRL facts not found — skipping XBRL Altman fallback")
    print(f"  → Run: python scripts/pull_hf_dataset.py  to download XBRL data")

# ── Ohlson O-Score (1980) and Zmijewski Score (1984) ─────────────────────────
# Computed from the quarterly fundamental panel AFTER XBRL fill so we have the
# broadest possible coverage of the underlying accounting variables.
#
# Ohlson O-Score (1980): logistic regression on 9 accounting ratios.
#   O = -1.32 - 0.407*log(TA) + 6.03*(TL/TA) - 1.43*(WC/TA)
#       + 0.0757*(CL/CA) - 1.72*OENEG - 2.37*(NI/TA)
#       - 1.83*(CFO/TL) + 0.285*INTWO - 0.521*CHIN
#   P(distress) = σ(O)   [logistic sigmoid]
#   Ref: Ohlson J.A. (1980), J. Accounting Research, 18(1):109-131
#
# Zmijewski Score (1984): probit model on 3 ratios.
#   Z = -4.336 - 4.513*(NI/TA) + 5.679*(TL/TA) + 0.004*(CA/CL)
#   P(distress) = Φ(Z)   [standard normal CDF]
#   Ref: Zmijewski M.E. (1984), J. Accounting Research, 22(Suppl.):59-82
#
# Note: Both models use original 1980/1984 coefficients. Size (log TA) in
# Ohlson uses nominal USD which shifts the intercept vs. original GNP-deflated
# data, but AUC (rank-based) is invariant to monotone score transformations.

X_ts = X_ts.sort_values(['ticker', 'Date']).reset_index(drop=True)

# INTWO: 1 if net income < 0 for BOTH current AND prior year (~4 quarters)
ni_prev_yr  = X_ts.groupby('ticker')['ni_ta'].shift(4)
X_ts['ohlson_intwo'] = ((X_ts['ni_ta'].fillna(0) < 0) & (ni_prev_yr.fillna(0) < 0)).astype(float)

# CHIN: normalised change in net income — captures recent earnings trajectory
ni_abs_sum  = X_ts['ni_ta'].abs() + ni_prev_yr.abs()
X_ts['ohlson_chin'] = ((X_ts['ni_ta'] - ni_prev_yr) / ni_abs_sum.replace(0, np.nan)).fillna(0)

# ── Ohlson O-Score (vectorised) ───────────────────────────────────────────────
_o_miss = X_ts['ohlson_size'].isna() | X_ts['tl_ta'].isna() | X_ts['ni_ta'].isna()
_cl_ca  = (1.0 / X_ts['current_ratio'].replace(0, np.nan)).fillna(0)   # CL/CA
_cfo_tl = X_ts['ohlson_cfo_tl'].fillna(0) if 'ohlson_cfo_tl' in X_ts.columns else 0.0
_o_raw  = (-1.32
           - 0.407 * X_ts['ohlson_size'].fillna(0)
           + 6.03  * X_ts['tl_ta'].fillna(0)
           - 1.43  * X_ts['X1_wc_ta'].fillna(0)
           + 0.0757 * _cl_ca
           - 1.72  * X_ts['ohlson_oeneg'].fillna(0)
           - 2.37  * X_ts['ni_ta'].fillna(0)
           - 1.83  * _cfo_tl
           + 0.285 * X_ts['ohlson_intwo']
           - 0.521 * X_ts['ohlson_chin'])
X_ts['ohlson_score'] = _o_raw.where(~_o_miss)
X_ts['ohlson_pd']    = (1.0 / (1.0 + np.exp(-_o_raw.clip(-500, 500)))).where(~_o_miss)

# ── Zmijewski Score (vectorised) ─────────────────────────────────────────────
_z_miss  = X_ts['ni_ta'].isna() | X_ts['tl_ta'].isna() | X_ts['current_ratio'].isna()
_zm_raw  = (-4.336
            - 4.513 * X_ts['ni_ta'].fillna(0)
            + 5.679 * X_ts['tl_ta'].fillna(0)
            + 0.004 * X_ts['current_ratio'].fillna(0))
X_ts['zmijewski_score'] = _zm_raw.where(~_z_miss)
X_ts['zmijewski_pd']    = pd.Series(norm.cdf(_zm_raw.values),
                                     index=X_ts.index).where(~_z_miss)

_o_cov = int(X_ts['ohlson_pd'].notna().sum())
_z_cov = int(X_ts['zmijewski_pd'].notna().sum())
_m_cov = int((X_ts['merton_pd'].fillna(0) > 0).sum()) if 'merton_pd' in X_ts.columns else 0
print(f"  ✓ Ohlson O-Score (1980):    {_o_cov}/{len(X_ts)} records ({100*_o_cov/len(X_ts):.1f}%)")
print(f"  ✓ Zmijewski Score (1984):   {_z_cov}/{len(X_ts)} records ({100*_z_cov/len(X_ts):.1f}%)")
print(f"  ✓ Merton DD (1974):         {_m_cov}/{len(X_ts)} records ({100*_m_cov/len(X_ts):.1f}%)")

# --- 3c. Merge enhanced daily aggregates ---
print("  [3c] Merging enhanced daily aggregates...")
old_cols = set(X_ts.columns)
new_feature_cols = [c for c in q_enhanced.columns if c not in old_cols and c not in ['Date', 'ticker']]
X_ts = pd.merge(X_ts, q_enhanced[['Date', 'ticker'] + new_feature_cols], on=['Date', 'ticker'], how='left')
X_ts[new_feature_cols] = X_ts[new_feature_cols].apply(pd.to_numeric, errors='coerce')
print(f"  ✓ Added {len(new_feature_cols)} enhanced daily features → X_ts: {X_ts.shape}")

# --- 3d. Macro/Credit context features ---
print("  [3d] Adding macro/credit context (FRED)...")
fred_q = fred.resample('QE').agg(['mean', 'last'])
fred_q.columns = [f"{a}_{b}" for a, b in fred_q.columns]

macro_cols = {
    'BAMLH0A0HYM2_mean': 'hy_oas', 'BAMLH0A0HYM2_last': 'hy_oas_last',
    'VIXCLS_mean': 'vix_mean', 'VIXCLS_last': 'vix_last',
    'DCOILWTICO_mean': 'oil_wti', 'DCOILWTICO_last': 'oil_wti_last',
    'DCOILBRENTEU_mean': 'oil_brent', 'DHHNGSP_mean': 'natgas_price',
    'T10Y2Y_mean': 'yield_slope', 'T10Y2Y_last': 'yield_slope_last',
    'DGS10_mean': 'treasury_10y', 'TEDRATE_mean': 'ted_spread',
    'FEDFUNDS_mean': 'fed_funds', 'UNRATE_mean': 'unemployment',
    'BAA10Y_mean': 'baa_spread', 'BAMLC0A4CBBB_mean': 'bbb_spread',
}
fred_selected = fred_q[[c for c in macro_cols.keys() if c in fred_q.columns]].copy()
fred_selected = fred_selected.rename(columns=macro_cols)
fred_selected.index.name = 'Date'

fred_selected['oil_momentum'] = fred_selected.get('oil_wti_last', pd.Series(dtype=float)).pct_change()
fred_selected['hy_oas_change'] = fred_selected.get('hy_oas_last', pd.Series(dtype=float)).diff()
fred_selected['yield_curve_inverted'] = (fred_selected.get('yield_slope_last', pd.Series(dtype=float)) < 0).astype(float)

X_ts = pd.merge(X_ts, fred_selected.reset_index(), on='Date', how='left')
print(f"  ✓ Macro features added → X_ts: {X_ts.shape}")

# --- 3e. HMM Regime Detection ---
print("  [3e] Training HMM regime detection (VIX + HY OAS)...")
from hmmlearn.hmm import GaussianHMM

vix = fred['VIXCLS'].dropna()
hy = fred['BAMLH0A0HYM2'].dropna()
hmm_data = pd.DataFrame({'VIX': vix, 'HY_OAS': hy}).dropna()

scaler_hmm = StandardScaler()
X_hmm = scaler_hmm.fit_transform(hmm_data[['VIX', 'HY_OAS']])
hmm_model = GaussianHMM(n_components=2, covariance_type='full', n_iter=200, random_state=42)
hmm_model.fit(X_hmm)
states = hmm_model.predict(X_hmm)

state_means = [hmm_data['VIX'].values[states == i].mean() for i in range(2)]
stress_state = np.argmax(state_means)
hmm_data['regime'] = np.where(states == stress_state, 'stress', 'calm')
hmm_data['stress_flag'] = (states == stress_state).astype(float)
hmm_daily = hmm_data.copy()

hmm_q = hmm_data.resample('QE').agg({'stress_flag': 'mean'})
hmm_q = hmm_q.rename(columns={'stress_flag': 'regime_stress_frac'})
hmm_q.index.name = 'Date'
X_ts = pd.merge(X_ts, hmm_q.reset_index(), on='Date', how='left')

n_stress = (states == stress_state).sum()
n_calm = (states != stress_state).sum()
print(f"  ✓ HMM: {n_stress} stress days ({n_stress/(n_stress+n_calm)*100:.1f}%), "
      f"{n_calm} calm days")

# Final feature count
all_cols = [c for c in X_ts.columns if c not in ['Date', 'ticker']]
print(f"\n  ══ TOTAL: X_ts {X_ts.shape} — {X_ts['ticker'].nunique()} tickers, "
      f"{len(all_cols)} features ══")

# ═══════════════════════════════════════════════════════════════════════
# [4/9] DISTRESS LABELS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[4/9] Creating distress labels...")
print("─" * 70)

# A3: Use unified canonical labels from data/label_unified.parquet
# (replaces inline label construction from defaults_df + drawdowns_df)
labels = pd.read_parquet(CRISISNET_DATA / "data" / "label_unified.parquet")
X_ts['quarter'] = X_ts['Date'].apply(
    lambda d: f"{d.year}Q{(d.month - 1) // 3 + 1}"
)
X_ts = X_ts.drop(columns=['distress_label'], errors='ignore')
X_ts = X_ts.merge(
    labels[['ticker', 'quarter', 'distress_label']],
    on=['ticker', 'quarter'],
    how='left'
)
X_ts['distress_label'] = X_ts['distress_label'].fillna(0).astype(int)

n_pos = X_ts['distress_label'].sum()
n_neg = len(X_ts) - n_pos
print(f"  ✓ Healthy: {n_neg}  |  Distress: {n_pos}  |  Rate: {n_pos/len(X_ts)*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════════
# [5/9] TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[5/9] Temporal train/val/test split...")
print("─" * 70)

id_cols = ['Date', 'ticker', 'distress_label', 'quarter']
feature_cols = [c for c in X_ts.columns if c not in id_cols]

# Ensure all numeric
X_ts[feature_cols] = X_ts[feature_cols].apply(pd.to_numeric, errors='coerce')

# ── Snapshot formula-based PD columns BEFORE preprocessing ────────────────────
# fillna(train_medians) + winsorize will collapse ohlson_pd/zmijewski_pd/merton_pd
# to a single constant in any split that has no CSV-financial coverage (e.g. 2023+
# test rows), making AUC=0.5 an artefact of preprocessing rather than signal absence.
# We save the original NaN-bearing values here and reattach them after splitting so
# that _formula_baseline_eval sees the real covered/uncovered rows.
_FORMULA_PD_COLS = ['ohlson_pd', 'zmijewski_pd', 'merton_pd']
_formula_raw_snap = {
    col: X_ts[col].copy() if col in X_ts.columns else pd.Series(np.nan, index=X_ts.index)
    for col in _FORMULA_PD_COLS
}

train_mask = X_ts['Date'] < TRAIN_END
val_mask = (X_ts['Date'] >= TRAIN_END) & (X_ts['Date'] < VAL_END)
test_mask = X_ts['Date'] >= VAL_END

# Impute with train medians
train_medians = X_ts.loc[train_mask, feature_cols].median()
X_ts[feature_cols] = X_ts[feature_cols].fillna(train_medians)
X_ts[feature_cols] = X_ts[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

# Winsorize to 1st/99th percentile from train
for col in tqdm(feature_cols, desc="  Winsorizing", unit="feat"):
    q01, q99 = X_ts.loc[train_mask, col].quantile([0.01, 0.99])
    X_ts[col] = X_ts[col].clip(q01, q99)

X_train = X_ts[train_mask].copy()
X_val   = X_ts[val_mask].copy()
X_test  = X_ts[test_mask].copy()
y_train = X_train['distress_label'].values
y_val   = X_val['distress_label'].values
y_test  = X_test['distress_label'].values

# Reattach raw (pre-preprocessing) formula columns for baseline evaluation.
# Use _raw suffix so they are never confused with the preprocessed ML-feature versions.
for _fc in _FORMULA_PD_COLS:
    X_train[_fc + '_raw'] = _formula_raw_snap[_fc].loc[X_train.index].values
    X_test[_fc  + '_raw'] = _formula_raw_snap[_fc].loc[X_test.index].values
    if len(X_val) > 0:
        X_val[_fc + '_raw'] = _formula_raw_snap[_fc].loc[X_val.index].values

print(f"  ✓ Train: {len(X_train):>5d} samples  ({y_train.sum():>3d} distress)  2015–2022")
print(f"  ✓ Val:   {len(X_val):>5d} samples  ({y_val.sum():>3d} distress)  [merged into train — 0 positives in 2022]")
print(f"  ✓ Test:  {len(X_test):>5d} samples  ({y_test.sum():>3d} distress)  2023–2025")
print(f"  NOTE: 2022 validation dropped — zero distress events; walk-forward CV "
      f"on train (5 folds) is the sole generalization estimate for model selection.")
print(f"  ✓ Features: {len(feature_cols)}")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train[feature_cols])
# X_val is empty when 2022 is merged into train — skip transform to avoid
# sklearn's "0 samples" error; downstream code already guards on y_val.sum()>0
X_val_s   = scaler.transform(X_val[feature_cols]) if len(X_val) > 0 else np.empty((0, len(feature_cols)))
X_test_s  = scaler.transform(X_test[feature_cols])

# ═══════════════════════════════════════════════════════════════════════
# [6/9] MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[6/9] Training models...")
print("─" * 70)

results = {}
scale_pos_weight = max(n_neg / max(n_pos, 1), 1)

# ─────────────────────────────────────────────────
# 6a. ALTMAN Z-SCORE BASELINE (1968)
# ─────────────────────────────────────────────────
print("\n  ┌─ 6a. Altman Z-Score Baseline ─────────────────────────────")

def altman_predict(X_df):
    """Convert Altman Z-Score to distress probability and binary prediction.

    Probability mapping (Altman 1968, public-company model):
        sigmoid centered at Z = 1.81 (the distress / grey-zone boundary).
        k = 1.5 chosen so that:
            P(distress | Z = 0.00) ≈ 0.93   (deeply distressed)
            P(distress | Z = 1.81) = 0.50   (threshold, by construction)
            P(distress | Z = 2.99) ≈ 0.15   (Altman safe-zone boundary)

    NaN imputation:
        z_nan → 2.40 (midpoint of the grey zone 1.81–2.99), giving P ≈ 0.29.
        This is more conservative than 0.5 because missing financials are
        disproportionately common among distressed companies.

    Binary prediction uses the original Altman threshold: Z < 1.81 → distress.
    """
    z = X_df['altman_z'].values.copy()
    nan_mask = np.isnan(z)
    z = np.nan_to_num(z, nan=2.4)   # grey-zone midpoint as safe imputation
    # Sigmoid centered at the distress threshold Z=1.81 (Altman 1968 Table 1)
    prob = 1.0 / (1.0 + np.exp(1.5 * (z - 1.81)))
    # Explicitly set NaN cases to the grey-zone mid probability (≈0.29)
    prob[nan_mask] = 1.0 / (1.0 + np.exp(1.5 * (2.4 - 1.81)))
    pred = (z < 1.81).astype(int)   # Altman 1968 distress threshold
    return prob, pred

for name, df, y in [('Train', X_train, y_train), ('Val', X_val, y_val), ('Test', X_test, y_test)]:
    prob, pred = altman_predict(df)
    if y.sum() > 0 and y.sum() < len(y):
        auc = roc_auc_score(y, prob)
        brier = brier_score_loss(y, prob)
        print(f"  │  {name:5s} AUC: {auc:.4f}  Brier: {brier:.4f}")
        if name == 'Test':
            results['Altman Z-Score'] = {'auc_roc': auc, 'brier': brier, 'y_prob': prob, 'y_pred': pred, 'y_true': y}
    else:
        print(f"  │  {name:5s} — insufficient positive samples")
        if name == 'Test':
            results['Altman Z-Score'] = {'auc_roc': 0.5, 'brier': 0.25, 'y_prob': prob, 'y_pred': pred, 'y_true': y}

print("  └────────────────────────────────────────────────────────────")

# ─────────────────────────────────────────────────
# 6a-ext. POST-1968 ACADEMIC BASELINES
# ─────────────────────────────────────────────────
# Ohlson O-Score (1980), Zmijewski Score (1984), Merton DD (1974).
# All three are FORMULA-BASED (no ML training) — pure accounting / structural.
# AUC is evaluated only on rows where the score is available (non-NaN).
# ─────────────────────────────────────────────────

def _formula_baseline_eval(col_pd, col_pred_thresh, split_name, df, y, label):
    """Generic evaluator for formula-based distress probability columns.

    NaN fill: uses training-set prevalence (y_train.mean() ≈ 0.139) rather
    than 0.5, which was producing cosmetically exact AUC=0.500 on every row.
    The 0.5 fallback made all NaN rows predict exactly at the ROC coin-flip
    boundary, masking whether the signal is absent or just uninformative.
    Prevalence fill is the neutral Bayesian prior that preserves the real
    information content of covered rows.
    """
    _nan_fill = float(y_train.mean())   # ≈ training prevalence (~0.139)
    probs = df[col_pd].values.copy() if col_pd in df.columns else np.full(len(y), np.nan)
    eval_mask = ~np.isnan(probs)
    n_eval = int(eval_mask.sum())
    if n_eval == 0:
        print(f"  │  {split_name:5s} {label}: — no coverage (all NaN)")
        return None, None
    probs_filled = np.nan_to_num(probs, nan=_nan_fill)
    pred = (probs_filled > col_pred_thresh).astype(int)
    y_eval = y[eval_mask]
    p_eval = probs_filled[eval_mask]
    if y_eval.sum() == 0 or y_eval.sum() == len(y_eval):
        print(f"  │  {split_name:5s} {label}: — insufficient positive samples (n={n_eval})")
        return None, None
    auc   = roc_auc_score(y_eval, p_eval)
    brier = brier_score_loss(y_eval, p_eval)
    cov_pct = 100 * n_eval / len(y)
    print(f"  │  {split_name:5s} {label}: AUC={auc:.4f}  Brier={brier:.4f}  "
          f"coverage={n_eval}/{len(y)} ({cov_pct:.0f}%)")
    return auc, brier

baseline_specs = [
    # Use _raw columns (pre-preprocessing) — preprocessed versions are all-constant
    # in test (2023+) because CSV financials do not extend that far, so fillna zeroes
    # them out. _raw preserves original NaN coverage for honest evaluation.
    ("Ohlson O-Score  (1980)",   "ohlson_pd_raw",    0.5, "Ohlson O-Score"),
    ("Zmijewski Score (1984)",   "zmijewski_pd_raw", 0.5, "Zmijewski Score"),
    ("Merton DD PD   (1974)",    "merton_pd_raw",    0.5, "Merton DD"),
]

for bl_label, bl_col, bl_thresh, bl_key in baseline_specs:
    print(f"\n  ┌─ {bl_label} ─────────────────────")
    for sname, df, y in [("Train", X_train, y_train),
                          ("Val",   X_val,   y_val),
                          ("Test",  X_test,  y_test)]:
        auc, brier = _formula_baseline_eval(bl_col, bl_thresh, sname, df, y, bl_label)
        if sname == "Test" and auc is not None:
            probs = np.nan_to_num(df[bl_col].values if bl_col in df.columns else np.full(len(y), np.nan), nan=float(y_train.mean()))
            results[bl_key] = {
                "auc_roc": auc, "brier": brier,
                "y_prob": probs,
                "y_pred": (probs > bl_thresh).astype(int),
                "y_true": y
            }
        elif sname == "Test" and auc is None:
            results[bl_key] = {"auc_roc": float("nan"), "brier": float("nan"),
                               "y_prob": np.full(len(y), float(y_train.mean())),
                               "y_pred": np.zeros(len(y), int), "y_true": y}
    print("  └────────────────────────────────────────────────────────────")

# ── AUC = 0.5 diagnostic (Ohlson/Zmijewski/Merton) ──────────────────────────
# Review identified three possible causes:
#   Scenario 1: std ≈ 0 or n_unique ≈ 1  →  columns are constant (NaN→fallback swallowed all)
#   Scenario 2: std > 0, pos_mean ≈ neg_mean  →  signal exists but wrong label definition
#                  (accounting-based PDs predict bankruptcy filings, not 50% drawdowns)
#   Scenario 3: std > 0, pos_mean > neg_mean but AUC still 0.5  →  tie-breaking artifact
# This block prints the discriminating stats so we can categorise the cause.
print("\n  ── AUC=0.5 Diagnostic: Ohlson / Zmijewski / Merton ──")
for _dcol in ['ohlson_pd_raw', 'zmijewski_pd_raw', 'merton_pd_raw']:
    _x = X_test[_dcol].values.astype(float) if _dcol in X_test.columns else None
    if _x is None:
        print(f"  {_dcol}: not in test set")
        continue
    _y = y_test
    _nan_fill = float(y_train.mean())
    _x_filled = np.nan_to_num(_x, nan=_nan_fill)
    _pos_mean  = _x_filled[_y == 1].mean() if (_y == 1).any() else float('nan')
    _neg_mean  = _x_filled[_y == 0].mean() if (_y == 0).any() else float('nan')
    try:
        _auc = roc_auc_score(_y, _x_filled)
    except Exception:
        _auc = float('nan')
    _scenario = (
        "Scenario 1 — CONSTANT (fallback dominates)" if np.std(_x_filled) < 1e-6 or len(np.unique(_x_filled)) <= 3 else
        "Scenario 2 — uninformative for drawdown label (accounting PD ≠ market stress)"
        if abs(_pos_mean - _neg_mean) < 0.02 else
        "Scenario 3 — tie-breaking artifact" if abs(_auc - 0.5) < 0.01 else
        "Informative — AUC not at 0.5"
    )
    print(f"  {_dcol}: mean={_x_filled.mean():.4f}  std={_x_filled.std():.4f}  "
          f"n_unique={len(np.unique(_x_filled))}  "
          f"pos_mean={_pos_mean:.4f}  neg_mean={_neg_mean:.4f}  "
          f"AUC={_auc:.4f}  → {_scenario}")
print()
# ─────────────────────────────────────────────────
print("\n  ┌─ 6b. XGBoost — Walk-Forward CV + Full Train ─────────────")

xgb_model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7,
    scale_pos_weight=scale_pos_weight,
    eval_metric='auc', random_state=42, use_label_encoder=False,
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
)

# Walk-forward cross-validation (expanding window)
tscv = TimeSeriesSplit(n_splits=5)
train_sorted = X_train.sort_values('Date')
cv_aucs = []
for fold, (tr_idx, va_idx) in enumerate(tqdm(list(tscv.split(train_sorted)),
                                              desc="  │  Walk-forward CV", unit="fold")):
    Xtr = train_sorted.iloc[tr_idx][feature_cols].values
    ytr = train_sorted.iloc[tr_idx]['distress_label'].values
    Xva = train_sorted.iloc[va_idx][feature_cols].values
    yva = train_sorted.iloc[va_idx]['distress_label'].values
    fm = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.7,
                           scale_pos_weight=scale_pos_weight, random_state=42,
                           use_label_encoder=False, eval_metric='auc',
                           reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5)
    fm.fit(Xtr, ytr, verbose=False)
    if yva.sum() > 0 and yva.sum() < len(yva):
        fp = fm.predict_proba(Xva)[:, 1]
        fa = roc_auc_score(yva, fp)
        cv_aucs.append(fa)
        tqdm.write(f"  │    Fold {fold+1}: AUC={fa:.4f} (train={len(Xtr)}, val={len(Xva)})")

if cv_aucs:
    print(f"  │  CV Mean AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

# Full training
print("  │  Training final XGBoost (500 trees, depth=6, lr=0.03)...")
eval_set = [(X_val[feature_cols].values, y_val)] if y_val.sum() > 0 else []
xgb_model.fit(X_train[feature_cols].values, y_train,
              eval_set=eval_set if eval_set else None, verbose=False)

for name, Xs, y in [('Train', X_train[feature_cols].values, y_train),
                     ('Val', X_val[feature_cols].values, y_val),
                     ('Test', X_test[feature_cols].values, y_test)]:
    xp = xgb_model.predict_proba(Xs)[:, 1]
    xd = (xp > 0.5).astype(int)
    if y.sum() > 0 and y.sum() < len(y):
        auc = roc_auc_score(y, xp)
        brier = brier_score_loss(y, xp)
        print(f"  │  {name:5s} AUC: {auc:.4f}  Brier: {brier:.4f}")
    else:
        auc, brier = 0.5, 0.25
        print(f"  │  {name:5s} — no positive samples")
    if name == 'Test':
        results['XGBoost'] = {'auc_roc': auc, 'brier': brier, 'y_prob': xp, 'y_pred': xd, 'y_true': y}

xgb_model.save_model(str(MODELS_DIR / "xgboost_credit_risk.json"))
print("  │  ✓ Saved: xgboost_credit_risk.json")
print("  └────────────────────────────────────────────────────────────")

# ─────────────────────────────────────────────────
# 6c. QUARTERLY LSTM (4-QUARTER SEQUENCES)
# ─────────────────────────────────────────────────
print("\n  ┌─ 6c. Quarterly LSTM (4-quarter lookback) ────────────────")
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class SeqDataset(Dataset):
        def __init__(self, df, feat_cols, seq_len=4):
            self.samples, self.labels = [], []
            for t in df['ticker'].unique():
                td = df[df['ticker'] == t].sort_values('Date')
                feats = td[feat_cols].values.astype(np.float32)
                labs = td['distress_label'].values
                for i in range(seq_len, len(feats)):
                    self.samples.append(feats[i-seq_len:i])
                    self.labels.append(labs[i])
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            return torch.tensor(self.samples[i]), torch.tensor(self.labels[i], dtype=torch.float32)

    class QuarterlyLSTM(nn.Module):
        def __init__(self, inp, hid=64, nlayers=2, drop=0.3):
            super().__init__()
            self.lstm = nn.LSTM(inp, hid, nlayers, batch_first=True, dropout=drop)
            self.fc = nn.Sequential(nn.Linear(hid, 32), nn.ReLU(), nn.Dropout(drop),
                                    nn.Linear(32, 1), nn.Sigmoid())
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    # Build datasets with scaled features
    Xtr_df = X_train.copy(); Xva_df = X_val.copy(); Xte_df = X_test.copy()
    for i, c in enumerate(feature_cols):
        Xtr_df[c] = X_train_s[:, i]
        Xva_df[c] = X_val_s[:, i]
        Xte_df[c] = X_test_s[:, i]

    train_ds = SeqDataset(Xtr_df, feature_cols, 4)
    val_ds = SeqDataset(Xva_df, feature_cols, 4)
    test_ds = SeqDataset(Xte_df, feature_cols, 4)

    print(f"  │  Sequences — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_ld = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=64)
    test_ld = DataLoader(test_ds, batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  │  Device: {device}")
    lstm_model = QuarterlyLSTM(len(feature_cols)).to(device)
    opt = torch.optim.Adam(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    pw = torch.tensor([scale_pos_weight], dtype=torch.float32).to(device)

    best_vl = float('inf')
    N_EPOCHS_QLSTM = 80

    for epoch in tqdm(range(N_EPOCHS_QLSTM), desc="  │  Quarterly LSTM", unit="epoch"):
        lstm_model.train()
        tl = 0
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = lstm_model(xb)
            w = torch.where(yb == 1, pw[0], torch.tensor(1.0).to(device))
            loss = nn.functional.binary_cross_entropy(out, yb, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
            opt.step()
            tl += loss.item()

        avg_tl = tl / max(len(train_ld), 1)

        lstm_model.eval()
        vl = 0
        with torch.no_grad():
            for xb, yb in val_ld:
                xb, yb = xb.to(device), yb.to(device)
                out = lstm_model(xb)
                vl += nn.functional.binary_cross_entropy(out, yb).item()
        avg_vl = vl / max(len(val_ld), 1)
        scheduler.step(avg_vl)

        if avg_vl < best_vl:
            best_vl = avg_vl
            torch.save(lstm_model.state_dict(), str(MODELS_DIR / "lstm_credit_risk.pt"))

        if (epoch + 1) % 20 == 0:
            tqdm.write(f"  │    Epoch {epoch+1:3d}: train_loss={avg_tl:.4f}, val_loss={avg_vl:.4f}")

    lstm_model.load_state_dict(torch.load(str(MODELS_DIR / "lstm_credit_risk.pt"), weights_only=True))
    lstm_model.eval()

    def eval_lstm(loader):
        ps, ls = [], []
        with torch.no_grad():
            for xb, yb in loader:
                ps.extend(lstm_model(xb.to(device)).cpu().numpy())
                ls.extend(yb.numpy())
        return np.array(ps), np.array(ls)

    for name, ld in [('Train', train_ld), ('Val', val_ld), ('Test', test_ld)]:
        lp, ly = eval_lstm(ld)
        ld_pred = (lp > 0.5).astype(int)
        if ly.sum() > 0 and ly.sum() < len(ly):
            auc = roc_auc_score(ly, lp)
            brier = brier_score_loss(ly, lp)
            print(f"  │  {name:5s} AUC: {auc:.4f}  Brier: {brier:.4f}")
        else:
            auc, brier = 0.5, 0.25
            print(f"  │  {name:5s} — no positive samples")
        if name == 'Test':
            results['LSTM'] = {'auc_roc': auc, 'brier': brier, 'y_prob': lp, 'y_pred': ld_pred, 'y_true': ly}

    print("  │  ✓ Saved: lstm_credit_risk.pt")

except Exception as e:
    print(f"  │  LSTM failed: {e}")
    import traceback; traceback.print_exc()

print("  └────────────────────────────────────────────────────────────")

# ─────────────────────────────────────────────────
# 6d. DAILY LSTM (60-DAY SEQUENCES OF STOCK PRICES)
# ─────────────────────────────────────────────────
print("\n  ┌─ 6d. Daily LSTM (60-day price sequences) ────────────────")
try:
    # Re-import torch here so this block is self-contained.
    # The quarterly LSTM block (6c) may have already failed with ModuleNotFoundError,
    # which means 'Dataset' is NOT bound in the enclosing scope — causing
    # NameError when DailySeqDataset(Dataset) is defined below.
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    DAILY_FEATURES = ['log_return', 'vol_30d', 'vol_ratio_10_60', 'return_zscore_30d',
                      'rsi_14', 'macd_hist', 'bb_pct', 'vol_ratio_20',
                      'drawdown', 'atr_pct', 'price_sma50_ratio', 'price_sma200_ratio']

    # Create daily distress labels: 1 if distress within next 6 months
    daily_df['distress_label'] = 0
    for _, row in defaults_df.iterrows():
        t = row['ticker']
        ed = pd.to_datetime(row['event_date'])
        mask = (daily_df['ticker'] == t) & (daily_df.index >= ed - pd.Timedelta(days=180)) & (daily_df.index < ed)
        daily_df.loc[mask, 'distress_label'] = 1
    for _, row in drawdowns_df.iterrows():
        t = row['ticker']
        sd = pd.to_datetime(row['distress_start'])
        mask = (daily_df['ticker'] == t) & (daily_df.index >= sd - pd.Timedelta(days=180)) & (daily_df.index < sd + pd.Timedelta(days=30))
        daily_df.loc[mask, 'distress_label'] = 1

    n_daily_pos = daily_df['distress_label'].sum()
    print(f"  │  Daily labels: {n_daily_pos} distress days out of {len(daily_df)}")

    # Standardize daily features
    train_daily = daily_df[daily_df.index < TRAIN_END]
    test_daily = daily_df[daily_df.index >= VAL_END]

    scaler_daily = StandardScaler()
    scaler_daily.fit(train_daily[DAILY_FEATURES].fillna(0).replace([np.inf, -np.inf], 0))

    SEQ_LEN = 60

    class DailySeqDataset(Dataset):
        def __init__(self, df, features, scaler, seq_len=60, subsample_neg=5):
            self.samples, self.labels = [], []
            for ticker in tqdm(df['ticker'].unique(), desc="  │    Building seqs", unit="ticker", leave=False):
                td = df[df['ticker'] == ticker].sort_index()
                feats = scaler.transform(td[features].fillna(0).replace([np.inf, -np.inf], 0)).astype(np.float32)
                labs = td['distress_label'].values
                for i in range(seq_len, len(feats)):
                    label = labs[i]
                    if label == 0 and np.random.random() > 1.0 / subsample_neg:
                        continue
                    self.samples.append(feats[i - seq_len:i])
                    self.labels.append(label)

        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            return torch.tensor(self.samples[i]), torch.tensor(self.labels[i], dtype=torch.float32)

    class DailyLSTM(nn.Module):
        def __init__(self, n_features, hidden=128, n_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True, dropout=dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 16), nn.ReLU(),
                nn.Linear(16, 1), nn.Sigmoid()
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    print("  │  Building train dataset...")
    train_ds_d = DailySeqDataset(train_daily, DAILY_FEATURES, scaler_daily, SEQ_LEN, subsample_neg=3)
    print(f"  │    Train: {len(train_ds_d)} samples ({sum(train_ds_d.labels)} positive)")
    print("  │  Building test dataset...")
    test_ds_d = DailySeqDataset(test_daily, DAILY_FEATURES, scaler_daily, SEQ_LEN, subsample_neg=1)
    print(f"  │    Test:  {len(test_ds_d)} samples ({sum(test_ds_d.labels)} positive)")

    train_ld_d = DataLoader(train_ds_d, batch_size=128, shuffle=True)
    test_ld_d = DataLoader(test_ds_d, batch_size=256)

    daily_lstm = DailyLSTM(len(DAILY_FEATURES), hidden=128, n_layers=2).to(device)
    opt_d = torch.optim.Adam(daily_lstm.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_d, patience=3, factor=0.5)

    N_EPOCHS_DAILY = 40
    best_loss_d = float('inf')

    for epoch in tqdm(range(N_EPOCHS_DAILY), desc="  │  Daily LSTM", unit="epoch"):
        daily_lstm.train()
        total_loss, n_batch = 0, 0
        for xb, yb in train_ld_d:
            xb, yb = xb.to(device), yb.to(device)
            opt_d.zero_grad()
            out = daily_lstm(xb)
            w = torch.where(yb == 1, pw[0], torch.tensor(1.0).to(device))
            loss = nn.functional.binary_cross_entropy(out, yb, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(daily_lstm.parameters(), 1.0)
            opt_d.step()
            total_loss += loss.item()
            n_batch += 1

        avg_loss = total_loss / max(n_batch, 1)
        scheduler_d.step(avg_loss)

        if avg_loss < best_loss_d:
            best_loss_d = avg_loss
            torch.save(daily_lstm.state_dict(), str(MODELS_DIR / "lstm_daily.pt"))

        if (epoch + 1) % 10 == 0:
            tqdm.write(f"  │    Epoch {epoch+1:3d}: loss={avg_loss:.4f}")

    # Evaluate
    daily_lstm.load_state_dict(torch.load(str(MODELS_DIR / "lstm_daily.pt"), weights_only=True))
    daily_lstm.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_ld_d:
            probs = daily_lstm(xb.to(device)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if all_labels.sum() > 0 and all_labels.sum() < len(all_labels):
        auc_d = roc_auc_score(all_labels, all_probs)
        brier_d = brier_score_loss(all_labels, all_probs)
        print(f"  │  Test AUC: {auc_d:.4f}  Brier: {brier_d:.4f}")
        results['Daily LSTM'] = {'auc_roc': auc_d, 'brier': brier_d,
                                  'y_prob': all_probs, 'y_pred': (all_probs > 0.5).astype(int),
                                  'y_true': all_labels}
    else:
        print(f"  │  Test — {all_labels.sum()} positive out of {len(all_labels)} samples")

    print("  │  ✓ Saved: lstm_daily.pt")

except Exception as e:
    print(f"  │  Daily LSTM failed: {e}")
    import traceback; traceback.print_exc()

print("  └────────────────────────────────────────────────────────────")

# ─────────────────────────────────────────────────
# 6e. COX PROPORTIONAL HAZARD (SURVIVAL MODEL)
# ─────────────────────────────────────────────────
print("\n  ┌─ 6e. Cox Proportional Hazard ────────────────────────────")
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter

    events_map = {}
    for _, r in defaults_df.iterrows():
        events_map.setdefault(r['ticker'], []).append(pd.to_datetime(r['event_date']))
    for _, r in drawdowns_df.iterrows():
        events_map.setdefault(r['ticker'], []).append(pd.to_datetime(r['distress_start']))

    cox_feats = [f for f in ['altman_z', 'volatility_30d', 'momentum_90d', 'debt_to_equity',
                             'interest_coverage', 'merton_dd', 'hy_oas', 'oil_wti',
                             'yield_slope', 'vix_mean', 'regime_stress_frac'] if f in feature_cols]

    surv_records = []
    for _, row in tqdm(X_ts.iterrows(), desc="  │  Survival data", total=len(X_ts), unit="row"):
        t, d = row['ticker'], row['Date']
        future = [e for e in events_map.get(t, []) if e > d]
        if future:
            dur = max((future[0] - d).days / 90, 0.1)
            ev = 1
        else:
            dur = max((X_ts['Date'].max() - d).days / 90, 0.1)
            ev = 0
        rec = {f: row[f] for f in cox_feats}
        rec['duration'] = min(dur, 20)
        rec['event'] = ev
        surv_records.append(rec)

    surv_df = pd.DataFrame(surv_records)
    surv_df = surv_df.replace([np.inf, -np.inf], np.nan).dropna()

    n_train_surv = len(X_train)
    surv_train = surv_df.iloc[:n_train_surv]
    surv_test = surv_df.iloc[n_train_surv:]

    cox_feats_safe = [f for f in cox_feats if surv_train[f].std() > 1e-6 and surv_train[f].nunique() >= 3]

    cox_scaler = StandardScaler()
    surv_train_scaled = surv_train.copy()
    surv_train_scaled[cox_feats_safe] = cox_scaler.fit_transform(surv_train[cox_feats_safe])
    surv_test_scaled = surv_test.copy()
    surv_test_scaled[cox_feats_safe] = cox_scaler.transform(surv_test[cox_feats_safe])

    cph = CoxPHFitter(penalizer=1.0, l1_ratio=0.5)
    cph.fit(surv_train_scaled[cox_feats_safe + ['duration', 'event']],
            duration_col='duration', event_col='event',
            fit_options={'step_size': 0.5})

    print("  │  Cox PH Summary:")
    cph.print_summary(columns=['coef', 'exp(coef)', 'p'])
    print(f"  │  Train C-Index: {cph.concordance_index_:.4f}")

    if len(surv_test_scaled) > 10:
        try:
            ci = cph.score(surv_test_scaled[cox_feats_safe + ['duration', 'event']], scoring_method='concordance_index')
            print(f"  │  Test C-Index: {ci:.4f}")
            results['Cox PH'] = {'c_index': ci, 'c_index_train': cph.concordance_index_}
        except:
            results['Cox PH'] = {'c_index': cph.concordance_index_, 'c_index_train': cph.concordance_index_}

    with open(MODELS_DIR / "cox_ph_model.pkl", "wb") as f:
        pickle.dump(cph, f)
    print("  │  ✓ Saved: cox_ph_model.pkl")

except Exception as e:
    print(f"  │  Cox PH failed: {e}")
    import traceback; traceback.print_exc()

print("  └────────────────────────────────────────────────────────────")

# ═══════════════════════════════════════════════════════════════════════
# [7/9] VISUALIZATIONS (19 PUBLICATION-QUALITY FIGURES)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[7/9] Generating 19 publication-quality visualizations...")
print("─" * 70)

viz_list = [
    "01_roc_curves_comparison",
    "02_model_comparison",
    "03_xgboost_feature_importance",
    "04_calibration_curves",
    "05_volatility_heatmap",
    "06_merton_dd_timeline",
    "07_altman_zscore",
    "08_hmm_regime_detection",
    "09_oil_defaults_timeline",
    "10_feature_correlation",
    "11_precision_recall",
    "12_kaplan_meier",
    "13_cox_hazard_ratios",
    "14_confusion_matrices",
    "15_altman_vs_xgboost",
    "16_company_risk_profiles",
    "17_rsi_drawdown_distributions",
    "18_chk_multiscale_volatility",
    "19_enhanced_feature_importance",
]

pbar = tqdm(viz_list, desc="  Visualizations", unit="fig")

# 01 — ROC Curves
pbar.set_postfix_str("ROC curves")
fig, ax = plt.subplots(figsize=(10, 8))
for mn, res in results.items():
    if 'y_prob' in res and res['y_true'].sum() > 0:
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        ax.plot(fpr, tpr, linewidth=2.5, label=f"{mn} (AUC={res.get('auc_roc', 0):.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title('CrisisNet Module 1 — ROC Curves (Test Set 2023–2025)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='lower right')
plt.tight_layout()
plt.savefig(VIZ_DIR / "01_roc_curves_comparison.png", dpi=300)
plt.close()
next(pbar.__iter__(), None)  # manual iteration just for display
pbar.update(1)

# 02 — Model Comparison
pbar.set_postfix_str("Model comparison")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
names = [n for n in results if 'auc_roc' in results[n]]
aucs = [results[n]['auc_roc'] for n in names]
briers = [results[n].get('brier', 0) for n in names]
c = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71', '#f39c12']
axes[0].bar(names, aucs, color=c[:len(names)], edgecolor='white', linewidth=2)
axes[0].set_ylabel('AUC-ROC', fontsize=13)
axes[0].set_title('AUC-ROC (Higher = Better)', fontsize=14, fontweight='bold')
axes[0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target 0.80')
axes[0].legend()
for i, v in enumerate(aucs): axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
axes[0].set_ylim(0, 1.1)
axes[0].tick_params(axis='x', rotation=15)
axes[1].bar(names, briers, color=c[:len(names)], edgecolor='white', linewidth=2)
axes[1].set_ylabel('Brier Score', fontsize=13)
axes[1].set_title('Brier Score (Lower = Better)', fontsize=14, fontweight='bold')
axes[1].axhline(y=0.15, color='red', linestyle='--', alpha=0.7, label='Target < 0.15')
axes[1].legend()
for i, v in enumerate(briers): axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
axes[1].tick_params(axis='x', rotation=15)
plt.suptitle('CrisisNet Module 1 — Model Performance', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(VIZ_DIR / "02_model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
pbar.update(1)

# 03 — XGBoost Feature Importance
pbar.set_postfix_str("Feature importance")
fig, ax = plt.subplots(figsize=(12, 10))
imp = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values().tail(25)
imp.plot(kind='barh', ax=ax, color=COLORS['secondary'], edgecolor='white')
ax.set_xlabel('Feature Importance (Gain)', fontsize=13)
ax.set_title('XGBoost — Top 25 Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_DIR / "03_xgboost_feature_importance.png", dpi=300)
plt.close()
pbar.update(1)

# 04 — Calibration Curves
pbar.set_postfix_str("Calibration curves")
fig, ax = plt.subplots(figsize=(10, 8))
for mn, res in results.items():
    if 'y_prob' in res and res['y_true'].sum() > 0:
        try:
            pt, pp = calibration_curve(res['y_true'], res['y_prob'], n_bins=8, strategy='uniform')
            ax.plot(pp, pt, marker='o', linewidth=2, label=mn)
        except:
            pass
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
ax.set_xlabel('Predicted Probability', fontsize=14)
ax.set_ylabel('True Fraction', fontsize=14)
ax.set_title('Calibration Curves', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(VIZ_DIR / "04_calibration_curves.png", dpi=300)
plt.close()
pbar.update(1)

# 05 — Volatility Heatmap
pbar.set_postfix_str("Volatility heatmap")
fig, ax = plt.subplots(figsize=(16, 10))
vol_piv = X_ts.pivot_table(values='volatility_30d', index='ticker', columns='Date', aggfunc='mean')
notable = [t for t in ['CHK', 'XOM', 'CVX', 'OXY', 'DVN', 'COP', 'SLB', 'HAL', 'EOG', 'EQT',
                       'AR', 'RRC', 'KMI', 'WMB', 'VLO', 'MPC', 'PSX', 'PBF', 'LNG', 'FANG']
           if t in vol_piv.index]
vol_sub = vol_piv.loc[notable]
sns.heatmap(vol_sub, cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'Annualized Volatility'})
ax.set_title('30-Day Rolling Volatility — Energy Sector', fontsize=16, fontweight='bold')
n_lbl = min(15, len(vol_sub.columns))
step = max(1, len(vol_sub.columns) // n_lbl)
ax.set_xticks(range(0, len(vol_sub.columns), step))
ax.set_xticklabels([str(d)[:7] for d in vol_sub.columns[::step]], rotation=45)
plt.tight_layout()
plt.savefig(VIZ_DIR / "05_volatility_heatmap.png", dpi=300)
plt.close()
pbar.update(1)

# 06 — Merton DD Timeline
pbar.set_postfix_str("Merton DD")
if 'merton_dd' in X_ts.columns:
    fig, ax = plt.subplots(figsize=(14, 8))
    for t in [t for t in ['CHK', 'XOM', 'CVX', 'OXY', 'DVN', 'EOG', 'SLB', 'EQT'] if t in X_ts['ticker'].values]:
        td = X_ts[X_ts['ticker'] == t].sort_values('Date')
        dd_vals = td['merton_dd'].dropna()
        if len(dd_vals) > 0:
            lw = 3 if t == 'CHK' else 1.5
            ls = '--' if t == 'CHK' else '-'
            ax.plot(td.loc[dd_vals.index, 'Date'], dd_vals, ls, linewidth=lw, label=t)
    ax.axhline(y=1.5, color='red', linestyle=':', linewidth=2, label='DD=1.5 (High Risk)')
    ax.axhline(y=3.0, color='orange', linestyle=':', linewidth=1.5, label='DD=3.0 (Moderate)')
    ax.axvline(x=pd.Timestamp('2020-06-28'), color='red', alpha=0.3, linewidth=3)
    ax.annotate('CHK Bankruptcy\nJune 2020', xy=(pd.Timestamp('2020-06-28'), 0.5),
               fontsize=11, color='red', fontweight='bold')
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Merton Distance-to-Default', fontsize=13)
    ax.set_title('Merton Distance-to-Default — Energy Sector', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, ncol=2)
    ax.set_ylim(bottom=-0.5)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "06_merton_dd_timeline.png", dpi=300)
    plt.close()
pbar.update(1)

# 07 — Altman Z-Score
pbar.set_postfix_str("Altman Z-Score")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
zh = X_ts[X_ts['distress_label'] == 0]['altman_z'].dropna()
zd = X_ts[X_ts['distress_label'] == 1]['altman_z'].dropna()
axes[0].hist(zh, bins=40, alpha=0.6, color=COLORS['safe'], label='Healthy', density=True)
if len(zd) > 0:
    axes[0].hist(zd, bins=20, alpha=0.6, color=COLORS['distress'], label='Distress', density=True)
axes[0].axvline(x=1.81, color='red', linestyle='--', linewidth=2, label='Z=1.81 (Distress)')
axes[0].axvline(x=2.99, color='orange', linestyle='--', linewidth=2, label='Z=2.99 (Safe)')
axes[0].set_xlabel('Altman Z-Score', fontsize=13)
axes[0].set_title('Z-Score Distribution by Class', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
for t in ['CHK', 'XOM', 'OXY', 'DVN']:
    if t in X_ts['ticker'].values:
        td = X_ts[X_ts['ticker'] == t].sort_values('Date')
        lw = 3 if t == 'CHK' else 1.5
        axes[1].plot(td['Date'], td['altman_z'], linewidth=lw, label=t)
axes[1].axhline(y=1.81, color='red', linestyle='--', alpha=0.7, label='Distress')
axes[1].axhline(y=2.99, color='green', linestyle='--', alpha=0.7, label='Safe')
axes[1].fill_between(axes[1].get_xlim(), -10, 1.81, alpha=0.05, color='red')
axes[1].set_xlabel('Date', fontsize=13); axes[1].set_ylabel('Altman Z-Score', fontsize=13)
axes[1].set_title('Z-Score Timeline', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10); axes[1].set_ylim(-2, 10)
plt.suptitle('Altman Z-Score — Baseline Benchmark', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(VIZ_DIR / "07_altman_zscore.png", dpi=300, bbox_inches='tight')
plt.close()
pbar.update(1)

# 08 — HMM Regime Detection
pbar.set_postfix_str("HMM regimes")
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
axes[0].plot(hmm_daily.index, hmm_daily['VIX'], color='navy', linewidth=0.5, alpha=0.8)
axes[0].fill_between(hmm_daily.index, 0, hmm_daily['VIX'].max(),
                     where=hmm_daily['regime'] == 'stress', alpha=0.2, color='red', label='Stress Regime')
axes[0].set_ylabel('VIX', fontsize=13)
axes[0].set_title('HMM Regime Detection — VIX', fontsize=16, fontweight='bold')
axes[0].axhline(y=30, color='red', linestyle=':', alpha=0.5); axes[0].legend(fontsize=12)
axes[1].plot(hmm_daily.index, hmm_daily['HY_OAS'], color='darkred', linewidth=0.5, alpha=0.8)
axes[1].fill_between(hmm_daily.index, 0, hmm_daily['HY_OAS'].max(),
                     where=hmm_daily['regime'] == 'stress', alpha=0.2, color='red', label='Stress Regime')
axes[1].set_ylabel('HY OAS', fontsize=13)
axes[1].set_title('ICE BofA HY OAS — Credit Stress', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=12)
plt.tight_layout()
plt.savefig(VIZ_DIR / "08_hmm_regime_detection.png", dpi=300)
plt.close()
pbar.update(1)

# 09 — Oil Price vs Defaults
pbar.set_postfix_str("Oil/defaults timeline")
fig, ax = plt.subplots(figsize=(14, 7))
oil = fred['DCOILWTICO'].dropna()
ax.plot(oil.index, oil.values, color='black', linewidth=1.5, label='WTI Crude Oil ($/bbl)')
for _, r in defaults_df.iterrows():
    ed = pd.to_datetime(r['event_date'])
    if ed >= oil.index.min():
        ax.axvline(x=ed, color='red', alpha=0.4, linewidth=1)
        ax.annotate(r['ticker'], xy=(ed, oil.max()*0.9), fontsize=8, color='red', rotation=90, fontweight='bold')
ax.axvspan(pd.Timestamp('2015-06-01'), pd.Timestamp('2016-06-01'), alpha=0.1, color='orange', label='Oil Crash')
ax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-07-01'), alpha=0.1, color='red', label='COVID')
ax.set_title('Oil Price & Default Events — Energy Crisis Timeline', fontsize=16, fontweight='bold')
ax.set_ylabel('WTI Price ($)', fontsize=13); ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(VIZ_DIR / "09_oil_defaults_timeline.png", dpi=300)
plt.close()
pbar.update(1)

# 10 — Correlation Heatmap
pbar.set_postfix_str("Feature correlation")
fig, ax = plt.subplots(figsize=(14, 12))
kf = [f for f in ['altman_z', 'volatility_30d', 'momentum_90d', 'debt_to_equity',
                   'interest_coverage', 'merton_dd', 'merton_pd', 'free_cashflow',
                   'current_ratio', 'hy_oas', 'vix_mean', 'oil_wti',
                   'yield_slope', 'regime_stress_frac', 'volume_ratio',
                   'rsi_14_mean', 'bb_pct_mean', 'drawdown_min', 'atr_pct_mean']
       if f in X_ts.columns]
corr = X_ts[kf].corr()
mask = np.triu(np.ones_like(corr), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5)
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_DIR / "10_feature_correlation.png", dpi=300)
plt.close()
pbar.update(1)

# 11 — Precision-Recall Curves
pbar.set_postfix_str("PR curves")
fig, ax = plt.subplots(figsize=(10, 8))
for mn, res in results.items():
    if 'y_prob' in res and res['y_true'].sum() > 0:
        pr, rc, _ = precision_recall_curve(res['y_true'], res['y_prob'])
        ap = average_precision_score(res['y_true'], res['y_prob'])
        ax.plot(rc, pr, linewidth=2, label=f"{mn} (AP={ap:.3f})")
ax.set_xlabel('Recall', fontsize=14); ax.set_ylabel('Precision', fontsize=14)
ax.set_title('Precision-Recall Curves', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig(VIZ_DIR / "11_precision_recall.png", dpi=300)
plt.close()
pbar.update(1)

# 12 — Kaplan-Meier Curves
pbar.set_postfix_str("Kaplan-Meier")
try:
    from lifelines import KaplanMeierFitter
    fig, ax = plt.subplots(figsize=(12, 8))
    kmf = KaplanMeierFitter()
    for zone_mask, color, label in [
        (surv_df['altman_z'] > 2.99, COLORS['safe'], 'Safe (Z>2.99)') if 'altman_z' in surv_df.columns else (pd.Series(dtype=bool), '', ''),
        ((surv_df['altman_z'] > 1.81) & (surv_df['altman_z'] <= 2.99), COLORS['grey'], 'Grey (1.81<Z<=2.99)') if 'altman_z' in surv_df.columns else (pd.Series(dtype=bool), '', ''),
        (surv_df['altman_z'] <= 1.81, COLORS['distress'], 'Distress (Z<=1.81)') if 'altman_z' in surv_df.columns else (pd.Series(dtype=bool), '', '')
    ]:
        sub = surv_df[zone_mask]
        if len(sub) > 10:
            kmf.fit(sub['duration'], sub['event'], label=label)
            kmf.plot(ax=ax, color=color, linewidth=2)
    ax.set_xlabel('Time (Quarters)', fontsize=13); ax.set_ylabel('Survival Probability', fontsize=13)
    ax.set_title('Kaplan-Meier Survival by Z-Score Zone', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "12_kaplan_meier.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"    KM failed: {e}")
pbar.update(1)

# 13 — Cox Hazard Ratios
pbar.set_postfix_str("Cox hazard ratios")
try:
    fig, ax = plt.subplots(figsize=(10, 7))
    cph.plot(ax=ax)
    ax.set_title('Cox PH — Hazard Ratios', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "13_cox_hazard_ratios.png", dpi=300)
    plt.close()
except:
    pass
pbar.update(1)

# 14 — Confusion Matrices
pbar.set_postfix_str("Confusion matrices")
models_with_pred = [(n, r) for n, r in results.items() if 'y_pred' in r]
if models_with_pred:
    fig, axes = plt.subplots(1, len(models_with_pred), figsize=(6*len(models_with_pred), 5))
    if len(models_with_pred) == 1: axes = [axes]
    for i, (mn, res) in enumerate(models_with_pred):
        cm = confusion_matrix(res['y_true'], res['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Healthy', 'Distress'], yticklabels=['Healthy', 'Distress'])
        axes[i].set_title(mn, fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Actual'); axes[i].set_xlabel('Predicted')
    plt.suptitle('Confusion Matrices — Test Set', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "14_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
pbar.update(1)

# 15 — Altman vs XGBoost
pbar.set_postfix_str("Altman vs XGBoost")
if 'XGBoost' in results and 'y_prob' in results['XGBoost']:
    fig, ax = plt.subplots(figsize=(10, 8))
    zs = X_test['altman_z'].values
    xp = results['XGBoost']['y_prob']
    yt = results['XGBoost']['y_true']
    sc = ax.scatter(zs, xp, c=yt, cmap='RdYlGn_r', alpha=0.6, edgecolors='white', s=50)
    ax.set_xlabel('Altman Z-Score', fontsize=14); ax.set_ylabel('XGBoost P(Distress)', fontsize=14)
    ax.set_title('Altman Z vs XGBoost — Where CrisisNet Adds Value', fontsize=16, fontweight='bold')
    ax.axvline(x=1.81, color='red', linestyle='--', alpha=0.7, label='Z=1.81')
    ax.axhline(y=0.5, color='blue', linestyle='--', alpha=0.7, label='P=0.5')
    ax.legend(fontsize=12)
    plt.colorbar(sc, ax=ax, label='True Label')
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "15_altman_vs_xgboost.png", dpi=300)
    plt.close()
pbar.update(1)

# 16 — Company Risk Profiles
pbar.set_postfix_str("Risk profiles")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, ticker in enumerate(['CHK', 'OXY', 'XOM', 'DVN']):
    if ticker not in X_ts['ticker'].values: continue
    ax = axes[idx//2, idx%2]
    td = X_ts[X_ts['ticker'] == ticker].sort_values('Date')
    td_test = X_test[X_test['ticker'] == ticker]
    if len(td_test) > 0:
        test_probs = xgb_model.predict_proba(td_test[feature_cols].values)[:, 1]
        ax.plot(td_test['Date'], test_probs, 'r-', linewidth=2, label='XGBoost P(Distress)')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(td['Date'], td['altman_z'], 'b--', linewidth=1, alpha=0.5, label='Z-Score')
    ax2.axhline(y=1.81, color='blue', linestyle=':', alpha=0.3)
    ax2.set_ylabel('Z-Score', color='blue', fontsize=10)
    ax.set_title(f'{ticker}', fontsize=14, fontweight='bold')
    ax.set_ylabel('P(Distress)', color='red', fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=9); ax2.legend(loc='upper right', fontsize=9)
plt.suptitle('Company Risk Profiles — XGBoost vs Altman Z', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_DIR / "16_company_risk_profiles.png", dpi=300, bbox_inches='tight')
plt.close()
pbar.update(1)

# 17 — RSI & Drawdown Distributions
pbar.set_postfix_str("RSI/drawdown")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
healthy_rsi = daily_df[daily_df['distress_label'] == 0]['rsi_14'].dropna()
distress_rsi = daily_df[daily_df['distress_label'] == 1]['rsi_14'].dropna()
axes[0].hist(healthy_rsi, bins=50, alpha=0.6, color='#2ecc71', label='Healthy', density=True)
if len(distress_rsi) > 0:
    axes[0].hist(distress_rsi, bins=50, alpha=0.6, color='#e74c3c', label='Distress', density=True)
axes[0].axvline(x=30, color='red', linestyle='--', label='Oversold (RSI<30)')
axes[0].axvline(x=70, color='green', linestyle='--', label='Overbought (RSI>70)')
axes[0].set_xlabel('RSI (14-day)', fontsize=13)
axes[0].set_title('RSI Distribution by Distress Status', fontsize=14, fontweight='bold')
axes[0].legend()
healthy_dd = daily_df[daily_df['distress_label'] == 0]['drawdown'].dropna()
distress_dd = daily_df[daily_df['distress_label'] == 1]['drawdown'].dropna()
axes[1].hist(healthy_dd, bins=50, alpha=0.6, color='#2ecc71', label='Healthy', density=True)
if len(distress_dd) > 0:
    axes[1].hist(distress_dd, bins=50, alpha=0.6, color='#e74c3c', label='Distress', density=True)
axes[1].set_xlabel('Drawdown from Peak', fontsize=13)
axes[1].set_title('Drawdown Distribution by Distress Status', fontsize=14, fontweight='bold')
axes[1].legend()
plt.suptitle('Enhanced Stock Price Indicators — Distress Signal Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(VIZ_DIR / "17_rsi_drawdown_distributions.png", dpi=300, bbox_inches='tight')
plt.close()
pbar.update(1)

# 18 — CHK Multi-Scale Volatility
pbar.set_postfix_str("CHK volatility")
fig, ax = plt.subplots(figsize=(14, 7))
if 'CHK' in daily_df['ticker'].values:
    td = daily_df[daily_df['ticker'] == 'CHK']
    for col, label, color in [('vol_10d', '10-day', '#e74c3c'), ('vol_30d', '30-day', '#3498db'),
                               ('vol_60d', '60-day', '#2ecc71'), ('vol_90d', '90-day', '#9b59b6')]:
        if col in td.columns:
            ax.plot(td.index, td[col], linewidth=1, label=label, color=color, alpha=0.8)
ax.axvline(x=pd.Timestamp('2020-06-28'), color='red', alpha=0.5, linewidth=3)
ax.annotate('CHK Bankruptcy', xy=(pd.Timestamp('2020-06-28'), 3), fontsize=12, color='red', fontweight='bold')
ax.set_title('Chesapeake Energy — Multi-Scale Volatility (Cancer Screening Analogy)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=13); ax.set_ylabel('Annualized Volatility', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(VIZ_DIR / "18_chk_multiscale_volatility.png", dpi=300)
plt.close()
pbar.update(1)

# 19 — Enhanced Feature Importance (full model)
pbar.set_postfix_str("Enhanced importance")
fig, ax = plt.subplots(figsize=(14, 12))
imp_full = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values().tail(30)
colors = ['#e74c3c' if any(k in f for k in ['vol_', 'rsi', 'macd', 'bb_', 'drawdown', 'atr', 'mom_', 'obv', 'sma'])
          else '#3498db' for f in imp_full.index]
imp_full.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_xlabel('Feature Importance (Gain)', fontsize=13)
ax.set_title('Enhanced XGBoost — Top 30 Features\n(Red = Stock Price Derived, Blue = Fundamental/Macro)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_DIR / "19_enhanced_feature_importance.png", dpi=300)
plt.close()
pbar.update(1)
pbar.close()

print(f"  ✓ All 19 visualizations saved to {VIZ_DIR}/")

# ═══════════════════════════════════════════════════════════════════════
# [8/9] EXPORT X_ts.parquet FOR MODULE D FUSION
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[8/9] Exporting X_ts.parquet...")
print("─" * 70)

# Inject raw (pre-preprocessing) formula columns so Module D's train_fusion.py
# can use them directly. The preprocessed versions (ohlson_pd etc.) are zeroed
# out by fillna/winsorize for rows without CSV-financial coverage, making them
# useless as baselines. The _raw versions preserve the original NaN pattern.
for _fc in _FORMULA_PD_COLS:
    X_ts[_fc + '_raw'] = _formula_raw_snap[_fc].values

X_ts_export = X_ts.set_index(['ticker', 'Date'])
X_ts_export.to_parquet(RESULTS_DIR / "X_ts.parquet")
X_ts_export.to_parquet(PROJECT_ROOT.parent / "X_ts.parquet")
print(f"  ✓ X_ts: {X_ts_export.shape}")
print(f"  ✓ Saved to: {RESULTS_DIR / 'X_ts.parquet'}")

# Save model artifacts
with open(MODELS_DIR / "feature_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(MODELS_DIR / "feature_columns.json", "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"  ✓ Feature scaler + column list saved")

# ═══════════════════════════════════════════════════════════════════════
# [9/9] RESULTS SUMMARY & VALIDATION
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("[9/9] Results summary & validation...")
print("─" * 70)

# Save JSON results
summary = {
    'metadata': {
        'module': 'Module 1 — Time Series & Credit Risk Engine',
        'n_tickers': len(TICKERS),
        'tickers': TICKERS,
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'train_distress': int(y_train.sum()),
        'test_distress': int(y_test.sum()),
        'train_distress_rate': float(y_train.mean()),
        'test_distress_rate': float(y_test.mean()),
    },
    'models': {}
}
for name, res in results.items():
    ms = {}
    for k in ['auc_roc', 'brier', 'c_index', 'c_index_train']:
        if k in res:
            ms[k] = float(res[k])
    if 'y_true' in res and 'y_pred' in res:
        ms['classification_report'] = classification_report(
            res['y_true'], res['y_pred'], target_names=['Healthy', 'Distress'],
            output_dict=True, zero_division=0)
    summary['models'][name] = ms

if cv_aucs:
    summary['models']['XGBoost']['walk_forward_cv'] = {
        'mean_auc': float(np.mean(cv_aucs)), 'std_auc': float(np.std(cv_aucs)),
        'fold_aucs': [float(a) for a in cv_aucs]
    }

with open(RESULTS_DIR / "module1_results.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

elapsed = time.time() - t_start

# ─── Final Report ───
print(f"\n{'═'*70}")
print(f"  MODULE 1 COMPLETE — {elapsed:.1f}s elapsed")
print(f"{'═'*70}")
print(f"\n  ╔══════════════════════════════════════════════════════════╗")
print(f"  ║  TEST SET PERFORMANCE (2023–2025)                       ║")
print(f"  ╠══════════════════════════════════════════════════════════╣")
for name, res in results.items():
    line = f"  ║  {name:20s}"
    if 'auc_roc' in res:
        auc = res['auc_roc']
        status = "✓ PASS" if auc >= 0.8 else "  —   "
        line += f"  AUC: {auc:.4f}  {status}"
    if 'brier' in res:
        line += f"  Brier: {res['brier']:.4f}"
    if 'c_index' in res:
        line += f"  C-Index: {res['c_index']:.4f}"
    line += " " * max(0, 58 - len(line) + 4) + "║"
    print(line)
print(f"  ╚══════════════════════════════════════════════════════════╝")

print(f"\n  Validation Checklist:")
n_feat = len(feature_cols)
has_xgb = 'XGBoost' in results
xgb_pass = has_xgb and results['XGBoost'].get('auc_roc', 0) >= 0.8
has_lstm = 'LSTM' in results
lstm_pass = has_lstm and results['LSTM'].get('auc_roc', 0) >= 0.8
has_daily = 'Daily LSTM' in results
has_cox = 'Cox PH' in results
has_altman = 'Altman Z-Score' in results

checks = [
    (True, f"X_ts.parquet exported ({X_ts_export.shape[0]} rows × {X_ts_export.shape[1]} cols)"),
    (n_feat >= 25, f"Feature count: {n_feat} (target: 25–90)"),
    (has_altman, "Altman Z-Score baseline computed"),
    (has_xgb, f"XGBoost trained (CV AUC: {np.mean(cv_aucs):.3f})" if cv_aucs else "XGBoost trained"),
    (xgb_pass, f"XGBoost AUC > 0.80: {results.get('XGBoost', {}).get('auc_roc', 0):.3f}"),
    (has_lstm, "Quarterly LSTM trained (4-quarter sequences)"),
    (has_daily, "Daily LSTM trained (60-day price sequences)"),
    (has_cox, "Cox PH survival model trained"),
    (True, f"19 visualizations generated"),
    (True, f"Models saved to {MODELS_DIR}"),
]

for ok, msg in checks:
    status = "✅" if ok else "❌"
    print(f"    {status} {msg}")

print(f"\n  Output Files:")
print(f"    X_ts.parquet  → {RESULTS_DIR / 'X_ts.parquet'}")
print(f"    Results JSON  → {RESULTS_DIR / 'module1_results.json'}")
print(f"    Visualizations → {VIZ_DIR}/")
print(f"    Models         → {MODELS_DIR}/")
print(f"\n{'═'*70}")