#!/usr/bin/env python3
"""
CrisisNet Module 1 — Enhanced Stock Price Feature Engineering
==============================================================
Adds granular daily-price-derived features that the quarterly pipeline misses:
- Rolling z-scores of returns
- Bollinger Band signals
- Volume-weighted metrics
- RSI (Relative Strength Index)
- MACD signals
- Daily LSTM training on price sequences
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json, pickle, warnings
from pathlib import Path
from scipy.stats import norm

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MARKET_DIR = DATA_DIR / "market_data"
CREDIT_DIR = DATA_DIR / "credit_spreads"
LABELS_DIR = DATA_DIR / "labels"
RESULTS_DIR = PROJECT_ROOT / "results"
VIZ_DIR = RESULTS_DIR / "visualizations"
MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 70)
print("ENHANCED STOCK PRICE FEATURE ENGINEERING")
print("=" * 70)

# ===================== LOAD DATA =====================
prices_raw = pd.read_parquet(MARKET_DIR / "all_prices.parquet")
prices_raw.index = pd.to_datetime(prices_raw.index)
TICKERS = prices_raw.columns.get_level_values(0).unique().tolist()

fred = pd.read_parquet(CREDIT_DIR / "fred_all_series.parquet")
fred.index = pd.to_datetime(fred.index)
fred = fred.sort_index().ffill()

defaults_df = pd.read_csv(LABELS_DIR / "energy_defaults_curated.csv")
drawdowns_df = pd.read_csv(LABELS_DIR / "distress_from_drawdowns.csv")

print(f"Loaded {len(TICKERS)} tickers, {len(prices_raw)} trading days")

# ===================== ENHANCED DAILY FEATURES =====================
print("\n[1/5] Computing enhanced daily stock features...")

def compute_rsi(series, period=14):
    """Relative Strength Index."""
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

for ticker in TICKERS:
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

    # Rolling volatilities at multiple windows
    df['vol_10d'] = log_ret.rolling(10).std() * np.sqrt(252)
    df['vol_30d'] = log_ret.rolling(30).std() * np.sqrt(252)
    df['vol_60d'] = log_ret.rolling(60).std() * np.sqrt(252)
    df['vol_90d'] = log_ret.rolling(90).std() * np.sqrt(252)

    # Volatility ratio (short-term vs long-term) — spikes indicate regime change
    df['vol_ratio_10_60'] = df['vol_10d'] / (df['vol_60d'] + 1e-8)

    # Rolling z-score of returns (how unusual is today's return?)
    df['return_zscore_30d'] = (log_ret - log_ret.rolling(30).mean()) / (log_ret.rolling(30).std() + 1e-8)

    # Momentum at multiple horizons
    df['mom_5d'] = close.pct_change(5)
    df['mom_20d'] = close.pct_change(20)
    df['mom_60d'] = close.pct_change(60)
    df['mom_120d'] = close.pct_change(120)
    df['mom_250d'] = close.pct_change(250)

    # Bollinger Bands
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    df['bb_upper'] = sma_20 + 2 * std_20
    df['bb_lower'] = sma_20 - 2 * std_20
    df['bb_pct'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma_20 + 1e-8)

    # RSI
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
    df['obv'] = (np.sign(log_ret) * volume).cumsum()  # On-Balance Volume
    df['obv_slope_20'] = df['obv'].diff(20) / (df['obv'].shift(20).abs() + 1e-8)

    # Price relative to moving averages
    df['price_sma50_ratio'] = close / close.rolling(50).mean()
    df['price_sma200_ratio'] = close / close.rolling(200).mean()
    df['sma50_sma200_cross'] = (close.rolling(50).mean() - close.rolling(200).mean()) / close  # Golden/Death cross

    # Drawdown from peak
    rolling_max = close.rolling(252, min_periods=1).max()
    df['drawdown'] = (close - rolling_max) / rolling_max
    df['days_since_high'] = close.groupby((close == rolling_max).cumsum()).cumcount()

    # Intraday features
    df['true_range'] = pd.concat([high - low, (high - close.shift(1)).abs(),
                                   (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df['atr_14'] = df['true_range'].rolling(14).mean()
    df['atr_pct'] = df['atr_14'] / (close + 1e-8)

    all_daily_features.append(df)

daily_df = pd.concat(all_daily_features)
daily_df.index.name = 'Date'
print(f"  Daily features: {daily_df.shape} ({daily_df['ticker'].nunique()} tickers)")

# ===================== AGGREGATE TO QUARTERLY =====================
print("\n[2/5] Aggregating to quarterly with richer statistics...")

agg_funcs = {
    'close': ['last', 'mean', 'std'],
    'log_return': ['mean', 'std', 'min', 'max', 'skew'],
    'vol_30d': ['mean', 'last', 'max'],
    'vol_60d': ['mean', 'last'],
    'vol_ratio_10_60': ['mean', 'max'],
    'return_zscore_30d': ['mean', 'std', 'min', 'max'],
    'mom_20d': 'last',
    'mom_60d': 'last',
    'mom_120d': 'last',
    'mom_250d': 'last',
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
for ticker in daily_df['ticker'].unique():
    td = daily_df[daily_df['ticker'] == ticker].copy()
    td_numeric = td.drop(columns=['ticker'])

    q = td_numeric.resample('QE').agg(agg_funcs)
    q.columns = ['_'.join(c).strip('_') for c in q.columns]
    q['ticker'] = ticker
    quarterly_enhanced.append(q)

q_enhanced = pd.concat(quarterly_enhanced).reset_index()
q_enhanced.columns = [str(c).replace(' ', '_') for c in q_enhanced.columns]
print(f"  Quarterly enhanced: {q_enhanced.shape}")

# ===================== MERGE WITH EXISTING X_ts =====================
print("\n[3/5] Merging with existing X_ts features...")

X_ts_old = pd.read_parquet(RESULTS_DIR / "X_ts.parquet").reset_index()
# Keep only the new columns that don't overlap
old_cols = set(X_ts_old.columns)
new_feature_cols = [c for c in q_enhanced.columns if c not in old_cols and c not in ['Date', 'ticker']]
print(f"  Adding {len(new_feature_cols)} new features to X_ts")

X_ts = pd.merge(X_ts_old, q_enhanced[['Date', 'ticker'] + new_feature_cols], on=['Date', 'ticker'], how='left')
X_ts[new_feature_cols] = X_ts[new_feature_cols].apply(pd.to_numeric, errors='coerce')

# Fill NaN with median
medians = X_ts[new_feature_cols].median()
X_ts[new_feature_cols] = X_ts[new_feature_cols].fillna(medians)
X_ts[new_feature_cols] = X_ts[new_feature_cols].replace([np.inf, -np.inf], 0)

print(f"  Enhanced X_ts: {X_ts.shape}")

# ===================== RETRAIN XGBOOST WITH ENHANCED FEATURES =====================
print("\n[4/5] Retraining XGBoost with enhanced features...")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, classification_report
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

id_cols = ['Date', 'ticker', 'distress_label']
feature_cols = [c for c in X_ts.columns if c not in id_cols]
X_ts[feature_cols] = X_ts[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

# Winsorize on train
train_mask = X_ts['Date'] < '2022-01-01'
for col in feature_cols:
    q01, q99 = X_ts.loc[train_mask, col].quantile([0.01, 0.99])
    X_ts[col] = X_ts[col].clip(q01, q99)

X_train = X_ts[train_mask]
X_val = X_ts[(X_ts['Date'] >= '2022-01-01') & (X_ts['Date'] < '2023-01-01')]
X_test = X_ts[X_ts['Date'] >= '2023-01-01']
y_train = X_train['distress_label'].values
y_val = X_val['distress_label'].values
y_test = X_test['distress_label'].values

n_pos = y_train.sum()
scale_pos = (len(y_train) - n_pos) / max(n_pos, 1)

print(f"  Train: {len(X_train)} ({n_pos} distress)")
print(f"  Test:  {len(X_test)} ({y_test.sum()} distress)")
print(f"  Features: {len(feature_cols)}")

# Walk-forward CV
tscv = TimeSeriesSplit(n_splits=5)
train_sorted = X_train.sort_values('Date')
cv_aucs = []
for fold, (tr_idx, va_idx) in enumerate(tscv.split(train_sorted)):
    Xtr = train_sorted.iloc[tr_idx][feature_cols].values
    ytr = train_sorted.iloc[tr_idx]['distress_label'].values
    Xva = train_sorted.iloc[va_idx][feature_cols].values
    yva = train_sorted.iloc[va_idx]['distress_label'].values
    fm = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.7,
                           scale_pos_weight=scale_pos, random_state=42,
                           use_label_encoder=False, eval_metric='auc',
                           reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5)
    fm.fit(Xtr, ytr, verbose=False)
    if yva.sum() > 0 and yva.sum() < len(yva):
        fp = fm.predict_proba(Xva)[:, 1]
        fa = roc_auc_score(yva, fp)
        cv_aucs.append(fa)
        print(f"    Fold {fold+1}: AUC={fa:.4f}")

if cv_aucs:
    print(f"  Walk-Forward CV AUC: {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f}")

# Train final model
xgb_enhanced = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7,
    scale_pos_weight=scale_pos, random_state=42,
    use_label_encoder=False, eval_metric='auc',
    reg_alpha=0.1, reg_lambda=1.0, min_child_weight=5,
)
xgb_enhanced.fit(X_train[feature_cols].values, y_train, verbose=False)

# Evaluate
for name, Xs, y in [('Train', X_train[feature_cols].values, y_train),
                     ('Test', X_test[feature_cols].values, y_test)]:
    xp = xgb_enhanced.predict_proba(Xs)[:, 1]
    if y.sum() > 0 and y.sum() < len(y):
        auc = roc_auc_score(y, xp)
        brier = brier_score_loss(y, xp)
        print(f"  {name:5s} AUC: {auc:.4f}  Brier: {brier:.4f}")
    else:
        print(f"  {name:5s} — insufficient positive samples")

# Save enhanced model
xgb_enhanced.save_model(str(MODELS_DIR / "xgboost_enhanced.json"))

# ===================== RETRAIN LSTM ON DAILY SEQUENCES =====================
print("\n[5/5] Training LSTM on daily price sequences...")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    # Build daily sequences: 60-day windows of price features
    DAILY_FEATURES = ['log_return', 'vol_30d', 'vol_ratio_10_60', 'return_zscore_30d',
                      'rsi_14', 'macd_hist', 'bb_pct', 'vol_ratio_20',
                      'drawdown', 'atr_pct', 'price_sma50_ratio', 'price_sma200_ratio']

    # Create labels for daily data: 1 if distress within next 6 months
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

    print(f"  Daily labels: {daily_df['distress_label'].sum()} distress days out of {len(daily_df)}")

    # Standardize features
    daily_feats = daily_df[DAILY_FEATURES].copy()
    daily_feats = daily_feats.apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0)

    train_daily = daily_df[daily_df.index < '2022-01-01']
    test_daily = daily_df[daily_df.index >= '2023-01-01']

    scaler_daily = StandardScaler()
    scaler_daily.fit(train_daily[DAILY_FEATURES].fillna(0).replace([np.inf, -np.inf], 0))

    SEQ_LEN = 60  # 60 trading days = ~3 months

    class DailySeqDataset(Dataset):
        def __init__(self, df, features, scaler, seq_len=60, subsample_neg=5):
            self.samples, self.labels = [], []
            for ticker in df['ticker'].unique():
                td = df[df['ticker'] == ticker].sort_index()
                feats = scaler.transform(td[features].fillna(0).replace([np.inf, -np.inf], 0)).astype(np.float32)
                labs = td['distress_label'].values
                for i in range(seq_len, len(feats)):
                    label = labs[i]
                    # Subsample negatives to balance
                    if label == 0 and np.random.random() > 1.0/subsample_neg:
                        continue
                    self.samples.append(feats[i-seq_len:i])
                    self.labels.append(label)
            print(f"    Dataset: {len(self.samples)} samples, {sum(self.labels)} positive")

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

    train_ds = DailySeqDataset(train_daily, DAILY_FEATURES, scaler_daily, SEQ_LEN, subsample_neg=3)
    test_ds = DailySeqDataset(test_daily, DAILY_FEATURES, scaler_daily, SEQ_LEN, subsample_neg=1)

    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=128)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    daily_lstm = DailyLSTM(len(DAILY_FEATURES), hidden=128, n_layers=2).to(device)
    opt = torch.optim.Adam(daily_lstm.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    pw = torch.tensor([scale_pos], dtype=torch.float32).to(device)

    best_loss = float('inf')
    for epoch in range(30):
        daily_lstm.train()
        total_loss, n_batch = 0, 0
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = daily_lstm(xb)
            w = torch.where(yb == 1, pw[0], torch.tensor(1.0).to(device))
            loss = nn.functional.binary_cross_entropy(out, yb, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(daily_lstm.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batch += 1

        avg_loss = total_loss / max(n_batch, 1)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(daily_lstm.state_dict(), str(MODELS_DIR / "lstm_daily_enhanced.pt"))

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")

    # Evaluate
    daily_lstm.load_state_dict(torch.load(str(MODELS_DIR / "lstm_daily_enhanced.pt"), weights_only=True))
    daily_lstm.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_ld:
            probs = daily_lstm(xb.to(device)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if all_labels.sum() > 0 and all_labels.sum() < len(all_labels):
        auc = roc_auc_score(all_labels, all_probs)
        brier = brier_score_loss(all_labels, all_probs)
        print(f"  Daily LSTM Test AUC: {auc:.4f}  Brier: {brier:.4f}")
    else:
        print(f"  Daily LSTM — {all_labels.sum()} positive out of {len(all_labels)} test samples")

except Exception as e:
    print(f"  Daily LSTM failed: {e}")
    import traceback; traceback.print_exc()

# ===================== ENHANCED VISUALIZATIONS =====================
print("\n[BONUS] Generating enhanced visualizations...")

# RSI distribution by class
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

# Drawdown distribution
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
print("  17_rsi_drawdown_distributions.png")

# Multi-scale volatility for CHK
fig, ax = plt.subplots(figsize=(14, 7))
for ticker in ['CHK']:
    if ticker in daily_df['ticker'].values:
        td = daily_df[daily_df['ticker'] == ticker]
        for col, label, color in [('vol_10d', '10-day', '#e74c3c'),
                                   ('vol_30d', '30-day', '#3498db'),
                                   ('vol_60d', '60-day', '#2ecc71'),
                                   ('vol_90d', '90-day', '#9b59b6')]:
            if col in td.columns:
                ax.plot(td.index, td[col], linewidth=1, label=label, color=color, alpha=0.8)

ax.axvline(x=pd.Timestamp('2020-06-28'), color='red', alpha=0.5, linewidth=3)
ax.annotate('CHK Bankruptcy', xy=(pd.Timestamp('2020-06-28'), 3), fontsize=12, color='red', fontweight='bold')
ax.set_title('Chesapeake Energy — Multi-Scale Volatility (The Cancer Screening Analogy)', fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=13)
ax.set_ylabel('Annualized Volatility', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(VIZ_DIR / "18_chk_multiscale_volatility.png", dpi=300)
plt.close()
print("  18_chk_multiscale_volatility.png")

# XGBoost enhanced feature importance
fig, ax = plt.subplots(figsize=(14, 12))
imp = pd.Series(xgb_enhanced.feature_importances_, index=feature_cols).sort_values().tail(25)
imp.plot(kind='barh', ax=ax, color='#3498db', edgecolor='white')
ax.set_xlabel('Feature Importance (Gain)', fontsize=13)
ax.set_title('Enhanced XGBoost — Top 25 Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_DIR / "19_enhanced_feature_importance.png", dpi=300)
plt.close()
print("  19_enhanced_feature_importance.png")

# Save enhanced X_ts
X_ts_export = X_ts.set_index(['ticker', 'Date'])
X_ts_export.to_parquet(RESULTS_DIR / "X_ts.parquet")
X_ts_export.to_parquet(PROJECT_ROOT.parent / "X_ts.parquet")

# Save enhanced feature list
with open(MODELS_DIR / "feature_columns_enhanced.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"\n{'='*70}")
print("ENHANCED PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"  Enhanced X_ts: {X_ts_export.shape}")
print(f"  Features: {len(feature_cols)}")
print(f"  Models: xgboost_enhanced.json, lstm_daily_enhanced.pt")
print(f"  New visualizations: 17, 18, 19")
