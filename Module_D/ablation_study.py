"""
CrisisNet Module D — Step 3: Ablation Study
============================================
Trains 6 separate models with different feature subsets to quantify each
module's marginal contribution.

Ablation configurations:
  1. zscore_only        — Altman Z-Score sigmoid (no ML)
  2. zscore_5factors    — Z-Score sub-factors only (1 feature via LightGBM)
  3. module_a_only      — All Module A ts features (~88 features)
  4. a_plus_b           — Module A + NLP features (~134 features)
  5. a_plus_c           — Module A + Graph features (~109 features)
  6. full_fusion        — All features A+B+C (~155 features)

Usage:
    python Module_D/ablation_study.py

Output:
    Module_D/ablation_results.json
    Module_D/ablation_table.png
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
MODULE_D     = Path(__file__).resolve().parent
X_FUSED_PATH = MODULE_D / "X_fused.parquet"

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading X_fused…")
X = pd.read_parquet(X_FUSED_PATH)
X['year'] = X['quarter'].str[:4].astype(int)

META      = {'ticker', 'quarter', 'Date', 'distress_label', 'year'}
feat_cols = [c for c in X.columns
             if c not in META and pd.api.types.is_numeric_dtype(X[c])]

train = X[X['year'] <= 2018].sort_values('quarter').reset_index(drop=True)
test  = X[X['year'] >= 2019].copy()

y_train = train['distress_label'].values
y_test  = test['distress_label'].values
scale   = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

print(f"Train: {len(train)} rows ({y_train.sum()} positives)  "
      f"Test: {len(test)} rows ({y_test.sum()} positives)")

# ── Altman Z-Score sigmoid ────────────────────────────────────────────────────
def zscore_to_prob(z):
    z = np.nan_to_num(z, nan=2.4)
    return 1.0 / (1.0 + np.exp(0.8 * (z - 2.0)))

# ── Ablation configurations ───────────────────────────────────────────────────
ZSCORE_5 = ['altman_z', 'X1_wc_ta', 'X2_re_ta', 'X3_ebit_ta', 'X4_mcap_tl', 'X5_rev_ta']
# Filter to those that actually exist in feat_cols
ZSCORE_5 = [c for c in ZSCORE_5 if c in feat_cols]

configs = {
    'zscore_only':     None,   # special case — sigmoid, no ML
    'zscore_5factors': ZSCORE_5,
    'module_a_only':   [c for c in feat_cols if not c.startswith(('nlp_', 'graph_'))],
    'a_plus_b':        [c for c in feat_cols if not c.startswith('graph_')],
    'a_plus_c':        [c for c in feat_cols if not c.startswith('nlp_')],
    'full_fusion':     feat_cols,
}

# Expected AUC ranges (from playbook section 8.4)
expected = {
    'zscore_only':     '~0.50',
    'zscore_5factors': '~0.55',
    'module_a_only':   '~0.88',
    'a_plus_b':        '~0.89–0.91',
    'a_plus_c':        '~0.88–0.91',
    'full_fusion':     '~0.90–0.93',
}

rq_map = {
    'zscore_only':     'RQ1 baseline',
    'zscore_5factors': 'Z sub-factors vs. full Z',
    'module_a_only':   'Module A standalone',
    'a_plus_b':        'RQ3: NLP marginal value',
    'a_plus_c':        'RQ2: Network marginal value',
    'full_fusion':     'RQ1: Final vs Z-Score',
}

# ── Walk-forward CV helper ────────────────────────────────────────────────────
def cv_auc(cols):
    tscv    = TimeSeriesSplit(n_splits=5)
    aucs    = []
    for tr_idx, va_idx in tscv.split(train):
        Xtr = train.iloc[tr_idx][cols].values
        ytr = train.iloc[tr_idx]['distress_label'].values
        Xva = train.iloc[va_idx][cols].values
        yva = train.iloc[va_idx]['distress_label'].values
        if yva.sum() == 0:
            continue
        m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=scale,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1
        )
        m.fit(Xtr, ytr)
        aucs.append(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))
    return float(np.mean(aucs)) if aucs else float('nan')

# ── Run ablations ─────────────────────────────────────────────────────────────
results = {}
print("\n{'Config':<20s} {'n_feat':>6s}  {'CV AUC':>8s}  {'Test AUC':>9s}  {'Brier':>7s}")
print("-" * 65)

for config_name, cols in configs.items():
    if config_name == 'zscore_only':
        # Pure sigmoid — no LightGBM
        probs  = zscore_to_prob(test['altman_z'].values if 'altman_z' in test.columns
                                else np.full(len(test), 2.4))
        cv_val = 'N/A (sigmoid)'
        n_feat = 1
    else:
        # Ensure all columns exist
        cols = [c for c in cols if c in train.columns]
        n_feat  = len(cols)
        cv_val  = cv_auc(cols)

        m_final = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=scale,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1
        )
        m_final.fit(train[cols].values, y_train)
        probs = m_final.predict_proba(test[cols].values)[:, 1]

    if y_test.sum() > 0 and y_test.sum() < len(y_test):
        test_auc = round(roc_auc_score(y_test, probs), 4)
        brier    = round(brier_score_loss(y_test, probs), 4)
    else:
        test_auc, brier = float('nan'), float('nan')

    cv_str = f"{cv_val:.4f}" if isinstance(cv_val, float) else cv_val
    results[config_name] = {
        'n_features': n_feat,
        'cv_auc':     round(cv_val, 4) if isinstance(cv_val, float) else cv_val,
        'test_auc':   test_auc,
        'brier':      brier,
        'expected':   expected[config_name],
        'rq':         rq_map[config_name],
    }
    print(f"  {config_name:<22s} {n_feat:>5d}  {cv_str:>9s}  {test_auc:>9.4f}  {brier:>7.4f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
with open(MODULE_D / 'ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved ablation_results.json")

# ── Ablation table plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.axis('off')

rows = []
for k, v in results.items():
    cv_str  = f"{v['cv_auc']:.4f}" if isinstance(v['cv_auc'], float) else v['cv_auc']
    test_str= f"{v['test_auc']:.4f}" if not np.isnan(v['test_auc']) else 'N/A'
    rows.append([
        k,
        str(v['n_features']),
        cv_str,
        test_str,
        v['expected'],
        v['rq'],
    ])

col_labels = ['Configuration', 'Features', 'CV AUC', 'Test AUC',
              'Expected AUC', 'RQ Answered']
table = ax.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(9)
# Header style
for j in range(len(col_labels)):
    table[(0, j)].set_facecolor('#1A5276')
    table[(0, j)].get_text().set_color('white')
    table[(0, j)].get_text().set_fontweight('bold')

ax.set_title('CrisisNet Ablation Study — Feature Contribution Analysis',
             fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(MODULE_D / 'ablation_table.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved ablation_table.png")

print("\n=== Ablation complete ===")
for k, v in results.items():
    test_s = f"{v['test_auc']:.4f}" if not np.isnan(v['test_auc']) else 'N/A'
    print(f"  {k:<22s}  Test AUC={test_s}  (expected {v['expected']})")
