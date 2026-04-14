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
from sklearn.metrics import (roc_auc_score, brier_score_loss, average_precision_score,
                             fbeta_score, recall_score)
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
# Same exclusion as train_fusion.py — contemporaneous drawdown features are
# label-leaky (labels partly defined from >50% drawdown events)
LEAKY_COLS = {'max_drawdown_6m', 'drawdown_mean', 'drawdown_min'}
feat_cols = [c for c in X.columns
             if c not in META
             and c not in LEAKY_COLS
             and pd.api.types.is_numeric_dtype(X[c])]

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
altman_train_mask = (train['altman_z'].fillna(0) != 0).values if 'altman_z' in train.columns else np.zeros(len(train), dtype=bool)
altman_test_mask = (test['altman_z'].fillna(0) != 0).values if 'altman_z' in test.columns else np.zeros(len(test), dtype=bool)
print(f"Original Altman coverage: train={int(altman_train_mask.sum())}/{len(train)}, "
      f"test={int(altman_test_mask.sum())}/{len(test)} "
      f"(test positives={int(y_test[altman_test_mask].sum())})")

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
ABLATION_THRESHOLD = 0.07   # Same threshold as train_fusion.py

results = {}
print(f"\n  {'Config':<24s} {'n_feat':>6s}  {'CV AUC':>8s}  {'Test AUC':>9s}  {'PR-AUC':>7s}  {'F2':>6s}  {'Recall':>6s}  {'Brier':>7s}")
print("  " + "-" * 85)

for config_name, cols in configs.items():
    eval_mask = np.ones(len(test), dtype=bool)
    if config_name == 'zscore_only':
        # Pure sigmoid; evaluate only rows with restored nonzero Altman coverage.
        eval_mask = altman_test_mask
        probs  = zscore_to_prob(test['altman_z'].values)
        preds  = (probs > 0.5).astype(int)
        cv_val = 'N/A'
        n_feat = 1
    elif config_name == 'zscore_5factors' and altman_train_mask.sum() == 0:
        eval_mask = np.zeros(len(test), dtype=bool)
        probs  = np.full(len(test), np.nan)
        preds  = np.zeros(len(test), dtype=int)
        cv_val = 'N/A'
        n_feat = len(cols)
    else:
        cols   = [c for c in cols if c in train.columns]
        n_feat = len(cols)
        cv_val = cv_auc(cols)

        m_final = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=scale,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1
        )
        m_final.fit(train[cols].values, y_train)
        probs = m_final.predict_proba(test[cols].values)[:, 1]
        preds = (probs > ABLATION_THRESHOLD).astype(int)

    y_eval = y_test[eval_mask]
    p_eval = probs[eval_mask]
    d_eval = preds[eval_mask]
    if len(y_eval) > 0 and y_eval.sum() > 0 and y_eval.sum() < len(y_eval) and not np.isnan(p_eval).all():
        test_auc = round(roc_auc_score(y_eval, p_eval), 4)
        pr_auc   = round(average_precision_score(y_eval, p_eval), 4)
        brier    = round(brier_score_loss(y_eval, p_eval), 4)
        f2_val   = round(fbeta_score(y_eval, d_eval, beta=2), 4)
        rec_val  = round(recall_score(y_eval, d_eval), 4)
    else:
        test_auc, pr_auc, brier, f2_val, rec_val = [float('nan')] * 5

    cv_str = f"{cv_val:.4f}" if isinstance(cv_val, float) else cv_val
    results[config_name] = {
        'n_features': n_feat,
        'cv_auc':     round(cv_val, 4) if isinstance(cv_val, float) else cv_val,
        'test_auc':   test_auc,
        'pr_auc':     pr_auc,
        'f2_score':   f2_val,
        'recall':     rec_val,
        'brier':      brier,
        'n_eval':     int(len(y_eval)),
        'expected':   expected[config_name],
        'rq':         rq_map[config_name],
    }
    print(f"  {config_name:<24s} {n_feat:>5d}  {cv_str:>9s}  {test_auc:>9.4f}"
          f"  {pr_auc:>7.4f}  {f2_val:>6.4f}  {rec_val:>6.4f}  {brier:>7.4f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
with open(MODULE_D / 'ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved ablation_results.json")

# ── NLP contribution summary ───────────────────────────────────────────────────
print("\n  *** NLP Contribution Summary ***")
if 'module_a_only' in results and 'a_plus_b' in results:
    a   = results['module_a_only']
    ab  = results['a_plus_b']
    print(f"  Module A only  :  ROC-AUC={a['test_auc']:.4f}  F2={a['f2_score']:.4f}  Recall={a['recall']:.4f}")
    print(f"  A + NLP (B)    :  ROC-AUC={ab['test_auc']:.4f}  F2={ab['f2_score']:.4f}  Recall={ab['recall']:.4f}")
    delta_auc = ab['test_auc'] - a['test_auc']
    delta_f2  = ab['f2_score'] - a['f2_score']
    delta_rec = ab['recall']   - a['recall']
    print(f"  NLP marginal   :  ΔAUC={delta_auc:+.4f}  ΔF2={delta_f2:+.4f}  ΔRecall={delta_rec:+.4f}")
    if delta_rec > 0:
        print(f"  → NLP improves recall by {delta_rec:+.1%} and F2 by {delta_f2:+.4f}")
        print(f"  → For an early-warning system, recall improvement is the key metric.")
    elif delta_f2 > 0:
        print(f"  → NLP improves F2 by {delta_f2:+.4f}; recall is unchanged at this threshold.")
        print(f"  → For an early-warning system, F2 captures the recall-weighted trade-off.")

print("\n=== Ablation complete ===")
for k, v in results.items():
    f2_s  = f"{v['f2_score']:.4f}" if isinstance(v.get('f2_score'), float) and not np.isnan(v['f2_score']) else 'N/A'
    rec_s = f"{v['recall']:.4f}"   if isinstance(v.get('recall'),   float) and not np.isnan(v['recall'])   else 'N/A'
    test_s= f"{v['test_auc']:.4f}" if not np.isnan(v['test_auc'])   else 'N/A'
    print(f"  {k:<22s}  ROC-AUC={test_s}  F2={f2_s}  Recall={rec_s}")

# ── Ablation table plot ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.axis('off')

rows = []
for k, v in results.items():
    cv_str   = f"{v['cv_auc']:.4f}" if isinstance(v['cv_auc'], float) else v['cv_auc']
    test_str = f"{v['test_auc']:.4f}" if not np.isnan(v['test_auc']) else 'N/A'
    pr_str   = f"{v['pr_auc']:.4f}"   if not np.isnan(v['pr_auc'])   else 'N/A'
    f2_str   = f"{v['f2_score']:.4f}" if not np.isnan(v['f2_score']) else 'N/A'
    rec_str  = f"{v['recall']:.4f}"   if not np.isnan(v['recall'])   else 'N/A'
    rows.append([
        k,
        str(v['n_features']),
        cv_str,
        test_str,
        pr_str,
        f2_str,
        rec_str,
        v['rq'],
    ])

col_labels = ['Configuration', 'Features', 'CV AUC', 'ROC-AUC', 'PR-AUC',
              'F2-Score', 'Recall', 'RQ Answered']
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
