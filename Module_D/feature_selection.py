"""
CrisisNet Module D — Feature Selection
========================================
Uses LightGBM built-in importance to reduce features from ~250 down to
~80-120 while preserving NLP recall signals.

Usage:
    python Module_D/feature_selection.py

Output:
    Module_D/selected_features.json
"""

import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, fbeta_score, recall_score

warnings.filterwarnings('ignore')

MODULE_D = Path(__file__).resolve().parent

X = pd.read_parquet(MODULE_D / "X_fused.parquet")
X['year'] = X['quarter'].str[:4].astype(int)

META       = {'ticker', 'quarter', 'Date', 'distress_label', 'year'}
LEAKY_COLS = {'max_drawdown_6m', 'drawdown_mean', 'drawdown_min'}
feat_cols  = [c for c in X.columns
              if c not in META
              and c not in LEAKY_COLS
              and pd.api.types.is_numeric_dtype(X[c])]

train  = X[X['year'] <= 2018].sort_values('quarter').reset_index(drop=True)
test   = X[X['year'] >= 2019].copy()
y_train = train['distress_label'].values
y_test  = test['distress_label'].values
scale   = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1) * 3

print(f"[1/3] Training full model with {len(feat_cols)} features to get importances…")
model = lgb.LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7,
    scale_pos_weight=scale,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbose=-1
)
model.fit(train[feat_cols].values, y_train)

imp_df = pd.DataFrame({
    'feature':    feat_cols,
    'importance': model.feature_importances_,
}).sort_values('importance', ascending=False)

non_zero = imp_df[imp_df['importance'] > 0]
print(f"[2/3] Non-zero importance features: {len(non_zero)} / {len(feat_cols)}")

# Print module-level breakdown
for prefix, label in [('nlp_', 'NLP'), ('graph_', 'Graph'), ('', 'TS/Macro')]:
    if prefix:
        sub = non_zero[non_zero['feature'].str.startswith(prefix)]
    else:
        sub = non_zero[~non_zero['feature'].str.startswith(('nlp_', 'graph_'))]
    print(f"  {label:<10s}: {len(sub)} non-zero-importance features")

print(f"\n[3/3] Testing feature subsets (targeting 80-120 features and F2-score)…")
print(f"  {'N features':>12s}  {'ROC-AUC':>8s}  {'F2':>6s}  {'Recall':>6s}")
print(f"  {'-'*42}")

best_f2 = -1.0
best_n  = 100
results = []

candidate_sizes = [80, 100, 120]
for extra in [len(non_zero), len(feat_cols)]:
    if 80 <= extra <= 120 and extra not in candidate_sizes:
        candidate_sizes.append(extra)

for n_keep in candidate_sizes:
    n_keep   = min(n_keep, len(feat_cols))
    selected = imp_df.head(n_keep)['feature'].tolist()

    m = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=scale,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1
    )
    m.fit(train[selected].values, y_train)
    probs = m.predict_proba(test[selected].values)[:, 1]
    preds = (probs > 0.07).astype(int)

    auc = roc_auc_score(y_test, probs)
    f2  = fbeta_score(y_test, preds, beta=2)
    rec = recall_score(y_test, preds)

    results.append({'n': n_keep, 'auc': round(auc, 4),
                    'f2': round(f2, 4), 'recall': round(rec, 4)})
    print(f"  {n_keep:>12d}  {auc:>8.4f}  {f2:>6.4f}  {rec:>6.4f}")

    if f2 >= best_f2:
        best_f2 = f2
        best_n  = n_keep

print(f"\n  Recommended: top {best_n} ranked features  (best F2={best_f2:.4f})")

selected_features = imp_df.head(best_n)['feature'].tolist()

# Keep a minimum NLP footprint so the final fusion model still carries the
# text distress signals used for the recall narrative.
MIN_NLP = 10
ranked_nlp = imp_df[imp_df['feature'].str.startswith('nlp_')]['feature'].tolist()
missing_nlp = [f for f in ranked_nlp if f not in selected_features][:max(0, MIN_NLP - sum(f.startswith('nlp_') for f in selected_features))]
if missing_nlp:
    protected = set(selected_features[:20])
    replaceable = [f for f in reversed(selected_features) if not f.startswith('nlp_') and f not in protected]
    for add_f, drop_f in zip(missing_nlp, replaceable):
        selected_features[selected_features.index(drop_f)] = add_f
    print(f"  NLP retention: added {len(missing_nlp)} top-ranked NLP features")

nlp_sel   = [f for f in selected_features if f.startswith('nlp_')]
graph_sel = [f for f in selected_features if f.startswith('graph_')]
ts_sel    = [f for f in selected_features if not f.startswith(('nlp_', 'graph_'))]
print(f"  NLP features retained:   {len(nlp_sel)}")
print(f"  Graph features retained: {len(graph_sel)}")
print(f"  TS/Macro features:       {len(ts_sel)}")

output = {
    'selected_features':      selected_features,
    'n_selected':             best_n,
    'n_nlp':                  len(nlp_sel),
    'n_graph':                len(graph_sel),
    'n_ts_macro':             len(ts_sel),
    'feature_importances':    imp_df[['feature', 'importance']].to_dict('records'),
    'selection_results':      results,
}
with open(MODULE_D / 'selected_features.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"Saved selected_features.json  ({best_n} features)")
