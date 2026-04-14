"""
CrisisNet Module D — Model Failure Analysis
=============================================
Identifies which companies/quarters the model gets wrong and diagnoses
common failure patterns. Outputs to failure_analysis.json.

Usage:
    python Module_D/failure_analysis.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import fbeta_score, recall_score

MODULE_D = Path(__file__).resolve().parent

test_pred = pd.read_parquet(MODULE_D / 'test_predictions.parquet')
X_fused   = pd.read_parquet(MODULE_D / 'X_fused.parquet')

# ── NLP coverage flag ─────────────────────────────────────────────────────────
nlp_cols = [
    c for c in X_fused.columns
    if c.startswith('nlp_') and pd.api.types.is_numeric_dtype(X_fused[c])
]
if nlp_cols:
    nlp_cov = X_fused[['ticker', 'quarter'] + nlp_cols].copy()
    nlp_cov['has_nlp'] = (nlp_cov[nlp_cols].abs().sum(axis=1) > 0).astype(int)
    test_pred = test_pred.merge(nlp_cov[['ticker', 'quarter', 'has_nlp']],
                                on=['ticker', 'quarter'], how='left')
    test_pred['has_nlp'] = test_pred['has_nlp'].fillna(0).astype(int)
else:
    test_pred['has_nlp'] = 0

# ── False Negatives (missed distress) ─────────────────────────────────────────
fn = test_pred[(test_pred['distress_label'] == 1) & (test_pred['predicted_label'] == 0)]
print(f"=== FALSE NEGATIVES (Missed Distress): {len(fn)} ===")
print(f"  With NLP coverage: {fn['has_nlp'].sum()}")
print(f"  Without NLP:       {(fn['has_nlp'] == 0).sum()}")
print(f"\n  By ticker:")
fn_by_ticker = fn.groupby('ticker').size().sort_values(ascending=False)
for ticker, count in fn_by_ticker.items():
    grp = fn[fn['ticker'] == ticker]
    print(f"    {ticker:<8s}: {count} missed  |  NLP={grp['has_nlp'].sum()}/{count}  |  "
          f"quarters: {', '.join(sorted(grp['quarter'].tolist()))}")

# ── False Positives (false alarms) ────────────────────────────────────────────
fp = test_pred[(test_pred['distress_label'] == 0) & (test_pred['predicted_label'] == 1)]
print(f"\n=== FALSE POSITIVES (False Alarms): {len(fp)} ===")
fp_by_ticker = fp.groupby('ticker').size().sort_values(ascending=False)
print(f"\n  By ticker (top 10):")
for ticker, count in fp_by_ticker.head(10).items():
    print(f"    {ticker:<8s}: {count} false alarms")

# ── Per-ticker recall ─────────────────────────────────────────────────────────
print(f"\n=== PER-TICKER RECALL (companies with distress events) ===")
per_ticker = []
for ticker, group in test_pred.groupby('ticker'):
    pos = group['distress_label'].sum()
    if pos > 0:
        rec = recall_score(group['distress_label'], group['predicted_label'])
        f2  = fbeta_score(group['distress_label'], group['predicted_label'], beta=2)
        nlp_pct = group['has_nlp'].mean() * 100
        per_ticker.append({
            'ticker':           ticker,
            'positives':        int(pos),
            'recall':           round(rec, 3),
            'f2':               round(f2,  3),
            'nlp_coverage_pct': round(nlp_pct, 1),
        })
        print(f"  {ticker:<8s}: recall={rec:.2f}  F2={f2:.2f}  "
              f"({pos} distress qtrs, NLP={nlp_pct:.0f}%)")

# ── NLP impact on recall ───────────────────────────────────────────────────────
print(f"\n=== NLP IMPACT ON RECALL ===")
for has_nlp_val, label in [(1, "With NLP features"), (0, "Without NLP features")]:
    sub = test_pred[test_pred['has_nlp'] == has_nlp_val]
    if sub['distress_label'].sum() > 0:
        rec = recall_score(sub['distress_label'], sub['predicted_label'])
        f2  = fbeta_score(sub['distress_label'], sub['predicted_label'], beta=2)
        print(f"  {label:<22s}: recall={rec:.4f}  F2={f2:.4f}  "
              f"(n={len(sub)}, pos={sub['distress_label'].sum()})")

# ── Save results ──────────────────────────────────────────────────────────────
analysis = {
    'false_negatives': {
        'total':        int(len(fn)),
        'with_nlp':     int(fn['has_nlp'].sum()),
        'without_nlp':  int((fn['has_nlp'] == 0).sum()),
        'by_ticker':    fn_by_ticker.to_dict(),
    },
    'false_positives': {
        'total':           int(len(fp)),
        'by_ticker_top10': fp_by_ticker.head(10).to_dict(),
    },
    'per_ticker_recall': per_ticker,
}

with open(MODULE_D / 'failure_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)
print(f"\nSaved failure_analysis.json")
