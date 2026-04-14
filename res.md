# CrisisNet E0259 — Session Results

## NLP Contribution — Recall-Focused Analysis

Module B is now fused through `Module_B/results/X_nlp_finbert.parquet` rather than the older basic lexicon file. The fused model includes FinBERT/text distress features such as `nlp_going_concern_flag`, `nlp_covenant_flag`, `nlp_distress_phrase_rate`, `nlp_topic_kl_shift`, `nlp_readability_fog_approx`, and NLP interaction features with volatility and HY OAS.

| Metric | Without NLP (A-only) | With NLP (A+B) | Delta |
|---|---:|---:|---:|
| ROC-AUC | 0.7911 | 0.8000 | +0.0089 |
| PR-AUC | 0.6808 | 0.7012 | +0.0204 |
| F2-Score | 0.5920 | 0.6314 | +0.0394 |
| Recall | 0.6722 | 0.6889 | +0.0167 |

Interpretation: after restoring SEC EDGAR Altman coverage, NLP improves the recall-weighted F2 trade-off and recall at the fixed 0.07 threshold. The honest presentation is: **NLP helps the early-warning objective through F2, PR-AUC, and fewer missed distress cases.** Full fusion with graph features reaches recall 0.6944 in the ablation; the feature-selected final model reaches recall 0.6944.

Key NLP features retained/used by the final model include `nlp_tenk_score_lag2q`, `nlp_topic_0`, `nlp_tenk_score_4q_mean_lag2q`, `nlp_yoy_cosine_sim`, and `nlp_yoy_cosine_dist`, all appearing in the top-20 SHAP list after training.

## Model Performance

| Metric | Value |
|---|---:|
| Selected feature count | 120 |
| Selected NLP features | 13 |
| CV Walk-Forward AUC | 0.8988 |
| Test ROC-AUC | 0.8162 |
| Test ROC-AUC 95% CI | [0.7688, 0.8581] |
| Test PR-AUC | 0.7142 |
| Test PR-AUC 95% CI | [0.6471, 0.7751] |
| Test F2-score | 0.6437 |
| Test recall | 0.6944 |
| Test precision | 0.4980 |
| Test Brier score | 0.0811 |
| Original Altman Z ROC-AUC | 0.7145 on 505 covered test rows |
| Original Altman Z PR-AUC | 0.3812 on 505 covered test rows |
| LR baseline ROC-AUC | 0.7483 |
| LR baseline PR-AUC | 0.5637 |

The original Altman Z-score path is restored in `build_x_fused.py`: when Module A exports zeroed Altman columns, fusion recomputes the original 1968 Altman score and X1-X5 components from SEC EDGAR XBRL companyfacts, falling back to raw quarterly financial statement CSVs if needed. This restored nonzero Altman values for 657 rows after forward-fill in the current run.

Important coverage caveat: Altman coverage is `152/610` train rows and `505/1145` test rows, with 87 positives in the covered test subset. `zscore_only` is evaluated only on covered rows, not on zero-filled missing rows.

## Corrected Ablation

| Configuration | Features | CV AUC | ROC-AUC | PR-AUC | F2 | Recall | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| zscore_only | 1 | N/A | 0.7145 | 0.3812 | 0.4987 | 0.9080 | 0.4065 |
| zscore_5factors | 6 | 0.8457 | 0.5948 | 0.3247 | 0.4179 | 0.6389 | 0.1839 |
| module_a_only | 187 | 0.8787 | 0.7911 | 0.6808 | 0.5920 | 0.6722 | 0.1041 |
| a_plus_b | 257 | 0.8741 | 0.8000 | 0.7012 | 0.6314 | 0.6889 | 0.0866 |
| a_plus_c | 209 | 0.8929 | 0.8059 | 0.6980 | 0.6176 | 0.7000 | 0.0894 |
| full_fusion | 279 | 0.8909 | 0.8110 | 0.7182 | 0.6463 | 0.6944 | 0.0751 |

## Per-Period Performance

| Period | n | Positives | ROC-AUC | F2 | Recall |
|---|---:|---:|---:|---:|---:|
| 2019-2020 crisis | 312 | 174 | 0.7933 | 0.7092 | 0.6897 |
| 2021+ post-crisis | 833 | 6 | 0.9831 | 0.2273 | 1.0000 |

The test positives remain concentrated in 2019-2020: 174 of 180 positives are in that window. Report the per-period split transparently.

## Failure Analysis

| Failure type | Count |
|---|---:|
| False negatives | 55 |
| False positives | 126 |
| False negatives with NLP coverage | 49 |
| False negatives without NLP coverage | 6 |

Top false-negative tickers: CHK 4, PSX 4, COP 3, NOV 3, VLO 3, then several tickers with 2 missed quarters each. CHK still has zero NLP coverage and remains a key data-coverage limitation.

Top false-positive tickers: AR 13, APA 11, DVN 11, OVV 11, PR 11.

## Files Added or Updated

| File | Change |
|---|---|
| `Module_D/build_x_fused.py` | FinBERT NLP forced; NLP lag/delta and interaction features; original Altman fallback from SEC EDGAR XBRL |
| `Module_D/feature_selection.py` | Selects 80-120 feature subset; current output is 120 features with 13 NLP features |
| `Module_D/train_fusion.py` | F2, recall, precision, bootstrap CIs, calibration plot, PR curve, per-period metrics, selected features, original Altman baseline |
| `Module_D/ablation_study.py` | Adds F2 and recall; NLP contribution summary; ablation table updated |
| `Module_D/failure_analysis.py` | Failure analysis JSON; fixed numeric NLP coverage handling |
| `dashboard/app.py` | Adds 7th Model Diagnostics tab |
| `Makefile` | Reproducible `make all` pipeline |
| `scripts/download_required_datasets.sh` | Dataset path preparation and SEC XBRL download hook |
| `scripts/download_sec_xbrl.py` | Versioned SEC EDGAR companyfacts downloader with company-list CIK fallback |
| `scripts/push_to_github.sh` | GitHub push helper |
