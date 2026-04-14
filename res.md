# CrisisNet E0259 — Session Results

## NLP Contribution — Recall-Focused Analysis

Module B is now fused through `Module_B/results/X_nlp_finbert.parquet` rather than the older basic lexicon file. The fused model includes FinBERT/text distress features such as `nlp_going_concern_flag`, `nlp_covenant_flag`, `nlp_distress_phrase_rate`, `nlp_topic_kl_shift`, `nlp_readability_fog_approx`, and NLP interaction features with volatility and HY OAS.

| Metric | Without NLP (A-only) | With NLP (A+B) | Delta |
|---|---:|---:|---:|
| ROC-AUC | 0.7890 | 0.8060 | +0.0170 |
| PR-AUC | 0.6943 | 0.7107 | +0.0164 |
| F2-Score | 0.5967 | 0.6175 | +0.0208 |
| Recall | 0.6889 | 0.6889 | +0.0000 |

Interpretation: in this rerun, NLP improves the recall-weighted F2 trade-off and ranking metrics, while marginal recall is unchanged at the fixed 0.07 threshold. The honest presentation is: **NLP helps the early-warning objective through F2 and PR-AUC, and recall remains protected rather than degraded.** Full fusion with graph features reaches recall 0.6944 in the ablation; the feature-selected final model reaches recall 0.7000.

Key NLP features retained/used by the final model include `nlp_tenk_score_lag2q`, `nlp_topic_0`, `nlp_tenk_score_4q_mean_lag2q`, `nlp_yoy_cosine_sim`, and `nlp_yoy_cosine_dist`, all appearing in the top-20 SHAP list after training.

## Model Performance

| Metric | Value |
|---|---:|
| Selected feature count | 100 |
| Selected NLP features | 10 |
| CV Walk-Forward AUC | 0.9166 |
| Test ROC-AUC | 0.8198 |
| Test ROC-AUC 95% CI | [0.7724, 0.8614] |
| Test PR-AUC | 0.7259 |
| Test PR-AUC 95% CI | [0.6588, 0.7855] |
| Test F2-score | 0.6442 |
| Test recall | 0.7000 |
| Test precision | 0.4884 |
| Test Brier score | 0.0766 |
| Original Altman Z ROC-AUC | 0.6359 on 198 covered test rows |
| Original Altman Z PR-AUC | 0.0264 on 198 covered test rows |
| LR baseline ROC-AUC | 0.8402 |
| LR baseline PR-AUC | 0.6912 |

The original Altman Z-score path is restored in `build_x_fused.py`: when Module A exports zeroed Altman columns, fusion recomputes the original 1968 Altman score and X1-X5 components from raw quarterly financial statements under `crisisnet-data/Module_1/market_data/financials` or `crisisnet-data/Module_A/market_data/financials`. This restored nonzero Altman values for 198 rows in the current run.

Important coverage caveat: Altman coverage is `0/610` train rows and `198/1145` test rows, with only 3 positives in the covered test subset. Therefore:
- `zscore_only` is evaluated only on covered rows, not on zero-filled missing rows.
- `zscore_5factors` is marked unavailable because there is no covered train data for fitting.
- The `zscore_only` recall of 1.0000 is not strong evidence; it is based on only 3 covered positives and has very poor F2/PR-AUC.

## Corrected Ablation

| Configuration | Features | CV AUC | ROC-AUC | PR-AUC | F2 | Recall | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| zscore_only | 1 | N/A | 0.6359 | 0.0264 | 0.0877 | 1.0000 | 0.3744 |
| zscore_5factors | 6 | N/A | N/A | N/A | N/A | N/A | N/A |
| module_a_only | 187 | 0.8801 | 0.7890 | 0.6943 | 0.5967 | 0.6889 | 0.1173 |
| a_plus_b | 257 | 0.8965 | 0.8060 | 0.7107 | 0.6175 | 0.6889 | 0.0935 |
| a_plus_c | 209 | 0.8949 | 0.8086 | 0.7072 | 0.6244 | 0.6944 | 0.0930 |
| full_fusion | 279 | 0.8925 | 0.8171 | 0.7263 | 0.6300 | 0.6944 | 0.0772 |

## Per-Period Performance

| Period | n | Positives | ROC-AUC | F2 | Recall |
|---|---:|---:|---:|---:|---:|
| 2019-2020 crisis | 312 | 174 | 0.7933 | 0.7092 | 0.6897 |
| 2021+ post-crisis | 833 | 6 | 0.9831 | 0.2273 | 1.0000 |

The test positives remain concentrated in 2019-2020: 174 of 180 positives are in that window. Report the per-period split transparently.

## Failure Analysis

| Failure type | Count |
|---|---:|
| False negatives | 54 |
| False positives | 132 |
| False negatives with NLP coverage | 48 |
| False negatives without NLP coverage | 6 |

Top false-negative tickers: CHK 4, PSX 4, LNG 3, VLO 3, then several tickers with 2 missed quarters each. CHK still has zero NLP coverage and remains a key data-coverage limitation.

Top false-positive tickers: AR 14, OVV 13, APA 11, DVN 10, RRC 10.

## Files Added or Updated

| File | Change |
|---|---|
| `Module_D/build_x_fused.py` | FinBERT NLP forced; NLP lag/delta and interaction features; original Altman fallback from raw financials |
| `Module_D/feature_selection.py` | Selects 80-120 feature subset; current output is 100 features with 10 NLP features |
| `Module_D/train_fusion.py` | F2, recall, precision, bootstrap CIs, calibration plot, PR curve, per-period metrics, selected features, original Altman baseline |
| `Module_D/ablation_study.py` | Adds F2 and recall; NLP contribution summary; ablation table updated |
| `Module_D/failure_analysis.py` | Failure analysis JSON; fixed numeric NLP coverage handling |
| `dashboard/app.py` | Adds 7th Model Diagnostics tab |
| `Makefile` | Reproducible `make all` pipeline |
| `scripts/download_required_datasets.sh` | Dataset path preparation and SEC XBRL download hook |
| `scripts/push_to_github.sh` | GitHub push helper |
