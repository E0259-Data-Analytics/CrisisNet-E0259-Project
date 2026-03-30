# CrisisNet Module 2 (NLP) — Results Report

Date: 2026-03-30 (Asia/Kolkata)

Summary
This report consolidates the Module 2 (NLP) pipeline build, artifacts, evaluations, and known issues.

Scope
- 10-K filings (2015–2024) from `crisisnet-data/Module_2/10k_extracted/10-K/`
- Filings metadata `crisisnet-data/Module_2/filings_metadata.csv`
- Earnings call Q&A `crisisnet-data/Module_2/transcripts/huggingface/train.jsonl`

Pipeline Overview
- Topic modeling: LDA on 10-K MD&A (`item_7`)
- Sentiment: FinBERT for sentence-level sentiment on risk factors (`item_1a`) and earnings calls
- Output features: quarterly topic proportions + sentiment aggregates + 4-quarter rolling sentiment

Artifacts (Saved)
- Feature sets:
  - `results/X_nlp.parquet` (lexicon sentiment baseline)
  - `results/X_nlp_finbert.parquet` (FinBERT sentiment)
- Evaluations:
  - `results/eval_nlp.txt`
  - `results/eval_nlp_finbert.txt`
- Model artifacts (repo root):
  - `lda_model_module2.joblib`
  - `lda_vectorizer_module2.joblib`
  - `module2_artifacts_meta.json`
  - `finbert_model/` (downloaded from Hugging Face)

Evaluation Results
Lexicon sentiment baseline (from `results/eval_nlp.txt`):
- Score column: `tenk_score_4q_mean`
- Samples: 318
- Positives: 57
- Negatives: 261
- AUC: 0.469

FinBERT sentiment (from `results/eval_nlp_finbert.txt`):
- Score column: `tenk_score_4q_mean`
- Samples: 318
- Positives: 57
- Negatives: 261
- AUC: 0.431

Validation and Test Logs
- Full data checks logged to:
  - `tests/analysis_log.txt`
  - `tests/analysis_log.jsonl`
- Latest log timestamp: 2026-03-28 05:55:53 UTC

Known Data Issues (from checks)
- 22 energy-default tickers missing from `stock_prices` train split.
- Validation split has 0 distress/default labels.
- Macro series missingness roughly 26–31%.

Model/Feature Notes
- FinBERT produces sentence-level sentiment; aggregation is quarterly mean + 4-quarter rolling mean.
- Topic modeling uses LDA for interpretability; embeddings-based BERTopic is available as an upgrade.

Repro Commands
- Build features (FinBERT, full pass):
  - `python scripts/build_module2_features.py --finbert-model finbert_model --auto-device --output results/X_nlp_finbert.parquet --save-artifacts`
- Evaluate:
  - `python scripts/eval_nlp_features.py --features results/X_nlp_finbert.parquet > results/eval_nlp_finbert.txt`

Next Steps (Optional)
- Evaluate `calls_score_4q_mean` and topic features directly against labels.
- Extend evaluation to validation/test splits once labels exist.
- Consider BERTopic + Sentence-BERT for richer topic semantics.
