TODO — Module 2 (NLP: 10-K + Earnings Calls)

Goal
Produce the Module 2 feature set `X_nlp` as described in the report: topic proportions per quarter and sentiment trends over trailing 4 quarters per company, using 10-K filings and earnings call Q&A.

Status Checklist
[x] Install core NLP deps (`requirements.txt`)
[x] Run `tests/crisisnet_checks.py` and store logs in `tests/analysis_log.*`
[x] Scaffold Module 2 pipeline code (`module2_nlp_pipeline.py`)
[x] Add build script (`scripts/build_module2_features.py`)
[x] Add eval script (`scripts/eval_nlp_features.py`)
[x] Build sample features (lexicon sentiment) and save to `results/`
[x] Download and save FinBERT model to `finbert_model/`
[ ] Run full FinBERT pass over all earnings calls and save `results/X_nlp_finbert.parquet`
[ ] Save FinBERT evaluation output to `results/eval_nlp_finbert.txt`

Data Inputs (from report)
1. 10-K filings: `crisisnet-data/Module_2/10k_extracted/10-K/` (353 JSON files, 2015–2024).
2. Filings metadata: `crisisnet-data/Module_2/filings_metadata.csv`.
3. Earnings call Q&A: `crisisnet-data/Module_2/transcripts/huggingface/train.jsonl` (large file, ~3.6 GB).
4. Label reference: `crisisnet-data/splits/labels/distress_drawdowns/`.
5. Ticker-CIK map: `crisisnet-data/Module_1/sec_xbrl/ticker_cik_mapping.csv`.

Step-by-step Execution Plan
1. Verify environment
1. Confirm Python can import: `pandas`, `numpy`, `scikit-learn`, `torch`, `transformers`, `sentencepiece`, `bertopic`, `gensim`.
1. If GPU is available, confirm with `nvidia-smi` and `python -c "import torch; print(torch.cuda.is_available())"`.
1. Set a reproducible seed for any training or sampling.

2. Validate data presence and basic integrity
1. Confirm counts and file presence:
1. `10k` JSON count is 353.
1. `filings_metadata.csv` rows are 353.
1. `train.jsonl` exists and is >3 GB.
1. Confirm tickers in 10-Ks map to CIKs and tickers in `ticker_cik_mapping.csv`.

3. Re-run required tests and compare with prior logs
1. Run `python tests/crisisnet_checks.py`.
1. Ensure `tests/analysis_log.txt` and `tests/analysis_log.jsonl` updated.
1. Compare new metrics to the latest log block (2026-03-25 08:18:22 UTC) and explain any drift.
1. Flag the following known issues if they still appear:
1. Missing energy-default tickers in prices: 22 tickers absent from `stock_prices` train split.
1. Validation split has 0 distress/default labels.
1. High missing macro data (~26–31%).

4. Build 10-K processing pipeline
1. Load JSON, extract `item_1`, `item_1a`, `item_7`, `item_7a`, `item_8`.
1. Clean and normalize text (lowercase, remove boilerplate, de-duplicate lines).
1. Sentence-split per section to support sentence-level sentiment.
1. Aggregate per (ticker, filing year, section).

5. Earnings calls processing pipeline
1. Load Q&A from `train.jsonl` in streaming mode (file is large).
1. Standardize fields: `ticker`, `date`, `question`, `answer`.
1. Filter to tickers in the project’s 40-company universe.
1. Build quarterly buckets from `date`.

6. Topic modeling
1. Start with LDA on 10-K `item_7` (per report).
1. Evaluate topic coherence; pick a stable topic count.
1. Optional upgrade: BERTopic or LDA + Sentence-BERT embeddings for better semantic clusters.

7. Sentiment modeling
1. Use FinBERT for sentence-level sentiment on 10-K `item_1a` and earnings call Q&A.
1. Aggregate sentiment per (ticker, quarter) as mean and trend over trailing 4 quarters.

8. Feature assembly (`X_nlp`)
1. Build `X_nlp` with:
1. Topic proportions per quarter (from 10-K `item_7`).
1. Sentiment stats per quarter from 10-K and earnings calls.
1. Rolling 4-quarter sentiment trend.
1. Align all features to quarterly time index used in Module 1.

9. Baseline validation
1. Recompute `nlp_risk_baseline` in `tests/crisisnet_checks.py`.
1. Confirm AUCs are roughly stable:
1. `keyword_rate_auc` around 0.60.
1. `neg_rate_auc` around 0.41.
1. If they shift materially, diagnose changes in text extraction or label alignment.

10. Deliverables
1. Write the feature DataFrame to a stable path (e.g., `crisisnet-data/Module_2/features/X_nlp.parquet`).
1. Document:
1. model choices,
1. topic count,
1. aggregation windows,
1. tokenization and filtering rules,
1. any label alignment assumptions.

Recommended Models (GPU-friendly)
1. Sentiment
1. FinBERT (baseline, domain-specific).
1. DeBERTa or RoBERTa variants fine-tuned on financial sentiment (if available).
1. Longformer or BigBird if you want to model long 10-K sections end-to-end.

2. Topic modeling
1. BERTopic + Sentence-BERT embeddings (GPU accelerates embeddings).
1. LDA as a baseline for interpretability.

3. Earnings-call Q&A classification
1. FinBERT sentence-level on Q/A, aggregated to quarter.
1. Optional: Zero-shot or instruction-tuned models for risk-tagging (run with GPU, then distill).

Notes
1. The report explicitly ties Module 2 to `item_1a` (risk factors) and `item_7` (MD&A). Use those sections first.
1. The expected output is a quarterly feature vector `X_nlp` aligned with Module 1’s timeline.
