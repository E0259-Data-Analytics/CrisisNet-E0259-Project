CrisisNet
Module 2 — The Language Analysis Engine (NLP Text Data)

Detailed Analysis Report
Generated: April 6, 2026
Dataset: HuggingFace Sashank-810/crisisnet-dataset (Module_2)
Universe: 40 S&P 500 Energy Companies, 2015–2024
Data Analytics Course (E0 259)
Indian Institute of Science, Bangalore

"Just as a radiologist looks for subtle changes in tissue texture to catch early cancer, Module 2 looks for subtle
changes in managerial language to catch early financial distress."

Table of Contents
1. Executive Summary
2. Role in the CrisisNet Pipeline
3. Data Sources & Coverage
4. Text Extraction & Cleaning
5. Feature Engineering (X_nlp)
6. Models & Algorithms
7. Evaluation Methodology
8. Results & Diagnostics
9. Artifacts & Reproducibility
10. Limitations
11. Recommended Next Improvements
12. Output for Module D

1. Executive Summary
Module 2 is the NLP intelligence layer of CrisisNet. It processes two sources of corporate language: 10-K risk
factors (Item 1a) and earnings-call Q&A. It uses topic modeling (LDA) on MD&A (Item 7) and sentiment scoring
(FinBERT) on Item 1a and calls, producing a quarterly feature vector X_nlp. The latest run produces 318
(ticker, quarter) rows and 29 feature columns. The strongest single signal is the topic shift feature (topic_kl_shift,
AUC 0.613) and the rolling 4-quarter 10-K sentiment mean (AUC 0.570). Earnings-call features are sparse and
currently unreliable due to low positive label counts in those quarters.

2. Role in the CrisisNet Pipeline
Module 2 provides language-based leading indicators that complement Module 1’s numerical time-series
signals and Module C’s network contagion metrics. The output X_nlp is aligned to quarterly time bins and
is consumed directly by the final fusion model (Module D). It is explicitly designed to detect early shifts in
managerial tone and discussion topics 2–4 quarters before distress events.

3. Data Sources & Coverage
10-K Annual Filings
- Source: crisisnet-data/Module_2/10k_extracted/10-K/
- Count: 353 JSON files
- Coverage: 2015–2024
- Sections used: Item 1a (Risk Factors), Item 7 (MD&A)

Filings Metadata
- Source: crisisnet-data/Module_2/filings_metadata.csv
- Count: 353 rows (CIK, filing date, etc.)

Earnings Call Q&A
- Source: crisisnet-data/Module_2/transcripts/huggingface/train.jsonl
- Size: ~3.89 GB
- Coverage: 2015–2024 (streamed; filtered to 40-company universe)

Labels
- Baseline labels: crisisnet-data/splits/labels/distress_drawdowns/train.parquet
- Chapter 11 2022–2025 labels: deferred (not available in repo)

4. Text Extraction & Cleaning
- Parsed 10-K JSON and extracted Item 1a and Item 7 sections.
- Cleaned text by lowercasing, collapsing whitespace, and removing empty lines.
- Earnings calls loaded in streaming mode; question and answer text concatenated per record.

5. Feature Engineering (X_nlp)
Output features (29 columns, excluding ticker/quarter):
- Topic proportions: topic_0 … topic_11 (12 features)
- 10-K FinBERT sentiment: tenk_pos, tenk_neg, tenk_neu, tenk_score
- Earnings-call FinBERT sentiment: calls_pos, calls_neg, calls_neu, calls_score
- Uncertainty lexicon rate: tenk_uncertainty_rate, calls_uncertainty_rate
- Rolling 4-quarter means: tenk_score_4q_mean, calls_score_4q_mean,
  tenk_uncertainty_4q_mean, calls_uncertainty_4q_mean
- Sentiment deltas (QoQ change): tenk_score_delta, calls_score_delta
- Topic shift: topic_kl_shift (KL divergence between consecutive quarters)

Feature matrix summary:
- Rows: 318
- Columns: 31 total (29 features + ticker + quarter)
- Calls channel sparsity: ~86% of calls features are missing due to limited quarterly matches

6. Models & Algorithms
Topic Modeling
- LDA (sklearn) with 12 topics
- Vectorizer: CountVectorizer, max_features=6000, min_df=5, stop_words=english
- Trained on Item 7 (MD&A)

Sentiment Modeling
- FinBERT (ProsusAI/finbert)
- Sentence-level scoring for Item 1a and calls
- Aggregated mean sentiment per (ticker, quarter)

Uncertainty Feature
- Loughran-McDonald-inspired uncertainty lexicon
- Rate computed as term count / token count

7. Evaluation Methodology
- AUC computed using Mann-Whitney U rank test
- Lookahead window: 365 days (annual risk horizon)
- Labels from distress_drawdowns train split (2015–2021)

8. Results & Diagnostics
Top features by AUC (latest run):
- calls_neg: 0.826 (only 2 positives; unreliable)
- topic_kl_shift: 0.613
- topic_1: 0.578
- tenk_score_4q_mean: 0.570
- topic_7: 0.547

Observations:
- Topic shift (topic_kl_shift) is the best stable single feature.
- Calls features are sparse; calls-based AUCs are unstable due to low positives.
- LDA topics are dominated by generic financial terms; a clear distress topic is not yet isolated.

9. Artifacts & Reproducibility
Core artifacts:
- results/X_nlp_finbert.parquet (latest feature set)
- results/eval_nlp_finbert.txt (baseline eval)
- /tmp/eval_all_cols.txt (all-column eval output)
- results/lda_topics.txt (top words per topic)
- lda_model_module2.joblib
- lda_vectorizer_module2.joblib
- finbert_model/ (weights)

Repro command:
python3 scripts/build_module2_features.py \
  --finbert-model finbert_model --auto-device \
  --output results/X_nlp_finbert.parquet --save-artifacts

10. Limitations
- Quarterly alignment is noisy for annual 10-K features; no 10-Q ingestion yet.
- Earnings-call features have high missingness for several quarters.
- LDA topics are dominated by boilerplate and company-specific terms.
- Chapter 11 labels (2022–2025) not available locally; out-of-sample validation pending.

11. Recommended Next Improvements
- Add 10-Q filings to densify quarterly coverage.
- Replace LDA with BERTopic + sentence embeddings for cleaner topics.
- Add syntactic/hedging features (modal verbs, uncertainty patterns).
- Evaluate against Chapter 11 labels when available.

12. Output for Module D
The output X_nlp is a quarterly feature matrix aligned to Module 1’s timeline and ready for fusion:
- Path: results/X_nlp_finbert.parquet
- Index columns: ticker, quarter
- Features: 29 NLP features as listed above

