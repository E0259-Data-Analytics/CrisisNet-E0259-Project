# Module 1 — The Financial Heartbeat Monitor
## Time Series & Credit Risk Engine

> *"Just as a hospital monitors heart rate, blood pressure, and oxygen — Module 1 continuously monitors a company's financial vital signs and flags abnormal patterns."*

---

## Overview

Module 1 is the **time series credit risk engine** of CrisisNet. It processes three types of financial data — daily stock prices, quarterly accounting statements, and macroeconomic indicators — to produce a comprehensive feature vector (**X_ts**) that captures the financial health of 35 S&P 500 Energy sector companies.

The module trains four predictive models and benchmarks them against the industry-standard **Altman Z-Score** (1968), demonstrating that modern ML approaches achieve significantly higher accuracy.

---

## Key Results

| Model | Test AUC-ROC | Brier Score | Description |
|-------|-------------|-------------|-------------|
| Altman Z-Score | 0.500 | 0.688 | 5-factor linear model (1968 baseline) |
| **XGBoost** | **0.829** | **0.056** | Gradient boosted trees, walk-forward CV |
| **LSTM** | **0.869** | 0.593 | 4-quarter sequence model (PyTorch) |
| Cox PH | C-Index: 0.684 | — | Survival analysis (time-to-default) |

**Both XGBoost and LSTM exceed the project's 0.80 AUC-ROC target.**

---

## Directory Structure

```
Module_A/
├── README.md                    # This file
├── notebooks/
│   └── module1_pipeline.py      # Complete pipeline script
├── data/
│   ├── market_data/
│   │   ├── all_prices.parquet   # Daily stock prices (40 tickers)
│   │   └── financials/          # Quarterly statements per ticker
│   ├── credit_spreads/
│   │   └── fred_all_series.parquet  # 22 FRED macro series
│   ├── labels/
│   │   ├── energy_defaults_curated.csv
│   │   └── distress_from_drawdowns.csv
│   └── splits/                  # Pre-computed temporal splits
├── models/
│   ├── xgboost_credit_risk.json # Serialized XGBoost model
│   ├── lstm_credit_risk.pt      # PyTorch LSTM state dict
│   ├── cox_ph_model.pkl         # Fitted Cox PH model
│   ├── feature_scaler.pkl       # StandardScaler for inference
│   └── feature_columns.json     # Ordered feature list
└── results/
    ├── X_ts.parquet             # OUTPUT: Feature vector for Module D
    ├── module1_results.json     # All numerical results
    ├── Module_A_DETAILED_REPORT.md
    └── visualizations/          # 16 publication-quality plots
        ├── 01_roc_curves_comparison.png
        ├── 02_model_comparison.png
        ├── 03_xgboost_feature_importance.png
        ├── 04_calibration_curves.png
        ├── 05_volatility_heatmap.png
        ├── 06_merton_dd_timeline.png
        ├── 07_altman_zscore.png
        ├── 08_hmm_regime_detection.png
        ├── 09_oil_defaults_timeline.png
        ├── 10_feature_correlation.png
        ├── 11_precision_recall.png
        ├── 12_kaplan_meier.png
        ├── 13_cox_hazard_ratios.png
        ├── 14_confusion_matrices.png
        ├── 15_altman_vs_xgboost.png
        └── 16_company_risk_profiles.png
```

---

## How to Run

### Prerequisites
```bash
pip install pandas pyarrow numpy scikit-learn xgboost lifelines hmmlearn matplotlib seaborn plotly torch
```

### Execute
```bash
cd Module_A/notebooks
python module1_pipeline.py
```

The pipeline runs end-to-end in approximately 3–5 minutes and produces:
- All trained models in `models/`
- All visualizations in `results/visualizations/`
- The `X_ts.parquet` feature vector in `results/`
- A comprehensive `module1_results.json` with all metrics

---

## Feature Engineering (43 Features)

### Market Features (from daily stock prices)
- **30-day rolling volatility** — annualized std of log returns
- **60/90-day momentum** — price change over 60 and 90 trading days
- **Volume ratio** — recent volume vs 90-day average
- **Intraday range** — (High-Low)/Close spread
- **Max drawdown** — worst peak-to-trough decline in trailing 6 months

### Fundamental Ratios (from quarterly financials)
- **Altman Z-Score** (5 sub-components: X1 through X5)
- **Debt-to-Equity**, **Interest Coverage Ratio**, **Current Ratio**
- **Free Cash Flow**, **FCF-to-Debt**, **Debt-to-Assets**

### Structural Credit Model
- **Merton Distance-to-Default** — Nobel Prize-winning framework
- **Merton Default Probability** — risk-neutral PD from the Merton model
- **Asset Volatility**, **Leverage Ratio** — intermediate Merton outputs

### Macro/Credit Indicators (from FRED)
- **HY OAS** (ICE BofA High Yield spread) — primary credit stress barometer
- **VIX** — equity market fear index
- **WTI/Brent crude oil** — sector-specific critical input
- **Yield curve slope** (10Y - 2Y) — recession predictor
- **TED spread**, **Fed Funds**, **Unemployment** — macro controls
- **Oil/gas momentum** — QoQ price changes
- **Yield curve inversion flag** — binary recession warning

### HMM Regime Features
- **Regime stress fraction** — % of days in the quarter the 2-state HMM classified as "stress"
- Trained on VIX + HY OAS daily data

---

## Models in Detail

### 1. Altman Z-Score (Baseline, 1968)
The industry's oldest bankruptcy predictor. Five accounting ratios combined linearly:
```
Z = 1.2 × (WC/TA) + 1.4 × (RE/TA) + 3.3 × (EBIT/TA) + 0.6 × (MCap/TL) + 1.0 × (Rev/TA)
```
- Z < 1.81 = Distress Zone, Z > 2.99 = Safe Zone
- Limitation: backward-looking only, no market data, no macro context

### 2. XGBoost (Primary Model)
- 300 trees, depth 5, learning rate 0.05
- Walk-forward cross-validation (5 folds): **Mean AUC = 0.745 +/- 0.091**
- Trained with class imbalance weighting (scale_pos_weight = 5.84)
- Best overall calibration (Brier = 0.056)

### 3. LSTM (Sequence Model)
- 2-layer LSTM (hidden=64) with 4-quarter lookback window
- Learns temporal degradation patterns (e.g., 4 consecutive quarters of declining metrics)
- Highest test AUC (0.869) — captures patterns static models miss

### 4. Cox Proportional Hazard (Survival Model)
- Semi-parametric survival model from the `lifelines` library
- Outputs **time-to-default distributions**, not just binary predictions
- Concordance Index: 0.744 (train), 0.684 (test)
- Elastic Net regularization (penalizer=1.0, l1_ratio=0.5)

---

## Temporal Train/Val/Test Split

| Split | Period | Size | Purpose |
|-------|--------|------|---------|
| **Train** | 2015–2021 | 940 samples (195 distress) | Covers oil crash + COVID |
| **Validation** | 2022 | 140 samples (0 distress) | Hyperparameter tuning |
| **Test** | 2023–2025 | 455 samples (3 distress) | Held-out final evaluation |

The split is **temporal** to prevent data leakage — future information never enters training.

---

## Cancer Screening Analogy

| Cancer Screening | CrisisNet Module 1 |
|-----------------|-------------------|
| Blood pressure reading | 30-day stock volatility |
| Blood test panel | Quarterly financial ratios |
| X-ray/MRI scan | Merton structural credit model |
| Environmental exposure | Macro credit spreads (HY OAS, VIX) |
| Family history risk factor | HMM regime detection |
| Cancer staging (I–IV) | Altman Z zones (Safe/Grey/Distress) |
| Survival curve | Cox PH time-to-default distribution |
| AI-assisted diagnosis | XGBoost + LSTM ensemble |

---

## Output for Module D Fusion

The X_ts feature vector is exported as `results/X_ts.parquet` with:
- **Multi-index**: (ticker, Date)
- **43 feature columns** covering market, fundamental, structural, macro, and regime indicators
- **1,535 observations** (35 tickers × ~44 quarters)

This vector will be concatenated with X_nlp (Module 2) and X_graph (Module 3) for the final LightGBM fusion model in Module D.

---

## Technology Stack

| Library | Version | Purpose |
|---------|---------|---------|
| pandas + pyarrow | Latest | Data processing |
| numpy + scipy | Latest | Numerical computation |
| scikit-learn | Latest | Preprocessing, CV, metrics |
| xgboost | Latest | Gradient boosted classifier |
| PyTorch | Latest | LSTM neural network |
| lifelines | Latest | Cox PH survival analysis |
| hmmlearn | Latest | HMM regime detection |
| matplotlib + seaborn | Latest | Publication-quality visualizations |

---

*CrisisNet Module 1 | Data Analytics Course (E0 259) | 2025*
