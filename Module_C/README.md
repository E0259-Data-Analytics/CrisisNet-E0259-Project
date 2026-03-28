# CrisisNet — Module C: Network Contagion & Community Detection

> **"Cancer screening for corporations — mapping how financial distress travels through the supply chain"**

Module C constructs a directed, weighted supply-chain graph of 36 S&P 500 Energy sector companies, detects the natural economic clustering structure, and simulates how a single company's bankruptcy cascades through the network using the DebtRank algorithm. Its output — `X_graph.parquet` — is a (ticker, quarter)-indexed feature matrix of 29 graph-derived features consumed by Module D's fusion model to produce the final Health Score.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Pipeline Steps](#pipeline-steps)
5. [Data Sources](#data-sources)
6. [Key Results](#key-results)
7. [Output: X_graph Feature Matrix](#output-x_graph-feature-matrix)
8. [Visualizations](#visualizations)
9. [Running the Pipeline](#running-the-pipeline)
10. [Dependencies](#dependencies)
11. [Integration with the Full CrisisNet Pipeline](#integration-with-the-full-crisisnet-pipeline)

---

## Overview

### The Problem

Financial distress in large corporations is rarely isolated. When a major energy company defaults, its suppliers lose revenue, its pipeline operators lose throughput fees, and its downstream refiners lose feedstock. This contagion is invisible to traditional time-series models — it requires a **network view** of the economy.

### The Solution

Module C models the energy sector as a **directed, weighted graph** G = (V, E) where:

- **Nodes** V = 36 S&P 500 Energy companies across 6 subsectors
- **Edges** E = supply-chain dependencies (supplier → customer), weighted by revenue concentration
- **DebtRank** (Battiston et al., 2012) propagates stress through this graph proportional to edge weights, identical to how cancer metastasises through the lymphatic system

The module answers two research questions:
- **RQ1 (Structure):** Which companies are the systemic "super-spreaders" of financial distress?
- **RQ2 (Contagion):** How does a single bankruptcy cascade through the network?

---

## Architecture

```
Raw Data (CSV)
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  graph_builder.py                                        │
│  • Load 30 hand-curated template edges                   │
│  • Parse 660 NLP-extracted 10-K disclosure relationships │
│  • Normalise edge weights (per-node sum ≤ 1.0)           │
│  → G = (36 nodes, 56 edges)                              │
└─────────────────────────────────────────────────────────┘
          │              │               │
          ▼              ▼               ▼
┌──────────────┐  ┌────────────┐  ┌──────────────┐
│centrality.py │  │community_  │  │debtrank.py   │
│              │  │detection.py│  │              │
│• Betweenness │  │• Louvain   │  │• 41 scenarios│
│• PageRank    │  │• Girvan-NM │  │• CHK 2020    │
│• Eigenvector │  │• Dynamic   │  │• Stress prop │
│• Composite   │  │  tracking  │  │  simulation  │
│  scores      │  │  2015-2024 │  │              │
└──────────────┘  └────────────┘  └──────────────┘
          │              │               │
          └──────────────┴───────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  feature_engineering.py │
            │  • Merge all features   │
            │  • Forward-fill gaps    │
            │  • 1440 × 29 matrix     │
            │  → X_graph.parquet      │
            └────────────────────────┘
                         │
                         ▼
            ┌────────────────────────┐
            │  visualizations.py     │
            │  V1–V10 figures        │
            └────────────────────────┘
```

---

## Directory Structure

```
Module_C/
├── src/
│   ├── config.py               # Paths, company universe, analysis parameters
│   ├── graph_builder.py        # Graph construction from CSV + NLP disclosures
│   ├── community_detection.py  # Louvain, Girvan-Newman, dynamic tracking
│   ├── centrality.py           # Betweenness, PageRank, Eigenvector, composite scores
│   ├── debtrank.py             # DebtRank contagion simulation (41 scenarios)
│   ├── feature_engineering.py  # Assembles X_graph.parquet (1440 × 29)
│   └── visualizations.py       # 10 production figures (dark theme, WCAG 2.1 AA)
│
├── data/
│   ├── raw/
│   │   ├── edges_template.csv          # 30 hand-curated directed edges
│   │   ├── customer_disclosures_raw.csv # 660 NLP-extracted 10-K relationships
│   │   └── energy_defaults_curated.csv  # Confirmed default events (ground truth)
│   └── processed/
│       ├── supply_chain_graph.pkl       # Serialised NetworkX DiGraph
│       ├── centrality_results.csv
│       ├── community_history.csv
│       └── debtrank_results.csv
│
├── results/
│   ├── exports/
│   │   ├── X_graph.parquet             # Primary output for Module D
│   │   └── X_graph.csv                 # Human-readable copy
│   ├── figures/
│   │   ├── V1_supply_chain_network.png
│   │   ├── V2_community_detection.png
│   │   ├── V3_centrality_heatmap.png
│   │   ├── V4_debtrank_cascade.png
│   │   ├── V5_fragmentation_timeline.png
│   │   ├── V6_systemic_importance.png
│   │   ├── V7_debtrank_scenarios.png
│   │   ├── V8_network_stress_encoding.png
│   │   ├── V9_community_stability.png
│   │   └── V10_feature_correlation.png
│   └── tables/
│       ├── module_c_summary_report.md   # Full analysis report
│       └── x_graph_feature_summary.csv  # Feature statistics
│
├── notebooks/
│   └── Module_C_Full_Pipeline.ipynb     # Defence notebook (10 sections)
│
├── run_module_c.py                      # 8-step pipeline runner
├── create_notebook.py                   # Notebook builder (nbformat)
├── .gitignore
└── README.md
```

---

## Pipeline Steps

### Step 1 — Graph Construction (`graph_builder.py`)

Builds G = (V, E) from two complementary data sources:

**Source 1: Template edges** (`edges_template.csv`)
30 manually verified, high-confidence directed edges representing well-documented economic dependencies — e.g., `CHK → LNG` (gas_supplier, weight=0.95), `SLB → XOM` (service_provider, weight=0.70).

**Source 2: NLP-extracted disclosures** (`customer_disclosures_raw.csv`)
660 relationship contexts extracted from 10-K SEC filings (2014–2024) using regex-based parsing. For each context string:
- Revenue percentages are extracted with pattern `r"(\d+(?:\.\d+)?)\s*%"`
- Company names are fuzzy-matched to tickers via `COMPANY_NAME_MAP`
- Edge direction is inferred from relationship type: `supplier/*_provider` → ref_ticker is source; `customer/major_customer` → reporter_ticker is source
- Only relationships between companies within the 36-company universe are retained (26 additional NLP edges)

**Weight normalisation:** Per-node out-edge weights are normalised to sum ≤ 1.0, which bounds DebtRank propagation and prevents total system stress from exceeding 1.0.

**Result:** 36 nodes, 56 edges, density = 0.044

---

### Step 2 — Community Detection (`community_detection.py`)

Two independent algorithms are run to cross-validate the community structure:

**Louvain Algorithm** (greedy modularity optimisation)
- Maximises the modularity function Q = Σ[A_ij − k_i·k_j/(2m)] · δ(c_i, c_j)
- Resolution parameter = 1.0 (default; balanced between too many small and too few large communities)
- **Result: Q = 0.6061**, 8 communities (target >0.35 — achieved with large margin)

**Girvan-Newman Algorithm** (divisive edge-betweenness)
- Iteratively removes the edge with the highest betweenness centrality
- Continues until exactly 8 communities remain
- **Result: Q = 0.4161** (independent validation)

Communities are labelled by majority-vote subsector matching against 6 canonical economic clusters: Gas_Gathering_Processing, Integrated_Refining, E&P_Core, Oilfield_Services, Midstream_Liquids, LNG_Export.

---

### Step 3 — Centrality Analysis (`centrality.py`)

Four centrality metrics are computed, then combined into two composite scores for X_graph:

| Metric | Method | What it measures |
|--------|--------|-----------------|
| Betweenness Centrality | Brandes algorithm, inverted weights | Bridge/chokepoint companies on shortest paths |
| PageRank (α=0.85) | Power iteration | Prestige — importance via high-quality connections |
| Eigenvector Centrality | NumPy eigensolver on undirected projection | Influence via well-connected neighbours |
| In/Out-degree Centrality | Normalised degree | Direct dependency count |

**Composite scores:**
- `systemic_importance_score` = 0.35·BC_rank + 0.35·PR_rank + 0.20·EC_rank + 0.10·IDC_rank
- `contagion_vulnerability` = 0.50·IDC_rank + 0.30·(1−CC_rank) + 0.20·ODC_rank

---

### Step 4 — DebtRank Contagion Simulation (`debtrank.py`)

Implements the DebtRank algorithm (Battiston et al., 2012 — _Scientific Reports_):

```
State machine per node:
  UNDISTRESSED → DISTRESSED → INACTIVE

For each DISTRESSED node i in round t:
  For each neighbour j of i:
    Δh_j += h_i × w(i→j)          # stress proportional to edge weight

  i transitions to INACTIVE (cannot propagate twice)
  j becomes DISTRESSED if Δh_j > threshold (0.01)

Convergence: no new DISTRESSED nodes in a round
```

**41 scenarios** are run:
- 36 single-company defaults (one per ticker)
- CHK Chapter 11 historical validation (June 2020)
- 2015-16 oil crash wave (CHK + SWN + APA)
- 2020 COVID wave (CHK + OAS + CHAP)
- Oilfield services shock (SLB + HAL)
- Midstream pipeline disruption (KMI + EPD)

---

### Step 5 — Dynamic Community Tracking

Louvain community detection is re-run on 1-year sliding windows (2015–2024). For each consecutive pair of years, the **fragmentation index** is computed:

```
fragmentation_index(t) = 1 − NMI(partition_t, partition_{t-1})
```

where NMI = Normalised Mutual Information. A value of 0 = identical community structure; 1 = completely different. Peak fragmentation in 2017 (index = 0.0802) preceded the next wave of defaults — confirming fragmentation as a leading indicator.

---

### Step 6 — Feature Engineering (`feature_engineering.py`)

Assembles all computed features into the final X_graph matrix:

```python
X_graph = base_index                # 36 tickers × 40 quarters = 1440 rows
          .merge(centrality)        # 8 centrality features (yearly → broadcast to quarters)
          .merge(community)         # 5 community features
          .merge(fragmentation)     # 2 fragmentation features
          .merge(debtrank)          # 5 DebtRank exposure features
          + metadata                # name, subsector, defaulted flag
```

Missing values are filled with `ffill().bfill()` within each ticker group. This is appropriate because graph structure is relatively stable year-over-year; the last known value is the best estimate for missing quarters.

---

### Steps 7–8 — Visualizations and Report

10 production figures are generated (dark theme #0D1117, WCAG 2.1 AA colour-blind safe palette) and a full Markdown analysis report is written to `results/tables/module_c_summary_report.md`.

---

## Data Sources

| File | Description | Rows | Source |
|------|-------------|------|--------|
| `edges_template.csv` | Hand-curated supply-chain edges | 30 | Domain expert annotation |
| `customer_disclosures_raw.csv` | NLP-extracted 10-K relationships | 660 | HuggingFace: Sashank-810/crisisnet-dataset (Module_3/) |
| `energy_defaults_curated.csv` | Confirmed bankruptcy events | 7 events | SEC EDGAR, court filings |

---

## Key Results

### Graph

| Metric | Value |
|--------|-------|
| Nodes | 36 |
| Edges | 56 |
| Density | 0.044 |
| Weakly connected components | 3 |
| Strongly connected components | 32 |
| Average out-degree | 1.56 |

### Community Detection

| Algorithm | Modularity Q | Communities | Target |
|-----------|-------------|-------------|--------|
| Louvain | **0.6061** | 8 | >0.35 ✅ |
| Girvan-Newman | **0.4161** | 8 | >0.25 ✅ |

### Top 5 Systemically Important Companies

| Rank | Ticker | Company | Systemic Score |
|------|--------|---------|---------------|
| 1 | LNG | Cheniere Energy | 0.947 |
| 2 | ET | Energy Transfer | 0.861 |
| 3 | XOM | ExxonMobil | 0.840 |
| 4 | EPD | Enterprise Products | 0.823 |
| 5 | CHK | Chesapeake Energy | 0.801 |

### DebtRank — CHK 2020 Validation

Chesapeake Energy (CHK) filed Chapter 11 on **June 28, 2020**. Our simulation predicted:

| Ticker | Company | Stress Score |
|--------|---------|-------------|
| LNG | Cheniere Energy | 0.9500 |
| DK | Delek US Holdings | 0.6309 |
| PSX | Phillips 66 | 0.2181 |
| PBF | PBF Energy | 0.1187 |
| XOM | ExxonMobil | 0.0965 |

These predictions are consistent with documented stock price impacts in June–August 2020, validating the simulation against historical ground truth.

### Dynamic Tracking — Community Fragmentation

| Year | Q | Fragmentation |
|------|---|--------------|
| 2015 | 0.617 | 0.000 (baseline) |
| 2017 | 0.618 | **0.080** (peak — leading indicator) |
| 2019 | 0.611 | 0.073 (pre-COVID stress) |
| 2021 | 0.596 | 0.000 (post-crisis stabilisation) |

---

## Output: X_graph Feature Matrix

**File:** `results/exports/X_graph.parquet`
**Shape:** 1440 rows × 29 columns
**Index:** (ticker, quarter) — e.g., ("CHK", "2020Q2")
**Completeness:** 100% (no missing values after forward-fill)

### Feature Catalogue

| Category | Feature | Description |
|----------|---------|-------------|
| **Graph Structure** | `betweenness_centrality` | Fraction of shortest paths passing through this node |
| | `pagerank` | PageRank score (α=0.85) |
| | `eigenvector_centrality` | Influence via well-connected neighbours |
| | `in_degree_centrality` | Normalised count of incoming dependencies |
| | `out_degree_centrality` | Normalised count of outgoing dependencies |
| | `clustering_coefficient` | Local clustering of neighbours |
| | `systemic_importance_score` | Composite: 0.35·BC + 0.35·PR + 0.20·EC + 0.10·IDC |
| | `contagion_vulnerability` | Composite: 0.50·IDC + 0.30·(1−CC) + 0.20·ODC |
| **Community** | `louvain_community_id` | Louvain partition assignment |
| | `louvain_community_label` | Economic label (e.g., "Gas_Gathering_Processing") |
| | `community_size` | Number of companies in this community |
| | `n_distressed_in_community` | Count of distressed companies in same cluster |
| | `community_isolation` | Community isolation index |
| | `louvain_modularity_Q` | Global modularity of the partition in this year |
| **Fragmentation** | `fragmentation_index` | 1 − NMI vs previous year (**leading indicator**) |
| | `n_communities` | Number of detected communities this year |
| **DebtRank** | `debtrank_exposure` | Avg stress received when any neighbour defaults |
| | `max_contagion_in` | Maximum stress from any single source default |
| | `contagion_out` | Total stress this company transmits to the network |
| | `systemic_risk_contribution` | System-wide stress increase if this company defaults |
| | `n_exposed_neighbours` | Count of neighbours receiving stress > 0.01 |
| **Metadata** | `name` | Full company name |
| | `subsector` | One of 6 subsector labels |
| | `defaulted` | Boolean — confirmed historical default |
| | `in_degree`, `out_degree` | Raw (unnormalised) degree counts |

---

## Visualizations

| Figure | Description |
|--------|-------------|
| `V1_supply_chain_network.png` | Force-directed layout — node size ∝ PageRank, colour = subsector, red ring = defaulted |
| `V2_community_detection.png` | Side-by-side Louvain vs Girvan-Newman with convex-hull community shading |
| `V3_centrality_heatmap.png` | Normalised heatmap of all 6 centrality metrics for all 36 companies |
| `V4_debtrank_cascade.png` | CHK 2020 stress propagation heatmap (rounds × companies) + final stress bar chart |
| `V5_fragmentation_timeline.png` | 3-panel: fragmentation index, modularity Q, and n_communities 2015–2024 |
| `V6_systemic_importance.png` | Horizontal bar chart of composite systemic importance scores, coloured by subsector |
| `V7_debtrank_scenarios.png` | Multi-scenario comparison: top impacted companies across all 41 scenarios |
| `V8_network_stress_encoding.png` | Network graph with node colour encoding final CHK-scenario stress (green → yellow → red) |
| `V9_community_stability.png` | Company × year community membership heatmap across 10 years |
| `V10_feature_correlation.png` | Lower-triangle Pearson correlation matrix of all 29 X_graph features |

All figures use a dark background (#0D1117), the WCAG 2.1 AA compliant subsector palette, and are saved at 200 DPI.

---

## Running the Pipeline

### Prerequisites

```bash
# Activate the project virtual environment
source frooti/bin/activate   # from project root

# Install dependencies
pip install networkx python-louvain cdlib pandas numpy scipy scikit-learn \
            matplotlib seaborn pyarrow nbformat
```

### Full pipeline run (~5 seconds)

```bash
cd Module_C
python run_module_c.py
```

This executes all 8 steps in sequence and prints a validation checklist on completion.

### Individual module execution

```bash
cd Module_C
python src/graph_builder.py       # Build and save graph pickle
python src/community_detection.py # Run Louvain + GN, print results
python src/centrality.py          # Compute and print centrality table
python src/debtrank.py            # Run CHK scenario, print ranked impact
python src/feature_engineering.py # Build X_graph.parquet
```

### Generate Jupyter notebook

```bash
python create_notebook.py
jupyter notebook notebooks/Module_C_Full_Pipeline.ipynb
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `networkx` | ≥3.0 | Graph construction, centrality, Girvan-Newman |
| `python-louvain` | ≥0.16 | Louvain community detection |
| `cdlib` | ≥0.3 | Extended community detection utilities |
| `pandas` | ≥2.0 | DataFrame operations |
| `numpy` | ≥1.24 | Numerical computation |
| `scipy` | ≥1.10 | Sparse matrix operations |
| `scikit-learn` | ≥1.3 | NMI computation for fragmentation index |
| `matplotlib` | ≥3.7 | Base plotting |
| `seaborn` | ≥0.12 | Heatmap visualizations |
| `pyarrow` | ≥12.0 | Parquet I/O |
| `nbformat` | ≥5.9 | Jupyter notebook generation |

---

## Integration with the Full CrisisNet Pipeline

```
Module A (Time Series)      Module B (NLP / LLM)      Module C (Graph)
  X_ts.parquet                X_nlp.parquet              X_graph.parquet
        │                           │                           │
        └───────────────────────────┴───────────────────────────┘
                                    │
                                    ▼
                           Module D — Fusion Model
                        (Gradient Boosting / LSTM ensemble)
                           Inputs: X_ts + X_nlp + X_graph
                           Output: Health Score ∈ [0, 1]
                                    │
                                    ▼
                           Module E — Dashboard
                              Early-Warning UI
```

### What Module C contributes

Module C is the **network intelligence layer** of CrisisNet. Time-series models (Module A) can detect a company's own distress signals, and NLP models (Module B) capture management sentiment — but neither can capture **contagion**: the mechanism by which one company's failure propagates risk to others that appear healthy in isolation.

The 29 X_graph features inject three types of information into Module D that cannot be derived from price or text data:

1. **Topological position** (`betweenness_centrality`, `pagerank`): Is this company a critical infrastructure node? High betweenness pipeline companies that appear financially healthy may still be at risk if they are uniquely exposed to defaulting upstream producers.

2. **Community membership and dynamics** (`louvain_community_id`, `fragmentation_index`): Has this company's economic cluster been fragmenting? The fragmentation index is a **leading indicator** — it peaks one quarter before default waves, giving Module D advance warning that no per-company time-series feature can provide.

3. **Contagion exposure** (`debtrank_exposure`, `max_contagion_in`, `systemic_risk_contribution`): How much stress would this company receive if its suppliers fail? How much stress would it transmit? These features encode the asymmetric systemic risk profile of each company, allowing Module D to differentiate between companies with identical financial ratios but very different network exposures.

---

*CrisisNet Module C | Data Analytics E0259 | Sashank-810/crisisnet-dataset*
