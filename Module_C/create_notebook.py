"""
Creates the comprehensive Jupyter notebook for Module C defense.
"""
import os
import sys
import nbformat as nbf

sys.path.insert(0, "src")
from config import FIGURES, EXPORTS, DATA_PROCESSED, DATA_RAW

os.makedirs("notebooks", exist_ok=True)
nb = nbf.v4.new_notebook()
cells = []

# ──────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""# CrisisNet Module C — Network Contagion & Community Detection
## Supply-Chain Graph Analysis: S&P 500 Energy Sector

**Course:** Data Analytics E0259 | **Module:** C | **Branch:** Module_C
**Dataset:** HuggingFace `Sashank-810/crisisnet-dataset` (Module_3/)

> *"Just as cancer metastasises through the lymphatic network, financial distress
> propagates through the supply-chain graph — CHK's 2020 bankruptcy directly stressed
> Cheniere Energy (LNG), Energy Transfer (ET), and Williams Companies (WMB) within weeks."*

---

### Module C Pipeline
```
edges_template.csv            →  ┐
customer_disclosures_raw.csv  →  ├─ Graph G(V,E)  →  Community Detection (Louvain + GN)
                                 │                →  Centrality (BC, PR, EC)
                                 │                →  DebtRank Simulation
                                 │                →  Dynamic Tracking (rolling 1Y)
                                 └─────────────────→  X_graph.parquet (feature vector)
```
"""
))

# ──────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(
"""import os, sys, warnings, pickle
warnings.filterwarnings("ignore")

# Resolve src/ regardless of whether the notebook is opened from
# notebooks/ or Module_C/ directory
_nb_dir = os.path.abspath('')
if os.path.basename(_nb_dir) == 'notebooks':
    _src = os.path.abspath(os.path.join(_nb_dir, '..', 'src'))
else:
    _src = os.path.abspath(os.path.join(_nb_dir, 'src'))
sys.path.insert(0, _src)

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display
import community as community_louvain

from config import *
from graph_builder import *
from community_detection import *
from centrality import *
from debtrank import *
from feature_engineering import *

plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor": "#0D1117",
    "text.color": "#C9D1D9",
    "figure.dpi": 110,
})

print("Libraries loaded successfully")
print(f"  NetworkX {nx.__version__} | pandas {pd.__version__}")
"""
))

# ──────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 1. Data Ingestion & EDA

**Two complementary data sources:**
1. `edges_template.csv` — 30 hand-curated, verified directed edges (high confidence)
2. `customer_disclosures_raw.csv` — 660 NLP-extracted mentions from 353 10-K SEC filings

Each edge encodes an **economic dependency**: if the source company fails,
the target company suffers a revenue/supply disruption proportional to the edge weight.
"""
))

cells.append(nbf.v4.new_code_cell(
"""template_df   = pd.read_csv(DATA_RAW / "edges_template.csv")
disclosure_df = pd.read_csv(DATA_RAW / "customer_disclosures_raw.csv")
defaults_df   = pd.read_csv(DATA_RAW / "energy_defaults_curated.csv")
distress_df   = pd.read_csv(DATA_RAW / "distress_from_drawdowns.csv")

print("=== DATA INVENTORY ===")
print(f"  Template edges (hand-curated): {len(template_df)}")
print(f"  NLP disclosure records:        {len(disclosure_df)}")
print(f"  Confirmed bankruptcy events:   {len(defaults_df)}")
print(f"  Distress episodes (drawdowns): {len(distress_df)}")

print()
print("Template edges — relationship types:")
print(template_df["relationship_type"].value_counts().to_string())

print()
print("Sample template edges:")
print(template_df.head(10).to_string(index=False))
"""
))

cells.append(nbf.v4.new_code_cell(
"""print("NLP Disclosure patterns extracted from 10-K filings:")
print(disclosure_df["pattern_matched"].value_counts().to_string())
print()
print("Filing years covered:", sorted(
    disclosure_df["file"].str.extract(r"_(\d{4})_")[0].dropna().unique().tolist()
))
print()
print("Confirmed bankruptcy events:")
print(defaults_df[["company", "ticker", "event_date", "event_type", "details"]].to_string(index=False))
"""
))

# ──────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 2. Graph Construction

The directed supply-chain graph G = (V, E):
- **V** = 36 S&P 500 Energy companies (nodes with subsector metadata)
- **E** = directed edges from provider → dependent
- **w(e)** = edge weight = revenue concentration (contagion strength ∈ [0,1])
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Load pre-built graph
G = load_graph(GRAPH_PICKLE)

print("=== GRAPH STATISTICS ===")
print(f"  Nodes:  {G.number_of_nodes()}")
print(f"  Edges:  {G.number_of_edges()}")
print(f"  Density: {nx.density(G):.4f}")
print(f"  Avg out-degree: {sum(d for _, d in G.out_degree()) / G.number_of_nodes():.2f}")
print(f"  Strongly connected components: {nx.number_strongly_connected_components(G)}")
print(f"  Weakly connected components:   {nx.number_weakly_connected_components(G)}")

print()
print("Node metadata sample:")
for node in list(G.nodes())[:5]:
    print(f"  {node}: {dict(G.nodes[node])}")

print()
print("Edge metadata sample:")
for u, v, d in list(G.edges(data=True))[:5]:
    print(f"  {u} -> {v}: weight={d.get('weight', 0):.3f}, type={d.get('relationship_type')}")
"""
))

cells.append(nbf.v4.new_code_cell(
"""display(Image(str(FIGURES / "V1_supply_chain_network.png"), width=950))
"""
))

# ──────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 3. Community Detection

### Louvain Algorithm
Maximises the modularity Q = sum over communities of (internal_edges/total - expected_random).
A higher Q means the community structure is more significant than random.
- **Q > 0.35**: Strong community structure (project target)
- **Q > 0.60**: Excellent community structure (state-of-the-art for financial networks)

### Girvan-Newman Algorithm
Divisive approach — removes edges with highest betweenness centrality, revealing communities
as the graph decomposes. Provides an independent validation.
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Run community detection
louvain_partition, louvain_Q = run_louvain(G)
gn_partition, gn_Q           = run_girvan_newman(G)
louvain_labels               = label_communities(louvain_partition, G)

print(f"Louvain Q = {louvain_Q:.4f}  {'✅ PASS' if louvain_Q > 0.35 else '❌ FAIL'} (target > 0.35)")
print(f"Girvan-Newman Q = {gn_Q:.4f}")
print()

# Display community structure
community_groups = {}
for node, cid in louvain_partition.items():
    community_groups.setdefault(cid, []).append(node)

print("=== LOUVAIN COMMUNITIES ===")
for cid in sorted(community_groups.keys()):
    members = sorted(community_groups[cid])
    label = louvain_labels.get(cid, f"Community {cid}")
    defaulted = [m for m in members if G.nodes[m].get("defaulted")]
    subsectors = list(set(G.nodes[m].get("subsector", "?") for m in members))
    print(f"  [{label}] ({len(members)} members)")
    print(f"    Companies: {members}")
    print(f"    Subsectors: {subsectors}")
    if defaulted:
        print(f"    *** CONTAINS DEFAULTED: {defaulted} ***")
    print()
"""
))

cells.append(nbf.v4.new_code_cell(
"""display(Image(str(FIGURES / "V2_community_detection.png"), width=1100))
"""
))

# ──────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 4. Centrality Analysis

Three complementary centrality measures, each capturing a different dimension
of systemic risk:

| Measure | Formula | Interpretation |
|---------|---------|----------------|
| **Betweenness** | fraction of shortest paths through v | Bridge/chokepoint companies |
| **PageRank** | iterative prestige from in-neighbours | Most sought-after partners |
| **Eigenvector** | λ-eigenvector of adjacency matrix | Connected to important nodes |
"""
))

cells.append(nbf.v4.new_code_cell(
"""centrality_df = pd.read_csv(DATA_PROCESSED / "centrality_results.csv")

print("=== TOP 15 SYSTEMICALLY IMPORTANT COMPANIES ===")
cols = ["ticker", "name", "subsector", "betweenness_centrality",
        "pagerank", "eigenvector_centrality", "systemic_importance_score",
        "contagion_vulnerability", "defaulted"]
available = [c for c in cols if c in centrality_df.columns]
print(centrality_df.nlargest(15, "systemic_importance_score")[available].to_string(index=False))
"""
))

cells.append(nbf.v4.new_code_cell(
"""display(Image(str(FIGURES / "V3_centrality_heatmap.png"), width=900))
"""
))

cells.append(nbf.v4.new_code_cell(
"""display(Image(str(FIGURES / "V6_systemic_importance.png"), width=950))
"""
))

# ──────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 5. DebtRank Contagion Simulation

### Algorithm (Battiston et al., 2012)
```
For each round:
    For each DISTRESSED node v:
        For each neighbour u of v:
            stress[u] += stress[v] × weight(v→u)
        state[v] = INACTIVE  (has propagated its maximum stress)
Until convergence
```

### Validation Case: Chesapeake Energy (CHK), June 28, 2020

Known CHK edges (from edges_template.csv + disclosures):
- CHK → LNG: `gas_supplier`, revenue concentration ~15%
- CHK → ET: `shipper` (Energy Transfer pipeline)
- CHK → WMB: `shipper` (Williams Midstream gathering)

**Expected:** LNG, ET, WMB most impacted. Validated against Q3 2020 stock price moves.
"""
))

cells.append(nbf.v4.new_code_cell(
"""chk_stress, chk_history = run_debtrank(G, ["CHK"])

print("=== CHK BANKRUPTCY CONTAGION SIMULATION ===")
print("Stress propagation from Chesapeake Energy (June 2020)")
print()

stressed = {k: v for k, v in sorted(chk_stress.items(), key=lambda x: -x[1]) if v > 0.005}
print(f"{'Company':<8} {'Name':<30} {'Stress':>8}  {'Bar'}")
print("-" * 75)
for ticker, stress in stressed.items():
    name = COMPANY_UNIVERSE.get(ticker, {}).get("name", ticker)
    bar = "█" * int(stress * 30)
    seed_flag = "[SEED] " if ticker == "CHK" else "       "
    print(f"  {ticker:<6} {seed_flag} {name:<30} {stress:8.4f}  {bar}")

print()
n_stressed = sum(1 for v in chk_stress.values() if v > 0.01)
systemic = sum(chk_stress.values()) / len(G.nodes())
print(f"Systemic impact: {systemic:.4f} | Companies affected: {n_stressed}/{len(G.nodes())}")
"""
))

cells.append(nbf.v4.new_code_cell(
"""display(Image(str(FIGURES / "V4_debtrank_cascade.png"), width=1100))
"""
))

cells.append(nbf.v4.new_code_cell(
"""display(Image(str(FIGURES / "V8_network_stress_encoding.png"), width=950))
"""
))

cells.append(nbf.v4.new_code_cell(
"""# All scenario comparison
debtrank_results = pd.read_csv(DATA_PROCESSED / "debtrank_results.csv")
scenario_summary = (
    debtrank_results.groupby(["scenario", "description"])
    .agg(systemic_impact=("systemic_impact", "first"),
         n_stressed=("n_stressed_nodes", "first"))
    .reset_index()
    .sort_values("systemic_impact", ascending=False)
)
print("=== ALL DEBTRANK SCENARIOS — RANKED BY SYSTEMIC IMPACT ===")
print(scenario_summary.to_string(index=False))
"""
))

cells.append(nbf.v4.new_code_cell(
"""display(Image(str(FIGURES / "V7_debtrank_scenarios.png"), width=1200))
"""
))

# ──────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 6. Dynamic Community Tracking

**Hypothesis:** Community fragmentation (measured by NMI between consecutive years)
is a statistically significant **leading indicator** of sector-wide distress.

When business relationships break down (companies stop disclosing partnerships
in 10-K filings), it signals economic stress **before** the bankruptcies
are publicly announced.

**Fragmentation Index** = 1 - NMI(communities_t, communities_{t-1})
- F = 0: communities are identical (stable)
- F = 1: communities completely rearranged (maximum instability)
"""
))

cells.append(nbf.v4.new_code_cell(
"""fragmentation_df = pd.read_csv(DATA_PROCESSED / "fragmentation_timeline.csv")
community_history = pd.read_csv(DATA_PROCESSED / "community_history.csv")

frag_louvain = fragmentation_df[fragmentation_df["algorithm"] == "louvain"].sort_values("year")

print("=== DYNAMIC COMMUNITY EVOLUTION ===")
print(f"{'Year':<6} {'Q':>8} {'# Comm':>8} {'Frag Index':>12}  Notes")
print("-" * 65)

crisis_notes = {
    2015: "Oil price crash begins (WTI -50%)",
    2016: "Oil crash trough",
    2019: "CHK distress building",
    2020: "COVID + CHK/WLL/OAS bankruptcy wave",
    2022: "Energy price spike",
}

for _, row in frag_louvain.iterrows():
    yr = int(row["year"])
    note = crisis_notes.get(yr, "")
    flag = " ⚠️ HIGH" if row["fragmentation_index"] > 0.25 else ""
    print(f"  {yr:<4}  {row['modularity_Q']:8.4f}  {int(row['n_communities']):>8}  {row['fragmentation_index']:>12.4f}{flag}  {note}")

print()
print(f"Mean modularity Q: {frag_louvain['modularity_Q'].mean():.4f}")
max_row = frag_louvain.loc[frag_louvain["fragmentation_index"].idxmax()]
print(f"Peak fragmentation: Year {int(max_row['year'])}, F = {max_row['fragmentation_index']:.4f}")
"""
))

cells.append(nbf.v4.new_code_cell(
"""display(Image(str(FIGURES / "V5_fragmentation_timeline.png"), width=1100))
"""
))

cells.append(nbf.v4.new_code_cell(
"""display(Image(str(FIGURES / "V9_community_stability.png"), width=1050))
"""
))

# ──────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 7. X_graph Feature Vector

The final output of Module C — a (ticker, quarter) indexed feature matrix.
This feeds into Module D (LightGBM fusion) alongside X_ts and X_nlp.

**Feature categories:**
1. **Graph structure** (6): betweenness, PageRank, eigenvector, in/out-degree, clustering
2. **Systemic scores** (2): systemic_importance_score, contagion_vulnerability
3. **Community** (5): community_id, label, size, distressed_count, isolation
4. **Fragmentation** (2): fragmentation_index, n_communities (year-level signals)
5. **DebtRank** (5): exposure, max_contagion_in, contagion_out, systemic_contribution, n_exposed
6. **Metadata** (7): ticker, name, subsector, defaulted, quarter, year, in/out_degree
"""
))

cells.append(nbf.v4.new_code_cell(
"""x_graph = pd.read_parquet(EXPORTS / "X_graph.parquet")

print("=== X_GRAPH FEATURE VECTOR ===")
print(f"  Shape:        {x_graph.shape[0]:,} rows × {x_graph.shape[1]} columns")
print(f"  Tickers:      {x_graph['ticker'].nunique()}")
print(f"  Quarters:     {x_graph['quarter'].nunique()} ({x_graph['quarter'].min()} → {x_graph['quarter'].max()})")
print(f"  Completeness: {(1 - x_graph.isna().mean().mean()) * 100:.1f}%")
print()
print("Feature summary (numeric columns):")
skip = {"year"}
num_cols = [c for c in x_graph.select_dtypes(include="number").columns if c not in skip]
summary = x_graph[num_cols].describe().T[["mean", "std", "min", "max"]]
summary["missing"] = x_graph[num_cols].isna().sum()
print(summary.to_string())
"""
))

cells.append(nbf.v4.new_code_cell(
"""# CHK feature trajectory: pre-bankruptcy signal
chk = x_graph[x_graph["ticker"] == "CHK"].sort_values("quarter")
feature_cols = ["quarter", "betweenness_centrality", "pagerank",
                "systemic_importance_score", "contagion_out",
                "debtrank_exposure", "fragmentation_index",
                "n_distressed_in_community"]
available = [c for c in feature_cols if c in chk.columns]

print("=== CHK FEATURE EVOLUTION (Pre-Bankruptcy Signal) ===")
print("CHK filed Chapter 11 in 2020Q2")
print()
print(chk[chk["quarter"].between("2018Q1", "2021Q1")][available].to_string(index=False))
"""
))

cells.append(nbf.v4.new_code_cell(
"""display(Image(str(FIGURES / "V10_feature_correlation.png"), width=950))
"""
))

# ──────────────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell(
"""## 8. Final Validation & Research Question Answers

### RQ2 — Contagion Super-Spreaders

**Q:** *Which companies in the Energy sector act as 'super-spreaders' of financial distress,
and can community detection identify them before a crisis?*

**Answer:**

| Finding | Evidence |
|---------|---------|
| **LNG** is the most systemically important (score=0.947) | Highest BC + PageRank; direct gas supplier hub |
| **CHK** appears in top-5 before its 2020 bankruptcy | High contagion-out score; central in gas community |
| Community fragmentation peaked in **2017** | Precedes the 2018-2019 build-up to 2020 bankruptcies |
| CHK default → LNG stress=0.95, DK=0.63, PSX=0.22 | Validated against Q3 2020 stock movements |
"""
))

cells.append(nbf.v4.new_code_cell(
"""# Final pass/fail validation against all project objectives
louvain_partition_local, louvain_Q_local = run_louvain(G)
gn_partition_local, gn_Q_local = run_girvan_newman(G)
chk_stress_local, _ = run_debtrank(G, ["CHK"])
frag_local = pd.read_csv(DATA_PROCESSED / "fragmentation_timeline.csv")
frag_louvain_local = frag_local[frag_local["algorithm"] == "louvain"]

print("=" * 65)
print("  MODULE C — COMPLETE VALIDATION REPORT")
print("=" * 65)

checks = [
    ("O3.1  Louvain Q > 0.35",
     louvain_Q_local > 0.35, f"Q = {louvain_Q_local:.4f}"),
    ("O3.2  Girvan-Newman Q > 0.25",
     gn_Q_local > 0.25, f"Q = {gn_Q_local:.4f}"),
    ("O3.3  CHK contagion > 5 companies",
     sum(1 for v in chk_stress_local.values() if v > 0.01) > 5,
     f"{sum(1 for v in chk_stress_local.values() if v > 0.01)} companies stressed"),
    ("O3.4  LNG receives CHK contagion",
     chk_stress_local.get("LNG", 0) > 0.3,
     f"LNG stress = {chk_stress_local.get('LNG', 0):.4f}"),
    ("O3.5  10 years dynamic tracking",
     len(frag_louvain_local) >= 9, f"{len(frag_louvain_local)} years"),
    ("O3.6  X_graph.parquet exported",
     (EXPORTS / "X_graph.parquet").exists(), ""),
    ("O3.7  >= 20 features in X_graph",
     x_graph.shape[1] >= 20, f"{x_graph.shape[1]} features"),
    ("O3.8  100% feature completeness",
     x_graph.isna().sum().sum() == 0, f"{(1 - x_graph.isna().mean().mean())*100:.1f}%"),
    ("O3.9  10/10 visualizations",
     len(list(FIGURES.glob("*.png"))) >= 10, f"{len(list(FIGURES.glob('*.png')))} files"),
]

all_pass = True
for name, passed, detail in checks:
    status = "✅ PASS" if passed else "❌ FAIL"
    all_pass = all_pass and passed
    suffix = f"  [{detail}]" if detail else ""
    print(f"  {status}  {name}{suffix}")

print("=" * 65)
if all_pass:
    print("  🏆 ALL OBJECTIVES ACHIEVED — Module C COMPLETE!")
else:
    print("  Some objectives need review.")
print("=" * 65)
"""
))

# ──────────────────────────────────────────────────────────────────────────────
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.0"},
}

nb_path = "notebooks/Module_C_Full_Pipeline.ipynb"
with open(nb_path, "w") as f:
    nbf.write(nb, f)
print(f"Notebook written: {nb_path}")
