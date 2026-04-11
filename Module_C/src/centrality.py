"""
CrisisNet Module C — Centrality Metrics
=========================================
Computes three centrality measures for every node in the supply-chain graph:

1. Betweenness Centrality
   Identifies 'broker' companies that sit on many shortest paths.
   A company with high betweenness is a 'bridge' — if it fails,
   it disconnects large parts of the network.
   → High betweenness = systemic risk multiplier.

2. PageRank (adapted from Google's algorithm)
   A company has high PageRank if it is pointed to by many other
   important companies. High PageRank = most sought-after customer
   or service provider in the network.
   → High PageRank = systemically important regardless of direct connections.

3. Eigenvector Centrality
   Measures influence in the network — similar to PageRank but
   emphasises being connected to well-connected nodes.
   Integrated oil companies (XOM, CVX) score highest because
   they are connected to every subsector.

All three metrics are computed on the directed graph.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import COMPANY_UNIVERSE, DATA_PROCESSED, CENTRALITY_RESULTS


def compute_betweenness_centrality(G: nx.DiGraph, normalized: bool = True) -> Dict[str, float]:
    """
    Betweenness centrality: fraction of shortest paths that pass through each node.

    Uses edge weights as distances (higher weight = stronger relationship = shorter
    effective distance in supply-chain terms). We invert weights so high-weight
    edges represent short paths (easy contagion pathways).
    """
    # Invert weights: strong supply dependency = short path
    G_inv = G.copy()
    for u, v, d in G_inv.edges(data=True):
        G_inv[u][v]["inv_weight"] = 1.0 / max(d.get("weight", 0.1), 0.01)

    bc = nx.betweenness_centrality(
        G_inv,
        weight="inv_weight",
        normalized=normalized,
        endpoints=False,
    )
    return bc


def compute_pagerank(
    G: nx.DiGraph,
    alpha: float = 0.85,  # damping factor (standard)
    weight: str = "weight",
) -> Dict[str, float]:
    """
    PageRank on the directed graph.
    A company that is depended upon by many other important companies
    scores high here.
    """
    pr = nx.pagerank(G, alpha=alpha, weight=weight, max_iter=200, tol=1e-8)
    return pr


def compute_eigenvector_centrality(
    G: nx.DiGraph,
    max_iter: int = 1000,
) -> Dict[str, float]:
    """
    Eigenvector centrality: influence propagation on the undirected projection.
    """
    # Use undirected version (eigenvector centrality works better on undirected graphs)
    U = G.to_undirected()
    try:
        ec = nx.eigenvector_centrality_numpy(U, weight="weight")
    except Exception:
        # Fallback for disconnected graphs
        ec = nx.eigenvector_centrality(
            U, max_iter=max_iter, weight="weight", tol=1e-6
        )
    return ec


def compute_in_degree_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """
    In-degree centrality (weighted): total incoming dependency weight.
    Companies that are heavily depended upon by others.
    """
    in_deg = {}
    for node in G.nodes():
        in_deg[node] = sum(
            d.get("weight", 0.5)
            for _, _, d in G.in_edges(node, data=True)
        )
    # Normalise by max
    max_val = max(in_deg.values()) if in_deg else 1.0
    return {n: v / max_val for n, v in in_deg.items()}


def compute_out_degree_centrality(G: nx.DiGraph) -> Dict[str, float]:
    """
    Out-degree centrality (weighted): total outgoing dependency weight.
    Companies that themselves depend heavily on others.
    """
    out_deg = {}
    for node in G.nodes():
        out_deg[node] = sum(
            d.get("weight", 0.5)
            for _, _, d in G.out_edges(node, data=True)
        )
    max_val = max(out_deg.values()) if out_deg else 1.0
    return {n: v / max_val for n, v in out_deg.items()}


def compute_clustering_coefficient(G: nx.DiGraph) -> Dict[str, float]:
    """
    Local clustering coefficient on undirected projection.
    Low clustering = node sits at a bridge between otherwise disconnected groups
    → potential contagion chokepoint.
    """
    U = G.to_undirected()
    cc = nx.clustering(U, weight="weight")
    return cc


def compute_all_centrality_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Compute all six centrality metrics for every node. Returns a DataFrame.
    """
    log.info("Computing betweenness centrality...")
    bc  = compute_betweenness_centrality(G)
    log.info("Computing PageRank...")
    pr  = compute_pagerank(G)
    log.info("Computing eigenvector centrality...")
    ec  = compute_eigenvector_centrality(G)
    log.info("Computing weighted in/out-degree centrality...")
    idc = compute_in_degree_centrality(G)
    odc = compute_out_degree_centrality(G)
    log.info("Computing clustering coefficients...")
    cc  = compute_clustering_coefficient(G)

    rows = []
    for node in G.nodes():
        meta = COMPANY_UNIVERSE.get(node, {})
        rows.append({
            "ticker":                  node,
            "name":                    meta.get("name", node),
            "subsector":               meta.get("subsector", G.nodes[node].get("subsector", "Unknown")),
            "betweenness_centrality":  bc.get(node, 0.0),
            "pagerank":                pr.get(node, 0.0),
            "eigenvector_centrality":  ec.get(node, 0.0),
            "in_degree_centrality":    idc.get(node, 0.0),
            "out_degree_centrality":   odc.get(node, 0.0),
            "clustering_coefficient":  cc.get(node, 0.0),
            "in_degree":               G.in_degree(node),
            "out_degree":              G.out_degree(node),
            "defaulted":               G.nodes[node].get("defaulted", False),
        })

    df = pd.DataFrame(rows).set_index("ticker")

    # Composite systemic risk score (equal-weighted combination of centrality measures)
    df["systemic_importance_score"] = (
        0.35 * df["betweenness_centrality"].rank(pct=True) +
        0.35 * df["pagerank"].rank(pct=True) +
        0.20 * df["eigenvector_centrality"].rank(pct=True) +
        0.10 * df["in_degree_centrality"].rank(pct=True)
    )

    # Contagion vulnerability score (how exposed a company is to contagion from others)
    df["contagion_vulnerability"] = (
        0.50 * df["in_degree_centrality"].rank(pct=True) +
        0.30 * (1 - df["clustering_coefficient"].rank(pct=True)) +  # low clustering = more exposed
        0.20 * df["out_degree_centrality"].rank(pct=True)
    )

    log.info(f"Centrality metrics computed for {len(df)} nodes")
    log.info(f"\nTop 5 by Systemic Importance:")
    top5 = df.nlargest(5, "systemic_importance_score")[
        ["name", "subsector", "systemic_importance_score", "betweenness_centrality", "pagerank"]
    ]
    log.info(f"\n{top5.to_string()}")

    return df.reset_index()


def compute_yearly_centrality(
    G_full: nx.DiGraph,
    years: list,
) -> pd.DataFrame:
    """
    Compute centrality metrics for each year's subgraph.
    Used for the dynamic feature vector X_graph.
    """
    from graph_builder import build_yearly_subgraph

    all_dfs = []
    for year in years:
        G_year = build_yearly_subgraph(G_full, year)
        df_year = compute_all_centrality_metrics(G_year)
        df_year["year"] = year
        all_dfs.append(df_year)

    df_all = pd.concat(all_dfs, ignore_index=True)
    log.info(f"Yearly centrality computed: {len(years)} years × {G_full.number_of_nodes()} nodes")
    return df_all


if __name__ == "__main__":
    from graph_builder import load_graph
    from config import GRAPH_PICKLE

    G = load_graph(GRAPH_PICKLE)
    df = compute_all_centrality_metrics(G)
    df.to_csv(CENTRALITY_RESULTS, index=False)
    print(df.sort_values("systemic_importance_score", ascending=False).head(10).to_string())
