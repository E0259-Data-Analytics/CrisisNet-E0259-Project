"""
CrisisNet Module C — Community Detection
==========================================
Implements two community detection algorithms on the supply-chain graph:

1. Louvain Algorithm  — modularity-maximising greedy optimisation
   (python-louvain / community package)
2. Girvan-Newman Algorithm — divisive edge-betweenness method
   (networkx implementation)

Both are run on the undirected projection of the supply-chain graph.

Key outputs:
  - Community partition (node → community_id)
  - Modularity score Q  (target > 0.35 per project spec)
  - Named community labels (economic interpretation)
  - Community fragmentation index per rolling window
"""

import logging
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain          # python-louvain
from networkx.algorithms.community import girvan_newman
from itertools import islice
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    COMPANY_UNIVERSE, LOUVAIN_RANDOM_STATE, GN_NUM_COMMUNITIES,
    DATA_PROCESSED, COMMUNITY_HISTORY
)


# ── Known community structure (economic ground truth for labelling) ────────────
ECONOMIC_COMMUNITIES = {
    "Gas_Gathering_Processing": {"EQT", "AR", "AM", "WMB", "RRC", "SWN", "CHK"},
    "Integrated_Refining":      {"XOM", "CVX", "KMI", "VLO", "PSX", "OXY"},
    "E&P_Core":                 {"COP", "EOG", "DVN", "FANG", "APA", "OVV", "CTRA"},
    "Oilfield_Services":        {"SLB", "HAL", "BKR", "NOV", "FTI"},
    "Midstream_Liquids":        {"EPD", "ET", "OKE", "TRGP", "MPC"},
    "LNG_Export":               {"LNG", "ET", "EQT"},
}


def graph_to_undirected_weighted(G: nx.DiGraph) -> nx.Graph:
    """
    Convert the directed supply-chain graph to undirected for community detection.
    When edges exist in both directions (a→b and b→a), sum their weights.
    """
    U = nx.Graph()
    U.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 0.5)
        if U.has_edge(u, v):
            U[u][v]["weight"] += w
        else:
            U.add_edge(u, v, weight=w, **{
                k: v2 for k, v2 in d.items() if k != "weight"
            })
    return U


def run_louvain(
    G: nx.DiGraph,
    resolution: float = 1.0,
    random_state: int = LOUVAIN_RANDOM_STATE,
) -> Tuple[Dict[str, int], float]:
    """
    Run the Louvain community detection algorithm.

    Parameters
    ----------
    G           : Supply-chain DiGraph
    resolution  : Resolution parameter (higher → more, smaller communities)
    random_state: For reproducibility

    Returns
    -------
    partition : dict mapping node → community_id
    modularity: float (Q value, higher = better; target > 0.35)
    """
    U = graph_to_undirected_weighted(G)
    partition = community_louvain.best_partition(
        U,
        weight="weight",
        resolution=resolution,
        random_state=random_state,
    )
    modularity = community_louvain.modularity(partition, U, weight="weight")
    n_communities = len(set(partition.values()))
    log.info(f"Louvain: {n_communities} communities, Q = {modularity:.4f}")
    return partition, modularity


def run_girvan_newman(
    G: nx.DiGraph,
    n_communities: int = GN_NUM_COMMUNITIES,
) -> Tuple[Dict[str, int], float]:
    """
    Run the Girvan-Newman algorithm (edge-betweenness divisive method).

    Parameters
    ----------
    G            : Supply-chain DiGraph
    n_communities: Target number of communities

    Returns
    -------
    partition : dict mapping node → community_id
    modularity: float (Q value)
    """
    U = graph_to_undirected_weighted(G)

    # Use weighted edge betweenness
    comp = girvan_newman(U)

    # Advance until we reach the desired number of communities
    # islice is safe: stops at the last partition if target exceeds max
    best_partition_sets = None
    best_q = -1.0

    for communities in islice(comp, n_communities * 2):
        if len(communities) >= n_communities:
            # Build partition dict
            partition = {}
            for cid, members in enumerate(communities):
                for node in members:
                    partition[node] = cid
            # Compute modularity
            try:
                q = community_louvain.modularity(partition, U, weight="weight")
                if q > best_q:
                    best_q = q
                    best_partition_sets = communities
            except Exception:
                pass
            if len(communities) >= n_communities:
                break

    if best_partition_sets is None:
        # Fallback: return all-in-one community
        return {n: 0 for n in G.nodes()}, 0.0

    partition = {}
    for cid, members in enumerate(best_partition_sets):
        for node in members:
            partition[node] = cid

    log.info(f"Girvan-Newman: {len(set(partition.values()))} communities, Q = {best_q:.4f}")
    return partition, best_q


def label_communities(
    partition: Dict[str, int],
    G: nx.DiGraph,
) -> Dict[int, str]:
    """
    Assign human-readable economic labels to each community by majority-voting
    against the known economic community structure.

    Returns
    -------
    dict mapping community_id → economic label
    """
    community_nodes: Dict[int, List[str]] = {}
    for node, cid in partition.items():
        community_nodes.setdefault(cid, []).append(node)

    labels = {}
    for cid, nodes in community_nodes.items():
        node_set = set(nodes)
        # Find the best-matching economic community
        best_label = f"Community_{cid}"
        best_overlap = 0
        for econ_label, econ_members in ECONOMIC_COMMUNITIES.items():
            overlap = len(node_set & econ_members)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = econ_label
        labels[cid] = best_label
    return labels


def compute_community_stats(
    partition: Dict[str, int],
    G: nx.DiGraph,
    modularity: float,
    algorithm: str,
    year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame summarising community membership and statistics.
    """
    rows = []
    community_nodes: Dict[int, List[str]] = {}
    for node, cid in partition.items():
        community_nodes.setdefault(cid, []).append(node)

    labels = label_communities(partition, G)

    for cid, nodes in community_nodes.items():
        subg = G.subgraph(nodes)
        internal_edges = subg.number_of_edges()
        # Internal vs external connectivity
        total_edges_out = sum(G.out_degree(n) for n in nodes)
        isolation = internal_edges / max(total_edges_out, 1)

        # Subsector composition
        subsectors = [G.nodes[n].get("subsector", "Unknown") for n in nodes]
        top_subsector = max(set(subsectors), key=subsectors.count)

        # Distressed companies in community
        distressed = [n for n in nodes if G.nodes[n].get("defaulted", False)]

        for node in nodes:
            rows.append({
                "algorithm":          algorithm,
                "year":               year,
                "node":               node,
                "community_id":       cid,
                "community_label":    labels[cid],
                "community_size":     len(nodes),
                "community_members":  ", ".join(sorted(nodes)),
                "internal_edges":     internal_edges,
                "isolation_index":    isolation,
                "top_subsector":      top_subsector,
                "n_distressed":       len(distressed),
                "distressed_members": ", ".join(sorted(distressed)),
                "modularity_Q":       modularity,
                "subsector":          G.nodes[node].get("subsector", "Unknown"),
                "defaulted":          G.nodes[node].get("defaulted", False),
            })

    return pd.DataFrame(rows)


def compute_fragmentation_index(
    partition_t1: Dict[str, int],
    partition_t2: Dict[str, int],
) -> float:
    """
    Compute the community fragmentation index between two time periods.

    Fragmentation is defined as:
        F = 1 - NMI(partition_t1, partition_t2)

    where NMI is Normalized Mutual Information.
    F = 0 means communities are identical.
    F = 1 means communities are completely rearranged (maximum fragmentation).

    High fragmentation preceding a crisis is a known leading indicator.
    """
    from sklearn.metrics import normalized_mutual_info_score

    nodes = sorted(set(partition_t1.keys()) & set(partition_t2.keys()))
    if not nodes:
        return 1.0

    labels_t1 = [partition_t1[n] for n in nodes]
    labels_t2 = [partition_t2[n] for n in nodes]

    nmi = normalized_mutual_info_score(labels_t1, labels_t2, average_method="arithmetic")
    return 1.0 - nmi


def run_dynamic_community_tracking(
    G_full: nx.DiGraph,
    years: List[int],
    algorithm: str = "louvain",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run community detection on rolling 1-year windows (one graph per year).

    Tracks:
    - Community membership evolution per node per year
    - Fragmentation index between consecutive years (F_t = fragmentation t vs t-1)

    Returns
    -------
    community_history_df : per-node-per-year community membership
    fragmentation_df     : per-year fragmentation index
    """
    from graph_builder import build_yearly_subgraph

    partitions: Dict[int, Dict[str, int]] = {}
    all_stats  = []
    frag_rows  = []

    prev_partition = None
    for year in years:
        G_year = build_yearly_subgraph(G_full, year)

        if algorithm == "louvain":
            partition, modularity = run_louvain(G_year)
        else:
            partition, modularity = run_girvan_newman(G_year)

        partitions[year] = partition
        stats_df = compute_community_stats(partition, G_year, modularity, algorithm, year)
        all_stats.append(stats_df)

        # Fragmentation vs previous year
        if prev_partition is not None:
            frag = compute_fragmentation_index(prev_partition, partition)
        else:
            frag = 0.0
        frag_rows.append({
            "year":                year,
            "fragmentation_index": frag,
            "modularity_Q":        modularity,
            "n_communities":       len(set(partition.values())),
            "algorithm":           algorithm,
        })
        log.info(f"  Year {year}: Q={modularity:.4f}, "
                 f"communities={len(set(partition.values()))}, frag={frag:.4f}")

        prev_partition = partition

    community_history_df = pd.concat(all_stats, ignore_index=True)
    fragmentation_df     = pd.DataFrame(frag_rows)

    log.info(f"Dynamic tracking complete: {len(years)} years, algorithm={algorithm}")
    return community_history_df, fragmentation_df


if __name__ == "__main__":
    from graph_builder import load_graph
    from config import GRAPH_PICKLE

    G = load_graph(GRAPH_PICKLE)
    partition, modularity = run_louvain(G)
    print(f"Louvain Q = {modularity:.4f}")
    labels = label_communities(partition, G)
    for cid, label in sorted(labels.items()):
        members = [n for n, c in partition.items() if c == cid]
        print(f"  {label}: {sorted(members)}")
