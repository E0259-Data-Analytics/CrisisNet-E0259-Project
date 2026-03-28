"""
CrisisNet Module C — Feature Engineering
==========================================
Assembles the X_graph feature vector: a (company_id, quarter) indexed
DataFrame with all graph-derived features for use in Module D (fusion).

Each row represents a company's network position and contagion exposure
at a specific quarter. The features are:

Graph Structure Features (static per year):
  - betweenness_centrality
  - pagerank
  - eigenvector_centrality
  - in_degree_centrality
  - out_degree_centrality
  - clustering_coefficient
  - systemic_importance_score
  - contagion_vulnerability

Community Detection Features (per year):
  - louvain_community_id
  - louvain_community_label
  - louvain_modularity_Q
  - community_size
  - n_distressed_in_community
  - fragmentation_index          (vs previous year — LEADING INDICATOR)

DebtRank Contagion Features (static):
  - debtrank_exposure            (avg stress received from neighbour defaults)
  - max_contagion_in             (worst-case stress from any single source)
  - contagion_out                (total stress this company can transmit)
  - systemic_risk_contribution   (system-level impact of this company's default)
  - n_exposed_neighbours

The features are exported to X_graph.parquet with a (ticker, quarter) MultiIndex.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    COMPANY_UNIVERSE, QUARTERS, ANALYSIS_START_YEAR, ANALYSIS_END_YEAR,
    X_GRAPH_PARQUET, DATA_PROCESSED, EXPORTS
)


def year_to_quarters(year: int) -> List[str]:
    return [f"{year}Q{q}" for q in range(1, 5)]


def build_x_graph(
    centrality_yearly: pd.DataFrame,
    community_history: pd.DataFrame,
    fragmentation_df: pd.DataFrame,
    debtrank_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the final X_graph feature matrix.

    Parameters
    ----------
    centrality_yearly  : output of centrality.compute_yearly_centrality()
    community_history  : output of community_detection.run_dynamic_community_tracking()
    fragmentation_df   : per-year fragmentation index
    debtrank_features  : output of debtrank.compute_debtrank_exposure_features()

    Returns
    -------
    X_graph : DataFrame indexed by (ticker, quarter)
    """
    all_tickers = list(COMPANY_UNIVERSE.keys())
    all_quarters = QUARTERS
    # Build the full index
    rows = []
    for ticker in all_tickers:
        for quarter in all_quarters:
            year = int(quarter[:4])
            rows.append({"ticker": ticker, "quarter": quarter, "year": year})
    base_df = pd.DataFrame(rows)

    # ── Merge centrality features ──────────────────────────────────────────────
    cent_cols = [
        "ticker", "year", "betweenness_centrality", "pagerank",
        "eigenvector_centrality", "in_degree_centrality", "out_degree_centrality",
        "clustering_coefficient", "systemic_importance_score", "contagion_vulnerability",
        "in_degree", "out_degree",
    ]
    available_cent = [c for c in cent_cols if c in centrality_yearly.columns]
    df = base_df.merge(
        centrality_yearly[available_cent],
        on=["ticker", "year"],
        how="left",
    )

    # ── Merge community features ───────────────────────────────────────────────
    louvain_comm = community_history[
        community_history["algorithm"] == "louvain"
    ][["node", "year", "community_id", "community_label",
       "community_size", "n_distressed", "modularity_Q", "isolation_index"]].copy()
    louvain_comm = louvain_comm.rename(columns={
        "node":          "ticker",
        "community_id":  "louvain_community_id",
        "community_label":"louvain_community_label",
        "n_distressed":  "n_distressed_in_community",
        "modularity_Q":  "louvain_modularity_Q",
        "isolation_index":"community_isolation",
    })
    # Drop duplicate rows (keep one per ticker per year)
    louvain_comm = louvain_comm.drop_duplicates(subset=["ticker", "year"])

    df = df.merge(louvain_comm, on=["ticker", "year"], how="left")

    # ── Merge fragmentation index ──────────────────────────────────────────────
    frag_louvain = fragmentation_df[
        fragmentation_df["algorithm"] == "louvain"
    ][["year", "fragmentation_index", "n_communities"]].copy()
    df = df.merge(frag_louvain, on="year", how="left")

    # ── Merge DebtRank features ────────────────────────────────────────────────
    dr_cols = [
        "ticker", "year", "debtrank_exposure", "max_contagion_in",
        "contagion_out", "systemic_risk_contribution", "n_exposed_neighbours",
    ]
    available_dr = [c for c in dr_cols if c in debtrank_features.columns]
    if available_dr:
        df = df.merge(
            debtrank_features[available_dr].drop_duplicates(subset=["ticker", "year"]),
            on=["ticker", "year"],
            how="left",
        )

    # ── Add metadata ──────────────────────────────────────────────────────────
    df["name"]        = df["ticker"].map(lambda t: COMPANY_UNIVERSE.get(t, {}).get("name", t))
    df["subsector"]   = df["ticker"].map(lambda t: COMPANY_UNIVERSE.get(t, {}).get("subsector", "Unknown"))
    df["defaulted"]   = df["ticker"].map(lambda t: COMPANY_UNIVERSE.get(t, {}).get("defaulted", False))

    # ── Forward-fill centrality values within each ticker ─────────────────────
    # (Graph structure is relatively stable; use last known value for missing quarters)
    fill_cols = [
        "betweenness_centrality", "pagerank", "eigenvector_centrality",
        "in_degree_centrality", "out_degree_centrality", "clustering_coefficient",
        "systemic_importance_score", "contagion_vulnerability",
        "debtrank_exposure", "max_contagion_in", "contagion_out",
        "systemic_risk_contribution", "n_exposed_neighbours",
        "louvain_community_id", "louvain_community_label", "community_size",
        "n_distressed_in_community", "louvain_modularity_Q", "community_isolation",
    ]
    for col in fill_cols:
        if col in df.columns:
            df[col] = df.groupby("ticker")[col].transform(
                lambda x: x.ffill().bfill()
            )

    # ── Fragmentation index: fill 0 for first year ────────────────────────────
    if "fragmentation_index" in df.columns:
        df["fragmentation_index"] = df["fragmentation_index"].fillna(0.0)

    # ── Set MultiIndex ────────────────────────────────────────────────────────
    df = df.sort_values(["ticker", "quarter"]).reset_index(drop=True)

    log.info(f"X_graph shape: {df.shape}")
    log.info(f"Features: {[c for c in df.columns if c not in ['ticker', 'quarter', 'year', 'name', 'subsector', 'defaulted']]}")

    # Summary stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing = df[numeric_cols].isna().sum()
    if missing.sum() > 0:
        log.warning(f"Missing values:\n{missing[missing > 0]}")

    return df


def save_x_graph(df: pd.DataFrame, path: Path = X_GRAPH_PARQUET) -> None:
    """Save X_graph as parquet (primary) and CSV (human-readable)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    csv_path = path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    log.info(f"X_graph saved: {path} ({df.shape[0]} rows × {df.shape[1]} cols)")
    log.info(f"CSV copy: {csv_path}")


def generate_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary statistics table for the feature vector."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df[numeric_cols].describe().T
    summary["missing_pct"] = df[numeric_cols].isna().mean() * 100
    return summary


if __name__ == "__main__":
    import pickle
    from graph_builder import load_graph, build_and_save
    from community_detection import run_dynamic_community_tracking
    from centrality import compute_yearly_centrality
    from debtrank import compute_debtrank_exposure_features
    from config import GRAPH_PICKLE

    years = list(range(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR + 1))

    # Build or load graph
    try:
        G = load_graph(GRAPH_PICKLE)
    except Exception:
        G = build_and_save()

    # Compute features
    log.info("Computing yearly centrality...")
    centrality_df = compute_yearly_centrality(G, years)

    log.info("Running dynamic community tracking...")
    community_history, fragmentation_df = run_dynamic_community_tracking(G, years)

    log.info("Computing DebtRank exposure features...")
    debtrank_df = compute_debtrank_exposure_features(G, years)

    # Build and save X_graph
    x_graph = build_x_graph(centrality_df, community_history, fragmentation_df, debtrank_df)
    save_x_graph(x_graph)

    print(f"\nX_graph shape: {x_graph.shape}")
    print(x_graph.dtypes)
    print(x_graph.head(5).to_string())
