"""
CrisisNet Module C — Graph Builder
====================================
Constructs the directed supply-chain graph G = (V, E) from two data sources:
  1. edges_template.csv  — 30 hand-curated, verified directed edges
  2. customer_disclosures_raw.csv — 660 NLP-extracted relationships from 10-K filings

The graph is weighted; edge weights represent revenue concentration (contagion strength).
Every node carries metadata: subsector, CIK, default flag, centrality scores (added later).
"""

import re
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Import local config
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EDGES_TEMPLATE, CUSTOMER_DISCLOSURES, COMPANY_UNIVERSE,
    CIK_TO_TICKER, RELATIONSHIP_WEIGHTS, DEFAULT_EVENTS,
    GRAPH_PICKLE, DATA_PROCESSED
)

# ── Company name → ticker fuzzy lookup ────────────────────────────────────────
COMPANY_NAME_MAP = {
    "exxonmobil": "XOM", "exxon mobil": "XOM", "exxon": "XOM",
    "chevron": "CVX",
    "occidental": "OXY", "oxy": "OXY",
    "conocophillips": "COP", "conoco": "COP",
    "eog resources": "EOG", "eog": "EOG",
    "devon energy": "DVN", "devon": "DVN",
    "diamondback": "FANG",
    "apa corporation": "APA", "apache": "APA",
    "ovintiv": "OVV", "encana": "OVV",
    "coterra": "CTRA", "cabot oil": "CTRA",
    "schlumberger": "SLB", "slb": "SLB",
    "halliburton": "HAL",
    "baker hughes": "BKR",
    "technipfmc": "FTI", "fmc technologies": "FTI",
    "nov inc": "NOV", "national oilwell": "NOV",
    "valero": "VLO",
    "marathon petroleum": "MPC",
    "phillips 66": "PSX",
    "delek": "DK",
    "pbf energy": "PBF",
    "kinder morgan": "KMI",
    "williams": "WMB", "williams companies": "WMB",
    "oneok": "OKE",
    "energy transfer": "ET",
    "enterprise products": "EPD",
    "targa resources": "TRGP",
    "antero midstream": "AM",
    "eqt corporation": "EQT", "eqt": "EQT",
    "antero resources": "AR", "antero": "AR",
    "range resources": "RRC",
    "chesapeake energy": "CHK", "chesapeake": "CHK",
    "southwestern energy": "SWN",
    "cheniere energy": "LNG", "cheniere": "LNG",
    "hess": "HES",
    "marathon oil": "MRO",
    "pioneer natural resources": "PXD", "pioneer": "PXD",
}


def extract_revenue_pct_from_context(text: str) -> Optional[float]:
    """
    Extract the first revenue percentage mentioned in a disclosure text.
    Handles patterns like '28% of our revenues', 'approximately 15 percent',
    'accounted for 10%', etc.
    """
    patterns = [
        r'(\d{1,3}(?:\.\d+)?)\s*%\s*(?:of\s+(?:our\s+)?(?:total\s+)?(?:revenues?|sales|net\s+sales))',
        r'accounted\s+for\s+(?:approximately\s+)?(\d{1,3}(?:\.\d+)?)\s*%',
        r'approximately\s+(\d{1,3}(?:\.\d+)?)\s*(?:percent|%)',
        r'(\d{1,3}(?:\.\d+)?)\s*percent\s+of\s+(?:our\s+)?(?:revenues?|sales)',
        r'(\d{1,3}(?:\.\d+)?)\s*%',  # fallback: any percentage
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            pct = float(m.group(1)) / 100.0
            # Sanity check: revenue concentration should be between 5% and 100%
            if 0.05 <= pct <= 1.0:
                return pct
    return None


def extract_company_ticker_from_context(text: str) -> Optional[str]:
    """
    Fuzzy-match company names from disclosure text to known tickers.
    Returns the first matched ticker, or None.
    """
    text_lower = text.lower()
    for name, ticker in COMPANY_NAME_MAP.items():
        if name in text_lower:
            return ticker
    # Also look for ticker symbols in ALL-CAPS (e.g., "(XOM)", "ExxonMobil (XOM)")
    ticker_pattern = r'\b(' + '|'.join(COMPANY_UNIVERSE.keys()) + r')\b'
    m = re.search(ticker_pattern, text)
    if m:
        return m.group(1)
    return None


def infer_relationship_type(pattern_matched: str, match_text: str) -> str:
    """Map NLP pattern to relationship type category."""
    p = pattern_matched.lower()
    t = match_text.lower()
    if 'supply agreement' in p or 'supply agreement' in t:
        return 'supply_agreement'
    if 'major customer' in p or 'significant customer' in p or 'largest customer' in p:
        return 'major_customer'
    if '10%' in p or 'accounted for' in p:
        return 'customer'
    if 'supplier' in p or 'vendor' in p:
        return 'service_provider'
    if 'concentration' in t:
        return 'customer'
    return 'unknown'


def load_template_edges(path: Path) -> pd.DataFrame:
    """Load and validate the 30 hand-curated edges from edges_template.csv."""
    df = pd.read_csv(path)
    required = {"source", "target", "relationship_type", "description"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    # Assign default contagion weight based on relationship type
    df["weight"] = df["relationship_type"].map(
        lambda r: RELATIONSHIP_WEIGHTS.get(r, RELATIONSHIP_WEIGHTS["unknown"])
    )
    df["source_file"]  = "edges_template"
    df["year"]         = None
    df["revenue_pct"]  = df["weight"]  # use relationship weight as proxy
    df["confidence"]   = "high"        # hand-curated = high confidence
    log.info(f"Loaded {len(df)} template edges")
    return df


def load_and_parse_disclosure_edges(path: Path, min_confidence: str = "medium") -> pd.DataFrame:
    """
    Parse customer_disclosures_raw.csv to extract additional directed edges.

    Each row is an NLP match from a 10-K filing. We:
      1. Extract the reporting company ticker from CIK in the filename.
      2. Extract the customer/supplier ticker from the match_text/context.
      3. Extract revenue concentration (edge weight) from context.
      4. Infer the relationship direction and type.

    Returns a DataFrame of edges (source, target, weight, year, relationship_type, ...).
    """
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df)} raw disclosure records")

    # Extract CIK and year from filename
    df["cik"]  = df["file"].str.extract(r'^(\d+)_')
    df["year"] = df["file"].str.extract(r'_(\d{4})_').astype(float).astype("Int64")

    # Map CIK → reporting company ticker
    df["reporter_ticker"] = df["cik"].map(CIK_TO_TICKER)
    n_unmapped = df["reporter_ticker"].isna().sum()
    if n_unmapped:
        log.warning(f"{n_unmapped} rows have unmapped CIKs → dropped")
    df = df.dropna(subset=["reporter_ticker"])

    # Extract revenue percentage from context
    df["revenue_pct"] = df["context"].apply(extract_revenue_pct_from_context)

    # Extract referenced company ticker from match_text + context
    df["ref_ticker"] = df.apply(
        lambda r: extract_company_ticker_from_context(r["match_text"] + " " + r["context"]),
        axis=1
    )

    # Drop rows where we couldn't identify both endpoints OR the same company references itself
    df = df.dropna(subset=["ref_ticker"])
    df = df[df["reporter_ticker"] != df["ref_ticker"]]

    # Infer relationship type
    df["relationship_type"] = df.apply(
        lambda r: infer_relationship_type(r["pattern_matched"], r["match_text"]), axis=1
    )

    # Determine edge direction:
    # If relationship is supplier/vendor → ref_ticker provides services TO reporter_ticker
    #   edge: ref_ticker → reporter_ticker  (ref is source/provider)
    # If relationship is customer → reporter_ticker sells TO ref_ticker
    #   edge: reporter_ticker → ref_ticker  (reporter is source/provider)
    supplier_patterns = {"supply_agreement", "service_provider", "equipment_supplier"}
    customer_patterns = {"major_customer", "customer"}

    edges = []
    for _, row in df.iterrows():
        rtype = row["relationship_type"]
        rep   = row["reporter_ticker"]
        ref   = row["ref_ticker"]
        pct   = row["revenue_pct"] if pd.notna(row["revenue_pct"]) else None
        yr    = row["year"]

        if rtype in supplier_patterns:
            # ref_ticker is the supplier → ref → reporter
            source, target = ref, rep
        else:
            # reporter_ticker is the supplier → reporter → ref
            source, target = rep, ref

        # Only include edges between companies in our universe
        if source in COMPANY_UNIVERSE and target in COMPANY_UNIVERSE:
            rel_weight = RELATIONSHIP_WEIGHTS.get(rtype, RELATIONSHIP_WEIGHTS["unknown"])
            final_weight = pct if pct is not None else rel_weight * 0.6  # lower confidence without pct
            confidence = "high" if pct is not None else "medium"

            edges.append({
                "source":            source,
                "target":            target,
                "relationship_type": rtype,
                "description":       row["match_text"][:80],
                "weight":            final_weight,
                "revenue_pct":       pct,
                "year":              int(yr) if pd.notna(yr) else None,
                "source_file":       row["file"],
                "confidence":        confidence,
            })

    edges_df = pd.DataFrame(edges)
    log.info(f"Extracted {len(edges_df)} disclosure edges "
             f"({edges_df['confidence'].eq('high').sum()} high-confidence)")
    return edges_df


def build_full_graph(
    template_df: pd.DataFrame,
    disclosure_df: pd.DataFrame,
) -> nx.DiGraph:
    """
    Merge template and disclosure edges into a single directed weighted graph.

    Deduplication strategy:
      - If the same (source→target) pair appears in both sources, take the maximum weight.
      - Template edges always take priority for relationship_type and description.

    Node attributes: name, subsector, cik, defaulted (bool).
    Edge attributes: weight, relationship_type, description, confidence, years_active (list).
    """
    G = nx.DiGraph()

    # Add all nodes from universe with metadata
    for ticker, meta in COMPANY_UNIVERSE.items():
        G.add_node(
            ticker,
            name        = meta["name"],
            subsector   = meta["subsector"],
            cik         = meta.get("cik", ""),
            defaulted   = meta.get("defaulted", False),
            default_date= DEFAULT_EVENTS.get(ticker, {}).get("date", None),
        )

    # ── Pass 1: template edges (high confidence skeleton) ─────────────────────
    for _, row in template_df.iterrows():
        src, tgt = row["source"], row["target"]
        if src not in COMPANY_UNIVERSE or tgt not in COMPANY_UNIVERSE:
            continue
        G.add_edge(
            src, tgt,
            weight           = row["weight"],
            relationship_type= row["relationship_type"],
            description      = row["description"],
            confidence       = "high",
            years_active     = list(range(2015, 2025)),
            revenue_pct      = row["weight"],
        )

    # ── Pass 2: disclosure edges (aggregate by source→target) ─────────────────
    if len(disclosure_df):
        agg = (
            disclosure_df
            .groupby(["source", "target"])
            .agg(
                weight            = ("weight",            "max"),
                revenue_pct       = ("revenue_pct",       "mean"),
                relationship_type = ("relationship_type", "first"),
                description       = ("description",       "first"),
                years_active      = ("year",              lambda x: sorted(x.dropna().unique().tolist())),
                confidence        = ("confidence",        "first"),
            )
            .reset_index()
        )
        for _, row in agg.iterrows():
            src, tgt = row["source"], row["target"]
            if G.has_edge(src, tgt):
                # Update weight (take maximum) but keep template metadata
                existing_w = G[src][tgt]["weight"]
                G[src][tgt]["weight"] = max(existing_w, row["weight"])
                # Extend years_active
                existing_years = G[src][tgt].get("years_active", [])
                G[src][tgt]["years_active"] = sorted(
                    set(existing_years) | set(row["years_active"])
                )
            else:
                # New edge discovered from disclosures
                yrs = row["years_active"] if isinstance(row["years_active"], list) else []
                G.add_edge(
                    src, tgt,
                    weight            = row["weight"],
                    relationship_type = row["relationship_type"],
                    description       = str(row["description"]),
                    confidence        = row["confidence"],
                    years_active      = yrs,
                    revenue_pct       = row["revenue_pct"] if pd.notna(row["revenue_pct"]) else row["weight"],
                )

    # ── Final normalisation ────────────────────────────────────────────────────
    # Normalise weights so that out-degree weight sum per node ≤ 1.0
    # This ensures DebtRank propagation is bounded.
    for node in G.nodes():
        out_edges = list(G.out_edges(node, data=True))
        total_w = sum(d["weight"] for _, _, d in out_edges)
        if total_w > 1.0:
            for _, tgt, d in out_edges:
                G[node][tgt]["weight"] /= total_w  # normalise

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_template = sum(1 for _, _, d in G.edges(data=True) if d.get("confidence") == "high")
    log.info(f"Graph built: {n_nodes} nodes, {n_edges} edges "
             f"({n_template} high-confidence, {n_edges - n_template} NLP-extracted)")
    return G


def build_yearly_subgraph(
    G_full: nx.DiGraph,
    year: int,
    template_always_active: bool = True,
) -> nx.DiGraph:
    """
    Extract a subgraph containing only edges that were active in a given year.
    Template edges are always included (they represent stable structural relationships).
    Disclosure edges are only included if `year` appears in their years_active list.
    """
    G_year = nx.DiGraph()
    G_year.add_nodes_from(G_full.nodes(data=True))

    for src, tgt, d in G_full.edges(data=True):
        is_template = d.get("source_file") == "edges_template" or d.get("confidence") == "high"
        years = d.get("years_active", [])
        if template_always_active and is_template:
            G_year.add_edge(src, tgt, **d)
        elif year in years:
            G_year.add_edge(src, tgt, **d)

    return G_year


def save_graph(G: nx.DiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    log.info(f"Graph saved to {path}")


def load_graph(path: Path) -> nx.DiGraph:
    with open(path, "rb") as f:
        G = pickle.load(f)
    log.info(f"Graph loaded from {path} — {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── Main entry point ───────────────────────────────────────────────────────────
def build_and_save() -> nx.DiGraph:
    log.info("=== Module C: Graph Construction ===")

    template_df    = load_template_edges(EDGES_TEMPLATE)
    disclosure_df  = load_and_parse_disclosure_edges(CUSTOMER_DISCLOSURES)
    G              = build_full_graph(template_df, disclosure_df)

    save_graph(G, GRAPH_PICKLE)

    # Print graph summary
    log.info(f"\nGraph Summary:")
    log.info(f"  Nodes: {G.number_of_nodes()}")
    log.info(f"  Edges: {G.number_of_edges()}")
    log.info(f"  Density: {nx.density(G):.4f}")
    log.info(f"  Avg out-degree: {sum(d for _, d in G.out_degree()) / G.number_of_nodes():.2f}")
    log.info(f"  Strongly connected components: {nx.number_strongly_connected_components(G)}")
    log.info(f"  Weakly connected components: {nx.number_weakly_connected_components(G)}")

    return G


if __name__ == "__main__":
    build_and_save()
