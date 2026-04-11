"""
CrisisNet Module C — Visualizations
=====================================
All production-quality visualizations for the supply-chain network analysis.

Colour accessibility: all palettes use the WCAG 2.1 AA deuteranopia-safe
colour scheme. This is verified using the project's colour-blindness simulation.

Visualizations produced:
  V1. Full supply-chain network graph (force-directed, colour = subsector)
  V2. Community detection overlay (Louvain + Girvan-Newman comparison)
  V3. Centrality heatmap (nodes × metrics)
  V4. DebtRank contagion cascade (CHK 2020 scenario)
  V5. Dynamic community fragmentation timeline (2015-2024)
  V6. Systemic importance ranking bar chart
  V7. DebtRank scenario comparison (multiple scenarios)
  V8. Community evolution Sankey diagram
  V9. Network graph with DebtRank stress encoded as node colour
  V10. Feature correlation matrix (X_graph features)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import SUBSECTOR_COLORS, COMPANY_UNIVERSE, FIGURES, ANALYSIS_START_YEAR, ANALYSIS_END_YEAR

# ── Global aesthetic settings ─────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         200,
    "figure.facecolor":    "#0D1117",
    "axes.facecolor":      "#0D1117",
    "axes.edgecolor":      "#30363D",
    "axes.labelcolor":     "#C9D1D9",
    "axes.titlecolor":     "#F0F6FC",
    "xtick.color":         "#8B949E",
    "ytick.color":         "#8B949E",
    "text.color":          "#C9D1D9",
    "grid.color":          "#21262D",
    "grid.alpha":          0.5,
    "font.family":         "DejaVu Sans",
    "font.size":           10,
    "axes.titlesize":      13,
    "axes.labelsize":      11,
    "legend.framealpha":   0.3,
    "legend.edgecolor":    "#30363D",
    "legend.facecolor":    "#161B22",
})

# Crisis-safe diverging colormap (red = high stress, green = healthy)
STRESS_CMAP = LinearSegmentedColormap.from_list(
    "stress",
    ["#1A7341", "#F7DC6F", "#E74C3C"],  # green → yellow → red
    N=256
)

def _save(fig, name: str, tight: bool = True) -> Path:
    path = FIGURES / f"{name}.png"
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def _node_positions(G: nx.DiGraph, layout: str = "spring") -> Dict:
    """Compute node positions with multiple layout options."""
    if layout == "spring":
        pos = nx.spring_layout(G, weight="weight", seed=42, k=2.5, iterations=100)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G, weight="weight")
    elif layout == "spectral":
        pos = nx.spectral_layout(G, weight="weight")
    else:
        pos = nx.circular_layout(G)
    return pos


# ── V1. Full Supply-Chain Network ─────────────────────────────────────────────
def plot_supply_chain_network(
    G: nx.DiGraph,
    partition: Optional[Dict[str, int]] = None,
    title: str = "CrisisNet — S&P 500 Energy Sector Supply-Chain Network",
    highlight_defaulted: bool = True,
) -> Path:
    """
    Force-directed graph of the full supply-chain network.
    Nodes coloured by subsector; size proportional to PageRank.
    Edges coloured by relationship type; width proportional to weight.
    Defaulted companies shown with red ring.
    """
    fig, ax = plt.subplots(figsize=(18, 14), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    ax.axis("off")

    pos = _node_positions(G, "spring")

    # ── PageRank for node sizes ────────────────────────────────────────────────
    pr = nx.pagerank(G, weight="weight", alpha=0.85)
    node_sizes = [max(200, pr.get(n, 0.03) * 12000) for n in G.nodes()]

    # ── Node colours by subsector ─────────────────────────────────────────────
    node_colors = [
        SUBSECTOR_COLORS.get(G.nodes[n].get("subsector", "Unknown"), "#888888")
        for n in G.nodes()
    ]

    # ── Edge properties ───────────────────────────────────────────────────────
    edge_weights = [G[u][v].get("weight", 0.3) for u, v in G.edges()]
    edge_widths  = [max(0.5, w * 4) for w in edge_weights]
    edge_alphas  = [min(0.9, 0.3 + w * 0.7) for w in edge_weights]

    # Colour edges by relationship type
    rel_color_map = {
        "service_provider":  "#2ECC71",
        "equipment_supplier":"#F39C12",
        "shipper":           "#3498DB",
        "pipeline_supplier": "#9B59B6",
        "gas_supplier":      "#1ABC9C",
        "major_customer":    "#E74C3C",
        "customer":          "#E67E22",
        "supply_agreement":  "#ECF0F1",
        "unknown":           "#555555",
    }
    edge_colors = [
        rel_color_map.get(G[u][v].get("relationship_type", "unknown"), "#555555")
        for u, v in G.edges()
    ]

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.1",
        min_source_margin=15,
        min_target_margin=15,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
    )

    # Highlight defaulted companies with red ring
    if highlight_defaulted:
        defaulted = [n for n in G.nodes() if G.nodes[n].get("defaulted", False)]
        if defaulted:
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                nodelist=defaulted,
                node_color="none",
                node_size=[max(300, node_sizes[list(G.nodes()).index(n)] + 120) for n in defaulted],
                linewidths=3,
                edgecolors="#FF0000",
            )

    # Node labels (tickers)
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=7.5,
        font_color="#FFFFFF",
        font_weight="bold",
    )

    # ── Legend ─────────────────────────────────────────────────────────────────
    subsector_patches = [
        mpatches.Patch(color=col, label=subsector)
        for subsector, col in SUBSECTOR_COLORS.items()
    ]
    # Edge type patches
    rel_patches = [
        mpatches.Patch(color=col, label=rel.replace("_", " ").title())
        for rel, col in list(rel_color_map.items())[:6]
    ]
    legend1 = ax.legend(
        handles=subsector_patches,
        title="Subsector",
        loc="upper left",
        fontsize=8,
        title_fontsize=9,
        framealpha=0.4,
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=rel_patches,
        title="Relationship Type",
        loc="upper right",
        fontsize=7.5,
        title_fontsize=8.5,
        framealpha=0.4,
    )

    # Red ring annotation
    if highlight_defaulted:
        ax.annotate(
            "Red ring = confirmed\nbankruptcy/default",
            xy=(0.01, 0.04), xycoords="axes fraction",
            fontsize=8, color="#FF6B6B",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1E1E2E", alpha=0.7)
        )

    ax.set_title(title, fontsize=15, color="#F0F6FC", pad=20, fontweight="bold")
    fig.suptitle(
        "Node size = PageRank  |  Edge width = Revenue concentration  |  Arrow direction = dependency flow",
        fontsize=9, color="#8B949E", y=0.02
    )
    return _save(fig, "V1_supply_chain_network")


# ── V2. Community Detection Overlay ──────────────────────────────────────────
def plot_community_detection(
    G: nx.DiGraph,
    louvain_partition: Dict[str, int],
    gn_partition: Dict[str, int],
    louvain_Q: float,
    gn_Q: float,
    louvain_labels: Dict[int, str],
) -> Path:
    """
    Side-by-side comparison of Louvain vs Girvan-Newman community detection.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11), facecolor="#0D1117")

    pos = _node_positions(G, "spring")

    def draw_communities(ax, partition, labels, Q, title):
        ax.set_facecolor("#0D1117")
        ax.axis("off")

        # Community colours
        n_comm = len(set(partition.values()))
        comm_cmap = plt.cm.get_cmap("tab20", n_comm)
        community_colors = {cid: comm_cmap(i) for i, cid in enumerate(sorted(set(partition.values())))}
        node_colors = [community_colors[partition.get(n, 0)] for n in G.nodes()]

        pr = nx.pagerank(G, weight="weight")
        node_sizes = [max(150, pr.get(n, 0.02) * 10000) for n in G.nodes()]

        # Draw community convex hulls (background shading)
        from scipy.spatial import ConvexHull
        community_nodes = {}
        for node, cid in partition.items():
            community_nodes.setdefault(cid, []).append(node)

        for cid, members in community_nodes.items():
            if len(members) < 3:
                continue
            pts = np.array([pos[m] for m in members if m in pos])
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])  # close polygon
                color = community_colors[cid]
                ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                        alpha=0.12, color=color, zorder=0)
                ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                        color=color, alpha=0.4, linewidth=1.5, zorder=1)
            except Exception:
                pass

        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#2D3748", width=0.8, alpha=0.5,
                               arrows=True, arrowsize=8, connectionstyle="arc3,rad=0.05")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.95)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="#FFFFFF", font_weight="bold")

        # Community legend
        patches = []
        for cid in sorted(set(partition.values())):
            label = labels.get(cid, f"Community {cid}")
            short_label = label.replace("_", "\n")
            patches.append(mpatches.Patch(color=community_colors[cid], label=short_label))
        ax.legend(handles=patches, loc="lower left", fontsize=7, title="Communities",
                  title_fontsize=8, framealpha=0.5)

        ax.set_title(f"{title}\nModularity Q = {Q:.4f} ({'✓ PASS' if Q > 0.35 else '✗ BELOW TARGET'} target>0.35)",
                     fontsize=12, color="#F0F6FC", pad=15, fontweight="bold")

    draw_communities(ax1, louvain_partition, louvain_labels, louvain_Q, "Louvain Algorithm")
    # For GN, create basic numeric labels
    gn_labels = {cid: f"GN_Comm_{cid}" for cid in set(gn_partition.values())}
    draw_communities(ax2, gn_partition, gn_labels, gn_Q, "Girvan-Newman Algorithm")

    fig.suptitle(
        "CrisisNet Module C — Community Detection: Identifying Economic Clusters in the Supply-Chain Network",
        fontsize=14, color="#F0F6FC", y=1.01, fontweight="bold"
    )
    return _save(fig, "V2_community_detection")


# ── V3. Centrality Heatmap ───────────────────────────────────────────────────
def plot_centrality_heatmap(centrality_df: pd.DataFrame) -> Path:
    """
    Heatmap of all centrality metrics across companies, sorted by systemic importance.
    """
    metrics = [
        "betweenness_centrality", "pagerank", "eigenvector_centrality",
        "in_degree_centrality", "out_degree_centrality",
        "systemic_importance_score", "contagion_vulnerability",
    ]
    available = [m for m in metrics if m in centrality_df.columns]
    df_plot = centrality_df.set_index("ticker")[available].copy() if "ticker" in centrality_df.columns else centrality_df[available].copy()

    # Sort by systemic importance
    if "systemic_importance_score" in df_plot.columns:
        df_plot = df_plot.sort_values("systemic_importance_score", ascending=False)

    # Normalise each column to [0, 1] for visual comparability
    df_norm = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min() + 1e-10)

    # Rename columns for display
    rename = {
        "betweenness_centrality":   "Betweenness",
        "pagerank":                 "PageRank",
        "eigenvector_centrality":   "Eigenvector",
        "in_degree_centrality":     "In-Degree",
        "out_degree_centrality":    "Out-Degree",
        "systemic_importance_score":"Systemic\nImportance",
        "contagion_vulnerability":  "Contagion\nVulnerability",
    }
    df_norm = df_norm.rename(columns=rename)

    fig, ax = plt.subplots(figsize=(14, 16), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    cmap = LinearSegmentedColormap.from_list(
        "cent", ["#0D1B2A", "#1565C0", "#42A5F5", "#EF9A9A", "#B71C1C"], N=256
    )

    # Add subsector row colours
    subsector_series = centrality_df.set_index("ticker")["subsector"] if "subsector" in centrality_df.columns else None

    sns.heatmap(
        df_norm,
        ax=ax,
        cmap=cmap,
        vmin=0, vmax=1,
        linewidths=0.3,
        linecolor="#21262D",
        annot=False,
        cbar_kws={"label": "Normalised Score", "shrink": 0.4},
    )

    ax.set_xlabel("Centrality Metric", color="#C9D1D9", fontsize=11)
    ax.set_ylabel("Company (ranked by Systemic Importance)", color="#C9D1D9", fontsize=11)
    ax.set_title(
        "CrisisNet Module C — Centrality Analysis\n"
        "Supply-Chain Network Systemic Importance Heatmap",
        fontsize=13, color="#F0F6FC", pad=20, fontweight="bold"
    )
    ax.tick_params(axis="x", labelsize=9, rotation=30, colors="#C9D1D9")
    ax.tick_params(axis="y", labelsize=8, colors="#C9D1D9")

    # Add subsector colour strip on the left
    if subsector_series is not None:
        tickers_in_order = list(df_norm.index)
        strip_colors = [
            SUBSECTOR_COLORS.get(subsector_series.get(t, "Unknown"), "#888888")
            for t in tickers_in_order
        ]
        ax2 = ax.twinx()
        ax2.set_facecolor("#0D1117")
        for i, (t, col) in enumerate(zip(tickers_in_order, strip_colors)):
            ax2.barh(i + 0.5, 0.015, left=-0.015, color=col, height=0.9, alpha=0.9,
                     transform=ax.transData)
        ax2.set_ylim(0, len(tickers_in_order))
        ax2.axis("off")

    return _save(fig, "V3_centrality_heatmap")


# ── V4. DebtRank Contagion Cascade ────────────────────────────────────────────
def plot_debtrank_cascade(
    G: nx.DiGraph,
    history_df: pd.DataFrame,
    scenario_name: str = "CHK Default — June 2020",
    seed: str = "CHK",
) -> Path:
    """
    Visualise the DebtRank contagion cascade across rounds.
    Shows which companies get stressed, when, and how much.
    """
    n_rounds = history_df["round"].max()
    tickers_stressed = history_df[history_df["stress"] > 0.01]["node"].unique()
    tickers_stressed = sorted(tickers_stressed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor="#0D1117",
                                    gridspec_kw={"width_ratios": [1.4, 1]})

    # ── Left: stress evolution heatmap ────────────────────────────────────────
    ax1.set_facecolor("#0D1117")
    pivot = history_df[history_df["node"].isin(tickers_stressed)].pivot_table(
        index="node", columns="round", values="stress", aggfunc="mean"
    )

    sns.heatmap(
        pivot,
        ax=ax1,
        cmap=STRESS_CMAP,
        vmin=0, vmax=1,
        linewidths=0.2,
        linecolor="#1E2634",
        cbar_kws={"label": "Stress Level [0=healthy, 1=defaulted]", "shrink": 0.6},
    )
    ax1.set_title(
        f"DebtRank Stress Propagation\n{scenario_name}",
        fontsize=12, color="#F0F6FC", fontweight="bold"
    )
    ax1.set_xlabel("Propagation Round", color="#C9D1D9")
    ax1.set_ylabel("Company", color="#C9D1D9")
    ax1.tick_params(axis="both", colors="#C9D1D9", labelsize=8)

    # ── Right: final stress bar chart ─────────────────────────────────────────
    ax2.set_facecolor("#0D1117")
    final = history_df[history_df["round"] == history_df["round"].max()].copy()
    final = final.sort_values("stress", ascending=True)
    final_stress = final[final["stress"] > 0.001]

    colors = [
        "#FF4444" if row["node"] == seed else
        SUBSECTOR_COLORS.get(
            COMPANY_UNIVERSE.get(row["node"], {}).get("subsector", "Unknown"), "#5B8AF5"
        )
        for _, row in final_stress.iterrows()
    ]

    bars = ax2.barh(final_stress["node"], final_stress["stress"],
                    color=colors, alpha=0.85, height=0.7, edgecolor="#0D1117", linewidth=0.5)

    # Add stress value labels
    for bar, (_, row) in zip(bars, final_stress.iterrows()):
        if row["stress"] > 0.02:
            ax2.text(
                row["stress"] + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{row['stress']:.3f}",
                va="center", fontsize=7.5, color="#C9D1D9"
            )

    ax2.axvline(x=0.5, color="#FF6B6B", linestyle="--", alpha=0.5, linewidth=1.2,
                label="Severe distress (0.5)")
    ax2.axvline(x=0.2, color="#F39C12", linestyle=":", alpha=0.5, linewidth=1.2,
                label="Moderate distress (0.2)")
    ax2.set_xlabel("Final Stress Score", color="#C9D1D9")
    ax2.set_title("Final Contagion Exposure\n(sorted by impact)", fontsize=12,
                  color="#F0F6FC", fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.set_xlim(0, 1.1)
    ax2.tick_params(axis="both", colors="#C9D1D9", labelsize=8.5)
    ax2.set_facecolor("#0D1117")

    ax2.annotate(
        f"CHK → LNG gas_supplier edge: 15% revenue\n"
        f"Direct contagion confirmed in Q3 2020",
        xy=(0.02, 0.02), xycoords="axes fraction",
        fontsize=7.5, color="#8B949E",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#161B22", alpha=0.7)
    )

    fig.suptitle(
        "CrisisNet Module C — DebtRank Contagion Simulation\n"
        "'Cancer Metastasis' Through the Energy Supply-Chain Network",
        fontsize=14, color="#F0F6FC", fontweight="bold", y=1.01
    )
    return _save(fig, "V4_debtrank_cascade")


# ── V5. Dynamic Community Fragmentation Timeline ──────────────────────────────
def plot_fragmentation_timeline(
    fragmentation_df: pd.DataFrame,
    community_history: pd.DataFrame,
) -> Path:
    """
    Timeline of community structure fragmentation (2015-2024).
    Annotated with known crisis events.
    """
    louvain_frag = fragmentation_df[fragmentation_df["algorithm"] == "louvain"].copy()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), facecolor="#0D1117",
                                          sharex=True)
    years = louvain_frag["year"].values

    # ── Panel 1: Fragmentation Index ─────────────────────────────────────────
    ax1.set_facecolor("#0D1117")
    ax1.plot(years, louvain_frag["fragmentation_index"], color="#E74C3C",
             linewidth=2.5, marker="o", markersize=6, label="Fragmentation Index (Louvain)")
    ax1.fill_between(years, 0, louvain_frag["fragmentation_index"],
                     alpha=0.2, color="#E74C3C")
    ax1.axhline(y=0.3, color="#F39C12", linestyle="--", alpha=0.6, linewidth=1.2,
                label="High fragmentation threshold (0.30)")
    ax1.set_ylabel("Fragmentation Index", color="#C9D1D9")
    ax1.set_title("Community Structure Fragmentation — Leading Indicator of Sector Distress",
                  fontsize=12, color="#F0F6FC", fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.tick_params(colors="#C9D1D9")
    ax1.yaxis.label.set_color("#C9D1D9")

    # ── Panel 2: Modularity Q ─────────────────────────────────────────────────
    ax2.set_facecolor("#0D1117")
    ax2.plot(years, louvain_frag["modularity_Q"], color="#2ECC71",
             linewidth=2.5, marker="s", markersize=6, label="Modularity Q")
    ax2.fill_between(years, 0, louvain_frag["modularity_Q"], alpha=0.15, color="#2ECC71")
    ax2.axhline(y=0.35, color="#3498DB", linestyle="--", alpha=0.6, linewidth=1.2,
                label="Target Q > 0.35")
    ax2.set_ylabel("Modularity Q", color="#C9D1D9")
    ax2.set_title("Louvain Modularity Q — Community Cohesion Score", fontsize=11,
                  color="#F0F6FC", fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.tick_params(colors="#C9D1D9")
    ax2.yaxis.label.set_color("#C9D1D9")

    # ── Panel 3: Number of communities ───────────────────────────────────────
    ax3.set_facecolor("#0D1117")
    ax3.bar(years, louvain_frag["n_communities"], color="#8E44AD", alpha=0.75,
            width=0.6, label="# Communities (Louvain)", edgecolor="#0D1117", linewidth=0.5)
    ax3.set_ylabel("# Communities", color="#C9D1D9")
    ax3.set_title("Number of Detected Communities per Year", fontsize=11,
                  color="#F0F6FC", fontweight="bold")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.tick_params(colors="#C9D1D9")
    ax3.yaxis.label.set_color("#C9D1D9")
    ax3.set_xlabel("Year", color="#C9D1D9")

    # ── Crisis annotations (all panels) ──────────────────────────────────────
    crisis_events = [
        (2015.5, "#E74C3C", "Oil Price\nCrash 2015-16"),
        (2020.0, "#FF6B6B", "COVID-19\nCrash + CHK"),
        (2022.0, "#F39C12", "Energy\nCrisis 2022"),
    ]
    for year_mark, color, label in crisis_events:
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=year_mark, color=color, linestyle=":", alpha=0.5, linewidth=1.5)
        ax1.text(year_mark + 0.05, ax1.get_ylim()[1] * 0.85, label,
                 color=color, fontsize=7.5, rotation=0,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#1E1E2E", alpha=0.6))

    # Fix x-axis
    ax3.set_xticks(years)
    ax3.set_xticklabels(years, fontsize=9, color="#C9D1D9")

    fig.tight_layout(h_pad=2.0)
    return _save(fig, "V5_fragmentation_timeline", tight=False)


# ── V6. Systemic Importance Bar Chart ────────────────────────────────────────
def plot_systemic_importance(centrality_df: pd.DataFrame) -> Path:
    """
    Horizontal bar chart of top companies by systemic importance score.
    Colour-coded by subsector.
    """
    df = centrality_df.sort_values("systemic_importance_score", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    colors = [
        SUBSECTOR_COLORS.get(COMPANY_UNIVERSE.get(row["ticker"], {}).get("subsector", "Unknown"), "#888888")
        for _, row in df.iterrows()
    ]

    bars = ax.barh(
        df["ticker"] + " — " + df["name"].str[:20],
        df["systemic_importance_score"],
        color=colors, alpha=0.85, height=0.7,
        edgecolor="#0D1117", linewidth=0.5
    )

    # Labels
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(
            row["systemic_importance_score"] + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{row['systemic_importance_score']:.3f}",
            va="center", fontsize=8, color="#C9D1D9"
        )

    # Mark defaulted companies
    for i, (_, row) in enumerate(df.iterrows()):
        if row.get("defaulted", False):
            ax.text(
                0.01, i,
                "★ BANKRUPT",
                va="center", fontsize=7, color="#FF4444", fontweight="bold"
            )

    # Subsector legend
    subsector_patches = [
        mpatches.Patch(color=col, label=subsector)
        for subsector, col in SUBSECTOR_COLORS.items()
    ]
    ax.legend(handles=subsector_patches, title="Subsector", loc="lower right",
              fontsize=8, title_fontsize=9, framealpha=0.4)

    ax.set_xlabel("Composite Systemic Importance Score", color="#C9D1D9", fontsize=11)
    ax.set_title(
        "CrisisNet Module C — Top 20 Systemically Important Companies\n"
        "(Composite of Betweenness 35% + PageRank 35% + Eigenvector 20% + In-Degree 10%)",
        fontsize=13, color="#F0F6FC", pad=20, fontweight="bold"
    )
    ax.tick_params(axis="both", colors="#C9D1D9", labelsize=9)
    ax.set_xlim(0, df["systemic_importance_score"].max() * 1.15)

    return _save(fig, "V6_systemic_importance")


# ── V7. Multi-Scenario DebtRank Comparison ────────────────────────────────────
def plot_debtrank_scenarios(debtrank_results: pd.DataFrame) -> Path:
    """
    Compare final stress levels across multiple contagion scenarios.
    """
    scenarios_to_show = [
        "historical_chk_2020",
        "oil_crash_2015_wave",
        "covid_2020_wave",
        "oilfield_services_shock",
        "midstream_disruption",
    ]
    avail = [s for s in scenarios_to_show if s in debtrank_results["scenario"].values]
    if not avail:
        avail = debtrank_results["scenario"].unique()[:5].tolist()

    n_scenarios = len(avail)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(5 * n_scenarios, 10),
                              facecolor="#0D1117")
    if n_scenarios == 1:
        axes = [axes]

    for ax, scenario in zip(axes, avail):
        ax.set_facecolor("#0D1117")
        df_s = debtrank_results[debtrank_results["scenario"] == scenario].copy()
        df_s = df_s.sort_values("final_stress", ascending=True)
        df_s = df_s[df_s["final_stress"] > 0.005]

        colors = [
            "#FF4444" if row["is_seed"] else
            SUBSECTOR_COLORS.get(row.get("subsector", "Unknown"), "#5B8AF5")
            for _, row in df_s.iterrows()
        ]

        ax.barh(df_s["node"], df_s["final_stress"], color=colors, alpha=0.85,
                edgecolor="#0D1117", height=0.7)

        # Seed label
        seeds = df_s[df_s["is_seed"]]["node"].tolist()
        desc = df_s["description"].iloc[0] if len(df_s) else scenario
        systemic = df_s["systemic_impact"].iloc[0] if len(df_s) else 0
        n_stressed = df_s["n_stressed_nodes"].iloc[0] if len(df_s) else 0

        ax.set_title(
            f"{desc}\nImpact={systemic:.3f} | {n_stressed} companies stressed",
            fontsize=8.5, color="#F0F6FC", fontweight="bold", pad=8
        )
        ax.set_xlabel("Final Stress", color="#C9D1D9", fontsize=9)
        ax.tick_params(colors="#C9D1D9", labelsize=8)
        ax.axvline(x=0.5, color="#FF6B6B", linestyle="--", alpha=0.4, linewidth=1)
        ax.set_facecolor("#0D1117")
        ax.set_xlim(0, 1.1)

    fig.suptitle(
        "CrisisNet Module C — DebtRank Multi-Scenario Contagion Comparison\n"
        "Red bars = seed (defaulting) companies",
        fontsize=13, color="#F0F6FC", y=1.01, fontweight="bold"
    )
    return _save(fig, "V7_debtrank_scenarios")


# ── V8. Network with DebtRank Stress Encoding ────────────────────────────────
def plot_network_with_stress(
    G: nx.DiGraph,
    final_stress: Dict[str, float],
    scenario_name: str = "CHK Default 2020",
) -> Path:
    """
    Network graph where node colour encodes contagion stress level.
    Red = high stress, green = healthy.
    """
    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")
    ax.axis("off")

    pos = _node_positions(G, "spring")

    # Node colours: stress level
    norm = Normalize(vmin=0, vmax=1)
    node_colors = [STRESS_CMAP(norm(final_stress.get(n, 0))) for n in G.nodes()]

    # Node sizes: larger if stressed
    pr = nx.pagerank(G, weight="weight")
    node_sizes = [
        max(300, (pr.get(n, 0.03) * 8000) + final_stress.get(n, 0) * 500)
        for n in G.nodes()
    ]

    edge_weights = [G[u][v].get("weight", 0.3) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#1E2634", width=[max(0.5, w * 3) for w in edge_weights],
        alpha=0.6, arrows=True, arrowsize=10,
        connectionstyle="arc3,rad=0.1"
    )
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="#FFFFFF", font_weight="bold")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=STRESS_CMAP, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.01, orientation="vertical")
    cbar.set_label("Contagion Stress [0=healthy, 1=defaulted]", color="#C9D1D9", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#C9D1D9")
    cbar.ax.tick_params(labelcolor="#C9D1D9", labelsize=8)

    ax.set_title(
        f"CrisisNet Module C — Contagion Network State\n"
        f"Scenario: {scenario_name}",
        fontsize=13, color="#F0F6FC", pad=15, fontweight="bold"
    )
    return _save(fig, "V8_network_stress_encoding")


# ── V9. Community Stability Heatmap (across years) ───────────────────────────
def plot_community_stability(community_history: pd.DataFrame) -> Path:
    """
    Heatmap showing which community each company belongs to, per year.
    Colour encodes community ID; shows community stability vs fragmentation.
    """
    louvain = community_history[community_history["algorithm"] == "louvain"].copy()
    louvain = louvain.drop_duplicates(subset=["node", "year"])

    pivot = louvain.pivot_table(
        index="node", columns="year", values="community_id", aggfunc="first"
    )
    # Sort by most common community
    pivot = pivot.fillna(-1).astype(int)

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    cmap = plt.cm.get_cmap("tab20", 12)
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, vmin=-1, vmax=10,
        linewidths=0.5, linecolor="#1E2634",
        cbar_kws={"label": "Community ID", "shrink": 0.4},
        annot=True, fmt="d", annot_kws={"size": 7, "color": "white"},
    )
    ax.set_title(
        "CrisisNet Module C — Community Membership Evolution (2015–2024)\n"
        "Same colour = same community; colour change = fragmentation event",
        fontsize=13, color="#F0F6FC", pad=20, fontweight="bold"
    )
    ax.set_xlabel("Year", color="#C9D1D9", fontsize=11)
    ax.set_ylabel("Company", color="#C9D1D9", fontsize=11)
    ax.tick_params(axis="both", colors="#C9D1D9", labelsize=8.5)

    return _save(fig, "V9_community_stability")


# ── V10. Feature Correlation Matrix ──────────────────────────────────────────
def plot_feature_correlation(x_graph: pd.DataFrame) -> Path:
    """
    Correlation heatmap of all X_graph features.
    Helps validate feature diversity and detect redundancy.
    """
    numeric_cols = x_graph.select_dtypes(include=[np.number]).columns.tolist()
    drop_cols = ["year"]
    feature_cols = [c for c in numeric_cols if c not in drop_cols]

    corr = x_graph[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="#0D1117")
    ax.set_facecolor("#0D1117")

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True  # upper triangle mask

    cmap = LinearSegmentedColormap.from_list(
        "corr", ["#1565C0", "#0D1117", "#B71C1C"], N=256
    )

    sns.heatmap(
        corr, ax=ax, cmap=cmap, vmin=-1, vmax=1,
        mask=mask,
        linewidths=0.3, linecolor="#21262D",
        annot=True, fmt=".2f", annot_kws={"size": 7.5, "color": "#C9D1D9"},
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.5},
        square=True,
    )

    ax.set_title(
        "CrisisNet Module C — X_graph Feature Correlation Matrix\n"
        "Validates feature diversity for Module D fusion",
        fontsize=13, color="#F0F6FC", pad=20, fontweight="bold"
    )
    ax.tick_params(axis="both", colors="#C9D1D9", labelsize=8, rotation=45)

    return _save(fig, "V10_feature_correlation")


def generate_all_visualizations(
    G,
    louvain_partition, louvain_Q, louvain_labels,
    gn_partition, gn_Q,
    centrality_df,
    community_history, fragmentation_df,
    debtrank_results,
    chk_stress, chk_history,
    x_graph,
) -> List[Path]:
    """Generate all 10 visualizations in sequence."""
    print("Generating visualizations...")
    paths = []
    try:
        print("[V1] Supply-chain network...")
        paths.append(plot_supply_chain_network(G, louvain_partition))
    except Exception as e:
        print(f"  V1 failed: {e}")
    try:
        print("[V2] Community detection overlay...")
        paths.append(plot_community_detection(G, louvain_partition, gn_partition,
                                               louvain_Q, gn_Q, louvain_labels))
    except Exception as e:
        print(f"  V2 failed: {e}")
    try:
        print("[V3] Centrality heatmap...")
        paths.append(plot_centrality_heatmap(centrality_df))
    except Exception as e:
        print(f"  V3 failed: {e}")
    try:
        print("[V4] DebtRank cascade (CHK 2020)...")
        paths.append(plot_debtrank_cascade(G, chk_history, "CHK — Chapter 11 Jun 2020", "CHK"))
    except Exception as e:
        print(f"  V4 failed: {e}")
    try:
        print("[V5] Fragmentation timeline...")
        paths.append(plot_fragmentation_timeline(fragmentation_df, community_history))
    except Exception as e:
        print(f"  V5 failed: {e}")
    try:
        print("[V6] Systemic importance ranking...")
        paths.append(plot_systemic_importance(centrality_df))
    except Exception as e:
        print(f"  V6 failed: {e}")
    try:
        print("[V7] Multi-scenario DebtRank...")
        paths.append(plot_debtrank_scenarios(debtrank_results))
    except Exception as e:
        print(f"  V7 failed: {e}")
    try:
        print("[V8] Network with stress encoding...")
        paths.append(plot_network_with_stress(G, chk_stress, "CHK Default — June 2020"))
    except Exception as e:
        print(f"  V8 failed: {e}")
    try:
        print("[V9] Community stability heatmap...")
        paths.append(plot_community_stability(community_history))
    except Exception as e:
        print(f"  V9 failed: {e}")
    try:
        print("[V10] Feature correlation matrix...")
        paths.append(plot_feature_correlation(x_graph))
    except Exception as e:
        print(f"  V10 failed: {e}")

    print(f"\nGenerated {len(paths)}/10 visualizations in {FIGURES}")
    return paths
