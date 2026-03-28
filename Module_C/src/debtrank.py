"""
CrisisNet Module C — DebtRank Contagion Simulation
====================================================
DebtRank is an algorithm originally developed by Battiston et al. (2012)
to model financial system fragility after the 2008 financial crisis.

Analogy: Just as cancer metastasises through the lymphatic network,
a financially distressed firm propagates stress through the supply-chain
network proportional to the revenue dependency (edge weight).

Algorithm:
  1. Mark a seed company as 'defaulted' (stress = 1.0)
  2. Each company receiving stress from a neighbour increases its own
     stress level proportionally to the edge weight
  3. Propagate in rounds until system stabilises
  4. Output: final stress score for every company in the network

Validation:
  - CHK (Chesapeake Energy, June 2020 bankruptcy) is used as the
    primary validation case. Companies most impacted by CHK's stress
    (LNG, ET, WMB) should show the highest post-simulation stress scores.
    This is corroborated by stock price movements in June-August 2020.

Reference:
  Battiston, S., Puliga, M., Kaushik, R., Tasca, P., & Caldarelli, G. (2012).
  DebtRank: Too central to fail? Financial networks, the FED and systemic risk.
  Scientific Reports, 2(1), 541.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    COMPANY_UNIVERSE, DEFAULT_EVENTS,
    DEBTRANK_MAX_ROUNDS, DEBTRANK_THRESHOLD,
    DATA_PROCESSED, DEBTRANK_RESULTS
)


# ── Node states ────────────────────────────────────────────────────────────────
UNDISTRESSED  = "U"  # no stress
DISTRESSED    = "D"  # receiving/propagating stress
INACTIVE      = "I"  # has already propagated its maximum stress (cannot propagate again)


def run_debtrank(
    G: nx.DiGraph,
    seed_companies: List[str],
    initial_stress: Optional[Dict[str, float]] = None,
    max_rounds: int = DEBTRANK_MAX_ROUNDS,
    threshold: float = DEBTRANK_THRESHOLD,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Run the DebtRank contagion simulation from one or more seed companies.

    Parameters
    ----------
    G               : The supply-chain directed graph
    seed_companies  : List of tickers to mark as initial defaulters
    initial_stress  : Optional dict of ticker → initial stress (default: 1.0 for seeds)
    max_rounds      : Maximum propagation rounds
    threshold       : Ignore stress increments below this value

    Returns
    -------
    final_stress : dict of ticker → final stress score [0, 1]
    history_df   : DataFrame of stress evolution per round per node
    """
    # Initialise stress levels
    h = {node: 0.0 for node in G.nodes()}
    state = {node: UNDISTRESSED for node in G.nodes()}

    for seed in seed_companies:
        if seed in G.nodes():
            h[seed] = initial_stress.get(seed, 1.0) if initial_stress else 1.0
            state[seed] = DISTRESSED

    # Track history
    history = []
    for node in G.nodes():
        history.append({
            "round":  0,
            "node":   node,
            "stress": h[node],
            "state":  state[node],
        })

    # Propagate in rounds
    for round_num in range(1, max_rounds + 1):
        delta = {node: 0.0 for node in G.nodes()}
        active_this_round = False

        for node in G.nodes():
            if state[node] != DISTRESSED:
                continue
            # Propagate stress through outgoing edges (supply-chain dependency)
            for _, neighbour, d in G.out_edges(node, data=True):
                w = d.get("weight", 0.1)
                stress_received = h[node] * w

                if stress_received < threshold:
                    continue

                if state[neighbour] == UNDISTRESSED:
                    # First time this neighbour is stressed
                    delta[neighbour] += stress_received
                    active_this_round = True
                elif state[neighbour] == DISTRESSED:
                    # Additional stress from multiple sources
                    new_stress = h[neighbour] + stress_received
                    if new_stress - h[neighbour] >= threshold:
                        delta[neighbour] += stress_received
                        active_this_round = True

        # Apply updates
        state_changed = False
        for node in G.nodes():
            if state[node] == DISTRESSED:
                # After propagating, move to INACTIVE
                state[node] = INACTIVE

        for node in G.nodes():
            if delta[node] > threshold:
                old_h = h[node]
                h[node] = min(1.0, h[node] + delta[node])
                if state[node] == UNDISTRESSED:
                    state[node] = DISTRESSED
                    state_changed = True

        # Record round
        for node in G.nodes():
            history.append({
                "round":  round_num,
                "node":   node,
                "stress": h[node],
                "state":  state[node],
            })

        # Convergence check
        if not active_this_round:
            log.debug(f"  DebtRank converged at round {round_num}")
            break

    # Compute systemic impact: total stress absorbed by the system
    systemic_impact = sum(h.values()) / len(G.nodes())
    n_stressed = sum(1 for v in h.values() if v > threshold)
    log.info(f"  Systemic impact: {systemic_impact:.4f}, stressed nodes: {n_stressed}/{len(G.nodes())}")

    history_df = pd.DataFrame(history)
    return h, history_df


def run_all_scenarios(
    G: nx.DiGraph,
    scenarios: Optional[Dict[str, Dict]] = None,
) -> pd.DataFrame:
    """
    Run DebtRank for multiple seed scenarios and aggregate results.

    Default scenarios include:
      - Each company as a single-seed defaulter
      - CHK (historical 2020 bankruptcy)
      - WLL (historical 2020 bankruptcy)
      - Multi-company stress scenarios (2015-16 oil crash, 2020 COVID)

    Returns
    -------
    results_df : DataFrame with columns [scenario, node, final_stress, systemic_impact]
    """
    if scenarios is None:
        # All single-company scenarios
        scenarios = {}
        for ticker in G.nodes():
            scenarios[f"single_{ticker}"] = {
                "seeds": [ticker],
                "description": f"{ticker} default",
            }
        # Historical event scenarios
        scenarios["historical_chk_2020"] = {
            "seeds": ["CHK"],
            "description": "Chesapeake Energy Chapter 11 (Jun 2020)",
            "initial_stress": {"CHK": 1.0},
        }
        scenarios["oil_crash_2015_wave"] = {
            "seeds": ["CHK", "SWN", "APA"],
            "description": "2015-16 oil crash: multiple E&P distress",
            "initial_stress": {"CHK": 0.7, "SWN": 0.6, "APA": 0.5},
        }
        scenarios["covid_2020_wave"] = {
            "seeds": ["CHK", "OAS", "CHAP"],
            "description": "2020 COVID wave: CHK + Oasis + Chaparral",
            "initial_stress": {"CHK": 1.0, "OAS": 0.8, "CHAP": 0.7},
        }
        scenarios["oilfield_services_shock"] = {
            "seeds": ["SLB", "HAL"],
            "description": "Oilfield services sector shock",
            "initial_stress": {"SLB": 0.8, "HAL": 0.7},
        }
        scenarios["midstream_disruption"] = {
            "seeds": ["KMI", "EPD"],
            "description": "Midstream pipeline disruption",
            "initial_stress": {"KMI": 0.6, "EPD": 0.6},
        }

    all_rows = []
    for scenario_name, cfg in scenarios.items():
        seeds = cfg["seeds"]
        valid_seeds = [s for s in seeds if s in G.nodes()]
        if not valid_seeds:
            continue

        init_stress = cfg.get("initial_stress", None)
        final_stress, history_df = run_debtrank(G, valid_seeds, init_stress)

        systemic_impact = sum(final_stress.values()) / len(G.nodes())
        n_stressed = sum(1 for v in final_stress.values() if v > DEBTRANK_THRESHOLD)

        for node, stress in final_stress.items():
            meta = COMPANY_UNIVERSE.get(node, {})
            all_rows.append({
                "scenario":         scenario_name,
                "description":      cfg["description"],
                "seed_companies":   ", ".join(sorted(valid_seeds)),
                "node":             node,
                "name":             meta.get("name", node),
                "subsector":        meta.get("subsector", "Unknown"),
                "final_stress":     stress,
                "is_seed":          node in valid_seeds,
                "systemic_impact":  systemic_impact,
                "n_stressed_nodes": n_stressed,
                "defaulted":        G.nodes[node].get("defaulted", False),
            })
        log.info(f"  Scenario '{scenario_name}': impact={systemic_impact:.4f}, "
                 f"stressed={n_stressed}/{len(G.nodes())}")

    results_df = pd.DataFrame(all_rows)
    log.info(f"DebtRank: {len(scenarios)} scenarios completed")
    return results_df


def compute_debtrank_exposure_features(
    G: nx.DiGraph,
    years: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Compute per-company DebtRank exposure features for use in X_graph.

    For each company c and year t:
      - debtrank_exposure: average stress received when any neighbour defaults
      - max_contagion_in:  maximum stress received from any single source
      - contagion_out:     total stress c can transmit to the network
      - systemic_risk_contribution: how much system stress increases if c defaults

    These features capture c's position in the contagion network.
    """
    if years is None:
        years = list(range(2015, 2025))

    feature_rows = []

    # Run single-company scenarios for all companies to get exposure profiles
    all_results = run_all_scenarios(G, {
        f"single_{t}": {"seeds": [t], "description": f"{t} stress"}
        for t in G.nodes()
    })

    for ticker in G.nodes():
        meta = COMPANY_UNIVERSE.get(ticker, {})

        # How much stress does this company receive when each neighbour defaults?
        incoming_stress = all_results[
            (all_results["node"] == ticker) & (~all_results["is_seed"])
        ]["final_stress"]

        # How much stress does this company cause to others when it defaults?
        outgoing_scenario = all_results[
            (all_results["scenario"] == f"single_{ticker}") &
            (all_results["node"] != ticker)
        ]["final_stress"]

        for year in years:
            feature_rows.append({
                "ticker":                    ticker,
                "year":                      year,
                "name":                      meta.get("name", ticker),
                "subsector":                 meta.get("subsector", "Unknown"),
                "debtrank_exposure":         float(incoming_stress.mean()) if len(incoming_stress) else 0.0,
                "max_contagion_in":          float(incoming_stress.max()) if len(incoming_stress) else 0.0,
                "contagion_out":             float(outgoing_scenario.sum()) if len(outgoing_scenario) else 0.0,
                "systemic_risk_contribution":float(all_results[
                    all_results["scenario"] == f"single_{ticker}"
                ]["systemic_impact"].mean()) if len(all_results[
                    all_results["scenario"] == f"single_{ticker}"
                ]) else 0.0,
                "n_exposed_neighbours":      int(
                    (outgoing_scenario > DEBTRANK_THRESHOLD).sum()
                ),
            })

    df = pd.DataFrame(feature_rows)
    log.info(f"DebtRank exposure features: {len(df)} rows")
    return df


if __name__ == "__main__":
    from graph_builder import load_graph
    from config import GRAPH_PICKLE

    G = load_graph(GRAPH_PICKLE)

    log.info("Running CHK default scenario (2020 validation)...")
    final_stress, history = run_debtrank(G, ["CHK"])

    # Ranked contagion exposure
    stressed = {k: v for k, v in sorted(final_stress.items(), key=lambda x: -x[1]) if v > 0.01}
    print("\nCHK Default — Contagion Impact (ranked):")
    for ticker, stress in stressed.items():
        name = COMPANY_UNIVERSE.get(ticker, {}).get("name", ticker)
        print(f"  {ticker:6s} ({name:30s}): stress = {stress:.4f}")
