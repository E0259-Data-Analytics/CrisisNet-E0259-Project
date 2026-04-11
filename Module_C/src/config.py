"""
CrisisNet Module C — Configuration
===================================
Central configuration for the Supply-Chain Network Analysis pipeline.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_C_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW       = MODULE_C_ROOT / "data" / "raw"
DATA_PROCESSED = MODULE_C_ROOT / "data" / "processed"
RESULTS        = MODULE_C_ROOT / "results"
FIGURES        = RESULTS / "figures"
TABLES         = RESULTS / "tables"
EXPORTS        = RESULTS / "exports"

for p in [DATA_PROCESSED, RESULTS, FIGURES, TABLES, EXPORTS]:
    p.mkdir(parents=True, exist_ok=True)

# ── Input files ────────────────────────────────────────────────────────────────
EDGES_TEMPLATE        = DATA_RAW / "edges_template.csv"
CUSTOMER_DISCLOSURES  = DATA_RAW / "customer_disclosures_raw.csv"
ENERGY_DEFAULTS       = DATA_RAW / "energy_defaults_curated.csv"
DISTRESS_DRAWDOWNS    = DATA_RAW / "distress_from_drawdowns.csv"

# ── Output files ───────────────────────────────────────────────────────────────
GRAPH_PICKLE          = DATA_PROCESSED / "supply_chain_graph.pkl"
X_GRAPH_PARQUET       = EXPORTS / "X_graph.parquet"
COMMUNITY_HISTORY     = DATA_PROCESSED / "community_history.csv"
DEBTRANK_RESULTS      = DATA_PROCESSED / "debtrank_results.csv"
CENTRALITY_RESULTS    = DATA_PROCESSED / "centrality_results.csv"

# ── Universe of 40 companies ──────────────────────────────────────────────────
COMPANY_UNIVERSE = {
    # Integrated Oil
    "XOM": {"name": "ExxonMobil",            "subsector": "Integrated Oil",    "cik": "34088"},
    "CVX": {"name": "Chevron",               "subsector": "Integrated Oil",    "cik": "93410"},
    "OXY": {"name": "Occidental Petroleum",  "subsector": "Integrated Oil",    "cik": "797468"},
    # E&P
    "COP": {"name": "ConocoPhillips",        "subsector": "E&P",               "cik": "1163165"},
    "EOG": {"name": "EOG Resources",         "subsector": "E&P",               "cik": "821189"},
    "DVN": {"name": "Devon Energy",          "subsector": "E&P",               "cik": "1090012"},
    "FANG":{"name": "Diamondback Energy",    "subsector": "E&P",               "cik": "1539838"},
    "APA": {"name": "APA Corporation",       "subsector": "E&P",               "cik": "6769"},
    "OVV": {"name": "Ovintiv",               "subsector": "E&P",               "cik": "1520006"},
    "CTRA":{"name": "Coterra Energy",        "subsector": "E&P",               "cik": "858470"},
    # Oilfield Services
    "SLB": {"name": "SLB (Schlumberger)",    "subsector": "Oilfield Services", "cik": "87347"},
    "HAL": {"name": "Halliburton",           "subsector": "Oilfield Services", "cik": "45012"},
    "BKR": {"name": "Baker Hughes",          "subsector": "Oilfield Services", "cik": "808362"},
    "FTI": {"name": "TechnipFMC",            "subsector": "Oilfield Services", "cik": "892553"},
    "NOV": {"name": "NOV Inc.",              "subsector": "Oilfield Services", "cik": "1021860"},
    # Refining
    "VLO": {"name": "Valero Energy",         "subsector": "Refining",          "cik": "1035002"},
    "MPC": {"name": "Marathon Petroleum",    "subsector": "Refining",          "cik": "1510295"},
    "PSX": {"name": "Phillips 66",           "subsector": "Refining",          "cik": "1534701"},
    "DK":  {"name": "Delek US Holdings",     "subsector": "Refining",          "cik": "1694426"},
    "PBF": {"name": "PBF Energy",            "subsector": "Refining",          "cik": "1534504"},
    # Midstream / Pipelines
    "KMI": {"name": "Kinder Morgan",         "subsector": "Midstream",         "cik": "1110805"},
    "WMB": {"name": "Williams Companies",    "subsector": "Midstream",         "cik": "107263"},
    "OKE": {"name": "ONEOK",                 "subsector": "Midstream",         "cik": "1040792"},
    "ET":  {"name": "Energy Transfer",       "subsector": "Midstream",         "cik": "1276187"},
    "EPD": {"name": "Enterprise Products",   "subsector": "Midstream",         "cik": "1061219"},
    "TRGP":{"name": "Targa Resources",       "subsector": "Midstream",         "cik": "1389170"},
    "AM":  {"name": "Antero Midstream",      "subsector": "Midstream",         "cik": "1623925"},
    # Natural Gas / LNG
    "EQT": {"name": "EQT Corporation",       "subsector": "Natural Gas",       "cik": "33213"},
    "AR":  {"name": "Antero Resources",      "subsector": "Natural Gas",       "cik": "1492691"},
    "RRC": {"name": "Range Resources",       "subsector": "Natural Gas",       "cik": "315852"},
    "CHK": {"name": "Chesapeake Energy",     "subsector": "Natural Gas",       "cik": "895126", "defaulted": True},
    "SWN": {"name": "Southwestern Energy",   "subsector": "Natural Gas",       "cik": "7332",   "defaulted": True},
    "LNG": {"name": "Cheniere Energy",       "subsector": "LNG",               "cik": "1486159"},
    # Acquired / Merged companies retained as positive distress examples
    "HES": {"name": "Hess Corporation",      "subsector": "E&P",               "cik": "4447"},
    "MRO": {"name": "Marathon Oil",          "subsector": "E&P",               "cik": "101778"},
    "PXD": {"name": "Pioneer Natural Resources","subsector": "E&P",            "cik": "1038357"},
    # C1: 4 missing tickers added to reach full 40-ticker universe
    "CHRD":{"name": "Chord Energy",          "subsector": "E&P",               "cik": "0"},
    "DTM": {"name": "DT Midstream",          "subsector": "Midstream",         "cik": "0"},
    "MTDR":{"name": "Matador Resources",     "subsector": "E&P",               "cik": "1520358"},
    "PR":  {"name": "Permian Resources",     "subsector": "E&P",               "cik": "0"},
}

# Reverse CIK → ticker mapping (for parsing customer_disclosures_raw)
CIK_TO_TICKER = {v["cik"]: k for k, v in COMPANY_UNIVERSE.items()}
# Also add alternative CIKs found in the actual data
CIK_TO_TICKER.update({
    "1021860": "NOV",
    "1035002": "VLO",
    "1061219": "EPD",
    "107263":  "WMB",
    "1090012": "DVN",
    "1163165": "COP",
    "1276187": "ET",
    "1389170": "TRGP",
    "1486159": "LNG",
    "1510295": "MPC",
    "1534701": "PSX",
    "315852":  "RRC",
    "33213":   "EQT",
    "45012":   "HAL",
    "797468":  "OXY",
    "821189":  "EOG",
    "858470":  "CTRA",
    "87347":   "SLB",
    "1433270": "FANG",
    "1506307": "OVV",
    "1520006": "OVV",
    "1539838": "FANG",
    "1623925": "AM",
    "1658566": "AR",
    "1681459": "BKR",
    "1694426": "DK",
    "1701605": "PBF",
    "1792580": "PXD",
    "1841666": "MRO",
    "1842022": "HES",
    "3570":    "DK",
})

# ── Confirmed default events (for contagion validation) ───────────────────────
DEFAULT_EVENTS = {
    "CHK": {"date": "2020-06-28", "type": "Chapter 11"},
    "WLL": {"date": "2020-04-01", "type": "Chapter 11"},
    "SN":  {"date": "2019-08-11", "type": "Chapter 11"},
    "CHAP":{"date": "2020-08-16", "type": "Chapter 11"},
    "DNR": {"date": "2020-07-29", "type": "Chapter 11"},
    "OAS": {"date": "2020-09-30", "type": "Chapter 11"},
    "WFT": {"date": "2019-07-01", "type": "Chapter 11"},
}

# ── Edge relationship type weights (contagion strength multiplier) ─────────────
RELATIONSHIP_WEIGHTS = {
    "shipper":           1.0,   # direct revenue dependency → strongest contagion
    "gas_supplier":      0.95,  # long-term offtake contract
    "pipeline_supplier": 0.85,  # physical delivery chain
    "service_provider":  0.70,  # operational dependency
    "equipment_supplier":0.55,  # second-order dependency
    "customer":          0.80,  # general customer relationship
    "supply_agreement":  0.75,  # formal supply agreement
    "major_customer":    0.85,  # significant revenue concentration
    "unknown":           0.50,  # fallback weight
}

# ── Subsector color palette (colour-blind safe, WCAG 2.1 AA compliant) ────────
SUBSECTOR_COLORS = {
    "Integrated Oil":    "#1A5276",   # deep navy blue
    "E&P":               "#E74C3C",   # vermilion red
    "Oilfield Services": "#28B463",   # forest green
    "Refining":          "#F39C12",   # amber orange
    "Midstream":         "#8E44AD",   # purple
    "Natural Gas":       "#2E86C1",   # ocean blue
    "LNG":               "#16A085",   # teal
}

# ── Analysis parameters ───────────────────────────────────────────────────────
ROLLING_WINDOW_YEARS    = 1       # years for dynamic community tracking
DEBTRANK_MAX_ROUNDS     = 100     # maximum propagation rounds
DEBTRANK_THRESHOLD      = 0.01    # stress below this is ignored
MIN_REVENUE_PCT         = 0.05    # minimum edge weight (5% revenue concentration)
LOUVAIN_RANDOM_STATE    = 42
GN_NUM_COMMUNITIES      = 8       # target communities for Girvan-Newman

ANALYSIS_START_YEAR = 2015
ANALYSIS_END_YEAR   = 2025
QUARTERS = [f"{y}Q{q}" for y in range(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR + 1)
            for q in range(1, 5)]
