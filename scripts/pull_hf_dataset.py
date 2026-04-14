#!/usr/bin/env python3
"""
Pull CrisisNet dataset from HuggingFace and set up the crisisnet-data/ directory.

Usage:
    # Pull only (no push):
    python scripts/pull_hf_dataset.py

    # Pull, augment, and push back (requires HF write token):
    HF_TOKEN=hf_xxx python scripts/pull_hf_dataset.py --push

What this does:
    1. Downloads all files from Sashank-810/crisisnet-dataset → crisisnet-data/
    2. Creates the Module_A symlink (crisisnet-data/Module_A → Module_1)
    3. Generates the missing label_unified.parquet from the two label CSVs
    4. Optionally pushes the augmented dataset back to HuggingFace

Missing data confirmed from HF dataset inspection:
    ✗ crisisnet-data/data/label_unified.parquet   (needed by build_x_fused.py)
    ✓ Module_1/market_data/financials/*.csv        (needed by module1_pipeline.py)
    ✓ Module_1/sec_xbrl/company_facts/*_facts.json (Altman XBRL fallback)
    ✓ Labels/energy_defaults_curated.csv           (hard default labels)
    ✓ Labels/distress_from_drawdowns.csv           (soft distress labels)
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "crisisnet-data"
HF_REPO_ID = "Sashank-810/crisisnet-dataset"


def _hf_available() -> bool:
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
        return True
    except ImportError:
        return False


def install_hf_hub():
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "huggingface_hub", "-q", "--break-system-packages"])


# ── 1. Pull all dataset files from HuggingFace ────────────────────────────────
def pull_dataset(token: str | None = None):
    """Download every file in the HuggingFace repo to crisisnet-data/."""
    if not _hf_available():
        print("huggingface_hub not installed — installing…")
        install_hf_hub()

    from huggingface_hub import list_repo_files, hf_hub_download

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[1/4] Listing files in {HF_REPO_ID}…")

    try:
        files = list(list_repo_files(HF_REPO_ID, repo_type="dataset", token=token))
    except Exception as exc:
        print(f"  ERROR listing repo files: {exc}")
        print("  Check network connectivity and HF token.")
        sys.exit(1)

    print(f"  Found {len(files)} files.")
    success = 0
    failed = []

    for rel_path in files:
        dest = DATA_ROOT / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            print(f"  ✓ (cached) {rel_path}")
            success += 1
            continue

        try:
            local = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=rel_path,
                token=token,
                local_dir=str(DATA_ROOT),
            )
            print(f"  ↓ {rel_path}")
            success += 1
            time.sleep(0.05)
        except Exception as exc:
            print(f"  ✗ FAILED {rel_path}: {exc}")
            failed.append(rel_path)

    print(f"\n  Downloaded {success}/{len(files)} files.")
    if failed:
        print(f"  {len(failed)} failed: {failed[:5]}{'…' if len(failed)>5 else ''}")

    return success, failed


# ── 2. Create Module_A compatibility symlink ──────────────────────────────────
def create_symlink():
    """crisisnet-data/Module_A → Module_1 (the codebase was renamed)."""
    module_a = DATA_ROOT / "Module_A"
    module_1 = DATA_ROOT / "Module_1"

    if module_a.exists() or module_a.is_symlink():
        print("[2/4] Module_A symlink already exists — skipping")
        return

    if not module_1.exists():
        print("[2/4] WARNING: Module_1 directory not found; cannot create symlink")
        return

    try:
        module_a.symlink_to("Module_1")
        print(f"[2/4] Created symlink: crisisnet-data/Module_A → Module_1")
    except OSError as exc:
        print(f"[2/4] Could not create symlink ({exc}); copying instead…")
        import shutil
        shutil.copytree(str(module_1), str(module_a))
        print(f"[2/4] Copied Module_1 → Module_A")


# ── 3. Generate label_unified.parquet (MISSING from HF dataset) ───────────────
def generate_label_unified():
    """
    Merge energy_defaults_curated.csv + distress_from_drawdowns.csv into a
    single (ticker, quarter) label table used by build_x_fused.py.

    Logic:
        hard_default  (Chapter 11, etc.)          → distress_label = 1
        distress_drawdown (>50% drawdown in 6m)   → distress_label = 1
        Everything else                            → distress_label = 0

    The label window extends 4 quarters BEFORE each event (early-warning).
    This matches the labelling convention used when the existing
    X_fused.parquet was built.
    """
    out_path = DATA_ROOT / "data" / "label_unified.parquet"
    if out_path.exists():
        print("[3/4] label_unified.parquet already exists — skipping generation")
        return

    # Source files
    defaults_path = DATA_ROOT / "Labels" / "energy_defaults_curated.csv"
    drawdowns_path = DATA_ROOT / "Labels" / "distress_from_drawdowns.csv"
    company_path = DATA_ROOT / "data" / "company_list.csv"

    if not defaults_path.exists() or not drawdowns_path.exists():
        print("[3/4] WARNING: Label CSVs not found — cannot generate label_unified.parquet")
        return

    print("[3/4] Generating label_unified.parquet…")

    companies = pd.read_csv(company_path)
    tickers = companies["ticker"].tolist()

    # Build complete (ticker × quarter) grid 2015Q1–2025Q4
    quarters = []
    for year in range(2015, 2026):
        for q in range(1, 5):
            quarters.append(f"{year}Q{q}")

    grid = pd.MultiIndex.from_product([tickers, quarters], names=["ticker", "quarter"])
    df_label = pd.DataFrame({"distress_label": 0}, index=grid).reset_index()

    def _date_to_quarter(date_str: str) -> str:
        try:
            d = pd.to_datetime(date_str, errors="coerce")
            if pd.isna(d):
                return None
            return f"{d.year}Q{(d.month - 1)//3 + 1}"
        except Exception:
            return None

    def _mark_distress(ticker: str, event_date_str: str, lead_quarters: int = 4):
        """Mark the event quarter and the N quarters prior as distress."""
        eq = _date_to_quarter(event_date_str)
        if eq is None:
            return
        # Build a sorted list of all quarters
        year, q = int(eq[:4]), int(eq[-1])
        for offset in range(lead_quarters + 1):
            q2 = q - offset
            y2 = year
            while q2 < 1:
                q2 += 4
                y2 -= 1
            qkey = f"{y2}Q{q2}"
            mask = (df_label["ticker"] == ticker) & (df_label["quarter"] == qkey)
            df_label.loc[mask, "distress_label"] = 1

    # Hard defaults
    defaults = pd.read_csv(defaults_path)
    for _, row in defaults.iterrows():
        t = str(row.get("ticker", "")).strip().upper()
        if t not in tickers:
            continue
        evt_date = str(row.get("event_date", row.get("date", "")))
        _mark_distress(t, evt_date, lead_quarters=4)

    # Soft drawdown labels
    drawdowns = pd.read_csv(drawdowns_path)
    # Column names may vary — normalise
    col_map = {}
    for c in drawdowns.columns:
        cl = c.lower().strip()
        if "ticker" in cl:
            col_map["ticker"] = c
        elif "date" in cl or "start" in cl:
            col_map["date"] = c
    if "ticker" in col_map and "date" in col_map:
        for _, row in drawdowns.iterrows():
            t = str(row[col_map["ticker"]]).strip().upper()
            if t not in tickers:
                continue
            _mark_distress(t, str(row[col_map["date"]]), lead_quarters=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_label.to_parquet(out_path, index=False)

    pos = int(df_label["distress_label"].sum())
    total = len(df_label)
    print(f"  ✓ label_unified.parquet: {total} rows, {pos} distress ({pos/total:.1%} prevalence)")
    print(f"  ✓ Saved → {out_path}")


# ── 4. (Optional) Push augmented dataset back to HuggingFace ─────────────────
def push_to_hf(token: str):
    """Upload the generated label_unified.parquet to the HF dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    label_path = DATA_ROOT / "data" / "label_unified.parquet"

    if not label_path.exists():
        print("[4/4] label_unified.parquet not found — skipping push")
        return

    print("[4/4] Pushing label_unified.parquet to HuggingFace…")
    try:
        api.upload_file(
            path_or_fileobj=str(label_path),
            path_in_repo="data/label_unified.parquet",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=token,
            commit_message="Add generated label_unified.parquet (merged defaults + drawdowns)",
        )
        print(f"  ✓ Pushed data/label_unified.parquet to {HF_REPO_ID}")
    except Exception as exc:
        print(f"  ✗ Push failed: {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pull and augment CrisisNet HF dataset")
    parser.add_argument("--push", action="store_true",
                        help="Push augmented files back to HuggingFace (requires HF_TOKEN)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip HF download (only generate/push augmented files)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CrisisNet HuggingFace Dataset Setup")
    print("=" * 60)

    if not args.skip_download:
        pull_dataset(token=args.token)
    else:
        print("[1/4] Skipping download (--skip-download)")

    create_symlink()
    generate_label_unified()

    if args.push:
        if not args.token:
            print("[4/4] ERROR: --push requires a HuggingFace token "
                  "(--token or HF_TOKEN env var)")
            sys.exit(1)
        push_to_hf(token=args.token)
    else:
        print("[4/4] Skipping push (pass --push to upload augmented files)")

    print("\n" + "=" * 60)
    print("  Setup complete.")
    print("  Next steps:")
    print("    1. cd Module_A/notebooks && python module1_pipeline.py")
    print("    2. python Module_C/src/graph_pipeline.py   (if needed)")
    print("    3. python Module_D/build_x_fused.py")
    print("    4. python Module_D/train_fusion.py")
    print("=" * 60)


if __name__ == "__main__":
    main()