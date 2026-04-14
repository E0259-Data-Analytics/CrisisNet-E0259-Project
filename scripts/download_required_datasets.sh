#!/bin/bash
# Prepare or download datasets needed for the CrisisNet pipeline.
# Run from repo root: bash scripts/download_required_datasets.sh

set -euo pipefail

echo "=== Preparing CrisisNet data paths ==="

if [ ! -d "crisisnet-data" ]; then
  echo "ERROR: crisisnet-data/ is missing. Clone or restore the dataset directory first."
  exit 1
fi

# The codebase was renamed Module_1 -> Module_A, while the checked-in dataset
# may still use the original Module_1 directory. Keep a compatibility symlink
# so Module A can use the original financial statements and Altman inputs.
if [ ! -e "crisisnet-data/Module_A" ] && [ -d "crisisnet-data/Module_1" ]; then
  ln -s Module_1 crisisnet-data/Module_A
  echo "Linked crisisnet-data/Module_A -> Module_1"
fi

if [ ! -d "crisisnet-data/Module_A/sec_xbrl/company_facts" ]; then
  echo "SEC XBRL company facts missing; downloading from EDGAR..."
  crisis/bin/python crisisnet-data/scripts/03_download_sec_xbrl.py
else
  echo "SEC XBRL company facts present"
fi

if [ ! -f "Module_B/results/X_nlp_finbert.parquet" ]; then
  echo "WARNING: Module_B/results/X_nlp_finbert.parquet is missing."
  echo "Run the Module B FinBERT feature builder before fusion."
fi

echo "=== Data preparation complete ==="
