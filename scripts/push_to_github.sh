#!/bin/bash
# Push the full codebase to GitHub.
# Run from repo root: bash scripts/push_to_github.sh

set -euo pipefail

echo "=== Preparing GitHub push ==="

if [ ! -f .gitignore ]; then
  cp scripts/templates/gitignore.github .gitignore
  echo "Created .gitignore from scripts/templates/gitignore.github"
fi

git add -A
git commit -m "feat: Full CrisisNet pipeline with FinBERT NLP diagnostics"
git push origin main

echo "=== Done ==="
