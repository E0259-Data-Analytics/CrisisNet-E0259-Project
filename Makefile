# CrisisNet - Full Pipeline Makefile
# Usage: make all PY=crisis/bin/python

PY ?= crisis/bin/python
STREAMLIT ?= crisis/bin/streamlit

.PHONY: all fuse select train ablate analyze dashboard data clean

all: fuse select train ablate analyze
	@echo "=== CrisisNet pipeline complete ==="

data:
	@echo "=== Preparing required datasets ==="
	bash scripts/download_required_datasets.sh

fuse:
	@echo "=== Step 1: Building X_fused.parquet ==="
	$(PY) Module_D/build_x_fused.py

select:
	@echo "=== Step 2: Feature selection ==="
	$(PY) Module_D/feature_selection.py

train:
	@echo "=== Step 3: Training fusion model ==="
	$(PY) Module_D/train_fusion.py

ablate:
	@echo "=== Step 4: Ablation study ==="
	$(PY) Module_D/ablation_study.py

analyze:
	@echo "=== Step 5: Failure analysis ==="
	$(PY) Module_D/failure_analysis.py

dashboard:
	@echo "=== Launching dashboard ==="
	$(STREAMLIT) run dashboard/app.py

clean:
	rm -f Module_D/X_fused.parquet Module_D/lgbm_fusion.txt
	rm -f Module_D/health_scores.parquet Module_D/test_predictions.parquet
	rm -f Module_D/shap_values.npy Module_D/shap_feat_cols.json
	rm -f Module_D/metrics.json Module_D/ablation_results.json
	rm -f Module_D/selected_features.json Module_D/failure_analysis.json
	rm -f Module_D/*.png
