.PHONY: help install synthetic smoke test ledgers figures figures-all check-figures replicate clean
PY ?= python3
help:
	@echo "install       pip install -r requirements.txt"
	@echo "synthetic     write a synthetic dataset for smoke tests"
	@echo "smoke         run step3 + step5 on synthetic data (minutes, no licensed data needed)"
	@echo "test          pytest (ledger consistency + smoke)"
	@echo "ledgers       rebuild results/*.csv from the transcribed report numbers"
	@echo "figures       rebuild the 7 ledger-based figures (no raw data needed) into figures/"
	@echo "figures-all   also rebuild the 5 data-driven figures (needs licensed data)"
	@echo "check-figures list which of the 12 manuscript figures are present"
	@echo "replicate     full pipeline on the licensed export (RAW=... bash run_all.sh)"
install:
	$(PY) -m pip install -r requirements.txt
synthetic:
	$(PY) tests/make_synthetic_data.py data/synthetic/synthetic_with_cluster.csv 3000
smoke: synthetic
	cd scripts/pipeline && RV_RAW=../../data/synthetic/synthetic_with_cluster.csv RV_OUT=../../outputs/smoke RV_BOOT=19 RV_BOOT_ATT=5 $(PY) -u step5_full_pipeline.py
test:
	$(PY) -m pytest -q tests
ledgers:
	$(PY) scripts/utils/build_results_ledgers.py
figures:
	$(PY) scripts/figures/make_figures.py figures_raw results/robustness/C3_leave_one_institution_out.csv
	bash scripts/figures/collect_figures.sh figures_raw
figures-all: figures
	$(PY) scripts/figures/make_data_figures.py --raw data/raw/Transport_CN_Scholarly_Works.csv --clustered data/processed/Transport_CN_Scholarly_Works_with_cluster.csv --out figures_raw
	bash scripts/figures/collect_figures.sh figures_raw
check-figures:
	$(PY) scripts/utils/check_figures.py
replicate:
	bash run_all.sh
clean:
	rm -rf outputs figures_raw __pycache__ scripts/pipeline/__pycache__ .pytest_cache
