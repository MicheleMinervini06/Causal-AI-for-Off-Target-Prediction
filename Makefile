UV ?= uv
PYTHON := $(UV) run python

.PHONY: install data train eval test

install:
	$(UV) sync --extra dev

data:
	$(PYTHON) -m dag.features --input data/raw --output data/processed/features/features.parquet

train:
	$(UV) run crispr-exp01
	$(UV) run crispr-exp02
	$(UV) run crispr-exp03

eval:
	$(UV) run crispr-benchmark --results-dir experiments/results --output experiments/results/benchmark_metrics.csv

test:
	$(UV) run pytest -q
