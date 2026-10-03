# floorscan - every reported number is regenerable from these targets.
PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
CAPTURE ?=

.PHONY: setup setup-models weights test test-all synthetic benchmark benchmark-real reproduce fixloop run report clean

setup:                       ## clean machine -> working CLI (LiDAR tier needs nothing else)
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

setup-models:                ## learned backends for video/photo tiers (needs network once)
	$(PIP) install -e ".[models]"
	$(PIP) install "mapanything @ git+https://github.com/facebookresearch/map-anything.git"
	$(MAKE) weights

weights:
	$(PY) scripts/fetch_weights.py

test:                        ## fast seam tests
	$(PY) -m pytest -m "not slow"

test-all:                    ## incl. end-to-end CLI tests on rendered captures (~3 min)
	$(PY) -m pytest

run:                         ## make run CAPTURE=path/to/capture
	$(PY) -m scan.cli run $(CAPTURE)

synthetic:                   ## render the synthetic benchmark (deterministic, ~20 min)
	$(PY) scripts/make_synthetic_benchmark.py

benchmark: synthetic         ## run + score every capture, write reports/synthetic/benchmark.md
	$(PY) -m eval.run --manifest data/benchmark_synthetic.yaml --out reports/synthetic --reuse

benchmark-real:              ## real captures (data/benchmark_real.yaml); refits the shipped calibration
	$(PY) -m eval.run --manifest data/benchmark_real.yaml --out reports/real --write-calibration

fixloop:                     ## regenerate the before/after runs of the fix loop from their git tags
	bash fixloop/run_fixloop.sh

reproduce: benchmark fixloop ## everything in the reports from raw inputs
	@if [ -f data/benchmark_real.yaml ]; then $(MAKE) benchmark-real; fi

report:
	$(PY) scripts/build_report.py

clean:
	rm -rf out .cache
