# Makefile to run experiments, generate results, and build the paper
# Usage:
#   make all        # run everything
#   make artifacts  # run experiments and generate results tex
#   make paper      # compile docs/paper.tex with latexrun
#   make clean      # clean LaTeX build outputs

PYTHON := python
LATEXRUN := latexrun

ARTIFACTS := artifacts/summary_metrics.json docs/results_generated.tex

.PHONY: all artifacts experiments results paper clean

all: artifacts paper

artifacts: experiments results

experiments:
	$(PYTHON) scripts/run_wilson_experiments.py

results: experiments
	$(PYTHON) scripts/generate_results_tex.py

paper: $(ARTIFACTS)
	cd docs && $(LATEXRUN) paper.tex

clean:
	cd docs && rm -rf _latexrun_tmp *.aux *.log *.out *.bbl *.blg *.toc *.pdf
