# ============================================================
# DAS Gradiometry (I-FDG) Pipeline
# Author: Benz Poobua
# ============================================================

# -----------------------
# Config files
# -----------------------
GRAD_CFG    ?= configs/urban_grad.yaml
EVAL_OUT    ?= data/benchmarks/synth

# -----------------------
# Python executable
# -----------------------
VENV        ?= das_grad
PYTHON      ?= $(VENV)/bin/python

# -----------------------
# Default target
# -----------------------
.DEFAULT_GOAL := help

# -----------------------
# Phony targets
# -----------------------
.PHONY: help grad eval test paths

# ============================================================
# HELP
# ============================================================
help:
	@echo ""
	@echo "DAS Gradiometry Pipeline (Lightweight Makefile)"
	@echo "-----------------------------------------------"
	@echo "Available targets:"
	@echo "  make grad       Run I-FDG on das_ani VSGs (config-driven)"
	@echo "  make eval       Run the synthetic-recovery benchmark"
	@echo "  make test       Run the pytest suite"
	@echo "  make paths      Print current variable configuration"
	@echo ""
	@echo "Override examples:"
	@echo "  make grad GRAD_CFG=configs/urban_grad.yaml"
	@echo "  make eval EVAL_OUT=data/benchmarks/synth_noise"
	@echo ""

# ============================================================
# Sanity print
# ============================================================
paths:
	@echo "GRAD_CFG = $(GRAD_CFG)"
	@echo "EVAL_OUT = $(EVAL_OUT)"
	@echo "PYTHON   = $(PYTHON)"

# ============================================================
# GRADIOMETRY (I-FDG)
# ============================================================
grad:
	@echo ">>> Running I-FDG gradiometry"
	@mkdir -p data/grad
	$(PYTHON) -m src.grad --config $(GRAD_CFG) --verbose

# ============================================================
# SYNTHETIC BENCHMARK
# ============================================================
eval:
	@echo ">>> Running synthetic-recovery benchmark"
	@mkdir -p $(EVAL_OUT)
	$(PYTHON) -m src.eval --outdir $(EVAL_OUT)

# ============================================================
# TESTS
# ============================================================
test:
	@echo ">>> Running pytest"
	$(PYTHON) -m pytest tests -q
