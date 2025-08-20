# Makefile for Aikagrya-ALIGNMENTMANDALA Research

.PHONY: help install test mmip mmip-full clean lint format check all

# Default target
help:
	@echo "Aikagrya-ALIGNMENTMANDALA Research - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install     Install package in development mode"
	@echo "  make install-all Install with all optional dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test        Run test suite"
	@echo "  make test-cov    Run tests with coverage report"
	@echo ""
	@echo "MMIP Experiments:"
	@echo "  make mmip        Run quick MMIP trials (n=10)"
	@echo "  make mmip-full   Run full MMIP trials (n=100)"
	@echo "  make mmip-sweep  Run parameter sweep"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint        Check code style with ruff"
	@echo "  make format      Format code with black"
	@echo "  make check       Run all checks (lint + test)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       Remove build artifacts and caches"
	@echo "  make clean-all   Deep clean including results"

# Installation
install:
	pip install -e .
	@echo "✅ Package installed in development mode"

install-all:
	pip install -e ".[all]"
	@echo "✅ Package installed with all optional dependencies"

install-dev:
	pip install -e ".[dev]"
	@echo "✅ Development dependencies installed"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/aikagrya --cov-report=term-missing

test-quick:
	pytest tests/ -v -k "not slow"

# MMIP Experiments
mmip:
	@echo "Running quick MMIP trials (n=10)..."
	python -m aikagrya.mmip --trials 10 --dim 512

mmip-full:
	@echo "Running full MMIP trials (n=100)..."
	python -m aikagrya.mmip --trials 100 --dim 512 --perturb

mmip-sweep:
	@echo "Running parameter sweep..."
	python -m aikagrya.mmip --sweep --trials 20

# L4 Protocol Tests
l4-test:
	@echo "Running L4 protocol test..."
	cd alignment-lattice/thermo-L4 && python l4_reveal_verify_v22.py

# Code Quality
lint:
	ruff src/ tests/

format:
	black src/ tests/

format-check:
	black --check src/ tests/

mypy:
	mypy src/

check: lint test
	@echo "✅ All checks passed"

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/
	@echo "✅ Build artifacts cleaned"

clean-results:
	rm -rf runs/
	rm -rf analysis/*.jsonl
	@echo "⚠️  Result files cleaned"

clean-all: clean clean-results
	@echo "✅ Deep clean complete"

# Development workflow
dev: format lint test
	@echo "✅ Ready for commit"

# Run everything
all: install format lint test mmip
	@echo "✅ Full pipeline complete"

# Git hooks (optional)
install-hooks:
	@echo "#!/bin/sh\nmake check" > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✅ Git pre-commit hook installed"

.DEFAULT_GOAL := help
