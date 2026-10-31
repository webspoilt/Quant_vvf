.PHONY: help test-phase0 test-phase1 test-phase2 test-phase3 test-phase4 test-phase5 test-all lint clean

PYTHON ?= python
PYTEST ?= pytest

help:
	@echo "Quant_vvf Engine - Make targets:"
	@echo "  test-phase0  - Run Phase 0 Data Feasibility and Numeraire Accounting Kill Gate"
	@echo "  test-phase1  - Run Phase 1 SVI Calibration and Baseline VRP Regression Kill Gate"
	@echo "  test-phase2  - Run Phase 2 Realized Measures and HAR-RV Benchmark Kill Gate"
	@echo "  test-phase3  - Run Phase 3 Deterministic Tail Timing and Risk Veto Kill Gate"
	@echo "  test-phase4  - Run Phase 4 Greek Profile and Solver Tests"
	@echo "  test-phase5  - Run Phase 5 Nautilus Execution Actor and Greeks Monitor Tests"
	@echo "  test-all     - Run full test suite across all phases"
	@echo "  lint         - Run mypy type checking"

test-phase0:
	$(PYTEST) -v -s tests/bias_checks/test_lookahead.py tests/unit/test_inverse_payoffs.py tests/regression/test_phase0_numeraire.py

test-phase1:
	$(PYTEST) -v -s tests/unit/test_svi_surface.py tests/regression/test_phase1_vrp_regression.py

test-phase2:
	$(PYTEST) -v -s tests/unit/test_realized_measures.py tests/regression/test_phase2_benchmark.py

test-phase3:
	$(PYTEST) -v -s tests/unit/test_tail_triggers.py tests/unit/test_risk_constants.py tests/regression/test_phase3_tail_timing.py

test-phase4:
	$(PYTEST) -v -s tests/unit/test_greek_strategy.py

test-phase5:
	$(PYTEST) -v -s tests/unit/test_nautilus_execution.py

test-all:
	$(PYTEST) -v

lint:
	$(PYTHON) -m mypy libs/core libs/data strategies risk execution monitoring simulation

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
