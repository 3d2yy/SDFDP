# UHF-PD Detection System - Executive Technical Overview

## Scope

This project provides a complete software stack for UHF partial discharge signal processing, from preprocessing to validation, with both:

- a **research-grade numerical framework** (Phases 1–4), and
- a **production-oriented operational workflow** compatible with current GUI tools.

## Implemented Capability Map

### A) Numerical / Asymptotic Framework

#### Phase 1 - Stochastic Optimization
- Grid search over wavelets: `db4`, `sym8`, `coif3`
- Threshold mode/rule combinations
- AWGN Monte Carlo simulation (`N=1000`)
- Objective: minimize expected RMSE under variance constraint `Var(RMSE) < ε`

#### Phase 2 - Variable Isolation
- Pulse detection and extraction of 1D inter-pulse intervals `Δt`
- Explicit suppression of amplitude propagation in the primary research pathway

#### Phase 3 - Tracking Evaluation
- 1D Kalman filter
- Adaptive EWMA
- CUSUM
- Unified execution API for consistent comparative output

#### Phase 4 - Quantification
- Empirical complexity estimation via `t(n) = a * n^b`
- Big-O approximation from fitted exponent `b`
- Convergence-latency vs false-positive matrices across stochastic event-rate variation

### B) Operational Workflow (Legacy-Compatible)

- Descriptor-based feature set for severity assessment
- Weighted severity index
- Traffic-light classification
- Validation metrics and reporting functions used by existing scripts

## File-Level Responsibility

- `preprocessing.py`: signal conditioning + stochastic optimization
- `descriptors.py`: `Δt` extraction + legacy descriptors
- `blind_algorithms.py`: tracking algorithms
- `validation.py`: quantification and validation reports
- `severity.py`: severity scoring and state mapping
- `main.py`: orchestration and demonstrations
- `app.py` / `start_gui.py`: GUI and application startup
- `gui/`: interface modules (unchanged in this numerical refactor)

## Runtime Entry Points

```bash
python main.py
python app.py
python start_gui.py
python test_system.py
```

## Quality Status

- Core numerical modules import successfully.
- End-to-end Phase 1–4 smoke testing completed.
- Backward compatibility with `main.py` checked.

## Engineering Notes

- The repository now supports publication-style validation analyses while retaining existing diagnostic workflows.
- The numerical path is intentionally decoupled from UI concerns.
- Documentation and module-level narrative were updated to English.

## Suggested Roadmap

1. Add deterministic benchmark fixtures for reproducibility.
2. Add CI jobs for complexity/failure-rate drift alerts.
3. Add real-measurement case studies with labeled fault progression.
