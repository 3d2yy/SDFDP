# System Summary

## General Description

SDFDP is a UHF partial discharge processing framework that now includes a doctoral-level numerical validation path and a legacy-compatible operational path.

- **Numerical path (Phases 1–4)**: stochastic optimization, variable isolation (Δt), tracking algorithms, and asymptotic quantification.
- **Operational path**: descriptor-based severity scoring and traffic-light classification used by existing scripts and GUI workflows.

## Architecture Overview

### 1) Preprocessing (`preprocessing.py`)
- Band-pass filtering
- Signal normalization
- Hilbert envelope extraction
- Wavelet denoising
- LMS adaptive filtering
- **Phase 1**: Monte Carlo + grid search optimization over wavelet families and threshold rules

### 2) Descriptor Layer (`descriptors.py`)
- **Primary research output**: 1D inter-pulse interval vector `Δt`
- Pulse detection and inter-event time calculation
- Legacy descriptors retained for compatibility (`compute_all_descriptors`)

### 3) Tracking Algorithms (`blind_algorithms.py`)
- **1D Kalman tracker** for Δt
- **Adaptive EWMA** for nonstationary event rates
- **CUSUM** change detector
- Unified runner: `apply_delta_t_tracking`

### 4) Validation and Quantification (`validation.py`)
- **Phase 4 complexity analysis**: empirical `O(n^b)` fitting
- Convergence-latency and false-positive confusion matrices over stochastic event-rate variation
- Report generator: `generate_phase4_report`
- Legacy validation helpers preserved

### 5) Severity Layer (`severity.py`)
- Descriptor normalization to baseline
- Weighted severity index
- Traffic-light state assignment (green/yellow/orange/red)

### 6) Integration (`main.py`)
- Synthetic signal generation
- End-to-end processing demo
- Legacy comparison/reporting functions

## Typical Research Workflow (Phases 1–4)

1. Generate/obtain clean reference UHF-PD signal.
2. Run stochastic wavelet optimization under AWGN Monte Carlo.
3. Extract `Δt` from processed signal.
4. Apply Kalman, adaptive EWMA, and CUSUM.
5. Quantify complexity and robustness with confusion matrices.

## Typical Operational Workflow

1. Acquire signal (live or file).
2. Preprocess signal.
3. Compute descriptors.
4. Estimate severity index.
5. Classify state using traffic-light thresholds.

## Current Technical Notes

- The Phase 1–4 numerical framework is active in core modules.
- Legacy APIs remain available to avoid breaking existing scripts.
- Type annotations were expanded across the numerical stack (PEP 484 style).

## Recommended Next Steps

- Add CI benchmarks for complexity regression.
- Add dataset-specific calibration profiles for real equipment.
- Add reproducible experiment notebooks for publication workflows.
