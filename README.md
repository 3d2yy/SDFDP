# SDFDP

Signal-based UHF Partial Discharge (UHF-PD) processing and validation framework for doctoral research.

## Overview

Four-phase numerical pipeline for evaluating UHF-PD detection algorithms:

1. **Phase 1 — Stochastic Wavelet Optimisation**: Monte Carlo grid search across wavelet families (db4, sym8, coif3) and thresholding rules (soft/hard × universal/minimax/sqtwolog). Selects the configuration that minimises E[RMSE] subject to Var[RMSE] < ε.

2. **Phase 2 — Variable Isolation (Δt Extraction)**: Detects PD pulses in the denoised signal and computes a 1-D inter-pulse interval vector Δt. All other amplitude/spectral descriptors are deliberately excluded.

3. **Phase 3 — Tracking Evaluation**: Three algorithms are applied to the Δt vector:
   - 1-D Kalman Filter
   - Adaptive EWMA (smoothing factor driven by local variance)
   - Two-sided CUSUM (Page 1954) change-point detector

4. **Phase 4 — Quantification**:
   - Empirical Big-O complexity estimation via power-law fitting `t(n) = a·n^b`
   - Convergence-latency vs false-positive-rate confusion matrix across stochastic event-rate variation levels

## Signal Generation Model

The synthetic UHF-PD signal is generated via a physics-based model:

- **PD current pulse**: Gemant-Philippoff double-exponential `i(t) = I₀·(exp(−t/τ₁) − exp(−t/τ₂))`
- **Dielectric channel**: Complex permittivity `ε*(f) = ε_r·ε₀·(1−j·tan δ)` with configurable `ε_r`, `tan δ`, and propagation distance
- **Vivaldi antenna**: Butterworth bandpass model (300 MHz – 3 GHz) approximating the UWB reception window

## Project Structure

```
SDFDP-main/
├── preprocessing.py      # Phase 1: wavelet optimisation + signal generation
├── descriptors.py         # Phase 2: Δt extraction
├── blind_algorithms.py    # Phase 3: Kalman / EWMA / CUSUM trackers
├── validation.py          # Phase 4: Big-O + convergence/FPR
├── main.py                # Full pipeline entry point
├── test_system.py         # System verification tests
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Full pipeline execution

```bash
python main.py
```

### Programmatic usage

```python
from main import run_phase1, run_phase2, run_phase3, run_phase4

# Phase 1 — Wavelet optimisation
mc_result, clean, noisy = run_phase1(n_samples=4096, fs=1e9, seed=42)

# Phase 2 — Δt extraction
delta_t, denoised = run_phase2(noisy, fs=1e9, mc_result=mc_result)

# Phase 3 — Tracking
tracking = run_phase3(delta_t)

# Phase 4 — Quantification
complexity, confusion, report = run_phase4()
print(report)
```

### System tests

```bash
python test_system.py
```

## Dependencies

- numpy ≥ 1.24
- scipy ≥ 1.10
- PyWavelets ≥ 1.4
- pandas ≥ 2.0
- matplotlib ≥ 3.7

## License

MIT (see LICENSE / license.md).
