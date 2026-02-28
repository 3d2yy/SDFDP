# SDFDP

Signal-based UHF Partial Discharge (UHF-PD) processing and validation framework.

## Overview

This repository contains two complementary layers:

1. **Numerical Core (research pipeline)**
   - Phase 1: Stochastic wavelet optimization (Monte Carlo + grid search)
   - Phase 2: Variable isolation via inter-pulse interval extraction (Δt)
   - Phase 3: Tracking with Kalman, adaptive EWMA, and CUSUM
   - Phase 4: Quantification (empirical Big-O + convergence/FPR confusion matrix)

2. **Operational Layer (legacy-compatible diagnostics + GUI integration)**
   - Preprocessing, descriptor computation, severity index, and traffic-light classification.

## Key Features

- **Advanced preprocessing**
  - Butterworth band-pass filtering
  - Signal normalization (z-score, min-max, robust)
  - Hilbert envelope extraction
  - Wavelet denoising and adaptive filtering

- **Doctoral-level validation framework**
  - AWGN Monte Carlo simulation with constrained optimization
  - Δt-only descriptor pathway for variable isolation
  - Exclusive tracking algorithms: 1D Kalman, adaptive EWMA, CUSUM
  - Complexity estimation and asymptotic characterization

- **Backward compatibility**
  - Legacy feature descriptors and severity scoring remain available
  - Existing application scripts continue to run

## Project Structure

```
SDFDP-main/
├── preprocessing.py
├── descriptors.py
├── blind_algorithms.py
├── validation.py
├── severity.py
├── main.py
├── app.py
├── start_gui.py
├── test_system.py
├── requirements.txt
└── gui/
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Usage

### 1) Legacy full demo

```bash
python main.py
```

### 2) New Phase 1–4 numerical workflow

```python
import numpy as np
from preprocessing import generate_uhf_reference_signal, monte_carlo_wavelet_optimization
from descriptors import extract_delta_t_vector
from blind_algorithms import apply_delta_t_tracking
from validation import (
    measure_all_tracking_complexities,
    generate_convergence_confusion_matrix,
    generate_phase4_report,
)

# Phase 1
clean, noisy = generate_uhf_reference_signal(n_samples=4096, seed=42)
mc = monte_carlo_wavelet_optimization(reference_clean=clean, n_iterations=1000)

# Phase 2
fs = 1e9
delta_t = extract_delta_t_vector(noisy, fs, threshold_sigma=3.0)

# Phase 3
tracking = apply_delta_t_tracking(delta_t)

# Phase 4
complexity = measure_all_tracking_complexities()
confusion = generate_convergence_confusion_matrix()
print(generate_phase4_report(complexity, confusion))
```

## Main Outputs

- Optimized wavelet family and threshold configuration
- 1D Δt vector (inter-pulse intervals)
- Tracking outputs (filtered trajectories, residuals, alarms)
- Empirical complexity estimates in form `O(n^b)`
- Convergence-latency vs false-positive-rate matrices across event-rate variability

## Dependencies

- numpy
- scipy
- PyWavelets
- pandas
- dash / plotly (GUI layer)

## License

MIT (see LICENSE / license.md).
