# SDFDP - Unified English Documentation

This document consolidates all English project documentation into a single reference.

## Contents

1. Core Project Overview (from README.md)
2. Quick Start Guide (from INICIO_RAPIDO.md)
3. GUI Reference (from GUI_README.md)
4. Executive Technical Overview (from SISTEMA_COMPLETO.md)
5. Architecture Summary (from RESUMEN.md)
6. UI/UX Design Notes (from MEJORAS_DISEÑO.md)
7. File Inventory (from ARCHIVOS_CREADOS.md)
8. License (from license.md)

---

## 1) Core Project Overview

Source: README.md

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

---

## 2) Quick Start Guide

Source: INICIO_RAPIDO.md

# 🚀 Quick Start Guide - UHF-PD System

## ✅ Current Status

The project is installed and ready to run.

---

## 🎯 Start the Application

### Option 1: Simple start
```bash
python app.py
```

### Option 2: Start script (recommended)
```bash
python start_gui.py
```

### Option 3: Custom options
```bash
python start_gui.py --port 8080
python start_gui.py --debug
python start_gui.py --port 8080 --debug
```

---

## 🌐 Open the UI

After startup, open:

**http://localhost:8050**

For remote servers:

**http://[SERVER_IP]:8050**

---

## 📋 Available Tabs

### 📡 Live Capture
- Real-time monitoring
- NI PXIe-5185 hardware mode
- Simulation mode

### 📂 File Analysis
- Offline analysis of saved data
- Supports CSV, HDF5, and MATLAB files

### ⚙️ Signal Generator
- Generate synthetic datasets
- Export in multiple formats

### 🎯 Threshold Configuration
- Calibrate severity thresholds
- Run validation tests

### 📚 Documentation
- In-app reference and technical notes

---

## 🧪 Quick Validation Steps

### 1) Verify installation
```bash
python test_system.py
```

### 2) First simulation test
1. Run `python app.py`
2. Open **📡 Live Capture**
3. Select **Simulation Mode**
4. Select **🟢 Green**
5. Click **Start Capture**
6. Confirm live plots update continuously

### 3) Generator test
1. Open **⚙️ Signal Generator**
2. Select **🔴 Red**
3. Set duration, number of discharges, and amplitude
4. Click **Generate Signal**
5. Review spectrum/statistics and optionally export

### 4) Threshold test
1. Open **🎯 Threshold Configuration**
2. Keep default thresholds (or edit)
3. Run full validation
4. Review confusion matrix and metrics

---

## 🔧 Real Hardware Setup (NI PXIe-5185)

1. Install NI-DAQmx driver from National Instruments.
2. Install Python bindings:
```bash
pip install nidaqmx
```
3. Check device name in NI MAX (for example `PXI1Slot2`).

In **📡 Live Capture**:
- Select hardware mode
- Set device/channel/sample rate
- Start capture

If capture fails:
- check cabling
- verify NI MAX device name
- validate permissions
- test simulation mode first

---

## 📊 File Input Formats

### CSV
Expected layout example:
```csv
time,signal
0.0000,0.0123
0.0001,0.0245
```

### HDF5
Provide dataset name containing the signal.

### MATLAB
Provide variable name containing the signal.

---

## 🎓 Result Interpretation

| State | Meaning | Typical Action |
|-------|---------|----------------|
| 🟢 Green | Normal | Routine monitoring |
| 🟡 Yellow | Caution | Increase monitoring frequency |
| 🟠 Orange | Alert | Plan maintenance |
| 🔴 Red | Critical | Immediate intervention |

---

## 🆘 Troubleshooting

**App does not start**
```bash
pip install -r requirements.txt
python test_system.py
```

**Plots do not refresh**
- hard refresh browser
- clear cache
- inspect browser console

**File upload fails**
- verify format (CSV/H5/MAT)
- verify signal field/column name

---

## 📌 Recommended Next Steps

1. Explore simulation mode
2. Analyze existing recordings
3. Generate synthetic data for benchmarking
4. Calibrate thresholds to your equipment
5. Move to real-hardware acquisition

---

## 3) GUI Reference

Source: GUI_README.md

# 🔌 UHF Partial Discharge Detection System - Graphical Interface

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-2.14+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Professional platform for real-time monitoring and offline analysis of partial discharge signals**

</div>

---

## 🚀 Main Features

### 📡 **Live Capture**
- **Real Hardware**: Compatible with NI PXIe-5185 (12.5 GS/s, 3 GHz BW, 8-bit)
- **Simulation Mode**: Synthetic generation for no-hardware testing
- **Real-Time Monitoring**: Continuous plotting of signals and descriptors
- **Automatic Classification**: Traffic-light severity states (Green/Yellow/Orange/Red)

### 📂 **File Analysis**
- **Multiple Formats**: CSV, HDF5 (.h5), MATLAB (.mat)
- **Full Visualizations**: Signal, spectrum, descriptors, radar chart
- **Advanced Processing**: Filtering, normalization, envelope extraction
- **Severity Evaluation**: Automatic classification with detailed outputs

### ⚙️ **Signal Generator**
- **Custom Parameters**: State, amplitude, frequency, noise
- **Noise Types**: Gaussian, Pink, Brown, Uniform
- **Multi-Format Export**: CSV, HDF5, MAT with metadata
- **Immediate Analysis**: Statistics, spectrum, histograms

### 🎯 **Threshold Configuration**
- **Custom Thresholds**: Adjust classification boundaries
- **Descriptor Weights**: Control relative importance
- **Interactive Tests**: Generate and classify in real time
- **Full Validation**: Confusion matrix and accuracy metrics

### 📚 **Integrated Documentation**
- Step-by-step user guidance
- Technical specifications
- Best practices

---

## 📦 Installation

### 1. Clone or download the repository

```bash
cd /workspaces/V2DP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. (Optional) Install NI hardware support

If you plan to use National Instruments hardware:

```bash
pip install nidaqmx
```

---

## 🎯 Quick Use

### Start the application

```bash
python app.py
```

The interface is available at: **http://localhost:8050**

### Recommended workflow

1. **📚 Documentation**: Understand system behavior
2. **🎯 Threshold Configuration**: Adjust parameters as needed
3. **⚙️ Generator**: Create synthetic test signals
4. **📂 File Analysis**: Analyze existing recordings
5. **📡 Live Capture**: Move to real-time monitoring

---

## 🔧 Configuration

### NI PXIe-5185 Hardware

To use real hardware, in **Live Capture**:

1. Select "NI PXIe-5185 Hardware"
2. Configure:
   - **Device**: Device name (for example `PXI1Slot2`)
   - **Channel**: Analog channel number (for example `0`)
   - **Sampling Rate**: In GS/s (for example `12.5`)
3. Start capture

### Simulation Mode

For no-hardware testing:

1. Select "Simulation Mode"
2. Choose state:
   - 🟢 Green (Normal)
   - 🟡 Yellow (Caution)
   - 🟠 Orange (Alert)
   - 🔴 Red (Critical)
3. Tune noise level
4. Start capture

---

## 📊 Computed Descriptors

The operational path computes nine descriptors:

| # | Descriptor | Description |
|---|------------|-------------|
| 1 | **Total Energy** | Sum of squared signal amplitudes |
| 2 | **RMS** | Root mean square value |
| 3 | **Kurtosis** | Tail/peakedness indicator |
| 4 | **Skewness** | Distribution asymmetry |
| 5 | **Crest Factor** | Peak-to-RMS ratio |
| 6 | **Peak Count** | Number of significant peaks |
| 7 | **Spectral Entropy** | Spectral disorder |
| 8 | **Spectral Stability** | Inter-window spectral consistency |
| 9 | **Zero-Crossing Rate** | Sign-change frequency |

---

## 🎨 Project Structure

```
V2DP/
├── app.py                      # Main Dash application
├── gui/
│   ├── __init__.py
│   ├── live_capture.py         # Real-time capture tab
│   ├── file_analysis.py        # File analysis tab
│   ├── signal_generator.py     # Signal generator tab
│   ├── threshold_config.py     # Threshold configuration tab
│   └── documentation.py        # In-app docs tab
├── main.py                     # Backend processing layer
├── preprocessing.py            # Signal preprocessing + MC optimization
├── descriptors.py              # Δt extraction + legacy descriptors
├── severity.py                 # Severity scoring and traffic-light mapping
├── blind_algorithms.py         # Δt tracking algorithms
├── validation.py               # Complexity and validation metrics
└── requirements.txt            # Dependencies
```

---

## 🔬 Technical Specifications

### Acquisition System

| Component | Specification |
|------------|----------------|
| **System** | NI PXIe-1071 |
| **Controller** | NI PXIe-8135 (Embedded) |
| **Digitizer** | NI PXIe-5185 |
| **Bandwidth** | 3 GHz |
| **Sampling Rate** | 12.5 GS/s |
| **Resolution** | 8 bits |

### Signal Processing

- **Filtering**: Band-pass (1% - 40% of fs)
- **Normalization**: Adaptive
- **Envelope**: Hilbert transform
- **Denoising**: Wavelets

---

## 📖 Usage Examples

### Example 1: Analyze a CSV file

```python
# In the "File Analysis" tab:
# 1. Upload a CSV signal file
# 2. Set fs = 10000 Hz
# 3. Set data column = "signal"
# 4. Click "Analyze Signal"
# 5. Review classification and descriptors
```

### Example 2: Generate a synthetic dataset

```python
# In the "Signal Generator" tab:
# 1. State = "Orange"
# 2. Duration = 5000 samples
# 3. Discharges = 30
# 4. Amplitude = 4.0
# 5. Click "Generate Signal"
# 6. Export as HDF5 with metadata
```

### Example 3: Calibrate thresholds

```python
# In the "Threshold Configuration" tab:
# 1. Set Green→Yellow = 0.3
# 2. Set Yellow→Orange = 0.6
# 3. Set Orange→Red = 0.8
# 4. Click "Run Full Test"
# 5. Review confusion matrix and accuracy
```

---

## 🐛 Troubleshooting

### Error: "nidaqmx is not installed"

```bash
pip install nidaqmx
```

### Error: "h5py not found"

```bash
pip install h5py
```

### Application does not start

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### NI hardware is not detected

1. Verify NI-DAQmx driver installation
2. Confirm the device name in NI MAX
3. Use the exact device name in settings

---

## 🤝 Contributing

Contributions are welcome. Please open an issue before large structural changes.

---

## 📄 License

See LICENSE / license.md.

---

## 🙏 Acknowledgements

Built with:
- **Dash & Plotly** for interactive visualization
- **NumPy & SciPy** for scientific processing
- **NI-DAQmx** for instrumentation integration
- **Bootstrap** for responsive UI design

---

<div align="center">

**🔌 UHF Partial Discharge Detection System**

*Professional real-time monitoring for high-voltage assets*

</div>

---

## 4) Executive Technical Overview

Source: SISTEMA_COMPLETO.md

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

---

## 5) Architecture Summary

Source: RESUMEN.md

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

---

## 6) UI/UX Design Notes

Source: MEJORAS_DISEÑO.md

# Professional UI/UX Design Improvements

## Purpose

This document summarizes major visual and usability improvements delivered for the web interface layer.

## Implemented Improvements

### 1) Visual Theme Modernization
- Professional dark-theme baseline
- Improved contrast and readability
- Consistent color usage for operational states

### 2) Typography and Hierarchy
- Cleaner heading/body hierarchy
- Better spacing between blocks and controls
- Improved scanability for live diagnostics

### 3) Interaction Quality
- Smoother hover/transition behavior
- Better button/input feedback
- More coherent layout behavior across tabs

### 4) Plot Presentation
- Cleaner plot styling and grid behavior
- Better color consistency across charts
- Improved real-time readability

### 5) Navigation Clarity
- More explicit tab semantics
- Stronger visual distinction for active sections
- Better organization of controls and outputs

## File Impact Summary

- `app.py`: shell layout and styling updates
- `gui/live_capture.py`: visualization and status presentation improvements
- `gui/plot_utils.py`: shared plotting style utilities
- `start_gui.py`: startup compatibility adjustments

## Design Principles Used

- Clarity over ornament
- State-driven visual cues
- Consistency across all tabs
- Fast interpretation in monitoring scenarios

## Next Design Iterations (Optional)

- Theme toggle (dark/light)
- Accessibility pass (WCAG contrast + keyboard focus refinement)
- Export-ready reporting views
- Compact mode for dense diagnostic sessions

---

## 7) File Inventory

Source: ARCHIVOS_CREADOS.md

# Created and Updated Files Inventory

## Objective

This document tracks the principal files involved in the SDFDP implementation and refactor, including the numerical validation framework and GUI support scripts.

## Core Numerical Modules

- `preprocessing.py`
  - Signal conditioning utilities
  - Phase 1 stochastic optimization (Monte Carlo + wavelet grid search)

- `descriptors.py`
  - Phase 2 variable isolation via `Δt` extraction
  - Legacy descriptor functions retained for compatibility

- `blind_algorithms.py`
  - Phase 3 tracking algorithms
  - 1D Kalman, adaptive EWMA, CUSUM

- `validation.py`
  - Phase 4 quantification
  - Complexity estimation, confusion matrix generation, reporting

- `severity.py`
  - Severity index and traffic-light classification

- `main.py`
  - Legacy integration and demonstration workflow

## Application and Utility Scripts

- `app.py` - Dash application entry point
- `start_gui.py` - Launch helper with runtime options
- `demo.py` - Interactive terminal demos
- `test_system.py` - System checks

## GUI Package (Interface Layer)

- `gui/__init__.py`
- `gui/live_capture.py`
- `gui/file_analysis.py`
- `gui/signal_generator.py`
- `gui/threshold_config.py`
- `gui/documentation.py`
- `gui/plot_utils.py`

## Documentation Files

- `README.md` - Main project documentation (updated, English)
- `INICIO_RAPIDO.md` - Quick start guide (updated, English)
- `GUI_README.md` - GUI documentation (updated, English)
- `SISTEMA_COMPLETO.md` - Executive technical overview (updated, English)
- `RESUMEN.md` - Condensed architecture summary (updated, English)
- `MEJORAS_DISEÑO.md` - UI/UX design notes (updated, English)
- `ARCHIVOS_CREADOS.md` - This file

## Dependency File

- `requirements.txt`
  - Numerical stack + GUI dependencies
  - Optional NI hardware dependency noted

## Notes

- The numerical refactor was implemented without editing interface logic under `gui/`.
- Legacy APIs remain available to preserve compatibility with existing workflows.

---

## 8) License

Source: license.md

MIT License

Copyright (c) 2026 Benito Yair Acuña Meléndez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
