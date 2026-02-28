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
