"""
Módulo de cálculo de descriptores para análisis de descargas parciales.

Phase 2 — Variable Isolation:
    The primary interface of this module is :func:`extract_delta_t_vector`.
    It detects UHF-PD pulses in the (pre-processed) signal and returns a
    **single one-dimensional vector** containing the time differences Δt
    between consecutive pulses.  Amplitude data is **not** propagated.

    Legacy energy / spectral / statistical descriptors are retained in a
    ``_legacy`` namespace for backward compatibility but are **excluded**
    from the doctoral-level validation pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import signal, stats
from scipy.fft import fft, fftfreq

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Signal = NDArray[np.floating[Any]]


# ===================================================================
# Phase 2 — Pulse detection & Δt extraction  (PRIMARY INTERFACE)
# ===================================================================

def detect_pulses(
    signal_data: Signal,
    fs: float,
    threshold_sigma: float = 3.0,
    min_separation_s: float = 0.0,
    method: str = "threshold",
) -> NDArray[np.intp]:
    """Detect PD pulses in a pre-processed UHF signal.

    Parameters
    ----------
    signal_data : Signal
        Pre-processed (envelope / denoised) signal.
    fs : float
        Sampling frequency in Hz.
    threshold_sigma : float
        Number of standard deviations above the mean used as the peak
        detection threshold (only for ``method='threshold'``).
    min_separation_s : float
        Minimum time separation (in seconds) between consecutive pulses.
        Translated to samples internally.
    method : str
        ``'threshold'`` — simple amplitude threshold on the absolute signal.
        ``'scipy_peaks'`` — ``scipy.signal.find_peaks`` with prominence.

    Returns
    -------
    pulse_indices : ndarray of int
        Sample indices where pulses were detected, sorted ascending.
    """
    data = np.asarray(signal_data, dtype=np.float64)
    abs_data = np.abs(data)

    min_distance: int = max(1, int(min_separation_s * fs))

    if method == "threshold":
        mu = np.mean(abs_data)
        sigma = np.std(abs_data)
        height = mu + threshold_sigma * sigma
        peaks, _ = signal.find_peaks(abs_data, height=height, distance=min_distance)
    elif method == "scipy_peaks":
        # Use prominence-based detection (more robust for UHF-PD)
        prominence = np.std(abs_data) * threshold_sigma * 0.5
        peaks, _ = signal.find_peaks(
            abs_data,
            prominence=prominence,
            distance=min_distance,
        )
    else:
        raise ValueError(f"Unknown detection method: {method!r}")

    return np.sort(peaks)


def compute_delta_t(
    pulse_indices: NDArray[np.intp],
    fs: float,
) -> Signal:
    """Compute Δt — time differences between consecutive detected pulses.

    Parameters
    ----------
    pulse_indices : ndarray of int
        Sorted sample indices of detected pulses.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    delta_t : Signal
        1-D vector of inter-pulse time intervals in **seconds**.
        Length is ``len(pulse_indices) - 1``.

    Raises
    ------
    ValueError
        If fewer than two pulses are provided.
    """
    if len(pulse_indices) < 2:
        raise ValueError(
            f"At least 2 pulses are required to compute Δt "
            f"(got {len(pulse_indices)})."
        )
    idx = np.sort(pulse_indices)
    delta_samples: NDArray[np.intp] = np.diff(idx)
    delta_t: Signal = delta_samples.astype(np.float64) / fs
    return delta_t


def extract_delta_t_vector(
    signal_data: Signal,
    fs: float,
    threshold_sigma: float = 3.0,
    min_separation_s: float = 0.0,
    detection_method: str = "threshold",
) -> Signal:
    """Primary descriptor interface — returns a 1-D Δt vector.

    This is the **single output** prescribed by Phase 2 of the validation
    framework.  Amplitude information is deliberately excluded.

    Parameters
    ----------
    signal_data : Signal
        Pre-processed UHF-PD signal.
    fs : float
        Sampling frequency in Hz.
    threshold_sigma : float
        Detection threshold in units of σ.
    min_separation_s : float
        Minimum inter-pulse gap in seconds.
    detection_method : str
        ``'threshold'`` or ``'scipy_peaks'``.

    Returns
    -------
    delta_t : Signal
        1-D vector ``[Δt₁, Δt₂, …, Δtₙ₋₁]`` in seconds.
    """
    pulse_idx = detect_pulses(
        signal_data,
        fs,
        threshold_sigma=threshold_sigma,
        min_separation_s=min_separation_s,
        method=detection_method,
    )
    return compute_delta_t(pulse_idx, fs)


