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


# ===================================================================
# Legacy descriptors (kept for backward compatibility)
# ===================================================================


def energy_total(signal_data: Signal) -> float:
    """
    Calcula la energía total de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    energy : float
        Energía total de la señal
    """
    return np.sum(np.square(signal_data))


def energy_windowed(signal_data: Signal, window_size: int) -> Signal:
    """
    Calcula la energía en ventanas deslizantes.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    window_size : int
        Tamaño de la ventana
    
    Retorna:
    --------
    energies : ndarray
        Vector de energías por ventana
    """
    signal_data = np.asarray(signal_data)
    n = len(signal_data)
    n_windows = n - window_size + 1
    
    energies = np.zeros(n_windows)
    for i in range(n_windows):
        window = signal_data[i:i+window_size]
        energies[i] = np.sum(np.square(window))
    
    return energies


def energy_spectral_bands(signal_data: Signal, fs: float, bands: Optional[List[Tuple[float, float]]] = None) -> Dict[str, float]:
    """
    Calcula la energía en bandas de frecuencia específicas.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    fs : float
        Frecuencia de muestreo
    bands : list of tuples, opcional
        Lista de bandas de frecuencia [(f1_low, f1_high), (f2_low, f2_high), ...]
        Si es None, usa bandas predeterminadas
    
    Retorna:
    --------
    band_energies : dict
        Diccionario con energías por banda
    """
    signal_data = np.asarray(signal_data)
    n = len(signal_data)
    
    # Calcular FFT
    fft_values = fft(signal_data)
    fft_freq = fftfreq(n, 1/fs)
    power_spectrum = np.abs(fft_values)**2
    
    # Usar solo frecuencias positivas
    positive_freq_idx = fft_freq >= 0
    fft_freq = fft_freq[positive_freq_idx]
    power_spectrum = power_spectrum[positive_freq_idx]
    
    # Bandas predeterminadas si no se especifican
    if bands is None:
        bands = [
            (0, fs/8),       # Banda baja
            (fs/8, fs/4),    # Banda media-baja
            (fs/4, 3*fs/8),  # Banda media-alta
            (3*fs/8, fs/2)   # Banda alta
        ]
    
    band_energies = {}
    for i, (f_low, f_high) in enumerate(bands):
        band_mask = (fft_freq >= f_low) & (fft_freq < f_high)
        band_energy = np.sum(power_spectrum[band_mask])
        band_energies[f'band_{i+1}_{f_low:.1f}_{f_high:.1f}Hz'] = band_energy
    
    return band_energies


def rms_value(signal_data: Signal) -> float:
    """
    Calcula el valor RMS (Root Mean Square) de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    rms : float
        Valor RMS de la señal
    """
    return np.sqrt(np.mean(np.square(signal_data)))


def kurtosis(signal_data: Signal) -> float:
    """
    Calcula la curtosis de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    kurt : float
        Curtosis de la señal
    """
    return stats.kurtosis(signal_data, fisher=True)


def skewness(signal_data: Signal) -> float:
    """
    Calcula la asimetría de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    skew : float
        Asimetría de la señal
    """
    return stats.skew(signal_data)


def crest_factor(signal_data: Signal) -> float:
    """
    Calcula el factor de cresta de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    cf : float
        Factor de cresta
    """
    peak = np.max(np.abs(signal_data))
    rms = rms_value(signal_data)
    
    if rms == 0:
        return 0
    
    return peak / rms


def spectral_entropy(signal_data: Signal, fs: float) -> float:
    """
    Calcula la entropía espectral de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    fs : float
        Frecuencia de muestreo
    
    Retorna:
    --------
    entropy : float
        Entropía espectral
    """
    signal_data = np.asarray(signal_data)
    n = len(signal_data)
    
    # Calcular espectro de potencia
    fft_values = fft(signal_data)
    power_spectrum = np.abs(fft_values[:n//2])**2
    
    # Normalizar como distribución de probabilidad
    power_spectrum = power_spectrum / np.sum(power_spectrum)
    
    # Evitar log(0)
    power_spectrum = power_spectrum[power_spectrum > 0]
    
    # Calcular entropía
    entropy = -np.sum(power_spectrum * np.log2(power_spectrum))
    
    return entropy


def spectral_stability(signal_data: Signal, fs: float, window_size: Optional[int] = None) -> float:
    """
    Calcula la estabilidad espectral entre ventanas consecutivas.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    fs : float
        Frecuencia de muestreo
    window_size : int, opcional
        Tamaño de la ventana. Si es None, se divide en 4 ventanas
    
    Retorna:
    --------
    stability : float
        Medida de estabilidad espectral (0-1, donde 1 es más estable)
    """
    signal_data = np.asarray(signal_data)
    n = len(signal_data)
    
    if window_size is None:
        window_size = n // 4
    
    if window_size < 16:
        window_size = min(16, n // 2)
    
    # Dividir señal en ventanas
    n_windows = n // window_size
    if n_windows < 2:
        return 1.0  # Solo una ventana, perfectamente estable
    
    spectra = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window = signal_data[start:end]
        
        # Calcular espectro normalizado
        fft_values = fft(window)
        spectrum = np.abs(fft_values[:window_size//2])
        spectrum = spectrum / (np.sum(spectrum) + 1e-10)
        spectra.append(spectrum)
    
    # Calcular correlación promedio entre espectros consecutivos
    correlations = []
    for i in range(len(spectra) - 1):
        corr = np.corrcoef(spectra[i], spectra[i+1])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    if len(correlations) == 0:
        return 1.0
    
    stability = np.mean(correlations)
    
    # Asegurar que esté en el rango [0, 1]
    stability = max(0, min(1, stability))
    
    return stability


def residual_energy(original_signal: Signal, filtered_signal: Signal) -> float:
    """
    Calcula la energía residual después del filtrado.
    
    Parámetros:
    -----------
    original_signal : array-like
        Señal original
    filtered_signal : array-like
        Señal filtrada
    
    Retorna:
    --------
    residual : float
        Energía del residuo
    """
    residual = np.asarray(original_signal) - np.asarray(filtered_signal)
    return energy_total(residual)


def peak_count(signal_data: Signal, threshold: Optional[float] = None) -> int:
    """
    Cuenta el número de picos en la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    threshold : float, opcional
        Umbral para detección de picos. Si es None, usa la media + 2*std
    
    Retorna:
    --------
    n_peaks : int
        Número de picos detectados
    """
    signal_data = np.asarray(signal_data)
    
    if threshold is None:
        threshold = np.mean(signal_data) + 2 * np.std(signal_data)
    
    peaks, _ = signal.find_peaks(signal_data, height=threshold)
    
    return len(peaks)


def zero_crossing_rate(signal_data: Signal) -> float:
    """
    Calcula la tasa de cruces por cero.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    zcr : float
        Tasa de cruces por cero (normalizada)
    """
    signal_data = np.asarray(signal_data)
    
    # Contar cambios de signo
    signs = np.sign(signal_data)
    zero_crossings = np.sum(np.abs(np.diff(signs))) / 2
    
    # Normalizar por la longitud
    zcr = zero_crossings / (len(signal_data) - 1)
    
    return zcr


def compute_all_descriptors(signal_data: Signal, fs: float, original_signal: Optional[Signal] = None) -> Dict[str, float]:
    """
    Calcula todos los descriptores de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal procesada
    fs : float
        Frecuencia de muestreo
    original_signal : array-like, opcional
        Señal original para calcular energía residual
    
    Retorna:
    --------
    descriptors : dict
        Diccionario con todos los descriptores
    """
    descriptors = {}
    
    # Descriptores energéticos
    descriptors['energy_total'] = energy_total(signal_data)
    
    # Energía por bandas espectrales
    band_energies = energy_spectral_bands(signal_data, fs)
    descriptors.update(band_energies)
    
    # Descriptores estadísticos
    descriptors['rms'] = rms_value(signal_data)
    descriptors['kurtosis'] = kurtosis(signal_data)
    descriptors['skewness'] = skewness(signal_data)
    descriptors['crest_factor'] = crest_factor(signal_data)
    
    # Descriptores espectrales
    descriptors['spectral_entropy'] = spectral_entropy(signal_data, fs)
    descriptors['spectral_stability'] = spectral_stability(signal_data, fs)
    
    # Otros descriptores
    descriptors['peak_count'] = peak_count(signal_data)
    descriptors['zero_crossing_rate'] = zero_crossing_rate(signal_data)
    
    # Energía residual si se proporciona señal original
    if original_signal is not None:
        descriptors['residual_energy'] = residual_energy(original_signal, signal_data)
    
    return descriptors
