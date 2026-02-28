"""
Módulo de preprocesamiento de señales UHF para detección de descargas parciales.

Este módulo proporciona funciones para:
- Filtrado pasabanda
- Normalización de señales
- Extracción de envolvente mediante transformada de Hilbert
- Eliminación de ruido mediante wavelets
- Optimización estocástica de parámetros wavelet (Monte Carlo + Grid Search)

Phase 1 — Stochastic Optimization:
    Implements a grid search across wavelet families {db4, sym8, coif3} and
    thresholding rules {soft, hard} × {universal, minimax, sqtwolog}.  A Monte
    Carlo simulation (N=1000) injects AWGN into a reference UHF-PD signal and
    selects the configuration that minimises E[RMSE] subject to Var[RMSE] < ε.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.signal import hilbert
import pywt

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Signal = NDArray[np.floating[Any]]


# ===================================================================
# Phase 1 — Stochastic wavelet optimisation data structures
# ===================================================================

@dataclass
class WaveletGridPoint:
    """Single point in the wavelet parameter grid."""

    wavelet: str
    threshold_mode: str          # 'soft' | 'hard'
    threshold_rule: str          # 'universal' | 'minimax' | 'sqtwolog'
    rmse_mean: float = 0.0
    rmse_var: float = 0.0
    rmse_samples: List[float] = field(default_factory=list)


@dataclass
class MonteCarloResult:
    """Aggregate result of the Monte-Carlo grid search."""

    best_wavelet: str
    best_threshold_mode: str
    best_threshold_rule: str
    best_rmse_mean: float
    best_rmse_var: float
    grid: List[WaveletGridPoint] = field(default_factory=list)
    n_iterations: int = 1000
    epsilon: float = 1e-3
    converged: bool = True


# ===================================================================
# Helper — reference UHF-PD signal generation
# ===================================================================

def generate_uhf_pd_signal_physical(
    n_samples: int = 4096,
    fs: float = 1e9,
    n_pulses: int = 12,
    snr_db: float = 30.0,
    seed: Optional[int] = None,
    # --- Dielectric channel parameters ---
    epsilon_r: float = 2.2,
    tan_delta: float = 0.005,
    propagation_distance_m: float = 0.3,
    # --- Vivaldi antenna parameters ---
    f_low_hz: float = 300e6,
    f_high_hz: float = 3e9,
    antenna_order: int = 4,
    # --- PD current-pulse parameters ---
    tau1_range_ns: Tuple[float, float] = (0.5, 2.0),
    tau2_range_ns: Tuple[float, float] = (5.0, 20.0),
    amplitude_range: Tuple[float, float] = (0.5, 2.0),
) -> Tuple[Signal, Signal]:
    """Generate a physics-based UHF partial-discharge signal.

    The model chains three physically-motivated stages:

    1. **PD current pulse** — Gemant-Philippoff double-exponential:
       ``i(t) = I₀ · (exp(−t/τ₁) − exp(−t/τ₂))``, with rise/fall time
       constants drawn from the ranges ``tau1_range_ns`` / ``tau2_range_ns``.

    2. **Dielectric channel transfer function** — derived from the complex
       permittivity ``ε*(f) = ε_r·ε₀·(1 − j·tan δ)`` of the insulating
       medium (default: mineral oil in power transformers).  The channel
       introduces frequency-dependent attenuation ``α(f)`` and phase
       ``β(f)`` over ``propagation_distance_m``.

    3. **Antipodal Vivaldi antenna** — modelled as a Butterworth bandpass
       of order ``antenna_order`` spanning ``[f_low_hz, f_high_hz]``,
       capturing the UWB reception window of a typical Vivaldi probe.

    Parameters
    ----------
    n_samples : int
        Length of the output signal vector.
    fs : float
        Sampling frequency in Hz (default 1 GHz for UHF).
    n_pulses : int
        Number of PD pulses to inject.
    snr_db : float
        Signal-to-noise ratio of the *clean* received signal in dB.
    seed : int, optional
        Random seed for reproducibility.
    epsilon_r : float
        Relative permittivity of the dielectric medium.
    tan_delta : float
        Loss tangent of the dielectric medium.
    propagation_distance_m : float
        Propagation distance through the dielectric in metres.
    f_low_hz, f_high_hz : float
        Lower and upper −3 dB frequencies of the Vivaldi antenna.
    antenna_order : int
        Butterworth order for the antenna transfer function model.
    tau1_range_ns, tau2_range_ns : tuple of float
        (min, max) range in nanoseconds for the rise (τ₁) and fall (τ₂)
        time constants of the Gemant-Philippoff current pulse.
    amplitude_range : tuple of float
        (min, max) current-pulse amplitude.

    Returns
    -------
    clean : Signal
        Noise-free received signal (after channel + antenna).
    noisy : Signal
        Signal contaminated with AWGN at the specified SNR.
    """
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)  # one-sided

    # ------------------------------------------------------------------
    # Stage 1 — Generate PD current pulses (Gemant-Philippoff)
    # ------------------------------------------------------------------
    current_signal: Signal = np.zeros(n_samples, dtype=np.float64)

    for _ in range(n_pulses):
        pos = rng.integers(50, n_samples - 200)
        tau1 = rng.uniform(*tau1_range_ns) * 1e-9          # rise
        tau2 = rng.uniform(*tau2_range_ns) * 1e-9          # fall
        amp = rng.uniform(*amplitude_range)
        dur = min(int(50e-9 * fs), n_samples - pos)        # 50 ns window
        t_local = np.arange(dur) / fs
        pulse = amp * (np.exp(-t_local / tau2) - np.exp(-t_local / tau1))
        current_signal[pos:pos + dur] += pulse

    # ------------------------------------------------------------------
    # Stage 2 — Dielectric channel transfer function  H_d(f)
    #
    #   ε*(f) = ε_r · ε₀ · (1 − j·tan δ)
    #   k(f)  = 2πf · √(μ₀ · ε*(f))  =  β(f) − j·α(f)
    #   H_d(f) = exp(−j · k(f) · d)
    # ------------------------------------------------------------------
    eps_0 = 8.854187817e-12       # F/m
    mu_0 = 4.0 * np.pi * 1e-7    # H/m
    d = propagation_distance_m

    eps_complex = epsilon_r * eps_0 * (1.0 - 1j * tan_delta)

    # Avoid DC singularity
    freqs_safe = freqs.copy()
    freqs_safe[0] = 1.0  # placeholder, DC bin zeroed later

    omega = 2.0 * np.pi * freqs_safe
    k = omega * np.sqrt(mu_0 * eps_complex)           # complex wavenumber
    H_dielectric = np.exp(-1j * k * d)
    H_dielectric[0] = 0.0  # suppress DC

    # ------------------------------------------------------------------
    # Stage 3 — Vivaldi antenna transfer function  H_ant(f)
    #
    #   Butterworth bandpass of order `antenna_order` evaluated on the
    #   frequency axis.  This approximates the UWB reception window.
    # ------------------------------------------------------------------
    f_center = np.sqrt(f_low_hz * f_high_hz)
    bw = f_high_hz - f_low_hz
    s = 1j * freqs_safe / f_center                    # normalised frequency
    # Butterworth magnitude-squared for bandpass
    n_ord = antenna_order
    H_ant_mag2 = 1.0 / (1.0 + ((freqs_safe**2 - f_center**2) / (freqs_safe * bw))**(2 * n_ord))
    H_antenna = np.sqrt(H_ant_mag2)
    H_antenna[0] = 0.0

    # ------------------------------------------------------------------
    # Apply channel in frequency domain
    # ------------------------------------------------------------------
    I_f = np.fft.rfft(current_signal)
    V_received_f = I_f * H_dielectric * H_antenna
    clean: Signal = np.fft.irfft(V_received_f, n=n_samples)

    # ------------------------------------------------------------------
    # Add AWGN at specified SNR
    # ------------------------------------------------------------------
    sig_power = np.mean(clean ** 2) + 1e-30
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), n_samples)
    noisy: Signal = clean + noise

    return clean, noisy


# Backward-compatible alias
generate_uhf_reference_signal = generate_uhf_pd_signal_physical


# ===================================================================
# Helper — RMSE
# ===================================================================

def compute_rmse(reference: Signal, estimate: Signal) -> float:
    """Root Mean Square Error between *reference* and *estimate*."""
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


# ===================================================================
# Helper — wavelet threshold value by rule
# ===================================================================

def _threshold_value(
    coeffs: List[NDArray[np.floating[Any]]],
    n: int,
    rule: str,
) -> float:
    """Compute a wavelet threshold value according to *rule*.

    Parameters
    ----------
    coeffs : list of ndarray
        Wavelet coefficients (output of ``pywt.wavedec``).
    n : int
        Length of the original signal.
    rule : {'universal', 'minimax', 'sqtwolog'}
        Threshold selection rule.

    Returns
    -------
    float
        Threshold value λ.
    """
    # Robust noise-level estimate (MAD of finest detail coefficients)
    sigma: float = float(np.median(np.abs(coeffs[-1]))) / 0.6745

    if rule == "universal":
        # Donoho-Johnstone universal threshold
        return sigma * np.sqrt(2.0 * np.log(n))
    elif rule == "sqtwolog":
        # √(2 log n) rule (same formula, different name in some literature)
        return sigma * np.sqrt(2.0 * np.log(n))
    elif rule == "minimax":
        # Minimax threshold (Donoho & Johnstone 1994)
        if n <= 32:
            return 0.0  # No thresholding for very short signals
        return sigma * (0.3936 + 0.1829 * np.log2(n))
    else:
        raise ValueError(f"Unknown threshold rule: {rule!r}")


# ===================================================================
# Core — wavelet denoising with configurable rule
# ===================================================================

def wavelet_denoise_parametric(
    signal_data: Signal,
    wavelet: str = "db4",
    threshold_mode: str = "soft",
    threshold_rule: str = "universal",
    level: Optional[int] = None,
) -> Signal:
    """Denoise *signal_data* using the DWT with full parametric control.

    Parameters
    ----------
    signal_data : Signal
        Input (noisy) signal.
    wavelet : str
        Wavelet family identifier (e.g. ``'db4'``, ``'sym8'``, ``'coif3'``).
    threshold_mode : str
        ``'soft'`` or ``'hard'`` thresholding.
    threshold_rule : str
        ``'universal'``, ``'minimax'``, or ``'sqtwolog'``.
    level : int, optional
        Decomposition level.  ``None`` → automatic.

    Returns
    -------
    Signal
        Denoised signal.
    """
    data = np.asarray(signal_data, dtype=np.float64)
    n = len(data)

    if level is None:
        level = min(pywt.dwt_max_level(n, wavelet), 6)

    coeffs = pywt.wavedec(data, wavelet, level=level)
    lam = _threshold_value(coeffs, n, threshold_rule)

    coeffs_t = [coeffs[0]]  # keep approximation
    for c in coeffs[1:]:
        coeffs_t.append(pywt.threshold(c, lam, mode=threshold_mode))

    rec: Signal = pywt.waverec(coeffs_t, wavelet)
    # Trim / pad to original length
    if len(rec) > n:
        rec = rec[:n]
    elif len(rec) < n:
        rec = np.pad(rec, (0, n - len(rec)), mode="edge")
    return rec


# ===================================================================
# Phase 1 — Monte-Carlo grid search
# ===================================================================

def monte_carlo_wavelet_optimization(
    reference_clean: Optional[Signal] = None,
    reference_noisy: Optional[Signal] = None,
    fs: float = 1e9,
    wavelet_families: Sequence[str] = ("db4", "sym8", "coif3"),
    threshold_modes: Sequence[str] = ("soft", "hard"),
    threshold_rules: Sequence[str] = ("universal", "minimax", "sqtwolog"),
    n_iterations: int = 1000,
    snr_range_db: Tuple[float, float] = (5.0, 25.0),
    epsilon: float = 1e-3,
    seed: Optional[int] = 42,
    verbose: bool = False,
) -> MonteCarloResult:
    """Stochastic optimisation of wavelet denoising parameters.

    Performs a full-factorial grid search over *wavelet_families* ×
    *threshold_modes* × *threshold_rules*.  For each grid point, **N**
    Monte-Carlo iterations inject AWGN at a random SNR drawn uniformly from
    ``snr_range_db`` into the reference signal and measure the RMSE of the
    reconstructed (denoised) signal against the clean reference.

    The selection criterion minimises **E[RMSE]** subject to
    **Var[RMSE] < ε**.

    Parameters
    ----------
    reference_clean : Signal, optional
        Clean reference UHF-PD signal.  If ``None`` a synthetic one is
        generated internally.
    reference_noisy : Signal, optional
        Ignored when *reference_clean* is given (noise is injected by MC).
    fs : float
        Sampling frequency in Hz.
    wavelet_families : sequence of str
        Wavelet identifiers to search.
    threshold_modes : sequence of str
        ``'soft'`` and/or ``'hard'``.
    threshold_rules : sequence of str
        ``'universal'``, ``'minimax'``, ``'sqtwolog'``.
    n_iterations : int
        Number of Monte-Carlo realisations per grid point.
    snr_range_db : tuple of float
        (min, max) SNR in dB for AWGN injection.
    epsilon : float
        Upper bound on acceptable RMSE variance.
    seed : int, optional
        RNG seed for reproducibility.
    verbose : bool
        Print progress information.

    Returns
    -------
    MonteCarloResult
        Aggregated result including the optimal configuration and the full
        grid of evaluated points.
    """
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Generate (or accept) a reference clean signal
    # ------------------------------------------------------------------
    if reference_clean is None:
        reference_clean, _ = generate_uhf_reference_signal(
            n_samples=4096, fs=fs, n_pulses=12, snr_db=40.0, seed=seed
        )

    n = len(reference_clean)
    sig_power: float = float(np.mean(reference_clean ** 2)) + 1e-30

    # ------------------------------------------------------------------
    # Build parameter grid
    # ------------------------------------------------------------------
    grid: List[WaveletGridPoint] = []
    combos = list(itertools.product(wavelet_families, threshold_modes, threshold_rules))

    for idx, (wv, tmode, trule) in enumerate(combos):
        gp = WaveletGridPoint(wavelet=wv, threshold_mode=tmode, threshold_rule=trule)

        if verbose:
            print(
                f"  [{idx + 1}/{len(combos)}] wavelet={wv}, "
                f"mode={tmode}, rule={trule} — running {n_iterations} MC iterations …"
            )

        rmse_samples: List[float] = []
        for _ in range(n_iterations):
            snr_db = rng.uniform(*snr_range_db)
            noise_power = sig_power / (10.0 ** (snr_db / 10.0))
            noise = rng.normal(0.0, np.sqrt(noise_power), n)
            noisy = reference_clean + noise

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                denoised = wavelet_denoise_parametric(
                    noisy, wavelet=wv, threshold_mode=tmode, threshold_rule=trule
                )

            rmse_samples.append(compute_rmse(reference_clean, denoised))

        gp.rmse_samples = rmse_samples
        gp.rmse_mean = float(np.mean(rmse_samples))
        gp.rmse_var = float(np.var(rmse_samples))
        grid.append(gp)

    # ------------------------------------------------------------------
    # Selection: minimise E[RMSE]  s.t.  Var[RMSE] < ε
    # ------------------------------------------------------------------
    feasible = [gp for gp in grid if gp.rmse_var < epsilon]
    converged = len(feasible) > 0

    if not converged:
        # Relax: choose from all grid points sorted by variance, then RMSE
        feasible = sorted(grid, key=lambda g: (g.rmse_var, g.rmse_mean))

    best = min(feasible, key=lambda g: g.rmse_mean)

    result = MonteCarloResult(
        best_wavelet=best.wavelet,
        best_threshold_mode=best.threshold_mode,
        best_threshold_rule=best.threshold_rule,
        best_rmse_mean=best.rmse_mean,
        best_rmse_var=best.rmse_var,
        grid=grid,
        n_iterations=n_iterations,
        epsilon=epsilon,
        converged=converged,
    )

    if verbose:
        tag = "✓ converged" if converged else "⚠ relaxed (no feasible point under ε)"
        print(
            f"\n  Optimal: wavelet={best.wavelet}, mode={best.threshold_mode}, "
            f"rule={best.threshold_rule}  |  E[RMSE]={best.rmse_mean:.6f}, "
            f"Var[RMSE]={best.rmse_var:.2e}  [{tag}]"
        )

    return result


# ===================================================================
# Original preprocessing functions (preserved, type-annotated)
# ===================================================================


def bandpass_filter(signal_data: Signal, fs: float, lowcut: float, highcut: float, order: int = 5) -> Signal:
    """
    Aplica un filtro pasabanda Butterworth a la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    fs : float
        Frecuencia de muestreo en Hz
    lowcut : float
        Frecuencia de corte inferior en Hz
    highcut : float
        Frecuencia de corte superior en Hz
    order : int, opcional
        Orden del filtro (por defecto 5)
    
    Retorna:
    --------
    filtered_signal : ndarray
        Señal filtrada
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Asegurar que las frecuencias estén en el rango válido
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    
    if low >= high:
        raise ValueError("La frecuencia de corte inferior debe ser menor que la superior")
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal: Signal = signal.filtfilt(b, a, signal_data)
    
    return filtered_signal


def normalize_signal(signal_data: Signal, method: str = 'zscore') -> Signal:
    """
    Normaliza la señal usando diferentes métodos.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    method : str, opcional
        Método de normalización: 'zscore', 'minmax', 'robust' (por defecto 'zscore')
    
    Retorna:
    --------
    normalized_signal : ndarray
        Señal normalizada
    """
    signal_data = np.asarray(signal_data)
    
    if method == 'zscore':
        # Normalización Z-score (media 0, desviación estándar 1)
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        if std == 0:
            return signal_data - mean
        normalized_signal = (signal_data - mean) / std
        
    elif method == 'minmax':
        # Normalización Min-Max [0, 1]
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        if max_val == min_val:
            return signal_data - min_val
        normalized_signal = (signal_data - min_val) / (max_val - min_val)
        
    elif method == 'robust':
        # Normalización robusta usando mediana y MAD
        median = np.median(signal_data)
        mad = np.median(np.abs(signal_data - median))
        if mad == 0:
            return signal_data - median
        normalized_signal = (signal_data - median) / (1.4826 * mad)
        
    else:
        raise ValueError(f"Método '{method}' no reconocido. Use 'zscore', 'minmax' o 'robust'")
    
    return normalized_signal


def get_envelope(signal_data: Signal) -> Signal:
    """
    Extrae la envolvente de la señal usando la transformada de Hilbert.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    envelope : ndarray
        Envolvente de la señal
    """
    analytic_signal = hilbert(signal_data)
    envelope = np.abs(analytic_signal)
    
    return envelope


def wavelet_denoise(
    signal_data: Signal,
    wavelet: str = 'db4',
    level: Optional[int] = None,
    threshold_method: str = 'soft',
) -> Signal:
    """
    Elimina ruido de la señal usando transformada wavelet.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    wavelet : str, opcional
        Tipo de wavelet a usar (por defecto 'db4')
    level : int, opcional
        Nivel de descomposición. Si es None, se calcula automáticamente
    threshold_method : str, opcional
        Método de umbralización: 'soft' o 'hard' (por defecto 'soft')
    
    Retorna:
    --------
    denoised_signal : ndarray
        Señal con ruido reducido
    """
    signal_data = np.asarray(signal_data)
    
    # Calcular nivel óptimo si no se proporciona
    if level is None:
        level = min(pywt.dwt_max_level(len(signal_data), wavelet), 6)
    
    # Descomposición wavelet
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    
    # Calcular umbral usando regla universal de Donoho-Johnstone
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
    
    # Aplicar umbralización a los coeficientes de detalle
    coeffs_thresholded = [coeffs[0]]  # Mantener coeficientes de aproximación
    for i in range(1, len(coeffs)):
        if threshold_method == 'soft':
            coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        else:
            coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
    
    # Reconstruir señal
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)
    
    # Ajustar longitud si es necesario
    if len(denoised_signal) > len(signal_data):
        denoised_signal = denoised_signal[:len(signal_data)]
    elif len(denoised_signal) < len(signal_data):
        denoised_signal = np.pad(denoised_signal, (0, len(signal_data) - len(denoised_signal)), 'edge')
    
    return denoised_signal


def adaptive_filter_lms(
    signal_data: Signal,
    reference: Optional[Signal] = None,
    mu: float = 0.01,
    filter_order: int = 32,
) -> Signal:
    """
    Filtro adaptativo LMS simple para eliminación de ruido.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada contaminada
    reference : array-like, opcional
        Señal de referencia de ruido. Si es None, se usa versión retardada de la señal
    mu : float, opcional
        Tasa de aprendizaje (por defecto 0.01)
    filter_order : int, opcional
        Orden del filtro adaptativo (por defecto 32)
    
    Retorna:
    --------
    filtered_signal : ndarray
        Señal filtrada
    """
    signal_data = np.asarray(signal_data)
    n = len(signal_data)
    
    # Si no hay señal de referencia, usar versión retardada
    if reference is None:
        reference = np.roll(signal_data, filter_order)
    
    # Inicializar pesos del filtro
    weights = np.zeros(filter_order)
    filtered_signal = np.zeros(n)
    
    # Algoritmo LMS
    for i in range(filter_order, n):
        # Vector de entrada
        x = reference[i-filter_order:i][::-1]
        
        # Salida del filtro
        y = np.dot(weights, x)
        
        # Error
        e = signal_data[i] - y
        
        # Actualizar pesos
        weights = weights + 2 * mu * e * x
        
        filtered_signal[i] = e
    
    return filtered_signal


def preprocess_signal(
    signal_data: Signal,
    fs: float,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
    normalize: bool = True,
    envelope: bool = True,
    denoise: bool = True,
) -> Tuple[Signal, Dict[str, Any]]:
    """
    Pipeline completo de preprocesamiento de señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada en bruto
    fs : float
        Frecuencia de muestreo en Hz
    lowcut : float, opcional
        Frecuencia de corte inferior del filtro pasabanda
    highcut : float, opcional
        Frecuencia de corte superior del filtro pasabanda
    normalize : bool, opcional
        Si aplicar normalización (por defecto True)
    envelope : bool, opcional
        Si extraer la envolvente (por defecto True)
    denoise : bool, opcional
        Si aplicar eliminación de ruido (por defecto True)
    
    Retorna:
    --------
    processed_signal : ndarray
        Señal procesada
    processing_info : dict
        Información sobre los pasos de procesamiento aplicados
    """
    signal_data = np.asarray(signal_data)
    processing_info = {}
    
    # Filtro pasabanda si se especifican frecuencias
    if lowcut is not None and highcut is not None:
        signal_data = bandpass_filter(signal_data, fs, lowcut, highcut)
        processing_info['bandpass_filter'] = {'lowcut': lowcut, 'highcut': highcut}
    
    # Normalización
    if normalize:
        signal_data = normalize_signal(signal_data, method='zscore')
        processing_info['normalized'] = True
    
    # Extracción de envolvente
    if envelope:
        signal_data = get_envelope(signal_data)
        processing_info['envelope'] = True
    
    # Eliminación de ruido
    if denoise:
        signal_data = wavelet_denoise(signal_data)
        processing_info['denoised'] = True
    
    return signal_data, processing_info
