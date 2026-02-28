"""
Módulo de validación del algoritmo de detección.

Phase 4 — Quantification:
    1. **Asymptotic time-complexity (Big-O)** of each Phase-3 algorithm
       (Kalman, adaptive EWMA, CUSUM) is measured empirically by timing
       execution across geometrically spaced input sizes and fitting the
       exponent of a power-law model t(n) = a·n^b.
    2. **Confusion matrix** contrasting *convergence latency* (number of
       samples required for the Kalman gain / EWMA α to stabilise within a
       tolerance) against the **false-positive rate** of each tracker,
       parameterised by stochastic variation of the event rate.
    3. Full PEP 484 type annotations throughout.

Legacy validation utilities (``calculate_confusion_matrix``,
``calculate_false_positive_rate``, etc.) are preserved below.
"""

from __future__ import annotations

import time
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
from scipy import stats
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Signal = NDArray[np.floating[Any]]
Labels = Union[NDArray[np.str_], NDArray[np.object_], List[str]]


# ===================================================================
# Phase-4 — §1  Asymptotic time-complexity measurement (Big-O)
# ===================================================================

@dataclass
class BigOEstimate:
    """Result of empirical Big-O estimation for a single algorithm."""

    algorithm_name: str
    exponent_b: float               # fitted exponent in t(n) = a * n^b
    coefficient_a: float            # fitted coefficient
    r_squared: float                # goodness of fit
    sizes: NDArray[np.int64]        # tested input sizes
    wall_times: NDArray[np.float64] # measured wall times (seconds)
    big_o_label: str                # human-readable, e.g. "O(n^1.01)"


def _power_law(n: NDArray[np.floating[Any]], a: float, b: float) -> NDArray[np.floating[Any]]:
    """Model function ``t(n) = a * n^b``."""
    return a * np.power(n.astype(np.float64), b)


def measure_algorithm_complexity(
    algorithm_fn: Callable[[Signal], Any],
    sizes: Sequence[int] = (256, 512, 1024, 2048, 4096, 8192, 16384),
    n_repeats: int = 5,
    seed: Optional[int] = 42,
    algorithm_name: str = "unknown",
) -> BigOEstimate:
    """Empirically estimate the Big-O complexity of *algorithm_fn*.

    For each size in *sizes* the function is called *n_repeats* times on
    random Δt vectors and the **median** wall-clock time is recorded.
    A power-law ``t(n) = a·n^b`` is then fitted in log-log space via
    non-linear least squares.

    Parameters
    ----------
    algorithm_fn : callable
        ``f(delta_t: Signal) -> Any``.  The return value is discarded.
    sizes : sequence of int
        Input sizes to benchmark.
    n_repeats : int
        Repetitions per size (takes median).
    seed : int, optional
        RNG seed.
    algorithm_name : str
        Label for the result.

    Returns
    -------
    BigOEstimate
    """
    rng = np.random.default_rng(seed)
    median_times: List[float] = []
    actual_sizes: List[int] = []

    for n in sizes:
        times: List[float] = []
        for _ in range(n_repeats):
            dt: Signal = rng.exponential(scale=1e-4, size=n).astype(np.float64)
            t0 = time.perf_counter()
            algorithm_fn(dt)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        median_times.append(float(np.median(times)))
        actual_sizes.append(n)

    s_arr = np.array(actual_sizes, dtype=np.float64)
    t_arr = np.array(median_times, dtype=np.float64)

    # Fit power law in log-log space
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            popt, _ = curve_fit(
                _power_law,
                s_arr,
                t_arr,
                p0=[1e-8, 1.0],
                maxfev=10000,
            )
            a_fit, b_fit = float(popt[0]), float(popt[1])
        except RuntimeError:
            # Fallback: linear regression in log-log
            log_s = np.log(s_arr)
            log_t = np.log(t_arr + 1e-30)
            slope, intercept, _, _, _ = stats.linregress(log_s, log_t)
            b_fit = float(slope)
            a_fit = float(np.exp(intercept))

    # R² in original space
    t_pred = _power_law(s_arr, a_fit, b_fit)
    ss_res = float(np.sum((t_arr - t_pred) ** 2))
    ss_tot = float(np.sum((t_arr - np.mean(t_arr)) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot

    label = f"O(n^{b_fit:.2f})"

    return BigOEstimate(
        algorithm_name=algorithm_name,
        exponent_b=b_fit,
        coefficient_a=a_fit,
        r_squared=r2,
        sizes=np.array(actual_sizes, dtype=np.int64),
        wall_times=t_arr,
        big_o_label=label,
    )


def measure_all_tracking_complexities(
    sizes: Sequence[int] = (256, 512, 1024, 2048, 4096, 8192, 16384),
    n_repeats: int = 5,
    seed: Optional[int] = 42,
) -> Dict[str, BigOEstimate]:
    """Measure Big-O for every Phase-3 algorithm.

    Returns
    -------
    dict
        ``{'Kalman': BigOEstimate, 'AdaptiveEWMA': …, 'CUSUM': …}``
    """
    # Import here to avoid circular dependency at module level
    from blind_algorithms import (
        KalmanDeltaTTracker,
        AdaptiveEWMATracker,
        CUSUMDetector,
    )

    kalman = KalmanDeltaTTracker()
    ewma = AdaptiveEWMATracker()
    cusum = CUSUMDetector()

    results: Dict[str, BigOEstimate] = {}

    results["Kalman"] = measure_algorithm_complexity(
        lambda dt: kalman.track(dt),
        sizes=sizes,
        n_repeats=n_repeats,
        seed=seed,
        algorithm_name="Kalman",
    )
    results["AdaptiveEWMA"] = measure_algorithm_complexity(
        lambda dt: ewma.track(dt),
        sizes=sizes,
        n_repeats=n_repeats,
        seed=seed,
        algorithm_name="AdaptiveEWMA",
    )
    results["CUSUM"] = measure_algorithm_complexity(
        lambda dt: cusum.detect(dt),
        sizes=sizes,
        n_repeats=n_repeats,
        seed=seed,
        algorithm_name="CUSUM",
    )

    return results


# ===================================================================
# Phase-4 — §2  Convergence latency & FPR confusion matrix
# ===================================================================

@dataclass
class ConvergenceMetrics:
    """Per-algorithm convergence latency and false-positive rate."""

    algorithm_name: str
    convergence_latency: int        # samples to converge
    false_positive_rate: float
    true_positive_rate: float
    n_alarms: int
    total_samples: int


@dataclass
class ConvergenceConfusionMatrix:
    """Confusion-matrix–style comparison across algorithms and event-rate
    variation levels."""

    algorithms: List[str]
    variation_levels: NDArray[np.float64]
    latency_matrix: NDArray[np.float64]     # shape (n_algos, n_var_levels)
    fpr_matrix: NDArray[np.float64]         # shape (n_algos, n_var_levels)
    tpr_matrix: NDArray[np.float64]         # shape (n_algos, n_var_levels)
    raw_metrics: List[List[ConvergenceMetrics]]


def _estimate_convergence_latency_kalman(
    delta_t: Signal,
    Q: float = 1e-5,
    R: float = 1e-2,
    tol: float = 0.01,
) -> int:
    """Convergence latency = first index k where |K[k] − K_∞| < tol."""
    from blind_algorithms import KalmanDeltaTTracker

    result = KalmanDeltaTTracker(process_variance=Q, measurement_variance=R).track(delta_t)
    K_inf = result.steady_state_gain
    for k, g in enumerate(result.kalman_gains):
        if abs(g - K_inf) < tol:
            return k
    return len(delta_t)


def _estimate_convergence_latency_ewma(
    delta_t: Signal,
    alpha_0: float = 0.2,
    var_window: int = 10,
    tol: float = 0.02,
) -> int:
    """Convergence = first index where |α[k] − median(α)| < tol."""
    from blind_algorithms import AdaptiveEWMATracker

    result = AdaptiveEWMATracker(alpha_0=alpha_0, variance_window=var_window).track(delta_t)
    alpha_med = float(np.median(result.alpha_sequence))
    for k, a in enumerate(result.alpha_sequence):
        if abs(a - alpha_med) < tol:
            return k
    return len(delta_t)


def _estimate_convergence_latency_cusum(
    delta_t: Signal,
    threshold: float = 5.0,
    drift: float = 0.5,
) -> int:
    """Convergence = first alarm index (or len if no alarm)."""
    from blind_algorithms import CUSUMDetector

    result = CUSUMDetector(threshold=threshold, drift=drift).detect(delta_t)
    if result.n_alarms > 0:
        return int(result.alarm_indices[0])
    return len(delta_t)


def _compute_fpr_cusum(
    delta_t_stationary: Signal,
    cusum_threshold: float = 5.0,
    cusum_drift: float = 0.5,
) -> Tuple[float, int]:
    """FPR for CUSUM on a known-stationary series = alarms / total."""
    from blind_algorithms import CUSUMDetector

    result = CUSUMDetector(threshold=cusum_threshold, drift=cusum_drift).detect(delta_t_stationary)
    n = len(delta_t_stationary)
    fpr: float = result.n_alarms / max(1, n)
    return fpr, result.n_alarms


def _compute_fpr_tracker(
    delta_t_stationary: Signal,
    tracker_fn: Callable[[Signal], Any],
    residual_key: str,
    sigma_multiplier: float = 3.0,
) -> Tuple[float, int]:
    """FPR for Kalman / EWMA: fraction of residuals exceeding ±3σ on stationary data."""
    result = tracker_fn(delta_t_stationary)
    residuals: Signal = getattr(result, residual_key)
    mu = float(np.mean(residuals))
    sigma = float(np.std(residuals)) + 1e-30
    alarms = np.abs(residuals - mu) > sigma_multiplier * sigma
    n_alarms = int(np.sum(alarms))
    fpr: float = n_alarms / max(1, len(residuals))
    return fpr, n_alarms


def generate_convergence_confusion_matrix(
    base_rate: float = 1e-4,
    n_samples: int = 2000,
    variation_levels: Sequence[float] = (0.0, 0.1, 0.25, 0.5, 1.0),
    n_monte_carlo: int = 50,
    seed: Optional[int] = 42,
    cusum_threshold: float = 5.0,
    cusum_drift: float = 0.5,
) -> ConvergenceConfusionMatrix:
    """Build a confusion matrix of convergence latency vs FPR.

    For each *variation_level* (coefficient of variation of the
    inter-event time), a Monte-Carlo ensemble of Δt vectors is generated.
    The three Phase-3 algorithms are applied and their **convergence
    latency** and **false-positive rate** are averaged.

    Parameters
    ----------
    base_rate : float
        Mean inter-event time (seconds) for the exponential Δt generator.
    n_samples : int
        Length of each synthetic Δt vector.
    variation_levels : sequence of float
        Coefficients of variation σ/μ of the Gamma-distributed Δt.
    n_monte_carlo : int
        Number of MC realisations per variation level.
    seed : int, optional
        RNG seed.
    cusum_threshold, cusum_drift : float
        CUSUM detector parameters.

    Returns
    -------
    ConvergenceConfusionMatrix
    """
    from blind_algorithms import (
        KalmanDeltaTTracker,
        AdaptiveEWMATracker,
        CUSUMDetector,
    )

    rng = np.random.default_rng(seed)
    algo_names = ["Kalman", "AdaptiveEWMA", "CUSUM"]
    n_algos = len(algo_names)
    n_var = len(variation_levels)

    lat_accum = np.zeros((n_algos, n_var), dtype=np.float64)
    fpr_accum = np.zeros((n_algos, n_var), dtype=np.float64)
    tpr_accum = np.zeros((n_algos, n_var), dtype=np.float64)
    raw_all: List[List[ConvergenceMetrics]] = [[] for _ in range(n_var)]

    for vi, cv in enumerate(variation_levels):
        for _ in range(n_monte_carlo):
            # Generate Δt from a Gamma distribution with mean = base_rate
            # and coefficient of variation = cv
            if cv <= 0.0:
                delta_t = np.full(n_samples, base_rate, dtype=np.float64)
            else:
                shape = 1.0 / (cv ** 2)
                scale = base_rate / shape
                delta_t = rng.gamma(shape, scale, size=n_samples).astype(np.float64)

            # --- Kalman ---
            lat_k = _estimate_convergence_latency_kalman(delta_t)
            fpr_k, n_alarms_k = _compute_fpr_tracker(
                delta_t,
                KalmanDeltaTTracker().track,
                "residuals",
            )

            # --- Adaptive EWMA ---
            lat_e = _estimate_convergence_latency_ewma(delta_t)
            fpr_e, n_alarms_e = _compute_fpr_tracker(
                delta_t,
                AdaptiveEWMATracker().track,
                "residuals",
            )

            # --- CUSUM ---
            lat_c = _estimate_convergence_latency_cusum(
                delta_t, threshold=cusum_threshold, drift=cusum_drift
            )
            fpr_c, n_alarms_c = _compute_fpr_cusum(
                delta_t, cusum_threshold=cusum_threshold, cusum_drift=cusum_drift
            )

            lat_accum[0, vi] += lat_k
            lat_accum[1, vi] += lat_e
            lat_accum[2, vi] += lat_c
            fpr_accum[0, vi] += fpr_k
            fpr_accum[1, vi] += fpr_e
            fpr_accum[2, vi] += fpr_c

            metrics_list = [
                ConvergenceMetrics("Kalman", lat_k, fpr_k, 1.0 - fpr_k, n_alarms_k, n_samples),
                ConvergenceMetrics("AdaptiveEWMA", lat_e, fpr_e, 1.0 - fpr_e, n_alarms_e, n_samples),
                ConvergenceMetrics("CUSUM", lat_c, fpr_c, 1.0 - fpr_c, n_alarms_c, n_samples),
            ]
            raw_all[vi].extend(metrics_list)

    # Average
    lat_accum /= n_monte_carlo
    fpr_accum /= n_monte_carlo
    tpr_accum = 1.0 - fpr_accum

    return ConvergenceConfusionMatrix(
        algorithms=algo_names,
        variation_levels=np.array(variation_levels, dtype=np.float64),
        latency_matrix=lat_accum,
        fpr_matrix=fpr_accum,
        tpr_matrix=tpr_accum,
        raw_metrics=raw_all,
    )


# -------------------------------------------------------------------
# Formatted report for Phase-4 results
# -------------------------------------------------------------------

def generate_phase4_report(
    complexity: Dict[str, BigOEstimate],
    confusion: ConvergenceConfusionMatrix,
) -> str:
    """Human-readable report of Phase-4 quantification results.

    Parameters
    ----------
    complexity : dict
        Output of :func:`measure_all_tracking_complexities`.
    confusion : ConvergenceConfusionMatrix
        Output of :func:`generate_convergence_confusion_matrix`.

    Returns
    -------
    str
        Multi-line formatted report.
    """
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("  PHASE 4 — QUANTIFICATION REPORT (Doctoral Validation Framework)")
    lines.append("=" * 78)
    lines.append("")

    # §1 — Big-O
    lines.append("§1  ASYMPTOTIC TIME-COMPLEXITY (empirical Big-O)")
    lines.append("-" * 78)
    for name, est in complexity.items():
        lines.append(
            f"  {name:<18s}  {est.big_o_label:<14s}  "
            f"b={est.exponent_b:.4f}  a={est.coefficient_a:.2e}  R²={est.r_squared:.4f}"
        )
    lines.append("")

    # §2 — Confusion matrix
    lines.append("§2  CONVERGENCE-LATENCY vs FALSE-POSITIVE RATE CONFUSION MATRIX")
    lines.append("-" * 78)

    # Header
    hdr = f"  {'CV(Δt)':<10s}"
    for alg in confusion.algorithms:
        hdr += f" | {'Lat':>6s}  {'FPR':>7s}  {'TPR':>7s}"
    lines.append(hdr)
    lines.append("  " + "-" * (10 + len(confusion.algorithms) * 28))

    for vi, cv in enumerate(confusion.variation_levels):
        row = f"  {cv:<10.2f}"
        for ai in range(len(confusion.algorithms)):
            lat = confusion.latency_matrix[ai, vi]
            fpr = confusion.fpr_matrix[ai, vi]
            tpr = confusion.tpr_matrix[ai, vi]
            row += f" | {lat:6.1f}  {fpr:7.4f}  {tpr:7.4f}"
        lines.append(row)

    lines.append("")
    lines.append("  CV(Δt) = coefficient of variation of the stochastic event rate")
    lines.append("  Lat    = mean convergence latency (samples)")
    lines.append("  FPR    = mean false-positive rate on stationary data")
    lines.append("  TPR    = 1 − FPR (true-positive complement)")
    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


# ===================================================================
# Legacy validation utilities (preserved, PEP 484–annotated)
# ===================================================================


def calculate_confusion_matrix(
    true_labels: Labels,
    predicted_labels: Labels,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calcula la matriz de confusión.
    
    Parámetros:
    -----------
    true_labels : array-like
        Etiquetas verdaderas
    predicted_labels : array-like
        Etiquetas predichas
    classes : list, opcional
        Lista de clases. Si es None, se infiere de los datos
    
    Retorna:
    --------
    confusion_matrix : dict
        Matriz de confusión estructurada
    """
    if classes is None:
        classes = sorted(set(true_labels) | set(predicted_labels))
    
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for true, pred in zip(true_labels, predicted_labels):
        i = class_to_idx[true]
        j = class_to_idx[pred]
        matrix[i, j] += 1
    
    confusion_matrix = {
        'matrix': matrix,
        'classes': classes
    }
    
    return confusion_matrix


def calculate_false_positive_rate(
    true_labels: Labels,
    predicted_labels: Labels,
    positive_class: str = 'rojo',
) -> float:
    """
    Calcula la tasa de falsos positivos.
    
    Parámetros:
    -----------
    true_labels : array-like
        Etiquetas verdaderas
    predicted_labels : array-like
        Etiquetas predichas
    positive_class : str, opcional
        Clase considerada como positiva (por defecto 'rojo')
    
    Retorna:
    --------
    fpr : float
        Tasa de falsos positivos
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    
    # Verdaderos negativos (no son clase positiva y no se predicen como tal)
    true_negative = np.sum((true_labels != positive_class) & (predicted_labels != positive_class))
    
    # Falsos positivos (no son clase positiva pero se predicen como tal)
    false_positive = np.sum((true_labels != positive_class) & (predicted_labels == positive_class))
    
    if (true_negative + false_positive) == 0:
        return 0.0
    
    fpr = false_positive / (false_positive + true_negative)
    
    return fpr


def calculate_false_negative_rate(
    true_labels: Labels,
    predicted_labels: Labels,
    positive_class: str = 'rojo',
) -> float:
    """
    Calcula la tasa de falsos negativos.
    
    Parámetros:
    -----------
    true_labels : array-like
        Etiquetas verdaderas
    predicted_labels : array-like
        Etiquetas predichas
    positive_class : str, opcional
        Clase considerada como positiva (por defecto 'rojo')
    
    Retorna:
    --------
    fnr : float
        Tasa de falsos negativos
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    
    # Verdaderos positivos (son clase positiva y se predicen como tal)
    true_positive = np.sum((true_labels == positive_class) & (predicted_labels == positive_class))
    
    # Falsos negativos (son clase positiva pero no se predicen como tal)
    false_negative = np.sum((true_labels == positive_class) & (predicted_labels != positive_class))
    
    if (true_positive + false_negative) == 0:
        return 0.0
    
    fnr = false_negative / (true_positive + false_negative)
    
    return fnr


def calculate_accuracy(
    true_labels: Labels,
    predicted_labels: Labels,
) -> float:
    """
    Calcula la precisión de clasificación.
    
    Parámetros:
    -----------
    true_labels : array-like
        Etiquetas verdaderas
    predicted_labels : array-like
        Etiquetas predichas
    
    Retorna:
    --------
    accuracy : float
        Precisión (proporción de predicciones correctas)
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    
    correct = np.sum(true_labels == predicted_labels)
    total = len(true_labels)
    
    if total == 0:
        return 0.0
    
    accuracy = correct / total
    
    return accuracy


def calculate_class_separation(
    descriptor_values_by_class: Dict[str, List[float]],
) -> Dict[str, Any]:
    """
    Calcula la separación entre clases usando análisis discriminante.
    
    Parámetros:
    -----------
    descriptor_values_by_class : dict
        Diccionario {clase: [valores]}
    
    Retorna:
    --------
    separation_metrics : dict
        Métricas de separación entre clases
    """
    classes = list(descriptor_values_by_class.keys())
    
    # Calcular medias por clase
    means = {cls: np.mean(values) for cls, values in descriptor_values_by_class.items()}
    
    # Calcular varianzas por clase
    variances = {cls: np.var(values) for cls, values in descriptor_values_by_class.items()}
    
    # Separación entre clases adyacentes
    separations = {}
    for i in range(len(classes) - 1):
        cls1 = classes[i]
        cls2 = classes[i + 1]
        
        mean_diff = abs(means[cls2] - means[cls1])
        pooled_std = np.sqrt((variances[cls1] + variances[cls2]) / 2)
        
        if pooled_std > 0:
            # Distancia de Cohen (effect size)
            cohen_d = mean_diff / pooled_std
        else:
            cohen_d = 0
        
        separations[f'{cls1}_vs_{cls2}'] = cohen_d
    
    # Separación global (varianza entre clases / varianza dentro de clases)
    all_values = []
    class_labels = []
    for cls, values in descriptor_values_by_class.items():
        all_values.extend(values)
        class_labels.extend([cls] * len(values))
    
    if len(all_values) > 0:
        # Varianza total
        total_variance = np.var(all_values)
        
        # Varianza dentro de clases (promedio ponderado)
        within_class_variance = np.mean(list(variances.values()))
        
        if within_class_variance > 0:
            f_ratio = (total_variance - within_class_variance) / within_class_variance
        else:
            f_ratio = 0
    else:
        f_ratio = 0
    
    separation_metrics = {
        'pairwise_separations': separations,
        'f_ratio': f_ratio,
        'class_means': means,
        'class_variances': variances
    }
    
    return separation_metrics


def calculate_threshold_stability(
    severity_indices: Signal,
    window_size: int = 10,
) -> Dict[str, float]:
    """
    Calcula la estabilidad del umbral de decisión.
    
    Parámetros:
    -----------
    severity_indices : array-like
        Serie temporal de índices de severidad
    window_size : int, opcional
        Tamaño de ventana para calcular variación (por defecto 10)
    
    Retorna:
    --------
    stability_metrics : dict
        Métricas de estabilidad
    """
    severity_indices = np.asarray(severity_indices)
    
    # Variación local (desviación estándar en ventanas)
    n = len(severity_indices)
    local_stds = []
    
    for i in range(0, n - window_size + 1, window_size // 2):
        window = severity_indices[i:i+window_size]
        local_stds.append(np.std(window))
    
    if len(local_stds) > 0:
        avg_local_std = np.mean(local_stds)
        max_local_std = np.max(local_stds)
    else:
        avg_local_std = 0
        max_local_std = 0
    
    # Coeficiente de variación global
    mean_severity = np.mean(severity_indices)
    if mean_severity > 0:
        cv = np.std(severity_indices) / mean_severity
    else:
        cv = 0
    
    # Número de cambios de tendencia
    if len(severity_indices) > 1:
        diff = np.diff(severity_indices)
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
    else:
        sign_changes = 0
    
    stability_metrics = {
        'avg_local_std': avg_local_std,
        'max_local_std': max_local_std,
        'coefficient_of_variation': cv,
        'trend_changes': sign_changes,
        'stability_score': 1.0 / (1.0 + cv)  # Mayor score = más estable
    }
    
    return stability_metrics


def calculate_effective_snr(
    signal_data: Signal,
    noise_estimate: Optional[Signal] = None,
) -> float:
    """
    Calcula el SNR efectivo de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    noise_estimate : array-like, opcional
        Estimación del ruido. Si es None, se estima de la señal
    
    Retorna:
    --------
    snr_db : float
        SNR en decibelios
    """
    signal_data = np.asarray(signal_data)
    
    if noise_estimate is None:
        # Estimar ruido usando diferencias de alto orden
        noise_estimate = np.diff(signal_data, n=2)
    
    signal_power = np.mean(signal_data**2)
    noise_power = np.mean(noise_estimate**2)
    
    if noise_power == 0:
        return np.inf
    
    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)
    
    return snr_db


def calculate_descriptor_variation_by_state(
    descriptors_by_state: Dict[str, List[Dict[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Calcula la variación de descriptores por estado operativo.
    
    Parámetros:
    -----------
    descriptors_by_state : dict
        Diccionario {estado: [lista de diccionarios de descriptores]}
    
    Retorna:
    --------
    variation_metrics : dict
        Métricas de variación por descriptor y estado
    """
    variation_metrics = {}
    
    # Recopilar todos los descriptores únicos
    all_descriptors = set()
    for state, desc_list in descriptors_by_state.items():
        for desc_dict in desc_list:
            all_descriptors.update(desc_dict.keys())
    
    # Calcular variación para cada descriptor
    for descriptor in all_descriptors:
        variation_metrics[descriptor] = {}
        
        for state, desc_list in descriptors_by_state.items():
            values = [d[descriptor] for d in desc_list if descriptor in d]
            
            if len(values) > 0:
                variation_metrics[descriptor][state] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / (np.mean(values) + 1e-10),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values)
                }
    
    return variation_metrics


def validate_detection_system(
    true_labels: Labels,
    predicted_labels: Labels,
    severity_indices: Signal,
    descriptors_by_state: Optional[Dict[str, List[Dict[str, float]]]] = None,
    signal_data: Optional[Signal] = None,
) -> Dict[str, Any]:
    """
    Validación completa del sistema de detección.
    
    Parámetros:
    -----------
    true_labels : array-like
        Etiquetas verdaderas
    predicted_labels : array-like
        Etiquetas predichas por el sistema
    severity_indices : array-like
        Índices de severidad calculados
    descriptors_by_state : dict, opcional
        Descriptores agrupados por estado
    signal_data : array-like, opcional
        Señal original para cálculo de SNR
    
    Retorna:
    --------
    validation_results : dict
        Resultados completos de validación
    """
    validation_results = {}
    
    # Métricas de clasificación
    validation_results['accuracy'] = calculate_accuracy(true_labels, predicted_labels)
    validation_results['false_positive_rate'] = calculate_false_positive_rate(
        true_labels, predicted_labels, positive_class='rojo'
    )
    validation_results['false_negative_rate'] = calculate_false_negative_rate(
        true_labels, predicted_labels, positive_class='rojo'
    )
    
    # Matriz de confusión
    validation_results['confusion_matrix'] = calculate_confusion_matrix(
        true_labels, predicted_labels
    )
    
    # Estabilidad del umbral
    validation_results['threshold_stability'] = calculate_threshold_stability(severity_indices)
    
    # Separación entre clases
    if descriptors_by_state is not None:
        # Usar el primer descriptor como ejemplo
        descriptor_values_by_class = {}
        first_descriptor = None
        
        for state, desc_list in descriptors_by_state.items():
            if len(desc_list) > 0 and first_descriptor is None:
                first_descriptor = list(desc_list[0].keys())[0]
            
            if first_descriptor:
                descriptor_values_by_class[state] = [
                    d[first_descriptor] for d in desc_list if first_descriptor in d
                ]
        
        if descriptor_values_by_class:
            validation_results['class_separation'] = calculate_class_separation(
                descriptor_values_by_class
            )
        
        # Variación de descriptores por estado
        validation_results['descriptor_variation'] = calculate_descriptor_variation_by_state(
            descriptors_by_state
        )
    
    # SNR efectivo
    if signal_data is not None:
        validation_results['effective_snr_db'] = calculate_effective_snr(signal_data)
    
    return validation_results


def generate_validation_report(validation_results: Dict[str, Any]) -> str:
    """
    Genera un reporte legible de los resultados de validación.
    
    Parámetros:
    -----------
    validation_results : dict
        Resultados de validación
    
    Retorna:
    --------
    report : str
        Reporte formateado
    """
    report = []
    report.append("=" * 70)
    report.append("REPORTE DE VALIDACIÓN DEL SISTEMA DE DETECCIÓN")
    report.append("=" * 70)
    report.append("")
    
    # Métricas de clasificación
    report.append("MÉTRICAS DE CLASIFICACIÓN:")
    report.append("-" * 70)
    report.append(f"  Precisión (Accuracy):        {validation_results['accuracy']:.4f}")
    report.append(f"  Tasa de Falsos Positivos:    {validation_results['false_positive_rate']:.4f}")
    report.append(f"  Tasa de Falsos Negativos:    {validation_results['false_negative_rate']:.4f}")
    report.append("")
    
    # Estabilidad del umbral
    if 'threshold_stability' in validation_results:
        stab = validation_results['threshold_stability']
        report.append("ESTABILIDAD DEL UMBRAL:")
        report.append("-" * 70)
        report.append(f"  Puntuación de Estabilidad:   {stab['stability_score']:.4f}")
        report.append(f"  Coef. de Variación:          {stab['coefficient_of_variation']:.4f}")
        report.append(f"  Cambios de Tendencia:        {stab['trend_changes']}")
        report.append("")
    
    # Separación entre clases
    if 'class_separation' in validation_results:
        sep = validation_results['class_separation']
        report.append("SEPARACIÓN ENTRE CLASES:")
        report.append("-" * 70)
        report.append(f"  F-ratio:                     {sep['f_ratio']:.4f}")
        report.append("  Separaciones por pares:")
        for pair, cohen_d in sep['pairwise_separations'].items():
            report.append(f"    {pair:30s} {cohen_d:.4f}")
        report.append("")
    
    # SNR efectivo
    if 'effective_snr_db' in validation_results:
        report.append("RELACIÓN SEÑAL-RUIDO:")
        report.append("-" * 70)
        report.append(f"  SNR Efectivo:                {validation_results['effective_snr_db']:.2f} dB")
        report.append("")
    
    report.append("=" * 70)
    
    return "\n".join(report)
