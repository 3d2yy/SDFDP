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

