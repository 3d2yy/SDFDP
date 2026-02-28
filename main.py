"""
Main entry point — Doctoral UHF-PD validation pipeline.

Executes the four-phase numerical workflow:
    Phase 1: Stochastic wavelet optimisation (Monte Carlo + grid search)
    Phase 2: Variable isolation via inter-pulse interval extraction (Δt)
    Phase 3: Tracking with Kalman, adaptive EWMA, and CUSUM
    Phase 4: Quantification (empirical Big-O + convergence/FPR confusion matrix)
"""

from __future__ import annotations

import sys
import numpy as np

from preprocessing import (
    generate_uhf_pd_signal_physical,
    monte_carlo_wavelet_optimization,
    wavelet_denoise_parametric,
)
from descriptors import extract_delta_t_vector
from blind_algorithms import apply_delta_t_tracking
from validation import (
    measure_all_tracking_complexities,
    generate_convergence_confusion_matrix,
    generate_phase4_report,
)


# ===================================================================
# Pipeline helpers
# ===================================================================

def run_phase1(
    n_samples: int = 4096,
    fs: float = 1e9,
    n_iterations: int = 500,
    seed: int = 42,
    verbose: bool = True,
):
    """Phase 1 — Stochastic wavelet optimisation.

    Returns
    -------
    mc_result : MonteCarloResult
        Optimal wavelet configuration and full grid.
    clean : ndarray
        Clean reference signal used for optimisation.
    noisy : ndarray
        Noisy copy used for optimisation.
    """
    if verbose:
        print("=" * 70)
        print("PHASE 1 — Stochastic Wavelet Optimisation")
        print("=" * 70)

    clean, noisy = generate_uhf_pd_signal_physical(
        n_samples=n_samples, fs=fs, seed=seed,
    )

    mc_result = monte_carlo_wavelet_optimization(
        reference_clean=clean,
        fs=fs,
        n_iterations=n_iterations,
        seed=seed,
        verbose=verbose,
    )

    if verbose:
        print(f"\n  Optimal config: wavelet={mc_result.best_wavelet}, "
              f"mode={mc_result.best_threshold_mode}, "
              f"rule={mc_result.best_threshold_rule}")
        print(f"  E[RMSE]={mc_result.best_rmse_mean:.6f}, "
              f"Var[RMSE]={mc_result.best_rmse_var:.2e}, "
              f"converged={mc_result.converged}")

    return mc_result, clean, noisy


def run_phase2(
    noisy_signal,
    fs: float = 1e9,
    mc_result=None,
    threshold_sigma: float = 3.0,
    verbose: bool = True,
):
    """Phase 2 — Δt vector extraction (variable isolation).

    Parameters
    ----------
    noisy_signal : ndarray
        Signal to process.
    fs : float
        Sampling frequency.
    mc_result : MonteCarloResult, optional
        If provided, applies the optimal wavelet denoising before extraction.
    threshold_sigma : float
        Pulse detection threshold in multiples of σ.

    Returns
    -------
    delta_t : ndarray
        1-D inter-pulse interval vector [Δt₁, Δt₂, …] in seconds.
    denoised : ndarray
        Denoised signal (if mc_result provided) or original.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 2 — Δt Vector Extraction (Variable Isolation)")
        print("=" * 70)

    # Apply optimal wavelet denoising from Phase 1
    if mc_result is not None:
        denoised = wavelet_denoise_parametric(
            noisy_signal,
            wavelet=mc_result.best_wavelet,
            threshold_mode=mc_result.best_threshold_mode,
            threshold_rule=mc_result.best_threshold_rule,
        )
        if verbose:
            print(f"  Denoised with {mc_result.best_wavelet} / "
                  f"{mc_result.best_threshold_mode} / "
                  f"{mc_result.best_threshold_rule}")
    else:
        denoised = noisy_signal

    delta_t = extract_delta_t_vector(
        denoised, fs, threshold_sigma=threshold_sigma,
    )

    if verbose:
        print(f"  Extracted Δt vector: {len(delta_t)} intervals")
        if len(delta_t) > 0:
            print(f"  Δt statistics: mean={np.mean(delta_t):.4e} s, "
                  f"std={np.std(delta_t):.4e} s, "
                  f"min={np.min(delta_t):.4e} s, "
                  f"max={np.max(delta_t):.4e} s")

    return delta_t, denoised


def run_phase3(delta_t, verbose: bool = True):
    """Phase 3 — Tracking evaluation (Kalman, EWMA, CUSUM).

    Returns
    -------
    tracking_result : DeltaTTrackingResult
        Aggregated tracking results from all three algorithms.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 3 — Δt Tracking (Kalman / EWMA / CUSUM)")
        print("=" * 70)

    tracking_result = apply_delta_t_tracking(delta_t)

    if verbose:
        print(f"  Kalman: steady-state gain = "
              f"{tracking_result.kalman.steady_state_gain:.6f}")
        print(f"  EWMA: final α = "
              f"{tracking_result.ewma.alpha_sequence[-1]:.4f}")
        print(f"  CUSUM: {tracking_result.cusum.n_alarms} alarms "
              f"(threshold={tracking_result.cusum.threshold:.2f})")

    return tracking_result


def run_phase4(
    sizes=(256, 512, 1024, 2048, 4096, 8192),
    n_repeats: int = 5,
    seed: int = 42,
    verbose: bool = True,
):
    """Phase 4 — Quantification (Big-O + convergence/FPR).

    Returns
    -------
    complexity : dict
        Per-algorithm Big-O estimates.
    confusion : ConvergenceConfusionMatrix
        Convergence-latency vs FPR matrix.
    report : str
        Human-readable Phase 4 report.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 4 — Asymptotic Quantification")
        print("=" * 70)

    if verbose:
        print("  Measuring empirical Big-O complexities …")
    complexity = measure_all_tracking_complexities(
        sizes=sizes, n_repeats=n_repeats, seed=seed,
    )

    for name, est in complexity.items():
        if verbose:
            print(f"    {name}: O(n^{est.exponent_b:.2f})  "
                  f"[R²={est.r_squared:.4f}]")

    if verbose:
        print("  Building convergence/FPR confusion matrix …")
    confusion = generate_convergence_confusion_matrix(seed=seed)

    report = generate_phase4_report(complexity, confusion)

    if verbose:
        print("\n" + report)

    return complexity, confusion, report


# ===================================================================
# Main
# ===================================================================

def main(
    n_samples: int = 4096,
    fs: float = 1e9,
    mc_iterations: int = 500,
    seed: int = 42,
    verbose: bool = True,
):
    """Execute the full four-phase doctoral pipeline.

    Returns
    -------
    results : dict
        Dictionary containing outputs from all four phases.
    """
    # Phase 1 — Wavelet optimisation
    mc_result, clean, noisy = run_phase1(
        n_samples=n_samples, fs=fs, n_iterations=mc_iterations,
        seed=seed, verbose=verbose,
    )

    # Phase 2 — Δt extraction
    delta_t, denoised = run_phase2(
        noisy, fs=fs, mc_result=mc_result, verbose=verbose,
    )

    if len(delta_t) < 3:
        print("\n⚠ Fewer than 3 Δt intervals detected. "
              "Increase n_samples or n_pulses for meaningful tracking.")
        return {
            "phase1_mc_result": mc_result,
            "phase2_delta_t": delta_t,
            "phase2_denoised": denoised,
        }

    # Phase 3 — Tracking
    tracking = run_phase3(delta_t, verbose=verbose)

    # Phase 4 — Quantification
    complexity, confusion, report = run_phase4(seed=seed, verbose=verbose)

    results = {
        "phase1_mc_result": mc_result,
        "phase1_clean": clean,
        "phase1_noisy": noisy,
        "phase2_delta_t": delta_t,
        "phase2_denoised": denoised,
        "phase3_tracking": tracking,
        "phase4_complexity": complexity,
        "phase4_confusion": confusion,
        "phase4_report": report,
    }

    if verbose:
        print("\n" + "=" * 70)
        print("✓ PIPELINE COMPLETE — All four phases executed successfully.")
        print("=" * 70)

    return results


if __name__ == "__main__":
    results = main()
