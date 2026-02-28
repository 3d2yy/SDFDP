"""
Quick system test â€” Doctoral UHF-PD validation pipeline.

Verifies all four phases execute without errors:
    Phase 1: Stochastic wavelet optimisation
    Phase 2: Î”t vector extraction
    Phase 3: Tracking (Kalman, EWMA, CUSUM)
    Phase 4: Asymptotic quantification (Big-O + convergence/FPR)
"""

import sys
import numpy as np

print("=" * 70)
print("SDFDP â€” DOCTORAL PIPELINE SYSTEM TEST")
print("=" * 70)
print()

# -----------------------------------------------------------------------
# Test 1: Import core modules
# -----------------------------------------------------------------------
print("Test 1: Importing core modules â€¦")
try:
    from preprocessing import (
        generate_uhf_pd_signal_physical,
        monte_carlo_wavelet_optimization,
        wavelet_denoise_parametric,
        bandpass_filter,
        normalize_signal,
        get_envelope,
    )
    from descriptors import detect_pulses, compute_delta_t, extract_delta_t_vector
    from blind_algorithms import (
        KalmanDeltaTTracker,
        AdaptiveEWMATracker,
        CUSUMDetector,
        apply_delta_t_tracking,
    )
    from validation import (
        measure_all_tracking_complexities,
        generate_convergence_confusion_matrix,
        generate_phase4_report,
    )
    print("âœ“ All core modules imported successfully")
except Exception as e:
    print(f"âœ— Import error: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Test 2: Physics-based signal generation
# -----------------------------------------------------------------------
print("\nTest 2: Physics-based UHF-PD signal generation â€¦")
try:
    fs = 1e9
    clean, noisy = generate_uhf_pd_signal_physical(
        n_samples=4096, fs=fs, n_pulses=15, snr_db=20.0, seed=42,
    )
    assert clean.shape == (4096,), f"Bad shape: {clean.shape}"
    assert noisy.shape == (4096,), f"Bad shape: {noisy.shape}"
    print(f"âœ“ Signal generated: {len(clean)} samples, "
          f"clean power={np.mean(clean**2):.4e}, "
          f"noisy power={np.mean(noisy**2):.4e}")
except Exception as e:
    print(f"âœ— Signal generation failed: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Test 3: Phase 1 â€” Monte Carlo wavelet optimisation (reduced iterations)
# -----------------------------------------------------------------------
print("\nTest 3: Phase 1 â€” Wavelet optimisation (N=10, quick) â€¦")
try:
    mc = monte_carlo_wavelet_optimization(
        reference_clean=clean, fs=fs, n_iterations=10, seed=42,
    )
    print(f"âœ“ Optimal: wavelet={mc.best_wavelet}, "
          f"mode={mc.best_threshold_mode}, rule={mc.best_threshold_rule}")
    print(f"  E[RMSE]={mc.best_rmse_mean:.6f}, converged={mc.converged}")
except Exception as e:
    print(f"âœ— Phase 1 failed: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Test 4: Phase 2 â€” Î”t extraction
# -----------------------------------------------------------------------
print("\nTest 4: Phase 2 â€” Î”t vector extraction â€¦")
try:
    denoised = wavelet_denoise_parametric(
        noisy, wavelet=mc.best_wavelet,
        threshold_mode=mc.best_threshold_mode,
        threshold_rule=mc.best_threshold_rule,
    )
    delta_t = extract_delta_t_vector(denoised, fs, threshold_sigma=3.0)
    print(f"âœ“ Extracted {len(delta_t)} Î”t intervals")
    if len(delta_t) > 0:
        print(f"  mean={np.mean(delta_t):.4e} s, std={np.std(delta_t):.4e} s")
except Exception as e:
    print(f"âœ— Phase 2 failed: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Test 5: Phase 3 â€” Tracking algorithms
# -----------------------------------------------------------------------
print("\nTest 5: Phase 3 â€” Tracking (Kalman / EWMA / CUSUM) â€¦")
try:
    if len(delta_t) >= 3:
        tracking = apply_delta_t_tracking(delta_t)
        print(f"âœ“ Kalman: steady-state gain = {tracking.kalman.steady_state_gain:.6f}")
        print(f"  EWMA: final Î± = {tracking.ewma.alpha_sequence[-1]:.4f}")
        print(f"  CUSUM: {tracking.cusum.n_alarms} alarms")
    else:
        print("âš  Insufficient Î”t intervals for tracking (need â‰¥ 3)")
except Exception as e:
    print(f"âœ— Phase 3 failed: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Test 6: Phase 4 â€” Big-O complexity (small sizes for speed)
# -----------------------------------------------------------------------
print("\nTest 6: Phase 4 â€” Big-O complexity measurement â€¦")
try:
    complexity = measure_all_tracking_complexities(
        sizes=(128, 256, 512), n_repeats=2, seed=42,
    )
    for name, est in complexity.items():
        print(f"  {name}: {est.big_o_label}  (RÂ²={est.r_squared:.4f})")
    print("âœ“ Complexity measurement complete")
except Exception as e:
    print(f"âœ— Phase 4 complexity failed: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Test 7: Phase 4 â€” Convergence/FPR confusion matrix (reduced MC)
# -----------------------------------------------------------------------
print("\nTest 7: Phase 4 â€” Convergence/FPR confusion matrix â€¦")
try:
    confusion = generate_convergence_confusion_matrix(
        n_samples=200, n_monte_carlo=5, seed=42,
        variation_levels=(0.0, 0.5, 1.0),
    )
    assert confusion.latency_matrix.shape == (3, 3)
    assert confusion.fpr_matrix.shape == (3, 3)
    print(f"âœ“ Confusion matrix: {confusion.latency_matrix.shape} "
          f"({len(confusion.algorithms)} algorithms Ã— "
          f"{len(confusion.variation_levels)} CV levels)")
except Exception as e:
    print(f"âœ— Phase 4 confusion matrix failed: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Test 8: Phase 4 â€” Report generation
# -----------------------------------------------------------------------
print("\nTest 8: Phase 4 â€” Report generation â€¦")
try:
    report = generate_phase4_report(complexity, confusion)
    assert len(report) > 100, "Report too short"
    print("âœ“ Phase 4 report generated successfully")
except Exception as e:
    print(f"âœ— Report generation failed: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------
# Test 9: Full import chain sanity check
# -----------------------------------------------------------------------
print("\nTest 9: Full import chain verification â€¦")
try:
    from main import run_phase1, run_phase2, run_phase3, run_phase4
    print("âœ“ Main pipeline entry points accessible")
except Exception as e:
    print(f"âœ— Main import failed: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("âœ… ALL TESTS PASSED â€” Doctoral pipeline is operational.")
print("=" * 70)
print()
print("Run the full pipeline with:  python main.py")
print()

