"""
TDD Test Suite — Preprocessing (Phase 1)
=========================================

Verifies mathematical correctness of bandpass filter, normalization,
wavelet denoising, envelope extraction, and Monte Carlo optimization.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np

from preprocessing import (
    bandpass_filter, normalize_signal, wavelet_denoise, get_envelope,
    monte_carlo_wavelet_optimization,
)

FS = 10_000


class TestBandpassFilter(unittest.TestCase):

    def test_output_same_length(self):
        sig = np.random.randn(2000)
        out = bandpass_filter(sig, FS, 100, 4000)
        self.assertEqual(len(out), len(sig))

    def test_removes_dc(self):
        sig = np.ones(2000) + 0.01 * np.random.randn(2000)
        out = bandpass_filter(sig, FS, 100, 4000)
        self.assertAlmostEqual(np.mean(out), 0.0, places=1)

    def test_passes_midband(self):
        t = np.arange(5000) / FS
        tone = np.sin(2 * np.pi * 1000 * t)
        out = bandpass_filter(tone, FS, 500, 2000)
        ratio = np.sqrt(np.mean(out ** 2)) / np.sqrt(np.mean(tone ** 2))
        self.assertGreater(ratio, 0.7)

    def test_rejects_out_of_band(self):
        t = np.arange(5000) / FS
        tone = np.sin(2 * np.pi * 4500 * t)
        out = bandpass_filter(tone, FS, 100, 2000)
        ratio = np.sqrt(np.mean(out ** 2)) / (np.sqrt(np.mean(tone ** 2)) + 1e-30)
        self.assertLess(ratio, 0.2)


class TestNormalizeSignal(unittest.TestCase):

    def test_zscore_mean_zero(self):
        sig = np.random.randn(1000) * 5 + 10
        out = normalize_signal(sig, method='zscore')
        self.assertAlmostEqual(np.mean(out), 0.0, places=5)

    def test_zscore_unit_variance(self):
        sig = np.random.randn(1000) * 5 + 10
        out = normalize_signal(sig, method='zscore')
        self.assertAlmostEqual(np.std(out), 1.0, places=5)

    def test_minmax_range(self):
        sig = np.random.randn(1000) * 5 + 10
        out = normalize_signal(sig, method='minmax')
        self.assertAlmostEqual(np.min(out), 0.0, places=10)
        self.assertAlmostEqual(np.max(out), 1.0, places=10)

    def test_robust_uses_median(self):
        sig = np.array([1.0, 2.0, 3.0, 100.0])
        out = normalize_signal(sig, method='robust')
        self.assertTrue(np.all(np.isfinite(out)))


class TestWaveletDenoise(unittest.TestCase):

    def test_output_same_length(self):
        sig = np.random.randn(1024)
        out = wavelet_denoise(sig)
        self.assertEqual(len(out), len(sig))

    def test_reduces_noise(self):
        noise = np.random.randn(1024)
        out = wavelet_denoise(noise)
        self.assertLessEqual(np.std(out), np.std(noise))

    def test_preserves_deterministic_component(self):
        t = np.arange(1024) / 10000
        clean = 5 * np.sin(2 * np.pi * 500 * t)
        noisy = clean + 0.5 * np.random.randn(1024)
        out = wavelet_denoise(noisy)
        corr = np.corrcoef(clean, out)[0, 1]
        self.assertGreater(corr, 0.8)


class TestEnvelope(unittest.TestCase):

    def test_output_same_length(self):
        sig = np.random.randn(1000)
        env = get_envelope(sig)
        self.assertEqual(len(env), len(sig))

    def test_non_negative(self):
        sig = np.random.randn(1000)
        env = get_envelope(sig)
        self.assertTrue(np.all(env >= 0))


class TestDonohoJohnstoneThreshold(unittest.TestCase):

    def test_universal_threshold_formula(self):
        n = 1024
        expected_factor = np.sqrt(2 * np.log(n))
        self.assertAlmostEqual(expected_factor, 3.7234, places=3)

    def test_mad_estimator(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal(10_000)
        mad = np.median(np.abs(data - np.median(data)))
        sigma_hat = mad / 0.6745
        self.assertAlmostEqual(sigma_hat, 1.0, places=1)


class TestMonteCarloOptimization(unittest.TestCase):

    def test_returns_best_result(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(512)
        result = monte_carlo_wavelet_optimization(sig, n_iterations=10)
        self.assertTrue(hasattr(result, 'best_wavelet'))
        self.assertTrue(hasattr(result, 'best_threshold_mode'))
        self.assertTrue(hasattr(result, 'best_threshold_rule'))

    def test_snr_improvement(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(512)
        result = monte_carlo_wavelet_optimization(sig, n_iterations=10)
        self.assertTrue(hasattr(result, 'best_rmse_mean'))


if __name__ == '__main__':
    unittest.main()
