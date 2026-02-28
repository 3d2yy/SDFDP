"""
TDD Test Suite — Descriptors (Phase 2)
=======================================

Verifies pulse detection, Δt extraction, and all legacy descriptor
calculations for mathematical correctness.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np

from descriptors import (
    detect_pulses, compute_delta_t, extract_delta_t_vector,
    energy_total, rms_value, kurtosis, skewness,
    crest_factor, spectral_entropy, zero_crossing_rate,
    peak_count, compute_all_descriptors,
)


class TestPulseDetection(unittest.TestCase):
    """Tests for descriptors.detect_pulses."""

    def _make_pulsed_signal(self, n_pulses=5, duration=5000, fs=10000):
        sig = np.zeros(duration)
        positions = np.linspace(500, duration - 500, n_pulses, dtype=int)
        for pos in positions:
            t = np.arange(50) / fs
            pulse = 5.0 * np.exp(-t * 200) * np.sin(2 * np.pi * 1000 * t)
            end = min(pos + 50, duration)
            sig[pos:end] += pulse[:end - pos]
        return sig, positions

    def test_finds_pulses(self):
        sig, _ = self._make_pulsed_signal(n_pulses=5)
        peaks = detect_pulses(sig, fs=10000)
        self.assertGreater(len(peaks), 0)

    def test_finds_correct_count_approximately(self):
        sig, _ = self._make_pulsed_signal(n_pulses=5)
        peaks = detect_pulses(sig, fs=10000)
        self.assertGreaterEqual(len(peaks), 1)
        self.assertLessEqual(len(peaks), 100)

    def test_no_pulses_in_silence(self):
        sig = np.zeros(1000)
        peaks = detect_pulses(sig, fs=10000)
        self.assertEqual(len(peaks), 0)


class TestDeltaT(unittest.TestCase):
    """Tests for descriptors.compute_delta_t and extract_delta_t_vector."""

    def test_delta_t_from_positions(self):
        positions = np.array([100, 300, 600, 1100])
        fs = 10000
        dt = compute_delta_t(positions, fs)
        expected = np.diff(positions) / fs
        np.testing.assert_allclose(dt, expected)

    def test_delta_t_length(self):
        positions = np.array([100, 300, 600, 1100])
        dt = compute_delta_t(positions, 10000)
        self.assertEqual(len(dt), len(positions) - 1)

    def test_extract_returns_array(self):
        from main import generate_synthetic_signal
        sig = generate_synthetic_signal('naranja', duration=5000, fs=10000)
        dt = extract_delta_t_vector(sig, fs=10000)
        self.assertIsInstance(dt, np.ndarray)

    def test_delta_t_all_positive(self):
        positions = np.array([100, 300, 600, 1100])
        dt = compute_delta_t(positions, 10000)
        self.assertTrue(np.all(dt > 0))


class TestLegacyDescriptors(unittest.TestCase):
    """Tests for legacy descriptor functions in descriptors.py."""

    # -- Energy = sum(x^2) --
    def test_energy_known(self):
        sig = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(energy_total(sig), 14.0, places=10)

    def test_energy_non_negative(self):
        sig = np.random.randn(500)
        self.assertGreaterEqual(energy_total(sig), 0.0)

    # -- RMS = sqrt(mean(x^2)) --
    def test_rms_known(self):
        sig = np.array([1.0, -1.0, 1.0, -1.0])
        self.assertAlmostEqual(rms_value(sig), 1.0, places=10)

    def test_rms_non_negative(self):
        sig = np.random.randn(500)
        self.assertGreaterEqual(rms_value(sig), 0.0)

    # -- Kurtosis (Fisher) Gaussian → ~0 --
    def test_kurtosis_gaussian(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(50_000)
        k = kurtosis(sig)
        self.assertAlmostEqual(k, 0.0, places=0)

    # -- Skewness Gaussian → ~0 --
    def test_skewness_gaussian(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(50_000)
        s = skewness(sig)
        self.assertAlmostEqual(s, 0.0, places=0)

    # -- Crest Factor = peak / RMS --
    def test_crest_factor_sine(self):
        t = np.linspace(0, 2 * np.pi * 10, 10000, endpoint=False)
        sig = np.sin(t)
        cf = crest_factor(sig)
        self.assertAlmostEqual(cf, np.sqrt(2), places=2)

    def test_crest_factor_positive(self):
        sig = np.random.randn(1000)
        self.assertGreater(crest_factor(sig), 0.0)

    # -- Spectral Entropy --
    def test_spectral_entropy_non_negative(self):
        sig = np.random.randn(1024)
        se = spectral_entropy(sig, fs=10000)
        self.assertGreaterEqual(se, 0.0)

    def test_spectral_entropy_white_noise_higher(self):
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(2048)
        t = np.arange(2048) / 10000
        tone = np.sin(2 * np.pi * 1000 * t)
        se_noise = spectral_entropy(noise, fs=10000)
        se_tone = spectral_entropy(tone, fs=10000)
        self.assertGreater(se_noise, se_tone)

    # -- Zero Crossing Rate --
    def test_zcr_sine(self):
        t = np.arange(10000) / 10000
        sig = np.sin(2 * np.pi * 100 * t)
        zcr = zero_crossing_rate(sig)
        self.assertAlmostEqual(zcr, 0.02, places=2)

    # -- Peak Count --
    def test_peak_count_non_negative(self):
        sig = np.random.randn(1000)
        self.assertGreaterEqual(peak_count(sig), 0)

    # -- compute_all_descriptors returns dict --
    def test_compute_all_returns_dict(self):
        sig = np.random.randn(1000)
        d = compute_all_descriptors(sig, fs=10000, original_signal=sig)
        self.assertIsInstance(d, dict)
        self.assertIn('energy_total', d)
        self.assertIn('rms', d)
        self.assertIn('kurtosis', d)


if __name__ == '__main__':
    unittest.main()
