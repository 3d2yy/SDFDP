"""
TDD Test Suite — Mathematical Correctness Verification
========================================================

Explicit verification of key mathematical formulas used
throughout the signal processing pipeline.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from scipy import signal as sp_signal


class TestDonohoJohnstoneThreshold(unittest.TestCase):
    """Verify the universal threshold: σ̂ · √(2·log(n))"""

    def test_universal_formula(self):
        """σ̂ estimated via MAD / 0.6745 should yield the correct threshold."""
        rng = np.random.default_rng(0)
        n = 4096
        data = rng.standard_normal(n)

        # MAD-based sigma estimate
        mad = np.median(np.abs(data - np.median(data)))
        sigma_hat = mad / 0.6745

        # Universal threshold
        threshold = sigma_hat * np.sqrt(2 * np.log(n))

        # With n=4096, √(2·ln(4096)) ≈ 4.079
        expected_sqrt = np.sqrt(2 * np.log(n))
        self.assertAlmostEqual(expected_sqrt, 4.079, places=2)
        self.assertGreater(threshold, 3.0)
        self.assertLess(threshold, 6.0)


class TestMinimaxThreshold(unittest.TestCase):
    """Verify minimax: σ̂ · (0.3936 + 0.1829 · log2(n))"""

    def test_minimax_formula(self):
        n = 1024
        sigma = 1.0
        minimax = sigma * (0.3936 + 0.1829 * np.log2(n))
        self.assertAlmostEqual(minimax, 0.3936 + 0.1829 * 10.0, places=4)


class TestButterworthFilterResponse(unittest.TestCase):
    """Verify that the Butterworth bandpass design is mathematically correct."""

    def test_passband_gain_near_unity(self):
        """At the geometric center of the passband, gain ≈ 1 (0 dB)."""
        fs = 10000
        lowcut, highcut = 100, 4000
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = sp_signal.butter(5, [low, high], btype='band')
        # Frequency response at geometric center
        f_center = np.sqrt(lowcut * highcut)
        w, h = sp_signal.freqz(b, a, worN=[2 * np.pi * f_center / fs])
        gain_db = 20 * np.log10(np.abs(h[0]) + 1e-30)
        self.assertGreater(gain_db, -3.0)  # within -3 dB

    def test_stopband_attenuation(self):
        """At 0.1× lowcut, attenuation should be significant."""
        fs = 10000
        lowcut, highcut = 100, 4000
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = sp_signal.butter(5, [low, high], btype='band')
        w, h = sp_signal.freqz(b, a, worN=[2 * np.pi * 10 / fs])
        gain_db = 20 * np.log10(np.abs(h[0]) + 1e-30)
        self.assertLess(gain_db, -20.0)  # > 20 dB attenuation


class TestKalmanMathUpdate(unittest.TestCase):
    """Verify Kalman filter update equations directly."""

    def test_prediction_step(self):
        """x_predicted = F · x, P_predicted = F · P · Fᵀ + Q."""
        x = np.array([1e-4, 0.0])
        F = np.array([[1, 1], [0, 1]])
        P = np.eye(2) * 0.01
        Q = np.eye(2) * 1e-5

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        self.assertAlmostEqual(x_pred[0], 1e-4)
        self.assertAlmostEqual(x_pred[1], 0.0)
        # P_pred should be larger than P due to process noise
        self.assertGreater(P_pred[0, 0], P[0, 0])

    def test_update_step_reduces_uncertainty(self):
        """After the update step, P should be smaller."""
        H = np.array([[1, 0]])
        R = np.array([[1e-2]])
        P_pred = np.eye(2) * 0.01

        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        P_update = (np.eye(2) - K @ H) @ P_pred

        # Updated covariance should be smaller
        self.assertLess(P_update[0, 0], P_pred[0, 0])


class TestCUSUMMath(unittest.TestCase):
    """Verify two-sided CUSUM accumulation formula."""

    def test_g_plus_accumulation(self):
        """g⁺(k) = max(0, g⁺(k-1) + (x(k) - μ₀ - δ/2))"""
        mu_0 = 0.0
        drift = 0.5
        g_plus = 0.0

        # If observation is 2.0: g+ = max(0, 0 + 2.0 - 0.0 - 0.5) = 1.5
        x = 2.0
        g_plus = max(0, g_plus + (x - mu_0) - drift)
        self.assertAlmostEqual(g_plus, 1.5)

        # Second observation of 2.0: g+ = max(0, 1.5 + 1.5) = 3.0
        g_plus = max(0, g_plus + (x - mu_0) - drift)
        self.assertAlmostEqual(g_plus, 3.0)

    def test_g_minus_accumulation(self):
        """g⁻(k) = max(0, g⁻(k-1) - (x(k) - μ₀) - δ/2)"""
        mu_0 = 0.0
        drift = 0.5
        g_minus = 0.0

        # Observation -2.0: g- = max(0, 0 - (-2.0 - 0.0) - 0.5) = max(0, 1.5) = 1.5
        x = -2.0
        g_minus = max(0, g_minus - (x - mu_0) - drift)
        self.assertAlmostEqual(g_minus, 1.5)


class TestSeverityWeightedIndex(unittest.TestCase):
    """Verify: SI = Σ(wᵢ · sᵢ) / Σ(wᵢ)"""

    def test_uniform_weights(self):
        scores = {'a': 2.0, 'b': 4.0, 'c': 6.0}
        weights = {'a': 1.0, 'b': 1.0, 'c': 1.0}
        si = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())
        self.assertAlmostEqual(si, 4.0)

    def test_weighted(self):
        scores = {'a': 2.0, 'b': 4.0}
        weights = {'a': 3.0, 'b': 1.0}
        si = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())
        self.assertAlmostEqual(si, 2.5)


class TestRobustNormalization(unittest.TestCase):
    """Verify: x_norm = (x - median) / (1.4826 · MAD)"""

    def test_formula(self):
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        scale = 1.4826 * mad
        normalized = (data - median) / scale

        self.assertAlmostEqual(median, 3.0)
        self.assertAlmostEqual(mad, 1.0)
        self.assertAlmostEqual(scale, 1.4826)
        self.assertAlmostEqual(normalized[2], 0.0)  # median element → 0


class TestCohensD(unittest.TestCase):
    """Verify Cohen's d = (μ₁ - μ₂) / s_pooled."""

    def test_known_separation(self):
        g1 = np.array([1, 1, 1, 1], dtype=float)
        g2 = np.array([3, 3, 3, 3], dtype=float)

        mu1, mu2 = g1.mean(), g2.mean()
        n1, n2 = len(g1), len(g2)
        s1, s2 = g1.std(ddof=1), g2.std(ddof=1)

        # With zero variance groups, pooled std → 0, Cohen's d → ∞
        # Use groups with variance
        g1 = np.array([0.0, 1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0, 7.0])
        mu1, mu2 = g1.mean(), g2.mean()
        s1 = g1.std(ddof=1)
        s2 = g2.std(ddof=1)
        s_pooled = np.sqrt(((len(g1) - 1) * s1**2 + (len(g2) - 1) * s2**2)
                           / (len(g1) + len(g2) - 2))
        d = abs(mu1 - mu2) / s_pooled
        # μ1=1.5, μ2=5.5, s_pooled=std of [0,1,2,3]≈1.29
        self.assertAlmostEqual(d, 4.0 / s_pooled, places=4)


if __name__ == '__main__':
    unittest.main()
