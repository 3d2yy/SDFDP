"""
TDD Test Suite — Blind Algorithms (Phase 3)
=============================================

Verifies Kalman Δt tracker, Adaptive EWMA, CUSUM detector, and the
unified ``apply_delta_t_tracking`` entry point.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np

from blind_algorithms import (
    KalmanDeltaTTracker, KalmanDeltaTResult,
    AdaptiveEWMATracker, AdaptiveEWMAResult,
    CUSUMDetector, CUSUMResult,
    apply_delta_t_tracking, DeltaTTrackingResult,
    EWMA, SimpleMovingAverage, KalmanFilter1D, AdaptiveLMS, AdaptiveRLS,
)


class TestKalmanDeltaTTracker(unittest.TestCase):

    def test_output_lengths(self):
        dt = np.ones(100) * 1e-4
        r = KalmanDeltaTTracker().track(dt)
        self.assertEqual(len(r.filtered), 100)
        self.assertEqual(len(r.residuals), 100)
        self.assertEqual(len(r.kalman_gains), 100)

    def test_converges_to_constant(self):
        dt = np.ones(200) * 1e-4
        r = KalmanDeltaTTracker().track(dt)
        np.testing.assert_allclose(r.filtered[-1], 1e-4, rtol=0.01)

    def test_kalman_gain_decreases(self):
        dt = np.ones(200) * 1e-4
        r = KalmanDeltaTTracker().track(dt)
        self.assertLess(r.kalman_gains[-1], r.kalman_gains[0])

    def test_steady_state_gain_positive(self):
        dt = np.ones(200) * 1e-4
        r = KalmanDeltaTTracker().track(dt)
        self.assertGreater(r.steady_state_gain, 0)

    def test_residuals_near_zero_stationary(self):
        dt = np.ones(200) * 1e-4
        r = KalmanDeltaTTracker().track(dt)
        self.assertAlmostEqual(np.mean(np.abs(r.residuals[-50:])), 0.0, places=6)


class TestAdaptiveEWMA(unittest.TestCase):

    def test_output_lengths(self):
        dt = np.random.exponential(1e-4, 100)
        r = AdaptiveEWMATracker().track(dt)
        self.assertEqual(len(r.smoothed), 100)
        self.assertEqual(len(r.residuals), 100)
        self.assertEqual(len(r.alpha_sequence), 100)

    def test_alpha_bounded(self):
        dt = np.random.exponential(1e-4, 200)
        tracker = AdaptiveEWMATracker(alpha_min=0.05, alpha_max=0.8)
        r = tracker.track(dt)
        self.assertTrue(np.all(r.alpha_sequence >= 0.05 - 1e-12))
        self.assertTrue(np.all(r.alpha_sequence <= 0.80 + 1e-12))

    def test_smoothed_tracks_step(self):
        dt = np.concatenate([np.ones(100) * 1e-4, np.ones(100) * 5e-4])
        r = AdaptiveEWMATracker(alpha_0=0.3).track(dt)
        self.assertGreater(r.smoothed[-1], 3e-4)


class TestCUSUM(unittest.TestCase):

    def test_no_alarms_stationary(self):
        dt = np.ones(500) * 1e-4
        r = CUSUMDetector(threshold=50.0, drift=0.5).detect(dt)
        self.assertEqual(r.n_alarms, 0)

    def test_detects_step_change(self):
        """A large step change should trigger alarms."""
        dt = np.concatenate([np.ones(200) * 1e-4, np.ones(200) * 100.0])
        r = CUSUMDetector(threshold=5.0, drift=0.5).detect(dt)
        self.assertGreater(r.n_alarms, 0)

    def test_g_plus_g_minus_non_negative(self):
        dt = np.random.exponential(1e-4, 300)
        r = CUSUMDetector().detect(dt)
        self.assertTrue(np.all(r.g_plus >= 0))
        self.assertTrue(np.all(r.g_minus >= 0))

    def test_output_shapes(self):
        dt = np.random.exponential(1e-4, 300)
        r = CUSUMDetector().detect(dt)
        self.assertEqual(len(r.g_plus), 300)
        self.assertEqual(len(r.g_minus), 300)
        self.assertEqual(len(r.alarms), 300)


class TestApplyDeltaTTracking(unittest.TestCase):

    def test_returns_all_three(self):
        dt = np.random.exponential(1e-4, 200)
        result = apply_delta_t_tracking(dt)
        self.assertTrue(hasattr(result, 'kalman'))
        self.assertTrue(hasattr(result, 'ewma'))
        self.assertTrue(hasattr(result, 'cusum'))

    def test_consistent_lengths(self):
        dt = np.random.exponential(1e-4, 150)
        r = apply_delta_t_tracking(dt)
        self.assertEqual(len(r.kalman.filtered), 150)
        self.assertEqual(len(r.ewma.smoothed), 150)
        self.assertEqual(len(r.cusum.g_plus), 150)


class TestLegacyAlgorithms(unittest.TestCase):
    """Basic smoke tests for legacy EWMA, SMA, KalmanFilter1D, LMS, RLS."""

    def test_ewma_score(self):
        from blind_algorithms import EWMA
        sig = np.random.randn(500)
        score = EWMA(alpha=0.2).calculate_score(sig)
        self.assertGreater(score, 0)

    def test_sma_score(self):
        from blind_algorithms import SimpleMovingAverage
        sig = np.random.randn(500)
        score = SimpleMovingAverage(window_size=10).calculate_score(sig)
        self.assertGreater(score, 0)

    def test_kalman1d_score(self):
        from blind_algorithms import KalmanFilter1D
        sig = np.random.randn(500)
        score = KalmanFilter1D().calculate_score(sig)
        self.assertGreater(score, 0)

    def test_lms_score(self):
        from blind_algorithms import AdaptiveLMS
        sig = np.random.randn(500)
        score = AdaptiveLMS().calculate_score(sig)
        self.assertIsInstance(score, float)

    def test_rls_score(self):
        from blind_algorithms import AdaptiveRLS
        sig = np.random.randn(500)
        score = AdaptiveRLS().calculate_score(sig)
        self.assertIsInstance(score, float)


if __name__ == '__main__':
    unittest.main()
