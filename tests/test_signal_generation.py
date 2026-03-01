"""
TDD Test Suite — Signal Generation (Phase 0)
=============================================

Verifies that ``generate_synthetic_signal`` and ``generate_custom_signal``
produce signals with correct physical characteristics for all four
traffic-light states.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np

from main import generate_synthetic_signal
from gui.signal_generator import generate_custom_signal

FS = 10_000
DUR = 2_000


class TestGenerateSyntheticSignal(unittest.TestCase):
    """Tests for main.generate_synthetic_signal."""

    def test_output_length(self):
        sig = generate_synthetic_signal('verde', duration=DUR, fs=FS)
        self.assertEqual(len(sig), DUR)

    def test_output_is_float_array(self):
        sig = generate_synthetic_signal('verde')
        self.assertTrue(np.issubdtype(sig.dtype, np.floating))

    def test_severity_ordering_rms(self):
        """RMS should increase with severity: verde < amarillo < naranja < rojo."""
        rms = {}
        for state in ('verde', 'amarillo', 'naranja', 'rojo'):
            sig = generate_synthetic_signal(state, duration=5000, fs=FS)
            rms[state] = np.sqrt(np.mean(sig ** 2))
        self.assertLess(rms['verde'],    rms['amarillo'])
        self.assertLess(rms['amarillo'], rms['naranja'])
        self.assertLess(rms['naranja'],  rms['rojo'])

    def test_no_nan_or_inf(self):
        for state in ('verde', 'amarillo', 'naranja', 'rojo'):
            sig = generate_synthetic_signal(state, duration=3000, fs=FS)
            self.assertFalse(np.any(np.isnan(sig)), f"NaN in state {state}")
            self.assertFalse(np.any(np.isinf(sig)), f"Inf in state {state}")

    def test_noise_level_effect(self):
        sig_low  = generate_synthetic_signal('verde', duration=5000, noise_level=0.01)
        sig_high = generate_synthetic_signal('verde', duration=5000, noise_level=1.0)
        self.assertLess(np.std(sig_low), np.std(sig_high))

    def test_invalid_state_falls_back_to_rojo(self):
        """Unknown states fall through to the else (rojo) branch."""
        sig = generate_synthetic_signal('unknown_state')
        self.assertIsInstance(sig, np.ndarray)


class TestGenerateCustomSignal(unittest.TestCase):
    """Tests for gui.signal_generator.generate_custom_signal."""

    def test_output_length(self):
        sig, meta = generate_custom_signal('verde', 1000, 10000, 0.1, 'gaussian', 5, 1.0, 1000)
        self.assertEqual(len(sig), 1000)

    def test_returns_metadata(self):
        _, meta = generate_custom_signal('verde', 500, 10000, 0.1, 'gaussian', 3, 1.0, 1000)
        self.assertIn('state', meta)
        self.assertIn('sample_rate', meta)
        self.assertIn('discharge_positions', meta)
        self.assertEqual(len(meta['discharge_positions']), 3)

    def test_noise_types(self):
        for nt in ('gaussian', 'pink', 'brown', 'uniform'):
            sig, _ = generate_custom_signal('verde', 1000, 10000, 0.1, nt, 0, 1.0, 1000)
            self.assertEqual(len(sig), 1000)
            self.assertFalse(np.any(np.isnan(sig)), f"NaN with noise_type={nt}")

    def test_zero_discharges(self):
        sig, meta = generate_custom_signal('verde', 1000, 10000, 0.1, 'gaussian', 0, 1.0, 1000)
        self.assertEqual(len(meta['discharge_positions']), 0)


if __name__ == '__main__':
    unittest.main()
