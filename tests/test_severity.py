"""
TDD Test Suite — Severity (Classification)
============================================

Verifies severity scoring, threshold determination, and traffic-light
classification logic.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np

from severity import (
    normalize_descriptor, calculate_severity_index, classify_traffic_light,
    assess_severity, determine_thresholds_percentile, determine_thresholds_statistical,
)


class TestNormalizeDescriptor(unittest.TestCase):

    def test_zscore_zero_when_equal(self):
        self.assertAlmostEqual(normalize_descriptor(5.0, 5.0, 1.0), 0.0)

    def test_zscore_one_sigma(self):
        self.assertAlmostEqual(normalize_descriptor(6.0, 5.0, 1.0), 1.0)

    def test_zero_std_returns_zero(self):
        self.assertAlmostEqual(normalize_descriptor(10.0, 5.0, 0.0), 0.0)


class TestSeverityIndex(unittest.TestCase):

    def test_empty_scores(self):
        self.assertEqual(calculate_severity_index({}), 0.0)

    def test_uniform_weights(self):
        scores = {'a': 2.0, 'b': 4.0}
        self.assertAlmostEqual(calculate_severity_index(scores), 3.0)

    def test_custom_weights(self):
        scores = {'a': 2.0, 'b': 4.0}
        weights = {'a': 1.0, 'b': 3.0}
        self.assertAlmostEqual(calculate_severity_index(scores, weights), 3.5)


class TestClassifyTrafficLight(unittest.TestCase):

    def test_verde(self):
        self.assertEqual(classify_traffic_light(0.5), 'verde')

    def test_amarillo(self):
        self.assertEqual(classify_traffic_light(3.0), 'amarillo')

    def test_naranja(self):
        self.assertEqual(classify_traffic_light(10.0), 'naranja')

    def test_rojo(self):
        self.assertEqual(classify_traffic_light(20.0), 'rojo')

    def test_boundary_green_yellow(self):
        self.assertEqual(classify_traffic_light(2.0), 'amarillo')

    def test_custom_thresholds(self):
        thresholds = {'green_yellow': 1.0, 'yellow_orange': 2.0, 'orange_red': 3.0}
        self.assertEqual(classify_traffic_light(0.5, thresholds), 'verde')
        self.assertEqual(classify_traffic_light(1.5, thresholds), 'amarillo')
        self.assertEqual(classify_traffic_light(2.5, thresholds), 'naranja')
        self.assertEqual(classify_traffic_light(3.5, thresholds), 'rojo')


class TestAssessSeverity(unittest.TestCase):

    def test_returns_required_keys(self):
        desc = {'energy_total': 1.0, 'rms': 0.5, 'spectral_entropy': 0.3}
        result = assess_severity(desc)
        self.assertIn('severity_index', result)
        self.assertIn('traffic_light_state', result)
        self.assertIn('descriptor_scores', result)

    def test_low_descriptors_green(self):
        desc = {'energy_total': 0.01, 'rms': 0.01, 'spectral_entropy': 0.01}
        result = assess_severity(desc)
        self.assertEqual(result['traffic_light_state'], 'verde')


class TestThresholdDetermination(unittest.TestCase):

    def test_percentile_ordering(self):
        vals = np.arange(100, dtype=float)
        t = determine_thresholds_percentile(vals)
        self.assertLess(t['green_yellow'], t['yellow_orange'])
        self.assertLess(t['yellow_orange'], t['orange_red'])

    def test_statistical_ordering(self):
        vals = np.random.randn(200)
        t = determine_thresholds_statistical(vals)
        self.assertLess(t['green_yellow'], t['yellow_orange'])
        self.assertLess(t['yellow_orange'], t['orange_red'])

    def test_statistical_formula(self):
        vals = np.array([0.0] * 100)
        vals[0] = 10.0
        mean = np.mean(vals)
        std = np.std(vals)
        t = determine_thresholds_statistical(vals, sigma_multipliers=[1.5, 4.0, 8.0])
        self.assertAlmostEqual(t['green_yellow'], mean + 1.5 * std, places=8)
        self.assertAlmostEqual(t['yellow_orange'], mean + 4.0 * std, places=8)
        self.assertAlmostEqual(t['orange_red'],    mean + 8.0 * std, places=8)


if __name__ == '__main__':
    unittest.main()
