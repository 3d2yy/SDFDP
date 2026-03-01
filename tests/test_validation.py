"""
TDD Test Suite — Validation (Phase 4)
=======================================

Verifies confusion matrix, FPR/FNR, accuracy, threshold stability,
and the full ``validate_detection_system`` pipeline.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np

from validation import (
    calculate_confusion_matrix,
    calculate_false_positive_rate,
    calculate_false_negative_rate,
    calculate_accuracy,
    calculate_threshold_stability,
)
from main import process_and_analyze_signal, evaluate_multiple_states


class TestConfusionMatrix(unittest.TestCase):

    def test_perfect(self):
        labels = ['verde', 'amarillo', 'naranja', 'rojo']
        cm = calculate_confusion_matrix(labels, labels)
        diag = np.diag(cm['matrix'])
        self.assertTrue(np.all(diag == 1))
        self.assertEqual(np.sum(cm['matrix']), 4)

    def test_shape(self):
        true = ['verde', 'verde', 'rojo', 'rojo']
        pred = ['verde', 'rojo',  'rojo', 'verde']
        cm = calculate_confusion_matrix(true, pred)
        n_classes = len(cm['classes'])
        self.assertEqual(cm['matrix'].shape, (n_classes, n_classes))


class TestFPR(unittest.TestCase):

    def test_no_false_positives(self):
        true = ['verde', 'verde', 'rojo', 'rojo']
        pred = ['verde', 'verde', 'rojo', 'rojo']
        self.assertAlmostEqual(calculate_false_positive_rate(true, pred, 'rojo'), 0.0)

    def test_all_false_positives(self):
        true = ['verde', 'verde']
        pred = ['rojo',  'rojo']
        self.assertAlmostEqual(calculate_false_positive_rate(true, pred, 'rojo'), 1.0)


class TestFNR(unittest.TestCase):

    def test_no_false_negatives(self):
        true = ['rojo', 'rojo', 'verde']
        pred = ['rojo', 'rojo', 'verde']
        self.assertAlmostEqual(calculate_false_negative_rate(true, pred, 'rojo'), 0.0)

    def test_all_false_negatives(self):
        true = ['rojo', 'rojo']
        pred = ['verde', 'verde']
        self.assertAlmostEqual(calculate_false_negative_rate(true, pred, 'rojo'), 1.0)


class TestAccuracy(unittest.TestCase):

    def test_perfect(self):
        labels = ['a', 'b', 'c', 'd']
        self.assertAlmostEqual(calculate_accuracy(labels, labels), 1.0)

    def test_half(self):
        true = ['a', 'b', 'a', 'b']
        pred = ['a', 'a', 'a', 'a']
        self.assertAlmostEqual(calculate_accuracy(true, pred), 0.5)


class TestThresholdStability(unittest.TestCase):

    def test_constant_signal_high_stability(self):
        vals = np.ones(100)
        s = calculate_threshold_stability(vals)
        self.assertEqual(s['coefficient_of_variation'], 0.0)
        self.assertEqual(s['stability_score'], 1.0)

    def test_returns_keys(self):
        vals = np.random.randn(100)
        s = calculate_threshold_stability(vals)
        for key in ('avg_local_std', 'max_local_std',
                     'coefficient_of_variation', 'trend_changes', 'stability_score'):
            self.assertIn(key, s)


class TestEndToEndPipeline(unittest.TestCase):
    """Integration test: generate -> preprocess -> descriptors -> severity -> validate."""

    def test_full_pipeline(self):
        from main import generate_synthetic_signal
        sig = generate_synthetic_signal('rojo', duration=3000, fs=10000)
        result = process_and_analyze_signal(sig, fs=10000)
        self.assertIn('traffic_light_state', result)
        self.assertIn('severity_index', result)
        self.assertIn('descriptors', result)

    def test_evaluate_multiple_states(self):
        results = evaluate_multiple_states(n_samples_per_state=2, fs=10000)
        self.assertIn('all_results', results)
        self.assertIn('validation', results)


if __name__ == '__main__':
    unittest.main()
