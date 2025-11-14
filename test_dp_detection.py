"""
Unit tests for the UHF Partial Discharge Detection System.
"""

import unittest
import numpy as np
from signal_processing import SignalProcessor
from feature_extraction import FeatureExtractor
from traffic_light_classifier import TrafficLightClassifier
from adaptive_filters import AdaptiveFilters
from validation import FilterValidator
from dp_detection_system import DPDetectionSystem, generate_synthetic_uhf_signal


class TestSignalProcessing(unittest.TestCase):
    """Test signal processing module."""
    
    def setUp(self):
        self.processor = SignalProcessor(sampling_rate=1e9)
        self.test_signal = np.random.randn(1000)
    
    def test_bandpass_filter(self):
        """Test bandpass filtering."""
        filtered = self.processor.bandpass_filter(self.test_signal)
        self.assertEqual(len(filtered), len(self.test_signal))
        self.assertIsInstance(filtered, np.ndarray)
    
    def test_normalize(self):
        """Test normalization methods."""
        # Z-score normalization
        normalized = self.processor.normalize(self.test_signal, method='zscore')
        self.assertAlmostEqual(np.mean(normalized), 0.0, places=10)
        self.assertAlmostEqual(np.std(normalized), 1.0, places=10)
        
        # MinMax normalization
        normalized = self.processor.normalize(self.test_signal, method='minmax')
        self.assertAlmostEqual(np.min(normalized), 0.0, places=10)
        self.assertAlmostEqual(np.max(normalized), 1.0, places=10)
    
    def test_envelope(self):
        """Test envelope detection."""
        envelope = self.processor.get_envelope(self.test_signal)
        self.assertEqual(len(envelope), len(self.test_signal))
        self.assertTrue(np.all(envelope >= 0))
    
    def test_process_signal(self):
        """Test complete signal processing pipeline."""
        processed, envelope = self.processor.process_signal(self.test_signal)
        self.assertIsNotNone(processed)
        self.assertIsNotNone(envelope)
        self.assertEqual(len(processed), len(self.test_signal))


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction module."""
    
    def setUp(self):
        self.extractor = FeatureExtractor(sampling_rate=1e9)
        self.test_signal = np.random.randn(1000)
    
    def test_calculate_energy(self):
        """Test energy calculation."""
        energy = self.extractor.calculate_energy(self.test_signal)
        self.assertIsInstance(energy, (int, float))
        self.assertGreater(energy, 0)
    
    def test_calculate_rms(self):
        """Test RMS calculation."""
        rms = self.extractor.calculate_rms(self.test_signal)
        self.assertIsInstance(rms, (int, float))
        self.assertGreater(rms, 0)
    
    def test_calculate_kurtosis(self):
        """Test kurtosis calculation."""
        kurtosis = self.extractor.calculate_kurtosis(self.test_signal)
        self.assertIsInstance(kurtosis, (int, float))
    
    def test_calculate_skewness(self):
        """Test skewness calculation."""
        skewness = self.extractor.calculate_skewness(self.test_signal)
        self.assertIsInstance(skewness, (int, float))
    
    def test_extract_all_features(self):
        """Test extraction of all features."""
        features = self.extractor.extract_all_features(self.test_signal)
        self.assertIsInstance(features, dict)
        self.assertIn('energy', features)
        self.assertIn('rms', features)
        self.assertIn('kurtosis', features)
        self.assertIn('skewness', features)


class TestTrafficLightClassifier(unittest.TestCase):
    """Test traffic light classification."""
    
    def setUp(self):
        self.classifier = TrafficLightClassifier()
    
    def test_classify(self):
        """Test classification."""
        # Test green
        classification, level = self.classifier.classify(0.1)
        self.assertEqual(classification, 'green')
        self.assertEqual(level, 0)
        
        # Test yellow
        classification, level = self.classifier.classify(0.4)
        self.assertEqual(classification, 'yellow')
        self.assertEqual(level, 1)
        
        # Test orange
        classification, level = self.classifier.classify(0.6)
        self.assertEqual(classification, 'orange')
        self.assertEqual(level, 2)
        
        # Test red
        classification, level = self.classifier.classify(0.9)
        self.assertEqual(classification, 'red')
        self.assertEqual(level, 3)
    
    def test_combine_descriptors(self):
        """Test combining descriptors."""
        features = {
            'energy': 100.0,
            'rms': 10.0,
            'kurtosis': 3.0,
            'skewness': 0.5,
            'spectral_stability': 0.1,
            'residual': 5.0,
            'band_1_300-600MHz': 50.0
        }
        combined_index, contributions = self.classifier.combine_descriptors(features)
        self.assertIsInstance(combined_index, float)
        self.assertGreaterEqual(combined_index, 0)
        self.assertLessEqual(combined_index, 1)
        self.assertIsInstance(contributions, dict)


class TestAdaptiveFilters(unittest.TestCase):
    """Test adaptive filters."""
    
    def setUp(self):
        self.filters = AdaptiveFilters()
        self.test_signal = np.random.randn(1000)
    
    def test_ewma_filter(self):
        """Test EWMA filter."""
        filtered = self.filters.ewma_filter(self.test_signal)
        self.assertEqual(len(filtered), len(self.test_signal))
    
    def test_moving_average_filter(self):
        """Test moving average filter."""
        filtered = self.filters.moving_average_filter(self.test_signal)
        self.assertEqual(len(filtered), len(self.test_signal))
    
    def test_kalman_filter(self):
        """Test Kalman filter."""
        filtered = self.filters.kalman_filter(self.test_signal)
        self.assertEqual(len(filtered), len(self.test_signal))
    
    def test_lms_filter(self):
        """Test LMS filter."""
        filtered, weights = self.filters.lms_filter(self.test_signal)
        self.assertEqual(len(filtered), len(self.test_signal))
        self.assertIsInstance(weights, np.ndarray)
    
    def test_rls_filter(self):
        """Test RLS filter."""
        filtered, weights = self.filters.rls_filter(self.test_signal)
        self.assertEqual(len(filtered), len(self.test_signal))
        self.assertIsInstance(weights, np.ndarray)
    
    def test_apply_all_filters(self):
        """Test applying all filters."""
        results = self.filters.apply_all_filters(self.test_signal)
        self.assertIsInstance(results, dict)
        self.assertIn('ewma', results)
        self.assertIn('moving_average', results)
        self.assertIn('kalman', results)
        self.assertIn('lms', results)
        self.assertIn('rls', results)


class TestValidation(unittest.TestCase):
    """Test validation module."""
    
    def setUp(self):
        self.validator = FilterValidator()
    
    def test_calculate_snr(self):
        """Test SNR calculation."""
        signal = np.ones(100)
        noise = np.random.randn(100) * 0.1
        snr = self.validator.calculate_snr(signal, noise)
        self.assertIsInstance(snr, float)
        self.assertGreater(snr, 0)
    
    def test_calculate_mse(self):
        """Test MSE calculation."""
        signal = np.random.randn(100)
        reference = signal + np.random.randn(100) * 0.1
        mse = self.validator.calculate_mse(signal, reference)
        self.assertIsInstance(mse, float)
        self.assertGreaterEqual(mse, 0)
    
    def test_detect_events(self):
        """Test event detection."""
        signal = np.zeros(100)
        signal[20:30] = 5.0
        signal[60:70] = 5.0
        events = self.validator.detect_events(signal, threshold=2.0)
        self.assertEqual(len(events), 2)


class TestDPDetectionSystem(unittest.TestCase):
    """Test main DP detection system."""
    
    def setUp(self):
        self.detector = DPDetectionSystem(sampling_rate=1e9)
    
    def test_generate_synthetic_signal(self):
        """Test synthetic signal generation."""
        signal, events = generate_synthetic_uhf_signal(
            duration=1e-4,
            sampling_rate=1e9,
            num_discharges=3
        )
        self.assertEqual(len(signal), 100000)
        self.assertEqual(len(events), 3)
    
    def test_process_and_diagnose(self):
        """Test complete processing and diagnosis."""
        signal, _ = generate_synthetic_uhf_signal(
            duration=1e-4,
            sampling_rate=1e9,
            num_discharges=2
        )
        diagnosis = self.detector.process_and_diagnose(
            signal,
            apply_filters=False
        )
        self.assertIsInstance(diagnosis, dict)
        self.assertIn('processed_signal', diagnosis)
        self.assertIn('envelope', diagnosis)
        self.assertIn('features', diagnosis)
        self.assertIn('classification', diagnosis)
    
    def test_generate_diagnostic_report(self):
        """Test diagnostic report generation."""
        signal, _ = generate_synthetic_uhf_signal(
            duration=1e-4,
            sampling_rate=1e9,
            num_discharges=2
        )
        diagnosis = self.detector.process_and_diagnose(
            signal,
            apply_filters=False
        )
        report = self.detector.generate_diagnostic_report(diagnosis)
        self.assertIsInstance(report, str)
        self.assertIn('CLASSIFICATION RESULTS', report)
        self.assertIn('EXTRACTED FEATURES', report)


if __name__ == '__main__':
    unittest.main()
