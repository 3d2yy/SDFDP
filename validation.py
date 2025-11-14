"""
Validation and comparison module for adaptive filters.
Calculates False Positives (FP), False Negatives (FN), SNR improvements,
and generates comparison tables.
"""

import numpy as np
import pandas as pd


class FilterValidator:
    """
    Validate and compare performance of different adaptive filters.
    """
    
    def __init__(self):
        """
        Initialize validator.
        """
        pass
    
    def calculate_snr(self, signal, noise):
        """
        Calculate Signal-to-Noise Ratio (SNR).
        
        Parameters:
        -----------
        signal : array-like
            Clean signal
        noise : array-like
            Noise signal
            
        Returns:
        --------
        snr_db : float
            SNR in decibels
        """
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr) if snr > 0 else -float('inf')
        
        return snr_db
    
    def calculate_snr_improvement(self, original_signal, filtered_signal, ground_truth):
        """
        Calculate SNR improvement after filtering.
        
        Parameters:
        -----------
        original_signal : array-like
            Original noisy signal
        filtered_signal : array-like
            Filtered signal
        ground_truth : array-like
            Ground truth clean signal
            
        Returns:
        --------
        snr_improvement : float
            SNR improvement in dB
        """
        original_noise = original_signal - ground_truth
        filtered_noise = filtered_signal - ground_truth
        
        snr_original = self.calculate_snr(ground_truth, original_noise)
        snr_filtered = self.calculate_snr(ground_truth, filtered_noise)
        
        return snr_filtered - snr_original
    
    def calculate_mse(self, signal, reference):
        """
        Calculate Mean Squared Error.
        
        Parameters:
        -----------
        signal : array-like
            Signal to evaluate
        reference : array-like
            Reference signal
            
        Returns:
        --------
        mse : float
            Mean squared error
        """
        return np.mean((signal - reference)**2)
    
    def calculate_rmse(self, signal, reference):
        """
        Calculate Root Mean Squared Error.
        
        Parameters:
        -----------
        signal : array-like
            Signal to evaluate
        reference : array-like
            Reference signal
            
        Returns:
        --------
        rmse : float
            Root mean squared error
        """
        return np.sqrt(self.calculate_mse(signal, reference))
    
    def detect_events(self, signal, threshold, min_duration=5):
        """
        Detect events (e.g., partial discharges) in signal.
        
        Parameters:
        -----------
        signal : array-like
            Signal to analyze
        threshold : float
            Detection threshold
        min_duration : int
            Minimum duration for an event (samples)
            
        Returns:
        --------
        events : list
            List of (start, end) tuples for detected events
        """
        signal = np.asarray(signal)
        above_threshold = np.abs(signal) > threshold
        
        events = []
        in_event = False
        start = 0
        
        for i in range(len(above_threshold)):
            if above_threshold[i] and not in_event:
                start = i
                in_event = True
            elif not above_threshold[i] and in_event:
                if i - start >= min_duration:
                    events.append((start, i))
                in_event = False
        
        # Handle case where signal ends during an event
        if in_event and len(signal) - start >= min_duration:
            events.append((start, len(signal)))
        
        return events
    
    def calculate_detection_metrics(self, detected_events, true_events, tolerance=10):
        """
        Calculate detection metrics (TP, FP, FN).
        
        Parameters:
        -----------
        detected_events : list
            List of detected event intervals
        true_events : list
            List of true event intervals
        tolerance : int
            Tolerance in samples for matching events
            
        Returns:
        --------
        metrics : dict
            Dictionary with TP, FP, FN counts
        """
        tp = 0  # True Positives
        fp = 0  # False Positives
        fn = 0  # False Negatives
        
        matched_true = set()
        
        # Check each detected event
        for det_start, det_end in detected_events:
            matched = False
            for i, (true_start, true_end) in enumerate(true_events):
                # Check if events overlap (with tolerance)
                if (det_start <= true_end + tolerance and 
                    det_end >= true_start - tolerance):
                    matched = True
                    matched_true.add(i)
                    break
            
            if matched:
                tp += 1
            else:
                fp += 1
        
        # Count false negatives (true events not detected)
        fn = len(true_events) - len(matched_true)
        
        metrics = {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 0
        }
        
        # Calculate F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                 (metrics['precision'] + metrics['recall'])
        
        return metrics
    
    def compare_filters(self, original_signal, filter_results, ground_truth=None, 
                       true_events=None, threshold=None):
        """
        Compare performance of different filters.
        
        Parameters:
        -----------
        original_signal : array-like
            Original noisy signal
        filter_results : dict
            Dictionary of filtered signals from different methods
        ground_truth : array-like
            Ground truth clean signal (optional)
        true_events : list
            List of true event intervals (optional)
        threshold : float
            Detection threshold (optional)
            
        Returns:
        --------
        comparison_table : DataFrame
            Comparison table with metrics for each filter
        """
        results = []
        
        for filter_name, filtered_signal in filter_results.items():
            if filter_name.endswith('_weights'):
                continue  # Skip weight arrays
            
            metrics = {'Filter': filter_name}
            
            # Basic statistics
            metrics['Mean'] = np.mean(filtered_signal)
            metrics['Std'] = np.std(filtered_signal)
            metrics['RMS'] = np.sqrt(np.mean(filtered_signal**2))
            
            # Comparison with ground truth if available
            if ground_truth is not None:
                metrics['MSE'] = self.calculate_mse(filtered_signal, ground_truth)
                metrics['RMSE'] = self.calculate_rmse(filtered_signal, ground_truth)
                metrics['SNR_Improvement_dB'] = self.calculate_snr_improvement(
                    original_signal, filtered_signal, ground_truth
                )
            
            # Detection metrics if true events provided
            if true_events is not None and threshold is not None:
                detected_events = self.detect_events(filtered_signal, threshold)
                detection_metrics = self.calculate_detection_metrics(
                    detected_events, true_events
                )
                metrics['True_Positives'] = detection_metrics['true_positives']
                metrics['False_Positives'] = detection_metrics['false_positives']
                metrics['False_Negatives'] = detection_metrics['false_negatives']
                metrics['Precision'] = detection_metrics['precision']
                metrics['Recall'] = detection_metrics['recall']
                metrics['F1_Score'] = detection_metrics['f1_score']
            
            results.append(metrics)
        
        comparison_table = pd.DataFrame(results)
        return comparison_table
    
    def generate_summary_report(self, comparison_table):
        """
        Generate summary report from comparison table.
        
        Parameters:
        -----------
        comparison_table : DataFrame
            Comparison table from compare_filters
            
        Returns:
        --------
        report : dict
            Summary report with best performing filters
        """
        report = {}
        
        # Find best filter for each metric
        numeric_columns = comparison_table.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # For error metrics (lower is better)
            if any(x in col.lower() for x in ['mse', 'rmse', 'false']):
                best_idx = comparison_table[col].idxmin()
            # For performance metrics (higher is better)
            else:
                best_idx = comparison_table[col].idxmax()
            
            best_filter = comparison_table.loc[best_idx, 'Filter']
            best_value = comparison_table.loc[best_idx, col]
            
            report[f'best_{col}'] = {
                'filter': best_filter,
                'value': best_value
            }
        
        return report
