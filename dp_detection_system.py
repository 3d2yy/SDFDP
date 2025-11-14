"""
Main diagnostic system for UHF Partial Discharge Detection.
Integrates signal processing, feature extraction, classification,
adaptive filtering, and validation.
"""

import numpy as np
import pandas as pd
from signal_processing import SignalProcessor
from feature_extraction import FeatureExtractor
from traffic_light_classifier import TrafficLightClassifier
from adaptive_filters import AdaptiveFilters
from validation import FilterValidator


class DPDetectionSystem:
    """
    Complete system for detecting and diagnosing partial discharges in UHF signals.
    """
    
    def __init__(self, sampling_rate=1e9):
        """
        Initialize DP detection system.
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling rate in Hz (default: 1 GHz)
        """
        self.sampling_rate = sampling_rate
        self.processor = SignalProcessor(sampling_rate)
        self.extractor = FeatureExtractor(sampling_rate)
        self.classifier = TrafficLightClassifier()
        self.filters = AdaptiveFilters()
        self.validator = FilterValidator()
        
        self.historical_data = {}
        self.historical_indices = []
    
    def process_and_diagnose(self, signal_data, apply_filters=True, 
                            ground_truth=None, true_events=None):
        """
        Process signal and generate complete diagnosis.
        
        Parameters:
        -----------
        signal_data : array-like
            Raw UHF signal data
        apply_filters : bool
            Whether to apply and compare adaptive filters
        ground_truth : array-like
            Ground truth signal for validation (optional)
        true_events : list
            List of true event intervals for validation (optional)
            
        Returns:
        --------
        diagnosis : dict
            Complete diagnosis with all results
        """
        diagnosis = {}
        
        # Step 1: Signal Processing
        processed_signal, envelope = self.processor.process_signal(
            signal_data,
            filter_band=True,
            normalize_signal=True,
            extract_envelope=True,
            reduce_noise_flag=True
        )
        
        diagnosis['processed_signal'] = processed_signal
        diagnosis['envelope'] = envelope
        
        # Step 2: Feature Extraction
        features = self.extractor.extract_all_features(processed_signal, envelope)
        diagnosis['features'] = features
        
        # Step 3: Classification
        combined_index, contributions = self.classifier.combine_descriptors(
            features, self.historical_data
        )
        
        classification_result = self.classifier.get_diagnosis(
            combined_index, features, contributions
        )
        
        diagnosis['classification'] = classification_result
        
        # Update historical data
        self.historical_indices.append(combined_index)
        if len(self.historical_indices) > 100:
            self.historical_indices = self.historical_indices[-100:]
        
        # Adjust thresholds dynamically if enough history
        if len(self.historical_indices) >= 10:
            self.classifier.adjust_thresholds_dynamically(self.historical_indices)
        
        # Step 4: Adaptive Filtering (if requested)
        if apply_filters:
            filter_results = self.filters.apply_all_filters(processed_signal)
            diagnosis['filter_results'] = filter_results
            
            # Step 5: Validation and Comparison
            comparison_table = self.validator.compare_filters(
                processed_signal,
                filter_results,
                ground_truth=ground_truth,
                true_events=true_events,
                threshold=np.std(processed_signal) * 3
            )
            
            diagnosis['comparison_table'] = comparison_table
            
            # Generate summary report
            summary_report = self.validator.generate_summary_report(comparison_table)
            diagnosis['summary_report'] = summary_report
        
        return diagnosis
    
    def generate_diagnostic_report(self, diagnosis):
        """
        Generate human-readable diagnostic report.
        
        Parameters:
        -----------
        diagnosis : dict
            Diagnosis results from process_and_diagnose
            
        Returns:
        --------
        report : str
            Formatted diagnostic report
        """
        report = []
        report.append("=" * 80)
        report.append("UHF PARTIAL DISCHARGE DETECTION SYSTEM - DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Classification Results
        classification = diagnosis['classification']
        report.append("CLASSIFICATION RESULTS:")
        report.append("-" * 80)
        report.append(f"Status: {classification['classification'].upper()}")
        report.append(f"Severity Level: {classification['severity_level']}/3")
        report.append(f"Combined Index: {classification['combined_index']:.4f}")
        report.append(f"Message: {classification['message']}")
        report.append("")
        
        # Thresholds
        report.append("Dynamic Thresholds:")
        for color, threshold in classification['thresholds'].items():
            report.append(f"  {color.capitalize()}: {threshold:.4f}")
        report.append("")
        
        # Top Contributors
        report.append("Top Contributing Features:")
        for contrib in classification['top_contributors']:
            report.append(f"  {contrib['feature']}: {contrib['contribution']:.4f}")
        report.append("")
        
        # Features
        report.append("EXTRACTED FEATURES:")
        report.append("-" * 80)
        features = diagnosis['features']
        for feature_name, value in features.items():
            if isinstance(value, (int, float)):
                report.append(f"  {feature_name}: {value:.6e}")
        report.append("")
        
        # Filter Comparison (if available)
        if 'comparison_table' in diagnosis:
            report.append("ADAPTIVE FILTER COMPARISON:")
            report.append("-" * 80)
            comparison_df = diagnosis['comparison_table']
            report.append(comparison_df.to_string(index=False))
            report.append("")
            
            # Best performers
            if 'summary_report' in diagnosis:
                report.append("BEST PERFORMING FILTERS:")
                report.append("-" * 80)
                summary = diagnosis['summary_report']
                
                # Group by metric type
                if 'best_SNR_Improvement_dB' in summary:
                    snr_info = summary['best_SNR_Improvement_dB']
                    report.append(f"  Best SNR Improvement: {snr_info['filter']} "
                                f"({snr_info['value']:.2f} dB)")
                
                if 'best_F1_Score' in summary:
                    f1_info = summary['best_F1_Score']
                    report.append(f"  Best Detection F1 Score: {f1_info['filter']} "
                                f"({f1_info['value']:.4f})")
                
                if 'best_RMSE' in summary:
                    rmse_info = summary['best_RMSE']
                    report.append(f"  Lowest RMSE: {rmse_info['filter']} "
                                f"({rmse_info['value']:.6e})")
                
                report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def update_historical_data(self, features):
        """
        Update historical statistics for feature normalization.
        
        Parameters:
        -----------
        features : dict
            Extracted features
        """
        for feature_name, value in features.items():
            if isinstance(value, (int, float)):
                if feature_name not in self.historical_data:
                    self.historical_data[feature_name] = {
                        'min': value,
                        'max': value,
                        'values': [value]
                    }
                else:
                    stats = self.historical_data[feature_name]
                    stats['min'] = min(stats['min'], value)
                    stats['max'] = max(stats['max'], value)
                    stats['values'].append(value)
                    
                    # Keep only recent history
                    if len(stats['values']) > 100:
                        stats['values'] = stats['values'][-100:]
                        stats['min'] = min(stats['values'])
                        stats['max'] = max(stats['values'])


def generate_synthetic_uhf_signal(duration=1e-3, sampling_rate=1e9, 
                                   num_discharges=5, noise_level=0.1):
    """
    Generate synthetic UHF signal with partial discharges for testing.
    
    Parameters:
    -----------
    duration : float
        Signal duration in seconds
    sampling_rate : float
        Sampling rate in Hz
    num_discharges : int
        Number of partial discharge events
    noise_level : float
        Noise level (standard deviation)
        
    Returns:
    --------
    signal : ndarray
        Synthetic UHF signal
    true_events : list
        List of true discharge event intervals
    """
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Base noise
    signal = np.random.normal(0, noise_level, n_samples)
    
    # Add partial discharge pulses
    true_events = []
    discharge_positions = np.random.choice(
        range(int(0.1*n_samples), int(0.9*n_samples)), 
        num_discharges, 
        replace=False
    )
    
    for pos in sorted(discharge_positions):
        # Create damped oscillation (typical PD signature)
        pulse_duration = int(100e-9 * sampling_rate)  # 100 ns pulse
        pulse_t = np.arange(pulse_duration) / sampling_rate
        
        # Frequency around 800 MHz (UHF range)
        freq = 800e6
        amplitude = np.random.uniform(0.5, 2.0)
        damping = 5e7
        
        pulse = amplitude * np.exp(-damping * pulse_t) * np.sin(2 * np.pi * freq * pulse_t)
        
        # Add pulse to signal
        end_pos = min(pos + pulse_duration, n_samples)
        actual_duration = end_pos - pos
        signal[pos:end_pos] += pulse[:actual_duration]
        
        true_events.append((pos, end_pos))
    
    return signal, true_events


def main():
    """
    Main function demonstrating the DP detection system.
    """
    print("=" * 80)
    print("UHF Partial Discharge Detection System")
    print("=" * 80)
    print()
    
    # Generate synthetic UHF signal
    print("Generating synthetic UHF signal with partial discharges...")
    signal, true_events = generate_synthetic_uhf_signal(
        duration=1e-3,
        sampling_rate=1e9,
        num_discharges=5,
        noise_level=0.05
    )
    
    # Generate ground truth (clean signal)
    clean_signal, _ = generate_synthetic_uhf_signal(
        duration=1e-3,
        sampling_rate=1e9,
        num_discharges=5,
        noise_level=0.0
    )
    
    print(f"Signal length: {len(signal)} samples")
    print(f"Number of true discharge events: {len(true_events)}")
    print()
    
    # Initialize detection system
    print("Initializing DP detection system...")
    detector = DPDetectionSystem(sampling_rate=1e9)
    print()
    
    # Process and diagnose
    print("Processing signal and generating diagnosis...")
    diagnosis = detector.process_and_diagnose(
        signal,
        apply_filters=True,
        ground_truth=clean_signal,
        true_events=true_events
    )
    print()
    
    # Generate and print report
    report = detector.generate_diagnostic_report(diagnosis)
    print(report)
    
    # Save comparison table
    if 'comparison_table' in diagnosis:
        comparison_table = diagnosis['comparison_table']
        comparison_table.to_csv('filter_comparison.csv', index=False)
        print("\nComparison table saved to 'filter_comparison.csv'")
    
    return diagnosis


if __name__ == "__main__":
    main()
