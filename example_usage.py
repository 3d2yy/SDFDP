"""
Example usage scripts for the UHF Partial Discharge Detection System.
"""

import numpy as np
from dp_detection_system import DPDetectionSystem, generate_synthetic_uhf_signal


def example_1_basic_detection():
    """
    Example 1: Basic partial discharge detection with synthetic signal.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Partial Discharge Detection")
    print("="*80 + "\n")
    
    # Generate synthetic signal
    signal, true_events = generate_synthetic_uhf_signal(
        duration=1e-3,
        sampling_rate=1e9,
        num_discharges=3,
        noise_level=0.05
    )
    
    print(f"Generated signal with {len(signal)} samples")
    print(f"Number of discharge events: {len(true_events)}\n")
    
    # Initialize detector
    detector = DPDetectionSystem(sampling_rate=1e9)
    
    # Process and diagnose
    diagnosis = detector.process_and_diagnose(
        signal,
        apply_filters=False  # Skip filter comparison for speed
    )
    
    # Print results
    classification = diagnosis['classification']
    print(f"Classification: {classification['classification'].upper()}")
    print(f"Severity Level: {classification['severity_level']}/3")
    print(f"Combined Index: {classification['combined_index']:.4f}")
    print(f"Message: {classification['message']}")
    
    return diagnosis


def example_2_filter_comparison():
    """
    Example 2: Compare different adaptive filters.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Adaptive Filter Comparison")
    print("="*80 + "\n")
    
    # Generate signal with more noise
    signal, true_events = generate_synthetic_uhf_signal(
        duration=1e-3,
        sampling_rate=1e9,
        num_discharges=5,
        noise_level=0.15
    )
    
    # Generate clean version for comparison
    clean_signal, _ = generate_synthetic_uhf_signal(
        duration=1e-3,
        sampling_rate=1e9,
        num_discharges=5,
        noise_level=0.0
    )
    
    print(f"Generated noisy signal (noise level: 0.15)")
    print(f"Number of discharge events: {len(true_events)}\n")
    
    # Initialize detector
    detector = DPDetectionSystem(sampling_rate=1e9)
    
    # Process with filter comparison
    diagnosis = detector.process_and_diagnose(
        signal,
        apply_filters=True,
        ground_truth=clean_signal,
        true_events=true_events
    )
    
    # Print comparison table
    if 'comparison_table' in diagnosis:
        print("Filter Comparison Results:")
        print("-"*80)
        print(diagnosis['comparison_table'].to_string(index=False))
        print()
        
        # Print best performers
        if 'summary_report' in diagnosis:
            summary = diagnosis['summary_report']
            print("Best Performing Filters:")
            print("-"*80)
            
            if 'best_SNR_Improvement_dB' in summary:
                snr_info = summary['best_SNR_Improvement_dB']
                print(f"  Best SNR Improvement: {snr_info['filter']} ({snr_info['value']:.2f} dB)")
            
            if 'best_F1_Score' in summary:
                f1_info = summary['best_F1_Score']
                print(f"  Best F1 Score: {f1_info['filter']} ({f1_info['value']:.4f})")
    
    return diagnosis


def example_3_multiple_signals():
    """
    Example 3: Process multiple signals and track historical data.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Multiple Signal Processing with Historical Data")
    print("="*80 + "\n")
    
    # Initialize detector
    detector = DPDetectionSystem(sampling_rate=1e9)
    
    # Process multiple signals with varying discharge levels
    discharge_counts = [1, 3, 5, 7, 10]
    results = []
    
    for i, num_discharges in enumerate(discharge_counts):
        signal, _ = generate_synthetic_uhf_signal(
            duration=1e-3,
            sampling_rate=1e9,
            num_discharges=num_discharges,
            noise_level=0.08
        )
        
        diagnosis = detector.process_and_diagnose(
            signal,
            apply_filters=False
        )
        
        classification = diagnosis['classification']
        results.append({
            'signal': i+1,
            'discharges': num_discharges,
            'classification': classification['classification'],
            'severity': classification['severity_level'],
            'index': classification['combined_index']
        })
        
        print(f"Signal {i+1}: {num_discharges} discharges -> "
              f"{classification['classification'].upper()} "
              f"(index: {classification['combined_index']:.4f})")
    
    print("\nThresholds after processing (dynamically adjusted):")
    for color, threshold in detector.classifier.thresholds.items():
        print(f"  {color.capitalize()}: {threshold:.4f}")
    
    return results


def example_4_custom_signal():
    """
    Example 4: Process a custom user-defined signal.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Signal Processing")
    print("="*80 + "\n")
    
    # Create a custom signal (sine wave with impulses)
    sampling_rate = 1e9
    duration = 1e-3
    n_samples = int(duration * sampling_rate)
    t = np.linspace(0, duration, n_samples)
    
    # Base signal: low frequency sine wave
    signal = 0.1 * np.sin(2 * np.pi * 10e6 * t)
    
    # Add some impulse-like disturbances
    impulse_positions = [int(0.2*n_samples), int(0.5*n_samples), int(0.8*n_samples)]
    for pos in impulse_positions:
        # Create short high-frequency burst
        burst_length = 100
        burst = 2.0 * np.sin(2 * np.pi * 800e6 * t[pos:pos+burst_length])
        signal[pos:pos+burst_length] += burst
    
    # Add noise
    signal += np.random.normal(0, 0.05, n_samples)
    
    print("Processing custom signal with impulse-like disturbances...")
    
    # Initialize detector
    detector = DPDetectionSystem(sampling_rate=sampling_rate)
    
    # Process
    diagnosis = detector.process_and_diagnose(
        signal,
        apply_filters=True
    )
    
    # Generate and print full report
    report = detector.generate_diagnostic_report(diagnosis)
    print(report)
    
    return diagnosis


def example_5_feature_analysis():
    """
    Example 5: Detailed feature analysis.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Detailed Feature Analysis")
    print("="*80 + "\n")
    
    # Generate signal
    signal, _ = generate_synthetic_uhf_signal(
        duration=1e-3,
        sampling_rate=1e9,
        num_discharges=5,
        noise_level=0.1
    )
    
    # Initialize detector
    detector = DPDetectionSystem(sampling_rate=1e9)
    
    # Process
    diagnosis = detector.process_and_diagnose(signal, apply_filters=False)
    
    # Extract and display features
    features = diagnosis['features']
    classification = diagnosis['classification']
    
    print("Extracted Features:")
    print("-"*80)
    for feature_name, value in sorted(features.items()):
        if isinstance(value, (int, float)):
            print(f"  {feature_name:25s}: {value:15.6e}")
    
    print("\nFeature Contributions to Classification:")
    print("-"*80)
    for contrib in classification['top_contributors']:
        print(f"  {contrib['feature']:25s}: {contrib['contribution']:8.4f}")
    
    return diagnosis


if __name__ == "__main__":
    """
    Run all examples.
    """
    print("\n" + "="*80)
    print("UHF PARTIAL DISCHARGE DETECTION SYSTEM - EXAMPLES")
    print("="*80)
    
    # Run examples
    example_1_basic_detection()
    example_2_filter_comparison()
    example_3_multiple_signals()
    example_4_custom_signal()
    example_5_feature_analysis()
    
    print("\n" + "="*80)
    print("All examples completed successfully!")
    print("="*80 + "\n")
