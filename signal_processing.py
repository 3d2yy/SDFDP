"""
Signal processing module for UHF Partial Discharge detection.
Includes filtering, normalization, envelope detection, and noise reduction.
"""

import numpy as np
from scipy import signal
from scipy.signal import hilbert, butter, filtfilt, savgol_filter


class SignalProcessor:
    """
    Class for processing UHF signals to detect partial discharges.
    """
    
    def __init__(self, sampling_rate=1e9):
        """
        Initialize signal processor.
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling rate in Hz (default: 1 GHz for UHF signals)
        """
        self.sampling_rate = sampling_rate
    
    def bandpass_filter(self, signal_data, lowcut=300e6, highcut=1.5e9, order=5):
        """
        Apply bandpass filter to UHF signal.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
        lowcut : float
            Lower cutoff frequency (default: 300 MHz)
        highcut : float
            Upper cutoff frequency (default: 1.5 GHz)
        order : int
            Filter order
            
        Returns:
        --------
        filtered_signal : ndarray
            Filtered signal
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure cutoff frequencies are valid
        low = max(0.001, min(low, 0.999))
        high = max(0.001, min(high, 0.999))
        
        if low >= high:
            high = min(low + 0.1, 0.999)
        
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal_data)
        return filtered_signal
    
    def lowpass_filter(self, signal_data, cutoff=50e6, order=5):
        """
        Apply lowpass filter.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
        cutoff : float
            Cutoff frequency (default: 50 MHz)
        order : int
            Filter order
            
        Returns:
        --------
        filtered_signal : ndarray
            Filtered signal
        """
        nyquist = 0.5 * self.sampling_rate
        normal_cutoff = min(cutoff / nyquist, 0.999)
        b, a = butter(order, normal_cutoff, btype='low')
        filtered_signal = filtfilt(b, a, signal_data)
        return filtered_signal
    
    def normalize(self, signal_data, method='zscore'):
        """
        Normalize signal.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
        method : str
            Normalization method: 'zscore', 'minmax', or 'rms'
            
        Returns:
        --------
        normalized_signal : ndarray
            Normalized signal
        """
        signal_data = np.asarray(signal_data)
        
        if method == 'zscore':
            mean = np.mean(signal_data)
            std = np.std(signal_data)
            if std == 0:
                return signal_data - mean
            return (signal_data - mean) / std
        
        elif method == 'minmax':
            min_val = np.min(signal_data)
            max_val = np.max(signal_data)
            if max_val == min_val:
                return np.zeros_like(signal_data)
            return (signal_data - min_val) / (max_val - min_val)
        
        elif method == 'rms':
            rms = np.sqrt(np.mean(signal_data**2))
            if rms == 0:
                return signal_data
            return signal_data / rms
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def get_envelope(self, signal_data):
        """
        Extract signal envelope using Hilbert transform.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
            
        Returns:
        --------
        envelope : ndarray
            Signal envelope
        """
        analytic_signal = hilbert(signal_data)
        envelope = np.abs(analytic_signal)
        return envelope
    
    def reduce_noise(self, signal_data, method='savgol', window_length=11, polyorder=3):
        """
        Reduce noise in signal.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
        method : str
            Noise reduction method: 'savgol' or 'median'
        window_length : int
            Window length for Savitzky-Golay filter
        polyorder : int
            Polynomial order for Savitzky-Golay filter
            
        Returns:
        --------
        denoised_signal : ndarray
            Denoised signal
        """
        signal_data = np.asarray(signal_data)
        
        if method == 'savgol':
            # Ensure window_length is odd and valid
            window_length = min(window_length, len(signal_data))
            if window_length % 2 == 0:
                window_length -= 1
            window_length = max(polyorder + 2, window_length)
            
            if window_length > len(signal_data):
                window_length = len(signal_data) if len(signal_data) % 2 == 1 else len(signal_data) - 1
            
            if window_length < polyorder + 2:
                return signal_data
            
            denoised_signal = savgol_filter(signal_data, window_length, polyorder)
            return denoised_signal
        
        elif method == 'median':
            from scipy.ndimage import median_filter
            denoised_signal = median_filter(signal_data, size=window_length)
            return denoised_signal
        
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")
    
    def process_signal(self, signal_data, filter_band=True, normalize_signal=True, 
                      extract_envelope=True, reduce_noise_flag=True):
        """
        Complete signal processing pipeline.
        
        Parameters:
        -----------
        signal_data : array-like
            Raw input signal
        filter_band : bool
            Apply bandpass filter
        normalize_signal : bool
            Normalize the signal
        extract_envelope : bool
            Extract envelope
        reduce_noise_flag : bool
            Reduce noise
            
        Returns:
        --------
        processed_signal : ndarray
            Processed signal
        envelope : ndarray
            Signal envelope (if extract_envelope=True)
        """
        processed = np.asarray(signal_data).copy()
        
        # Apply bandpass filter
        if filter_band:
            processed = self.bandpass_filter(processed)
        
        # Reduce noise
        if reduce_noise_flag:
            processed = self.reduce_noise(processed)
        
        # Normalize
        if normalize_signal:
            processed = self.normalize(processed)
        
        # Extract envelope
        envelope = None
        if extract_envelope:
            envelope = self.get_envelope(processed)
        
        if extract_envelope:
            return processed, envelope
        else:
            return processed
