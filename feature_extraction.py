"""
Feature extraction module for partial discharge detection.
Calculates energy, frequency bands, kurtosis, skewness, RMS, 
spectral stability, and residual features.
"""

import numpy as np
from scipy import stats
from scipy.fft import fft, fftfreq


class FeatureExtractor:
    """
    Extract features from processed signals for DP detection.
    """
    
    def __init__(self, sampling_rate=1e9):
        """
        Initialize feature extractor.
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
    
    def calculate_energy(self, signal_data):
        """
        Calculate signal energy.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
            
        Returns:
        --------
        energy : float
            Signal energy
        """
        return np.sum(signal_data**2)
    
    def calculate_frequency_bands(self, signal_data, bands=None):
        """
        Calculate energy in frequency bands.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
        bands : list of tuples
            Frequency bands in Hz [(low1, high1), (low2, high2), ...]
            Default: UHF bands
            
        Returns:
        --------
        band_energies : dict
            Energy in each frequency band
        """
        if bands is None:
            # Default UHF bands for partial discharge
            bands = [
                (300e6, 600e6),   # Low UHF
                (600e6, 1000e6),  # Mid UHF
                (1000e6, 1500e6)  # High UHF
            ]
        
        # Perform FFT
        n = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(n, 1/self.sampling_rate)
        
        # Calculate power spectrum
        power = np.abs(yf)**2
        
        band_energies = {}
        for i, (low, high) in enumerate(bands):
            # Find indices within band
            band_mask = (np.abs(xf) >= low) & (np.abs(xf) <= high)
            band_energy = np.sum(power[band_mask])
            band_energies[f'band_{i+1}_{int(low/1e6)}-{int(high/1e6)}MHz'] = band_energy
        
        return band_energies
    
    def calculate_kurtosis(self, signal_data):
        """
        Calculate kurtosis (measure of tailedness).
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
            
        Returns:
        --------
        kurtosis : float
            Kurtosis value
        """
        return stats.kurtosis(signal_data, fisher=True)
    
    def calculate_skewness(self, signal_data):
        """
        Calculate skewness (measure of asymmetry).
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
            
        Returns:
        --------
        skewness : float
            Skewness value
        """
        return stats.skew(signal_data)
    
    def calculate_rms(self, signal_data):
        """
        Calculate RMS (Root Mean Square) value.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
            
        Returns:
        --------
        rms : float
            RMS value
        """
        return np.sqrt(np.mean(signal_data**2))
    
    def calculate_spectral_stability(self, signal_data, window_size=None):
        """
        Calculate spectral stability (variation of spectrum over time).
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
        window_size : int
            Window size for STFT (default: 10% of signal length)
            
        Returns:
        --------
        stability : float
            Spectral stability metric (lower is more stable)
        """
        if window_size is None:
            window_size = max(256, len(signal_data) // 10)
        
        # Ensure window_size is valid
        window_size = min(window_size, len(signal_data) // 2)
        
        if window_size < 2:
            return 0.0
        
        # Compute spectrogram using STFT
        from scipy.signal import spectrogram
        f, t, Sxx = spectrogram(signal_data, fs=self.sampling_rate, 
                                nperseg=window_size, noverlap=window_size//2)
        
        # Calculate variance of spectral content over time
        spectral_variance = np.var(Sxx, axis=1)
        stability = np.mean(spectral_variance)
        
        return stability
    
    def calculate_residual(self, signal_data, reference_data=None):
        """
        Calculate residual (difference from reference or baseline).
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
        reference_data : array-like
            Reference signal (if None, use mean as baseline)
            
        Returns:
        --------
        residual : float
            Residual metric
        """
        if reference_data is None:
            # Use mean as baseline
            baseline = np.mean(signal_data)
            residual = np.sqrt(np.mean((signal_data - baseline)**2))
        else:
            # Compare with reference
            if len(signal_data) != len(reference_data):
                raise ValueError("Signal and reference must have same length")
            residual = np.sqrt(np.mean((signal_data - reference_data)**2))
        
        return residual
    
    def extract_all_features(self, signal_data, envelope=None):
        """
        Extract all features from signal.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
        envelope : array-like
            Signal envelope (optional)
            
        Returns:
        --------
        features : dict
            Dictionary containing all extracted features
        """
        features = {}
        
        # Basic features
        features['energy'] = self.calculate_energy(signal_data)
        features['rms'] = self.calculate_rms(signal_data)
        features['kurtosis'] = self.calculate_kurtosis(signal_data)
        features['skewness'] = self.calculate_skewness(signal_data)
        
        # Frequency band features
        band_energies = self.calculate_frequency_bands(signal_data)
        features.update(band_energies)
        
        # Spectral stability
        features['spectral_stability'] = self.calculate_spectral_stability(signal_data)
        
        # Residual
        features['residual'] = self.calculate_residual(signal_data)
        
        # Envelope features if provided
        if envelope is not None:
            features['envelope_energy'] = self.calculate_energy(envelope)
            features['envelope_rms'] = self.calculate_rms(envelope)
            features['envelope_kurtosis'] = self.calculate_kurtosis(envelope)
        
        return features
