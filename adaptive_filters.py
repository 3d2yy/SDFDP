"""
Adaptive filtering algorithms for partial discharge detection.
Implements EWMA, Moving Average, Kalman Filter, LMS, and RLS.
"""

import numpy as np


class AdaptiveFilters:
    """
    Collection of adaptive filtering algorithms.
    """
    
    def __init__(self):
        """
        Initialize adaptive filters.
        """
        pass
    
    def ewma_filter(self, signal_data, alpha=0.3):
        """
        Exponentially Weighted Moving Average (EWMA) filter.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
        alpha : float
            Smoothing factor (0 < alpha <= 1)
            Higher alpha means more weight on recent values
            
        Returns:
        --------
        filtered_signal : ndarray
            Filtered signal
        """
        signal_data = np.asarray(signal_data)
        filtered = np.zeros_like(signal_data, dtype=float)
        filtered[0] = signal_data[0]
        
        for i in range(1, len(signal_data)):
            filtered[i] = alpha * signal_data[i] + (1 - alpha) * filtered[i-1]
        
        return filtered
    
    def moving_average_filter(self, signal_data, window_size=10):
        """
        Simple Moving Average filter.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
        window_size : int
            Size of the moving window
            
        Returns:
        --------
        filtered_signal : ndarray
            Filtered signal
        """
        signal_data = np.asarray(signal_data)
        window_size = min(window_size, len(signal_data))
        
        filtered = np.convolve(signal_data, np.ones(window_size)/window_size, mode='same')
        
        # Fix edges
        for i in range(window_size//2):
            filtered[i] = np.mean(signal_data[:i+window_size//2+1])
            filtered[-(i+1)] = np.mean(signal_data[-(i+window_size//2+1):])
        
        return filtered
    
    def kalman_filter(self, signal_data, process_variance=1e-5, measurement_variance=1e-2):
        """
        Kalman filter for signal smoothing.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal (measurements)
        process_variance : float
            Process noise variance (Q)
        measurement_variance : float
            Measurement noise variance (R)
            
        Returns:
        --------
        filtered_signal : ndarray
            Filtered signal (state estimates)
        """
        signal_data = np.asarray(signal_data)
        n = len(signal_data)
        
        # Initialize
        filtered = np.zeros(n)
        P = np.zeros(n)  # Error covariance
        K = np.zeros(n)  # Kalman gain
        
        # Initial values
        filtered[0] = signal_data[0]
        P[0] = 1.0
        
        for i in range(1, n):
            # Prediction
            x_pred = filtered[i-1]
            P_pred = P[i-1] + process_variance
            
            # Update
            K[i] = P_pred / (P_pred + measurement_variance)
            filtered[i] = x_pred + K[i] * (signal_data[i] - x_pred)
            P[i] = (1 - K[i]) * P_pred
        
        return filtered
    
    def lms_filter(self, signal_data, reference=None, filter_order=10, mu=0.01):
        """
        Least Mean Squares (LMS) adaptive filter.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal (desired signal)
        reference : array-like
            Reference signal (if None, uses delayed version of input)
        filter_order : int
            Number of filter taps
        mu : float
            Step size (learning rate)
            
        Returns:
        --------
        filtered_signal : ndarray
            Filtered signal (error signal)
        weights : ndarray
            Final filter weights
        """
        signal_data = np.asarray(signal_data)
        n = len(signal_data)
        
        # If no reference, use delayed version of input
        if reference is None:
            reference = np.concatenate([np.zeros(1), signal_data[:-1]])
        else:
            reference = np.asarray(reference)
        
        # Initialize
        weights = np.zeros(filter_order)
        filtered = np.zeros(n)
        
        for i in range(filter_order, n):
            # Get input vector
            x = reference[i-filter_order:i][::-1]
            
            # Filter output
            y = np.dot(weights, x)
            
            # Error
            e = signal_data[i] - y
            filtered[i] = e
            
            # Update weights
            weights += 2 * mu * e * x
        
        # Fill initial values
        filtered[:filter_order] = signal_data[:filter_order]
        
        return filtered, weights
    
    def rls_filter(self, signal_data, reference=None, filter_order=10, lambda_factor=0.99):
        """
        Recursive Least Squares (RLS) adaptive filter.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal (desired signal)
        reference : array-like
            Reference signal (if None, uses delayed version of input)
        filter_order : int
            Number of filter taps
        lambda_factor : float
            Forgetting factor (0 < lambda <= 1)
            
        Returns:
        --------
        filtered_signal : ndarray
            Filtered signal (error signal)
        weights : ndarray
            Final filter weights
        """
        signal_data = np.asarray(signal_data)
        n = len(signal_data)
        
        # If no reference, use delayed version of input
        if reference is None:
            reference = np.concatenate([np.zeros(1), signal_data[:-1]])
        else:
            reference = np.asarray(reference)
        
        # Initialize
        weights = np.zeros(filter_order)
        P = np.eye(filter_order) / 0.01  # Inverse correlation matrix
        filtered = np.zeros(n)
        
        for i in range(filter_order, n):
            # Get input vector
            x = reference[i-filter_order:i][::-1].reshape(-1, 1)
            
            # Filter output
            y = np.dot(weights, x.flatten())
            
            # Error
            e = signal_data[i] - y
            filtered[i] = e
            
            # Update
            k = np.dot(P, x) / (lambda_factor + np.dot(x.T, np.dot(P, x)))
            P = (P - np.dot(k, np.dot(x.T, P))) / lambda_factor
            weights += (e * k.flatten())
        
        # Fill initial values
        filtered[:filter_order] = signal_data[:filter_order]
        
        return filtered, weights
    
    def apply_all_filters(self, signal_data):
        """
        Apply all adaptive filters to the signal.
        
        Parameters:
        -----------
        signal_data : array-like
            Input signal
            
        Returns:
        --------
        results : dict
            Dictionary containing results from all filters
        """
        results = {}
        
        # EWMA
        results['ewma'] = self.ewma_filter(signal_data)
        
        # Moving Average
        results['moving_average'] = self.moving_average_filter(signal_data)
        
        # Kalman
        results['kalman'] = self.kalman_filter(signal_data)
        
        # LMS
        lms_filtered, lms_weights = self.lms_filter(signal_data)
        results['lms'] = lms_filtered
        results['lms_weights'] = lms_weights
        
        # RLS
        rls_filtered, rls_weights = self.rls_filter(signal_data)
        results['rls'] = rls_filtered
        results['rls_weights'] = rls_weights
        
        return results
