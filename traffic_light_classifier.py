"""
Traffic light classification system for partial discharge detection.
Combines feature descriptors into a unified index and assigns 
severity levels: green/yellow/orange/red with dynamic thresholds.
"""

import numpy as np


class TrafficLightClassifier:
    """
    Classify partial discharge severity using traffic light system.
    """
    
    def __init__(self):
        """
        Initialize classifier with default thresholds.
        """
        # Default threshold ranges (will be adjusted dynamically)
        self.thresholds = {
            'green': 0.25,    # No significant DP
            'yellow': 0.50,   # Low DP activity
            'orange': 0.75,   # Moderate DP activity
            'red': 1.0        # High DP activity (critical)
        }
        
        # Feature weights for combined index
        self.feature_weights = {
            'energy': 0.20,
            'rms': 0.15,
            'kurtosis': 0.15,
            'skewness': 0.10,
            'spectral_stability': 0.15,
            'residual': 0.10,
            'band_energy': 0.15
        }
    
    def normalize_feature_value(self, value, feature_name, historical_data=None):
        """
        Normalize feature value to [0, 1] range.
        
        Parameters:
        -----------
        value : float
            Feature value to normalize
        feature_name : str
            Name of the feature
        historical_data : dict
            Historical statistics for dynamic normalization
            
        Returns:
        --------
        normalized_value : float
            Normalized value between 0 and 1
        """
        if historical_data and feature_name in historical_data:
            stats = historical_data[feature_name]
            min_val = stats.get('min', 0)
            max_val = stats.get('max', 1)
            
            if max_val == min_val:
                return 0.5
            
            normalized = (value - min_val) / (max_val - min_val)
            return np.clip(normalized, 0, 1)
        else:
            # Default normalization using absolute value
            if feature_name in ['kurtosis', 'skewness']:
                # These can be negative
                return np.clip(abs(value) / 10.0, 0, 1)
            else:
                # Assume positive values
                return np.clip(value / (value + 1), 0, 1)
    
    def combine_descriptors(self, features, historical_data=None):
        """
        Combine feature descriptors into a unified index.
        
        Parameters:
        -----------
        features : dict
            Dictionary of extracted features
        historical_data : dict
            Historical feature statistics for normalization
            
        Returns:
        --------
        combined_index : float
            Combined index value between 0 and 1
        feature_contributions : dict
            Individual feature contributions to the index
        """
        contributions = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Energy contribution
        if 'energy' in features:
            norm_value = self.normalize_feature_value(features['energy'], 'energy', historical_data)
            weight = self.feature_weights['energy']
            contributions['energy'] = norm_value * weight
            weighted_sum += contributions['energy']
            total_weight += weight
        
        # RMS contribution
        if 'rms' in features:
            norm_value = self.normalize_feature_value(features['rms'], 'rms', historical_data)
            weight = self.feature_weights['rms']
            contributions['rms'] = norm_value * weight
            weighted_sum += contributions['rms']
            total_weight += weight
        
        # Kurtosis contribution
        if 'kurtosis' in features:
            norm_value = self.normalize_feature_value(features['kurtosis'], 'kurtosis', historical_data)
            weight = self.feature_weights['kurtosis']
            contributions['kurtosis'] = norm_value * weight
            weighted_sum += contributions['kurtosis']
            total_weight += weight
        
        # Skewness contribution
        if 'skewness' in features:
            norm_value = self.normalize_feature_value(features['skewness'], 'skewness', historical_data)
            weight = self.feature_weights['skewness']
            contributions['skewness'] = norm_value * weight
            weighted_sum += contributions['skewness']
            total_weight += weight
        
        # Spectral stability contribution
        if 'spectral_stability' in features:
            norm_value = self.normalize_feature_value(features['spectral_stability'], 
                                                     'spectral_stability', historical_data)
            weight = self.feature_weights['spectral_stability']
            contributions['spectral_stability'] = norm_value * weight
            weighted_sum += contributions['spectral_stability']
            total_weight += weight
        
        # Residual contribution
        if 'residual' in features:
            norm_value = self.normalize_feature_value(features['residual'], 'residual', historical_data)
            weight = self.feature_weights['residual']
            contributions['residual'] = norm_value * weight
            weighted_sum += contributions['residual']
            total_weight += weight
        
        # Band energy contribution (average of all bands)
        band_keys = [k for k in features.keys() if k.startswith('band_')]
        if band_keys:
            band_values = [features[k] for k in band_keys]
            avg_band_energy = np.mean(band_values)
            norm_value = self.normalize_feature_value(avg_band_energy, 'band_energy', historical_data)
            weight = self.feature_weights['band_energy']
            contributions['band_energy'] = norm_value * weight
            weighted_sum += contributions['band_energy']
            total_weight += weight
        
        # Calculate combined index
        if total_weight > 0:
            combined_index = weighted_sum / total_weight
        else:
            combined_index = 0.0
        
        return combined_index, contributions
    
    def adjust_thresholds_dynamically(self, historical_indices):
        """
        Adjust thresholds dynamically based on historical data.
        
        Parameters:
        -----------
        historical_indices : array-like
            Historical combined index values
        """
        if len(historical_indices) < 10:
            return  # Need enough data for statistical analysis
        
        indices = np.array(historical_indices)
        
        # Calculate percentiles for dynamic thresholds
        self.thresholds['green'] = np.percentile(indices, 25)
        self.thresholds['yellow'] = np.percentile(indices, 50)
        self.thresholds['orange'] = np.percentile(indices, 75)
        self.thresholds['red'] = np.percentile(indices, 90)
        
        # Ensure proper ordering
        self.thresholds['green'] = min(self.thresholds['green'], 0.25)
        self.thresholds['yellow'] = max(self.thresholds['yellow'], self.thresholds['green'] + 0.1)
        self.thresholds['orange'] = max(self.thresholds['orange'], self.thresholds['yellow'] + 0.1)
        self.thresholds['red'] = max(self.thresholds['red'], self.thresholds['orange'] + 0.1)
    
    def classify(self, combined_index):
        """
        Classify partial discharge severity using traffic light system.
        
        Parameters:
        -----------
        combined_index : float
            Combined index value
            
        Returns:
        --------
        classification : str
            Classification: 'green', 'yellow', 'orange', or 'red'
        severity_level : int
            Severity level: 0 (green), 1 (yellow), 2 (orange), 3 (red)
        """
        if combined_index < self.thresholds['green']:
            return 'green', 0
        elif combined_index < self.thresholds['yellow']:
            return 'yellow', 1
        elif combined_index < self.thresholds['orange']:
            return 'orange', 2
        else:
            return 'red', 3
    
    def get_diagnosis(self, combined_index, features, contributions):
        """
        Generate detailed diagnosis message.
        
        Parameters:
        -----------
        combined_index : float
            Combined index value
        features : dict
            Extracted features
        contributions : dict
            Feature contributions to the index
            
        Returns:
        --------
        diagnosis : dict
            Detailed diagnosis information
        """
        classification, severity_level = self.classify(combined_index)
        
        # Determine primary contributors
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        top_contributors = sorted_contributions[:3]
        
        diagnosis = {
            'classification': classification,
            'severity_level': severity_level,
            'combined_index': combined_index,
            'thresholds': self.thresholds.copy(),
            'top_contributors': [
                {'feature': name, 'contribution': value} 
                for name, value in top_contributors
            ],
            'features': features,
            'message': self._generate_message(classification, severity_level, top_contributors)
        }
        
        return diagnosis
    
    def _generate_message(self, classification, severity_level, top_contributors):
        """
        Generate human-readable diagnosis message.
        
        Parameters:
        -----------
        classification : str
            Traffic light color
        severity_level : int
            Severity level
        top_contributors : list
            Top contributing features
            
        Returns:
        --------
        message : str
            Diagnosis message
        """
        messages = {
            'green': 'No significant partial discharge activity detected. System is operating normally.',
            'yellow': 'Low partial discharge activity detected. Monitor the system and schedule inspection.',
            'orange': 'Moderate partial discharge activity detected. Investigation recommended soon.',
            'red': 'High partial discharge activity detected! Immediate inspection and maintenance required.'
        }
        
        base_message = messages.get(classification, 'Unknown classification')
        
        if top_contributors:
            contributors_str = ', '.join([f"{name}" for name, _ in top_contributors])
            base_message += f" Primary indicators: {contributors_str}."
        
        return base_message
