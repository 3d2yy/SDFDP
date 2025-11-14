"""
Módulo de evaluación de severidad y clasificación por semáforo.

Este módulo determina el nivel de severidad basado en descriptores
y asigna automáticamente un estado operativo (verde, amarillo, naranja, rojo).
"""

import numpy as np


def normalize_descriptor(value, baseline_mean, baseline_std, method='zscore'):
    """
    Normaliza un descriptor respecto a condiciones base.
    
    Parámetros:
    -----------
    value : float
        Valor del descriptor
    baseline_mean : float
        Media de la línea base
    baseline_std : float
        Desviación estándar de la línea base
    method : str, opcional
        Método de normalización (por defecto 'zscore')
    
    Retorna:
    --------
    normalized : float
        Valor normalizado
    """
    if method == 'zscore':
        if baseline_std == 0:
            return 0.0
        return (value - baseline_mean) / baseline_std
    else:
        raise ValueError(f"Método '{method}' no reconocido")


def calculate_descriptor_scores(descriptors, baseline_stats=None):
    """
    Calcula puntuaciones individuales para cada descriptor.
    
    Parámetros:
    -----------
    descriptors : dict
        Diccionario de descriptores
    baseline_stats : dict, opcional
        Estadísticas de línea base {descriptor: {'mean': val, 'std': val}}
    
    Retorna:
    --------
    scores : dict
        Puntuaciones normalizadas por descriptor
    """
    scores = {}
    
    # Descriptores que aumentan con problemas (más es peor)
    increasing_descriptors = [
        'energy_total', 'rms', 'crest_factor',
        'spectral_entropy', 'peak_count', 'residual_energy'
    ]
    
    # Descriptores que disminuyen con problemas (menos es peor)
    decreasing_descriptors = ['spectral_stability']
    
    # Descriptores donde la desviación absoluta importa
    absolute_deviation_descriptors = ['kurtosis', 'skewness']
    
    for key, value in descriptors.items():
        # Saltar descriptores de banda si no están en las listas
        if key.startswith('band_'):
            increasing_descriptors.append(key)
        
        if key in increasing_descriptors:
            # Para descriptores crecientes, normalizar y usar valor absoluto
            if baseline_stats and key in baseline_stats:
                base_mean = baseline_stats[key]['mean']
                base_std = baseline_stats[key]['std']
                score = normalize_descriptor(value, base_mean, base_std)
            else:
                # Sin línea base, usar el valor directo normalizado
                score = value
            
            # Convertir a puntuación positiva (mayor valor = mayor problema)
            scores[key] = max(0, score)
            
        elif key in decreasing_descriptors:
            # Para descriptores decrecientes, invertir la puntuación
            if baseline_stats and key in baseline_stats:
                base_mean = baseline_stats[key]['mean']
                base_std = baseline_stats[key]['std']
                score = normalize_descriptor(value, base_mean, base_std)
            else:
                score = 1.0 - value  # Invertir
            
            # Mayor desviación negativa = mayor problema
            scores[key] = max(0, -score)
            
        elif key in absolute_deviation_descriptors:
            # Para estos descriptores, cualquier desviación es un problema
            if baseline_stats and key in baseline_stats:
                base_mean = baseline_stats[key]['mean']
                base_std = baseline_stats[key]['std']
                score = normalize_descriptor(value, base_mean, base_std)
            else:
                score = value
            
            # Usar valor absoluto de la desviación
            scores[key] = abs(score)
    
    return scores


def calculate_severity_index(scores, weights=None):
    """
    Calcula un índice de severidad unificado.
    
    Parámetros:
    -----------
    scores : dict
        Puntuaciones por descriptor
    weights : dict, opcional
        Pesos para cada descriptor. Si es None, usa pesos uniformes
    
    Retorna:
    --------
    severity_index : float
        Índice de severidad combinado
    """
    if not scores:
        return 0.0
    
    # Si no hay pesos, usar pesos uniformes
    if weights is None:
        weights = {key: 1.0 for key in scores.keys()}
    
    # Calcular índice ponderado
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for key, score in scores.items():
        weight = weights.get(key, 1.0)
        weighted_sum += score * weight
        weight_sum += weight
    
    if weight_sum == 0:
        return 0.0
    
    severity_index = weighted_sum / weight_sum
    
    return severity_index


def determine_thresholds_percentile(baseline_values, percentiles=[50, 75, 90]):
    """
    Determina umbrales dinámicamente usando percentiles.
    
    Parámetros:
    -----------
    baseline_values : array-like
        Valores de línea base
    percentiles : list, opcional
        Percentiles para los umbrales [verde->amarillo, amarillo->naranja, naranja->rojo]
    
    Retorna:
    --------
    thresholds : dict
        Umbrales para cada nivel
    """
    baseline_values = np.asarray(baseline_values)
    
    thresholds = {
        'green_yellow': np.percentile(baseline_values, percentiles[0]),
        'yellow_orange': np.percentile(baseline_values, percentiles[1]),
        'orange_red': np.percentile(baseline_values, percentiles[2])
    }
    
    return thresholds


def determine_thresholds_statistical(baseline_values, sigma_multipliers=[1.5, 4.0, 8.0]):
    """
    Determina umbrales usando reglas estadísticas (múltiplos de sigma).
    
    Parámetros:
    -----------
    baseline_values : array-like
        Valores de línea base
    sigma_multipliers : list, opcional
        Múltiplos de desviación estándar [verde->amarillo, amarillo->naranja, naranja->rojo]
    
    Retorna:
    --------
    thresholds : dict
        Umbrales para cada nivel
    """
    baseline_values = np.asarray(baseline_values)
    mean = np.mean(baseline_values)
    std = np.std(baseline_values)
    
    thresholds = {
        'green_yellow': mean + sigma_multipliers[0] * std,
        'yellow_orange': mean + sigma_multipliers[1] * std,
        'orange_red': mean + sigma_multipliers[2] * std
    }
    
    return thresholds


def classify_traffic_light(severity_index, thresholds=None):
    """
    Clasifica el estado operativo en verde, amarillo, naranja o rojo.
    
    Parámetros:
    -----------
    severity_index : float
        Índice de severidad
    thresholds : dict, opcional
        Umbrales personalizados. Si es None, usa valores predeterminados
    
    Retorna:
    --------
    state : str
        Estado: 'verde', 'amarillo', 'naranja' o 'rojo'
    """
    # Umbrales predeterminados si no se proporcionan
    if thresholds is None:
        thresholds = {
            'green_yellow': 2.0,
            'yellow_orange': 6.0,
            'orange_red': 15.0
        }
    
    if severity_index < thresholds['green_yellow']:
        return 'verde'
    elif severity_index < thresholds['yellow_orange']:
        return 'amarillo'
    elif severity_index < thresholds['orange_red']:
        return 'naranja'
    else:
        return 'rojo'


def assess_severity(descriptors, baseline_stats=None, baseline_severities=None, 
                   threshold_method='statistical', custom_weights=None):
    """
    Evaluación completa de severidad con clasificación automática.
    
    Parámetros:
    -----------
    descriptors : dict
        Descriptores calculados de la señal
    baseline_stats : dict, opcional
        Estadísticas de línea base por descriptor
    baseline_severities : array-like, opcional
        Índices de severidad de línea base para cálculo de umbrales
    threshold_method : str, opcional
        Método para determinar umbrales: 'statistical' o 'percentile'
    custom_weights : dict, opcional
        Pesos personalizados para cada descriptor
    
    Retorna:
    --------
    result : dict
        Diccionario con índice de severidad, puntuaciones, umbrales y clasificación
    """
    # Calcular puntuaciones por descriptor
    scores = calculate_descriptor_scores(descriptors, baseline_stats)
    
    # Calcular índice de severidad
    severity_index = calculate_severity_index(scores, custom_weights)
    
    # Determinar umbrales
    if baseline_severities is not None:
        if threshold_method == 'percentile':
            thresholds = determine_thresholds_percentile(baseline_severities)
        else:  # statistical
            thresholds = determine_thresholds_statistical(baseline_severities)
    else:
        # Usar umbrales predeterminados
        thresholds = None
    
    # Clasificar estado
    traffic_light_state = classify_traffic_light(severity_index, thresholds)
    
    result = {
        'severity_index': severity_index,
        'descriptor_scores': scores,
        'thresholds': thresholds,
        'traffic_light_state': traffic_light_state
    }
    
    return result


def create_baseline_profile(descriptor_list):
    """
    Crea un perfil de línea base a partir de múltiples mediciones.
    
    Parámetros:
    -----------
    descriptor_list : list of dict
        Lista de diccionarios de descriptores de mediciones en condiciones normales
    
    Retorna:
    --------
    baseline_stats : dict
        Estadísticas de línea base {descriptor: {'mean': val, 'std': val}}
    """
    if not descriptor_list:
        return {}
    
    # Recopilar todos los descriptores únicos
    all_keys = set()
    for descriptors in descriptor_list:
        all_keys.update(descriptors.keys())
    
    baseline_stats = {}
    
    for key in all_keys:
        values = [d[key] for d in descriptor_list if key in d]
        if values:
            baseline_stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return baseline_stats
