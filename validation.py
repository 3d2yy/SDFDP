"""
Módulo de validación del algoritmo de detección.

Este módulo proporciona funciones para evaluar el rendimiento del sistema,
incluyendo métricas de clasificación, estabilidad y separación entre clases.
"""

import numpy as np
from scipy import stats


def calculate_confusion_matrix(true_labels, predicted_labels, classes=None):
    """
    Calcula la matriz de confusión.
    
    Parámetros:
    -----------
    true_labels : array-like
        Etiquetas verdaderas
    predicted_labels : array-like
        Etiquetas predichas
    classes : list, opcional
        Lista de clases. Si es None, se infiere de los datos
    
    Retorna:
    --------
    confusion_matrix : dict
        Matriz de confusión estructurada
    """
    if classes is None:
        classes = sorted(set(true_labels) | set(predicted_labels))
    
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for true, pred in zip(true_labels, predicted_labels):
        i = class_to_idx[true]
        j = class_to_idx[pred]
        matrix[i, j] += 1
    
    confusion_matrix = {
        'matrix': matrix,
        'classes': classes
    }
    
    return confusion_matrix


def calculate_false_positive_rate(true_labels, predicted_labels, positive_class='rojo'):
    """
    Calcula la tasa de falsos positivos.
    
    Parámetros:
    -----------
    true_labels : array-like
        Etiquetas verdaderas
    predicted_labels : array-like
        Etiquetas predichas
    positive_class : str, opcional
        Clase considerada como positiva (por defecto 'rojo')
    
    Retorna:
    --------
    fpr : float
        Tasa de falsos positivos
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    
    # Verdaderos negativos (no son clase positiva y no se predicen como tal)
    true_negative = np.sum((true_labels != positive_class) & (predicted_labels != positive_class))
    
    # Falsos positivos (no son clase positiva pero se predicen como tal)
    false_positive = np.sum((true_labels != positive_class) & (predicted_labels == positive_class))
    
    if (true_negative + false_positive) == 0:
        return 0.0
    
    fpr = false_positive / (false_positive + true_negative)
    
    return fpr


def calculate_false_negative_rate(true_labels, predicted_labels, positive_class='rojo'):
    """
    Calcula la tasa de falsos negativos.
    
    Parámetros:
    -----------
    true_labels : array-like
        Etiquetas verdaderas
    predicted_labels : array-like
        Etiquetas predichas
    positive_class : str, opcional
        Clase considerada como positiva (por defecto 'rojo')
    
    Retorna:
    --------
    fnr : float
        Tasa de falsos negativos
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    
    # Verdaderos positivos (son clase positiva y se predicen como tal)
    true_positive = np.sum((true_labels == positive_class) & (predicted_labels == positive_class))
    
    # Falsos negativos (son clase positiva pero no se predicen como tal)
    false_negative = np.sum((true_labels == positive_class) & (predicted_labels != positive_class))
    
    if (true_positive + false_negative) == 0:
        return 0.0
    
    fnr = false_negative / (true_positive + false_negative)
    
    return fnr


def calculate_accuracy(true_labels, predicted_labels):
    """
    Calcula la precisión de clasificación.
    
    Parámetros:
    -----------
    true_labels : array-like
        Etiquetas verdaderas
    predicted_labels : array-like
        Etiquetas predichas
    
    Retorna:
    --------
    accuracy : float
        Precisión (proporción de predicciones correctas)
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    
    correct = np.sum(true_labels == predicted_labels)
    total = len(true_labels)
    
    if total == 0:
        return 0.0
    
    accuracy = correct / total
    
    return accuracy


def calculate_class_separation(descriptor_values_by_class):
    """
    Calcula la separación entre clases usando análisis discriminante.
    
    Parámetros:
    -----------
    descriptor_values_by_class : dict
        Diccionario {clase: [valores]}
    
    Retorna:
    --------
    separation_metrics : dict
        Métricas de separación entre clases
    """
    classes = list(descriptor_values_by_class.keys())
    
    # Calcular medias por clase
    means = {cls: np.mean(values) for cls, values in descriptor_values_by_class.items()}
    
    # Calcular varianzas por clase
    variances = {cls: np.var(values) for cls, values in descriptor_values_by_class.items()}
    
    # Separación entre clases adyacentes
    separations = {}
    for i in range(len(classes) - 1):
        cls1 = classes[i]
        cls2 = classes[i + 1]
        
        mean_diff = abs(means[cls2] - means[cls1])
        pooled_std = np.sqrt((variances[cls1] + variances[cls2]) / 2)
        
        if pooled_std > 0:
            # Distancia de Cohen (effect size)
            cohen_d = mean_diff / pooled_std
        else:
            cohen_d = 0
        
        separations[f'{cls1}_vs_{cls2}'] = cohen_d
    
    # Separación global (varianza entre clases / varianza dentro de clases)
    all_values = []
    class_labels = []
    for cls, values in descriptor_values_by_class.items():
        all_values.extend(values)
        class_labels.extend([cls] * len(values))
    
    if len(all_values) > 0:
        # Varianza total
        total_variance = np.var(all_values)
        
        # Varianza dentro de clases (promedio ponderado)
        within_class_variance = np.mean(list(variances.values()))
        
        if within_class_variance > 0:
            f_ratio = (total_variance - within_class_variance) / within_class_variance
        else:
            f_ratio = 0
    else:
        f_ratio = 0
    
    separation_metrics = {
        'pairwise_separations': separations,
        'f_ratio': f_ratio,
        'class_means': means,
        'class_variances': variances
    }
    
    return separation_metrics


def calculate_threshold_stability(severity_indices, window_size=10):
    """
    Calcula la estabilidad del umbral de decisión.
    
    Parámetros:
    -----------
    severity_indices : array-like
        Serie temporal de índices de severidad
    window_size : int, opcional
        Tamaño de ventana para calcular variación (por defecto 10)
    
    Retorna:
    --------
    stability_metrics : dict
        Métricas de estabilidad
    """
    severity_indices = np.asarray(severity_indices)
    
    # Variación local (desviación estándar en ventanas)
    n = len(severity_indices)
    local_stds = []
    
    for i in range(0, n - window_size + 1, window_size // 2):
        window = severity_indices[i:i+window_size]
        local_stds.append(np.std(window))
    
    if len(local_stds) > 0:
        avg_local_std = np.mean(local_stds)
        max_local_std = np.max(local_stds)
    else:
        avg_local_std = 0
        max_local_std = 0
    
    # Coeficiente de variación global
    mean_severity = np.mean(severity_indices)
    if mean_severity > 0:
        cv = np.std(severity_indices) / mean_severity
    else:
        cv = 0
    
    # Número de cambios de tendencia
    if len(severity_indices) > 1:
        diff = np.diff(severity_indices)
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
    else:
        sign_changes = 0
    
    stability_metrics = {
        'avg_local_std': avg_local_std,
        'max_local_std': max_local_std,
        'coefficient_of_variation': cv,
        'trend_changes': sign_changes,
        'stability_score': 1.0 / (1.0 + cv)  # Mayor score = más estable
    }
    
    return stability_metrics


def calculate_effective_snr(signal_data, noise_estimate=None):
    """
    Calcula el SNR efectivo de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    noise_estimate : array-like, opcional
        Estimación del ruido. Si es None, se estima de la señal
    
    Retorna:
    --------
    snr_db : float
        SNR en decibelios
    """
    signal_data = np.asarray(signal_data)
    
    if noise_estimate is None:
        # Estimar ruido usando diferencias de alto orden
        noise_estimate = np.diff(signal_data, n=2)
    
    signal_power = np.mean(signal_data**2)
    noise_power = np.mean(noise_estimate**2)
    
    if noise_power == 0:
        return np.inf
    
    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr)
    
    return snr_db


def calculate_descriptor_variation_by_state(descriptors_by_state):
    """
    Calcula la variación de descriptores por estado operativo.
    
    Parámetros:
    -----------
    descriptors_by_state : dict
        Diccionario {estado: [lista de diccionarios de descriptores]}
    
    Retorna:
    --------
    variation_metrics : dict
        Métricas de variación por descriptor y estado
    """
    variation_metrics = {}
    
    # Recopilar todos los descriptores únicos
    all_descriptors = set()
    for state, desc_list in descriptors_by_state.items():
        for desc_dict in desc_list:
            all_descriptors.update(desc_dict.keys())
    
    # Calcular variación para cada descriptor
    for descriptor in all_descriptors:
        variation_metrics[descriptor] = {}
        
        for state, desc_list in descriptors_by_state.items():
            values = [d[descriptor] for d in desc_list if descriptor in d]
            
            if len(values) > 0:
                variation_metrics[descriptor][state] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / (np.mean(values) + 1e-10),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values)
                }
    
    return variation_metrics


def validate_detection_system(true_labels, predicted_labels, severity_indices,
                              descriptors_by_state=None, signal_data=None):
    """
    Validación completa del sistema de detección.
    
    Parámetros:
    -----------
    true_labels : array-like
        Etiquetas verdaderas
    predicted_labels : array-like
        Etiquetas predichas por el sistema
    severity_indices : array-like
        Índices de severidad calculados
    descriptors_by_state : dict, opcional
        Descriptores agrupados por estado
    signal_data : array-like, opcional
        Señal original para cálculo de SNR
    
    Retorna:
    --------
    validation_results : dict
        Resultados completos de validación
    """
    validation_results = {}
    
    # Métricas de clasificación
    validation_results['accuracy'] = calculate_accuracy(true_labels, predicted_labels)
    validation_results['false_positive_rate'] = calculate_false_positive_rate(
        true_labels, predicted_labels, positive_class='rojo'
    )
    validation_results['false_negative_rate'] = calculate_false_negative_rate(
        true_labels, predicted_labels, positive_class='rojo'
    )
    
    # Matriz de confusión
    validation_results['confusion_matrix'] = calculate_confusion_matrix(
        true_labels, predicted_labels
    )
    
    # Estabilidad del umbral
    validation_results['threshold_stability'] = calculate_threshold_stability(severity_indices)
    
    # Separación entre clases
    if descriptors_by_state is not None:
        # Usar el primer descriptor como ejemplo
        descriptor_values_by_class = {}
        first_descriptor = None
        
        for state, desc_list in descriptors_by_state.items():
            if len(desc_list) > 0 and first_descriptor is None:
                first_descriptor = list(desc_list[0].keys())[0]
            
            if first_descriptor:
                descriptor_values_by_class[state] = [
                    d[first_descriptor] for d in desc_list if first_descriptor in d
                ]
        
        if descriptor_values_by_class:
            validation_results['class_separation'] = calculate_class_separation(
                descriptor_values_by_class
            )
        
        # Variación de descriptores por estado
        validation_results['descriptor_variation'] = calculate_descriptor_variation_by_state(
            descriptors_by_state
        )
    
    # SNR efectivo
    if signal_data is not None:
        validation_results['effective_snr_db'] = calculate_effective_snr(signal_data)
    
    return validation_results


def generate_validation_report(validation_results):
    """
    Genera un reporte legible de los resultados de validación.
    
    Parámetros:
    -----------
    validation_results : dict
        Resultados de validación
    
    Retorna:
    --------
    report : str
        Reporte formateado
    """
    report = []
    report.append("=" * 70)
    report.append("REPORTE DE VALIDACIÓN DEL SISTEMA DE DETECCIÓN")
    report.append("=" * 70)
    report.append("")
    
    # Métricas de clasificación
    report.append("MÉTRICAS DE CLASIFICACIÓN:")
    report.append("-" * 70)
    report.append(f"  Precisión (Accuracy):        {validation_results['accuracy']:.4f}")
    report.append(f"  Tasa de Falsos Positivos:    {validation_results['false_positive_rate']:.4f}")
    report.append(f"  Tasa de Falsos Negativos:    {validation_results['false_negative_rate']:.4f}")
    report.append("")
    
    # Estabilidad del umbral
    if 'threshold_stability' in validation_results:
        stab = validation_results['threshold_stability']
        report.append("ESTABILIDAD DEL UMBRAL:")
        report.append("-" * 70)
        report.append(f"  Puntuación de Estabilidad:   {stab['stability_score']:.4f}")
        report.append(f"  Coef. de Variación:          {stab['coefficient_of_variation']:.4f}")
        report.append(f"  Cambios de Tendencia:        {stab['trend_changes']}")
        report.append("")
    
    # Separación entre clases
    if 'class_separation' in validation_results:
        sep = validation_results['class_separation']
        report.append("SEPARACIÓN ENTRE CLASES:")
        report.append("-" * 70)
        report.append(f"  F-ratio:                     {sep['f_ratio']:.4f}")
        report.append("  Separaciones por pares:")
        for pair, cohen_d in sep['pairwise_separations'].items():
            report.append(f"    {pair:30s} {cohen_d:.4f}")
        report.append("")
    
    # SNR efectivo
    if 'effective_snr_db' in validation_results:
        report.append("RELACIÓN SEÑAL-RUIDO:")
        report.append("-" * 70)
        report.append(f"  SNR Efectivo:                {validation_results['effective_snr_db']:.2f} dB")
        report.append("")
    
    report.append("=" * 70)
    
    return "\n".join(report)
