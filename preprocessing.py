"""
Módulo de preprocesamiento de señales UHF para detección de descargas parciales.

Este módulo proporciona funciones para:
- Filtrado pasabanda
- Normalización de señales
- Extracción de envolvente mediante transformada de Hilbert
- Eliminación de ruido mediante wavelets
"""

import numpy as np
from scipy import signal
from scipy.signal import hilbert
import pywt


def bandpass_filter(signal_data, fs, lowcut, highcut, order=5):
    """
    Aplica un filtro pasabanda Butterworth a la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    fs : float
        Frecuencia de muestreo en Hz
    lowcut : float
        Frecuencia de corte inferior en Hz
    highcut : float
        Frecuencia de corte superior en Hz
    order : int, opcional
        Orden del filtro (por defecto 5)
    
    Retorna:
    --------
    filtered_signal : ndarray
        Señal filtrada
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Asegurar que las frecuencias estén en el rango válido
    low = max(0.001, min(low, 0.999))
    high = max(0.001, min(high, 0.999))
    
    if low >= high:
        raise ValueError("La frecuencia de corte inferior debe ser menor que la superior")
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, signal_data)
    
    return filtered_signal


def normalize_signal(signal_data, method='zscore'):
    """
    Normaliza la señal usando diferentes métodos.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    method : str, opcional
        Método de normalización: 'zscore', 'minmax', 'robust' (por defecto 'zscore')
    
    Retorna:
    --------
    normalized_signal : ndarray
        Señal normalizada
    """
    signal_data = np.asarray(signal_data)
    
    if method == 'zscore':
        # Normalización Z-score (media 0, desviación estándar 1)
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        if std == 0:
            return signal_data - mean
        normalized_signal = (signal_data - mean) / std
        
    elif method == 'minmax':
        # Normalización Min-Max [0, 1]
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        if max_val == min_val:
            return signal_data - min_val
        normalized_signal = (signal_data - min_val) / (max_val - min_val)
        
    elif method == 'robust':
        # Normalización robusta usando mediana y MAD
        median = np.median(signal_data)
        mad = np.median(np.abs(signal_data - median))
        if mad == 0:
            return signal_data - median
        normalized_signal = (signal_data - median) / (1.4826 * mad)
        
    else:
        raise ValueError(f"Método '{method}' no reconocido. Use 'zscore', 'minmax' o 'robust'")
    
    return normalized_signal


def get_envelope(signal_data):
    """
    Extrae la envolvente de la señal usando la transformada de Hilbert.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    envelope : ndarray
        Envolvente de la señal
    """
    analytic_signal = hilbert(signal_data)
    envelope = np.abs(analytic_signal)
    
    return envelope


def wavelet_denoise(signal_data, wavelet='db4', level=None, threshold_method='soft'):
    """
    Elimina ruido de la señal usando transformada wavelet.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    wavelet : str, opcional
        Tipo de wavelet a usar (por defecto 'db4')
    level : int, opcional
        Nivel de descomposición. Si es None, se calcula automáticamente
    threshold_method : str, opcional
        Método de umbralización: 'soft' o 'hard' (por defecto 'soft')
    
    Retorna:
    --------
    denoised_signal : ndarray
        Señal con ruido reducido
    """
    signal_data = np.asarray(signal_data)
    
    # Calcular nivel óptimo si no se proporciona
    if level is None:
        level = min(pywt.dwt_max_level(len(signal_data), wavelet), 6)
    
    # Descomposición wavelet
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    
    # Calcular umbral usando regla universal de Donoho-Johnstone
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
    
    # Aplicar umbralización a los coeficientes de detalle
    coeffs_thresholded = [coeffs[0]]  # Mantener coeficientes de aproximación
    for i in range(1, len(coeffs)):
        if threshold_method == 'soft':
            coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        else:
            coeffs_thresholded.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
    
    # Reconstruir señal
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)
    
    # Ajustar longitud si es necesario
    if len(denoised_signal) > len(signal_data):
        denoised_signal = denoised_signal[:len(signal_data)]
    elif len(denoised_signal) < len(signal_data):
        denoised_signal = np.pad(denoised_signal, (0, len(signal_data) - len(denoised_signal)), 'edge')
    
    return denoised_signal


def adaptive_filter_lms(signal_data, reference=None, mu=0.01, filter_order=32):
    """
    Filtro adaptativo LMS simple para eliminación de ruido.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada contaminada
    reference : array-like, opcional
        Señal de referencia de ruido. Si es None, se usa versión retardada de la señal
    mu : float, opcional
        Tasa de aprendizaje (por defecto 0.01)
    filter_order : int, opcional
        Orden del filtro adaptativo (por defecto 32)
    
    Retorna:
    --------
    filtered_signal : ndarray
        Señal filtrada
    """
    signal_data = np.asarray(signal_data)
    n = len(signal_data)
    
    # Si no hay señal de referencia, usar versión retardada
    if reference is None:
        reference = np.roll(signal_data, filter_order)
    
    # Inicializar pesos del filtro
    weights = np.zeros(filter_order)
    filtered_signal = np.zeros(n)
    
    # Algoritmo LMS
    for i in range(filter_order, n):
        # Vector de entrada
        x = reference[i-filter_order:i][::-1]
        
        # Salida del filtro
        y = np.dot(weights, x)
        
        # Error
        e = signal_data[i] - y
        
        # Actualizar pesos
        weights = weights + 2 * mu * e * x
        
        filtered_signal[i] = e
    
    return filtered_signal


def preprocess_signal(signal_data, fs, lowcut=None, highcut=None, 
                     normalize=True, envelope=True, denoise=True):
    """
    Pipeline completo de preprocesamiento de señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada en bruto
    fs : float
        Frecuencia de muestreo en Hz
    lowcut : float, opcional
        Frecuencia de corte inferior del filtro pasabanda
    highcut : float, opcional
        Frecuencia de corte superior del filtro pasabanda
    normalize : bool, opcional
        Si aplicar normalización (por defecto True)
    envelope : bool, opcional
        Si extraer la envolvente (por defecto True)
    denoise : bool, opcional
        Si aplicar eliminación de ruido (por defecto True)
    
    Retorna:
    --------
    processed_signal : ndarray
        Señal procesada
    processing_info : dict
        Información sobre los pasos de procesamiento aplicados
    """
    signal_data = np.asarray(signal_data)
    processing_info = {}
    
    # Filtro pasabanda si se especifican frecuencias
    if lowcut is not None and highcut is not None:
        signal_data = bandpass_filter(signal_data, fs, lowcut, highcut)
        processing_info['bandpass_filter'] = {'lowcut': lowcut, 'highcut': highcut}
    
    # Normalización
    if normalize:
        signal_data = normalize_signal(signal_data, method='zscore')
        processing_info['normalized'] = True
    
    # Extracción de envolvente
    if envelope:
        signal_data = get_envelope(signal_data)
        processing_info['envelope'] = True
    
    # Eliminación de ruido
    if denoise:
        signal_data = wavelet_denoise(signal_data)
        processing_info['denoised'] = True
    
    return signal_data, processing_info
