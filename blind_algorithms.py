"""
Módulo de algoritmos ciegos para comparación.

Este módulo implementa algoritmos de filtrado y detección sin conocimiento previo
del sistema, incluyendo EWMA, media móvil, Kalman y filtros adaptativos.
"""

import numpy as np


class EWMA:
    """
    Exponentially Weighted Moving Average (EWMA).
    """
    
    def __init__(self, alpha=0.2):
        """
        Inicializa el filtro EWMA.
        
        Parámetros:
        -----------
        alpha : float, opcional
            Factor de suavizado (0 < alpha <= 1). Por defecto 0.2
        """
        self.alpha = alpha
        self.ewma_value = None
    
    def update(self, value):
        """
        Actualiza el EWMA con un nuevo valor.
        
        Parámetros:
        -----------
        value : float
            Nuevo valor a incorporar
        
        Retorna:
        --------
        ewma : float
            Valor EWMA actualizado
        """
        if self.ewma_value is None:
            self.ewma_value = value
        else:
            self.ewma_value = self.alpha * value + (1 - self.alpha) * self.ewma_value
        
        return self.ewma_value
    
    def process_signal(self, signal_data):
        """
        Procesa una señal completa.
        
        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada
        
        Retorna:
        --------
        ewma_signal : ndarray
            Señal filtrada con EWMA
        """
        self.ewma_value = None
        ewma_signal = np.zeros(len(signal_data))
        
        for i, value in enumerate(signal_data):
            ewma_signal[i] = self.update(value)
        
        return ewma_signal
    
    def calculate_score(self, signal_data):
        """
        Calcula un puntaje de anomalía basado en desviación del EWMA.
        
        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada
        
        Retorna:
        --------
        score : float
            Puntaje de anomalía
        """
        ewma_signal = self.process_signal(signal_data)
        residuals = signal_data - ewma_signal
        
        # Usar RMS del residuo como puntaje
        score = np.sqrt(np.mean(residuals**2))
        
        return score


class SimpleMovingAverage:
    """
    Media móvil simple (SMA).
    """
    
    def __init__(self, window_size=10):
        """
        Inicializa el filtro de media móvil.
        
        Parámetros:
        -----------
        window_size : int, opcional
            Tamaño de la ventana (por defecto 10)
        """
        self.window_size = window_size
        self.buffer = []
    
    def update(self, value):
        """
        Actualiza la media móvil con un nuevo valor.
        
        Parámetros:
        -----------
        value : float
            Nuevo valor
        
        Retorna:
        --------
        sma : float
            Media móvil actualizada
        """
        self.buffer.append(value)
        
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        
        return np.mean(self.buffer)
    
    def process_signal(self, signal_data):
        """
        Procesa una señal completa.
        
        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada
        
        Retorna:
        --------
        sma_signal : ndarray
            Señal filtrada con media móvil
        """
        self.buffer = []
        sma_signal = np.zeros(len(signal_data))
        
        for i, value in enumerate(signal_data):
            sma_signal[i] = self.update(value)
        
        return sma_signal
    
    def calculate_score(self, signal_data):
        """
        Calcula un puntaje de anomalía.
        
        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada
        
        Retorna:
        --------
        score : float
            Puntaje de anomalía
        """
        sma_signal = self.process_signal(signal_data)
        residuals = signal_data - sma_signal
        
        score = np.sqrt(np.mean(residuals**2))
        
        return score


class KalmanFilter1D:
    """
    Filtro de Kalman 1D simple.
    """
    
    def __init__(self, process_variance=1e-5, measurement_variance=1e-2, 
                 initial_estimate=0.0, initial_error=1.0):
        """
        Inicializa el filtro de Kalman.
        
        Parámetros:
        -----------
        process_variance : float, opcional
            Varianza del ruido del proceso (Q)
        measurement_variance : float, opcional
            Varianza del ruido de medición (R)
        initial_estimate : float, opcional
            Estimación inicial del estado
        initial_error : float, opcional
            Error de estimación inicial
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_estimate
        self.error = initial_error
    
    def update(self, measurement):
        """
        Actualiza el filtro con una nueva medición.
        
        Parámetros:
        -----------
        measurement : float
            Nueva medición
        
        Retorna:
        --------
        estimate : float
            Estimación filtrada
        """
        # Predicción
        prediction_error = self.error + self.process_variance
        
        # Actualización
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = self.estimate + kalman_gain * (measurement - self.estimate)
        self.error = (1 - kalman_gain) * prediction_error
        
        return self.estimate
    
    def process_signal(self, signal_data):
        """
        Procesa una señal completa.
        
        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada
        
        Retorna:
        --------
        filtered_signal : ndarray
            Señal filtrada con Kalman
        """
        # Reiniciar con el primer valor
        self.estimate = signal_data[0]
        self.error = 1.0
        
        filtered_signal = np.zeros(len(signal_data))
        
        for i, measurement in enumerate(signal_data):
            filtered_signal[i] = self.update(measurement)
        
        return filtered_signal
    
    def calculate_score(self, signal_data):
        """
        Calcula un puntaje de anomalía.
        
        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada
        
        Retorna:
        --------
        score : float
            Puntaje de anomalía basado en innovación
        """
        filtered_signal = self.process_signal(signal_data)
        innovations = signal_data - filtered_signal
        
        score = np.sqrt(np.mean(innovations**2))
        
        return score


class AdaptiveLMS:
    """
    Filtro adaptativo LMS (Least Mean Squares).
    """
    
    def __init__(self, filter_order=8, mu=0.01):
        """
        Inicializa el filtro LMS.
        
        Parámetros:
        -----------
        filter_order : int, opcional
            Orden del filtro (por defecto 8)
        mu : float, opcional
            Tasa de aprendizaje (por defecto 0.01)
        """
        self.filter_order = filter_order
        self.mu = mu
        self.weights = np.zeros(filter_order)
        self.buffer = np.zeros(filter_order)
    
    def update(self, desired, input_sample):
        """
        Actualiza el filtro con nueva muestra.
        
        Parámetros:
        -----------
        desired : float
            Señal deseada
        input_sample : float
            Muestra de entrada
        
        Retorna:
        --------
        output : float
            Salida filtrada
        error : float
            Error de predicción
        """
        # Actualizar buffer
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = input_sample
        
        # Calcular salida
        output = np.dot(self.weights, self.buffer)
        
        # Calcular error
        error = desired - output
        
        # Actualizar pesos
        self.weights = self.weights + 2 * self.mu * error * self.buffer
        
        return output, error
    
    def process_signal(self, signal_data):
        """
        Procesa una señal completa.
        
        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada
        
        Retorna:
        --------
        filtered_signal : ndarray
            Señal filtrada
        errors : ndarray
            Errores de predicción
        """
        signal_data = np.asarray(signal_data)
        n = len(signal_data)
        
        # Reiniciar
        self.weights = np.zeros(self.filter_order)
        self.buffer = np.zeros(self.filter_order)
        
        filtered_signal = np.zeros(n)
        errors = np.zeros(n)
        
        # Usar señal retardada como referencia
        for i in range(n):
            if i < self.filter_order:
                filtered_signal[i] = signal_data[i]
                errors[i] = 0
            else:
                output, error = self.update(signal_data[i], signal_data[i-1])
                filtered_signal[i] = output
                errors[i] = error
        
        return filtered_signal, errors
    
    def calculate_score(self, signal_data):
        """
        Calcula un puntaje de anomalía.
        
        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada
        
        Retorna:
        --------
        score : float
            Puntaje basado en error de predicción
        """
        _, errors = self.process_signal(signal_data)
        
        # Usar RMS del error
        score = np.sqrt(np.mean(errors[self.filter_order:]**2))
        
        return score


class AdaptiveRLS:
    """
    Filtro adaptativo RLS (Recursive Least Squares) simplificado.
    """
    
    def __init__(self, filter_order=8, forgetting_factor=0.99):
        """
        Inicializa el filtro RLS.
        
        Parámetros:
        -----------
        filter_order : int, opcional
            Orden del filtro (por defecto 8)
        forgetting_factor : float, opcional
            Factor de olvido (lambda), típicamente 0.95-0.99
        """
        self.filter_order = filter_order
        self.lambda_factor = forgetting_factor
        self.weights = np.zeros(filter_order)
        self.P = np.eye(filter_order) * 1000  # Matriz de covarianza inversa
        self.buffer = np.zeros(filter_order)
    
    def update(self, desired, input_sample):
        """
        Actualiza el filtro con nueva muestra.
        
        Parámetros:
        -----------
        desired : float
            Señal deseada
        input_sample : float
            Muestra de entrada
        
        Retorna:
        --------
        output : float
            Salida filtrada
        error : float
            Error de predicción
        """
        # Actualizar buffer
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = input_sample
        
        # Calcular salida
        output = np.dot(self.weights, self.buffer)
        
        # Calcular error a priori
        error = desired - output
        
        # Cálculo del gain vector
        k = (self.P @ self.buffer) / (self.lambda_factor + self.buffer @ self.P @ self.buffer)
        
        # Actualizar pesos
        self.weights = self.weights + k * error
        
        # Actualizar matriz P
        self.P = (self.P - np.outer(k, self.buffer @ self.P)) / self.lambda_factor
        
        return output, error
    
    def process_signal(self, signal_data):
        """
        Procesa una señal completa.
        
        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada
        
        Retorna:
        --------
        filtered_signal : ndarray
            Señal filtrada
        errors : ndarray
            Errores de predicción
        """
        signal_data = np.asarray(signal_data)
        n = len(signal_data)
        
        # Reiniciar
        self.weights = np.zeros(self.filter_order)
        self.P = np.eye(self.filter_order) * 1000
        self.buffer = np.zeros(self.filter_order)
        
        filtered_signal = np.zeros(n)
        errors = np.zeros(n)
        
        for i in range(n):
            if i < self.filter_order:
                filtered_signal[i] = signal_data[i]
                errors[i] = 0
            else:
                output, error = self.update(signal_data[i], signal_data[i-1])
                filtered_signal[i] = output
                errors[i] = error
        
        return filtered_signal, errors
    
    def calculate_score(self, signal_data):
        """
        Calcula un puntaje de anomalía.
        
        Parámetros:
        -----------
        signal_data : array-like
            Señal de entrada
        
        Retorna:
        --------
        score : float
            Puntaje basado en error de predicción
        """
        _, errors = self.process_signal(signal_data)
        
        score = np.sqrt(np.mean(errors[self.filter_order:]**2))
        
        return score


def compare_algorithms(signal_data, algorithms=None):
    """
    Compara el rendimiento de múltiples algoritmos ciegos.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    algorithms : dict, opcional
        Diccionario de algoritmos {nombre: instancia}
        Si es None, usa todos los algoritmos predeterminados
    
    Retorna:
    --------
    results : dict
        Resultados comparativos por algoritmo
    """
    if algorithms is None:
        algorithms = {
            'EWMA': EWMA(alpha=0.2),
            'SMA': SimpleMovingAverage(window_size=10),
            'Kalman': KalmanFilter1D(),
            'LMS': AdaptiveLMS(filter_order=8, mu=0.01),
            'RLS': AdaptiveRLS(filter_order=8)
        }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        score = algorithm.calculate_score(signal_data)
        results[name] = {
            'score': score,
            'algorithm': name
        }
    
    return results
