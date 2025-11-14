"""
Módulo de cálculo de descriptores para análisis de descargas parciales.

Este módulo calcula descriptores energéticos, estadísticos y espectrales
que caracterizan el estado de la señal de descarga parcial.
"""

import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq


def energy_total(signal_data):
    """
    Calcula la energía total de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    energy : float
        Energía total de la señal
    """
    return np.sum(np.square(signal_data))


def energy_windowed(signal_data, window_size):
    """
    Calcula la energía en ventanas deslizantes.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    window_size : int
        Tamaño de la ventana
    
    Retorna:
    --------
    energies : ndarray
        Vector de energías por ventana
    """
    signal_data = np.asarray(signal_data)
    n = len(signal_data)
    n_windows = n - window_size + 1
    
    energies = np.zeros(n_windows)
    for i in range(n_windows):
        window = signal_data[i:i+window_size]
        energies[i] = np.sum(np.square(window))
    
    return energies


def energy_spectral_bands(signal_data, fs, bands=None):
    """
    Calcula la energía en bandas de frecuencia específicas.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    fs : float
        Frecuencia de muestreo
    bands : list of tuples, opcional
        Lista de bandas de frecuencia [(f1_low, f1_high), (f2_low, f2_high), ...]
        Si es None, usa bandas predeterminadas
    
    Retorna:
    --------
    band_energies : dict
        Diccionario con energías por banda
    """
    signal_data = np.asarray(signal_data)
    n = len(signal_data)
    
    # Calcular FFT
    fft_values = fft(signal_data)
    fft_freq = fftfreq(n, 1/fs)
    power_spectrum = np.abs(fft_values)**2
    
    # Usar solo frecuencias positivas
    positive_freq_idx = fft_freq >= 0
    fft_freq = fft_freq[positive_freq_idx]
    power_spectrum = power_spectrum[positive_freq_idx]
    
    # Bandas predeterminadas si no se especifican
    if bands is None:
        bands = [
            (0, fs/8),       # Banda baja
            (fs/8, fs/4),    # Banda media-baja
            (fs/4, 3*fs/8),  # Banda media-alta
            (3*fs/8, fs/2)   # Banda alta
        ]
    
    band_energies = {}
    for i, (f_low, f_high) in enumerate(bands):
        band_mask = (fft_freq >= f_low) & (fft_freq < f_high)
        band_energy = np.sum(power_spectrum[band_mask])
        band_energies[f'band_{i+1}_{f_low:.1f}_{f_high:.1f}Hz'] = band_energy
    
    return band_energies


def rms_value(signal_data):
    """
    Calcula el valor RMS (Root Mean Square) de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    rms : float
        Valor RMS de la señal
    """
    return np.sqrt(np.mean(np.square(signal_data)))


def kurtosis(signal_data):
    """
    Calcula la curtosis de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    kurt : float
        Curtosis de la señal
    """
    return stats.kurtosis(signal_data, fisher=True)


def skewness(signal_data):
    """
    Calcula la asimetría de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    skew : float
        Asimetría de la señal
    """
    return stats.skew(signal_data)


def crest_factor(signal_data):
    """
    Calcula el factor de cresta de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    cf : float
        Factor de cresta
    """
    peak = np.max(np.abs(signal_data))
    rms = rms_value(signal_data)
    
    if rms == 0:
        return 0
    
    return peak / rms


def spectral_entropy(signal_data, fs):
    """
    Calcula la entropía espectral de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    fs : float
        Frecuencia de muestreo
    
    Retorna:
    --------
    entropy : float
        Entropía espectral
    """
    signal_data = np.asarray(signal_data)
    n = len(signal_data)
    
    # Calcular espectro de potencia
    fft_values = fft(signal_data)
    power_spectrum = np.abs(fft_values[:n//2])**2
    
    # Normalizar como distribución de probabilidad
    power_spectrum = power_spectrum / np.sum(power_spectrum)
    
    # Evitar log(0)
    power_spectrum = power_spectrum[power_spectrum > 0]
    
    # Calcular entropía
    entropy = -np.sum(power_spectrum * np.log2(power_spectrum))
    
    return entropy


def spectral_stability(signal_data, fs, window_size=None):
    """
    Calcula la estabilidad espectral entre ventanas consecutivas.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    fs : float
        Frecuencia de muestreo
    window_size : int, opcional
        Tamaño de la ventana. Si es None, se divide en 4 ventanas
    
    Retorna:
    --------
    stability : float
        Medida de estabilidad espectral (0-1, donde 1 es más estable)
    """
    signal_data = np.asarray(signal_data)
    n = len(signal_data)
    
    if window_size is None:
        window_size = n // 4
    
    if window_size < 16:
        window_size = min(16, n // 2)
    
    # Dividir señal en ventanas
    n_windows = n // window_size
    if n_windows < 2:
        return 1.0  # Solo una ventana, perfectamente estable
    
    spectra = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window = signal_data[start:end]
        
        # Calcular espectro normalizado
        fft_values = fft(window)
        spectrum = np.abs(fft_values[:window_size//2])
        spectrum = spectrum / (np.sum(spectrum) + 1e-10)
        spectra.append(spectrum)
    
    # Calcular correlación promedio entre espectros consecutivos
    correlations = []
    for i in range(len(spectra) - 1):
        corr = np.corrcoef(spectra[i], spectra[i+1])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    if len(correlations) == 0:
        return 1.0
    
    stability = np.mean(correlations)
    
    # Asegurar que esté en el rango [0, 1]
    stability = max(0, min(1, stability))
    
    return stability


def residual_energy(original_signal, filtered_signal):
    """
    Calcula la energía residual después del filtrado.
    
    Parámetros:
    -----------
    original_signal : array-like
        Señal original
    filtered_signal : array-like
        Señal filtrada
    
    Retorna:
    --------
    residual : float
        Energía del residuo
    """
    residual = np.asarray(original_signal) - np.asarray(filtered_signal)
    return energy_total(residual)


def peak_count(signal_data, threshold=None):
    """
    Cuenta el número de picos en la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    threshold : float, opcional
        Umbral para detección de picos. Si es None, usa la media + 2*std
    
    Retorna:
    --------
    n_peaks : int
        Número de picos detectados
    """
    signal_data = np.asarray(signal_data)
    
    if threshold is None:
        threshold = np.mean(signal_data) + 2 * np.std(signal_data)
    
    peaks, _ = signal.find_peaks(signal_data, height=threshold)
    
    return len(peaks)


def zero_crossing_rate(signal_data):
    """
    Calcula la tasa de cruces por cero.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal de entrada
    
    Retorna:
    --------
    zcr : float
        Tasa de cruces por cero (normalizada)
    """
    signal_data = np.asarray(signal_data)
    
    # Contar cambios de signo
    signs = np.sign(signal_data)
    zero_crossings = np.sum(np.abs(np.diff(signs))) / 2
    
    # Normalizar por la longitud
    zcr = zero_crossings / (len(signal_data) - 1)
    
    return zcr


def compute_all_descriptors(signal_data, fs, original_signal=None):
    """
    Calcula todos los descriptores de la señal.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal procesada
    fs : float
        Frecuencia de muestreo
    original_signal : array-like, opcional
        Señal original para calcular energía residual
    
    Retorna:
    --------
    descriptors : dict
        Diccionario con todos los descriptores
    """
    descriptors = {}
    
    # Descriptores energéticos
    descriptors['energy_total'] = energy_total(signal_data)
    
    # Energía por bandas espectrales
    band_energies = energy_spectral_bands(signal_data, fs)
    descriptors.update(band_energies)
    
    # Descriptores estadísticos
    descriptors['rms'] = rms_value(signal_data)
    descriptors['kurtosis'] = kurtosis(signal_data)
    descriptors['skewness'] = skewness(signal_data)
    descriptors['crest_factor'] = crest_factor(signal_data)
    
    # Descriptores espectrales
    descriptors['spectral_entropy'] = spectral_entropy(signal_data, fs)
    descriptors['spectral_stability'] = spectral_stability(signal_data, fs)
    
    # Otros descriptores
    descriptors['peak_count'] = peak_count(signal_data)
    descriptors['zero_crossing_rate'] = zero_crossing_rate(signal_data)
    
    # Energía residual si se proporciona señal original
    if original_signal is not None:
        descriptors['residual_energy'] = residual_energy(original_signal, signal_data)
    
    return descriptors
