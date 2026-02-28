#!/usr/bin/env python
"""
Script de ejemplo para demostrar el uso del sistema de detección de fallas.
"""

import numpy as np
from preprocessing import preprocess_signal
from descriptors import compute_all_descriptors
from severity import assess_severity

def ejemplo_simple():
    """Ejemplo simple de uso del sistema."""
    print("=== Ejemplo de Uso del Sistema de Detección de DP ===\n")
    
    # 1. Generar señal sintética simple
    fs = 10000  # Hz
    duration = 1000
    t = np.arange(duration) / fs
    
    # Señal con ruido y algunos pulsos
    signal = 0.1 * np.random.randn(duration)
    
    # Añadir algunos pulsos de descarga parcial
    for i in [200, 400, 600, 800]:
        pulse_duration = 50
        t_pulse = np.arange(pulse_duration) / fs
        pulse = 3.0 * np.sin(2 * np.pi * 1500 * t_pulse) * np.exp(-t_pulse * 500)
        signal[i:i+pulse_duration] += pulse
    
    print(f"Señal generada: {len(signal)} muestras a {fs} Hz")
    
    # 2. Preprocesar
    print("\nPreprocesando señal...")
    processed_signal, info = preprocess_signal(
        signal, fs,
        lowcut=100,
        highcut=4000,
        normalize=True,
        envelope=True,
        denoise=True
    )
    print(f"Pasos aplicados: {list(info.keys())}")
    
    # 3. Calcular descriptores
    print("\nCalculando descriptores...")
    descriptors = compute_all_descriptors(processed_signal, fs, signal)
    
    print(f"Descriptores calculados: {len(descriptors)}")
    print(f"  - Energía total: {descriptors['energy_total']:.4f}")
    print(f"  - RMS: {descriptors['rms']:.4f}")
    print(f"  - Curtosis: {descriptors['kurtosis']:.4f}")
    print(f"  - Conteo de picos: {descriptors['peak_count']}")
    
    # 4. Evaluar severidad
    print("\nEvaluando severidad...")
    severity_result = assess_severity(descriptors)
    
    print(f"  - Índice de severidad: {severity_result['severity_index']:.4f}")
    print(f"  - Estado: {severity_result['traffic_light_state'].upper()}")
    
    # Símbolo de semáforo
    symbols = {'verde': '🟢', 'amarillo': '🟡', 'naranja': '🟠', 'rojo': '🔴'}
    symbol = symbols.get(severity_result['traffic_light_state'], '⚪')
    print(f"\n{symbol} Estado operativo: {severity_result['traffic_light_state'].upper()}")
    
    print("\n=== Ejemplo completado exitosamente ===")

if __name__ == "__main__":
    ejemplo_simple()
