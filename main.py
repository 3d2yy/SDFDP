"""
Módulo principal para el sistema de detección de fallas mediante descargas parciales.

Este módulo integra todos los componentes del sistema y proporciona una interfaz
para procesar señales, calcular descriptores, evaluar severidad y comparar algoritmos.
"""

import numpy as np
from preprocessing import preprocess_signal
from descriptors import compute_all_descriptors
from severity import assess_severity, create_baseline_profile
from blind_algorithms import compare_algorithms, EWMA, SimpleMovingAverage, KalmanFilter1D, AdaptiveLMS, AdaptiveRLS
from validation import validate_detection_system, generate_validation_report


def generate_synthetic_signal(state='verde', duration=1000, fs=10000, noise_level=0.1):
    """
    Genera una señal sintética de descarga parcial para diferentes estados operativos.
    
    Parámetros:
    -----------
    state : str
        Estado operativo: 'verde', 'amarillo', 'naranja', 'rojo'
    duration : int
        Número de muestras
    fs : float
        Frecuencia de muestreo
    noise_level : float
        Nivel de ruido base
    
    Retorna:
    --------
    signal : ndarray
        Señal sintética generada
    """
    t = np.arange(duration) / fs
    
    # Señal base con ruido
    signal = noise_level * np.random.randn(duration)
    
    # Parámetros según el estado
    if state == 'verde':
        # Estado normal: pocas descargas de baja amplitud
        n_discharges = 5
        amplitude = 1.0
        frequency = 1000  # Hz
        
    elif state == 'amarillo':
        # Estado precaución: más descargas, amplitud moderada
        n_discharges = 15
        amplitude = 2.5
        frequency = 1500
        
    elif state == 'naranja':
        # Estado alerta: muchas descargas, amplitud alta
        n_discharges = 30
        amplitude = 4.5
        frequency = 2000
        
    else:  # rojo
        # Estado crítico: descargas frecuentes, amplitud muy alta
        n_discharges = 50
        amplitude = 7.0
        frequency = 2500
    
    # Añadir pulsos de descarga parcial
    for _ in range(n_discharges):
        # Posición aleatoria
        pos = np.random.randint(100, duration - 100)
        
        # Duración del pulso (microsegundos simulados)
        pulse_duration = int(0.01 * fs)  # 10 ms
        
        # Generar pulso oscilatorio amortiguado
        t_pulse = np.arange(pulse_duration) / fs
        decay = np.exp(-t_pulse * 500)
        pulse = amplitude * np.sin(2 * np.pi * frequency * t_pulse) * decay
        
        # Añadir a la señal
        end_pos = min(pos + pulse_duration, duration)
        signal[pos:end_pos] += pulse[:end_pos-pos]
    
    return signal


def process_and_analyze_signal(signal_data, fs, baseline_profile=None):
    """
    Procesa y analiza una señal completa.
    
    Parámetros:
    -----------
    signal_data : array-like
        Señal en bruto
    fs : float
        Frecuencia de muestreo
    baseline_profile : dict, opcional
        Perfil de línea base para comparación
    
    Retorna:
    --------
    results : dict
        Resultados completos del análisis
    """
    results = {}
    
    # 1. Preprocesamiento
    lowcut = fs * 0.01  # 1% de fs
    highcut = fs * 0.4   # 40% de fs
    
    processed_signal, processing_info = preprocess_signal(
        signal_data, fs, 
        lowcut=lowcut, 
        highcut=highcut,
        normalize=True,
        envelope=True,
        denoise=True
    )
    
    results['processing_info'] = processing_info
    results['processed_signal'] = processed_signal
    
    # 2. Cálculo de descriptores
    descriptors = compute_all_descriptors(processed_signal, fs, signal_data)
    results['descriptors'] = descriptors
    
    # 3. Evaluación de severidad
    baseline_stats = baseline_profile.get('stats') if baseline_profile else None
    baseline_severities = baseline_profile.get('severities') if baseline_profile else None
    
    severity_results = assess_severity(
        descriptors,
        baseline_stats=baseline_stats,
        baseline_severities=baseline_severities,
        threshold_method='statistical'
    )
    
    results['severity_index'] = severity_results['severity_index']
    results['descriptor_scores'] = severity_results['descriptor_scores']
    results['thresholds'] = severity_results['thresholds']
    results['traffic_light_state'] = severity_results['traffic_light_state']
    
    # 4. Comparación con algoritmos ciegos
    blind_results = compare_algorithms(processed_signal)
    results['blind_algorithms'] = blind_results
    
    return results


def evaluate_multiple_states(n_samples_per_state=10, fs=10000):
    """
    Evalúa el sistema con múltiples muestras de diferentes estados.
    
    Parámetros:
    -----------
    n_samples_per_state : int
        Número de muestras por estado
    fs : float
        Frecuencia de muestreo
    
    Retorna:
    --------
    evaluation_results : dict
        Resultados de la evaluación completa
    """
    states = ['verde', 'amarillo', 'naranja', 'rojo']
    
    # Generar señales y crear perfil de línea base (usando estado verde)
    baseline_descriptors = []
    baseline_severities = []
    
    print("Generando perfil de línea base...")
    for i in range(n_samples_per_state):
        signal = generate_synthetic_signal('verde', duration=1000, fs=fs)
        lowcut = fs * 0.01
        highcut = fs * 0.4
        processed_signal, _ = preprocess_signal(signal, fs, lowcut, highcut, True, True, True)
        descriptors = compute_all_descriptors(processed_signal, fs, signal)
        baseline_descriptors.append(descriptors)
    
    baseline_profile = {
        'stats': create_baseline_profile(baseline_descriptors),
        'severities': None  # Se calculará después
    }
    
    # Procesar todas las señales y recopilar resultados
    all_results = {state: [] for state in states}
    true_labels = []
    predicted_labels = []
    severity_indices = []
    descriptors_by_state = {state: [] for state in states}
    
    print("\nProcesando señales por estado...")
    for state in states:
        print(f"  Estado: {state}")
        for i in range(n_samples_per_state):
            signal = generate_synthetic_signal(state, duration=1000, fs=fs)
            results = process_and_analyze_signal(signal, fs, baseline_profile)
            
            all_results[state].append(results)
            true_labels.append(state)
            predicted_labels.append(results['traffic_light_state'])
            severity_indices.append(results['severity_index'])
            descriptors_by_state[state].append(results['descriptors'])
    
    # Actualizar baseline con severidades
    baseline_profile['severities'] = severity_indices[:n_samples_per_state]
    
    # Validación del sistema
    print("\nValidando sistema...")
    validation_results = validate_detection_system(
        true_labels,
        predicted_labels,
        severity_indices,
        descriptors_by_state=descriptors_by_state
    )
    
    # Comparación de algoritmos ciegos
    print("\nComparando algoritmos ciegos...")
    blind_comparison = compare_blind_algorithms_across_states(all_results, states)
    
    evaluation_results = {
        'all_results': all_results,
        'validation': validation_results,
        'blind_comparison': blind_comparison,
        'baseline_profile': baseline_profile
    }
    
    return evaluation_results


def compare_blind_algorithms_across_states(all_results, states):
    """
    Compara el rendimiento de algoritmos ciegos en diferentes estados.
    
    Parámetros:
    -----------
    all_results : dict
        Resultados por estado
    states : list
        Lista de estados
    
    Retorna:
    --------
    comparison : dict
        Comparación de algoritmos
    """
    algorithms = ['EWMA', 'SMA', 'Kalman', 'LMS', 'RLS']
    
    comparison = {alg: {state: [] for state in states} for alg in algorithms}
    
    for state in states:
        for result in all_results[state]:
            blind_results = result['blind_algorithms']
            for alg in algorithms:
                if alg in blind_results:
                    comparison[alg][state].append(blind_results[alg]['score'])
    
    # Calcular estadísticas por algoritmo y estado
    statistics = {}
    for alg in algorithms:
        statistics[alg] = {}
        for state in states:
            scores = comparison[alg][state]
            if scores:
                statistics[alg][state] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
    
    return {
        'raw_scores': comparison,
        'statistics': statistics
    }


def generate_comparative_table(blind_comparison, states):
    """
    Genera una tabla comparativa de algoritmos ciegos.
    
    Parámetros:
    -----------
    blind_comparison : dict
        Resultados de comparación de algoritmos
    states : list
        Lista de estados
    
    Retorna:
    --------
    table : str
        Tabla formateada
    """
    statistics = blind_comparison['statistics']
    algorithms = list(statistics.keys())
    
    table = []
    table.append("=" * 90)
    table.append("TABLA COMPARATIVA DE ALGORITMOS CIEGOS")
    table.append("=" * 90)
    table.append("")
    
    # Encabezado
    header = f"{'Algoritmo':<15s}"
    for state in states:
        header += f" | {state:>18s}"
    table.append(header)
    table.append("-" * 90)
    
    # Datos por algoritmo
    for alg in algorithms:
        row = f"{alg:<15s}"
        for state in states:
            if state in statistics[alg]:
                mean = statistics[alg][state]['mean']
                std = statistics[alg][state]['std']
                row += f" | {mean:8.4f} ± {std:6.4f}"
            else:
                row += f" | {'N/A':>18s}"
        table.append(row)
    
    table.append("=" * 90)
    table.append("")
    table.append("Nota: Los valores representan puntuación media ± desviación estándar")
    table.append("      Valores más bajos indican mejor capacidad de filtrado")
    table.append("")
    
    return "\n".join(table)


def print_diagnostic_summary(results):
    """
    Imprime un resumen diagnóstico de los resultados.
    
    Parámetros:
    -----------
    results : dict
        Resultados del análisis
    """
    print("\n" + "=" * 70)
    print("DIAGNÓSTICO DEL SISTEMA DE DETECCIÓN DE DESCARGAS PARCIALES")
    print("=" * 70)
    print()
    
    print("ESTADO OPERATIVO:")
    print("-" * 70)
    state = results['traffic_light_state']
    severity = results['severity_index']
    
    # Símbolo de semáforo
    symbols = {'verde': '🟢', 'amarillo': '🟡', 'naranja': '🟠', 'rojo': '🔴'}
    symbol = symbols.get(state, '⚪')
    
    print(f"  Estado:                    {symbol} {state.upper()}")
    print(f"  Índice de Severidad:       {severity:.4f}")
    print()
    
    print("DESCRIPTORES PRINCIPALES:")
    print("-" * 70)
    desc = results['descriptors']
    print(f"  Energía Total:             {desc['energy_total']:.4f}")
    print(f"  RMS:                       {desc['rms']:.4f}")
    print(f"  Curtosis:                  {desc['kurtosis']:.4f}")
    print(f"  Asimetría:                 {desc['skewness']:.4f}")
    print(f"  Factor de Cresta:          {desc['crest_factor']:.4f}")
    print(f"  Entropía Espectral:        {desc['spectral_entropy']:.4f}")
    print(f"  Estabilidad Espectral:     {desc['spectral_stability']:.4f}")
    print(f"  Conteo de Picos:           {desc['peak_count']}")
    print()
    
    if results['thresholds']:
        print("UMBRALES DE CLASIFICACIÓN:")
        print("-" * 70)
        thresholds = results['thresholds']
        print(f"  Verde → Amarillo:          {thresholds['green_yellow']:.4f}")
        print(f"  Amarillo → Naranja:        {thresholds['yellow_orange']:.4f}")
        print(f"  Naranja → Rojo:            {thresholds['orange_red']:.4f}")
        print()
    
    print("ALGORITMOS CIEGOS:")
    print("-" * 70)
    blind = results['blind_algorithms']
    for alg_name, alg_result in blind.items():
        print(f"  {alg_name:<20s}   Score: {alg_result['score']:.4f}")
    
    print()
    print("=" * 70)


def main():
    """
    Función principal de demostración.
    """
    print("=" * 70)
    print("SISTEMA DE DETECCIÓN DE FALLAS POR DESCARGAS PARCIALES UHF")
    print("=" * 70)
    print()
    
    # Parámetros
    fs = 10000  # Hz
    n_samples_per_state = 10
    
    # Evaluar sistema con múltiples estados
    evaluation_results = evaluate_multiple_states(n_samples_per_state, fs)
    
    # Mostrar resultados
    print("\n" + "=" * 70)
    print("RESULTADOS DE LA EVALUACIÓN")
    print("=" * 70)
    print()
    
    # Reporte de validación
    validation_report = generate_validation_report(evaluation_results['validation'])
    print(validation_report)
    
    # Tabla comparativa de algoritmos ciegos
    states = ['verde', 'amarillo', 'naranja', 'rojo']
    comparison_table = generate_comparative_table(
        evaluation_results['blind_comparison'],
        states
    )
    print(comparison_table)
    
    # Ejemplo de diagnóstico individual
    print("\n" + "=" * 70)
    print("EJEMPLO DE DIAGNÓSTICO INDIVIDUAL")
    print("=" * 70)
    
    # Tomar una muestra de cada estado para mostrar
    for state in states:
        print(f"\n--- Ejemplo: Estado {state.upper()} ---")
        example_result = evaluation_results['all_results'][state][0]
        print_diagnostic_summary(example_result)
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print()
    
    val = evaluation_results['validation']
    print(f"Precisión Global del Sistema:     {val['accuracy']:.2%}")
    print(f"Tasa de Falsos Positivos:         {val['false_positive_rate']:.2%}")
    print(f"Tasa de Falsos Negativos:         {val['false_negative_rate']:.2%}")
    
    if 'threshold_stability' in val:
        stab = val['threshold_stability']
        print(f"Puntuación de Estabilidad:        {stab['stability_score']:.4f}")
    
    print()
    print("✓ Sistema de detección validado exitosamente")
    print("=" * 70)
    
    return evaluation_results


if __name__ == "__main__":
    # Ejecutar sistema
    results = main()
