"""
Script de Prueba Rápida del Sistema

Este script verifica que todos los componentes del sistema funcionen correctamente
sin iniciar la interfaz gráfica completa.
"""

import sys
import numpy as np

print("=" * 70)
print("PRUEBA RÁPIDA DEL SISTEMA")
print("=" * 70)
print()

# Test 1: Importar módulos principales
print("Test 1: Importando módulos principales...")
try:
    from main import generate_synthetic_signal, process_and_analyze_signal
    from preprocessing import preprocess_signal
    from descriptors import compute_all_descriptors
    from severity import assess_severity
    print("✓ Módulos principales importados correctamente")
except Exception as e:
    print(f"❌ Error al importar módulos: {e}")
    sys.exit(1)

# Test 2: Generar señal sintética
print("\nTest 2: Generando señal sintética...")
try:
    fs = 10000
    signal = generate_synthetic_signal('verde', duration=1000, fs=fs)
    print(f"✓ Señal generada: {len(signal)} muestras")
except Exception as e:
    print(f"❌ Error al generar señal: {e}")
    sys.exit(1)

# Test 3: Preprocesar señal
print("\nTest 3: Preprocesando señal...")
try:
    lowcut = fs * 0.01
    highcut = fs * 0.4
    processed_signal, info = preprocess_signal(signal, fs, lowcut, highcut, True, True, True)
    print(f"✓ Señal procesada: {len(processed_signal)} muestras")
except Exception as e:
    print(f"❌ Error al procesar señal: {e}")
    sys.exit(1)

# Test 4: Calcular descriptores
print("\nTest 4: Calculando descriptores...")
try:
    descriptors = compute_all_descriptors(processed_signal, fs, signal)
    print(f"✓ Descriptores calculados: {len(descriptors)} descriptores")
    print(f"   - Energía: {descriptors['energy_total']:.6f}")
    print(f"   - RMS: {descriptors['rms']:.6f}")
    print(f"   - Picos: {descriptors['peak_count']}")
except Exception as e:
    print(f"❌ Error al calcular descriptores: {e}")
    sys.exit(1)

# Test 5: Evaluar severidad
print("\nTest 5: Evaluando severidad...")
try:
    severity = assess_severity(descriptors)
    print(f"✓ Severidad evaluada:")
    print(f"   - Estado: {severity['traffic_light_state']}")
    print(f"   - Índice: {severity['severity_index']:.4f}")
except Exception as e:
    print(f"❌ Error al evaluar severidad: {e}")
    sys.exit(1)

# Test 6: Importar módulos GUI
print("\nTest 6: Verificando módulos GUI...")
try:
    from gui import live_capture, file_analysis, signal_generator
    from gui import threshold_config, documentation
    print("✓ Módulos GUI importados correctamente")
except Exception as e:
    print(f"❌ Error al importar módulos GUI: {e}")
    sys.exit(1)

# Test 7: Verificar Dash
print("\nTest 7: Verificando Dash y dependencias...")
try:
    import dash
    import dash_bootstrap_components as dbc
    import plotly.graph_objs as go
    print("✓ Dash y dependencias disponibles")
    print(f"   - Dash version: {dash.__version__}")
except Exception as e:
    print(f"❌ Error con Dash: {e}")
    sys.exit(1)

# Test 8: Análisis completo
print("\nTest 8: Análisis completo de señal...")
try:
    result = process_and_analyze_signal(signal, fs)
    print(f"✓ Análisis completo exitoso")
    print(f"   - Descriptores: {len(result['descriptors'])}")
    print(f"   - Severidad: {result['severity_index']:.4f}")
    print(f"   - Estado: {result['traffic_light_state']}")
except Exception as e:
    print(f"❌ Error en análisis completo: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("✅ TODOS LOS TESTS COMPLETADOS EXITOSAMENTE")
print("=" * 70)
print()
print("El sistema está listo para usar. Inicie la interfaz con:")
print("  python start_gui.py")
print()
print("O directamente:")
print("  python app.py")
print()
