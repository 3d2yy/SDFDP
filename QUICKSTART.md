# V2DP - Guía Rápida de Uso

## Instalación Rápida

```bash
git clone https://github.com/3d2yy/V2DP.git
cd V2DP
pip install -r requirements.txt
```

## Uso Inmediato

### 1. Ejecutar Demo Completo
```bash
python dp_detection_system.py
```
Genera un reporte completo con señal sintética y comparación de todos los filtros.

### 2. Ejecutar Todos los Ejemplos
```bash
python example_usage.py
```
Demuestra 5 escenarios diferentes de uso.

### 3. Ejecutar Tests
```bash
python -m unittest test_dp_detection -v
```
Verifica que todo funcione correctamente (23 tests).

## Uso Programático

### Ejemplo Mínimo
```python
from dp_detection_system import DPDetectionSystem, generate_synthetic_uhf_signal

# Generar señal de prueba
signal, events = generate_synthetic_uhf_signal(
    duration=1e-3,      # 1 ms
    sampling_rate=1e9,  # 1 GHz
    num_discharges=5    # 5 descargas parciales
)

# Detectar
detector = DPDetectionSystem(sampling_rate=1e9)
diagnosis = detector.process_and_diagnose(signal)

# Ver resultado
print(diagnosis['classification']['message'])
print(f"Estado: {diagnosis['classification']['classification'].upper()}")
```

### Con Señal Real
```python
import numpy as np
from dp_detection_system import DPDetectionSystem

# Cargar tu señal UHF (ejemplo con archivo)
signal = np.loadtxt('mi_señal_uhf.txt')

# Detectar
detector = DPDetectionSystem(sampling_rate=1e9)  # Ajustar sampling_rate
diagnosis = detector.process_and_diagnose(
    signal,
    apply_filters=True
)

# Reporte completo
report = detector.generate_diagnostic_report(diagnosis)
print(report)

# Guardar tabla comparativa
if 'comparison_table' in diagnosis:
    diagnosis['comparison_table'].to_csv('comparacion_filtros.csv')
```

### Procesamiento Múltiple
```python
from dp_detection_system import DPDetectionSystem

detector = DPDetectionSystem(sampling_rate=1e9)

# Procesar múltiples señales
for i, señal in enumerate(lista_señales):
    diagnosis = detector.process_and_diagnose(señal)
    
    clasificacion = diagnosis['classification']
    print(f"Señal {i+1}: {clasificacion['classification'].upper()}")
    print(f"  Índice: {clasificacion['combined_index']:.4f}")
    print(f"  Nivel: {clasificacion['severity_level']}/3")
    print()

# Los umbrales se ajustan automáticamente con más datos
print("Umbrales ajustados:")
for color, umbral in detector.classifier.thresholds.items():
    print(f"  {color}: {umbral:.4f}")
```

## Personalización

### Ajustar Pesos de Características
```python
detector = DPDetectionSystem(sampling_rate=1e9)

# Modificar pesos del clasificador
detector.classifier.feature_weights = {
    'energy': 0.25,           # Más peso a energía
    'rms': 0.20,              # Más peso a RMS
    'kurtosis': 0.15,
    'skewness': 0.10,
    'spectral_stability': 0.10,
    'residual': 0.10,
    'band_energy': 0.10
}
```

### Ajustar Filtros
```python
from signal_processing import SignalProcessor

processor = SignalProcessor(sampling_rate=1e9)

# Filtro paso-banda personalizado
filtered = processor.bandpass_filter(
    signal,
    lowcut=400e6,   # 400 MHz
    highcut=1200e6, # 1.2 GHz
    order=6
)

# Reducción de ruido personalizada
denoised = processor.reduce_noise(
    signal,
    method='savgol',
    window_length=15,
    polyorder=4
)
```

### Ajustar Parámetros de Filtros Adaptativos
```python
from adaptive_filters import AdaptiveFilters

filters = AdaptiveFilters()

# EWMA con más suavizado
ewma_result = filters.ewma_filter(signal, alpha=0.1)

# Kalman con diferentes varianzas
kalman_result = filters.kalman_filter(
    signal,
    process_variance=1e-6,
    measurement_variance=1e-1
)

# LMS con tasa de aprendizaje ajustada
lms_result, weights = filters.lms_filter(
    signal,
    filter_order=20,
    mu=0.001  # Menor tasa = más estable
)
```

## Interpretación de Resultados

### Clasificación por Semáforo
- **🟢 Verde (0)**: Sin acción necesaria
- **🟡 Amarillo (1)**: Programar revisión próxima
- **🟠 Naranja (2)**: Investigar pronto
- **🔴 Rojo (3)**: Acción inmediata requerida

### Índice Combinado
- 0.0 - 0.25: Muy baja actividad
- 0.25 - 0.50: Baja actividad
- 0.50 - 0.75: Actividad moderada
- 0.75 - 1.00: Alta actividad

### Métricas de Filtros
- **SNR Improvement**: Mejora en dB (mayor es mejor)
- **RMSE**: Error cuadrático medio (menor es mejor)
- **F1-Score**: Balance precisión/recall (mayor es mejor)

## Troubleshooting

### Error: "Window length must be odd"
El filtro Savitzky-Golay requiere ventana impar. El código lo ajusta automáticamente.

### Warning: "Cutoff frequencies invalid"
Verifica que `lowcut < highcut` y ambos estén en el rango válido (0, Nyquist).

### Señal muy corta
Para señales < 100 muestras, algunas características pueden no ser confiables.

### Memoria insuficiente
Para señales muy largas (>10M muestras), procesar en segmentos:

```python
segment_size = 1000000
for i in range(0, len(signal), segment_size):
    segment = signal[i:i+segment_size]
    diagnosis = detector.process_and_diagnose(segment)
    # Procesar diagnosis...
```

## Más Información

- **README.md**: Documentación general y ejemplos
- **TECHNICAL_DOCS.md**: Documentación técnica detallada
- **example_usage.py**: 5 ejemplos completos
- **test_dp_detection.py**: Tests unitarios como ejemplos

## Soporte

Para reportar problemas o sugerencias, crear un issue en GitHub:
https://github.com/3d2yy/V2DP/issues
