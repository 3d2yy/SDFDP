# V2DP - UHF Partial Discharge Detection System

Sistema completo en Python para detectar descargas parciales (DP) en señales UHF.

## Características

### Procesamiento de Señales
- **Filtrado**: Filtros paso-banda y paso-bajo para señales UHF (300 MHz - 1.5 GHz)
- **Normalización**: Múltiples métodos (z-score, minmax, RMS)
- **Envolvente**: Detección de envolvente mediante transformada de Hilbert
- **Reducción de ruido**: Filtros Savitzky-Golay y mediana

### Extracción de Características
- **Energía**: Cálculo de energía total de la señal
- **Bandas de frecuencia**: Análisis de energía en bandas UHF específicas
- **Curtosis**: Medida de la forma de la distribución
- **Asimetría (Skewness)**: Medida de la asimetría de la distribución
- **RMS**: Valor eficaz de la señal
- **Estabilidad espectral**: Variación del espectro en el tiempo
- **Residual**: Diferencia respecto a señal de referencia

### Sistema de Clasificación (Semáforo)
- **Índice combinado**: Combina múltiples descriptores con pesos ajustables
- **Umbrales dinámicos**: Se ajustan automáticamente basados en datos históricos
- **Niveles de severidad**:
  - 🟢 **Verde**: Sin actividad significativa de DP
  - 🟡 **Amarillo**: Actividad baja de DP (monitoreo)
  - 🟠 **Naranja**: Actividad moderada de DP (investigación recomendada)
  - 🔴 **Rojo**: Actividad alta de DP (inspección inmediata requerida)

### Filtros Adaptativos
Implementa y compara múltiples algoritmos:
- **EWMA** (Exponentially Weighted Moving Average)
- **Media Móvil** (Moving Average)
- **Filtro de Kalman** (Kalman Filter)
- **LMS** (Least Mean Squares)
- **RLS** (Recursive Least Squares)

### Validación y Comparación
- **Métricas de detección**: Verdaderos Positivos (TP), Falsos Positivos (FP), Falsos Negativos (FN)
- **Precisión y Recall**: Evaluación de la calidad de detección
- **Mejora de SNR**: Cálculo de mejora en relación señal-ruido
- **Tabla comparativa**: Comparación detallada de todos los filtros

## Instalación

```bash
pip install -r requirements.txt
```

## Uso Básico

```python
from dp_detection_system import DPDetectionSystem, generate_synthetic_uhf_signal

# Generar señal sintética de prueba
signal, true_events = generate_synthetic_uhf_signal(
    duration=1e-3,        # 1 ms
    sampling_rate=1e9,    # 1 GHz
    num_discharges=5,     # 5 descargas parciales
    noise_level=0.1       # Nivel de ruido
)

# Inicializar sistema de detección
detector = DPDetectionSystem(sampling_rate=1e9)

# Procesar y diagnosticar
diagnosis = detector.process_and_diagnose(
    signal,
    apply_filters=True,
    ground_truth=None,
    true_events=None
)

# Generar reporte
report = detector.generate_diagnostic_report(diagnosis)
print(report)
```

## Ejemplo Completo

Ejecutar el script principal:

```bash
python dp_detection_system.py
```

Esto generará:
1. Un reporte completo en consola con:
   - Clasificación de severidad (verde/amarillo/naranja/rojo)
   - Índice combinado y umbrales dinámicos
   - Características principales contribuyentes
   - Todas las características extraídas
   - Comparación de filtros adaptativos
   - Mejor desempeño por métrica

2. Un archivo CSV (`filter_comparison.csv`) con tabla comparativa detallada

## Estructura del Proyecto

```
V2DP/
├── signal_processing.py         # Procesamiento de señales UHF
├── feature_extraction.py        # Extracción de características
├── traffic_light_classifier.py  # Sistema de clasificación por semáforo
├── adaptive_filters.py          # Filtros adaptativos (EWMA, MA, Kalman, LMS, RLS)
├── validation.py                # Validación y comparación
├── dp_detection_system.py       # Sistema principal integrado
├── example_usage.py             # Ejemplos de uso
├── requirements.txt             # Dependencias
└── README.md                    # Este archivo
```

## Módulos Principales

### SignalProcessor
Procesamiento de señales UHF con filtrado, normalización y reducción de ruido.

### FeatureExtractor
Extracción de características relevantes para detección de DP.

### TrafficLightClassifier
Sistema de clasificación con umbrales dinámicos y semáforo de severidad.

### AdaptiveFilters
Colección de algoritmos de filtrado adaptativo para comparación.

### FilterValidator
Validación y comparación de desempeño de filtros.

### DPDetectionSystem
Sistema principal que integra todos los módulos.

## Características Técnicas

### Sin usar Δt
El sistema no requiere información explícita de intervalos de tiempo (Δt) entre eventos. Todas las métricas y algoritmos operan directamente sobre las muestras de la señal.

### Umbrales Dinámicos
Los umbrales para la clasificación se ajustan automáticamente basados en datos históricos usando percentiles (25%, 50%, 75%, 90%).

### Validación Completa
- Cálculo de FP, FN, TP
- Precisión, Recall, F1-Score
- Mejora de SNR en dB
- RMSE y MSE

## Ejemplo de Salida

```
================================================================================
UHF PARTIAL DISCHARGE DETECTION SYSTEM - DIAGNOSTIC REPORT
================================================================================

CLASSIFICATION RESULTS:
--------------------------------------------------------------------------------
Status: ORANGE
Severity Level: 2/3
Combined Index: 0.6543
Message: Moderate partial discharge activity detected. Investigation recommended soon. 
Primary indicators: energy, rms, spectral_stability.

Dynamic Thresholds:
  Green: 0.2500
  Yellow: 0.5000
  Orange: 0.7500
  Red: 0.9000

EXTRACTED FEATURES:
--------------------------------------------------------------------------------
  energy: 2.345678e+02
  rms: 4.567890e-01
  kurtosis: 3.456789e+00
  ...

ADAPTIVE FILTER COMPARISON:
--------------------------------------------------------------------------------
         Filter      Mean       Std       RMS  SNR_Improvement_dB  ...
           ewma  0.000123  0.456789  0.456789               12.34  ...
  moving_average  0.000234  0.345678  0.345678               15.67  ...
         kalman  0.000345  0.234567  0.234567               18.90  ...
            lms  0.000456  0.123456  0.123456               21.23  ...
            rls  0.000567  0.098765  0.098765               23.45  ...

BEST PERFORMING FILTERS:
--------------------------------------------------------------------------------
  Best SNR Improvement: rls (23.45 dB)
  Best Detection F1 Score: kalman (0.9234)
  Lowest RMSE: rls (9.876543e-03)
```

## Licencia

Ver archivo LICENSE.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request.