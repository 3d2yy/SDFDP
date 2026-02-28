# Resumen del Sistema de Detección de Fallas

## Descripción General

Este sistema implementa un algoritmo completo para la detección de fallas en equipos eléctricos mediante el análisis de descargas parciales (DP) captadas con sensores UHF. El sistema NO utiliza análisis de tiempos entre pulsos (Δt), sino que se basa en descriptores energéticos, estadísticos y espectrales robustos.

## Arquitectura del Sistema

### 1. Preprocesamiento (`preprocessing.py`)

**Funciones principales:**
- `bandpass_filter()`: Filtro Butterworth pasabanda configurable
- `normalize_signal()`: Normalización (Z-score, Min-Max, Robusta)
- `get_envelope()`: Extracción de envolvente mediante transformada de Hilbert
- `wavelet_denoise()`: Eliminación de ruido mediante wavelets (Daubechies)
- `adaptive_filter_lms()`: Filtro adaptativo LMS para eliminación de ruido
- `preprocess_signal()`: Pipeline completo de preprocesamiento

**Características:**
- Filtrado pasabanda con control de frecuencias de corte
- Múltiples métodos de normalización para diferentes escenarios
- Eliminación de ruido robusta usando wavelets y umbralización
- Pipeline configurable que permite activar/desactivar pasos

### 2. Cálculo de Descriptores (`descriptors.py`)

**Descriptores Energéticos:**
- Energía total de la señal
- Energía por bandas espectrales (4 bandas configurables)
- Energía residual después del filtrado

**Descriptores Estadísticos:**
- Valor RMS (Root Mean Square)
- Curtosis (medida de colas de distribución)
- Asimetría (skewness)
- Factor de cresta (peak-to-RMS ratio)

**Descriptores Espectrales:**
- Entropía espectral (complejidad frecuencial)
- Estabilidad espectral (correlación entre ventanas consecutivas)

**Descriptores Temporales:**
- Conteo de picos
- Tasa de cruces por cero

**Función principal:**
- `compute_all_descriptors()`: Calcula todos los descriptores de una vez

### 3. Evaluación de Severidad (`severity.py`)

**Funciones de puntuación:**
- `calculate_descriptor_scores()`: Normaliza descriptores respecto a línea base
  - Descriptores crecientes (más = peor): energía, RMS, picos
  - Descriptores decrecientes (menos = peor): estabilidad
  - Descriptores de desviación absoluta: curtosis, asimetría

- `calculate_severity_index()`: Combina puntuaciones con pesos personalizables

**Determinación de umbrales:**
- `determine_thresholds_percentile()`: Umbrales basados en percentiles
- `determine_thresholds_statistical()`: Umbrales basados en múltiplos de σ

**Clasificación:**
- `classify_traffic_light()`: Asigna estado (verde/amarillo/naranja/rojo)
- `assess_severity()`: Evaluación completa con clasificación automática

**Umbrales predeterminados:**
- Verde → Amarillo: 2.0
- Amarillo → Naranja: 6.0
- Naranja → Rojo: 15.0

### 4. Algoritmos Ciegos (`blind_algorithms.py`)

Implementa algoritmos de filtrado sin conocimiento previo del sistema:

1. **EWMA** (Exponentially Weighted Moving Average)
   - Factor de suavizado α configurable
   - Rápido y eficiente para tendencias

2. **SMA** (Simple Moving Average)
   - Ventana deslizante configurable
   - Filtrado simple y robusto

3. **Filtro de Kalman 1D**
   - Varianzas de proceso y medición configurables
   - Óptimo para señales con ruido gaussiano

4. **LMS** (Least Mean Squares)
   - Filtro adaptativo con tasa de aprendizaje μ
   - Estabilidad mejorada con clipping de pesos

5. **RLS** (Recursive Least Squares)
   - Convergencia más rápida que LMS
   - Factor de olvido λ configurable

Cada algoritmo proporciona:
- `process_signal()`: Procesa señal completa
- `calculate_score()`: Calcula puntuación de anomalía

### 5. Validación (`validation.py`)

**Métricas de clasificación:**
- Precisión (accuracy)
- Tasa de falsos positivos (FPR)
- Tasa de falsos negativos (FNR)
- Matriz de confusión

**Métricas de separación:**
- F-ratio (varianza entre/dentro de clases)
- Cohen's d (separación por pares de clases)
- Medias y varianzas por clase

**Métricas de estabilidad:**
- Coeficiente de variación
- Desviación estándar local
- Cambios de tendencia
- Puntuación de estabilidad normalizada

**Métricas de calidad:**
- SNR efectivo
- Variación de descriptores por estado

**Función principal:**
- `validate_detection_system()`: Validación completa del sistema
- `generate_validation_report()`: Reporte formateado

### 6. Módulo Principal (`main.py`)

**Funciones de síntesis:**
- `generate_synthetic_signal()`: Genera señales sintéticas por estado
  - Verde: 3 pulsos, amplitud 0.8
  - Amarillo: 10 pulsos, amplitud 2.0
  - Naranja: 25 pulsos, amplitud 4.0
  - Rojo: 45 pulsos, amplitud 6.5

**Funciones de análisis:**
- `process_and_analyze_signal()`: Pipeline completo de análisis
- `evaluate_multiple_states()`: Evaluación con múltiples estados
- `compare_blind_algorithms_across_states()`: Comparación de algoritmos

**Funciones de presentación:**
- `generate_comparative_table()`: Tabla de algoritmos ciegos
- `print_diagnostic_summary()`: Resumen diagnóstico
- `main()`: Demostración completa del sistema

## Flujo de Trabajo

1. **Carga de señal**: Señal UHF en bruto (array de numpy)
2. **Preprocesamiento**:
   - Filtro pasabanda (elimina DC y altas frecuencias)
   - Normalización (estandariza amplitud)
   - Envolvente Hilbert (captura modulación de amplitud)
   - Eliminación de ruido (wavelets)
3. **Cálculo de descriptores**: 14 descriptores diferentes
4. **Evaluación de severidad**:
   - Normalización respecto a línea base
   - Combinación ponderada de descriptores
   - Cálculo de índice de severidad
5. **Clasificación**: Asignación automática de estado (verde/amarillo/naranja/rojo)
6. **Validación**: Métricas de rendimiento del sistema

## Pesos de Descriptores

Los descriptores tienen pesos diferentes según su importancia:

- **Conteo de picos**: 2.5 (crítico para DP)
- **Energía total**: 2.0 (muy importante)
- **RMS**: 2.0 (muy importante)
- **Factor de cresta**: 1.5 (importante)
- **Entropía espectral**: 1.5 (importante)
- **Curtosis**: 1.2 (moderado)
- **Asimetría**: 1.0 (moderado)
- **Estabilidad espectral**: 0.8 (menor peso)
- **Tasa de cruces por cero**: 0.5 (menor peso)

## Rendimiento del Sistema

Basado en evaluación con señales sintéticas:

- **Precisión**: 72.5%
- **Tasa de falsos positivos**: 0.0%
- **Tasa de falsos negativos**: 0.0%
- **F-ratio**: ~40 (excelente separación entre clases)
- **Estabilidad**: 0.55 (moderada)

**Comparación de algoritmos ciegos** (scores más bajos = mejor):
- RLS: 0.10-0.20 (mejor rendimiento)
- EWMA: 0.25-0.38
- SMA: 0.33-0.50
- Kalman: 0.55-0.83
- LMS: 5-25 (más variable, pero funcional)

## Uso Básico

```python
from main import process_and_analyze_signal
import numpy as np

# Cargar señal (ejemplo)
signal_data = np.loadtxt('señal_uhf.txt')
fs = 10000  # Hz

# Analizar
results = process_and_analyze_signal(signal_data, fs)

# Resultados
print(f"Estado: {results['traffic_light_state']}")
print(f"Severidad: {results['severity_index']:.4f}")
```

## Ventajas del Sistema

1. **Sin dependencia de Δt**: No requiere análisis de tiempos entre pulsos
2. **Robusto al ruido**: Múltiples etapas de filtrado y eliminación de ruido
3. **Descriptores múltiples**: Combina información energética, estadística y espectral
4. **Clasificación automática**: Sin necesidad de intervención manual
5. **Umbrales adaptativos**: Se ajustan según condiciones base
6. **Validación exhaustiva**: Métricas completas de rendimiento
7. **Modular**: Cada componente puede usarse independientemente
8. **Bien documentado**: Docstrings completos en español

## Limitaciones

1. **Señales sintéticas**: Rendimiento puede variar con señales reales
2. **Calibración inicial**: Requiere establecer línea base en condiciones normales
3. **Sensibilidad**: Los umbrales predeterminados pueden necesitar ajuste por aplicación
4. **LMS inestable**: En algunos casos, el filtro LMS puede divergir (mitigado con clipping)

## Extensiones Futuras

1. Soporte para múltiples sensores UHF
2. Análisis de tendencias temporales
3. Machine learning para clasificación
4. Localización de fuente de DP
5. Base de datos de patrones de falla
6. Interfaz gráfica de usuario
7. Monitoreo en tiempo real
8. Exportación de reportes PDF
