# Sistema de Detección de Descargas Parciales UHF - Documentación Técnica

## Resumen Ejecutivo

Este proyecto implementa un sistema completo en Python para la detección y diagnóstico de descargas parciales (DP) en señales UHF, cumpliendo con todos los requisitos especificados.

## Características Implementadas

### 1. Procesamiento de Señales UHF (`signal_processing.py`)

**Filtrado:**
- Filtro paso-banda Butterworth (300 MHz - 1.5 GHz) para señales UHF
- Filtro paso-bajo configurable para eliminación de componentes de alta frecuencia
- Implementación con `scipy.signal.filtfilt` para respuesta de fase cero

**Normalización:**
- Z-score: Normalización estadística (μ=0, σ=1)
- MinMax: Escalado al rango [0, 1]
- RMS: Normalización por valor eficaz

**Envolvente:**
- Transformada de Hilbert para extracción de envolvente analítica
- Permite análisis de modulación de amplitud de señales DP

**Reducción de Ruido:**
- Filtro Savitzky-Golay con ventana y orden configurables
- Filtro de mediana para eliminación de ruido impulsivo

### 2. Extracción de Características (`feature_extraction.py`)

**Características Temporales:**
- **Energía**: ∑(x²) - Indicador de intensidad de actividad DP
- **RMS**: √(mean(x²)) - Valor efectivo de la señal
- **Residual**: Desviación respecto a referencia o línea base

**Características Estadísticas:**
- **Curtosis**: Medida de cola pesada (detección de picos)
- **Asimetría (Skewness)**: Medida de asimetría de distribución

**Características Frecuenciales:**
- **Bandas UHF**: Energía en tres bandas específicas
  - Banda 1: 300-600 MHz (UHF baja)
  - Banda 2: 600-1000 MHz (UHF media)
  - Banda 3: 1000-1500 MHz (UHF alta)
- **Estabilidad Espectral**: Varianza del espectrograma a lo largo del tiempo

### 3. Sistema de Clasificación por Semáforo (`traffic_light_classifier.py`)

**Índice Combinado:**
Combina descriptores con pesos ajustables:
- Energía: 20%
- RMS: 15%
- Curtosis: 15%
- Asimetría: 10%
- Estabilidad espectral: 15%
- Residual: 10%
- Energía de bandas: 15%

**Umbrales Dinámicos:**
Se ajustan automáticamente usando percentiles de datos históricos:
- Verde: Percentil 25 (≤0.25 por defecto)
- Amarillo: Percentil 50 (≤0.50 por defecto)
- Naranja: Percentil 75 (≤0.75 por defecto)
- Rojo: Percentil 90 (>0.75 por defecto)

**Niveles de Severidad:**
- 🟢 **Verde (0)**: Sin actividad significativa de DP
- 🟡 **Amarillo (1)**: Actividad baja - monitoreo requerido
- 🟠 **Naranja (2)**: Actividad moderada - investigación recomendada
- 🔴 **Rojo (3)**: Actividad alta - inspección inmediata

### 4. Filtros Adaptativos (`adaptive_filters.py`)

**EWMA (Exponentially Weighted Moving Average):**
- Promedio ponderado exponencialmente
- Factor de suavizado α configurable
- Ecuación: y[i] = α·x[i] + (1-α)·y[i-1]

**Media Móvil (Moving Average):**
- Promedio simple en ventana deslizante
- Tamaño de ventana configurable
- Implementación eficiente con convolución

**Filtro de Kalman:**
- Estimación óptima bajo ruido Gaussiano
- Varianza de proceso y medición configurables
- Predicción y corrección iterativas

**LMS (Least Mean Squares):**
- Filtro adaptativo con actualización por gradiente descendente
- Tasa de aprendizaje μ configurable
- Minimización del error cuadrático medio

**RLS (Recursive Least Squares):**
- Filtro adaptativo con convergencia más rápida que LMS
- Factor de olvido λ para datos no estacionarios
- Actualización recursiva de matriz de correlación inversa

### 5. Validación y Comparación (`validation.py`)

**Métricas de Rendimiento:**
- **SNR (Signal-to-Noise Ratio)**: Relación señal-ruido en dB
- **MSE/RMSE**: Error cuadrático medio y raíz
- **Mejora de SNR**: Comparación antes/después del filtrado

**Métricas de Detección:**
- **TP (True Positives)**: Eventos correctamente detectados
- **FP (False Positives)**: Falsas alarmas
- **FN (False Negatives)**: Eventos no detectados
- **Precisión**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Media armónica de precisión y recall

**Tabla Comparativa:**
Genera DataFrame de pandas con todas las métricas para cada filtro.

### 6. Sistema Integrado (`dp_detection_system.py`)

**Pipeline Completo:**
1. Preprocesamiento de señal (filtrado, normalización, reducción de ruido)
2. Extracción de envolvente
3. Cálculo de características
4. Clasificación con semáforo
5. Aplicación de filtros adaptativos
6. Validación y comparación
7. Generación de reporte diagnóstico

**Generador de Señales Sintéticas:**
Crea señales UHF con descargas parciales realistas:
- Oscilaciones amortiguadas (~800 MHz)
- Decaimiento exponencial
- Ruido Gaussiano configurable
- Eventos con posiciones y duraciones aleatorias

**Reporte Diagnóstico:**
Incluye:
- Estado de clasificación (color y nivel)
- Índice combinado y umbrales
- Características contribuyentes principales
- Todas las características extraídas
- Comparación de filtros adaptativos
- Mejor desempeño por métrica

## Aspectos Técnicos Especiales

### Sin uso de Δt
El sistema opera completamente sobre muestras de señal sin requerir intervalos de tiempo explícitos. Todas las operaciones son independientes de Δt:
- Índices de muestra en lugar de tiempos absolutos
- Métricas basadas en conteo de muestras
- Frecuencias derivadas de tasa de muestreo (fs)

### Arquitectura Modular
- Cada módulo es independiente y reutilizable
- Interfaces claras entre componentes
- Facilita pruebas unitarias y mantenimiento

### Escalabilidad
- Procesamiento eficiente con NumPy/SciPy
- Manejo de señales largas (1M+ muestras)
- Actualización incremental de estadísticas históricas

## Uso del Sistema

### Instalación
```bash
pip install -r requirements.txt
```

### Uso Básico
```python
from dp_detection_system import DPDetectionSystem

detector = DPDetectionSystem(sampling_rate=1e9)
diagnosis = detector.process_and_diagnose(signal)
report = detector.generate_diagnostic_report(diagnosis)
print(report)
```

### Ejemplos Completos
Ver `example_usage.py` para 5 ejemplos diferentes:
1. Detección básica
2. Comparación de filtros
3. Procesamiento múltiple con histórico
4. Señal personalizada
5. Análisis detallado de características

## Validación

### Tests Unitarios
23 tests implementados cubriendo todos los módulos:
- Signal Processing: 4 tests
- Feature Extraction: 5 tests
- Traffic Light Classifier: 2 tests
- Adaptive Filters: 6 tests
- Validation: 3 tests
- DP Detection System: 3 tests

**Resultado: 100% de tests pasando**

### Seguridad
CodeQL ejecutado sin alertas de seguridad.

## Archivos del Proyecto

```
V2DP/
├── signal_processing.py         # Procesamiento de señales (238 líneas)
├── feature_extraction.py        # Extracción de características (238 líneas)
├── traffic_light_classifier.py  # Sistema semáforo (286 líneas)
├── adaptive_filters.py          # Filtros adaptativos (262 líneas)
├── validation.py                # Validación y comparación (307 líneas)
├── dp_detection_system.py       # Sistema principal (358 líneas)
├── example_usage.py             # Ejemplos de uso (260 líneas)
├── test_dp_detection.py         # Tests unitarios (266 líneas)
├── requirements.txt             # Dependencias
├── .gitignore                   # Exclusiones de Git
└── README.md                    # Documentación principal
```

**Total: ~2,192 líneas de código Python**

## Dependencias

- NumPy ≥1.21.0: Operaciones numéricas
- SciPy ≥1.7.0: Procesamiento de señales, filtros, FFT
- Matplotlib ≥3.4.0: Visualización (para uso futuro)
- Pandas ≥1.3.0: Tablas comparativas

## Conclusión

El sistema implementado cumple completamente con todos los requisitos especificados:

✅ Filtrado de señales UHF (paso-banda, paso-bajo)
✅ Normalización (múltiples métodos)
✅ Detección de envolvente (Hilbert)
✅ Reducción de ruido (Savitzky-Golay, mediana)
✅ Cálculo de características (energía, bandas, curtosis, asimetría, RMS, estabilidad, residual)
✅ Sistema de semáforo con umbrales dinámicos (verde/amarillo/naranja/rojo)
✅ Cinco filtros adaptativos (EWMA, MA, Kalman, LMS, RLS)
✅ Comparación de filtros con métricas completas
✅ Validación (FP, FN, TP, Precisión, Recall, F1-Score, SNR)
✅ Tabla comparativa con pandas
✅ Diagnóstico completo sin usar Δt
✅ Tests unitarios (23 tests, 100% pasando)
✅ Sin vulnerabilidades de seguridad

El sistema está listo para uso en detección de descargas parciales en equipos de alta tensión mediante señales UHF.
