# V2DP

Sistema de Detección de Fallas mediante Descargas Parciales (DP) usando Señales UHF

## Descripción

Este sistema implementa un algoritmo completo de detección y clasificación de fallas en equipos eléctricos mediante el análisis de descargas parciales captadas por sensores UHF. El sistema procesa señales en bruto, calcula múltiples descriptores energéticos, estadísticos y espectrales, y determina automáticamente el estado operativo del equipo usando una clasificación tipo semáforo (verde, amarillo, naranja, rojo).

## Características

- **Preprocesamiento avanzado de señales**:
  - Filtrado pasabanda Butterworth
  - Normalización (Z-score, Min-Max, Robust)
  - Extracción de envolvente mediante transformada de Hilbert
  - Eliminación de ruido mediante wavelets y filtros adaptativos

- **Descriptores robustos**:
  - Energéticos: energía total, energía por bandas espectrales, energía residual
  - Estadísticos: RMS, curtosis, asimetría, factor de cresta
  - Espectrales: entropía espectral, estabilidad espectral
  - Temporales: conteo de picos, tasa de cruces por cero

- **Evaluación de severidad automática**:
  - Cálculo de índice de severidad unificado
  - Determinación dinámica de umbrales (percentiles y reglas estadísticas)
  - Clasificación automática en cuatro estados (verde/amarillo/naranja/rojo)

- **Comparación con algoritmos ciegos**:
  - EWMA (Exponentially Weighted Moving Average)
  - Media Móvil Simple (SMA)
  - Filtro de Kalman 1D
  - Filtros adaptativos LMS y RLS

- **Validación exhaustiva**:
  - Tasas de falsos positivos y negativos
  - Métricas de separación entre clases
  - Análisis de estabilidad del umbral
  - Cálculo de SNR efectivo
  - Variación de descriptores por estado

## Estructura del Proyecto

```
V2DP/
├── preprocessing.py        # Módulo de preprocesamiento de señales
├── descriptors.py         # Cálculo de descriptores
├── severity.py           # Evaluación de severidad y clasificación
├── blind_algorithms.py   # Algoritmos ciegos para comparación
├── validation.py         # Validación del sistema
├── main.py              # Módulo principal e integración
├── requirements.txt     # Dependencias del proyecto
└── README.md           # Este archivo
```

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/3d2yy/V2DP.git
cd V2DP
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Ejecución del sistema completo

Para ejecutar una demostración completa con señales sintéticas:

```bash
python main.py
```

Esto generará:
- Señales sintéticas para los cuatro estados operativos
- Perfil de línea base a partir de señales en estado verde
- Análisis completo con descriptores y clasificación
- Comparación con algoritmos ciegos
- Reporte de validación con métricas de rendimiento
- Tabla comparativa de algoritmos

### Uso programático

```python
from main import process_and_analyze_signal, generate_synthetic_signal
import numpy as np

# Generar señal sintética (o usar señal real)
fs = 10000  # Frecuencia de muestreo en Hz
signal = generate_synthetic_signal('amarillo', duration=1000, fs=fs)

# Procesar y analizar
results = process_and_analyze_signal(signal, fs)

# Acceder a resultados
print(f"Estado: {results['traffic_light_state']}")
print(f"Índice de severidad: {results['severity_index']:.4f}")
print(f"Descriptores: {results['descriptors']}")
```

### Procesamiento de señales reales

```python
from preprocessing import preprocess_signal
from descriptors import compute_all_descriptors
from severity import assess_severity

# Cargar señal real (por ejemplo, desde archivo)
# signal_data = np.loadtxt('señal_uhf.txt')
signal_data = np.random.randn(1000)  # Ejemplo
fs = 10000

# Preprocesar
processed_signal, _ = preprocess_signal(
    signal_data, fs,
    lowcut=100,      # Hz
    highcut=4000,    # Hz
    normalize=True,
    envelope=True,
    denoise=True
)

# Calcular descriptores
descriptors = compute_all_descriptors(processed_signal, fs, signal_data)

# Evaluar severidad
severity_results = assess_severity(descriptors)
print(f"Estado: {severity_results['traffic_light_state']}")
```

## Salida del Sistema

El sistema proporciona:

1. **Diagnóstico detallado**:
   - Estado operativo (🟢 verde, 🟡 amarillo, 🟠 naranja, 🔴 rojo)
   - Índice de severidad
   - Valores de todos los descriptores
   - Umbrales de clasificación

2. **Métricas de validación**:
   - Precisión del sistema
   - Tasas de falsos positivos y negativos
   - Estabilidad del umbral
   - Separación entre clases (Cohen's d, F-ratio)
   - SNR efectivo

3. **Comparación de algoritmos**:
   - Tabla con puntuaciones de cada algoritmo ciego
   - Estadísticas por estado operativo

## Características técnicas

- **Sin dependencia de tiempos entre pulsos**: El sistema no utiliza análisis de Δt
- **Descriptores robustos**: Resistentes a variaciones de ruido y condiciones operativas
- **Clasificación automática**: Sin necesidad de intervención manual
- **Umbrales adaptativos**: Se ajustan dinámicamente según condiciones base
- **Validación exhaustiva**: Métricas completas de rendimiento

## Dependencias

- numpy >= 1.24.0
- scipy >= 1.10.0
- PyWavelets >= 1.4.0
- matplotlib >= 3.7.0

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abra un issue para discutir cambios importantes antes de crear un pull request.

## Licencia

Ver archivo LICENSE para más detalles.

## Autores

Desarrollado para detección de fallas en equipos eléctricos mediante análisis de descargas parciales UHF.
