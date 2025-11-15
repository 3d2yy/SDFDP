# 🔌 Sistema de Detección de Descargas Parciales UHF - Interfaz Gráfica

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-2.14+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Sistema profesional de monitoreo en tiempo real y análisis offline de descargas parciales**

</div>

---

## 🚀 Características Principales

### 📡 **Captura en Vivo**
- **Hardware Real**: Compatible con NI PXIe-5185 (12.5 GS/s, 3 GHz BW, 8-bit)
- **Modo Simulación**: Generación sintética para pruebas sin hardware
- **Monitoreo en Tiempo Real**: Visualización continua de señales y descriptores
- **Clasificación Automática**: Sistema tipo semáforo (Verde/Amarillo/Naranja/Rojo)

### 📂 **Análisis de Archivos**
- **Formatos Múltiples**: CSV, HDF5 (.h5), MATLAB (.mat)
- **Visualizaciones Completas**: Señal, espectro, descriptores, radar chart
- **Procesamiento Avanzado**: Filtrado, normalización, extracción de envolvente
- **Evaluación de Severidad**: Clasificación automática con detalles

### ⚙️ **Generador de Señales**
- **Parámetros Personalizables**: Estado, amplitud, frecuencia, ruido
- **Tipos de Ruido**: Gaussiano, Rosa, Marrón, Uniforme
- **Exportación Múltiple**: CSV, HDF5, MAT con metadatos
- **Análisis Inmediato**: Estadísticas, espectro, histogramas

### 🎯 **Configuración de Umbrales**
- **Umbrales Personalizables**: Ajuste de límites de clasificación
- **Pesos de Descriptores**: Control sobre importancia relativa
- **Pruebas Interactivas**: Generación y clasificación en vivo
- **Validación Completa**: Matriz de confusión y métricas de precisión

### 📚 **Documentación Integrada**
- Guía de uso paso a paso
- Especificaciones técnicas
- Mejores prácticas

---

## 📦 Instalación

### 1. Clonar o descargar el repositorio

```bash
cd /workspaces/V2DP
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. (Opcional) Instalar soporte para hardware NI

Si va a usar hardware National Instruments:

```bash
pip install nidaqmx
```

---

## 🎯 Uso Rápido

### Iniciar la aplicación:

```bash
python app.py
```

La interfaz estará disponible en: **http://localhost:8050**

### Flujo de trabajo recomendado:

1. **📚 Documentación**: Familiarícese con el sistema
2. **🎯 Configuración de Umbrales**: Ajuste parámetros si es necesario
3. **⚙️ Generador**: Cree señales de prueba
4. **📂 Análisis de Archivos**: Analice datos existentes
5. **📡 Captura en Vivo**: Monitoreo en tiempo real

---

## 🔧 Configuración

### Hardware NI PXIe-5185

Para usar con hardware real, en la pestaña **Captura en Vivo**:

1. Seleccione "Hardware NI PXIe-5185"
2. Configure:
   - **Device**: Nombre del dispositivo (ej: `PXI1Slot2`)
   - **Canal**: Número de canal analógico (ej: `0`)
   - **Frecuencia de Muestreo**: En GS/s (ej: `12.5`)
3. Inicie la captura

### Modo Simulación

Para pruebas sin hardware:

1. Seleccione "Modo Simulación"
2. Elija el estado a simular:
   - 🟢 Verde (Normal)
   - 🟡 Amarillo (Precaución)
   - 🟠 Naranja (Alerta)
   - 🔴 Rojo (Crítico)
3. Ajuste el nivel de ruido
4. Inicie la captura

---

## 📊 Descriptores Calculados

El sistema calcula 9 descriptores para caracterizar las señales:

| # | Descriptor | Descripción |
|---|------------|-------------|
| 1 | **Energía Total** | Suma de cuadrados de la señal |
| 2 | **RMS** | Valor cuadrático medio |
| 3 | **Curtosis** | Medida de "picos" en distribución |
| 4 | **Asimetría** | Sesgo de la distribución |
| 5 | **Factor de Cresta** | Relación pico/RMS |
| 6 | **Conteo de Picos** | Número de picos significativos |
| 7 | **Entropía Espectral** | Desorden en el espectro |
| 8 | **Estabilidad Espectral** | Consistencia del espectro |
| 9 | **Tasa de Cruces por Cero** | Frecuencia de cambios de signo |

---

## 🎨 Estructura del Proyecto

```
V2DP/
├── app.py                      # Aplicación principal Dash
├── gui/
│   ├── __init__.py
│   ├── live_capture.py         # Captura en tiempo real
│   ├── file_analysis.py        # Análisis de archivos
│   ├── signal_generator.py     # Generador de señales
│   ├── threshold_config.py     # Configuración de umbrales
│   └── documentation.py        # Documentación
├── main.py                     # Sistema de backend
├── preprocessing.py            # Preprocesamiento de señales
├── descriptors.py              # Cálculo de descriptores
├── severity.py                 # Evaluación de severidad
├── blind_algorithms.py         # Algoritmos ciegos
├── validation.py               # Validación del sistema
└── requirements.txt            # Dependencias
```

---

## 🔬 Especificaciones Técnicas

### Sistema de Adquisición

| Componente | Especificación |
|------------|----------------|
| **Sistema** | NI PXIe-1071 |
| **Controlador** | NI PXIe-8135 (Embebido) |
| **Tarjeta** | NI PXIe-5185 |
| **Ancho de Banda** | 3 GHz |
| **Frecuencia de Muestreo** | 12.5 GS/s |
| **Resolución** | 8 bits |

### Procesamiento de Señal

- **Filtrado**: Pasa-banda (1% - 40% de fs)
- **Normalización**: Adaptativa
- **Envolvente**: Transformada de Hilbert
- **Reducción de Ruido**: Wavelets

---

## 📖 Ejemplos de Uso

### Ejemplo 1: Análisis de archivo CSV

```python
# En la pestaña "Análisis de Archivos":
# 1. Cargar archivo CSV con señal
# 2. Configurar fs = 10000 Hz
# 3. Columna de datos = "signal"
# 4. Clic en "Analizar Señal"
# 5. Ver clasificación y descriptores
```

### Ejemplo 2: Generar dataset sintético

```python
# En la pestaña "Generador de Señales":
# 1. Estado = "Naranja"
# 2. Duración = 5000 muestras
# 3. Descargas = 30
# 4. Amplitud = 4.0
# 5. Clic en "Generar Señal"
# 6. Exportar como HDF5 con metadatos
```

### Ejemplo 3: Calibrar umbrales

```python
# En la pestaña "Configuración de Umbrales":
# 1. Ajustar Verde→Amarillo = 0.3
# 2. Ajustar Amarillo→Naranja = 0.6
# 3. Ajustar Naranja→Rojo = 0.8
# 4. Clic en "Ejecutar Prueba Completa"
# 5. Ver matriz de confusión y precisión
```

---

## 🐛 Solución de Problemas

### Error: "nidaqmx no está instalado"

```bash
pip install nidaqmx
```

### Error: "h5py no encontrado"

```bash
pip install h5py
```

### La aplicación no inicia

Verifique que todas las dependencias estén instaladas:

```bash
pip install -r requirements.txt
```

### No se detecta el hardware NI

1. Verifique que el controlador NI-DAQmx esté instalado
2. Confirme el nombre del dispositivo en NI MAX
3. Use el nombre correcto en la configuración

---

## 🤝 Contribuciones

Este es un sistema profesional de detección de descargas parciales. Para mejoras o reportar problemas, consulte la documentación del proyecto.

---

## 📄 Licencia

Ver archivo LICENSE en el repositorio.

---

## 🙏 Agradecimientos

Sistema desarrollado utilizando:
- **Dash & Plotly**: Visualizaciones interactivas
- **NumPy & SciPy**: Procesamiento científico
- **NI-DAQmx**: Integración con hardware profesional
- **Bootstrap**: Diseño responsivo

---

<div align="center">

**🔌 Sistema de Detección de Descargas Parciales UHF**

*Monitoreo profesional en tiempo real para equipos de alta tensión*

</div>
