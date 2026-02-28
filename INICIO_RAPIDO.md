# 🚀 Guía de Inicio Rápido - Sistema PD-UHF

## ✅ Sistema Instalado y Funcional

Tu sistema de detección de descargas parciales ya está completamente configurado y listo para usar.

---

## 🎯 Inicio Rápido

### Opción 1: Inicio Simple
```bash
python app.py
```

### Opción 2: Inicio con Script
```bash
python start_gui.py
```

### Opción 3: Inicio con Opciones
```bash
# Puerto personalizado
python start_gui.py --port 8080

# Modo debug
python start_gui.py --debug

# Ambos
python start_gui.py --port 8080 --debug
```

---

## 🌐 Acceso a la Interfaz

Una vez iniciado, abra su navegador en:

**http://localhost:8050**

O si está en un contenedor/servidor remoto:

**http://[IP_DEL_SERVIDOR]:8050**

---

## 📋 Pestañas Disponibles

### 📡 Captura en Vivo
**Para:** Monitoreo en tiempo real
- **Modo Hardware:** Usar con NI PXIe-5185
- **Modo Simulación:** Pruebas sin hardware

### 📂 Análisis de Archivos
**Para:** Analizar datos guardados
- Formatos: CSV, HDF5, MATLAB
- Carga y analiza en segundos

### ⚙️ Generador de Señales
**Para:** Crear datasets sintéticos
- Señales personalizables
- Exportación en múltiples formatos

### 🎯 Configuración de Umbrales
**Para:** Calibrar el sistema
- Ajustar límites de clasificación
- Probar configuración

### 📚 Documentación
**Para:** Aprender a usar el sistema
- Guía completa
- Especificaciones técnicas

---

## 🧪 Probar el Sistema

### 1. Verificar Instalación
```bash
python test_system.py
```

### 2. Primera Prueba (Modo Simulación)

1. Inicie la aplicación: `python app.py`
2. Vaya a **📡 Captura en Vivo**
3. Seleccione **"Modo Simulación"**
4. Estado: **"🟢 Verde"**
5. Clic en **"Iniciar Captura"**
6. Observe las gráficas actualizándose en tiempo real
7. El semáforo debe mostrar **🟢 Verde - Normal**

### 3. Segunda Prueba (Generador)

1. Vaya a **⚙️ Generador de Señales**
2. Seleccione estado: **"🔴 Rojo"**
3. Configure:
   - Duración: 1000
   - Descargas: 40
   - Amplitud: 6.0
4. Clic en **"Generar Señal"**
5. Vea la señal, espectro y estadísticas
6. Exporte como CSV o HDF5

### 4. Tercera Prueba (Umbrales)

1. Vaya a **🎯 Configuración de Umbrales**
2. Ajuste los umbrales o use valores por defecto
3. Clic en **"Ejecutar Prueba Completa"**
4. Vea la matriz de confusión y precisión

---

## 🔧 Uso con Hardware Real (NI PXIe-5185)

### Requisitos Previos

1. Instalar driver NI-DAQmx:
   - Descargue de: https://www.ni.com/es-mx/support/downloads/drivers/download.ni-daqmx.html

2. Instalar Python bindings:
```bash
pip install nidaqmx
```

3. Verificar dispositivo:
   - Abra NI MAX (Measurement & Automation Explorer)
   - Verifique que el dispositivo aparezca (ej: PXI1Slot2)
   - Anote el nombre exacto del dispositivo

### Configuración

1. Vaya a **📡 Captura en Vivo**
2. Seleccione **"Hardware NI PXIe-5185"**
3. Configure:
   - **Device:** Nombre del NI MAX (ej: `PXI1Slot2`)
   - **Canal:** `0` (o el canal que use)
   - **Frecuencia de Muestreo:** `12.5` GS/s
4. Clic en **"Iniciar Captura"**

### Solución de Problemas Hardware

Si aparece error al capturar:

1. **Verifique conexiones físicas**
2. **Confirme nombre de dispositivo en NI MAX**
3. **Verifique permisos de acceso**
4. **Pruebe con modo simulación primero**

---

## 📊 Análisis de Archivos Existentes

### CSV

1. Prepare su archivo CSV con formato:
```csv
time,signal
0.0000,0.0123
0.0001,0.0245
...
```

2. En **📂 Análisis de Archivos**:
   - Cargue el archivo
   - Columna de datos: `signal`
   - Si tiene tiempo: active "Con columna de tiempo"
   - Columna de tiempo: `time`
   - Frecuencia: Su fs en Hz

### HDF5

1. Su archivo .h5 debe tener la señal como dataset
2. En **📂 Análisis de Archivos**:
   - Cargue el archivo
   - Campo de datos: nombre del dataset (ej: `signal`)
   - Frecuencia: Su fs en Hz

### MATLAB

1. Su archivo .mat debe tener la señal como variable
2. En **📂 Análisis de Archivos**:
   - Cargue el archivo
   - Campo de datos: nombre de variable (ej: `signal`)
   - Frecuencia: Su fs en Hz

---

## 🎓 Interpretación de Resultados

### Semáforo de Estado

| Estado | Símbolo | Significado | Acción Recomendada |
|--------|---------|-------------|-------------------|
| **Verde** | 🟢 | Normal | Operación normal, monitoreo rutinario |
| **Amarillo** | 🟡 | Precaución | Incrementar frecuencia de monitoreo |
| **Naranja** | 🟠 | Alerta | Planear mantenimiento próximo |
| **Rojo** | 🔴 | Crítico | Acción inmediata necesaria |

### Índice de Severidad

- **0.0 - 0.25:** Verde
- **0.25 - 0.50:** Amarillo
- **0.50 - 0.75:** Naranja
- **0.75 - 1.0:** Rojo

*Nota: Estos umbrales son configurables en la pestaña de Configuración*

### Descriptores Clave

- **Energía Total / RMS:** Magnitud de actividad
- **Conteo de Picos:** Frecuencia de descargas
- **Factor de Cresta:** Relación pico/promedio
- **Entropía Espectral:** Complejidad de la señal

---

## 💡 Mejores Prácticas

### 1. Establecer Línea Base

Antes de usar en producción:
1. Capture datos en condiciones normales (estado verde)
2. Genere 20-50 muestras
3. Use estos datos para calibrar umbrales

### 2. Monitoreo Continuo

- Configure captura automática cada X horas
- Guarde descriptores históricos
- Identifique tendencias

### 3. Documentar Eventos

Cuando detecte estado Naranja/Rojo:
- Capture la señal completa
- Anote condiciones operativas
- Guarde para análisis posterior

### 4. Validación Periódica

Cada 3-6 meses:
- Re-ejecute prueba completa de umbrales
- Ajuste si es necesario
- Documente cambios

---

## 🆘 Soporte

### Problemas Comunes

**P: La aplicación no inicia**
```bash
# Verificar dependencias
pip install -r requirements.txt

# Probar sistema
python test_system.py
```

**P: Gráficas no se actualizan**
- Refresque el navegador (F5)
- Limpie caché del navegador
- Verifique consola del navegador (F12)

**P: Error en captura de hardware**
- Verifique NI-DAQmx instalado
- Confirme nombre de dispositivo
- Pruebe con modo simulación

**P: Archivo no se carga**
- Verifique formato (CSV, H5, MAT)
- Confirme estructura de datos
- Revise nombre de columna/variable

---

## 📈 Próximos Pasos

1. ✅ **Familiarícese con la interfaz** → Use modo simulación
2. ✅ **Pruebe con datos existentes** → Análisis de archivos
3. ✅ **Genere datasets sintéticos** → Para entrenamiento
4. ✅ **Calibre umbrales** → Según su aplicación
5. ✅ **Conecte hardware real** → Si está disponible
6. ✅ **Monitoreo continuo** → En producción

---

## 📞 Información Adicional

- **Documentación Completa:** Ver pestaña 📚 en la aplicación
- **Especificaciones Técnicas:** Ver GUI_README.md
- **Código Fuente:** Todos los archivos están comentados

---

<div align="center">

**🔌 Sistema de Detección de Descargas Parciales UHF**

*¡Su sistema está listo para detectar y clasificar descargas parciales profesionalmente!*

**Inicie ahora:** `python app.py`

</div>
