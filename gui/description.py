"""
Módulo de descripción técnica detallada del sistema.

Este módulo proporciona la documentación técnica completa del sistema
de detección de descargas parciales UHF.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc


def create_description_layout():
    """
    Crea el layout de la pestaña de descripción técnica.
    
    Retorna:
    --------
    layout : dash component
        Layout de la pestaña
    """
    return dbc.Container([
        # Header
        html.Div([
            html.H2([
                html.I(className="fas fa-info-circle me-3"),
                "Descripción Técnica del Sistema"
            ], className="text-center mb-4"),
            html.Hr(className="mb-4"),
        ]),
        
        # Propósito General
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-bullseye me-2"),
                    "🎯 Propósito General del Sistema"
                ])
            ]),
            dbc.CardBody([
                html.P([
                    "Este es un ", html.Strong("sistema completo de diagnóstico automático"), 
                    " para detectar y clasificar el estado de equipos eléctricos mediante el análisis de señales de ",
                    html.Strong("descargas parciales (PD)"), " en el rango de ",
                    html.Strong("ultra alta frecuencia (UHF)"), 
                    ". El sistema clasifica automáticamente el estado operativo en 4 niveles de severidad:"
                ], className="mb-3"),
                html.Div([
                    dbc.Badge("🟢 Verde - Normal", color="success", className="me-2 mb-2", style={"fontSize": "1rem"}),
                    dbc.Badge("🟡 Amarillo - Precaución", color="warning", className="me-2 mb-2", style={"fontSize": "1rem"}),
                    dbc.Badge("🟠 Naranja - Alerta", color="danger", className="me-2 mb-2", style={"fontSize": "1rem", "background": "#ff8c00"}),
                    dbc.Badge("🔴 Rojo - Crítico", color="danger", className="me-2 mb-2", style={"fontSize": "1rem"}),
                ])
            ])
        ], className="mb-4 glass-card"),
        
        # Arquitectura Modular
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-sitemap me-2"),
                    "📁 Arquitectura Modular"
                ])
            ]),
            dbc.CardBody([
                # Módulo 1: Preprocessing
                html.H5([
                    html.I(className="fas fa-filter me-2"),
                    "1. preprocessing.py - Preprocesamiento de Señales"
                ], className="mt-3"),
                html.P([
                    html.Strong("Función: "), "Limpia y prepara las señales UHF crudas para análisis."
                ]),
                html.Ul([
                    html.Li([html.Code("bandpass_filter()"), " - Filtro Butterworth pasabanda para eliminar frecuencias fuera del rango de interés"]),
                    html.Li([html.Code("normalize_signal()"), " - 3 métodos: zscore (μ=0, σ=1), minmax [0,1], robust (mediana/MAD)"]),
                    html.Li([html.Code("get_envelope()"), " - Extrae envolvente con transformada de Hilbert"]),
                    html.Li([html.Code("wavelet_denoise()"), " - Eliminación de ruido con wavelets Daubechies db4"]),
                    html.Li([html.Code("preprocess_signal()"), " - Pipeline completo: filtrado → normalización → envolvente → denoising"]),
                ]),
                
                html.Hr(),
                
                # Módulo 2: Descriptors
                html.H5([
                    html.I(className="fas fa-chart-bar me-2"),
                    "2. descriptors.py - Extracción de Características"
                ], className="mt-3"),
                html.P([
                    html.Strong("Función: "), "Calcula 15+ descriptores que caracterizan la señal."
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H6("Descriptores Energéticos:", className="text-info"),
                        html.Ul([
                            html.Li([html.Code("energy_total()"), " - Energía total: E = Σx²"]),
                            html.Li([html.Code("energy_spectral_bands()"), " - Energía en 4 bandas de frecuencia"]),
                            html.Li([html.Code("rms_value()"), " - Valor RMS"]),
                        ]),
                    ], md=6),
                    dbc.Col([
                        html.H6("Descriptores Estadísticos:", className="text-info"),
                        html.Ul([
                            html.Li([html.Code("kurtosis()"), " - Medida de picudez (descargas)"]),
                            html.Li([html.Code("skewness()"), " - Asimetría de distribución"]),
                            html.Li([html.Code("crest_factor()"), " - Razón pico/RMS"]),
                        ]),
                    ], md=6),
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H6("Descriptores Espectrales:", className="text-info"),
                        html.Ul([
                            html.Li([html.Code("spectral_entropy()"), " - Entropía del espectro"]),
                            html.Li([html.Code("spectral_stability()"), " - Correlación entre ventanas"]),
                        ]),
                    ], md=6),
                    dbc.Col([
                        html.H6("Otros Descriptores:", className="text-info"),
                        html.Ul([
                            html.Li([html.Code("peak_count()"), " - Número de picos detectados"]),
                            html.Li([html.Code("zero_crossing_rate()"), " - Tasa de cruces por cero"]),
                        ]),
                    ], md=6),
                ]),
                
                html.Hr(),
                
                # Módulo 3: Severity
                html.H5([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    "3. severity.py - Evaluación de Severidad"
                ], className="mt-3"),
                html.P([
                    html.Strong("Función: "), "Clasifica automáticamente el estado operativo basado en descriptores."
                ]),
                
                html.H6("Proceso de Clasificación:", className="text-warning mt-3"),
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Strong("1. calculate_descriptor_scores()"), 
                        " - Normaliza cada descriptor usando Z-score contra línea base"
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("2. calculate_severity_index()"), 
                        " - Combina scores ponderados: SI = Σ(wᵢ·scoreᵢ) / Σwᵢ"
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("3. determine_thresholds_statistical()"), 
                        " - Calcula umbrales dinámicos: umbral = μ + k·σ"
                    ]),
                    dbc.ListGroupItem([
                        html.Strong("4. classify_traffic_light()"), 
                        " - Asigna color del semáforo según índice de severidad"
                    ]),
                ], className="mb-3"),
                
                dbc.Alert([
                    html.H6("Umbrales de Clasificación:", className="mb-2"),
                    html.Ul([
                        html.Li("🟢 Verde → Amarillo: μ + 1.5σ (SI < 2.0)"),
                        html.Li("🟡 Amarillo → Naranja: μ + 4.0σ (SI < 6.0)"),
                        html.Li("🟠 Naranja → Rojo: μ + 8.0σ (SI < 15.0)"),
                        html.Li("🔴 Rojo: SI ≥ 15.0"),
                    ], className="mb-0")
                ], color="info"),
                
                html.Hr(),
                
                # Módulo 4: Validation
                html.H5([
                    html.I(className="fas fa-check-circle me-2"),
                    "4. validation.py - Validación del Sistema"
                ], className="mt-3"),
                html.P([
                    html.Strong("Función: "), "Evalúa el rendimiento del algoritmo de detección."
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H6("Métricas de Clasificación:", className="text-success"),
                        html.Ul([
                            html.Li([html.Code("calculate_accuracy()"), " - Accuracy = correctas/total"]),
                            html.Li([html.Code("calculate_false_positive_rate()"), " - Tasa de falsas alarmas"]),
                            html.Li([html.Code("calculate_false_negative_rate()"), " - Fallas no detectadas"]),
                            html.Li([html.Code("calculate_confusion_matrix()"), " - Matriz 4×4"]),
                        ]),
                    ], md=6),
                    dbc.Col([
                        html.H6("Métricas de Separación:", className="text-success"),
                        html.Ul([
                            html.Li([html.Code("calculate_class_separation()"), " - Distancia de Cohen"]),
                            html.Li([html.Code("calculate_threshold_stability()"), " - Estabilidad temporal"]),
                        ]),
                        html.H6("Coeficientes:", className="text-success mt-2"),
                        html.Ul([
                            html.Li("CV = σ/μ (coeficiente de variación)"),
                            html.Li("Estabilidad = 1/(1+CV)"),
                        ]),
                    ], md=6),
                ]),
                
                html.Hr(),
                
                # Módulo 5: Main
                html.H5([
                    html.I(className="fas fa-cogs me-2"),
                    "5. main.py - Integración y Ejecución"
                ], className="mt-3"),
                html.P([
                    html.Strong("Función: "), "Orquesta todo el sistema y provee funciones de alto nivel."
                ]),
                
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.P([
                            html.Strong("generate_synthetic_signal()"), 
                            " - Genera señales sintéticas para testing con características específicas por estado:"
                        ]),
                        html.Ul([
                            html.Li("🟢 Verde: 3 descargas, amplitud 0.8, frecuencia 1 kHz"),
                            html.Li("🟡 Amarillo: 10 descargas, amplitud 2.0, frecuencia 1.5 kHz"),
                            html.Li("🟠 Naranja: 25 descargas, amplitud 4.0, frecuencia 2 kHz"),
                            html.Li("🔴 Rojo: 45 descargas, amplitud 6.5, frecuencia 2.5 kHz"),
                        ]),
                        html.P("Cada descarga es un pulso oscilatorio amortiguado: y(t) = A·sin(2πft)·e⁻⁵⁰⁰ᵗ", className="mt-2 text-muted"),
                    ], title="Generación de Señales"),
                    
                    dbc.AccordionItem([
                        html.P("Pipeline completo de análisis:"),
                        dbc.Alert([
                            "Señal cruda → Preprocesamiento → Descriptores → Severidad → Resultado"
                        ], color="secondary", className="text-center mb-0"),
                    ], title="process_and_analyze_signal()"),
                    
                    dbc.AccordionItem([
                        html.Ul([
                            html.Li("Genera N señales por estado (defecto: 10)"),
                            html.Li("Crea perfil de línea base (estado verde)"),
                            html.Li("Calcula umbrales adaptativos"),
                            html.Li("Valida clasificación con métricas estadísticas"),
                        ]),
                    ], title="evaluate_multiple_states()"),
                    
                    dbc.AccordionItem([
                        html.Ul([
                            html.Li("Ejecuta evaluación completa del sistema"),
                            html.Li("Genera reportes de validación"),
                            html.Li("Muestra ejemplos de diagnóstico por estado"),
                            html.Li("Imprime resumen de desempeño global"),
                        ]),
                    ], title="main()"),
                ]),
            ])
        ], className="mb-4 glass-card"),
        
        # Flujo de Ejecución
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-project-diagram me-2"),
                    "🔄 Flujo de Ejecución Completo"
                ])
            ]),
            dbc.CardBody([
                html.H6("Modo de uso típico:", className="text-primary"),
                dbc.Alert([
                    html.Pre([
                        html.Code("""# 1. Capturar señal UHF del equipo
signal_raw = capture_from_hardware(fs=12.5e9)  # 12.5 GS/s

# 2. Preprocesar
signal_clean, info = preprocess_signal(
    signal_raw, 
    fs=12.5e9,
    lowcut=100e6,   # 100 MHz
    highcut=5e9     # 5 GHz
)

# 3. Calcular descriptores
descriptors = compute_all_descriptors(signal_clean, fs=12.5e9)

# 4. Evaluar severidad
result = assess_severity(descriptors, baseline_profile)

# 5. Obtener diagnóstico
print(f"Estado: {result['traffic_light_state']}")
print(f"Severidad: {result['severity_index']:.2f}")""", style={"color": "#00ff88"})
                    ], className="mb-0", style={"background": "rgba(0,0,0,0.3)", "padding": "15px", "borderRadius": "8px"})
                ], color="dark", className="mb-0"),
            ])
        ], className="mb-4 glass-card"),
        
        # Conceptos Clave
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-graduation-cap me-2"),
                    "🎓 Conceptos Clave"
                ])
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Descargas Parciales (PD)", className="bg-primary text-white"),
                            dbc.CardBody([
                                html.P("Pequeñas chispas eléctricas dentro del aislamiento que indican degradación progresiva. Precursores de fallas catastróficas.")
                            ])
                        ], className="mb-3 h-100"),
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Rango UHF (300 MHz - 3 GHz)", className="bg-info text-white"),
                            dbc.CardBody([
                                html.P("Las descargas emiten pulsos electromagnéticos en este rango, detectables con antenas especializadas.")
                            ])
                        ], className="mb-3 h-100"),
                    ], md=6),
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Clasificación por Semáforo", className="bg-warning text-dark"),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li("🟢 Verde: Operación normal, sin acción"),
                                    html.Li("🟡 Amarillo: Monitoreo incrementado"),
                                    html.Li("🟠 Naranja: Mantenimiento programado"),
                                    html.Li("🔴 Rojo: Intervención inmediata"),
                                ], className="mb-0")
                            ])
                        ], className="mb-3 h-100"),
                    ], md=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Línea Base (Baseline)", className="bg-success text-white"),
                            dbc.CardBody([
                                html.P("Perfil estadístico del equipo en condiciones normales, usado como referencia para detectar desviaciones.", className="mb-0")
                            ])
                        ], className="mb-3 h-100"),
                    ], md=6),
                ]),
            ])
        ], className="mb-4 glass-card"),
        
        # Aplicaciones
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-industry me-2"),
                    "🔬 Aplicaciones Industriales"
                ])
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.I(className="fas fa-plug me-2 text-primary"),
                                html.Strong("Transformadores de potencia"), 
                                html.Br(),
                                html.Small("Detección de degradación del aislamiento")
                            ]),
                            dbc.ListGroupItem([
                                html.I(className="fas fa-bolt me-2 text-warning"),
                                html.Strong("Switchgear"), 
                                html.Br(),
                                html.Small("Monitoreo de conexiones eléctricas")
                            ]),
                        ]),
                    ], md=6),
                    dbc.Col([
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.I(className="fas fa-cable-car me-2 text-info"),
                                html.Strong("Cables de alto voltaje"), 
                                html.Br(),
                                html.Small("Identificación de defectos")
                            ]),
                            dbc.ListGroupItem([
                                html.I(className="fas fa-building me-2 text-success"),
                                html.Strong("Subestaciones"), 
                                html.Br(),
                                html.Small("Supervisión continua de activos críticos")
                            ]),
                        ]),
                    ], md=6),
                ]),
                
                dbc.Alert([
                    html.Strong("💡 Mantenimiento Predictivo: "),
                    "El sistema permite detectar problemas antes de que causen fallas, optimizando costos y confiabilidad."
                ], color="success", className="mt-3 mb-0"),
            ])
        ], className="mb-4 glass-card"),
        
        # Especificaciones Técnicas
        dbc.Card([
            dbc.CardHeader([
                html.H4([
                    html.I(className="fas fa-microchip me-2"),
                    "⚙️ Especificaciones Técnicas"
                ])
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("Hardware Compatible:", className="text-primary"),
                        html.Ul([
                            html.Li("NI PXIe-5185: 12.5 GS/s, 3 GHz BW, 8-bit"),
                            html.Li("Antenas UHF: 300 MHz - 3 GHz"),
                            html.Li("Acopladores capacitivos"),
                        ]),
                    ], md=6),
                    dbc.Col([
                        html.H6("Formatos Soportados:", className="text-primary"),
                        html.Ul([
                            html.Li("CSV: Archivos de texto con pandas"),
                            html.Li("HDF5: Archivos binarios con h5py"),
                            html.Li("MATLAB: Archivos .mat con scipy.io"),
                        ]),
                    ], md=6),
                ]),
                
                dbc.Row([
                    dbc.Col([
                        html.H6("Librerías Python:", className="text-primary"),
                        dbc.Badge("NumPy", className="me-2 mb-2", color="secondary"),
                        dbc.Badge("SciPy", className="me-2 mb-2", color="secondary"),
                        dbc.Badge("PyWavelets", className="me-2 mb-2", color="secondary"),
                        dbc.Badge("Plotly", className="me-2 mb-2", color="secondary"),
                        dbc.Badge("Dash", className="me-2 mb-2", color="secondary"),
                    ], md=12),
                ]),
            ])
        ], className="mb-4 glass-card"),
        
        # Footer
        html.Div([
            html.Hr(),
            html.P([
                html.I(className="fas fa-info-circle me-2"),
                "Sistema de Detección de Descargas Parciales UHF v1.0 - Noviembre 2025"
            ], className="text-center text-muted mb-0"),
        ], className="mt-4"),
        
    ], fluid=True, className="p-4")


def register_callbacks(app):
    """
    Registra los callbacks necesarios para la pestaña de descripción.
    
    Parámetros:
    -----------
    app : dash.Dash
        Instancia de la aplicación Dash
    """
    # Esta pestaña es principalmente estática, no requiere callbacks complejos
    pass
