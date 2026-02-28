"""
Pestaña de Documentación
=========================

Documentación y guía de uso del sistema.
"""

from dash import html
import dash_bootstrap_components as dbc


# ============================================================================
# LAYOUT
# ============================================================================

def create_layout():
    """Crear layout de documentación."""
    return dbc.Container([
        # Encabezado
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H1([
                        html.I(className="fas fa-bolt text-warning me-3"),
                        "Sistema de Detección de Descargas Parciales UHF"
                    ], className="mb-3"),
                    html.H4("Documentación y Guía de Uso", className="text-muted")
                ], className="text-center mb-5")
            ])
        ]),
        
        # Introducción
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3([
                        html.I(className="fas fa-info-circle me-2"),
                        "Introducción"
                    ])),
                    dbc.CardBody([
                        html.P([
                            "Este sistema está diseñado para la detección y análisis de ",
                            html.Strong("descargas parciales (DP)"),
                            " en equipos de alta tensión mediante tecnología UHF (Ultra High Frequency). "
                            "El sistema clasifica automáticamente el estado operativo del equipo en cuatro niveles "
                            "de severidad usando un sistema tipo semáforo."
                        ], className="lead"),
                        
                        html.Hr(),
                        
                        html.H5("Niveles de Severidad:", className="fw-bold mb-3"),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.H5("🟢 Verde - Normal", className="mb-2"),
                                html.P("Estado óptimo de operación. Pocas descargas de baja amplitud. "
                                      "No se requiere acción.")
                            ], color="success"),
                            dbc.ListGroupItem([
                                html.H5("🟡 Amarillo - Precaución", className="mb-2"),
                                html.P("Descargas moderadas detectadas. Se recomienda monitoreo continuo "
                                      "y planificar mantenimiento preventivo.")
                            ], color="warning"),
                            dbc.ListGroupItem([
                                html.H5("🟠 Naranja - Alerta", className="mb-2"),
                                html.P("Nivel alto de descargas. Revisión necesaria. "
                                      "Considerar reducir carga o programar intervención.")
                            ], color="warning"),
                            dbc.ListGroupItem([
                                html.H5("🔴 Rojo - Crítico", className="mb-2"),
                                html.P("Descargas frecuentes de alta amplitud. ¡Acción inmediata requerida! "
                                      "Riesgo de falla inminente.")
                            ], color="danger")
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Características del Sistema
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3([
                        html.I(className="fas fa-star me-2"),
                        "Características Principales"
                    ])),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-microchip fa-3x text-primary mb-3"),
                                    html.H5("Hardware Profesional", className="fw-bold"),
                                    html.P("Compatible con NI PXIe-5185: 12.5 GS/s, "
                                          "3 GHz de ancho de banda, resolución de 8 bits.")
                                ], className="text-center")
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-chart-line fa-3x text-success mb-3"),
                                    html.H5("Análisis Avanzado", className="fw-bold"),
                                    html.P("9 descriptores estadísticos y espectrales para "
                                          "caracterización completa de señales.")
                                ], className="text-center")
                            ], width=4),
                            dbc.Col([
                                html.Div([
                                    html.I(className="fas fa-brain fa-3x text-info mb-3"),
                                    html.H5("Clasificación Automática", className="fw-bold"),
                                    html.P("Sistema inteligente de umbralización adaptativa "
                                          "con personalización de pesos.")
                                ], className="text-center")
                            ], width=4)
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Guía de Uso - Captura en Vivo
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3([
                        html.I(className="fas fa-broadcast-tower me-2"),
                        "📡 Captura en Vivo"
                    ])),
                    dbc.CardBody([
                        html.H5("Propósito:", className="fw-bold"),
                        html.P("Monitoreo en tiempo real de señales de descargas parciales."),
                        
                        html.H5("Cómo Usar:", className="fw-bold mt-3"),
                        html.Ol([
                            html.Li([
                                html.Strong("Seleccione el Modo de Captura:"),
                                html.Ul([
                                    html.Li([
                                        html.Strong("Hardware NI PXIe-5185: "),
                                        "Para captura real. Configure el nombre del dispositivo (ej: PXI1Slot2), "
                                        "canal y frecuencia de muestreo."
                                    ]),
                                    html.Li([
                                        html.Strong("Modo Simulación: "),
                                        "Para pruebas sin hardware. Seleccione el estado a simular y nivel de ruido."
                                    ])
                                ])
                            ]),
                            html.Li([
                                html.Strong("Inicie la Captura: "),
                                "Haga clic en 'Iniciar Captura'. El sistema comenzará a adquirir y procesar datos."
                            ]),
                            html.Li([
                                html.Strong("Monitoree en Tiempo Real: "),
                                "Observe la señal, descriptores y el indicador de severidad que se actualizan continuamente."
                            ]),
                            html.Li([
                                html.Strong("Detenga la Captura: "),
                                "Haga clic en 'Detener Captura' cuando termine. Use 'Limpiar Buffers' para reiniciar."
                            ])
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Guía de Uso - Análisis de Archivos
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3([
                        html.I(className="fas fa-folder-open me-2"),
                        "📂 Análisis de Archivos"
                    ])),
                    dbc.CardBody([
                        html.H5("Propósito:", className="fw-bold"),
                        html.P("Análisis offline de datos previamente capturados."),
                        
                        html.H5("Formatos Soportados:", className="fw-bold mt-3"),
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.Strong("CSV: "),
                                "Archivos de texto con columnas. Especifique el nombre de la columna con la señal."
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("HDF5 (.h5): "),
                                "Formato jerárquico. Indique el nombre del dataset."
                            ]),
                            dbc.ListGroupItem([
                                html.Strong("MATLAB (.mat): "),
                                "Archivos MATLAB. Especifique el nombre de la variable."
                            ])
                        ], className="mb-3"),
                        
                        html.H5("Cómo Usar:", className="fw-bold mt-3"),
                        html.Ol([
                            html.Li("Arrastre o seleccione un archivo soportado."),
                            html.Li("Configure la frecuencia de muestreo y nombre de columna/variable."),
                            html.Li("Si su archivo incluye tiempo, active 'Con columna de tiempo'."),
                            html.Li("Haga clic en 'Analizar Señal'."),
                            html.Li("Revise las visualizaciones: señal procesada, espectro, descriptores y clasificación.")
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Guía de Uso - Generador de Señales
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3([
                        html.I(className="fas fa-cogs me-2"),
                        "⚙️ Generador de Señales"
                    ])),
                    dbc.CardBody([
                        html.H5("Propósito:", className="fw-bold"),
                        html.P("Crear señales sintéticas de descargas parciales con parámetros personalizables."),
                        
                        html.H5("Parámetros Configurables:", className="fw-bold mt-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Ul([
                                    html.Li([html.Strong("Estado Operativo: "), "Verde, Amarillo, Naranja o Rojo"]),
                                    html.Li([html.Strong("Duración: "), "Número de muestras"]),
                                    html.Li([html.Strong("Frecuencia de Muestreo: "), "En Hz"]),
                                    html.Li([html.Strong("Número de Descargas: "), "Cantidad de pulsos"]),
                                ])
                            ], width=6),
                            dbc.Col([
                                html.Ul([
                                    html.Li([html.Strong("Amplitud de Pulsos: "), "Intensidad"]),
                                    html.Li([html.Strong("Frecuencia de Oscilación: "), "En Hz"]),
                                    html.Li([html.Strong("Tipo de Ruido: "), "Gaussiano, Rosa, Marrón, Uniforme"]),
                                    html.Li([html.Strong("Nivel de Ruido: "), "0 a 1"]),
                                ])
                            ], width=6)
                        ]),
                        
                        html.H5("Cómo Usar:", className="fw-bold mt-3"),
                        html.Ol([
                            html.Li("Ajuste los parámetros deseados o use 'Parámetros Aleatorios'."),
                            html.Li("Haga clic en 'Generar Señal'."),
                            html.Li("Revise la señal, espectro, histograma y estadísticas."),
                            html.Li("Para exportar: seleccione formato (CSV, H5, MAT), nombre y metadatos."),
                            html.Li("Haga clic en 'Exportar'. El archivo se guardará en /tmp/")
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Guía de Uso - Configuración de Umbrales
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3([
                        html.I(className="fas fa-sliders-h me-2"),
                        "🎯 Configuración de Umbrales"
                    ])),
                    dbc.CardBody([
                        html.H5("Propósito:", className="fw-bold"),
                        html.P("Personalizar los criterios de clasificación según las necesidades específicas."),
                        
                        html.H5("Funcionalidades:", className="fw-bold mt-3"),
                        html.Ul([
                            html.Li([
                                html.Strong("Ajuste de Umbrales: "),
                                "Configure los valores de transición entre estados (Verde→Amarillo, Amarillo→Naranja, Naranja→Rojo)."
                            ]),
                            html.Li([
                                html.Strong("Pesos de Descriptores: "),
                                "Asigne importancia relativa a cada descriptor en el cálculo de severidad."
                            ]),
                            html.Li([
                                html.Strong("Pruebas Individuales: "),
                                "Genere señales de cada estado y verifique la clasificación."
                            ]),
                            html.Li([
                                html.Strong("Prueba Completa: "),
                                "Ejecute una evaluación con 20 muestras por estado y vea la matriz de confusión."
                            ])
                        ]),
                        
                        dbc.Alert([
                            html.I(className="fas fa-lightbulb me-2"),
                            html.Strong("Consejo: "),
                            "Los valores por defecto están optimizados para la mayoría de casos. "
                            "Ajústelos solo si tiene requisitos específicos o experiencia en su aplicación."
                        ], color="info")
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Especificaciones Técnicas
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H3([
                        html.I(className="fas fa-cog me-2"),
                        "Especificaciones Técnicas"
                    ])),
                    dbc.CardBody([
                        html.H5("Hardware Compatible:", className="fw-bold"),
                        dbc.Table([
                            html.Tbody([
                                html.Tr([html.Td("Sistema:", className="fw-bold"), 
                                        html.Td("NI PXIe-1071")]),
                                html.Tr([html.Td("Controlador:", className="fw-bold"), 
                                        html.Td("NI PXIe-8135 (Embebido)")]),
                                html.Tr([html.Td("Tarjeta de Adquisición:", className="fw-bold"), 
                                        html.Td("NI PXIe-5185")]),
                                html.Tr([html.Td("Ancho de Banda:", className="fw-bold"), 
                                        html.Td("3 GHz")]),
                                html.Tr([html.Td("Frecuencia de Muestreo:", className="fw-bold"), 
                                        html.Td("12.5 GS/s")]),
                                html.Tr([html.Td("Resolución Vertical:", className="fw-bold"), 
                                        html.Td("8 bits")]),
                            ])
                        ], bordered=True, className="mb-3"),
                        
                        html.H5("Descriptores Calculados:", className="fw-bold mt-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Ol([
                                    html.Li("Energía Total"),
                                    html.Li("RMS (Root Mean Square)"),
                                    html.Li("Curtosis"),
                                    html.Li("Asimetría"),
                                    html.Li("Factor de Cresta"),
                                ])
                            ], width=6),
                            dbc.Col([
                                html.Ol(start=6, children=[
                                    html.Li("Conteo de Picos"),
                                    html.Li("Entropía Espectral"),
                                    html.Li("Estabilidad Espectral"),
                                    html.Li("Tasa de Cruces por Cero")
                                ])
                            ], width=6)
                        ]),
                        
                        html.H5("Procesamiento de Señal:", className="fw-bold mt-3"),
                        html.Ul([
                            html.Li("Filtrado pasa-banda (1% - 40% de fs)"),
                            html.Li("Normalización adaptativa"),
                            html.Li("Extracción de envolvente (Transformada de Hilbert)"),
                            html.Li("Reducción de ruido (Wavelets)"),
                        ])
                    ])
                ], className="mb-4")
            ])
        ]),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.Div([
                    html.P([
                        html.I(className="fas fa-code me-2"),
                        "Sistema desarrollado con Python, Dash, Plotly y librerías científicas."
                    ], className="text-muted text-center"),
                    html.P([
                        html.I(className="fas fa-github me-2"),
                        "Para soporte técnico o reportar problemas, consulte la documentación del proyecto."
                    ], className="text-muted text-center")
                ], className="mb-4")
            ])
        ])
        
    ], fluid=True, className="py-4")


def register_callbacks(app):
    """No se requieren callbacks para documentación."""
    pass
