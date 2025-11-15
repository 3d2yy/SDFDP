"""
Pestaña de Configuración de Umbrales
=====================================

Permite configurar y visualizar umbrales de clasificación para el sistema de detección.
"""

import numpy as np
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# Importar funciones del sistema
import sys
sys.path.append('..')
from main import generate_synthetic_signal
from preprocessing import preprocess_signal
from descriptors import compute_all_descriptors
from severity import assess_severity, create_baseline_profile


# ============================================================================
# LAYOUT
# ============================================================================

def create_layout():
    """Crear layout de configuración de umbrales."""
    return dbc.Container([
        dbc.Row([
            # Panel de control
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.I(className="fas fa-sliders-h me-2"),
                        "Configuración de Umbrales"
                    ])),
                    dbc.CardBody([
                        html.P("Ajuste los umbrales de clasificación para cada nivel de severidad.",
                              className="text-muted"),
                        
                        html.Hr(),
                        
                        # Umbral Verde → Amarillo
                        html.Label("🟢 Verde → 🟡 Amarillo", className="fw-bold"),
                        dcc.Slider(
                            id="threshold-green-yellow",
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.25,
                            marks={i/10: str(i/10) for i in range(0, 11, 2)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Small("Nivel de severidad para transición a precaución", 
                                  className="text-muted"),
                        
                        html.Br(), html.Br(),
                        
                        # Umbral Amarillo → Naranja
                        html.Label("🟡 Amarillo → 🟠 Naranja", className="fw-bold mt-3"),
                        dcc.Slider(
                            id="threshold-yellow-orange",
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.5,
                            marks={i/10: str(i/10) for i in range(0, 11, 2)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Small("Nivel de severidad para transición a alerta", 
                                  className="text-muted"),
                        
                        html.Br(), html.Br(),
                        
                        # Umbral Naranja → Rojo
                        html.Label("🟠 Naranja → 🔴 Rojo", className="fw-bold mt-3"),
                        dcc.Slider(
                            id="threshold-orange-red",
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.75,
                            marks={i/10: str(i/10) for i in range(0, 11, 2)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Small("Nivel de severidad para transición a crítico", 
                                  className="text-muted"),
                        
                        html.Hr(),
                        
                        # Pesos de descriptores
                        html.H5("Pesos de Descriptores", className="fw-bold mt-3"),
                        html.P("Ajuste la importancia de cada descriptor en el cálculo de severidad.",
                              className="text-muted small"),
                        
                        html.Label("Energía Total:", className="mt-2"),
                        dcc.Slider(
                            id="weight-energy",
                            min=0, max=3, step=0.1, value=2.0,
                            marks={0: '0', 1.5: '1.5', 3: '3'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        
                        html.Label("RMS:", className="mt-2"),
                        dcc.Slider(
                            id="weight-rms",
                            min=0, max=3, step=0.1, value=2.0,
                            marks={0: '0', 1.5: '1.5', 3: '3'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        
                        html.Label("Conteo de Picos:", className="mt-2"),
                        dcc.Slider(
                            id="weight-peaks",
                            min=0, max=3, step=0.1, value=2.5,
                            marks={0: '0', 1.5: '1.5', 3: '3'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        
                        html.Label("Factor de Cresta:", className="mt-2"),
                        dcc.Slider(
                            id="weight-crest",
                            min=0, max=3, step=0.1, value=1.5,
                            marks={0: '0', 1.5: '1.5', 3: '3'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        
                        html.Label("Entropía Espectral:", className="mt-2"),
                        dcc.Slider(
                            id="weight-entropy",
                            min=0, max=3, step=0.1, value=1.5,
                            marks={0: '0', 1.5: '1.5', 3: '3'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        
                        html.Hr(),
                        
                        # Botones
                        dbc.Button(
                            [html.I(className="fas fa-sync me-2"), "Aplicar y Probar"],
                            id="apply-thresholds-btn",
                            color="primary",
                            className="w-100 mb-2"
                        ),
                        
                        dbc.Button(
                            [html.I(className="fas fa-undo me-2"), "Restaurar Valores por Defecto"],
                            id="reset-thresholds-btn",
                            color="secondary",
                            className="w-100"
                        )
                    ])
                ])
            ], width=4),
            
            # Panel de visualización
            dbc.Col([
                # Visualización de umbrales
                dbc.Card([
                    dbc.CardHeader(html.H5("Visualización de Umbrales")),
                    dbc.CardBody([
                        dcc.Graph(
                            id="threshold-visualization",
                            config={'displayModeBar': False},
                            style={'height': '300px'}
                        )
                    ])
                ], className="mb-3"),
                
                # Prueba con datos sintéticos
                dbc.Card([
                    dbc.CardHeader(html.H5("Prueba con Datos Sintéticos")),
                    dbc.CardBody([
                        html.P("Genere señales sintéticas y vea cómo se clasifican con los umbrales actuales."),
                        
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "🟢 Generar Verde",
                                    id="test-verde-btn",
                                    color="success",
                                    outline=True,
                                    className="w-100 mb-2"
                                ),
                            ], width=3),
                            dbc.Col([
                                dbc.Button(
                                    "🟡 Generar Amarillo",
                                    id="test-amarillo-btn",
                                    color="warning",
                                    outline=True,
                                    className="w-100 mb-2"
                                ),
                            ], width=3),
                            dbc.Col([
                                dbc.Button(
                                    "🟠 Generar Naranja",
                                    id="test-naranja-btn",
                                    color="warning",
                                    outline=True,
                                    className="w-100 mb-2"
                                ),
                            ], width=3),
                            dbc.Col([
                                dbc.Button(
                                    "🔴 Generar Rojo",
                                    id="test-rojo-btn",
                                    color="danger",
                                    outline=True,
                                    className="w-100 mb-2"
                                ),
                            ], width=3),
                        ]),
                        
                        html.Hr(),
                        
                        html.Div(id="test-result-display")
                    ])
                ], className="mb-3"),
                
                # Matriz de confusión
                dbc.Card([
                    dbc.CardHeader(html.H5("Prueba de Clasificación")),
                    dbc.CardBody([
                        dbc.Button(
                            [html.I(className="fas fa-flask me-2"), "Ejecutar Prueba Completa"],
                            id="run-full-test-btn",
                            color="info",
                            className="mb-3"
                        ),
                        
                        html.Div(id="confusion-matrix-display")
                    ])
                ])
            ], width=8)
        ])
    ], fluid=True)


# ============================================================================
# CALLBACKS (se registran en app.py)
# ============================================================================

def register_callbacks(app):
    """Registrar callbacks de la pestaña."""
    
    @app.callback(
        [Output('threshold-green-yellow', 'value'),
         Output('threshold-yellow-orange', 'value'),
         Output('threshold-orange-red', 'value'),
         Output('weight-energy', 'value'),
         Output('weight-rms', 'value'),
         Output('weight-peaks', 'value'),
         Output('weight-crest', 'value'),
         Output('weight-entropy', 'value')],
        Input('reset-thresholds-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def reset_to_defaults(n_clicks):
        """Restaurar valores por defecto."""
        return 0.25, 0.5, 0.75, 2.0, 2.0, 2.5, 1.5, 1.5
    
    
    @app.callback(
        Output('threshold-visualization', 'figure'),
        [Input('threshold-green-yellow', 'value'),
         Input('threshold-yellow-orange', 'value'),
         Input('threshold-orange-red', 'value')]
    )
    def update_threshold_visualization(t1, t2, t3):
        """Actualizar visualización de umbrales."""
        fig = go.Figure()
        
        # Áreas de severidad
        fig.add_trace(go.Scatter(
            x=[0, t1, t1, 0, 0],
            y=[0, 0, 1, 1, 0],
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.2)',
            line=dict(width=0),
            name='Verde',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=[t1, t2, t2, t1, t1],
            y=[0, 0, 1, 1, 0],
            fill='toself',
            fillcolor='rgba(255, 255, 0, 0.2)',
            line=dict(width=0),
            name='Amarillo',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=[t2, t3, t3, t2, t2],
            y=[0, 0, 1, 1, 0],
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.2)',
            line=dict(width=0),
            name='Naranja',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=[t3, 1, 1, t3, t3],
            y=[0, 0, 1, 1, 0],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(width=0),
            name='Rojo',
            hoverinfo='skip'
        ))
        
        # Líneas de umbral
        fig.add_vline(x=t1, line_dash="dash", line_color="green", 
                     annotation_text=f"Verde-Amarillo: {t1:.2f}")
        fig.add_vline(x=t2, line_dash="dash", line_color="orange",
                     annotation_text=f"Amarillo-Naranja: {t2:.2f}")
        fig.add_vline(x=t3, line_dash="dash", line_color="red",
                     annotation_text=f"Naranja-Rojo: {t3:.2f}")
        
        fig.update_layout(
            xaxis_title="Índice de Severidad",
            yaxis_title="",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(showticklabels=False),
            template="plotly_white",
            margin=dict(l=50, r=20, t=20, b=40),
            showlegend=True
        )
        
        return fig
    
    
    @app.callback(
        Output('test-result-display', 'children'),
        [Input('test-verde-btn', 'n_clicks'),
         Input('test-amarillo-btn', 'n_clicks'),
         Input('test-naranja-btn', 'n_clicks'),
         Input('test-rojo-btn', 'n_clicks')],
        [State('threshold-green-yellow', 'value'),
         State('threshold-yellow-orange', 'value'),
         State('threshold-orange-red', 'value'),
         State('weight-energy', 'value'),
         State('weight-rms', 'value'),
         State('weight-peaks', 'value'),
         State('weight-crest', 'value'),
         State('weight-entropy', 'value')],
        prevent_initial_call=True
    )
    def test_classification(n_verde, n_amarillo, n_naranja, n_rojo,
                          t1, t2, t3, w_energy, w_rms, w_peaks, w_crest, w_entropy):
        """Probar clasificación con señal sintética."""
        
        ctx = callback_context
        if not ctx.triggered:
            return ""
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Determinar estado a generar
        state_map = {
            'test-verde-btn': 'verde',
            'test-amarillo-btn': 'amarillo',
            'test-naranja-btn': 'naranja',
            'test-rojo-btn': 'rojo'
        }
        
        state = state_map.get(button_id, 'verde')
        
        # Generar y procesar señal
        fs = 10000
        signal = generate_synthetic_signal(state, duration=1000, fs=fs)
        
        lowcut = fs * 0.01
        highcut = fs * 0.4
        processed_signal, _ = preprocess_signal(signal, fs, lowcut, highcut, True, True, True)
        
        # Calcular descriptores
        descriptors = compute_all_descriptors(processed_signal, fs, signal)
        
        # Pesos personalizados
        custom_weights = {
            'energy_total': w_energy,
            'rms': w_rms,
            'peak_count': w_peaks,
            'crest_factor': w_crest,
            'spectral_entropy': w_entropy
        }
        
        # Evaluar severidad con umbrales personalizados
        severity_results = assess_severity(
            descriptors,
            custom_weights=custom_weights
        )
        
        severity_idx = severity_results['severity_index']
        
        # Clasificar según umbrales personalizados
        if severity_idx < t1:
            predicted = 'verde'
            color = 'success'
            symbol = '🟢'
        elif severity_idx < t2:
            predicted = 'amarillo'
            color = 'warning'
            symbol = '🟡'
        elif severity_idx < t3:
            predicted = 'naranja'
            color = 'warning'
            symbol = '🟠'
        else:
            predicted = 'rojo'
            color = 'danger'
            symbol = '🔴'
        
        # Resultado
        is_correct = (predicted == state)
        
        return dbc.Alert([
            html.H4([symbol, f" Resultado: {predicted.capitalize()}"], className="mb-3"),
            html.Hr(),
            html.P([
                html.Strong("Estado Generado: "),
                state.capitalize()
            ]),
            html.P([
                html.Strong("Estado Predicho: "),
                predicted.capitalize()
            ]),
            html.P([
                html.Strong("Índice de Severidad: "),
                f"{severity_idx:.4f}"
            ]),
            html.Hr(),
            html.Div([
                html.I(className=f"fas fa-{'check' if is_correct else 'times'}-circle me-2"),
                html.Strong("Clasificación Correcta" if is_correct else "Clasificación Incorrecta")
            ], className=f"text-{'success' if is_correct else 'danger'}")
        ], color=color)
    
    
    @app.callback(
        Output('confusion-matrix-display', 'children'),
        Input('run-full-test-btn', 'n_clicks'),
        [State('threshold-green-yellow', 'value'),
         State('threshold-yellow-orange', 'value'),
         State('threshold-orange-red', 'value'),
         State('weight-energy', 'value'),
         State('weight-rms', 'value'),
         State('weight-peaks', 'value'),
         State('weight-crest', 'value'),
         State('weight-entropy', 'value')],
        prevent_initial_call=True
    )
    def run_full_test(n_clicks, t1, t2, t3, w_energy, w_rms, w_peaks, w_crest, w_entropy):
        """Ejecutar prueba completa con múltiples muestras."""
        
        states = ['verde', 'amarillo', 'naranja', 'rojo']
        n_samples = 20
        fs = 10000
        
        confusion_matrix = {state: {'verde': 0, 'amarillo': 0, 'naranja': 0, 'rojo': 0} 
                           for state in states}
        
        custom_weights = {
            'energy_total': w_energy,
            'rms': w_rms,
            'peak_count': w_peaks,
            'crest_factor': w_crest,
            'spectral_entropy': w_entropy
        }
        
        # Generar y clasificar muestras
        for true_state in states:
            for _ in range(n_samples):
                signal = generate_synthetic_signal(true_state, duration=1000, fs=fs)
                
                lowcut = fs * 0.01
                highcut = fs * 0.4
                processed_signal, _ = preprocess_signal(signal, fs, lowcut, highcut, True, True, True)
                
                descriptors = compute_all_descriptors(processed_signal, fs, signal)
                severity_results = assess_severity(descriptors, custom_weights=custom_weights)
                severity_idx = severity_results['severity_index']
                
                # Clasificar con umbrales personalizados
                if severity_idx < t1:
                    predicted = 'verde'
                elif severity_idx < t2:
                    predicted = 'amarillo'
                elif severity_idx < t3:
                    predicted = 'naranja'
                else:
                    predicted = 'rojo'
                
                confusion_matrix[true_state][predicted] += 1
        
        # Calcular métricas
        total = n_samples * len(states)
        correct = sum(confusion_matrix[state][state] for state in states)
        accuracy = correct / total
        
        # Crear tabla de matriz de confusión
        matrix_table = dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Real \\ Predicho"),
                    html.Th("🟢 Verde"),
                    html.Th("🟡 Amarillo"),
                    html.Th("🟠 Naranja"),
                    html.Th("🔴 Rojo"),
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(f"🟢 Verde", className="fw-bold"),
                    html.Td(confusion_matrix['verde']['verde'], 
                           style={'backgroundColor': '#d4edda' if confusion_matrix['verde']['verde'] else ''}),
                    html.Td(confusion_matrix['verde']['amarillo']),
                    html.Td(confusion_matrix['verde']['naranja']),
                    html.Td(confusion_matrix['verde']['rojo']),
                ]),
                html.Tr([
                    html.Td(f"🟡 Amarillo", className="fw-bold"),
                    html.Td(confusion_matrix['amarillo']['verde']),
                    html.Td(confusion_matrix['amarillo']['amarillo'],
                           style={'backgroundColor': '#fff3cd' if confusion_matrix['amarillo']['amarillo'] else ''}),
                    html.Td(confusion_matrix['amarillo']['naranja']),
                    html.Td(confusion_matrix['amarillo']['rojo']),
                ]),
                html.Tr([
                    html.Td(f"🟠 Naranja", className="fw-bold"),
                    html.Td(confusion_matrix['naranja']['verde']),
                    html.Td(confusion_matrix['naranja']['amarillo']),
                    html.Td(confusion_matrix['naranja']['naranja'],
                           style={'backgroundColor': '#fff3cd' if confusion_matrix['naranja']['naranja'] else ''}),
                    html.Td(confusion_matrix['naranja']['rojo']),
                ]),
                html.Tr([
                    html.Td(f"🔴 Rojo", className="fw-bold"),
                    html.Td(confusion_matrix['rojo']['verde']),
                    html.Td(confusion_matrix['rojo']['amarillo']),
                    html.Td(confusion_matrix['rojo']['naranja']),
                    html.Td(confusion_matrix['rojo']['rojo'],
                           style={'backgroundColor': '#f8d7da' if confusion_matrix['rojo']['rojo'] else ''}),
                ])
            ])
        ], bordered=True, hover=True)
        
        return html.Div([
            dbc.Alert([
                html.H4(f"📊 Precisión Global: {accuracy:.2%}"),
                html.P(f"Correctas: {correct} / {total}")
            ], color="info"),
            
            html.H5("Matriz de Confusión", className="mt-3 mb-3"),
            matrix_table,
            
            html.Small("Cada celda muestra el número de clasificaciones. "
                      "Las celdas resaltadas indican clasificaciones correctas.",
                      className="text-muted")
        ])
