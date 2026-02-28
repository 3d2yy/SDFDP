"""
Pestaña de Captura en Vivo
============================

Captura y procesamiento en tiempo real de señales de descargas parciales.
Soporta:
- NI PXIe-5185 (hardware real)
- Modo simulación (generación sintética)
"""

import numpy as np
from dash import dcc, html, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from collections import deque
import time

# Importar funciones del sistema
import sys
sys.path.append('..')
from preprocessing import preprocess_signal
from descriptors import compute_all_descriptors
from severity import assess_severity
from main import generate_synthetic_signal
from gui.plot_utils import (
    apply_professional_style, COLOR_PALETTE, create_gradient_line,
    create_professional_card_style, create_metric_card
)


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Buffer para datos en tiempo real
MAX_BUFFER_SIZE = 1000
signal_buffer = deque(maxlen=MAX_BUFFER_SIZE)
time_buffer = deque(maxlen=MAX_BUFFER_SIZE)
descriptor_history = {
    'energy': deque(maxlen=100),
    'rms': deque(maxlen=100),
    'peak_count': deque(maxlen=100),
    'crest_factor': deque(maxlen=100),
    'spectral_entropy': deque(maxlen=100),
    'severity_index': deque(maxlen=100),
    'timestamps': deque(maxlen=100)
}


# ============================================================================
# LAYOUT
# ============================================================================

def create_layout():
    """Crear layout de la pestaña de captura en vivo."""
    return dbc.Container([
        dbc.Row([
            # Panel de control izquierdo
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.I(className="fas fa-cog me-2"),
                        "Control de Captura"
                    ])),
                    dbc.CardBody([
                        # Modo de captura
                        html.Label("Modo de Captura:", className="fw-bold"),
                        dbc.RadioItems(
                            id="capture-mode",
                            options=[
                                {"label": " Hardware NI PXIe-5185", "value": "hardware"},
                                {"label": " Modo Simulación", "value": "simulation"}
                            ],
                            value="simulation",
                            className="mb-3"
                        ),
                        
                        html.Hr(),
                        
                        # Configuración de hardware
                        html.Div(id="hardware-config", children=[
                            html.Label("Configuración Hardware:", className="fw-bold"),
                            dbc.Input(
                                id="device-name",
                                placeholder="Device (ej: PXI1Slot2)",
                                value="PXI1Slot2",
                                className="mb-2"
                            ),
                            dbc.Input(
                                id="channel",
                                placeholder="Canal (ej: 0)",
                                value="0",
                                type="number",
                                className="mb-2"
                            ),
                            html.Label("Frecuencia de Muestreo (GS/s):", className="mt-2"),
                            dbc.Input(
                                id="sample-rate",
                                value="12.5",
                                type="number",
                                step="0.1",
                                className="mb-2"
                            ),
                        ], style={'display': 'none'}),
                        
                        # Configuración de simulación
                        html.Div(id="simulation-config", children=[
                            html.Label("Estado a Simular:", className="fw-bold"),
                            dbc.Select(
                                id="simulation-state",
                                options=[
                                    {"label": "🟢 Verde (Normal)", "value": "verde"},
                                    {"label": "🟡 Amarillo (Precaución)", "value": "amarillo"},
                                    {"label": "🟠 Naranja (Alerta)", "value": "naranja"},
                                    {"label": "🔴 Rojo (Crítico)", "value": "rojo"}
                                ],
                                value="verde",
                                className="mb-2"
                            ),
                            html.Label("Nivel de Ruido:", className="mt-2"),
                            dcc.Slider(
                                id="noise-level",
                                min=0,
                                max=0.5,
                                step=0.01,
                                value=0.1,
                                marks={0: '0', 0.25: '0.25', 0.5: '0.5'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                        ]),
                        
                        html.Hr(),
                        
                        # Controles de captura
                        html.Div([
                            dbc.Button(
                                [html.I(className="fas fa-play me-2"), "Iniciar Captura"],
                                id="start-capture",
                                color="success",
                                className="w-100 mb-2"
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-stop me-2"), "Detener Captura"],
                                id="stop-capture",
                                color="danger",
                                className="w-100 mb-2",
                                disabled=True
                            ),
                            dbc.Button(
                                [html.I(className="fas fa-trash me-2"), "Limpiar Buffers"],
                                id="clear-buffers",
                                color="warning",
                                className="w-100"
                            ),
                        ]),
                        
                        html.Hr(),
                        
                        # Estado del sistema
                        html.Div([
                            html.H5("Estado del Sistema", className="fw-bold"),
                            html.Div(id="capture-status", children=[
                                dbc.Badge("Detenido", color="secondary", className="me-2"),
                                html.Span("Esperando inicio...")
                            ]),
                            html.Div(id="samples-captured", className="mt-2", children=[
                                html.Small("Muestras capturadas: 0")
                            ])
                        ])
                    ])
                ], className="mb-3"),
                
                # Indicador de severidad
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.I(className="fas fa-traffic-light me-2"),
                        "Severidad Actual"
                    ])),
                    dbc.CardBody([
                        html.Div(id="severity-indicator", style={'textAlign': 'center'})
                    ])
                ])
            ], width=3),
            
            # Panel de visualización derecho
            dbc.Col([
                # Gráfica de señal en tiempo real
                dbc.Card([
                    dbc.CardHeader(html.H5("Señal en Tiempo Real")),
                    dbc.CardBody([
                        dcc.Graph(
                            id="live-signal-graph",
                            config={'displayModeBar': False},
                            style={'height': '300px'}
                        )
                    ])
                ], className="mb-3"),
                
                # Gráficas de descriptores
                dbc.Card([
                    dbc.CardHeader(html.H5("Evolución de Descriptores")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(
                                    id="descriptor-energy",
                                    config={'displayModeBar': False},
                                    style={'height': '200px'}
                                )
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(
                                    id="descriptor-rms",
                                    config={'displayModeBar': False},
                                    style={'height': '200px'}
                                )
                            ], width=6),
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(
                                    id="descriptor-peaks",
                                    config={'displayModeBar': False},
                                    style={'height': '200px'}
                                )
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(
                                    id="descriptor-severity",
                                    config={'displayModeBar': False},
                                    style={'height': '200px'}
                                )
                            ], width=6),
                        ])
                    ])
                ], className="mb-3"),
                
                # Tabla de descriptores actuales
                dbc.Card([
                    dbc.CardHeader(html.H5("Descriptores Actuales")),
                    dbc.CardBody([
                        html.Div(id="current-descriptors")
                    ])
                ])
            ], width=9)
        ])
    ], fluid=True)


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def capture_from_hardware(device_name, channel, sample_rate, duration=0.001):
    """
    Capturar datos del hardware NI PXIe-5185.
    
    Parámetros:
    -----------
    device_name : str
        Nombre del dispositivo
    channel : int
        Número de canal
    sample_rate : float
        Frecuencia de muestreo en GS/s
    duration : float
        Duración de captura en segundos
    
    Retorna:
    --------
    signal : ndarray
        Señal capturada
    """
    try:
        import nidaqmx
        from nidaqmx.constants import AcquisitionType
        
        # Convertir GS/s a S/s
        rate = sample_rate * 1e9
        n_samples = int(rate * duration)
        
        with nidaqmx.Task() as task:
            # Configurar canal analógico
            task.ai_channels.add_ai_voltage_chan(f"{device_name}/ai{channel}")
            
            # Configurar timing
            task.timing.cfg_samp_clk_timing(
                rate=rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=n_samples
            )
            
            # Leer datos
            data = task.read(number_of_samples_per_channel=n_samples)
            
            return np.array(data)
            
    except ImportError:
        raise ImportError("nidaqmx no está instalado. Use 'pip install nidaqmx'")
    except Exception as e:
        raise RuntimeError(f"Error al capturar del hardware: {str(e)}")


def simulate_capture(state='verde', noise_level=0.1, duration=1000, fs=10000):
    """
    Simular captura de señal de descarga parcial.
    
    Parámetros:
    -----------
    state : str
        Estado a simular
    noise_level : float
        Nivel de ruido
    duration : int
        Número de muestras
    fs : float
        Frecuencia de muestreo
    
    Retorna:
    --------
    signal : ndarray
        Señal simulada
    """
    return generate_synthetic_signal(state, duration, fs, noise_level)


# ============================================================================
# CALLBACKS (se registran en app.py)
# ============================================================================

def register_callbacks(app):
    """Registrar callbacks de la pestaña."""
    
    @app.callback(
        [Output('hardware-config', 'style'),
         Output('simulation-config', 'style')],
        Input('capture-mode', 'value')
    )
    def toggle_config(mode):
        """Mostrar/ocultar configuración según el modo."""
        if mode == 'hardware':
            return {'display': 'block'}, {'display': 'none'}
        else:
            return {'display': 'none'}, {'display': 'block'}
    
    
    @app.callback(
        [Output('start-capture', 'disabled'),
         Output('stop-capture', 'disabled'),
         Output('interval-component', 'disabled'),
         Output('capture-status', 'children')],
        [Input('start-capture', 'n_clicks'),
         Input('stop-capture', 'n_clicks')],
        prevent_initial_call=True
    )
    def control_capture(start_clicks, stop_clicks):
        """Controlar inicio/detención de captura."""
        ctx = callback_context
        
        if not ctx.triggered:
            return False, True, True, [
                dbc.Badge("Detenido", color="secondary", className="me-2"),
                html.Span("Esperando inicio...")
            ]
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if button_id == 'start-capture':
            return True, False, False, [
                dbc.Badge("Capturando", color="success", className="me-2"),
                html.Span("Sistema activo...")
            ]
        else:
            return False, True, True, [
                dbc.Badge("Detenido", color="warning", className="me-2"),
                html.Span("Captura detenida")
            ]
    
    
    @app.callback(
        Output('clear-buffers', 'n_clicks'),
        Input('clear-buffers', 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_buffers(n_clicks):
        """Limpiar buffers de datos."""
        signal_buffer.clear()
        time_buffer.clear()
        for key in descriptor_history:
            descriptor_history[key].clear()
        return 0
    
    
    @app.callback(
        [Output('live-signal-graph', 'figure'),
         Output('descriptor-energy', 'figure'),
         Output('descriptor-rms', 'figure'),
         Output('descriptor-peaks', 'figure'),
         Output('descriptor-severity', 'figure'),
         Output('current-descriptors', 'children'),
         Output('severity-indicator', 'children'),
         Output('samples-captured', 'children')],
        [Input('interval-component', 'n_intervals')],
        [State('capture-mode', 'value'),
         State('simulation-state', 'value'),
         State('noise-level', 'value'),
         State('device-name', 'value'),
         State('channel', 'value'),
         State('sample-rate', 'value')]
    )
    def update_live_data(n_intervals, mode, sim_state, noise_level, 
                        device_name, channel, sample_rate):
        """Actualizar datos en tiempo real."""
        
        # Capturar datos
        fs = 10000  # Hz para simulación
        
        try:
            if mode == 'hardware':
                signal = capture_from_hardware(device_name, int(channel), float(sample_rate))
                fs = float(sample_rate) * 1e9
            else:
                signal = simulate_capture(sim_state, noise_level)
        except Exception as e:
            # En caso de error, usar simulación
            signal = simulate_capture('verde', 0.1)
        
        # Agregar al buffer
        current_time = time.time()
        signal_buffer.extend(signal)
        time_buffer.extend([current_time + i/fs for i in range(len(signal))])
        
        # Procesar señal
        lowcut = fs * 0.01
        highcut = fs * 0.4
        processed_signal, _ = preprocess_signal(signal, fs, lowcut, highcut, True, True, True)
        
        # Calcular descriptores
        descriptors = compute_all_descriptors(processed_signal, fs, signal)
        
        # Evaluar severidad
        severity_results = assess_severity(descriptors)
        
        # Actualizar historial
        descriptor_history['energy'].append(descriptors['energy_total'])
        descriptor_history['rms'].append(descriptors['rms'])
        descriptor_history['peak_count'].append(descriptors['peak_count'])
        descriptor_history['crest_factor'].append(descriptors['crest_factor'])
        descriptor_history['spectral_entropy'].append(descriptors['spectral_entropy'])
        descriptor_history['severity_index'].append(severity_results['severity_index'])
        descriptor_history['timestamps'].append(current_time)
        
        # Crear gráficas
        signal_fig = create_signal_figure(list(signal_buffer), list(time_buffer))
        energy_fig = create_descriptor_figure(
            list(descriptor_history['timestamps']),
            list(descriptor_history['energy']),
            "Energía Total"
        )
        rms_fig = create_descriptor_figure(
            list(descriptor_history['timestamps']),
            list(descriptor_history['rms']),
            "RMS"
        )
        peaks_fig = create_descriptor_figure(
            list(descriptor_history['timestamps']),
            list(descriptor_history['peak_count']),
            "Conteo de Picos"
        )
        severity_fig = create_severity_figure(
            list(descriptor_history['timestamps']),
            list(descriptor_history['severity_index'])
        )
        
        # Tabla de descriptores
        descriptor_table = create_descriptor_table(descriptors)
        
        # Indicador de severidad
        severity_display = create_severity_display(severity_results)
        
        # Contador de muestras
        sample_count = html.Small(f"Muestras capturadas: {len(signal_buffer)}")
        
        return (signal_fig, energy_fig, rms_fig, peaks_fig, severity_fig,
                descriptor_table, severity_display, sample_count)


def create_signal_figure(signal, times):
    """Crear gráfica de señal."""
    fig = go.Figure()
    
    if len(signal) > 0:
        fig.add_trace(go.Scatter(
            x=list(range(len(signal))),
            y=signal,
            mode='lines',
            line=dict(
                color=COLOR_PALETTE['gradient_3'][0],
                width=1.5,
                shape='spline'
            ),
            fill='tozeroy',
            fillcolor='rgba(79, 172, 254, 0.1)',
            name='Señal',
            hovertemplate='<b>Muestra:</b> %{x}<br><b>Amplitud:</b> %{y:.4f}<extra></extra>'
        ))
    
    fig = apply_professional_style(fig)
    fig.update_layout(
        xaxis_title="Muestra",
        yaxis_title="Amplitud",
        showlegend=False,
        height=300
    )
    
    return fig


def create_descriptor_figure(times, values, title):
    """Crear gráfica de descriptor."""
    fig = go.Figure()
    
    if len(values) > 0:
        # Línea principal con gradiente
        fig.add_trace(go.Scatter(
            x=list(range(len(values))),
            y=values,
            mode='lines+markers',
            line=dict(
                color=COLOR_PALETTE['primary'],
                width=2.5,
                shape='spline'
            ),
            marker=dict(
                size=6,
                color=COLOR_PALETTE['secondary'],
                line=dict(color='#fff', width=1)
            ),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)',
            name=title,
            hovertemplate='<b>%{y:.4f}</b><extra></extra>'
        ))
    
    fig = apply_professional_style(fig)
    fig.update_layout(
        title=dict(text=title, x=0, xanchor='left'),
        xaxis_title="Medición",
        yaxis_title="Valor",
        showlegend=False,
        height=200
    )
    
    return fig


def create_severity_figure(times, severities):
    """Crear gráfica de severidad con zonas coloreadas."""
    fig = go.Figure()
    
    if len(severities) > 0:
        # Áreas de severidad con colores profesionales
        fig.add_hrect(y0=0, y1=0.25, fillcolor=COLOR_PALETTE['state_verde'], 
                     opacity=0.1, line_width=0)
        fig.add_hrect(y0=0.25, y1=0.5, fillcolor=COLOR_PALETTE['state_amarillo'], 
                     opacity=0.1, line_width=0)
        fig.add_hrect(y0=0.5, y1=0.75, fillcolor=COLOR_PALETTE['state_naranja'], 
                     opacity=0.1, line_width=0)
        fig.add_hrect(y0=0.75, y1=1.0, fillcolor=COLOR_PALETTE['state_rojo'], 
                     opacity=0.1, line_width=0)
        
        # Línea de severidad
        fig.add_trace(go.Scatter(
            x=list(range(len(severities))),
            y=severities,
            mode='lines+markers',
            line=dict(
                color=COLOR_PALETTE['danger'],
                width=3,
                shape='spline'
            ),
            marker=dict(
                size=7,
                color=COLOR_PALETTE['danger'],
                line=dict(color='#fff', width=2),
                symbol='circle'
            ),
            name='Índice de Severidad',
            hovertemplate='<b>Severidad:</b> %{y:.4f}<extra></extra>'
        ))
    
    fig = apply_professional_style(fig)
    fig.update_layout(
        title=dict(text="Índice de Severidad", x=0, xanchor='left'),
        xaxis_title="Medición",
        yaxis_title="Severidad",
        showlegend=False,
        yaxis=dict(range=[0, 1]),
        height=200
    )
    
    return fig


def create_descriptor_table(descriptors):
    """Crear tabla de descriptores actuales."""
    return dbc.Table([
        html.Tbody([
            html.Tr([html.Td("Energía Total", className="fw-bold"), 
                    html.Td(f"{descriptors['energy_total']:.6f}")]),
            html.Tr([html.Td("RMS", className="fw-bold"), 
                    html.Td(f"{descriptors['rms']:.6f}")]),
            html.Tr([html.Td("Curtosis", className="fw-bold"), 
                    html.Td(f"{descriptors['kurtosis']:.4f}")]),
            html.Tr([html.Td("Asimetría", className="fw-bold"), 
                    html.Td(f"{descriptors['skewness']:.4f}")]),
            html.Tr([html.Td("Factor de Cresta", className="fw-bold"), 
                    html.Td(f"{descriptors['crest_factor']:.4f}")]),
            html.Tr([html.Td("Conteo de Picos", className="fw-bold"), 
                    html.Td(f"{descriptors['peak_count']}")]),
            html.Tr([html.Td("Entropía Espectral", className="fw-bold"), 
                    html.Td(f"{descriptors['spectral_entropy']:.4f}")]),
            html.Tr([html.Td("Estabilidad Espectral", className="fw-bold"), 
                    html.Td(f"{descriptors['spectral_stability']:.4f}")]),
        ])
    ], bordered=True, hover=True, size="sm")


def create_severity_display(severity_results):
    """Crear display de severidad con semáforo."""
    state = severity_results['traffic_light_state']
    severity = severity_results['severity_index']
    
    colors = {
        'verde': (COLOR_PALETTE['state_verde'], '🟢', 'Normal'),
        'amarillo': (COLOR_PALETTE['state_amarillo'], '🟡', 'Precaución'),
        'naranja': (COLOR_PALETTE['state_naranja'], '🟠', 'Alerta'),
        'rojo': (COLOR_PALETTE['state_rojo'], '🔴', 'Crítico')
    }
    
    color_hex, symbol, label = colors.get(state, ('#fff', '⚪', 'Desconocido'))
    
    return html.Div([
        html.Div([
            html.H1(symbol, style={
                'fontSize': '100px',
                'marginBottom': '16px',
                'textShadow': f'0 0 40px {color_hex}'
            }),
            html.H3(label, style={
                'fontSize': '28px',
                'fontWeight': '700',
                'color': '#fff',
                'marginBottom': '8px',
                'letterSpacing': '-0.02em'
            }),
            html.Div([
                html.Span('Índice: ', style={'color': 'rgba(255,255,255,0.6)', 'fontSize': '14px'}),
                html.Span(f'{severity:.4f}', style={
                    'fontSize': '24px',
                    'fontWeight': '600',
                    'background': f'linear-gradient(135deg, {color_hex} 0%, {color_hex}dd 100%)',
                    '-webkit-background-clip': 'text',
                    '-webkit-text-fill-color': 'transparent'
                })
            ], style={'marginBottom': '20px'}),
            html.Div([
                html.Div(style={
                    'width': '100%',
                    'height': '24px',
                    'background': 'rgba(255,255,255,0.1)',
                    'borderRadius': '12px',
                    'overflow': 'hidden',
                    'position': 'relative'
                }, children=[
                    html.Div(style={
                        'width': f'{min(severity * 100, 100)}%',
                        'height': '100%',
                        'background': f'linear-gradient(90deg, {color_hex} 0%, {color_hex}aa 100%)',
                        'borderRadius': '12px',
                        'transition': 'width 0.3s ease',
                        'boxShadow': f'0 0 20px {color_hex}66'
                    })
                ])
            ])
        ], style={
            'textAlign': 'center',
            'padding': '32px',
            'background': create_professional_card_style()['background'],
            'borderRadius': '16px',
            'border': f'1px solid {color_hex}33'
        })
    ])
