"""
Pestaña de Análisis de Archivos
=================================

Carga y análisis de archivos de señales en diferentes formatos:
- CSV
- HDF5 (.h5)
- MATLAB (.mat)
"""

import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import base64
import io

# Importar funciones del sistema
import sys
sys.path.append('..')
from preprocessing import preprocess_signal
from descriptors import compute_all_descriptors
from severity import assess_severity, create_baseline_profile
from gui.plot_utils import (
    apply_professional_style, COLOR_PALETTE,
    create_professional_card_style, create_metric_card,
)


# ============================================================================
# LAYOUT
# ============================================================================

def create_layout():
    """Crear layout de la pestaña de análisis de archivos."""
    return dbc.Container([
        dbc.Row([
            # Panel de carga de archivos
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.I(className="fas fa-upload me-2"),
                        "Cargar Archivo"
                    ])),
                    dbc.CardBody([
                        # Upload component
                        dcc.Upload(
                            id='upload-signal-file',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt fa-3x mb-3"),
                                html.Br(),
                                'Arrastre o ',
                                html.A('seleccione un archivo', style={'color': '#007bff'})
                            ]),
                            style={
                                'width': '100%',
                                'height': '150px',
                                'lineHeight': '150px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '12px',
                                'textAlign': 'center',
                                'backgroundColor': 'rgba(255,255,255,0.03)',
                                'borderColor': 'rgba(102,126,234,0.35)',
                                'color': 'rgba(255,255,255,0.5)',
                                'transition': 'all 300ms ease',
                            },
                            multiple=False
                        ),
                        
                        html.Div(id='file-upload-status', className='mt-3'),
                        
                        html.Hr(),
                        
                        # Configuración de carga
                        html.Label("Configuración de Lectura:", className="fw-bold mt-3"),
                        
                        html.Label("Frecuencia de Muestreo (Hz):", className="mt-2"),
                        dbc.Input(
                            id="file-sample-rate",
                            type="number",
                            value=10000,
                            placeholder="Ej: 10000"
                        ),
                        
                        html.Label("Columna/Campo de Datos:", className="mt-2"),
                        dbc.Input(
                            id="file-data-column",
                            placeholder="Para CSV: nombre de columna, MAT: nombre de variable",
                            value="signal"
                        ),
                        
                        html.Label("Formato de Tiempo:", className="mt-2"),
                        dbc.RadioItems(
                            id="time-format",
                            options=[
                                {"label": "Sin columna de tiempo", "value": "none"},
                                {"label": "Con columna de tiempo", "value": "column"}
                            ],
                            value="none",
                            className="mb-3"
                        ),
                        
                        html.Div(id="time-column-config", children=[
                            dbc.Input(
                                id="file-time-column",
                                placeholder="Nombre de columna de tiempo",
                                value="time"
                            )
                        ], style={'display': 'none'}),
                        
                        html.Hr(),
                        
                        # Botón de análisis
                        dbc.Button(
                            [html.I(className="fas fa-chart-line me-2"), "Analizar Señal"],
                            id="analyze-file-signal",
                            color="primary",
                            className="w-100 mt-3",
                            disabled=True
                        ),
                        
                        html.Hr(),
                        
                        # Información del archivo
                        html.Div(id='file-info', className='mt-3')
                    ])
                ], className="mb-3"),
                
                # Resultado de severidad
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.I(className="fas fa-exclamation-triangle me-2"),
                        "Resultado del Análisis"
                    ])),
                    dbc.CardBody([
                        html.Div(id="file-severity-result")
                    ])
                ])
            ], width=3),
            
            # Panel de visualización
            dbc.Col([
                # Señal original y procesada
                dbc.Card([
                    dbc.CardHeader(html.H5("Señales")),
                    dbc.CardBody([
                        dcc.Graph(
                            id="file-signal-comparison",
                            config={'displayModeBar': True},
                            style={'height': '400px'}
                        )
                    ])
                ], className="mb-3"),
                
                # Espectro de frecuencia
                dbc.Card([
                    dbc.CardHeader(html.H5("Análisis Espectral")),
                    dbc.CardBody([
                        dcc.Graph(
                            id="file-spectrum",
                            config={'displayModeBar': True},
                            style={'height': '300px'}
                        )
                    ])
                ], className="mb-3"),
                
                # Descriptores
                dbc.Card([
                    dbc.CardHeader(html.H5("Descriptores Calculados")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="file-descriptors-table")
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(
                                    id="file-descriptors-radar",
                                    config={'displayModeBar': False},
                                    style={'height': '300px'}
                                )
                            ], width=6)
                        ])
                    ])
                ])
            ], width=9)
        ])
    ], fluid=True)


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def parse_csv_file(contents, filename, data_column, time_column=None):
    """Parsear archivo CSV."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), comment='#')
        except UnicodeDecodeError:
            df = pd.read_csv(io.StringIO(decoded.decode('iso-8859-1')), comment='#')
        
        if data_column not in df.columns:
            # Intentar con la primera columna numérica
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                data_column = numeric_cols[0]
            else:
                raise ValueError(f"Columna '{data_column}' no encontrada")
        
        signal = df[data_column].values
        
        time = None
        if time_column and time_column in df.columns:
            time = df[time_column].values
        
        return signal, time, f"CSV cargado: {len(signal)} muestras"
        
    except Exception as e:
        raise ValueError(f"Error al parsear CSV: {str(e)}")


def parse_h5_file(contents, filename, data_column):
    """Parsear archivo HDF5."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        import h5py
        import tempfile
        import os
        
        # Guardar temporalmente
        temp_file = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_file, 'wb') as f:
            f.write(decoded)
        
        # Leer HDF5
        with h5py.File(temp_file, 'r') as f:
            if data_column in f:
                signal = f[data_column][:]
            else:
                # Intentar con el primer dataset
                keys = list(f.keys())
                if len(keys) > 0:
                    signal = f[keys[0]][:]
                else:
                    raise ValueError("No se encontraron datasets")
        
        return signal, None, f"HDF5 cargado: {len(signal)} muestras"
        
    except ImportError:
        raise ValueError("h5py no está instalado. Use 'pip install h5py'")
    except Exception as e:
        raise ValueError(f"Error al parsear HDF5: {str(e)}")


def parse_mat_file(contents, filename, data_column):
    """Parsear archivo MATLAB."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        from scipy.io import loadmat
        import tempfile
        import os
        
        # Guardar temporalmente
        temp_file = os.path.join(tempfile.gettempdir(), filename)
        with open(temp_file, 'wb') as f:
            f.write(decoded)
        
        # Leer MAT
        mat_data = loadmat(temp_file)
        
        if data_column in mat_data:
            signal = mat_data[data_column].flatten()
        else:
            # Buscar primera variable que no sea metadata
            for key in mat_data.keys():
                if not key.startswith('__'):
                    signal = mat_data[key].flatten()
                    break
            else:
                raise ValueError("No se encontraron variables de datos")
        
        return signal, None, f"MATLAB cargado: {len(signal)} muestras"
        
    except Exception as e:
        raise ValueError(f"Error al parsear MAT: {str(e)}")


def load_signal_file(contents, filename, data_column, time_column=None):
    """
    Cargar archivo de señal en cualquier formato soportado.
    
    Retorna:
    --------
    signal : ndarray
        Señal cargada
    time : ndarray or None
        Vector de tiempo (si está disponible)
    info : str
        Información sobre la carga
    """
    if filename.endswith('.csv'):
        return parse_csv_file(contents, filename, data_column, time_column)
    elif filename.endswith('.h5') or filename.endswith('.hdf5'):
        return parse_h5_file(contents, filename, data_column)
    elif filename.endswith('.mat'):
        return parse_mat_file(contents, filename, data_column)
    else:
        raise ValueError(f"Formato de archivo no soportado: {filename}")


# ============================================================================
# CALLBACKS (se registran en app.py)
# ============================================================================

def register_callbacks(app):
    """Registrar callbacks de la pestaña."""
    
    @app.callback(
        Output('time-column-config', 'style'),
        Input('time-format', 'value')
    )
    def toggle_time_column(time_format):
        """Mostrar/ocultar configuración de columna de tiempo."""
        if time_format == 'column':
            return {'display': 'block', 'marginTop': '10px'}
        return {'display': 'none'}
    
    
    @app.callback(
        [Output('file-upload-status', 'children'),
         Output('analyze-file-signal', 'disabled'),
         Output('file-info', 'children')],
        Input('upload-signal-file', 'contents'),
        State('upload-signal-file', 'filename')
    )
    def handle_file_upload(contents, filename):
        """Manejar carga de archivo."""
        if contents is None:
            return "", True, ""
        
        # Verificar formato
        supported_formats = ['.csv', '.h5', '.hdf5', '.mat']
        if not any(filename.endswith(fmt) for fmt in supported_formats):
            return dbc.Alert(
                "❌ Formato no soportado. Use CSV, HDF5 o MAT",
                color="danger"
            ), True, ""
        
        status = dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            f"✓ Archivo cargado: {filename}"
        ], color="success")
        
        file_info = dbc.ListGroup([
            dbc.ListGroupItem([
                html.Strong("Nombre: "),
                filename
            ]),
            dbc.ListGroupItem([
                html.Strong("Formato: "),
                filename.split('.')[-1].upper()
            ]),
            dbc.ListGroupItem([
                html.Strong("Estado: "),
                html.Span("Listo para analizar", className="text-success")
            ])
        ])
        
        return status, False, file_info
    
    
    @app.callback(
        [Output('file-signal-comparison', 'figure'),
         Output('file-spectrum', 'figure'),
         Output('file-descriptors-table', 'children'),
         Output('file-descriptors-radar', 'figure'),
         Output('file-severity-result', 'children')],
        Input('analyze-file-signal', 'n_clicks'),
        [State('upload-signal-file', 'contents'),
         State('upload-signal-file', 'filename'),
         State('file-sample-rate', 'value'),
         State('file-data-column', 'value'),
         State('time-format', 'value'),
         State('file-time-column', 'value')],
        prevent_initial_call=True
    )
    def analyze_signal_file(n_clicks, contents, filename, fs, 
                           data_column, time_format, time_column):
        """Analizar archivo de señal."""
        
        if contents is None:
            return {}, {}, "", {}, ""
        
        try:
            # Cargar señal
            time_col = time_column if time_format == 'column' else None
            signal, time_vec, load_info = load_signal_file(
                contents, filename, data_column, time_col
            )
            
            # Preprocesar
            lowcut = fs * 0.01
            highcut = fs * 0.4
            processed_signal, proc_info = preprocess_signal(
                signal, fs, lowcut, highcut, True, True, True
            )
            
            # Calcular descriptores
            descriptors = compute_all_descriptors(processed_signal, fs, signal)
            
            # Evaluar severidad
            severity_results = assess_severity(descriptors)
            
            # Crear visualizaciones
            comparison_fig = create_signal_comparison_figure(signal, processed_signal, fs)
            spectrum_fig = create_spectrum_figure(processed_signal, fs)
            descriptors_table = create_file_descriptors_table(descriptors)
            radar_fig = create_descriptors_radar(descriptors)
            severity_display = create_file_severity_display(severity_results, descriptors)
            
            return (comparison_fig, spectrum_fig, descriptors_table, 
                   radar_fig, severity_display)
            
        except Exception as e:
            error_msg = dbc.Alert([
                html.H5("❌ Error en el Análisis"),
                html.Hr(),
                html.P(str(e))
            ], color="danger")
            
            return {}, {}, error_msg, {}, error_msg


def create_signal_comparison_figure(original, processed, fs):
    """Crear figura comparando señal original y procesada."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Señal Original", "Señal Procesada"),
        vertical_spacing=0.12
    )
    
    t_original = np.arange(len(original)) / fs
    t_processed = np.arange(len(processed)) / fs
    
    # Señal original
    fig.add_trace(
        go.Scattergl(x=t_original, y=original, mode='lines',
                  line=dict(color=COLOR_PALETTE['primary'], width=1.5),
                  name='Original'),
        row=1, col=1
    )
    
    # Señal procesada
    fig.add_trace(
        go.Scattergl(x=t_processed, y=processed, mode='lines',
                  line=dict(color=COLOR_PALETTE['success'], width=1.5),
                  name='Procesada'),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitud", row=1, col=1)
    fig.update_yaxes(title_text="Amplitud", row=2, col=1)
    
    fig = apply_professional_style(fig, height=400)
    fig.update_layout(showlegend=False)
    
    return fig


def create_spectrum_figure(signal, fs):
    """Crear figura de espectro de frecuencia."""
    # FFT
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, 1/fs)
    
    # Solo frecuencias positivas
    pos_mask = fft_freq >= 0
    freqs = fft_freq[pos_mask]
    magnitude = np.abs(fft_vals[pos_mask])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattergl(
        x=freqs,
        y=magnitude,
        mode='lines',
        line=dict(color=COLOR_PALETTE['warning'], width=1.5),
        fill='tozeroy',
        fillcolor='rgba(251,191,36,0.18)',
        name='Magnitud',
    ))
    
    fig = apply_professional_style(fig, height=300)
    fig.update_layout(
        xaxis_title="Frecuencia (Hz)",
        yaxis_title="Magnitud",
        hovermode='x',
    )
    
    return fig


def create_file_descriptors_table(descriptors):
    """Crear tabla de descriptores."""
    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Descriptor"),
                html.Th("Valor")
            ])
        ]),
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
            html.Tr([html.Td("Tasa de Cruces por Cero", className="fw-bold"), 
                    html.Td(f"{descriptors['zero_crossing_rate']:.4f}")]),
        ])
    ], bordered=True, hover=True, striped=True, size="sm")


def create_descriptors_radar(descriptors):
    """Crear gráfica de radar de descriptores normalizados."""
    # Normalizar descriptores para visualización
    desc_names = ['Energía', 'RMS', 'Curtosis', 'Asimetría', 
                  'Cresta', 'Entropía', 'Estabilidad']
    
    # Valores normalizados (0-1)
    values = [
        min(descriptors['energy_total'] * 10, 1),
        min(descriptors['rms'] * 10, 1),
        min(descriptors['kurtosis'] / 10, 1),
        min(abs(descriptors['skewness']) / 3, 1),
        min(descriptors['crest_factor'] / 10, 1),
        descriptors['spectral_entropy'],
        descriptors['spectral_stability']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=desc_names,
        fill='toself',
        fillcolor='rgba(118,75,162,0.30)',
        line_color=COLOR_PALETTE['secondary'],
        name='Descriptores',
    ))
    
    fig = apply_professional_style(fig, height=300)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor='rgba(255,255,255,0.08)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
            bgcolor='rgba(0,0,0,0)',
        ),
        showlegend=False,
    )
    
    return fig


def create_file_severity_display(severity_results, descriptors):
    """Crear display de resultado de severidad."""
    state = severity_results['traffic_light_state']
    severity = severity_results['severity_index']
    
    colors = {
        'verde': ('success', '🟢', 'Normal', 'Estado óptimo de operación'),
        'amarillo': ('warning', '🟡', 'Precaución', 'Monitoreo recomendado'),
        'naranja': ('warning', '🟠', 'Alerta', 'Revisión necesaria'),
        'rojo': ('danger', '🔴', 'Crítico', '¡Acción inmediata requerida!')
    }
    
    color, symbol, label, description = colors.get(state, ('secondary', '⚪', 'Desconocido', ''))
    
    return html.Div([
        html.Div([
            html.H1(symbol, style={'fontSize': '100px', 'marginBottom': '10px'}),
            html.H2(label, className=f"text-{color}"),
            html.P(description, className="text-muted"),
            html.Hr(),
            html.H4(f"Índice de Severidad: {severity:.4f}"),
            dbc.Progress(
                value=severity * 100,
                color=color,
                className="mt-3 mb-3",
                style={'height': '25px'},
                label=f"{severity*100:.1f}%"
            ),
        ], style={'textAlign': 'center', 'padding': '20px'}),
        
        dbc.Alert([
            html.H5("Descriptores Principales:", className="mb-3"),
            html.Ul([
                html.Li(f"Energía Total: {descriptors['energy_total']:.6f}"),
                html.Li(f"RMS: {descriptors['rms']:.6f}"),
                html.Li(f"Conteo de Picos: {descriptors['peak_count']}"),
                html.Li(f"Factor de Cresta: {descriptors['crest_factor']:.4f}"),
            ])
        ], color="info")
    ])
