"""
Pestaña de Generador de Señales
=================================

Generación de señales sintéticas de descargas parciales con configuración avanzada.
Exportación en múltiples formatos: CSV, HDF5, MAT
"""

import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime

# Importar funciones del sistema
import sys
sys.path.append('..')
from main import generate_synthetic_signal


# ============================================================================
# LAYOUT
# ============================================================================

def create_layout():
    """Crear layout de la pestaña de generación de señales."""
    return dbc.Container([
        dbc.Row([
            # Panel de configuración
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.I(className="fas fa-wrench me-2"),
                        "Configuración de Señal"
                    ])),
                    dbc.CardBody([
                        # Estado operativo
                        html.Label("Estado Operativo:", className="fw-bold"),
                        dbc.Select(
                            id="gen-state",
                            options=[
                                {"label": "🟢 Verde - Normal", "value": "verde"},
                                {"label": "🟡 Amarillo - Precaución", "value": "amarillo"},
                                {"label": "🟠 Naranja - Alerta", "value": "naranja"},
                                {"label": "🔴 Rojo - Crítico", "value": "rojo"}
                            ],
                            value="verde",
                            className="mb-3"
                        ),
                        
                        html.Hr(),
                        
                        # Parámetros temporales
                        html.H5("Parámetros Temporales", className="fw-bold"),
                        
                        html.Label("Duración (muestras):", className="mt-2"),
                        dbc.Input(
                            id="gen-duration",
                            type="number",
                            value=1000,
                            min=100,
                            max=100000,
                            step=100
                        ),
                        
                        html.Label("Frecuencia de Muestreo (Hz):", className="mt-2"),
                        dbc.Input(
                            id="gen-sample-rate",
                            type="number",
                            value=10000,
                            min=1000,
                            max=1000000,
                            step=1000
                        ),
                        
                        html.Hr(),
                        
                        # Parámetros de descargas
                        html.H5("Parámetros de Descargas", className="fw-bold"),
                        
                        html.Label("Número de Descargas:", className="mt-2"),
                        dbc.Input(
                            id="gen-n-discharges",
                            type="number",
                            value=5,
                            min=0,
                            max=100
                        ),
                        
                        html.Label("Amplitud de Pulsos:", className="mt-2"),
                        dcc.Slider(
                            id="gen-amplitude",
                            min=0.1,
                            max=10,
                            step=0.1,
                            value=1.0,
                            marks={0.1: '0.1', 5: '5', 10: '10'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        
                        html.Label("Frecuencia de Oscilación (Hz):", className="mt-3"),
                        dbc.Input(
                            id="gen-frequency",
                            type="number",
                            value=1000,
                            min=100,
                            max=5000,
                            step=100
                        ),
                        
                        html.Hr(),
                        
                        # Parámetros de ruido
                        html.H5("Parámetros de Ruido", className="fw-bold"),
                        
                        html.Label("Tipo de Ruido:", className="mt-2"),
                        dbc.Select(
                            id="gen-noise-type",
                            options=[
                                {"label": "Gaussiano Blanco", "value": "gaussian"},
                                {"label": "Rosa (1/f)", "value": "pink"},
                                {"label": "Marrón (1/f²)", "value": "brown"},
                                {"label": "Uniforme", "value": "uniform"}
                            ],
                            value="gaussian"
                        ),
                        
                        html.Label("Nivel de Ruido:", className="mt-2"),
                        dcc.Slider(
                            id="gen-noise-level",
                            min=0,
                            max=1,
                            step=0.01,
                            value=0.1,
                            marks={0: '0', 0.5: '0.5', 1: '1'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        
                        html.Hr(),
                        
                        # Botones de acción
                        dbc.Button(
                            [html.I(className="fas fa-sync me-2"), "Generar Señal"],
                            id="generate-signal-btn",
                            color="primary",
                            className="w-100 mb-2"
                        ),
                        
                        dbc.Button(
                            [html.I(className="fas fa-random me-2"), "Parámetros Aleatorios"],
                            id="randomize-params-btn",
                            color="info",
                            className="w-100"
                        ),
                    ])
                ], className="mb-3"),
                
                # Exportación
                dbc.Card([
                    dbc.CardHeader(html.H4([
                        html.I(className="fas fa-download me-2"),
                        "Exportar Señal"
                    ])),
                    dbc.CardBody([
                        html.Label("Formato de Archivo:", className="fw-bold"),
                        dbc.RadioItems(
                            id="export-format",
                            options=[
                                {"label": "CSV", "value": "csv"},
                                {"label": "HDF5 (.h5)", "value": "h5"},
                                {"label": "MATLAB (.mat)", "value": "mat"}
                            ],
                            value="csv",
                            className="mb-3"
                        ),
                        
                        html.Label("Nombre de Archivo:", className="mt-2"),
                        dbc.Input(
                            id="export-filename",
                            placeholder="mi_señal",
                            value=f"pd_signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        ),
                        
                        html.Label("Incluir Metadatos:", className="mt-3"),
                        dbc.Checklist(
                            id="export-metadata",
                            options=[
                                {"label": "Parámetros de generación", "value": "params"},
                                {"label": "Vector de tiempo", "value": "time"},
                                {"label": "Estadísticas", "value": "stats"}
                            ],
                            value=["params", "time"],
                            className="mb-3"
                        ),
                        
                        html.Hr(),
                        
                        dbc.Button(
                            [html.I(className="fas fa-file-export me-2"), "Exportar"],
                            id="export-signal-btn",
                            color="success",
                            className="w-100",
                            disabled=True
                        ),
                        
                        html.Div(id="export-status", className="mt-3")
                    ])
                ])
            ], width=3),
            
            # Panel de visualización
            dbc.Col([
                # Señal generada
                dbc.Card([
                    dbc.CardHeader(html.H5("Señal Generada")),
                    dbc.CardBody([
                        dcc.Graph(
                            id="generated-signal-plot",
                            config={'displayModeBar': True},
                            style={'height': '350px'}
                        )
                    ])
                ], className="mb-3"),
                
                # Análisis de la señal
                dbc.Card([
                    dbc.CardHeader(html.H5("Análisis de la Señal")),
                    dbc.CardBody([
                        dbc.Row([
                            # Espectro
                            dbc.Col([
                                dcc.Graph(
                                    id="generated-spectrum",
                                    config={'displayModeBar': True},
                                    style={'height': '250px'}
                                )
                            ], width=6),
                            # Histograma
                            dbc.Col([
                                dcc.Graph(
                                    id="generated-histogram",
                                    config={'displayModeBar': True},
                                    style={'height': '250px'}
                                )
                            ], width=6)
                        ])
                    ])
                ], className="mb-3"),
                
                # Estadísticas
                dbc.Card([
                    dbc.CardHeader(html.H5("Estadísticas de la Señal")),
                    dbc.CardBody([
                        html.Div(id="generated-statistics")
                    ])
                ])
            ], width=9)
        ])
    ], fluid=True)


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def generate_custom_signal(state, duration, fs, noise_level, noise_type,
                          n_discharges, amplitude, frequency):
    """
    Generar señal personalizada con parámetros específicos.
    
    Retorna:
    --------
    signal : ndarray
        Señal generada
    metadata : dict
        Metadatos de generación
    """
    t = np.arange(duration) / fs
    
    # Generar ruido base
    if noise_type == 'gaussian':
        signal = noise_level * np.random.randn(duration)
    elif noise_type == 'pink':
        # Ruido rosa (aproximación)
        white = np.random.randn(duration)
        signal = noise_level * np.cumsum(white) / np.sqrt(duration)
    elif noise_type == 'brown':
        # Ruido marrón
        white = np.random.randn(duration)
        signal = noise_level * np.cumsum(np.cumsum(white)) / duration
    else:  # uniform
        signal = noise_level * (np.random.rand(duration) - 0.5) * 2
    
    # Añadir pulsos de descarga parcial
    discharge_positions = []
    for _ in range(n_discharges):
        # Posición aleatoria
        pos = np.random.randint(100, duration - 100)
        discharge_positions.append(pos)
        
        # Duración del pulso
        pulse_duration = int(0.01 * fs)
        
        # Generar pulso oscilatorio amortiguado
        t_pulse = np.arange(pulse_duration) / fs
        decay = np.exp(-t_pulse * 500)
        pulse = amplitude * np.sin(2 * np.pi * frequency * t_pulse) * decay
        
        # Añadir a la señal
        end_pos = min(pos + pulse_duration, duration)
        signal[pos:end_pos] += pulse[:end_pos-pos]
    
    # Metadatos
    metadata = {
        'state': state,
        'duration': duration,
        'sample_rate': fs,
        'noise_level': noise_level,
        'noise_type': noise_type,
        'n_discharges': n_discharges,
        'amplitude': amplitude,
        'frequency': frequency,
        'discharge_positions': discharge_positions,
        'generation_time': datetime.now().isoformat()
    }
    
    return signal, metadata


def export_signal(signal, metadata, filename, format_type, include_metadata):
    """
    Exportar señal a archivo.
    
    Parámetros:
    -----------
    signal : ndarray
        Señal a exportar
    metadata : dict
        Metadatos
    filename : str
        Nombre del archivo (sin extensión)
    format_type : str
        Formato: 'csv', 'h5', 'mat'
    include_metadata : list
        Lista de metadatos a incluir
    
    Retorna:
    --------
    filepath : str
        Ruta del archivo generado
    """
    # Preparar datos
    data = {'signal': signal}
    
    if 'time' in include_metadata:
        fs = metadata['sample_rate']
        data['time'] = np.arange(len(signal)) / fs
    
    # Exportar según formato
    if format_type == 'csv':
        filepath = f"/tmp/{filename}.csv"
        df = pd.DataFrame(data)
        
        # Agregar metadatos como comentarios
        if 'params' in include_metadata:
            with open(filepath, 'w') as f:
                f.write(f"# Señal de Descarga Parcial Sintética\n")
                f.write(f"# Generado: {metadata['generation_time']}\n")
                f.write(f"# Estado: {metadata['state']}\n")
                f.write(f"# Muestras: {metadata['duration']}\n")
                f.write(f"# Frecuencia Muestreo: {metadata['sample_rate']} Hz\n")
                f.write(f"# Descargas: {metadata['n_discharges']}\n")
                f.write(f"# Amplitud: {metadata['amplitude']}\n")
                f.write(f"# Frecuencia Oscilación: {metadata['frequency']} Hz\n")
                f.write(f"# Tipo Ruido: {metadata['noise_type']}\n")
                f.write(f"# Nivel Ruido: {metadata['noise_level']}\n")
                f.write("\n")
            
            df.to_csv(filepath, mode='a', index=False)
        else:
            df.to_csv(filepath, index=False)
    
    elif format_type == 'h5':
        import h5py
        filepath = f"/tmp/{filename}.h5"
        
        with h5py.File(filepath, 'w') as f:
            # Guardar señal
            f.create_dataset('signal', data=signal)
            
            if 'time' in include_metadata:
                f.create_dataset('time', data=data['time'])
            
            # Guardar metadatos
            if 'params' in include_metadata:
                meta_group = f.create_group('metadata')
                for key, value in metadata.items():
                    if key != 'discharge_positions':
                        meta_group.attrs[key] = value
            
            # Guardar estadísticas
            if 'stats' in include_metadata:
                stats_group = f.create_group('statistics')
                stats_group.attrs['mean'] = np.mean(signal)
                stats_group.attrs['std'] = np.std(signal)
                stats_group.attrs['min'] = np.min(signal)
                stats_group.attrs['max'] = np.max(signal)
                stats_group.attrs['rms'] = np.sqrt(np.mean(signal**2))
    
    else:  # mat
        from scipy.io import savemat
        filepath = f"/tmp/{filename}.mat"
        
        mat_data = {'signal': signal}
        
        if 'time' in include_metadata:
            mat_data['time'] = data['time']
        
        if 'params' in include_metadata:
            mat_data['metadata'] = metadata
        
        if 'stats' in include_metadata:
            mat_data['statistics'] = {
                'mean': np.mean(signal),
                'std': np.std(signal),
                'min': np.min(signal),
                'max': np.max(signal),
                'rms': np.sqrt(np.mean(signal**2))
            }
        
        savemat(filepath, mat_data)
    
    return filepath


# ============================================================================
# CALLBACKS (se registran en app.py)
# ============================================================================

def register_callbacks(app):
    """Registrar callbacks de la pestaña."""
    
    @app.callback(
        [Output('gen-n-discharges', 'value'),
         Output('gen-amplitude', 'value'),
         Output('gen-frequency', 'value'),
         Output('gen-noise-level', 'value')],
        Input('randomize-params-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def randomize_parameters(n_clicks):
        """Generar parámetros aleatorios."""
        n_discharges = np.random.randint(1, 50)
        amplitude = np.random.uniform(0.5, 8.0)
        frequency = np.random.randint(500, 3000)
        noise_level = np.random.uniform(0.05, 0.5)
        
        return n_discharges, amplitude, frequency, noise_level
    
    
    @app.callback(
        [Output('generated-signal-plot', 'figure'),
         Output('generated-spectrum', 'figure'),
         Output('generated-histogram', 'figure'),
         Output('generated-statistics', 'children'),
         Output('export-signal-btn', 'disabled')],
        Input('generate-signal-btn', 'n_clicks'),
        [State('gen-state', 'value'),
         State('gen-duration', 'value'),
         State('gen-sample-rate', 'value'),
         State('gen-noise-level', 'value'),
         State('gen-noise-type', 'value'),
         State('gen-n-discharges', 'value'),
         State('gen-amplitude', 'value'),
         State('gen-frequency', 'value')],
        prevent_initial_call=True
    )
    def generate_and_display_signal(n_clicks, state, duration, fs, noise_level,
                                   noise_type, n_discharges, amplitude, frequency):
        """Generar y mostrar señal."""
        
        # Generar señal
        signal, metadata = generate_custom_signal(
            state, duration, fs, noise_level, noise_type,
            n_discharges, amplitude, frequency
        )
        
        # Almacenar en memoria para exportación
        app.generated_signal = signal
        app.generated_metadata = metadata
        
        # Crear visualizaciones
        signal_fig = create_generated_signal_figure(signal, fs)
        spectrum_fig = create_generated_spectrum_figure(signal, fs)
        histogram_fig = create_generated_histogram_figure(signal)
        statistics_table = create_generated_statistics_table(signal)
        
        return signal_fig, spectrum_fig, histogram_fig, statistics_table, False
    
    
    @app.callback(
        Output('export-status', 'children'),
        Input('export-signal-btn', 'n_clicks'),
        [State('export-format', 'value'),
         State('export-filename', 'value'),
         State('export-metadata', 'value')],
        prevent_initial_call=True
    )
    def export_generated_signal(n_clicks, format_type, filename, metadata_options):
        """Exportar señal generada."""
        
        try:
            signal = app.generated_signal
            metadata = app.generated_metadata
            
            filepath = export_signal(
                signal, metadata, filename, 
                format_type, metadata_options
            )
            
            return dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"✓ Archivo exportado: {filepath}"
            ], color="success")
            
        except Exception as e:
            return dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                f"Error al exportar: {str(e)}"
            ], color="danger")


def create_generated_signal_figure(signal, fs):
    """Crear figura de señal generada."""
    t = np.arange(len(signal)) / fs
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=t * 1000,  # Convertir a ms
        y=signal,
        mode='lines',
        line=dict(color='#1f77b4', width=1),
        name='Señal'
    ))
    
    fig.update_layout(
        xaxis_title="Tiempo (ms)",
        yaxis_title="Amplitud",
        template="plotly_white",
        margin=dict(l=50, r=20, t=20, b=40),
        hovermode='x'
    )
    
    return fig


def create_generated_spectrum_figure(signal, fs):
    """Crear figura de espectro."""
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, 1/fs)
    
    pos_mask = fft_freq >= 0
    freqs = fft_freq[pos_mask]
    magnitude = np.abs(fft_vals[pos_mask])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=freqs,
        y=magnitude,
        mode='lines',
        line=dict(color='#ff7f0e', width=1),
        fill='tozeroy',
        name='Espectro'
    ))
    
    fig.update_layout(
        title="Espectro de Frecuencia",
        xaxis_title="Frecuencia (Hz)",
        yaxis_title="Magnitud",
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode='x'
    )
    
    return fig


def create_generated_histogram_figure(signal):
    """Crear histograma de amplitudes."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=signal,
        nbinsx=50,
        marker_color='#2ca02c',
        name='Distribución'
    ))
    
    fig.update_layout(
        title="Distribución de Amplitudes",
        xaxis_title="Amplitud",
        yaxis_title="Frecuencia",
        template="plotly_white",
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=False
    )
    
    return fig


def create_generated_statistics_table(signal):
    """Crear tabla de estadísticas."""
    stats = {
        'Media': np.mean(signal),
        'Desviación Estándar': np.std(signal),
        'Mínimo': np.min(signal),
        'Máximo': np.max(signal),
        'RMS': np.sqrt(np.mean(signal**2)),
        'Energía': np.sum(signal**2),
        'Curtosis': float(pd.Series(signal).kurtosis()),
        'Asimetría': float(pd.Series(signal).skew())
    }
    
    return dbc.Table([
        html.Tbody([
            html.Tr([
                html.Td(name, className="fw-bold"),
                html.Td(f"{value:.6f}")
            ]) for name, value in stats.items()
        ])
    ], bordered=True, hover=True, striped=True)
