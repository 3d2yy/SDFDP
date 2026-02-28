"""
Aplicación Dash para Sistema de Detección de Descargas Parciales UHF
======================================================================

Interfaz gráfica profesional para análisis en tiempo real y offline de descargas parciales.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime

# Importar módulos de la interfaz
from gui import live_capture, file_analysis, signal_generator, documentation, threshold_config, description

# Inicializar la aplicación Dash con tema oscuro profesional
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,  # Tema oscuro profesional
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="Sistema PD-UHF | Detección Profesional de Descargas Parciales",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

# CSS personalizado para un look más profesional
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }
            body {
                background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                margin: 0;
                padding: 0;
            }
            .card {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .card:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
            }
            .card-header {
                background: rgba(255, 255, 255, 0.03);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px 16px 0 0 !important;
                padding: 1.25rem 1.5rem;
            }
            .nav-tabs .nav-link {
                border: none;
                border-radius: 12px;
                margin-right: 8px;
                padding: 12px 24px;
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.7);
                transition: all 0.3s ease;
                font-weight: 500;
            }
            .nav-tabs .nav-link:hover {
                background: rgba(255, 255, 255, 0.1);
                color: #fff;
            }
            .nav-tabs .nav-link.active {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #fff;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            }
            .btn {
                border-radius: 10px;
                padding: 10px 24px;
                font-weight: 500;
                transition: all 0.3s ease;
                border: none;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            }
            .form-control, .form-select {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                color: #fff;
                transition: all 0.3s ease;
            }
            .form-control:focus, .form-select:focus {
                background: rgba(255, 255, 255, 0.08);
                border-color: #667eea;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
                color: #fff;
            }
            .badge {
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: 500;
            }
            h1, h2, h3, h4, h5, h6 {
                font-weight: 600;
                letter-spacing: -0.02em;
            }
            .navbar {
                background: rgba(0, 0, 0, 0.3) !important;
                backdrop-filter: blur(20px);
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============================================================================
# LAYOUT PRINCIPAL
# ============================================================================

def create_header():
    """Crear encabezado de la aplicación."""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-bolt", 
                                  style={
                                      'fontSize': '32px',
                                      'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                      '-webkit-background-clip': 'text',
                                      '-webkit-text-fill-color': 'transparent',
                                      'marginRight': '16px'
                                  }),
                        ], style={'display': 'inline-block'}),
                        html.Div([
                            html.Div("Sistema PD-UHF", 
                                    style={
                                        'fontSize': '24px',
                                        'fontWeight': '700',
                                        'color': '#fff',
                                        'lineHeight': '1.2',
                                        'letterSpacing': '-0.02em'
                                    }),
                            html.Div("Detección Profesional de Descargas Parciales",
                                    style={
                                        'fontSize': '12px',
                                        'color': 'rgba(255,255,255,0.6)',
                                        'fontWeight': '400'
                                    })
                        ], style={'display': 'inline-block', 'verticalAlign': 'middle'})
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], width=8),
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-circle", 
                                  style={
                                      'fontSize': '8px',
                                      'color': '#00ff88',
                                      'marginRight': '8px',
                                      'animation': 'pulse 2s infinite'
                                  }),
                            html.Span("Sistema Activo", 
                                     id="system-status",
                                     style={'fontWeight': '500', 'color': '#fff'})
                        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end'})
                    ])
                ], width=4)
            ], align="center", className="w-100")
        ], fluid=True, style={'padding': '16px 0'}),
        style={
            'background': 'rgba(0, 0, 0, 0.3)',
            'backdropFilter': 'blur(20px)',
            'borderBottom': '1px solid rgba(255, 255, 255, 0.1)',
            'marginBottom': '32px'
        },
        dark=True
    )


def create_tabs():
    """Crear pestañas de navegación."""
    tab_style = {
        'borderRadius': '12px',
        'padding': '12px 24px',
        'fontWeight': '500',
        'fontSize': '14px',
        'marginRight': '8px',
        'border': 'none',
        'transition': 'all 0.3s ease'
    }
    
    active_tab_style = {
        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'color': '#fff',
        'borderRadius': '12px',
        'padding': '12px 24px',
        'fontWeight': '500',
        'fontSize': '14px',
        'border': 'none',
        'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.4)'
    }
    
    return dbc.Tabs([
        dbc.Tab(
            label="📡 Captura en Vivo",
            tab_id="tab-live",
            tab_style=tab_style,
            active_label_style=active_tab_style
        ),
        dbc.Tab(
            label="📂 Análisis de Archivos",
            tab_id="tab-files",
            tab_style=tab_style,
            active_label_style=active_tab_style
        ),
        dbc.Tab(
            label="⚙️ Generador de Señales",
            tab_id="tab-generator",
            tab_style=tab_style,
            active_label_style=active_tab_style
        ),
        dbc.Tab(
            label="🎯 Configuración de Umbrales",
            tab_id="tab-thresholds",
            tab_style=tab_style,
            active_label_style=active_tab_style
        ),
        dbc.Tab(
            label="📚 Documentación",
            tab_id="tab-docs",
            tab_style=tab_style,
            active_label_style=active_tab_style
        ),
        dbc.Tab(
            label="📖 Descripción Técnica",
            tab_id="tab-description",
            tab_style=tab_style,
            active_label_style=active_tab_style
        ),
    ], id="tabs", active_tab="tab-live", className="mb-4", 
       style={'background': 'transparent'})


# Layout principal
app.layout = dbc.Container([
    # Encabezado
    create_header(),
    
    # Pestañas de navegación
    create_tabs(),
    
    # Contenedor de contenido
    html.Div(id="tab-content", className="mt-3"),
    
    # Intervalo para actualización en tiempo real
    dcc.Interval(
        id='interval-component',
        interval=1000,  # 1 segundo
        n_intervals=0,
        disabled=True
    ),
    
    # Store para datos compartidos
    dcc.Store(id='baseline-profile'),
    dcc.Store(id='threshold-config'),
    dcc.Store(id='live-capture-state'),
    
], fluid=True, style={
    'maxWidth': '1400px',
    'padding': '0 24px 48px 24px'
})


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab')
)
def render_tab_content(active_tab):
    """Renderizar el contenido de la pestaña activa."""
    if active_tab == "tab-live":
        return live_capture.create_layout()
    elif active_tab == "tab-files":
        return file_analysis.create_layout()
    elif active_tab == "tab-generator":
        return signal_generator.create_layout()
    elif active_tab == "tab-thresholds":
        return threshold_config.create_layout()
    elif active_tab == "tab-docs":
        return documentation.create_layout()
    elif active_tab == "tab-description":
        return description.create_description_layout()
    return html.Div("Seleccione una pestaña")


# Registrar callbacks de cada módulo
live_capture.register_callbacks(app)
file_analysis.register_callbacks(app)
signal_generator.register_callbacks(app)
threshold_config.register_callbacks(app)
documentation.register_callbacks(app)
description.register_callbacks(app)


# ============================================================================
# EJECUTAR APLICACIÓN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SISTEMA DE DETECCIÓN DE DESCARGAS PARCIALES UHF")
    print("=" * 70)
    print()
    print("🚀 Iniciando aplicación web...")
    print("📊 Interfaz disponible en: http://localhost:8050")
    print()
    print("✓ Sistema listo para operar")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=8050)
