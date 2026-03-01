"""
Aplicación Dash para Sistema de Detección de Descargas Parciales UHF
======================================================================

Interfaz SPA profesional para análisis en tiempo real y offline de
descargas parciales.  Preparada para despliegue en Google AI Studio con
NI PXIe-5185 (12.5 GS/s, 3 GHz BW, 8-bit).

WCAG AAA compliant:
  - Contrast ratio ≥ 7:1 for normal text
  - Focus-visible outlines on all interactive elements
  - Skip-to-content link
  - aria-labels on all controls
  - Keyboard-navigable tabs
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from datetime import datetime

# Importar módulos de la interfaz
from gui import (
    live_capture, file_analysis, signal_generator,
    documentation, threshold_config, description,
    time_series,
)
from gui.i18n import t, TRANSLATIONS
from gui.pdf_export import generate_pdf_report, pdf_bytes_to_data_uri

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="Sistema PD-UHF | Detección Profesional de Descargas Parciales",
    update_title="Procesando…",
    meta_tags=[
        {"name": "viewport",
         "content": "width=device-width, initial-scale=1, shrink-to-fit=no"},
        {"name": "description",
         "content": "Sistema profesional de detección de descargas parciales "
                    "UHF con NI PXIe-5185.  Análisis Phase-1→4, semáforo de "
                    "severidad, tracking Kalman/EWMA/CUSUM."},
        {"name": "theme-color", "content": "#0f0c29"},
        {"property": "og:title",
         "content": "Sistema PD-UHF | Detección Profesional"},
        {"property": "og:type", "content": "website"},
    ],
)

server = app.server  # WSGI entry-point for production (gunicorn / render / GCP)

# ============================================================================
# CUSTOM INDEX  –  glassmorphism CSS + keyframe animations + responsive
# ============================================================================

app.index_string = '''
<!DOCTYPE html>
<html lang="es">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* ── Base typography ──────────────────────────────────────── */
            * {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                             'Segoe UI', sans-serif;
                box-sizing: border-box;
            }
            body {
                background: linear-gradient(135deg,
                    #0f0c29 0%, #302b63 50%, #24243e 100%);
                background-attachment: fixed;
                margin: 0; padding: 0;
                min-height: 100vh;
                color: #f0f0f4;   /* WCAG AAA ≥ 7:1 on #302b63 */
            }
            /* ── WCAG AAA: Skip-to-content link ──────────────────────── */
            .skip-link {
                position: absolute;
                top: -100px;
                left: 50%;
                transform: translateX(-50%);
                background: #667eea;
                color: #fff;
                padding: 10px 24px;
                border-radius: 0 0 8px 8px;
                z-index: 10000;
                font-weight: 600;
                text-decoration: none;
                transition: top 200ms ease;
            }
            .skip-link:focus {
                top: 0;
                outline: 3px solid #ffd700;
                outline-offset: 2px;
            }
            /* ── WCAG AAA: Focus-visible outlines ────────────────────── */
            :focus-visible {
                outline: 3px solid #ffd700 !important;
                outline-offset: 2px !important;
            }
            a:focus-visible, button:focus-visible,
            [role="tab"]:focus-visible,
            input:focus-visible, select:focus-visible {
                outline: 3px solid #ffd700 !important;
                outline-offset: 2px !important;
                box-shadow: 0 0 0 4px rgba(255,215,0,0.3) !important;
            }
            /* ── Scrollbar ───────────────────────────────────────────── */
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: rgba(0,0,0,0.15); }
            ::-webkit-scrollbar-thumb {
                background: rgba(102,126,234,0.35);
                border-radius: 4px;
            }
            ::-webkit-scrollbar-thumb:hover {
                background: rgba(102,126,234,0.6);
            }
            /* ── Keyframes ───────────────────────────────────────────── */
            @keyframes pulse {
                0%   { opacity: 1; transform: scale(1); }
                50%  { opacity: 0.55; transform: scale(1.15); }
                100% { opacity: 1; transform: scale(1); }
            }
            @keyframes glow {
                0%   { box-shadow: 0 0 4px rgba(102,126,234,0.3); }
                50%  { box-shadow: 0 0 18px rgba(102,126,234,0.55); }
                100% { box-shadow: 0 0 4px rgba(102,126,234,0.3); }
            }
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(12px); }
                to   { opacity: 1; transform: translateY(0); }
            }
            @keyframes shimmer {
                0%   { background-position: -200% 0; }
                100% { background-position: 200% 0; }
            }
            /* ── Glassmorphism cards ─────────────────────────────────── */
            .card {
                background: rgba(255,255,255,0.05);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.37);
                transition: transform 350ms cubic-bezier(.4,0,.2,1),
                            box-shadow 350ms cubic-bezier(.4,0,.2,1);
                animation: fadeInUp 0.4s ease-out both;
            }
            .card:hover {
                transform: translateY(-3px);
                box-shadow: 0 14px 44px rgba(0,0,0,0.5);
            }
            .card-header {
                background: rgba(255,255,255,0.03);
                border-bottom: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px 16px 0 0 !important;
                padding: 1rem 1.25rem;
            }
            /* ── Nav tabs ────────────────────────────────────────────── */
            .nav-tabs {
                border-bottom: none;
                gap: 6px;
                flex-wrap: wrap;
            }
            .nav-tabs .nav-link {
                border: none;
                border-radius: 12px;
                padding: 10px 20px;
                background: rgba(255,255,255,0.05);
                color: #f0f0f4;   /* WCAG AAA ≥ 7:1 */
                transition: all 300ms cubic-bezier(.4,0,.2,1);
                font-weight: 500;
                font-size: 13px;
                white-space: nowrap;
            }
            .nav-tabs .nav-link:hover {
                background: rgba(255,255,255,0.12);
                color: #fff;
            }
            .nav-tabs .nav-link.active {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: #fff;
                box-shadow: 0 4px 15px rgba(102,126,234,0.4);
                animation: glow 3s ease-in-out infinite;
            }
            /* ── Buttons ─────────────────────────────────────────────── */
            .btn {
                border-radius: 10px;
                padding: 10px 24px;
                font-weight: 500;
                transition: all 300ms cubic-bezier(.4,0,.2,1);
                border: none;
                letter-spacing: 0.01em;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.3);
            }
            .btn:active { transform: translateY(0); }
            /* ── Forms ───────────────────────────────────────────────── */
            .form-control, .form-select {
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 8px;
                color: #f0f0f4;
                transition: all 300ms ease;
            }
            .form-control:focus, .form-select:focus {
                background: rgba(255,255,255,0.08);
                border-color: #667eea;
                box-shadow: 0 0 0 0.2rem rgba(102,126,234,0.25);
                color: #fff;
            }
            .form-control::placeholder { color: rgba(255,255,255,0.4); }
            /* ── Misc typography  (WCAG AAA high-contrast) ───────────── */
            .badge { padding: 6px 12px; border-radius: 6px; font-weight: 500; }
            h1,h2,h3,h4,h5,h6 {
                font-weight: 600;
                letter-spacing: -0.02em;
                color: #ffffff;
            }
            .text-muted { color: rgba(255,255,255,0.6) !important; }
            label { color: #f0f0f4; font-weight: 500; }
            /* ── Navbar ──────────────────────────────────────────────── */
            .navbar {
                background: rgba(0,0,0,0.3) !important;
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                box-shadow: 0 4px 30px rgba(0,0,0,0.15);
            }
            /* ── Upload area (dark-friendly) ─────────────────────────── */
            .dash-upload {
                border: 2px dashed rgba(102,126,234,0.35) !important;
                background: rgba(255,255,255,0.03) !important;
                border-radius: 12px !important;
                transition: all 300ms ease;
            }
            .dash-upload:hover {
                border-color: rgba(102,126,234,0.7) !important;
                background: rgba(102,126,234,0.06) !important;
            }
            /* ── Table dark overrides ────────────────────────────────── */
            .table { color: #f0f0f4; }
            .table > :not(caption) > * > * {
                background-color: transparent;
                border-bottom-color: rgba(255,255,255,0.06);
            }
            .table-hover > tbody > tr:hover > * {
                background-color: rgba(255,255,255,0.04);
            }
            /* ── Slider handle ───────────────────────────────────────── */
            .rc-slider-handle {
                border-color: #667eea !important;
                background-color: #667eea !important;
                box-shadow: 0 0 6px rgba(102,126,234,0.5);
            }
            .rc-slider-track { background-color: #667eea !important; }
            /* ── Language toggle button ───────────────────────────────── */
            .lang-btn {
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.15);
                color: #f0f0f4;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 200ms ease;
            }
            .lang-btn:hover {
                background: rgba(255,255,255,0.15);
                color: #fff;
            }
            /* ── PDF export button ───────────────────────────────────── */
            .pdf-btn {
                background: linear-gradient(135deg, #e74c3c, #c0392b);
                border: none;
                color: #fff;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                transition: all 200ms ease;
            }
            .pdf-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(231,76,60,0.4);
            }
            /* ── Responsive ──────────────────────────────────────────── */
            @media (max-width: 992px) {
                .nav-tabs .nav-link { padding: 8px 14px; font-size: 12px; }
            }
            @media (max-width: 576px) {
                .nav-tabs .nav-link { padding: 6px 10px; font-size: 11px; }
                .card { border-radius: 12px; }
            }
            /* WCAG AAA: ensure reduced-motion preference is respected */
            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                }
            }
        </style>
    </head>
    <body>
        <a class="skip-link" href="#main-content">Skip to main content</a>
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
    """Crear encabezado con toggle de idioma y botón PDF."""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-bolt",
                                   style={
                                       'fontSize': '28px',
                                       'background': 'linear-gradient(135deg, #667eea, #764ba2)',
                                       '-webkit-background-clip': 'text',
                                       '-webkit-text-fill-color': 'transparent',
                                       'marginRight': '14px',
                                   }),
                        ], style={'display': 'inline-block'}),
                        html.Div([
                            html.Div(id="app-title",
                                     children="Sistema PD-UHF",
                                     style={
                                         'fontSize': '22px',
                                         'fontWeight': '700',
                                         'color': '#fff',
                                         'lineHeight': '1.2',
                                         'letterSpacing': '-0.02em',
                                     }),
                            html.Div(id="app-subtitle",
                                     children="Detección Profesional — NI PXIe-5185 · 12.5 GS/s",
                                     style={
                                         'fontSize': '11px',
                                         'color': 'rgba(255,255,255,0.6)',
                                         'fontWeight': '400',
                                     }),
                        ], style={'display': 'inline-block',
                                  'verticalAlign': 'middle'}),
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], md=6, xs=12),
                dbc.Col([
                    html.Div([
                        # PDF export button
                        html.Button(
                            [html.I(className="fas fa-file-pdf",
                                    style={'marginRight': '6px'}),
                             "PDF"],
                            id="btn-pdf-export",
                            className="pdf-btn",
                            n_clicks=0,
                            **{"aria-label": "Export analysis report as PDF"},
                            style={'marginRight': '10px'},
                        ),
                        # Language toggle
                        html.Button(
                            id="btn-lang-toggle",
                            children="EN",
                            className="lang-btn",
                            n_clicks=0,
                            **{"aria-label": "Switch language"},
                        ),
                        # Status indicator
                        html.Div([
                            html.I(className="fas fa-circle",
                                   style={
                                       'fontSize': '8px',
                                       'color': '#34d399',
                                       'marginRight': '8px',
                                       'animation': 'pulse 2s infinite',
                                   }),
                            html.Span(id="system-status",
                                      children="Sistema Activo",
                                      style={'fontWeight': '500',
                                             'color': '#fff',
                                             'fontSize': '13px'}),
                        ], style={'display': 'flex',
                                  'alignItems': 'center',
                                  'marginLeft': '14px'}),
                    ], style={'display': 'flex',
                              'alignItems': 'center',
                              'justifyContent': 'flex-end'}),
                ], md=6, className="d-none d-md-block"),
            ], align="center", className="w-100"),
        ], fluid=True, style={'padding': '14px 0'}),
        style={
            'background': 'rgba(0,0,0,0.3)',
            'backdropFilter': 'blur(20px)',
            'WebkitBackdropFilter': 'blur(20px)',
            'borderBottom': '1px solid rgba(255,255,255,0.08)',
            'marginBottom': '24px',
        },
        dark=True,
    )


def create_tabs():
    """Crear pestañas de navegación."""
    tab_style = {
        'borderRadius': '12px',
        'padding': '10px 20px',
        'fontWeight': '500',
        'fontSize': '13px',
        'border': 'none',
        'transition': 'all 300ms cubic-bezier(.4,0,.2,1)',
    }

    active_tab_style = {
        'background': 'linear-gradient(135deg, #667eea, #764ba2)',
        'color': '#fff',
        'borderRadius': '12px',
        'padding': '10px 20px',
        'fontWeight': '600',
        'fontSize': '13px',
        'border': 'none',
        'boxShadow': '0 4px 15px rgba(102,126,234,0.4)',
    }

    tabs = [
        ("📡 Captura en Vivo",           "tab-live"),
        ("📂 Análisis de Archivos",      "tab-files"),
        ("📈 Time-Series & Δt",          "tab-timeseries"),
        ("⚙️ Generador de Señales",      "tab-generator"),
        ("🎯 Configuración de Umbrales", "tab-thresholds"),
        ("📚 Documentación",             "tab-docs"),
        ("📖 Descripción Técnica",       "tab-description"),
    ]

    return dbc.Tabs(
        [dbc.Tab(label=label, tab_id=tid,
                 tab_style=tab_style,
                 active_label_style=active_tab_style)
         for label, tid in tabs],
        id="tabs",
        active_tab="tab-live",
        className="mb-4",
        style={'background': 'transparent'},
    )


# Layout principal
app.layout = dbc.Container([
    # Language store (persisted in session)
    dcc.Store(id='lang-store', data='es', storage_type='session'),

    # PDF download component
    dcc.Download(id='download-pdf'),

    create_header(),
    create_tabs(),
    html.Div(id="tab-content", className="mt-3",
             style={'animation': 'fadeInUp 0.35s ease-out'},
             role="main"),

    # Intervalo para actualización en tiempo real
    dcc.Interval(id='interval-component',
                 interval=1000, n_intervals=0, disabled=True),

    # Stores compartidos
    dcc.Store(id='baseline-profile'),
    dcc.Store(id='threshold-config'),
    dcc.Store(id='live-capture-state'),
    dcc.Store(id='analysis-results-store'),
], fluid=True, id="main-content", style={
    'maxWidth': '1440px',
    'padding': '0 20px 48px 20px',
})


# ============================================================================
# CALLBACKS
# ============================================================================

# ── Language toggle ──────────────────────────────────────────────────────────

@app.callback(
    Output('lang-store', 'data'),
    Output('btn-lang-toggle', 'children'),
    Output('app-title', 'children'),
    Output('app-subtitle', 'children'),
    Output('system-status', 'children'),
    Input('btn-lang-toggle', 'n_clicks'),
    State('lang-store', 'data'),
    prevent_initial_call=True,
)
def toggle_language(n_clicks, current_lang):
    """Flip between 'es' and 'en'."""
    new_lang = 'en' if current_lang == 'es' else 'es'
    btn_label = 'ES' if new_lang == 'en' else 'EN'
    return (
        new_lang,
        btn_label,
        t('app.title', new_lang),
        t('app.subtitle', new_lang),
        t('app.status_active', new_lang),
    )


# ── PDF export ───────────────────────────────────────────────────────────────

@app.callback(
    Output('download-pdf', 'data'),
    Input('btn-pdf-export', 'n_clicks'),
    State('analysis-results-store', 'data'),
    State('lang-store', 'data'),
    prevent_initial_call=True,
)
def export_pdf(n_clicks, results_json, lang):
    """Generate and download a PDF report from the current analysis state."""
    if not n_clicks:
        return no_update

    results = results_json if isinstance(results_json, dict) else {}
    pdf_bytes = generate_pdf_report(results, lang=lang or 'es')

    filename = f"pd_uhf_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return dcc.send_bytes(pdf_bytes, filename)


# ── Tab routing ──────────────────────────────────────────────────────────────

@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab'),
)
def render_tab_content(active_tab):
    """Renderizar el contenido de la pestaña activa."""
    router = {
        "tab-live":        live_capture.create_layout,
        "tab-files":       file_analysis.create_layout,
        "tab-timeseries":  time_series.create_layout,
        "tab-generator":   signal_generator.create_layout,
        "tab-thresholds":  threshold_config.create_layout,
        "tab-docs":        documentation.create_layout,
        "tab-description": description.create_description_layout,
    }
    factory = router.get(active_tab)
    if factory:
        return factory()
    return html.Div("Seleccione una pestaña")


# Registrar callbacks de cada módulo
live_capture.register_callbacks(app)
file_analysis.register_callbacks(app)
signal_generator.register_callbacks(app)
threshold_config.register_callbacks(app)
documentation.register_callbacks(app)
description.register_callbacks(app)
time_series.register_callbacks(app)


# ============================================================================
# EJECUTAR APLICACIÓN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("SISTEMA DE DETECCIÓN DE DESCARGAS PARCIALES UHF")
    print("NI PXIe-5185 · 12.5 GS/s · 3 GHz BW · 8-bit")
    print("=" * 70)
    print()
    print("  Iniciando aplicación web…")
    print("  Interfaz disponible en: http://localhost:8050")
    print()
    print("  Sistema listo para operar")
    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=8050)
