"""
Utilidades de visualización para el sistema PD-UHF
===================================================

Professional dark-theme Plotly templates, colour palette, and reusable
chart helpers.  WebGL-accelerated traces for real-time & large-data use.
"""

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import html


# ============================================================================
# PROFESSIONAL DARK TEMPLATE
# ============================================================================

PROFESSIONAL_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(255,255,255,0.02)',
        'font': {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
            'color': 'rgba(255,255,255,0.87)',
            'size': 12,
        },
        'title': {
            'font': {
                'size': 15,
                'color': 'rgba(255,255,255,0.95)',
                'family': 'Inter, sans-serif',
            },
            'x': 0.02,
            'xanchor': 'left',
            'pad': {'b': 8},
        },
        'xaxis': {
            'gridcolor': 'rgba(255,255,255,0.06)',
            'zerolinecolor': 'rgba(255,255,255,0.12)',
            'color': 'rgba(255,255,255,0.6)',
            'showline': True,
            'linecolor': 'rgba(255,255,255,0.12)',
            'linewidth': 1,
            'tickfont': {'size': 11},
            'title': {'font': {'size': 12, 'color': 'rgba(255,255,255,0.6)'}},
        },
        'yaxis': {
            'gridcolor': 'rgba(255,255,255,0.06)',
            'zerolinecolor': 'rgba(255,255,255,0.12)',
            'color': 'rgba(255,255,255,0.6)',
            'showline': True,
            'linecolor': 'rgba(255,255,255,0.12)',
            'linewidth': 1,
            'tickfont': {'size': 11},
            'title': {'font': {'size': 12, 'color': 'rgba(255,255,255,0.6)'}},
        },
        'legend': {
            'bgcolor': 'rgba(15,12,41,0.75)',
            'bordercolor': 'rgba(255,255,255,0.08)',
            'borderwidth': 1,
            'font': {'color': 'rgba(255,255,255,0.8)', 'size': 11},
        },
        'hoverlabel': {
            'bgcolor': 'rgba(15,12,41,0.95)',
            'bordercolor': 'rgba(102,126,234,0.5)',
            'font': {'color': '#fff', 'family': 'Inter, sans-serif', 'size': 12},
        },
    }
}


# ============================================================================
# COLOUR PALETTE — Enriched
# ============================================================================

COLOR_PALETTE = {
    # Core
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#a78bfa',
    'surface': 'rgba(255,255,255,0.05)',
    'surface_hover': 'rgba(255,255,255,0.08)',
    'border': 'rgba(255,255,255,0.10)',
    'border_focus': 'rgba(102,126,234,0.5)',

    # Semantic
    'success': '#34d399',
    'warning': '#fbbf24',
    'danger': '#f43f5e',
    'info': '#38bdf8',

    # Gradients
    'gradient_1': ['#667eea', '#764ba2'],
    'gradient_2': ['#f093fb', '#f5576c'],
    'gradient_3': ['#4facfe', '#00f2fe'],
    'gradient_4': ['#43e97b', '#38f9d7'],
    'gradient_5': ['#fa709a', '#fee140'],

    # Traffic-light states
    'state_verde':    '#34d399',
    'state_amarillo': '#fbbf24',
    'state_naranja':  '#fb923c',
    'state_rojo':     '#f43f5e',

    # Sequential palette for multi-trace plots
    'seq': [
        '#667eea', '#38bdf8', '#34d399', '#fbbf24',
        '#fb923c', '#f43f5e', '#a78bfa', '#f472b6',
    ],
}


def apply_professional_style(fig, height=None):
    """Apply the professional dark template to any Plotly figure."""
    fig.update_layout(PROFESSIONAL_TEMPLATE['layout'])
    fig.update_layout(
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(15,12,41,0.95)',
            bordercolor='rgba(102,126,234,0.5)',
            font_size=12,
            font_family='Inter, sans-serif',
        ),
        margin=dict(l=56, r=24, t=52, b=44),
        transition={'duration': 350, 'easing': 'cubic-in-out'},
    )
    if height:
        fig.update_layout(height=height)
    return fig


def create_gradient_line(x, y, name='', gradient_colors=None, fill=True):
    """Create a smooth Scatter trace with optional fill-to-zero."""
    if gradient_colors is None:
        gradient_colors = COLOR_PALETTE['gradient_1']
    c = gradient_colors[0]
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    kwargs = dict(
        x=x, y=y,
        mode='lines',
        name=name,
        line=dict(color=c, width=2, shape='spline'),
        hovertemplate='<b>%{y:.4f}</b><extra></extra>',
    )
    if fill:
        kwargs['fill'] = 'tozeroy'
        kwargs['fillcolor'] = f'rgba({r},{g},{b},0.12)'
    return go.Scatter(**kwargs)


def create_professional_card_style(glow_color=None):
    """Return a glassmorphism card style dict."""
    style = {
        'background': 'rgba(255, 255, 255, 0.04)',
        'backdropFilter': 'blur(16px)',
        'WebkitBackdropFilter': 'blur(16px)',
        'border': '1px solid rgba(255, 255, 255, 0.08)',
        'borderRadius': '16px',
        'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.25)',
        'padding': '24px',
        'marginBottom': '24px',
        'transition': 'transform 0.25s ease, box-shadow 0.25s ease',
    }
    if glow_color:
        style['boxShadow'] = f'0 8px 32px rgba(0,0,0,0.25), 0 0 24px {glow_color}22'
    return style


def kpi_card_style(accent_color=None):
    """Compact KPI card used in dashboards."""
    base = create_professional_card_style(accent_color)
    base.update({
        'padding': '20px',
        'minHeight': '120px',
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
    })
    return base


def create_metric_card(title, value, subtitle='', icon='fas fa-chart-line',
                       color='primary'):
    """Render a KPI metric card as a Dash html.Div."""
    color_map = {
        'primary':  'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'success':  'linear-gradient(135deg, #34d399 0%, #059669 100%)',
        'warning':  'linear-gradient(135deg, #fbbf24 0%, #d97706 100%)',
        'danger':   'linear-gradient(135deg, #f43f5e 0%, #dc2626 100%)',
        'info':     'linear-gradient(135deg, #38bdf8 0%, #0284c7 100%)',
    }
    return html.Div([
        html.Div([
            html.I(className=icon, style={
                'fontSize': '28px',
                'background': color_map.get(color, color_map['primary']),
                '-webkit-background-clip': 'text',
                'WebkitBackgroundClip': 'text',
                '-webkit-text-fill-color': 'transparent',
                'WebkitTextFillColor': 'transparent',
                'marginBottom': '10px',
            }),
            html.P(title, style={
                'fontSize': '11px',
                'color': 'rgba(255,255,255,0.5)',
                'fontWeight': '600',
                'marginBottom': '6px',
                'textTransform': 'uppercase',
                'letterSpacing': '0.08em',
            }),
            html.H3(value, style={
                'fontSize': '28px',
                'fontWeight': '700',
                'color': '#fff',
                'marginBottom': '4px',
                'letterSpacing': '-0.02em',
                'lineHeight': '1.1',
            }),
            html.P(subtitle, style={
                'fontSize': '11px',
                'color': 'rgba(255,255,255,0.4)',
                'marginBottom': '0',
            }) if subtitle else None,
        ], style=kpi_card_style()),
    ])


def create_animated_gauge(value, title='', max_value=100, color_ranges=None):
    """Create a professional indicator gauge with coloured severity arcs."""
    if color_ranges is None:
        color_ranges = [
            (0,    0.25, COLOR_PALETTE['state_verde']),
            (0.25, 0.50, COLOR_PALETTE['state_amarillo']),
            (0.50, 0.75, COLOR_PALETTE['state_naranja']),
            (0.75, 1.00, COLOR_PALETTE['state_rojo']),
        ]
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=value,
        title={'text': title, 'font': {'size': 15, 'color': 'rgba(255,255,255,0.8)'}},
        number={'font': {'size': 36, 'color': '#fff'}, 'suffix': '',
                'valueformat': '.2f'},
        gauge={
            'axis': {
                'range': [0, max_value],
                'tickcolor': 'rgba(255,255,255,0.4)',
                'tickfont': {'size': 10, 'color': 'rgba(255,255,255,0.5)'},
            },
            'bar': {'color': COLOR_PALETTE['primary'], 'thickness': 0.25},
            'bgcolor': 'rgba(255,255,255,0.03)',
            'borderwidth': 0,
            'steps': [
                {
                    'range': [r[0] * max_value, r[1] * max_value],
                    'color': f'{r[2]}33',
                }
                for r in color_ranges
            ],
            'threshold': {
                'line': {'color': '#fff', 'width': 3},
                'thickness': 0.8,
                'value': value,
            },
        },
    ))
    fig = apply_professional_style(fig, height=260)
    fig.update_layout(margin=dict(l=30, r=30, t=50, b=20))
    return fig


def create_3d_surface_plot(data, title=''):
    """Professional 3-D surface plot for dark-theme dashboards."""
    fig = go.Figure(data=[go.Surface(
        z=data,
        colorscale=[
            [0, COLOR_PALETTE['gradient_3'][0]],
            [0.5, COLOR_PALETTE['primary']],
            [1, COLOR_PALETTE['gradient_2'][1]],
        ],
        showscale=True,
        colorbar=dict(
            tickfont={'color': 'rgba(255,255,255,0.6)', 'size': 10},
            title={'text': 'Intensity', 'side': 'right',
                   'font': {'color': 'rgba(255,255,255,0.7)', 'size': 12}},
        ),
    )])
    axis_common = dict(
        backgroundcolor='rgba(0,0,0,0)',
        gridcolor='rgba(255,255,255,0.06)',
        showbackground=True,
        title={'font': {'color': 'rgba(255,255,255,0.6)'}},
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis=axis_common, yaxis=axis_common, zaxis=axis_common,
                   bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'rgba(255,255,255,0.87)'},
    )
    return fig


# ============================================================================
# SEVERITY ZONE SHAPES
# ============================================================================

def create_severity_zone_shapes(fig, y_max=1.0):
    """Add coloured horizontal zones for the four severity bands."""
    zones = [
        (0,    0.25, COLOR_PALETTE['state_verde'],    0.08),
        (0.25, 0.50, COLOR_PALETTE['state_amarillo'], 0.08),
        (0.50, 0.75, COLOR_PALETTE['state_naranja'],  0.08),
        (0.75, y_max, COLOR_PALETTE['state_rojo'],    0.08),
    ]
    for y0, y1, color, opa in zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=opa, line_width=0)
    return fig


# ============================================================================
# TIME-SERIES DUAL-AXIS FIGURE
# ============================================================================

def create_dual_axis_figure(x, y1, y2, name1='Signal', name2='Smoothed',
                            title='', height=300):
    """Overlay two traces sharing the x-axis; second on a secondary y-axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scattergl(x=x, y=y1, mode='lines', name=name1,
                     line=dict(color=COLOR_PALETTE['seq'][0], width=1.5)),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scattergl(x=x, y=y2, mode='lines', name=name2,
                     line=dict(color=COLOR_PALETTE['seq'][2], width=2)),
        secondary_y=True,
    )
    fig = apply_professional_style(fig, height=height)
    fig.update_layout(title=dict(text=title))
    return fig


# ============================================================================
# WATERFALL SPECTRUM (SPECTROGRAM)
# ============================================================================

def create_spectrogram(signal_data, fs, nperseg=256, title='Spectrogram'):
    """Return a Plotly Heatmap spectrogram for an input signal."""
    from scipy.signal import spectrogram as sp_spectrogram
    f, t, Sxx = sp_spectrogram(np.asarray(signal_data), fs=fs, nperseg=nperseg)
    fig = go.Figure(go.Heatmap(
        z=10 * np.log10(Sxx + 1e-20),
        x=t, y=f,
        colorscale=[
            [0.0, '#0f0c29'],
            [0.25, '#302b63'],
            [0.5, '#667eea'],
            [0.75, '#a78bfa'],
            [1.0, '#f5576c'],
        ],
        colorbar=dict(
            title='dB',
            tickfont={'color': 'rgba(255,255,255,0.6)'},
            titlefont={'color': 'rgba(255,255,255,0.7)'},
        ),
    ))
    fig = apply_professional_style(fig, height=300)
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        hovermode='closest',
    )
    return fig


# ============================================================================
# CONFUSION-MATRIX HEATMAP
# ============================================================================

def create_confusion_heatmap(matrix_dict, title='Confusion Matrix'):
    """Plotly annotated heatmap from a nested-dict confusion matrix."""
    classes = ['verde', 'amarillo', 'naranja', 'rojo']
    labels  = ['Verde', 'Amarillo', 'Naranja', 'Rojo']
    z = [[matrix_dict[r][c] for c in classes] for r in classes]
    text = [[str(v) for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate='%{text}',
        textfont={'size': 16, 'color': '#fff'},
        colorscale=[
            [0.0, 'rgba(255,255,255,0.03)'],
            [0.5, 'rgba(102,126,234,0.35)'],
            [1.0, 'rgba(102,126,234,0.8)'],
        ],
        showscale=False,
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>',
    ))
    fig = apply_professional_style(fig, height=360)
    fig.update_layout(
        title=title,
        xaxis_title='Predicted',
        yaxis_title='True',
        yaxis=dict(autorange='reversed'),
    )
    return fig
