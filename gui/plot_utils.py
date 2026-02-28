"""
Utilidades de visualización para el sistema PD-UHF
===================================================

Templates y funciones para crear gráficas profesionales con tema oscuro.
"""

import plotly.graph_objs as go
from plotly.subplots import make_subplots


# ============================================================================
# TEMPLATE PROFESIONAL DE PLOTLY
# ============================================================================

PROFESSIONAL_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(255,255,255,0.02)',
        'font': {
            'family': 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
            'color': '#fff',
            'size': 12
        },
        'title': {
            'font': {
                'size': 16,
                'color': '#fff',
                'family': 'Inter'
            },
            'x': 0.02,
            'xanchor': 'left'
        },
        'xaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'zerolinecolor': 'rgba(255,255,255,0.2)',
            'color': 'rgba(255,255,255,0.7)',
            'showline': True,
            'linecolor': 'rgba(255,255,255,0.2)',
        },
        'yaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'zerolinecolor': 'rgba(255,255,255,0.2)',
            'color': 'rgba(255,255,255,0.7)',
            'showline': True,
            'linecolor': 'rgba(255,255,255,0.2)',
        },
        'legend': {
            'bgcolor': 'rgba(0,0,0,0.3)',
            'bordercolor': 'rgba(255,255,255,0.2)',
            'borderwidth': 1,
            'font': {'color': '#fff'}
        },
        'hoverlabel': {
            'bgcolor': 'rgba(0,0,0,0.8)',
            'bordercolor': 'rgba(255,255,255,0.3)',
            'font': {'color': '#fff', 'family': 'Inter'}
        }
    }
}


# ============================================================================
# PALETA DE COLORES PROFESIONAL
# ============================================================================

COLOR_PALETTE = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#00ff88',
    'warning': '#ffa600',
    'danger': '#ff006e',
    'info': '#00d9ff',
    'gradient_1': ['#667eea', '#764ba2'],
    'gradient_2': ['#f093fb', '#f5576c'],
    'gradient_3': ['#4facfe', '#00f2fe'],
    'gradient_4': ['#43e97b', '#38f9d7'],
    'state_verde': '#00ff88',
    'state_amarillo': '#ffd700',
    'state_naranja': '#ff8c00',
    'state_rojo': '#ff006e',
}


def apply_professional_style(fig):
    """
    Aplicar estilo profesional a una figura de Plotly.
    
    Parámetros:
    -----------
    fig : plotly.graph_objs.Figure
        Figura a estilizar
    
    Retorna:
    --------
    fig : plotly.graph_objs.Figure
        Figura estilizada
    """
    fig.update_layout(PROFESSIONAL_TEMPLATE['layout'])
    
    # Agregar efectos adicionales
    fig.update_layout(
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(0,0,0,0.9)',
            bordercolor='rgba(102, 126, 234, 0.5)',
            font_size=12,
            font_family='Inter'
        ),
        margin=dict(l=60, r=30, t=60, b=50)
    )
    
    return fig


def create_gradient_line(x, y, name='', gradient_colors=None):
    """
    Crear línea con efecto de gradiente.
    
    Parámetros:
    -----------
    x : array-like
        Valores X
    y : array-like
        Valores Y
    name : str
        Nombre de la serie
    gradient_colors : list
        Lista de 2 colores para el gradiente
    
    Retorna:
    --------
    trace : plotly.graph_objs.Scatter
        Traza con gradiente
    """
    if gradient_colors is None:
        gradient_colors = COLOR_PALETTE['gradient_1']
    
    return go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name=name,
        line=dict(
            color=gradient_colors[0],
            width=2.5,
            shape='spline'
        ),
        fill='tozeroy',
        fillcolor=f'rgba({int(gradient_colors[0][1:3], 16)}, {int(gradient_colors[0][3:5], 16)}, {int(gradient_colors[0][5:7], 16)}, 0.2)',
        hovertemplate='<b>%{y:.4f}</b><extra></extra>'
    )


def create_professional_card_style():
    """
    Retornar estilo CSS para cards profesionales.
    
    Retorna:
    --------
    style : dict
        Diccionario de estilos
    """
    return {
        'background': 'rgba(255, 255, 255, 0.05)',
        'backdropFilter': 'blur(10px)',
        'border': '1px solid rgba(255, 255, 255, 0.1)',
        'borderRadius': '16px',
        'boxShadow': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
        'padding': '24px',
        'marginBottom': '24px'
    }


def create_metric_card(title, value, subtitle='', icon='fas fa-chart-line', color='primary'):
    """
    Crear card de métrica profesional.
    
    Parámetros:
    -----------
    title : str
        Título de la métrica
    value : str
        Valor principal
    subtitle : str
        Texto secundario
    icon : str
        Clase de icono FontAwesome
    color : str
        Color del tema (primary, success, etc.)
    
    Retorna:
    --------
    card : dash component
        Componente HTML
    """
    from dash import html
    
    color_map = {
        'primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'success': 'linear-gradient(135deg, #00ff88 0%, #00cc6a 100%)',
        'warning': 'linear-gradient(135deg, #ffa600 0%, #ff8c00 100%)',
        'danger': 'linear-gradient(135deg, #ff006e 0%, #d90058 100%)',
        'info': 'linear-gradient(135deg, #00d9ff 0%, #00a8cc 100%)'
    }
    
    return html.Div([
        html.Div([
            html.I(className=icon, style={
                'fontSize': '32px',
                'background': color_map.get(color, color_map['primary']),
                '-webkit-background-clip': 'text',
                '-webkit-text-fill-color': 'transparent',
                'marginBottom': '12px'
            }),
            html.H3(title, style={
                'fontSize': '14px',
                'color': 'rgba(255,255,255,0.6)',
                'fontWeight': '500',
                'marginBottom': '8px',
                'textTransform': 'uppercase',
                'letterSpacing': '0.05em'
            }),
            html.H2(value, style={
                'fontSize': '32px',
                'fontWeight': '700',
                'color': '#fff',
                'marginBottom': '4px',
                'letterSpacing': '-0.02em'
            }),
            html.P(subtitle, style={
                'fontSize': '12px',
                'color': 'rgba(255,255,255,0.5)',
                'marginBottom': '0'
            }) if subtitle else None
        ], style=create_professional_card_style())
    ])


def create_animated_gauge(value, title='', max_value=100, color_ranges=None):
    """
    Crear gauge animado profesional.
    
    Parámetros:
    -----------
    value : float
        Valor actual
    title : str
        Título del gauge
    max_value : float
        Valor máximo
    color_ranges : list
        Rangos de colores
    
    Retorna:
    --------
    fig : plotly.graph_objs.Figure
        Figura con gauge
    """
    if color_ranges is None:
        color_ranges = [
            (0, 0.25, COLOR_PALETTE['state_verde']),
            (0.25, 0.5, COLOR_PALETTE['state_amarillo']),
            (0.5, 0.75, COLOR_PALETTE['state_naranja']),
            (0.75, 1.0, COLOR_PALETTE['state_rojo'])
        ]
    
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=value,
        title={'text': title, 'font': {'size': 18, 'color': '#fff'}},
        delta={'reference': max_value * 0.5, 'increasing': {'color': COLOR_PALETTE['danger']}},
        gauge={
            'axis': {'range': [None, max_value], 'tickcolor': 'rgba(255,255,255,0.6)'},
            'bar': {'color': COLOR_PALETTE['primary']},
            'bgcolor': 'rgba(255,255,255,0.05)',
            'borderwidth': 2,
            'bordercolor': 'rgba(255,255,255,0.1)',
            'steps': [
                {'range': [r[0] * max_value, r[1] * max_value], 
                 'color': r[2], 
                 'name': f'{r[0]}-{r[1]}'} 
                for r in color_ranges
            ],
            'threshold': {
                'line': {'color': '#fff', 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig = apply_professional_style(fig)
    fig.update_layout(height=300)
    
    return fig


def create_3d_surface_plot(data, title=''):
    """
    Crear gráfica 3D de superficie profesional.
    
    Parámetros:
    -----------
    data : array-like
        Datos 2D para la superficie
    title : str
        Título de la gráfica
    
    Retorna:
    --------
    fig : plotly.graph_objs.Figure
        Figura 3D
    """
    fig = go.Figure(data=[go.Surface(
        z=data,
        colorscale=[
            [0, COLOR_PALETTE['gradient_3'][0]],
            [0.5, COLOR_PALETTE['primary']],
            [1, COLOR_PALETTE['gradient_2'][1]]
        ],
        showscale=True,
        colorbar=dict(
            tickfont={'color': '#fff'},
            title={'text': 'Intensidad', 'side': 'right', 'font': {'color': '#fff'}}
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='rgba(255,255,255,0.1)',
                showbackground=True,
                title={'font': {'color': '#fff'}}
            ),
            yaxis=dict(
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='rgba(255,255,255,0.1)',
                showbackground=True,
                title={'font': {'color': '#fff'}}
            ),
            zaxis=dict(
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='rgba(255,255,255,0.1)',
                showbackground=True,
                title={'font': {'color': '#fff'}}
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fff'}
    )
    
    return fig
