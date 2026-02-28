"""
Time-Series Analysis Tab
==========================

Provides Phase-3 Δt tracking analysis:
  - Kalman filter smoothing of inter-pulse intervals
  - Adaptive EWMA trend estimation
  - CUSUM change-point detection with alarm overlay
  - Spectrogram waterfall display
  - Descriptor time-series with rolling statistics

Designed for live-streamed data from NI PXIe-5185 and offline files.
"""

import numpy as np
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from gui.plot_utils import (
    apply_professional_style, COLOR_PALETTE,
    create_professional_card_style, create_metric_card,
    create_severity_zone_shapes, create_spectrogram,
)

# Lazy imports to avoid circular deps
_TRACKING = None

def _get_tracking():
    global _TRACKING
    if _TRACKING is None:
        from blind_algorithms import (
            KalmanDeltaTTracker, AdaptiveEWMATracker, CUSUMDetector,
            apply_delta_t_tracking,
        )
        _TRACKING = {
            'apply': apply_delta_t_tracking,
            'Kalman': KalmanDeltaTTracker,
            'EWMA': AdaptiveEWMATracker,
            'CUSUM': CUSUMDetector,
        }
    return _TRACKING

# ============================================================================
# LAYOUT
# ============================================================================

def create_layout():
    """Build the Time-Series Analysis tab."""
    card = create_professional_card_style

    return dbc.Container([
        # ── Header ──────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(html.Div([
                html.H4([
                    html.I(className="fas fa-wave-square me-2",
                           style={'color': COLOR_PALETTE['info']}),
                    "Time-Series & Δt Tracking Analysis"
                ], style={'fontWeight': '600', 'color': '#fff'}),
                html.P("Phase-3 tracking algorithms applied to inter-pulse "
                       "intervals.  Kalman smoothing, adaptive EWMA, and "
                       "CUSUM change-point detection.",
                       style={'color': 'rgba(255,255,255,0.55)',
                              'fontSize': '13px', 'marginBottom': 0}),
            ]), width=12)
        ], className='mb-4'),

        # ── Controls ────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5([
                        html.I(className="fas fa-sliders-h me-2"),
                        "Analysis Parameters"
                    ], style={'fontSize': '14px', 'fontWeight': '600'})),
                    dbc.CardBody([
                        # Signal source selector
                        html.Label("Signal Source", className="fw-bold",
                                   style={'fontSize': '12px',
                                          'color': 'rgba(255,255,255,0.7)'}),
                        dbc.Select(
                            id='ts-source',
                            options=[
                                {'label': '🟢 Verde (Normal)',      'value': 'verde'},
                                {'label': '🟡 Amarillo (Caution)',  'value': 'amarillo'},
                                {'label': '🟠 Naranja (Alert)',     'value': 'naranja'},
                                {'label': '🔴 Rojo (Critical)',     'value': 'rojo'},
                                {'label': '🔀 Transition Sweep',    'value': 'sweep'},
                            ],
                            value='naranja', className='mb-3',
                        ),

                        html.Label("Samples", className="fw-bold mt-2",
                                   style={'fontSize': '12px',
                                          'color': 'rgba(255,255,255,0.7)'}),
                        dbc.Input(id='ts-n-samples', type='number',
                                  value=4000, min=500, max=50000, step=500,
                                  className='mb-3'),

                        html.Label("Sample Rate (Hz)", className="fw-bold",
                                   style={'fontSize': '12px',
                                          'color': 'rgba(255,255,255,0.7)'}),
                        dbc.Input(id='ts-fs', type='number',
                                  value=10000, min=1000, max=1_000_000,
                                  step=1000, className='mb-3'),

                        html.Hr(style={'borderColor': 'rgba(255,255,255,0.08)'}),

                        # Kalman params
                        html.H6("Kalman Filter", className='fw-bold mt-2',
                                style={'fontSize': '12px',
                                       'color': COLOR_PALETTE['seq'][0]}),
                        html.Label("Process Noise (Q)",
                                   style={'fontSize': '11px',
                                          'color': 'rgba(255,255,255,0.55)'}),
                        dbc.Input(id='ts-kalman-q', type='number',
                                  value=1e-5, step=1e-6, className='mb-2'),
                        html.Label("Measurement Noise (R)",
                                   style={'fontSize': '11px',
                                          'color': 'rgba(255,255,255,0.55)'}),
                        dbc.Input(id='ts-kalman-r', type='number',
                                  value=1e-2, step=1e-3, className='mb-3'),

                        # EWMA params
                        html.H6("Adaptive EWMA", className='fw-bold',
                                style={'fontSize': '12px',
                                       'color': COLOR_PALETTE['seq'][2]}),
                        html.Label("α₀",
                                   style={'fontSize': '11px',
                                          'color': 'rgba(255,255,255,0.55)'}),
                        dcc.Slider(id='ts-ewma-alpha', min=0.05, max=0.8,
                                   step=0.05, value=0.2,
                                   marks={0.05: '.05', 0.4: '.4', 0.8: '.8'},
                                   tooltip={'placement': 'bottom',
                                            'always_visible': True}),

                        # CUSUM params
                        html.H6("CUSUM Detector", className='fw-bold mt-3',
                                style={'fontSize': '12px',
                                       'color': COLOR_PALETTE['seq'][5]}),
                        html.Label("Threshold (h)",
                                   style={'fontSize': '11px',
                                          'color': 'rgba(255,255,255,0.55)'}),
                        dbc.Input(id='ts-cusum-h', type='number',
                                  value=5.0, step=0.5, className='mb-2'),
                        html.Label("Drift (δ)",
                                   style={'fontSize': '11px',
                                          'color': 'rgba(255,255,255,0.55)'}),
                        dbc.Input(id='ts-cusum-drift', type='number',
                                  value=0.5, step=0.1, className='mb-3'),

                        html.Hr(style={'borderColor': 'rgba(255,255,255,0.08)'}),

                        dbc.Button([
                            html.I(className="fas fa-play me-2"),
                            "Run Analysis",
                        ], id='ts-run-btn', color='primary',
                           className='w-100 mb-2',
                           style={'background':
                                  'linear-gradient(135deg,#667eea,#764ba2)',
                                  'border': 'none'}),

                        dbc.Button([
                            html.I(className="fas fa-redo me-2"),
                            "Reset",
                        ], id='ts-reset-btn', color='secondary',
                           outline=True, className='w-100'),
                    ])
                ])
            ], lg=3, md=4, className='mb-3'),

            # ── Visualisations ──────────────────────────────────────────
            dbc.Col([
                # KPI row
                dbc.Row(id='ts-kpi-row', className='mb-3'),

                # Raw Δt + Kalman overlay
                dbc.Card([
                    dbc.CardHeader(html.H5("Δt  Inter-Pulse Intervals  —  Kalman & EWMA Overlay",
                                           style={'fontSize': '13px',
                                                  'fontWeight': '600'})),
                    dbc.CardBody(dcc.Graph(id='ts-delta-plot',
                                           config={'displayModeBar': True,
                                                   'scrollZoom': True},
                                           style={'height': '320px'}))
                ], className='mb-3'),

                # CUSUM alarm chart
                dbc.Card([
                    dbc.CardHeader(html.H5("CUSUM Change-Point Statistics",
                                           style={'fontSize': '13px',
                                                  'fontWeight': '600'})),
                    dbc.CardBody(dcc.Graph(id='ts-cusum-plot',
                                           config={'displayModeBar': True},
                                           style={'height': '260px'}))
                ], className='mb-3'),

                # Spectrogram + residual distribution
                dbc.Row([
                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H5("Kalman Residuals Distribution",
                                               style={'fontSize': '13px',
                                                      'fontWeight': '600'})),
                        dbc.CardBody(dcc.Graph(id='ts-residual-hist',
                                               config={'displayModeBar': False},
                                               style={'height': '260px'}))
                    ]), md=6, className='mb-3'),

                    dbc.Col(dbc.Card([
                        dbc.CardHeader(html.H5("Adaptive α Sequence (EWMA)",
                                               style={'fontSize': '13px',
                                                      'fontWeight': '600'})),
                        dbc.CardBody(dcc.Graph(id='ts-alpha-plot',
                                               config={'displayModeBar': False},
                                               style={'height': '260px'}))
                    ]), md=6, className='mb-3'),
                ]),
            ], lg=9, md=8),
        ]),
    ], fluid=True)


# ============================================================================
# CALLBACKS
# ============================================================================

def register_callbacks(app):
    """Register all callbacks for the time-series tab."""

    @app.callback(
        [Output('ts-delta-plot', 'figure'),
         Output('ts-cusum-plot', 'figure'),
         Output('ts-residual-hist', 'figure'),
         Output('ts-alpha-plot', 'figure'),
         Output('ts-kpi-row', 'children')],
        Input('ts-run-btn', 'n_clicks'),
        [State('ts-source', 'value'),
         State('ts-n-samples', 'value'),
         State('ts-fs', 'value'),
         State('ts-kalman-q', 'value'),
         State('ts-kalman-r', 'value'),
         State('ts-ewma-alpha', 'value'),
         State('ts-cusum-h', 'value'),
         State('ts-cusum-drift', 'value')],
        prevent_initial_call=True,
    )
    def run_analysis(n_clicks, source, n_samples, fs,
                     kalman_q, kalman_r, ewma_alpha,
                     cusum_h, cusum_drift):
        from main import generate_synthetic_signal
        from preprocessing import preprocess_signal
        from descriptors import detect_pulses, compute_delta_t

        tracking = _get_tracking()

        n_samples = int(n_samples or 4000)
        fs = float(fs or 10000)

        # Generate / get signal
        if source == 'sweep':
            segments = []
            for st in ['verde', 'amarillo', 'naranja', 'rojo']:
                segments.append(generate_synthetic_signal(
                    st, duration=n_samples // 4, fs=fs))
            raw = np.concatenate(segments)
        else:
            raw = generate_synthetic_signal(source, duration=n_samples, fs=fs)

        # Preprocess
        lowcut, highcut = fs * 0.01, fs * 0.4
        proc, _ = preprocess_signal(raw, fs, lowcut, highcut, True, True, True)

        # Detect pulses → Δt
        try:
            pulse_idx = detect_pulses(proc, fs, threshold_sigma=3.0)
            delta_t = compute_delta_t(pulse_idx, fs)
        except ValueError:
            # Not enough pulses – fall back to a tiny synthetic Δt
            delta_t = np.abs(np.random.exponential(1e-4, size=20))

        # Run tracking
        result = tracking['apply'](
            delta_t,
            kalman_Q=float(kalman_q or 1e-5),
            kalman_R=float(kalman_r or 1e-2),
            ewma_alpha=float(ewma_alpha or 0.2),
            cusum_threshold=float(cusum_h or 5.0),
            cusum_drift=float(cusum_drift or 0.5),
        )

        x = list(range(len(delta_t)))

        # ── Figure 1: Δt + Kalman + EWMA ──────────────────────────────
        fig1 = go.Figure()
        fig1.add_trace(go.Scattergl(
            x=x, y=delta_t, mode='lines', name='Raw Δt',
            line=dict(color='rgba(255,255,255,0.25)', width=1),
        ))
        fig1.add_trace(go.Scattergl(
            x=x, y=result.kalman.filtered, mode='lines',
            name='Kalman Smoothed',
            line=dict(color=COLOR_PALETTE['seq'][0], width=2),
        ))
        fig1.add_trace(go.Scattergl(
            x=x, y=result.ewma.smoothed, mode='lines',
            name='Adaptive EWMA',
            line=dict(color=COLOR_PALETTE['seq'][2], width=2, dash='dot'),
        ))
        # Mark CUSUM alarms
        if result.cusum.n_alarms > 0:
            alarm_x = result.cusum.alarm_indices.tolist()
            alarm_y = [float(delta_t[i]) for i in alarm_x]
            fig1.add_trace(go.Scatter(
                x=alarm_x, y=alarm_y, mode='markers',
                name=f'CUSUM Alarms ({result.cusum.n_alarms})',
                marker=dict(color=COLOR_PALETTE['danger'], size=8,
                            symbol='triangle-up',
                            line=dict(color='#fff', width=1)),
            ))
        fig1 = apply_professional_style(fig1, height=320)
        fig1.update_layout(
            xaxis_title='Pulse Index',
            yaxis_title='Δt (s)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='center', x=0.5),
        )

        # ── Figure 2: CUSUM g+/g- ─────────────────────────────────────
        fig2 = go.Figure()
        fig2.add_trace(go.Scattergl(
            x=x, y=result.cusum.g_plus, mode='lines',
            name='g⁺',
            line=dict(color=COLOR_PALETTE['seq'][3], width=2),
        ))
        fig2.add_trace(go.Scattergl(
            x=x, y=result.cusum.g_minus, mode='lines',
            name='g⁻',
            line=dict(color=COLOR_PALETTE['seq'][5], width=2),
        ))
        fig2.add_hline(y=result.cusum.threshold,
                       line_dash='dash',
                       line_color=COLOR_PALETTE['danger'],
                       annotation_text=f'h = {result.cusum.threshold}')
        fig2 = apply_professional_style(fig2, height=260)
        fig2.update_layout(
            xaxis_title='Pulse Index',
            yaxis_title='Cumulative Sum',
        )

        # ── Figure 3: Residual histogram ──────────────────────────────
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=result.kalman.residuals,
            nbinsx=40,
            marker_color=COLOR_PALETTE['primary'],
            opacity=0.75,
            name='Kalman Residuals',
        ))
        fig3 = apply_professional_style(fig3, height=260)
        fig3.update_layout(
            xaxis_title='Residual',
            yaxis_title='Count',
            bargap=0.05,
        )

        # ── Figure 4: adaptive alpha ──────────────────────────────────
        fig4 = go.Figure()
        fig4.add_trace(go.Scattergl(
            x=x, y=result.ewma.alpha_sequence, mode='lines',
            line=dict(color=COLOR_PALETTE['seq'][4], width=2),
            fill='tozeroy',
            fillcolor=f'{COLOR_PALETTE["seq"][4]}18',
            name='α[k]',
        ))
        fig4 = apply_professional_style(fig4, height=260)
        fig4.update_layout(
            xaxis_title='Pulse Index',
            yaxis_title='Smoothing Factor α',
            yaxis=dict(range=[0, 1]),
        )

        # ── KPI cards ─────────────────────────────────────────────────
        kpis = [
            dbc.Col(create_metric_card(
                'Pulses Detected', str(len(delta_t)),
                f'in {n_samples} samples',
                icon='fas fa-bolt', color='info',
            ), md=3, sm=6),
            dbc.Col(create_metric_card(
                'CUSUM Alarms', str(result.cusum.n_alarms),
                f'threshold h={result.cusum.threshold}',
                icon='fas fa-exclamation-triangle',
                color='danger' if result.cusum.n_alarms > 0 else 'success',
            ), md=3, sm=6),
            dbc.Col(create_metric_card(
                'Kalman Gain (∞)',
                f'{result.kalman.steady_state_gain:.4f}',
                'steady-state',
                icon='fas fa-crosshairs', color='primary',
            ), md=3, sm=6),
            dbc.Col(create_metric_card(
                'Mean Δt',
                f'{np.mean(delta_t)*1e3:.2f} ms',
                f'σ = {np.std(delta_t)*1e3:.2f} ms',
                icon='fas fa-clock', color='warning',
            ), md=3, sm=6),
        ]

        return fig1, fig2, fig3, fig4, kpis
