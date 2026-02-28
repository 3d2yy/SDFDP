"""
Módulo GUI para el sistema de detección de descargas parciales.

Submodules
----------
plot_utils        – Visualization helpers, COLOR_PALETTE, professional styling
hardware_bridge   – NI PXIe-5185 abstraction (hardware + simulation fallback)
live_capture      – Real-time capture tab
file_analysis     – File upload & analysis tab
signal_generator  – Synthetic signal generation tab
threshold_config  – Threshold configuration & confusion-matrix evaluation
time_series       – Δt time-series tracking (Kalman / EWMA / CUSUM)
documentation     – Usage guide tab
description       – Technical description tab
i18n              – EN/ES internationalization strings & helper
pdf_export        – PDF report generation (reportlab or text fallback)
"""
