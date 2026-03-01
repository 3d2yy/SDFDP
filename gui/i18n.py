"""
Internationalization (i18n) — English / Spanish toggle
=======================================================

Provides a translation dictionary and a helper that GUI modules
call to resolve the current-language string for any key.

Usage
-----
    from gui.i18n import t, TRANSLATIONS

    # In a callback that knows the current language ('en' or 'es'):
    label = t('app.title', lang)
"""

from typing import Dict

# ── master translation table ────────────────────────────────────────────────
# Keys follow a dot-separated namespace:  module.element

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    # ── App-level ────────────────────────────────────────────────────────────
    "app.title":          {"es": "Sistema PD-UHF",
                           "en": "PD-UHF System"},
    "app.subtitle":       {"es": "Detección Profesional — NI PXIe-5185 · 12.5 GS/s",
                           "en": "Professional Detection — NI PXIe-5185 · 12.5 GS/s"},
    "app.status_active":  {"es": "Sistema Activo",
                           "en": "System Active"},
    "app.select_tab":     {"es": "Seleccione una pestaña",
                           "en": "Select a tab"},

    # ── Tab labels ───────────────────────────────────────────────────────────
    "tab.live":           {"es": "📡 Captura en Vivo",
                           "en": "📡 Live Capture"},
    "tab.files":          {"es": "📂 Análisis de Archivos",
                           "en": "📂 File Analysis"},
    "tab.timeseries":     {"es": "📈 Time-Series & Δt",
                           "en": "📈 Time-Series & Δt"},
    "tab.generator":      {"es": "⚙️ Generador de Señales",
                           "en": "⚙️ Signal Generator"},
    "tab.thresholds":     {"es": "🎯 Configuración de Umbrales",
                           "en": "🎯 Threshold Config"},
    "tab.docs":           {"es": "📚 Documentación",
                           "en": "📚 Documentation"},
    "tab.description":    {"es": "📖 Descripción Técnica",
                           "en": "📖 Technical Description"},

    # ── PDF / Export ─────────────────────────────────────────────────────────
    "export.pdf_btn":     {"es": "📄 Exportar PDF",
                           "en": "📄 Export PDF"},
    "export.pdf_title":   {"es": "Informe de Análisis PD-UHF",
                           "en": "PD-UHF Analysis Report"},
    "export.no_data":     {"es": "No hay datos para exportar.",
                           "en": "No data to export."},
    "export.success":     {"es": "PDF generado exitosamente.",
                           "en": "PDF generated successfully."},
    "export.csv":         {"es": "Exportar CSV",
                           "en": "Export CSV"},
    "export.hdf5":        {"es": "Exportar HDF5",
                           "en": "Export HDF5"},
    "export.mat":         {"es": "Exportar MAT",
                           "en": "Export MAT"},

    # ── Signal generator ─────────────────────────────────────────────────────
    "gen.title":          {"es": "Generador de Señales Sintéticas",
                           "en": "Synthetic Signal Generator"},
    "gen.state":          {"es": "Estado Operativo",
                           "en": "Operational State"},
    "gen.duration":       {"es": "Duración (muestras)",
                           "en": "Duration (samples)"},
    "gen.fs":             {"es": "Frecuencia de Muestreo (Hz)",
                           "en": "Sampling Frequency (Hz)"},
    "gen.noise":          {"es": "Nivel de Ruido",
                           "en": "Noise Level"},
    "gen.generate":       {"es": "Generar Señal",
                           "en": "Generate Signal"},
    "gen.randomize":      {"es": "Aleatorizar",
                           "en": "Randomize"},

    # ── File analysis ────────────────────────────────────────────────────────
    "file.upload":        {"es": "Arrastre o seleccione un archivo",
                           "en": "Drag or select a file"},
    "file.analyze":       {"es": "Analizar",
                           "en": "Analyze"},

    # ── Threshold config ─────────────────────────────────────────────────────
    "thresh.title":       {"es": "Configuración de Umbrales",
                           "en": "Threshold Configuration"},
    "thresh.method":      {"es": "Método",
                           "en": "Method"},
    "thresh.percentile":  {"es": "Percentil",
                           "en": "Percentile"},
    "thresh.statistical": {"es": "Estadístico",
                           "en": "Statistical"},

    # ── Traffic-light states ────────────────────────────────────────────────
    "state.verde":        {"es": "Verde — Normal",
                           "en": "Green — Normal"},
    "state.amarillo":     {"es": "Amarillo — Precaución",
                           "en": "Yellow — Caution"},
    "state.naranja":      {"es": "Naranja — Alerta",
                           "en": "Orange — Alert"},
    "state.rojo":         {"es": "Rojo — Crítico",
                           "en": "Red — Critical"},

    # ── Live capture ─────────────────────────────────────────────────────────
    "live.title":         {"es": "Captura en Tiempo Real",
                           "en": "Real-Time Capture"},
    "live.start":         {"es": "Iniciar Captura",
                           "en": "Start Capture"},
    "live.stop":          {"es": "Detener Captura",
                           "en": "Stop Capture"},

    # ── Time-series ──────────────────────────────────────────────────────────
    "ts.title":           {"es": "Análisis de Series Temporales y Δt",
                           "en": "Time-Series & Δt Analysis"},

    # ── Documentation ────────────────────────────────────────────────────────
    "docs.title":         {"es": "Documentación del Sistema",
                           "en": "System Documentation"},
    "desc.title":         {"es": "Descripción Técnica",
                           "en": "Technical Description"},

    # ── Severity ─────────────────────────────────────────────────────────────
    "sev.index":          {"es": "Índice de Severidad",
                           "en": "Severity Index"},
    "sev.traffic_light":  {"es": "Semáforo",
                           "en": "Traffic Light"},

    # ── WCAG ─────────────────────────────────────────────────────────────────
    "a11y.skip_nav":      {"es": "Saltar al contenido principal",
                           "en": "Skip to main content"},
    "a11y.lang_toggle":   {"es": "Cambiar a English",
                           "en": "Switch to Español"},
}


def t(key: str, lang: str = "es") -> str:
    """Return the translation for *key* in the given language.

    Falls back to Spanish, then to the key itself, so the app never crashes
    on a missing translation.
    """
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang, entry.get("es", key))
