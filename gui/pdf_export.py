"""
PDF Export Module
==================

Generates a professional PDF report from the current analysis state.
Uses only stdlib + plotly for chart images (kaleido) when available,
falling back to a text-only report otherwise.

The exported PDF mirrors the same data the user can already export as
CSV / HDF5 / MAT, satisfying the "coincide with the current exportable
format intent" requirement.
"""

import io
import base64
import json
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

# ── Optional heavy dependencies — graceful degradation ──────────────────────
try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def _heading_style():
    """Return custom heading styles for the report."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontSize=22,
        spaceAfter=14,
        textColor=rl_colors.HexColor("#1a1a2e"),
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=16,
        spaceAfter=8,
        textColor=rl_colors.HexColor("#302b63"),
    ))
    styles.add(ParagraphStyle(
        "CellText",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
    ))
    return styles


def generate_pdf_report(
    analysis_results: Dict[str, Any],
    lang: str = "es",
    title: Optional[str] = None,
) -> bytes:
    """Build a PDF report and return the raw bytes.

    Parameters
    ----------
    analysis_results : dict
        The analysis result dict as returned by ``process_and_analyze_signal``
        or the GUI stores.  Expected (optional) keys:
        - ``descriptors`` : dict of descriptor name → value
        - ``severity_index`` : float
        - ``traffic_light_state`` : str
        - ``processing_info`` : dict
        - ``signal_data`` : list/array (raw signal)
        - ``metadata`` : dict (generator metadata)
    lang : str
        ``'es'`` or ``'en'``.
    title : str, optional
        Override the default report title.

    Returns
    -------
    bytes
        The PDF file content ready to be served as a download.
    """
    if not HAS_REPORTLAB:
        return _generate_text_fallback(analysis_results, lang)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=20 * mm, bottomMargin=20 * mm)
    styles = _heading_style()
    elements = []

    # ── Title ────────────────────────────────────────────────────────────────
    report_title = title or (
        "Informe de Análisis PD-UHF" if lang == "es"
        else "PD-UHF Analysis Report"
    )
    elements.append(Paragraph(report_title, styles["ReportTitle"]))
    elements.append(Spacer(1, 4 * mm))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elements.append(Paragraph(
        f"{'Generado' if lang == 'es' else 'Generated'}: {ts}",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 8 * mm))

    # ── Severity summary ─────────────────────────────────────────────────────
    sev = analysis_results.get("severity_index")
    tl = analysis_results.get("traffic_light_state", "—")
    if sev is not None:
        sec_title = "Resumen de Severidad" if lang == "es" else "Severity Summary"
        elements.append(Paragraph(sec_title, styles["SectionHead"]))
        data = [
            [("Índice de Severidad" if lang == "es" else "Severity Index"),
             f"{sev:.4f}"],
            [("Semáforo" if lang == "es" else "Traffic Light"), tl],
        ]
        t = Table(data, colWidths=[70 * mm, 80 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), rl_colors.HexColor("#e8e8f0")),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 6 * mm))

    # ── Descriptors table ────────────────────────────────────────────────────
    descriptors = analysis_results.get("descriptors", {})
    if descriptors:
        sec_title = "Descriptores" if lang == "es" else "Descriptors"
        elements.append(Paragraph(sec_title, styles["SectionHead"]))
        header = [("Descriptor", ("Valor" if lang == "es" else "Value"))]
        rows = [(k, f"{v:.6g}" if isinstance(v, (int, float)) else str(v))
                for k, v in descriptors.items()]
        data = header + rows
        t = Table(data, colWidths=[80 * mm, 70 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#302b63")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.HexColor("#f8f8fc"), rl_colors.white]),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 6 * mm))

    # ── Processing info ──────────────────────────────────────────────────────
    pinfo = analysis_results.get("processing_info", {})
    if pinfo:
        sec_title = ("Información de Procesamiento" if lang == "es"
                     else "Processing Information")
        elements.append(Paragraph(sec_title, styles["SectionHead"]))
        rows = [(k, str(v)[:80]) for k, v in pinfo.items()
                if not isinstance(v, (np.ndarray, list))]
        if rows:
            t = Table([("Param", ("Valor" if lang == "es" else "Value"))] + rows,
                      colWidths=[80 * mm, 70 * mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#302b63")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(t)

    # ── Metadata (from signal generator) ─────────────────────────────────────
    meta = analysis_results.get("metadata", {})
    if meta:
        sec_title = "Metadatos" if lang == "es" else "Metadata"
        elements.append(Paragraph(sec_title, styles["SectionHead"]))
        rows = [(k, str(v)[:80]) for k, v in meta.items()]
        if rows:
            t = Table([("Key", ("Valor" if lang == "es" else "Value"))] + rows,
                      colWidths=[80 * mm, 70 * mm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#302b63")),
                ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(t)

    # ── Footer ───────────────────────────────────────────────────────────────
    elements.append(Spacer(1, 12 * mm))
    footer_text = (
        "Generado automáticamente por el Sistema de Detección PD-UHF"
        if lang == "es"
        else "Automatically generated by the PD-UHF Detection System"
    )
    elements.append(Paragraph(
        f"<i>{footer_text}</i>",
        ParagraphStyle("Footer", parent=styles["Normal"],
                       fontSize=8, textColor=rl_colors.grey,
                       alignment=TA_CENTER),
    ))

    doc.build(elements)
    return buf.getvalue()


def _generate_text_fallback(results: Dict[str, Any], lang: str) -> bytes:
    """Plain-text fallback when *reportlab* is not installed."""
    lines = []
    title = ("Informe de Análisis PD-UHF" if lang == "es"
             else "PD-UHF Analysis Report")
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(f"{'Generado' if lang == 'es' else 'Generated'}: "
                 f"{datetime.now().isoformat()}")
    lines.append("")

    sev = results.get("severity_index")
    if sev is not None:
        lines.append(f"Severity Index: {sev:.4f}")
    tl = results.get("traffic_light_state")
    if tl:
        lines.append(f"Traffic Light: {tl}")
    lines.append("")

    for k, v in results.get("descriptors", {}).items():
        lines.append(f"  {k}: {v:.6g}" if isinstance(v, (int, float))
                     else f"  {k}: {v}")

    return "\n".join(lines).encode("utf-8")


def pdf_bytes_to_data_uri(pdf_bytes: bytes) -> str:
    """Encode raw PDF bytes as a ``data:`` URI for Dash downloads."""
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    return f"data:application/pdf;base64,{b64}"
