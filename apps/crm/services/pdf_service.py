"""
CRM Vitale — PDF Report Generator
====================================
Produces a professional, multi-page executive PDF report
with KPI tables, zone rankings, delegate stats, and analytics insights.
"""

from io import BytesIO
from datetime import datetime


try:
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────
C_PRIMARY = colors.HexColor("#0D47A1")
C_PRIMARY_LIGHT = colors.HexColor("#1565C0")
C_ACCENT = colors.HexColor("#1976D2")
C_GREEN = colors.HexColor("#2E7D32")
C_ORANGE = colors.HexColor("#E64A19")
C_RED = colors.HexColor("#C62828")
C_TEAL = colors.HexColor("#00695C")
C_PURPLE = colors.HexColor("#6A1B9A")
C_DARK = colors.HexColor("#1A2744")
C_LIGHT_BG = colors.HexColor("#F5F7FA")
C_WHITE = colors.white
C_TEXT = colors.HexColor("#263238")
C_TEXT_DIM = colors.HexColor("#546E7A")
C_BORDER = colors.HexColor("#B0BEC5")


def _fmt_number(n, decimals=1):
    """Format large numbers with M/K suffixes."""
    if abs(n) >= 1e6:
        return f"{n / 1e6:,.{decimals}f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:,.{decimals}f}K"
    return f"{n:,.{decimals}f}"


def generate_pdf(title, data, report_data):
    """
    Generate a professional PDF report.

    Args:
        title:       Report title string
        data:        dict with raw KPI values (ca, transactions, zones, visits, pharmacies)
        report_data: dict with 'summary_text' and 'analytics' (from build_overview_ai)
                     For backward compat, can also be a plain string.
    """
    if not _HAS_REPORTLAB:
        buffer = BytesIO()
        buffer.write(b"reportlab is not installed. Run: pip install reportlab")
        buffer.seek(0)
        return buffer

    # Handle backward compat: if report_data is a string, wrap it
    if isinstance(report_data, str):
        summary_text = report_data
        analytics = {}
    else:
        summary_text = report_data.get("summary_text", "")
        analytics = report_data.get("analytics", {})

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=25 * mm,
        leftMargin=25 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    # ─────────────────────────────────────────────
    # CUSTOM STYLES
    # ─────────────────────────────────────────────
    s_title = ParagraphStyle(
        "report_title", parent=styles["Title"],
        fontName="Helvetica-Bold", fontSize=22, leading=28,
        textColor=C_PRIMARY, spaceAfter=4,
    )
    s_subtitle = ParagraphStyle(
        "report_subtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=10, leading=14,
        textColor=C_TEXT_DIM, spaceAfter=16,
    )
    s_section = ParagraphStyle(
        "report_section", parent=styles["Heading2"],
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=C_PRIMARY, spaceBefore=18, spaceAfter=10,
    )
    s_subsection = ParagraphStyle(
        "report_subsection", parent=styles["Heading3"],
        fontName="Helvetica-Bold", fontSize=11, leading=15,
        textColor=C_ACCENT, spaceBefore=12, spaceAfter=6,
    )
    s_body = ParagraphStyle(
        "report_body", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9.5, leading=13.5,
        textColor=C_TEXT, spaceAfter=4,
    )
    s_body_bold = ParagraphStyle(
        "report_body_bold", parent=s_body,
        fontName="Helvetica-Bold",
    )
    s_bullet = ParagraphStyle(
        "report_bullet", parent=s_body,
        leftIndent=14, bulletIndent=6, spaceAfter=3,
    )
    s_kpi_value = ParagraphStyle(
        "kpi_value", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=16, leading=20,
        textColor=C_PRIMARY, alignment=TA_CENTER,
    )
    s_kpi_label = ParagraphStyle(
        "kpi_label", parent=styles["Normal"],
        fontName="Helvetica", fontSize=7.5, leading=10,
        textColor=C_TEXT_DIM, alignment=TA_CENTER,
    )
    s_footer = ParagraphStyle(
        "footer_style", parent=styles["Normal"],
        fontName="Helvetica", fontSize=7, leading=9,
        textColor=C_TEXT_DIM, alignment=TA_CENTER,
    )

    content = []

    # ═════════════════════════════════════════════
    # COVER / HEADER
    # ═════════════════════════════════════════════
    # Header bar
    header_data = [[
        Paragraph("💊 CRM VITALE", ParagraphStyle("hdr", parent=s_title, fontSize=18, textColor=C_WHITE)),
        Paragraph("Executive Report", ParagraphStyle("hdr2", parent=s_subtitle, textColor=colors.HexColor("#90CAF9"), alignment=TA_RIGHT)),
    ]]
    header_table = Table(header_data, colWidths=[doc.width * 0.55, doc.width * 0.45])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, -1), C_WHITE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING", (0, 0), (0, 0), 16),
        ("RIGHTPADDING", (-1, -1), (-1, -1), 16),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
    ]))
    content.append(header_table)
    content.append(Spacer(1, 6))

    # Title & metadata
    content.append(Paragraph(title, s_title))
    now = datetime.now().strftime("%d/%m/%Y à %H:%M")
    content.append(Paragraph(
        f"Generated on {now} · Pharmaceutical Market Tunisia · 2019–2026",
        s_subtitle
    ))
    content.append(HRFlowable(width="100%", thickness=1, color=C_BORDER, spaceBefore=4, spaceAfter=12))

    # ═════════════════════════════════════════════
    # KPI SUMMARY CARDS (as a table)
    # ═════════════════════════════════════════════
    content.append(Paragraph("📊 Key Performance Indicators", s_section))

    kpi_items = [
        (_fmt_number(analytics.get("ca_total", data.get("ca", 0))), "Total Revenue\n(TND)", C_PRIMARY),
        (f"{analytics.get('nb_transactions', data.get('transactions', 0)):,}", "Transactions", C_GREEN),
        (f"{analytics.get('nb_clients', 0):,}", "Unique Clients", C_ORANGE),
        (str(analytics.get("nb_zones", data.get("zones", 0))), "Active Zones", C_TEAL),
        (f"{analytics.get('nb_visites', data.get('visits', 0)):,}", "Field Visits", C_PURPLE),
        (str(analytics.get("nb_pharma", data.get("pharmacies", 0))), "Pharmacies", C_RED),
    ]

    kpi_row_values = []
    kpi_row_labels = []
    for val, label, _ in kpi_items:
        kpi_row_values.append(Paragraph(val, s_kpi_value))
        kpi_row_labels.append(Paragraph(label, s_kpi_label))

    col_w = doc.width / len(kpi_items)
    kpi_table = Table(
        [kpi_row_values, kpi_row_labels],
        colWidths=[col_w] * len(kpi_items),
    )
    kpi_style = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), C_LIGHT_BG),
        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]
    # Color the top border of each KPI cell
    for i, (_, _, color) in enumerate(kpi_items):
        kpi_style.append(("LINEABOVE", (i, 0), (i, 0), 3, color))
    kpi_table.setStyle(TableStyle(kpi_style))
    content.append(kpi_table)
    content.append(Spacer(1, 8))

    # Additional derived KPIs
    if analytics:
        ca_client = analytics.get("ca_per_client", 0)
        ca_trans = analytics.get("ca_per_transaction", 0)
        avg_compl = analytics.get("avg_completion", 0)
        avg_vis = analytics.get("avg_visits_per_deleg", 0)

        derived = [
            [Paragraph("<b>Avg Revenue/Client</b>", s_body), Paragraph(f"{_fmt_number(ca_client)} TND", s_body)],
            [Paragraph("<b>Avg Revenue/Transaction</b>", s_body), Paragraph(f"{_fmt_number(ca_trans)} TND", s_body)],
            [Paragraph("<b>Avg Completion Rate</b>", s_body), Paragraph(f"{avg_compl:.1f}%", s_body)],
            [Paragraph("<b>Avg Visits/Delegate</b>", s_body), Paragraph(f"{avg_vis:.0f}", s_body)],
        ]
        d_table = Table(derived, colWidths=[doc.width * 0.45, doc.width * 0.55])
        d_table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("BACKGROUND", (0, 0), (0, -1), C_LIGHT_BG),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        content.append(d_table)
        content.append(Spacer(1, 6))

    # ═════════════════════════════════════════════
    # REVENUE BY YEAR TABLE
    # ═════════════════════════════════════════════
    if analytics.get("ca_by_year"):
        content.append(Paragraph("📈 Annual Revenue Breakdown", s_section))

        rev_header = [
            Paragraph("<b>Year</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
            Paragraph("<b>Revenue (TND)</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
            Paragraph("<b>Growth</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
        ]
        rev_rows = [rev_header]

        ca_year = analytics["ca_by_year"]
        sorted_years = sorted(ca_year.keys())
        prev_ca = None
        for y in sorted_years:
            ca = ca_year[y]
            if prev_ca and prev_ca > 0:
                growth = ((ca - prev_ca) / prev_ca) * 100
                growth_str = f"{growth:+.1f}%"
                growth_color = C_GREEN if growth >= 0 else C_RED
            else:
                growth_str = "—"
                growth_color = C_TEXT_DIM
            rev_rows.append([
                Paragraph(str(y), s_body),
                Paragraph(f"{_fmt_number(ca)} TND", s_body),
                Paragraph(f"<font color='{growth_color.hexval()}'>{growth_str}</font>", s_body),
            ])
            prev_ca = ca

        rev_table = Table(rev_rows, colWidths=[doc.width * 0.2, doc.width * 0.45, doc.width * 0.35])
        rev_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT_BG]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        content.append(rev_table)
        content.append(Spacer(1, 6))

    # ═════════════════════════════════════════════
    # TOP ZONES TABLE
    # ═════════════════════════════════════════════
    if analytics.get("top5_zones"):
        content.append(Paragraph("🗺️ Top 5 Revenue Zones", s_section))

        zone_header = [
            Paragraph("<b>Rank</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
            Paragraph("<b>Zone</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
            Paragraph("<b>Revenue (TND)</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
            Paragraph("<b>Share %</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
        ]
        zone_rows = [zone_header]
        total = analytics.get("ca_total", 1)

        for i, (zone, ca) in enumerate(analytics["top5_zones"].items(), 1):
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            rank_str = medals.get(i, f"#{i}")
            share = (ca / total) * 100
            zone_rows.append([
                Paragraph(rank_str, s_body),
                Paragraph(f"<b>{zone}</b>", s_body),
                Paragraph(f"{_fmt_number(ca)} TND", s_body),
                Paragraph(f"{share:.1f}%", s_body),
            ])

        zone_table = Table(zone_rows, colWidths=[doc.width * 0.12, doc.width * 0.33, doc.width * 0.33, doc.width * 0.22])
        zone_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_TEAL),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT_BG]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        content.append(zone_table)
        content.append(Spacer(1, 4))
        content.append(Paragraph(
            f"<i>Top 5 zones concentration: {analytics.get('top5_share', 0):.1f}% of total revenue</i>",
            ParagraphStyle("note", parent=s_body, textColor=C_TEXT_DIM, fontSize=8),
        ))
        content.append(Spacer(1, 6))

    # ═════════════════════════════════════════════
    # DELEGATE PERFORMANCE TABLE
    # ═════════════════════════════════════════════
    if analytics.get("top5_delegates"):
        content.append(PageBreak())
        content.append(Paragraph("👥 Top 5 Delegates Performance", s_section))

        deleg_header = [
            Paragraph("<b>Delegate ID</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
            Paragraph("<b>Total Visits</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
            Paragraph("<b>Completion Rate</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
            Paragraph("<b>Performance</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
        ]
        deleg_rows = [deleg_header]

        for d_id, stats in analytics["top5_delegates"].items():
            rate = stats["rate"]
            if rate >= 80:
                perf = "🟢 Excellent"
            elif rate >= 60:
                perf = "🟡 Good"
            else:
                perf = "🔴 Needs Improvement"
            deleg_rows.append([
                Paragraph(f"Delegate {d_id}", s_body),
                Paragraph(f"{stats['visits']:,}", s_body),
                Paragraph(f"{rate}%", s_body),
                Paragraph(perf, s_body),
            ])

        deleg_table = Table(deleg_rows, colWidths=[doc.width * 0.25, doc.width * 0.2, doc.width * 0.25, doc.width * 0.3])
        deleg_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_PURPLE),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT_BG]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        content.append(deleg_table)
        content.append(Spacer(1, 6))

        # Delegate summary
        content.append(Paragraph(
            f"<b>{analytics['nb_delegates']}</b> active delegates · "
            f"<b>{analytics['nb_visites']:,}</b> total visits · "
            f"<b>{analytics['avg_visits_per_deleg']:.0f}</b> avg visits/delegate · "
            f"<b>{analytics['avg_completion']:.1f}%</b> avg completion rate",
            s_body
        ))
        content.append(Spacer(1, 6))

    # ═════════════════════════════════════════════
    # PHARMACY SEGMENTATION
    # ═════════════════════════════════════════════
    if analytics.get("nb_pharma"):
        content.append(Paragraph("🏥 Pharmacy Segmentation", s_section))

        seg_header = [
            Paragraph("<b>Segment</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
            Paragraph("<b>Criteria</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
            Paragraph("<b>Count</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
            Paragraph("<b>Share</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
        ]

        nb_p = analytics["nb_pharma"]
        segments = [
            ("🟢 Very Active", "≥ 20 visits", analytics["very_active"]),
            ("🔵 Active", "10–19 visits", analytics["active"]),
            ("🟡 Low Activity", "5–9 visits", analytics["low_activity"]),
            ("🔴 Inactive", "< 5 visits", analytics["inactive"]),
        ]

        seg_rows = [seg_header]
        for label, criteria, count in segments:
            pct = (count / nb_p * 100) if nb_p > 0 else 0
            seg_rows.append([
                Paragraph(label, s_body),
                Paragraph(criteria, s_body),
                Paragraph(str(count), s_body),
                Paragraph(f"{pct:.1f}%", s_body),
            ])
        # Total row
        seg_rows.append([
            Paragraph("<b>Total</b>", s_body_bold),
            Paragraph("", s_body),
            Paragraph(f"<b>{nb_p}</b>", s_body_bold),
            Paragraph("<b>100%</b>", s_body_bold),
        ])

        seg_table = Table(seg_rows, colWidths=[doc.width * 0.28, doc.width * 0.22, doc.width * 0.2, doc.width * 0.3])
        seg_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_ORANGE),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2), [C_WHITE, C_LIGHT_BG]),
            ("BACKGROUND", (0, -1), (-1, -1), C_LIGHT_BG),
            ("LINEABOVE", (0, -1), (-1, -1), 1.5, C_PRIMARY),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        content.append(seg_table)
        content.append(Spacer(1, 4))
        content.append(Paragraph(
            f"<i>Max visits to a single pharmacy: {analytics.get('max_visits', '—')}</i>",
            ParagraphStyle("note", parent=s_body, textColor=C_TEXT_DIM, fontSize=8),
        ))
        content.append(Spacer(1, 6))

    # ═════════════════════════════════════════════
    # SEASONALITY
    # ═════════════════════════════════════════════
    if analytics.get("best_month"):
        content.append(Paragraph("📅 Seasonality Analysis", s_section))

        season_data = [
            [Paragraph("<b>Metric</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold")),
             Paragraph("<b>Value</b>", ParagraphStyle("th", parent=s_body, textColor=C_WHITE, fontName="Helvetica-Bold"))],
            [Paragraph("Best performing month", s_body),
             Paragraph(f"<b>{analytics['best_month']}</b> (avg {_fmt_number(analytics['best_month_ca'])} TND)", s_body)],
            [Paragraph("Weakest performing month", s_body),
             Paragraph(f"<b>{analytics['worst_month']}</b> (avg {_fmt_number(analytics['worst_month_ca'])} TND)", s_body)],
            [Paragraph("Best growth year", s_body),
             Paragraph(f"<b>{analytics.get('best_growth_year', '—')}</b> ({analytics.get('best_growth_pct', 0):+.1f}%)", s_body)],
            [Paragraph("Worst growth year", s_body),
             Paragraph(f"<b>{analytics.get('worst_growth_year', '—')}</b> ({analytics.get('worst_growth_pct', 0):+.1f}%)", s_body)],
            [Paragraph("Average annual growth", s_body),
             Paragraph(f"<b>{analytics.get('avg_growth', 0):+.1f}%</b>", s_body)],
        ]
        season_table = Table(season_data, colWidths=[doc.width * 0.4, doc.width * 0.6])
        season_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
            ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_LIGHT_BG]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("ROUNDEDCORNERS", [4, 4, 4, 4]),
        ]))
        content.append(season_table)
        content.append(Spacer(1, 6))

    # ═════════════════════════════════════════════
    # INSIGHTS & ANALYSIS
    # ═════════════════════════════════════════════
    content.append(PageBreak())
    content.append(Paragraph("🧠 Analytics Insights & Recommendations", s_section))

    # Parse summary text into structured sections
    lines = summary_text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            content.append(Spacer(1, 4))
            continue

        # Section headers
        upper = line.upper()
        if upper.startswith(("KEY INSIGHTS", "TRENDS", "ANOMALIES",
                             "RECOMMENDATIONS", "RISKS", "AI EXECUTIVE")):
            # Section divider
            content.append(Spacer(1, 6))
            section_color = C_PRIMARY
            if "RISK" in upper:
                section_color = C_RED
            elif "RECOMMEND" in upper:
                section_color = C_GREEN
            elif "TREND" in upper:
                section_color = C_ACCENT

            content.append(HRFlowable(width="100%", thickness=1, color=section_color, spaceBefore=2, spaceAfter=4))
            content.append(Paragraph(
                line.rstrip(":"),
                ParagraphStyle("insight_section", parent=s_subsection, textColor=section_color),
            ))
            continue

        # Bullet lines
        if line.startswith("-"):
            clean = line[1:].strip()
            content.append(Paragraph(f"• {clean}", s_bullet))
        else:
            content.append(Paragraph(line, s_body))

    # ═════════════════════════════════════════════
    # FOOTER
    # ═════════════════════════════════════════════
    content.append(Spacer(1, 24))
    content.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceBefore=8, spaceAfter=8))
    content.append(Paragraph(
        f"CRM Vitale · VITAL SA · Generated on {now} · Confidential",
        s_footer
    ))

    # Build PDF
    doc.build(content)
    buffer.seek(0)
    return buffer
