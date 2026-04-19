from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO


def generate_pdf(title, data, summary):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    # ─────────────────────────────
    # STYLES
    # ─────────────────────────────
    title_style = ParagraphStyle(
        "title",
        parent=styles["Title"],
        textColor=colors.HexColor("#1976D2"),
        spaceAfter=20
    )

    section_style = ParagraphStyle(
        "section",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#0D47A1"),
        spaceBefore=12,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "normal",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        "bullet",
        parent=styles["Normal"],
        fontSize=10,
        leftIndent=12,
        leading=14,
        spaceAfter=2
    )

    content = []

    # ─────────────────────────────
    # TITLE
    # ─────────────────────────────
    content.append(Paragraph(title, title_style))
    content.append(Spacer(1, 12))

    # ─────────────────────────────
    # DATA SECTION
    # ─────────────────────────────
    content.append(Paragraph("DATA SUMMARY", section_style))

    for k, v in data.items():
        content.append(Paragraph(f"<b>{k}</b>: {v}", normal_style))

    content.append(Spacer(1, 12))

    # ─────────────────────────────
    # AI SECTION
    # ─────────────────────────────
    content.append(Paragraph("AI INSIGHTS", section_style))

    # safer parsing
    lines = summary.split("\n")

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # section headers
        if line.upper().startswith(("KEY INSIGHTS", "TRENDS", "ANOMALIES", "RECOMMENDATIONS")):
            content.append(Spacer(1, 6))
            content.append(Paragraph(line.upper(), section_style))
            continue

        # bullet lines
        if line.startswith("-"):
            clean = line[1:].strip()
            content.append(Paragraph(f"• {clean}", bullet_style))
        else:
            content.append(Paragraph(line, normal_style))

    doc.build(content)
    buffer.seek(0)
    return buffer