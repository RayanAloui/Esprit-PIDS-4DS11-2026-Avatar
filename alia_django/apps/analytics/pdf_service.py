"""
ALIA Analytics — PDF Report Generator
"""
import io
from datetime import datetime

try:
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


# Colors matching Analytics UI
C_CYAN = colors.HexColor("#00d4ff")
C_GOLD = colors.HexColor("#f0c040")
C_GREEN = colors.HexColor("#00e676")
C_RED = colors.HexColor("#ff4444")
C_ORANGE = colors.HexColor("#ff9800")
C_NAVY = colors.HexColor("#0a1628")
C_WHITE = colors.white
C_TEXT = colors.HexColor("#263238")
C_BORDER = colors.HexColor("#cfd8dc")
C_BG = colors.HexColor("#f8f9fa")

def _decode_b64(base64_str):
    if not base64_str:
        return None
    import base64
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        decoded = base64.b64decode(base64_str)
        return io.BytesIO(decoded)
    except Exception:
        return None

def generate_analytics_pdf(data, action_plan=None, images=None):
    """
    Generate a professional PDF report for Analytics.
    :param data: Extracted payload from `_build_dashboard_data`
    :param action_plan: Extracted payload (List of dicts) representing the action plan
    :param images: dict of base64 strings coming from frontend charts
    """
    images = images or {}
    if not _HAS_REPORTLAB:
        buffer = io.BytesIO()
        buffer.write(b"Reportlab non disponible. Installez-le.")
        buffer.seek(0)
        return buffer

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=25*mm, leftMargin=25*mm,
        topMargin=20*mm, bottomMargin=20*mm,
    )
    styles = getSampleStyleSheet()

    # Styles
    s_title = ParagraphStyle(
        "title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, leading=26, textColor=C_NAVY, spaceAfter=8,
    )
    s_subtitle = ParagraphStyle(
        "subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=10, textColor=colors.gray, spaceAfter=16,
    )
    s_section = ParagraphStyle(
        "section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, textColor=C_CYAN, spaceBefore=20, spaceAfter=10,
    )
    s_body = ParagraphStyle(
        "body", parent=styles["Normal"], fontName="Helvetica", fontSize=10, leading=14, textColor=C_TEXT, spaceAfter=6,
    )
    
    content = []
    username = data.get("username", "Délégué")
    
    # ── En-tête
    header_data = [[
        Paragraph("<b>ALIA</b> Analytics", ParagraphStyle("h1", parent=s_title, textColor=C_WHITE, fontSize=16)),
        Paragraph("Bilan Compétences Sim", ParagraphStyle("h2", parent=s_subtitle, textColor=C_WHITE, alignment=TA_RIGHT)),
    ]]
    t_header = Table(header_data, colWidths=[doc.width*0.5, doc.width*0.5])
    t_header.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_NAVY),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 12), ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('ROUNDEDCORNERS', [4,4,4,4]),
    ]))
    content.append(t_header)
    content.append(Spacer(1, 15))

    # ── Titre
    content.append(Paragraph(f"Rapport de Progression — {username}", s_title))
    content.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", s_subtitle))
    content.append(HRFlowable(width="100%", thickness=1, color=C_BORDER, spaceBefore=4, spaceAfter=15))

    stats = data.get("summary_stats", {})

    # ── KPIS
    content.append(Paragraph("📊 Statistiques Clés", s_section))
    kpi_items = [
        ("Sessions Testées", str(stats.get("total", 0)), C_CYAN),
        ("Score Moyen", f"{stats.get('avg_score', 0)}/10", C_CYAN),
        ("Niveau Régulier", stats.get("niveau_dominant", ""), C_GOLD),
        ("Taux Excellent", f"{stats.get('taux_excellent', 0)}%", C_GREEN),
        ("Taux Conformité", f"{stats.get('taux_conforme', 0)}%", C_GREEN if stats.get("taux_conforme",0)>=80 else C_RED),
    ]

    row_vals = [Paragraph(f"<b>{val}</b>", ParagraphStyle("v", parent=s_body, fontSize=14, textColor=col, alignment=TA_CENTER)) for _, val, col in kpi_items]
    row_lbls = [Paragraph(lbl, ParagraphStyle("l", parent=s_body, fontSize=8, textColor=colors.gray, alignment=TA_CENTER)) for lbl, _, _ in kpi_items]
    
    t_kpis = Table([row_vals, row_lbls], colWidths=[doc.width/5.0]*5)
    t_kpis.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), C_BG), ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('TOPPADDING', (0,0), (-1,-1), 10), ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    content.append(t_kpis)

    # ── GRAPHIQUES (Si disponibles)
    from reportlab.platypus import Image as RLImage

    img_prog = _decode_b64(images.get("progression"))
    img_radar = _decode_b64(images.get("radar"))
    img_niveau = _decode_b64(images.get("niveau"))
    img_qual = _decode_b64(images.get("qualite"))
    
    if img_prog or img_radar:
        content.append(Spacer(1, 15))
        content.append(Paragraph("📈 Visualisations de Progression", s_section))
        graphs_row = []
        if img_prog:
            graphs_row.append(RLImage(img_prog, width=90*mm, height=45*mm))
        else:
            graphs_row.append(Paragraph("Métrique indisp.", s_body))
            
        if img_radar:
            graphs_row.append(RLImage(img_radar, width=65*mm, height=45*mm))
        else:
            graphs_row.append(Paragraph("Métrique indisp.", s_body))
            
        t_graphs1 = Table([graphs_row], colWidths=[doc.width*0.55, doc.width*0.45])
        t_graphs1.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
        ]))
        content.append(t_graphs1)
    
    if img_niveau or img_qual:
        content.append(Spacer(1, 10))
        graphs_row2 = []
        if img_niveau:
            graphs_row2.append(RLImage(img_niveau, width=80*mm, height=40*mm))
        if img_qual:
            graphs_row2.append(RLImage(img_qual, width=60*mm, height=40*mm))
            
        t_graphs2 = Table([graphs_row2], colWidths=[doc.width/2.0]*2)
        t_graphs2.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
        ]))
        content.append(t_graphs2)

    # ── Objections les plus fréquentes
    clusters = data.get("objection_clusters", {})
    if clusters:
        content.append(Spacer(1, 10))
        content.append(Paragraph("💡 Objections Rencontrées", s_section))
        obj_data = [[Paragraph("<b>Catégorie</b>", s_body), Paragraph("<b>Occurrences</b>", s_body), Paragraph("<b>Aisance /10</b>", s_body)]]
        
        for cat, info in sorted(clusters.items(), key=lambda x: x[1]['count'], reverse=True):
            avg = info['avg_score']
            color = C_GREEN if avg >= 7 else C_GOLD if avg >= 5 else C_RED
            obj_data.append([
                Paragraph(cat, s_body), 
                Paragraph(str(info['count']), s_body), 
                Paragraph(f"<font color='{color.hexval()}'><b>{avg}</b></font>", s_body)
            ])
            
        t_obj = Table(obj_data, colWidths=[doc.width*0.4, doc.width*0.3, doc.width*0.3])
        t_obj.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
            ('BACKGROUND', (0,0), (-1,0), C_NAVY), ('TEXTCOLOR', (0,0), (-1,0), C_WHITE),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [C_WHITE, C_BG]),
            ('TOPPADDING', (0,0), (-1,-1), 8), ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        content.append(t_obj)

    # ── PLAN D'ACTION (IA)
    if action_plan:
        content.append(PageBreak())
        content.append(Paragraph("✨ Plan d'Action Personnalisé (Coach IA)", s_section))
        content.append(Paragraph("Conçu par le moteur RAG d'ALIA suite à l'analyse de l'évolution de vos entraînements.", s_body))
        content.append(Spacer(1, 10))
        
        for idx, item in enumerate(action_plan, 1):
            priorite = str(item.get("priorite", "")).lower()
            p_color = C_RED if "haut" in priorite else C_ORANGE if "moyen" in priorite else C_CYAN
            
            p_box = [[
                Paragraph(f"<b>Axe {idx} : {item.get('axe', '')}</b>", ParagraphStyle("ax", parent=s_body, textColor=C_WHITE, fontSize=12)),
                Paragraph(f"[Priorité : {priorite.upper()}]", ParagraphStyle("pr", parent=s_body, textColor=C_WHITE, alignment=TA_RIGHT))
            ]]
            t_plan = Table(p_box, colWidths=[doc.width*0.7, doc.width*0.3])
            t_plan.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), p_color),
                ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ('ROUNDEDCORNERS', [4,4,4,4]),
            ]))
            content.append(t_plan)
            content.append(Spacer(1, 4))
            
            content.append(Paragraph(f"<b>Stratégie :</b> {item.get('description', '')}", s_body))
            content.append(Paragraph(f"<b>Objectif chiffré :</b> <font color='{C_CYAN.hexval()}'>{item.get('seuil_cible', '')}</font>", s_body))
            content.append(Spacer(1, 15))


    # ── Footer
    content.append(Spacer(1, 30))
    content.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceBefore=4, spaceAfter=8))
    content.append(Paragraph("VITAL SA — Rapport Simulation ALIA", ParagraphStyle("f", parent=s_body, fontSize=8, textColor=colors.gray, alignment=TA_CENTER)))

    doc.build(content)
    buffer.seek(0)
    return buffer
