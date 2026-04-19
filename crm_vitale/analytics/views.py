"""
DSO 3 - CRM Vitale | Django Views
===================================
All analytics views with Plotly charts
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from django.shortcuts import render
from django.http import HttpResponse
from .services.ai_report_service import build_overview_ai
from .services.pdf_service import generate_pdf


# ─────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "data", "models")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    kpi_zone    = pd.read_csv(f"{DATA_DIR}/kpi_ca_zone_annee.csv")
    kpi_mensuel = pd.read_csv(f"{DATA_DIR}/kpi_ca_mensuel.csv")
    kpi_top     = pd.read_csv(f"{DATA_DIR}/kpi_top_zones.csv")
    kpi_deleg   = pd.read_csv(f"{DATA_DIR}/kpi_visites_delegue.csv")
    kpi_pharma  = pd.read_csv(f"{DATA_DIR}/kpi_freq_pharmacie.csv")
    return kpi_zone, kpi_mensuel, kpi_top, kpi_deleg, kpi_pharma

# ─────────────────────────────────────────────
# CHART CONFIG
# ─────────────────────────────────────────────
COLORS = {
    "bg":    "#0F1923",
    "card":  "#1A2744",
    "text":  "#E8EDF2",
    "blue":  "#1976D2",
    "green": "#2E7D32",
    "orange":"#E64A19",
    "red":   "#C62828",
    "chart": ["#1976D2","#26A69A","#F57C00","#7B1FA2","#C62828","#00838F"],
}

def base_layout(title="", height=400):
    return dict(
        paper_bgcolor=COLORS["card"],
        plot_bgcolor =COLORS["card"],
        height=height,
        title=dict(text=title, font=dict(color=COLORS["text"], size=14)),
        font=dict(family="Inter", color=COLORS["text"], size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text"])),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", color=COLORS["text"],
                   tickfont=dict(color=COLORS["text"])),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color=COLORS["text"],
                   tickfont=dict(color=COLORS["text"])),
        margin=dict(t=55, b=35, l=40, r=30),
    )

def fig_to_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False)

# ─────────────────────────────────────────────
# VIEW 1 — OVERVIEW
# ─────────────────────────────────────────────
def overview(request):
    kpi_zone, kpi_mensuel, kpi_top, kpi_deleg, kpi_pharma = load_data()

    # KPIs
    ca_total   = kpi_zone["ca_ttc"].sum()
    nb_trans   = kpi_zone["nb_transactions"].sum()
    nb_clients = kpi_zone["nb_clients"].sum()
    nb_zones   = kpi_zone["zone"].nunique()
    nb_vis     = kpi_deleg["nb_visites"].sum()
    nb_pharma  = kpi_pharma["id_pharmay"].nunique()
    nb_deleg   = kpi_deleg["id_annimatrice"].nunique()
    ca_cl      = ca_total / nb_clients if nb_clients > 0 else 0
    taux_fin   = kpi_deleg["taux_finalisation"].mean()

    # Chart 1 — Revenue by year
    ca_an = kpi_zone.groupby("annee").agg(
        ca=("ca_ttc","sum"),
        trans=("nb_transactions","sum")
    ).reset_index()

    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Bar(
        x=ca_an["annee"], y=ca_an["ca"], name="Revenue",
        marker_color=COLORS["blue"],
        text=[f"{v/1e6:.1f}M" for v in ca_an["ca"]],
        textposition="outside", textfont=dict(color=COLORS["text"])
    ), secondary_y=False)
    fig1.add_trace(go.Scatter(
        x=ca_an["annee"], y=ca_an["trans"], name="Transactions",
        mode="lines+markers", line=dict(color=COLORS["orange"], width=2.5),
        marker=dict(size=9, color=COLORS["orange"])
    ), secondary_y=True)
    l1 = base_layout("Annual Revenue & Transactions", 400)
    l1["legend"] = dict(orientation="h", y=1.1, font=dict(color=COLORS["text"]))
    fig1.update_layout(**l1)
    fig1.update_yaxes(secondary_y=False, gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color=COLORS["text"]))
    fig1.update_yaxes(secondary_y=True,  gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color=COLORS["orange"]))
    chart1 = fig_to_html(fig1)

    # Chart 2 — Heatmap
    pivot = kpi_mensuel.pivot(index="annee", columns="mois", values="ca_ttc").fillna(0)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [months[i-1] for i in pivot.columns]
    fig2 = px.imshow(pivot,
        color_continuous_scale=[[0,COLORS["card"]],[0.4,"#0D47A1"],[1,"#90CAF9"]],
        aspect="auto", title="Monthly Revenue Heatmap",
        labels=dict(x="Month", y="Year", color="Revenue"))
    fig2.update_layout(**base_layout("Monthly Revenue Heatmap", 400))
    chart2 = fig_to_html(fig2)

    # Chart 3 — Growth rate
    ca_gr = kpi_zone.groupby("annee")["ca_ttc"].sum().reset_index()
    ca_gr["growth"] = ca_gr["ca_ttc"].pct_change() * 100
    ca_gr = ca_gr.dropna()
    bar_colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in ca_gr["growth"]]
    fig3 = go.Figure(go.Bar(
        x=ca_gr["annee"], y=ca_gr["growth"],
        marker_color=bar_colors,
        text=[f"{v:+.1f}%" for v in ca_gr["growth"]],
        textposition="outside", textfont=dict(color=COLORS["text"])
    ))
    fig3.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.25)")
    fig3.update_layout(**base_layout("Revenue Growth Rate (%)", 300))
    chart3 = fig_to_html(fig3)

    context = {
        "ca_total":   f"{ca_total/1e6:.1f}M",
        "nb_trans":   f"{nb_trans:,.0f}",
        "nb_clients": f"{nb_clients:,.0f}",
        "nb_vis":     f"{nb_vis:,}",
        "nb_zones":   nb_zones,
        "nb_pharma":  nb_pharma,
        "nb_deleg":   nb_deleg,
        "ca_cl":      f"{ca_cl:,.0f}",
        "taux_fin":   f"{taux_fin:.1f}",
        "chart1":     chart1,
        "chart2":     chart2,
        "chart3":     chart3,
    }
    return render(request, "analytics/overview.html", context)

# ─────────────────────────────────────────────
# VIEW 2 — ZONES
# ─────────────────────────────────────────────
def zones(request):
    kpi_zone, _, kpi_top, _, _ = load_data()

    # Top 15 zones
    top15 = (kpi_zone.groupby("zone")["ca_ttc"].sum().reset_index()
             .sort_values("ca_ttc", ascending=False).head(15).sort_values("ca_ttc"))
    fig1 = go.Figure(go.Bar(
        x=top15["ca_ttc"], y=top15["zone"], orientation="h",
        marker=dict(color=top15["ca_ttc"],
            colorscale=[[0,"#0D47A1"],[0.5,"#1976D2"],[1,"#90CAF9"]], showscale=False),
        text=[f"  {v/1e6:.2f}M" for v in top15["ca_ttc"]],
        textposition="outside", textfont=dict(color=COLORS["text"])
    ))
    fig1.update_layout(**base_layout("Top 15 Zones by Revenue", 500))
    chart1 = fig_to_html(fig1)

    # Pie chart
    top10 = (kpi_zone.groupby("zone")["ca_ttc"].sum().reset_index()
             .sort_values("ca_ttc", ascending=False))
    pie_d = pd.concat([
        top10.head(8),
        pd.DataFrame([{"zone":"Others","ca_ttc":top10.iloc[8:]["ca_ttc"].sum()}])
    ], ignore_index=True)
    fig2 = px.pie(pie_d, values="ca_ttc", names="zone", hole=0.45,
        title="Revenue Share — Top 8 Zones",
        color_discrete_sequence=COLORS["chart"])
    fig2.update_traces(textposition="outside", textinfo="percent+label",
        textfont=dict(color=COLORS["text"]))
    fig2.update_layout(**{**base_layout("Revenue Share", 420), "showlegend":False})
    chart2 = fig_to_html(fig2)

    # Evolution top 5
    top5_noms = top10.head(5)["zone"].tolist()
    evo = kpi_zone[kpi_zone["zone"].isin(top5_noms)]
    fig3 = px.line(evo, x="annee", y="ca_ttc", color="zone", markers=True,
        title="Revenue Evolution — Top 5 Zones",
        labels={"ca_ttc":"Revenue (TND)","annee":"Year","zone":"Zone"},
        color_discrete_sequence=COLORS["chart"])
    fig3.update_traces(line=dict(width=2.5), marker=dict(size=9))
    l3 = base_layout("Revenue Evolution — Top 5 Zones", 420)
    l3["legend"] = dict(orientation="h", y=1.1, font=dict(color=COLORS["text"]))
    fig3.update_layout(**l3)
    chart3 = fig_to_html(fig3)

    # Top zone KPI
    top_zone     = top10.iloc[0]
    top_zone_pct = top_zone["ca_ttc"] / top10["ca_ttc"].sum() * 100

    context = {
        "chart1":        chart1,
        "chart2":        chart2,
        "chart3":        chart3,
        "top_zone":      top_zone["zone"],
        "top_zone_ca":   f"{top_zone['ca_ttc']/1e6:.1f}M",
        "top_zone_pct":  f"{top_zone_pct:.1f}",
        "nb_zones":      kpi_zone["zone"].nunique(),
    }
    return render(request, "analytics/zones.html", context)

# ─────────────────────────────────────────────
# VIEW 3 — DELEGATES
# ─────────────────────────────────────────────
def delegates(request):
    _, _, _, kpi_deleg, _ = load_data()

    td = (kpi_deleg.groupby("id_annimatrice")
          .agg(total_visits=("nb_visites","sum"),
               total_pharmacies=("nb_pharmacies","max"),
               completion_rate=("taux_finalisation","mean"))
          .reset_index().sort_values("total_visits", ascending=False))

    # Bar chart top 15
    t15 = td.head(15).sort_values("total_visits").copy()
    t15["label"] = "Delegate " + t15["id_annimatrice"].astype(str)
    fig1 = go.Figure(go.Bar(
        x=t15["total_visits"], y=t15["label"], orientation="h",
        marker=dict(color=t15["completion_rate"],
            colorscale="RdYlGn", showscale=True,
            colorbar=dict(title="Rate %",
                title_font=dict(color=COLORS["text"]),
                tickfont=dict(color=COLORS["text"]), thickness=12)),
        text=t15["total_visits"], textposition="outside",
        textfont=dict(color=COLORS["text"])
    ))
    fig1.update_layout(**base_layout("Top 15 Delegates — Visits & Completion Rate", 520))
    chart1 = fig_to_html(fig1)

    # Scatter
    t15b = td.head(15).copy()
    t15b["label"] = "Delegate " + t15b["id_annimatrice"].astype(str)
    fig2 = px.scatter(t15b,
        x="total_visits", y="total_pharmacies",
        size="total_visits", color="completion_rate", text="label",
        title="Coverage: Visits vs Pharmacies",
        labels={"total_visits":"Visits","total_pharmacies":"Pharmacies","completion_rate":"Rate %"},
        color_continuous_scale="RdYlGn", size_max=45)
    fig2.update_traces(textposition="top center", textfont=dict(color=COLORS["text"]))
    fig2.update_coloraxes(colorbar=dict(tickfont=dict(color=COLORS["text"]),
        title=dict(font=dict(color=COLORS["text"]))))
    fig2.update_layout(**base_layout("Coverage: Visits vs Pharmacies", 500))
    chart2 = fig_to_html(fig2)

    # Visits over time
    vis_an = kpi_deleg.groupby("annee")["nb_visites"].sum().reset_index()
    fig3 = go.Figure(go.Scatter(
        x=vis_an["annee"], y=vis_an["nb_visites"],
        fill="tozeroy", fillcolor="rgba(25,118,210,0.15)",
        line=dict(color=COLORS["blue"], width=3),
        mode="lines+markers+text",
        marker=dict(size=11, color=COLORS["blue"]),
        text=vis_an["nb_visites"], textposition="top center",
        textfont=dict(color=COLORS["text"])
    ))
    fig3.update_layout(**base_layout("Field Visits per Year", 360))
    chart3 = fig_to_html(fig3)

    context = {
        "chart1":          chart1,
        "chart2":          chart2,
        "chart3":          chart3,
        "nb_deleg":        len(td),
        "total_visits":    f"{td['total_visits'].sum():,}",
        "avg_visits":      f"{td['total_visits'].mean():.0f}",
        "avg_completion":  f"{td['completion_rate'].mean():.1f}",
    }
    return render(request, "analytics/delegates.html", context)

# ─────────────────────────────────────────────
# VIEW 4 — PHARMACIES
# ─────────────────────────────────────────────
def pharmacies(request):
    _, _, _, _, kpi_pharma = load_data()

    kpi_pharma["segment"] = kpi_pharma["total_visites"].apply(
        lambda v: "Very Active" if v>=20 else ("Active" if v>=10 else ("Low Activity" if v>=5 else "Inactive"))
    )

    # Bar segmentation
    order = ["Very Active","Active","Low Activity","Inactive"]
    seg = kpi_pharma["segment"].value_counts().reindex(order).reset_index()
    seg.columns = ["segment","count"]
    fig1 = go.Figure(go.Bar(
        x=seg["segment"], y=seg["count"],
        marker_color=[COLORS["chart"][1], COLORS["chart"][0], COLORS["orange"], COLORS["red"]],
        text=seg["count"], textposition="outside",
        textfont=dict(color=COLORS["text"])
    ))
    fig1.update_layout(**base_layout("Pharmacy Segmentation by Activity", 420))
    chart1 = fig_to_html(fig1)

    # Histogram
    fig2 = px.histogram(kpi_pharma, x="total_visites", nbins=25,
        title="Visit Distribution per Pharmacy",
        labels={"total_visites":"Number of Visits"},
        color_discrete_sequence=[COLORS["blue"]])
    fig2.update_layout(**base_layout("Visit Distribution per Pharmacy", 420))
    chart2 = fig_to_html(fig2)

    # Pie
    fig3 = px.pie(seg, values="count", names="segment", hole=0.5,
        color_discrete_sequence=[COLORS["chart"][1], COLORS["chart"][0], COLORS["orange"], COLORS["red"]])
    fig3.update_traces(textposition="outside", textinfo="percent+label",
        textfont=dict(color=COLORS["text"]))
    fig3.update_layout(**{**base_layout("Segment Breakdown", 380), "showlegend":True,
        "legend":dict(font=dict(color=COLORS["text"]), orientation="h", y=-0.12)})
    chart3 = fig_to_html(fig3)

    inactives    = (kpi_pharma["total_visites"] < 5).sum()
    tres_actives = (kpi_pharma["total_visites"] >= 20).sum()

    context = {
        "chart1":        chart1,
        "chart2":        chart2,
        "chart3":        chart3,
        "total_pharma":  len(kpi_pharma),
        "very_active":   tres_actives,
        "inactive":      inactives,
        "pct_inactive":  f"{inactives/len(kpi_pharma)*100:.1f}",
        "max_visits":    kpi_pharma["total_visites"].max(),
    }
    return render(request, "analytics/pharmacies.html", context)

# ─────────────────────────────────────────────
# VIEW 5 — PREDICTIONS
# ─────────────────────────────────────────────
def predictions(request):
    try:
        df_pred = pd.read_csv(f"{MODEL_DIR}/revenue_predictions.csv")
        df_imp  = pd.read_csv(f"{MODEL_DIR}/feature_importance.csv")
        df_met  = pd.read_csv(f"{MODEL_DIR}/model_metrics.csv")

        # Predictions chart
        fig1 = px.bar(
            df_pred, x="zone", y="predicted_ca", color="annee",
            barmode="group", title="Predicted Revenue by Zone (2025 & 2026)",
            labels={"predicted_ca":"Predicted Revenue (TND)","zone":"Zone","annee":"Year"},
            color_discrete_sequence=COLORS["chart"]
        )
        fig1.update_layout(**base_layout("Predicted Revenue by Zone", 500))
        chart1 = fig_to_html(fig1)

        # Top 10 predictions 2025
        top_pred = df_pred[df_pred["annee"]==2025].nlargest(10,"predicted_ca")
        fig2 = go.Figure(go.Bar(
            x=top_pred["predicted_ca"], y=top_pred["zone"], orientation="h",
            marker_color=COLORS["blue"],
            text=[f"{v/1e6:.2f}M" for v in top_pred["predicted_ca"]],
            textposition="outside", textfont=dict(color=COLORS["text"])
        ))
        fig2.update_layout(**base_layout("Top 10 Zones — 2025 Revenue Prediction", 420))
        chart2 = fig_to_html(fig2)

        # Feature importance
        fig3 = go.Figure(go.Bar(
            x=df_imp["importance"], y=df_imp["feature"], orientation="h",
            marker_color=COLORS["chart"][1],
            text=[f"{v:.3f}" for v in df_imp["importance"]],
            textposition="outside", textfont=dict(color=COLORS["text"])
        ))
        fig3.update_layout(**base_layout("Feature Importance (Random Forest)", 380))
        chart3 = fig_to_html(fig3)

        # Metrics
        total_2025 = df_pred[df_pred["annee"]==2025]["predicted_ca"].sum()
        total_2026 = df_pred[df_pred["annee"]==2026]["predicted_ca"].sum()

        best_model = df_met.loc[df_met["R2"].idxmax()]

        context = {
            "chart1":       chart1,
            "chart2":       chart2,
            "chart3":       chart3,
            "total_2025":   f"{total_2025/1e6:.2f}M",
            "total_2026":   f"{total_2026/1e6:.2f}M",
            "best_model":   best_model["model"],
            "r2":           f"{best_model['R2']:.4f}",
            "mae":          f"{best_model['MAE']:,.0f}",
            "model_ready":  True,
        }
    except FileNotFoundError:
        context = {"model_ready": False}

    return render(request, "analytics/predictions.html", context)
def overview_pdf(request):
    kpi_zone, kpi_mensuel, kpi_top, kpi_deleg, kpi_pharma = load_data()

    summary = build_overview_ai(
        kpi_zone, kpi_mensuel, kpi_deleg, kpi_pharma
    )

    data = {
        "ca": kpi_zone["ca_ttc"].sum(),
        "transactions": kpi_zone["nb_transactions"].sum(),
        "zones": kpi_zone["zone"].nunique(),
        "visits": kpi_deleg["nb_visites"].sum(),
        "pharmacies": kpi_pharma["id_pharmay"].nunique(),
    }

    pdf = generate_pdf("Overview Dashboard Report", data, summary)

    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="overview_report.pdf"'

    return response