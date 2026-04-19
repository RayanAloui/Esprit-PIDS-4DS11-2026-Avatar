"""
CRM Vitale — AI Report Service
================================
Builds a comprehensive analytics summary from raw KPI DataFrames.
Works offline (no LLM needed) by computing statistical insights directly.
If google-genai is available, enriches the report with an AI executive summary.
"""

import numpy as np
from .llm_service import summarize_dashboard


def _fmt(n, decimals=1):
    """Format large numbers with M/K suffixes."""
    if abs(n) >= 1e6:
        return f"{n/1e6:,.{decimals}f}M"
    if abs(n) >= 1e3:
        return f"{n/1e3:,.{decimals}f}K"
    return f"{n:,.{decimals}f}"


def build_overview_ai(kpi_zone, kpi_mensuel, kpi_deleg, kpi_pharma):
    """
    Return a dict with:
        - 'summary_text': AI or fallback text (for backward compat)
        - 'analytics': dict of computed analytics for the PDF
    """

    # ── Revenue analytics ─────────────────────────────────────
    ca_total = kpi_zone["ca_ttc"].sum()
    nb_trans = kpi_zone["nb_transactions"].sum()
    nb_clients = kpi_zone["nb_clients"].sum()
    nb_zones = kpi_zone["zone"].nunique()

    ca_by_year = kpi_zone.groupby("annee")["ca_ttc"].sum().sort_index()
    years = sorted(ca_by_year.index)
    latest_year = years[-1]
    latest_ca = ca_by_year.iloc[-1]

    # Growth rates
    growth_rates = ca_by_year.pct_change().dropna() * 100
    avg_growth = growth_rates.mean()
    best_year = growth_rates.idxmax()
    best_growth = growth_rates.max()
    worst_year = growth_rates.idxmin()
    worst_growth = growth_rates.min()

    # ── Zone analytics ────────────────────────────────────────
    ca_by_zone = kpi_zone.groupby("zone")["ca_ttc"].sum().sort_values(ascending=False)
    top5_zones = ca_by_zone.head(5)
    top_zone_name = top5_zones.index[0]
    top_zone_ca = top5_zones.iloc[0]
    top_zone_pct = (top_zone_ca / ca_total) * 100
    bottom5_zones = ca_by_zone.tail(5)

    # Zone concentration: top 5 zones share
    top5_share = top5_zones.sum() / ca_total * 100

    # ── Delegate analytics ────────────────────────────────────
    nb_visites = kpi_deleg["nb_visites"].sum()
    nb_deleg = kpi_deleg["id_annimatrice"].nunique()
    avg_completion = kpi_deleg["taux_finalisation"].mean()

    deleg_totals = (kpi_deleg.groupby("id_annimatrice")
                    .agg(total_visits=("nb_visites", "sum"),
                         avg_rate=("taux_finalisation", "mean"))
                    .sort_values("total_visits", ascending=False))
    top_deleg = deleg_totals.head(5)
    bottom_deleg = deleg_totals.tail(5)
    avg_visits_per_deleg = deleg_totals["total_visits"].mean()

    # ── Pharmacy analytics ────────────────────────────────────
    nb_pharma = kpi_pharma["id_pharmay"].nunique()
    total_visits_pharma = kpi_pharma["total_visites"].sum()
    very_active = (kpi_pharma["total_visites"] >= 20).sum()
    active = ((kpi_pharma["total_visites"] >= 10) & (kpi_pharma["total_visites"] < 20)).sum()
    low_activity = ((kpi_pharma["total_visites"] >= 5) & (kpi_pharma["total_visites"] < 10)).sum()
    inactive = (kpi_pharma["total_visites"] < 5).sum()
    pct_inactive = inactive / nb_pharma * 100 if nb_pharma > 0 else 0
    max_visits = kpi_pharma["total_visites"].max()

    # ── Monthly seasonality ───────────────────────────────────
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ca_by_month = kpi_mensuel.groupby("mois")["ca_ttc"].mean().sort_index()
    best_month_idx = ca_by_month.idxmax()
    worst_month_idx = ca_by_month.idxmin()
    best_month = month_names[best_month_idx - 1] if 1 <= best_month_idx <= 12 else str(best_month_idx)
    worst_month = month_names[worst_month_idx - 1] if 1 <= worst_month_idx <= 12 else str(worst_month_idx)

    # ── Build analytics dict ──────────────────────────────────
    analytics = {
        # Revenue
        "ca_total": ca_total,
        "ca_total_fmt": _fmt(ca_total),
        "nb_transactions": nb_trans,
        "nb_clients": nb_clients,
        "nb_zones": nb_zones,
        "ca_per_client": ca_total / nb_clients if nb_clients > 0 else 0,
        "ca_per_transaction": ca_total / nb_trans if nb_trans > 0 else 0,
        "latest_year": int(latest_year),
        "latest_ca": latest_ca,
        "latest_ca_fmt": _fmt(latest_ca),

        # Growth
        "avg_growth": avg_growth,
        "best_growth_year": int(best_year),
        "best_growth_pct": best_growth,
        "worst_growth_year": int(worst_year),
        "worst_growth_pct": worst_growth,
        "ca_by_year": {int(y): float(v) for y, v in ca_by_year.items()},

        # Zones
        "top_zone": top_zone_name,
        "top_zone_ca": top_zone_ca,
        "top_zone_pct": top_zone_pct,
        "top5_zones": {z: float(v) for z, v in top5_zones.items()},
        "top5_share": top5_share,
        "bottom5_zones": {z: float(v) for z, v in bottom5_zones.items()},

        # Delegates
        "nb_delegates": nb_deleg,
        "nb_visites": nb_visites,
        "avg_completion": avg_completion,
        "avg_visits_per_deleg": avg_visits_per_deleg,
        "top5_delegates": {int(d): {"visits": int(r["total_visits"]), "rate": round(r["avg_rate"], 1)}
                           for d, r in top_deleg.iterrows()},

        # Pharmacies
        "nb_pharma": nb_pharma,
        "very_active": int(very_active),
        "active": int(active),
        "low_activity": int(low_activity),
        "inactive": int(inactive),
        "pct_inactive": pct_inactive,
        "max_visits": int(max_visits),

        # Seasonality
        "best_month": best_month,
        "best_month_ca": float(ca_by_month.max()),
        "worst_month": worst_month,
        "worst_month_ca": float(ca_by_month.min()),
    }

    # ── Build textual key insights ────────────────────────────
    insights = []
    insights.append(f"Total cumulative revenue across {len(years)} years ({years[0]}-{years[-1]}): {_fmt(ca_total)} TND")
    insights.append(f"The most recent year ({int(latest_year)}) generated {_fmt(latest_ca)} TND in revenue")
    insights.append(f"Average annual growth rate: {avg_growth:+.1f}%")
    insights.append(f"Best growth was in {int(best_year)} at {best_growth:+.1f}%, worst in {int(worst_year)} at {worst_growth:+.1f}%")
    insights.append(f"{top_zone_name} is the #1 zone contributing {top_zone_pct:.1f}% of total revenue ({_fmt(top_zone_ca)} TND)")
    insights.append(f"Top 5 zones account for {top5_share:.1f}% of all revenue — high geographic concentration")
    insights.append(f"{nb_deleg} active delegates performed {nb_visites:,} field visits with {avg_completion:.1f}% avg completion rate")
    insights.append(f"{pct_inactive:.0f}% of pharmacies ({inactive}/{nb_pharma}) are inactive (< 5 visits) — opportunity for re-engagement")
    insights.append(f"Strongest sales month: {best_month} (avg {_fmt(ca_by_month.max())} TND), weakest: {worst_month} (avg {_fmt(ca_by_month.min())} TND)")

    # Try LLM enrichment
    data_summary = {
        "ca": ca_total,
        "transactions": nb_trans,
        "zones": nb_zones,
        "visits": nb_visites,
        "pharmacies": nb_pharma,
    }
    llm_text = summarize_dashboard(data_summary, "CRM Overview Dashboard")

    # Build final summary text
    summary_lines = ["KEY INSIGHTS:"]
    for ins in insights:
        summary_lines.append(f"- {ins}")
    summary_lines.append("")

    # If LLM provided real content (not the fallback), append it
    if "AI summary is not available" not in llm_text:
        summary_lines.append("AI EXECUTIVE SUMMARY:")
        summary_lines.append(llm_text)
    else:
        # Add computed trends/risks instead
        summary_lines.append("TRENDS:")
        if avg_growth > 0:
            summary_lines.append(f"- Revenue shows a positive trajectory with {avg_growth:.1f}% average annual growth")
        else:
            summary_lines.append(f"- Revenue is declining with {avg_growth:.1f}% average annual contraction")

        if len(years) >= 3:
            recent_growth = growth_rates.iloc[-1] if len(growth_rates) > 0 else 0
            if recent_growth > avg_growth:
                summary_lines.append("- Recent growth is accelerating compared to historical average")
            else:
                summary_lines.append("- Recent growth is decelerating compared to historical average")

        summary_lines.append(f"- {best_month} consistently outperforms other months in revenue generation")
        summary_lines.append("")

        summary_lines.append("RISKS:")
        if pct_inactive > 40:
            summary_lines.append(f"- High pharmacy inactivity rate ({pct_inactive:.0f}%) signals weak field coverage")
        else:
            summary_lines.append(f"- Pharmacy inactivity at {pct_inactive:.0f}% — targeted re-engagement recommended")

        if top5_share > 50:
            summary_lines.append(f"- Revenue is highly concentrated: top 5 zones = {top5_share:.0f}% of total")
        summary_lines.append(f"- Completion rate of {avg_completion:.1f}% indicates room for improvement in delegate follow-ups")

        summary_lines.append("")
        summary_lines.append("RECOMMENDATIONS:")
        summary_lines.append("- Focus field visits on inactive pharmacies to expand active customer base")
        summary_lines.append("- Diversify revenue across underperforming zones to reduce geographic concentration risk")
        summary_lines.append(f"- Leverage {best_month} season for targeted sales campaigns and promotions")

    summary_text = "\n".join(summary_lines)

    return {"summary_text": summary_text, "analytics": analytics}
