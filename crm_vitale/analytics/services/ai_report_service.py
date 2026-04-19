from .llm_service import summarize_dashboard

def build_overview_ai(kpi_zone, kpi_mensuel, kpi_deleg, kpi_pharma):
    data_summary = {
        "total_ca": kpi_zone["ca_ttc"].sum(),
        "transactions": kpi_zone["nb_transactions"].sum(),
        "zones": kpi_zone["zone"].nunique(),
        "visits": kpi_deleg["nb_visites"].sum(),
        "pharmacies": kpi_pharma["id_pharmay"].nunique(),
    }

    return summarize_dashboard(data_summary, "CRM Overview Dashboard")