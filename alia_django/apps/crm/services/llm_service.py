"""
CRM Vitale — LLM Service
==========================
Optional AI enrichment via Google Gemini.
Falls back gracefully if the package or API key is unavailable.
"""

import os

try:
    from google import genai
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False

_FALLBACK = (
    "KEY INSIGHTS:\n"
    "- AI summary is not available\n\n"
    "TRENDS:\n"
    "- Data loaded successfully\n\n"
    "RISKS:\n"
    "- No AI analysis available"
)


def summarize_dashboard(data, title):
    """
    Try to call Gemini for an executive summary.
    Returns a plain-text summary string, or a fallback if unavailable.
    """
    if not _HAS_GENAI:
        return _FALLBACK

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return _FALLBACK

    try:
        client = genai.Client(api_key=api_key)

        prompt = f"""
You are a senior business data analyst.

Write a clean executive report.

STRICT RULES:
- No markdown symbols
- No technical variable names
- No recommendations
- Use simple business language
- Keep sentences short

FORMAT:

KEY INSIGHTS:
- sentence
- sentence
- sentence

TRENDS:
- sentence
- sentence
- sentence

RISKS:
- sentence
- sentence
- sentence

DATA SUMMARY:
Revenue: {data.get("ca")}
Transactions: {data.get("transactions")}
Zones: {data.get("zones")}
Visits: {data.get("visits")}
Pharmacies: {data.get("pharmacies")}

Dashboard: {title}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text

    except Exception:
        # Any failure (network, auth, rate-limit…) → fallback silently
        return _FALLBACK
