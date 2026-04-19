import os
from google import genai

client = genai.Client(api_key=os.getenv("AIzaSyCi2igSwyyl0LGXyq2yry6YQGCcrkgxijs"))

def summarize_dashboard(data, title):
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
        contents=prompt
    )

    return response.text