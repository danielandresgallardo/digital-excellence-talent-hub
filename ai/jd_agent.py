import os
from google import genai
from google.genai import types

# Shared client used for JD analysis and embedding calls.
client = genai.Client(api_key=os.environ.get("GEMINI_KEY_JD"))

def get_client():
    """Helper to initialize and return the Gemini Client using the JD key."""
    api_key = os.environ.get("GEMINI_KEY_JD")
    if not api_key:
        raise ValueError("GEMINI_KEY_JD not found in environment variables.")
    return genai.Client(api_key=api_key)

def run_jd_agent(job_description):
    # ... (client init code)
    prompt = f"""
    You are a Fact-Based Technical Analyst for BMW. 
    Your task is to synthesize the provided Job Description and Discovery Chat data into a dense, embedding-optimized summary.

    STRICT RULES:
    1. ZERO FABRICATION: Do not invent requirements, technologies, or plant details not present in the input.
    2. NO HALLUCINATION: If a detail (like years of experience) is not mentioned, do not guess. 
    3. DATA GROUNDING: Every sentence in the 'detailed_jd' must be traceable back to the USER INPUT or CHAT HISTORY.
    4. STRUCTURE: The 'detailed_jd' should be a 150-200 word technical synthesis of the PROVIDED DATA only.

    Return ONLY a JSON object with these EXACT keys:
    {{
      "job_title": "Position Name from data",
      "detailed_jd": "A comprehensive, fact-grounded technical summary of the provided input...",
      "urgency_score": 9,
      "criteria_list": [
        {{"label": "Skill from data", "weight": 8}},
        {{"label": "Requirement from data", "weight": 10}}
      ]
    }}

    USER INPUT & CHAT DATA: 
    {job_description}
    """
    
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )
    return response.text, client