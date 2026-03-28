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
    Analyze this Job Description for a BMW plant crisis. 
    Return ONLY a JSON object with these EXACT keys:
    
    {{
      "job_title": "Short title",
      "detailed_jd": "A 2-sentence summary of the crisis role",
      "urgency_score": "score from 1-10 on how urgently this role needs to be filled",
      "criteria_list": [
        {{"label": "Criteria 1", "weight": "score from 1-10 on importance"}},
        {{"label": "Criteria 2", "weight": "score from 1-10 on importance"}}
      ]
    }}

    JD to analyze: {job_description}
    """
    
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite-preview',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )
    return response.text, client