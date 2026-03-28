import os
from google import genai
from google.genai import types

# Shared client used for JD analysis and embedding calls.
client = genai.Client(api_key=os.environ.get("GEMINI_KEY_JD"))

def run_jd_agent(job_description):
    print(f"[DEBUG] -> JD Agent starting...")
    # Keep output schema strict so callers can parse JSON reliably.
    prompt = f"""
    You are an expert HR Crisis Manager. Analyze this job description and output 
    the core requirements and an urgency score (1-10).
    
    Job Description: {job_description}
    
    Return ONLY valid JSON in this format:
    {{"criteria": "Must have supply chain and crisis management experience", "urgency_score": 8}}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    # Return both raw JSON text and the initialized client for downstream use.
    return response.text, client