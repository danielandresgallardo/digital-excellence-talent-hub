import os
import asyncio
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_KEY_CV"))

async def score_single_candidate(candidate, criteria_json):
    print(f"[DEBUG] -> CV Agent evaluating {candidate['id']}...")
    prompt = f"""
    Evaluate this candidate against the crisis criteria. 
    Criteria: {criteria_json}
    Candidate Profile: {candidate['resume_text']}
    
    Return ONLY valid JSON:
    {{"candidate_id": "{candidate['id']}", "fit_score": 85, "tradeoff_reasoning": "High skill, but slow onboarding."}}
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, 
        lambda: client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
    )
    return response.text