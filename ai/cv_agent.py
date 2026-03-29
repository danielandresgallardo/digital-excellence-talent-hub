import os
import asyncio
from google import genai
from google.genai import types

# Reuse a single client instance for all candidate evaluations.
client = genai.Client(api_key=os.environ.get("GEMINI_KEY_CV"))

async def score_single_candidate(candidate, criteria_json):
    print(f"[DEBUG] -> CV Agent evaluating {candidate['id']}...")
    # Constrain the model output to a strict JSON payload for downstream parsing.
    prompt = f"""
    Evaluate this candidate for the BMW crisis role.
    Criteria: {criteria_json}
    Candidate Profile: {candidate['resume_text']}
    
    Return ONLY valid JSON with these EXACT keys:
    {{
      "candidate_id": "{candidate['id']}",
      "fit_score": "score from 0 to 100 indicating overall fit for the role",
      "tradeoff_reasoning": "Explain why they fit or the risks involved."
    }}
    """
    loop = asyncio.get_event_loop()
    # Run the blocking SDK call in a thread so async callers are not blocked.
    response = await loop.run_in_executor(
        None, 
        lambda: client.models.generate_content(
            model='gemini-3.1-flash-lite-preview',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
    )
    return response.text