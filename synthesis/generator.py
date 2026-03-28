import os
import json
from google import genai
from google.genai import types

from synthesis.config import GEMINI_KEY_SY, GENERATION_MODEL, DEFAULT_TEMPERATURE
from synthesis.models import CandidateList

class ResumeSynthesizer:
    def __init__(self):
        if not GEMINI_KEY_SY:
            raise ValueError("Missing Gemini API Key in environment.")
        self.client = genai.Client(api_key=GEMINI_KEY_SY)

    def generate_batch(self, skeletons: list) -> list:
        skeletons_json = json.dumps(skeletons, indent=2)
        schema_definition = json.dumps(CandidateList.model_json_schema(), indent=2)
        
        prompt = f"""
        You are an expert HR Data Simulator.
        I will provide a JSON array of candidate 'skeletons' with basic target properties. 
        For EACH skeleton, generate a full, highly detailed CandidateProfile matching the schema.
        
        Input Skeletons:
        {skeletons_json}

        Rules for Internal Candidate (source="internal"):
        - Fill out department, tenure_years, performance_rating (e.g., "Exceeds Expectations"), promotion_readiness (e.g., "Ready 1-2 years"), mobility_interest.
        - Set current_company, location, notice_period_days, and salary_expectation to null.
        
        Rules for External Candidate (source="external"):
        - Fill out current_company, location, notice_period_days, salary_expectation.
        - Set department, tenure_years, performance_rating, promotion_readiness, and mobility_interest to null.
        
        Return the entire result strictly as a valid JSON object matching the following JSON Schema:
        {schema_definition}
        """
        
        print(f"Sending batch of {len(skeletons)} to LLM for full profile synthesis...")
        
        try:
            response = self.client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=DEFAULT_TEMPERATURE
                ),
            )
            
            data = json.loads(response.text)
            return data.get("candidates", [])
        except Exception as e:
            print(f"Batch generation failed: {e}")
            return []
