import os
import asyncio
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from supabase import create_client, Client

app = Flask(__name__)
CORS(app) 

# ==========================================
# 1. INITIALIZE CLIENTS
# ==========================================
try:
    jd_client = genai.Client(api_key=os.environ.get("GEMINI_KEY_JD"))
    cv_client = genai.Client(api_key=os.environ.get("GEMINI_KEY_CV"))
except Exception as e:
    print("Warning: Gemini API keys not set.")

# Initialize Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("Warning: Supabase credentials not set.")

# ==========================================
# 2. AGENT LOGIC
# ==========================================
def run_jd_agent(job_description):
    """Agent 1: Parses JD and outputs criteria."""
    prompt = f"""
    You are an expert HR Crisis Manager. Analyze this job description and output 
    the core requirements and an urgency score (1-10).
    
    Job Description: {job_description}
    
    Return ONLY valid JSON in this format:
    {{"criteria": "Must have supply chain and crisis management experience", "urgency_score": 8}}
    """
    
    response = jd_client.models.generate_content(
        model='gemini-2.5-pro',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return response.text

async def score_single_candidate(candidate, criteria_json):
    """Agent 2: Scores candidate against criteria."""
    prompt = f"""
    Evaluate this candidate against the crisis criteria. 
    Criteria: {criteria_json}
    Candidate Profile: {candidate['resume_text']}
    
    Return ONLY valid JSON in this format:
    {{"candidate_id": "{candidate['id']}", "fit_score": 85, "tradeoff_reasoning": "High skill, but slow onboarding."}}
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, 
        lambda: cv_client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
    )
    return response.text

# ==========================================
# 3. API ENDPOINT
# ==========================================
@app.route('/api/evaluate-crisis', methods=['POST'])
def evaluate_crisis():
    data = request.json
    job_description = data.get('job_description')
    
    if not job_description:
        return jsonify({"error": "No job description provided"}), 400

    try:
        # Step 1: Get structured criteria from JD Agent
        criteria_json_str = run_jd_agent(job_description)
        criteria_dict = json.loads(criteria_json_str)
        criteria_text = criteria_dict.get("criteria", job_description)

        # Step 2: Embed the criteria to search Supabase
        embedding_response = jd_client.models.embed_content(
            model='text-embedding-004',
            contents=criteria_text
        )
        query_vector = embedding_response.embeddings[0].values

        # Step 3: Vector Search in Supabase (Get top 5)
        rpc_response = supabase.rpc('match_candidates', {
            'query_embedding': query_vector,
            'match_threshold': 0.1, # Keep low to catch broad matches initially
            'match_count': 5
        }).execute()
        
        top_candidates = rpc_response.data
        
        if not top_candidates:
             return jsonify({"error": "No candidates found in database."}), 404

        # Step 4: Asynchronous Batch Scoring with LLM
        async def run_batch_scoring():
            tasks = [score_single_candidate(c, criteria_json_str) for c in top_candidates]
            return await asyncio.gather(*tasks)
        
        batch_results_raw = asyncio.run(run_batch_scoring())
        
        # Clean up JSON strings into objects for the frontend
        batch_results = [json.loads(res) for res in batch_results_raw]
        
        # Sort by fit_score descending
        batch_results.sort(key=lambda x: x.get('fit_score', 0), reverse=True)

        return jsonify({
            "status": "success",
            "jd_analysis": criteria_dict,
            "candidate_scores": batch_results 
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)