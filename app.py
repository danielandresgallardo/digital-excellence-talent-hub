import os
import asyncio
import json
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app) 

# ==========================================
# 1. INITIALIZE CLIENTS
# ==========================================
try:
    jd_client = genai.Client(api_key=os.environ.get("GEMINI_KEY_JD"))
    cv_client = genai.Client(api_key=os.environ.get("GEMINI_KEY_CV"))
    print("[DEBUG] Gemini clients initialized successfully.")
except Exception as e:
    print(f"[DEBUG] ERROR: Gemini API keys not set or invalid. Details: {e}")

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def cosine_similarity(v1, v2):
    dot_product = sum(x * y for x, y in zip(v1, v2))
    magnitude1 = math.sqrt(sum(x * x for x in v1))
    magnitude2 = math.sqrt(sum(x * x for x in v2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

# ==========================================
# 3. AGENT LOGIC
# ==========================================
def run_jd_agent(job_description):
    print(f"[DEBUG] -> JD Agent starting for description: {job_description[:50]}...")
    prompt = f"""
    You are an expert HR Crisis Manager. Analyze this job description and output 
    the core requirements and an urgency score (1-10).
    
    Job Description: {job_description}
    
    Return ONLY valid JSON in this format without markdown blocks:
    {{"criteria": "Must have supply chain and crisis management experience", "urgency_score": 8}}
    """
    
    response = jd_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    print(f"[DEBUG] <- JD Agent returned raw text: {response.text}")
    return response.text

async def score_single_candidate(candidate, criteria_json):
    print(f"[DEBUG] -> CV Agent evaluating candidate: {candidate['id']}")
    prompt = f"""
    Evaluate this candidate against the crisis criteria. 
    Criteria: {criteria_json}
    Candidate Profile: {candidate['resume_text']}
    
    Return ONLY valid JSON in this format without markdown blocks:
    {{"candidate_id": "{candidate['id']}", "fit_score": 85, "tradeoff_reasoning": "High skill, but slow onboarding."}}
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, 
        lambda: cv_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
    )
    print(f"[DEBUG] <- CV Agent returned for {candidate['id']}: {response.text}")
    return response.text

# ==========================================
# 4. FRONTEND ENDPOINT
# ==========================================
from flask import send_file

@app.route('/', methods=['GET'])
def serve_frontend():
    try:
        return send_file('index.html')
    except Exception as e:
        return f"Error loading frontend: {e}"

# ==========================================
# 5. API ENDPOINT
# ==========================================
@app.route('/api/evaluate-crisis', methods=['POST'])
def evaluate_crisis():
    print("\n[DEBUG] ======================================")
    print("[DEBUG] Incoming Request to /api/evaluate-crisis")
    
    data = request.json
    job_description = data.get('job_description')
    
    if not job_description:
        print("[DEBUG] ERROR: No job description provided in payload.")
        return jsonify({"error": "No job description provided"}), 400

    try:
        # Step 1: Load local candidates
        print("[DEBUG] Loading candidates.json...")
        with open('candidates.json', 'r') as file:
            all_candidates = json.load(file)
        print(f"[DEBUG] Successfully loaded {len(all_candidates)} candidates.")

        # Step 2: Get structured criteria
        criteria_json_str = run_jd_agent(job_description)
        try:
            criteria_dict = json.loads(criteria_json_str)
            print("[DEBUG] Successfully parsed JD Agent JSON.")
        except json.JSONDecodeError as e:
            print(f"[DEBUG] CRITICAL ERROR: JD Agent JSON parsing failed! Error: {e}")
            raise Exception("JD Agent returned invalid JSON")

        criteria_text = criteria_dict.get("criteria", job_description)

        # Step 3: Embed the JD criteria
        print("[DEBUG] Getting embeddings for JD criteria...")
        jd_embedding_response = jd_client.models.embed_content(
            model='gemini-embedding-001',
            contents=criteria_text
        )
        query_vector = jd_embedding_response.embeddings[0].values
        print(f"[DEBUG] Vector generated. Length: {len(query_vector)}")

        # Step 4: Vector Search (Local Python Math)
        print("[DEBUG] Starting local vector similarity search...")
        scored_candidates = []
        for candidate in all_candidates:
            cand_embedding = jd_client.models.embed_content(
                model='gemini-embedding-001',
                contents=candidate['resume_text']
            ).embeddings[0].values
            
            sim_score = cosine_similarity(query_vector, cand_embedding)
            print(f"[DEBUG] Similarity for {candidate['id']}: {sim_score:.4f}")
            scored_candidates.append({
                "candidate": candidate, 
                "similarity": sim_score
            })

        # Sort by similarity and grab top 3
        scored_candidates.sort(key=lambda x: x['similarity'], reverse=True)
        top_candidates = [item['candidate'] for item in scored_candidates[:3]]
        print(f"[DEBUG] Top 3 candidates selected: {[c['id'] for c in top_candidates]}")

        # Step 5: Asynchronous Batch Scoring
        print("[DEBUG] Starting async batch scoring with LLMs...")
        async def run_batch_scoring():
            tasks = [score_single_candidate(c, criteria_json_str) for c in top_candidates]
            return await asyncio.gather(*tasks)
        
        batch_results_raw = asyncio.run(run_batch_scoring())
        
        # Clean up JSON strings into objects
        print("[DEBUG] Parsing CV Agent outputs into JSON objects...")
        batch_results = []
        for res in batch_results_raw:
            try:
                batch_results.append(json.loads(res))
            except json.JSONDecodeError as e:
                print(f"[DEBUG] ERROR parsing one of the CV Agent results: {res} | Error: {e}")
                # We append a fallback object so the whole request doesn't crash if one fails
                batch_results.append({"candidate_id": "Error", "fit_score": 0, "tradeoff_reasoning": "LLM JSON parse error."})
        
        batch_results.sort(key=lambda x: x.get('fit_score', 0), reverse=True)
        
        print("[DEBUG] Pipeline complete. Sending response to frontend.")
        print("[DEBUG] ======================================\n")

        return jsonify({
            "status": "success",
            "jd_analysis": criteria_dict,
            "candidate_scores": batch_results 
        })

    except Exception as e:
        print(f"[DEBUG] CATASTROPHIC FAILURE: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5005))
    app.run(host='0.0.0.0', port=port, debug=True)