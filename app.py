import os, json, asyncio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

# Local Imports
from ai.jd_agent import run_jd_agent, get_client
from ai.cv_agent import score_single_candidate
# We no longer need local cosine_similarity since Supabase handles it
from db.db_functions import get_closest_candidates

app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_frontend():
    return send_file('index.html')

# --- STEP 1: PARSE THE JD ---
@app.route('/api/parse-jd', methods=['POST'])
def parse_jd():
    data = request.json
    jd_text = data.get('job_description')
    try:
        print(f"[DEBUG] Step 1: Agent 1 analyzing JD...")
        criteria_str, _ = run_jd_agent(jd_text)
        return jsonify(json.loads(criteria_str))
    except Exception as e:
        print(f"[DEBUG] Step 1 Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- STEP 1.5: DISCOVERY CHAT ---
@app.route('/api/chat-discovery', methods=['POST'])
def chat_discovery():
    data = request.json
    history = data.get('history', [])  # The conversation so far
    jd_text = data.get('job_description')
    
    prompt = f"""
    You are a Senior Technical Recruiter for BMW. You're a "Gatekeeper of Quality," but you talk like a human, not a corporate script. 
    Avoid starting every response with generic phrases like "To ensure we align the right talent..." 

    INPUT DATA (Initial JD + All Chat History): 
    {jd_text}
    {history}

    STEP 1: ROLE COMPLEXITY ASSESSMENT
    - Is this a "General/Entry" role (e.g., Janitor, Intern, General Labor)? 
    - Or a "Specialized/Technical" role (e.g., Engineer, SAP Consultant, Crisis Lead)?

    STEP 2: ADAPTIVE CHECKLIST
    1. [LOCATION]: Always required.
    2. [TECHNICAL DOMAIN]: 
       - For General roles: Only ask if the specific department/plant is missing. 
       - For Technical roles: Must have specific tools/languages/certs (SAP, C++, ISO).
    3. [SENIORITY/EXPERIENCE]: 
       - For General roles: A simple "any experience level" is enough to pass.
       - For Technical roles: Must define years or level (Senior, Lead).
    4. [URGENCY]: Always required to prioritize the pipeline.

    DECISION LOGIC:
    - If the role is "General" and you have Location and Urgency, return {{"status": "READY"}}.
    - If the role is "Technical" and is missing specific stack/tools or seniority, return {{"status": "CHAT", "message": "Ask a sharp, professional, and natural follow-up question."}}.
    - If the user pushes back or says "I don't have more details," just return {{"status": "READY"}}.
    - If all items are clearly defined for the role type, return {{"status": "READY"}}.

    STRICT RULES:
    - NO REPETITIVE INTROS: Start directly with the question or a very brief, natural acknowledgment.
    - Do not demand technical certifications for non-technical roles.
    - Maintain a professional, direct, and grounded BMW-consultant tone.

    Return ONLY valid JSON.
    """
    
    from ai.jd_agent import get_client
    client = get_client()
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite-preview',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    print(f"[DEBUG] Gatekeeper Decision: {response.text}")
    return jsonify(json.loads(response.text))

# --- STEP 2: RANK BASED ON VERIFIED DATA ---
@app.route('/api/rank-candidates', methods=['POST'])
def rank_candidates():
    data = request.json
    # This matches the detailed summary from your Verify Page
    verified_criteria = data.get('criteria') 
    
    if not verified_criteria:
        return jsonify({"error": "No criteria provided"}), 400

    try:
        # 1. Get Embedding for the verified requirements
        print(f"[DEBUG] Step 2: Embedding verified criteria...")
        jd_client = get_client()
        res = jd_client.models.embed_content(
            model='gemini-embedding-001', 
            contents=verified_criteria
        )
        query_vec = res.embeddings[0].values

        # 2. Vector Search via Supabase RPC
        print(f"[DEBUG] Querying Supabase match_candidates...")
        top_candidates = get_closest_candidates(query_vec, k=3)

        if not top_candidates:
            print("[DEBUG] No candidates returned from DB.")
            return jsonify({"status": "success", "candidate_scores": []})

        # 3. Deep Scoring with Agent 2 (CV Agent)
        print(f"[DEBUG] Scoring {len(top_candidates)} candidates with Agent 2...")
        async def run_scoring():
            tasks = [score_single_candidate(c, verified_criteria) for c in top_candidates]
            return await asyncio.gather(*tasks)
        
        results_raw = asyncio.run(run_scoring())
        
        # 4. Parse AI reasoning and sort by fit_score
        parsed_results = []
        for r in results_raw:
            try:
                parsed_results.append(json.loads(r))
            except Exception as e:
                print(f"[DEBUG] CV Agent JSON parse error: {e} | Raw: {r}")

        parsed_results.sort(key=lambda x: x.get('fit_score', 0), reverse=True)

        print(f"[DEBUG] Sending {len(parsed_results)} results to frontend.")
        return jsonify({
            "status": "success", 
            "candidate_scores": parsed_results
        })

    except Exception as e:
        print(f"[DEBUG] Step 2 Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)