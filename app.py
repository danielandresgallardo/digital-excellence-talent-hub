import os, json, asyncio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

# Local Imports
from ai.jd_agent import run_jd_agent, get_client
from ai.cv_agent import score_single_candidate
from db.db_functions import get_closest_candidates, get_all_candidates

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

# --- STEP 2: DISCOVERY CHAT ---
@app.route('/api/chat-discovery', methods=['POST'])
def chat_discovery():
    data = request.json
    history = data.get('history', [])
    jd_text = data.get('job_description')
    
    prompt = f"""
    You are a Senior Recruitment Consultant at BMW. You are talking to a Hiring Manager.
    Your goal is precision. A vague JD leads to poor talent matches. 
    
    CRITICAL: For Internships and Junior roles, "Python" is not enough. You must know the 
    application domain (e.g., Computer Vision, DevOps, Web, or Data Science).

    INPUT DATA: {jd_text}
    HISTORY: {history}

    DIAGNOSTIC CRITERIA:
    1. ROLE TYPE: Technical/Specialized, Intern/Junior, or General/Support?
    2. SUFFICIENCY: Is there enough context to generate a 200-word *specific* summary?
       - BAD (Generic): "Intern will code in Python and help the team."
       - GOOD (Specific): "Intern will use Python/OpenCV to optimize battery thermal simulations."

    DECISION LOGIC:
    - If the input is generic (e.g., just 'Python Intern' + 'Munich'), you MUST ask for the 
      specific project focus or department goals.
    - For Interns: Do not return 'READY' until you have at least one specific project 
      context or a sub-technology (e.g. PyTorch, Django, AWS).
    - For Technical/Senior roles: Maintain the high bar for tech stack and seniority.
    - If the user provides a specific project/domain or pushes back, return {{"status": "READY"}}.

    IMPORTANT JSON KEYS:
    - Use "status": "CHAT" or "status": "READY"
    - If status is CHAT, use the key "message" for your question.
    
    CHAT RULES:
    - NO FILLER: Skip "To ensure we source..." 
    - BE NATURALLY CURIOUS: Ask like a real recruiter: "Python is a broad field—what's the main project this intern will be tackling?"
    - MAX 2-3 QUESTIONS: Don't interrogate forever, but don't accept "vague" as an answer.

    Return ONLY JSON.
    """
    
    client = get_client()
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite-preview', # Using stable Flash for discovery
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        print(f"[DEBUG] Gatekeeper Decision: {response.text}")
        return jsonify(json.loads(response.text))
    except Exception as e:
        print(f"[ERROR] Discovery: {e}")
        return jsonify({"status": "READY"})

# --- STEP 3: RANK BASED ON VERIFIED DATA ---
@app.route('/api/rank-candidates', methods=['POST'])
def rank_candidates():
    data = request.json
    verified_criteria = data.get('criteria') 
    
    if not verified_criteria:
        return jsonify({"error": "No criteria provided"}), 400

    try:
        print(f"[DEBUG] Step 2: Embedding verified criteria...")
        jd_client = get_client()
        res = jd_client.models.embed_content(
            model='gemini-embedding-001', 
            contents=verified_criteria
        )
        query_vec = res.embeddings[0].values

        print(f"[DEBUG] Querying Supabase match_candidates...")
        top_candidates = get_closest_candidates(query_vec, k=3)

        if not top_candidates:
            return jsonify({"status": "success", "candidate_scores": []})

        async def run_scoring():
            # Pass the full candidate dict to the scoring agent
            tasks = [score_single_candidate(c, verified_criteria) for c in top_candidates]
            return await asyncio.gather(*tasks)
        
        results_raw = asyncio.run(run_scoring())
        
        parsed_results = []
        for i, r in enumerate(results_raw):
            try:
                ai_analysis = json.loads(r)
                # --- NEW: MAP METADATA TO RESULTS ---
                original_meta = top_candidates[i].get('metadata', {})
                
                # Combine AI reasoning with DB metadata
                enriched_result = {
                    "full_name": original_meta.get("full_name", "Internal Candidate"),
                    "current_title": original_meta.get("current_title", "Specialist"),
                    "years_experience": original_meta.get("years_experience", "N/A"),
                    "location": original_meta.get("location") or "Munich (Main)",
                    "skills": original_meta.get("skills", [])[:5], # Top 5
                    "fit_score": ai_analysis.get("fit_score", 0),
                    "tradeoff_reasoning": ai_analysis.get("tradeoff_reasoning", "No reasoning provided.")
                }
                parsed_results.append(enriched_result)
            except Exception as e:
                print(f"[DEBUG] Result parsing error: {e}")

        parsed_results.sort(key=lambda x: x.get('fit_score', 0), reverse=True)
        return jsonify({"status": "success", "candidate_scores": parsed_results})

    except Exception as e:
        print(f"[DEBUG] Step 2 Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/list-candidates', methods=['GET'])
def list_candidates():
    return get_all_candidates()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)