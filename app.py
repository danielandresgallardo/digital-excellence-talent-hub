import os, json, asyncio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Local Imports
from ai.jd_agent import run_jd_agent, get_client
from ai.cv_agent import score_single_candidate
from ai.utils import cosine_similarity

app = Flask(__name__)
CORS(app)

# Helper to load local candidates
def load_local_db():
    # This finds candidates.json in the same folder as app.py
    base_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_path, 'candidates.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find candidates.json at {json_path}")
        
    with open(json_path, 'r') as f:
        return json.load(f)

@app.route('/')
def serve_frontend():
    return send_file('index.html')

# --- STEP 1: PARSE THE JD ---
@app.route('/api/parse-jd', methods=['POST'])
def parse_jd():
    data = request.json
    jd_text = data.get('job_description')
    try:
        # JD Agent returns structured JSON (Title, Detailed JD, Criteria List)
        criteria_str, _ = run_jd_agent(jd_text)
        return jsonify(json.loads(criteria_str))
    except Exception as e:
        print(f"[DEBUG] Step 1 Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- STEP 2: RANK BASED ON VERIFIED DATA ---
@app.route('/api/rank-candidates', methods=['POST'])
def rank_candidates():
    data = request.json
    # This is the "Weighted Summary" string built by your new Frontend
    verified_criteria = data.get('criteria') 
    
    if not verified_criteria:
        return jsonify({"error": "No criteria provided"}), 400

    try:
        # 1. Load Local JSON "Database"
        all_candidates = load_local_db()

        # 2. Embed the Human-Verified Criteria
        jd_client = get_client()
        res = jd_client.models.embed_content(
            model='gemini-embedding-001', 
            contents=verified_criteria
        )
        query_vec = res.embeddings[0].values

        # 3. Local Vector Search (Replacing Supabase)
        scored = []
        for c in all_candidates:
            # Assumes you ran seed_db.py and candidates.json has 'embedding' keys
            sim = cosine_similarity(query_vec, c['embedding'])
            scored.append({"candidate": c, "similarity": sim})

        # Sort by similarity and take top 3
        scored.sort(key=lambda x: x['similarity'], reverse=True)
        top_3 = [x['candidate'] for x in scored[:3]]

        # 4. Deep Scoring with Agent 2 (CV Agent)
        async def run_scoring():
            tasks = [score_single_candidate(c, verified_criteria) for c in top_3]
            return await asyncio.gather(*tasks)
        
        results_raw = asyncio.run(run_scoring())
        
        # Parse and sort by fit_score
        parsed_results = []
        for r in results_raw:
            try:
                parsed_results.append(json.loads(r))
            except:
                print(f"[DEBUG] CV Agent output parsing failed for: {r}")

        parsed_results.sort(key=lambda x: x.get('fit_score', 0), reverse=True)

        return jsonify({
            "status": "success", 
            "candidate_scores": parsed_results
        })

    except Exception as e:
        print(f"[DEBUG] Step 2 Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Using 5005 as requested
    app.run(host='0.0.0.0', port=5005, debug=True)