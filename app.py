import os, json, asyncio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# Local Imports
from ai.jd_agent import run_jd_agent
from ai.cv_agent import score_single_candidate
from ai.utils import cosine_similarity

app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_frontend():
    return send_file('index.html')

@app.route('/api/evaluate-crisis', methods=['POST'])
def evaluate_crisis():
    data = request.json
    jd_text = data.get('job_description')

    try:
        # 1. Load Data
        with open('candidates.json', 'r') as f:
            all_candidates = json.load(f)

        # 2. Analyze JD
        criteria_str, jd_client = run_jd_agent(jd_text)
        criteria_dict = json.loads(criteria_str)

        # 3. Vector Search (Using pre-baked embeddings in candidates.json)
        jd_emb = jd_client.models.embed_content(model='gemini-embedding-001', contents=criteria_dict['criteria'])
        query_vec = jd_emb.embeddings[0].values

        scored = []
        for c in all_candidates:
            sim = cosine_similarity(query_vec, c['embedding'])
            scored.append({"candidate": c, "similarity": sim})
        
        scored.sort(key=lambda x: x['similarity'], reverse=True)
        top_3 = [x['candidate'] for x in scored[:3]]

        # 4. Deep Scoring
        async def run_scoring():
            tasks = [score_single_candidate(c, criteria_str) for c in top_3]
            return await asyncio.gather(*tasks)
        
        results = asyncio.run(run_scoring())
        parsed_results = [json.loads(r) for r in results]
        parsed_results.sort(key=lambda x: x['fit_score'], reverse=True)

        return jsonify({"status": "success", "jd_analysis": criteria_dict, "candidate_scores": parsed_results})

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)