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
        print(f"\n[DEBUG] --- Step 1: Agent 1 (Parser) starting ---")
        print(f"[DEBUG] Input Text Length: {len(jd_text)}")
        criteria_str, _ = run_jd_agent(jd_text)
        print(f"[DEBUG] Parser Output: {criteria_str[:100]}...") # Print first 100 chars
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
    
    print(f"\n[DEBUG] --- Step 1.5: Discovery Chat Turn ---")
    print(f"[DEBUG] History Length: {len(history)}")

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
    3. DECISION LOGIC: Return 'READY' if sufficient, 'CHAT' otherwise.

    Return ONLY JSON with "status" and "message".
    """
    
    client = get_client()
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite-preview', 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        print(f"[DEBUG] Gatekeeper Decision Raw: {response.text}")
        return jsonify(json.loads(response.text))
    except Exception as e:
        print(f"[ERROR] Discovery: {e}")
        return jsonify({"status": "READY"})

# --- STEP 3: RANK BASED ON VERIFIED DATA ---
@app.route('/api/rank-candidates', methods=['POST'])
def rank_candidates():
    data = request.json
    verified_criteria = data.get('criteria') 
    urgency = int(data.get('urgency', 5))

    print(f"\n[DEBUG] --- Step 2: Ranking & Vector Search ---")
    print(f"[DEBUG] Urgency Level: {urgency}")

    if not verified_criteria:
        print("[DEBUG] Error: No criteria provided from frontend.")
        return jsonify({"error": "No criteria provided"}), 400

    try:
        print(f"[DEBUG] Generating Embedding for query...")
        jd_client = get_client()
        res = jd_client.models.embed_content(
            model='gemini-embedding-001', 
            contents=verified_criteria
        )
        query_vec = res.embeddings[0].values

        print(f"[DEBUG] Querying Supabase for Top 3 candidates...")
        top_candidates = get_closest_candidates(query_vec, k=3)
        print(f"[DEBUG] Supabase returned {len(top_candidates)} raw candidates.")

        if not top_candidates:
            print("[DEBUG] No matches found in DB. Triggering external suggestion logic.")
            return jsonify({
                "status": "success", 
                "candidate_scores": [],
                "suggest_external": True,
                "recommendation_type": "CRITICAL" if urgency > 7 else "STRATEGIC"
            })

        async def run_scoring():
            print(f"[DEBUG] Deep Scoring started for {len(top_candidates)} candidates...")
            tasks = [score_single_candidate(c, verified_criteria) for c in top_candidates]
            return await asyncio.gather(*tasks)
        
        results_raw = asyncio.run(run_scoring())
        
        parsed_results = []
        for i, r in enumerate(results_raw):
            try:
                ai_analysis = json.loads(r)
                original_meta = top_candidates[i].get('metadata', {})
                
                enriched_result = {
                    "full_name": original_meta.get("full_name", "Internal Candidate"),
                    "current_title": original_meta.get("current_title", "Specialist"),
                    "years_experience": original_meta.get("years_experience", "N/A"),
                    "location": original_meta.get("location") or "Munich (Main)",
                    "skills": original_meta.get("skills", [])[:5],
                    "fit_score": ai_analysis.get("fit_score", 0),
                    "tradeoff_reasoning": ai_analysis.get("tradeoff_reasoning", "No reasoning provided.")
                }
                parsed_results.append(enriched_result)
                print(f"[DEBUG] Scored {enriched_result['full_name']}: {enriched_result['fit_score']}%")
            except Exception as e:
                print(f"[DEBUG] Result parsing error at index {i}: {e}")

        parsed_results.sort(key=lambda x: x.get('fit_score', 0), reverse=True)
        
        # Calculate max score for external logic
        max_score = parsed_results[0]['fit_score'] if parsed_results else 0
    
        suggest_external = False
        recommendation_type = None

        if not parsed_results or max_score < 70:
            suggest_external = True
            recommendation_type = "CRITICAL" if urgency > 7 else "STRATEGIC"
            print(f"[DEBUG] Quality threshold not met ({max_score}%). Suggesting external ({recommendation_type}).")

        print(f"[DEBUG] Final Response: {len(parsed_results)} candidates, suggest_external={suggest_external}")
        
        return jsonify({
            "status": "success", 
            "candidate_scores": parsed_results,
            "suggest_external": suggest_external,
            "recommendation_type": recommendation_type
        })

    except Exception as e:
        print(f"[DEBUG] Step 2 Exception: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate-ad', methods=['POST'])
def generate_ad():
    data = request.json
    title = data.get('title')
    jd_context = data.get('jd_context')
    
    prompt = f"""
    Act as a Senior Employer Branding Specialist at BMW Group. 
    Convert these internal requirements into a FULL-LENGTH, high-energy LinkedIn Job Posting.
    
    STRUCTURE REQUIREMENTS:
    - HEADER: "About the job"
    - BRAND HOOK: "THE BEST {title} IN THEORY - AND IN PRACTICE. SHARE YOUR PASSION."
    - INTRO: 4-5 sentences about BMW's culture of innovation and taking ideas from the drawing board to the road.
    - SECTION: "What awaits you?" (Provide 7 detailed, sophisticated bullet points using words like 'Furthermore', 'Moreover', 'In addition').
    - SECTION: "What should you bring along?" (Detailed bullets for Education, Tech Stack, and Mindset).
    - SECTION: "What do we offer?" (List: Mentoring, Mobile work, Flexible hours, Fair compensation, Student apartments).
    - FOOTER: Standard BMW equal opportunity and selection process statement.

    DATA:
    ROLE: {title}
    CONTEXT: {jd_context}
    
    OUTPUT ONLY THE TEXT OF THE ADVERTISEMENT.
    """
    
    client = get_client()
    try:
        # Using a higher temperature (0.7) makes the writing more "creative" and less "summary-like"
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite-preview',
            contents=prompt
        )
        return jsonify({"ad_text": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/list-candidates', methods=['GET'])
def list_candidates():
    return get_all_candidates()

if __name__ == '__main__':
    print("\n[SYSTEM] RapidResolve AI Server started on http://0.0.0.0:5005")
    app.run(host='0.0.0.0', port=5005, debug=True)