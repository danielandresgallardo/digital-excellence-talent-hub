import os, json, asyncio
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from google.genai import types

load_dotenv()

# Local Imports
from ai.jd_agent import run_jd_agent, get_client
from ai.cv_agent import score_single_candidate
from db.db_functions import get_candidate_count, get_closest_candidates, get_all_candidates

app = Flask(__name__)
CORS(app)

@app.route('/')
def serve_frontend():
    return send_file('index.html')

# --- STEP 1: PARSE THE LEADERSHIP MANDATE ---
@app.route('/api/parse-jd', methods=['POST'])
def parse_jd():
    data = request.json
    jd_text = data.get('job_description')
    try:
        print(f"\n[DEBUG] --- Step 1: Agent 1 (Executive Parser) starting ---")
        print(f"[DEBUG] Input Text Length: {len(jd_text)}")
        # Instructing the parser to look through an executive lens
        executive_context = f"EXECUTIVE SEARCH CONTEXT (Board/SVP/VP Level): {jd_text}"
        criteria_str, _ = run_jd_agent(executive_context)
        print(f"[DEBUG] Parser Output: {criteria_str[:100]}...") 
        return jsonify(json.loads(criteria_str))
    except Exception as e:
        print(f"[DEBUG] Step 1 Error: {e}")
        return jsonify({"error": str(e)}), 500

# --- STEP 2: STRATEGIC DISCOVERY CHAT ---
@app.route('/api/chat-discovery', methods=['POST'])
def chat_discovery():
    data = request.json
    history = data.get('history', [])
    jd_text = data.get('job_description')
    
    print(f"\n[DEBUG] --- Step 1.5: Executive Discovery Chat Turn ---")
    print(f"[DEBUG] History Length: {len(history)}")

    prompt = f"""
    You are a Strategic Leadership Consultant for BMW Executive HR. 
    You are advising the 'HR for Senior Executives' department on a Board/VP level appointment.
    Your goal is absolute alignment. A vague mandate leads to poor executive successions.
    
    CRITICAL: For Board, SVP, and VP roles, general experience is not enough. You must know the 
    strategic mandate (e.g., Turnaround, M&A Integration, Digital Transformation, Market Expansion).

    INPUT DATA: {jd_text}
    HISTORY: {history}

    DIAGNOSTIC CRITERIA:
    1. STRATEGIC MANDATE: Is the core business mission clearly defined?
    2. SCOPE: Is the P&L responsibility (€) or global oversight scale clear?
    3. DECISION LOGIC: Return 'READY' if sufficient context exists, 'CHAT' otherwise to ask the Consultant for these missing executive details.

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

# --- STEP 3: RANK BASED ON STRATEGIC FIT ---
@app.route('/api/rank-candidates', methods=['POST'])
def rank_candidates():
    data = request.json
    verified_criteria = data.get('criteria') 
    urgency = int(data.get('urgency', 5))

    print(f"\n[DEBUG] --- Step 2: Executive Ranking & Search ---")
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

        num_of_candidates = get_candidate_count()
        if num_of_candidates <= 30:
            num_top_candidates = num_of_candidates//2
        else:
            num_top_candidates = 15
            
        print(f"[DEBUG] Querying Supabase for Top candidates...")
        top_candidates = get_closest_candidates(query_vec, k=num_top_candidates)
        print(f"[DEBUG] Supabase returned {len(top_candidates)} raw candidates.")

        if not top_candidates:
            print("[DEBUG] No matches found in DB. Triggering external suggestion logic.")
            return jsonify({
                "status": "success", 
                "candidate_scores": [],
                "suggest_external": True,
                "recommendation_type": "CRITICAL" if urgency > 8 else "STRATEGIC"
            })

        async def run_scoring():
            print(f"[DEBUG] Deep Scoring started for {len(top_candidates)} candidates...")
            
            sem = asyncio.Semaphore(3) 

            async def bounded_scoring(c):
                async with sem:
                    await asyncio.sleep(0.5) 
                    # Point the agent to look for executive leadership
                    exec_prompt = f"EVALUATE FOR EXECUTIVE LEADERSHIP (SVP/VP/Board): {verified_criteria}"
                    return await score_single_candidate(c, exec_prompt)

            tasks = [bounded_scoring(c) for c in top_candidates]
            return await asyncio.gather(*tasks)
        
        results_raw = asyncio.run(run_scoring())
        
        parsed_results = []
        for i, r in enumerate(results_raw):
            try:
                ai_analysis = json.loads(r)
                original_meta = top_candidates[i].get('metadata', {})
                
                raw_score = ai_analysis.get("fit_score", 0)
                try:
                    clean_score = int(str(raw_score).replace('%', '').strip())
                except ValueError:
                    clean_score = 0

                # Fallback safely to 'skills' if your DB hasn't been updated to 'leadership_competencies' yet
                exec_skills = original_meta.get("leadership_competencies", original_meta.get("skills", []))[:5]

                enriched_result = {
                    "full_name": original_meta.get("full_name", "Confidential Candidate"),
                    "current_title": original_meta.get("current_title", "Senior Executive"),
                    "years_experience": original_meta.get("years_experience", "N/A"),
                    "location": original_meta.get("location") or "Munich HQ",
                    "skills": exec_skills,
                    "fit_score": clean_score,
                    "tradeoff_reasoning": ai_analysis.get("tradeoff_reasoning", "No reasoning provided.")
                }
                parsed_results.append(enriched_result)
                print(f"[DEBUG] Scored {enriched_result['full_name']}: {enriched_result['fit_score']}%")
            except Exception as e:
                print(f"[DEBUG] Result parsing error at index {i}: {e}")

        parsed_results.sort(key=lambda x: x.get('fit_score', 0), reverse=True)
        
        max_score = parsed_results[0]['fit_score'] if parsed_results else 0
    
        suggest_external = False
        recommendation_type = None

        # Raised threshold to 75% and Urgency to 8 for the Board-level standard
        if not parsed_results or max_score < 75:
            suggest_external = True
            recommendation_type = "CRITICAL" if urgency > 8 else "STRATEGIC"
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

# --- STEP 4: EXECUTIVE BRANDING AD GENERATOR ---
@app.route('/api/generate-ad', methods=['POST'])
def generate_ad():
    data = request.json
    title = data.get('title')
    jd_context = data.get('jd_context')
    
    prompt = f"""
    Act as a Senior Executive Branding Specialist for BMW Group Board of Management. 
    Convert these strategic requirements into a highly professional Executive Job Advertisement.
    
    STRUCTURE REQUIREMENTS:
    - HEADER: "About the Position"
    - BRAND HOOK: "THE BEST {title.upper()} IN THEORY - AND IN PRACTICE. SHARE YOUR PASSION."
    - THE VISION: 4-5 sentences about BMW's vision for global leadership and premium mobility.
    - YOUR MISSION: (P&L Responsibility, strategic transformation goals, stakeholder management).
    - YOUR PROFILE: (Executive track record, global mindset, intercultural leadership).
    - OUR OFFER: (Long-term impact, executive benefits, BMW leadership culture).
    - FOOTER: Standard BMW equal opportunity and selection process statement.

    DATA:
    ROLE: {title}
    CONTEXT: {jd_context}
    
    OUTPUT ONLY THE TEXT OF THE ADVERTISEMENT.
    """
    
    client = get_client()
    try:
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