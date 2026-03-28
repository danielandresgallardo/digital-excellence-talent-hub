import os
from supabase import create_client, Client
from google import genai

from synthesis.config import GEMINI_KEY_SY, SUPABASE_URL, SUPABASE_KEY, EMBEDDING_MODEL, DEFAULT_TARGET_COUNT, DEFAULT_CHUNK_SIZE
from synthesis.planner import CandidatePlanner
from synthesis.generator import ResumeSynthesizer

def compile_searchable_text(c: dict) -> str:
    """Concatenates candidate details into a rich text block for embedding."""
    parts = []
    parts.append(f"Name: {c.get('full_name')} | Title: {c.get('current_title')} | Type: {c.get('source')} | Exp: {c.get('years_experience')} years")
    parts.append(f"Skills: {', '.join(c.get('skills', []))}")
    parts.append(f"Summary: {c.get('summary', '')}")
    
    parts.append("Work History:")
    for wh in c.get('work_history', []):
        achievements = " ".join(wh.get('achievements', []))
        parts.append(f"- {wh.get('title')} at {wh.get('company')} ({wh.get('duration_years')} yrs, domain: {wh.get('domain')}): {achievements}")
    
    parts.append(f"Project Highlights: {', '.join(c.get('project_highlights', []))}")
    
    return "\n".join(parts)

def embed_and_store(candidates):
    if not SUPABASE_URL or not SUPABASE_KEY or not GEMINI_KEY_SY:
         print("Missing essential API keys or Supabase credentials.")
         return

    jd_client = genai.Client(api_key=GEMINI_KEY_SY)
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    print(f"Embedding & storing {len(candidates)} rich candidates into Supabase...")
    for c in candidates:
        try:
            full_text = compile_searchable_text(c)
            
            res = jd_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=full_text
            )
            vector = res.embeddings[0].values
            
            # Use the entire generated object as metadata
            metadata = c.copy()
            metadata["type"] = "synthetic_rich_pool"
            
            supabase.table('candidates').insert({
                "resume_text": c.get("summary", ""),
                "metadata": metadata,
                "embedding": vector
            }).execute()
            print(f"Inserted: {c.get('full_name')} [{metadata['source']}]")
        except Exception as e:
            print(f"Error inserting candidate: {e}")

if __name__ == "__main__":
    print("--- Starting Rich Synthetic Data Pipeline ---")
    planner = CandidatePlanner(target_count=DEFAULT_TARGET_COUNT)
    skeletons = planner.generate_skeletons()
    
    synthesizer = ResumeSynthesizer()
    
    chunk_size = DEFAULT_CHUNK_SIZE
    all_profiles = []
    
    for i in range(0, len(skeletons), chunk_size):
        chunk = skeletons[i:i+chunk_size]
        profiles = synthesizer.generate_batch(chunk)
        if profiles:
            all_profiles.extend(profiles)
            
    if all_profiles:
        embed_and_store(all_profiles)
        print("--- Pipeline Complete ---")
    else:
        print("Failed to generate any profiles.")
