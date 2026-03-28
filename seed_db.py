import os
from google import genai
from supabase import create_client, Client

jd_client = genai.Client(api_key=os.environ.get("GEMINI_KEY_JD"))
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

# Synthetic Hackathon Data
candidates = [
    {"id": "c_001", "resume_text": "Internal employee. 8 years at BMW Leipzig plant. Expert in battery assembly. Knows crisis protocols."},
    {"id": "c_002", "resume_text": "External hire. 12 years at Tesla. Fast execution, but requires 4 weeks notice period."},
    {"id": "c_003", "resume_text": "Internal HR manager. Great people skills, no production line experience."}
]

print("Embedding and uploading candidates to Supabase...")

for c in candidates:
    # 1. Generate the vector embedding
    res = jd_client.models.embed_content(
        model='text-embedding-004',
        contents=c['resume_text']
    )
    vector = res.embeddings[0].values
    
    # 2. Insert into Supabase
    supabase.table('candidates').insert({
        "id": c["id"],
        "resume_text": c["resume_text"],
        "metadata": {"type": "synthetic_hackathon_data"},
        "embedding": vector
    }).execute()
    
    print(f"Inserted {c['id']}")

print("Done! Database is seeded.")