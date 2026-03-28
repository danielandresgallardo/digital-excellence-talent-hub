import os, json
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_KEY_JD"))

candidates = [
    {"id": "c_001", "resume_text": "Internal employee. 8 years at BMW Leipzig plant. Expert in battery assembly. Knows crisis protocols."},
    {"id": "c_002", "resume_text": "External hire. 12 years at Tesla. Fast execution, but requires 4 weeks notice period."},
    {"id": "c_003", "resume_text": "Internal HR manager. Great people skills, no production line experience."},
    {"id": "c_004", "resume_text": "Supply Chain Guru. 15 years experience. Available for immediate relocation."}
]

print("Baking embeddings into candidates.json...")
for c in candidates:
    res = client.models.embed_content(model='gemini-embedding-001', contents=c['resume_text'])
    c['embedding'] = res.embeddings[0].values
    print(f"Baked {c['id']}")

with open('candidates.json', 'w') as f:
    json.dump(candidates, f, indent=2)
print("Done!")