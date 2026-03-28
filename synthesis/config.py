import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_KEY_SY = os.environ.get("GEMINI_KEY_SY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Models
GENERATION_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# Generation Parameters
DEFAULT_TARGET_COUNT = 10
DEFAULT_CHUNK_SIZE = 5
DEFAULT_TEMPERATURE = 0.7

# Data Dictionaries
PROFESSIONS = {
    "Software Engineer": "Engineering",
    "Data Scientist": "Data",
    "Product Manager": "Product",
    "HR Specialist": "Human Resources",
    "Supply Chain Manager": "Operations",
    "Marketing Coordinator": "Marketing",
    "Financial Analyst": "Finance",
    "Operations Director": "Operations",
    "Mechanical Engineer": "Engineering",
    "Customer Support Lead": "Customer Support",
    "Legal Counsel": "Legal",
    "Sales Executive": "Sales"
}

SENIORITY_EXP = {
    "Junior": (1, 3),
    "Mid-Level": (4, 7),
    "Senior": (8, 12),
    "Principal/Lead": (12, 18),
    "Executive": (15, 25)
}

LOCATIONS = [
    "New York, NY", "Austin, TX", "London, UK", "Remote", 
    "Berlin, GER", "San Francisco, CA", "Singapore, SG", "Munich, GER"
]
