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
DEFAULT_TARGET_COUNT = 20
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
    "Mechanical Engineer": "Engineering",
    "Electrical Engineer": "Engineering",
    "Battery Systems Engineer": "Engineering",
    "Powertrain Engineer": "Engineering",
    "HV Battery Manufacturing Engineer": "Engineering",
    "Vehicle Integration Engineer": "Engineering",
    "Embedded Software Engineer": "Engineering",
    "ADAS Engineer": "Engineering",
    "Autonomous Driving Engineer": "Engineering",
    "Validation Engineer": "Engineering",
    "Test Engineer": "Engineering",
    "Systems Engineer": "Engineering",
    "Controls Engineer": "Engineering",
    "Prototype Engineer": "Engineering",
    "Homologation Engineer": "Engineering",
    "Regulatory Engineer": "Engineering",
    "Sustainability Engineer": "Engineering",
    "Materials Engineer": "Engineering",

    "Production Operator": "Operations",
    "Assembly Technician": "Operations",
    "Manufacturing Engineer": "Operations",
    "Process Engineer": "Operations",
    "Industrial Engineer": "Operations",
    "Production Planner": "Operations",
    "Maintenance Technician": "Operations",
    "Mechatronics Technician": "Operations",
    "Tooling Engineer": "Operations",
    "Paint Shop Specialist": "Operations",
    "Body Shop Specialist": "Operations",
    "Press Shop Specialist": "Operations",
    "Shift Supervisor": "Operations",
    "Plant Operations Manager": "Operations",

    "Supply Chain Planner": "Operations",
    "Logistics Specialist": "Operations",
    "Inbound Logistics Coordinator": "Operations",
    "Warehouse Operations Manager": "Operations",
    "Procurement Specialist": "Operations",
    "Strategic Buyer": "Operations",
    "Supplier Development Manager": "Operations",
    "Demand Planner": "Operations",
    "Inventory Analyst": "Operations",

    "Plant Quality Engineer": "Quality",
    "Supplier Quality Engineer": "Quality",
    "Quality Manager": "Quality",
    "Quality Inspector": "Quality",
    "Supplier Quality Manager": "Quality",

    "Compliance Specialist": "Legal",
    "Legal Counsel": "Legal",

    "Functional Safety Engineer": "Engineering",
    "Environmental Compliance Manager": "Operations",

    "Cybersecurity Compliance Analyst": "Data",
    "Cybersecurity Engineer": "Data",
    "Software Developer": "Data",
    "Cloud Engineer": "Data",
    "Platform Engineer": "Data",
    "Data Engineer": "Data",
    "Data Scientist": "Data",
    "AI Engineer": "Data",
    "Machine Learning Engineer": "Data",
    "SAP Consultant": "Data",
    "Digital Product Manager": "Data",
    "Enterprise Architect": "Data",
    "DevOps Engineer": "Data",
    "Business Systems Analyst": "Data",

    "Automotive Designer": "Design",
    "UX Designer": "Design",
    "UI Designer": "Design",
    "Service Designer": "Design",
    "Design Researcher": "Design",
    "Clay Model Specialist": "Design",
    "Color and Materials Designer": "Design",

    "Sales Manager": "Sales",
    "Regional Sales Analyst": "Sales",
    "Dealer Network Manager": "Sales",
    "Aftersales Manager": "Sales",
    "Service Advisor": "Sales",

    "Customer Experience Manager": "Marketing",
    "Marketing Manager": "Marketing",
    "Product Marketing Manager": "Marketing",
    "CRM Specialist": "Marketing",
    "Market Intelligence Analyst": "Marketing",
    "Communications Manager": "Marketing",

    "HR Business Partner": "Human Resources",
    "Talent Acquisition Specialist": "Human Resources",
    "Learning and Development Manager": "Human Resources",

    "Finance Analyst": "Finance",
    "Controller": "Finance",
    "Risk Manager": "Finance",
    "Internal Auditor": "Finance",
    "Sustainability Reporting Analyst": "Finance",

    "Corporate Strategy Manager": "Strategy",
}

SENIORITY_EXP = {
    "Junior": (1, 3),
    "Mid-Level": (4, 7),
    "Senior": (8, 12),
    "Principal/Lead": (12, 18),
    "Executive": (15, 25)
}

LOCATIONS = [
    "New York, USA", 
    "Austin, USA", 
    "London, UK", 
    "Remote", 
    "Berlin, GER", 
    "San Francisco, USA", 
    "Singapore, SG", 
    "Munich, GER",
    "Dingolfing, GER",
    "Legenzburg, GER",
    "Landshut, GER",
    "Berlin, GER",
    "Wackersdorf, GER",
    "Eisenach, GER",
    "Ulm, GER",
    "Bonn, GER",
    "Frankfurt, GER",
    "Dusseldorf, GER",
    "Dresden, GER",
    "Dortmund, GER",
    "Essen, GER",
    "Steyr, AUT",
    "Vienna, AUT",
    "Salzburg, AUT",
    "Oxford, UK",
    "Goodwood, UK",
    "London, UK",
    "Spartanburg, USA",
    "Greer, USA",
    "Woodruff, USA",
    "Woodcliff Lake, USA",
    "Mountain View, USA",
    "San Luis Potosi, MEX",
    "Rosslyn, RSA",
    "Pretoria, RSA",
    "Shenyang, CHN",
    "Beijing, CHN",
    "Shanghai, CHN",
    "Debrecen, HUN",
]
