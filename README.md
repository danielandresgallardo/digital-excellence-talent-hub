# Digital Excellence Talent Hub - Smart HR AI Agent

The **Digital Excellence Talent Hub** is an advanced multi-agent HR tool designed to assist recruitment for high-stakes scenarios. It uses Google's Gemini models and a vector database (Supabase) to perform intelligent job description analysis, iterative candidate discovery, and precision ranking of both internal and external talent.

## 🚀 Core Features

-   **Multi-Agent Workflow**:
    -   **JD Agent**: Analyzes raw job descriptions to extract core criteria and urgency.
    -   **Discovery Agent**: Engages in a back-and-forth chat with the HR manager to refine vague JDs into high-precision technical requirements.
    -   **Ranking Agent**: Perform semantic vector search followed by detailed LLM-based scoring and tradeoff reasoning.
-   **Synthetic Data Pipeline**: A modular "Hybrid" generator that populates the database with realistic, logically consistent candidate profiles (internal/external).
-   **Semantic Search**: Powered by `pgvector` for lightning-fast candidate matching based on intent and skills rather than just keywords.

## 📂 Project Structure

```text
.
├── app.py                # Main Flask API orchestration
├── synthesis/            # Synthetic Data Generation Module
├── ai/                   # AI Agent logic (JD Analysis, CV Scoring)
├── db/                   # Database interaction layer (Supabase)
├── index.html            # Main frontend entry point
├── supabase_schema.sql   # Database schema & match_candidates RPC
└── .env                  # Environment variables (API keys)
```

## 🛠 Setup Instructions

### 1. Prerequisites
-   Python 3.10+
-   `uv` (recommended) or `pip`
-   A Supabase account with a `pgvector` enabled database.

### 2. Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_KEY_SY=your_gemini_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

### 3. Database Schema
Run the contents of [supabase_schema.sql](supabase_schema.sql) in your Supabase SQL Editor to create the `candidates` table and the necessary RPC functions.

### 4. Seed the Database
Run the synthetic data pipeline to populate your database with initial candidates:
```bash
uv run python -m synthesis.pipeline
```

### 5. Run the Application
Start the Flask backend:
```bash
python app.py
```
The app will be available at `http://localhost:5005`.

## 🧪 Development & Improvements

### Analysis
A detailed analysis of the generation pipeline architecture and potential future improvements can be found in `synthesis_analysis.md`.

### API Reference
Detailed documentation for the backend endpoints is available in `api_documentation.md`.

## 🛡 License
Internal Use Only.
