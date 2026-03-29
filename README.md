# Digital Excellence Talent Hub - Smart HR AI Agent

The **Digital Excellence Talent Hub** is an advanced multi-agent HR tool designed to assist recruitment for high-stakes scenarios. It uses Google's Gemini models and a vector database (Supabase) to perform intelligent job description analysis, iterative candidate discovery, and precision ranking of both internal and external talent.

## 🚀 Core Features

- **Multi-Agent Workflow**:
  - **Discovery Agent**: Engages in a back-and-forth chat with the HR manager to refine vague JDs into high-precision technical requirements.
  - **JD Agent**: Analyzes raw job descriptions to extract core criteria and urgency.
  - **Scoring Agent**: Perform LLM-based scoring of a single candidate according to the generated criteria. These scores are used to rank candidates.
  - **Job Posting Agent**: In case there is no good fit and low urgency, this agent can be used to generate a job description suitable for posting.
  - **Filtering Using Vector Embeddings**: To avoid running the Scoring agent on each candidate, we added a filtering step that uses semantic search using vector embeddings of candidates (which are stored in the database) against the embedding of the output of the JD Agent. The top few candidates from this search are then scored using the scoring agent.
- **Synthetic Data Pipeline**: A modular "Hybrid" generator that populates the database with realistic, logically consistent candidate profiles (internal/external).
- **Semantic Search**: Powered by `pgvector` for lightning-fast candidate matching based on intent and skills rather than just keywords.

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

- Python 3.10+
- `uv` (recommended) or `pip`
- A Supabase account with a `pgvector` enabled database.

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_KEY_SY=your_gemini_api_key_1
GEMINI_KEY_CV=your_gemini_api_key_2
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
```

### 3. Database Schema

Run the contents of [supabase_schema.sql](supabase_schema.sql) in your Supabase SQL Editor to create the `candidates` table and the necessary RPC functions.

### 4. Install Dependencies

```bash
uv sync
```

### 4. Seed the Database

Run the synthetic data pipeline to populate your database with initial candidates:

```bash
uv run python -m synthesis.pipeline
```

### 5. Run the Application

Start the Flask backend:

```bash
uv run app.py
```

The app will be available at `http://localhost:5005`.

## 🧪 Development & Improvements

### Analysis

A detailed analysis of the generation pipeline architecture and potential future improvements can be found in `synthesis_analysis.md`.

### API Reference

Detailed documentation for the backend endpoints is available in `api_documentation.md`.

## 🛡 License

Internal Use Only.
