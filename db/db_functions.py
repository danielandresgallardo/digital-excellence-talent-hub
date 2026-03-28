"""Supabase helper utilities.

This module provides a lightweight bootstrap for a Supabase client
and a couple of convenience helpers used by the app.

Configuration:
- SUPABASE_URL: Supabase project URL
- SUPABASE_KEY: Supabase anon or service role key

The module loads environment variables from a local `.env` if present.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
	"""Return a cached Supabase client, creating it from env vars if needed.

	Raises:
		EnvironmentError: when `SUPABASE_URL` or `SUPABASE_KEY` are not set.
	"""
	global _supabase_client
	if _supabase_client is None:
		url = os.getenv("SUPABASE_URL")
		key = os.getenv("SUPABASE_KEY")
		if not url or not key:
			raise EnvironmentError(
				"SUPABASE_URL and SUPABASE_KEY must be set in the environment or .env"
			)
		_supabase_client = create_client(url, key)
	return _supabase_client

def get_closest_candidates(embedding: List[float], k = 3) -> List[Dict[str, Any]]:
	"""Query Supabase for the candidates with the closest embeddings to the input."""
	client = get_supabase_client()
	response = (
        client.rpc("match_candidates", {"query_embedding": embedding, "match_count": k})
        .execute()
    )
	return response.data if response.data else []

def get_all_candidates() -> List[Dict[str, Any]]:
    """Fetch all candidates from the database."""
    client = get_supabase_client()
    response = client.table("candidates").select("full_name", "current_title", "years_experience", "location", "skills", "source").execute()
    return response.data if response.data else []