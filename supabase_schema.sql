create table public.candidates (
  id bigint generated always as identity not null,
  resume_text text null,
  metadata jsonb null,
  embedding extensions.vector null,
  full_name text null,
  current_title text null,
  years_experience bigint null,
  skills text[] null,
  summary text null,
  source text null,
  location text null,
  constraint candidates_pkey primary key (id)
) TABLESPACE pg_default;



create or replace function match_candidates (
  query_embedding extensions.vector(512),
  match_count int
)
returns table(
  id bigint,
  resume_text text,
  metadata jsonb
)
language sql
as $$
  select id, resume_text, metadata
  from candidates
  order by candidates.embedding <=> query_embedding
  limit least(match_count, 200);
$$;

