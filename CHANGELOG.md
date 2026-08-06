# Changelog

## Unreleased

- Added a local OpenAI + pgvector RAG evaluation mode with labeled semantic cases, ranking and refusal metrics, index consistency checks, and sanitized result snapshots.
- Increased embedding storage from 16 to 256 dimensions after the same semantic suite improved from 40% to 93.3% Hit@5 and from 40% to 75% overall pass rate.
- Added a recruiter-focused project preview, concise technical highlights, and CI status to the README.

## 0.1.0 - 2026-08-05

- Added semantic source retrieval with OpenAI embeddings and pgvector.
- Added context-grounded answers with validated citations and refusal handling.
- Added a dependency-free browser interface for grounded queries.
- Added bounded, read-only operational triage with local evidence validation.
- Added deterministic providers and RAG and triage evaluation harnesses.
- Added request tracing, sanitized errors, Docker packaging, and a reproducible demo.

The offline RAG baseline measures deterministic harness behavior rather than semantic retrieval quality. The current triage fixtures measure structural and grounding invariants; labeled expected-answer metrics remain `N/A`.
