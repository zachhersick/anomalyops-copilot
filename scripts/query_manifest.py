import argparse
from pathlib import Path

from copilot.answering.grounded import build_grounded_answer
from copilot.ingestion.manifest import load_chunk_manifest
from copilot.retrieval.search import retrieve_relevant_chunks
from copilot.schemas.answer import GroundedAnswer
from copilot.schemas.retrieval import ScoredChunk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--answer", action="store_true")
    args = parser.parse_args(argv)
    
    source_chunks = load_chunk_manifest(Path(args.manifest_path))
    scored_chunks = retrieve_relevant_chunks(args.query, source_chunks, args.top_k)
    
    if args.answer:
        grounded_answer = build_grounded_answer(args.query, scored_chunks)
        print_grounded_answer(grounded_answer)
    else:
        print_raw_results(scored_chunks)
    
    return 0


def print_raw_results(scored_chunks: list[ScoredChunk]) -> None:
    for scored_chunk in scored_chunks:
        chunk = scored_chunk.chunk
        print(f"Score: {scored_chunk.score}")
        print(f"Source: {chunk.source_path}:{chunk.start_line}-{chunk.end_line}")
        print(f"Preview: {chunk.content[:200]}")
        print()
        
        
def print_grounded_answer(grounded_answer: GroundedAnswer) -> None:
    print(f"Answer: {grounded_answer.answer}")
    print(f"Confidence: {grounded_answer.confidence:.4f}")
    
    if grounded_answer.refusal_reason is not None:
        print(f"Refusal reason: {grounded_answer.refusal_reason}")
        
    print("Citations: ")
    
    for citation in grounded_answer.citations:
        print(
            f"[{citation.citation_id}] "
            f"{citation.source_path}:{citation.start_line}-{citation.end_line}"
        )
        

if __name__ == "__main__":
    raise SystemExit(main())