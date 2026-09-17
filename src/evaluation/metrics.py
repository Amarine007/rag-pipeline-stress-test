"""Retrieval metrics, computed against exact fact spans.

**Relevance.** A chunk is relevant to a question if it comes from that
question's gold document and overlaps the planted fact's character span by at
least `min_overlap_ratio` of the fact's length. Overlap is used rather than
chunk identity because chunk boundaries move every time chunk size changes;
spans do not.

**Why both recall@k and hit_rate@k.** Textbook recall@k divides retrieved
relevant items by *total* relevant items. That number is misleading in a
chunk-size sweep: a 200-character chunking might split one fact across three
chunks while an 800-character chunking puts it in one, so the denominator itself
moves with the variable under test, and the metric stops being comparable across
conditions.

`hit_rate@k` -- did at least one answer-bearing chunk make the top k -- has a
denominator of 1 per question by construction, so it stays comparable. It is
also the quantity that actually predicts whether generation can succeed. Both
are reported; hit_rate@k is the one the chunk-size results should be read from.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.chunking.base import Chunk
from src.ingestion.documents import Document, EvalQuestion
from src.retrieval.vector_store import SearchHit

DEFAULT_MIN_OVERLAP_RATIO = 0.5


def is_relevant(
    chunk: Chunk,
    question: EvalQuestion,
    documents_by_id: dict[str, Document],
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> bool:
    """True if `chunk` carries enough of the question's planted fact to answer it."""
    if chunk.doc_id != question.gold_doc_id:
        return False

    document = documents_by_id.get(question.gold_doc_id)
    if document is None:
        return False

    span = document.fact(question.gold_fact_id)
    if span is None:
        return False

    fact_length = span.end - span.start
    if fact_length <= 0:
        return False

    overlap = span.overlap_chars(chunk.start, chunk.end)
    return (overlap / fact_length) >= min_overlap_ratio


def count_relevant_chunks(
    chunks: list[Chunk],
    question: EvalQuestion,
    documents_by_id: dict[str, Document],
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> int:
    """How many chunks in the whole corpus are relevant -- the recall denominator."""
    return sum(
        1 for c in chunks if is_relevant(c, question, documents_by_id, min_overlap_ratio)
    )


@dataclass(frozen=True)
class QuestionRetrievalResult:
    """Per-question retrieval outcome. Aggregated by `aggregate_retrieval`."""

    question_id: str
    hit: bool
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    first_relevant_rank: int | None
    n_retrieved: int
    n_relevant_in_corpus: int


def score_retrieval(
    question: EvalQuestion,
    hits: list[SearchHit],
    all_chunks: list[Chunk],
    documents_by_id: dict[str, Document],
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
) -> QuestionRetrievalResult:
    """Score one question's retrieval results."""
    relevant_ranks = [
        hit.rank
        for hit in hits
        if is_relevant(hit.chunk, question, documents_by_id, min_overlap_ratio)
    ]
    n_relevant_corpus = count_relevant_chunks(
        all_chunks, question, documents_by_id, min_overlap_ratio
    )

    n_retrieved = len(hits)
    precision = len(relevant_ranks) / n_retrieved if n_retrieved else 0.0
    recall = len(relevant_ranks) / n_relevant_corpus if n_relevant_corpus else 0.0
    first_rank = min(relevant_ranks) if relevant_ranks else None
    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0

    return QuestionRetrievalResult(
        question_id=question.question_id,
        hit=bool(relevant_ranks),
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        first_relevant_rank=first_rank,
        n_retrieved=n_retrieved,
        n_relevant_in_corpus=n_relevant_corpus,
    )


def aggregate_retrieval(results: list[QuestionRetrievalResult]) -> dict[str, float]:
    """Mean metrics over the eval set. `n` is carried so results can be read with it."""
    n = len(results)
    if n == 0:
        return {"n": 0, "hit_rate_at_k": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    return {
        "n": n,
        "hit_rate_at_k": sum(r.hit for r in results) / n,
        "precision_at_k": sum(r.precision_at_k for r in results) / n,
        "recall_at_k": sum(r.recall_at_k for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
    }
