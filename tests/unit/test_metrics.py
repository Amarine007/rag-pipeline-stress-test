"""Retrieval metric tests, including the rank-sensitivity MRR depends on."""

from __future__ import annotations

import pytest

from src.chunking.base import Chunk
from src.evaluation.metrics import (
    aggregate_retrieval,
    is_relevant,
    score_retrieval,
)
from src.ingestion.documents import Document, EvalQuestion, FactSpan
from src.retrieval.vector_store import SearchHit

FACT = "Halcyon Index rebuilds its primary shard every 36 hours."
PREFIX = "Filler sentence about something else. "
TEXT = PREFIX + FACT + " Trailing filler."

GOLD_DOC = Document(
    doc_id="doc-0000",
    text=TEXT,
    fact_spans=(
        FactSpan(
            fact_id="doc-0000:rebuild_cadence",
            start=len(PREFIX),
            end=len(PREFIX) + len(FACT),
            text=FACT,
        ),
    ),
)
OTHER_DOC = Document(doc_id="doc-0001", text="An unrelated system with unrelated values.")
DOCS_BY_ID = {d.doc_id: d for d in (GOLD_DOC, OTHER_DOC)}

QUESTION = EvalQuestion(
    question_id="q-0000",
    question="How often does Halcyon Index rebuild its primary shard?",
    answer="every 36 hours",
    gold_doc_id="doc-0000",
    gold_fact_id="doc-0000:rebuild_cadence",
)


def chunk(doc_id: str, start: int, end: int, index: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::{index}",
        doc_id=doc_id,
        text=DOCS_BY_ID[doc_id].text[start:end],
        start=start,
        end=end,
        chunk_index=index,
    )


GOLD_CHUNK = chunk("doc-0000", len(PREFIX), len(PREFIX) + len(FACT))
WRONG_DOC_CHUNK = chunk("doc-0001", 0, 20)
NON_OVERLAPPING_CHUNK = chunk("doc-0000", 0, len(PREFIX))


def hits(*chunks: Chunk) -> list[SearchHit]:
    return [SearchHit(chunk=c, score=0.9 - i * 0.1, rank=i + 1) for i, c in enumerate(chunks)]


def test_chunk_containing_fact_is_relevant():
    assert is_relevant(GOLD_CHUNK, QUESTION, DOCS_BY_ID)


def test_chunk_from_other_document_is_not_relevant():
    assert not is_relevant(WRONG_DOC_CHUNK, QUESTION, DOCS_BY_ID)


def test_chunk_missing_the_fact_is_not_relevant():
    assert not is_relevant(NON_OVERLAPPING_CHUNK, QUESTION, DOCS_BY_ID)


def test_partial_overlap_respects_threshold():
    """A chunk holding only a sliver of the fact should not count as answer-bearing."""
    sliver = chunk("doc-0000", len(PREFIX), len(PREFIX) + 10)
    assert not is_relevant(sliver, QUESTION, DOCS_BY_ID, min_overlap_ratio=0.5)
    assert is_relevant(sliver, QUESTION, DOCS_BY_ID, min_overlap_ratio=0.1)


def test_mrr_is_rank_sensitive():
    """The same gold chunk at rank 1 vs rank 3 must not score the same."""
    all_chunks = [GOLD_CHUNK, WRONG_DOC_CHUNK, NON_OVERLAPPING_CHUNK]

    first = score_retrieval(QUESTION, hits(GOLD_CHUNK, WRONG_DOC_CHUNK), all_chunks, DOCS_BY_ID)
    third = score_retrieval(
        QUESTION,
        hits(WRONG_DOC_CHUNK, NON_OVERLAPPING_CHUNK, GOLD_CHUNK),
        all_chunks,
        DOCS_BY_ID,
    )

    assert first.reciprocal_rank == pytest.approx(1.0)
    assert third.reciprocal_rank == pytest.approx(1 / 3)
    assert first.first_relevant_rank == 1
    assert third.first_relevant_rank == 3
    # Both retrieved the fact, so both are hits -- which is exactly why hit_rate
    # and MRR are reported separately.
    assert first.hit and third.hit


def test_miss_scores_zero_across_the_board():
    result = score_retrieval(
        QUESTION, hits(WRONG_DOC_CHUNK, NON_OVERLAPPING_CHUNK), [GOLD_CHUNK], DOCS_BY_ID
    )
    assert not result.hit
    assert result.reciprocal_rank == 0.0
    assert result.precision_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.first_relevant_rank is None


def test_precision_counts_all_retrieved():
    result = score_retrieval(
        QUESTION, hits(GOLD_CHUNK, WRONG_DOC_CHUNK), [GOLD_CHUNK], DOCS_BY_ID
    )
    assert result.precision_at_k == pytest.approx(0.5)
    assert result.recall_at_k == pytest.approx(1.0)


def test_empty_hits_do_not_divide_by_zero():
    result = score_retrieval(QUESTION, [], [GOLD_CHUNK], DOCS_BY_ID)
    assert result.precision_at_k == 0.0
    assert result.n_retrieved == 0


def test_aggregate_reports_n():
    all_chunks = [GOLD_CHUNK, WRONG_DOC_CHUNK]
    results = [
        score_retrieval(QUESTION, hits(GOLD_CHUNK), all_chunks, DOCS_BY_ID),
        score_retrieval(QUESTION, hits(WRONG_DOC_CHUNK), all_chunks, DOCS_BY_ID),
    ]
    agg = aggregate_retrieval(results)
    assert agg["n"] == 2
    assert agg["hit_rate_at_k"] == pytest.approx(0.5)
    assert agg["mrr"] == pytest.approx(0.5)


def test_aggregate_of_nothing_is_zero_not_a_crash():
    assert aggregate_retrieval([])["n"] == 0
