"""Retrieval metric tests, including the rank-sensitivity MRR depends on."""

from __future__ import annotations

import pytest

from src.chunking.base import Chunk
from src.evaluation.metrics import (
    aggregate_retrieval,
    is_relevant,
    score_retrieval,
)
from src.evaluation.cost import format_cost_report, price_usage
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


# -- Multi-sentence facts: the value sentence must be present whole ------

MULTI_DOC = Document(
    doc_id="doc-multi",
    text=(
        "Halcyon Index maintains a primary shard that is rebuilt on a fixed schedule. "  # 0-77
        "That rebuild runs every 36 hours. "  # 78-111
        "Operators are paged if two consecutive rebuilds are missed."  # 112-170
    ),
    fact_spans=(
        FactSpan(
            fact_id="doc-multi:rebuild_cadence",
            start=0,
            end=170,
            text=(
                "Halcyon Index maintains a primary shard that is rebuilt on a fixed schedule. "
                "That rebuild runs every 36 hours. "
                "Operators are paged if two consecutive rebuilds are missed."
            ),
            required_start=78,
            required_end=111,
            required_text="That rebuild runs every 36 hours.",
        ),
    ),
)
MULTI_DOCS_BY_ID = {MULTI_DOC.doc_id: MULTI_DOC}
MULTI_QUESTION = EvalQuestion(
    question_id="q-multi",
    question="How often does Halcyon Index rebuild its primary shard?",
    answer="every 36 hours",
    gold_doc_id="doc-multi",
    gold_fact_id="doc-multi:rebuild_cadence",
)


def _multi_chunk(start: int, end: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc-multi:{start}",
        doc_id="doc-multi",
        text=MULTI_DOC.text[start:end],
        start=start,
        end=end,
        chunk_index=0,
    )


def test_chunk_with_whole_passage_is_relevant():
    assert is_relevant(_multi_chunk(0, 170), MULTI_QUESTION, MULTI_DOCS_BY_ID)


def test_chunk_missing_the_value_sentence_is_not_relevant():
    """Setup plus elaboration clears the overlap ratio but cannot state the answer."""
    chunk = _multi_chunk(0, 105)  # cuts mid-way through the value sentence
    span = MULTI_DOC.fact_spans[0]
    assert span.overlap_chars(chunk.start, chunk.end) / (span.end - span.start) >= 0.5
    assert not is_relevant(chunk, MULTI_QUESTION, MULTI_DOCS_BY_ID)


def test_chunk_with_only_the_value_sentence_is_not_relevant():
    """The value alone is unattributable: nothing says which system it describes."""
    assert not is_relevant(_multi_chunk(78, 111), MULTI_QUESTION, MULTI_DOCS_BY_ID)


def test_chunk_with_binding_and_value_is_relevant():
    assert is_relevant(_multi_chunk(0, 111), MULTI_QUESTION, MULTI_DOCS_BY_ID)


# -- cost accounting ------------------------------------------------------


def test_price_usage_matches_published_rates():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "api_calls": 2}
    estimate = price_usage(usage, "claude-opus-5")
    assert estimate.usd == pytest.approx(30.00)  # $5 in + $25 out


def test_batch_transport_halves_the_price():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "api_calls": 2}
    assert price_usage(usage, "claude-opus-5", batched=True).usd == pytest.approx(15.00)


def test_unknown_model_raises_rather_than_costing_zero():
    with pytest.raises(ValueError, match="No price on file"):
        price_usage({"input_tokens": 10}, "claude-imaginary-9")


def test_scaling_a_pilot_projects_the_full_grid():
    pilot = price_usage(
        {"input_tokens": 10_000, "output_tokens": 5_000, "api_calls": 20}, "claude-opus-5"
    )
    full = pilot.scaled(4.0)
    assert full.usd == pytest.approx(pilot.usd * 4)
    assert full.api_calls == 80


def test_cost_report_says_nothing_was_spent_when_fully_cached():
    usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "cached_calls": 40}
    report = format_cost_report([usage], "claude-opus-5", False, 40, 160)
    assert "spent nothing" in report
    assert "Projected" not in report
