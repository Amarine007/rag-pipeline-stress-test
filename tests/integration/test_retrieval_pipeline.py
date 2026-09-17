"""Integration tests for the retrieval half of the pipeline.

These load the real embedding model and build a real FAISS index, but make no
API calls -- generation is not exercised here. Marked `integration` so the unit
suite stays fast:

    pytest -m "not integration"     # fast
    pytest                          # everything
"""

from __future__ import annotations

import pytest

from src.chunking.base import chunk_corpus, verify_offsets
from src.chunking.strategies import build_chunker
from src.config import ChunkingConfig, CorpusConfig, EmbeddingConfig, RetrievalConfig
from src.embeddings.embedder import Embedder
from src.evaluation.metrics import aggregate_retrieval, score_retrieval
from src.ingestion.synthetic import generate_corpus
from src.retrieval.retriever import Retriever

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder():
    return Embedder(EmbeddingConfig())


@pytest.fixture(scope="module")
def corpus():
    return generate_corpus(CorpusConfig(n_relevant_docs=12, distractor_ratio=1.0, seed=101))


def build_retriever(embedder, documents, chunking: ChunkingConfig, k: int = 5):
    chunker = build_chunker(chunking.strategy, chunking.chunk_size, chunking.chunk_overlap)
    chunks = chunk_corpus(chunker, documents)
    verify_offsets(chunks, documents)
    retriever = Retriever(embedder, RetrievalConfig(k=k))
    retriever.index_chunks(chunks)
    return retriever, chunks


def test_embeddings_are_normalized(embedder):
    import numpy as np

    vectors = embedder.encode(["a test sentence", "another one"])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embeddings_are_deterministic(embedder):
    """The reproducibility claim for the retrieval half rests on this."""
    first = embedder.encode(["Halcyon Index rebuilds its primary shard every 36 hours."])
    second = embedder.encode(["Halcyon Index rebuilds its primary shard every 36 hours."])
    assert (first == second).all()


def test_index_holds_every_chunk(embedder, corpus):
    documents, _ = corpus
    retriever, chunks = build_retriever(embedder, documents, ChunkingConfig(chunk_size=400))
    assert len(retriever) == len(chunks)


def test_retrieval_returns_k_results_ranked(embedder, corpus):
    documents, questions = corpus
    retriever, _ = build_retriever(embedder, documents, ChunkingConfig(chunk_size=400), k=5)

    hits = retriever.retrieve(questions[0].question)
    assert len(hits) == 5
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    # Scores must be non-increasing, or MRR is meaningless.
    assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))


def test_k_larger_than_corpus_does_not_crash(embedder):
    documents, questions = generate_corpus(CorpusConfig(n_relevant_docs=2, seed=5))
    retriever, chunks = build_retriever(
        embedder, documents, ChunkingConfig(chunk_size=10_000), k=50
    )
    hits = retriever.retrieve(questions[0].question)
    assert len(hits) == len(chunks)


def test_retrieval_finds_planted_facts_on_a_clean_corpus(embedder, corpus):
    """Sanity floor: with sane settings the pipeline should mostly work.

    This is the control. If retrieval cannot find planted facts under favourable
    conditions, then any degradation the experiments measure would be a bug in
    the harness rather than a property of RAG.
    """
    documents, questions = corpus
    retriever, chunks = build_retriever(
        embedder, documents, ChunkingConfig(strategy="recursive", chunk_size=400), k=5
    )
    documents_by_id = {d.doc_id: d for d in documents}

    results = [
        score_retrieval(q, retriever.retrieve(q.question), chunks, documents_by_id)
        for q in questions
    ]
    aggregate = aggregate_retrieval(results)

    assert aggregate["n"] == len(questions)
    assert aggregate["hit_rate_at_k"] >= 0.75, f"retrieval floor breached: {aggregate}"
    assert aggregate["mrr"] > 0.5, f"gold chunks ranking too low: {aggregate}"


def test_tiny_chunks_degrade_retrieval(embedder, corpus):
    """The chunk-size effect should be visible, not merely hypothesized.

    Chunks far smaller than a planted fact sever it across boundaries, so no
    single chunk remains answer-bearing. This asserts the mechanism the
    chunk-size experiment is built to quantify actually exists in this corpus.
    """
    documents, questions = corpus
    documents_by_id = {d.doc_id: d for d in documents}

    def hit_rate(chunk_size: int) -> float:
        # Overlap stays at zero: a severed fact must stay severed for the
        # mechanism under test to be visible at all.
        retriever, chunks = build_retriever(
            embedder, documents, ChunkingConfig(chunk_size=chunk_size, chunk_overlap=0), k=5
        )
        results = [
            score_retrieval(q, retriever.retrieve(q.question), chunks, documents_by_id)
            for q in questions
        ]
        return aggregate_retrieval(results)["hit_rate_at_k"]

    assert hit_rate(32) < hit_rate(400)
