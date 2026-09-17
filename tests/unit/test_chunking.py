"""Chunking tests.

The offset guarantees checked here are what every retrieval metric rests on: if
a chunk's claimed character range does not match its text, gold labels attach to
the wrong chunks and every number in the paper is wrong.
"""

from __future__ import annotations

import pytest

from src.chunking.base import chunk_corpus, verify_offsets
from src.chunking.strategies import FixedSizeChunker, RecursiveChunker, build_chunker
from src.ingestion.documents import Document

SAMPLE = (
    "Halcyon Index rebuilds its primary shard every 36 hours. "
    "Query latency is budgeted at 45 milliseconds for the 99th percentile. "
    "The console requires hardware token authentication. "
    "Deleted records are retained for 90 days before purging."
)


def make_doc(text: str = SAMPLE) -> Document:
    return Document(doc_id="doc-0000", text=text, topic="Halcyon Index")


@pytest.mark.parametrize("chunker_cls", [FixedSizeChunker, RecursiveChunker])
@pytest.mark.parametrize("size,overlap", [(50, 0), (100, 20), (400, 50), (10_000, 0)])
def test_offsets_match_text(chunker_cls, size, overlap):
    doc = make_doc()
    chunks = chunker_cls(chunk_size=size, chunk_overlap=overlap).chunk_document(doc)
    verify_offsets(chunks, [doc])  # raises if any offset is wrong
    assert chunks


@pytest.mark.parametrize("chunker_cls", [FixedSizeChunker, RecursiveChunker])
def test_full_coverage(chunker_cls):
    """Every character of the document must appear in at least one chunk.

    A gap would mean a planted fact could vanish from the index entirely, making
    a retrieval failure look like a ranking problem when it is really a
    chunking bug.
    """
    doc = make_doc()
    chunks = chunker_cls(chunk_size=60, chunk_overlap=10).chunk_document(doc)
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.start, chunk.end))
    assert covered == set(range(len(doc.text)))


def test_fixed_respects_chunk_size():
    chunks = FixedSizeChunker(chunk_size=50, chunk_overlap=0).chunk_document(make_doc())
    assert all(len(c.text) <= 50 for c in chunks)


def test_recursive_prefers_sentence_boundaries():
    """With room for a sentence, recursive chunking should not cut mid-sentence."""
    chunks = RecursiveChunker(chunk_size=80, chunk_overlap=0).chunk_document(make_doc())
    # Every chunk but the last should end at a sentence terminator.
    assert all(c.text.rstrip().endswith((".", "?", "!")) for c in chunks[:-1])


def test_fixed_chunker_cuts_mid_sentence():
    """The naive baseline is expected to sever sentences -- that is the point of it."""
    chunks = FixedSizeChunker(chunk_size=30, chunk_overlap=0).chunk_document(make_doc())
    assert any(not c.text.rstrip().endswith(".") for c in chunks[:-1])


def test_overlap_is_applied():
    chunks = FixedSizeChunker(chunk_size=50, chunk_overlap=20).chunk_document(make_doc())
    assert len(chunks) >= 2
    # Consecutive chunks should share their overlap region.
    assert chunks[0].text[-20:] == chunks[1].text[:20]


def test_empty_document_yields_no_chunks():
    assert FixedSizeChunker(chunk_size=100).chunk_document(make_doc("")) == []
    assert RecursiveChunker(chunk_size=100).chunk_document(make_doc("")) == []


def test_document_shorter_than_chunk_size_is_one_chunk():
    doc = make_doc("Short.")
    for chunker in (FixedSizeChunker(chunk_size=500), RecursiveChunker(chunk_size=500)):
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Short."


@pytest.mark.parametrize("chunker_cls", [FixedSizeChunker, RecursiveChunker])
def test_overlap_must_be_smaller_than_size(chunker_cls):
    """Overlap >= size would stall the window and loop forever."""
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunker_cls(chunk_size=50, chunk_overlap=50)


@pytest.mark.parametrize("chunker_cls", [FixedSizeChunker, RecursiveChunker])
def test_rejects_nonpositive_size(chunker_cls):
    with pytest.raises(ValueError, match="must be positive"):
        chunker_cls(chunk_size=0)


def test_build_chunker_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        build_chunker("semantic-magic", 400, 50)


def test_chunk_corpus_preserves_document_order():
    docs = [make_doc(), Document(doc_id="doc-0001", text="Another system entirely.")]
    chunks = chunk_corpus(FixedSizeChunker(chunk_size=40), docs)
    assert [c.doc_id for c in chunks] == sorted([c.doc_id for c in chunks])
    assert {c.doc_id for c in chunks} == {"doc-0000", "doc-0001"}


def test_verify_offsets_catches_corruption():
    doc = make_doc()
    chunks = FixedSizeChunker(chunk_size=50).chunk_document(doc)
    broken = [type(chunks[0])(**{**chunks[0].__dict__, "start": chunks[0].start + 3})]
    with pytest.raises(ValueError, match="offsets are wrong"):
        verify_offsets(broken, [doc])
