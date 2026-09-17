"""Chunk data model and the chunker interface.

Chunks carry the character range they occupy in their source document. That is
what lets `src/evaluation` decide whether a retrieved chunk actually contains a
question's planted fact, independent of which chunking strategy produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.ingestion.documents import Document


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document."""

    chunk_id: str
    doc_id: str
    text: str
    start: int
    end: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


@runtime_checkable
class Chunker(Protocol):
    """Anything that turns documents into chunks.

    Implementations must produce chunks whose `start`/`end` are true offsets into
    `document.text`; the evaluation layer trusts them.
    """

    name: str

    def chunk_document(self, document: Document) -> list[Chunk]: ...


def chunk_corpus(chunker: Chunker, documents: list[Document]) -> list[Chunk]:
    """Chunk every document, preserving corpus order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunker.chunk_document(document))
    return chunks


def verify_offsets(chunks: list[Chunk], documents: list[Document]) -> None:
    """Assert every chunk's text matches the span it claims to occupy.

    An off-by-one here would silently corrupt every retrieval metric in the
    project, so this is called from the pipeline rather than left to tests.
    """
    by_id = {d.doc_id: d for d in documents}
    for chunk in chunks:
        source = by_id.get(chunk.doc_id)
        if source is None:
            raise ValueError(f"Chunk {chunk.chunk_id} references unknown document {chunk.doc_id}")
        actual = source.text[chunk.start : chunk.end]
        if actual != chunk.text:
            raise ValueError(
                f"Chunk {chunk.chunk_id} offsets are wrong.\n"
                f"  claims [{chunk.start}:{chunk.end}]\n"
                f"  expected: {actual[:80]!r}\n"
                f"  actual:   {chunk.text[:80]!r}"
            )
