"""Two hand-written chunking strategies.

`FixedSizeChunker` is the naive baseline: a sliding character window that cuts
wherever it lands, sentences be damned. `RecursiveChunker` prefers to break on
structural boundaries -- paragraphs, then sentences, then words -- and only
splits mid-word as a last resort.

The chunk-size experiment compares them precisely because the naive one is
expected to sever planted facts across a boundary, and severing a fact is one of
the concrete mechanisms by which retrieval recall collapses.
"""

from __future__ import annotations

from src.chunking.base import Chunk
from src.ingestion.documents import Document

# Tried in order: paragraph, line, sentence, clause, word, then raw characters.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


class FixedSizeChunker:
    """Fixed-width sliding window over characters, ignoring text structure."""

    name = "fixed"

    def __init__(self, chunk_size: int, chunk_overlap: int = 0):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size "
                f"({chunk_size}); otherwise the window never advances."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        stride = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{index}",
                    doc_id=document.doc_id,
                    text=text[start:end],
                    start=start,
                    end=end,
                    chunk_index=index,
                    metadata={"strategy": self.name},
                )
            )
            index += 1
            if end == len(text):
                break
            start += stride
        return chunks


def _split_keep_offsets(text: str, start: int, separator: str) -> list[tuple[str, int]]:
    """Split on `separator`, keeping it attached to the preceding piece.

    Keeping the separator means concatenating the pieces reconstructs the source
    exactly, so offsets stay contiguous and verifiable.
    """
    if separator == "":
        return [(ch, start + i) for i, ch in enumerate(text)]

    pieces: list[tuple[str, int]] = []
    cursor = 0
    while True:
        found = text.find(separator, cursor)
        if found == -1:
            pieces.append((text[cursor:], start + cursor))
            break
        boundary = found + len(separator)
        pieces.append((text[cursor:boundary], start + cursor))
        cursor = boundary
    return [(t, o) for t, o in pieces if t]


def _split_recursively(
    text: str, start: int, max_size: int, separators: list[str]
) -> list[tuple[str, int]]:
    """Break text into atoms no larger than `max_size`, preferring earlier separators."""
    if len(text) <= max_size or not separators:
        return [(text, start)]

    head, tail = separators[0], separators[1:]
    pieces = _split_keep_offsets(text, start, head)

    # This separator did not divide the text; try the next one down.
    if len(pieces) <= 1:
        return _split_recursively(text, start, max_size, tail)

    atoms: list[tuple[str, int]] = []
    for piece_text, piece_start in pieces:
        if len(piece_text) <= max_size:
            atoms.append((piece_text, piece_start))
        else:
            atoms.extend(_split_recursively(piece_text, piece_start, max_size, tail))
    return atoms


class RecursiveChunker:
    """Split on the most natural boundary that fits, then greedily repack."""

    name = "recursive"

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int = 0,
        separators: list[str] | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators if separators is not None else list(DEFAULT_SEPARATORS)

    def chunk_document(self, document: Document) -> list[Chunk]:
        text = document.text
        if not text:
            return []

        atoms = _split_recursively(text, 0, self.chunk_size, self.separators)

        # Greedily merge adjacent atoms until adding the next would overflow.
        merged: list[tuple[int, int]] = []  # (start, end) in document coordinates
        current_start: int | None = None
        current_end = 0
        for atom_text, atom_start in atoms:
            atom_end = atom_start + len(atom_text)
            if current_start is None:
                current_start, current_end = atom_start, atom_end
            elif atom_end - current_start <= self.chunk_size:
                current_end = atom_end
            else:
                merged.append((current_start, current_end))
                current_start, current_end = atom_start, atom_end
        if current_start is not None:
            merged.append((current_start, current_end))

        chunks: list[Chunk] = []
        for index, (start, end) in enumerate(merged):
            # Overlap is applied by reaching backwards into the document, which
            # keeps offsets true without duplicating or re-slicing atoms.
            overlapped_start = max(0, start - self.chunk_overlap) if index > 0 else start
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::{index}",
                    doc_id=document.doc_id,
                    text=text[overlapped_start:end],
                    start=overlapped_start,
                    end=end,
                    chunk_index=index,
                    metadata={"strategy": self.name},
                )
            )
        return chunks


def build_chunker(strategy: str, chunk_size: int, chunk_overlap: int):
    """Construct a chunker from its config name."""
    strategies = {"fixed": FixedSizeChunker, "recursive": RecursiveChunker}
    if strategy not in strategies:
        raise ValueError(
            f"Unknown chunking strategy {strategy!r}. Available: {sorted(strategies)}"
        )
    return strategies[strategy](chunk_size=chunk_size, chunk_overlap=chunk_overlap)
