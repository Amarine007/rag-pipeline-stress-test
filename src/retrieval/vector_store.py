"""A thin, direct wrapper over FAISS.

FAISS is used as a bare index -- `IndexFlatIP` over L2-normalized vectors, which
makes inner product equal cosine similarity. There is no framework abstraction
between this code and the library, per the project's constraints.

Exact (flat) search is the default on purpose. An approximate index would
introduce recall loss of its own, which would confound experiments whose entire
subject is retrieval recall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.chunking.base import Chunk


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk and the similarity that retrieved it."""

    chunk: Chunk
    score: float
    rank: int
    """1-indexed position in the result list -- MRR depends on this."""


class FaissVectorStore:
    """An in-memory FAISS index paired with the chunks its rows correspond to."""

    def __init__(self, dimension: int, index_type: str = "flat"):
        if index_type != "flat":
            raise ValueError(
                f"Unsupported index_type {index_type!r}. Only 'flat' (exact search) is "
                "implemented; approximate indexes would confound the retrieval metrics "
                "these experiments measure."
            )
        self.dimension = dimension
        self.index_type = index_type
        self._index = None
        self._chunks: list[Chunk] = []

    @property
    def index(self):
        if self._index is None:
            import faiss

            self._index = faiss.IndexFlatIP(self.dimension)
        return self._index

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Add chunks and their vectors. Row i of `vectors` must describe `chunks[i]`."""
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"Got {len(chunks)} chunks but {vectors.shape[0]} vectors -- these must "
                "correspond one-to-one, or every retrieval result will be mislabeled."
            )
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vectors are {vectors.shape[1]}-dimensional but the index expects "
                f"{self.dimension}."
            )
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int) -> list[SearchHit]:
        """Return the top-`k` most similar chunks, best first."""
        if len(self._chunks) == 0:
            return []

        effective_k = min(k, len(self._chunks))
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        scores, indices = self.index.search(query, effective_k)

        hits: list[SearchHit] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            # FAISS pads with -1 when it cannot fill k results.
            if idx == -1:
                continue
            hits.append(SearchHit(chunk=self._chunks[int(idx)], score=float(score), rank=rank))
        return hits
