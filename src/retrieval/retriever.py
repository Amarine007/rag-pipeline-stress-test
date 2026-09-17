"""The retriever: embed a query, search the index, return ranked chunks."""

from __future__ import annotations

from src.chunking.base import Chunk
from src.config import RetrievalConfig
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import FaissVectorStore, SearchHit


class Retriever:
    """Owns an embedder and a vector store, and the index built from them."""

    def __init__(
        self,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
    ):
        self.embedder = embedder
        self.config = config or RetrievalConfig()
        self.store = FaissVectorStore(
            dimension=embedder.dimension, index_type=self.config.index_type
        )

    def index_chunks(self, chunks: list[Chunk], show_progress: bool = False) -> None:
        """Embed and index a corpus of chunks."""
        if not chunks:
            return
        vectors = self.embedder.encode([c.text for c in chunks], show_progress=show_progress)
        self.store.add(chunks, vectors)

    def retrieve(self, query: str, k: int | None = None) -> list[SearchHit]:
        """Return the top-k chunks for a query, best first."""
        effective_k = self.config.k if k is None else k
        query_vector = self.embedder.encode_one(query)
        return self.store.search(query_vector, effective_k)

    def __len__(self) -> int:
        return len(self.store)
