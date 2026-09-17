"""Local embedding generation.

Embeddings run locally through `sentence-transformers`. That is a deliberate
reproducibility choice: the same text produces the same vector on every run,
forever, at no cost -- which matters when the experiment grid re-embeds the
corpus for every chunk-size and every distractor ratio.

Vectors are L2-normalized by default so that an inner-product FAISS index
computes cosine similarity directly.
"""

from __future__ import annotations

import numpy as np

from src.config import EmbeddingConfig


class Embedder:
    """Wraps a sentence-transformers model, loading it lazily."""

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._model = None

    @property
    def model(self):
        # Lazy: importing torch is slow, and config-only code paths (tests,
        # corpus generation) should not pay for it.
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        # sentence-transformers 6 renamed this; support both so the pinned
        # range in requirements.txt keeps working either way.
        getter = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        return int(getter())

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Embed a list of texts into a (len(texts), dim) float32 matrix."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)

        if self.config.normalize:
            vectors = l2_normalize(vectors)
        return vectors

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving zero rows untouched."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # A zero vector has no direction; dividing would produce NaNs that then
    # silently poison every similarity score it touches.
    norms[norms == 0] = 1.0
    return vectors / norms
