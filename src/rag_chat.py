"""Lightweight local Q&A over a judgment using MiniLM embeddings + FAISS.

Fully offline and extractive: the answer is the most relevant passage(s) of the
document, so there is no hallucination and no large LLM download.
"""
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def chunk_text(text: str, words_per_chunk: int = 120, overlap: int = 20) -> list[str]:
    """Split text into overlapping word windows for retrieval."""
    words = text.split()
    step = max(words_per_chunk - overlap, 1)
    chunks = [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), step)
    ]
    return [c for c in chunks if c.strip()]


class JudgmentChat:
    """Builds a FAISS index over one judgment and answers questions from it."""

    def __init__(self, text: str):
        self.chunks = chunk_text(text)
        embeddings = _embedder().encode(
            self.chunks, normalize_embeddings=True, show_progress_bar=False
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def answer(self, question: str, top_k: int = 3) -> str:
        """Return the most relevant passages joined into a contextual answer."""
        if not self.chunks:
            return "The document has no readable text to answer from."
        query = _embedder().encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )
        query = np.asarray(query, dtype="float32")
        k = min(top_k, len(self.chunks))
        _, idx = self.index.search(query, k)
        passages = [self.chunks[i] for i in idx[0]]
        return "\n\n".join(f"• {p}" for p in passages)
