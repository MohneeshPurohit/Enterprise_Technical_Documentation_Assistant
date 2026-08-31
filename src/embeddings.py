"""
Phase 3: Vector Embeddings Module
=================================
Converts document text chunks and user queries into dense numerical vectors.
Uses SentenceTransformers ('all-MiniLM-L6-v2') producing 384-dimensional embeddings.
Vectors are L2-normalized to enable Cosine Similarity via Inner Product dot products.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """Vector Embedding Engine using Hugging Face Sentence Transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Encodes a list of text strings into an L2-normalized float32 NumPy array.
        Shape: (len(texts), 384)
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)

        # L2 Normalize vectors so Inner Product equals Cosine Similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_embeddings = embeddings / norms

        return normalized_embeddings

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Encodes a single query string into a normalized 2D NumPy vector shape (1, dim)."""
        return self.generate_embeddings([query])