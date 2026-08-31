"""
Phase 4 & 5: Vector Database Indexing & Semantic Retrieval Engine
===============================================================
Indexes dense chunk vectors using FAISS (IndexFlatIP for Cosine Similarity).
Persists vector index and chunk metadata payloads to disk (`vectorstore/`).
Provides real-time semantic retrieval of top-K relevant documentation snippets.
"""

import sys, os
import pickle
from typing import List, Dict, Any
import numpy as np
import faiss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.embeddings import EmbeddingGenerator

class TechnicalRetriever:
    """FAISS Vector Store Manager & Semantic Retriever."""

    def __init__(self, embedding_generator: EmbeddingGenerator = None):
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.embedding_dim = self.embedding_generator.embedding_dim
        # IndexFlatIP calculates Inner Product (Cosine Similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks_payload: List[Dict[str, Any]] = []

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Phase 4: Generates embeddings for chunks and populates FAISS vector database."""
        if not chunks:
            print("No chunks provided to build vector index.")
            return

        print(f"Generating embeddings for {len(chunks)} text chunks...")
        texts = [c["content"] for c in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)

        # Reset index and add new vectors
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self.chunks_payload = chunks

        print(f"Phase 4 Complete: Successfully indexed {self.index.ntotal} vectors in FAISS database.")

    def save(self, vectorstore_dir: str = "vectorstore"):
        """Saves FAISS binary index and metadata pickle store to disk."""
        os.makedirs(vectorstore_dir, exist_ok=True)
        index_path = os.path.join(vectorstore_dir, "faiss_index.bin")
        meta_path = os.path.join(vectorstore_dir, "metadata.pkl")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks_payload, f)

        print(f"Vectorstore saved to disk: '{index_path}' and '{meta_path}'.")

    @classmethod
    def load(cls, vectorstore_dir: str = "vectorstore", embedding_generator: EmbeddingGenerator = None):
        """Loads persistent FAISS index and metadata store from disk."""
        index_path = os.path.join(vectorstore_dir, "faiss_index.bin")
        meta_path = os.path.join(vectorstore_dir, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Vectorstore files not found in '{vectorstore_dir}'. Build index first.")

        retriever = cls(embedding_generator=embedding_generator)
        retriever.index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            retriever.chunks_payload = pickle.load(f)

        print(f"Loaded vectorstore with {retriever.index.ntotal} indexed chunks from '{vectorstore_dir}'.")
        return retriever

    def search(self, query: str, top_k: int = 4, score_threshold: float = 0.2) -> List[Dict[str, Any]]:
        """Phase 5: Performs top-K Cosine Similarity search over FAISS database."""
        if self.index.ntotal == 0:
            return []

        # 1. Embed query string
        query_vector = self.embedding_generator.generate_query_embedding(query)

        # 2. Search FAISS index
        k_to_search = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k_to_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) >= score_threshold:
                match = self.chunks_payload[idx].copy()
                match["similarity_score"] = round(float(score), 4)
                results.append(match)

        return results