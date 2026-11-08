"""Local Vector Index for Semantic Memory Retrieval in HAKS.

Calibrated for resource-constrained laptop hardware (Intel i5-7200U, 8 GB RAM).
Provides fast 384-dimensional subword hashing vectorization, L2-normalized cosine
similarity matching, duplicate detection, and independent persistence for
Personal Memory and Work Memory.
"""

import json
import math
from pathlib import Path
import re
from typing import Optional

from core.logger import get_logger

logger = get_logger("memory.vector_index")

VECTOR_DIM = 384


class LocalVectorIndex:
    """Independent vector search index for a memory domain."""

    def __init__(self, index_file: Path, encryption_service: Optional[object] = None) -> None:
        self.index_file = index_file
        self.encryption = encryption_service
        # Maps doc_id -> {"vector": list[float], "text": str, "metadata": dict}
        self._entries: dict[str, dict] = {}
        self.load()

    def _hash_token(self, token: str, dim: int = VECTOR_DIM) -> int:
        """Deterministic string hash into vector bucket."""
        h = 5381
        for char in token:
            h = ((h << 5) + h) + ord(char)
        return abs(h) % dim

    def vectorize(self, text: str) -> list[float]:
        """Convert arbitrary text into an L2-normalized 384-dimensional dense feature vector."""
        vec = [0.0] * VECTOR_DIM
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = cleaned.split()

        if not tokens:
            return vec

        # Unigrams & Bigrams
        for i, token in enumerate(tokens):
            # Token unigram
            idx1 = self._hash_token(token)
            vec[idx1] += 1.0

            # Subword trigrams for robust typo & Tamil phonetic variation tolerance
            if len(token) >= 3:
                for j in range(len(token) - 2):
                    sub = token[j : j + 3]
                    s_idx = self._hash_token(sub)
                    vec[s_idx] += 0.4

            # Bigram
            if i < len(tokens) - 1:
                bigram = f"{token}_{tokens[i + 1]}"
                idx2 = self._hash_token(bigram)
                vec[idx2] += 1.5

        # L2 Normalization
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]

        return vec

    def cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity between two normalized vectors."""
        if not v1 or not v2:
            return 0.0
        return sum(a * b for a, b in zip(v1, v2))

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity score between two raw text strings (0.0 to 1.0)."""
        v1 = self.vectorize(text1)
        v2 = self.vectorize(text2)
        return max(0.0, min(1.0, self.cosine_similarity(v1, v2)))

    def add_document(self, doc_id: str, text: str, metadata: Optional[dict] = None) -> None:
        """Index a document with its vector representation."""
        vector = self.vectorize(text)
        self._entries[doc_id] = {
            "vector": vector,
            "text": text,
            "metadata": metadata or {},
        }
        self.save()

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index."""
        if doc_id in self._entries:
            del self._entries[doc_id]
            self.save()
            return True
        return False

    def search(
        self, query: str, top_k: int = 5, min_score: float = 0.15
    ) -> list[tuple[str, float, dict]]:
        """Search the index for the top-k most semantically similar documents.

        Returns:
            List of tuples: (doc_id, similarity_score, metadata)
        """
        if not self._entries or not query.strip():
            return []

        query_vec = self.vectorize(query)
        scored: list[tuple[str, float, dict]] = []

        for doc_id, entry in self._entries.items():
            score = self.cosine_similarity(query_vec, entry["vector"])
            if score >= min_score:
                scored.append((doc_id, score, entry["metadata"]))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def clear(self) -> None:
        """Clear all entries in this index."""
        self._entries.clear()
        self.save()

    def save(self) -> None:
        """Persist index to filesystem, encrypting sensitive text at rest if cipher is configured."""
        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)
            serializable = {}
            for doc_id, data in self._entries.items():
                stored_text = data["text"]
                if self.encryption and hasattr(self.encryption, "encrypt_field"):
                    stored_text = self.encryption.encrypt_field(stored_text)
                serializable[doc_id] = {
                    "vector": [round(x, 5) for x in data["vector"]],
                    "text": stored_text,
                    "metadata": data["metadata"],
                }
            self.index_file.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to save vector index to %s: %s", self.index_file.name, exc)

    def load(self) -> None:
        """Load index from filesystem, transparently decrypting text fields."""
        if not self.index_file.exists():
            self._entries = {}
            return
        try:
            raw = json.loads(self.index_file.read_text(encoding="utf-8"))
            self._entries = {}
            for doc_id, entry in raw.items():
                text = entry.get("text", "")
                if self.encryption and hasattr(self.encryption, "decrypt_field"):
                    text = self.encryption.decrypt_field(text)
                self._entries[doc_id] = {
                    "vector": entry.get("vector", []),
                    "text": text,
                    "metadata": entry.get("metadata", {}),
                }
            logger.debug("Loaded %d vectors from %s", len(self._entries), self.index_file.name)
        except Exception as exc:
            logger.warning("Could not load vector index %s: %s", self.index_file.name, exc)
            self._entries = {}

    def __len__(self) -> int:
        return len(self._entries)
