"""FAISS CPU Vector Index Manager for HAKS File Intelligence (Phase 9.2).

Stores and queries 1024-dimensional BGE-M3 dense embeddings using Inner Product (cosine similarity).
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

import faiss
import numpy as np

logger = logging.getLogger("haks.file_intelligence.faiss")


class FaissDenseIndex:
    """Manages dense FAISS vector index with persistent chunk ID mappings."""

    DIM = 1024

    def __init__(self, index_path: Optional[str | Path] = None) -> None:
        self.index_path = Path(index_path) if index_path else None
        self._index: Optional[faiss.Index] = None
        self._id_to_chunk_map: dict[int, str] = {}  # int_id -> chunk_id
        self._chunk_to_id_map: dict[str, int] = {}  # chunk_id -> int_id
        self._chunk_to_file_map: dict[str, str] = {}  # chunk_id -> file_path
        self._next_id: int = 0

        self._init_or_load()

    def _init_or_load(self) -> None:
        """Initialize new IndexFlatIP or load from disk."""
        if self.index_path and self.index_path.exists():
            try:
                self.load(self.index_path)
                return
            except Exception as exc:
                logger.error("Failed to load existing FAISS index (%s); creating fresh index.", exc)

        # Flat Inner Product index (equivalent to cosine similarity on unit-normalized vectors)
        raw_index = faiss.IndexFlatIP(self.DIM)
        self._index = faiss.IndexIDMap2(raw_index)
        self._id_to_chunk_map = {}
        self._chunk_to_id_map = {}
        self._chunk_to_file_map = {}
        self._next_id = 0

    @property
    def total_vectors(self) -> int:
        """Return total number of vectors in the index."""
        return self._index.ntotal if self._index is not None else 0

    def add_vectors(
        self,
        vectors: np.ndarray,
        chunk_ids: list[str],
        file_paths: list[str],
    ) -> int:
        """Add batch of 1024-d normalized vectors with their corresponding chunk IDs."""
        if len(vectors) == 0:
            return 0

        if vectors.shape[1] != self.DIM:
            raise ValueError(f"Expected vector dimension {self.DIM}, got {vectors.shape[1]}")

        # Ensure float32 and contiguous
        vecs = np.ascontiguousarray(vectors, dtype=np.float32)

        assigned_ids = []
        for i, chunk_id in enumerate(chunk_ids):
            # If chunk already in index, reuse ID or assign new ID
            if chunk_id in self._chunk_to_id_map:
                int_id = self._chunk_to_id_map[chunk_id]
            else:
                int_id = self._next_id
                self._next_id += 1

            self._id_to_chunk_map[int_id] = chunk_id
            self._chunk_to_id_map[chunk_id] = int_id
            self._chunk_to_file_map[chunk_id] = file_paths[i]
            assigned_ids.append(int_id)

        id_array = np.array(assigned_ids, dtype=np.int64)
        self._index.add_with_ids(vecs, id_array)
        logger.debug("Added %d vectors to FAISS index. Total now: %d", len(chunk_ids), self.total_vectors)
        return len(chunk_ids)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[Tuple[str, float]]:
        """Search nearest neighbor chunks for a normalized query vector."""
        if self.total_vectors == 0:
            return []

        # Shape (1, 1024)
        if query_vector.ndim == 1:
            q_vec = query_vector.reshape(1, -1)
        else:
            q_vec = query_vector

        q_vec = np.ascontiguousarray(q_vec, dtype=np.float32)

        k = min(top_k, self.total_vectors)
        distances, indices = self._index.search(q_vec, k)

        results: list[Tuple[str, float]] = []
        for dist, int_id in zip(distances[0], indices[0]):
            if int_id == -1 or int_id not in self._id_to_chunk_map:
                continue
            chunk_id = self._id_to_chunk_map[int_id]
            results.append((chunk_id, float(dist)))

        return results

    def remove_file(self, file_path: str) -> int:
        """Remove all chunks associated with a file path."""
        chunks_to_remove = [c for c, f in self._chunk_to_file_map.items() if f == file_path]
        if not chunks_to_remove:
            return 0

        ids_to_remove = [self._chunk_to_id_map[c] for c in chunks_to_remove if c in self._chunk_to_id_map]
        if ids_to_remove:
            id_arr = np.array(ids_to_remove, dtype=np.int64)
            self._index.remove_ids(id_arr)

        for c in chunks_to_remove:
            int_id = self._chunk_to_id_map.pop(c, None)
            if int_id is not None:
                self._id_to_chunk_map.pop(int_id, None)
            self._chunk_to_file_map.pop(c, None)

        logger.debug("Removed %d chunks for file '%s' from FAISS.", len(chunks_to_remove), file_path)
        return len(chunks_to_remove)

    def save(self, output_path: str | Path) -> None:
        """Serialize FAISS index and ID maps to disk."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(out))

        # Save metadata mapping alongside index
        map_path = out.with_suffix(".meta.json")
        meta_payload = {
            "next_id": self._next_id,
            "id_to_chunk": {str(k): v for k, v in self._id_to_chunk_map.items()},
            "chunk_to_file": self._chunk_to_file_map,
        }
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f)

        logger.info("Saved FAISS index (%d vectors) to %s", self.total_vectors, out)

    def load(self, index_path: str | Path) -> None:
        """Load serialized FAISS index and metadata mappings from disk."""
        idx_p = Path(index_path).resolve()
        map_p = idx_p.with_suffix(".meta.json")

        if not idx_p.exists():
            raise FileNotFoundError(f"FAISS index file not found: {idx_p}")

        self._index = faiss.read_index(str(idx_p))

        if map_p.exists():
            with open(map_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._next_id = meta.get("next_id", 0)
            self._id_to_chunk_map = {int(k): v for k, v in meta.get("id_to_chunk", {}).items()}
            self._chunk_to_id_map = {v: k for k, v in self._id_to_chunk_map.items()}
            self._chunk_to_file_map = meta.get("chunk_to_file", {})
        else:
            self._id_to_chunk_map = {}
            self._chunk_to_id_map = {}
            self._chunk_to_file_map = {}

        logger.info("Loaded FAISS index with %d vectors from %s", self.total_vectors, idx_p)

