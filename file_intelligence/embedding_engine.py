"""BGE-M3 Dense (1024-d) and Sparse Embedding Engine for HAKS File Intelligence (Phase 9.2).

Provides CPU-first multi-lingual embedding extraction with lazy loading, batching,
memory monitoring, auto-unloading after inactivity, and deterministic fallback.
"""

import hashlib
import logging
import os
import threading
import time
from typing import Any, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger("haks.file_intelligence.embedding")


class BGEM3EmbeddingEngine:
    """Manages BAAI/bge-m3 dense and sparse representations with memory-friendly CPU inference."""

    DENSE_DIM = 1024

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 2,
        idle_timeout_seconds: int = 300,
        use_mock_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.idle_timeout_seconds = idle_timeout_seconds
        self.use_mock_fallback = use_mock_fallback

        self._model: Optional[Any] = None
        self._lock = threading.Lock()
        self._last_used_time: float = 0.0
        self._unload_timer: Optional[threading.Timer] = None

    @property
    def is_loaded(self) -> bool:
        """Return True if the underlying model is currently in memory."""
        return self._model is not None

    def get_memory_info(self) -> dict[str, float]:
        """Return current process RAM utilization metrics."""
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        sys_mem = psutil.virtual_memory()
        return {
            "process_rss_mb": round(mem.rss / (1024 * 1024), 2),
            "system_available_mb": round(sys_mem.available / (1024 * 1024), 2),
            "system_percent_used": sys_mem.percent,
        }

    def _ensure_model_loaded(self) -> None:
        """Lazy loader: loads model into RAM only upon actual inference call."""
        with self._lock:
            # Cancel any pending inactivity timer
            if self._unload_timer:
                self._unload_timer.cancel()
                self._unload_timer = None

            if self._model is not None:
                self._last_used_time = time.time()
                return

            if self.use_mock_fallback:
                logger.debug("Operating with deterministic fallback embedding engine.")
                return

            mem_info = self.get_memory_info()
            logger.info("Loading BGE-M3 model on %s (Process RAM: %.1f MB, System Available: %.1f MB)...",
                        self.device, mem_info["process_rss_mb"], mem_info["system_available_mb"])

            try:
                from FlagEmbedding import BGEM3FlagModel
                # CPU-optimized: use_fp16=False, batch_size=2
                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=False,
                    devices=self.device,
                    batch_size=self.batch_size,
                    passage_max_length=self.max_length,
                    query_max_length=self.max_length,
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                )
                self._last_used_time = time.time()
                new_mem = self.get_memory_info()
                logger.info("BGE-M3 loaded successfully. RAM changed: %.1f MB -> %.1f MB",
                            mem_info["process_rss_mb"], new_mem["process_rss_mb"])
            except Exception as exc:
                logger.warning("Could not load native BGE-M3 model (%s). Reverting to deterministic CPU fallback.", exc)
                self.use_mock_fallback = True
                self._model = None

    def _schedule_unload_timer(self) -> None:
        """Schedule automatic model unloading after inactivity."""
        if self.idle_timeout_seconds <= 0 or self._model is None:
            return

        with self._lock:
            if self._unload_timer:
                self._unload_timer.cancel()
            self._unload_timer = threading.Timer(self.idle_timeout_seconds, self.unload_model)
            self._unload_timer.daemon = True
            self._unload_timer.start()

    def unload_model(self) -> None:
        """Explicitly purge model from RAM to reclaim system memory."""
        with self._lock:
            if self._model is not None:
                logger.info("Unloading BGE-M3 model from memory.")
                self._model = None
                import gc
                gc.collect()
            if self._unload_timer:
                self._unload_timer.cancel()
                self._unload_timer = None

    # --------------------------------------------------------------------------
    # Inference APIs (Dense + Sparse)
    # --------------------------------------------------------------------------

    def encode_documents(self, documents: list[str]) -> Tuple[np.ndarray, list[dict[str, float]]]:
        """Compute dense vectors (N, 1024) and sparse token weights for document chunks."""
        if not documents:
            return np.empty((0, self.DENSE_DIM), dtype=np.float32), []

        self._ensure_model_loaded()
        t0 = time.perf_counter()

        if self._model is not None and not self.use_mock_fallback:
            try:
                output = self._model.encode(
                    documents,
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                    return_dense=True,
                    return_sparse=True,
                )
                dense_vecs = np.array(output["dense_vecs"], dtype=np.float32)
                # Normalize dense vectors to unit L2 norm for fast inner-product search
                norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                dense_vecs = dense_vecs / norms

                sparse_weights = output.get("lexical_weights", [])
                self._schedule_unload_timer()
                return dense_vecs, sparse_weights
            except Exception as exc:
                logger.error("Error in BGE-M3 model inference: %s. Falling back.", exc)

        # Fallback deterministic projection
        dense_vecs, sparse_weights = self._fallback_encode(documents)
        self._schedule_unload_timer()
        return dense_vecs, sparse_weights

    def encode_queries(self, queries: list[str]) -> Tuple[np.ndarray, list[dict[str, float]]]:
        """Compute dense vectors (N, 1024) and sparse token weights for search queries."""
        return self.encode_documents(queries)

    def _fallback_encode(self, texts: list[str]) -> Tuple[np.ndarray, list[dict[str, float]]]:
        """Deterministic 1024-d hash projection and TF-based sparse representation."""
        dense_list = []
        sparse_list = []

        for text in texts:
            words = text.lower().split()
            # 1. Sparse token frequencies
            sparse_dict: dict[str, float] = {}
            for w in words:
                clean_w = "".join(c for c in w if c.isalnum() or ord(c) > 127)
                if len(clean_w) >= 2:
                    sparse_dict[clean_w] = sparse_dict.get(clean_w, 0.0) + 1.0

            # Normalize sparse weights
            total_w = sum(sparse_dict.values()) or 1.0
            for k in sparse_dict:
                sparse_dict[k] = round(sparse_dict[k] / total_w, 4)

            # 2. Dense 1024-d projection via multi-hash seeds
            dense = np.zeros(self.DENSE_DIM, dtype=np.float32)
            if words:
                for w in words:
                    h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
                    idx = h % self.DENSE_DIM
                    sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
                    dense[idx] += sign

                norm = np.linalg.norm(dense)
                if norm > 0:
                    dense = dense / norm
            dense_list.append(dense)
            sparse_list.append(sparse_dict)

        return np.array(dense_list, dtype=np.float32), sparse_list

