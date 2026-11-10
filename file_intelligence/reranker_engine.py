"""BGE Reranker v2 M3 Engine for HAKS File Intelligence (Phase 9.2).

Performs second-stage cross-encoder scoring of top 10-20 hybrid candidates
with lazy loading, CPU optimization, and auto-unloading after inactivity.
"""

import logging
import os
import threading
import time
from typing import Any, Optional

import psutil

from file_intelligence.models import SearchResultItem

logger = logging.getLogger("haks.file_intelligence.reranker")


class BGERerankerEngine:
    """Manages BAAI/bge-reranker-v2-m3 cross-encoder inference."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 4,
        idle_timeout_seconds: int = 180,
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
        return self._model is not None

    def _ensure_model_loaded(self) -> None:
        """Lazy loader: loads model into RAM only upon actual rerank call."""
        with self._lock:
            if self._unload_timer:
                self._unload_timer.cancel()
                self._unload_timer = None

            if self._model is not None:
                self._last_used_time = time.time()
                return

            if self.use_mock_fallback:
                return

            proc = psutil.Process(os.getpid())
            mem_start = proc.memory_info().rss / (1024 * 1024)
            logger.info("Loading BGE-Reranker-v2-m3 on %s (Process RAM: %.1f MB)...", self.device, mem_start)

            try:
                from FlagEmbedding import FlagReranker
                self._model = FlagReranker(
                    self.model_name,
                    use_fp16=False,
                    devices=self.device,
                    batch_size=self.batch_size,
                    max_length=self.max_length,
                )
                self._last_used_time = time.time()
                mem_end = proc.memory_info().rss / (1024 * 1024)
                logger.info("BGE-Reranker loaded successfully. Process RAM: %.1f MB -> %.1f MB", mem_start, mem_end)
            except Exception as exc:
                logger.warning("Could not load native BGE-Reranker (%s). Using fallback reranking.", exc)
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
        """Purge reranker from memory."""
        with self._lock:
            if self._model is not None:
                logger.info("Unloading BGE-Reranker model from memory.")
                self._model = None
                import gc
                gc.collect()
            if self._unload_timer:
                self._unload_timer.cancel()
                self._unload_timer = None

    def rerank(self, query: str, candidates: list[SearchResultItem], top_n: int = 10) -> list[SearchResultItem]:
        """Score candidate pairs (query, chunk_text) and return re-ordered top_n results."""
        if not candidates:
            return []

        # Only rerank up to 20 candidates per specification
        eval_candidates = candidates[:20]
        self._ensure_model_loaded()

        pairs = [[query, c.snippet or c.file_name] for c in eval_candidates]

        if self._model is not None and not self.use_mock_fallback:
            try:
                raw_scores = self._model.compute_score(pairs, normalize=True)
                # Compute_score returns a single float if len(pairs) == 1
                if isinstance(raw_scores, float):
                    raw_scores = [raw_scores]
                for item, score in zip(eval_candidates, raw_scores):
                    item.rerank_score = float(score)
                    item.score = float(score)
                    item.match_source = "reranker"
                eval_candidates.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
                self._schedule_unload_timer()
                return eval_candidates[:top_n]
            except Exception as exc:
                logger.error("Error computing BGE-Reranker scores: %s", exc)

        # Fallback keyword overlap reranker
        q_words = set(query.lower().split())
        for item in eval_candidates:
            text = (item.snippet or item.file_name).lower()
            overlap = sum(1 for w in q_words if w in text)
            fb_score = (overlap / (len(q_words) or 1)) * 0.5 + (item.score * 0.5)
            item.rerank_score = round(fb_score, 4)
            item.score = round(fb_score, 4)
            item.match_source = "reranker_fallback"

        eval_candidates.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)
        self._schedule_unload_timer()
        return eval_candidates[:top_n]

