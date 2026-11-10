"""Hybrid Document and File Retrieval Engine with RRF and Reranking (Phase 9.2).

Combines BGE-M3 Dense Vector search, BGE-M3 Sparse lexical search, SQLite FTS5/BM25,
and native Windows Search using Reciprocal Rank Fusion (RRF), followed by BGE Reranker v2 M3.
"""

from collections import defaultdict
import logging
from pathlib import Path
from typing import Optional

from file_intelligence.embedding_engine import BGEM3EmbeddingEngine
from file_intelligence.faiss_index import FaissDenseIndex
from file_intelligence.fts_bm25 import FTSAndSparseEngine
from file_intelligence.models import SearchMode, SearchQuery, SearchResultItem
from file_intelligence.privacy_filter import PrivacyFilter
from file_intelligence.reranker_engine import BGERerankerEngine
from file_intelligence.windows_search import WindowsSearchEngine

logger = logging.getLogger("haks.file_intelligence.retriever")


class HybridRetriever:
    """Orchestrates multi-modal retrieval streams with RRF fusion and reranking."""

    def __init__(
        self,
        embedding_engine: BGEM3EmbeddingEngine,
        faiss_index: FaissDenseIndex,
        fts_engine: FTSAndSparseEngine,
        reranker_engine: Optional[BGERerankerEngine] = None,
        windows_search: Optional[WindowsSearchEngine] = None,
        privacy_filter: Optional[PrivacyFilter] = None,
        rrf_k: int = 60,
    ) -> None:
        self.embedding = embedding_engine
        self.faiss = faiss_index
        self.fts = fts_engine
        self.reranker = reranker_engine or BGERerankerEngine()
        self.win_search = windows_search or WindowsSearchEngine()
        self.privacy = privacy_filter or PrivacyFilter()
        self.rrf_k = rrf_k

    def search(self, query: SearchQuery) -> list[SearchResultItem]:
        """Execute search according to requested modality (Metadata, Lexical, Semantic, or Hybrid)."""
        mode = query.search_mode

        if mode == SearchMode.METADATA:
            return self._search_metadata(query)
        elif mode == SearchMode.LEXICAL:
            return self._search_lexical(query)
        elif mode == SearchMode.SEMANTIC:
            return self._search_semantic(query)
        elif mode == SearchMode.WINDOWS:
            return self._search_windows(query)
        else:
            return self._search_hybrid(query)

    # --------------------------------------------------------------------------
    # Individual Stream Retrievers
    # --------------------------------------------------------------------------

    def _search_metadata(self, query: SearchQuery) -> list[SearchResultItem]:
        """Search by filename and metadata via Windows Search and FTS filename match."""
        candidates = self.win_search.search_by_name(
            query.normalized_query,
            directory_scope=query.directory_scope,
            max_results=query.top_k,
        )
        return self._filter_and_format(candidates, query)

    def _search_lexical(self, query: SearchQuery) -> list[SearchResultItem]:
        """Search document text using SQLite FTS5 BM25."""
        ranked_chunks = self.fts.search_fts(query.normalized_query, top_k=query.top_k * 2)
        items = []
        for chunk_id, score in ranked_chunks:
            chunk = self.fts.get_chunk_by_id(chunk_id)
            if chunk:
                items.append(
                    SearchResultItem(
                        file_path=chunk.file_path,
                        file_name=chunk.file_name,
                        file_type=chunk.file_type,
                        chunk_id=chunk.chunk_id,
                        page_or_section=chunk.page_or_section,
                        snippet=chunk.text[:280] + ("..." if len(chunk.text) > 280 else ""),
                        score=score,
                        lexical_score=score,
                        match_source="fts",
                    )
                )
        return self._filter_and_format(items, query)[:query.top_k]

    def _search_semantic(self, query: SearchQuery) -> list[SearchResultItem]:
        """Search using BGE-M3 Dense vector retrieval + Sparse weights."""
        q_dense, q_sparse = self.embedding.encode_queries([query.normalized_query])
        dense_results = self.faiss.search(q_dense[0], top_k=query.top_k * 2)

        items = []
        for chunk_id, d_score in dense_results:
            chunk = self.fts.get_chunk_by_id(chunk_id)
            if chunk:
                items.append(
                    SearchResultItem(
                        file_path=chunk.file_path,
                        file_name=chunk.file_name,
                        file_type=chunk.file_type,
                        chunk_id=chunk.chunk_id,
                        page_or_section=chunk.page_or_section,
                        snippet=chunk.text[:280] + ("..." if len(chunk.text) > 280 else ""),
                        score=d_score,
                        dense_score=d_score,
                        match_source="dense",
                    )
                )
        return self._filter_and_format(items, query)[:query.top_k]

    def _search_windows(self, query: SearchQuery) -> list[SearchResultItem]:
        """Search using native Windows Search index."""
        candidates = self.win_search.search_by_content(
            query.normalized_query,
            directory_scope=query.directory_scope,
            max_results=query.top_k,
        )
        return self._filter_and_format(candidates, query)

    # --------------------------------------------------------------------------
    # Hybrid Retrieval (Combination B: Dense + Sparse + FTS + Windows Search + Reranker)
    # --------------------------------------------------------------------------

    def _search_hybrid(self, query: SearchQuery) -> list[SearchResultItem]:
        """Execute full hybrid retrieval pipeline: RRF fusion across all streams + BGE Reranker."""
        # 1. Compute query embeddings (dense + sparse)
        q_dense, q_sparse = self.embedding.encode_queries([query.normalized_query])
        query_dense_vec = q_dense[0]
        query_sparse_dict = q_sparse[0] if q_sparse else {}

        # Stream A: Dense FAISS (top 30)
        dense_hits = self.faiss.search(query_dense_vec, top_k=30)

        # Stream B: Sparse BGE-M3 (top 30)
        sparse_hits = self.fts.search_sparse(query_sparse_dict, top_k=30)

        # Stream C: SQLite FTS5 / BM25 (top 30)
        fts_hits = self.fts.search_fts(query.normalized_query, top_k=30)

        # Stream D: Native Windows Search (top 15)
        win_hits = self.win_search.search_by_content(
            query.normalized_query,
            directory_scope=query.directory_scope,
            max_results=15,
        )

        # 2. Reciprocal Rank Fusion (RRF)
        # RRF Score(d) = sum(1 / (k + rank_m(d)))
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_dense_scores: dict[str, float] = {}
        chunk_sparse_scores: dict[str, float] = {}
        chunk_lexical_scores: dict[str, float] = {}

        # Add Dense stream ranks
        for rank, (cid, score) in enumerate(dense_hits, 1):
            rrf_scores[cid] += 1.0 / (self.rrf_k + rank)
            chunk_dense_scores[cid] = score

        # Add Sparse stream ranks
        for rank, (cid, score) in enumerate(sparse_hits, 1):
            rrf_scores[cid] += 1.0 / (self.rrf_k + rank)
            chunk_sparse_scores[cid] = score

        # Add FTS5 ranks
        for rank, (cid, score) in enumerate(fts_hits, 1):
            rrf_scores[cid] += 1.0 / (self.rrf_k + rank)
            chunk_lexical_scores[cid] = score

        # Build candidate items from chunk DB
        candidates: list[SearchResultItem] = []
        for cid, rrf_val in rrf_scores.items():
            chunk = self.fts.get_chunk_by_id(cid)
            if not chunk:
                continue

            candidates.append(
                SearchResultItem(
                    file_path=chunk.file_path,
                    file_name=chunk.file_name,
                    file_type=chunk.file_type,
                    chunk_id=chunk.chunk_id,
                    page_or_section=chunk.page_or_section,
                    snippet=chunk.text[:280] + ("..." if len(chunk.text) > 280 else ""),
                    score=round(rrf_val, 5),
                    dense_score=chunk_dense_scores.get(cid),
                    sparse_score=chunk_sparse_scores.get(cid),
                    lexical_score=chunk_lexical_scores.get(cid),
                    match_source="rrf",
                )
            )

        # Merge Windows Search file results if not already represented
        seen_paths = {c.file_path for c in candidates}
        for rank, w_item in enumerate(win_hits, 1):
            if w_item.file_path not in seen_paths:
                w_item.score = 1.0 / (self.rrf_k + rank)
                candidates.append(w_item)
                seen_paths.add(w_item.file_path)

        # Sort by RRF score descending
        candidates.sort(key=lambda x: x.score, reverse=True)

        # 3. Security & Permission Filtering
        filtered_candidates = self._filter_and_format(candidates, query)

        # 4. BGE Reranker v2 M3 on top 10-20 candidates
        if query.rerank and filtered_candidates and len(filtered_candidates) > 1:
            reranked = self.reranker.rerank(
                query.raw_query,
                filtered_candidates[:20],
                top_n=query.top_k,
            )
            return reranked

        return filtered_candidates[:query.top_k]

    def _filter_and_format(self, items: list[SearchResultItem], query: SearchQuery) -> list[SearchResultItem]:
        """Apply directory scope, file extension filters, and privacy safety checks."""
        filtered = []
        for item in items:
            p = Path(item.file_path)
            # 1. Existence check
            if not p.exists():
                continue

            # 2. Privacy safety check
            safe, _ = self.privacy.is_path_safe_to_access(p)
            if not safe:
                continue

            # 3. Scope check
            if query.directory_scope:
                try:
                    p.resolve().relative_to(Path(query.directory_scope).resolve())
                except ValueError:
                    continue

            # 4. File extension filter
            if query.file_types:
                if p.suffix.lower() not in [ext.lower() for ext in query.file_types]:
                    continue

            # 5. Minimum score filter
            if item.score < query.min_score:
                continue

            filtered.append(item)

        return filtered

