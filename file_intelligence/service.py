"""HAKS File Intelligence Service Coordinator (Phase 9.2).

Combines BGE-M3 Dense + Sparse retrieval, SQLite FTS5/BM25, native Windows Search,
BGE Reranker v2 M3, incremental indexing, and safe Windows filesystem operations.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

from config.settings import AppConfig, get_settings
from file_intelligence.chunker import DocumentChunker
from file_intelligence.document_extractor import DocumentExtractor
from file_intelligence.embedding_engine import BGEM3EmbeddingEngine
from file_intelligence.faiss_index import FaissDenseIndex
from file_intelligence.filesystem_ops import FilesystemOperationsService
from file_intelligence.fts_bm25 import FTSAndSparseEngine
from file_intelligence.hybrid_retriever import HybridRetriever
from file_intelligence.indexing_service import IndexingService
from file_intelligence.models import (
    DiskUsageSummary,
    DocumentChunk,
    DuplicateGroup,
    FileMetadata,
    SearchMode,
    SearchQuery,
    SearchResultItem,
)
from file_intelligence.privacy_filter import PrivacyFilter
from file_intelligence.query_normalizer import QueryNormalizer
from file_intelligence.reranker_engine import BGERerankerEngine
from file_intelligence.windows_search import WindowsSearchEngine

logger = logging.getLogger("haks.file_intelligence.service")


class FileIntelligenceService:
    """Central interface for industrial file intelligence, hybrid retrieval, and safe file operations."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        use_mock_fallback: bool = False,
    ) -> None:
        self.config = config or get_settings()
        self.use_mock_fallback = use_mock_fallback

        # Subsystems
        self.privacy = PrivacyFilter(config=self.config)
        self.extractor = DocumentExtractor()
        self.chunker = DocumentChunker()
        self.normalizer = QueryNormalizer()
        self.fs_ops = FilesystemOperationsService(privacy_filter=self.privacy)
        self.windows_search = WindowsSearchEngine()

        # Storage directory
        storage_dir = self.config.paths.data_dir / "file_intelligence"
        storage_dir.mkdir(parents=True, exist_ok=True)
        db_path = storage_dir / "file_metadata.db"
        faiss_path = storage_dir / "faiss_dense.index"

        # AI Models and Vector Stores
        self.embedding = BGEM3EmbeddingEngine(use_mock_fallback=self.use_mock_fallback)
        self.reranker = BGERerankerEngine(use_mock_fallback=self.use_mock_fallback)
        self.faiss = FaissDenseIndex(index_path=faiss_path)
        self.fts = FTSAndSparseEngine(db_path=db_path)

        # High-level pipeline engines
        self.retriever = HybridRetriever(
            embedding_engine=self.embedding,
            faiss_index=self.faiss,
            fts_engine=self.fts,
            reranker_engine=self.reranker,
            windows_search=self.windows_search,
            privacy_filter=self.privacy,
        )

        self.indexer = IndexingService(
            config=self.config,
            embedding_engine=self.embedding,
            faiss_index=self.faiss,
            fts_engine=self.fts,
            privacy_filter=self.privacy,
            extractor=self.extractor,
            chunker=self.chunker,
        )

    # --------------------------------------------------------------------------
    # 1. Search APIs
    # --------------------------------------------------------------------------

    def search(
        self,
        natural_query: str,
        search_mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 10,
        rerank: bool = True,
        directory_scope: Optional[str] = None,
    ) -> list[SearchResultItem]:
        """Execute multilingual query through normalization, hybrid retrieval, and reranking."""
        query_obj = self.normalizer.normalize(natural_query, search_mode=search_mode)
        query_obj.top_k = top_k
        query_obj.rerank = rerank
        query_obj.directory_scope = directory_scope

        logger.info(
            "Executing %s search for '%s' (Normalized: '%s', Lang: %s, Scope: %s)...",
            query_obj.search_mode.value,
            query_obj.raw_query,
            query_obj.normalized_query,
            query_obj.language,
            query_obj.directory_scope,
        )

        return self.retriever.search(query_obj)

    # --------------------------------------------------------------------------
    # 2. Filesystem & Deduplication APIs
    # --------------------------------------------------------------------------

    def get_file_info(self, file_path: str | Path) -> Optional[FileMetadata]:
        return self.fs_ops.get_file_info(file_path)

    def find_duplicates(
        self,
        root_dir: str | Path,
        use_blake3_benchmark: bool = False,
    ) -> Tuple[list[DuplicateGroup], dict[str, float]]:
        return self.fs_ops.find_duplicates(root_dir, use_blake3_benchmark=use_blake3_benchmark)

    def find_large_files(
        self,
        root_dir: str | Path,
        min_size_mb: float = 50.0,
        top_n: int = 20,
    ) -> list[FileMetadata]:
        return self.fs_ops.find_large_files(root_dir, min_size_mb=min_size_mb, top_n=top_n)

    def analyze_disk_usage(self, root_dir: str | Path) -> DiskUsageSummary:
        return self.fs_ops.analyze_disk_usage(root_dir)

    def send_to_recycle_bin(self, target_path: str | Path) -> Tuple[bool, str]:
        return self.fs_ops.send_to_recycle_bin(target_path)

    def copy_file(self, source: str | Path, destination: str | Path, overwrite: bool = False) -> Tuple[bool, str]:
        return self.fs_ops.copy_file(source, destination, overwrite=overwrite)

    def move_file(self, source: str | Path, destination: str | Path, overwrite: bool = False) -> Tuple[bool, str]:
        return self.fs_ops.move_file(source, destination, overwrite=overwrite)

    def rename_file(self, target: str | Path, new_name: str) -> Tuple[bool, str]:
        return self.fs_ops.rename_file(target, new_name)

    # --------------------------------------------------------------------------
    # 3. Indexing & Maintenance APIs
    # --------------------------------------------------------------------------

    def index_directory(
        self,
        directory_path: str | Path,
        force_reindex: bool = False,
    ) -> dict[str, int]:
        return self.indexer.index_directory(directory_path, force_reindex=force_reindex)

    def unload_models(self) -> None:
        """Explicitly unload both BGE-M3 and BGE-Reranker from RAM to free memory."""
        self.embedding.unload_model()
        self.reranker.unload_model()

