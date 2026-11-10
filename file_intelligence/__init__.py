"""HAKS Industrial File Intelligence Package (Phase 9.2).

Combines BGE-M3 Dense (1024-d) + Sparse retrieval, SQLite FTS5/BM25,
native Windows Search, BGE Reranker v2 M3, and safe filesystem operations.
"""

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
    EvaluationMetrics,
    FileMetadata,
    SearchMode,
    SearchQuery,
    SearchResultItem,
)
from file_intelligence.privacy_filter import PrivacyFilter
from file_intelligence.query_normalizer import QueryNormalizer
from file_intelligence.reranker_engine import BGERerankerEngine
from file_intelligence.service import FileIntelligenceService
from file_intelligence.windows_search import WindowsSearchEngine

__all__ = [
    "BGEM3EmbeddingEngine",
    "BGERerankerEngine",
    "DiskUsageSummary",
    "DocumentChunk",
    "DocumentChunker",
    "DocumentExtractor",
    "DuplicateGroup",
    "EvaluationMetrics",
    "FaissDenseIndex",
    "FileIntelligenceService",
    "FileMetadata",
    "FilesystemOperationsService",
    "FTSAndSparseEngine",
    "HybridRetriever",
    "IndexingService",
    "PrivacyFilter",
    "QueryNormalizer",
    "SearchMode",
    "SearchQuery",
    "SearchResultItem",
    "WindowsSearchEngine",
]

