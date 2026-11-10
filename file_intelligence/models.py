"""Data models for HAKS Industrial File Intelligence (Phase 9.2).

Defines document chunks, file metadata, search queries, hybrid retrieval items,
duplicate groupings, storage summaries, and evaluation metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class SearchMode(str, Enum):
    """File intelligence search modalities."""

    METADATA = "metadata"      # Filename, extension, size, mtime search
    LEXICAL = "lexical"        # BM25 / SQLite FTS5 exact keyword search
    SEMANTIC = "semantic"      # BGE-M3 Dense + Sparse vector retrieval
    WINDOWS = "windows"        # Native Windows Search index query
    HYBRID = "hybrid"          # Reciprocal Rank Fusion (RRF) of all available streams + Reranker


@dataclass
class DocumentChunk:
    """A granular chunk of extracted document text with structural provenance."""

    chunk_id: str
    file_path: str
    file_name: str
    file_type: str  # pdf, docx, xlsx, pptx, txt, md, csv, html
    chunk_index: int
    text: str
    token_count: int
    char_start: int = 0
    char_end: int = 0
    page_or_section: Optional[str] = None  # e.g. "Page 3", "Slide 5", "Sheet: Q3_Budget"
    mtime: float = 0.0
    embedding_dense: Optional[list[float]] = None
    embedding_sparse: Optional[dict[str, float]] = None


@dataclass
class FileMetadata:
    """Filesystem inspection metadata."""

    path: str
    name: str
    extension: str
    size_bytes: int
    created_time: float
    modified_time: float
    sha256_hash: Optional[str] = None
    blake3_hash: Optional[str] = None
    mime_type: str = "application/octet-stream"
    is_directory: bool = False
    chunk_count: int = 0
    indexed: bool = False
    last_indexed_time: Optional[float] = None

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 2)

    @property
    def modified_iso(self) -> str:
        return datetime.fromtimestamp(self.modified_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class SearchQuery:
    """User search request parameters with normalization."""

    raw_query: str
    normalized_query: str = ""
    language: str = "en"  # en, ta, tanglish, mixed
    search_mode: SearchMode = SearchMode.HYBRID
    top_k: int = 10
    rerank: bool = True
    file_types: Optional[list[str]] = None
    directory_scope: Optional[str] = None
    min_score: float = 0.0

    def __post_init__(self):
        if not self.normalized_query:
            self.normalized_query = self.raw_query.strip().lower()


@dataclass
class SearchResultItem:
    """Ranked search result item returned from hybrid retrieval or reranking."""

    file_path: str
    file_name: str
    file_type: str
    chunk_id: Optional[str] = None
    page_or_section: Optional[str] = None
    snippet: str = ""
    score: float = 0.0
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    lexical_score: Optional[float] = None
    rerank_score: Optional[float] = None
    match_source: str = "hybrid"  # windows_search, fts, dense, sparse, rrf, reranker
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DuplicateGroup:
    """Group of identical files discovered via hash match."""

    size_bytes: int
    sha256_hash: str
    blake3_hash: Optional[str] = None
    file_paths: list[str] = field(default_factory=list)

    @property
    def duplicate_count(self) -> int:
        return max(0, len(self.file_paths) - 1)

    @property
    def wasted_bytes(self) -> int:
        return self.size_bytes * self.duplicate_count

    @property
    def wasted_mb(self) -> float:
        return round(self.wasted_bytes / (1024 * 1024), 2)


@dataclass
class DiskUsageSummary:
    """Hierarchical storage and file-type analysis."""

    root_path: str
    total_files: int
    total_directories: int
    total_size_bytes: int
    by_extension: dict[str, int] = field(default_factory=dict)         # ext -> bytes
    by_extension_counts: dict[str, int] = field(default_factory=dict)  # ext -> count
    largest_files: list[FileMetadata] = field(default_factory=list)

    @property
    def total_size_mb(self) -> float:
        return round(self.total_size_bytes / (1024 * 1024), 2)

    @property
    def total_size_gb(self) -> float:
        return round(self.total_size_bytes / (1024 * 1024 * 1024), 2)


@dataclass
class EvaluationMetrics:
    """Retrieval benchmark metrics across languages."""

    category: str  # english, tamil, tanglish
    query_count: int
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    precision_at_5: float
    avg_latency_ms: float

