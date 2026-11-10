"""Incremental Indexing Orchestrator for HAKS File Intelligence (Phase 9.2).

Coordinates document extraction, chunking, embedding generation, FAISS indexing,
SQLite FTS5 updates, hash tracking, changed-file detection, and stale-file cleanup.
"""

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Callable, Optional

from config.settings import AppConfig, get_settings
from file_intelligence.chunker import DocumentChunker
from file_intelligence.document_extractor import DocumentExtractor
from file_intelligence.embedding_engine import BGEM3EmbeddingEngine
from file_intelligence.faiss_index import FaissDenseIndex
from file_intelligence.filesystem_ops import FilesystemOperationsService
from file_intelligence.fts_bm25 import FTSAndSparseEngine
from file_intelligence.models import DocumentChunk, FileMetadata
from file_intelligence.privacy_filter import PrivacyFilter

logger = logging.getLogger("haks.file_intelligence.indexer")


class IndexingService:
    """Manages incremental file discovery, extraction, embedding, and index synchronization."""

    INDEX_SCHEMA_VERSION = "9.2.1"

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        embedding_engine: Optional[BGEM3EmbeddingEngine] = None,
        faiss_index: Optional[FaissDenseIndex] = None,
        fts_engine: Optional[FTSAndSparseEngine] = None,
        privacy_filter: Optional[PrivacyFilter] = None,
        extractor: Optional[DocumentExtractor] = None,
        chunker: Optional[DocumentChunker] = None,
    ) -> None:
        self.config = config or get_settings()
        self.storage_dir: Path = self.config.paths.data_dir / "file_intelligence"
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.storage_dir / "file_metadata.db"
        self.faiss_path = self.storage_dir / "faiss_dense.index"
        self.state_path = self.storage_dir / "indexing_state.json"

        # Initialize component engines
        self.privacy = privacy_filter or PrivacyFilter(config=self.config)
        self.extractor = extractor or DocumentExtractor()
        self.chunker = chunker or DocumentChunker()
        self.embedding = embedding_engine or BGEM3EmbeddingEngine()
        self.faiss = faiss_index or FaissDenseIndex(index_path=self.faiss_path)
        self.fts = fts_engine or FTSAndSparseEngine(db_path=self.db_path)
        self.fs_ops = FilesystemOperationsService(privacy_filter=self.privacy)

        self._lock = threading.Lock()
        self._is_indexing: bool = False
        self._init_metadata_table()

    def _init_metadata_table(self) -> None:
        """Create file tracking metadata table in SQLite."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS indexed_files (
                    path TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    modified_time REAL NOT NULL,
                    sha256_hash TEXT,
                    chunk_count INTEGER NOT NULL,
                    last_indexed REAL NOT NULL,
                    status TEXT NOT NULL
                );
                """
            )
            conn.commit()

    @property
    def is_indexing(self) -> bool:
        return self._is_indexing

    # --------------------------------------------------------------------------
    # Incremental Indexing Pipeline
    # --------------------------------------------------------------------------

    def index_directory(
        self,
        directory_path: str | Path,
        force_reindex: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> dict[str, int]:
        """Scan directory and incrementally index new or modified supported documents."""
        with self._lock:
            self._is_indexing = True

        root = Path(directory_path).resolve()
        stats = {"added": 0, "updated": 0, "deleted": 0, "unchanged": 0, "errors": 0}

        try:
            # 1. Discover all candidate files
            candidate_files = []
            for dirpath, dirnames, filenames in os.walk(str(root)):
                dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("node_modules", "__pycache__")]
                for fname in filenames:
                    fp = Path(dirpath) / fname
                    if self.privacy.is_file_indexable(fp) and self.extractor.is_supported(fp):
                        candidate_files.append(fp)

            total_candidates = len(candidate_files)
            logger.info("Found %d candidate documents to index in '%s'.", total_candidates, root)

            # 2. Get existing index state for files in this scope
            existing_records = self._get_indexed_file_records(root)
            current_paths = {str(f.resolve()) for f in candidate_files}

            # 3. Clean up deleted files
            deleted_paths = set(existing_records.keys()) - current_paths
            for d_path in deleted_paths:
                self.remove_file_from_index(d_path)
                stats["deleted"] += 1

            # 4. Process new and modified files
            for idx, fp in enumerate(candidate_files, 1):
                str_p = str(fp)
                if progress_callback:
                    progress_callback(fp.name, idx, total_candidates)

                try:
                    st = fp.stat()
                    rec = existing_records.get(str_p)

                    # Check if unchanged
                    if rec and not force_reindex:
                        if rec["modified_time"] == st.st_mtime and rec["size_bytes"] == st.st_size:
                            stats["unchanged"] += 1
                            continue

                    # Extract sections & chunks
                    chunks_for_file = []
                    chunk_idx = 0
                    for text_section, section_ref in self.extractor.extract_sections(fp):
                        section_chunks = self.chunker.chunk_section(
                            text=text_section,
                            file_path=str_p,
                            file_name=fp.name,
                            file_type=fp.suffix.lower(),
                            page_or_section=section_ref,
                            starting_chunk_index=chunk_idx,
                            mtime=st.st_mtime,
                        )
                        chunks_for_file.extend(section_chunks)
                        chunk_idx += len(section_chunks)

                    if not chunks_for_file:
                        stats["unchanged"] += 1
                        continue

                    # Generate Dense + Sparse embeddings in batches of 2
                    chunk_texts = [c.text for c in chunks_for_file]
                    dense_vecs, sparse_weights = self.embedding.encode_documents(chunk_texts)

                    for c, d_vec, s_w in zip(chunks_for_file, dense_vecs, sparse_weights):
                        c.embedding_dense = d_vec.tolist()
                        c.embedding_sparse = s_w

                    # If updating existing file, remove old chunks first
                    if rec:
                        self.faiss.remove_file(str_p)
                        self.fts.remove_file(str_p)
                        stats["updated"] += 1
                    else:
                        stats["added"] += 1

                    # Index into FAISS and FTS5
                    chunk_ids = [c.chunk_id for c in chunks_for_file]
                    file_paths = [str_p] * len(chunks_for_file)
                    self.faiss.add_vectors(dense_vecs, chunk_ids, file_paths)
                    self.fts.index_chunks(chunks_for_file)

                    # Update metadata record
                    sha = self.fs_ops.calculate_file_hash(fp, algorithm="sha256")
                    self._record_indexed_file(
                        path=str_p,
                        name=fp.name,
                        extension=fp.suffix.lower(),
                        size_bytes=st.st_size,
                        modified_time=st.st_mtime,
                        sha256_hash=sha,
                        chunk_count=len(chunks_for_file),
                    )
                except Exception as exc:
                    logger.error("Error indexing file '%s': %s", fp.name, exc)
                    stats["errors"] += 1

            # Save FAISS index
            self.faiss.save(self.faiss_path)
            self._save_state(stats)
            logger.info("Indexing completed: %s", stats)
        finally:
            with self._lock:
                self._is_indexing = False

        return stats

    def remove_file_from_index(self, file_path: str) -> bool:
        """Purge all index entries (FAISS, FTS5, metadata) for a deleted file."""
        str_p = str(Path(file_path).resolve())
        self.faiss.remove_file(str_p)
        self.fts.remove_file(str_p)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM indexed_files WHERE path = ?;", (str_p,))
            conn.commit()
        logger.info("Purged file from index: %s", str_p)
        return True

    def rebuild_index(self, directory_path: str | Path) -> dict[str, int]:
        """Wipe existing index database and rebuild from scratch."""
        logger.warning("Rebuilding entire file intelligence index for '%s'...", directory_path)
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM indexed_files;")
            conn.execute("DELETE FROM chunks;")
            conn.execute("DELETE FROM chunks_fts;")
            conn.commit()

        # Reinitialize FAISS
        self.faiss._init_or_load()
        return self.index_directory(directory_path, force_reindex=True)

    # --------------------------------------------------------------------------
    # Storage & State Helpers
    # --------------------------------------------------------------------------

    def _get_indexed_file_records(self, root: Path) -> dict[str, dict]:
        """Fetch previously indexed files matching root directory."""
        records: dict[str, dict] = {}
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT path, size_bytes, modified_time, sha256_hash, chunk_count FROM indexed_files;")
            for row in cursor.fetchall():
                p = row[0]
                try:
                    Path(p).resolve().relative_to(root)
                    records[p] = {
                        "path": p,
                        "size_bytes": row[1],
                        "modified_time": row[2],
                        "sha256_hash": row[3],
                        "chunk_count": row[4],
                    }
                except ValueError:
                    continue
        return records

    def _record_indexed_file(
        self,
        path: str,
        name: str,
        extension: str,
        size_bytes: int,
        modified_time: float,
        sha256_hash: str,
        chunk_count: int,
    ) -> None:
        """Insert or replace file metadata record in SQLite."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO indexed_files (
                    path, name, extension, size_bytes, modified_time,
                    sha256_hash, chunk_count, last_indexed, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    path,
                    name,
                    extension,
                    size_bytes,
                    modified_time,
                    sha256_hash,
                    chunk_count,
                    time.time(),
                    "indexed",
                ),
            )
            conn.commit()

    def _save_state(self, last_stats: dict) -> None:
        """Save indexing state metadata to JSON."""
        state = {
            "schema_version": self.INDEX_SCHEMA_VERSION,
            "embedding_model": self.embedding.model_name,
            "dense_dim": self.embedding.DENSE_DIM,
            "total_vectors": self.faiss.total_vectors,
            "total_chunks": self.fts.count_chunks(),
            "last_index_time": time.time(),
            "last_stats": last_stats,
        }
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

