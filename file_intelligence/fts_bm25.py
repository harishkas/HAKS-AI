"""SQLite FTS5 and BM25 Lexical + BGE-M3 Sparse Retrieval Engine (Phase 9.2).

Provides full-text lexical indexing via SQLite FTS5, BM25Okapi scoring,
and token-level sparse lexical inverted indexing matching BGE-M3 specifications.
"""

from collections import defaultdict
import json
import logging
from pathlib import Path
import sqlite3
from typing import Optional, Tuple

from rank_bm25 import BM25Okapi

from file_intelligence.models import DocumentChunk, SearchResultItem

logger = logging.getLogger("haks.file_intelligence.fts_bm25")


class FTSAndSparseEngine:
    """Manages SQLite FTS5 full-text database, BM25Okapi scoring, and sparse token inverted index."""

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        self.db_path = Path(db_path) if db_path else None
        self._conn: Optional[sqlite3.Connection] = None

        # Sparse Inverted Index for BGE-M3: token -> {chunk_id: weight}
        self._sparse_inverted_index: dict[str, dict[str, float]] = defaultdict(dict)
        self._sparse_chunk_lengths: dict[str, float] = {}

        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database with FTS5 virtual table and metadata tables."""
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        else:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)

        cursor = self._conn.cursor()
        # 1. Chunks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page_or_section TEXT,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                char_start INTEGER,
                char_end INTEGER,
                mtime REAL NOT NULL,
                sparse_json TEXT
            );
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);")

        # 2. SQLite FTS5 virtual table for full-text search
        cursor.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                file_path UNINDEXED,
                file_name,
                page_or_section,
                text,
                tokenize = 'unicode61 remove_diacritics 2'
            );
            """
        )
        self._conn.commit()

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Insert or replace chunks into FTS5 table and update sparse inverted index."""
        if not chunks or not self._conn:
            return 0

        cursor = self._conn.cursor()
        for c in chunks:
            sparse_json = json.dumps(c.embedding_sparse) if c.embedding_sparse else None

            # Upsert into chunks table
            cursor.execute(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, file_path, file_name, file_type, chunk_index,
                    page_or_section, text, token_count, char_start, char_end, mtime, sparse_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    c.chunk_id,
                    c.file_path,
                    c.file_name,
                    c.file_type,
                    c.chunk_index,
                    c.page_or_section,
                    c.text,
                    c.token_count,
                    c.char_start,
                    c.char_end,
                    c.mtime,
                    sparse_json,
                ),
            )

            # Delete old FTS entry if exists, then insert
            cursor.execute("DELETE FROM chunks_fts WHERE chunk_id = ?;", (c.chunk_id,))
            cursor.execute(
                """
                INSERT INTO chunks_fts (chunk_id, file_path, file_name, page_or_section, text)
                VALUES (?, ?, ?, ?, ?);
                """,
                (c.chunk_id, c.file_path, c.file_name, c.page_or_section, c.text),
            )

            # Update in-memory BGE-M3 Sparse Inverted Index
            if c.embedding_sparse:
                norm_sq = sum(w * w for w in c.embedding_sparse.values())
                self._sparse_chunk_lengths[c.chunk_id] = max(1e-6, norm_sq ** 0.5)
                for token, weight in c.embedding_sparse.items():
                    self._sparse_inverted_index[token][c.chunk_id] = float(weight)

        self._conn.commit()
        return len(chunks)

    def search_fts(self, query: str, top_k: int = 15) -> list[Tuple[str, float]]:
        """Search SQLite FTS5 using BM25 ranking."""
        if not self._conn or not query.strip():
            return []

        # Sanitize query terms for FTS5 syntax
        clean_terms = []
        for word in query.replace('"', " ").replace("'", " ").split():
            clean = "".join(c for c in word if c.isalnum() or ord(c) > 127)
            if clean:
                clean_terms.append(f'"{clean}"')

        if not clean_terms:
            return []

        fts_query = " OR ".join(clean_terms)

        cursor = self._conn.cursor()
        try:
            cursor.execute(
                """
                SELECT chunk_id, bm25(chunks_fts) as rank_score
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank_score
                LIMIT ?;
                """,
                (fts_query, top_k),
            )
            rows = cursor.fetchall()
            # FTS5 bm25 returns negative scores where more negative = better match
            results = []
            for chunk_id, raw_score in rows:
                score = -float(raw_score)  # Convert to positive higher-is-better
                results.append((chunk_id, score))
            return results
        except sqlite3.OperationalError as exc:
            logger.debug("FTS5 query notice for '%s': %s", query, exc)
            return []

    def search_sparse(self, query_sparse_weights: dict[str, float], top_k: int = 15) -> list[Tuple[str, float]]:
        """Score chunks via dot product over BGE-M3 sparse lexical weights."""
        if not query_sparse_weights or not self._sparse_inverted_index:
            return []

        scores: dict[str, float] = defaultdict(float)

        for token, q_weight in query_sparse_weights.items():
            if token in self._sparse_inverted_index:
                postings = self._sparse_inverted_index[token]
                for chunk_id, doc_weight in postings.items():
                    scores[chunk_id] += q_weight * doc_weight

        # Normalize by doc vector length
        ranked = []
        for chunk_id, dot_prod in scores.items():
            doc_len = self._sparse_chunk_lengths.get(chunk_id, 1.0)
            ranked.append((chunk_id, float(dot_prod / doc_len)))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def remove_file(self, file_path: str) -> int:
        """Remove all chunks associated with a file path from FTS and sparse tables."""
        if not self._conn:
            return 0

        cursor = self._conn.cursor()
        cursor.execute("SELECT chunk_id FROM chunks WHERE file_path = ?;", (file_path,))
        chunk_ids = [r[0] for r in cursor.fetchall()]

        if not chunk_ids:
            return 0

        for chunk_id in chunk_ids:
            cursor.execute("DELETE FROM chunks WHERE chunk_id = ?;", (chunk_id,))
            cursor.execute("DELETE FROM chunks_fts WHERE chunk_id = ?;", (chunk_id,))
            self._sparse_chunk_lengths.pop(chunk_id, None)

            # Purge from inverted index
            for token in list(self._sparse_inverted_index.keys()):
                self._sparse_inverted_index[token].pop(chunk_id, None)
                if not self._sparse_inverted_index[token]:
                    del self._sparse_inverted_index[token]

        self._conn.commit()
        return len(chunk_ids)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        """Retrieve full DocumentChunk details by chunk_id."""
        if not self._conn:
            return None

        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT chunk_id, file_path, file_name, file_type, chunk_index,
                   page_or_section, text, token_count, char_start, char_end, mtime, sparse_json
            FROM chunks WHERE chunk_id = ?;
            """,
            (chunk_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        sparse_dict = json.loads(row[11]) if row[11] else None

        return DocumentChunk(
            chunk_id=row[0],
            file_path=row[1],
            file_name=row[2],
            file_type=row[3],
            chunk_index=row[4],
            page_or_section=row[5],
            text=row[6],
            token_count=row[7],
            char_start=row[8],
            char_end=row[9],
            mtime=row[10],
            embedding_sparse=sparse_dict,
        )

    def count_chunks(self) -> int:
        """Return total indexed chunks."""
        if not self._conn:
            return 0
        cursor = self._conn.cursor()
        cursor.execute("SELECT count(*) FROM chunks;")
        return int(cursor.fetchone()[0])

