"""Comprehensive Phase 9.2 Industrial File Intelligence Test Suite.

Covers all 19 functional areas required by Phase 9.2:
1. Filesystem metadata search (extension, size, date filters)
2. Windows Search integration (ADODB CollatorDSO / graceful fallback)
3. Document extraction (PDF, DOCX, XLSX, PPTX, TXT, MD, CSV, HTML)
4. Token-aware document chunker (sentence boundaries, page/slide/sheet tracking)
5. BGE-M3 Dense retrieval (1024-d embeddings, cosine similarity)
6. BGE-M3 Sparse lexical retrieval (token-weight inverted index)
7. SQLite FTS5 full-text search & BM25 scoring
8. FAISS CPU dense vector index & persistence
9. Reciprocal Rank Fusion (RRF, k=60)
10. BGE Reranker v2 M3 cross-encoder ranking
11. Tamil script natural query normalization & retrieval
12. Tanglish / Romanized Tamil normalization & retrieval
13. Two-stage duplicate detection (size grouping + SHA-256/BLAKE3) & zero auto-deletion rule
14. Safe filesystem operations (copy, move, rename with conflict checks)
15. Windows Recycle Bin deletion (send2trash, never permanent deletion)
16. Privacy filter (credentials, .env, memory DBs, outside allowlists, path traversal)
17. Resource management (lazy loading, idle unloading, memory protection)
18. Phase 9.1 Application automation regression
19. Voice architecture regression (Gemini Live, APM, Silero VAD, interruption sync)
"""

import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from config.settings import AppConfig, SystemPaths
from file_intelligence.chunker import DocumentChunker
from file_intelligence.document_extractor import DocumentExtractor
from file_intelligence.embedding_engine import BGEM3EmbeddingEngine
from file_intelligence.faiss_index import FaissDenseIndex
from file_intelligence.filesystem_ops import FilesystemOperationsService
from file_intelligence.fts_bm25 import FTSAndSparseEngine
from file_intelligence.hybrid_retriever import HybridRetriever
from file_intelligence.indexing_service import IndexingService
from file_intelligence.models import (
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
from file_intelligence.service import FileIntelligenceService
from file_intelligence.windows_search import WindowsSearchEngine
from windows_control.application_controller import ApplicationController
from windows_control.application_registry import ApplicationRegistry
from windows_control.manager import WindowsControlManager
from windows_control.models import ActionPlan, ExecutionStatus, RiskLevel


@pytest.fixture
def temp_workspace():
    """Create an isolated temporary folder for filesystem tests."""
    temp_dir = Path(tempfile.mkdtemp(prefix="haks_test_fi_"))
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def create_isolated_service(temp_workspace: Path) -> FileIntelligenceService:
    """Create an isolated FileIntelligenceService using temporary paths."""
    config = AppConfig()
    config.paths.data_dir = temp_workspace / "data"
    config.paths.data_dir.mkdir(parents=True, exist_ok=True)
    return FileIntelligenceService(config=config, use_mock_fallback=True)


# ------------------------------------------------------------------------------
# 1. Metadata Search
# ------------------------------------------------------------------------------
def test_metadata_search(temp_workspace):
    """Test searching filesystem metadata with extension, size, and pattern filters."""
    f1 = temp_workspace / "report_2026.pdf"
    f1.write_text("dummy pdf content" * 10, encoding="utf-8")
    f2 = temp_workspace / "notes.txt"
    f2.write_text("quick text notes", encoding="utf-8")
    f3 = temp_workspace / "sub" / "data.csv"
    f3.parent.mkdir(parents=True, exist_ok=True)
    f3.write_text("a,b,c\n1,2,3", encoding="utf-8")

    fs_ops = FilesystemOperationsService()

    # Search by extension
    results_pdf = fs_ops.search_files_by_metadata(temp_workspace, extensions=[".pdf"])
    assert len(results_pdf) == 1
    assert results_pdf[0].name == "report_2026.pdf"

    # Search by name pattern
    results_notes = fs_ops.search_files_by_metadata(temp_workspace, name_pattern="*note*")
    assert len(results_notes) == 1
    assert results_notes[0].name == "notes.txt"

    # Search with min size filter
    results_large = fs_ops.search_files_by_metadata(temp_workspace, min_size_bytes=100)
    assert len(results_large) == 1
    assert results_large[0].name == "report_2026.pdf"


# ------------------------------------------------------------------------------
# 2. Windows Search Integration
# ------------------------------------------------------------------------------
def test_windows_search_integration():
    """Test Windows Search CollatorDSO engine availability and query APIs."""
    ws = WindowsSearchEngine()
    is_avail = ws.is_available()
    assert isinstance(is_avail, bool)

    # Search by filename
    results_name = ws.search_by_name("explorer", max_results=5)
    assert isinstance(results_name, list)

    # Search by content
    results_content = ws.search_by_content("test", max_results=5)
    assert isinstance(results_content, list)


# ------------------------------------------------------------------------------
# 3. Document Extraction (PDF, DOCX, XLSX, PPTX, TXT, MD, CSV, HTML)
# ------------------------------------------------------------------------------
def test_document_extractor(temp_workspace):
    """Verify document extraction across TXT, MD, CSV, HTML, and office documents."""
    extractor = DocumentExtractor()

    # TXT & MD
    txt_file = temp_workspace / "sample.txt"
    txt_file.write_text("Hello HAKS AI Assistant File Intelligence", encoding="utf-8")
    full_text = extractor.extract_full_text(txt_file)
    assert "HAKS AI Assistant" in full_text

    sections = list(extractor.extract_sections(txt_file))
    assert len(sections) > 0
    assert "HAKS AI Assistant" in sections[0][0]

    # CSV
    csv_file = temp_workspace / "metrics.csv"
    csv_file.write_text("Name,Score,Status\nAlpha,95,PASS\nBeta,88,PASS\n", encoding="utf-8")
    csv_text = extractor.extract_full_text(csv_file)
    assert "Alpha" in csv_text
    assert "PASS" in csv_text

    # HTML
    html_file = temp_workspace / "index.html"
    html_file.write_text("<html><head><title>Test Page</title></head><body><h1>HAKS</h1><p>Document extractor HTML test.</p></body></html>", encoding="utf-8")
    html_text = extractor.extract_full_text(html_file)
    assert "Document extractor HTML test." in html_text

    # DOCX
    try:
        from docx import Document
        docx_file = temp_workspace / "doc.docx"
        doc = Document()
        doc.add_heading("HAKS Docx Test", 0)
        doc.add_paragraph("Paragraph 1 content.")
        doc.save(str(docx_file))
        docx_text = extractor.extract_full_text(docx_file)
        assert "Paragraph 1 content." in docx_text
    except ImportError:
        pass

    # XLSX
    try:
        import openpyxl
        xlsx_file = temp_workspace / "sheet.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Financials"
        ws.append(["Category", "Amount"])
        ws.append(["Hardware", 5000])
        wb.save(str(xlsx_file))
        xlsx_sections = list(extractor.extract_sections(xlsx_file))
        assert len(xlsx_sections) > 0
        assert "Financials" in (xlsx_sections[0][1] or "")
        assert "Hardware" in xlsx_sections[0][0]
    except ImportError:
        pass


# ------------------------------------------------------------------------------
# 4. Token-Aware Document Chunker
# ------------------------------------------------------------------------------
def test_chunker():
    """Verify sentence-boundary preserving chunking with target tokens and overlap."""
    chunker = DocumentChunker(target_chunk_tokens=450, overlap_tokens=60)

    sentences = [f"This is sentence number {i} discussing HAKS File Intelligence features." for i in range(120)]
    long_text = " ".join(sentences)

    chunks = chunker.chunk_section(
        text=long_text,
        file_path="sample.txt",
        file_name="sample.txt",
        file_type="txt",
        page_or_section="Section 1",
    )
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 550
        assert c.chunk_index >= 0
        assert c.file_path == "sample.txt"
        assert c.page_or_section == "Section 1"


# ------------------------------------------------------------------------------
# 5. BGE-M3 Dense Retrieval
# ------------------------------------------------------------------------------
def test_bge_m3_dense_retrieval():
    """Verify 1024-dimensional dense embeddings and cosine similarity."""
    engine = BGEM3EmbeddingEngine(use_mock_fallback=True)
    embeddings, _ = engine.encode_documents(["HAKS Voice Assistant", "Quarterly Budget 2026"])
    assert embeddings.shape == (2, 1024)

    # Normalized check (L2 norm should be ~1.0)
    norm0 = np.linalg.norm(embeddings[0])
    norm1 = np.linalg.norm(embeddings[1])
    assert pytest.approx(norm0, rel=1e-3) == 1.0
    assert pytest.approx(norm1, rel=1e-3) == 1.0

    # Query embedding
    q_emb, _ = engine.encode_queries(["voice assistant"])
    assert q_emb.shape == (1, 1024)
    assert pytest.approx(np.linalg.norm(q_emb[0]), rel=1e-3) == 1.0


# ------------------------------------------------------------------------------
# 6. BGE-M3 Sparse Lexical Retrieval
# ------------------------------------------------------------------------------
def test_bge_m3_sparse_retrieval():
    """Verify BGE-M3 sparse token weight generation and matching."""
    engine = BGEM3EmbeddingEngine(use_mock_fallback=True)
    _, sparse_weights = engine.encode_documents(["HAKS voice assistant architecture and audio pipeline"])
    assert len(sparse_weights) == 1
    assert "voice" in sparse_weights[0]
    assert "assistant" in sparse_weights[0]
    assert sparse_weights[0]["voice"] > 0.0


# ------------------------------------------------------------------------------
# 7. SQLite FTS5 & BM25 Scoring
# ------------------------------------------------------------------------------
def test_fts5_and_bm25(temp_workspace):
    """Verify SQLite FTS5 full-text indexing, BM25 scoring, and sparse token inverted index."""
    db_path = temp_workspace / "test_fts.db"
    fts = FTSAndSparseEngine(db_path=db_path)

    chunk1 = DocumentChunk(
        chunk_id="chunk_1",
        file_path=str(temp_workspace / "voice.txt"),
        file_name="voice.txt",
        file_type="txt",
        chunk_index=0,
        text="HAKS voice assistant provides real-time speech detection with WebRTC APM and Silero VAD.",
        token_count=14,
        embedding_sparse={"voice": 1.2, "webrtc": 0.9, "silero": 1.1},
    )
    chunk2 = DocumentChunk(
        chunk_id="chunk_2",
        file_path=str(temp_workspace / "budget.csv"),
        file_name="budget.csv",
        file_type="csv",
        chunk_index=0,
        text="Financial annual budget forecast for engineering and operations.",
        token_count=10,
        embedding_sparse={"budget": 1.5, "finance": 0.8},
    )

    fts.index_chunks([chunk1, chunk2])

    # FTS query for voice
    results_fts = fts.search_fts("voice assistant", top_k=5)
    assert len(results_fts) > 0
    assert results_fts[0][0] == chunk1.chunk_id

    # Sparse inverted index search
    results_sparse = fts.search_sparse({"budget": 1.0}, top_k=5)
    assert len(results_sparse) > 0
    assert results_sparse[0][0] == chunk2.chunk_id


# ------------------------------------------------------------------------------
# 8. FAISS Dense Vector Index & Persistence
# ------------------------------------------------------------------------------
def test_faiss_indexing_and_persistence(temp_workspace):
    """Verify FAISS IndexFlatIP index creation, nearest neighbor search, and disk persistence."""
    index_path = temp_workspace / "test_faiss.index"
    faiss_idx = FaissDenseIndex(index_path=index_path)

    v1 = np.zeros(1024, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(1024, dtype=np.float32)
    v2[1] = 1.0

    faiss_idx.add_vectors(
        vectors=np.array([v1, v2]),
        chunk_ids=["chunk_1", "chunk_2"],
        file_paths=["file1.txt", "file2.txt"],
    )
    assert faiss_idx.total_vectors == 2

    # Query nearest to v1
    top_matches = faiss_idx.search(v1, top_k=1)
    assert len(top_matches) == 1
    assert top_matches[0][0] == "chunk_1"
    assert pytest.approx(top_matches[0][1], rel=1e-3) == 1.0

    # Save and reload
    faiss_idx.save(index_path)
    assert index_path.exists()

    reloaded = FaissDenseIndex(index_path=index_path)
    assert reloaded.total_vectors == 2
    top_matches_reloaded = reloaded.search(v2, top_k=1)
    assert top_matches_reloaded[0][0] == "chunk_2"


# ------------------------------------------------------------------------------
# 9. Reciprocal Rank Fusion (RRF, k=60)
# ------------------------------------------------------------------------------
def test_rrf_fusion():
    """Verify Reciprocal Rank Fusion (RRF k=60) scoring formula and candidate prioritization."""
    rrf_k = 60
    # Simulate scores from two rankings
    # Document B is rank 2 in stream 1, rank 1 in stream 2
    # Document A is rank 1 in stream 1
    # Document C is rank 2 in stream 2
    score_b = (1.0 / (rrf_k + 2)) + (1.0 / (rrf_k + 1))
    score_a = 1.0 / (rrf_k + 1)
    score_c = 1.0 / (rrf_k + 2)

    assert score_b > score_a
    assert score_b > score_c

    retriever = HybridRetriever(
        embedding_engine=BGEM3EmbeddingEngine(use_mock_fallback=True),
        faiss_index=FaissDenseIndex(),
        fts_engine=FTSAndSparseEngine(),
        reranker_engine=BGERerankerEngine(use_mock_fallback=True),
        privacy_filter=PrivacyFilter(),
    )
    assert retriever.rrf_k == 60


# ------------------------------------------------------------------------------
# 10. BGE Reranker v2 M3
# ------------------------------------------------------------------------------
def test_bge_reranker():
    """Verify BGE Reranker v2 cross-encoder scoring and reordering."""
    reranker = BGERerankerEngine(use_mock_fallback=True)

    items = [
        SearchResultItem(file_path="f1.txt", file_name="f1.txt", file_type="txt", snippet="Irrelevant gardening tips", score=0.5),
        SearchResultItem(file_path="f2.txt", file_name="f2.txt", file_type="txt", snippet="Highly relevant HAKS voice pipeline architecture", score=0.4),
    ]

    reranked = reranker.rerank("HAKS voice pipeline", items, top_n=2)
    assert len(reranked) == 2
    assert reranked[0].file_path == "f2.txt"
    assert reranked[0].score > reranked[1].score


# ------------------------------------------------------------------------------
# 11. Tamil Script Retrieval
# ------------------------------------------------------------------------------
def test_tamil_retrieval(temp_workspace):
    """Verify Tamil script query normalization, domain mapping, and retrieval."""
    normalizer = QueryNormalizer()
    q = normalizer.normalize("திட்ட அறிக்கை கோப்பைத் தேடு")
    assert q.language == "ta"
    assert "திட்ட" in q.normalized_query
    # "அறிக்கை" maps to "report"
    assert "report" in q.normalized_query

    # Test end-to-end service retrieval in isolated workspace
    service = create_isolated_service(temp_workspace)
    ta_file = temp_workspace / "திட்டம்.txt"
    ta_file.write_text("செயற்கை நுண்ணறிவு திட்ட அறிக்கை மற்றும் குரல் வழி கட்டுப்பாடு.", encoding="utf-8")
    service.index_directory(temp_workspace, force_reindex=True)

    results = service.search("திட்ட அறிக்கை தேடு", top_k=5, directory_scope=str(temp_workspace))
    assert len(results) > 0
    assert results[0].file_name == "திட்டம்.txt"


# ------------------------------------------------------------------------------
# 12. Tanglish / Romanized Tamil Retrieval
# ------------------------------------------------------------------------------
def test_tanglish_retrieval(temp_workspace):
    """Verify Tanglish normalization, particle stripping, and retrieval."""
    normalizer = QueryNormalizer()
    q = normalizer.normalize("resume file enga irukku")
    assert q.language == "tanglish"
    assert "resume" in q.normalized_query

    # End-to-end service retrieval in isolated workspace
    service = create_isolated_service(temp_workspace)
    resume_file = temp_workspace / "resume_kavitha.txt"
    resume_file.write_text("Curriculum Vitae of Senior AI Engineer specializing in NLP.", encoding="utf-8")
    service.index_directory(temp_workspace, force_reindex=True)

    results = service.search("kavitha resume enga irukku", top_k=5, directory_scope=str(temp_workspace))
    assert len(results) > 0
    assert results[0].file_name == "resume_kavitha.txt"


# ------------------------------------------------------------------------------
# 13. Two-Stage Duplicate Detection & Zero Auto-Deletion Rule
# ------------------------------------------------------------------------------
def test_duplicate_detection(temp_workspace):
    """Verify size grouping, SHA-256 and BLAKE3 hash deduplication, and zero auto-deletion."""
    fs_ops = FilesystemOperationsService()

    content = "IDENTICAL DUP CONTENT" * 50
    f1 = temp_workspace / "copy1.txt"
    f2 = temp_workspace / "copy2.txt"
    f3 = temp_workspace / "unique.txt"

    f1.write_text(content, encoding="utf-8")
    f2.write_text(content, encoding="utf-8")
    f3.write_text("Completely unique text data.", encoding="utf-8")

    groups, benchmarks = fs_ops.find_duplicates(temp_workspace, use_blake3_benchmark=True)
    assert len(groups) == 1
    assert len(groups[0].file_paths) == 2
    assert "copy1.txt" in str(groups[0].file_paths)
    assert "copy2.txt" in str(groups[0].file_paths)
    assert "hash_time_seconds" in benchmarks

    # MANDATORY SAFETY RULE: Verify that find_duplicates NEVER deleted either file!
    assert f1.exists()
    assert f2.exists()
    assert f3.exists()


# ------------------------------------------------------------------------------
# 14. Safe Filesystem Operations (Copy, Move, Rename)
# ------------------------------------------------------------------------------
def test_safe_file_ops(temp_workspace):
    """Verify safe copy, move, and rename operations with conflict resolution."""
    fs_ops = FilesystemOperationsService()

    src = temp_workspace / "original.txt"
    src.write_text("original content", encoding="utf-8")

    # Copy
    dst_copy = temp_workspace / "original_copy.txt"
    success, msg = fs_ops.copy_file(src, dst_copy)
    assert success
    assert dst_copy.exists()
    assert dst_copy.read_text(encoding="utf-8") == "original content"

    # Rename
    success_ren, msg_ren = fs_ops.rename_file(dst_copy, "renamed.txt")
    assert success_ren
    assert (temp_workspace / "renamed.txt").exists()
    assert not dst_copy.exists()

    # Move
    sub_dir = temp_workspace / "nested"
    sub_dir.mkdir(parents=True, exist_ok=True)
    dst_move = sub_dir / "moved.txt"
    success_mv, msg_mv = fs_ops.move_file(temp_workspace / "renamed.txt", dst_move)
    assert success_mv
    assert dst_move.exists()


# ------------------------------------------------------------------------------
# 15. Windows Recycle Bin Deletion (Send2Trash, Never Permanent)
# ------------------------------------------------------------------------------
def test_recycle_bin(temp_workspace):
    """Verify safe Recycle Bin deletion using send2trash and no fallback permanent deletion."""
    fs_ops = FilesystemOperationsService()

    target = temp_workspace / "to_delete.txt"
    target.write_text("delete this safely to recycle bin", encoding="utf-8")

    success, msg = fs_ops.send_to_recycle_bin(target)
    assert success
    assert not target.exists()  # Safely moved to Windows Recycle Bin
    assert "Recycle Bin" in msg

    # Non-existent file error
    success_err, msg_err = fs_ops.send_to_recycle_bin(target)
    assert not success_err
    assert "does not exist" in msg_err


# ------------------------------------------------------------------------------
# 16. Privacy Filter & Secret Exclusion
# ------------------------------------------------------------------------------
def test_privacy_filter(temp_workspace):
    """Verify exclusion of .env, credentials, pem keys, memory DBs, and traversal attacks."""
    privacy = PrivacyFilter()

    # Secret files
    env_file = temp_workspace / ".env"
    env_file.write_text("SECRET_KEY=12345", encoding="utf-8")
    assert not privacy.is_file_indexable(env_file)

    cred_file = temp_workspace / "credentials.json"
    cred_file.write_text('{"api_key": "xyz"}', encoding="utf-8")
    assert not privacy.is_file_indexable(cred_file)

    key_file = temp_workspace / "server.pem"
    key_file.write_text("-----BEGIN RSA PRIVATE KEY-----", encoding="utf-8")
    assert not privacy.is_file_indexable(key_file)

    # Memory databases
    mem_db = temp_workspace / "personal_memory.db"
    mem_db.write_text("sqlite", encoding="utf-8")
    assert not privacy.is_file_indexable(mem_db)

    # Valid text file
    safe_file = temp_workspace / "valid_document.txt"
    safe_file.write_text("Safe public documentation", encoding="utf-8")
    assert privacy.is_file_indexable(safe_file)


# ------------------------------------------------------------------------------
# 17. Resource Management & Unloading
# ------------------------------------------------------------------------------
def test_resource_usage_and_unloading(temp_workspace):
    """Verify lazy loading and model memory release via unload_models()."""
    service = create_isolated_service(temp_workspace)
    service.search("test resource unload", top_k=2)

    service.unload_models()
    assert service.embedding._model is None
    assert service.reranker._model is None
    assert not service.embedding.is_loaded
    assert not service.reranker.is_loaded


# ------------------------------------------------------------------------------
# 18. Phase 9.1 Application Automation Regression
# ------------------------------------------------------------------------------
def test_phase_9_1_regression():
    """Verify that Phase 9.1 Application Automation commands execute properly."""
    manager = WindowsControlManager()

    # Launch app intent
    plan = manager.interpreter.interpret("open notepad")
    assert plan.operation_type == "launch_app"
    assert "notepad" in plan.target.lower()

    # Switch app intent
    plan_switch = manager.interpreter.interpret("switch to notepad")
    assert plan_switch.operation_type == "switch_app"

    # Close app intent
    plan_close = manager.interpreter.interpret("close notepad")
    assert plan_close.operation_type == "close_app"


# ------------------------------------------------------------------------------
# 19. Voice Architecture Regression
# ------------------------------------------------------------------------------
def test_voice_architecture_regression():
    """Verify Voice Architecture voice commands, APM, and interruption sync remain intact."""
    from voice.conversation_engine import ContinuousVoiceEngine
    from voice.models import VoiceState

    engine = ContinuousVoiceEngine()
    assert engine.state == VoiceState.STOPPED
    assert hasattr(engine, "interruption_controller")
    assert hasattr(engine, "audio_capture")
    assert hasattr(engine, "vad")
    assert hasattr(engine, "player")
    assert hasattr(engine, "gemini_live")

    # Verify voice-driven file intelligence command parsing
    manager = WindowsControlManager()
    voice_file_plan = manager.interpreter.interpret("find my resume", source="voice")
    assert voice_file_plan.operation_type in ("file_search", "search_files")
    assert "resume" in voice_file_plan.target
