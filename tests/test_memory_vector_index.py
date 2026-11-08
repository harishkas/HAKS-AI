"""Unit tests for the local vector index and semantic similarity."""

from pathlib import Path
from memory.vector_index import LocalVectorIndex


def test_vector_indexing_and_search(tmp_path: Path):
    """Verify document addition, semantic retrieval, and cosine ranking."""
    index_file = tmp_path / "test_index.json"
    index = LocalVectorIndex(index_file)

    index.add_document("doc1", "I love South Indian filter coffee in the morning", {"category": "preference"})
    index.add_document("doc2", "Architectural decision: Use SQLite with WAL mode", {"category": "architecture"})
    index.add_document("doc3", "I prefer speaking in Tamil rather than English", {"category": "language"})

    assert len(index) == 3

    # Query matching doc1
    results = index.search("filter coffee morning", top_k=2)
    assert len(results) > 0
    top_doc_id, score, meta = results[0]
    assert top_doc_id == "doc1"
    assert score > 0.4
    assert meta["category"] == "preference"

    # Query matching doc2
    results_arch = index.search("database sqlite architecture", top_k=1)
    assert len(results_arch) > 0
    assert results_arch[0][0] == "doc2"


def test_vector_index_persistence(tmp_path: Path):
    """Verify vector index saves to disk and reloads properly."""
    index_file = tmp_path / "persistent_index.json"
    index1 = LocalVectorIndex(index_file)
    index1.add_document("docA", "Machine learning quantized model", {"tag": "ai"})
    assert len(index1) == 1

    # Reload in new instance
    index2 = LocalVectorIndex(index_file)
    assert len(index2) == 1
    assert "docA" in index2._entries

    # Removal
    removed = index2.remove_document("docA")
    assert removed is True
    assert len(index2) == 0

    index3 = LocalVectorIndex(index_file)
    assert len(index3) == 0


def test_text_similarity_calculation(tmp_path: Path):
    """Verify text similarity scoring between identical, similar, and unrelated strings."""
    index = LocalVectorIndex(tmp_path / "sim_index.json")

    score_identical = index.calculate_text_similarity(
        "I prefer filter coffee in the morning",
        "I prefer filter coffee in the morning",
    )
    assert score_identical > 0.95

    score_similar = index.calculate_text_similarity(
        "I love drinking hot filter coffee every morning",
        "I prefer filter coffee in the morning",
    )
    assert score_similar > 0.35

    score_unrelated = index.calculate_text_similarity(
        "Quantum mechanics and gravitational wave detection",
        "I prefer filter coffee in the morning",
    )
    assert score_unrelated < score_similar
