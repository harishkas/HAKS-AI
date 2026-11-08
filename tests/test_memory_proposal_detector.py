"""Unit tests for automatic memory proposal detection."""

from memory.models import MemoryDomain, MemoryProposalStatus
from memory.proposal_detector import MemoryProposalDetector


def test_detect_personal_preference():
    """Verify detecting user personal preference statement."""
    detector = MemoryProposalDetector()
    text = "Please remember that I prefer speaking in Tamil when using voice."
    proposal = detector.detect_proposal(text, source_conversation_id="conv_123")

    assert proposal is not None
    assert proposal.domain == MemoryDomain.PERSONAL
    assert proposal.suggested_category == "preference"
    assert proposal.status == MemoryProposalStatus.PENDING
    assert proposal.source_conversation_id == "conv_123"
    assert proposal.suggested_importance >= 4


def test_detect_work_architecture_decision():
    """Verify detecting technical decision for project."""
    detector = MemoryProposalDetector()
    text = "For project HAKS, we decided to use a Modular Monolith architecture instead of microservices."
    proposal = detector.detect_proposal(text, source_conversation_id="conv_456")

    assert proposal is not None
    assert proposal.domain == MemoryDomain.WORK
    assert proposal.suggested_category == "architecture"
    assert proposal.suggested_project == "HAKS"
    assert proposal.suggested_importance >= 4


def test_ignore_negative_phrases():
    """Verify phrases with negative/temporary intent produce no proposal."""
    detector = MemoryProposalDetector()
    assert detector.detect_proposal("Please don't remember this temporary password.") is None
    assert detector.detect_proposal("This is just temporary scratch data, ignore this.") is None
    assert detector.detect_proposal("Short text") is None
