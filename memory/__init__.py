"""Memory Architecture package for HAKS Personal AI Assistant."""

from memory.context_builder import MemoryContextBuilder
from memory.encryption import MemoryEncryptionService, get_encryption_service
from memory.models import (
    MemoryBackupBundle,
    MemoryDomain,
    MemoryItem,
    MemoryProposal,
    MemoryProposalStatus,
    MemorySearchResult,
)
from memory.personal_memory import PersonalMemoryManager
from memory.proposal_detector import MemoryProposalDetector
from memory.proposal_manager import MemoryProposalManager
from memory.vector_index import LocalVectorIndex
from memory.work_memory import WorkMemoryManager

__all__ = [
    "LocalVectorIndex",
    "MemoryBackupBundle",
    "MemoryContextBuilder",
    "MemoryDomain",
    "MemoryEncryptionService",
    "MemoryItem",
    "MemoryProposal",
    "MemoryProposalDetector",
    "MemoryProposalManager",
    "MemoryProposalStatus",
    "MemorySearchResult",
    "PersonalMemoryManager",
    "WorkMemoryManager",
    "get_encryption_service",
]
