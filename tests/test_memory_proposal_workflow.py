"""Unit tests for the memory proposal approval/rejection workflow and context builder."""

from datetime import datetime, timezone
from config.settings import AppConfig
from memory.context_builder import MemoryContextBuilder
from memory.models import MemoryDomain, MemoryProposal, MemoryProposalStatus
from memory.personal_memory import PersonalMemoryManager
from memory.proposal_manager import MemoryProposalManager
from memory.work_memory import WorkMemoryManager
from storage.database import DatabaseManager


def test_proposal_approval_persists_memory(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify approving a proposal creates persistent memory and marks proposal APPROVED."""
    pm = PersonalMemoryManager(config=temp_workspace, db_manager=test_db_manager)
    wm = WorkMemoryManager(config=temp_workspace, db_manager=test_db_manager)
    prop_mgr = MemoryProposalManager(temp_workspace, test_db_manager, pm, wm)

    proposal = MemoryProposal(
        id="prop_test_appr",
        domain=MemoryDomain.PERSONAL,
        proposal_text="User prefers Tamil voice output",
        suggested_title="Tamil Voice Output",
        suggested_category="preference",
        suggested_tags=["tamil", "voice"],
        confidence=0.90,
        status=MemoryProposalStatus.PENDING,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    prop_mgr.create_proposal(proposal)

    # Verify in pending
    pending = prop_mgr.get_pending_proposals()
    assert any(p.id == "prop_test_appr" for p in pending)

    # Approve
    success, mem_id = prop_mgr.approve_proposal("prop_test_appr", importance=4)
    assert success is True
    assert mem_id is not None

    # Verify no longer in pending
    pending_after = prop_mgr.get_pending_proposals()
    assert not any(p.id == "prop_test_appr" for p in pending_after)

    # Verify proposal status is APPROVED
    updated_prop = prop_mgr.get_proposal("prop_test_appr")
    assert updated_prop.status == MemoryProposalStatus.APPROVED
    assert updated_prop.reviewed_at is not None

    # Verify memory is saved in Personal Memory
    saved_mem = pm.get_memory(mem_id)
    assert saved_mem is not None
    assert saved_mem.title == "Tamil Voice Output"
    assert saved_mem.importance == 4


def test_proposal_rejection_never_persists(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify rejecting a proposal marks REJECTED and NEVER writes to memory."""
    pm = PersonalMemoryManager(config=temp_workspace, db_manager=test_db_manager)
    wm = WorkMemoryManager(config=temp_workspace, db_manager=test_db_manager)
    prop_mgr = MemoryProposalManager(temp_workspace, test_db_manager, pm, wm)

    proposal = MemoryProposal(
        id="prop_test_rej",
        domain=MemoryDomain.WORK,
        proposal_text="Experimental hack for debugging",
        suggested_title="Debug Hack",
        confidence=0.70,
        status=MemoryProposalStatus.PENDING,
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    )
    prop_mgr.create_proposal(proposal)

    # Reject
    success = prop_mgr.reject_proposal("prop_test_rej")
    assert success is True

    # Verify proposal status is REJECTED
    updated_prop = prop_mgr.get_proposal("prop_test_rej")
    assert updated_prop.status == MemoryProposalStatus.REJECTED

    # Verify NO memory was created in Work Memory
    work_items = wm.get_all_memories()
    assert len(work_items) == 0


def test_memory_context_builder(temp_workspace: AppConfig, test_db_manager: DatabaseManager):
    """Verify MemoryContextBuilder retrieves relevant approved memories and avoids leakage."""
    pm = PersonalMemoryManager(config=temp_workspace, db_manager=test_db_manager)
    wm = WorkMemoryManager(config=temp_workspace, db_manager=test_db_manager)
    builder = MemoryContextBuilder(pm, wm)

    pm.add_memory(title="Language Preference", content="Prefers Tamil speech synthesis", category="preference")
    wm.add_memory(title="Architecture Spec", content="Use Modular Monolith pattern for HAKS", project_name="HAKS")

    # Relevant query
    ctx = builder.build_context("How should HAKS architecture be structured?")
    assert "Modular Monolith" in ctx
    assert "--- APPROVED MEMORY CONTEXT" in ctx

    # Completely irrelevant query should yield empty context (minimal disclosure rule)
    empty_ctx = builder.build_context("Quantum astrophysics astrophysics astrophysical orbital calculation")
    assert empty_ctx == ""
