"""Application-wide Qt signal dispatcher for HAKS.

Provides a thread-safe decoupled event bus using PySide6 Signals and Slots.
Ensures background worker threads (AI inference, audio capture, DB operations)
can communicate with the UI without thread violations or UI freezing.
"""

from PySide6.QtCore import QObject, Signal


class ApplicationEvents(QObject):
    """Central event bus emitting system-wide signals."""

    # UI & Navigation signals
    navigate_to_page = Signal(str)  # page_id (e.g., 'home', 'chat', 'memory')
    status_message_updated = Signal(str)  # message string

    # Voice pipeline state signals
    voice_state_changed = Signal(str)  # STANDBY, LISTENING, THINKING, etc.
    voice_transcript_received = Signal(str, str)  # (text, language)

    # AI engine lifecycle signals
    ai_generation_started = Signal(str)  # prompt/query
    ai_token_received = Signal(str)  # streamed chunk
    ai_generation_finished = Signal(str)  # full response
    ai_error_occurred = Signal(str)  # error description

    # Memory system signals
    memory_proposal_created = Signal(dict)  # proposal data
    memory_domain_updated = Signal(str)  # 'personal' or 'work'

    # Security & Permission signals
    permission_requested = Signal(dict)  # request details
    permission_resolved = Signal(str, bool)  # request_id, approved

    # System resource / hardware monitoring signals
    system_metrics_updated = Signal(dict)  # cpu, ram, storage metrics


# Singleton instance
_events_instance: ApplicationEvents | None = None


def get_app_events() -> ApplicationEvents:
    """Return the singleton instance of ApplicationEvents."""
    global _events_instance
    if _events_instance is None:
        _events_instance = ApplicationEvents()
    return _events_instance
