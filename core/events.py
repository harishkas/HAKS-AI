"""Application-wide Qt signal dispatcher for HAKS.

Provides a thread-safe decoupled event bus using PySide6 Signals and Slots.
Ensures background worker threads (AI inference, audio capture, DB operations)
and multiple UI surfaces (Main Window, Floating Assistant, System Tray)
stay synchronized without tight coupling or UI thread blocking.
"""

from typing import Optional
from PySide6.QtCore import QObject, Signal


class ApplicationEvents(QObject):
    """Central event bus emitting system-wide signals."""

    # UI & Navigation signals
    navigate_to_page = Signal(str)  # page_id (e.g., 'home', 'chat', 'memory')
    status_message_updated = Signal(str)  # message string
    theme_changed = Signal(str)  # 'dark' or 'light'
    show_main_window_requested = Signal()
    show_floating_requested = Signal()
    minimize_to_tray_requested = Signal()

    # Conversation & Message signals
    conversation_created = Signal(str, str)  # (conv_id, title)
    conversation_loaded = Signal(str)  # conv_id
    conversation_list_updated = Signal()
    message_received = Signal(str, str, str, str)  # (conv_id, message_id, role, content)
    message_chunk_received = Signal(str, str)  # (conv_id, chunk_text)

    # Voice pipeline state signals
    voice_state_changed = Signal(str)  # STANDBY, LISTENING, THINKING, etc.
    voice_transcript_received = Signal(str, str)  # (text, language)
    voice_mute_toggled = Signal(bool)  # is_muted
    voice_audio_level_updated = Signal(float)  # 0.0 to 1.0 live mic VU meter
    voice_speech_prob_updated = Signal(float)  # 0.0 to 1.0 live human speech probability
    voice_speech_started = Signal()
    voice_speech_ended = Signal()
    voice_speaking_started = Signal(str)  # text being spoken
    voice_speaking_finished = Signal()
    voice_interrupted = Signal()  # barge-in event
    voice_error_occurred = Signal(str)  # voice error message
    voice_live_status_changed = Signal(str)  # Gemini Live connection status
    voice_live_latency_updated = Signal(float)  # Gemini Live response latency in ms

    # AI engine lifecycle signals
    ai_generation_started = Signal(str)  # prompt/query
    ai_token_received = Signal(str)  # streamed chunk
    ai_generation_finished = Signal(str)  # full response
    ai_error_occurred = Signal(str)  # error description
    ai_engine_changed = Signal(str)  # "local" or "cloud"
    ai_models_updated = Signal(str)  # engine_type
    routing_decision_updated = Signal(dict)  # routing decision data packet

    # Memory system signals
    memory_proposal_created = Signal(dict)  # proposal data
    memory_domain_updated = Signal(str)  # 'personal' or 'work'

    # Security & Permission signals
    permission_requested = Signal(dict)  # request details
    permission_resolved = Signal(str, bool)  # request_id, approved

    # System resource / hardware monitoring signals
    system_metrics_updated = Signal(dict)  # cpu, ram, storage metrics

    # Phase 7 Windows Computer Control signals
    windows_command_output = Signal(str, str)  # (cmd_id, text_chunk)
    windows_command_finished = Signal(dict)  # (command_result_dict)
    windows_action_planned = Signal(dict)  # (action_plan_dict)
    windows_confirmation_requested = Signal(dict)  # (action_plan_dict)
    windows_confirmation_resolved = Signal(str, bool)  # (plan_id, approved)
    windows_elevation_requested = Signal(dict)  # (elevation_request_dict)
    windows_download_progress = Signal(dict)  # (download_item_dict)
    windows_emergency_stop_triggered = Signal()

    # Phase 8 Offline Operation & Disaster Recovery signals
    network_connectivity_changed = Signal(bool)  # is_online
    backup_completed = Signal(dict)  # (backup_manifest_dict)
    backup_failed = Signal(str)  # (error_message)
    recovery_operation_completed = Signal(dict)  # (recovery_report_dict)
    database_health_changed = Signal(dict)  # (database_health_dict)


# Singleton instance
_events_instance: Optional[ApplicationEvents] = None


def get_app_events() -> ApplicationEvents:
    """Return the singleton instance of ApplicationEvents."""
    global _events_instance
    if _events_instance is None:
        _events_instance = ApplicationEvents()
    return _events_instance
