"""UI Views for HAKS Application."""

from ui.views.chat_view import ChatView
from ui.views.history_view import HistoryView
from ui.views.personal_memory_view import PersonalMemoryView
from ui.views.work_memory_view import WorkMemoryView
from ui.views.voice_view import VoiceView
from ui.views.ai_engine_view import AIEngineView
from ui.views.settings_view import SettingsView

__all__ = [
    "ChatView",
    "HistoryView",
    "PersonalMemoryView",
    "WorkMemoryView",
    "VoiceView",
    "AIEngineView",
    "SettingsView",
]
