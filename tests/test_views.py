"""Unit tests for UI views, ChatView, and FloatingAssistant."""

import sys
import pytest
from PySide6.QtWidgets import QApplication

from config.settings import AppConfig
from core.app_controller import AppController
from storage.database import DatabaseManager
from ui.floating_assistant import FloatingAssistant
from ui.main_window import MainWindow
from ui.views.chat_view import ChatView
from ui.views.history_view import HistoryView
from ui.views.personal_memory_view import PersonalMemoryView
from ui.views.work_memory_view import WorkMemoryView
from ui.views.voice_view import VoiceView
from ui.views.ai_engine_view import AIEngineView
from ui.views.settings_view import SettingsView





@pytest.fixture
def controller(temp_workspace: AppConfig, test_db_manager: DatabaseManager) -> AppController:
    """Fixture providing an isolated AppController."""
    return AppController(config=temp_workspace, db_manager=test_db_manager)


def test_chat_view_messaging(qapp, controller: AppController):
    """Verify ChatView renders messages, receives input, and dispatches via controller."""
    chat = ChatView(controller)

    # Enter text and trigger send
    chat.text_input.setPlainText("Test question from pytest")
    chat._on_send_message()

    # Verify input cleared
    assert chat.text_input.toPlainText() == ""

    # Verify message recorded in active conversation
    conv_id = controller.get_active_conversation_id()
    msgs = controller.load_conversation(conv_id)
    assert any(m["content"] == "Test question from pytest" for m in msgs)

    # Test mic toggle
    chat._on_toggle_mic()
    assert controller.is_mic_muted() is True
    assert "Muted" in chat.btn_mic.text()
    chat._on_toggle_mic()
    assert controller.is_mic_muted() is False

    chat.close()


def test_floating_assistant_shared_controller(qapp, controller: AppController):
    """Verify FloatingAssistant interacts with the shared controller and syncs messages."""
    floating = FloatingAssistant(controller=controller)

    # Test quick input send
    floating.quick_input.setText("Message from Floating Assistant")
    floating._on_quick_send()

    assert floating.quick_input.text() == ""

    # Verify message appears in active conversation
    conv_id = controller.get_active_conversation_id()
    msgs = controller.load_conversation(conv_id)
    assert any(m["content"] == "Message from Floating Assistant" for m in msgs)

    # Test always on top toggle
    prev = controller.get_always_on_top()
    floating._toggle_always_on_top()
    assert controller.get_always_on_top() == (not prev)

    floating.close()


def test_history_view_operations(qapp, controller: AppController):
    """Verify HistoryView lists conversations, filters, and supports opening."""
    # Create two conversations
    c1 = controller.create_new_conversation("PySide6 UI Architecture")
    c2 = controller.create_new_conversation("Voice Tamil STT Plan")

    view = HistoryView(controller)
    assert view.list_widget.count() >= 2

    # Filter
    view._filter_list("Tamil")
    assert view.list_widget.count() == 1

    view._filter_list("")
    assert view.list_widget.count() >= 2

    view.close()


def test_personal_and_work_memory_views(qapp, controller: AppController):
    """Verify PersonalMemoryView and WorkMemoryView operate independently."""
    pm_view = PersonalMemoryView(controller)
    wm_view = WorkMemoryView(controller)

    # Add personal memory
    controller.add_personal_memory(title="Dietary Preference", content="Vegetarian food only")
    pm_view.refresh_memories()
    assert pm_view.list_widget.count() == 1

    # Add work memory
    controller.add_work_memory(project_name="HAKS", title="VAD Threshold", content="Set Silero VAD to 0.5")
    wm_view.refresh_memories()
    assert wm_view.list_widget.count() == 1

    # Clear work memory
    controller.clear_work_memory()
    wm_view.refresh_memories()
    pm_view.refresh_memories()

    assert wm_view.list_widget.count() == 0
    assert pm_view.list_widget.count() == 1  # Personal memory untouched!

    pm_view.close()
    wm_view.close()


def test_voice_view_and_state_transitions(qapp, controller: AppController):
    """Verify VoiceView transitions through voice states."""
    voice_view = VoiceView(controller)

    voice_view._set_state("LISTENING")
    assert controller.get_voice_state() == "LISTENING"
    assert "LISTENING" in voice_view.lbl_state_display.text()

    voice_view._set_state("THINKING")
    assert controller.get_voice_state() == "THINKING"

    voice_view._set_state("STANDBY")
    assert controller.get_voice_state() == "STANDBY"

    voice_view.close()


def test_settings_view_theme_toggle(qapp, controller: AppController):
    """Verify SettingsView toggles application theme."""
    settings_view = SettingsView(controller)

    settings_view._on_theme_changed(1)  # Light
    assert controller.get_theme() == "light"

    settings_view._on_theme_changed(0)  # Dark
    assert controller.get_theme() == "dark"

    settings_view.close()


def test_ai_engine_view_controls(qapp, controller: AppController, monkeypatch):
    """Verify AIEngineView controls, provider selection, and default engine configuration."""
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.StandardButton.Yes)

    view = AIEngineView(controller)

    # 1. Switch default engine to Cloud
    view.combo_default_engine.setCurrentIndex(1)
    assert controller.get_selected_ai_engine() == "cloud"

    # 2. Switch default engine back to Local
    view.combo_default_engine.setCurrentIndex(0)
    assert controller.get_selected_ai_engine() == "local"

    # 3. Cloud provider switching
    view.cloud_provider_combo.setCurrentIndex(2)  # Groq
    assert controller.get_ai_manager().cloud_engine.active_provider_id == "groq"

    # 4. Base URL update
    view.txt_cloud_base_url.setText("https://test.groq.endpoint/v1")
    view._on_save_cloud_base_url()
    adapter = controller.get_ai_manager().cloud_engine.get_active_adapter()
    assert adapter.base_url == "https://test.groq.endpoint/v1"

    # 5. Local model selection
    view.local_model_combo.setCurrentText("qwen2.5:1.5b")
    assert controller.get_ai_manager().local_engine.current_model == "qwen2.5:1.5b"

    view.close()
