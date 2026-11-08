"""UI Integration tests for ChatView streaming, engine selection, and cancellation."""

import time
import pytest
from PySide6.QtWidgets import QApplication

from config.settings import AppConfig
from core.app_controller import AppController
from storage.database import DatabaseManager
from ui.views.chat_view import ChatView


@pytest.fixture
def controller(temp_workspace: AppConfig, test_db_manager: DatabaseManager) -> AppController:
    """Fixture providing an isolated AppController."""
    return AppController(config=temp_workspace, db_manager=test_db_manager)


@pytest.fixture(autouse=True)
def mock_worker_start(monkeypatch):
    """Prevent unmocked background streaming threads during UI tests."""
    monkeypatch.setattr("ui.views.chat_view.AIInferenceWorker.start", lambda self: None)


def test_chat_view_engine_selector(qapp, controller: AppController):
    """Verify engine and model combo boxes update settings and reflect options."""
    chat = ChatView(controller)

    # Initial engine should default to local
    assert chat.combo_engine.count() == 2
    assert "Local AI" in chat.combo_engine.itemText(0)
    assert "Cloud AI" in chat.combo_engine.itemText(1)

    # Switch to Cloud
    chat.combo_engine.setCurrentIndex(1)
    assert controller.get_selected_ai_engine() == "cloud"
    assert chat.combo_model.count() >= 1

    # Switch back to Local
    chat.combo_engine.setCurrentIndex(0)
    assert controller.get_selected_ai_engine() == "local"
    assert "llama3.2:1b" in [chat.combo_model.itemText(i) for i in range(chat.combo_model.count())]

    chat.close()


def test_chat_view_streaming_token_accumulation(qapp, controller: AppController):
    """Verify streaming token chunks append smoothly to the active message bubble."""
    chat = ChatView(controller)

    # Manually trigger streaming placeholder
    chat._start_ai_inference()

    assert chat._is_generating is True
    assert chat.btn_send.isEnabled() is False
    assert not chat.btn_stop.isHidden()
    assert chat._current_streaming_lbl is not None

    # Simulate arrival of streaming chunks
    chat._on_worker_chunk("Hello")
    assert chat._current_streaming_text == "Hello"
    assert chat._current_streaming_lbl.text() == "Hello"

    chat._on_worker_chunk(" world!")
    assert chat._current_streaming_text == "Hello world!"
    assert chat._current_streaming_lbl.text() == "Hello world!"

    # Simulate completion
    chat._on_worker_finished("Hello world!", {"engine": "Local AI", "model": "llama3.2:1b"})

    assert chat._is_generating is False
    assert chat.btn_send.isEnabled() is True
    assert chat.btn_stop.isVisible() is False

    # Verify assistant message was saved to conversation database
    conv_id = controller.get_active_conversation_id()
    messages = controller.load_conversation(conv_id, emit_event=False)
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]
    assert any("Hello world!" in m["content"] for m in assistant_msgs)

    chat.close()


def test_chat_view_cancellation(qapp, controller: AppController):
    """Verify user clicking Stop button triggers cancellation and records partial message."""
    chat = ChatView(controller)

    chat._start_ai_inference()
    chat._on_worker_chunk("Partial response before user stop")

    # User clicks stop
    chat._on_worker_cancelled()

    assert chat._is_generating is False
    assert chat.btn_send.isEnabled() is True
    assert "Generation stopped by user" in chat._current_streaming_lbl.text()

    conv_id = controller.get_active_conversation_id()
    messages = controller.load_conversation(conv_id, emit_event=False)
    assert any("Generation stopped by user" in m["content"] for m in messages)

    chat.close()


def test_chat_view_error_presentation(qapp, controller: AppController):
    """Verify structured error messages are presented visually in chat bubble."""
    chat = ChatView(controller)

    chat._start_ai_inference()
    error_text = "Ollama connection refused at http://127.0.0.1:11434"
    chat._on_worker_error(error_text)

    assert chat._is_generating is False
    assert chat.btn_send.isEnabled() is True
    assert "AI Engine Error" in chat._current_streaming_lbl.text()
    assert error_text in chat._current_streaming_lbl.text()

    chat.close()


def test_conversation_context_privacy(controller: AppController):
    """Verify privacy guarantee: only conversation context is provided to AI engines.

    Personal Memory and Work Memory databases are NEVER included in AI context.
    """
    # 1. Populate Personal Memory
    controller.add_personal_memory(
        title="Secret Medical Info",
        content="Patient has allergy to penicillin",
        category="health",
    )

    # 2. Populate Work Memory
    controller.add_work_memory(
        project_name="Confidential Project",
        title="Proprietary Algorithm",
        content="Core patent secrets",
    )

    # 3. Create conversation messages
    controller.send_message("What is the weather today?", role="user")
    controller.send_message("It is sunny.", role="assistant")
    controller.send_message("Thank you!", role="user")

    # 4. Fetch context prepared for AI engine
    ai_msgs = controller.get_conversation_ai_messages()

    # Assert only conversation messages are present
    assert len(ai_msgs) == 3
    contents = [m.content for m in ai_msgs]
    assert "What is the weather today?" in contents
    assert "It is sunny." in contents
    assert "Thank you!" in contents

    # Strictly assert NO memory data was leaked into context
    for c in contents:
        assert "Secret Medical Info" not in c
        assert "penicillin" not in c
        assert "Proprietary Algorithm" not in c
        assert "patent secrets" not in c


def test_chat_view_retry_flow(qapp, controller: AppController):
    """Verify Retry button appears upon inference failure and can re-trigger inference."""
    view = ChatView(controller=controller)

    # Initial state: retry button hidden
    assert view.btn_retry.isHidden() is True

    # Simulate sending message
    view.text_input.setPlainText("Prompt that will fail")
    view._on_send_message()
    assert view._last_user_prompt == "Prompt that will fail"

    # Simulate worker error
    view._on_worker_error("Connection refused by Ollama daemon")
    assert not view.btn_retry.isHidden()

    # User clicks retry
    view.btn_retry.click()
    assert view.btn_retry.isHidden() is True
    assert view._is_generating is True

    view.close()
