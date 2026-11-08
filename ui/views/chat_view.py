"""Chat View component for HAKS.

Provides a real-time conversational interface with streaming AI responses,
message bubbles, non-blocking asynchronous execution, engine & model selectors,
and interactive cancellation control.
"""

import os
from datetime import datetime, timezone
from typing import Optional, Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai.models import AIMessage
from ai.worker import AIInferenceWorker
from core.app_controller import AppController, get_app_controller
from core.logger import get_logger
from privacy.consent_manager import ConsentDecision
from routing.models import RoutingDecision, RoutingMode

logger = get_logger("ui.views.chat_view")


class ChatInputEdit(QTextEdit):
    """Custom QTextEdit that triggers send on Enter without Shift."""

    def __init__(self, on_send_callback, parent=None):
        super().__init__(parent)
        self.on_send_callback = on_send_callback
        self.setPlaceholderText("Type a message to HAKS... (Enter to send, Shift+Enter for newline)")
        self.setFixedHeight(72)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                event.accept()
                self.on_send_callback()
                return
        super().keyPressEvent(event)


class ChatView(QWidget):
    """Interactive conversation interface with streaming AI integration."""

    def __init__(self, controller: Optional[AppController] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller or get_app_controller()
        self._worker: Optional[AIInferenceWorker] = None
        self._is_generating: bool = False
        self._current_streaming_text: str = ""
        self._current_streaming_lbl: Optional[QLabel] = None
        self._current_streaming_time_lbl: Optional[QLabel] = None
        self._just_finished_msg_id: Optional[str] = None
        self._last_user_prompt: str = ""

        self._init_ui()
        self._connect_signals()
        self.refresh_active_conversation()
        self._refresh_engine_and_models()

    def _init_ui(self) -> None:
        """Construct the chat view layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(10)

        # 1. Top Header Bar
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self.lbl_chat_title = QLabel("Conversation")
        self.lbl_chat_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #f0f6fc;")
        header_layout.addWidget(self.lbl_chat_title)

        header_layout.addSpacing(10)

        # Routing Mode selector
        lbl_routing = QLabel("Routing:")
        lbl_routing.setStyleSheet("color: #8b949e; font-size: 12px; font-weight: 600;")
        header_layout.addWidget(lbl_routing)

        self.combo_routing_mode = QComboBox()
        self.combo_routing_mode.addItem("🤖 Auto (Intelligent)", "auto")
        self.combo_routing_mode.addItem("⚙️ Manual", "manual")
        self.combo_routing_mode.addItem("🔒 Force Local-Only", "force_local")
        self.combo_routing_mode.setFixedWidth(160)
        self.combo_routing_mode.currentIndexChanged.connect(self._on_routing_mode_changed)
        header_layout.addWidget(self.combo_routing_mode)

        # Engine selector
        lbl_engine = QLabel("Engine:")
        lbl_engine.setStyleSheet("color: #8b949e; font-size: 12px; font-weight: 600;")
        header_layout.addWidget(lbl_engine)

        self.combo_engine = QComboBox()
        self.combo_engine.addItem("Local AI (Ollama)", "local")
        self.combo_engine.addItem("Cloud AI", "cloud")
        self.combo_engine.setFixedWidth(140)
        self.combo_engine.currentIndexChanged.connect(self._on_engine_selection_changed)
        header_layout.addWidget(self.combo_engine)

        # Model selector
        lbl_model = QLabel("Model:")
        lbl_model.setStyleSheet("color: #8b949e; font-size: 12px; font-weight: 600;")
        header_layout.addWidget(lbl_model)

        self.combo_model = QComboBox()
        self.combo_model.setFixedWidth(150)
        self.combo_model.currentIndexChanged.connect(self._on_model_selection_changed)
        header_layout.addWidget(self.combo_model)

        # Routing status badge
        self.lbl_routing_badge = QLabel("[Auto: Ready]")
        self.lbl_routing_badge.setStyleSheet(
            "color: #58a6ff; font-size: 11px; font-weight: 600; background-color: #161b22; "
            "border: 1px solid #30363d; border-radius: 4px; padding: 2px 6px;"
        )
        self.lbl_routing_badge.setToolTip("Intelligent AI router evaluates privacy classification and reasoning complexity.")
        header_layout.addWidget(self.lbl_routing_badge)

        # Health badge
        self.lbl_engine_health = QLabel("● Ready")
        self.lbl_engine_health.setStyleSheet("color: #3fb950; font-size: 11px; font-weight: bold; margin-left: 2px;")
        header_layout.addWidget(self.lbl_engine_health)

        header_layout.addStretch()

        self.btn_new_chat = QPushButton("+ New Chat")
        self.btn_new_chat.setProperty("class", "btn_secondary")
        self.btn_new_chat.clicked.connect(self._on_new_chat)
        header_layout.addWidget(self.btn_new_chat)

        self.btn_clear_chat = QPushButton("Clear")
        self.btn_clear_chat.setProperty("class", "btn_danger")
        self.btn_clear_chat.clicked.connect(self._on_clear_chat)
        header_layout.addWidget(self.btn_clear_chat)

        layout.addWidget(header)

        # 2. Scroll Area for Messages
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background: transparent;")

        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(0, 10, 0, 10)
        self.messages_layout.setSpacing(12)
        self.messages_layout.addStretch()

        self.scroll_area.setWidget(self.messages_container)
        layout.addWidget(self.scroll_area, stretch=1)

        # 3. Input Controls Area
        input_container = QFrame()
        input_container.setProperty("class", "stat_card")
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(8)

        # Memory Proposal Notification Banner (AI suggests; user approves)
        self.proposal_notification = QFrame()
        self.proposal_notification.setStyleSheet(
            "background-color: #1a2332; border: 1px solid #388bfd; border-radius: 6px; padding: 6px 10px;"
        )
        prop_notif_layout = QHBoxLayout(self.proposal_notification)
        prop_notif_layout.setContentsMargins(4, 2, 4, 2)
        self.lbl_prop_notif = QLabel("💡 New memory candidate detected awaiting your approval.")
        self.lbl_prop_notif.setStyleSheet("color: #58a6ff; font-size: 11px; font-weight: 600;")
        self.btn_review_prop = QPushButton("Review in Memory")
        self.btn_review_prop.setProperty("class", "btn_primary")
        self.btn_review_prop.setFixedHeight(26)
        self.btn_review_prop.clicked.connect(self._on_review_proposal_clicked)

        self.btn_dismiss_prop = QPushButton("✕")
        self.btn_dismiss_prop.setProperty("class", "btn_secondary")
        self.btn_dismiss_prop.setFixedSize(26, 26)
        self.btn_dismiss_prop.clicked.connect(lambda: self.proposal_notification.setVisible(False))

        prop_notif_layout.addWidget(self.lbl_prop_notif, stretch=1)
        prop_notif_layout.addWidget(self.btn_review_prop)
        prop_notif_layout.addWidget(self.btn_dismiss_prop)
        input_layout.addWidget(self.proposal_notification)
        self.proposal_notification.setVisible(False)

        # Multiline text input
        self.text_input = ChatInputEdit(self._on_send_message)
        input_layout.addWidget(self.text_input)

        # Bottom buttons row
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)

        self.btn_mic = QPushButton("🎤 Mic: Active")
        self.btn_mic.setProperty("class", "btn_secondary")
        self.btn_mic.clicked.connect(self._on_toggle_mic)
        btn_row.addWidget(self.btn_mic)

        self.lbl_input_hint = QLabel("Tamil, English & Mixed speech supported")
        self.lbl_input_hint.setStyleSheet("color: #7d8590; font-size: 11px;")
        btn_row.addWidget(self.lbl_input_hint)

        self.lbl_generation_status = QLabel("")
        self.lbl_generation_status.setStyleSheet("color: #58a6ff; font-size: 11px; font-weight: 600; margin-left: 10px;")
        btn_row.addWidget(self.lbl_generation_status)

        btn_row.addStretch()

        # Stop button (shown only during generation)
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setProperty("class", "btn_danger")
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self._on_stop_generation)
        btn_row.addWidget(self.btn_stop)

        # Retry button (shown on failure or cancellation)
        self.btn_retry = QPushButton("🔄 Retry")
        self.btn_retry.setProperty("class", "btn_secondary")
        self.btn_retry.setVisible(False)
        self.btn_retry.setToolTip("Retry generating the last message with the selected engine")
        self.btn_retry.clicked.connect(self._on_retry_last_message)
        btn_row.addWidget(self.btn_retry)

        self.btn_send = QPushButton("Send Message")
        self.btn_send.setProperty("class", "btn_primary")
        self.btn_send.clicked.connect(self._on_send_message)
        btn_row.addWidget(self.btn_send)

        input_layout.addLayout(btn_row)
        layout.addWidget(input_container)

    def _connect_signals(self) -> None:
        """Connect to application-wide events."""
        self.controller.events.message_received.connect(self._on_message_received)
        self.controller.events.conversation_loaded.connect(self._on_conversation_loaded)
        self.controller.events.voice_mute_toggled.connect(self._on_voice_mute_toggled)
        self.controller.events.ai_engine_changed.connect(self._on_ai_engine_event)
        self.controller.events.memory_proposal_created.connect(self._on_memory_proposal_created)

    def _on_memory_proposal_created(self, prop_data: dict) -> None:
        """Surface proposal banner when an automatic memory proposal is detected."""
        domain = prop_data.get("domain", "personal")
        snippet = prop_data.get("proposal_text", "")[:45] + "..."
        self.lbl_prop_notif.setText(f"💡 Detected {domain.capitalize()} Memory candidate: \"{snippet}\"")
        self._last_proposal_domain = domain
        self.proposal_notification.setVisible(True)

    def _on_review_proposal_clicked(self) -> None:
        """Navigate user to the appropriate memory view for review."""
        target_page = "personal_memory" if getattr(self, "_last_proposal_domain", "personal") == "personal" else "work_memory"
        self.proposal_notification.setVisible(False)
        self.controller.events.navigate_to_page.emit(target_page)

    def _on_routing_mode_changed(self, index: int) -> None:
        """Handle user changing routing mode."""
        mode_data = self.combo_routing_mode.currentData()
        if mode_data == "force_local":
            self.combo_engine.setEnabled(False)
            self.combo_engine.setCurrentIndex(0)
            self.lbl_routing_badge.setText("[Forced Local: Active]")
            self.lbl_routing_badge.setStyleSheet(
                "color: #f85149; font-size: 11px; font-weight: 600; background-color: #161b22; "
                "border: 1px solid #f85149; border-radius: 4px; padding: 2px 6px;"
            )
            self.lbl_routing_badge.setToolTip("Strict offline mode enforced: all queries are routed to Local AI.")
        elif mode_data == "manual":
            self.combo_engine.setEnabled(True)
            eng = "Local AI" if self.combo_engine.currentIndex() == 0 else "Cloud AI"
            self.lbl_routing_badge.setText(f"[Manual: {eng}]")
            self.lbl_routing_badge.setStyleSheet(
                "color: #d29922; font-size: 11px; font-weight: 600; background-color: #161b22; "
                "border: 1px solid #d29922; border-radius: 4px; padding: 2px 6px;"
            )
            self.lbl_routing_badge.setToolTip("Manual override mode: routing strictly follows the Engine selector.")
        else:
            self.combo_engine.setEnabled(True)
            self.lbl_routing_badge.setText("[Auto Route: Ready]")
            self.lbl_routing_badge.setStyleSheet(
                "color: #58a6ff; font-size: 11px; font-weight: 600; background-color: #161b22; "
                "border: 1px solid #30363d; border-radius: 4px; padding: 2px 6px;"
            )
            self.lbl_routing_badge.setToolTip("Intelligent AI router evaluates privacy classification and reasoning complexity.")

    def _update_routing_badge(self, decision: RoutingDecision) -> None:
        """Update routing status label and tooltip based on decision."""
        color_map = {
            "PUBLIC": "#3fb950",
            "PERSONAL": "#d29922",
            "SENSITIVE": "#f85149",
        }
        level_str = decision.privacy.level.value
        col = color_map.get(level_str, "#58a6ff")
        eng = "Local AI" if decision.selected_engine == "local" else "Cloud AI"
        self.lbl_routing_badge.setText(f"Route: {eng} | {level_str} | {decision.complexity.category.value}")
        self.lbl_routing_badge.setStyleSheet(
            f"color: {col}; font-size: 11px; font-weight: 600; background-color: #161b22; "
            f"border: 1px solid {col}; border-radius: 4px; padding: 2px 6px;"
        )
        self.lbl_routing_badge.setToolTip(f"Routing Reason: {decision.routing_reason}\n\nModel: {decision.selected_model}")

    def _get_current_engine_type(self) -> str:
        """Return 'local' or 'cloud'."""
        data = self.combo_engine.currentData()
        return str(data) if data else "local"

    def _on_engine_selection_changed(self, index: int) -> None:
        """Handle user changing engine via combo box."""
        engine_type = self._get_current_engine_type()
        self.controller.set_selected_ai_engine(engine_type)
        self._refresh_models_for_engine(engine_type)

    def _on_ai_engine_event(self, engine_type: str) -> None:
        """Sync UI when engine changes from elsewhere."""
        target_idx = 0 if engine_type == "local" else 1
        if self.combo_engine.currentIndex() != target_idx:
            self.combo_engine.blockSignals(True)
            self.combo_engine.setCurrentIndex(target_idx)
            self.combo_engine.blockSignals(False)
            self._refresh_models_for_engine(engine_type)

    def _refresh_engine_and_models(self) -> None:
        """Initialize engine selector and model options from config."""
        saved_engine = self.controller.get_selected_ai_engine()
        idx = 1 if saved_engine == "cloud" else 0
        self.combo_engine.blockSignals(True)
        self.combo_engine.setCurrentIndex(idx)
        self.combo_engine.blockSignals(False)
        self._refresh_models_for_engine(saved_engine)

    def _refresh_models_for_engine(self, engine_type: str) -> None:
        """Populate model dropdown based on active engine."""
        self.combo_model.blockSignals(True)
        self.combo_model.clear()

        mgr = self.controller.get_ai_manager()
        if engine_type == "local":
            current = mgr.local_engine.current_model
            installed = [m["name"] for m in mgr.local_engine._installed_models_cache if "name" in m]
            models = installed or ["llama3.2:1b", "qwen2.5:1.5b", "llama3.2:3b"]
            for m in models:
                self.combo_model.addItem(m, m)
            idx = self.combo_model.findText(current)
            if idx >= 0:
                self.combo_model.setCurrentIndex(idx)
        else:
            current = mgr.cloud_engine.current_model
            adapter = mgr.cloud_engine.get_active_adapter()
            models = adapter.get_supported_models() if adapter else ["gemini-1.5-flash", "gemini-1.5-pro"]
            for m in models:
                self.combo_model.addItem(m, m)
            idx = self.combo_model.findText(current)
            if idx >= 0:
                self.combo_model.setCurrentIndex(idx)

        self.combo_model.blockSignals(False)
        self._probe_health_async(engine_type)

    def _on_model_selection_changed(self, index: int) -> None:
        """Update active model in AI engine."""
        model_name = self.combo_model.currentText()
        if not model_name:
            return
        mgr = self.controller.get_ai_manager()
        engine_type = self._get_current_engine_type()
        if engine_type == "local":
            mgr.local_engine.set_model(model_name)
        else:
            mgr.cloud_engine.set_model(model_name)

    def _probe_health_async(self, engine_type: str) -> None:
        """Lightweight non-blocking health indicator check."""
        import asyncio

        async def _check():
            try:
                mgr = self.controller.get_ai_manager()
                engine = mgr.get_engine(engine_type)
                health = await engine.check_health()
                if health.is_healthy:
                    self.lbl_engine_health.setText("● Ready")
                    self.lbl_engine_health.setStyleSheet("color: #3fb950; font-size: 11px; font-weight: bold;")
                else:
                    self.lbl_engine_health.setText("● Offline / Not Configured")
                    self.lbl_engine_health.setStyleSheet("color: #d29922; font-size: 11px; font-weight: bold;")
            except Exception:
                self.lbl_engine_health.setText("● Offline")
                self.lbl_engine_health.setStyleSheet("color: #f85149; font-size: 11px; font-weight: bold;")

        def _schedule_probe():
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    asyncio.create_task(_check())
            except RuntimeError:
                pass

        from PySide6.QtCore import QCoreApplication, QThread
        if QCoreApplication.instance() is not None and QThread.currentThread().eventDispatcher() is not None:
            QTimer.singleShot(50, _schedule_probe)

    def refresh_active_conversation(self) -> None:
        """Load messages for the currently active conversation."""
        self._clear_bubbles_layout()

        conv_id = self.controller.get_active_conversation_id()
        messages = self.controller.load_conversation(conv_id, emit_event=False)
        for msg in messages:
            self._render_message_bubble(msg["role"], msg["content"], msg["timestamp"])

        self._scroll_to_bottom()

    def _clear_bubbles_layout(self) -> None:
        """Remove all widget bubbles from layout."""
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _render_message_bubble(self, role: str, content: str, timestamp: str = "") -> tuple[QLabel, QLabel]:
        """Create and append a message bubble. Returns (content_lbl, time_lbl)."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        bubble = QFrame()
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 8, 12, 8)
        bubble_layout.setSpacing(4)

        if role == "user":
            bubble.setProperty("class", "msg_bubble_user")
            role_lbl = QLabel("You")
            role_lbl.setProperty("class", "msg_role_label")
            row_layout.addStretch()
            row_layout.addWidget(bubble)
            bubble.setMaximumWidth(680)
        elif role == "system_error":
            bubble.setStyleSheet("background-color: #3b1d1d; border: 1px solid #f85149; border-radius: 8px;")
            role_lbl = QLabel("System Error")
            role_lbl.setStyleSheet("color: #ff7b72; font-weight: bold; font-size: 11px;")
            row_layout.addWidget(bubble)
            row_layout.addStretch()
            bubble.setMaximumWidth(680)
        else:
            bubble.setProperty("class", "msg_bubble_assistant")
            role_lbl = QLabel("HAKS")
            role_lbl.setProperty("class", "msg_role_label msg_role_assistant")
            row_layout.addWidget(bubble)
            row_layout.addStretch()
            bubble.setMaximumWidth(680)

        content_lbl = QLabel(content)
        content_lbl.setProperty("class", "msg_content_label")
        content_lbl.setWordWrap(True)
        content_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        time_lbl = QLabel(timestamp)
        time_lbl.setProperty("class", "msg_time_label")

        bubble_layout.addWidget(role_lbl)
        bubble_layout.addWidget(content_lbl)
        bubble_layout.addWidget(time_lbl)

        # Insert before stretch item at the end
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, row_widget)
        return content_lbl, time_lbl

    def _on_send_message(self) -> None:
        """Send message from input box and begin streaming inference."""
        if self._is_generating:
            return

        text = self.text_input.toPlainText().strip()
        if not text:
            return

        self.text_input.clear()
        self._last_user_prompt = text
        self.btn_retry.setVisible(False)

        # 1. Record user message in shared controller & conversation database
        self.controller.send_message(text, role="user")

        # 2. Check for computer control instruction
        ctrl = getattr(self.controller, "windows_control", None)
        if ctrl is not None:
            is_explicit_cmd = text.startswith(("/cmd ", "/ps ", "/powershell ", "/run ", "/control "))
            plan = ctrl.interpreter.interpret(text)
            if is_explicit_cmd or plan.is_actionable:
                self._execute_computer_instruction(text, plan)
                return

        # 3. Trigger asynchronous AI inference
        self._start_ai_inference()

    def _execute_computer_instruction(self, text: str, plan: Any) -> None:
        """Execute a Windows computer control instruction asynchronously and display output."""
        import asyncio
        from datetime import datetime, timezone

        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        summary_msg = f"⚙️ Executing system action: {plan.summary}..."
        content_lbl, time_lbl = self._render_message_bubble("assistant", summary_msg, now_str)
        self._scroll_to_bottom()

        async def _run():
            try:
                res = await self.controller.execute_windows_instruction(text, source="chat")
                out_parts = []
                if res.stdout:
                    out_parts.append(f"Output:\n```\n{res.stdout.strip()}\n```")
                if res.stderr:
                    out_parts.append(f"Error details:\n```\n{res.stderr.strip()}\n```")

                status_icon = "✅" if res.success else "❌"
                status_text = "Completed successfully" if res.success else f"Failed (exit code {res.exit_code})"
                if res.error:
                    status_text += f": {res.error}"

                body = f"{status_icon} **{plan.summary}** — {status_text}"
                if out_parts:
                    body += "\n\n" + "\n\n".join(out_parts)

                content_lbl.setText(body)
                msg_data = self.controller.send_message(body, role="assistant", metadata={"action_plan": plan.summary})
                self._just_finished_msg_id = msg_data["id"]
            except Exception as exc:
                err_text = f"❌ System execution error: {exc}"
                content_lbl.setText(err_text)
                msg_data = self.controller.send_message(err_text, role="system_error")
                self._just_finished_msg_id = msg_data["id"]
            finally:
                self._scroll_to_bottom()

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.create_task(_run())
        except RuntimeError:
            pass

    def _on_retry_last_message(self) -> None:
        """Retry generating the last message with currently selected engine/model."""
        if self._is_generating:
            return
        self.btn_retry.setVisible(False)
        self._start_ai_inference()

    def _start_ai_inference(self) -> None:
        """Evaluate intelligent routing, privacy, consent, and initiate streaming inference."""
        prompt = self._last_user_prompt

        # 1. Determine effective routing mode
        r_mode_str = self.combo_routing_mode.currentData() if hasattr(self, "combo_routing_mode") else "auto"
        if r_mode_str == "force_local":
            effective_mode = RoutingMode.FORCE_LOCAL
        elif r_mode_str == "manual":
            engine_choice = self._get_current_engine_type()
            effective_mode = RoutingMode.MANUAL_CLOUD if engine_choice == "cloud" else RoutingMode.MANUAL_LOCAL
        else:
            effective_mode = RoutingMode.AUTO

        # 2. Build preliminary memory context
        mem_context = ""
        if prompt:
            try:
                mem_context = self.controller.context_builder.build_context(
                    query=prompt,
                    include_personal=True,
                    include_work=True,
                    top_k_per_domain=2,
                    min_relevance=0.25,
                )
            except Exception as exc:
                logger.warning("Could not build preliminary memory context: %s", exc)

        # 3. Authoritative Routing Decision
        decision = self.controller.route_request(
            user_prompt=prompt,
            mode=effective_mode,
            memory_context=mem_context,
        )
        self._update_routing_badge(decision)

        # 4. Handle Engine Unavailable (NEVER silent fallback)
        target_engine = decision.selected_engine
        is_headless = bool(os.environ.get("PYTEST_CURRENT_TEST")) or (
            QApplication.instance() is not None
            and QApplication.instance().platformName() == "offscreen"
        )
        if effective_mode == RoutingMode.AUTO and decision.is_fallback_needed and prompt and not is_headless:
            from ui.dialogs.consent_dialog import FallbackApprovalDialog
            dialog = FallbackApprovalDialog(
                target_engine=decision.selected_engine,
                fallback_engine=decision.fallback_engine or "local",
                reason=decision.fallback_reason or "Service not reachable",
                parent=self,
            )
            dialog.exec()
            if not dialog.approved_fallback:
                now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
                self._render_message_bubble(
                    "system_error",
                    f"Request cancelled: Target engine '{decision.selected_engine}' is unavailable and fallback was declined.",
                    now_str,
                )
                self.btn_retry.setVisible(True)
                self._finish_generation_ui()
                return
            target_engine = decision.fallback_engine or "local"
            logger.info("User explicitly approved engine fallback to: %s", target_engine)
        elif effective_mode in (RoutingMode.MANUAL_LOCAL, RoutingMode.MANUAL_CLOUD, RoutingMode.FORCE_LOCAL):
            target_engine = "local" if effective_mode in (RoutingMode.MANUAL_LOCAL, RoutingMode.FORCE_LOCAL) else "cloud"

        # 5. Handle Consent & Redaction
        prompt_to_send = prompt
        if decision.requires_consent and decision.consent_request and target_engine == "cloud" and not is_headless:
            from ui.dialogs.consent_dialog import ConsentDialog
            consent_dlg = ConsentDialog(decision.consent_request, parent=self)
            consent_dlg.exec()
            decision_val = consent_dlg.decision

            if decision_val == ConsentDecision.APPROVED:
                self.controller.consent_manager.record_decision(decision.consent_request.id, ConsentDecision.APPROVED)
                self.controller.consent_manager.consume_consent(decision.consent_request.id)
                prompt_to_send = decision.consent_request.transmitted_data
                logger.info("Cloud transmission explicitly approved by user for request %s", decision.consent_request.id)
            elif decision_val == ConsentDecision.USE_LOCAL:
                self.controller.consent_manager.record_decision(decision.consent_request.id, ConsentDecision.USE_LOCAL)
                target_engine = "local"
                logger.info("User chose to use Local AI instead of cloud transmission.")
            else:
                self.controller.consent_manager.record_decision(decision.consent_request.id, ConsentDecision.CANCELLED)
                now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
                self._render_message_bubble(
                    "system_error",
                    "Cloud transmission cancelled by user.",
                    now_str,
                )
                self.btn_retry.setVisible(True)
                self._finish_generation_ui()
                return

        # 6. Rebuild memory context with final privacy and target engine constraints
        final_mem_context = ""
        try:
            final_mem_context = self.controller.context_builder.build_context(
                query=prompt_to_send,
                include_personal=True,
                include_work=True,
                top_k_per_domain=2,
                min_relevance=0.25,
                is_cloud_target=(target_engine == "cloud"),
                redact_for_cloud=decision.requires_redaction,
            )
        except Exception as exc:
            logger.warning("Could not build final memory context: %s", exc)

        # 7. Start generation UI
        self._is_generating = True
        self.btn_send.setEnabled(False)
        self.btn_stop.setVisible(True)

        engine_label = "Local AI" if target_engine == "local" else "Cloud AI"
        model_name = self.combo_model.currentText()
        self.lbl_generation_status.setText(f"● Generating ({engine_label} - {model_name})...")

        # Create placeholder assistant bubble
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self._current_streaming_text = ""
        self._current_streaming_lbl, self._current_streaming_time_lbl = self._render_message_bubble(
            "assistant", "● Thinking...", now_str
        )
        self._scroll_to_bottom()

        # Prepare typed AIMessage context
        messages = self.controller.get_conversation_ai_messages(limit=20)
        # Update last user message if sanitized
        if prompt_to_send != prompt and messages and messages[-1].role == "user":
            messages[-1].content = prompt_to_send

        if final_mem_context:
            from ai.models import AIMessage
            messages.insert(0, AIMessage(role="system", content=final_mem_context))

        # Launch background worker thread
        self._worker = AIInferenceWorker(
            messages=messages,
            engine_type=target_engine,
            manager=self.controller.get_ai_manager(),
            parent=self,
        )
        self._worker.signals.chunk_received.connect(self._on_worker_chunk)
        self._worker.signals.finished.connect(self._on_worker_finished)
        self._worker.signals.error.connect(self._on_worker_error)
        self._worker.signals.cancelled.connect(self._on_worker_cancelled)
        self._worker.start()

    def _on_worker_chunk(self, chunk_text: str) -> None:
        """Handle incoming token chunk from worker thread."""
        self._current_streaming_text += chunk_text
        if self._current_streaming_lbl:
            self._current_streaming_lbl.setText(self._current_streaming_text)
            self._scroll_to_bottom()

    def _on_worker_finished(self, full_text: str, metadata: dict) -> None:
        """Handle successful inference completion."""
        self._finish_generation_ui()
        self.btn_retry.setVisible(False)

        # Guard: handle empty AI response with diagnostic instead of saving empty message
        if not full_text.strip():
            engine_info = metadata.get("engine", "unknown")
            model_info = metadata.get("model", "unknown")
            err_display = (
                f"⚠️ AI engine returned an empty response.\n\n"
                f"Engine: {engine_info} | Model: {model_info}\n\n"
                f"Try sending your message again, or check engine health "
                f"in the AI Engine panel."
            )
            if self._current_streaming_lbl:
                self._current_streaming_lbl.setText(err_display)
                self._current_streaming_lbl.setStyleSheet("color: #ff7b72;")
            self.btn_retry.setVisible(True)
            logger.warning("Empty AI response received from %s (%s)", engine_info, model_info)
            return

        # Persist full assistant response into conversation database
        msg_data = self.controller.send_message(full_text, role="assistant", metadata=metadata)
        self._just_finished_msg_id = msg_data["id"]

        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        if self._current_streaming_time_lbl:
            self._current_streaming_time_lbl.setText(now_str)

        self._scroll_to_bottom()
        logger.info("Inference finished successfully (%d chars)", len(full_text))

    def _on_worker_error(self, error_message: str) -> None:
        """Handle inference failure."""
        self._finish_generation_ui()
        self.btn_retry.setVisible(True)

        err_display = (
            f"⚠️ **AI Engine Error**\n\n{error_message}\n\n"
            f"*Check engine health or switch engines using the selector above.*"
        )
        if self._current_streaming_lbl:
            self._current_streaming_lbl.setText(err_display)
            self._current_streaming_lbl.setStyleSheet("color: #ff7b72;")

        # Save notice to conversation so context remains coherent
        self.controller.send_message(err_display, role="assistant", metadata={"error": True})
        self._scroll_to_bottom()

    def _on_worker_cancelled(self) -> None:
        """Handle generation aborted by user."""
        self._finish_generation_ui()
        self.btn_retry.setVisible(True)

        partial = self._current_streaming_text.strip()
        final_text = f"{partial}\n\n*(Generation stopped by user)*" if partial else "*(Generation stopped by user)*"

        if self._current_streaming_lbl:
            self._current_streaming_lbl.setText(final_text)

        self.controller.send_message(final_text, role="assistant", metadata={"cancelled": True})
        self._scroll_to_bottom()

    def _on_stop_generation(self) -> None:
        """Stop active generation immediately."""
        if self._worker and self._worker.isRunning():
            self.lbl_generation_status.setText("● Stopping...")
            self.btn_stop.setEnabled(False)
            self._worker.cancel()

    def _finish_generation_ui(self) -> None:
        """Reset UI state after generation stops or finishes."""
        self._is_generating = False
        self.btn_send.setEnabled(True)
        self.btn_stop.setVisible(False)
        self.btn_stop.setEnabled(True)
        self.lbl_generation_status.setText("")

    def _on_message_received(self, conv_id: str, msg_id: str, role: str, content: str) -> None:
        """Handle message events, avoiding duplicate rendering for freshly streamed responses."""
        active_id = self.controller.get_active_conversation_id()
        if conv_id == active_id:
            # Skip if this is the assistant message we just rendered in the streaming bubble
            if self._just_finished_msg_id == msg_id:
                self._just_finished_msg_id = None
                return

            now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
            self._render_message_bubble(role, content, now_str)
            self._scroll_to_bottom()

    def _on_conversation_loaded(self, conv_id: str) -> None:
        """Reload conversation when selected from history."""
        self.refresh_active_conversation()

    def _on_new_chat(self) -> None:
        """Start a fresh conversation."""
        if self._is_generating:
            self._on_stop_generation()
        self.controller.create_new_conversation()
        self.refresh_active_conversation()

    def _on_clear_chat(self) -> None:
        """Clear current conversation messages."""
        if self._is_generating:
            self._on_stop_generation()
        conv_id = self.controller.get_active_conversation_id()
        self.controller.delete_conversation(conv_id)
        self.refresh_active_conversation()

    def _on_toggle_mic(self) -> None:
        """Toggle microphone mute."""
        is_muted = self.controller.toggle_mic_mute()
        self._update_mic_button(is_muted)

    def _on_voice_mute_toggled(self, is_muted: bool) -> None:
        """Update button when mute state changes from elsewhere."""
        self._update_mic_button(is_muted)

    def _update_mic_button(self, is_muted: bool) -> None:
        if is_muted:
            self.btn_mic.setText("🔇 Mic: Muted")
            self.btn_mic.setStyleSheet("color: #f85149; border-color: #da3633;")
        else:
            self.btn_mic.setText("🎤 Mic: Active")
            self.btn_mic.setStyleSheet("")

    def close(self) -> bool:
        """Ensure background worker is terminated upon close."""
        if self._worker:
            if self._worker.isRunning():
                self._worker.cancel()
                self._worker.wait(2000)
            self._worker = None
        return super().close()

    def closeEvent(self, event) -> None:
        """Cancel active inference worker and clean up thread on widget close."""
        if self._worker:
            if self._worker.isRunning():
                self._worker.cancel()
                self._worker.wait(2000)
            self._worker = None
        super().closeEvent(event)

    def _scroll_to_bottom(self) -> None:
        """Scroll message viewport to the latest message."""
        from PySide6.QtCore import QCoreApplication, QThread
        if QCoreApplication.instance() is None:
            return
        dispatcher = QThread.currentThread().eventDispatcher()
        if dispatcher is None:
            return
        QTimer.singleShot(20, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))
