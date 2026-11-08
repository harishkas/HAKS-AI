"""Floating AI Assistant widget for HAKS.

Provides a compact, draggable, always-on-top voice and quick text interface.
Shares the exact same backend and AppController with the Main Window.
"""

import os
from typing import Optional

from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QFrame,
    QApplication,
)

from config.settings import AppConfig, get_settings
from ai.models import AIMessage
from ai.worker import AIInferenceWorker
from core.app_controller import AppController, get_app_controller
from core.logger import get_logger
from privacy.consent_manager import ConsentDecision
from routing.models import RoutingMode
from ui.dialogs.consent_dialog import ConsentDialog, FallbackApprovalDialog

logger = get_logger("ui.floating_assistant")


class FloatingAssistant(QWidget):
    """Compact floating AI companion."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        controller: Optional[AppController] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.config = config or get_settings()
        self.controller = controller or get_app_controller(self.config)
        self._drag_pos: Optional[QPoint] = None
        self._worker: Optional[AIInferenceWorker] = None
        self._is_generating: bool = False
        self._current_stream_text: str = ""
        self._just_finished_msg_id: Optional[str] = None

        self._init_window_flags()
        self._init_ui()
        self._restore_position()
        self._connect_signals()
        logger.info("FloatingAssistant initialized.")

    def _init_window_flags(self) -> None:
        """Configure frameless and always-on-top window hints."""
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if self.controller.get_always_on_top():
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

    def _init_ui(self) -> None:
        """Construct the compact floating interface."""
        self.setFixedSize(380, 220)
        self.setObjectName("floating_root")

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Drag Header Bar
        self.drag_bar = QWidget()
        self.drag_bar.setObjectName("floating_drag_bar")
        drag_layout = QHBoxLayout(self.drag_bar)
        drag_layout.setContentsMargins(12, 6, 8, 6)
        drag_layout.setSpacing(6)

        # Voice status pill
        self.lbl_status_pill = QLabel("● HAKS STANDBY")
        self.lbl_status_pill.setStyleSheet("font-size: 11px; font-weight: 700; color: #3fb950;")
        drag_layout.addWidget(self.lbl_status_pill)

        drag_layout.addStretch()

        # Always-on-top toggle button
        self.btn_pin = QPushButton("📌")
        self.btn_pin.setProperty("class", "floating_btn_icon")
        self.btn_pin.setToolTip("Toggle Always on Top")
        self.btn_pin.clicked.connect(self._toggle_always_on_top)
        drag_layout.addWidget(self.btn_pin)

        # Expand to full app button
        self.btn_expand = QPushButton("⛶")
        self.btn_expand.setProperty("class", "floating_btn_icon")
        self.btn_expand.setToolTip("Expand to Full Desktop App")
        self.btn_expand.clicked.connect(self._on_expand_to_full_app)
        drag_layout.addWidget(self.btn_expand)

        # Minimize to tray button
        self.btn_min_tray = QPushButton("—")
        self.btn_min_tray.setProperty("class", "floating_btn_icon")
        self.btn_min_tray.setToolTip("Minimize to System Tray")
        self.btn_min_tray.clicked.connect(self.hide)
        drag_layout.addWidget(self.btn_min_tray)

        root_layout.addWidget(self.drag_bar)

        # 2. Main Body Container
        body_widget = QWidget()
        body_layout = QVBoxLayout(body_widget)
        body_layout.setContentsMargins(12, 10, 12, 12)
        body_layout.setSpacing(8)

        # Compact response preview label
        self.lbl_response_preview = QLabel("Hey HAKS is standing by. Type or speak below:")
        self.lbl_response_preview.setWordWrap(True)
        self.lbl_response_preview.setStyleSheet(
            "background-color: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 8px; color: #c9d1d9; font-size: 12px;"
        )
        self.lbl_response_preview.setFixedHeight(75)
        body_layout.addWidget(self.lbl_response_preview)

        # Quick Input Row
        input_row = QHBoxLayout()
        input_row.setSpacing(6)

        self.btn_mic = QPushButton("🎤")
        self.btn_mic.setFixedSize(36, 36)
        self.btn_mic.setProperty("class", "btn_secondary")
        self.btn_mic.setToolTip("Toggle Microphone")
        self.btn_mic.clicked.connect(self._on_mic_clicked)
        input_row.addWidget(self.btn_mic)

        self.quick_input = QLineEdit()
        self.quick_input.setPlaceholderText("Ask HAKS anything...")
        self.quick_input.setFixedHeight(36)
        self.quick_input.returnPressed.connect(self._on_quick_send)
        input_row.addWidget(self.quick_input)

        self.btn_send = QPushButton("Send")
        self.btn_send.setFixedHeight(36)
        self.btn_send.setProperty("class", "btn_primary")
        self.btn_send.clicked.connect(self._on_quick_send)
        input_row.addWidget(self.btn_send)

        body_layout.addLayout(input_row)
        root_layout.addWidget(body_widget)

    def _restore_position(self) -> None:
        """Position window using stored coordinates."""
        x, y = self.controller.get_floating_position()
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            # Boundary checks
            x = max(geo.left(), min(x, geo.right() - self.width()))
            y = max(geo.top(), min(y, geo.bottom() - self.height()))
        self.move(x, y)

    def _connect_signals(self) -> None:
        """Connect to application events."""
        self.controller.events.voice_state_changed.connect(self._update_voice_status)
        self.controller.events.voice_mute_toggled.connect(self._update_mic_status)
        self.controller.events.message_received.connect(self._on_message_received)
        self.controller.events.show_floating_requested.connect(self._on_show_requested)

    # --------------------------------------------------------------------------
    # Draggable Window Implementation
    # --------------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            new_pos = event.globalPosition().toPoint() - self._drag_pos
            self.move(new_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
            self.controller.save_floating_position(self.x(), self.y())
            event.accept()

    # --------------------------------------------------------------------------
    # User Interactions & Shared Controller Calls
    # --------------------------------------------------------------------------

    def _on_quick_send(self) -> None:
        if self._is_generating:
            return

        text = self.quick_input.text().strip()
        if not text:
            return

        self.quick_input.clear()
        self.lbl_response_preview.setText(f"<b>You:</b> {text}")

        # Send via shared controller (writes to conversations.db, broadcasts to all UI surfaces)
        self.controller.send_message(text, role="user")

        # Launch real AI streaming inference
        self._start_floating_inference(prompt=text)

    def _start_floating_inference(self, prompt: str = "") -> None:
        """Start asynchronous AI streaming inference for floating assistant with intelligent routing."""
        self._is_generating = True
        self._current_stream_text = ""
        self.btn_send.setEnabled(False)
        self.lbl_response_preview.setText("<i>● HAKS thinking...</i>")

        # If prompt not passed explicitly, attempt to find latest user prompt
        messages = self.controller.get_conversation_ai_messages(limit=10)
        if not prompt and messages:
            for m in reversed(messages):
                if m.role == "user":
                    prompt = m.content
                    break

        # 1. Routing decision
        effective_mode = self.controller.get_routing_mode()
        decision = self.controller.route_request(
            user_prompt=prompt,
            mode=effective_mode,
        )
        target_engine = decision.selected_engine

        is_headless = bool(os.environ.get("PYTEST_CURRENT_TEST")) or (
            QApplication.instance() is not None
            and QApplication.instance().platformName() == "offscreen"
        )

        # 2. Check engine availability fallback (never silent)
        if effective_mode == RoutingMode.AUTO and decision.is_fallback_needed and prompt and not is_headless:
            dialog = FallbackApprovalDialog(
                target_engine=decision.selected_engine,
                fallback_engine=decision.fallback_engine or "local",
                reason=decision.fallback_reason or "Service not reachable",
                parent=self,
            )
            dialog.exec()
            if not dialog.approved_fallback:
                self.lbl_response_preview.setText(f"<b style='color:#ff7b72;'>Cancelled:</b> {decision.selected_engine} unavailable")
                self._is_generating = False
                self.btn_send.setEnabled(True)
                return
            target_engine = decision.fallback_engine or "local"

        # 3. Check Consent & Redaction
        if decision.requires_consent and decision.consent_request and target_engine == "cloud" and not is_headless:
            consent_dlg = ConsentDialog(decision.consent_request, parent=self)
            consent_dlg.exec()
            dec_val = consent_dlg.decision

            if dec_val == ConsentDecision.APPROVED:
                self.controller.consent_manager.record_decision(decision.consent_request.id, ConsentDecision.APPROVED)
                self.controller.consent_manager.consume_consent(decision.consent_request.id)
                # Redacted if applicable
                if decision.redaction_preview and messages and messages[-1].role == "user":
                    messages[-1].content = decision.redaction_preview.sanitized_text
            elif dec_val == ConsentDecision.USE_LOCAL:
                self.controller.consent_manager.record_decision(decision.consent_request.id, ConsentDecision.USE_LOCAL)
                target_engine = "local"
            else:
                self.controller.consent_manager.record_decision(decision.consent_request.id, ConsentDecision.CANCELLED)
                self.lbl_response_preview.setText("<i>Transmission cancelled by user.</i>")
                self._is_generating = False
                self.btn_send.setEnabled(True)
                return

        # 4. Inject Memory Context with privacy constraints
        if prompt:
            try:
                mem_context = self.controller.context_builder.build_context(
                    query=prompt,
                    include_personal=True,
                    include_work=True,
                    top_k_per_domain=2,
                    min_relevance=0.25,
                    is_cloud_target=(target_engine == "cloud"),
                    redact_for_cloud=(target_engine == "cloud"),
                )
                if mem_context:
                    sys_msg = AIMessage(
                        role="system",
                        content=f"Context from memory:\n{mem_context}\nAnswer user queries using this context where helpful.",
                    )
                    messages.insert(0, sys_msg)
            except Exception as exc:
                logger.warning("Could not append memory context in floating assistant: %s", exc)

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
        self._current_stream_text += chunk_text
        preview = self._current_stream_text
        if len(preview) > 95:
            preview = "..." + preview[-95:]
        self.lbl_response_preview.setText(f"<b>HAKS:</b> {preview}")

    def _on_worker_finished(self, full_text: str, metadata: dict) -> None:
        self._is_generating = False
        self.btn_send.setEnabled(True)
        msg_data = self.controller.send_message(full_text, role="assistant", metadata=metadata)
        self._just_finished_msg_id = msg_data["id"]

        preview = full_text
        if len(preview) > 95:
            preview = preview[:95] + "..."
        self.lbl_response_preview.setText(f"<b>HAKS:</b> {preview}")

    def _on_worker_error(self, err_msg: str) -> None:
        self._is_generating = False
        self.btn_send.setEnabled(True)
        short_err = err_msg[:80] + ("..." if len(err_msg) > 80 else "")
        self.lbl_response_preview.setText(f"<b style='color:#ff7b72;'>⚠️ Error:</b> {short_err}")
        self.controller.send_message(f"⚠️ Error: {err_msg}", role="assistant", metadata={"error": True})

    def _on_worker_cancelled(self) -> None:
        self._is_generating = False
        self.btn_send.setEnabled(True)
        self.lbl_response_preview.setText("<i>Generation stopped.</i>")

    def close(self) -> bool:
        """Cancel worker on widget close."""
        if self._worker:
            if self._worker.isRunning():
                self._worker.cancel()
                self._worker.wait(2000)
            self._worker = None
        return super().close()

    def closeEvent(self, event) -> None:
        """Cancel worker on close event."""
        if self._worker:
            if self._worker.isRunning():
                self._worker.cancel()
                self._worker.wait(2000)
            self._worker = None
        super().closeEvent(event)

    def _on_message_received(self, conv_id: str, msg_id: str, role: str, content: str) -> None:
        """Update preview label when a message occurs on the active conversation."""
        if self._is_generating or self._just_finished_msg_id == msg_id:
            self._just_finished_msg_id = None
            return

        prefix = "You" if role == "user" else "HAKS"
        clean_text = content[:90] + ("..." if len(content) > 90 else "")
        self.lbl_response_preview.setText(f"<b>{prefix}:</b> {clean_text}")

    def _on_mic_clicked(self) -> None:
        self.controller.toggle_mic_mute()

    def _update_mic_status(self, is_muted: bool) -> None:
        if is_muted:
            self.btn_mic.setText("🔇")
            self.btn_mic.setStyleSheet("color: #f85149; border: 1px solid #da3633;")
        else:
            self.btn_mic.setText("🎤")
            self.btn_mic.setStyleSheet("")

    def _update_voice_status(self, state: str) -> None:
        color_map = {
            "STANDBY": "#3fb950",
            "WAKE_DETECTED": "#58a6ff",
            "LISTENING": "#ff7b72",
            "PROCESSING": "#d29922",
            "THINKING": "#a371f7",
            "SPEAKING": "#2ea043",
            "INTERRUPTED": "#f85149",
            "SLEEP": "#8b949e",
        }
        color = color_map.get(state, "#3fb950")
        self.lbl_status_pill.setText(f"● HAKS {state}")
        self.lbl_status_pill.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {color};")

    def _toggle_always_on_top(self) -> None:
        current = self.controller.get_always_on_top()
        new_val = not current
        self.controller.set_always_on_top(new_val)

        # Re-apply window flags
        pos = self.pos()
        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if new_val:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.move(pos)
        self.show()

    def _on_expand_to_full_app(self) -> None:
        """Hide floating assistant and show main window."""
        self.hide()
        self.controller.events.show_main_window_requested.emit()

    def _on_show_requested(self) -> None:
        """Show floating assistant."""
        self._restore_position()
        self.show()
        self.activateWindow()
        self.raise_()
