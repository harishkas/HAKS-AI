"""User Consent and Fallback Confirmation Dialogs for HAKS.

Presents explicit, request-specific approval interfaces for:
1. Cloud transmission of personal or redacted information.
2. Engine switching when an engine is unavailable (prevents silent fallback).
"""

from typing import Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QTabWidget,
    QTextEdit,
    QWidget,
)

from privacy.consent_manager import ConsentDecision, ConsentRequest


class ConsentDialog(QDialog):
    """Modal dialog presenting request-specific consent for cloud AI transmission."""

    def __init__(self, request: ConsentRequest, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.request = request
        self.consent_request = request
        self.decision: ConsentDecision = ConsentDecision.PENDING
        self.setWindowTitle("Cloud Transmission Consent Required — HAKS")
        self.setMinimumWidth(650)
        self.setMinimumHeight(480)
        self.setModal(True)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Header Title
        title_box = QHBoxLayout()
        icon_lbl = QLabel("🛡️")
        icon_lbl.setStyleSheet("font-size: 24px;")
        title_box.addWidget(icon_lbl)

        title_lbl = QLabel("Cloud Transmission Approval Required")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 700; color: #f0f6fc;")
        title_box.addWidget(title_lbl, stretch=1)
        layout.addLayout(title_box)

        # Overview Card
        card = QFrame()
        card.setStyleSheet("background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px;")
        c_layout = QVBoxLayout(card)
        c_layout.setSpacing(6)

        c_layout.addWidget(QLabel(f"<b>Destination Provider:</b> {self.request.destination_provider}"))
        c_layout.addWidget(QLabel(f"<b>Selected Model:</b> <code>{self.request.selected_model}</code>"))
        c_layout.addWidget(QLabel(f"<b>Privacy Classification:</b> <span style='color: #d29922; font-weight: bold;'>{self.request.privacy_level.value}</span>"))

        cats_str = ", ".join(self.request.detected_categories) if self.request.detected_categories else "None"
        c_layout.addWidget(QLabel(f"<b>Information Categories Detected:</b> {cats_str}"))

        redact_status = "<span style='color: #3fb950;'>Active (Masked sensitive identifiers)</span>" if self.request.is_redacted else "<span style='color: #8b949e;'>None</span>"
        c_layout.addWidget(QLabel(f"<b>Redaction Applied:</b> {redact_status}"))
        layout.addWidget(card)

        # Tabs for Sanitized vs Original Text Preview
        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { padding: 6px 12px; font-size: 12px; }")

        # Tab 1: Data to be transmitted
        tab_trans = QWidget()
        tt_layout = QVBoxLayout(tab_trans)
        tt_layout.setContentsMargins(4, 4, 4, 4)
        edit_trans = QTextEdit()
        edit_trans.setReadOnly(True)
        edit_trans.setPlainText(self.request.transmitted_data)
        edit_trans.setStyleSheet("background-color: #0d1117; color: #58a6ff; font-family: Consolas, monospace; font-size: 11px;")
        tt_layout.addWidget(edit_trans)
        tabs.addTab(tab_trans, "Data To Be Transmitted (Sanitized)")

        # Tab 2: Original Data
        tab_orig = QWidget()
        to_layout = QVBoxLayout(tab_orig)
        to_layout.setContentsMargins(4, 4, 4, 4)
        edit_orig = QTextEdit()
        edit_orig.setReadOnly(True)
        edit_orig.setPlainText(self.request.original_data)
        edit_orig.setStyleSheet("background-color: #0d1117; color: #e6edf3; font-family: Consolas, monospace; font-size: 11px;")
        to_layout.addWidget(edit_orig)
        tabs.addTab(tab_orig, "Original Input")

        layout.addWidget(tabs, stretch=1)

        # Action Buttons Row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_cancel = QPushButton("✕ Cancel")
        self.btn_cancel.setStyleSheet("background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 8px 16px; border-radius: 6px;")
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.btn_cancel)

        self.btn_use_local = QPushButton("💻 Use Local AI Instead")
        self.btn_use_local.setStyleSheet("background-color: #1f6feb; color: #ffffff; border: none; font-weight: 600; padding: 8px 16px; border-radius: 6px;")
        self.btn_use_local.clicked.connect(self._on_use_local)
        btn_row.addWidget(self.btn_use_local)

        self.btn_approve = QPushButton("✓ Approve This Request")
        self.btn_approve.setStyleSheet("background-color: #238636; color: #ffffff; border: none; font-weight: 600; padding: 8px 18px; border-radius: 6px;")
        self.btn_approve.clicked.connect(self._on_approve)
        btn_row.addWidget(self.btn_approve)

        layout.addLayout(btn_row)

    def _on_approve(self) -> None:
        self.decision = ConsentDecision.APPROVED
        self.accept()

    def _on_use_local(self) -> None:
        self.decision = ConsentDecision.USE_LOCAL
        self.accept()

    def _on_cancel(self) -> None:
        self.decision = ConsentDecision.CANCELLED
        self.reject()

    def reject(self) -> None:
        self.decision = ConsentDecision.CANCELLED
        super().reject()


class FallbackApprovalDialog(QDialog):
    """Explicit confirmation dialog before switching engines on failure (Prevents silent fallback)."""

    def __init__(
        self,
        target_engine: str,
        fallback_engine: str,
        reason: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.target_engine = target_engine
        self.fallback_engine = fallback_engine
        self.approved_fallback = False
        self.setWindowTitle("Engine Unavailable — Fallback Approval Required")
        self.setMinimumWidth(500)
        self.setModal(True)
        self._init_ui(reason)

    def _init_ui(self, reason: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        icon_title = QHBoxLayout()
        icon = QLabel("⚠️")
        icon.setStyleSheet("font-size: 24px;")
        icon_title.addWidget(icon)

        title = QLabel(f"Target Engine '{self.target_engine.upper()}' is Unavailable")
        title.setStyleSheet("font-size: 15px; font-weight: 700; color: #f85149;")
        icon_title.addWidget(title, stretch=1)
        layout.addLayout(icon_title)

        body = QLabel(
            f"<b>Failure Reason:</b> {reason}\n\n"
            f"HAKS privacy guidelines strictly prohibit silent engine switching.\n"
            f"Would you like to switch to <b>{self.fallback_engine.upper()} AI</b> for this request?"
        )
        body.setWordWrap(True)
        body.setStyleSheet("color: #e6edf3; font-size: 12px; line-height: 1.4;")
        layout.addWidget(body)

        btn_row = QHBoxLayout()
        self.btn_cancel = QPushButton("✕ Cancel Request")
        self.btn_cancel.setStyleSheet("background-color: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 8px 14px; border-radius: 6px;")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_switch = QPushButton(f"✓ Switch to {self.fallback_engine.title()} AI")
        self.btn_switch.setStyleSheet("background-color: #1f6feb; color: #ffffff; border: none; font-weight: 600; padding: 8px 16px; border-radius: 6px;")
        self.btn_switch.clicked.connect(self._on_switch)
        btn_row.addWidget(self.btn_switch)

        layout.addLayout(btn_row)

    def _on_switch(self) -> None:
        self.approved_fallback = True
        self.accept()
