"""AI Engine View for HAKS.

Displays Local AI (Ollama) and Cloud AI adapter status,
dynamic model management (pull with live progress, delete, unload),
secure Windows Credential Manager key storage, hardware-aware safeguards,
and cloud provider endpoint configuration.
"""

import asyncio
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QGridLayout,
    QLineEdit,
    QComboBox,
    QProgressBar,
    QMessageBox,
    QInputDialog,
    QScrollArea,
)

from ai.credentials import get_credential_manager
from ai.local_engine import RECOMMENDED_LIGHTWEIGHT_MODELS
from ai.worker import ModelDownloadWorker
from core.app_controller import AppController, get_app_controller
from core.logger import get_logger
from routing.models import RoutingMode

logger = get_logger("ui.views.ai_engine_view")


class AIEngineView(QWidget):
    """AI Engine and Router management view."""

    def __init__(self, controller: Optional[AppController] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller or get_app_controller()
        self.cred_manager = get_credential_manager()
        self._download_worker: Optional[ModelDownloadWorker] = None

        self._init_ui()
        self._load_current_state()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scrollable container so all sections fit on smaller laptop screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")

        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # 1. Header
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("AI Engine & Integration Management")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #f0f6fc;")
        subtitle = QLabel("Local AI (Ollama) + Secure Multi-Provider Cloud Adapter")
        subtitle.setStyleSheet("color: #8b949e; font-size: 12px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)

        header.addStretch()

        # Routing Policy Selector
        lbl_routing = QLabel("Routing Policy:")
        lbl_routing.setStyleSheet("color: #8b949e; font-size: 12px; font-weight: 600;")
        header.addWidget(lbl_routing)

        self.combo_default_routing_mode = QComboBox()
        self.combo_default_routing_mode.addItem("🤖 Auto (Intelligent)", "AUTO")
        self.combo_default_routing_mode.addItem("⚙️ Manual", "MANUAL_LOCAL")
        self.combo_default_routing_mode.addItem("🔒 Force Local-Only", "FORCE_LOCAL")
        self.combo_default_routing_mode.currentIndexChanged.connect(self._on_routing_policy_changed)
        header.addWidget(self.combo_default_routing_mode)

        # Default engine selector
        lbl_def = QLabel("Default Engine:")
        lbl_def.setStyleSheet("color: #8b949e; font-size: 12px; font-weight: 600;")
        header.addWidget(lbl_def)

        self.combo_default_engine = QComboBox()
        self.combo_default_engine.addItem("Local AI (Ollama)", "local")
        self.combo_default_engine.addItem("Cloud AI Adapter", "cloud")
        self.combo_default_engine.currentIndexChanged.connect(self._on_default_engine_changed)
        header.addWidget(self.combo_default_engine)

        self.btn_check_health = QPushButton("Check Engine Health")
        self.btn_check_health.setProperty("class", "btn_primary")
        self.btn_check_health.clicked.connect(self._on_check_health)
        header.addWidget(self.btn_check_health)

        layout.addLayout(header)

        # 2. Grid of Engines
        grid = QGridLayout()
        grid.setSpacing(14)

        # ----------------------------------------------------------------------
        # Card 1: Local AI (Ollama)
        # ----------------------------------------------------------------------
        local_card = QFrame()
        local_card.setProperty("class", "stat_card")
        lc_layout = QVBoxLayout(local_card)
        lc_layout.setSpacing(10)

        lc_title = QLabel("LOCAL AI RUNTIME (OLLAMA)")
        lc_title.setProperty("class", "card_title")
        self.lc_status = QLabel("Checking Ollama...")
        self.lc_status.setProperty("class", "card_value")
        self.lc_status.setStyleSheet("font-size: 14px; font-weight: 600; color: #8b949e;")

        lc_desc = QLabel("Offline, private, low-latency execution directly on your device CPU.")
        lc_desc.setProperty("class", "card_detail")

        # Hardware recommendation banner
        hw_banner = QLabel(
            "💻 Target Hardware Profile: Intel Core i5-7200U | 8 GB RAM\n"
            "Recommended quantized models: llama3.2:1b (1.3 GB) or qwen2.5:1.5b (1.6 GB).\n"
            "Warning: Models > 3.8 GB (7B+) are restricted to prevent RAM exhaustion and OS freezing."
        )
        hw_banner.setStyleSheet(
            "background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 8px 10px; color: #58a6ff; font-size: 11px;"
        )

        # Active Model Selector row
        lc_model_lbl = QLabel("Active Local Model:")
        lc_model_lbl.setStyleSheet("font-weight: 600; color: #c9d1d9; font-size: 12px;")
        self.local_model_combo = QComboBox()
        self.local_model_combo.currentIndexChanged.connect(self._on_local_model_changed)

        # Local Actions row (Unload and Delete)
        lc_act_row = QHBoxLayout()
        self.btn_unload_model = QPushButton("Unload from RAM")
        self.btn_unload_model.setProperty("class", "btn_secondary")
        self.btn_unload_model.setToolTip("Free system RAM immediately")
        self.btn_unload_model.clicked.connect(self._on_unload_model)

        self.btn_delete_model = QPushButton("Delete Model")
        self.btn_delete_model.setProperty("class", "btn_danger")
        self.btn_delete_model.setToolTip("Delete selected local model from SSD")
        self.btn_delete_model.clicked.connect(self._on_delete_model)

        lc_act_row.addWidget(self.btn_unload_model)
        lc_act_row.addWidget(self.btn_delete_model)
        lc_act_row.addStretch()

        # Model Puller Section
        pull_sep = QLabel("Install / Pull Model:")
        pull_sep.setStyleSheet("font-weight: 600; color: #c9d1d9; font-size: 12px; margin-top: 6px;")

        # Quick-pull buttons for recommended models
        quick_row = QHBoxLayout()
        quick_lbl = QLabel("Presets:")
        quick_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        quick_row.addWidget(quick_lbl)

        self.btn_preset_1b = QPushButton("llama3.2:1b")
        self.btn_preset_1b.setProperty("class", "btn_secondary")
        self.btn_preset_1b.clicked.connect(lambda: self.txt_pull_model.setText("llama3.2:1b"))
        quick_row.addWidget(self.btn_preset_1b)

        self.btn_preset_qwen = QPushButton("qwen2.5:1.5b")
        self.btn_preset_qwen.setProperty("class", "btn_secondary")
        self.btn_preset_qwen.clicked.connect(lambda: self.txt_pull_model.setText("qwen2.5:1.5b"))
        quick_row.addWidget(self.btn_preset_qwen)

        self.btn_preset_3b = QPushButton("llama3.2:3b")
        self.btn_preset_3b.setProperty("class", "btn_secondary")
        self.btn_preset_3b.clicked.connect(lambda: self.txt_pull_model.setText("llama3.2:3b"))
        quick_row.addWidget(self.btn_preset_3b)
        quick_row.addStretch()

        # Custom model pull row
        pull_input_row = QHBoxLayout()
        self.txt_pull_model = QLineEdit()
        self.txt_pull_model.setPlaceholderText("e.g., llama3.2:1b or qwen2.5:1.5b")
        self.btn_pull = QPushButton("Pull Model")
        self.btn_pull.setProperty("class", "btn_primary")
        self.btn_pull.clicked.connect(self._on_pull_model)
        pull_input_row.addWidget(self.txt_pull_model, stretch=1)
        pull_input_row.addWidget(self.btn_pull)

        # Download progress indicators
        self.pull_progress_bar = QProgressBar()
        self.pull_progress_bar.setRange(0, 100)
        self.pull_progress_bar.setValue(0)
        self.pull_progress_bar.setVisible(False)

        pull_status_row = QHBoxLayout()
        self.lbl_pull_status = QLabel("")
        self.lbl_pull_status.setStyleSheet("color: #58a6ff; font-size: 11px;")
        self.btn_cancel_pull = QPushButton("Cancel")
        self.btn_cancel_pull.setProperty("class", "btn_danger")
        self.btn_cancel_pull.setVisible(False)
        self.btn_cancel_pull.clicked.connect(self._on_cancel_pull)
        pull_status_row.addWidget(self.lbl_pull_status, stretch=1)
        pull_status_row.addWidget(self.btn_cancel_pull)

        lc_url_lbl = QLabel("Ollama API Endpoint:")
        lc_url_lbl.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.local_url_input = QLineEdit(self.controller.config.ai.ollama_base_url)
        self.local_url_input.setReadOnly(True)

        lc_layout.addWidget(lc_title)
        lc_layout.addWidget(self.lc_status)
        lc_layout.addWidget(lc_desc)
        lc_layout.addWidget(hw_banner)
        lc_layout.addWidget(lc_model_lbl)
        lc_layout.addWidget(self.local_model_combo)
        lc_layout.addLayout(lc_act_row)
        lc_layout.addWidget(pull_sep)
        lc_layout.addLayout(quick_row)
        lc_layout.addLayout(pull_input_row)
        lc_layout.addWidget(self.pull_progress_bar)
        lc_layout.addLayout(pull_status_row)
        lc_layout.addWidget(lc_url_lbl)
        lc_layout.addWidget(self.local_url_input)

        grid.addWidget(local_card, 0, 0)

        # ----------------------------------------------------------------------
        # Card 2: Cloud AI Adapter
        # ----------------------------------------------------------------------
        cloud_card = QFrame()
        cloud_card.setProperty("class", "stat_card")
        cc_layout = QVBoxLayout(cloud_card)
        cc_layout.setSpacing(10)

        cc_title = QLabel("CLOUD AI ADAPTER")
        cc_title.setProperty("class", "card_title")
        self.cc_status = QLabel("Provider-Independent Adapter")
        self.cc_status.setProperty("class", "card_value")
        self.cc_status.setStyleSheet("font-size: 14px; font-weight: 600; color: #8b949e;")

        cc_desc = QLabel("High-complexity reasoning, advanced code analysis, architecture design.")
        cc_desc.setProperty("class", "card_detail")

        # Provider combo
        cc_provider_lbl = QLabel("Active Provider:")
        cc_provider_lbl.setStyleSheet("font-weight: 600; color: #c9d1d9; font-size: 12px;")
        self.cloud_provider_combo = QComboBox()
        self.cloud_provider_combo.addItem("Google Gemini API", "gemini")
        self.cloud_provider_combo.addItem("OpenAI API", "openai")
        self.cloud_provider_combo.addItem("Groq Cloud (Ultra Fast)", "groq")
        self.cloud_provider_combo.addItem("OpenRouter (Universal)", "openrouter")
        self.cloud_provider_combo.addItem("Anthropic Claude API", "anthropic")
        self.cloud_provider_combo.addItem("Custom OpenAI-Compatible Endpoint", "custom")
        self.cloud_provider_combo.currentIndexChanged.connect(self._on_cloud_provider_changed)

        # Endpoint Base URL Row
        self.lbl_cloud_url = QLabel("API Base Endpoint URL:")
        self.lbl_cloud_url.setStyleSheet("color: #8b949e; font-size: 11px;")
        url_edit_row = QHBoxLayout()
        self.txt_cloud_base_url = QLineEdit()
        self.btn_save_cloud_url = QPushButton("Apply URL")
        self.btn_save_cloud_url.setProperty("class", "btn_secondary")
        self.btn_save_cloud_url.clicked.connect(self._on_save_cloud_base_url)
        url_edit_row.addWidget(self.txt_cloud_base_url, stretch=1)
        url_edit_row.addWidget(self.btn_save_cloud_url)

        # Model combo
        cc_model_lbl = QLabel("Cloud Model:")
        cc_model_lbl.setStyleSheet("font-weight: 600; color: #c9d1d9; font-size: 12px;")
        self.cloud_model_combo = QComboBox()
        self.cloud_model_combo.currentIndexChanged.connect(self._on_cloud_model_changed)

        # Secure API Key row
        cc_key_lbl = QLabel("Windows Credential Manager API Key:")
        cc_key_lbl.setStyleSheet("font-weight: 600; color: #c9d1d9; font-size: 12px;")
        self.lbl_key_status = QLabel("[Not Configured]")
        self.lbl_key_status.setStyleSheet("color: #d29922; font-family: monospace; font-size: 12px;")

        key_btn_row = QHBoxLayout()
        self.btn_set_key = QPushButton("Configure API Key")
        self.btn_set_key.setProperty("class", "btn_primary")
        self.btn_set_key.clicked.connect(self._on_configure_key)

        self.btn_delete_key = QPushButton("Remove Key")
        self.btn_delete_key.setProperty("class", "btn_danger")
        self.btn_delete_key.clicked.connect(self._on_delete_key)

        self.btn_test_cloud = QPushButton("Test Connection")
        self.btn_test_cloud.setProperty("class", "btn_secondary")
        self.btn_test_cloud.clicked.connect(self._on_test_cloud_connection)

        key_btn_row.addWidget(self.btn_set_key)
        key_btn_row.addWidget(self.btn_delete_key)
        key_btn_row.addWidget(self.btn_test_cloud)
        key_btn_row.addStretch()

        cc_policy_lbl = QLabel("Privacy Guarantee:")
        cc_policy_desc = QLabel(
            "🔒 Stored in Windows Credential Manager. Personal and Work Memory databases are NEVER uploaded to cloud AI."
        )
        cc_policy_desc.setStyleSheet("color: #3fb950; font-size: 11px;")
        cc_policy_desc.setWordWrap(True)

        cc_layout.addWidget(cc_title)
        cc_layout.addWidget(self.cc_status)
        cc_layout.addWidget(cc_desc)
        cc_layout.addWidget(cc_provider_lbl)
        cc_layout.addWidget(self.cloud_provider_combo)
        cc_layout.addWidget(self.lbl_cloud_url)
        cc_layout.addLayout(url_edit_row)
        cc_layout.addWidget(cc_model_lbl)
        cc_layout.addWidget(self.cloud_model_combo)
        cc_layout.addWidget(cc_key_lbl)
        cc_layout.addWidget(self.lbl_key_status)
        cc_layout.addLayout(key_btn_row)
        cc_layout.addWidget(cc_policy_lbl)
        cc_layout.addWidget(cc_policy_desc)

        grid.addWidget(cloud_card, 0, 1)

        layout.addLayout(grid)

        # 3. Router Policy & Privacy Card
        router_card = QFrame()
        router_card.setProperty("class", "stat_card")
        rc_layout = QVBoxLayout(router_card)
        rc_title = QLabel("AI ENGINE INTEGRATION & PRIVACY BOUNDARIES")
        rc_title.setProperty("class", "card_title")

        rule1 = QLabel("• Consistent Asynchronous Interface: Both Local and Cloud AI expose identical generate(), stream(), and check_health() APIs.")
        rule2 = QLabel("• Hardware Isolation: Inference runs in background threads, keeping PySide6 desktop UI fluid and 60fps responsive.")
        rule3 = QLabel("• Zero Silent Fallback: If Cloud AI fails or is unconfigured, HAKS reports the explicit error rather than silently leaking data or switching.")
        rule4 = QLabel("• Strict Context Boundary: Only explicit conversation messages are transmitted; local memory databases remain strictly on-device.")

        for r in (rule1, rule2, rule3, rule4):
            r.setStyleSheet("color: #e6edf3; font-size: 12px; padding: 2px 0;")
            r.setWordWrap(True)
            rc_layout.addWidget(r)

        layout.addWidget(router_card)
        layout.addStretch()

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

    def _load_current_state(self) -> None:
        """Populate widgets with active engine state."""
        mgr = self.controller.get_ai_manager()

        # Routing mode
        cur_mode = self.controller.get_routing_mode().value
        for i in range(self.combo_default_routing_mode.count()):
            if self.combo_default_routing_mode.itemData(i) == cur_mode:
                self.combo_default_routing_mode.setCurrentIndex(i)
                break

        # Default engine
        cur_def = self.controller.get_selected_ai_engine()
        idx_def = 1 if cur_def == "cloud" else 0
        self.combo_default_engine.blockSignals(True)
        self.combo_default_engine.setCurrentIndex(idx_def)
        self.combo_default_engine.blockSignals(False)

        # Local model list
        self._refresh_local_models()

        # Cloud provider
        cloud_prov = mgr.cloud_engine.active_provider_id
        for i in range(self.cloud_provider_combo.count()):
            if self.cloud_provider_combo.itemData(i) == cloud_prov:
                self.cloud_provider_combo.setCurrentIndex(i)
                break

        self._refresh_cloud_models_and_key()

        # Quick health probe
        from PySide6.QtCore import QCoreApplication, QThread
        if QCoreApplication.instance() is not None and QThread.currentThread().eventDispatcher() is not None:
            QTimer.singleShot(200, self._quick_health_probe)

    def _on_routing_policy_changed(self, index: int) -> None:
        """Persist default routing policy."""
        mode_val = self.combo_default_routing_mode.currentData()
        try:
            self.controller.set_routing_mode(RoutingMode(mode_val))
        except Exception as exc:
            logger.warning("Could not set routing mode '%s': %s", mode_val, exc)

    def _refresh_local_models(self) -> None:
        """Refresh local models dropdown."""
        mgr = self.controller.get_ai_manager()
        cur_model = mgr.local_engine.current_model

        self.local_model_combo.blockSignals(True)
        self.local_model_combo.clear()

        # Populate with recommended models or installed models
        defaults = ["llama3.2:1b", "qwen2.5:1.5b", "llama3.2:3b"]
        for m in defaults:
            self.local_model_combo.addItem(m, m)

        idx = self.local_model_combo.findText(cur_model)
        if idx >= 0:
            self.local_model_combo.setCurrentIndex(idx)
        self.local_model_combo.blockSignals(False)

    def _refresh_cloud_models_and_key(self) -> None:
        """Update cloud models, base URL, and masked key display."""
        prov_id = self.cloud_provider_combo.currentData() or "gemini"
        mgr = self.controller.get_ai_manager()
        mgr.cloud_engine.set_provider(prov_id)

        adapter = mgr.cloud_engine.get_active_adapter()
        self.cloud_model_combo.blockSignals(True)
        self.cloud_model_combo.clear()
        if adapter:
            for m in adapter.get_supported_models():
                self.cloud_model_combo.addItem(m, m)
            idx = self.cloud_model_combo.findText(mgr.cloud_engine.current_model)
            if idx >= 0:
                self.cloud_model_combo.setCurrentIndex(idx)

            # Update base URL input
            if hasattr(adapter, "base_url"):
                self.txt_cloud_base_url.setText(adapter.base_url)
                self.txt_cloud_base_url.setEnabled(prov_id in ("custom", "groq", "openrouter", "openai"))
            else:
                self.txt_cloud_base_url.setText("Standard Cloud API")
                self.txt_cloud_base_url.setEnabled(False)

        self.cloud_model_combo.blockSignals(False)

        # Refresh key display
        key = self.cred_manager.get_api_key(prov_id)
        if key:
            self.lbl_key_status.setText(f"Active: {self.cred_manager.mask_key(key)}")
            self.lbl_key_status.setStyleSheet("color: #3fb950; font-family: monospace; font-size: 12px;")
        else:
            self.lbl_key_status.setText("[Not Configured in Windows Credential Manager]")
            self.lbl_key_status.setStyleSheet("color: #d29922; font-family: monospace; font-size: 12px;")

    def _on_default_engine_changed(self, index: int) -> None:
        eng_type = self.combo_default_engine.currentData()
        if eng_type:
            self.controller.set_selected_ai_engine(eng_type)

    def _on_local_model_changed(self, index: int) -> None:
        model = self.local_model_combo.currentText()
        if model:
            mgr = self.controller.get_ai_manager()
            mgr.local_engine.set_model(model)

    def _on_cloud_provider_changed(self, index: int) -> None:
        self._refresh_cloud_models_and_key()

    def _on_cloud_model_changed(self, index: int) -> None:
        model = self.cloud_model_combo.currentText()
        if model:
            mgr = self.controller.get_ai_manager()
            mgr.cloud_engine.set_model(model)

    def _on_save_cloud_base_url(self) -> None:
        """Update provider base URL for custom / configurable endpoints."""
        prov_id = self.cloud_provider_combo.currentData() or "gemini"
        url = self.txt_cloud_base_url.text().strip()
        if url:
            mgr = self.controller.get_ai_manager()
            mgr.cloud_engine.set_provider_base_url(prov_id, url)
            QMessageBox.information(self, "Endpoint Updated", f"Base URL for {prov_id.upper()} updated to:\n{url}")

    def _on_configure_key(self) -> None:
        """Prompt user for API key and store securely in Windows Credential Manager."""
        prov_id = self.cloud_provider_combo.currentData() or "gemini"
        prov_name = self.cloud_provider_combo.currentText()

        key, ok = QInputDialog.getText(
            self,
            f"Configure {prov_name} Key",
            f"Enter API key for {prov_name}:\n(It will be stored encrypted in Windows Credential Manager)",
            QLineEdit.EchoMode.Password,
        )
        if ok and key.strip():
            self.cred_manager.set_api_key(prov_id, key.strip())
            QMessageBox.information(
                self,
                "API Key Saved",
                f"API Key for {prov_name} has been securely saved to Windows Credential Manager.",
            )
            self._refresh_cloud_models_and_key()

    def _on_delete_key(self) -> None:
        """Remove API key for active cloud provider."""
        prov_id = self.cloud_provider_combo.currentData() or "gemini"
        prov_name = self.cloud_provider_combo.currentText()

        confirm = QMessageBox.question(
            self,
            "Remove Key",
            f"Are you sure you want to delete the stored key for {prov_name} from Windows Credential Manager?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.cred_manager.delete_api_key(prov_id)
            self._refresh_cloud_models_and_key()

    def _on_test_cloud_connection(self) -> None:
        """Test health and latency for currently selected cloud provider."""
        mgr = self.controller.get_ai_manager()
        prov_name = self.cloud_provider_combo.currentText()

        loop = asyncio.new_event_loop()
        try:
            health = loop.run_until_complete(mgr.cloud_engine.check_health())
        finally:
            loop.close()

        if health.is_healthy:
            lat = f"{health.latency_ms:.1f}ms" if health.latency_ms else "N/A"
            models_preview = ", ".join(health.available_models[:4])
            msg = (
                f"✅ {prov_name} Connected Successfully!\n\n"
                f"Status: {health.status_message}\n"
                f"Latency: {lat}\n"
                f"Available Models: {models_preview}..."
            )
            QMessageBox.information(self, f"{prov_name} Connection Test", msg)
        else:
            msg = (
                f"⚠️ Connection Test Failed for {prov_name}:\n\n"
                f"{health.status_message}\n\n"
                f"Please ensure an API key is configured and network is reachable."
            )
            QMessageBox.warning(self, f"{prov_name} Connection Test", msg)
        self._quick_health_probe()

    def _on_unload_model(self) -> None:
        """Explicitly unload local model from memory."""
        mgr = self.controller.get_ai_manager()

        loop = asyncio.new_event_loop()
        try:
            success = loop.run_until_complete(mgr.local_engine.unload_active_model())
        finally:
            loop.close()

        msg = "Model unloaded from RAM successfully." if success else "Ollama is not running or model was already unloaded."
        QMessageBox.information(self, "Model Unload", msg)

    def _on_delete_model(self) -> None:
        """Delete selected local model from disk."""
        model_name = self.local_model_combo.currentText().strip()
        if not model_name:
            return

        confirm = QMessageBox.question(
            self,
            "Delete Local Model",
            f"Are you sure you want to delete local model '{model_name}' to free SSD space?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            mgr = self.controller.get_ai_manager()
            loop = asyncio.new_event_loop()
            try:
                success = loop.run_until_complete(mgr.local_engine.delete_model(model_name))
            finally:
                loop.close()

            if success:
                QMessageBox.information(self, "Model Deleted", f"Model '{model_name}' has been deleted from Ollama.")
                self._refresh_local_models()
            else:
                QMessageBox.warning(self, "Delete Failed", f"Could not delete '{model_name}'. Is Ollama running?")

    def _on_pull_model(self) -> None:
        """Initiate model download with hardware safety check."""
        model_name = self.txt_pull_model.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Model Required", "Please enter a model name to pull (e.g. llama3.2:1b).")
            return

        # Hardware safety check for 8 GB RAM
        lower_name = model_name.lower()
        if any(large in lower_name for large in ["7b", "8b", "13b", "14b", "32b", "70b"]):
            confirm = QMessageBox.question(
                self,
                "⚠️ High RAM Warning (Hardware Safeguard)",
                f"Model '{model_name}' appears to be a 7B+ parameter model.\n\n"
                f"Target Machine: Intel Core i5-7200U with 8 GB RAM.\n"
                f"Running models 7B+ will consume >4.5 GB RAM and may cause severe OS paging or system freezing.\n\n"
                f"We strongly recommend 'llama3.2:1b' (1.3 GB) or 'qwen2.5:1.5b' (1.6 GB).\n\n"
                f"Do you still want to proceed with downloading '{model_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return

        # Launch download worker
        self.btn_pull.setEnabled(False)
        self.btn_cancel_pull.setVisible(True)
        self.pull_progress_bar.setVisible(True)
        self.pull_progress_bar.setValue(0)
        self.lbl_pull_status.setText(f"Starting download for '{model_name}'...")

        self._download_worker = ModelDownloadWorker(
            model_name=model_name,
            manager=self.controller.get_ai_manager(),
            parent=self,
        )
        self._download_worker.signals.progress.connect(self._on_download_progress)
        self._download_worker.signals.finished.connect(self._on_download_finished)
        self._download_worker.signals.error.connect(self._on_download_error)
        self._download_worker.signals.cancelled.connect(self._on_download_cancelled)
        self._download_worker.start()

    def _on_download_progress(self, status: str, completed: int, total: int, percent: float) -> None:
        """Update download progress bar and status text."""
        self.pull_progress_bar.setValue(int(percent))
        if total > 0:
            mb_done = completed / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self.lbl_pull_status.setText(f"{status} ({mb_done:.1f} MB / {mb_total:.1f} MB - {percent:.1f}%)")
        else:
            self.lbl_pull_status.setText(status)

    def _on_download_finished(self, model_name: str) -> None:
        """Handle download completion."""
        self._reset_download_ui()
        self.lbl_pull_status.setText(f"✅ Model '{model_name}' downloaded successfully!")
        self.lbl_pull_status.setStyleSheet("color: #3fb950; font-size: 11px;")
        self._refresh_local_models()

        # Set as active model in combo
        idx = self.local_model_combo.findText(model_name)
        if idx >= 0:
            self.local_model_combo.setCurrentIndex(idx)
        else:
            self.local_model_combo.addItem(model_name, model_name)
            self.local_model_combo.setCurrentText(model_name)

        QMessageBox.information(
            self,
            "Download Complete",
            f"Local model '{model_name}' is now installed and ready for inference.",
        )

    def _on_download_error(self, err: str) -> None:
        """Handle download error."""
        self._reset_download_ui()
        self.lbl_pull_status.setText(f"❌ Download failed: {err}")
        self.lbl_pull_status.setStyleSheet("color: #ff7b72; font-size: 11px;")
        QMessageBox.warning(self, "Download Error", f"Failed to pull model:\n{err}")

    def _on_download_cancelled(self) -> None:
        """Handle download cancelled by user."""
        self._reset_download_ui()
        self.lbl_pull_status.setText("Download cancelled by user.")
        self.lbl_pull_status.setStyleSheet("color: #8b949e; font-size: 11px;")

    def _on_cancel_pull(self) -> None:
        """Request cancel on active download worker."""
        if self._download_worker and self._download_worker.isRunning():
            self.lbl_pull_status.setText("Cancelling download...")
            self.btn_cancel_pull.setEnabled(False)
            self._download_worker.cancel()

    def _reset_download_ui(self) -> None:
        """Reset download progress widgets."""
        self.btn_pull.setEnabled(True)
        self.btn_cancel_pull.setVisible(False)
        self.btn_cancel_pull.setEnabled(True)
        self.pull_progress_bar.setVisible(False)
        self._download_worker = None

    def _quick_health_probe(self) -> None:
        """Update card status strings based on non-blocking health checks."""
        mgr = self.controller.get_ai_manager()

        async def _probe():
            try:
                results = await mgr.check_all_health()
                loc = results["local"]
                cld = results["cloud"]

                if loc.is_healthy:
                    self.lc_status.setText(f"● Ollama Connected ({len(loc.available_models)} models installed)")
                    self.lc_status.setStyleSheet("font-size: 14px; font-weight: 600; color: #3fb950;")
                else:
                    self.lc_status.setText("● Ollama Standby / Offline")
                    self.lc_status.setStyleSheet("font-size: 14px; font-weight: 600; color: #d29922;")

                if cld.is_healthy:
                    self.cc_status.setText("● Cloud Connected (Authenticated)")
                    self.cc_status.setStyleSheet("font-size: 14px; font-weight: 600; color: #3fb950;")
                else:
                    self.cc_status.setText("● Cloud Standby / Not Configured")
                    self.cc_status.setStyleSheet("font-size: 14px; font-weight: 600; color: #d29922;")
            except Exception as exc:
                logger.debug("Quick health probe error: %s", exc)

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.create_task(_probe())
        except RuntimeError:
            pass

    def _on_check_health(self) -> None:
        """Run full synchronous health check and display structured dialog."""
        mgr = self.controller.get_ai_manager()

        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(mgr.check_all_health())
        finally:
            loop.close()

        loc = results["local"]
        cld = results["cloud"]

        loc_icon = "✅" if loc.is_healthy else "⚠️"
        cld_icon = "✅" if cld.is_healthy else "⚠️"

        report = (
            f"=== HAKS AI Engine Health Check ===\n\n"
            f"{loc_icon} LOCAL AI (Ollama):\n"
            f"  Status: {'Healthy & Active' if loc.is_healthy else 'Offline / Standby'}\n"
            f"  Details: {loc.status_message}\n"
            f"  Installed Models: {', '.join(loc.available_models) if loc.available_models else 'None'}\n\n"
            f"{cld_icon} CLOUD AI ADAPTER:\n"
            f"  Status: {'Healthy & Connected' if cld.is_healthy else 'Unconfigured / Offline'}\n"
            f"  Details: {cld.status_message}\n"
            f"  Supported Models: {', '.join(cld.available_models[:4])}...\n\n"
            f"Security Status: Windows Credential Manager integration active.\n"
            f"Memory Isolation: Personal and Work DBs excluded from cloud transport."
        )

        QMessageBox.information(self, "AI Engine Diagnostics", report)
        self._quick_health_probe()

    def close(self) -> bool:
        """Clean up download worker on view close."""
        if self._download_worker and self._download_worker.isRunning():
            self._download_worker.cancel()
            self._download_worker.wait(1000)
        return super().close()

    def closeEvent(self, event) -> None:
        """Clean up download worker on event close."""
        if self._download_worker and self._download_worker.isRunning():
            self._download_worker.cancel()
            self._download_worker.wait(1000)
        super().closeEvent(event)
