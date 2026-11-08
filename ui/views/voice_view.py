"""Voice Assistant View for HAKS.

Provides interactive voice status visualization, continuous conversation controls,
wake phrase configuration ("Hey HAKS"), language selection, and manual voice commands.
"""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QFrame,
    QGridLayout,
    QCheckBox,
)

from core.app_controller import AppController, get_app_controller
from core.logger import get_logger

logger = get_logger("ui.views.voice_view")


class VoiceView(QWidget):
    """Voice Assistant control and status monitor."""

    def __init__(self, controller: Optional[AppController] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller or get_app_controller()
        self._init_ui()
        self._connect_signals()
        self._update_ui_state(self.controller.get_voice_state())

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # 1. Header
        header = QHBoxLayout()
        title = QLabel("Voice Assistant")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #f0f6fc;")
        subtitle = QLabel("Continuous Tamil & English voice conversation engine")
        subtitle.setStyleSheet("color: #8b949e; font-size: 12px; margin-left: 8px;")
        header.addWidget(title)
        header.addWidget(subtitle)
        header.addStretch()
        layout.addLayout(header)

        # 2. Main Visual Voice State Card
        self.state_card = QFrame()
        self.state_card.setProperty("class", "stat_card")
        sc_layout = QVBoxLayout(self.state_card)
        sc_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sc_layout.setSpacing(8)

        self.lbl_wake_phrase = QLabel('Wake Phrase: "Hey HAKS"')
        self.lbl_wake_phrase.setStyleSheet("font-size: 13px; font-weight: 600; color: #58a6ff;")

        self.lbl_state_display = QLabel("STANDBY")
        self.lbl_state_display.setStyleSheet(
            "font-size: 28px; font-weight: 800; color: #3fb950; padding: 12px;"
        )

        self.lbl_state_desc = QLabel("HAKS is idling and waiting for wake phrase or push-to-talk.")
        self.lbl_state_desc.setStyleSheet("color: #8b949e; font-size: 12px;")

        sc_layout.addWidget(self.lbl_wake_phrase, alignment=Qt.AlignmentFlag.AlignCenter)
        sc_layout.addWidget(self.lbl_state_display, alignment=Qt.AlignmentFlag.AlignCenter)
        sc_layout.addWidget(self.lbl_state_desc, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.state_card)

        # 3. Interactive Voice Action Buttons
        cmd_grid = QGridLayout()
        cmd_grid.setSpacing(10)

        self.btn_wake = QPushButton("Simulate Wake ('Hey HAKS')")
        self.btn_wake.setProperty("class", "btn_primary")
        self.btn_wake.clicked.connect(lambda: self._set_state("WAKE_DETECTED"))
        cmd_grid.addWidget(self.btn_wake, 0, 0)

        self.btn_listen = QPushButton("Start Listening")
        self.btn_listen.setProperty("class", "btn_secondary")
        self.btn_listen.clicked.connect(lambda: self._set_state("LISTENING"))
        cmd_grid.addWidget(self.btn_listen, 0, 1)

        self.btn_interrupt = QPushButton("Interrupt / Barge-in")
        self.btn_interrupt.setProperty("class", "btn_danger")
        self.btn_interrupt.clicked.connect(lambda: self._set_state("INTERRUPTED"))
        cmd_grid.addWidget(self.btn_interrupt, 0, 2)

        self.btn_sleep = QPushButton("Go to Sleep")
        self.btn_sleep.setProperty("class", "btn_secondary")
        self.btn_sleep.clicked.connect(self._toggle_sleep)
        cmd_grid.addWidget(self.btn_sleep, 1, 0)

        self.btn_mic_toggle = QPushButton("Mute Microphone")
        self.btn_mic_toggle.setProperty("class", "btn_secondary")
        self.btn_mic_toggle.clicked.connect(self._on_toggle_mic)
        cmd_grid.addWidget(self.btn_mic_toggle, 1, 1)

        self.btn_standby = QPushButton("Reset to Standby")
        self.btn_standby.setProperty("class", "btn_secondary")
        self.btn_standby.clicked.connect(lambda: self._set_state("STANDBY"))
        cmd_grid.addWidget(self.btn_standby, 1, 2)

        layout.addLayout(cmd_grid)

        # 4. Settings Card
        settings_card = QFrame()
        settings_card.setProperty("class", "stat_card")
        set_layout = QVBoxLayout(settings_card)
        set_layout.setSpacing(10)

        card_title = QLabel("VOICE ENGINE CONFIGURATION")
        card_title.setProperty("class", "card_title")
        set_layout.addWidget(card_title)

        # Language row
        lang_row = QHBoxLayout()
        lang_lbl = QLabel("Preferred Language:")
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["Mixed (Tamil + English)", "Tamil Only", "English Only"])
        lang_row.addWidget(lang_lbl)
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch()
        set_layout.addLayout(lang_row)

        # Continuous mode checkbox
        self.chk_continuous = QCheckBox("Enable Continuous Conversation (multi-turn voice without re-waking)")
        self.chk_continuous.setChecked(self.controller.config.voice.continuous_mode)
        set_layout.addWidget(self.chk_continuous)

        layout.addWidget(settings_card)
        layout.addStretch()

    def _connect_signals(self) -> None:
        self.controller.events.voice_state_changed.connect(self._update_ui_state)
        self.controller.events.voice_mute_toggled.connect(self._update_mic_ui)

    def _set_state(self, state: str) -> None:
        self.controller.set_voice_state(state)

    def _toggle_sleep(self) -> None:
        current = self.controller.get_voice_state()
        if current == "SLEEP":
            self._set_state("STANDBY")
            self.btn_sleep.setText("Go to Sleep")
        else:
            self._set_state("SLEEP")
            self.btn_sleep.setText("Wake Up from Sleep")

    def _on_toggle_mic(self) -> None:
        self.controller.toggle_mic_mute()

    def _update_mic_ui(self, is_muted: bool) -> None:
        if is_muted:
            self.btn_mic_toggle.setText("Unmute Microphone")
            self.btn_mic_toggle.setStyleSheet("color: #f85149; border-color: #da3633;")
        else:
            self.btn_mic_toggle.setText("Mute Microphone")
            self.btn_mic_toggle.setStyleSheet("")

    def _update_ui_state(self, state: str) -> None:
        self.lbl_state_display.setText(state)
        state_colors = {
            "STANDBY": ("#3fb950", "Waiting for wake phrase 'Hey HAKS' or push-to-talk."),
            "WAKE_DETECTED": ("#58a6ff", "Wake phrase detected! Initializing voice channel..."),
            "LISTENING": ("#ff7b72", "Microphone actively listening for speech..."),
            "PROCESSING": ("#d29922", "Voice Activity Detection segmenting audio buffer..."),
            "THINKING": ("#a371f7", "AI Router processing speech transcript..."),
            "SPEAKING": ("#2ea043", "Synthesizing Tamil/English audio response..."),
            "INTERRUPTED": ("#f85149", "Speech interrupted by user barge-in!"),
            "SLEEP": ("#8b949e", "Voice engine sleeping. Say 'Wake up' or press Wake Up."),
            "STOPPED": ("#f85149", "Voice pipeline stopped."),
        }
        color, desc = state_colors.get(state, ("#3fb950", "Active"))
        self.lbl_state_display.setStyleSheet(
            f"font-size: 28px; font-weight: 800; color: {color}; padding: 12px;"
        )
        self.lbl_state_desc.setText(desc)
