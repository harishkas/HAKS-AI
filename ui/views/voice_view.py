"""Voice Assistant View for HAKS.

Provides complete Phase 6 Voice controls:
- Live animated state badge (STANDBY, LISTENING, THINKING, SPEAKING, etc.)
- Real-time VU-meter microphone activity monitor
- Hardware device selection (Microphones from sounddevice)
- Wake word configuration ('Hey HAKS' local acoustic & Porcupine)
- Speech transcription & spoken response live transcripts
- Bilingual language routing & TTS voice selection
- Push-to-talk, mute, sleep, and instant barge-in controls
"""

from typing import Optional

from PySide6.QtCore import Qt, QTimer
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
    QProgressBar,
    QSlider,
    QScrollArea,
)

from core.app_controller import AppController, get_app_controller
from core.logger import get_logger
from voice.audio_capture import AudioCaptureService

logger = get_logger("ui.views.voice_view")


class VoiceView(QWidget):
    """Voice Assistant control, live status monitor, and audio hardware configuration."""

    def __init__(self, controller: Optional[AppController] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.controller = controller or get_app_controller()
        self._init_ui()
        self._connect_signals()
        self._load_initial_values()
        self._update_ui_state(self.controller.get_voice_state())

    def _init_ui(self) -> None:
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background: transparent;")

        self.container = QWidget()
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        # 1. Header with Start / Stop Master Switch
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Voice Assistant")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #f0f6fc;")
        subtitle = QLabel("Continuous bilingual Tamil & English voice conversation engine with barge-in")
        subtitle.setStyleSheet("color: #8b949e; font-size: 12px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        self.btn_master_power = QPushButton("▶ Start Voice Assistant")
        self.btn_master_power.setProperty("class", "btn_primary")
        self.btn_master_power.clicked.connect(self._toggle_master_power)
        header.addWidget(self.btn_master_power)

        layout.addLayout(header)

        # 2. Main Visual Voice State & VU Meter Card
        self.state_card = QFrame()
        self.state_card.setProperty("class", "stat_card")
        sc_layout = QVBoxLayout(self.state_card)
        sc_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sc_layout.setSpacing(8)

        self.lbl_wake_phrase = QLabel('Wake Phrase: "Hey HAKS" (ஹே ஹாக்ஸ்)')
        self.lbl_wake_phrase.setStyleSheet("font-size: 13px; font-weight: 600; color: #58a6ff;")
        sc_layout.addWidget(self.lbl_wake_phrase, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_state_display = QLabel("STANDBY")
        self.lbl_state_display.setStyleSheet(
            "font-size: 28px; font-weight: 800; color: #3fb950; padding: 10px;"
        )
        sc_layout.addWidget(self.lbl_state_display, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_state_desc = QLabel("HAKS is idling and waiting for wake phrase or push-to-talk.")
        self.lbl_state_desc.setStyleSheet("color: #8b949e; font-size: 12px;")
        sc_layout.addWidget(self.lbl_state_desc, alignment=Qt.AlignmentFlag.AlignCenter)

        # Live VU Meter (Mic Audio Activity)
        vu_box = QHBoxLayout()
        lbl_vu = QLabel("Mic Level:")
        lbl_vu.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.vu_meter = QProgressBar()
        self.vu_meter.setRange(0, 100)
        self.vu_meter.setValue(0)
        self.vu_meter.setFixedHeight(8)
        self.vu_meter.setTextVisible(False)
        self.vu_meter.setStyleSheet(
            """
            QProgressBar {
                background-color: #21262d;
                border-radius: 4px;
                border: 1px solid #30363d;
            }
            QProgressBar::chunk {
                background-color: #3fb950;
                border-radius: 4px;
            }
            """
        )
        vu_box.addWidget(lbl_vu)
        vu_box.addWidget(self.vu_meter, stretch=1)
        sc_layout.addLayout(vu_box)

        layout.addWidget(self.state_card)

        # 3. Live Transcripts Card
        conv_card = QFrame()
        conv_card.setProperty("class", "stat_card")
        cc_layout = QVBoxLayout(conv_card)
        cc_layout.setSpacing(8)

        cc_title = QLabel("LIVE CONVERSATION TRANSCRIPTS")
        cc_title.setProperty("class", "card_title")
        cc_layout.addWidget(cc_title)

        self.lbl_user_transcript = QLabel("<b>You:</b> <i>(Awaiting speech...)</i>")
        self.lbl_user_transcript.setStyleSheet("color: #c9d1d9; font-size: 13px; padding: 4px;")
        self.lbl_user_transcript.setWordWrap(True)
        cc_layout.addWidget(self.lbl_user_transcript)

        self.lbl_spoken_response = QLabel("<b>HAKS:</b> <i>(Standing by...)</i>")
        self.lbl_spoken_response.setStyleSheet("color: #58a6ff; font-size: 13px; padding: 4px;")
        self.lbl_spoken_response.setWordWrap(True)
        cc_layout.addWidget(self.lbl_spoken_response)

        layout.addWidget(conv_card)

        # 4. Interactive Voice Action Buttons
        cmd_grid = QGridLayout()
        cmd_grid.setSpacing(10)

        self.btn_ptt = QPushButton("🎙️ Push-to-Talk (Start Listening)")
        self.btn_ptt.setProperty("class", "btn_primary")
        self.btn_ptt.clicked.connect(self._on_ptt_clicked)
        cmd_grid.addWidget(self.btn_ptt, 0, 0)

        self.btn_interrupt = QPushButton("⏹️ Interrupt / Barge-in")
        self.btn_interrupt.setProperty("class", "btn_danger")
        self.btn_interrupt.clicked.connect(self._on_interrupt_clicked)
        cmd_grid.addWidget(self.btn_interrupt, 0, 1)

        self.btn_simulate_wake = QPushButton("⚡ Simulate 'Hey HAKS'")
        self.btn_simulate_wake.setProperty("class", "btn_secondary")
        self.btn_simulate_wake.clicked.connect(self._on_simulate_wake)
        cmd_grid.addWidget(self.btn_simulate_wake, 0, 2)

        self.btn_mic_toggle = QPushButton("🔇 Mute Microphone")
        self.btn_mic_toggle.setProperty("class", "btn_secondary")
        self.btn_mic_toggle.clicked.connect(self._on_toggle_mic)
        cmd_grid.addWidget(self.btn_mic_toggle, 1, 0)

        self.btn_sleep = QPushButton("💤 Go to Sleep")
        self.btn_sleep.setProperty("class", "btn_secondary")
        self.btn_sleep.clicked.connect(self._toggle_sleep)
        cmd_grid.addWidget(self.btn_sleep, 1, 1)

        self.btn_standby = QPushButton("🔄 Reset to Standby")
        self.btn_standby.setProperty("class", "btn_secondary")
        self.btn_standby.clicked.connect(lambda: self.controller.set_voice_state("STANDBY"))
        cmd_grid.addWidget(self.btn_standby, 1, 2)

        layout.addLayout(cmd_grid)

        # 5. Audio Hardware & Wake Word Configuration Card
        hw_card = QFrame()
        hw_card.setProperty("class", "stat_card")
        hw_layout = QVBoxLayout(hw_card)
        hw_layout.setSpacing(10)

        hw_title = QLabel("AUDIO HARDWARE & WAKE WORD ENGINE")
        hw_title.setProperty("class", "card_title")
        hw_layout.addWidget(hw_title)

        # Input device dropdown
        dev_row = QHBoxLayout()
        lbl_mic = QLabel("Microphone Device:")
        lbl_mic.setStyleSheet("color: #8b949e; font-size: 12px; width: 140px;")
        self.combo_devices = QComboBox()
        self.combo_devices.setMinimumWidth(320)
        self.combo_devices.currentIndexChanged.connect(self._on_device_changed)
        dev_row.addWidget(lbl_mic)
        dev_row.addWidget(self.combo_devices)
        dev_row.addStretch()
        hw_layout.addLayout(dev_row)

        # Wake word engine selector
        wake_row = QHBoxLayout()
        lbl_ww = QLabel("Wake Word Engine:")
        lbl_ww.setStyleSheet("color: #8b949e; font-size: 12px; width: 140px;")
        self.combo_wake_engine = QComboBox()
        self.combo_wake_engine.addItem("Local Acoustic ('Hey HAKS' - Offline CPU)", "local")
        self.combo_wake_engine.addItem("Picovoice Porcupine (Requires AccessKey & .ppn)", "porcupine")
        self.combo_wake_engine.currentIndexChanged.connect(self._on_wake_engine_changed)
        wake_row.addWidget(lbl_ww)
        wake_row.addWidget(self.combo_wake_engine)
        wake_row.addStretch()
        hw_layout.addLayout(wake_row)

        # Wake sensitivity slider
        sens_row = QHBoxLayout()
        lbl_sens = QLabel("Wake Sensitivity:")
        lbl_sens.setStyleSheet("color: #8b949e; font-size: 12px; width: 140px;")
        self.slider_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_sens.setRange(10, 100)
        self.slider_sens.setValue(int(self.controller.config.voice.wake_sensitivity * 100))
        self.lbl_sens_val = QLabel(f"{self.controller.config.voice.wake_sensitivity:.2f}")
        self.slider_sens.valueChanged.connect(self._on_sensitivity_changed)
        sens_row.addWidget(lbl_sens)
        sens_row.addWidget(self.slider_sens, stretch=1)
        sens_row.addWidget(self.lbl_sens_val)
        hw_layout.addLayout(sens_row)

        layout.addWidget(hw_card)

        # 6. Bilingual Speech & Voice Synthesis Card
        speech_card = QFrame()
        speech_card.setProperty("class", "stat_card")
        sp_layout = QVBoxLayout(speech_card)
        sp_layout.setSpacing(10)

        sp_title = QLabel("SPEECH RECOGNITION & TAMIL/ENGLISH TTS")
        sp_title.setProperty("class", "card_title")
        sp_layout.addWidget(sp_title)

        # Language preference
        lang_row = QHBoxLayout()
        lbl_lang = QLabel("Speech Language:")
        lbl_lang.setStyleSheet("color: #8b949e; font-size: 12px; width: 140px;")
        self.lang_combo = QComboBox()
        self.lang_combo.addItem("Mixed (Tamil + English)", "mixed")
        self.lang_combo.addItem("Tamil Only (தமிழ்)", "ta")
        self.lang_combo.addItem("English Only", "en")
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_row.addWidget(lbl_lang)
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch()
        sp_layout.addLayout(lang_row)

        # STT Model Size
        stt_row = QHBoxLayout()
        lbl_stt = QLabel("Whisper Model Size:")
        lbl_stt.setStyleSheet("color: #8b949e; font-size: 12px; width: 140px;")
        self.combo_stt_size = QComboBox()
        self.combo_stt_size.addItem("base (Recommended for i5-7200U, ~140 MB)", "base")
        self.combo_stt_size.addItem("tiny (Fastest CPU latency, ~75 MB)", "tiny")
        self.combo_stt_size.addItem("small (Higher accuracy, ~460 MB)", "small")
        self.combo_stt_size.currentIndexChanged.connect(self._on_stt_model_changed)
        stt_row.addWidget(lbl_stt)
        stt_row.addWidget(self.combo_stt_size)
        stt_row.addStretch()
        sp_layout.addLayout(stt_row)

        # TTS Engine & Tamil Voice
        tts_row = QHBoxLayout()
        lbl_tts = QLabel("TTS Voice Engine:")
        lbl_tts.setStyleSheet("color: #8b949e; font-size: 12px; width: 140px;")
        self.combo_tts_engine = QComboBox()
        self.combo_tts_engine.addItem("Microsoft Edge Neural (Authentic Tamil & English)", "edge")
        self.combo_tts_engine.addItem("Windows SAPI (100% Offline Local Speech)", "sapi")
        self.combo_tts_engine.addItem("Piper Local ONNX (Community Models)", "piper")
        self.combo_tts_engine.currentIndexChanged.connect(self._on_tts_engine_changed)
        tts_row.addWidget(lbl_tts)
        tts_row.addWidget(self.combo_tts_engine)
        tts_row.addStretch()
        sp_layout.addLayout(tts_row)

        # Speech Speed Slider
        speed_row = QHBoxLayout()
        lbl_speed = QLabel("Speaking Speed:")
        lbl_speed.setStyleSheet("color: #8b949e; font-size: 12px; width: 140px;")
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(75, 150)
        self.slider_speed.setValue(int(self.controller.config.voice.speech_speed * 100))
        self.lbl_speed_val = QLabel(f"{self.controller.config.voice.speech_speed:.2f}x")
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(lbl_speed)
        speed_row.addWidget(self.slider_speed, stretch=1)
        speed_row.addWidget(self.lbl_speed_val)
        sp_layout.addLayout(speed_row)

        # Continuous mode & barge-in checkboxes
        self.chk_continuous = QCheckBox("Enable Continuous Conversation (multi-turn voice without re-waking)")
        self.chk_continuous.setChecked(self.controller.config.voice.continuous_mode)
        self.chk_continuous.toggled.connect(self._on_continuous_toggled)
        sp_layout.addWidget(self.chk_continuous)

        self.chk_barge_in = QCheckBox("Enable Voice Interruption / Barge-in (stop speaking when user interrupts)")
        self.chk_barge_in.setChecked(self.controller.config.voice.enable_barge_in)
        self.chk_barge_in.toggled.connect(self._on_barge_in_toggled)
        sp_layout.addWidget(self.chk_barge_in)

        layout.addWidget(speech_card)

        # Set scroll widget layout
        self.scroll.setWidget(self.container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.scroll)

    def _connect_signals(self) -> None:
        self.controller.events.voice_state_changed.connect(self._update_ui_state)
        self.controller.events.voice_mute_toggled.connect(self._update_mic_ui)
        self.controller.events.voice_audio_level_updated.connect(self._update_vu_meter)
        self.controller.events.voice_transcript_received.connect(self._update_user_transcript)
        self.controller.events.voice_speaking_started.connect(self._update_spoken_response)

    def _load_initial_values(self) -> None:
        """Populate device lists and configuration controls."""
        # Populate input devices
        self.combo_devices.blockSignals(True)
        self.combo_devices.clear()
        devices = AudioCaptureService.list_input_devices()
        cur_idx = self.controller.config.voice.input_device_index
        selected_combo_idx = 0
        for i, dev in enumerate(devices):
            display = f"{dev.name} ({dev.host_api})"
            if dev.is_default_input:
                display += " [Default]"
            self.combo_devices.addItem(display, dev.index)
            if cur_idx is not None and dev.index == cur_idx:
                selected_combo_idx = i
            elif cur_idx is None and dev.is_default_input:
                selected_combo_idx = i

        if self.combo_devices.count() > 0:
            self.combo_devices.setCurrentIndex(selected_combo_idx)
        self.combo_devices.blockSignals(False)

        # Set Preferred Language
        pref_lang = self.controller.config.voice.preferred_language
        for i in range(self.lang_combo.count()):
            if self.lang_combo.itemData(i) == pref_lang:
                self.lang_combo.setCurrentIndex(i)
                break

    # --------------------------------------------------------------------------
    # UI Actions & Control Handlers
    # --------------------------------------------------------------------------

    def _toggle_master_power(self) -> None:
        state = self.controller.get_voice_state()
        if state == "STOPPED":
            success = self.controller.start_voice_assistant()
            if success:
                self.btn_master_power.setText("⏹ Stop Voice Assistant")
                self.btn_master_power.setProperty("class", "btn_danger")
        else:
            self.controller.stop_voice_assistant()
            self.btn_master_power.setText("▶ Start Voice Assistant")
            self.btn_master_power.setProperty("class", "btn_primary")
        self.btn_master_power.style().unpolish(self.btn_master_power)
        self.btn_master_power.style().polish(self.btn_master_power)

    def _on_ptt_clicked(self) -> None:
        self.controller.trigger_push_to_talk()

    def _set_state(self, state: str) -> None:
        """Helper to set voice state via controller."""
        self.controller.set_voice_state(state)

    def _on_interrupt_clicked(self) -> None:
        self.controller.interrupt_voice()

    def _on_simulate_wake(self) -> None:
        if hasattr(self.controller, "voice_engine"):
            self.controller.voice_engine._handle_wake_word_detected()
        else:
            self.controller.set_voice_state("WAKE_DETECTED")

    def _toggle_sleep(self) -> None:
        current = self.controller.get_voice_state()
        if current == "SLEEP":
            self.controller.set_voice_state("STANDBY")
            self.btn_sleep.setText("💤 Go to Sleep")
        else:
            self.controller.set_voice_state("SLEEP")
            self.btn_sleep.setText("☀️ Wake Up from Sleep")

    def _on_toggle_mic(self) -> None:
        self.controller.toggle_mic_mute()

    def _on_device_changed(self, idx: int) -> None:
        dev_index = self.combo_devices.itemData(idx)
        if hasattr(self.controller, "voice_engine"):
            self.controller.voice_engine.audio_capture.set_device_index(dev_index)
        self.controller.config.voice.input_device_index = dev_index

    def _on_wake_engine_changed(self, idx: int) -> None:
        eng = self.combo_wake_engine.itemData(idx)
        self.controller.config.voice.wake_word_engine = eng

    def _on_sensitivity_changed(self, val: int) -> None:
        sens = val / 100.0
        self.lbl_sens_val.setText(f"{sens:.2f}")
        self.controller.config.voice.wake_sensitivity = sens
        if hasattr(self.controller, "voice_engine"):
            self.controller.voice_engine.local_wake_detector.set_sensitivity(sens)
            self.controller.voice_engine.porcupine_wake_detector.set_sensitivity(sens)

    def _on_language_changed(self, idx: int) -> None:
        lang = self.lang_combo.itemData(idx)
        self.controller.config.voice.preferred_language = lang

    def _on_stt_model_changed(self, idx: int) -> None:
        size = self.combo_stt_size.itemData(idx)
        self.controller.config.voice.whisper_model_size = size
        if hasattr(self.controller, "voice_engine"):
            self.controller.voice_engine.stt.model_size = size

    def _on_tts_engine_changed(self, idx: int) -> None:
        eng = self.combo_tts_engine.itemData(idx)
        self.controller.config.voice.tts_engine = eng

    def _on_speed_changed(self, val: int) -> None:
        speed = val / 100.0
        self.lbl_speed_val.setText(f"{speed:.2f}x")
        self.controller.config.voice.speech_speed = speed

    def _on_continuous_toggled(self, checked: bool) -> None:
        self.controller.config.voice.continuous_mode = checked

    def _on_barge_in_toggled(self, checked: bool) -> None:
        self.controller.config.voice.enable_barge_in = checked

    # --------------------------------------------------------------------------
    # Signal Event Handlers
    # --------------------------------------------------------------------------

    def _update_mic_ui(self, is_muted: bool) -> None:
        if is_muted:
            self.btn_mic_toggle.setText("🔊 Unmute Microphone")
            self.btn_mic_toggle.setStyleSheet("color: #f85149; border-color: #da3633;")
        else:
            self.btn_mic_toggle.setText("🔇 Mute Microphone")
            self.btn_mic_toggle.setStyleSheet("")

    def _update_vu_meter(self, level: float) -> None:
        val = int(min(100, max(0, level * 100)))
        self.vu_meter.setValue(val)

    def _update_user_transcript(self, text: str, lang: str) -> None:
        lang_badge = f"<span style='color:#58a6ff;'>[{lang.upper()}]</span>"
        self.lbl_user_transcript.setText(f"<b>You:</b> {lang_badge} {text}")

    def _update_spoken_response(self, text: str) -> None:
        self.lbl_spoken_response.setText(f"<b>HAKS:</b> 🔊 {text}")

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
            "SLEEP": ("#8b949e", "Voice engine sleeping. Say 'Wake up HAKS' or press Wake Up."),
            "STOPPED": ("#8b949e", "Voice pipeline stopped. Click 'Start Voice Assistant' above."),
        }
        color, desc = state_colors.get(state, ("#3fb950", "Active"))
        self.lbl_state_display.setStyleSheet(
            f"font-size: 28px; font-weight: 800; color: {color}; padding: 10px;"
        )
        self.lbl_state_desc.setText(desc)

        if state == "STOPPED":
            self.btn_master_power.setText("▶ Start Voice Assistant")
            self.btn_master_power.setProperty("class", "btn_primary")
        else:
            self.btn_master_power.setText("⏹ Stop Voice Assistant")
            self.btn_master_power.setProperty("class", "btn_danger")
        self.btn_master_power.style().unpolish(self.btn_master_power)
        self.btn_master_power.style().polish(self.btn_master_power)
