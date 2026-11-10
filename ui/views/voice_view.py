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
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #111827;")
        subtitle = QLabel("Continuous bilingual Tamil & English voice conversation engine with barge-in")
        subtitle.setStyleSheet("color: #4b5563; font-size: 12px;")
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
        self.lbl_wake_phrase.setStyleSheet("font-size: 13px; font-weight: 600; color: #0067c0;")
        sc_layout.addWidget(self.lbl_wake_phrase, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_state_display = QLabel("STANDBY")
        self.lbl_state_display.setStyleSheet(
            "font-size: 28px; font-weight: 800; color: #15803d; padding: 10px;"
        )
        sc_layout.addWidget(self.lbl_state_display, alignment=Qt.AlignmentFlag.AlignCenter)

        self.lbl_state_desc = QLabel("HAKS is idling and waiting for wake phrase or push-to-talk.")
        self.lbl_state_desc.setStyleSheet("color: #4b5563; font-size: 12px;")
        sc_layout.addWidget(self.lbl_state_desc, alignment=Qt.AlignmentFlag.AlignCenter)

        # Live VU Meter (Mic Audio Activity)
        vu_box = QHBoxLayout()
        lbl_vu = QLabel("Mic Level:")
        lbl_vu.setStyleSheet("color: #4b5563; font-size: 11px; width: 110px;")
        self.vu_meter = QProgressBar()
        self.vu_meter.setRange(0, 100)
        self.vu_meter.setValue(0)
        self.vu_meter.setFixedHeight(8)
        self.vu_meter.setTextVisible(False)
        self.vu_meter.setStyleSheet(
            """
            QProgressBar {
                background-color: #f3f4f6;
                border-radius: 4px;
                border: 1px solid #d1d5db;
            }
            QProgressBar::chunk {
                background-color: #15803d;
                border-radius: 4px;
            }
            """
        )
        vu_box.addWidget(lbl_vu)
        vu_box.addWidget(self.vu_meter, stretch=1)
        sc_layout.addLayout(vu_box)

        # Live Speech Probability Meter
        prob_box = QHBoxLayout()
        lbl_prob = QLabel("Speech Prob:")
        lbl_prob.setStyleSheet("color: #4b5563; font-size: 11px;")
        self.prob_meter = QProgressBar()
        self.prob_meter.setRange(0, 100)
        self.prob_meter.setValue(0)
        self.prob_meter.setFixedHeight(8)
        self.prob_meter.setTextVisible(False)
        self.prob_meter.setStyleSheet(
            """
            QProgressBar {
                background-color: #f3f4f6;
                border-radius: 4px;
                border: 1px solid #d1d5db;
            }
            QProgressBar::chunk {
                background-color: #0284c7;
                border-radius: 4px;
            }
            """
        )
        self.lbl_prob_val = QLabel("0%")
        self.lbl_prob_val.setStyleSheet("color: #0284c7; font-size: 11px; font-weight: 600; min-width: 35px;")
        prob_box.addWidget(lbl_prob)
        prob_box.addWidget(self.prob_meter, stretch=1)
        prob_box.addWidget(self.lbl_prob_val)
        sc_layout.addLayout(prob_box)

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
        self.lbl_user_transcript.setStyleSheet("color: #111827; font-size: 13px; padding: 4px;")
        self.lbl_user_transcript.setWordWrap(True)
        cc_layout.addWidget(self.lbl_user_transcript)

        self.lbl_spoken_response = QLabel("<b>HAKS:</b> <i>(Standing by...)</i>")
        self.lbl_spoken_response.setStyleSheet("color: #0067c0; font-size: 13px; padding: 4px;")
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

        # 4b. Voice Mode & Gemini Live Architecture Card
        mode_card = QFrame()
        mode_card.setProperty("class", "stat_card")
        mc_layout = QVBoxLayout(mode_card)
        mc_layout.setSpacing(10)

        mc_title = QLabel("VOICE AI ARCHITECTURE & GEMINI LIVE")
        mc_title.setProperty("class", "card_title")
        mc_layout.addWidget(mc_title)

        # Mode selector row
        mode_row = QHBoxLayout()
        lbl_vmode = QLabel("Voice Engine Mode:")
        lbl_vmode.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
        self.combo_voice_mode = QComboBox()
        self.combo_voice_mode.setMinimumWidth(320)
        self.combo_voice_mode.addItem("Google Gemini Live (Real-Time Bidirectional Speech-to-Speech)", "gemini_live")
        self.combo_voice_mode.addItem("Pipeline Mode (Faster-Whisper STT + Gemini Text + Edge TTS)", "pipeline")
        self.combo_voice_mode.addItem("Offline Mode (Faster-Whisper + Ollama Local + SAPI/Piper)", "offline")
        self.combo_voice_mode.currentIndexChanged.connect(self._on_voice_mode_changed)
        mode_row.addWidget(lbl_vmode)
        mode_row.addWidget(self.combo_voice_mode)
        mode_row.addStretch()
        mc_layout.addLayout(mode_row)

        # Gemini Live Voice selector
        live_voice_row = QHBoxLayout()
        lbl_lvoice = QLabel("Gemini Live Voice:")
        lbl_lvoice.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
        self.combo_live_voice = QComboBox()
        for v in ["Aoede (Default, Warm & Expressive)", "Puck (Energetic)", "Charon (Calm)", "Kore (Bright)", "Fenrir (Authoritative)"]:
            raw_v = v.split()[0]
            self.combo_live_voice.addItem(v, raw_v)
        self.combo_live_voice.currentIndexChanged.connect(self._on_live_voice_changed)
        live_voice_row.addWidget(lbl_lvoice)
        live_voice_row.addWidget(self.combo_live_voice)
        live_voice_row.addStretch()
        mc_layout.addLayout(live_voice_row)

        # Live Status & Latency indicator
        status_row = QHBoxLayout()
        self.lbl_live_status = QLabel("Gemini Live Status: Disconnected")
        self.lbl_live_status.setStyleSheet("color: #0366d6; font-size: 12px; font-weight: 600;")
        self.lbl_live_latency = QLabel("Latency: -- ms")
        self.lbl_live_latency.setStyleSheet("color: #4b5563; font-size: 12px;")
        status_row.addWidget(self.lbl_live_status)
        status_row.addSpacing(20)
        status_row.addWidget(self.lbl_live_latency)
        status_row.addStretch()
        mc_layout.addLayout(status_row)

        layout.addWidget(mode_card)

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
        lbl_mic.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
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
        lbl_ww.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
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
        lbl_sens.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
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

        # 5.5 WebRTC APM & Silero VAD Configuration Card
        apm_card = QFrame()
        apm_card.setProperty("class", "stat_card")
        apm_layout = QVBoxLayout(apm_card)
        apm_layout.setSpacing(10)

        apm_title = QLabel("WEBRTC AUDIO PROCESSING (APM) & ADVANCED VAD")
        apm_title.setProperty("class", "card_title")
        apm_layout.addWidget(apm_title)

        # VAD Engine Selector
        vad_eng_row = QHBoxLayout()
        lbl_vad_eng = QLabel("VAD Engine:")
        lbl_vad_eng.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
        self.combo_vad_engine = QComboBox()
        self.combo_vad_engine.addItem("Silero VAD (Deep Learning - Neural)", "silero")
        self.combo_vad_engine.addItem("WebRTC VAD (Lightweight GMM Fallback)", "webrtc")
        self.combo_vad_engine.currentIndexChanged.connect(self._on_vad_engine_changed)
        vad_eng_row.addWidget(lbl_vad_eng)
        vad_eng_row.addWidget(self.combo_vad_engine)
        vad_eng_row.addStretch()
        apm_layout.addLayout(vad_eng_row)

        # AEC & NS Checkboxes
        self.chk_aec = QCheckBox("Enable Acoustic Echo Cancellation (AEC - speaker output loopback suppression)")
        self.chk_aec.setChecked(getattr(self.controller.config.voice, "enable_aec", True))
        self.chk_aec.toggled.connect(self._on_aec_toggled)
        apm_layout.addWidget(self.chk_aec)

        self.chk_ns = QCheckBox("Enable WebRTC Noise Suppression (NS - filter fan noise, typing, room hum)")
        self.chk_ns.setChecked(getattr(self.controller.config.voice, "enable_ns", True))
        self.chk_ns.toggled.connect(self._on_ns_toggled)
        apm_layout.addWidget(self.chk_ns)

        # VAD Sensitivity / Speech Threshold Slider
        sens_row = QHBoxLayout()
        lbl_v_sens = QLabel("VAD Sensitivity (Threshold):")
        lbl_v_sens.setStyleSheet("color: #4b5563; font-size: 12px; width: 180px;")
        self.slider_vad_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_vad_sens.setRange(10, 95)
        cur_thresh = getattr(self.controller.config.voice, "silero_threshold", 0.55)
        self.slider_vad_sens.setValue(int(cur_thresh * 100))
        self.lbl_vad_sens_val = QLabel(f"{cur_thresh:.2f}")
        self.slider_vad_sens.valueChanged.connect(self._on_vad_sens_changed)
        sens_row.addWidget(lbl_v_sens)
        sens_row.addWidget(self.slider_vad_sens, stretch=1)
        sens_row.addWidget(self.lbl_vad_sens_val)
        apm_layout.addLayout(sens_row)

        # Minimum Speech Duration Slider
        dur_row = QHBoxLayout()
        lbl_dur = QLabel("Minimum Speech Duration:")
        lbl_dur.setStyleSheet("color: #4b5563; font-size: 12px; width: 180px;")
        self.slider_min_speech = QSlider(Qt.Orientation.Horizontal)
        self.slider_min_speech.setRange(50, 500)
        cur_dur = getattr(self.controller.config.voice, "min_speech_duration_ms", 150)
        self.slider_min_speech.setValue(cur_dur)
        self.lbl_min_speech_val = QLabel(f"{cur_dur} ms")
        self.slider_min_speech.valueChanged.connect(self._on_min_speech_changed)
        dur_row.addWidget(lbl_dur)
        dur_row.addWidget(self.slider_min_speech, stretch=1)
        dur_row.addWidget(self.lbl_min_speech_val)
        apm_layout.addLayout(dur_row)

        # End of Speech Silence Timeout Slider
        sil_row = QHBoxLayout()
        lbl_sil = QLabel("End-of-Speech Silence:")
        lbl_sil.setStyleSheet("color: #4b5563; font-size: 12px; width: 180px;")
        self.slider_silence = QSlider(Qt.Orientation.Horizontal)
        self.slider_silence.setRange(150, 1200)
        cur_sil = getattr(self.controller.config.voice, "end_of_speech_silence_ms", 450)
        self.slider_silence.setValue(cur_sil)
        self.lbl_silence_val = QLabel(f"{cur_sil} ms")
        self.slider_silence.valueChanged.connect(self._on_silence_changed)
        sil_row.addWidget(lbl_sil)
        sil_row.addWidget(self.slider_silence, stretch=1)
        sil_row.addWidget(self.lbl_silence_val)
        apm_layout.addLayout(sil_row)

        # Live APM Status & CPU Inference Timing
        self.lbl_telemetry = QLabel("APM Status: Ready | Inference: 0.16 ms | Engine: Silero VAD (ONNX)")
        self.lbl_telemetry.setStyleSheet(
            "background-color: #f3f4f6; border-radius: 4px; padding: 6px 10px; color: #1f2937; font-size: 11px; font-family: monospace;"
        )
        apm_layout.addWidget(self.lbl_telemetry)

        layout.addWidget(apm_card)

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
        lbl_lang.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
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
        lbl_stt.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
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
        lbl_tts.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
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
        lbl_speed.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(75, 150)
        self.slider_speed.setValue(int(self.controller.config.voice.speech_speed * 100))
        self.lbl_speed_val = QLabel(f"{self.controller.config.voice.speech_speed:.2f}x")
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        speed_row.addWidget(lbl_speed)
        speed_row.addWidget(self.slider_speed, stretch=1)
        speed_row.addWidget(self.lbl_speed_val)
        sp_layout.addLayout(speed_row)

        # VAD Sensitivity Dropdown
        vad_row = QHBoxLayout()
        lbl_vadsens = QLabel("VAD Noise Rejection:")
        lbl_vadsens.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
        self.combo_vad_sens = QComboBox()
        self.combo_vad_sens.addItem("Mode 2 (Balanced Speech Detection - Recommended)", 2)
        self.combo_vad_sens.addItem("Mode 3 (Very Aggressive - High Ambient Noise)", 3)
        self.combo_vad_sens.addItem("Mode 1 (Moderate Sensitivity)", 1)
        self.combo_vad_sens.addItem("Mode 0 (Relaxed - Quiet Environment)", 0)
        self.combo_vad_sens.currentIndexChanged.connect(self._on_vad_sensitivity_changed)
        vad_row.addWidget(lbl_vadsens)
        vad_row.addWidget(self.combo_vad_sens)
        vad_row.addStretch()
        sp_layout.addLayout(vad_row)

        # End of Speech Silence Timeout Slider
        silence_row = QHBoxLayout()
        lbl_silence = QLabel("End-of-Speech Silence:")
        lbl_silence.setStyleSheet("color: #4b5563; font-size: 12px; width: 140px;")
        self.slider_silence = QSlider(Qt.Orientation.Horizontal)
        self.slider_silence.setRange(300, 1200)
        self.slider_silence.setValue(self.controller.config.voice.silence_timeout_ms)
        self.lbl_silence_val = QLabel(f"{self.controller.config.voice.silence_timeout_ms} ms")
        self.slider_silence.valueChanged.connect(self._on_silence_timeout_changed)
        silence_row.addWidget(lbl_silence)
        silence_row.addWidget(self.slider_silence, stretch=1)
        silence_row.addWidget(self.lbl_silence_val)
        sp_layout.addLayout(silence_row)

        # Echo cancellation and noise suppression
        self.chk_echo_cancel = QCheckBox("Enable Acoustic Echo Cancellation (prevent speaker feedback false barge-in)")
        self.chk_echo_cancel.setChecked(self.controller.config.voice.enable_echo_cancellation)
        self.chk_echo_cancel.toggled.connect(self._on_echo_cancel_toggled)
        sp_layout.addWidget(self.chk_echo_cancel)

        self.chk_noise_suppress = QCheckBox("Enable Multi-Band Spectral Noise Suppression (fans, clicks, hum)")
        self.chk_noise_suppress.setChecked(self.controller.config.voice.enable_noise_suppression)
        self.chk_noise_suppress.toggled.connect(self._on_noise_suppress_toggled)
        sp_layout.addWidget(self.chk_noise_suppress)

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

        # 7. Human Voice Isolation & Noise Cancellation Card
        dsp_card = QFrame()
        dsp_card.setProperty("class", "stat_card")
        dsp_layout = QVBoxLayout(dsp_card)
        dsp_layout.setSpacing(10)

        dsp_title = QLabel("HUMAN VOICE ISOLATION & NOISE CANCELLATION")
        dsp_title.setProperty("class", "card_title")
        dsp_layout.addWidget(dsp_title)

        # Master Toggles Row
        toggles_row = QHBoxLayout()
        self.chk_noise_suppression = QCheckBox("Real-time Background Noise Suppression")
        self.chk_noise_suppression.setChecked(self.controller.config.voice.enable_noise_suppression)
        self.chk_noise_suppression.toggled.connect(self._on_noise_suppression_toggled)

        self.chk_aec = QCheckBox("Acoustic Echo Cancellation (AEC)")
        self.chk_aec.setChecked(self.controller.config.voice.enable_echo_cancellation)
        self.chk_aec.toggled.connect(self._on_aec_toggled)

        toggles_row.addWidget(self.chk_noise_suppression)
        toggles_row.addSpacing(20)
        toggles_row.addWidget(self.chk_aec)
        toggles_row.addStretch()
        dsp_layout.addLayout(toggles_row)

        # Noise Suppression Strength row
        strength_row = QHBoxLayout()
        lbl_strength = QLabel("Noise Suppression Strength:")
        lbl_strength.setStyleSheet("color: #4b5563; font-size: 12px; width: 180px;")
        self.combo_ns_strength = QComboBox()
        self.combo_ns_strength.addItem("Low (6-10 dB, preserves room warmth)", "low")
        self.combo_ns_strength.addItem("Medium (12-18 dB, recommended for fan/AC)", "medium")
        self.combo_ns_strength.addItem("High (18-24 dB, loud traffic & mechanical noise)", "high")
        self.combo_ns_strength.addItem("Ultra (24+ dB, aggressive suppression)", "ultra")
        self.combo_ns_strength.currentIndexChanged.connect(self._on_ns_strength_changed)
        strength_row.addWidget(lbl_strength)
        strength_row.addWidget(self.combo_ns_strength)
        strength_row.addStretch()
        dsp_layout.addLayout(strength_row)

        # Human Speech Detection Sensitivity slider
        speech_sens_row = QHBoxLayout()
        lbl_ssens = QLabel("Speech Detection Sensitivity:")
        lbl_ssens.setStyleSheet("color: #4b5563; font-size: 12px; width: 180px;")
        self.slider_speech_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_speech_sens.setRange(10, 100)
        self.slider_speech_sens.setValue(int(self.controller.config.voice.human_speech_sensitivity * 100))
        self.lbl_speech_sens_val = QLabel(f"{self.controller.config.voice.human_speech_sensitivity:.2f}")
        self.slider_speech_sens.valueChanged.connect(self._on_speech_sens_changed)
        speech_sens_row.addWidget(lbl_ssens)
        speech_sens_row.addWidget(self.slider_speech_sens, stretch=1)
        speech_sens_row.addWidget(self.lbl_speech_sens_val)
        dsp_layout.addLayout(speech_sens_row)

        # Microphone Input Gain slider
        gain_row = QHBoxLayout()
        lbl_gain = QLabel("Microphone Input Gain:")
        lbl_gain.setStyleSheet("color: #4b5563; font-size: 12px; width: 180px;")
        self.slider_mic_gain = QSlider(Qt.Orientation.Horizontal)
        self.slider_mic_gain.setRange(20, 300)
        self.slider_mic_gain.setValue(int(self.controller.config.voice.mic_input_gain * 100))
        self.lbl_mic_gain_val = QLabel(f"{self.controller.config.voice.mic_input_gain:.2f}x")
        self.slider_mic_gain.valueChanged.connect(self._on_mic_gain_changed)
        gain_row.addWidget(lbl_gain)
        gain_row.addWidget(self.slider_mic_gain, stretch=1)
        gain_row.addWidget(self.lbl_mic_gain_val)
        dsp_layout.addLayout(gain_row)

        # Voice Interruption (Barge-in) Sensitivity slider
        int_sens_row = QHBoxLayout()
        lbl_isens = QLabel("Interruption Sensitivity:")
        lbl_isens.setStyleSheet("color: #4b5563; font-size: 12px; width: 180px;")
        self.slider_interruption_sens = QSlider(Qt.Orientation.Horizontal)
        self.slider_interruption_sens.setRange(10, 100)
        self.slider_interruption_sens.setValue(int(self.controller.config.voice.voice_interruption_sensitivity * 100))
        self.lbl_interruption_sens_val = QLabel(f"{self.controller.config.voice.voice_interruption_sensitivity:.2f}")
        self.slider_interruption_sens.valueChanged.connect(self._on_interruption_sens_changed)
        int_sens_row.addWidget(lbl_isens)
        int_sens_row.addWidget(self.slider_interruption_sens, stretch=1)
        int_sens_row.addWidget(self.lbl_interruption_sens_val)
        dsp_layout.addLayout(int_sens_row)

        # Noise-floor Calibration Row
        cal_row = QHBoxLayout()
        self.btn_calibrate = QPushButton("🎯 Calibrate Room Noise Floor")
        self.btn_calibrate.setProperty("class", "btn_secondary")
        self.btn_calibrate.clicked.connect(self._on_calibrate_clicked)
        self.lbl_cal_status = QLabel("Room noise floor: calibrated baseline ready")
        self.lbl_cal_status.setStyleSheet("color: #4b5563; font-size: 11px;")
        cal_row.addWidget(self.btn_calibrate)
        cal_row.addSpacing(16)
        cal_row.addWidget(self.lbl_cal_status)
        cal_row.addStretch()
        dsp_layout.addLayout(cal_row)

        layout.addWidget(dsp_card)

        # Set scroll widget layout
        self.scroll.setWidget(self.container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.scroll)

    def _connect_signals(self) -> None:
        self.controller.events.voice_state_changed.connect(self._update_ui_state)
        self.controller.events.voice_mute_toggled.connect(self._update_mic_ui)
        self.controller.events.voice_audio_level_updated.connect(self._update_vu_meter)
        if hasattr(self.controller.events, "voice_speech_prob_updated"):
            self.controller.events.voice_speech_prob_updated.connect(self._update_speech_prob)
        self.controller.events.voice_transcript_received.connect(self._update_user_transcript)
        self.controller.events.voice_speaking_started.connect(self._update_spoken_response)
        if hasattr(self.controller.events, "voice_live_status_changed"):
            self.controller.events.voice_live_status_changed.connect(self._update_live_status)
        if hasattr(self.controller.events, "voice_live_latency_updated"):
            self.controller.events.voice_live_latency_updated.connect(self._update_live_latency)
        if hasattr(self.controller.events, "voice_speech_confidence_updated"):
            self.controller.events.voice_speech_confidence_updated.connect(self._update_confidence_meter)
        if hasattr(self.controller.events, "voice_noise_floor_updated"):
            self.controller.events.voice_noise_floor_updated.connect(self._update_noise_floor_display)

        # Telemetry timer
        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.timeout.connect(self._update_telemetry)
        self._telemetry_timer.start(500)

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

        # Select active voice mode
        cur_vmode = self.controller.get_voice_mode()
        for i in range(self.combo_voice_mode.count()):
            if self.combo_voice_mode.itemData(i) == cur_vmode:
                self.combo_voice_mode.setCurrentIndex(i)
                break

        # Select live voice
        cur_lvoice = self.controller.config.voice.gemini_live_voice
        for i in range(self.combo_live_voice.count()):
            if self.combo_live_voice.itemData(i) == cur_lvoice:
                self.combo_live_voice.setCurrentIndex(i)
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
            self.controller.voice_engine.set_input_device(dev_index)
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

    def _on_voice_mode_changed(self, idx: int) -> None:
        mode = self.combo_voice_mode.itemData(idx)
        self.controller.set_voice_mode(mode)

    def _on_live_voice_changed(self, idx: int) -> None:
        v = self.combo_live_voice.itemData(idx)
        self.controller.config.voice.gemini_live_voice = v
        self.controller.set_setting("gemini_live_voice", v, section="voice")

    def _on_continuous_toggled(self, checked: bool) -> None:
        self.controller.config.voice.continuous_mode = checked

    def _on_barge_in_toggled(self, checked: bool) -> None:
        self.controller.config.voice.enable_barge_in = checked

    def _on_noise_suppression_toggled(self, checked: bool) -> None:
        self.controller.set_noise_suppression_enabled(checked)

    def _on_aec_toggled(self, checked: bool) -> None:
        self.controller.set_echo_cancellation_enabled(checked)

    def _on_ns_strength_changed(self, idx: int) -> None:
        val = self.combo_ns_strength.itemData(idx)
        if val:
            self.controller.set_noise_suppression_strength(val)

    def _on_speech_sens_changed(self, val: int) -> None:
        sens = val / 100.0
        self.lbl_speech_sens_val.setText(f"{sens:.2f}")
        self.controller.set_human_speech_sensitivity(sens)

    def _on_mic_gain_changed(self, val: int) -> None:
        gain = val / 100.0
        self.lbl_mic_gain_val.setText(f"{gain:.2f}x")
        self.controller.set_mic_gain(gain)

    def _on_interruption_sens_changed(self, val: int) -> None:
        sens = val / 100.0
        self.lbl_interruption_sens_val.setText(f"{sens:.2f}")
        self.controller.set_voice_interruption_sensitivity(sens)

    def _on_calibrate_clicked(self) -> None:
        self.lbl_cal_status.setText("Calibrating ambient room noise... (keep quiet for 1s)")
        self.btn_calibrate.setEnabled(False)
        self.controller.calibrate_noise_floor()

        # Timer to complete calibration after 1.2s
        QTimer.singleShot(1200, self._finish_calibration_ui)

    def _finish_calibration_ui(self) -> None:
        floor = self.controller.finish_noise_calibration()
        self.lbl_cal_status.setText(f"Calibrated noise floor baseline: {floor:.5f}")
        self.btn_calibrate.setEnabled(True)

    def _toggle_mic_test(self) -> None:
        """Toggle microphone capture for live confidence & audio testing."""
        state = self.controller.get_voice_state()
        if state == "STOPPED":
            success = self.controller.start_voice_assistant()
            if success:
                self.btn_test_mic.setText("⏹ Stop Mic Test")
                self.btn_test_mic.setStyleSheet("color: #dc2626;")
        else:
            self.btn_test_mic.setText("🎙️ Test Microphone & Confidence")
            self.btn_test_mic.setStyleSheet("")

    def _update_confidence_meter(self, confidence: float) -> None:
        val = int(min(100, max(0, confidence * 100)))
        self.conf_meter.setValue(val)
        if val >= 55:
            self.lbl_conf_badge.setText(f"{val}% (Speech Detected)")
            self.lbl_conf_badge.setStyleSheet("color: #15803d; font-size: 11px; font-weight: 700; min-width: 90px;")
        elif val >= 25:
            self.lbl_conf_badge.setText(f"{val}% (Uncertain)")
            self.lbl_conf_badge.setStyleSheet("color: #d97706; font-size: 11px; font-weight: 600; min-width: 90px;")
        else:
            self.lbl_conf_badge.setText(f"{val}% (Silence/Noise)")
            self.lbl_conf_badge.setStyleSheet("color: #6b7280; font-size: 11px; font-weight: 500; min-width: 90px;")

    def _update_noise_floor_display(self, floor: float) -> None:
        self.lbl_cal_status.setText(f"Calibrated noise floor: {floor:.5f}")

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

    def _update_speech_prob(self, prob: float) -> None:
        val = int(min(100, max(0, prob * 100)))
        self.prob_meter.setValue(val)
        self.lbl_prob_val.setText(f"{val}%")

    def _update_telemetry(self) -> None:
        """Periodic UI update of APM processing and VAD inference timing."""
        if not hasattr(self.controller, "voice_engine") or self.controller.voice_engine is None:
            return

        engine = self.controller.voice_engine
        apm_status = engine.apm.get_status()
        vad_engine_name = engine.vad.engine_name
        vad_latency = engine.vad.last_inference_latency_ms
        fallback_flag = " [FALLBACK ACTIVE]" if engine.vad.is_fallback_active else ""

        aec_str = "AEC: ON" if apm_status.aec_enabled else "AEC: OFF"
        ns_str = "NS: ON" if apm_status.ns_enabled else "NS: OFF"
        apm_lat = apm_status.last_process_latency_ms

        telemetry_text = (
            f"APM: {aec_str} | {ns_str} ({apm_lat:.2f}ms) | "
            f"VAD: {vad_engine_name}{fallback_flag} ({vad_latency:.2f}ms) | "
            f"Frames: {apm_status.frames_processed}"
        )
        self.lbl_telemetry.setText(telemetry_text)

    def _on_vad_engine_changed(self, idx: int) -> None:
        engine_name = self.combo_vad_engine.itemData(idx)
        self.controller.config.voice.vad_engine = engine_name
        if hasattr(self.controller, "voice_engine") and self.controller.voice_engine:
            self.controller.voice_engine.vad.set_engine(engine_name)

    def _on_aec_toggled(self, checked: bool) -> None:
        self.controller.config.voice.enable_aec = checked
        if hasattr(self.controller, "voice_engine") and self.controller.voice_engine:
            self.controller.voice_engine.apm.enable_aec = checked

    def _on_ns_toggled(self, checked: bool) -> None:
        self.controller.config.voice.enable_ns = checked
        if hasattr(self.controller, "voice_engine") and self.controller.voice_engine:
            self.controller.voice_engine.apm.enable_ns = checked

    def _on_vad_sens_changed(self, val: int) -> None:
        sens = val / 100.0
        self.lbl_vad_sens_val.setText(f"{sens:.2f}")
        self.controller.config.voice.silero_threshold = sens
        if hasattr(self.controller, "voice_engine") and self.controller.voice_engine:
            self.controller.voice_engine.vad.speech_threshold = sens
            self.controller.voice_engine.interruption_controller.speech_probability_threshold = sens

    def _on_min_speech_changed(self, val: int) -> None:
        self.lbl_min_speech_val.setText(f"{val} ms")
        self.controller.config.voice.min_speech_duration_ms = val
        if hasattr(self.controller, "voice_engine") and self.controller.voice_engine:
            self.controller.voice_engine.vad.min_speech_duration_ms = val

    def _on_silence_changed(self, val: int) -> None:
        self.lbl_silence_val.setText(f"{val} ms")
        self.controller.config.voice.end_of_speech_silence_ms = val
        if hasattr(self.controller, "voice_engine") and self.controller.voice_engine:
            self.controller.voice_engine.vad.set_silence_timeout(val)

    def _update_user_transcript(self, text: str, lang: str) -> None:
        lang_badge = f"<span style='color:#58a6ff;'>[{lang.upper()}]</span>"
        self.lbl_user_transcript.setText(f"<b>You:</b> {lang_badge} {text}")

    def _update_spoken_response(self, text: str) -> None:
        self.lbl_spoken_response.setText(f"<b>HAKS:</b> 🔊 {text}")

    def _update_live_status(self, status: str) -> None:
        color = "#15803d" if status in ("CONNECTED", "STREAMING") else ("#d97706" if status == "CONNECTING" else "#6b7280")
        self.lbl_live_status.setText(f"Gemini Live Status: {status}")
        self.lbl_live_status.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600;")

    def _update_live_latency(self, latency_ms: float) -> None:
        self.lbl_live_latency.setText(f"Latency: {latency_ms:.0f} ms")

    def _update_ui_state(self, state: str) -> None:
        self.lbl_state_display.setText(state)
        state_colors = {
            "STANDBY": ("#15803d", "Waiting for wake phrase 'Hey HAKS' or push-to-talk."),
            "WAKE_DETECTED": ("#0067c0", "Wake phrase detected! Initializing voice channel..."),
            "LISTENING": ("#b91c1c", "Microphone actively listening for speech..."),
            "PROCESSING": ("#b45309", "Voice Activity Detection segmenting audio buffer..."),
            "THINKING": ("#6d28d9", "AI Router processing speech transcript..."),
            "SPEAKING": ("#15803d", "Synthesizing Tamil/English audio response..."),
            "INTERRUPTED": ("#dc2626", "Speech interrupted by user barge-in!"),
            "SLEEP": ("#6b7280", "Voice engine sleeping. Say 'Wake up HAKS' or press Wake Up."),
            "STOPPED": ("#6b7280", "Voice pipeline stopped. Click 'Start Voice Assistant' above."),
        }
        color, desc = state_colors.get(state, ("#15803d", "Active"))
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
