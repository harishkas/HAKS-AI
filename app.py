"""Main entrypoint for HAKS Personal AI Assistant.

Usage:
    python app.py                # Launch desktop application
    python app.py --version      # Print version and exit
    python app.py --smoke-test   # Run initialization test and exit
"""

import argparse
import sys

from config.settings import get_settings
from core.app_controller import get_app_controller
from core.logger import get_logger, setup_logging
from storage.database import get_db_manager
from ui.app import HaksApplication


def main() -> int:
    """Parse arguments and start HAKS."""
    parser = argparse.ArgumentParser(
        description="HAKS - Personal AI Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="HAKS v0.1.0 (Phase 3 AI Engine Integration)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Perform non-interactive initialization smoke test and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug level logging",
    )

    args = parser.parse_args()

    # Load configuration
    config = get_settings()
    if args.debug:
        config.logging.level = "DEBUG"

    # Setup core logging
    setup_logging(config)
    logger = get_logger("app")
    logger.info("Initializing HAKS Personal AI Assistant...")

    if args.smoke_test:
        logger.info("Running Phase 2 UI smoke test...")
        # 1. Test database initialization
        db = get_db_manager(config)
        db.initialize()
        stats = db.get_stats()
        logger.info("Database verification passed: %d databases active.", len(stats))

        # 2. Test shared AppController
        controller = get_app_controller(config)
        conv_id = controller.get_active_conversation_id()
        logger.info("Shared AppController initialized with active conversation: %s", conv_id)

        # 3. Test Qt application and UI windows instantiation
        from PySide6.QtWidgets import QApplication
        qapp = QApplication.instance() or QApplication(sys.argv)

        from ui.main_window import MainWindow
        from ui.floating_assistant import FloatingAssistant

        win = MainWindow(config, controller)
        floating = FloatingAssistant(config, controller)

        logger.info("MainWindow instantiated: %dx%d", win.width(), win.height())
        logger.info("FloatingAssistant instantiated: %dx%d at pos (%d, %d)",
                    floating.width(), floating.height(), floating.x(), floating.y())

        # Test message dispatching through shared controller
        msg = controller.send_message("Phase 2 UI smoke test message", role="user")
        logger.info("Message dispatched via shared controller: %s", msg["id"])

        # 4. Test AI Engine Manager initialization
        ai_mgr = controller.get_ai_manager()
        logger.info("AI Engine Manager initialized: active engine = %s", ai_mgr.active_engine_type.value)
        local_caps = ai_mgr.get_capabilities("local")
        cloud_caps = ai_mgr.get_capabilities("cloud")
        logger.info("Local engine caps: streaming=%s, is_local=%s", local_caps.streaming, local_caps.is_local)
        logger.info("Cloud engine caps: streaming=%s, is_local=%s", cloud_caps.streaming, cloud_caps.is_local)

        # 5. Test Phase 5 Privacy & Intelligent Routing Subsystems
        router = controller.get_router()
        sens_dec = router.route("API key sk-1234567890abcdef1234567890abcdef test")
        assert sens_dec.selected_engine == "local"
        logger.info("Phase 5 Privacy Guard verified: sensitive prompt forced to Local AI.")

        redacted = controller.redactor.redact("Contact me at user@example.com with key sk-1234567890abcdef1234567890abcdef")
        assert redacted.has_redactions is True
        logger.info("Phase 5 Data Redactor verified: masked %d sensitive items.", redacted.redactions_count)

        # 6. Test Phase 6 Voice Assistant Subsystems
        from voice.models import VoiceCommandType
        assert controller.get_voice_state() == "STOPPED"
        assert controller.voice_engine._parse_voice_command("Stop HAKS") == VoiceCommandType.STOP
        assert controller.voice_engine._parse_voice_command("நிறுத்து") == VoiceCommandType.STOP
        assert controller.voice_engine.tts.detect_language("வணக்கம் உலகம்") == "ta"
        assert controller.voice_engine.tts.detect_language("Hello HAKS") == "en"
        logger.info("Phase 6 Voice Assistant verified: state machine, bilingual commands, and TTS routing active.")

        # 7. Test Phase 7 Windows Computer Control Subsystems
        ctrl = controller.get_windows_control()
        assert ctrl is not None
        plan_en = ctrl.interpreter.interpret("open notepad")
        assert plan_en.operation_type == "launch_app"
        plan_ta = ctrl.interpreter.interpret("நோட்பேட் திற")
        assert plan_ta.operation_type == "launch_app"
        drives = ctrl.list_system_drives()
        assert len(drives) >= 1
        hw = ctrl.get_complete_hardware_profile()
        assert hw.cpu.physical_cores >= 1
        stopped = controller.trigger_emergency_stop()
        assert isinstance(stopped, int)
        assert hasattr(win, "page_windows_control")
        logger.info("Phase 7 Windows Computer Control verified: interpreter (English+Tamil), policy, hardware profiler, and UI.")

        controller.voice_engine.shutdown()
        win.close()
        floating.close()
        logger.info("HAKS Release 0.1 Smoke test passed successfully across all phases!")
        return 0

    # Normal execution
    app = HaksApplication(config)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
