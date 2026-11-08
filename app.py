"""Main entrypoint for HAKS Personal AI Assistant.

Usage:
    python app.py                # Launch desktop application
    python app.py --version      # Print version and exit
    python app.py --smoke-test   # Run initialization test and exit
"""

import argparse
import os
import sys

# High DPI and Qt Platform logging rules
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window.warning=false")

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

        # 2. Qt application instance MUST exist before services with QTimers/QObjects initialize
        from PySide6.QtWidgets import QApplication
        qapp = QApplication.instance() or QApplication(sys.argv)

        # 3. Test shared AppController
        controller = get_app_controller(config)
        conv_id = controller.get_active_conversation_id()
        logger.info("Shared AppController initialized with active conversation: %s", conv_id)

        from ui.main_window import MainWindow
        from ui.floating_assistant import FloatingAssistant

        win = MainWindow(config, controller)
        floating = FloatingAssistant(config, controller)

        logger.info("MainWindow instantiated: %dx%d", win.width(), win.height())
        logger.info("FloatingAssistant instantiated: %dx%d at pos (%d, %d)",
                    floating.width(), floating.height(), floating.x(), floating.y())

        # Test message dispatching through shared controller without polluting persistent session
        orig_conv = controller.get_active_conversation_id()
        temp_conv_id = controller.create_new_conversation(title="[Smoke Test Session]")
        msg = controller.send_message("Phase 2 UI smoke test message", role="user")
        logger.info("Message dispatched via shared controller: %s", msg["id"])
        controller.delete_conversation(temp_conv_id)
        if orig_conv and orig_conv != temp_conv_id:
            controller.select_conversation(orig_conv)

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

        # 8. Test Phase 8 Offline Operation & Disaster Recovery Subsystems
        rm = controller.get_recovery_manager()
        assert rm is not None
        assert isinstance(rm.is_online, bool)
        # Verify SQLite integrity check
        db_integrity = rm.run_integrity_check()
        assert len(db_integrity) == 5
        assert all(st.is_healthy for st in db_integrity.values())
        # Verify full backup creation and verification
        bk_path = rm.create_full_backup(encrypt=True, notes="Smoke test backup")
        assert bk_path.exists()
        v_res = rm.backup.verify_backup(bk_path)
        assert v_res.is_valid is True
        assert v_res.verified_files_count >= 5
        # Verify Offline AI & Voice indicators
        voice_offline = rm.get_voice_offline_status()
        assert voice_offline["wake_word_offline"] is True
        assert voice_offline["tts_edge_neural_online_only"] is True
        assert voice_offline["tts_sapi_offline"] is True
        assert hasattr(win, "page_recovery")
        logger.info("Phase 8 Offline Operation & Disaster Recovery verified: integrity, encrypted backups, SHA-256 verification, and offline voice readiness.")

        controller.shutdown()
        win.close()
        floating.close()
        win.deleteLater()
        floating.deleteLater()
        qapp.processEvents()
        logger.info("HAKS Release 0.1 Smoke test passed successfully across all phases!")
        return 0

    # Normal execution
    app = HaksApplication(config)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
