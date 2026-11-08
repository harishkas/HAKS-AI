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

        win.close()
        floating.close()
        logger.info("Phase 3 Smoke test passed successfully!")
        return 0

    # Normal execution
    app = HaksApplication(config)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
