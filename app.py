"""Main entrypoint for HAKS Personal AI Assistant.

Usage:
    python app.py                # Launch desktop application
    python app.py --version      # Print version and exit
    python app.py --smoke-test   # Run initialization test and exit
"""

import argparse
import sys

from config.settings import get_settings
from core.logger import get_logger, setup_logging
from storage.database import get_db_manager
from ui.app import HaksApplication


def main() -> int:
    """Parse arguments and start HAKS."""
    parser = argparse.ArgumentParser(
        description="HAKS — Personal AI Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version="HAKS v0.1.0 (Phase 1 Foundation)",
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
        logger.info("Running Phase 1 smoke test...")
        # 1. Test database initialization
        db = get_db_manager(config)
        db.initialize()
        stats = db.get_stats()
        logger.info("Database verification passed: %d databases active.", len(stats))
        for name, info in stats.items():
            logger.info("  - %s: %s (tables: %s)", name, info["file_path"], list(info["tables"].keys()))

        # 2. Test Qt application instantiation
        from PySide6.QtWidgets import QApplication
        qapp = QApplication.instance() or QApplication(sys.argv)
        from ui.main_window import MainWindow
        win = MainWindow(config)
        logger.info("MainWindow instantiated successfully: title='%s', size=%dx%d",
                    win.windowTitle(), win.width(), win.height())
        win.close()
        logger.info("Smoke test passed successfully!")
        return 0

    # Normal GUI execution
    app = HaksApplication(config)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
