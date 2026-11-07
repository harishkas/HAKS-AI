"""UI Stylesheet and Design System for HAKS Desktop Application.

Provides a modern dark theme with clean typography, distinct elevation layers,
and high contrast readability.
"""

DARK_THEME_QSS = """
/* Global Window and Base Styles */
QMainWindow, QWidget#root_container {
    background-color: #0f1117;
    color: #e6edf3;
    font-family: 'Segoe UI', 'SF Pro Text', -apple-system, sans-serif;
    font-size: 13px;
}

/* Sidebar Navigation */
QWidget#sidebar {
    background-color: #161b22;
    border-right: 1px solid #30363d;
    min-width: 210px;
    max-width: 230px;
}

QLabel#brand_title {
    font-size: 18px;
    font-weight: 700;
    color: #58a6ff;
    padding: 16px 12px 4px 12px;
}

QLabel#brand_subtitle {
    font-size: 11px;
    color: #8b949e;
    padding: 0 12px 16px 12px;
}

/* Navigation Buttons */
QPushButton.nav_button {
    text-align: left;
    padding: 10px 14px;
    margin: 2px 8px;
    border: none;
    border-radius: 6px;
    color: #c9d1d9;
    background-color: transparent;
    font-size: 13px;
    font-weight: 500;
}

QPushButton.nav_button:hover {
    background-color: #21262d;
    color: #f0f6fc;
}

QPushButton.nav_button:checked, QPushButton.nav_button.active {
    background-color: #1f3a5f;
    color: #58a6ff;
    font-weight: 600;
}

/* Top Header Bar */
QWidget#top_bar {
    background-color: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 6px 16px;
    min-height: 44px;
}

QLabel#top_bar_title {
    font-size: 14px;
    font-weight: 600;
    color: #f0f6fc;
}

/* Status Badges */
QLabel.status_badge {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.badge_local {
    background-color: #1c3d2f;
    color: #3fb950;
    border: 1px solid #238636;
}

QLabel.badge_voice {
    background-color: #2d2254;
    color: #a371f7;
    border: 1px solid #8957e5;
}

QLabel.badge_offline {
    background-color: #382e18;
    color: #d29922;
    border: 1px solid #9e6a03;
}

/* Main Content Area */
QWidget#content_area {
    background-color: #0d1117;
}

/* Dashboard Cards */
QFrame.stat_card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px;
}

QFrame.stat_card:hover {
    border-color: #58a6ff;
}

QLabel.card_title {
    font-size: 12px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
}

QLabel.card_value {
    font-size: 20px;
    font-weight: 700;
    color: #f0f6fc;
    margin-top: 4px;
}

QLabel.card_detail {
    font-size: 11px;
    color: #7d8590;
    margin-top: 2px;
}

/* Primary Action Buttons */
QPushButton.btn_primary {
    background-color: #238636;
    color: #ffffff;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid rgba(240, 246, 252, 0.1);
}

QPushButton.btn_primary:hover {
    background-color: #2ea043;
}

QPushButton.btn_secondary {
    background-color: #21262d;
    color: #c9d1d9;
    font-weight: 500;
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid #30363d;
}

QPushButton.btn_secondary:hover {
    background-color: #30363d;
    color: #f0f6fc;
}

/* Status Bar */
QStatusBar {
    background-color: #161b22;
    color: #8b949e;
    border-top: 1px solid #21262d;
    font-size: 11px;
}

/* ScrollBars */
QScrollBar:vertical {
    background: #0d1117;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #30363d;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #484f58;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""
