"""UI Stylesheets and Design System for HAKS Desktop Application.

Provides modern Dark and Light themes with clean typography, high contrast,
and specialized styling for both the main desktop app and floating assistant.
"""

DARK_THEME_QSS = """
/* Global Window and Base Styles */
QMainWindow, QWidget#root_container, QWidget#content_area {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Segoe UI', -apple-system, sans-serif;
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
    font-size: 20px;
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

QPushButton.nav_button:checked {
    background-color: #1f3a5f;
    color: #58a6ff;
    font-weight: 600;
}

/* Top Header Bar */
QWidget#top_bar {
    background-color: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 6px 16px;
    min-height: 46px;
}

QLabel#top_bar_title {
    font-size: 15px;
    font-weight: 600;
    color: #f0f6fc;
}

/* Status Badges */
QLabel.status_badge {
    padding: 4px 10px;
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

QLabel.badge_voice_active {
    background-color: #591b1b;
    color: #ff7b72;
    border: 1px solid #da3633;
}

QLabel.badge_offline {
    background-color: #382e18;
    color: #d29922;
    border: 1px solid #9e6a03;
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
    font-size: 11px;
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
    font-size: 12px;
    color: #7d8590;
    margin-top: 2px;
}

/* Buttons */
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

QPushButton.btn_danger {
    background-color: #21262d;
    color: #f85149;
    font-weight: 500;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid #da3633;
}

QPushButton.btn_danger:hover {
    background-color: #b62324;
    color: #ffffff;
}

/* Chat Message Bubbles */
QFrame.msg_bubble_user {
    background-color: #1f3a5f;
    border: 1px solid #388bfd;
    border-radius: 12px;
    border-bottom-right-radius: 2px;
    padding: 10px 14px;
}

QFrame.msg_bubble_assistant {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    border-bottom-left-radius: 2px;
    padding: 10px 14px;
}

QLabel.msg_role_label {
    font-size: 11px;
    font-weight: 700;
    color: #58a6ff;
    margin-bottom: 2px;
}

QLabel.msg_role_assistant {
    color: #3fb950;
}

QLabel.msg_content_label {
    font-size: 13px;
    color: #e6edf3;
    line-height: 1.4;
}

QLabel.msg_time_label {
    font-size: 10px;
    color: #7d8590;
    margin-top: 4px;
}

/* Input Fields & Text Edits */
QLineEdit, QTextEdit {
    background-color: #0d1117;
    color: #f0f6fc;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 8px 12px;
    selection-background-color: #1f6feb;
}

QLineEdit:focus, QTextEdit:focus {
    border-color: #58a6ff;
}

/* Status Bar */
QStatusBar {
    background-color: #161b22;
    color: #8b949e;
    border-top: 1px solid #21262d;
    font-size: 11px;
}

/* Floating Assistant Specific Styles */
QWidget#floating_root {
    background-color: #161b22;
    border: 1px solid #388bfd;
    border-radius: 12px;
}

QWidget#floating_drag_bar {
    background-color: #21262d;
    border-top-left-radius: 11px;
    border-top-right-radius: 11px;
    padding: 6px 10px;
}

QPushButton.floating_btn_icon {
    background: transparent;
    border: none;
    color: #8b949e;
    font-size: 12px;
    padding: 2px 6px;
    border-radius: 4px;
}

QPushButton.floating_btn_icon:hover {
    background-color: #30363d;
    color: #f0f6fc;
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

LIGHT_THEME_QSS = """
/* Global Window and Base Styles */
QMainWindow, QWidget#root_container, QWidget#content_area {
    background-color: #f6f8fa;
    color: #1f2328;
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
}

/* Sidebar Navigation */
QWidget#sidebar {
    background-color: #ffffff;
    border-right: 1px solid #d0d7de;
    min-width: 210px;
    max-width: 230px;
}

QLabel#brand_title {
    font-size: 20px;
    font-weight: 700;
    color: #0969da;
    padding: 16px 12px 4px 12px;
}

QLabel#brand_subtitle {
    font-size: 11px;
    color: #656d76;
    padding: 0 12px 16px 12px;
}

/* Navigation Buttons */
QPushButton.nav_button {
    text-align: left;
    padding: 10px 14px;
    margin: 2px 8px;
    border: none;
    border-radius: 6px;
    color: #24292f;
    background-color: transparent;
    font-size: 13px;
    font-weight: 500;
}

QPushButton.nav_button:hover {
    background-color: #eaeef2;
    color: #1f2328;
}

QPushButton.nav_button:checked {
    background-color: #ddf4ff;
    color: #0969da;
    font-weight: 600;
}

/* Top Header Bar */
QWidget#top_bar {
    background-color: #ffffff;
    border-bottom: 1px solid #d0d7de;
    padding: 6px 16px;
    min-height: 46px;
}

QLabel#top_bar_title {
    font-size: 15px;
    font-weight: 600;
    color: #1f2328;
}

/* Status Badges */
QLabel.status_badge {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.badge_local {
    background-color: #dafbe1;
    color: #1a7f37;
    border: 1px solid #4ac26b;
}

QLabel.badge_voice {
    background-color: #fbefff;
    color: #8250df;
    border: 1px solid #bf8700;
}

QLabel.badge_voice_active {
    background-color: #ffebe9;
    color: #cf222e;
    border: 1px solid #ff8182;
}

QLabel.badge_offline {
    background-color: #fff8c5;
    color: #9a6700;
    border: 1px solid #d4a72c;
}

/* Dashboard Cards */
QFrame.stat_card {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    padding: 16px;
}

QFrame.stat_card:hover {
    border-color: #0969da;
}

QLabel.card_title {
    font-size: 11px;
    font-weight: 600;
    color: #656d76;
    text-transform: uppercase;
}

QLabel.card_value {
    font-size: 20px;
    font-weight: 700;
    color: #1f2328;
    margin-top: 4px;
}

QLabel.card_detail {
    font-size: 12px;
    color: #57606a;
    margin-top: 2px;
}

/* Buttons */
QPushButton.btn_primary {
    background-color: #1f883d;
    color: #ffffff;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid rgba(31, 35, 40, 0.15);
}

QPushButton.btn_primary:hover {
    background-color: #1a7f37;
}

QPushButton.btn_secondary {
    background-color: #f6f8fa;
    color: #24292f;
    font-weight: 500;
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid #d0d7de;
}

QPushButton.btn_secondary:hover {
    background-color: #eaeef2;
    color: #1f2328;
}

QPushButton.btn_danger {
    background-color: #f6f8fa;
    color: #cf222e;
    font-weight: 500;
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid #d0d7de;
}

QPushButton.btn_danger:hover {
    background-color: #cf222e;
    color: #ffffff;
}

/* Chat Message Bubbles */
QFrame.msg_bubble_user {
    background-color: #ddf4ff;
    border: 1px solid #54aeff;
    border-radius: 12px;
    border-bottom-right-radius: 2px;
    padding: 10px 14px;
}

QFrame.msg_bubble_assistant {
    background-color: #ffffff;
    border: 1px solid #d0d7de;
    border-radius: 12px;
    border-bottom-left-radius: 2px;
    padding: 10px 14px;
}

QLabel.msg_role_label {
    font-size: 11px;
    font-weight: 700;
    color: #0969da;
    margin-bottom: 2px;
}

QLabel.msg_role_assistant {
    color: #1a7f37;
}

QLabel.msg_content_label {
    font-size: 13px;
    color: #1f2328;
    line-height: 1.4;
}

QLabel.msg_time_label {
    font-size: 10px;
    color: #656d76;
    margin-top: 4px;
}

/* Input Fields & Text Edits */
QLineEdit, QTextEdit {
    background-color: #ffffff;
    color: #1f2328;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    padding: 8px 12px;
    selection-background-color: #0969da;
}

QLineEdit:focus, QTextEdit:focus {
    border-color: #0969da;
}

/* Status Bar */
QStatusBar {
    background-color: #ffffff;
    color: #656d76;
    border-top: 1px solid #d0d7de;
    font-size: 11px;
}

/* Floating Assistant Specific Styles */
QWidget#floating_root {
    background-color: #ffffff;
    border: 1px solid #0969da;
    border-radius: 12px;
}

QWidget#floating_drag_bar {
    background-color: #f6f8fa;
    border-top-left-radius: 11px;
    border-top-right-radius: 11px;
    padding: 6px 10px;
}

QPushButton.floating_btn_icon {
    background: transparent;
    border: none;
    color: #656d76;
    font-size: 12px;
    padding: 2px 6px;
    border-radius: 4px;
}

QPushButton.floating_btn_icon:hover {
    background-color: #eaeef2;
    color: #1f2328;
}

/* ScrollBars */
QScrollBar:vertical {
    background: #f6f8fa;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #d0d7de;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #afb8c1;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


def get_stylesheet(theme_name: str) -> str:
    """Return the stylesheet string corresponding to theme_name."""
    if theme_name.lower() == "light":
        return LIGHT_THEME_QSS
    return DARK_THEME_QSS
