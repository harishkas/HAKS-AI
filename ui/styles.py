"""UI Stylesheets and Design System for HAKS Desktop Application.

Provides a clean, modern Windows Light (White mode) theme as the default,
with crisp typography (Segoe UI), high contrast, rounded card surfaces,
and specialized styling for both the main desktop app and floating assistant.
Also provides a polished Dark theme fallback.
"""

LIGHT_THEME_QSS = """
/* ==========================================================================
   HAKS WINDOWS LIGHT THEME (WHITE MODE)
   Clean, modern, accessible Windows desktop aesthetic
   ========================================================================== */

/* Global Window and Base Styles */
QMainWindow, QWidget#root_container, QWidget#content_area {
    background-color: #f8f9fa;
    color: #111827;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Segoe UI Variable', system-ui, sans-serif;
    font-size: 13px;
}

QWidget {
    color: #111827;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, 'Segoe UI Variable', system-ui, sans-serif;
}

/* Sidebar Navigation (Windows Modern Rail) */
QWidget#sidebar {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
    min-width: 220px;
    max-width: 240px;
}

QLabel#brand_title {
    font-size: 20px;
    font-weight: 700;
    color: #0067c0;
    padding: 16px 14px 2px 14px;
}

QLabel#brand_subtitle {
    font-size: 11px;
    font-weight: 500;
    color: #6b7280;
    padding: 0 14px 14px 14px;
}

/* Navigation Buttons */
QPushButton.nav_button {
    text-align: left;
    padding: 9px 12px;
    margin: 2px 8px;
    border: 1px solid transparent;
    border-radius: 6px;
    color: #374151;
    background-color: transparent;
    font-size: 13px;
    font-weight: 500;
}

QPushButton.nav_button:hover {
    background-color: #f3f4f6;
    color: #111827;
}

QPushButton.nav_button:checked {
    background-color: #e0edff;
    color: #0067c0;
    font-weight: 600;
    border: 1px solid #b9d5ff;
}

/* Top Header Bar */
QWidget#top_bar {
    background-color: #ffffff;
    border-bottom: 1px solid #e5e7eb;
    padding: 6px 16px;
    min-height: 48px;
}

QLabel#top_bar_title {
    font-size: 16px;
    font-weight: 700;
    color: #111827;
}

/* Status Badges */
QLabel.status_badge {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.badge_local, QLabel.badge_online {
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #86efac;
}

QLabel.badge_voice {
    background-color: #ede9fe;
    color: #6d28d9;
    border: 1px solid #c4b5fd;
}

QLabel.badge_voice_active {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fca5a5;
}

QLabel.badge_offline {
    background-color: #fef3c7;
    color: #b45309;
    border: 1px solid #fcd34d;
}

/* Dashboard / Surface Cards */
QFrame.stat_card {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
}

QFrame.stat_card:hover {
    border-color: #0067c0;
}

QLabel.card_title {
    font-size: 11px;
    font-weight: 700;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

QLabel.card_value {
    font-size: 18px;
    font-weight: 700;
    color: #111827;
    margin-top: 4px;
}

QLabel.card_detail {
    font-size: 12px;
    color: #4b5563;
    margin-top: 2px;
    line-height: 1.4;
}

/* Action Quick-Cards */
QFrame.quick_action_card {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
}

QFrame.quick_action_card:hover {
    border: 1px solid #0067c0;
    background-color: #f9fbfe;
}

/* Buttons */
QPushButton {
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
    border-radius: 6px;
    padding: 7px 14px;
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #d1d5db;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #f9fafb;
    border-color: #9ca3af;
}

QPushButton:pressed {
    background-color: #f3f4f6;
}

QPushButton.btn_primary {
    background-color: #0067c0;
    color: #ffffff;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid #005da6;
}

QPushButton.btn_primary:hover {
    background-color: #005da6;
    border-color: #004f8c;
}

QPushButton.btn_primary:pressed {
    background-color: #004f8c;
}

QPushButton.btn_secondary {
    background-color: #ffffff;
    color: #1f2937;
    font-weight: 500;
    padding: 7px 14px;
    border-radius: 6px;
    border: 1px solid #d1d5db;
}

QPushButton.btn_secondary:hover {
    background-color: #f3f4f6;
    border-color: #9ca3af;
    color: #111827;
}

QPushButton.btn_secondary:pressed {
    background-color: #e5e7eb;
}

QPushButton.btn_danger {
    background-color: #fef2f2;
    color: #dc2626;
    font-weight: 600;
    padding: 7px 14px;
    border-radius: 6px;
    border: 1px solid #fca5a5;
}

QPushButton.btn_danger:hover {
    background-color: #dc2626;
    color: #ffffff;
    border-color: #b91c1c;
}

QPushButton.btn_danger:pressed {
    background-color: #b91c1c;
    color: #ffffff;
}

/* Chat Message Bubbles */
QFrame.msg_bubble_user {
    background-color: #e0edff;
    border: 1px solid #b9d5ff;
    border-radius: 12px;
    border-bottom-right-radius: 2px;
    padding: 10px 14px;
}

QFrame.msg_bubble_assistant {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    border-bottom-left-radius: 2px;
    padding: 10px 14px;
}

QLabel.msg_role_label {
    font-size: 11px;
    font-weight: 700;
    color: #0067c0;
    margin-bottom: 2px;
}

QLabel.msg_role_assistant {
    color: #15803d;
}

QLabel.msg_content_label {
    font-size: 13px;
    color: #111827;
    line-height: 1.4;
}

QLabel.msg_time_label {
    font-size: 10px;
    color: #6b7280;
    margin-top: 4px;
}

/* Input Fields & Text Edits */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 8px 12px;
    selection-background-color: #0067c0;
    selection-color: #ffffff;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border: 1px solid #0067c0;
    background-color: #ffffff;
}

/* ComboBox */
QComboBox {
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 6px 10px;
    min-height: 22px;
}

QComboBox:hover {
    border-color: #9ca3af;
}

QComboBox:focus {
    border-color: #0067c0;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #e5e7eb;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    selection-background-color: #e0edff;
    selection-color: #0067c0;
    padding: 4px;
}

/* List Widgets (Conversations, Memory, etc.) */
QListWidget {
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 6px;
}

QListWidget::item {
    padding: 10px 12px;
    border-bottom: 1px solid #f3f4f6;
    border-radius: 6px;
    color: #111827;
    margin-bottom: 2px;
}

QListWidget::item:hover {
    background-color: #f3f4f6;
    color: #111827;
}

QListWidget::item:selected {
    background-color: #e0edff;
    color: #0067c0;
    font-weight: 600;
}

/* Tables */
QTableWidget {
    background-color: #ffffff;
    color: #111827;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    gridline-color: #f3f4f6;
}

QHeaderView::section {
    background-color: #f9fafb;
    color: #4b5563;
    font-weight: 600;
    font-size: 11px;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #e5e7eb;
}

/* Tabs */
QTabWidget::pane {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    background-color: #ffffff;
}

QTabBar::tab {
    background-color: #f3f4f6;
    color: #4b5563;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #0067c0;
    font-weight: 600;
    border-bottom: 2px solid #0067c0;
}

/* Status Bar */
QStatusBar {
    background-color: #ffffff;
    color: #4b5563;
    border-top: 1px solid #e5e7eb;
    font-size: 12px;
    padding: 4px 10px;
}

/* ScrollBars */
QScrollBar:vertical {
    background: #f9fafb;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #f9fafb;
    height: 8px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background: #cbd5e1;
    min-width: 24px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background: #94a3b8;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* CheckBox & Radio */
QCheckBox, QRadioButton {
    color: #111827;
    spacing: 8px;
    font-size: 13px;
}

/* ToolTips */
QToolTip {
    background-color: #1f2937;
    color: #ffffff;
    border: 1px solid #374151;
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 12px;
}

/* Floating Assistant Specific Styles */
QWidget#floating_root {
    background-color: #ffffff;
    border: 1px solid #0067c0;
    border-radius: 12px;
}

QWidget#floating_drag_bar {
    background-color: #f3f4f6;
    border-top-left-radius: 11px;
    border-top-right-radius: 11px;
    border-bottom: 1px solid #e5e7eb;
    padding: 6px 10px;
}

QLabel#floating_preview {
    background-color: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 8px;
    color: #111827;
    font-size: 12px;
}

QPushButton.floating_btn_icon {
    background: transparent;
    border: none;
    color: #4b5563;
    font-size: 12px;
    padding: 2px 6px;
    border-radius: 4px;
}

QPushButton.floating_btn_icon:hover {
    background-color: #e5e7eb;
    color: #111827;
}
"""

DARK_THEME_QSS = """
/* ==========================================================================
   HAKS DARK THEME (OPTIONAL HIGH-CONTRAST DARK)
   ========================================================================== */

/* Global Window and Base Styles */
QMainWindow, QWidget#root_container, QWidget#content_area {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
}

QWidget {
    color: #f8fafc;
    font-family: 'Segoe UI', -apple-system, sans-serif;
}

/* Sidebar Navigation */
QWidget#sidebar {
    background-color: #1e293b;
    border-right: 1px solid #334155;
    min-width: 220px;
    max-width: 240px;
}

QLabel#brand_title {
    font-size: 20px;
    font-weight: 700;
    color: #38bdf8;
    padding: 16px 14px 2px 14px;
}

QLabel#brand_subtitle {
    font-size: 11px;
    color: #94a3b8;
    padding: 0 14px 14px 14px;
}

/* Navigation Buttons */
QPushButton.nav_button {
    text-align: left;
    padding: 9px 12px;
    margin: 2px 8px;
    border: 1px solid transparent;
    border-radius: 6px;
    color: #cbd5e1;
    background-color: transparent;
    font-size: 13px;
    font-weight: 500;
}

QPushButton.nav_button:hover {
    background-color: #334155;
    color: #f8fafc;
}

QPushButton.nav_button:checked {
    background-color: #0369a1;
    color: #ffffff;
    font-weight: 600;
    border: 1px solid #38bdf8;
}

/* Top Header Bar */
QWidget#top_bar {
    background-color: #1e293b;
    border-bottom: 1px solid #334155;
    padding: 6px 16px;
    min-height: 48px;
}

QLabel#top_bar_title {
    font-size: 16px;
    font-weight: 700;
    color: #f8fafc;
}

/* Status Badges */
QLabel.status_badge {
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.badge_local, QLabel.badge_online {
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
}

QLabel.badge_voice {
    background-color: #4c1d95;
    color: #c4b5fd;
    border: 1px solid #7c3aed;
}

QLabel.badge_voice_active {
    background-color: #7f1d1d;
    color: #fca5a5;
    border: 1px solid #dc2626;
}

QLabel.badge_offline {
    background-color: #78350f;
    color: #fcd34d;
    border: 1px solid #d97706;
}

/* Dashboard Cards */
QFrame.stat_card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px;
}

QFrame.stat_card:hover {
    border-color: #38bdf8;
}

QLabel.card_title {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
}

QLabel.card_value {
    font-size: 18px;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 4px;
}

QLabel.card_detail {
    font-size: 12px;
    color: #cbd5e1;
    margin-top: 2px;
}

/* Buttons */
QPushButton {
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
    border-radius: 6px;
    padding: 7px 14px;
    background-color: #1e293b;
    color: #f8fafc;
    border: 1px solid #475569;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #334155;
    border-color: #64748b;
}

QPushButton.btn_primary {
    background-color: #0284c7;
    color: #ffffff;
    font-weight: 600;
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid #38bdf8;
}

QPushButton.btn_primary:hover {
    background-color: #0369a1;
}

QPushButton.btn_secondary {
    background-color: #1e293b;
    color: #e2e8f0;
    font-weight: 500;
    padding: 7px 14px;
    border-radius: 6px;
    border: 1px solid #475569;
}

QPushButton.btn_secondary:hover {
    background-color: #334155;
    color: #f8fafc;
}

QPushButton.btn_danger {
    background-color: #7f1d1d;
    color: #fecaca;
    font-weight: 600;
    padding: 7px 14px;
    border-radius: 6px;
    border: 1px solid #ef4444;
}

QPushButton.btn_danger:hover {
    background-color: #dc2626;
    color: #ffffff;
}

/* Chat Message Bubbles */
QFrame.msg_bubble_user {
    background-color: #0369a1;
    border: 1px solid #38bdf8;
    border-radius: 12px;
    border-bottom-right-radius: 2px;
    padding: 10px 14px;
}

QFrame.msg_bubble_assistant {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    border-bottom-left-radius: 2px;
    padding: 10px 14px;
}

QLabel.msg_role_label {
    font-size: 11px;
    font-weight: 700;
    color: #38bdf8;
    margin-bottom: 2px;
}

QLabel.msg_role_assistant {
    color: #34d399;
}

QLabel.msg_content_label {
    font-size: 13px;
    color: #f8fafc;
    line-height: 1.4;
}

QLabel.msg_time_label {
    font-size: 10px;
    color: #94a3b8;
    margin-top: 4px;
}

/* Input Fields & Text Edits */
QLineEdit, QTextEdit, QPlainTextEdit {
    background-color: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 12px;
    selection-background-color: #0284c7;
    selection-color: #ffffff;
}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #38bdf8;
}

/* ComboBox */
QComboBox {
    background-color: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    color: #f8fafc;
    border: 1px solid #334155;
    selection-background-color: #0369a1;
}

/* List Widgets */
QListWidget {
    background-color: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 6px;
}

QListWidget::item {
    padding: 10px 12px;
    border-bottom: 1px solid #1e293b;
    border-radius: 6px;
    color: #f8fafc;
}

QListWidget::item:hover {
    background-color: #1e293b;
}

QListWidget::item:selected {
    background-color: #0369a1;
    color: #ffffff;
}

/* Status Bar */
QStatusBar {
    background-color: #1e293b;
    color: #94a3b8;
    border-top: 1px solid #334155;
    font-size: 12px;
}

/* Floating Assistant */
QWidget#floating_root {
    background-color: #1e293b;
    border: 1px solid #38bdf8;
    border-radius: 12px;
}

QWidget#floating_drag_bar {
    background-color: #0f172a;
    border-top-left-radius: 11px;
    border-top-right-radius: 11px;
    padding: 6px 10px;
}

QLabel#floating_preview {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px;
    color: #f8fafc;
    font-size: 12px;
}

QPushButton.floating_btn_icon {
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 12px;
    padding: 2px 6px;
    border-radius: 4px;
}

QPushButton.floating_btn_icon:hover {
    background-color: #334155;
    color: #f8fafc;
}

/* ScrollBars */
QScrollBar:vertical {
    background: #0f172a;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


def get_stylesheet(theme_name: str) -> str:
    """Return the stylesheet string corresponding to theme_name (default: light / white mode)."""
    if theme_name.lower() == "dark":
        return DARK_THEME_QSS
    return LIGHT_THEME_QSS
