"""pyat GUI application entry point.

Usage:
    python gui/app.py

Or from the project root:
    python -m gui.app
"""
import sys
import os

# Ensure the pyat backend is importable
SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from PySide6.QtCore import QSettings
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .utils.config import ensure_dirs


# ── Theme definitions ───────────────────────────────────────────────

DARK_STYLESHEET = """
    QToolTip { color: #d4d4d4; background-color: #252526; border: 1px solid #555; }
    QMenu { background-color: #2d2d2d; color: #d4d4d4; border: 1px solid #555; }
    QMenu::item:selected { background-color: #094771; }
    QTreeWidget { background-color: #252526; alternate-background-color: #2d2d2d; }
    QTreeWidget::item:selected { background-color: #094771; }
    QDockWidget { titlebar-close-icon: none; }
    QTabWidget::pane { border: 1px solid #3c3c3c; }
    QGroupBox { color: #d4d4d4; border: 1px solid #555; margin-top: 0.5em; padding-top: 0.5em; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; }
    QDoubleSpinBox { background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #555; }
    #coordLabel { background: #2d2d2d; color: #d4d4d4; padding: 0px 8px; font-family: Consolas; font-size: 10px; border-top: 1px solid #444; }
    #logView { background-color: #1e1e1e; color: #d4d4d4; }
"""

LIGHT_STYLESHEET = """
    QToolTip { color: #333; background-color: #fff; border: 1px solid #ccc; }
    QMenu { background-color: #fff; color: #333; border: 1px solid #ccc; }
    QMenu::item:selected { background-color: #0078d4; color: #fff; }
    QDockWidget { titlebar-close-icon: none; }
    QTabWidget::pane { border: 1px solid #ccc; }
    QGroupBox { color: #333; border: 1px solid #ccc; margin-top: 0.5em; padding-top: 0.5em; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; }
    QDoubleSpinBox { background-color: #ffffff; color: #333333; border: 1px solid #ccc; }
    #coordLabel { background: #ffffff; color: #333333; padding: 0px 8px; font-family: Consolas; font-size: 10px; border-top: 1px solid #ccc; }
    #logView { background-color: #ffffff; color: #333333; }
"""


def _dark_palette() -> QPalette:
    """VS Code-style dark palette."""
    p = QPalette()
    p.setColor(QPalette.Window, QColor(37, 37, 38))
    p.setColor(QPalette.WindowText, QColor(212, 212, 212))
    p.setColor(QPalette.Base, QColor(30, 30, 30))
    p.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    p.setColor(QPalette.ToolTipBase, QColor(37, 37, 38))
    p.setColor(QPalette.ToolTipText, QColor(212, 212, 212))
    p.setColor(QPalette.Text, QColor(212, 212, 212))
    p.setColor(QPalette.Button, QColor(45, 45, 45))
    p.setColor(QPalette.ButtonText, QColor(212, 212, 212))
    p.setColor(QPalette.BrightText, QColor(255, 0, 0))
    p.setColor(QPalette.Link, QColor(86, 156, 214))
    p.setColor(QPalette.Highlight, QColor(38, 79, 120))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
    return p


def _light_palette() -> QPalette:
    """Clean light palette."""
    p = QPalette()
    p.setColor(QPalette.Window, QColor(240, 240, 240))
    p.setColor(QPalette.WindowText, QColor(30, 30, 30))
    p.setColor(QPalette.Base, QColor(255, 255, 255))
    p.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    p.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    p.setColor(QPalette.ToolTipText, QColor(30, 30, 30))
    p.setColor(QPalette.Text, QColor(30, 30, 30))
    p.setColor(QPalette.Button, QColor(240, 240, 240))
    p.setColor(QPalette.ButtonText, QColor(30, 30, 30))
    p.setColor(QPalette.BrightText, QColor(200, 0, 0))
    p.setColor(QPalette.Link, QColor(0, 100, 200))
    p.setColor(QPalette.Highlight, QColor(0, 120, 212))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(160, 160, 160))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(160, 160, 160))
    return p


_THEMES = {
    "dark": (_dark_palette, DARK_STYLESHEET),
    "light": (_light_palette, LIGHT_STYLESHEET),
}


def apply_theme(app: QApplication, theme_name: str = "dark") -> None:
    """Apply a named theme to the application."""
    palette_fn, stylesheet = _THEMES.get(theme_name, _THEMES["dark"])
    app.setPalette(palette_fn())
    app.setStyleSheet(stylesheet)
    settings = QSettings("pyat", "gui")
    settings.setValue("theme", theme_name)


def get_theme_names() -> list:
    return list(_THEMES.keys())


# ── Entry point ─────────────────────────────────────────────────────

def main() -> None:
    """Launch the pyat GUI application."""
    ensure_dirs()

    app = QApplication(sys.argv)
    app.setApplicationName("pyat GUI")
    app.setOrganizationName("pyat")

    app.setStyle("Fusion")
    settings = QSettings("pyat", "gui")
    saved_theme = settings.value("theme", "dark")
    apply_theme(app, saved_theme if saved_theme in _THEMES else "dark")

    window = MainWindow()
    window.show()

    _auto_load_test_data(window)

    sys.exit(app.exec())


def _auto_load_test_data(window: MainWindow) -> None:
    """Auto-load XSF files from default test directories."""
    test_dirs = [r"D:\WorkSpace3\Cdata", r"D:\workspace11\test"]
    for test_dir in test_dirs:
        if os.path.isdir(test_dir):
            from .core.xsf_reader import scan_directory
            results = scan_directory(test_dir)
            if results:
                window._project_explorer._add_to_tree(results)
                window._project_explorer.files_added.emit(results)
                break


if __name__ == "__main__":
    main()
