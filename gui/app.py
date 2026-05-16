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

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .utils.config import ensure_dirs


def main() -> None:
    """Launch the pyat GUI application."""
    ensure_dirs()

    app = QApplication(sys.argv)
    app.setApplicationName("pyat GUI")
    app.setOrganizationName("pyat")

    # Apply dark theme styling
    app.setStyle("Fusion")
    _apply_dark_palette(app)

    window = MainWindow()
    window.show()

    # Auto-load test data if available
    _auto_load_test_data(window)

    sys.exit(app.exec())


def _apply_dark_palette(app: QApplication) -> None:
    """Apply a dark color palette similar to VS Code dark theme."""
    from PySide6.QtGui import QPalette, QColor

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(37, 37, 38))
    palette.setColor(QPalette.WindowText, QColor(212, 212, 212))
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
    palette.setColor(QPalette.ToolTipBase, QColor(37, 37, 38))
    palette.setColor(QPalette.ToolTipText, QColor(212, 212, 212))
    palette.setColor(QPalette.Text, QColor(212, 212, 212))
    palette.setColor(QPalette.Button, QColor(45, 45, 45))
    palette.setColor(QPalette.ButtonText, QColor(212, 212, 212))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Link, QColor(86, 156, 214))
    palette.setColor(QPalette.Highlight, QColor(38, 79, 120))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))

    app.setPalette(palette)
    # Additional stylesheet fixes
    app.setStyleSheet("""
        QToolTip { color: #d4d4d4; background-color: #252526; border: 1px solid #555; }
        QMenu { background-color: #2d2d2d; color: #d4d4d4; border: 1px solid #555; }
        QMenu::item:selected { background-color: #094771; }
        QTreeWidget { background-color: #252526; alternate-background-color: #2d2d2d; }
        QTreeWidget::item:selected { background-color: #094771; }
        QDockWidget { titlebar-close-icon: none; }
        QTabWidget::pane { border: 1px solid #3c3c3c; }
        QGroupBox { color: #d4d4d4; border: 1px solid #555; margin-top: 0.5em; padding-top: 0.5em; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; }
    """)


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
                break  # Only load from first available dir


if __name__ == "__main__":
    main()
