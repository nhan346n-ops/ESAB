"""pyat GUI application entry point.

Usage:
    python gui/app.py

Or from the project root:
    python -m gui.app
"""
import sys
import os

# Setup DLL paths on Windows first
if sys.platform == 'win32':
    py_bin = os.path.dirname(sys.executable)
    env_root = os.path.dirname(py_bin) if os.path.basename(py_bin).lower() == "scripts" else py_bin
    conda_lib_bin = os.path.join(env_root, "Library", "bin")
    if os.path.isdir(conda_lib_bin):
        path_list = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        first_path = os.path.normpath(path_list[0]).lower() if path_list else ""
        conda_lib_bin_norm = os.path.normpath(conda_lib_bin).lower()

        if first_path != conda_lib_bin_norm and not os.environ.get("PYAT_ENV_RESTARTED"):
            import subprocess
            os.environ["PATH"] = conda_lib_bin + os.pathsep + os.environ.get("PATH", "")
            os.environ["PYAT_ENV_RESTARTED"] = "1"
            os.environ.pop("USE_PATH_FOR_GDAL_PYTHON", None)
            sys.exit(subprocess.call([sys.executable] + sys.argv, env=os.environ.copy()))

        os.environ.pop("USE_PATH_FOR_GDAL_PYTHON", None)
        if sys.version_info >= (3, 8):
            try:
                os.add_dll_directory(conda_lib_bin)
            except Exception:
                pass

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

BLUE_WHITE_QSS = """
    /* === 全局基础 === */
    QWidget {
        font-family: 'Segoe UI', 'Microsoft YaHei UI', Arial, sans-serif;
        font-size: 12px;
        color: #1C2B3A;
        background-color: #F7F9FC;
    }
    QMainWindow { background-color: #EEF2F7; }

    /* === 工具栏 === */
    QToolBar {
        background: #FFFFFF;
        border-bottom: 1px solid #D0D9E4;
        spacing: 2px;
        padding: 2px 4px;
    }
    QToolBar::separator {
        width: 1px;
        background: #D0D9E4;
        margin: 4px 6px;
    }
    QToolButton {
        padding: 2px 4px;
        border-radius: 4px;
        color: #1C2B3A;
        border: 1px solid transparent;
    }
    QToolButton:hover {
        background-color: #E3EDF8;
        border: 1px solid #2E8FE8;
        color: #0D3A6E;
    }
    QToolButton:checked, QToolButton:pressed {
        background-color: #D0E6F8;
        border: 1px solid #1A6FBF;
        color: #0D3A6E;
    }

    /* === Dock 面板 === */
    QDockWidget {
        font-weight: bold;
        font-size: 12px;
        color: #0D3A6E;
        border: none;
    }
    QDockWidget::title {
        background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
            stop:0 #E3EDF8, stop:1 #D8E6F3);
        padding: 5px 8px;
        font-weight: bold;
        color: #0D3A6E;
        text-align: left;
        border-bottom: 2px solid #1A6FBF;
    }
    QDockWidget::close-button, QDockWidget::float-button {
        border: none;
        padding: 2px;
        background: transparent;
    }
    QDockWidget::close-button:hover, QDockWidget::float-button:hover {
        background: #C5D9EF;
        border-radius: 2px;
    }

    /* === 标签页 === */
    QTabWidget::pane {
        border: 1px solid #D0D9E4;
        background: #FFFFFF;
        top: -1px;
    }
    QTabBar::tab {
        background: #EEF2F7;
        padding: 5px 14px;
        border: 1px solid #D0D9E4;
        border-bottom: none;
        color: #5C7A99;
        margin-right: 2px;
    }
    QTabBar::tab:selected {
        background: #FFFFFF;
        font-weight: bold;
        color: #1A6FBF;
        border-top: 3px solid #1A6FBF;
    }
    QTabBar::tab:hover:!selected {
        background: #E3EDF8;
        color: #0D3A6E;
    }

    /* === 树形控件 === */
    QTreeWidget {
        background: #FFFFFF;
        border: 1px solid #D0D9E4;
        color: #1C2B3A;
        font-size: 12px;
        outline: none;
    }
    QTreeWidget::item { padding: 3px 4px; }
    QTreeWidget::item:hover { background: #E3EDF8; }
    QTreeWidget::item:selected {
        background: #C5D9EF;
        color: #0D3A6E;
        border-left: 3px solid #1A6FBF;
    }
    QTreeWidget::branch:has-children:closed {
        image: none;
    }

    /* === 表格 === */
    QTableWidget, QTableView {
        font-size: 12px;
        background: #FFFFFF;
        alternate-background-color: #F4F8FD;
        color: #1C2B3A;
        gridline-color: #E8EEF5;
        border: 1px solid #D0D9E4;
        selection-background-color: #C5D9EF;
        selection-color: #0D3A6E;
    }
    QTableWidget::item, QTableView::item {
        color: #1C2B3A;
        padding: 3px 4px;
    }
    QTableWidget::item:alternate, QTableView::item:alternate {
        background: #F4F8FD;
        color: #1C2B3A;
    }
    QTableWidget::item:selected, QTableView::item:selected {
        background: #C5D9EF;
        color: #0D3A6E;
    }
    QHeaderView::section {
        background: #EEF2F7;
        padding: 5px 8px;
        border: none;
        border-right: 1px solid #D0D9E4;
        border-bottom: 2px solid #1A6FBF;
        font-weight: bold;
        color: #0D3A6E;
    }

    /* === 按钮 === */
    QPushButton {
        background-color: #FFFFFF;
        border: 1px solid #C2D0E0;
        padding: 5px 14px;
        border-radius: 4px;
        color: #1C2B3A;
        font-weight: 500;
    }
    QPushButton:hover {
        background-color: #E3EDF8;
        border: 1px solid #1A6FBF;
        color: #0D3A6E;
    }
    QPushButton:pressed {
        background-color: #C5D9EF;
        border: 1px solid #1A6FBF;
    }
    QPushButton:disabled {
        background-color: #F0F3F7;
        color: #A0B4C8;
        border-color: #D8E2EC;
    }

    /* === 输入控件 === */
    QLineEdit {
        padding: 4px 6px;
        border: 1px solid #C2D0E0;
        border-radius: 3px;
        background: #FFFFFF;
        color: #1C2B3A;
        selection-background-color: #C5D9EF;
    }
    QLineEdit:hover {
        border: 1px solid #1A6FBF;
    }
    QLineEdit:focus {
        border: 1.5px solid #1A6FBF;
        outline: none;
    }

    /* === 滚动条 === */
    QScrollBar:vertical {
        background: #F0F4F8;
        width: 8px;
        margin: 0;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #B0C4D8;
        border-radius: 4px;
        min-height: 20px;
    }
    QScrollBar::handle:vertical:hover { background: #1A6FBF; }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    QScrollBar:horizontal {
        background: #F0F4F8;
        height: 8px;
        border-radius: 4px;
    }
    QScrollBar::handle:horizontal {
        background: #B0C4D8;
        border-radius: 4px;
        min-width: 20px;
    }
    QScrollBar::handle:horizontal:hover { background: #1A6FBF; }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

    /* === 状态栏 === */
    QStatusBar {
        background: #FFFFFF;
        border-top: 1px solid #D0D9E4;
        color: #5C7A99;
        font-size: 11px;
        padding: 2px 8px;
    }
    QStatusBar::item { border: none; }

    /* === 分隔器 === */
    QSplitter::handle {
        background: #D0D9E4;
    }
    QSplitter::handle:horizontal { width: 2px; }
    QSplitter::handle:vertical { height: 2px; }
    QSplitter::handle:hover { background: #1A6FBF; }

    /* === 菜单 === */
    QMenu {
        background: #FFFFFF;
        border: 1px solid #D0D9E4;
        border-radius: 4px;
        padding: 4px 0;
    }
    QMenu::item {
        padding: 6px 28px 6px 16px;
        color: #1C2B3A;
    }
    QMenu::item:selected {
        background: #E3EDF8;
        color: #0D3A6E;
    }
    QMenu::separator {
        height: 1px;
        background: #E0E8F0;
        margin: 3px 8px;
    }

    /* === 对话框 === */
    QDialog, QMessageBox { background-color: #F7F9FC; }

    /* === 分组框 === */
    QGroupBox {
        border: 1px solid #D0D9E4;
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 4px;
        color: #0D3A6E;
        font-weight: bold;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 8px;
        padding: 0 4px;
    }

    /* === 复选框 === */
    QCheckBox { color: #1C2B3A; spacing: 5px; }
    QCheckBox::indicator {
        width: 14px; height: 14px;
        border: 1px solid #C2D0E0;
        border-radius: 2px;
        background: #FFFFFF;
    }
    QCheckBox::indicator:checked {
        background: #1A6FBF;
        border-color: #1A6FBF;
    }
    QCheckBox::indicator:hover { border-color: #1A6FBF; }

    /* === 标签 === */
    QLabel { color: #1C2B3A; background: transparent; }

    /* === 独立特定控件补充（从原主题迁移的修复项） === */
    QDoubleSpinBox { background-color: #ffffff; color: #1C2B3A; border: 1px solid #C2D0E0; border-radius: 3px; padding: 3px; }
    #coordLabel { background: #ffffff; color: #1C2B3A; padding: 0px 8px; font-family: Consolas; font-size: 10px; border-top: 1px solid #D0D9E4; }
    #logView { background-color: #ffffff; color: #1C2B3A; }
"""

def _blue_white_palette() -> QPalette:
    """蓝白科研风调色板 (与 QSS 配合)"""
    p = QPalette()
    p.setColor(QPalette.Window, QColor(247, 249, 252))
    p.setColor(QPalette.WindowText, QColor(28, 43, 58))
    p.setColor(QPalette.Base, QColor(255, 255, 255))
    p.setColor(QPalette.AlternateBase, QColor(244, 248, 253))
    p.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    p.setColor(QPalette.ToolTipText, QColor(28, 43, 58))
    p.setColor(QPalette.Text, QColor(28, 43, 58))
    p.setColor(QPalette.Button, QColor(255, 255, 255))
    p.setColor(QPalette.ButtonText, QColor(28, 43, 58))
    p.setColor(QPalette.Highlight, QColor(197, 217, 239))
    p.setColor(QPalette.HighlightedText, QColor(13, 58, 110))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(160, 160, 160))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(160, 160, 160))
    return p

def apply_theme(app: QApplication) -> None:
    """Apply the BLUE_WHITE_QSS theme."""
    app.setPalette(_blue_white_palette())
    app.setStyleSheet(BLUE_WHITE_QSS)

# ── Entry point ─────────────────────────────────────────────────────

def main() -> None:
    """Launch the pyat GUI application."""
    ensure_dirs()

    app = QApplication(sys.argv)
    app.setApplicationName("pyat GUI")
    app.setOrganizationName("pyat")

    app.setStyle("Fusion")
    apply_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
