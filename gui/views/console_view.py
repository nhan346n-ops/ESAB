"""Console & Job Manager bottom panel.

Displays process stdout/stderr in real-time, job list with progress bars,
and supports log filtering by task.
"""
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QColor, QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSplitter,
    QTextEdit, QTreeWidget, QTreeWidgetItem,
    QPushButton, QLabel, QToolBar,
)

from ..core.task_manager import TaskManager, Task, TaskStatus


class ConsoleView(QWidget):
    """Combined console log + job manager panel."""

    cancel_requested = Signal()  # Emitted when user clicks Cancel

    def __init__(self, task_manager: TaskManager, parent=None):
        super().__init__(parent)
        self._task_manager = task_manager
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QToolBar()
        toolbar.addWidget(QLabel("\u63a7\u5236\u53f0\u4e0e\u4efb\u52a1\u7ba1\u7406"))
        self._clear_btn = QPushButton("\u6e05\u9664")
        self._clear_btn.clicked.connect(self._clear_log)
        toolbar.addWidget(self._clear_btn)
        cancel_btn = QPushButton("\u53d6\u6d88")
        cancel_btn.setStyleSheet("QPushButton { color: #f44747; }")
        cancel_btn.clicked.connect(self.cancel_requested.emit)
        toolbar.addWidget(cancel_btn)
        layout.addWidget(toolbar)

        # Splitter: log on left, job list on right
        splitter = QSplitter(Qt.Horizontal)

        # Log output
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Consolas", 9))
        self._log_view.setObjectName("logView")
        self._log_view.setStyleSheet("""
            QTextEdit {
                border: 1px solid #3c3c3c;
            }
        """)
        splitter.addWidget(self._log_view)

        # Job list
        job_widget = QWidget()
        job_layout = QVBoxLayout(job_widget)
        job_layout.setContentsMargins(4, 4, 4, 4)

        job_layout.addWidget(QLabel("Jobs"))

        self._job_tree = QTreeWidget()
        self._job_tree.setHeaderLabels(["Task", "Status", "Progress"])
        self._job_tree.setColumnWidth(0, 150)
        self._job_tree.setColumnWidth(1, 100)
        self._job_tree.setRootIsDecorated(False)
        self._job_tree.setAlternatingRowColors(True)
        job_layout.addWidget(self._job_tree)

        splitter.addWidget(job_widget)
        splitter.setSizes([500, 250])
        layout.addWidget(splitter)

    def _connect_signals(self) -> None:
        tm = self._task_manager
        tm.task_added.connect(self._on_task_added)
        tm.task_status_changed.connect(self._on_task_status_changed)
        tm.task_progress.connect(self._on_task_progress)
        tm.console_log.connect(self._on_log)

    def _on_task_added(self, task: Task) -> None:
        item = QTreeWidgetItem(self._job_tree)
        item.setData(0, Qt.UserRole, task.task_id)
        item.setText(0, task.tool_name)
        item.setText(1, task.status.value)
        self._add_log("INFO", f"Task [{task.task_id}] {task.tool_name} — QUEUED")

    def _on_task_status_changed(self, task: Task) -> None:
        for i in range(self._job_tree.topLevelItemCount()):
            item = self._job_tree.topLevelItem(i)
            if item.data(0, Qt.UserRole) == task.task_id:
                item.setText(1, task.status.value)
                if task.status == TaskStatus.COMPLETED:
                    item.setForeground(1, QColor("#4ec94e"))
                    self._add_log("OK", f"Task [{task.task_id}] COMPLETED")
                elif task.status == TaskStatus.FAILED:
                    item.setForeground(1, QColor("#f44747"))
                    self._add_log("ERROR", f"Task [{task.task_id}] FAILED: {task.error_message}")
                elif task.status == TaskStatus.RUNNING:
                    item.setForeground(1, QColor("#569cd6"))
                break

    def _on_task_progress(self, task_id: str, percent: int, message: str) -> None:
        self._add_log("INFO", f"[{task_id}] {percent}% — {message}")

    def _on_log(self, task_id: str, level: str, message: str) -> None:
        self._add_log(level, f"[{task_id}] {message}")

    def _add_log(self, level: str, message: str) -> None:
        color_map = {
            "ERROR": "#f44747",
            "WARNING": "#cca700",
            "OK": "#4ec94e",
            "INFO": "#d4d4d4",
            "DEBUG": "#808080",
        }
        color = color_map.get(level, "#d4d4d4")
        formatted = f'<span style="color:{color}">[{level}] {message}</span>'
        self._log_view.append(formatted)
        # Auto-scroll to bottom
        cursor = self._log_view.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._log_view.setTextCursor(cursor)

    def _clear_log(self) -> None:
        self._log_view.clear()

    def append_text(self, text: str, level: str = "INFO") -> None:
        """Public method to add text directly."""
        self._add_log(level, text)
