"""Editor View placeholder for Phase 3.

Will embed PyQtGraph PlotWidget for:
- BSAR angular response curve display (incidence + transmission)
- Spline filtering slider
- GSAB model parameter visualization
"""
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel,
)


class EditorView(QWidget):
    """Placeholder editor view. Full implementation in Phase 3."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        placeholder = QLabel()
        placeholder.setText(
            "Editor View\n\n"
            "PyQtGraph PlotWidget will be embedded here (Phase 3).\n"
            "Features: BSAR angular response curves, GSAB fitting, spline filter slider."
        )
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("""
            QLabel {
                background-color: #252526;
                color: #808080;
                font-size: 14px;
                border: 1px dashed #555;
            }
        """)
        layout.addWidget(placeholder)
