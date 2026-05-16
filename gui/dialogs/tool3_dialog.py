"""Tool 3: Grid Corrected Backscatter Mosaic dialog.

Parameters identical to Tool 1: projection, resolution, gap fill, elevation filter.
Generates JSON config for dtm_gridder.py backscatter mosaic + GeoTIFF export.
"""
from typing import Optional, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QLineEdit, QPushButton, QLabel,
    QGroupBox, QDialogButtonBox, QListWidget,
    QDoubleSpinBox, QFileDialog, QWidget,
)

from ..utils.config import PROJECTIONS, RESOLUTIONS, GAP_FILL_METHODS


class Tool3Dialog(QDialog):
    """Dialog for Tool 3: Grid Backscatter Mosaic & Export."""

    def __init__(self, selected_files: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selected_files = selected_files
        self._export_format = "GeoTIFF"
        self._setup_ui()
        self.setWindowTitle("Tool 3: Grid Backscatter Mosaic & Export")
        self.resize(480, 460)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Selected files
        layout.addWidget(QLabel(f"Input XSF Files (corrected, green): {len(self._selected_files)} selected"))
        fl = QListWidget()
        for f in self._selected_files[:8]:
            fl.addItem(f)
        fl.setMaximumHeight(60)
        layout.addWidget(fl)

        # Grid params (same as Tool 1)
        grid = QGroupBox("Grid Parameters (same as Tool 1)")
        gf = QFormLayout()

        self._projection = QComboBox()
        self._projection.addItems(PROJECTIONS)
        self._projection.setCurrentText("Auto Detect")
        gf.addRow("Projection:", self._projection)

        self._resolution = QComboBox()
        self._resolution.addItems(RESOLUTIONS)
        self._resolution.setCurrentText("2.0")
        self._resolution.setEditable(True)
        gf.addRow("Resolution (m):", self._resolution)

        self._gap_fill = QComboBox()
        self._gap_fill.addItems(GAP_FILL_METHODS)
        self._gap_fill.setCurrentText("None")
        gf.addRow("Gap Filling:", self._gap_fill)

        self._elev_min = QDoubleSpinBox()
        self._elev_min.setRange(-12000, 12000)
        self._elev_min.setValue(-12000)
        self._elev_min.setSpecialValueText("No limit")
        self._elev_min.setDecimals(1)
        gf.addRow("Elevation Min:", self._elev_min)

        self._elev_max = QDoubleSpinBox()
        self._elev_max.setRange(-12000, 12000)
        self._elev_max.setValue(12000)
        self._elev_max.setSpecialValueText("No limit")
        self._elev_max.setDecimals(1)
        gf.addRow("Elevation Max:", self._elev_max)

        grid.setLayout(gf)
        layout.addWidget(grid)

        # Export
        export = QGroupBox("Export")
        ef = QFormLayout()

        self._export_fmt = QComboBox()
        self._export_fmt.addItems(["GeoTIFF", "COG (Cloud Optimized)", "MBTiles", "ASCII Grid"])
        self._export_fmt.currentTextChanged.connect(lambda t: setattr(self, '_export_format', t))
        ef.addRow("Format:", self._export_fmt)

        out_row = QHBoxLayout()
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("Output directory")
        out_row.addWidget(self._out_dir)
        out_btn = QPushButton("Browse...")
        out_btn.clicked.connect(self._browse_out)
        out_row.addWidget(out_btn)
        ef.addRow("Output Dir:", out_row)

        export.setLayout(ef)
        layout.addWidget(export)

        # Color Ramp preview note
        layout.addWidget(QLabel(
            "Color Ramp and dB range adjustment are available on the Map View\n"
            "after the mosaic is loaded."
        ))

        layout.addStretch()

        # Buttons
        bb = QDialogButtonBox()
        bb.addButton(QPushButton("Save Config"), QDialogButtonBox.ActionRole).clicked.connect(lambda: self.done(2))
        run_btn = QPushButton("Generate & Export")
        run_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; }")
        run_btn.clicked.connect(self.accept)
        bb.addButton(run_btn, QDialogButtonBox.AcceptRole)
        bb.addButton(QPushButton("Cancel"), QDialogButtonBox.RejectRole)
        layout.addWidget(bb)

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d:
            self._out_dir.setText(d)

    def get_params(self) -> dict:
        return {
            "projection": self._projection.currentText(),
            "resolution": self._resolution.currentText(),
            "gap_fill": self._gap_fill.currentText(),
            "elev_min": self._elev_min.value() if self._elev_min.value() != self._elev_min.minimum() else None,
            "elev_max": self._elev_max.value() if self._elev_max.value() != self._elev_max.maximum() else None,
            "output_dir": self._out_dir.text() or None,
            "export_format": self._export_format,
        }
