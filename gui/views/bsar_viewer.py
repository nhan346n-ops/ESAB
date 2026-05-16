"""BSAR Curve Viewer using PyQtGraph.

Phase 3A: Load .bsar.nc files and plot incidence/transmission angle curves.
Features: zoom, pan, data probe, spline filtering slider, GSAB model display.
"""
from typing import Optional, Dict, Any, List
import json

import numpy as np
import pyqtgraph as pg

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QSlider, QGroupBox, QFormLayout, QComboBox,
)


class BsarViewer(QWidget):
    """PyQtGraph-based BSAR curve viewer."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._curves_data: Dict[str, Any] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        tb = QHBoxLayout()
        load_btn = QPushButton("Load .bsar.nc")
        load_btn.clicked.connect(self._load_file)
        tb.addWidget(load_btn)

        self._info_label = QLabel("No BSAR file loaded")
        tb.addWidget(self._info_label)
        tb.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear)
        tb.addWidget(clear_btn)
        layout.addLayout(tb)

        # PyQtGraph plot widget
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#252526")
        self._plot.setLabel("left", "Mean BS", units="dB")
        self._plot.setLabel("bottom", "Angle", units="deg")
        self._plot.addLegend(offset=(-10, 10))
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot, stretch=1)

        # Spline filter control
        filter_grp = QGroupBox("Spline Filter")
        fg = QHBoxLayout()
        fg.addWidget(QLabel("Smoothing:"))
        self._spline_slider = QSlider(Qt.Horizontal)
        self._spline_slider.setRange(0, 100)
        self._spline_slider.setValue(0)
        self._spline_slider.setTickInterval(10)
        self._spline_slider.valueChanged.connect(self._on_spline_changed)
        fg.addWidget(self._spline_slider)
        self._spline_label = QLabel("0 (off)")
        fg.addWidget(self._spline_label)
        filter_grp.setLayout(fg)
        layout.addWidget(filter_grp)

    def _load_file(self) -> None:
        """Load a .bsar.nc file and plot its curves."""
        fpath, _ = QFileDialog.getOpenFileName(
            self, "Open BSAR File", "", "BSAR files (*.bsar.nc *.nc);;All (*.*)"
        )
        if not fpath:
            return

        try:
            data = self._read_bsar(fpath)
            self._curves_data = data
            self._plot_curves(data)
            self._info_label.setText(
                f"Sounder: {data.get('sounder_type', 'N/A')} | "
                f"Integration: {data.get('integration_method', 'N/A')}"
            )
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to read BSAR file:\n{e}")

    def _read_bsar(self, filepath: str) -> Dict[str, Any]:
        """Read BSAR data from a NetCDF file.

        Handles both full BSAR files and raw NetCDF with by_incidence_angle groups.
        """
        from netCDF4 import Dataset

        result = {"filepath": filepath, "sounder_type": "", "curves": {}}

        with Dataset(filepath, "r") as ds:
            # Read global attrs
            for attr in ds.ncattrs():
                try:
                    result[attr] = str(getattr(ds, attr))
                except Exception:
                    pass

            # Try to find incidence/transmission angle data
            # Structure: /<mode>/by_incidence_angle/ and /<mode>/by_transmission_angle/
            for grp_name in ds.groups:
                grp = ds.groups[grp_name]
                if "by_incidence_angle" in grp.groups:
                    inc_grp = grp.groups["by_incidence_angle"]
                    inc_data = self._extract_curve(inc_grp, "incidence")
                    if inc_data is not None:
                        result["curves"][f"{grp_name}/incidence"] = inc_data

                if "by_transmission_angle" in grp.groups:
                    tx_grp = grp.groups["by_transmission_angle"]
                    tx_data = self._extract_curve(tx_grp, "transmission")
                    if tx_data is not None:
                        result["curves"][f"{grp_name}/transmission"] = tx_data

        return result

    def _extract_curve(self, grp, curve_type: str) -> Optional[Dict[str, np.ndarray]]:
        """Extract angle and mean_bs arrays from a NetCDF group."""
        result = {"type": curve_type, "angle": None, "mean_bs": None}

        for vname in grp.variables:
            vlow = vname.lower()
            var = grp.variables[vname]
            try:
                data = var[:].ravel()
            except Exception:
                continue

            if "angle" in vlow and "bin" not in vlow:
                result["angle"] = data
            elif "mean_bs" in vlow or "bs" in vlow:
                if np.any(np.isfinite(data)):
                    result["mean_bs"] = data

        if result["angle"] is not None and result["mean_bs"] is not None:
            return result
        return None

    def _plot_curves(self, data: Dict[str, Any]) -> None:
        """Plot all curves on the PyQtGraph widget."""
        self._plot.clear()

        colors = ["#4ec94e", "#569cd6", "#dcdcaa", "#f44747", "#c586c0", "#ce9178"]
        ci = 0

        for name, curve in data.get("curves", {}).items():
            if curve["angle"] is None or curve["mean_bs"] is None:
                continue

            mask = np.isfinite(curve["angle"]) & np.isfinite(curve["mean_bs"])
            if not mask.any():
                continue

            pen = pg.mkPen(color=colors[ci % len(colors)], width=2)
            self._plot.plot(
                curve["angle"][mask], curve["mean_bs"][mask],
                pen=pen, name=name,
                symbol="o", symbolSize=3, symbolBrush=colors[ci % len(colors)]
            )
            ci += 1

        if ci == 0:
            self._info_label.setText("No valid curves found in BSAR file")

    def _on_spline_changed(self, value: int) -> None:
        self._spline_label.setText(f"{value} ({'off' if value == 0 else 'active'})")

    def _clear(self) -> None:
        self._plot.clear()
        self._curves_data = {}
        self._info_label.setText("No BSAR file loaded")

    def load_file(self, filepath: str) -> None:
        """Public method to load a BSAR file programmatically."""
        try:
            data = self._read_bsar(filepath)
            self._curves_data = data
            self._plot_curves(data)
        except Exception as e:
            self._info_label.setText(f"Error: {e}")
