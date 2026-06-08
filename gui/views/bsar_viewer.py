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

        # Row 1: Toolbar
        tb = QHBoxLayout()
        load_btn = QPushButton("\u52a0\u8f7d .bsar.nc")
        load_btn.clicked.connect(self._load_file)
        tb.addWidget(load_btn)
        tb.addStretch()
        clear_btn = QPushButton("\u6e05\u9664")
        clear_btn.clicked.connect(self._clear)
        tb.addWidget(clear_btn)
        layout.addLayout(tb)

        # Row 2: Curve selector + info
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("\u663e\u793a\u66f2\u7ebf:"))
        self._curve_combo = QComboBox()
        self._curve_combo.setMinimumWidth(220)
        self._curve_combo.currentIndexChanged.connect(self._on_curve_filter)
        self._curve_combo.addItem("\u5168\u90e8")
        self._curve_combo.setEnabled(False)
        sel_row.addWidget(self._curve_combo)

        self._info_label = QLabel("\u672a\u52a0\u8f7d BSAR \u6587\u4ef6")
        sel_row.addWidget(self._info_label, stretch=1)
        layout.addLayout(sel_row)

        # PyQtGraph plot
        self._plot = pg.PlotWidget()
        self._plot.setBackground("#FFFFFF")
        self._plot.setLabel("left", "\u540e\u5411\u6563\u5c04\u5f3a\u5ea6", units="dB")
        self._plot.setLabel("bottom", "\u5165\u5c04\u89d2", units="\u00b0  (\u5de6\u8237/\u53f3\u8237)")
        self._plot.addLegend(offset=(-10, 10))
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot, stretch=1)

        filter_grp = QGroupBox("\u6837\u6761\u6ee4\u6ce2")
        fg = QHBoxLayout()
        fg.addWidget(QLabel("\u5e73\u6ed1:"))
        self._spline_slider = QSlider(Qt.Horizontal)
        self._spline_slider.setRange(0, 100)
        self._spline_slider.setValue(0)
        self._spline_slider.setTickInterval(10)
        self._spline_slider.valueChanged.connect(self._on_spline_changed)
        fg.addWidget(self._spline_slider)
        self._spline_label = QLabel("0 (\u5173\u95ed)")
        fg.addWidget(self._spline_label)
        filter_grp.setLayout(fg)
        layout.addWidget(filter_grp)

    def _load_file(self) -> None:
        """\u52a0\u8f7d\u5e76\u663e\u793a .bsar.nc \u6587\u4ef6\u3002"""
        fpath, _ = QFileDialog.getOpenFileName(
            self, "\u6253\u5f00 BSAR \u6587\u4ef6", "", "BSAR \u6587\u4ef6 (*.bsar.nc *.nc);;\u6240\u6709\u6587\u4ef6 (*.*)"
        )
        if not fpath:
            return

        import os
        if not os.path.isfile(fpath):
            QMessageBox.warning(self, "\u52a0\u8f7d\u5931\u8d25",
                                f"\u6587\u4ef6\u4e0d\u5b58\u5728:\n{fpath}")
            return

        try:
            data = self._read_bsar(fpath)
            if not data.get("curves"):
                QMessageBox.warning(self, "\u52a0\u8f7d\u5931\u8d25",
                                    "\u6587\u4ef6\u4e2d\u672a\u627e\u5230\u89d2\u5ea6\u54cd\u5e94\u66f2\u7ebf\u6570\u636e\u3002\n\n"
                                    "\u8bf7\u786e\u4fdd\u8be5\u6587\u4ef6\u662f\u901a\u8fc7\u672c\u8f6f\u4ef6\u751f\u6210\u7684 BSAR \u6587\u4ef6\u3002")
                return
            self._curves_data = data
            # Populate curve combo
            self._curve_combo.blockSignals(True)
            self._curve_combo.clear()
            self._curve_combo.addItem("\u5168\u90e8")
            for name in data.get("curves", {}):
                self._curve_combo.addItem(name)
            self._curve_combo.setCurrentIndex(0)
            self._curve_combo.setEnabled(True)
            self._curve_combo.blockSignals(False)
            # Plot with current filter
            self._replot_with_filter()
            sounder = data.get('sounder_type', 'N/A')
            integ = data.get('integration_method', 'N/A')
            self._info_label.setText(
                f"\u58f0\u7eb3\u7c7b\u578b: {sounder} | \u79ef\u5206: {integ}"
            )
        except OSError as e:
            self._info_label.setText("\u65e0\u6cd5\u8bfb\u53d6 BSAR \u6587\u4ef6")
            QMessageBox.warning(self, "\u52a0\u8f7d\u5931\u8d25",
                                f"\u65e0\u6cd5\u6253\u5f00 netCDF \u6587\u4ef6\u3002\n\n"
                                f"\u8def\u5f84: {fpath}\n\n"
                                f"\u539f\u56e0: {e}\n\n"
                                f"\u53ef\u80fd\u539f\u56e0:\n"
                                f"  \u2022 \u6587\u4ef6\u4e0d\u662f\u6709\u6548\u7684 netCDF \u683c\u5f0f\n"
                                f"  \u2022 \u6587\u4ef6\u5df2\u635f\u574f\u6216\u4e0d\u5b8c\u6574\n"
                                f"  \u2022 \u6587\u4ef6\u540d\u5305\u542b\u7279\u6b8a\u5b57\u7b26\n")
        except Exception as e:
            self._info_label.setText("\u8bfb\u53d6\u5931\u8d25")
            QMessageBox.warning(self, "\u52a0\u8f7d\u5931\u8d25",
                                f"\u65e0\u6cd5\u8bfb\u53d6 BSAR \u6587\u4ef6:\n{e}")

    def _read_bsar(self, filepath: str) -> Dict[str, Any]:
        """Read BSAR data from a NetCDF file.

        Falls back to mmap-based copy if direct netCDF open fails
        (Windows file locking issue with HDF5).

        BSAR netCDF structure (from ``MeanBSModel.save_to_netcdf``):
          ``/<mode>/by_incidence_angle/``  — 1-D variables [angle]
          ``/<mode>/by_transmission_angle/`` — 3-D variables [rx, tx, angle]
        """
        import os, tempfile, mmap

        def _open_nc(path):
            """Try opening a netCDF file; fallback: mmap → temp copy.

            Returns (dataset, cleanup_fn) tuple.  ``cleanup_fn`` must
            be called after the dataset is no longer needed.
            """
            from netCDF4 import Dataset
            import locale
            enc = locale.getlocale()[1]

            # Direct open
            try:
                return Dataset(path, "r", encoding=enc or "utf-8"), None

            # File locked or format unrecognised → mmap fallback
            except Exception:
                fd = os.open(path, os.O_RDONLY)
                try:
                    m = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                    raw = m[:]
                    m.close()
                finally:
                    os.close(fd)
                tmp = os.path.join(tempfile.gettempdir(),
                                   f"bsar_{os.path.basename(path)}")
                with open(tmp, "wb") as fh:
                    fh.write(raw)

                def _cleanup():
                    ds.close()
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

                return Dataset(tmp, "r"), _cleanup

        result = {"filepath": filepath, "sounder_type": "", "curves": {}}
        ds, cleanup_fn = _open_nc(filepath)
        try:
            for attr in ds.ncattrs():
                try:
                    result[attr] = str(getattr(ds, attr))
                except Exception:
                    pass

            for grp_name in ds.groups:
                grp = ds.groups[grp_name]
                if "by_incidence_angle" in grp.groups:
                    inc_grp = grp.groups["by_incidence_angle"]
                    inc_data = self._extract_curve(inc_grp, "incidence")
                    if inc_data is not None:
                        result["curves"][f"{grp_name}/\u5165\u5c04\u89d2"] = inc_data

                if "by_transmission_angle" in grp.groups:
                    tx_grp = grp.groups["by_transmission_angle"]
                    tx_data = self._extract_curve(tx_grp, "transmission")
                    if tx_data is not None:
                        result["curves"][f"{grp_name}/\u53d1\u5c04\u89d2"] = tx_data
        finally:
            if cleanup_fn:
                cleanup_fn()
            else:
                ds.close()

        return result

    def _extract_curve(self, grp, curve_type: str) -> Optional[Dict[str, np.ndarray]]:
        """Extract angle and mean_bs arrays from a NetCDF group.

        BSAR netCDF structure (from ``MeanBSModel.save_to_netcdf``):
          ``/<mode>/by_incidence_angle/``  — 1-D variables [angle]
          ``/<mode>/by_transmission_angle/`` — 3-D variables [rx, tx, angle]

        Variable priority (filtered over raw):
          ``mean_bs`` > ``raw_mean_bs`` > ``mean_residual_bs`` > ``raw_mean_residual_bs``
        """
        result = {"type": curve_type, "angle": None, "mean_bs": None}

        # Read angle coordinate (always 1-D)
        if "angle" in grp.variables:
            try:
                result["angle"] = grp.variables["angle"][:]
            except Exception:
                pass

        # Try variables in priority order; stop at the first valid match
        preferred = ["mean_bs", "raw_mean_bs", "mean_residual_bs", "raw_mean_residual_bs"]
        for vname in preferred:
            if vname not in grp.variables:
                continue
            try:
                data = grp.variables[vname][:]
            except Exception:
                continue
            # Transmission curves are 3-D [rx_antenna, tx_beam, angle];
            # average over rx and tx to obtain a single 1-D angle curve.
            if data.ndim == 3:
                data = np.nanmean(data, axis=(0, 1))
            data = data.ravel()
            if np.any(np.isfinite(data)):
                result["mean_bs"] = data
                break  # take the first (highest-priority) valid curve

        if result["angle"] is not None and result["mean_bs"] is not None:
            return result
        return None

    def _plot_curves(self, data: Dict[str, Any],
                      filter_name: Optional[str] = None) -> None:
        """Plot all curves on the PyQtGraph widget.

        Incidence curves (absolute angles 0..89°) are mirrored around
        0° so the X-axis runs from port (negative) to starboard
        (positive) with nadir (0°) at centre.
        Transmission curves already span signed angles (-80..+80°).
        """
        self._plot.clear()

        colors = ["#4ec94e", "#569cd6", "#dcdcaa", "#f44747", "#c586c0", "#ce9178"]
        ci = 0

        for name, curve in data.get("curves", {}).items():
            # Skip if a specific filter is active
            if filter_name is not None and name != filter_name:
                continue
            if curve["angle"] is None or curve["mean_bs"] is None:
                continue

            mask = np.isfinite(curve["angle"]) & np.isfinite(curve["mean_bs"])
            if not mask.any():
                continue

            x = curve["angle"][mask]
            y = curve["mean_bs"][mask]

            # Mirror incidence curves: port-negative / starboard-positive
            if curve.get("type") == "incidence" and np.all(x >= 0):
                # Incidence data runs 0..89° (absolute).
                # Mirror: [-89..-1, 0, 1..89] with symmetric BS values.
                rev_x = -x[::-1]   # [-89, -88, ..., -1]
                rev_y = y[::-1]    # corresponding BS values
                x = np.concatenate([rev_x, x])
                y = np.concatenate([rev_y, y])

            pen = pg.mkPen(color=colors[ci % len(colors)], width=2)
            self._plot.plot(
                x, y,
                pen=pen, name=name,
                symbol="o", symbolSize=3, symbolBrush=colors[ci % len(colors)]
            )
            ci += 1

        if ci == 0:
            self._info_label.setText("\u672a\u627e\u5230\u6709\u6548\u66f2\u7ebf\u6570\u636e")

    def _on_spline_changed(self, value: int) -> None:
        self._spline_label.setText(f"{value} ({'关闭' if value == 0 else '开启'})")

    def _replot_with_filter(self) -> None:
        """Re-plot using the currently selected combo filter."""
        sel = self._curve_combo.currentText()
        self._plot_curves(self._curves_data,
                          filter_name=None if sel == "\u5168\u90e8" else sel)

    def _on_curve_filter(self, _idx: int) -> None:
        if self._curves_data:
            self._replot_with_filter()

    def _clear(self) -> None:
        self._plot.clear()
        self._curves_data = {}
        self._curve_combo.blockSignals(True)
        self._curve_combo.clear()
        self._curve_combo.addItem("\u5168\u90e8")
        self._curve_combo.setEnabled(False)
        self._curve_combo.blockSignals(False)
        self._info_label.setText("\u672a\u52a0\u8f7d BSAR \u6587\u4ef6")

    def load_file(self, filepath: str) -> None:
        """Public method to load a BSAR file programmatically."""
        try:
            data = self._read_bsar(filepath)
            self._curves_data = data
            self._curve_combo.blockSignals(True)
            self._curve_combo.clear()
            self._curve_combo.addItem("\u5168\u90e8")
            for name in data.get("curves", {}):
                self._curve_combo.addItem(name)
            self._curve_combo.setCurrentIndex(0)
            self._curve_combo.setEnabled(True)
            self._curve_combo.blockSignals(False)
            self._replot_with_filter()
        except Exception as e:
            self._info_label.setText(f"\u9519\u8bef: {e}")
