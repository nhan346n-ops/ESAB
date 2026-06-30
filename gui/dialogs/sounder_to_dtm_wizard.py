"""\u591a\u9875\u5411\u5bfc\uff1a\u58f0\u7eb3\u6570\u636e\u81f3 DTM \u5bfc\u51fa\uff08\u5de5\u5177 1 / \u5de5\u5177 3\uff09\u3002

\u9875\u9762\uff1a
  1. \u8f93\u5165/\u8f93\u51fa \u2014 \u9009\u62e9 XSF \u6587\u4ef6\u548c\u8f93\u51fa\u76ee\u5f55
  2. \u6295\u5f71\u4e0e\u5206\u8fa8\u7387 \u2014 \u6295\u5f71\u3001\u5355\u5143\u683c\u5927\u5c0f\u3001\u56fe\u5c42
  3. \u9ad8\u7ea7\u7f51\u683c\u5316 \u2014 \u7f3a\u5931\u586b\u8865\u3001\u6df9\u6ca1\u533a\u5927\u5c0f\u3001\u6709\u6548\u56de\u58f0\u3001\u53cd\u6df7\u53e0
  4. \u6d77\u62d4\u6ee4\u6ce2 \u2014 \u6700\u5c0f/\u6700\u5927\u6c34\u6df1\u3001\u6700\u5c0f\u56de\u58f0\u6570
  5. \u5143\u6570\u636e \u2014 \u6807\u9898\u3001\u673a\u6784\u3001\u6570\u636e\u6765\u6e90\u7b49
  6. \u603b\u7ed3\u4e0e\u8fd0\u884c \u2014 \u68c0\u67e5\u6240\u6709\u53c2\u6570\u5e76\u6267\u884c
"""
import math
from typing import Dict, Optional, List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QListWidget, QFileDialog,
    QGroupBox, QWidget, QTextEdit,
)

from ..utils.config import PROJECTIONS


# ── \u9875\u9762 1\uff1a\u8f93\u5165 / \u8f93\u51fa ────────────────────────────────────────

class PageInputOutput(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("\u6b65\u9aa4 1\uff1a\u8f93\u5165\u4e0e\u8f93\u51fa")
        self.setSubTitle("\u9009\u62e9 XSF \u6587\u4ef6\u548c\u8f93\u51fa\u76ee\u6807")

        layout = QVBoxLayout(self)

        input_label = QLabel("\u8f93\u5165 XSF \u6587\u4ef6:")
        layout.addWidget(input_label)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ \u6dfb\u52a0\u6587\u4ef6")
        add_btn.clicked.connect(self._add_files)
        btn_row.addWidget(add_btn)
        remove_btn = QPushButton("- \u79fb\u9664\u9009\u4e2d")
        remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(remove_btn)
        clear_btn = QPushButton("\u6e05\u9664\u5168\u90e8")
        clear_btn.clicked.connect(self._clear_all)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._file_list = QListWidget()
        self._file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self._file_list.setAlternatingRowColors(True)
        layout.addWidget(self._file_list)

        self._separate = QCheckBox("\u6bcf\u4e2a\u8f93\u5165\u6587\u4ef6\u5355\u72ec\u751f\u6210 DTM")
        self._separate.setChecked(False)
        self._separate.setToolTip(
            "\u52fe\u9009\u65f6\uff0c\u6bcf\u4e2a\u6587\u4ef6\u4ea7\u751f\u81ea\u5df1\u7684 DTM\u3002\n"
            "\u672a\u52fe\u9009\u65f6\uff08\u9ed8\u8ba4\uff09\uff0c\u6240\u6709\u6587\u4ef6\u5408\u5e76\u4e3a\u4e00\u4e2a DTM\u3002"
        )
        layout.addWidget(self._separate)

        out_group = QGroupBox("\u8f93\u51fa")
        of = QFormLayout()
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("\u8f93\u51fa\u76ee\u5f55")
        browse_btn = QPushButton("\u6d4f\u89c8...")
        browse_btn.clicked.connect(self._browse_out)
        row = QHBoxLayout()
        row.addWidget(self._out_dir)
        row.addWidget(browse_btn)
        of.addRow("\u76ee\u5f55:", row)

        self._out_prefix = QLineEdit("bathy")
        of.addRow("\u6587\u4ef6\u524d\u7f00:", self._out_prefix)

        self._overwrite = QCheckBox("\u8986\u76d6\u5df2\u6709\u6587\u4ef6")
        of.addRow(self._overwrite)
        out_group.setLayout(of)
        layout.addWidget(out_group)

        self.setFileList([])

    def setFileList(self, files: List[str]) -> None:
        self._file_list.clear()
        for f in files:
            self._file_list.addItem(f)
        self._files = list(files)
        self.setTitle(f"\u6b65\u9aa4 1\uff1a\u8f93\u5165\u4e0e\u8f93\u51fa ({len(files)} \u4e2a\u6587\u4ef6)")

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "\u6dfb\u52a0 XSF \u6587\u4ef6", "",
            "XSF \u6587\u4ef6 (*.xsf.nc *.nc);;\u6240\u6709\u6587\u4ef6 (*.*)"
        )
        if files:
            for f in files:
                if f not in self._files:
                    self._file_list.addItem(f)
                    self._files.append(f)
            self.setTitle(f"\u6b65\u9aa4 1\uff1a\u8f93\u5165\u4e0e\u8f93\u51fa ({len(self._files)} \u4e2a\u6587\u4ef6)")

    def _remove_selected(self) -> None:
        items = self._file_list.selectedItems()
        if not items:
            return
        for item in items:
            self._files.remove(item.text())
            self._file_list.takeItem(self._file_list.row(item))
        self.setTitle(f"\u6b65\u9aa4 1\uff1a\u8f93\u5165\u4e0e\u8f93\u51fa ({len(self._files)} \u4e2a\u6587\u4ef6)")

    def _clear_all(self) -> None:
        self._file_list.clear()
        self._files.clear()
        self.setTitle("\u6b65\u9aa4 1\uff1a\u8f93\u5165\u4e0e\u8f93\u51fa (0 \u4e2a\u6587\u4ef6)")

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "\u9009\u62e9\u8f93\u51fa\u76ee\u5f55")
        if d:
            self._out_dir.setText(d)

    def getInputFiles(self) -> List[str]:
        return self._files

    def getSeparate(self) -> bool:
        return self._separate.isChecked()

    def getOutputDir(self) -> str:
        return self._out_dir.text()

    def getOutputPrefix(self) -> str:
        return self._out_prefix.text()

    def getOverwrite(self) -> bool:
        return self._overwrite.isChecked()


# ── \u9875\u9762 2\uff1a\u6295\u5f71\u4e0e\u5206\u8fa8\u7387 ───────────────────────────────────

class PageProjection(QWizardPage):
    def __init__(self, bounds: Optional[Tuple[float, float, float, float]] = None, parent=None):
        super().__init__(parent)
        self.setTitle("\u6b65\u9aa4 2\uff1a\u6295\u5f71\u4e0e\u5206\u8fa8\u7387")
        self.setSubTitle("\u8bbe\u7f6e\u8f93\u51fa\u5750\u6807\u7cfb\u548c\u5355\u5143\u683c\u5927\u5c0f")
        self._bounds = bounds

        layout = QFormLayout(self)

        self._proj = QComboBox()
        self._proj.addItems(PROJECTIONS)
        self._proj.setCurrentText("自动检测")
        self._proj.setEditable(True)
        self._proj.currentTextChanged.connect(self._on_proj_changed)
        layout.addRow("\u6295\u5f71:", self._proj)

        self._proj_def = QLineEdit()
        self._proj_def.setText("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
        self._proj_def.setPlaceholderText("PROJ.4 \u5b9a\u4e49\u5b57\u7b26\u4e32")
        self._proj_def.textChanged.connect(self._update_units)
        layout.addRow("PROJ.4 \u5b9a\u4e49:", self._proj_def)

        res_widget = QWidget()
        res_layout = QVBoxLayout(res_widget)
        res_layout.setContentsMargins(0, 0, 0, 0)

        row1 = QHBoxLayout()
        self._resolution = QDoubleSpinBox()
        self._resolution.setRange(0.000001, 10000)
        self._resolution.setDecimals(6)
        self._resolution.setValue(0.000277778)
        self._resolution.valueChanged.connect(self._on_resolution_changed)
        row1.addWidget(self._resolution, stretch=1)
        self._res_unit = QLabel("\u00b0")
        self._res_unit.setStyleSheet("color: #808080;")
        row1.addWidget(self._res_unit)
        auto_btn = QPushButton("\u81ea\u52a8")
        auto_btn.setToolTip("\u6839\u636e\u6570\u636e\u8303\u56f4\u4f30\u8ba1\u6700\u4f73\u5206\u8fa8\u7387\uff08\u7ea6 500 \u4e2a\u5355\u5143\u683c\uff09")
        auto_btn.clicked.connect(self._auto_resolution)
        auto_btn.setMaximumWidth(50)
        row1.addWidget(auto_btn)
        res_layout.addLayout(row1)

        self._res_meters = QLabel("\u2248 \u2014 m")
        self._res_meters.setStyleSheet("color: #808080; font-size: 10px;")
        res_layout.addWidget(self._res_meters)
        layout.addRow("\u5206\u8fa8\u7387:", res_widget)

        self._expand = QCheckBox("\u6269\u5c55\u5230\u6574\u6570\u5206\u8fa8\u7387\u7684\u7f51\u683c\u5927\u5c0f")
        self._expand.setChecked(True)
        self._expand.setToolTip(
            "\u8c03\u6574\u7f51\u683c\u8fb9\u754c\uff0c\u4f7f\u7f51\u683c\u5177\u6709\u6574\u6570\u884c\u548c\u5217\u6570\u3002\n"
            "\u672a\u52fe\u9009\u65f6\uff0c\u7f51\u683c\u8fb9\u7f18\u53ef\u80fd\u6709\u5c0f\u6570\u5355\u5143\u683c\u3002"
        )
        self._expand.toggled.connect(self._update_grid_size)
        layout.addRow(self._expand)

        self._grid_label = QLabel("\u7f51\u683c\u5927\u5c0f: \u2014")
        self._grid_label.setStyleSheet("color: #569cd6; font-weight: bold;")
        layout.addRow(self._grid_label)

        # Geographic bounds
        bounds_grp = QGroupBox("\u5730\u7406\u8fb9\u754c")
        bounds_grid = QGridLayout(bounds_grp)
        self._spins = {}
        for i, (key, label, lo, hi, dec) in enumerate([
            ("west", "\u897f (\u7ecf\u5ea6)", -180, 180, 6),
            ("south", "\u5357 (\u7eac\u5ea6)", -90, 90, 6),
            ("east", "\u4e1c (\u7ecf\u5ea6)", -180, 180, 6),
            ("north", "\u5317 (\u7eac\u5ea6)", -90, 90, 6),
        ]):
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(dec)
            sp.setSingleStep(0.001)
            sp.setWrapping(False)
            sp.setMaximumWidth(130)
            sp.setMaximumHeight(22)
            sp.setStyleSheet("QDoubleSpinBox { padding: 0px 2px; }")
            sp.valueChanged.connect(self._on_bounds_changed)
            self._spins[key] = sp
            bounds_grid.addWidget(QLabel(label), i // 2, (i % 2) * 2)
            bounds_grid.addWidget(sp, i // 2, (i % 2) * 2 + 1)
        layout.addRow(bounds_grp)

        # Pre-fill spinboxes from auto-detected bounds
        if self._bounds is not None:
            lon_min, lat_min, lon_max, lat_max = self._bounds
            self._spins["west"].setValue(lon_min)
            self._spins["south"].setValue(lat_min)
            self._spins["east"].setValue(lon_max)
            self._spins["north"].setValue(lat_max)

        layout.addRow(QLabel("\u5bfc\u51fa\u56fe\u5c42:"))
        self._layers_widget = QWidget()
        ll = QHBoxLayout(self._layers_widget)
        ll.setContentsMargins(0, 0, 0, 0)
        self._layer_checks = {}
        for name, label in [
            ("elevation", "\u6c34\u6df1\uff08\u9ed8\u8ba4\uff09"), ("backscatter", "\u540e\u5411\u6563\u5c04\u5f3a\u5ea6"),
            ("stdev", "\u6807\u51c6\u5dee"), ("elevation_min", "\u6700\u5c0f\u6c34\u6df1"),
            ("elevation_max", "\u6700\u5927\u6c34\u6df1"), ("filtered_sounding", "\u6709\u6548\u56de\u58f0\u6570"),
        ]:
            cb = QCheckBox(label)
            if name == "elevation":
                cb.setChecked(True)
                cb.setEnabled(False)
            elif name == "backscatter":
                cb.setChecked(True)
            ll.addWidget(cb)
            self._layer_checks[name] = cb
        layout.addRow(self._layers_widget)

        self._update_units()
        self._update_grid_size()

    def setBounds(self, bounds: Optional[Tuple[float, float, float, float]]) -> None:
        self._bounds = bounds
        if bounds is not None and hasattr(self, '_spins'):
            lon_min, lat_min, lon_max, lat_max = bounds
            self._spins["west"].setValue(lon_min)
            self._spins["south"].setValue(lat_min)
            self._spins["east"].setValue(lon_max)
            self._spins["north"].setValue(lat_max)
        self._update_grid_size()

    def _on_bounds_changed(self) -> None:
        self._bounds = (
            self._spins["west"].value(),
            self._spins["south"].value(),
            self._spins["east"].value(),
            self._spins["north"].value(),
        )
        self._update_grid_size()

    def getBounds(self) -> Optional[Dict[str, float]]:
        if self._bounds is None:
            return None
        return {
            "west": self._spins["west"].value(),
            "south": self._spins["south"].value(),
            "east": self._spins["east"].value(),
            "north": self._spins["north"].value(),
        }

    def _on_proj_changed(self, text: str) -> None:
        old_proj = self._proj_def.text()
        was_geo = "longlat" in old_proj or "latlong" in old_proj
        
        base = "+ellps=WGS84 +datum=WGS84 +no_defs"
        if text == "自动检测":
            self._proj_def.setText("+proj=longlat " + base)
        elif text == "墨卡托 (Mercator)":
            lat_ts = 0
            if self._bounds:
                _, lat_min, _, lat_max = self._bounds
                lat_ts = round((lat_min + lat_max) / 2)
            self._proj_def.setText(
                "+proj=merc +lon_0={} +lat_ts={} {}".format(0, lat_ts, base))
        elif text == "通用横轴墨卡托 (UTM)":
            zone = 1
            if self._bounds:
                lon_min, _, lon_max, _ = self._bounds
                center_lon = (lon_min + lon_max) / 2
                zone = int((center_lon + 180) / 6) + 1
            south = ""
            if self._bounds and self._bounds[1] < 0:
                south = " +south"
            self._proj_def.setText(
                "+proj=utm +zone={}{} {}".format(zone, south, base))
        elif text == "自定义 EPSG":
            pass
        
        # Auto-convert resolution when switching CRS units
        new_is_geo = "longlat" in self._proj_def.text() or "latlong" in self._proj_def.text()
        if was_geo and not new_is_geo:
            # lat/lon (deg) → projected (m)
            res_m = self._deg_to_m(self._resolution.value())
            self._resolution.setValue(round(res_m, 2))
        elif not was_geo and new_is_geo:
            # projected (m) → lat/lon (deg)
            m_per_deg = self._deg_to_m(1.0)
            res_deg = self._resolution.value() / m_per_deg if m_per_deg else 0.00001
            self._resolution.setValue(round(res_deg, 8))
        self._update_grid_size()

    def _update_units(self) -> None:
        proj = self._proj_def.text()
        if "longlat" in proj or "latlong" in proj:
            self._res_unit.setText("\u00b0")
            self._resolution.setDecimals(6)
            self._resolution.setSingleStep(0.0001)
            self._update_meter_display()
        else:
            self._res_unit.setText("m")
            self._resolution.setDecimals(2)
            self._resolution.setSingleStep(1.0)
            self._res_meters.setText("(\u7c73)")

    def _deg_to_m(self, res_deg: float) -> float:
        if not self._bounds:
            return res_deg * 111320
        _, lat_min, _, lat_max = self._bounds
        mean_lat = (lat_min + lat_max) / 2
        import math
        m_per_deg_lat = 111320
        m_per_deg_lon = 111320 * math.cos(math.radians(mean_lat))
        return res_deg * (m_per_deg_lat + m_per_deg_lon) / 2

    def _update_meter_display(self) -> None:
        if "longlat" in self._proj_def.text():
            res_m = self._deg_to_m(self._resolution.value())
            self._res_meters.setText(f"\u2248 {res_m:.1f} m")
        else:
            self._res_meters.setText("")

    def _auto_resolution(self) -> None:
        wiz = self.wizard()
        files = wiz._page1.getInputFiles() if wiz and hasattr(wiz, '_page1') else []
        if not files:
            self._grid_label.setText("\u65e0\u6587\u4ef6\u2014\u2014\u8bf7\u5148\u6dfb\u52a0 XSF \u6587\u4ef6")
            return

        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
            gws_path = os.path.join(os.path.dirname(__file__), "..", "..", "gws")
            if os.path.isdir(gws_path):
                sys.path.insert(0, gws_path)

            from pyat.function.evaluate_sounder_spatial_resolution import SpatialResolutionEvaluator
            res_meter_list = []
            res_degree_list = []
            for f in files:
                try:
                    evaluator = SpatialResolutionEvaluator(i_paths=[f])
                    rm, rd = evaluator.evaluate()
                    if rm > 0 and rd > 0:
                        res_meter_list.append(rm)
                        res_degree_list.append(rd)
                except Exception:
                    continue
            if not res_meter_list:
                self._grid_label.setText("无法估算分辨率：所有文件评估失败")
                return
            res_meter = max(res_meter_list)
            res_degree = max(res_degree_list)
        except ImportError as e:
            self._grid_label.setText(f"\u540e\u7aef\u5bfc\u5165\u9519\u8bef: {e}")
            return
        except Exception as e:
            self._grid_label.setText(f"\u540e\u7aef\u9519\u8bef: {e}")
            return

        proj = self._proj_def.text()
        if "longlat" in proj or "latlong" in proj:
            self._resolution.setValue(round(res_degree, 8))
        else:
            self._resolution.setValue(round(res_meter, 2))

    def _on_resolution_changed(self) -> None:
        self._update_meter_display()
        self._update_grid_size()

    def _update_grid_size(self) -> None:
        if not self._bounds:
            self._grid_label.setText("\u7f51\u683c\u5927\u5c0f: \u2014\uff08\u8bf7\u6dfb\u52a0\u8f93\u5165\u6587\u4ef6\u4ee5\u4f30\u7b97\uff09")
            return
        lon_min, lat_min, lon_max, lat_max = self._bounds
        res = self._resolution.value()
        if res <= 0:
            self._grid_label.setText("\u7f51\u683c\u5927\u5c0f: \u2014\uff08\u65e0\u6548\u5206\u8fa8\u7387\uff09")
            return
        
        # For projected CRS, resolution is in meters but bounds are in degrees.
        # Convert resolution to approximate degrees for display.
        proj = self._proj_def.text()
        if "longlat" not in proj and "latlong" not in proj:
            m_per_deg = self._deg_to_m(1.0)
            res = res / m_per_deg if m_per_deg else res
        
        dx = lon_max - lon_min
        dy = lat_max - lat_min
        cols = dx / res
        rows = dy / res
        if self._expand.isChecked():
            cols = math.ceil(cols)
            rows = math.ceil(rows)
        cells = int(cols * rows)
        self._grid_label.setText(
            f"\u7f51\u683c\u5927\u5c0f: {int(cols)} \u5217 \u00d7 {int(rows)} \u884c = {cells:,} \u4e2a\u5355\u5143\u683c"
        )

    def getProjDef(self) -> str:
        return self._proj_def.text()

    def getResolution(self) -> str:
        return str(self._resolution.value())

    def getExpand(self) -> bool:
        return self._expand.isChecked()

    def getLayers(self) -> List[str]:
        return [k for k, cb in self._layer_checks.items() if cb.isChecked()]


# ── \u9875\u9762 3\uff1a\u9ad8\u7ea7\u7f51\u683c\u5316 ───────────────────────────────────────

class PageAdvancedGridding(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("\u6b65\u9aa4 3\uff1a\u9ad8\u7ea7\u7f51\u683c\u5316\u9009\u9879")
        self.setSubTitle("\u7f3a\u5931\u586b\u8865\u3001\u6709\u6548\u56de\u58f0\u3001\u53cd\u6df7\u53e0")

        layout = QFormLayout(self)

        self._valid_sounds = QCheckBox("\u4ec5\u4f7f\u7528\u6709\u6548\u56de\u58f0")
        self._valid_sounds.setChecked(True)
        self._valid_sounds.setToolTip("\u6392\u9664\u88ab\u6807\u8bb0\u4e3a\u65e0\u6548/\u62d2\u7edd\u7684\u56de\u58f0\u6570\u636e")
        layout.addRow(self._valid_sounds)

        self._antialiasing = QCheckBox("\u5e94\u7528\u53cd\u6df7\u53e0\u5904\u7406")
        self._antialiasing.setChecked(False)
        self._antialiasing.setToolTip("\u5355\u5143\u683c\u8d28\u5fc3\u7684\u53cc\u7ebf\u6027\u5185\u63d2\u503c")
        layout.addRow(self._antialiasing)

        self._gap_fill = QCheckBox("\u8ba1\u7b97\u540e\u586b\u8865\u7f3a\u5931\u533a\u57df")
        self._gap_fill.setChecked(False)
        self._gap_fill.setToolTip(
            "\u901a\u8fc7\u5185\u63d2\u5904\u7406\u586b\u8865\u7a7a\u5355\u5143\u683c\u3002\n"
            "\u9700\u8981 GDAL netCDF \u63d2\u4ef6\u652f\u6301\u3002"
        )
        layout.addRow(self._gap_fill)

        self._mask_size = QSpinBox()
        self._mask_size.setRange(1, 50)
        self._mask_size.setValue(3)
        self._mask_size.setSuffix(" \u4e2a\u5355\u5143\u683c")
        self._mask_size.setToolTip("\u7f3a\u5931\u586b\u8865\u7684\u6269\u5f20\u6df9\u6ca1\u533a\u5927\u5c0f")
        self._gap_fill.toggled.connect(self._mask_size.setEnabled)
        self._mask_size.setEnabled(False)
        layout.addRow("\u6df9\u6ca1\u533a\u5927\u5c0f:", self._mask_size)

        self._quality = QCheckBox("\u8ba1\u7b97\u8d28\u91cf\u6307\u6807 (TIFF)")
        layout.addRow(self._quality)

    def getParams(self) -> dict:
        return {
            "valid_soundings_only": self._valid_sounds.isChecked(),
            "spatial_antialiasing": self._antialiasing.isChecked(),
            "gap_filling": self._gap_fill.isChecked(),
            "mask_size": self._mask_size.value() if self._gap_fill.isChecked() else None,
            "quality_indicator": self._quality.isChecked(),
        }


# ── \u9875\u9762 4\uff1a\u6d77\u62d4\u6ee4\u6ce2 ─────────────────────────────────────────────

class PageElevationFilters(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("\u6b65\u9aa4 4\uff1a\u6c34\u6df1\u4e0e\u56de\u58f0\u6ee4\u6ce2")
        self.setSubTitle("\u6309\u6c34\u6df1\u548c\u6700\u5c0f\u56de\u58f0\u6570\u7b5b\u9009\u8f93\u5165\u6570\u636e")

        layout = QFormLayout(self)

        self._min_elev = QDoubleSpinBox()
        self._min_elev.setRange(-12000, 12000)
        self._min_elev.setValue(-12000)
        self._min_elev.setSpecialValueText("\u65e0\u9650\u5236")
        self._min_elev.setDecimals(1)
        layout.addRow("\u6700\u5c0f\u6c34\u6df1 (m):", self._min_elev)

        self._max_elev = QDoubleSpinBox()
        self._max_elev.setRange(-12000, 12000)
        self._max_elev.setValue(12000)
        self._max_elev.setSpecialValueText("\u65e0\u9650\u5236")
        self._max_elev.setDecimals(1)
        layout.addRow("\u6700\u5927\u6c34\u6df1 (m):", self._max_elev)

        self._min_sounds = QSpinBox()
        self._min_sounds.setRange(1, 9999)
        self._min_sounds.setValue(1)
        self._min_sounds.setSpecialValueText("\u65e0\u6700\u5c0f\u503c")
        self._min_sounds.setToolTip("\u6bcf\u4e2a\u5355\u5143\u683c\u671f\u671b\u7684\u6700\u5c0f\u56de\u58f0\u6570")
        layout.addRow("\u6bcf\u5355\u5143\u683c\u6700\u5c0f\u56de\u58f0\u6570:", self._min_sounds)

    def getParams(self) -> dict:
        return {
            "min_elevation": self._min_elev.value() if self._min_elev.value() > -12000 else None,
            "max_elevation": self._max_elev.value() if self._max_elev.value() < 12000 else None,
            "min_sounds": self._min_sounds.value() if self._min_sounds.value() > 1 else None,
        }


# ── \u9875\u9762 5\uff1a\u5143\u6570\u636e ──────────────────────────────────────────────────

class PageMetadata(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("\u6b65\u9aa4 5\uff1a\u5143\u6570\u636e")
        self.setSubTitle("\u53ef\u9009 DTM \u5143\u6570\u636e\u5b57\u6bb5")

        layout = QFormLayout(self)
        self._title = QLineEdit()
        layout.addRow("\u6807\u9898:", self._title)
        self._inst = QLineEdit()
        layout.addRow("\u673a\u6784:", self._inst)
        self._source = QLineEdit()
        layout.addRow("\u6570\u636e\u6765\u6e90:", self._source)
        self._refs = QLineEdit()
        layout.addRow("\u53c2\u8003:", self._refs)
        self._comment = QTextEdit()
        self._comment.setMaximumHeight(60)
        layout.addRow("\u5907\u6ce8:", self._comment)

    def getParams(self) -> dict:
        return {
            "title": self._title.text(),
            "institution": self._inst.text(),
            "source": self._source.text(),
            "references": self._refs.text(),
            "comment": self._comment.toPlainText(),
        }


# ── \u9875\u9762 6\uff1a\u603b\u7ed3\u4e0e\u8fd0\u884c ────────────────────────────────────────

class PageSummary(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("\u6b65\u9aa4 6\uff1a\u603b\u7ed3\u4e0e\u8fd0\u884c")
        self.setSubTitle("\u68c0\u67e5\u53c2\u6570\u5e76\u6267\u884c")

        layout = QVBoxLayout(self)
        self._summary = QTextEdit()
        self._summary.setReadOnly(True)
        layout.addWidget(QLabel("\u914d\u7f6e\u603b\u7ed3:"))
        layout.addWidget(self._summary)

    def setSummary(self, text: str) -> None:
        self._summary.setText(text)


# ── \u4e3b\u5411\u5bfc ─────────────────────────────────────────────────────────────

class SounderToDtmWizard(QWizard):
    """\u591a\u9875\u5411\u5bfc\uff1a\u58f0\u7eb3\u6570\u636e\u81f3 DTM \u5bfc\u51fa\u3002"""

    def __init__(self, selected_files: List[str], bounds: Optional[Tuple] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("\u58f0\u7eb3\u6570\u636e\u81f3 DTM \u5bfc\u51fa\u5411\u5bfc")
        self.setButtonText(QWizard.NextButton, "下一步 >")
        self.setButtonText(QWizard.BackButton, "< 上一步")
        self.setButtonText(QWizard.CancelButton, "取消")
        self.setButtonText(QWizard.FinishButton, "完成")
        self.setMinimumSize(550, 500)
        self.setWizardStyle(QWizard.ModernStyle)

        self._page1 = PageInputOutput(self)
        self._page1.setFileList(selected_files)
        self.addPage(self._page1)

        self._page2 = PageProjection(bounds, self)
        self.addPage(self._page2)

        self._page3 = PageAdvancedGridding(self)
        self.addPage(self._page3)

        self._page4 = PageElevationFilters(self)
        self.addPage(self._page4)

        self._page5 = PageMetadata(self)
        self.addPage(self._page5)

        self._page6 = PageSummary(self)
        self.addPage(self._page6)

        self.currentIdChanged.connect(self._on_page_changed)
        self._on_page_changed(0)

    def _on_page_changed(self, page_id: int) -> None:
        if page_id == 5:
            self._page6.setSummary(self._build_summary())

    def _build_summary(self) -> str:
        p = self.getAllParams()
        lines = [f"XSF \u6587\u4ef6: {len(p['input_files'])}", ""]
        lines.append("=== \u6295\u5f71 ===")
        lines.append(f"  \u5b9a\u4e49: {p['proj_def']}")
        lines.append(f"  \u5206\u8fa8\u7387: {p['resolution']} {p.get('res_unit', '')}")
        lines.append(f"  \u6269\u5c55\u5230\u6574\u6570: {p['expand']}")
        lines.append(f"  \u56fe\u5c42: {', '.join(p['layers'])}")
        if p.get("coord"):
            c = p["coord"]
            lines.append(f"  \u8fb9\u754c: \u897f={c['west']:.4f} \u5357={c['south']:.4f} \u4e1c={c['east']:.4f} \u5317={c['north']:.4f}")
        lines.append("")
        lines.append("=== \u7f51\u683c\u5316 ===")
        lines.append(f"  \u4ec5\u6709\u6548\u56de\u58f0: {p['valid_soundings_only']}")
        lines.append(f"  \u53cd\u6df7\u53e0: {p['spatial_antialiasing']}")
        lines.append(f"  \u586b\u8865\u7f3a\u5931: {p['gap_filling']} (\u6df9\u6ca1\u533a: {p['mask_size']})")
        lines.append(f"  \u8d28\u91cf\u6307\u6807: {p['quality_indicator']}")
        lines.append("")
        lines.append("=== \u6ee4\u6ce2 ===")
        lines.append(f"  \u6700\u5c0f\u6c34\u6df1: {p['min_elevation']}")
        lines.append(f"  \u6700\u5927\u6c34\u6df1: {p['max_elevation']}")
        lines.append(f"  \u6bcf\u5355\u5143\u683c\u6700\u5c0f\u56de\u58f0\u6570: {p['min_sounds']}")
        lines.append("")
        lines.append("=== \u5143\u6570\u636e ===")
        for k in ('title', 'institution', 'source', 'comment'):
            if p.get(k):
                lines.append(f"  {k}: {p[k]}")
        return "\n".join(lines)

    def getAllParams(self) -> dict:
        params = {
            "input_files": self._page1.getInputFiles(),
            "output_dir": self._page1.getOutputDir(),
            "output_prefix": self._page1.getOutputPrefix(),
            "overwrite": self._page1.getOverwrite(),
            "separate": self._page1.getSeparate(),
            "proj_def": self._page2.getProjDef(),
            "resolution": self._page2.getResolution(),
            "expand": self._page2.getExpand(),
            "layers": self._page2.getLayers(),
            "coord": self._page2.getBounds(),
        }
        params.update(self._page3.getParams())
        params.update(self._page4.getParams())
        params.update(self._page5.getParams())
        return params
