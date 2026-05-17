"""水柱数据 2D 查看器 — 读取 .g3d.nc 文件，滑块浏览 Ping-to-Ping

关键设计：
  - 纵轴 = 实际水深 (m)，从 elevation 变量读出
  - 横轴 = 跨航迹距离 (m) 或波束索引
  - 图像使用 QImage + QTransform 保持正确的高宽比
"""

import os
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QTransform
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel,
    QListWidget, QListWidgetItem, QSplitter, QPushButton, QFileDialog,
)
import pyqtgraph as pg


class Wc2dViewer(QWidget):
    """水柱 2D 查看器：左侧文件列表 + 右侧热力图 + 底部滑块"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._g3d_path = None
        self._slices = []       # [{"type", "data", "x_min", ...}]
        self._loaded_files = {} # path -> slices list
        self._current_idx = 0
        self._colormap = pg.colormap.get('viridis')
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QHBoxLayout()
        btn_open = QPushButton("打开 .g3d.nc 文件")
        btn_open.clicked.connect(self._open_file)
        toolbar.addWidget(btn_open)
        toolbar.addStretch()
        self._info_label = QLabel("未加载文件")
        self._info_label.setStyleSheet("color:#808080;")
        toolbar.addWidget(self._info_label)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)

        self._file_list = QListWidget()
        self._file_list.setMaximumWidth(200)
        self._file_list.itemClicked.connect(self._on_file_selected)
        splitter.addWidget(self._file_list)

        # 右侧：PyQtGraph 热力图
        self._plot = pg.PlotWidget()
        # Aspect ratio will be locked dynamically for polar echograms
        self._plot.setLabel('left', '水深', units='m')
        self._plot.setLabel('bottom', '波束索引')
        self._img = pg.ImageItem()
        self._img.setColorMap(self._colormap)
        self._plot.addItem(self._img)
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        splitter.addWidget(self._plot)

        layout.addWidget(splitter, stretch=1)

        # 底部：滑块 + 色阶范围控制
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("切片:"))
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.valueChanged.connect(self._on_slider_changed)
        slider_row.addWidget(self._slider, stretch=1)
        self._slider_label = QLabel("0 / 0")
        self._slider_label.setMinimumWidth(80)
        slider_row.addWidget(self._slider_label)

        slider_row.addWidget(QLabel(" 色阶:"))
        self._level_label = QLabel("-60 ~ 0 dB")
        self._level_label.setMinimumWidth(100)
        slider_row.addWidget(self._level_label)
        layout.addLayout(slider_row)

    def load_file(self, path: str):
        """加载 .g3d.nc 文件"""
        import netCDF4 as nc
        self._g3d_path = path

        if path in self._loaded_files:
            self._slices = self._loaded_files[path]
        else:
            self._slices = []
            filename = os.path.basename(path).lower()

            with nc.Dataset(path) as ds:
                if len(ds.groups) == 0:
                    # Horizontal slice file (variables at root)
                    slice_vars = []
                    for vname in ds.variables.keys():
                        if 'backscatter_mean' in vname:
                            try:
                                depth_str = vname.split('z=')[-1].replace(')', '')
                                depth = float(depth_str)
                                # Force depth to be negative
                                depth = -abs(depth)
                                prefix = vname.split('_')[0]
                                idx = int(prefix) if prefix.isdigit() else len(slice_vars)
                                slice_vars.append((idx, depth, vname))
                            except Exception:
                                continue
                    slice_vars.sort(key=lambda x: x[0])

                    if 'lon' in ds.variables and 'lat' in ds.variables:
                        lon_min = float(np.nanmin(ds.variables['lon'][:]))
                        lon_max = float(np.nanmax(ds.variables['lon'][:]))
                        lat_min = float(np.nanmin(ds.variables['lat'][:]))
                        lat_max = float(np.nanmax(ds.variables['lat'][:]))
                    else:
                        lon_min = lon_max = lat_min = lat_max = 0.0

                    for idx, depth, vname in slice_vars:
                        data = ds.variables[vname][:]
                        data = np.array(data, dtype=float)
                        data[data <= -32767] = np.nan

                        self._slices.append({
                            "type": "horizontal",
                            "data": data,
                            "x_min": lon_min, "x_max": lon_max,
                            "y_min": lat_min, "y_max": lat_max,
                            "x_label": "经度", "x_unit": "°",
                            "y_label": "纬度", "y_unit": "°",
                            "title": f"水平切片 - 深度: {depth:.2f} m",
                            "elev_min": depth, "elev_max": depth
                        })
                else:
                    groups = sorted(ds.groups.keys(), key=lambda g: int(g))

                    if "polar" in filename:
                        stype = "polar"
                        x_label = "跨航迹距离"
                        x_unit = "m"
                    elif "longitudinal" in filename:
                        stype = "longitudinal"
                        x_label = "沿航迹样本序号"
                        x_unit = ""
                    else:
                        stype = "vertical"
                        x_label = "水平样本序号"
                        x_unit = ""

                    for gname in groups:
                        grp = ds.groups[gname]
                        if 'backscatter_mean' in grp.variables:
                            data = grp.variables['backscatter_mean'][:]
                            data = np.array(data, dtype=float)
                            data[data <= -32767] = np.nan

                            elev_min = elev_max = 0.0
                            if 'elevation' in grp.variables:
                                elev = grp.variables['elevation'][:]
                                elev_min = float(np.nanmin(elev))
                                elev_max = float(np.nanmax(elev))

                            # Ensure depth is negative downwards (0 to -depth)
                            # y_max is set to 0.0 so the image top aligns with sea level (0m)
                            if elev_min >= 0 and elev_max >= 0:
                                y_min = -elev_max
                            else:
                                y_min = min(elev_min, elev_max)
                            y_max = 0.0

                            title_suffix = "极坐标声图" if stype == "polar" else ("纵向剖面" if stype == "longitudinal" else "垂直剖面")

                            if stype == "polar":
                                x_min = float(getattr(grp, 'across_dist_L', -data.shape[1]/2))
                                x_max = float(getattr(grp, 'across_dist_R', data.shape[1]/2))
                            else:
                                x_min = 0
                                x_max = data.shape[1]

                            self._slices.append({
                                "type": stype,
                                "data": data,
                                "x_min": x_min, "x_max": x_max,
                                "y_min": y_min, "y_max": y_max,
                                "x_label": x_label, "x_unit": x_unit,
                                "y_label": "水深", "y_unit": "m",
                                "title": f"{title_suffix} - 切片 {gname} (水深 [{y_min:.1f} ~ {y_max:.1f}] m)",
                                "elev_min": y_min, "elev_max": y_max
                            })

            self._loaded_files[path] = self._slices

        if not self._slices:
            self._info_label.setText("文件中无 backscatter_mean 数据")
            return

        basename = os.path.basename(path)
        found = False
        self._file_list.blockSignals(True)
        for i in range(self._file_list.count()):
            item = self._file_list.item(i)
            if item.data(Qt.UserRole) == path or item.text() == basename:
                item.setData(Qt.UserRole, path)
                self._file_list.setCurrentRow(i)
                found = True
                break
        if not found:
            item = QListWidgetItem(basename)
            item.setData(Qt.UserRole, path)
            self._file_list.addItem(item)
            self._file_list.setCurrentRow(self._file_list.count() - 1)
        self._file_list.blockSignals(False)

        # 锁定极坐标显示的宽高比以保持正视标准扇面
        is_polar = any(sl.get("type") == "polar" for sl in self._slices)
        self._plot.setAspectLocked(is_polar)

        self._slider.setMaximum(len(self._slices) - 1)
        self._slider.setValue(0)
        self._info_label.setText(
            f"{basename}  —  {len(self._slices)} 个切片")
        self._show_slice(0, fit_view=True)

    def _show_slice(self, idx: int, fit_view: bool = False):
        if not self._slices or idx >= len(self._slices):
            return
        self._current_idx = idx
        sl = self._slices[idx]
        data = sl["data"]

        vmin = np.nanpercentile(data, 2)
        vmax = np.nanpercentile(data, 98)
        display = np.clip(data, vmin, vmax)
        display = (display - vmin) / max(vmax - vmin, 1e-6)

        h, w = data.shape
        x_min, x_max = sl["x_min"], sl["x_max"]
        y_min, y_max = sl["y_min"], sl["y_max"]

        dy = (y_max - y_min) / h if h > 0 and y_max > y_min else 1.0
        dx = (x_max - x_min) / w if w > 0 and x_max > x_min else 1.0
        self._img.setImage(display.T, autoLevels=False)
        tr = QTransform()
        tr.translate(x_min, y_min)
        tr.scale(dx, dy)
        self._img.setTransform(tr)

        self._img.setLevels((0, 1))

        self._level_label.setText(f"{vmin:.0f} ~ {vmax:.0f} dB")
        self._slider_label.setText(f"{idx + 1} / {len(self._slices)}")
        self._plot.setTitle(sl["title"])
        self._plot.setLabel('bottom', sl["x_label"], units=sl["x_unit"])
        self._plot.setLabel('left', sl["y_label"], units=sl["y_unit"])
        
        # Dynamically lock aspect ratio to 1.0 for polar slices to preserve physical geometry (1:1 ratio)
        if sl["type"] == "polar":
            self._plot.setAspectLocked(True, ratio=1.0)
        else:
            self._plot.setAspectLocked(False)

        if fit_view:
            self._plot.autoRange()

    def _on_slider_changed(self, value: int):
        self._show_slice(value, fit_view=False)

    def _on_file_selected(self, item):
        path = item.data(Qt.UserRole)
        if path and path != self._g3d_path:
            self.load_file(path)

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 .g3d.nc 文件", "",
            "g3d 文件 (*.g3d.nc *.nc);;所有文件 (*.*)")
        if path:
            self.load_file(path)
