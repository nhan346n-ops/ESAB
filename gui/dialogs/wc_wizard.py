"""水柱分析多页向导 — QWizard，每个工具独立初始化，完整对接后端参数。

页面：
  1. 输入/输出（XSF 文件 + 输出目录）
  2. 算法参数（工具特有，含 Compute 按钮 + 网格大小 + 地理边界）
  3. 输出选项（图层 + 归一化 + 滤波配置）
  4. 汇总与执行
"""

import os, json, tempfile, math
from typing import Optional, List
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QListWidget, QFileDialog,
    QGroupBox, QWidget, QTextEdit,
)


# ── 4 工具参数配置 ──

MODE_CONFIG = {
    "horizontal": {
        "title": "水平切片",
        "overview": "将水柱数据按指定深度层进行水平剖切，生成水体后向散射空间分布的 g3d 网格文件。",
        "gws_config": "sonar/wc/horizontal_section.json",
        "suffix": "WCHorizontalEcho.g3d.nc",
        "params": [
            {"key": "delta_elevation","label":"垂直层间距 (m)","type":"float","default":0.0,"auto":"0=自动"},
            {"key": "grid_count","label":"网格数量","type":"int","default":0,"auto":"0=自动"},
            {"key": "vertical_offset","label":"垂直偏移 (m)","type":"float","default":0.0},
            {"key": "vertical_reference","label":"垂向参考","type":"enum",
             "choices":["chart_datum","sea_floor"],
             "labels":{"chart_datum":"海图基准面","sea_floor":"海底面"},"default":"chart_datum"},
        ],
        "has_coord": True, "has_resolution": True,
    },
    "longitudinal": {
        "title": "纵向剖面",
        "overview": "沿航迹方向生成水柱垂直剖面，将后向散射按深度和航迹距离展开为二维声学图像。",
        "gws_config": "sonar/wc/longitudinal_section.json",
        "suffix": "WCLongitudinalEcho.g3d.nc",
        "params": [
            {"key": "delta_across","label":"跨测线间距 (m)","type":"float","default":0.0,"auto":"0=自动"},
            {"key": "grid_count","label":"网格数量","type":"int","default":0,"auto":"0=自动"},
            {"key": "delta_elevation","label":"垂直采样间距 (m)","type":"float","default":0.0,"auto":"0=自动"},
            {"key": "delta_along","label":"沿测线间距 (m)","type":"float","default":0.0,"auto":"0=自动"},
            {"key": "interpolate","label":"线性插值填补空隙","type":"bool","default":False},
        ],
        "has_coord": False, "has_resolution": False,
    },
    "polar": {
        "title": "极坐标声图",
        "overview": "将单 Ping 水柱回波按波束角和斜距展开为极坐标图像，呈现全角度散射结构。",
        "gws_config": "sonar/wc/polar_echograms.json",
        "suffix": "PolarEchograms.g3d.nc",
        "params": [
            {"key": "sample_resolution","label":"采样分辨率 (m)","type":"float","default":0.0,"auto":"0=自动"},
            {"key": "height","label":"图像高度 (像素)","type":"int","default":500,"auto":"0=自动"},
            {"key": "interpolate","label":"线性插值填补空隙","type":"bool","default":True},
        ],
        "has_coord": False, "has_resolution": False,
    },
    "vertical": {
        "title": "垂直积分",
        "overview": "将整个水柱后向散射能量沿深度积分投影到水平面，生成类似镶嵌图的栅格图像。",
        "gws_config": "sonar/wc/vertical_integration.json",
        "suffix": "WCVerticalEcho.tiff",
        "params": [
            {"key": "enable_normalization","label":"按斜距归一化补偿","type":"bool","default":False},
        ],
        "has_coord": True, "has_resolution": True,
    },
}

LAYER_CHOICES = [
    ("backscatter_mean", "后向散射均值"),
    ("backscatter_max", "后向散射最大值"),
    ("backscatter_comp_mean", "补偿后均值"),
    ("backscatter_comp_max", "补偿后最大值"),
]


# ── Page 1: 输入/输出 ──

class PageInputOutput(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("步骤 1: 输入文件与输出目录")
        self.setSubTitle("指定输入的 XSF 文件和输出位置。")

        layout = QFormLayout(self)

        self._files_list = QListWidget()
        self._files_list.setMaximumHeight(60)
        layout.addRow("XSF 文件:", self._files_list)
        file_row = QHBoxLayout()
        btn_add = QPushButton("添加文件")
        btn_add.clicked.connect(self._add_files)
        btn_clear = QPushButton("清空")
        btn_clear.clicked.connect(self._files_list.clear)
        file_row.addWidget(btn_add); file_row.addWidget(btn_clear)
        layout.addRow(file_row)

        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("留空则使用输入文件目录")
        btn_out = QPushButton("浏览...")
        btn_out.clicked.connect(self._browse_out)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_dir); out_row.addWidget(btn_out)
        layout.addRow("输出目录:", out_row)

        self._overwrite = QCheckBox("覆盖已有文件")
        self._overwrite.setChecked(True)
        layout.addRow(self._overwrite)

    def get_files(self): return [self._files_list.item(i).text() for i in range(self._files_list.count())]
    def get_output_dir(self): return self._out_dir.text()
    def get_overwrite(self): return self._overwrite.isChecked()
    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择 XSF 文件", "", "XSF 文件 (*.xsf.nc *.nc);;所有文件 (*.*)")
        for f in files: self._files_list.addItem(f)
    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d: self._out_dir.setText(d)


# ── Page 2: 算法参数（工具特有，含 Compute + 网格大小 + 地理边界） ──

class PageParams(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("步骤 2: 算法参数")
        self.setSubTitle("配置核心处理参数。")
        self._param_widgets = {}
        self._layout = QVBoxLayout(self)
        self._scroll = QWidget()
        self._form_layout = QVBoxLayout(self._scroll)
        self._layout.addWidget(self._scroll)
        self._layout.addStretch()

    def initializePage(self):
        wiz = self.wizard()
        mode = getattr(wiz, '_mode', 'horizontal') if wiz else 'horizontal'
        cfg = MODE_CONFIG.get(mode, list(MODE_CONFIG.values())[0])
        self._rebuild(cfg)

    def _rebuild(self, cfg):
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self._param_widgets.clear()

        # 算法参数
        grp = QGroupBox("算法参数")
        form = QFormLayout(grp)
        for p in cfg["params"]:
            lbl = QLabel(p["label"])
            w = self._create_widget(p)
            form.addRow(lbl, w)
            self._param_widgets[p["key"]] = w
        self._form_layout.addWidget(grp)

        # 地理边界
        if cfg.get("has_coord"):
            coord_grp = QGroupBox("地理边界")
            coord_form = QFormLayout(coord_grp)
            self._coord_spins = {}
            for pos, lo, hi in [("西",-180,180),("东",-180,180),("南",-90,90),("北",-90,90)]:
                sp = QDoubleSpinBox(); sp.setRange(lo, hi); sp.setDecimals(5)
                coord_form.addRow(QLabel(pos), sp)
                self._coord_spins[pos] = sp
            coord_form.addRow(QCheckBox("展开到整数弧分网格"))
            # 从 XSF 文件自动读取地理边界按钮
            btn_auto_bounds = QPushButton("从文件读取边界")
            btn_auto_bounds.clicked.connect(self._auto_fill_bounds)
            coord_form.addRow(btn_auto_bounds)
            # 自动填充：如果有文件已选，立即读取边界
            if self._get_input_files():
                self._auto_fill_bounds()
            self._form_layout.addWidget(coord_grp)

        # 空间分辨率
        if cfg.get("has_resolution"):
            res_grp = QGroupBox("空间分辨率")
            res_form = QFormLayout(res_grp)
            
            res_widget = QWidget()
            res_row = QHBoxLayout(res_widget)
            res_row.setContentsMargins(0,0,0,0)
            
            self._res_spin = QDoubleSpinBox()
            self._res_spin.setRange(0.0, 1000)
            self._res_spin.setDecimals(10)
            self._res_spin.setValue(0.00027778)
            self._res_spin.setToolTip("0 = 自动估算")
            
            res_row.addWidget(self._res_spin, stretch=2)
            
            # Unit combobox: 度 (°), 角分 ('), 角秒 (")
            self._res_unit_combo = QComboBox()
            self._res_unit_combo.addItems(["度 (°)", "角分 (')", "角秒 (\")"])
            res_row.addWidget(self._res_unit_combo, stretch=1)
            
            btn_auto_res = QPushButton("计算")
            btn_auto_res.setToolTip("根据输入文件估算推荐分辨率")
            btn_auto_res.clicked.connect(lambda: self._compute_param(self._res_spin, "target_resolution"))
            res_row.addWidget(btn_auto_res)
            
            res_form.addRow(QLabel("分辨率:"), res_widget)
            
            # Add labels for conversions and estimations
            self._degree_val_label = QLabel("度值: —")
            self._degree_val_label.setStyleSheet("color:#888;")
            res_form.addRow(self._degree_val_label)
            
            self._ns_est_label = QLabel("N-S 估算: —")
            self._ns_est_label.setStyleSheet("color:#aaa;")
            res_form.addRow(self._ns_est_label)
            
            self._we_est_label = QLabel("W-E 估算: —")
            self._we_est_label.setStyleSheet("color:#aaa;")
            res_form.addRow(self._we_est_label)
            
            # 网格大小显示
            self._grid_label = QLabel("网格大小: —")
            self._grid_label.setStyleSheet("color:#569cd6;font-weight:bold;")
            res_form.addRow(self._grid_label)
            
            self._prev_unit_index = 0
            
            # Connect slots
            self._res_spin.valueChanged.connect(self._update_grid_size)
            self._res_unit_combo.currentIndexChanged.connect(self._on_res_unit_changed)
            if cfg.get("has_coord"):
                for sp in self._coord_spins.values():
                    sp.valueChanged.connect(self._update_grid_size)
            self._form_layout.addWidget(res_grp)

    def _create_widget(self, p):
        key = p["key"]
        if p["type"] == "bool":
            cb = QCheckBox(p["label"]); cb.setChecked(p.get("default",False)); return cb
        if p["type"] == "enum":
            cb = QComboBox()
            labels = p.get("labels",{})
            for v in p.get("choices",[]): cb.addItem(labels.get(v,v), v)
            if p.get("default"): i = cb.findData(p["default"]); cb.setCurrentIndex(i) if i>=0 else None
            return cb
        if p["type"] == "int":
            w = QWidget(); row = QHBoxLayout(w); row.setContentsMargins(0,0,0,0)
            sp = QSpinBox(); sp.setRange(p.get("min",0), p.get("max",10**6))
            sp.setValue(int(p.get("default",0))); sp.setToolTip(p.get("auto",""))
            row.addWidget(sp, stretch=1)
            if p.get("auto"):
                btn = QPushButton("计"); btn.setMaximumWidth(22); btn.setMaximumHeight(22)
                btn.setToolTip("计算并填入推荐值")
                btn.clicked.connect(lambda checked, s=sp, k=key: self._compute_param(s, k))
                row.addWidget(btn)
            return w
        if p["type"] == "float":
            w = QWidget(); row = QHBoxLayout(w); row.setContentsMargins(0,0,0,0)
            sp = QDoubleSpinBox(); sp.setRange(p.get("min",-1e9), p.get("max",1e9)); sp.setDecimals(4)
            sp.setValue(float(p.get("default",0))); sp.setToolTip(p.get("auto",""))
            row.addWidget(sp, stretch=1)
            if p.get("auto"):
                btn = QPushButton("计"); btn.setMaximumWidth(22); btn.setMaximumHeight(22)
                btn.setToolTip("计算并填入推荐值")
                btn.clicked.connect(lambda checked, s=sp, k=key: self._compute_param(s, k))
                row.addWidget(btn)
            return w

    def _on_res_unit_changed(self, index):
        if not hasattr(self, '_prev_unit_index'):
            self._prev_unit_index = 0
        
        # Get current value
        val = self._res_spin.value()
        
        # Convert val to degrees first
        if self._prev_unit_index == 1: # arcminute
            val_deg = val / 60.0
        elif self._prev_unit_index == 2: # arcsecond
            val_deg = val / 3600.0
        else:
            val_deg = val
            
        # Convert val_deg to new unit
        if index == 1: # arcminute
            new_val = val_deg * 60.0
        elif index == 2: # arcsecond
            new_val = val_deg * 3600.0
        else:
            new_val = val_deg
            
        self._prev_unit_index = index
        
        self._res_spin.blockSignals(True)
        self._res_spin.setValue(new_val)
        self._res_spin.blockSignals(False)
        
        self._update_grid_size()

    def _compute_param(self, spinbox, key):
        """计算并填入参数推荐值（基于 backend 算法类）"""
        files = self._get_input_files()
        if not files:
            spinbox.setStyleSheet("QDoubleSpinBox{color:orange} QSpinBox{color:orange}")
            return
        
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QCursor
            self.setCursor(QCursor(Qt.WaitCursor))
            
            v = None
            if key == "target_resolution":
                from pyat.function.evaluate_sounder_spatial_resolution import SpatialResolutionEvaluator
                import pyat.sounder.sounder_driver_factory as sounder_driver_factory
                
                # Fetch center coordinates
                with sounder_driver_factory.open_sounder(files[0]) as driver:
                    nav_point = int(driver.sounder_file.swath_count / 2)
                    lons = driver.read_platform_longitudes()
                    lats = driver.read_platform_latitudes()
                    self._center_lon = float(lons.flat[nav_point])
                    self._center_lat = float(lats.flat[nav_point])
                
                se = SpatialResolutionEvaluator(files)
                res_meter, res_degree = se.evaluate()
                
                unit_idx = self._res_unit_combo.currentIndex()
                if unit_idx == 1: # arcminute
                    v = res_degree * 60.0
                elif unit_idx == 2: # arcsecond
                    v = res_degree * 3600.0
                else:
                    v = res_degree
                    
                spinbox.setValue(v)
                self._update_grid_size()
                spinbox.setToolTip(f"推荐值: {res_degree:.10f}°")
                return
                
            elif key in ("delta_elevation", "delta_across", "delta_along"):
                from pyat.function.evaluate_longitudinal_section_arguments import LongitudinalSectionArgumentsEvaluator
                evaluator = LongitudinalSectionArgumentsEvaluator(files)
                delta_elevation, delta_across, delta_along = evaluator._evaluate()
                if key == "delta_elevation":
                    v = delta_elevation
                elif key == "delta_across":
                    v = delta_across
                else:
                    v = delta_along
                    
            elif key in ("sample_resolution", "height"):
                from pyat.function.evaluate_polar_echograms_arguments import PolarEchogramsArgumentsEvaluator
                evaluator = PolarEchogramsArgumentsEvaluator(files)
                sample_res, height = evaluator.evaluate()
                if key == "sample_resolution":
                    v = sample_res
                else:
                    v = int(height)
                    
            elif key == "grid_count":
                import netCDF4 as nc
                with nc.Dataset(files[0]) as ds:
                    bg = ds.groups['声呐 (Sonar)'].groups['Beam_group1']
                    bathy = bg.groups.get('水深 (Bathymetry)')
                    depth_arr = np.array([])
                    if bathy and 'detection_z' in bathy.variables:
                        z = np.array(bathy.variables['detection_z'][:]).ravel()
                        z = z[np.isfinite(z) & (z < 0)]
                        if len(z) > 0: depth_arr = -z
                    if len(depth_arr) == 0 and 'tx_transducer_depth' in bg.variables:
                        z = np.array(bg.variables['tx_transducer_depth'][:]).ravel()
                        z = z[np.isfinite(z)]
                        if len(z) > 0: depth_arr = z
                    mean_dep = float(np.mean(depth_arr)) if len(depth_arr) > 0 else 30.0
                v = max(10, int(mean_dep / 2))
                
            if v is not None:
                if isinstance(spinbox, QDoubleSpinBox):
                    spinbox.setValue(float(v))
                else:
                    spinbox.setValue(int(v))
                spinbox.setToolTip(f"推荐值: {v}")
                
        except Exception as e:
            print("参数计算出错:", e)
            if isinstance(spinbox, QDoubleSpinBox):
                spinbox.setValue(1.0)
            else:
                spinbox.setValue(1)
        finally:
            self.unsetCursor()

    def _update_grid_size(self):
        if not hasattr(self, '_res_spin') or not hasattr(self, '_coord_spins'): return
        s = self._coord_spins
        west, east = s["西"].value(), s["东"].value()
        south, north = s["南"].value(), s["北"].value()
        
        val = self._res_spin.value()
        unit_idx = self._res_unit_combo.currentIndex()
        if unit_idx == 1: # arcminute
            res_deg = val / 60.0
        elif unit_idx == 2: # arcsecond
            res_deg = val / 3600.0
        else:
            res_deg = val
            
        if val <= 0:
            self._degree_val_label.setText("度值: 自动")
            self._ns_est_label.setText("N-S 估算: 自动")
            self._we_est_label.setText("W-E 估算: 自动")
            self._grid_label.setText("网格大小: 自动")
            return
            
        self._degree_val_label.setText(f"度值: {res_deg:.10f} °")
        
        from pyproj import Geod
        geod = Geod(ellps='WGS84')
        lon = getattr(self, '_center_lon', 0.0)
        lat = getattr(self, '_center_lat', 0.0)
        
        try:
            _, _, ns_dist = geod.inv(lon, lat, lon, lat + res_deg)
            _, _, we_dist = geod.inv(lon, lat, lon + res_deg, lat)
            self._ns_est_label.setText(f"N-S 估算: {ns_dist:.2f} m")
            self._we_est_label.setText(f"W-E 估算: {we_dist:.2f} m")
        except Exception:
            self._ns_est_label.setText("N-S 估算: 错误")
            self._we_est_label.setText("W-E 估算: 错误")
            
        if (east-west) <= 0 or (north-south) <= 0:
            self._grid_label.setText("网格大小: —")
            return
            
        cols = max(1, math.ceil((east - west) / res_deg))
        rows = max(1, math.ceil((north - south) / res_deg))
        self._grid_label.setText(f"网格大小: {cols} 列 × {rows} 行 = {cols*rows:,} 个单元")

    def _get_input_files(self):
        """从向导页 1 获取选中的输入文件"""
        wiz = self.wizard()
        if wiz and hasattr(wiz, '_page1'):
            return wiz._page1.get_files()
        return []

    def _auto_fill_bounds(self):
        """从 XSF 文件读取导航坐标并自动填入地理边界"""
        files = self._get_input_files()
        if not files or not hasattr(self, '_coord_spins'):
            return
        
        from pyat.function.evaluate_sounder_geobox import GeoboxEvaluator
        import pyat.sounder.sounder_driver_factory as sounder_driver_factory
        
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QCursor
            self.setCursor(QCursor(Qt.WaitCursor))
            
            with sounder_driver_factory.open_sounder(files[0]) as driver:
                nav_point = int(driver.sounder_file.swath_count / 2)
                lons = driver.read_platform_longitudes()
                lats = driver.read_platform_latitudes()
                self._center_lon = float(lons.flat[nav_point])
                self._center_lat = float(lats.flat[nav_point])
            
            evaluator = GeoboxEvaluator(files)
            res_dict = evaluator()
            if res_dict and "result" in res_dict:
                box = res_dict["result"]
                self._coord_spins["西"].setValue(box["left"])
                self._coord_spins["东"].setValue(box["right"])
                self._coord_spins["南"].setValue(box["bottom"])
                self._coord_spins["北"].setValue(box["top"])
                self._update_grid_size()
        except Exception as e:
            print("从文件填充边界出错:", e)
        finally:
            self.unsetCursor()

    def get_params(self):
        result = {}
        for k, w in self._param_widgets.items():
            if isinstance(w, QCheckBox): result[k] = w.isChecked()
            elif isinstance(w, QComboBox): result[k] = w.currentData() or w.currentText()
            elif isinstance(w, QSpinBox): result[k] = w.value()
            elif isinstance(w, QDoubleSpinBox): result[k] = w.value()
            elif hasattr(w, 'layout'):
                sp = w.layout().itemAt(0).widget() if w.layout() else None
                if isinstance(sp, QSpinBox): result[k] = sp.value()
                elif isinstance(sp, QDoubleSpinBox): result[k] = sp.value()
        if hasattr(self, '_coord_spins'):
            s = self._coord_spins
            result["coord"] = {"west": s["西"].value(), "south": s["南"].value(),
                               "east": s["东"].value(), "north": s["北"].value()}
        if hasattr(self, '_res_spin'):
            val = self._res_spin.value()
            unit_idx = self._res_unit_combo.currentIndex()
            if unit_idx == 1: # arcminute
                res_deg = val / 60.0
            elif unit_idx == 2: # arcsecond
                res_deg = val / 3600.0
            else:
                res_deg = val
            result["target_resolution"] = res_deg
        return result


# ── Page 3: 输出选项 ──

class PageOutput(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("步骤 3: 输出选项")
        self.setSubTitle("配置输出图层、归一化参考水平和滤波参数。")

        layout = QFormLayout(self)

        self._layer_checks = {}
        layer_w = QWidget(); layer_layout = QVBoxLayout(layer_w); layer_layout.setContentsMargins(0,0,0,0)
        for val, label in LAYER_CHOICES:
            cb = QCheckBox(label); cb.setChecked(val == "backscatter_mean"); layer_layout.addWidget(cb)
            self._layer_checks[val] = cb
        layout.addRow("输出图层:", layer_w)

        self._norm_offset = QDoubleSpinBox(); self._norm_offset.setRange(-100,100); self._norm_offset.setValue(0); self._norm_offset.setSuffix(" dB")
        layout.addRow("归一化参考水平:", self._norm_offset)

        filter_w = QWidget(); filter_row = QHBoxLayout(filter_w); filter_row.setContentsMargins(0,0,0,0)
        self._filter_path = QLineEdit(); self._filter_path.setPlaceholderText("点击配置按钮设置滤波参数")
        btn_filter = QPushButton("配置...")
        btn_filter.clicked.connect(self._open_filter_dialog)
        filter_row.addWidget(self._filter_path); filter_row.addWidget(btn_filter)
        layout.addRow("滤波配置:", filter_w)

    def _open_filter_dialog(self):
        from ..views.wc_panel import WcFilterDialog
        dlg = WcFilterDialog(parent=self)
        if dlg.exec() == QWizard.Accepted:
            config = dlg.get_config()
            tmp = os.path.join(tempfile.gettempdir(), f"wc_filt_{hash(str(config))&0xFFFFFFFF:x}.json")
            with open(tmp,"w",encoding="utf-8") as f: json.dump(config,f,indent=2,ensure_ascii=False)
            self._filter_path.setText(tmp)

    def get_layers(self): return [k for k,cb in self._layer_checks.items() if cb.isChecked()]
    def get_normalization_offset(self): return self._norm_offset.value()
    def get_filter_path(self): return self._filter_path.text()


# ── Page 4: 汇总 ──

class PageSummary(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("步骤 4: 汇总")
        self.setSubTitle("请确认以下配置，无误后点击「运行」。")
        self._text = QTextEdit(); self._text.setReadOnly(True)
        layout = QVBoxLayout(self); layout.addWidget(QLabel("配置摘要:")); layout.addWidget(self._text)

    def set_summary(self, text: str): self._text.setText(text)


# ── 主向导 ──

class WcWizard(QWizard):
    def __init__(self, mode: str = "horizontal", parent=None):
        config = MODE_CONFIG.get(mode)
        if config is None: raise ValueError(f"未知水柱模式: {mode}")
        self._mode = mode
        self._config = config
        super().__init__(parent)
        self.setWindowTitle(config["title"])
        self.setButtonText(QWizard.NextButton, "下一步 >")
        self.setButtonText(QWizard.BackButton, "< 上一步")
        self.setButtonText(QWizard.CancelButton, "取消")
        self.setButtonText(QWizard.FinishButton, "完成")
        self.setMinimumSize(580, 540)
        self.setWizardStyle(QWizard.ModernStyle)
        self._page1 = PageInputOutput(self)
        self._page2 = PageParams(self)
        self._page3 = PageOutput(self)
        self._page4 = PageSummary(self)
        self.addPage(self._page1); self.addPage(self._page2); self.addPage(self._page3); self.addPage(self._page4)
        self.currentIdChanged.connect(self._on_page_changed)

    def _on_page_changed(self, page_id: int):
        if page_id == 3:
            self._page4.set_summary(self._build_summary())

    def _build_summary(self) -> str:
        cfg = self._config; files = self._page1.get_files()
        params = self._page2.get_params(); layers = self._page3.get_layers()
        norm = self._page3.get_normalization_offset(); filt = self._page3.get_filter_path()
        lines = [f"模式: {cfg.get('title', self._mode)}", ""
                 f"输入文件: {len(files)} 个"]
        if self._page1.get_output_dir():
            lines.append(f"输出目录: {self._page1.get_output_dir()}")
        lines.append(f"覆盖已有: {'是' if self._page1.get_overwrite() else '否'}")
        lines.append(""); lines.append("=== 参数 ===")
        for k, v in params.items():
            if k == "coord":
                lines.append(f"  边界: 西={v['west']:.4f} 南={v['south']:.4f} 东={v['east']:.4f} 北={v['north']:.4f}")
            elif isinstance(v, bool): lines.append(f"  {k}: {'是' if v else '否'}")
            else: lines.append(f"  {k}: {v}")
        lines.append(""); lines.append("=== 输出 ===")
        lines.append(f"  图层: {', '.join(layers)}"); lines.append(f"  归一化参考水平: {norm} dB")
        lines.append(f"  滤波配置: {'有' if filt else '无'}")
        return "\n".join(lines)

    def get_all_params(self) -> dict:
        params = self._page2.get_params()
        r = {"mode": self._mode, "input_files": self._page1.get_files(),
             "output_dir": self._page1.get_output_dir(), "overwrite": self._page1.get_overwrite(),
             "layers": self._page3.get_layers(),
             "normalization_offset": self._page3.get_normalization_offset(),
             "filters": self._page3.get_filter_path(),
             "gws_config": self._config.get("gws_config",""), "suffix": self._config.get("suffix","")}
        for k, v in params.items():
            if k not in ("coord","target_resolution"): r[k] = v
        if "coord" in params: r["coord"] = params["coord"]
        if "target_resolution" in params: r["target_resolution"] = params["target_resolution"]
        return r
