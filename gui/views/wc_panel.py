"""水柱分析面板 — 左工具树 + 右参数表单 + 3D 视图。

三栏布局：
  左侧: QTreeWidget（工具选择树）
  中间: Wc3dView（QWebEngineView + Three.js）
  右侧: QScrollArea（动态参数表单 + 运行按钮）

所有用户可见文本均使用中文显示，后端参数键名保持英文。
"""

import json, os, tempfile
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QSplitter, QTreeWidget, QTreeWidgetItem,
    QScrollArea, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QPushButton, QFileDialog, QGroupBox,
    QMessageBox, QDialog, QDialogButtonBox,
)

# ── 4 工具参数配置（全中文标签，键名保持英文以匹配后端） ──

WC_TOOLS = {
    "horizontal": {
        "title": "水平切片 (Horizontal Slicer)",
        "overview": "水平切片工具将水柱数据按指定深度层进行水平剖切，生成包含水体后向散射强度空间分布的 g3d 网格文件。",
        "gws_config": "sonar/wc/horizontal_section.json",
        "output_suffix": "WCHorizontalEcho.g3d.nc",
        "groups": [
            {"title": "输入 / 输出", "fields": [
                {"key": "i_paths", "label": "输入文件", "desc": "XSF 文件列表 (.xsf.nc)",
                 "type": "infile", "required": True},
                {"key": "o_paths", "label": "输出目录", "desc": "输出文件存放位置",
                 "type": "outdir", "required": True},
                {"key": "overwrite", "label": "覆盖已有", "desc": "允许覆盖已存在的输出文件",
                 "type": "bool", "default": True},
            ]},
            {"title": "切片参数", "fields": [
                {"key": "delta_elevation","label": "垂直层间距","desc": "相邻切片的垂直距离 (m)",
                 "type": "float", "default": 0.0, "min": 0, "hint": "0 = 自动计算"},
                {"key": "grid_count","label": "网格数量","desc": "切片总数 (0 = 自动)",
                 "type": "int", "default": 0, "min": 0, "hint": "0 = 自动"},
                {"key": "vertical_offset","label": "垂直偏移","desc": "整体上移/下移切片 (m)",
                 "type": "float", "default": 0.0},
                {"key": "vertical_reference","label": "垂向参考","desc": "深度坐标基准面",
                 "type": "enum", "choices": ["chart_datum","sea_floor"],
                 "choice_labels": {"chart_datum":"海图基准面","sea_floor":"海底面"},
                 "default": "chart_datum"},
            ]},
            {"title": "地理边界", "fields": [
                {"key": "coord","label": "边界范围","desc": "西/南/东/北 (十进制度)",
                 "type": "geobox", "default": None},
                {"key": "target_resolution","label": "空间分辨率","desc": "输出网格分辨率 (0 = 自动)",
                 "type": "float", "default": 0.00027778, "hint": "0 = 自动"},
            ]},
            {"title": "输出选项", "fields": [
                {"key": "layers","label": "输出图层","desc": "选择后向散射图层",
                 "type": "layers_checkbox"},
                {"key": "normalization_offset","label": "归一化参考水平","desc": "按距离归一化参考值 (dB)",
                 "type": "float", "default": 0.0},
                {"key": "filters","label": "滤波配置 (JSON)","desc": "可选水柱数据滤波文件",
                 "type": "json_file"},
            ]},
        ],
    },
    "longitudinal": {
        "title": "纵向剖面 (Longitudinal Slicer)",
        "overview": "纵向剖面工具沿船舶航迹方向生成水柱数据垂直剖面，将水体后向散射按深度和航迹距离展开为二维声学图像。",
        "gws_config": "sonar/wc/longitudinal_section.json",
        "output_suffix": "WCLongitudinalEcho.g3d.nc",
        "groups": [
            {"title": "输入 / 输出", "fields": [
                {"key": "i_paths","label": "输入文件","desc": "XSF 文件列表 (.xsf.nc)",
                 "type": "infile", "required": True},
                {"key": "o_paths","label": "输出目录","desc": "输出文件存放位置",
                 "type": "outdir", "required": True},
                {"key": "overwrite","label": "覆盖已有","desc": "允许覆盖已存在的输出文件",
                 "type": "bool", "default": True},
            ]},
            {"title": "剖面参数", "fields": [
                {"key": "delta_elevation","label": "垂直采样间距","desc": "深度方向采样间距 (m)",
                 "type": "float", "default": 0.0, "min": 0, "hint": "0 = 自动"},
                {"key": "delta_across","label": "跨测线间距","desc": "相邻剖面水平距离 (m)",
                 "type": "float", "default": 0.0, "min": 0, "hint": "0 = 自动"},
                {"key": "grid_count","label": "网格数量","desc": "剖面总数 (替代跨测线间距, 0 = 自动)",
                 "type": "int", "default": 0, "min": 0, "hint": "0 = 自动"},
                {"key": "delta_along","label": "沿测线间距","desc": "沿航迹采样间距 (m)",
                 "type": "float", "default": 0.0, "min": 0, "hint": "0 = 自动"},
                {"key": "interpolate","label": "线性插值","desc": "用线性插值填补数据空隙",
                 "type": "bool", "default": False},
            ]},
            {"title": "输出选项", "fields": [
                {"key": "layers","label": "输出图层","desc": "选择后向散射图层",
                 "type": "layers_checkbox"},
                {"key": "normalization_offset","label": "归一化参考水平","desc": "按距离归一化参考值 (dB)",
                 "type": "float", "default": 0.0},
                {"key": "filters","label": "滤波配置 (JSON)","desc": "可选水柱数据滤波文件",
                 "type": "json_file"},
            ]},
        ],
    },
    "polar": {
        "title": "极坐标声图 (Polar Echograms)",
        "overview": "极坐标声图工具将单 Ping 水柱回波数据按波束角和斜距展开为极坐标图像，呈现水体全角度散射结构。",
        "gws_config": "sonar/wc/polar_echograms.json",
        "output_suffix": "PolarEchograms.g3d.nc",
        "groups": [
            {"title": "输入 / 输出", "fields": [
                {"key": "i_paths","label": "输入文件","desc": "XSF 文件列表 (.xsf.nc)",
                 "type": "infile", "required": True},
                {"key": "o_paths","label": "输出目录","desc": "输出文件存放位置",
                 "type": "outdir", "required": True},
                {"key": "overwrite","label": "覆盖已有","desc": "允许覆盖已存在输出文件",
                 "type": "bool", "default": True},
            ]},
            {"title": "声图参数", "fields": [
                {"key": "sample_resolution","label": "采样分辨率","desc": "回波采样间距 (m, 0=自动)",
                 "type": "float", "default": 0.0, "min": 0, "hint": "0 = 自动"},
                {"key": "height","label": "图像高度","desc": "用于计算默认分辨率的像素高度",
                 "type": "int", "default": 500, "min": 100, "max": 2000, "hint": "0 = 自动"},
                {"key": "interpolate","label": "线性插值","desc": "用线性插值填补数据空隙",
                 "type": "bool", "default": True},
            ]},
            {"title": "输出选项", "fields": [
                {"key": "layers","label": "输出图层","desc": "选择后向散射图层",
                 "type": "layers_checkbox"},
                {"key": "normalization_offset","label": "归一化参考水平","desc": "按距离归一化参考值 (dB)",
                 "type": "float", "default": 0.0},
                {"key": "filters","label": "滤波配置 (JSON)","desc": "可选水柱数据滤波文件",
                 "type": "json_file"},
            ]},
        ],
    },
    "vertical": {
        "title": "垂直积分 (Vertical Integration)",
        "overview": "垂直积分工具将整个水柱的后向散射能量沿深度方向积分投影到水平面，生成类似后向散射镶嵌图的栅格图像。",
        "gws_config": "sonar/wc/vertical_integration.json",
        "output_suffix": "WCVerticalEcho.tiff",
        "groups": [
            {"title": "输入 / 输出", "fields": [
                {"key": "i_paths","label": "输入文件","desc": "XSF 文件列表 (.xsf.nc)",
                 "type": "infile", "required": True},
                {"key": "o_paths","label": "输出目录","desc": "输出文件存放位置",
                 "type": "outdir", "required": True},
                {"key": "overwrite","label": "覆盖已有","desc": "允许覆盖已存在输出文件",
                 "type": "bool", "default": True},
            ]},
            {"title": "地理边界", "fields": [
                {"key": "coord","label": "边界范围","desc": "西/南/东/北 (十进制度)",
                 "type": "geobox", "default": None},
                {"key": "target_resolution","label": "空间分辨率","desc": "网格分辨率（0 = 自动估算）",
                 "type": "float", "default": 0.00027778, "hint": "0 = 自动"},
            ]},
            {"title": "积分选项", "fields": [
                {"key": "enable_normalization","label": "距离归一化","desc": "启用斜距归一化积分补偿，勾选后以下方参考水平进行能量补偿",
                 "type": "bool", "default": False},
                {"key": "normalization_offset","label": "归一化参考水平","desc": "归一化参考基准 (dB)",
                 "type": "float", "default": 0.0},
                {"key": "filters","label": "滤波配置 (JSON)","desc": "可选水柱数据滤波文件",
                 "type": "json_file"},
            ]},
        ],
    },
}


class WcPanel(QWidget):
    """水柱分析面板 —— 左树 + 右表单，含 3D 视图占位。"""

    # 当用户填写完参数点击运行后，发射带参数和工具名的信号
    run_requested = object  # will be set by main_window

    def __init__(self, wc3d_view: QWidget = None, parent=None):
        super().__init__(parent)
        self._wc3d_view = wc3d_view
        self._active_tool: Optional[str] = None
        self._form_widgets: dict = {}
        self._selected_files: list = []

        self._setup_ui()

        # 默认选中第一个工具
        first_tool = list(WC_TOOLS.keys())[0]
        self._select_tool(first_tool)

    def set_selected_files(self, paths: list):
        self._selected_files = paths
        # 更新输入文件显示
        infile_widget = self._form_widgets.get("i_paths")
        if infile_widget and hasattr(infile_widget, "layout"):
            edit = infile_widget.layout().itemAt(0).widget()
            if isinstance(edit, QLineEdit):
                edit.setText(f"已选 {len(paths)} 个文件")
                edit.setToolTip("\n".join(paths[:10]))

    # ── UI 构建 ──

    def _setup_ui(self):
        splitter = QSplitter(Qt.Horizontal, self)

        # 左: 工具树
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setMinimumWidth(200)
        self._tree.setMaximumWidth(280)
        self._tree.currentItemChanged.connect(self._on_tool_selected)
        self._populate_tree()
        splitter.addWidget(self._tree)

        # 中: 3D 视图 (由外部注入)
        if self._wc3d_view:
            splitter.addWidget(self._wc3d_view)

        # 右: 参数面板
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._title_label = QLabel("水柱分析")
        self._title_label.setStyleSheet(
            "font-size:15px; font-weight:bold; padding:8px 12px;")
        right_layout.addWidget(self._title_label)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumWidth(350)
        right_layout.addWidget(self._scroll, stretch=1)

        # 运行按钮
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(8, 6, 8, 6)
        btn_layout.addStretch()
        self._btn_run = QPushButton("运行...")
        self._btn_run.setMinimumHeight(34)
        self._btn_run.setMinimumWidth(100)
        self._btn_run.setStyleSheet(
            "QPushButton { background-color:#0e639c; color:white; font-weight:bold; "
            "border:none; border-radius:3px; } "
            "QPushButton:hover { background-color:#1177bb; }")
        self._btn_run.clicked.connect(self._on_run)
        btn_layout.addWidget(self._btn_run)
        right_layout.addWidget(btn_row)

        splitter.addWidget(right)
        splitter.setSizes([220, 500, 380])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def _populate_tree(self):
        root = QTreeWidgetItem(self._tree, ["工具箱"])
        root.setExpanded(True)
        wc = QTreeWidgetItem(root, ["水柱分析"])
        wc.setExpanded(True)
        tools = QTreeWidgetItem(wc, ["工具"])
        tools.setExpanded(True)
        for key, cfg in WC_TOOLS.items():
            item = QTreeWidgetItem(tools, [cfg["title"]])
            item.setData(0, Qt.UserRole, key)
        self._tree.expandAll()

    # ── 工具选择 → 动态表单 ──

    def _select_tool(self, key: str):
        """程序化选中工具 (用于默认选中首项)"""
        for i in range(self._tree.topLevelItemCount()):
            root = self._tree.topLevelItem(i)
            for j in range(root.childCount()):
                child = root.child(j)
                for k in range(child.childCount()):
                    sub = child.child(k)
                    for m in range(sub.childCount()):
                        item = sub.child(m)
                        if item.data(0, Qt.UserRole) == key:
                            self._tree.setCurrentItem(item)
                            return

    def _on_tool_selected(self, current, previous):
        key = current.data(0, Qt.UserRole) if current else None
        if key and key in WC_TOOLS:
            self._render_form(key)
            # 通知 3D 视图切换图层
            if self._wc3d_view and hasattr(self._wc3d_view, 'bridge'):
                self._wc3d_view.bridge.switch_layer(key)

    def _render_form(self, tool_key: str):
        cfg = WC_TOOLS[tool_key]
        self._active_tool = tool_key
        self._title_label.setText(cfg["title"])
        self._form_widgets.clear()
        self._grid_size_label = None  # reset

        form = QWidget()
        layout = QVBoxLayout(form)

        # Overview
        ov = QLabel(cfg["overview"])
        ov.setWordWrap(True)
        ov.setStyleSheet("color:#808080; padding:8px 12px; font-size:12px;")
        layout.addWidget(ov)

        # Parameter groups
        for group in cfg["groups"]:
            grp = QGroupBox(group["title"])
            fl = QFormLayout(grp)
            fl.setSpacing(6)
            for field in group["fields"]:
                w = self._create_widget(field)
                label = QLabel(field["label"])
                label.setToolTip(field.get("desc", ""))
                fl.addRow(label, w)
                self._form_widgets[field["key"]] = w
            layout.addWidget(grp)

            # After rendering geographic bounds group, add grid size display
            if group["title"] == "地理边界":
                # 分辨率转换显示（从复合控件中提取 QDoubleSpinBox）
                res_w = self._form_widgets.get("target_resolution")
                if res_w and res_w.layout():
                    inner = res_w.layout().itemAt(0)
                    if inner and inner.widget() and hasattr(inner.widget(), 'valueChanged'):
                        inner.widget().valueChanged.connect(self._update_resolution_display)
                # 网格大小显示
                self._grid_size_label = QLabel("网格大小: —")
                self._grid_size_label.setStyleSheet("color:#569cd6;font-weight:bold;padding:2px 8px;")
                fl.addRow(self._grid_size_label)
                # 展开到整数弧分选项
                self._expand_minute = QCheckBox("展开到整数弧分网格")
                self._expand_minute.toggled.connect(self._update_grid_size)
                fl.addRow(self._expand_minute)

        layout.addStretch()
        self._scroll.setWidget(form)

    # ── 网格大小与分辨率显示 ──

    def _get_spinbox(self, key: str):
        """从表单控件字典中提取 QDoubleSpinBox（可能是复合控件）"""
        w = self._form_widgets.get(key)
        if not w:
            return None
        if hasattr(w, 'valueChanged'):  # 直接就是 SpinBox
            return w
        if w.layout():  # 复合控件，取第一个子控件
            inner = w.layout().itemAt(0)
            if inner and inner.widget() and hasattr(inner.widget(), 'valueChanged'):
                return inner.widget()
        return None

    def _update_resolution_display(self):
        """分辨率转换显示：度 ↔ 弧分 ↔ 弧秒 ↔ 米"""
        sp = self._get_spinbox("target_resolution")
        if not sp:
            return
        res_deg = sp.value()
        arcmin = res_deg * 60
        arcsec = res_deg * 3600
        sp.setToolTip(
            f"{res_deg:.6f}° ≈ {arcmin:.3f}′ ≈ {arcsec:.2f}″")
        self._update_grid_size()

    def _update_grid_size(self):
        """根据地理边界和分辨率计算网格大小"""
        b = self._form_widgets.get("coord")
        r = self._get_spinbox("target_resolution")
        if not b or not r or not hasattr(b, '_geobox'):
            return
        s = b._geobox
        west, east = s["西"].value(), s["东"].value()
        south, north = s["南"].value(), s["北"].value()
        res = r.value()
        if res <= 0:
            return
        dx = east - west
        dy = north - south
        import math
        cols = math.ceil(dx / res)
        rows = math.ceil(dy / res)
        if self._expand_minute and self._expand_minute.isChecked():
            cols = max(1, int(dx / res))
            rows = max(1, int(dy / res))
        cells = cols * rows
        if self._grid_size_label:
            self._grid_size_label.setText(
                f"网格大小: {cols} 列 × {rows} 行 = {cells:,} 个单元")
            self._grid_size_label.setStyleSheet(
                "color:#569cd6;font-weight:bold;padding:2px 8px;")

    # ── 控件工厂 ──

    def _create_widget(self, field: dict) -> QWidget:
        ftype = field["type"]
        default = field.get("default")
        key = field["key"]

        # infile — 只读标签
        if ftype == "infile":
            w = QWidget()
            row = QHBoxLayout(w); row.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit(); edit.setReadOnly(True)
            edit.setPlaceholderText("从项目列表传入")
            btn = QPushButton("浏览...")
            btn.clicked.connect(lambda: self._browse_xsf(edit))
            row.addWidget(edit); row.addWidget(btn)
            return w

        # outdir — 目录选择
        if ftype == "outdir":
            w = QWidget()
            row = QHBoxLayout(w); row.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit()
            edit.setPlaceholderText("留空则使用输入文件所在目录")
            btn = QPushButton("浏览...")
            btn.clicked.connect(lambda: self._browse_dir(edit))
            row.addWidget(edit); row.addWidget(btn)
            return w

        # bool — 复选框
        if ftype == "bool":
            cb = QCheckBox()
            cb.setChecked(bool(default))
            return cb

        # enum — 下拉框 (中文标签)
        if ftype == "enum":
            cb = QComboBox()
            labels = field.get("choice_labels", {})
            for val in field.get("choices", []):
                cb.addItem(labels.get(val, val), val)
            if default:
                idx = cb.findData(str(default))
                if idx >= 0:
                    cb.setCurrentIndex(idx)
            return cb

        # float — 浮点输入（含自动计算按钮）
        if ftype == "float":
            w = QWidget()
            row = QHBoxLayout(w); row.setContentsMargins(0, 0, 0, 0)
            sp = QDoubleSpinBox()
            sp.setRange(field.get("min", -1e9), field.get("max", 1e9))
            sp.setDecimals(6)
            sp.setValue(float(default or 0))
            if hint := field.get("hint"):
                sp.setToolTip(hint)
            row.addWidget(sp, stretch=1)
            # 自动计算按钮
            if "自动" in field.get("hint", ""):
                btn = QPushButton("计")
                btn.setMaximumWidth(22)
                btn.setMaximumHeight(22)
                btn.setToolTip("点击设为 0（自动计算）")
                btn.clicked.connect(lambda checked, s=sp: s.setValue(0.0))
                row.addWidget(btn)
            return w

        # int — 整数输入（含自动计算按钮）
        if ftype == "int":
            w = QWidget()
            row = QHBoxLayout(w); row.setContentsMargins(0, 0, 0, 0)
            sp = QSpinBox()
            sp.setRange(field.get("min", 0), field.get("max", 10 ** 9))
            sp.setValue(int(default or 0))
            if hint := field.get("hint"):
                sp.setToolTip(hint)
            row.addWidget(sp, stretch=1)
            if "自动" in field.get("hint", ""):
                btn = QPushButton("计")
                btn.setMaximumWidth(22)
                btn.setMaximumHeight(22)
                btn.setToolTip("点击设为 0（自动计算）")
                btn.clicked.connect(lambda checked, s=sp: s.setValue(0))
                row.addWidget(btn)
            return w

        # geobox — 西/南/东/北 四象限
        if ftype == "geobox":
            w = QWidget()
            grid = QVBoxLayout(w); grid.setContentsMargins(0, 0, 0, 0)
            spins = {}
            for pos, lo, hi in [("西", -180, 180), ("东", -180, 180),
                                ("南", -90, 90), ("北", -90, 90)]:
                row = QHBoxLayout(); row.addWidget(QLabel(pos))
                sp = QDoubleSpinBox(); sp.setRange(lo, hi); sp.setDecimals(5)
                sp.valueChanged.connect(self._update_grid_size)
                row.addWidget(sp)
                grid.addLayout(row)
                spins[pos] = sp
            w._geobox = spins
            return w

        # layers_checkbox — 4 个复选框
        if ftype == "layers_checkbox":
            w = QWidget()
            row = QHBoxLayout(w); row.setContentsMargins(0, 0, 0, 0)
            checks = {}
            layer_labels = {
                "backscatter_mean": "均值",
                "backscatter_max": "最大值",
                "backscatter_comp_mean": "补偿均值",
                "backscatter_comp_max": "补偿最大值",
            }
            for name in ["backscatter_mean", "backscatter_max",
                         "backscatter_comp_mean", "backscatter_comp_max"]:
                cb = QCheckBox(layer_labels.get(name, name))
                cb.setChecked(name == "backscatter_mean")
                row.addWidget(cb)
                checks[name] = cb
            w._layer_checks = checks
            return w

        # json_file — 文本 + 浏览 + 配置按钮
        if ftype == "json_file":
            w = QWidget()
            row = QHBoxLayout(w); row.setContentsMargins(0, 0, 0, 0)
            edit = QLineEdit()
            edit.setPlaceholderText("滤波配置会自动保存到文件")
            btn_config = QPushButton("配置...")
            btn_config.setToolTip("打开滤波参数可视化配置界面")
            btn_config.clicked.connect(lambda: self._open_filter_dialog(edit))
            row.addWidget(edit); row.addWidget(btn_config)
            return w

        return QLabel(f"未知类型: {ftype}")

    # ── 值收集 ──

    def _collect_params(self) -> dict:
        result = {}
        for key, w in self._form_widgets.items():
            if isinstance(w, QCheckBox):
                result[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                result[key] = w.currentData() or w.currentText()
            elif isinstance(w, QSpinBox):
                result[key] = w.value()
            elif isinstance(w, QDoubleSpinBox):
                result[key] = w.value()
            elif isinstance(w, QLineEdit):
                result[key] = w.text()
            elif hasattr(w, '_geobox'):
                s = w._geobox
                result[key] = {
                    "west": s["西"].value(), "south": s["南"].value(),
                    "east": s["东"].value(), "north": s["北"].value(),
                }
            elif hasattr(w, '_layer_checks'):
                result[key] = [
                    k for k, cb in w._layer_checks.items() if cb.isChecked()
                ]
            elif isinstance(w, QWidget) and w.layout():
                inner = w.layout().itemAt(0)
                if inner and inner.widget():
                    inner_w = inner.widget()
                    if isinstance(inner_w, QLineEdit):
                        result[key] = inner_w.text()
        return result

    def get_params(self) -> dict:
        """对外暴露：返回当前工具键 + 收集的全部参数。
        返回格式: {"mode": "longitudinal", "params": {...}}
        """
        return {
            "mode": self._active_tool,
            "input_files": self._selected_files,
            "params": self._collect_params(),
        }

    # ── 运行按钮 ──

    def _on_run(self):
        if not self._active_tool:
            QMessageBox.warning(self, "提示", "请先在左侧树中选择一个水柱分析工具。")
            return

        # 如果尚未选择文件，弹出文件选择对话框
        if not self._selected_files:
            reply = QMessageBox.question(
                self, "选择文件",
                "尚未选择 XSF 文件。\n是否立即选择文件？",
                QMessageBox.Yes | QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择 XSF 文件", "",
                "XSF 文件 (*.xsf.nc *.nc);;所有文件 (*.*)")
            if not files:
                return
            self.set_selected_files(files)

        data = self.get_params()
        # 构建 JSON 并交给 ProcessManager 执行
        try:
            from ..core.json_builder import build_wc_json
            config_path = build_wc_json(
                mode=data["mode"],
                input_files=data["input_files"],
                output_dir="",
                **data["params"],
            )
            print(f"[WC] Config written to: {config_path}")
            # 通知主窗口执行
            main_win = self.parent()
            if main_win and hasattr(main_win, '_task_manager'):
                task_label = f"工具 WC: {WC_TOOLS[self._active_tool]['title']}"
                from ..core.task_manager import TaskStatus
                task = main_win._task_manager.create_task(task_label, config_path)
                main_win._current_task_id = task.task_id
                main_win._task_manager.update_status(task.task_id, TaskStatus.QUEUED)
                main_win._process_manager.run(config_path)
            else:
                print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        except Exception as e:
            QMessageBox.critical(self, "运行失败", f"无法启动后端进程：\n{e}")

    # ── 文件浏览 ──

    def _browse_xsf(self, edit=None):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择 XSF 文件", "",
            "XSF 文件 (*.xsf.nc *.nc);;所有文件 (*.*)")
        if files:
            self.set_selected_files(files)
            if edit:
                edit.setText(f"已选 {len(files)} 个文件")

    def _browse_dir(self, edit):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            edit.setText(d)

    def _open_filter_dialog(self, edit):
        """打开可视化滤波参数配置对话框"""
        # 读取已有配置作为初始值
        initial = None
        if edit.text():
            try:
                with open(edit.text()) as f:
                    initial = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        dlg = WcFilterDialog(initial, self)
        if dlg.exec() == QDialog.Accepted:
            config = dlg.get_config()
            tmp_path = os.path.join(tempfile.gettempdir(),
                f"wc_filters_{hash(str(config)) & 0xFFFFFFFF:x}.json")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            edit.setText(tmp_path)


class WcFilterDialog(QDialog):
    """水柱数据滤波参数可视化配置对话框"""

    def __init__(self, initial: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WC 滤波参数配置")
        self.setMinimumSize(600, 500)
        self._initial = initial or {}
        self._widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QVBoxLayout(content)

        # ── 子采样 ──
        grp = QGroupBox("子采样 (Subsampling)")
        gf = QFormLayout(grp)
        self._add_bool("sampling_enable", "启用子采样", gf)
        self._add_spin("sampling_sampling", "保留间隔", 1, 1000, 75, gf)
        form.addWidget(grp)

        # ── 阈值 ──
        grp = QGroupBox("阈值 (Threshold)")
        gf = QFormLayout(grp)
        self._add_bool("threshold_enable", "启用阈值滤波", gf)
        self._add_float("threshold_minValue", "最小值 (dB)", -100, 10, -75.0, gf)
        self._add_float("threshold_maxValue", "最大值 (dB)", -100, 100, 20.0, gf)
        form.addWidget(grp)

        # ── 镜面反射 ──
        grp = QGroupBox("镜面反射 (Specular)")
        gf = QFormLayout(grp)
        self._add_bool("specular_enable", "启用镜面反射滤波", gf)
        self._add_bool("specular_below", "移除镜面下方数据", gf)
        self._add_float("specular_tolerance", "容差", 0, 200, 75.0, gf)
        form.addWidget(grp)

        # ── 海底检测 ──
        grp = QGroupBox("海底检测 (Bottom detection)")
        gf = QFormLayout(grp)
        self._add_bool("bottom_enable", "启用海底滤波", gf)
        # 容差类型
        type_row = QHBoxLayout()
        self._type_combo = QComboBox()
        self._type_combo.addItem("距离百分比", "RANGEPERCENT")
        self._type_combo.addItem("采样数", "SAMPLE")
        type_row.addWidget(QLabel("容差类型:"))
        type_row.addWidget(self._type_combo)
        gf.addRow(type_row)
        self._add_float("bottom_tolerancePercent", "容差百分比 (%)", 0, 100, 20.0, gf)
        self._add_float("bottom_toleranceAbsolute", "绝对容差", 0, 1000, 0.0, gf)
        self._add_float("bottom_angleCoefficient", "角度系数", 0, 5, 0.10, gf)
        form.addWidget(grp)

        # ── 旁瓣 ──
        grp = QGroupBox("旁瓣 (Side lobe)")
        gf = QFormLayout(grp)
        self._add_bool("sidelobe_enable", "启用旁瓣滤波", gf)
        self._add_float("sidelobe_threshold", "阈值 (dB)", 0, 100, 20.0, gf)
        form.addWidget(grp)

        # ── 波束索引 ──
        grp = QGroupBox("波束索引 (Beam index)")
        gf = QFormLayout(grp)
        self._add_bool("beam_enable", "启用波束索引滤波", gf)
        self._add_spin("beam_minValue", "最小索引", 0, 1023, 0, gf)
        self._add_spin("beam_maxValue", "最大索引", 0, 1023, 511, gf)
        form.addWidget(grp)

        # ── 采样索引 ──
        grp = QGroupBox("采样索引 (Sample index)")
        gf = QFormLayout(grp)
        self._add_bool("sample_enable", "启用采样索引滤波", gf)
        self._add_spin("sample_minValue", "最小索引", 0, 10000, 0, gf)
        self._add_spin("sample_maxValue", "最大索引", 0, 10000, 1000, gf)
        form.addWidget(grp)

        # ── 水深 ──
        grp = QGroupBox("水深 (Depth)")
        gf = QFormLayout(grp)
        self._add_bool("depth_enable", "启用水深滤波", gf)
        self._add_float("depth_minValue", "最小水深 (m)", -100, 12000, 0.0, gf)
        self._add_float("depth_maxValue", "最大水深 (m)", -100, 12000, 10000.0, gf)
        form.addWidget(grp)

        # ── 跨测线距离 ──
        grp = QGroupBox("跨测线距离 (Across distance)")
        gf = QFormLayout(grp)
        self._add_bool("acrossDistance_enable", "启用距离滤波", gf)
        self._add_float("acrossDistance_minValue", "最小距离 (m)", -10000, 10000, -5000.0, gf)
        self._add_float("acrossDistance_maxValue", "最大距离 (m)", -10000, 10000, 5000.0, gf)
        form.addWidget(grp)

        # ── 多脉冲序列 ──
        grp = QGroupBox("多脉冲序列 (Multiping sequence)")
        gf = QFormLayout(grp)
        self._add_bool("multiping_enable", "启用多脉冲滤波", gf)
        self._add_spin("multiping_index", "脉冲 ID", 0, 255, 0, gf)
        form.addWidget(grp)

        # 加载初始值
        self._load_initial()

        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        buttons.button(QDialogButtonBox.Reset).clicked.connect(self._reset_all)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_bool(self, key, label, form):
        cb = QCheckBox(label)
        form.addRow(cb)
        self._widgets[key] = cb

    def _add_float(self, key, label, lo, hi, default, form):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(2)
        sp.setValue(default)
        sp.setMinimumWidth(90)
        form.addRow(QLabel(label), sp)
        self._widgets[key] = sp

    def _add_spin(self, key, label, lo, hi, default, form):
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(default)
        sp.setMinimumWidth(90)
        form.addRow(QLabel(label), sp)
        self._widgets[key] = sp

    def _load_initial(self):
        ini = self._initial
        if "sampling" in ini:
            self._set("sampling_sampling", ini["sampling"].get("sampling"))
        for prefix in ("threshold", "specular", "sidelobe",
                        "acrossDistance", "beam", "sample", "depth"):
            if prefix in ini:
                p = ini[prefix]
                self._set(f"{prefix}_enable", p.get("enable"))
                self._set(f"{prefix}_minValue", p.get("minValue"))
                self._set(f"{prefix}_maxValue", p.get("maxValue"))
                if prefix == "specular":
                    self._set(f"{prefix}_below", p.get("below"))
                    self._set(f"{prefix}_tolerance", p.get("tolerance"))
                if prefix == "sidelobe":
                    self._set(f"{prefix}_threshold", p.get("threshold"))
        if "bottom" in ini:
            p = ini["bottom"]
            self._set("bottom_enable", p.get("enable"))
            self._set("bottom_tolerancePercent", p.get("tolerancePercent"))
            self._set("bottom_toleranceAbsolute", p.get("toleranceAbsolute"))
            self._set("bottom_angleCoefficient", p.get("angleCoefficient"))
            idx = self._type_combo.findData(p.get("type", "RANGEPERCENT"))
            if idx >= 0:
                self._type_combo.setCurrentIndex(idx)
        if "multiping" in ini:
            self._set("multiping_enable", ini["multiping"].get("enable"))
            self._set("multiping_index", ini["multiping"].get("index"))

    def _set(self, key, value):
        w = self._widgets.get(key)
        if w is None or value is None:
            return
        if isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.setValue(float(value) if isinstance(value, float) else int(value))

    def _reset_all(self):
        """重置所有滤波参数为关闭状态"""
        for key, w in self._widgets.items():
            if isinstance(w, QCheckBox):
                w.setChecked(False)

    def get_config(self) -> dict:
        """构建符合 backend apply_filters() 期望的 JSON 配置"""
        conf = {}

        # 子采样
        if self._v("sampling_enable"):
            conf["sampling"] = {"sampling": self._v("sampling_sampling")}

        # 阈值
        if self._v("threshold_enable"):
            conf["threshold"] = {
                "enable": True,
                "minValue": self._v("threshold_minValue"),
                "maxValue": self._v("threshold_maxValue"),
            }

        # 镜面反射
        if self._v("specular_enable"):
            conf["specular"] = {
                "enable": True,
                "below": self._v("specular_below"),
                "tolerance": self._v("specular_tolerance"),
            }

        # 海底检测
        if self._v("bottom_enable"):
            conf["bottom"] = {
                "enable": True,
                "tolerancePercent": self._v("bottom_tolerancePercent"),
                "toleranceAbsolute": self._v("bottom_toleranceAbsolute"),
                "angleCoefficient": self._v("bottom_angleCoefficient"),
                "type": self._type_combo.currentData(),
            }

        # 旁瓣
        if self._v("sidelobe_enable"):
            conf["sidelobe"] = {
                "enable": True,
                "threshold": self._v("sidelobe_threshold"),
            }

        # 波束索引
        if self._v("beam_enable"):
            conf["beam"] = {
                "enable": True,
                "minValue": self._v("beam_minValue"),
                "maxValue": self._v("beam_maxValue"),
            }

        # 采样索引
        if self._v("sample_enable"):
            conf["sample"] = {
                "enable": True,
                "minValue": self._v("sample_minValue"),
                "maxValue": self._v("sample_maxValue"),
            }

        # 水深
        if self._v("depth_enable"):
            conf["depth"] = {
                "enable": True,
                "minValue": self._v("depth_minValue"),
                "maxValue": self._v("depth_maxValue"),
            }

        # 跨测线距离
        if self._v("acrossDistance_enable"):
            conf["acrossDistance"] = {
                "enable": True,
                "minValue": self._v("acrossDistance_minValue"),
                "maxValue": self._v("acrossDistance_maxValue"),
            }

        # 多脉冲序列
        if self._v("multiping_enable"):
            conf["multiping"] = {
                "enable": True,
                "index": self._v("multiping_index"),
            }

        return conf

    def _v(self, key):
        w = self._widgets.get(key)
        if w is None:
            return None
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QSpinBox):
            return w.value()
        if isinstance(w, QDoubleSpinBox):
            return w.value()
        return None
