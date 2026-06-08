"""\u4e3b\u7a97\u53e3\uff0c\u91c7\u7528 Eclipse RCP / QGIS \u98ce\u683c\u5e03\u5c40\u3002

\u56db\u9762\u677f\u5e03\u5c40\uff1a
  - \u5de6\u4fa7\uff1a\u9879\u76ee\u6d4f\u89c8\u5668 + \u5de5\u5177\u7bb1\uff08QTabWidget \u5728 QDockWidget \u4e2d\uff09
  - \u5e95\u90e8\uff1a\u63a7\u5236\u53f0 + \u4efb\u52a1\u7ba1\u7406\u5668\uff08QDockWidget\uff09
  - \u4e2d\u95f4\uff1a\u4e3b\u753b\u5e03\uff08QTabWidget \u542b\u5730\u56fe\u89c6\u56fe + BSAR \u89c6\u56fe\uff09
"""
import json
import os
import tempfile
from datetime import datetime
from typing import Optional, List

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget, QWidget,
    QStatusBar, QMessageBox, QWizard, QFileDialog, QApplication,
)

from .project_explorer import ProjectExplorer
from .views.console_view import ConsoleView
from .views.map_view import MapView
from .views.bsar_viewer import BsarViewer
from .views.dtm_renderer import dtm_layer_to_file
from .core.task_manager import TaskManager, TaskStatus
from .core.process_manager import ProcessManager, ProcessState
from .core.json_builder import (
    build_tool1_json, build_tool2a_json, build_tool2b_step2a_json,
    build_tool2b_step2b_json, build_sounder_to_dtm_json, build_wc_json,
)
from .core.xsf_reader import XsfMetadata, read_xsf_metadata
from .dialogs.tool1_dialog import Tool1Dialog
from .dialogs.sounder_to_dtm_wizard import SounderToDtmWizard
from .dialogs.tool2a_dialog import Tool2ADialog
from .dialogs.tool2b_s1_dialog import Tool2BS1Dialog
from .dialogs.tool2b_s2_dialog import Tool2BS2Dialog
from .dialogs.bsar_tools_dialog import BsarToolsDialog
from .dialogs.wc_wizard import WcWizard
from .dialogs.dtm_export_dialog import DtmExportDialog
from .dialogs.dtm_xyz_export_dialog import DtmXyzExportDialog
from .views.wc2d_viewer import Wc2dViewer


class MainWindow(QMainWindow):
    """pyat GUI \u4e3b\u7a97\u53e3\u3002"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("pyat GUI — 多波束回声处理")
        self.resize(1400, 900)

        # ── 设置窗口图标 ──
        from PySide6.QtGui import QIcon
        import os
        _app_icon_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources", "svg", "app_logo.svg"))
        if os.path.exists(_app_icon_path):
            self.setWindowIcon(QIcon(_app_icon_path))

        # Core services
        self._task_manager = TaskManager(self)
        self._process_manager = ProcessManager(self)

        # Connect process manager signals
        self._process_manager.state_changed.connect(self._on_process_state)
        self._process_manager.progress_changed.connect(self._on_process_progress)
        self._process_manager.log_received.connect(self._on_process_log)
        self._process_manager.error_occurred.connect(self._on_process_error)
        self._process_manager.finished.connect(self._on_process_finished)

        # Current running task
        self._current_task_id: str = ""

        # Views
        self._project_explorer = ProjectExplorer(self)
        self._console_view = ConsoleView(self._task_manager, self)
        self._map_view = MapView(self)
        self._bsar_viewer = BsarViewer(self)
        self._wc2d_viewer = Wc2dViewer(self)

        # Mask storage
        self._current_kml_mask: str = ""

        # Setup
        self._setup_menu()
        self._setup_docks()
        self._setup_central()
        self._setup_statusbar()
        self._connect_signals()
        self._restore_state()

    def _setup_menu(self) -> None:
        menu_bar = self.menuBar()

        # ── File ──
        file_menu = menu_bar.addMenu("\u6587\u4ef6(&F)")
        add_action = QAction("\u6dfb\u52a0 XSF \u6587\u4ef6...", self)
        add_action.triggered.connect(self._project_explorer._add_files)
        file_menu.addAction(add_action)
        add_folder_action = QAction("\u6dfb\u52a0 XSF \u6587\u4ef6\u5939...", self)
        add_folder_action.triggered.connect(self._project_explorer._add_folder)
        file_menu.addAction(add_folder_action)

        file_menu.addSeparator()
        import_dtm = QAction("\u5bfc\u5165 DTM (.dtm.nc)...", self)
        import_dtm.triggered.connect(self._import_dtm)
        file_menu.addAction(import_dtm)

        import_g3d = QAction("打开水柱 g3d 文件...", self)
        import_g3d.triggered.connect(lambda: self._wc2d_viewer._open_file())
        file_menu.addAction(import_g3d)
        file_menu.addSeparator()

        exit_action = QAction("\u9000\u51fa(&X)", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ── Tools ──
        tools_menu = menu_bar.addMenu("\u5de5\u5177(&T)")

        # 后向散射处理
        t1 = QAction("导出参考 DTM...", self)
        t1.triggered.connect(lambda: self._dispatch_tool("tool1"))
        tools_menu.addAction(t1)
        tools_menu.addSeparator()

        t2as2 = QAction("统计角响应 (BSAR)...", self)
        t2as2.triggered.connect(lambda: self._dispatch_tool("tool2b_step2"))
        tools_menu.addAction(t2as2)
        t2as1 = QAction("静态角度重规范化...", self)
        t2as1.triggered.connect(lambda: self._dispatch_tool("tool2b_step1"))
        tools_menu.addAction(t2as1)
        t2a = QAction("滑动角度重规范化...", self)
        t2a.triggered.connect(lambda: self._dispatch_tool("tool2a"))
        tools_menu.addAction(t2a)
        tools_menu.addSeparator()

        # 水柱分析子菜单
        wc_menu = tools_menu.addMenu("水柱分析")

        wc_h = QAction("水平切片...", self)
        wc_h.triggered.connect(lambda: self._open_wc_wizard("horizontal"))
        wc_menu.addAction(wc_h)
        wc_l = QAction("纵向剖面...", self)
        wc_l.triggered.connect(lambda: self._open_wc_wizard("longitudinal"))
        wc_menu.addAction(wc_l)
        wc_p = QAction("极坐标声图...", self)
        wc_p.triggered.connect(lambda: self._open_wc_wizard("polar"))
        wc_menu.addAction(wc_p)
        wc_v = QAction("垂直积分...", self)
        wc_v.triggered.connect(lambda: self._open_wc_wizard("vertical"))
        wc_menu.addAction(wc_v)
        tools_menu.addSeparator()

        bsar_aux = QAction("BSAR 辅助工具...", self)
        bsar_aux.triggered.connect(self._open_bsar_tools)
        tools_menu.addAction(bsar_aux)
        refresh_status = QAction("刷新 XSF 状态", self)
        refresh_status.triggered.connect(self._refresh_xsf_status)
        tools_menu.addAction(refresh_status)

        # ── View ──
        view_menu = menu_bar.addMenu("视图(&V)")
        reset_action = QAction("重置布局", self)
        reset_action.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_action)

        # ── Help ──
        help_menu = menu_bar.addMenu("\u5e2e\u52a9(&H)")
        about_action = QAction("\u5173\u4e8e pyat GUI", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_docks(self) -> None:
        self._left_dock = QDockWidget("\u9879\u76ee\u6d4f\u89c8\u5668", self)
        self._left_dock.setWidget(self._project_explorer)
        self._left_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._left_dock.setMinimumWidth(300)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._left_dock)

        self._bottom_dock = QDockWidget("\u63a7\u5236\u53f0 & \u4efb\u52a1\u7ba1\u7406", self)
        self._bottom_dock.setWidget(self._console_view)
        self._bottom_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self._bottom_dock.setMinimumHeight(150)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._bottom_dock)

    def _setup_central(self) -> None:
        self._central_tabs = QTabWidget()
        self._central_tabs.addTab(self._map_view, "\u5730\u56fe\u89c6\u56fe")
        self._central_tabs.addTab(self._bsar_viewer, "BSAR \u89c6\u56fe")
        self._central_tabs.addTab(self._wc2d_viewer, "\u6c34\u67f1 2D")
        self.setCentralWidget(self._central_tabs)

    def _setup_statusbar(self) -> None:
        self._statusbar = QStatusBar()
        self._statusbar.showMessage("\u5c31\u7eea \u2014 \u8bf7\u52a0\u8f7d XSF \u6587\u4ef6\u5f00\u59cb\u5904\u7406\u3002")
        self.setStatusBar(self._statusbar)

    def _connect_signals(self) -> None:
        pe = self._project_explorer
        pe.files_added.connect(self._on_files_added)
        pe.file_selected.connect(self._on_files_selected)
        pe.file_selected.connect(self._on_file_highlight)  # highlight nav track
        pe.tool_requested.connect(self._dispatch_tool)

        # Map View polygon drawer → save KML mask
        self._map_view.polygon_drawn.connect(self._on_polygon_drawn)

        # Console cancel → process manager
        self._console_view.cancel_requested.connect(self._process_manager.cancel)

        # Go to location
        self._project_explorer.go_to_location.connect(self._on_go_to_location)

        # Toggle nav track / DTM layer visibility
        self._project_explorer.file_toggled.connect(self._on_file_toggled)

        # DTM layer toggle
        self._project_explorer.dtm_layer_changed.connect(self._map_view.show_dtm_layer)
        self._project_explorer.dtm_gamma_changed.connect(self._map_view.set_dtm_gamma)
        self._project_explorer.dtm_shoulder_changed.connect(self._map_view.set_dtm_shoulder)

        # Product actions (GoTo / Export GeoTIFF)
        self._project_explorer.product_action.connect(self._on_product_action)

        # File removal from list
        self._project_explorer.file_removed.connect(self._on_file_removed)

    def _on_files_added(self, metadata_list: list) -> None:
        count = len(metadata_list)
        processed = sum(1 for m in metadata_list if m.is_processed)
        self._statusbar.showMessage(
            f"\u5df2\u52a0\u8f7d {count} \u4e2a XSF \u6587\u4ef6 \u2014 {processed} \u4e2a\u5df2\u5904\u7406\uff0c"
            f"{count - processed} \u4e2a\u672a\u5904\u7406\u3002"
        )
        self._show_nav_tracks(metadata_list)

    def _on_files_selected(self, metadata_list: list) -> None:
        if metadata_list:
            self._statusbar.showMessage(
                f"已选中 {len(metadata_list)} 个文件。"
                f"右键使用工具或使用工具箱面板。"
            )

    def _on_file_highlight(self, metadata_list: list) -> None:
        """Highlight nav track of the first selected XSF file on the map.

        Does NOT call add_file_track() because that would corrupt _visible_tracks
        (Qt fires itemChanged before itemClicked for checkbox toggles —
        the checkbox handler hides the track, then this re-shows it).
        Instead only cache nav data and draw the highlight on highlightLayer.
        """
        if not metadata_list:
            return
        meta = metadata_list[0]
        coords = self._extract_nav_coords(meta.filepath)
        if coords:
            # Cache track data without altering checkbox-controlled _visible_tracks
            if meta.filepath not in self._map_view._nav_tracks:
                self._map_view._nav_tracks[meta.filepath] = coords
            self._map_view.draw_nav_line(coords)

    def _on_file_toggled(self, filepath: str, visible: bool) -> None:
        """Toggle nav track (XSF) or DTM layer visibility."""
        if filepath.endswith(".dtm.nc"):
            self._map_view.show_dtm_file(filepath, visible)
        else:
            self._map_view.show_file_track(filepath, visible)

    def _on_file_removed(self, filepath: str) -> None:
        """Clean up map layers when file is removed from list."""
        if filepath.endswith(".dtm.nc"):
            self._map_view.show_dtm_file(filepath, False)
            self._map_view._run_js(f"delete dtmOverlays[{json.dumps(filepath)}]")
        else:
            # Remove nav track data and redraw
            self._map_view._nav_tracks.pop(filepath, None)
            self._map_view._visible_tracks.discard(filepath)
            self._map_view._redraw_all()

    def _on_polygon_drawn(self, geojson_str: str) -> None:
        """Save ROI polygon as KML mask file for use in BSAR/Stats tools."""
        try:
            geojson = json.loads(geojson_str)
            # Save as GeoJSON for mask.py (which reads via fiona/geopandas)
            mask_path = os.path.join(
                tempfile.gettempdir(),
                f"pyat_roi_mask_{datetime.now().strftime('%H%M%S')}.geojson"
            )
            with open(mask_path, "w") as f:
                json.dump(geojson, f, indent=2)
            self._current_kml_mask = mask_path
            self._statusbar.showMessage(f"ROI 边界已保存: {mask_path}")
            self._console_view.append_text(f"边界 ROI 已保存: {mask_path}", "OK")
        except Exception as e:
            self._console_view.append_text(f"保存 ROI 失败: {e}", "ERROR")

    def _show_nav_tracks(self, metadata_list: List[XsfMetadata]) -> None:
        """Extract navigation coords and add each as a named track on the map."""
        for meta in metadata_list:
            coords = self._extract_nav_coords(meta.filepath)
            if coords:
                self._map_view.add_file_track(meta.filepath, coords)

    def _extract_nav_coords(self, filepath: str) -> List[List[float]]:
        """Extract vessel track [lat, lon] pairs from XSF.
        
        Uses ping-level platform_latitude/longitude (Sonar/Beam_group1/).
        NOT detection_latitude (beam-level, would show swath coverage).
        Decimates to ~2000 points max for map display.
        """
        try:
            from netCDF4 import Dataset
            import numpy as np
            with Dataset(filepath, "r") as ds:
                # Navigate to Sonar/Beam_group1
                sonar = ds.groups.get("Sonar")
                if not sonar:
                    return []
                for gname in sonar.groups:
                    if not gname.lower().startswith("beam_group"):
                        continue
                    bg = sonar.groups[gname]
                    # Try platform_lat/lon (ping-level, SONAR-netCDF4 standard)
                    if "platform_latitude" in bg.variables and "platform_longitude" in bg.variables:
                        lats = bg.variables["platform_latitude"][:].ravel()
                        lons = bg.variables["platform_longitude"][:].ravel()
                    else:
                        # Fallback: detection center beam
                        bathy = bg.groups.get("Bathymetry")
                        if not bathy:
                            continue
                        dlats = bathy.variables.get("detection_latitude")
                        dlons = bathy.variables.get("detection_longitude")
                        if dlats is None or dlons is None:
                            continue
                        lats = dlats[:, dlats.shape[1] // 2].ravel()
                        lons = dlons[:, dlons.shape[1] // 2].ravel()
                    # Decimate
                    total = len(lats)
                    step = max(1, total // 2000)
                    lats = lats[::step]
                    lons = lons[::step]
                    mask = np.isfinite(lats) & np.isfinite(lons)
                    if mask.any():
                        return [[float(lats[i]), float(lons[i])] for i in range(len(lats)) if mask[i]]
            return []
        except Exception:
            return []

    # ── Tool dispatch ──────────────────────────────────────────────

    def _dispatch_tool(self, tool_name: str) -> None:
        selected = self._project_explorer.get_selected_files()

        if tool_name == "tool1":
            self._open_tool1(selected)
        elif tool_name == "tool2a":
            self._open_tool2a(selected)
        elif tool_name == "tool2b_step1":
            self._open_tool2b_s1(selected)
        elif tool_name == "tool2b_step2":
            self._open_tool2b_s2(selected)
        elif tool_name in ("wc_horizontal", "wc_longitudinal", "wc_polar", "wc_vertical_integration"):
            self._open_wc_tool(tool_name, selected)
        else:
            QMessageBox.warning(self, "未知工具", f"未知工具: {tool_name}")

    def _warn_no_files(self) -> bool:
        selected = self._project_explorer.get_selected_files()
        if not selected:
            QMessageBox.warning(self, "无文件", "请先选择 XSF 文件。")
            return True
        return False

    def _execute_tool(self, tool_name: str, config_path: str) -> None:
        if self._process_manager.is_running():
            QMessageBox.warning(self, "繁忙", "另一个进程正在运行。")
            return

        task = self._task_manager.create_task(tool_name, config_path)
        self._current_task_id = task.task_id
        self._task_manager.update_status(task.task_id, TaskStatus.RUNNING)
        self._process_manager.run(config_path)

    def _on_process_state(self, state: ProcessState) -> None:
        tid = self._current_task_id
        if tid:
            if state == ProcessState.RUNNING:
                self._task_manager.update_status(tid, TaskStatus.RUNNING)
            elif state == ProcessState.FINISHED:
                self._task_manager.update_status(tid, TaskStatus.COMPLETED)
                self._refresh_xsf_status()
                self._on_dtm_created(tid)
            elif state == ProcessState.ERROR:
                self._task_manager.update_status(tid, TaskStatus.FAILED)
            elif state == ProcessState.CANCELLED:
                self._task_manager.update_status(tid, TaskStatus.CANCELLED)

    def _on_dtm_created(self, task_id: str) -> None:
        """Load DTM from completed Tool 1 into the map view as overlay."""
        task = self._task_manager.get_task(task_id)
        if not task or "sounder_to_dtm" not in task.config_json_path:
            return
        try:
            import json
            with open(task.config_json_path) as f:
                cfg = json.load(f)
            out_files = cfg.get("o_paths", [])
            if not out_files:
                return
            dtm_path = out_files[0]  # first output file
            if not os.path.exists(dtm_path):
                return
            # Generate overlays for all available layers
            png_bs, data_url, bounds = dtm_layer_to_file(dtm_path, "backscatter", cmap="gray")
            self._map_view.set_dtm_overlay(dtm_path, "backscatter", data_url, bounds)
            try:
                png_el, data_el, _ = dtm_layer_to_file(dtm_path, "elevation", cmap="terrain", hillshade=True)
                self._map_view.set_dtm_overlay(dtm_path, "elevation", data_el, bounds)
            except Exception:
                pass  # elevation layer may not exist in all DTM files
            self._map_view.show_dtm_layer("backscatter")
            # Add to Project Explorer
            self._project_explorer.add_external_file(dtm_path, "product")
            self._console_view.append_text(f"DTM loaded on map: {dtm_path}", "OK")
        except Exception as e:
            self._console_view.append_text(f"DTM overlay failed: {e}", "ERROR")

    def _on_process_progress(self, percent: int, message: str) -> None:
        self._task_manager.update_progress(self._current_task_id, percent, message)

    def _on_process_log(self, level: str, message: str) -> None:
        self._task_manager.append_log(self._current_task_id, level, message)

    def _on_process_error(self, error_msg: str) -> None:
        self._task_manager.set_failed(self._current_task_id, error_msg)

    def _on_process_finished(self, exit_code: int, output_file: str) -> None:
        pass  # Handled in _on_process_state

    # ── Tool 1 ─────────────────────────────────────────────────────

    def _open_tool1(self, selected_files: list) -> None:
        if not selected_files:
            QMessageBox.warning(self, "No Files", "Select XSF files first.")
            return
        # Compute overall bounds for grid size estimation and coord param
        lon_min, lat_min = 180, 90
        lon_max, lat_max = -180, -90
        for f in selected_files:
            c = self._extract_nav_coords(f)
            if c:
                lons = [pt[1] for pt in c]
                lats = [pt[0] for pt in c]
                lon_min = min(lon_min, min(lons))
                lon_max = max(lon_max, max(lons))
                lat_min = min(lat_min, min(lats))
                lat_max = max(lat_max, max(lats))
        bounds = None if lon_max < -180 else (lon_min, lat_min, lon_max, lat_max)

        wiz = SounderToDtmWizard(selected_files, bounds, self)
        if wiz.exec() == QWizard.Accepted:
            p = wiz.getAllParams()
            self._execute_tool1_with_params(p)

    def _execute_tool1_with_params(self, p: dict) -> None:
        """Execute Tool 1 with wizard parameters (shared by normal and re-run)."""
        out_dir = p["output_dir"] or next(iter(p["input_files"]), "")
        if not out_dir:
            return
        out_dir = os.path.dirname(out_dir) if os.path.isfile(out_dir) else out_dir
        prefix = p["output_prefix"]

        if p.get("separate", False):
            out_files = []
            for f in p["input_files"]:
                base = os.path.splitext(os.path.basename(f))[0]
                out_name = f"{prefix}_{base}.dtm.nc"
                out_files.append(os.path.join(out_dir, out_name))
            coord = None
        else:
            out_files = [os.path.join(out_dir, f"{prefix}_combined.dtm.nc")]
            proj_def = p.get("proj_def", "")
            is_geo = "longlat" in proj_def or "latlong" in proj_def
            # coord is always in lat/lon degrees. For projected CRS (Mercator/UTM),
            # passing degrees + meter resolution makes estimate_col(deg/m→0)→1 cell.
            # Let the backend auto-compute bounds from data in the correct CRS.
            if is_geo:
                coord = p.get("coord")
                if coord is None:
                    l_min, la_min, l_max, la_max = 180, 90, -180, -90
                    for f in p["input_files"]:
                        c = self._extract_nav_coords(f)
                        if c:
                            for pt in c:
                                l_min = min(l_min, pt[1])
                                l_max = max(l_max, pt[1])
                                la_min = min(la_min, pt[0])
                                la_max = max(la_max, pt[0])
                    coord = {"west": l_min, "south": la_min, "east": l_max, "north": la_max}
                if p.get("expand", True) and p["resolution"]:
                    try:
                        res = float(p["resolution"])
                        l_min = coord["west"]
                        la_min = coord["south"]
                        l_max = coord["east"]
                        la_max = coord["north"]
                        dx = l_max - l_min; dy = la_max - la_min
                        import math
                        cols = math.ceil(dx / res); rows = math.ceil(dy / res)
                        coord["east"] = l_min + cols * res
                        coord["north"] = la_min + rows * res
                    except (ValueError, ZeroDivisionError, KeyError):
                        pass
            else:
                coord = None

        config_path = build_sounder_to_dtm_json(
            input_files=p["input_files"],
            output_files=out_files,
            target_resolution=p["resolution"],
            target_spatial_reference=p["proj_def"],
            layers=p["layers"],
            gap_filling=p["gap_filling"],
            mask_size=p["mask_size"] or 3,
            valid_sounds_only=p["valid_sounds_only"],
            spatial_antialiasing=p["spatial_antialiasing"],
            min_elevation=p["min_elevation"],
            max_elevation=p["max_elevation"],
            min_sounds=p["min_sounds"],
            overwrite=p["overwrite"],
            coord=coord,
            title=p["title"],
            institution=p["institution"],
            source=p.get("source", ""),
            references=p.get("references", ""),
            comment=p.get("comment", ""),
            quality_indicator=p["quality_indicator"],
        )
        task = self._task_manager.create_task("工具 1: 声纳至 DTM", config_path)
        self._current_task_id = task.task_id
        self._task_manager.update_status(task.task_id, TaskStatus.QUEUED)
        self._process_manager.run(config_path)

    # ── Tool 2A ────────────────────────────────────────────────────

    def _open_tool2a(self, selected_files: list) -> None:
        if not selected_files:
            QMessageBox.warning(self, "No Files", "Select XSF files first.")
            return
        try:
            dlg = Tool2ADialog(selected_files, self)
            result = dlg.exec()
        except Exception as e:
            self._console_view.append_text(
                f"无法打开 BSAR 对话框: {e}", "ERROR")
            QMessageBox.critical(
                self, "对话框错误",
                f"无法打开 BSAR 向导:\n{e}")
            return
        if result in (QMessageBox.Accepted, 2):
            params = dlg.get_params()
            bathy = dlg.bathy_nc
            config_path = build_tool2a_json(
                input_files=selected_files, bathy_nc=bathy, **params,
            )
            if result == QMessageBox.Accepted:
                self._execute_tool("工具 2A: 滑动角度重规范化", config_path)
            else:
                self._console_view.append_text(f"Config saved: {config_path}", "INFO")
            self._statusbar.showMessage(f"工具 2A 配置已保存: {config_path}")

    # ── Tool 2B Step 1 ─────────────────────────────────────────────

    def _open_tool2b_s1(self, selected_files: list) -> None:
        if not selected_files:
            QMessageBox.warning(self, "无文件", "请先选择 XSF 文件。")
            return
        try:
            wiz = Tool2BS1Dialog(selected_files, self)
            result = wiz.exec()
        except Exception as e:
            self._console_view.append_text(
                f"无法打开 BSAR 对话框: {e}", "ERROR")
            QMessageBox.critical(
                self, "对话框错误",
                f"无法打开 BSAR 向导:\n{e}")
            return
        if result == QWizard.Accepted:
            if not wiz.bsar_nc:
                QMessageBox.warning(self, "缺少 BSAR 模型",
                                    "需要 BSAR 模型文件。\n"
                                    "请在步骤 1 中选择 .bsar.nc 文件。")
                return
            params = wiz.get_params()
            config_path = build_tool2b_step2a_json(
                input_files=selected_files,
                bsar_nc=wiz.bsar_nc,
                bathy_nc=wiz.bathy_nc,
                output_dir=wiz.getOutputDir(),
                overwrite=wiz.getOverwrite(),
                **params,
            )
            self._execute_tool("静态角度重规范化", config_path)
            self._statusbar.showMessage(f"静态角度重规范化配置已保存: {config_path}")

    # ── Tool 2B Step 2 ─────────────────────────────────────────────

    def _open_tool2b_s2(self, selected_files: list) -> None:
        if not selected_files:
            QMessageBox.warning(self, "无文件", "请先选择 XSF 文件。")
            return
        try:
            wiz = Tool2BS2Dialog(selected_files, self)
            result = wiz.exec()
        except Exception as e:
            self._console_view.append_text(
                f"打开统计角响应（BSAR）对话框失败: {e}", "ERROR"
            )
            QMessageBox.critical(
                self, "对话框错误",
                f"无法打开参数对话框:\n{e}\n\n"
                "请确保在文件树中已选择 XSF 文件。"
            )
            return
        if result == QWizard.Accepted:
            if not wiz.getOutputBsar():
                QMessageBox.warning(self, "缺少输出路径",
                                    "需要 BSAR 输出文件路径。\n"
                                    "请在步骤 1 中设置 .bsar.nc 文件路径。")
                return
            params = wiz.get_params()
            config_path = build_tool2b_step2b_json(
                input_files=selected_files,
                bathy_nc=wiz.bathy_nc,
                output_bsar=wiz.getOutputBsar(),
                **params,
            )
            self._execute_tool("统计角响应（BSAR）", config_path)
            self._statusbar.showMessage(f"统计角响应 (BSAR) 配置已保存: {config_path}")
            self._refresh_xsf_status()

    # ── BSAR Aux Tools ─────────────────────────────────────────────

    def _open_bsar_tools(self) -> None:
        dlg = BsarToolsDialog(self)
        if dlg.exec() == QMessageBox.Accepted:
            params = dlg.get_active_params()
            tool = dlg.get_active_tool()
            task = self._task_manager.create_task(f"BSAR {tool}", "")
            self._task_manager.update_status(task.task_id, TaskStatus.COMPLETED)
            self._task_manager.append_log(task.task_id, "INFO",
                f"BSAR {tool} 配置: {params}")
            self._statusbar.showMessage(f"BSAR {tool} 已完成")

    # ── Status Refresh ─────────────────────────────────────────────

    def _refresh_xsf_status(self) -> None:
        """Re-read processing status for all loaded XSF files."""
        for fp, meta in self._project_explorer._metadata_cache.items():
            try:
                updated = read_xsf_metadata(fp)
                self._project_explorer._metadata_cache[fp] = updated
            except Exception:
                pass
        self._statusbar.showMessage("XSF 状态已刷新")
        self._console_view.append_text("XSF 处理状态已刷新。", "INFO")

    # ── Import DTM ──────────────────────────────────────────────────

    def _import_dtm(self) -> None:
        """Import one or more DTM.nc files and display their layers on the map."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import DTM File(s)", "",
            "DTM files (*.dtm.nc *.nc);;All (*.*)"
        )
        if not paths:
            return
        loaded = 0
        errors = []
        for path in paths:
            try:
                self._console_view.append_text(f"Importing DTM: {path}", "INFO")
                # Generate overlay PNG files
                png_bs, data_url, bounds = dtm_layer_to_file(path, "backscatter", cmap="gray")
                self._map_view.set_dtm_overlay(path, "backscatter", data_url, bounds)
                png_el, data_el, _ = dtm_layer_to_file(path, "elevation", cmap="terrain", hillshade=True)
                self._map_view.set_dtm_overlay(path, "elevation", data_el, bounds)
                self._project_explorer.add_external_file(path, "product")
                loaded += 1
            except Exception as e:
                errors.append(f"{os.path.basename(path)}: {e}")
        if loaded:
            self._map_view.show_dtm_layer("backscatter")
            self._project_explorer.dtm_layer_changed.emit("backscatter")
            self._statusbar.showMessage(f"已导入 {loaded} 个 DTM 文件。")
        if errors:
            QMessageBox.warning(self, "导入错误",
                                "以下文件加载失败:\n" + "\n".join(errors))

    def _on_product_action(self, action: str, filepath: str) -> None:
        """Handle DTM/Product right-click actions."""
        if action == "goto_dtm":
            self._on_go_to_dtm(filepath)
        elif action == "geotiff_dtm":
            self._on_export_dtm_geotiff(filepath)
        elif action == "xyz_dtm":
            self._on_export_dtm_xyz(filepath)

    def _on_go_to_dtm(self, filepath: str) -> None:
        """Go to a DTM file's location on the map."""
        try:
            from netCDF4 import Dataset
            with Dataset(filepath, "r") as ds:
                if "lon" in ds.variables and "lat" in ds.variables:
                    lon = ds.variables["lon"][:]
                    lat = ds.variables["lat"][:]
                    lons = [float(lon.min()), float(lon.max())]
                    lats = [float(lat.min()), float(lat.max())]
                elif "x" in ds.variables and "y" in ds.variables:
                    x_arr = ds.variables["x"][:]
                    y_arr = ds.variables["y"][:]
                    crs_wkt = getattr(ds.variables.get("crs"), "crs_wkt", None)
                    if crs_wkt:
                        from pyproj import CRS, Transformer
                        src = CRS.from_wkt(crs_wkt)
                        tgt = CRS.from_epsg(4326)
                        tr = Transformer.from_crs(src, tgt, always_xy=True)
                        sw = tr.transform(float(x_arr[0]), float(y_arr[0]))
                        ne = tr.transform(float(x_arr[-1]), float(y_arr[-1]))
                        lons = [sw[0], ne[0]]
                        lats = [sw[1], ne[1]]
                    else:
                        lons = [float(x_arr.min()), float(x_arr.max())]
                        lats = [float(y_arr.min()), float(y_arr.max())]
                else:
                    raise KeyError("No lon/lat or x/y variables found")
            self._map_view.fly_to_bounds(lats, lons)
            self._statusbar.showMessage(f"定位到 DTM: {os.path.basename(filepath)}")
        except Exception as e:
            QMessageBox.warning(self, "读取错误", f"无法读取 DTM 范围:\n{e}")

    def _on_export_dtm_geotiff(self, filepath: str) -> None:
        """Export DTM layers to GeoTIFF via the backend Dtm2Tiff."""
        try:
            from PySide6.QtWidgets import QDialog
            dlg = DtmExportDialog(filepath, self)
            if dlg.exec() != QDialog.Accepted:
                return
        except Exception as e:
            QMessageBox.critical(self, "对话框错误",
                                 f"无法打开导出对话框:\n{e}")
            return

        self._statusbar.showMessage("正在导出 GeoTIFF...请稍候")
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
            # GDAL netCDF plugin needs conda bin on PATH to find netcdf.dll etc.
            env_root = os.path.dirname(_sys.executable)
            if os.path.basename(env_root).lower() == "scripts":
                env_root = os.path.dirname(env_root)
            _conda_bin = os.path.join(env_root, "Library", "bin")
            if not os.path.isdir(_conda_bin):
                _conda_bin = r"C:\Users\GUO\AppData\Local\anaconda3\Library\bin"

            if os.path.isdir(_conda_bin):
                os.environ.pop("USE_PATH_FOR_GDAL_PYTHON", None)
                if "GDAL_DRIVER_PATH" not in os.environ:
                    os.environ["GDAL_DRIVER_PATH"] = os.path.join(
                        _conda_bin, "..", "lib", "gdalplugins")
                if "GDAL_DATA" not in os.environ:
                    _gdal_data = os.path.join(env_root, "Library", "share", "gdal")
                    if os.path.isdir(_gdal_data):
                        os.environ["GDAL_DATA"] = _gdal_data
                    else:
                        os.environ["GDAL_DATA"] = r"C:\Users\GUO\AppData\Local\anaconda3\Library\share\gdal"
                os.environ["PATH"] = _conda_bin + os.pathsep + os.environ.get("PATH", "")
            from pyat.dtm.export.dtm_to_tiff import Dtm2Tiff
            from pygws.service.progress_monitor import DefaultMonitor

            out_dir = dlg.getOutputDir()
            name_base = dlg.getFileName()
            o_path = os.path.join(out_dir, f"{name_base}.tif")

            exporter = Dtm2Tiff(
                i_paths=[filepath],
                o_paths=[o_path],
                layers=dlg.getLayers(),
                overwrite=dlg.getOverwrite(),
                target_compression=dlg.getCompression(),
                nan_fillvalue=dlg.getNanFill(),
                target_fillvalue=dlg.getFillValue() or 32767,
                monitor=DefaultMonitor,
            )
            exporter()
            result_files = exporter.resulting_files
            for rf in result_files:
                self._console_view.append_text(f"GeoTIFF 已导出: {rf}", "OK")
            self._statusbar.showMessage(
                f"已导出 {len(result_files)} 个 GeoTIFF 文件"
            )
        except Exception as e:
            self._console_view.append_text(f"导出失败: {e}", "ERROR")
            QMessageBox.warning(self, "导出失败",
                                f"GeoTIFF 导出失败:\n{e}")

    # ── Map Interactions ──────────────────────────────────────────────


    def _on_export_dtm_xyz(self, filepath: str) -> None:
        """Export DTM to XYZ ascii format via backend Dtm2Ascii."""
        try:
            from PySide6.QtWidgets import QDialog
            dlg = DtmXyzExportDialog(filepath, self)
            if dlg.exec() != QDialog.Accepted:
                return
        except Exception as e:
            QMessageBox.critical(self, "对话框错误",
                                 f"无法打开导出对话框:\n{e}")
            return

        self._statusbar.showMessage("正在导出 XYZ...请稍候")
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
            
            from pyat.dtm.export.dtm_to_ascii import Dtm2Ascii
            from pygws.service.progress_monitor import DefaultMonitor

            out_dir = dlg.getOutputDir()
            name_base = dlg.getFileName()
            o_path = os.path.join(out_dir, f"{name_base}.xyz")

            exporter = Dtm2Ascii(
                i_paths=[filepath],
                o_paths=[o_path],
                export_missing_values=dlg.getExportMissing(),
                overwrite=dlg.getOverwrite(),
                column_separator=dlg.getColumnSeparator(),
                other_separator=dlg.getOtherSeparator(),
                decimal_separator=dlg.getDecimalSeparator(),
                column_order=dlg.getColumnOrder(),
                monitor=DefaultMonitor,
            )
            exporter()
            
            # Backend process finished successfully
            self._console_view.append_text(f"XYZ 已导出: {o_path}", "OK")
            self._statusbar.showMessage("已导出 XYZ 文件")
        except Exception as e:
            self._console_view.append_text(f"导出 XYZ 失败: {e}", "ERROR")
            QMessageBox.critical(self, "导出错误", f"导出 XYZ 失败:\n{e}")
            self._statusbar.showMessage("导出 XYZ 失败")

    def _on_go_to_location(self, filepath: str) -> None:
        """Extract nav from a single file, draw its track, and fly to it."""
        # Handle DTM files: use lon/lat bounds
        if filepath.endswith(".dtm.nc"):
            self._on_go_to_dtm(filepath)
            return
        # Ensure all files' tracks are loaded
        loaded = 0
        for meta in self._project_explorer._metadata_cache.values():
            if meta.filepath not in self._map_view._nav_tracks:
                coords = self._extract_nav_coords(meta.filepath)
                if coords:
                    self._map_view.add_file_track(meta.filepath, coords)
                    loaded += 1

        # Now highlight + fly to the selected file
        coords = self._map_view._nav_tracks.get(filepath)
        if not coords:
            coords = self._extract_nav_coords(filepath)
            if not coords:
                return
            self._map_view.add_file_track(filepath, coords)

        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        # Redraw all + highlight in one atomic call
        parts = ["if(typeof navLayer !== 'undefined') { navLayer.clearLayers(); highlightLayer.clearLayers(); }"]
        for fp, c in self._map_view._nav_tracks.items():
            if fp in self._map_view._visible_tracks:
                pts = ",".join(f"[{x[0]},{x[1]}]" for x in c)
                parts.append(f"L.polyline([{pts}],{{color:'#ff4444',weight:2,opacity:.7}}).addTo(navLayer)")
        hl_pts = ",".join(f"[{c[0]},{c[1]}]" for c in coords)
        parts.append(f"L.polyline([{hl_pts}],{{color:'#ffcc00',weight:3,opacity:0.9}}).addTo(highlightLayer)")
        self._map_view._run_js(";".join(parts))
        self._map_view.fly_to_bounds(lats, lons)
        self._statusbar.showMessage(
            f"Located: {os.path.basename(filepath)} — {len(coords)} nav pts, {len(self._map_view._visible_tracks)} tracks shown"
        )

    # ── WC ──────────────────────────────────────────────────────────

    def _open_wc_wizard(self, mode: str = "horizontal"):
        """打开水柱分析向导对话框"""
        from .core.json_builder import build_wc_json
        from .core.task_manager import TaskStatus

        selected = self._project_explorer.get_selected_files()
        wiz = WcWizard(mode, self)
        # 如果有选中的文件，自动填入第一页
        if selected and hasattr(wiz, '_page1'):
            for f in selected[:10]:
                wiz._page1._files_list.addItem(f)

        if wiz.exec() == QWizard.Accepted:
            data = wiz.get_all_params()
            if not data["input_files"]:
                QMessageBox.warning(self, "提示", "请至少选择一个 XSF 文件。")
                return
            # 构建 JSON 并执行
            try:
                config_path = build_wc_json(
                    mode=data["mode"],
                    input_files=data["input_files"],
                    output_dir=data["output_dir"],
                    overwrite=data["overwrite"],
                    layers=data["layers"],
                    normalization_offset=data["normalization_offset"],
                    **{k: v for k, v in data.items()
                       if k not in ("mode","input_files","output_dir",
                                    "overwrite","layers","normalization_offset",
                                    "gws_config","suffix")},
                )
                task_label = f"工具 WC: {data['mode']}"
                task = self._task_manager.create_task(task_label, config_path)
                self._current_task_id = task.task_id
                self._task_manager.update_status(task.task_id, TaskStatus.QUEUED)
                self._process_manager.run(config_path)
            except Exception as e:
                QMessageBox.critical(self, "运行失败", str(e))

    def _open_wc_tool(self, tool_name: str, selected_files: list) -> None:
        """从工具箱按钮调用的 WC 入口"""
        mode_map = {
            "wc_horizontal": "horizontal",
            "wc_longitudinal": "longitudinal",
            "wc_polar": "polar",
            "wc_vertical_integration": "vertical",
        }
        self._open_wc_wizard(mode_map.get(tool_name, "horizontal"))

    # ── Misc ───────────────────────────────────────────────────────

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "\u5173\u4e8e pyat GUI",
            "pyat GUI \u2014 \u591a\u675f\u56de\u58f0\u5904\u7406\n\n"
            "\u7248\u672c 4\uff1a\u5de5\u5177 3\uff08\u56de\u58f0\u62fc\u63a5\uff09\u3001GeoTIFF/COG/MBTiles \u5bfc\u51fa\u3001"
            "\u8272\u5f69\u7eb1\u3001\u5206\u8d1d\u8303\u56f4\u6ed1\u5757\u3001\u6570\u636e\u63a2\u9488\u3002\n\n"
            "\u57fa\u4e8e PySide6 + pyat \u540e\u7aef\u3002"
        )

    def _toggle_theme(self) -> None:
        from .app import apply_theme
        settings = QSettings("pyat", "gui")
        current = settings.value("theme", "dark")
        new_theme = "light" if current == "dark" else "dark"
        app = QApplication.instance()
        apply_theme(app, new_theme)
        self._theme_action.setChecked(new_theme == "dark")

    @staticmethod
    def _is_dark_theme() -> bool:
        settings = QSettings("pyat", "gui")
        return settings.value("theme", "dark") != "light"

    def _reset_layout(self) -> None:
        self._left_dock.setFloating(False)
        self._bottom_dock.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._left_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._bottom_dock)
        self.resize(1400, 900)
        self._statusbar.showMessage("\u5e03\u5c40\u5df2\u91cd\u7f6e\u4e3a\u9ed8\u8ba4\u3002")

    def _restore_state(self) -> None:
        settings = QSettings("pyat", "gui")
        geometry = settings.value("geometry")
        state = settings.value("windowState")
        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state)
        
        # Restore last opened files
        last_files = settings.value("lastFiles", [])
        if last_files:
            if isinstance(last_files, str):
                last_files = [last_files]
            valid_files = [f for f in last_files if os.path.exists(f)]
            if valid_files:
                self._project_explorer._load_files(valid_files)

    def closeEvent(self, event) -> None:
        settings = QSettings("pyat", "gui")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        
        # Save last opened files
        files = list(self._project_explorer._metadata_cache.keys())
        settings.setValue("lastFiles", files)
        
        super().closeEvent(event)
