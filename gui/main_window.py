"""Main application window with Eclipse RCP / QGIS style layout.

Four-panel dock layout:
  - Left: Project Explorer + Toolbox (QTabWidget in QDockWidget)
  - Bottom: Console & Job Manager (QDockWidget)
  - Center: Main Canvas (QTabWidget with Map View + BSAR Viewer)
  - Left-bottom: Properties Panel (optional, Phase 2+)
"""
import json
import os
import tempfile
from datetime import datetime
from typing import Optional, List

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QTabWidget, QWidget,
    QStatusBar, QMessageBox,
)

from .project_explorer import ProjectExplorer
from .views.console_view import ConsoleView
from .views.map_view import MapView
from .views.bsar_viewer import BsarViewer
from .core.task_manager import TaskManager, TaskStatus
from .core.process_manager import ProcessManager, ProcessState
from .core.json_builder import (
    build_tool1_json, build_tool2a_json, build_tool2b_step2a_json,
    build_tool2b_step2b_json, build_tool3_json,
)
from .core.xsf_reader import XsfMetadata, read_xsf_metadata
from .dialogs.tool1_dialog import Tool1Dialog
from .dialogs.tool2a_dialog import Tool2ADialog
from .dialogs.tool2b_s1_dialog import Tool2BS1Dialog
from .dialogs.tool2b_s2_dialog import Tool2BS2Dialog
from .dialogs.bsar_tools_dialog import BsarToolsDialog
from .dialogs.tool3_dialog import Tool3Dialog


class MainWindow(QMainWindow):
    """pyat GUI main window."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("pyat GUI \u2014 Multibeam Backscatter Processing")
        self.resize(1400, 900)

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

        file_menu = menu_bar.addMenu("&File")
        add_action = QAction("Add XSF Files...", self)
        add_action.triggered.connect(self._project_explorer._add_files)
        file_menu.addAction(add_action)
        add_folder_action = QAction("Add XSF Folder...", self)
        add_folder_action.triggered.connect(self._project_explorer._add_folder)
        file_menu.addAction(add_folder_action)

        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        tools_menu = menu_bar.addMenu("&Tools")
        bsar_aux = QAction("BSAR Auxiliary Tools...", self)
        bsar_aux.triggered.connect(self._open_bsar_tools)
        tools_menu.addAction(bsar_aux)
        refresh_status = QAction("Refresh XSF Status", self)
        refresh_status.triggered.connect(self._refresh_xsf_status)
        tools_menu.addAction(refresh_status)

        view_menu = menu_bar.addMenu("&View")
        reset_action = QAction("Reset Layout", self)
        reset_action.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_action)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("About pyat GUI", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_docks(self) -> None:
        self._left_dock = QDockWidget("Project Explorer", self)
        self._left_dock.setWidget(self._project_explorer)
        self._left_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._left_dock.setMinimumWidth(300)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._left_dock)

        self._bottom_dock = QDockWidget("Console & Job Manager", self)
        self._bottom_dock.setWidget(self._console_view)
        self._bottom_dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        self._bottom_dock.setMinimumHeight(150)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._bottom_dock)

    def _setup_central(self) -> None:
        self._central_tabs = QTabWidget()
        self._central_tabs.addTab(self._map_view, "Map View")
        self._central_tabs.addTab(self._bsar_viewer, "BSAR Viewer")
        self.setCentralWidget(self._central_tabs)

    def _setup_statusbar(self) -> None:
        self._statusbar = QStatusBar()
        self._statusbar.showMessage("Ready \u2014 Load XSF files to begin.")
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

        # Toggle nav track visibility
        self._project_explorer.file_toggled.connect(self._map_view.show_file_track)

    def _on_files_added(self, metadata_list: list) -> None:
        count = len(metadata_list)
        processed = sum(1 for m in metadata_list if m.is_processed)
        self._statusbar.showMessage(
            f"Loaded {count} XSF file(s) \u2014 {processed} processed, "
            f"{count - processed} unprocessed."
        )
        # Extract and display navigation tracks
        self._show_nav_tracks(metadata_list)

    def _on_files_selected(self, metadata_list: list) -> None:
        if metadata_list:
            self._statusbar.showMessage(
                f"Selected {len(metadata_list)} file(s). "
                f"Right-click for tools or use Toolbox panel."
            )

    def _on_file_highlight(self, metadata_list: list) -> None:
        """Highlight nav track of the first selected XSF file on the map."""
        if not metadata_list:
            return
        meta = metadata_list[0]
        coords = self._extract_nav_coords(meta.filepath)
        if coords:
            self._map_view.add_file_track(meta.filepath, coords)
            self._map_view.draw_nav_line(coords)

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
            self._statusbar.showMessage(f"ROI mask saved: {mask_path}")
            self._console_view.append_text(f"ROI mask created: {mask_path}", "OK")
        except Exception as e:
            self._console_view.append_text(f"Failed to save mask: {e}", "ERROR")

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
        elif tool_name == "tool3":
            self._open_tool3(selected)
        else:
            QMessageBox.warning(self, "Unknown Tool", f"Unknown tool: {tool_name}")

    def _warn_no_files(self) -> bool:
        selected = self._project_explorer.get_selected_files()
        if not selected:
            QMessageBox.warning(self, "No Files", "Please select XSF files first.")
            return True
        return False

    # ── Process Execution ───────────────────────────────────────────

    def _execute_tool(self, tool_name: str, config_path: str) -> None:
        """Create a task and launch the pyat subprocess."""
        if self._process_manager.is_running():
            QMessageBox.warning(self, "Busy", "Another process is already running.")
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
            elif state == ProcessState.ERROR:
                self._task_manager.update_status(tid, TaskStatus.FAILED)
            elif state == ProcessState.CANCELLED:
                self._task_manager.update_status(tid, TaskStatus.CANCELLED)

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
        dlg = Tool1Dialog(selected_files, self)
        result = dlg.exec()
        if result in (QMessageBox.Accepted, 2):
            params = dlg.get_params()
            config_path = build_tool1_json(input_files=selected_files, **params)
            if result == QMessageBox.Accepted:
                self._execute_tool("Tool 1: Reference DTM", config_path)
            else:
                self._task_manager.create_task("Tool 1: Reference DTM (saved)", config_path)
                self._console_view.append_text(f"Config saved: {config_path}", "INFO")
            self._statusbar.showMessage(f"Tool 1 config saved: {config_path}")

    # ── Tool 2A ────────────────────────────────────────────────────

    def _open_tool2a(self, selected_files: list) -> None:
        if not selected_files:
            QMessageBox.warning(self, "No Files", "Select XSF files first.")
            return
        dlg = Tool2ADialog(selected_files, self)
        result = dlg.exec()
        if result in (QMessageBox.Accepted, 2):
            params = dlg.get_params()
            bathy = dlg.bathy_nc
            config_path = build_tool2a_json(
                input_files=selected_files, bathy_nc=bathy, **params,
            )
            if result == QMessageBox.Accepted:
                self._execute_tool("Tool 2A: Sliding Renorm", config_path)
            else:
                self._console_view.append_text(f"Config saved: {config_path}", "INFO")
            self._statusbar.showMessage(f"Tool 2A config saved: {config_path}")

    # ── Tool 2B Step 1 ─────────────────────────────────────────────

    def _open_tool2b_s1(self, selected_files: list) -> None:
        if not selected_files:
            QMessageBox.warning(self, "No Files", "Select XSF files first.")
            return
        dlg = Tool2BS1Dialog(selected_files, self)
        result = dlg.exec()
        if result in (QMessageBox.Accepted, 2):
            params = dlg.get_params()
            bathy = dlg.bathy_nc
            mask_files = params.pop("mask_files", None)
            config_path = build_tool2b_step2a_json(
                input_files=selected_files, bathy_nc=bathy,
                mask_files=mask_files, **params,
            )
            if result == QMessageBox.Accepted:
                self._execute_tool("Tool 2B: Statistical BSAR", config_path)
            else:
                self._console_view.append_text(f"Config saved: {config_path}", "INFO")
            self._statusbar.showMessage(f"Tool 2B Step 1 config saved: {config_path}")

    # ── Tool 2B Step 2 ─────────────────────────────────────────────

    def _open_tool2b_s2(self, selected_files: list) -> None:
        if not selected_files:
            QMessageBox.warning(self, "No Files", "Select XSF files first.")
            return
        dlg = Tool2BS2Dialog(selected_files, self)
        result = dlg.exec()
        if result in (QMessageBox.Accepted, 2):
            params = dlg.get_params()
            config_path = build_tool2b_step2b_json(
                input_files=selected_files,
                bsar_nc=dlg.bsar_nc,
                bathy_nc=dlg.bathy_nc,
                **params,
            )
            if result == QMessageBox.Accepted:
                self._execute_tool("Tool 2B: Apply BSAR", config_path)
            else:
                self._console_view.append_text(f"Config saved: {config_path}", "INFO")
            self._statusbar.showMessage(f"Tool 2B Step 2 config saved: {config_path}")
            # Refresh XSF status since processing may have updated it
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
                f"BSAR {tool} config: {params}")
            self._statusbar.showMessage(f"BSAR {tool} completed")

    # ── Status Refresh ─────────────────────────────────────────────

    def _refresh_xsf_status(self) -> None:
        """Re-read processing status for all loaded XSF files."""
        for fp, meta in self._project_explorer._metadata_cache.items():
            try:
                updated = read_xsf_metadata(fp)
                self._project_explorer._metadata_cache[fp] = updated
            except Exception:
                pass
        self._statusbar.showMessage("XSF status refreshed")
        self._console_view.append_text("XSF processing status refreshed.", "INFO")

    # ── Go to Location ──────────────────────────────────────────────

    def _on_go_to_location(self, filepath: str) -> None:
        """Extract nav from a single file, draw its track, and fly to it."""
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
        parts = ["navLayer.clearLayers()", "highlightLayer.clearLayers()"]
        for fp, c in self._map_view._nav_tracks.items():
            if fp in self._map_view._visible_tracks:
                pts = ",".join(f"[{x[0]},{x[1]}]" for x in c)
                parts.append(f"L.polyline([{pts}],{{color:'#4ec94e',weight:1,opacity:.5}}).addTo(navLayer)")
        hl_pts = ",".join(f"[{c[0]},{c[1]}]" for c in coords)
        parts.append(f"L.polyline([{hl_pts}],{{color:'#ffcc00',weight:3,opacity:0.9}}).addTo(highlightLayer)")
        self._map_view._run_js(";".join(parts))
        self._map_view.fly_to_bounds(lats, lons)
        self._statusbar.showMessage(
            f"Located: {os.path.basename(filepath)} — {len(coords)} nav pts, {len(self._map_view._visible_tracks)} tracks shown"
        )

    # ── Tool 3 ──────────────────────────────────────────────────────

    def _open_tool3(self, selected_files: list) -> None:
        if not selected_files:
            QMessageBox.warning(self, "No Files", "Select corrected XSF files (green icon) first.")
            return
        dlg = Tool3Dialog(selected_files, self)
        result = dlg.exec()
        if result in (QMessageBox.Accepted, 2):
            params = dlg.get_params()
            export_fmt = params.pop("export_format", "GeoTIFF")
            config_path = build_tool3_json(
                input_files=selected_files, **params,
            )
            if result == QMessageBox.Accepted:
                self._execute_tool(f"Tool 3: Mosaic ({export_fmt})", config_path)
            else:
                self._console_view.append_text(f"Config saved: {config_path}", "INFO")
            self._statusbar.showMessage(f"Tool 3 config saved — Export: {export_fmt}")

    # ── Misc ───────────────────────────────────────────────────────

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About pyat GUI",
            "pyat GUI \u2014 Multibeam Sonar Backscatter Processing\n\n"
            "Phase 4: Tool 3 (Backscatter Mosaic), GeoTIFF/COG/MBTiles export,\n"
            "Color Ramp palette, dB range slider, data probe.\n\n"
            "Powered by PySide6 + pyat backend."
        )

    def _reset_layout(self) -> None:
        self._left_dock.setFloating(False)
        self._bottom_dock.setFloating(False)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._left_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._bottom_dock)
        self.resize(1400, 900)
        self._statusbar.showMessage("Layout reset to default.")

    def _restore_state(self) -> None:
        settings = QSettings("pyat", "gui")
        geometry = settings.value("geometry")
        state = settings.value("windowState")
        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state)

    def closeEvent(self, event) -> None:
        settings = QSettings("pyat", "gui")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        super().closeEvent(event)
