"""Project Explorer panel with XSF file tree and Toolbox.

Left-side panel with QTabWidget:
  Tab 1: "Files" — QTreeWidget showing XSF files grouped by processing status
  Tab 2: "Toolbox" — processing tools organized by phase
"""
import os
from typing import Optional, List

from PySide6.QtCore import Qt, Signal, QItemSelectionModel
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QPushButton, QLabel,
    QFileDialog, QMessageBox, QMenu, QToolBar, QGroupBox,
    QHBoxLayout, QCheckBox, QSlider,
)

from .core.xsf_reader import XsfMetadata, read_xsf_metadata, scan_directory


# Item types for the tree
ITEM_ROOT = 0
ITEM_CATEGORY = 1
ITEM_FILE = 2


class _SelectionPreservingTreeWidget(QTreeWidget):
    """QTreeWidget that prevents right-click from visually changing the current multi-selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._right_click_saved = set()
        self._right_click_in_progress = False

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self._right_click_saved = set()
            for it in self.selectedItems():
                if it.data(0, Qt.UserRole + 1) is not None:
                    self._right_click_saved.add(it)
            self._right_click_in_progress = True
        super().mousePressEvent(event)
        if event.button() == Qt.RightButton and self._right_click_saved:
            self._restore_saved()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            pass
        super().mouseReleaseEvent(event)
        if event.button() == Qt.RightButton and self._right_click_saved:
            self._restore_saved()
            self._right_click_in_progress = False

    def _restore_saved(self):
        sel_model = self.selectionModel()
        if not sel_model or not self._right_click_saved:
            return
        sel_model.clearSelection()
        for item in self._right_click_saved:
            idx = self.indexFromItem(item)
            if idx.isValid():
                sel_model.select(idx, QItemSelectionModel.Select)


class ProjectExplorer(QWidget):
    """Left panel: XSF file browser + Toolbox."""

    # Signals
    file_selected = Signal(list)  # list of XsfMetadata
    tool_requested = Signal(str)  # tool name
    files_added = Signal(list)  # list of XsfMetadata
    go_to_location = Signal(str)  # filepath
    file_toggled = Signal(str, bool)  # filepath, visible
    file_removed = Signal(str)  # filepath (removed from list)
    dtm_layer_changed = Signal(str)  # 'elevation', 'backscatter', or '' for none
    product_action = Signal(str, str)  # action, filepath  (e.g., 'goto_dtm', 'geotiff_dtm')
    dtm_gamma_changed = Signal(float)  # gamma value
    dtm_shoulder_changed = Signal(float)  # shoulder compression strength [0, 1]

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._metadata_cache: dict = {}  # filepath -> XsfMetadata
        self._populating = False
        self._last_right_click_files: List[str] = []  # fallback for context menu
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QToolBar()
        add_btn = QPushButton("+ \u6dfb\u52a0\u6587\u4ef6")
        add_btn.clicked.connect(self._add_files)
        toolbar.addWidget(add_btn)
        add_dir_btn = QPushButton("+ \u6dfb\u52a0\u6587\u4ef6\u5939")
        add_dir_btn.clicked.connect(self._add_folder)
        toolbar.addWidget(add_dir_btn)
        layout.addWidget(toolbar)

        self._tabs = QTabWidget()

        # Tab 1
        self._file_tree = _SelectionPreservingTreeWidget()
        self._file_tree.setHeaderLabels(["\u540d\u79f0", "\u5927\u5c0f", "\u72b6\u6001"])
        self._file_tree.setColumnWidth(0, 250)
        self._file_tree.setColumnWidth(1, 80)
        self._file_tree.setColumnWidth(2, 80)
        self._file_tree.setRootIsDecorated(True)
        self._file_tree.setAlternatingRowColors(True)
        self._file_tree.setSelectionMode(QTreeWidget.ExtendedSelection)
        self._file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._file_tree.customContextMenuRequested.connect(self._context_menu)
        self._file_tree.itemClicked.connect(self._on_file_clicked)
        self._file_tree.itemChanged.connect(self._on_item_changed)
        self._file_tree.itemDoubleClicked.connect(self._on_double_click)

        self._root_xsf = QTreeWidgetItem(self._file_tree, ["XSF \u6587\u4ef6", "", ""])
        self._root_xsf.setData(0, Qt.UserRole, ITEM_ROOT)

        self._cat_processed = QTreeWidgetItem(self._root_xsf, ["\u5df2\u5904\u7406 (\u7eff\u8272)", "", ""])
        self._cat_processed.setData(0, Qt.UserRole, ITEM_CATEGORY)
        self._cat_processed.setForeground(0, QBrush(QColor("#4ec94e")))

        self._cat_unprocessed = QTreeWidgetItem(self._root_xsf, ["\u672a\u5904\u7406 (\u7070\u8272)", "", ""])
        self._cat_unprocessed.setData(0, Qt.UserRole, ITEM_CATEGORY)
        self._cat_unprocessed.setForeground(0, QBrush(QColor("#808080")))

        self._root_dtm = QTreeWidgetItem(self._file_tree, ["参考地形 (DTM)", "", ""])
        self._root_dtm.setData(0, Qt.UserRole, ITEM_ROOT)

        self._root_bsar = QTreeWidgetItem(self._file_tree, ["角度响应模型 (BSAR)", "", ""])
        self._root_bsar.setData(0, Qt.UserRole, ITEM_ROOT)

        self._root_products = QTreeWidgetItem(self._file_tree, ["输出结果 (镶嵌图/网格)", "", ""])
        self._root_products.setData(0, Qt.UserRole, ITEM_ROOT)

        self._file_tree.expandAll()
        self._tabs.addTab(self._file_tree, "\u6587\u4ef6")

        # Tab 2
        self._toolbox_widget = QWidget()
        toolbox_layout = QVBoxLayout(self._toolbox_widget)

        toolbox_layout.addWidget(QLabel("\u9636\u6bb5 1 \u2014 \u6570\u636e\u51c6\u5907"))
        btn_t1 = QPushButton("\u5bfc\u51fa\u53c2\u8003 DTM")
        btn_t1.clicked.connect(lambda: self.tool_requested.emit("tool1"))
        toolbox_layout.addWidget(btn_t1)

        toolbox_layout.addWidget(QLabel("\u9636\u6bb5 2 \u2014 \u89d2\u5ea6\u54cd\u5e94\u8865\u507f"))
        btn_stats = QPushButton("\u9759\u6001\u89d2\u5ea6\u91cd\u89c4\u8303\u5316")
        btn_stats.clicked.connect(lambda: self.tool_requested.emit("tool2b_step1"))
        toolbox_layout.addWidget(btn_stats)

        btn_t2a = QPushButton("\u6ed1\u52a8\u89d2\u5ea6\u91cd\u89c4\u8303\u5316")
        btn_t2a.clicked.connect(lambda: self.tool_requested.emit("tool2a"))
        toolbox_layout.addWidget(btn_t2a)

        btn_t2b_s2 = QPushButton("\u7edf\u8ba1\u89d2\u54cd\u5e94\uff08BSAR\uff09")
        btn_t2b_s2.clicked.connect(lambda: self.tool_requested.emit("tool2b_step2"))
        toolbox_layout.addWidget(btn_t2b_s2)

        toolbox_layout.addStretch()
        self._tabs.addTab(self._toolbox_widget, "\u5de5\u5177\u7bb1")

        layout.addWidget(self._tabs)

        dtm_grp = QGroupBox("DTM \u56fe\u5c42")
        dtm_layout = QVBoxLayout(dtm_grp)
        dtm_layout.setContentsMargins(4, 4, 4, 4)
        self._dtm_none = QCheckBox("\u65e0")
        self._dtm_none.setChecked(True)
        self._dtm_bs = QCheckBox("\u540e\u5411\u6563\u5c04")
        self._dtm_elev = QCheckBox("\u6c34\u6df1")
        self._dtm_none.toggled.connect(lambda: self._on_dtm_toggle(""))
        self._dtm_bs.toggled.connect(lambda: self._on_dtm_toggle("backscatter"))
        self._dtm_elev.toggled.connect(lambda: self._on_dtm_toggle("elevation"))
        dtm_layout.addWidget(self._dtm_none)
        dtm_layout.addWidget(self._dtm_bs)
        dtm_layout.addWidget(self._dtm_elev)
        dtm_layout.addWidget(QLabel("\u4f3d\u9a6c\u8c03\u8282:"))
        gamma_row = QHBoxLayout()
        self._gamma_slider = QSlider(Qt.Horizontal)
        self._gamma_slider.setRange(-100, 100)
        self._gamma_slider.setValue(0)
        self._gamma_slider.setTickInterval(10)
        self._gamma_slider.setToolTip("\u5de6\u6697\u53f3\u4eae\uff0c\u4e2d\u95f4=\u539f\u59cb")
        self._gamma_label = QLabel("1.0")
        self._gamma_label.setMinimumWidth(30)
        gamma_row.addWidget(self._gamma_slider)
        gamma_row.addWidget(self._gamma_label)
        dtm_layout.addLayout(gamma_row)
        self._gamma_slider.valueChanged.connect(self._on_gamma_changed)

        dtm_layout.addWidget(QLabel("\u80a9\u90e8\u538b\u7f29 (\u538b\u6697\u4eae\u5e26):"))
        shoulder_row = QHBoxLayout()
        self._shoulder_slider = QSlider(Qt.Horizontal)
        self._shoulder_slider.setRange(0, 100)
        self._shoulder_slider.setValue(0)
        self._shoulder_slider.setTickInterval(10)
        self._shoulder_slider.setToolTip("\u503c\u8d8a\u5927\u4eae\u5e26\u538b\u8d8a\u6697\uff0c0=\u5173\u95ed")
        self._shoulder_label = QLabel("0%")
        self._shoulder_label.setMinimumWidth(30)
        shoulder_row.addWidget(self._shoulder_slider)
        shoulder_row.addWidget(self._shoulder_label)
        dtm_layout.addLayout(shoulder_row)
        self._shoulder_slider.valueChanged.connect(self._on_shoulder_changed)
        layout.addWidget(dtm_grp)

    def _on_gamma_changed(self, value: int) -> None:
        """Map -100..100 → gamma 0.33..3.0, center 0 → 1.0."""
        gamma = 3.0 ** (value / 100.0)
        self._gamma_label.setText(f"{gamma:.2f}")
        self.dtm_gamma_changed.emit(gamma)

    def _on_shoulder_changed(self, value: int) -> None:
        """Map 0..100 → shoulder strength 0.0..1.0."""
        strength = value / 100.0
        self._shoulder_label.setText(f"{value}%")
        self.dtm_shoulder_changed.emit(strength)

    def _on_dtm_toggle(self, layer: str) -> None:
        """Uncheck others when one is selected."""
        for cb in (self._dtm_none, self._dtm_bs, self._dtm_elev):
            cb.blockSignals(True)
        if layer == "":
            self._dtm_none.setChecked(True)
            self._dtm_bs.setChecked(False)
            self._dtm_elev.setChecked(False)
            self._gamma_slider.setEnabled(False)
            self._shoulder_slider.setEnabled(False)
        elif layer == "backscatter":
            self._dtm_none.setChecked(False)
            self._dtm_bs.setChecked(True)
            self._dtm_elev.setChecked(False)
            self._gamma_slider.setEnabled(True)
            self._shoulder_slider.setEnabled(True)
        elif layer == "elevation":
            self._dtm_none.setChecked(False)
            self._dtm_bs.setChecked(False)
            self._dtm_elev.setChecked(True)
            self._gamma_slider.setEnabled(True)
            self._shoulder_slider.setEnabled(True)
        for cb in (self._dtm_none, self._dtm_bs, self._dtm_elev):
            cb.blockSignals(False)
        self.dtm_layer_changed.emit(layer)

    def _add_files(self) -> None:
        """\u6253\u5f00\u6587\u4ef6\u5bf9\u8bdd\u6846\u6dfb\u52a0 XSF \u6587\u4ef6\u3002"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "\u6dfb\u52a0 XSF \u6587\u4ef6", "",
            "XSF \u6587\u4ef6 (*.xsf.nc *.nc);;\u6240\u6709\u6587\u4ef6 (*.*)"
        )
        if files:
            self._load_files(files)

    def _add_folder(self) -> None:
        """\u6253\u5f00\u6587\u4ef6\u5939\u5bf9\u8bdd\u6846\u626b\u63cf XSF \u6587\u4ef6\u3002"""
        folder = QFileDialog.getExistingDirectory(self, "\u9009\u62e9\u542b\u6709 XSF \u6587\u4ef6\u7684\u6587\u4ef6\u5939")
        if folder:
            results = scan_directory(folder)
            if results:
                self._add_to_tree(results)
                self.files_added.emit(results)
            else:
                QMessageBox.information(self, "\u65e0\u6587\u4ef6", "\u5728\u6240\u9009\u6587\u4ef6\u5939\u4e2d\u672a\u627e\u5230 XSF \u6587\u4ef6\u3002")

    def _load_files(self, filepaths: List[str]) -> None:
        """Load metadata from a list of file paths."""
        results = []
        for fp in filepaths:
            if fp not in self._metadata_cache:
                try:
                    meta = read_xsf_metadata(fp)
                    self._metadata_cache[fp] = meta
                except Exception as e:
                    QMessageBox.warning(self, "\u8bfb\u53d6\u9519\u8bef", f"\u65e0\u6cd5\u8bfb\u53d6 {os.path.basename(fp)}:\n{e}")
                    continue
            results.append(self._metadata_cache[fp])

        if results:
            self._add_to_tree(results)
            self.files_added.emit(results)

    def _add_to_tree(self, metadata_list: List[XsfMetadata]) -> None:
        """Add metadata items to the file tree with checkboxes for nav toggle."""
        self._populating = True  # block itemChanged signals during population
        for meta in metadata_list:
            self._metadata_cache[meta.filepath] = meta  # populate cache for _selected_metadata()
            item = QTreeWidgetItem()
            item.setText(0, meta.filename)
            item.setText(1, f"{meta.filesize_mb:.0f} MB")
            item.setText(2, "PROC" if meta.is_processed else "RAW")
            item.setData(0, Qt.UserRole, ITEM_FILE)
            item.setData(0, Qt.UserRole + 1, meta.filepath)
            item.setToolTip(0, f"文件: {meta.filepath}\n声纳类型: {meta.sounder_type}\n"
                              f"版本: {meta.xsf_version}\n创建时间: {meta.date_created}")
            # Add checkbox in column 0 (checked = nav visible)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked)

            if meta.is_processed:
                item.setForeground(0, QBrush(QColor("#4ec94e")))
                self._cat_processed.addChild(item)
            else:
                item.setForeground(0, QBrush(QColor("#808080")))
                self._cat_unprocessed.addChild(item)
        self._populating = False  # re-enable signals

    def _context_menu(self, pos) -> None:
        """\u663e\u793a\u53f3\u952e\u4e0a\u4e0b\u6587\u83dc\u5355\u3002"""
        item = self._file_tree.itemAt(pos)
        if not item:
            return
        filepath = item.data(0, Qt.UserRole + 1)
        if not filepath:
            return
        is_product = filepath.endswith(".dtm.nc")

        saved_selection = set()
        for si in self._file_tree.selectedItems():
            fp = si.data(0, Qt.UserRole + 1)
            if fp:
                saved_selection.add(fp)
        if filepath not in saved_selection:
            saved_selection = {filepath}

        menu = QMenu(self)

        action_go = menu.addAction("\u5b9a\u4f4d\u5230\u4f4d\u7f6e")
        action_go.triggered.connect(lambda: self.go_to_location.emit(filepath))

        if is_product:
            menu.addSeparator()
            action_tiff = menu.addAction("\u5bfc\u51fa GeoTIFF")
            action_tiff.triggered.connect(lambda: self.product_action.emit("geotiff_dtm", filepath))
            action_xyz = menu.addAction("导出到 XYZ")
            action_xyz.triggered.connect(lambda: self.product_action.emit("xyz_dtm", filepath))

        menu.addSeparator()
        action_remove = menu.addAction("\u4ece\u5217\u8868\u4e2d\u79fb\u9664")
        action_remove.triggered.connect(
            lambda checked, fp_list=list(saved_selection): self._remove_files_from_list(fp_list))

        if not is_product:
            menu.addSeparator()
            self._last_right_click_files = list(saved_selection)
            self._file_tree.setCurrentItem(item, 0, QItemSelectionModel.Current)
            if filepath in self._metadata_cache:
                self.file_selected.emit([self._metadata_cache[filepath]])

            action_t1 = menu.addAction("\u5bfc\u51fa\u53c2\u8003 DTM")
            action_t1.triggered.connect(lambda: self.tool_requested.emit("tool1"))

            angle_menu = menu.addMenu("\u89d2\u5ea6\u54cd\u5e94")
            action_stats = angle_menu.addAction("\u9759\u6001\u89d2\u5ea6\u91cd\u89c4\u8303\u5316")
            action_stats.triggered.connect(lambda: self.tool_requested.emit("tool2b_step1"))
            action_sliding = angle_menu.addAction("\u6ed1\u52a8\u89d2\u5ea6\u91cd\u89c4\u8303\u5316")
            action_sliding.triggered.connect(lambda: self.tool_requested.emit("tool2a"))
            action_static = angle_menu.addAction("\u7edf\u8ba1\u89d2\u54cd\u5e94\uff08BSAR\uff09")
            action_static.triggered.connect(lambda: self.tool_requested.emit("tool2b_step2"))

        menu.exec(self._file_tree.viewport().mapToGlobal(pos))

    @staticmethod
    def _find_item_by_filepath(parent_item, filepath):
        """Recursively search tree for an item with the given filepath."""
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            if child.data(0, Qt.UserRole + 1) == filepath:
                return child
            found = ProjectExplorer._find_item_by_filepath(child, filepath)
            if found:
                return found
        return None

    def _on_file_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Emit file_selected when user LEFT-clicks a file in the tree."""
        # Skip right-clicks (itemClicked fires for all buttons; check widget flag)
        if getattr(self._file_tree, '_right_click_in_progress', False):
            return
        self._last_right_click_files = []
        filepath = item.data(0, Qt.UserRole + 1)
        if filepath and filepath in self._metadata_cache:
            self.file_selected.emit([self._metadata_cache[filepath]])

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        """Toggle nav track visibility when checkbox state changes."""
        if self._populating or column != 0:
            return
        filepath = item.data(0, Qt.UserRole + 1)
        if filepath:
            visible = item.checkState(0) == Qt.Checked
            self.file_toggled.emit(filepath, visible)

    def _on_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        """Show metadata popup on double-click."""
        filepath = item.data(0, Qt.UserRole + 1)
        if filepath and filepath in self._metadata_cache:
            meta = self._metadata_cache[filepath]
            info = (
                f"文件: {meta.filename}\n"
                f"路径: {meta.filepath}\n"
                f"大小: {meta.filesize_mb:.0f} MB\n"
                f"声纳类型: {meta.sounder_type}\n"
                f"XSF版本: {meta.xsf_version}\n"
                f"创建时间: {meta.date_created}\n"
                f"后向散射处理状态: {'已处理' if meta.is_processed else '未处理'}\n"
            )
            if meta.nav_bounds:
                nb = meta.nav_bounds
                info += (
                    f"导航边界:\n"
                    f"  经度: [{nb['lon_min']:.4f}, {nb['lon_max']:.4f}]\n"
                    f"  纬度: [{nb['lat_min']:.4f}, {nb['lat_max']:.4f}]\n"
                )
            QMessageBox.information(self, "XSF 元数据", info)

    def _remove_files_from_list(self, filepaths: List[str], is_product: bool = False) -> None:
        """Remove one or more files from the tree list after confirmation."""
        if not filepaths:
            return
        from PySide6.QtWidgets import QMessageBox, QCheckBox as QCBox

        # ── Confirmation dialog ──
        if len(filepaths) == 1:
            msg = QMessageBox(self)
            msg.setWindowTitle("从列表中移除文件")
            msg.setText(f"是否从项目中移除 {os.path.basename(filepaths[0])}？")
            cb = QCBox("同时从磁盘中删除文件（不可恢复）")
            cb.setChecked(False)
            msg.setCheckBox(cb)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.No)
            if msg.exec() != QMessageBox.Yes:
                return
            delete_from_disk = cb.isChecked()
        else:
            msg = QMessageBox(self)
            msg.setWindowTitle("从列表中移除多个文件")
            msg.setText(f"是否从项目中移除 {len(filepaths)} 个文件？")
            cb = QCBox("同时从磁盘中删除文件（不可恢复）")
            cb.setChecked(False)
            msg.setCheckBox(cb)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.No)
            if msg.exec() != QMessageBox.Yes:
                return
            delete_from_disk = cb.isChecked()

        for filepath in filepaths:
            # Find and remove the item from the tree (any depth)
            removed = False
            for i in range(self._file_tree.topLevelItemCount()):
                root = self._file_tree.topLevelItem(i)
                for j in range(root.childCount()):
                    cat = root.child(j)
                    # Direct child of root (e.g. DTM, BSAR, Products)
                    if cat.data(0, Qt.UserRole + 1) == filepath:
                        root.removeChild(cat)
                        removed = True
                        break
                    # Grand-children under a category (e.g. XSF files)
                    for k in range(cat.childCount()):
                        child = cat.child(k)
                        if child.data(0, Qt.UserRole + 1) == filepath:
                            cat.removeChild(child)
                            removed = True
                            break
                    if removed:
                        break
                if removed:
                    break
            # Also remove from metadata cache if present
            self._metadata_cache.pop(filepath, None)
            # Delete from disk if checked
            if delete_from_disk and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass
            self._last_right_click_files = [f for f in self._last_right_click_files if f != filepath]
            self.file_removed.emit(filepath)

    def _selected_metadata(self) -> List[XsfMetadata]:
        """Get metadata for selected items. Falls back to filepath-only entries."""
        results = []
        for item in self._file_tree.selectedItems():
            filepath = item.data(0, Qt.UserRole + 1)
            if not filepath:
                continue
            if filepath in self._metadata_cache:
                results.append(self._metadata_cache[filepath])
            else:
                # Create a minimal entry with just the filepath
                from .core.xsf_reader import XsfMetadata
                meta = XsfMetadata(filepath=filepath, filename=os.path.basename(filepath))
                results.append(meta)
        return results

    def get_selected_files(self) -> List[str]:
        """Get file paths of selected XSF files.
        
        After right-click, uses the context menu's stored file list
        (immune to menu.exec() event loop side effects).
        """
        if self._last_right_click_files:
            return list(self._last_right_click_files)
        return [m.filepath for m in self._selected_metadata()]

    def add_external_file(self, filepath: str, category: str = "") -> None:
        """Add an external file (DTM, BSAR, product) to the tree with checkbox."""
        if not os.path.exists(filepath):
            return
        fname = os.path.basename(filepath)
        root_map = {
            "dtm": self._root_dtm,
            "bsar": self._root_bsar,
            "product": self._root_products,
        }
        parent = root_map.get(category, self._root_products)
        item = QTreeWidgetItem(parent, [fname, "", ""])
        item.setData(0, Qt.UserRole + 1, filepath)
        item.setToolTip(0, filepath)
        # Add checkbox (default checked)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(0, Qt.Checked)
