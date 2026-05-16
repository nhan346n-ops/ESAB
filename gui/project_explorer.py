"""Project Explorer panel with XSF file tree and Toolbox.

Left-side panel with QTabWidget:
  Tab 1: "Files" — QTreeWidget showing XSF files grouped by processing status
  Tab 2: "Toolbox" — processing tools organized by phase
"""
import os
from typing import Optional, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget,
    QTreeWidget, QTreeWidgetItem, QPushButton, QLabel,
    QFileDialog, QMessageBox, QMenu, QToolBar,
)

from .core.xsf_reader import XsfMetadata, read_xsf_metadata, scan_directory


# Item types for the tree
ITEM_ROOT = 0
ITEM_CATEGORY = 1
ITEM_FILE = 2


class ProjectExplorer(QWidget):
    """Left panel: XSF file browser + Toolbox."""

    # Signals
    file_selected = Signal(list)  # list of XsfMetadata
    tool_requested = Signal(str)  # tool name
    files_added = Signal(list)  # list of XsfMetadata
    go_to_location = Signal(str)  # filepath
    file_toggled = Signal(str, bool)  # filepath, visible

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._metadata_cache: dict = {}  # filepath -> XsfMetadata
        self._populating = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QToolBar()
        toolbar.addWidget(QLabel("Project Explorer"))
        add_btn = QPushButton("+ Add Files")
        add_btn.clicked.connect(self._add_files)
        toolbar.addWidget(add_btn)
        add_dir_btn = QPushButton("+ Add Folder")
        add_dir_btn.clicked.connect(self._add_folder)
        toolbar.addWidget(add_dir_btn)
        layout.addWidget(toolbar)

        # Tab widget
        self._tabs = QTabWidget()

        # Tab 1: Files
        self._file_tree = QTreeWidget()
        self._file_tree.setHeaderLabels(["Name", "Size", "Status"])
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

        # Root items
        self._root_xsf = QTreeWidgetItem(self._file_tree, ["XSF Files", "", ""])
        self._root_xsf.setData(0, Qt.UserRole, ITEM_ROOT)

        self._cat_processed = QTreeWidgetItem(self._root_xsf, ["Processed (green)", "", ""])
        self._cat_processed.setData(0, Qt.UserRole, ITEM_CATEGORY)
        self._cat_processed.setForeground(0, QBrush(QColor("#4ec94e")))

        self._cat_unprocessed = QTreeWidgetItem(self._root_xsf, ["Unprocessed (gray)", "", ""])
        self._cat_unprocessed.setData(0, Qt.UserRole, ITEM_CATEGORY)
        self._cat_unprocessed.setForeground(0, QBrush(QColor("#808080")))

        self._root_dtm = QTreeWidgetItem(self._file_tree, ["Reference DTM", "", ""])
        self._root_dtm.setData(0, Qt.UserRole, ITEM_ROOT)

        self._root_bsar = QTreeWidgetItem(self._file_tree, ["BSAR Models", "", ""])
        self._root_bsar.setData(0, Qt.UserRole, ITEM_ROOT)

        self._root_products = QTreeWidgetItem(self._file_tree, ["Products", "", ""])
        self._root_products.setData(0, Qt.UserRole, ITEM_ROOT)

        self._file_tree.expandAll()
        self._tabs.addTab(self._file_tree, "Files")

        # Tab 2: Toolbox
        self._toolbox_widget = QWidget()
        toolbox_layout = QVBoxLayout(self._toolbox_widget)

        # Phase 1 tools
        toolbox_layout.addWidget(QLabel("Phase 1 — Data Preparation"))
        btn_t1 = QPushButton("Tool 1: Export Reference DTM")
        btn_t1.clicked.connect(lambda: self.tool_requested.emit("tool1"))
        btn_t1.setStyleSheet("QPushButton { text-align: left; padding: 6px; }")
        toolbox_layout.addWidget(btn_t1)

        toolbox_layout.addWidget(QLabel("Phase 2 — Angle Response Compensation"))
        btn_t2a = QPushButton("Tool 2A: Sliding Renormalization")
        btn_t2a.clicked.connect(lambda: self.tool_requested.emit("tool2a"))
        btn_t2a.setStyleSheet("QPushButton { text-align: left; padding: 6px; }")
        toolbox_layout.addWidget(btn_t2a)

        btn_t2b_s1 = QPushButton("Tool 2B ①: Statistical BSAR")
        btn_t2b_s1.clicked.connect(lambda: self.tool_requested.emit("tool2b_step1"))
        btn_t2b_s1.setStyleSheet("QPushButton { text-align: left; padding: 6px; }")
        toolbox_layout.addWidget(btn_t2b_s1)

        btn_t2b_s2 = QPushButton("Tool 2B ②: Apply BSAR Renormalization")
        btn_t2b_s2.clicked.connect(lambda: self.tool_requested.emit("tool2b_step2"))
        btn_t2b_s2.setStyleSheet("QPushButton { text-align: left; padding: 6px; }")
        toolbox_layout.addWidget(btn_t2b_s2)

        toolbox_layout.addWidget(QLabel("Phase 4 — Mosaic Output"))
        btn_t3 = QPushButton("Tool 3: Grid Backscatter Mosaic")
        btn_t3.clicked.connect(lambda: self.tool_requested.emit("tool3"))
        btn_t3.setStyleSheet("QPushButton { text-align: left; padding: 6px; }")
        toolbox_layout.addWidget(btn_t3)

        toolbox_layout.addStretch()
        self._tabs.addTab(self._toolbox_widget, "Toolbox")

        layout.addWidget(self._tabs)

    def _add_files(self) -> None:
        """Open file dialog to add XSF files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add XSF Files", "",
            "XSF Files (*.xsf.nc *.nc);;All Files (*.*)"
        )
        if files:
            self._load_files(files)

    def _add_folder(self) -> None:
        """Open folder dialog to scan for XSF files."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with XSF Files")
        if folder:
            results = scan_directory(folder)
            if results:
                self._add_to_tree(results)
                self.files_added.emit(results)
            else:
                QMessageBox.information(self, "No Files", "No XSF files found in the selected folder.")

    def _load_files(self, filepaths: List[str]) -> None:
        """Load metadata from a list of file paths."""
        results = []
        for fp in filepaths:
            if fp not in self._metadata_cache:
                try:
                    meta = read_xsf_metadata(fp)
                    self._metadata_cache[fp] = meta
                except Exception as e:
                    QMessageBox.warning(self, "Read Error", f"Failed to read {os.path.basename(fp)}:\n{e}")
                    continue
            results.append(self._metadata_cache[fp])

        if results:
            self._add_to_tree(results)
            self.files_added.emit(results)

    def _add_to_tree(self, metadata_list: List[XsfMetadata]) -> None:
        """Add metadata items to the file tree with checkboxes for nav toggle."""
        self._populating = True  # block itemChanged signals during population
        for meta in metadata_list:
            item = QTreeWidgetItem()
            item.setText(0, meta.filename)
            item.setText(1, f"{meta.filesize_mb:.0f} MB")
            item.setText(2, "PROC" if meta.is_processed else "RAW")
            item.setData(0, Qt.UserRole, ITEM_FILE)
            item.setData(0, Qt.UserRole + 1, meta.filepath)
            item.setToolTip(0, f"File: {meta.filepath}\nSounder: {meta.sounder_type}\n"
                              f"Version: {meta.xsf_version}\nCreated: {meta.date_created}")
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
        """Show right-click context menu on file items."""
        item = self._file_tree.itemAt(pos)
        if not item:
            return
        filepath = item.data(0, Qt.UserRole + 1)
        if not filepath:
            return  # Not a file node (root, category, etc.)

        menu = QMenu(self)

        action_go = menu.addAction("Go to Location")
        action_go.triggered.connect(lambda: self.go_to_location.emit(filepath))

        menu.addSeparator()

        # Select this item so tool handlers can find it via get_selected_files()
        item.setSelected(True)
        # Also emit file_selected so map highlights this track
        if filepath in self._metadata_cache:
            self.file_selected.emit([self._metadata_cache[filepath]])

        action_t1 = menu.addAction("Tool 1: Export Reference DTM")
        action_t1.triggered.connect(lambda: self.tool_requested.emit("tool1"))

        action_t2a = menu.addAction("Tool 2A: Sliding Renormalization")
        action_t2a.triggered.connect(lambda: self.tool_requested.emit("tool2a"))

        action_t2b = menu.addAction("Tool 2B: Static Renormalization")
        action_t2b.triggered.connect(lambda: self.tool_requested.emit("tool2b_step1"))

        menu.addSeparator()

        action_t3 = menu.addAction("Tool 3: Grid Backscatter Mosaic")
        action_t3.triggered.connect(lambda: self.tool_requested.emit("tool3"))

        menu.exec(self._file_tree.viewport().mapToGlobal(pos))

    def _on_file_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Emit file_selected when user clicks a file in the tree."""
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
                f"File: {meta.filename}\n"
                f"Path: {meta.filepath}\n"
                f"Size: {meta.filesize_mb:.0f} MB\n"
                f"Sounder Type: {meta.sounder_type}\n"
                f"XSF Version: {meta.xsf_version}\n"
                f"Created: {meta.date_created}\n"
                f"Backscatter Correction: {'PROCESSED' if meta.is_processed else 'UNPROCESSED'}\n"
            )
            if meta.nav_bounds:
                nb = meta.nav_bounds
                info += (
                    f"Navigation Bounds:\n"
                    f"  Lon: [{nb['lon_min']:.4f}, {nb['lon_max']:.4f}]\n"
                    f"  Lat: [{nb['lat_min']:.4f}, {nb['lat_max']:.4f}]\n"
                )
            QMessageBox.information(self, "XSF Metadata", info)

    def _selected_metadata(self) -> List[XsfMetadata]:
        """Get metadata for selected items."""
        results = []
        for item in self._file_tree.selectedItems():
            filepath = item.data(0, Qt.UserRole + 1)
            if filepath and filepath in self._metadata_cache:
                results.append(self._metadata_cache[filepath])
        return results

    def get_selected_files(self) -> List[str]:
        """Get file paths of selected XSF files."""
        return [m.filepath for m in self._selected_metadata()]

    def add_external_file(self, filepath: str, category: str = "") -> None:
        """Add an external file (DTM, BSAR, product) to the tree."""
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
