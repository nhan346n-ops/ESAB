"""BSAR Auxiliary Tools dialog.

Provides CSV import, BSAR merge, BSAR split, and mode summary export.
Backend: mean_bs_processes.py functions.
"""
from typing import Optional, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QCheckBox, QLabel, QLineEdit,
    QGroupBox, QDialogButtonBox, QFileDialog, QTabWidget,
    QWidget, QMessageBox,
)


class _CsvToBsarTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        l = QFormLayout(self)
        self._inp = QLineEdit()
        l.addRow("Input CSV:", self._inp_row(self._inp, "Select CSV file"))
        self._out = QLineEdit()
        l.addRow("Output BSAR:", self._out_row(self._out, "Save BSAR as"))

    def _inp_row(self, edit, title):
        w = QWidget(); r = QHBoxLayout(w); r.setContentsMargins(0,0,0,0)
        r.addWidget(edit)
        b = QPushButton("Browse..."); b.clicked.connect(lambda: self._browse(edit, title, "CSV (*.csv);;All (*.*)", False)); r.addWidget(b)
        return w

    def _out_row(self, edit, title):
        w = QWidget(); r = QHBoxLayout(w); r.setContentsMargins(0,0,0,0)
        r.addWidget(edit)
        b = QPushButton("Browse..."); b.clicked.connect(lambda: self._browse(edit, title, "NC (*.nc)", True)); r.addWidget(b)
        return w

    def _browse(self, edit, title, filt, save):
        if save:
            f, _ = QFileDialog.getSaveFileName(self, title, "", filt)
        else:
            f, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        if f: edit.setText(f)

    def get_params(self): return {"i_path": self._inp.text(), "o_path": self._out.text()}


class _MergeBsarTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        l = QFormLayout(self)
        self._inps = QLineEdit()
        self._inps.setPlaceholderText("file1.bsar.nc file2.bsar.nc ...")
        l.addRow("Input BSAR files (space-separated):", self._inps)
        self._out = QLineEdit()
        o = QWidget(); r = QHBoxLayout(o); r.setContentsMargins(0,0,0,0); r.addWidget(self._out)
        b = QPushButton("Browse..."); b.clicked.connect(lambda: self._browse()); r.addWidget(b)
        l.addRow("Output:", o)

    def _browse(self):
        f, _ = QFileDialog.getSaveFileName(self, "Output merged BSAR", "", "NC (*.nc)")
        if f: self._out.setText(f)

    def get_params(self): return {"i_paths": self._inps.text().split(), "o_path": self._out.text()}


class _SplitBsarTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        l = QFormLayout(self)
        self._inp = QLineEdit()
        r = QWidget(); h = QHBoxLayout(r); h.setContentsMargins(0,0,0,0); h.addWidget(self._inp)
        h.addWidget(QPushButton("Browse...", clicked=lambda: self._browse()))
        l.addRow("Input BSAR:", r)
        self._out = QLineEdit()
        r2 = QWidget(); h2 = QHBoxLayout(r2); h2.setContentsMargins(0,0,0,0); h2.addWidget(self._out)
        h2.addWidget(QPushButton("Browse...", clicked=lambda: self._browse_dir()))
        l.addRow("Output Dir:", r2)

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select BSAR", "", "NC (*.bsar.nc *.nc);;All (*.*)")
        if f: self._inp.setText(f)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d: self._out.setText(d)

    def get_params(self): return {"i_path": self._inp.text(), "o_dir": self._out.text()}


class BsarToolsDialog(QDialog):
    """Dialog for BSAR auxiliary tools."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("BSAR Auxiliary Tools")
        self.resize(500, 350)

        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        self._csv_tab = _CsvToBsarTab()
        tabs.addTab(self._csv_tab, "CSV \u2192 BSAR")

        self._merge_tab = _MergeBsarTab()
        tabs.addTab(self._merge_tab, "Merge BSAR")

        self._split_tab = _SplitBsarTab()
        tabs.addTab(self._split_tab, "Split by Mode")

        layout.addWidget(tabs)

        bb = QDialogButtonBox()
        run_btn = QPushButton("Run")
        run_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; }")
        run_btn.clicked.connect(self.accept)
        bb.addButton(run_btn, QDialogButtonBox.AcceptRole)
        bb.addButton(QPushButton("Cancel"), QDialogButtonBox.RejectRole)
        layout.addWidget(bb)

    def get_active_tool(self) -> str:
        """Return 'csv', 'merge', or 'split' based on active tab."""
        idx = self.findChild(QTabWidget).currentIndex()
        return ["csv", "merge", "split"][idx]

    def get_active_params(self) -> dict:
        tool = self.get_active_tool()
        if tool == "csv":
            return self._csv_tab.get_params()
        elif tool == "merge":
            return self._merge_tab.get_params()
        elif tool == "split":
            return self._split_tab.get_params()
        return {}
