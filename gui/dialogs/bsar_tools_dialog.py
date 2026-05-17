"""BSAR \u8f85\u52a9\u5de5\u5177\u5bf9\u8bdd\u6846\u3002

\u63d0\u4f9b CSV \u5bfc\u5165\u3001BSAR \u5408\u5e76\u3001BSAR \u62c6\u5206\u548c\u6a21\u5f0f\u6982\u8981\u5bfc\u51fa\u529f\u80fd\u3002
\u540e\u7aef\uff1amean_bs_processes.py
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
        l.addRow("\u8f93\u5165 CSV:", self._inp_row(self._inp, "\u9009\u62e9 CSV \u6587\u4ef6"))
        self._out = QLineEdit()
        l.addRow("\u8f93\u51fa BSAR:", self._out_row(self._out, "\u4fdd\u5b58 BSAR \u6587\u4ef6"))

    def _inp_row(self, edit, title):
        w = QWidget(); r = QHBoxLayout(w); r.setContentsMargins(0,0,0,0)
        r.addWidget(edit)
        b = QPushButton("\u6d4f\u89c8..."); b.clicked.connect(lambda: self._browse(edit, title, "CSV (*.csv);;All (*.*)", False)); r.addWidget(b)
        return w

    def _out_row(self, edit, title):
        w = QWidget(); r = QHBoxLayout(w); r.setContentsMargins(0,0,0,0)
        r.addWidget(edit)
        b = QPushButton("\u6d4f\u89c8..."); b.clicked.connect(lambda: self._browse(edit, title, "NC (*.nc)", True)); r.addWidget(b)
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
        l.addRow("\u8f93\u5165 BSAR \u6587\u4ef6\uff08\u7a7a\u683c\u5206\u9694\uff09:", self._inps)
        self._out = QLineEdit()
        o = QWidget(); r = QHBoxLayout(o); r.setContentsMargins(0,0,0,0); r.addWidget(self._out)
        b = QPushButton("\u6d4f\u89c8..."); b.clicked.connect(lambda: self._browse()); r.addWidget(b)
        l.addRow("\u8f93\u51fa:", o)

    def _browse(self):
        f, _ = QFileDialog.getSaveFileName(self, "\u8f93\u51fa\u5408\u5e76\u7684 BSAR", "", "NC (*.nc)")
        if f: self._out.setText(f)

    def get_params(self): return {"i_paths": self._inps.text().split(), "o_path": self._out.text()}


class _SplitBsarTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        l = QFormLayout(self)
        self._inp = QLineEdit()
        r = QWidget(); h = QHBoxLayout(r); h.setContentsMargins(0,0,0,0); h.addWidget(self._inp)
        h.addWidget(QPushButton("\u6d4f\u89c8...", clicked=lambda: self._browse()))
        l.addRow("\u8f93\u5165 BSAR:", r)
        self._out = QLineEdit()
        r2 = QWidget(); h2 = QHBoxLayout(r2); h2.setContentsMargins(0,0,0,0); h2.addWidget(self._out)
        h2.addWidget(QPushButton("\u6d4f\u89c8...", clicked=lambda: self._browse_dir()))
        l.addRow("\u8f93\u51fa\u76ee\u5f55:", r2)

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "\u9009\u62e9 BSAR \u6587\u4ef6", "", "NC (*.bsar.nc *.nc);;All (*.*)")
        if f: self._inp.setText(f)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "\u9009\u62e9\u8f93\u51fa\u76ee\u5f55")
        if d: self._out.setText(d)

    def get_params(self): return {"i_path": self._inp.text(), "o_dir": self._out.text()}


class BsarToolsDialog(QDialog):
    """BSAR \u8f85\u52a9\u5de5\u5177\u5bf9\u8bdd\u6846\u3002"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("BSAR \u8f85\u52a9\u5de5\u5177")
        self.resize(500, 350)

        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        self._csv_tab = _CsvToBsarTab()
        tabs.addTab(self._csv_tab, "CSV \u2192 BSAR")

        self._merge_tab = _MergeBsarTab()
        tabs.addTab(self._merge_tab, "\u5408\u5e76 BSAR")

        self._split_tab = _SplitBsarTab()
        tabs.addTab(self._split_tab, "\u6309\u6a21\u5f0f\u62c6\u5206")

        layout.addWidget(tabs)

        bb = QDialogButtonBox()
        run_btn = QPushButton("\u8fd0\u884c")
        run_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; }")
        run_btn.clicked.connect(self.accept)
        bb.addButton(run_btn, QDialogButtonBox.AcceptRole)
        bb.addButton(QPushButton("\u53d6\u6d88"), QDialogButtonBox.RejectRole)
        layout.addWidget(bb)

    def get_active_tool(self) -> str:
        """\u8fd4\u56de\u5f53\u524d\u6d3b\u52a8\u6807\u7b7e\u9875\u5bf9\u5e94\u7684\u5de5\u5177\u540d\u79f0\u3002"""
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
