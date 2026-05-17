"""\u5de5\u5177 2A\uff1a\u6ed1\u52a8\u89d2\u5ea6\u91cd\u89c4\u8303\u5316\u5bf9\u8bdd\u6846\u3002

\u72ec\u7acb\u7a97\u53e3\uff0c\u63d0\u4f9b\u6ed1\u52a8\u7a97\u53e3 BL4 \u6b63\u89c4\u5316\u53c2\u6570\u8bbe\u7f6e\u3002
\u540e\u7aef\uff1abs_sliding_angular_renormalization.json
"""
from typing import Optional, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout,
    QComboBox, QPushButton, QCheckBox, QLabel,
    QGroupBox, QDialogButtonBox, QListWidget,
    QSpinBox, QDoubleSpinBox, QMessageBox, QFileDialog, QLineEdit,
    QWidget, QHBoxLayout,
)

from ..utils.config import SOUNDER_TYPES


class Tool2ADialog(QDialog):
    """\u5de5\u5177 2A\u5bf9\u8bdd\u6846\uff1a\u6ed1\u52a8\u89d2\u5ea6\u91cd\u89c4\u8303\u5316\u3002"""

    def __init__(self, selected_files: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selected_files = selected_files
        self._bathy_nc = ""
        self._setup_ui()
        self.setWindowTitle("\u5de5\u5177 2A\uff1a\u6ed1\u52a8\u89d2\u5ea6\u91cd\u89c4\u8303\u5316")
        self.resize(520, 550)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(f"\u8f93\u5165 XSF \u6587\u4ef6: {len(self._selected_files)} \u4e2a\u5df2\u9009"))
        fl = QListWidget()
        for f in self._selected_files[:8]:
            fl.addItem(f)
        fl.setMaximumHeight(70)
        layout.addWidget(fl)

        dtm_layout = QHBoxLayout()
        dtm_layout.addWidget(QLabel("\u53c2\u8003 DTM (_bathy.nc):"))
        self._dtm_edit = QLineEdit()
        self._dtm_edit.setPlaceholderText("Tool 1 \u8f93\u51fa\u7684 _bathy.nc \u6587\u4ef6\u8def\u5f84")
        dtm_layout.addWidget(self._dtm_edit)
        dtm_btn = QPushButton("\u6d4f\u89c8...")
        dtm_btn.clicked.connect(self._browse_dtm)
        dtm_layout.addWidget(dtm_btn)
        layout.addLayout(dtm_layout)

        basic = QGroupBox("\u57fa\u672c\u53c2\u6570")
        bf = QFormLayout()
        self._sounder = QComboBox()
        self._sounder.addItems(SOUNDER_TYPES)
        self._sounder.setCurrentText("AUTO")
        bf.addRow("\u58f0\u7eb3\u7c7b\u578b:", self._sounder)

        self._window = QSpinBox()
        self._window.setRange(1, 120)
        self._window.setValue(10)
        self._window.setSuffix(" \u5206\u949f")
        self._window.setToolTip("\u4ee5\u5f53\u524d\u626b\u63cf\u7ebf\u4e3a\u4e2d\u5fc3\u7684\u6ed1\u52a8\u65f6\u95f4\u7a97\u53e3\uff08\u5206\u949f\uff09")
        bf.addRow("\u6ed1\u52a8\u65f6\u95f4\u7a97:", self._window)

        self._ref_min = QDoubleSpinBox()
        self._ref_min.setRange(0, 90)
        self._ref_min.setValue(30)
        self._ref_min.setSuffix("\u00b0")
        bf.addRow("\u53c2\u8003\u5165\u5c04\u89d2\u6700\u5c0f:", self._ref_min)

        self._ref_max = QDoubleSpinBox()
        self._ref_max.setRange(0, 90)
        self._ref_max.setValue(60)
        self._ref_max.setSuffix("\u00b0")
        bf.addRow("\u53c2\u8003\u5165\u5c04\u89d2\u6700\u5927:", self._ref_max)
        basic.setLayout(bf)
        layout.addWidget(basic)

        adv = QGroupBox("\u9ad8\u7ea7 \u2014 BL0/BL2 \u6821\u6b63")
        adv.setCheckable(True)
        adv.setChecked(False)
        af = QFormLayout()
        self._use_snippets = QCheckBox("\u4f7f\u7528\u7c92\u5ea6\u5747\u503c")
        af.addRow(self._use_snippets)
        self._use_svp = QCheckBox("\u4f7f\u7528 SVP \u6298\u5c04\u6821\u6b63")
        self._use_svp.setChecked(True)
        af.addRow(self._use_svp)
        self._use_insonified = QCheckBox("Ifremer \u7167\u5c04\u533a\u57df\u4f30\u7b97")
        self._use_insonified.setChecked(True)
        af.addRow(self._use_insonified)
        self._remove_cal = QCheckBox("\u79fb\u9664 BSCorr \u6821\u51c6")
        self._remove_cal.setChecked(True)
        af.addRow(self._remove_cal)
        adv.setLayout(af)
        layout.addWidget(adv)

        out_grp = QGroupBox("\u53ef\u9009\u8f93\u51fa")
        of = QFormLayout()
        self._out_bsar = QLineEdit()
        self._out_bsar.setPlaceholderText("\u53ef\u9009\uff1a\u8f93\u51fa .bsar.nc \u7528\u4e8e\u68c0\u67e5")
        out_bsar_btn = QPushButton("\u6d4f\u89c8...")
        out_bsar_btn.clicked.connect(self._browse_out_bsar)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_bsar)
        out_row.addWidget(out_bsar_btn)
        of.addRow("\u8f93\u51fa BSAR:", out_row)
        out_grp.setLayout(of)
        layout.addWidget(out_grp)

        out_dir_layout = QHBoxLayout()
        out_dir_layout.addWidget(QLabel("\u8f93\u51fa\u76ee\u5f55:"))
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("\u4e0d\u586b\u5219\u4e0e\u8f93\u5165\u6587\u4ef6\u540c\u76ee\u5f55")
        out_dir_layout.addWidget(self._out_dir)
        out_dir_btn = QPushButton("\u6d4f\u89c8...")
        out_dir_btn.clicked.connect(self._browse_out_dir)
        out_dir_layout.addWidget(out_dir_btn)
        layout.addLayout(out_dir_layout)

        layout.addStretch()

        bb = QDialogButtonBox()
        save_btn = QPushButton("\u4ec5\u4fdd\u5b58\u914d\u7f6e")
        save_btn.clicked.connect(lambda: self.done(2))
        bb.addButton(save_btn, QDialogButtonBox.ActionRole)
        run_btn = QPushButton("\u8fd0\u884c")
        run_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; }")
        run_btn.clicked.connect(self.accept)
        bb.addButton(run_btn, QDialogButtonBox.AcceptRole)
        bb.addButton(QPushButton("\u53d6\u6d88"), QDialogButtonBox.RejectRole)
        layout.addWidget(bb)

    def _browse_dtm(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "\u9009\u62e9 DTM \u6587\u4ef6", "", "NC files (*.nc);;All (*.*)")
        if f:
            self._dtm_edit.setText(f)

    def _browse_out_bsar(self) -> None:
        f, _ = QFileDialog.getSaveFileName(self, "\u8f93\u51fa BSAR \u6587\u4ef6", "", "NC files (*.nc)")
        if f:
            self._out_bsar.setText(f)

    def _browse_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "\u9009\u62e9\u6821\u6b63\u540e XSF \u7684\u8f93\u51fa\u76ee\u5f55")
        if d:
            self._out_dir.setText(d)

    @property
    def bathy_nc(self) -> str:
        return self._dtm_edit.text()

    def get_params(self) -> dict:
        p = {
            "sounder_type": self._sounder.currentText(),
            "sliding_window": self._window.value(),
            "ref_angle_min": self._ref_min.value(),
            "ref_angle_max": self._ref_max.value(),
            "use_snippets": self._use_snippets.isChecked(),
            "use_svp": self._use_svp.isChecked(),
            "use_insonified_area": self._use_insonified.isChecked(),
            "remove_calibration": self._remove_cal.isChecked(),
            "output_bsar": self._out_bsar.text() or None,
        }
        out_dir = self._out_dir.text().strip()
        if out_dir:
            p["output_dir"] = out_dir
        return p
