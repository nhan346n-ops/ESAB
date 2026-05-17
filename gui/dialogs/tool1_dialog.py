"""\u5de5\u5177 1\uff1a\u5bfc\u51fa\u53c2\u8003 DTM \u5bf9\u8bdd\u6846\u3002

\u53c2\u6570\uff1a\u58f0\u7eb3\u7c7b\u578b\u3001\u6295\u5f71\u3001\u5206\u8fa8\u7387\u3001\u7f3a\u5931\u586b\u8865\u3001
\u6d77\u62d4\u6ee4\u6ce2\u3001BL1/BL2 \u6821\u6b63\u6807\u5fd7\u3002
\u751f\u6210 JSON \u914d\u7f6e\u6587\u4ef6\u5e76\u542f\u52a8 pyat \u5b50\u8fdb\u7a0b\u3002
"""
from typing import Optional, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QLineEdit, QPushButton, QCheckBox, QLabel,
    QGroupBox, QDialogButtonBox, QListWidget, QDoubleSpinBox,
    QFileDialog, QWidget,
)

from ..utils.config import (
    SOUNDER_TYPES, PROJECTIONS, RESOLUTIONS,
    GAP_FILL_METHODS,
)


class Tool1Dialog(QDialog):
    """\u5de5\u5177 1\u5bf9\u8bdd\u6846\uff1a\u5bfc\u51fa\u53c2\u8003 DTM \u4e0e\u672a\u6821\u6b63\u540e\u5411\u6563\u5c04\u9884\u89c8\u3002"""

    def __init__(
        self,
        selected_files: List[str],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._selected_files = selected_files
        self._setup_ui()
        self.setWindowTitle("\u5de5\u5177 1\uff1a\u5bfc\u51fa\u53c2\u8003 DTM")
        self.resize(550, 600)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(f"\u5df2\u9009 XSF \u6587\u4ef6: {len(self._selected_files)}"))
        self._file_list = QListWidget()
        for f in self._selected_files[:10]:
            self._file_list.addItem(f)
        if len(self._selected_files) > 10:
            self._file_list.addItem(f"... \u4ee5\u53ca\u5176\u4ed6 {len(self._selected_files) - 10} \u4e2a")
        self._file_list.setMaximumHeight(80)
        layout.addWidget(self._file_list)

        basic_group = QGroupBox("\u57fa\u672c\u53c2\u6570")
        basic_form = QFormLayout()

        self._sounder_type = QComboBox()
        self._sounder_type.addItems(SOUNDER_TYPES)
        self._sounder_type.setCurrentText("AUTO")
        basic_form.addRow("\u58f0\u7eb3\u7c7b\u578b:", self._sounder_type)

        self._projection = QComboBox()
        self._projection.addItems(PROJECTIONS)
        self._projection.setCurrentText("Auto Detect")
        basic_form.addRow("\u6295\u5f71:", self._projection)

        self._resolution = QComboBox()
        self._resolution.addItems(RESOLUTIONS)
        self._resolution.setCurrentText("2.0")
        self._resolution.setEditable(True)
        basic_form.addRow("\u5206\u8fa8\u7387 (m):", self._resolution)

        self._gap_fill = QComboBox()
        self._gap_fill.addItems(GAP_FILL_METHODS)
        self._gap_fill.setCurrentText("None")
        basic_form.addRow("\u7f3a\u5931\u586b\u8865:", self._gap_fill)

        self._elev_min = QDoubleSpinBox()
        self._elev_min.setRange(-12000, 12000)
        self._elev_min.setValue(-12000)
        self._elev_min.setSpecialValueText("\u65e0\u9650\u5236")
        self._elev_min.setDecimals(1)
        basic_form.addRow("\u6700\u5c0f\u6c34\u6df1 (m):", self._elev_min)

        self._elev_max = QDoubleSpinBox()
        self._elev_max.setRange(-12000, 12000)
        self._elev_max.setValue(12000)
        self._elev_max.setSpecialValueText("\u65e0\u9650\u5236")
        self._elev_max.setDecimals(1)
        basic_form.addRow("\u6700\u5927\u6c34\u6df1 (m):", self._elev_max)

        basic_group.setLayout(basic_form)
        layout.addWidget(basic_group)

        adv_group = QGroupBox("\u9ad8\u7ea7 \u2014 BL1/BL2 \u6821\u6b63\u53c2\u6570")
        adv_group.setCheckable(True)
        adv_group.setChecked(False)
        adv_form = QFormLayout()

        self._use_snippets = QCheckBox("\u4f7f\u7528\u7c92\u5ea6\u5747\u503c\uff08\u632f\u5e45\u57df\uff09")
        self._use_snippets.setToolTip(
            "\u4ece\u6d77\u5e95\u56fe\u50cf\u7c92\u5ea6\u91cd\u65b0\u8ba1\u7b97\u540e\u5411\u6563\u5c04\u5747\u503c\uff0c"
            "\u800c\u975e\u4f7f\u7528\u9884\u8ba1\u7b97\u7684\u68c0\u6d4b\u503c\u3002"
        )
        adv_form.addRow(self._use_snippets)

        self._use_svp = QCheckBox("\u4f7f\u7528 SVP \u6298\u5c04\u6821\u6b63")
        self._use_svp.setChecked(True)
        self._use_svp.setToolTip(
            "\u5e94\u7528\u65af\u6c85\u5c14\u5b9a\u5f8b\u5bf9\u5165\u5c04\u89d2\u8fdb\u884c\u6298\u5c04\u6821\u6b63\uff0c"
            "\u4f7f\u7528\u6587\u4ef6\u4e2d\u5d4c\u5165\u7684\u58f0\u901f\u5206\u5e03\u6570\u636e\u3002"
        )
        adv_form.addRow(self._use_svp)

        self._use_insonified_area = QCheckBox("\u5e94\u7528 Ifremer \u7167\u5c04\u533a\u57df\u4f30\u7b97")
        self._use_insonified_area.setChecked(True)
        self._use_insonified_area.setToolTip(
            "\u66ff\u6362 Kongsberg \u5382\u5546\u7167\u5c04\u533a\u57df\u4f30\u7b97\uff0c"
            "\u4f7f\u7528 Ifremer \u7684\u6d77\u5e95\u5761\u5ea6\u81ea\u9002\u5e94\u8ba1\u7b97\u3002"
        )
        adv_form.addRow(self._use_insonified_area)

        self._remove_compensation = QCheckBox("\u79fb\u9664 Kongsberg \u8865\u507f\uff08Lambert + \u955c\u9762\u53cd\u5c04\uff09")
        self._remove_compensation.setChecked(True)
        self._remove_compensation.setToolTip(
            "\u79fb\u9664 Kongsberg \u5b9e\u65f6 Lambert TVG \u548c\u955c\u9762\u53cd\u5c04\u8865\u507f\u3002"
            "\u4ec5\u9002\u7528\u4e8e Kongsberg .all/.kmall \u6587\u4ef6\u3002"
        )
        adv_form.addRow(self._remove_compensation)

        self._remove_calibration = QCheckBox("\u79fb\u9664 BSCorr \u6821\u51c6")
        self._remove_calibration.setChecked(True)
        self._remove_calibration.setToolTip(
            "\u79fb\u9664 Kongsberg kmall \u6587\u4ef6\u4e2d\u7684 BSCorr \u6821\u51c6\u6570\u636e\u3002"
        )
        adv_form.addRow(self._remove_calibration)

        adv_group.setLayout(adv_form)
        layout.addWidget(adv_group)

        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("\u8f93\u51fa\u76ee\u5f55:"))
        self._output_dir = QLineEdit()
        self._output_dir.setReadOnly(True)
        out_layout.addWidget(self._output_dir)
        browse_btn = QPushButton("\u6d4f\u89c8...")
        browse_btn.clicked.connect(self._browse_output)
        out_layout.addWidget(browse_btn)
        layout.addLayout(out_layout)

        layout.addStretch()

        btn_box = QDialogButtonBox()
        save_btn = QPushButton("\u4ec5\u4fdd\u5b58\u914d\u7f6e")
        save_btn.clicked.connect(self._save_config)
        btn_box.addButton(save_btn, QDialogButtonBox.ActionRole)

        run_btn = QPushButton("\u8fd0\u884c")
        run_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; }")
        run_btn.clicked.connect(self._run)
        btn_box.addButton(run_btn, QDialogButtonBox.AcceptRole)

        cancel_btn = QPushButton("\u53d6\u6d88")
        cancel_btn.clicked.connect(self.reject)
        btn_box.addButton(cancel_btn, QDialogButtonBox.RejectRole)

        layout.addWidget(btn_box)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "\u9009\u62e9\u8f93\u51fa\u76ee\u5f55")
        if folder:
            self._output_dir.setText(folder)

    def _save_config(self) -> None:
        self.done(2)

    def _run(self) -> None:
        self.accept()

    def get_params(self) -> dict:
        return {
            "sounder_type": self._sounder_type.currentText(),
            "projection": self._projection.currentText(),
            "resolution": self._resolution.currentText(),
            "gap_fill": self._gap_fill.currentText(),
            "elev_min": self._elev_min.value() if self._elev_min.value() != self._elev_min.minimum() else None,
            "elev_max": self._elev_max.value() if self._elev_max.value() != self._elev_max.maximum() else None,
            "use_snippets": self._use_snippets.isChecked(),
            "use_svp": self._use_svp.isChecked(),
            "use_insonified_area": self._use_insonified_area.isChecked(),
            "remove_compensation": self._remove_compensation.isChecked(),
            "remove_calibration": self._remove_calibration.isChecked(),
            "output_dir": self._output_dir.text() or None,
        }

    def get_should_run(self) -> bool:
        return self.result() == QDialog.Accepted
