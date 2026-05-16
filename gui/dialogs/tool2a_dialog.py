"""Tool 2A: Sliding Angular Renormalization dialog.

Independent window with parameter form for sliding window BL4 normalization.
Backend: bs_sliding_angular_renormalization.json
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
    """Dialog for Tool 2A: Sliding Angular Renormalization."""

    def __init__(self, selected_files: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selected_files = selected_files
        self._bathy_nc = ""
        self._setup_ui()
        self.setWindowTitle("Tool 2A: Sliding Angular Renormalization")
        self.resize(520, 550)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Selected files
        layout.addWidget(QLabel(f"Input XSF Files: {len(self._selected_files)} selected"))
        fl = QListWidget()
        for f in self._selected_files[:8]:
            fl.addItem(f)
        fl.setMaximumHeight(70)
        layout.addWidget(fl)

        # Reference DTM
        dtm_layout = QHBoxLayout()
        dtm_layout.addWidget(QLabel("Reference DTM (_bathy.nc):"))
        self._dtm_edit = QLineEdit()
        self._dtm_edit.setPlaceholderText("Path to _bathy.nc from Tool 1")
        dtm_layout.addWidget(self._dtm_edit)
        dtm_btn = QPushButton("Browse...")
        dtm_btn.clicked.connect(self._browse_dtm)
        dtm_layout.addWidget(dtm_btn)
        layout.addLayout(dtm_layout)

        # Sounder type
        basic = QGroupBox("Basic Parameters")
        bf = QFormLayout()
        self._sounder = QComboBox()
        self._sounder.addItems(SOUNDER_TYPES)
        self._sounder.setCurrentText("AUTO")
        bf.addRow("Sounder Type:", self._sounder)

        self._window = QSpinBox()
        self._window.setRange(1, 120)
        self._window.setValue(10)
        self._window.setSuffix(" min")
        self._window.setToolTip("Sliding window centered on current swath (minutes)")
        bf.addRow("Sliding Window:", self._window)

        self._ref_min = QDoubleSpinBox()
        self._ref_min.setRange(0, 90)
        self._ref_min.setValue(30)
        self._ref_min.setSuffix("°")
        bf.addRow("Ref Angle Min:", self._ref_min)

        self._ref_max = QDoubleSpinBox()
        self._ref_max.setRange(0, 90)
        self._ref_max.setValue(60)
        self._ref_max.setSuffix("°")
        bf.addRow("Ref Angle Max:", self._ref_max)
        basic.setLayout(bf)
        layout.addWidget(basic)

        # Advanced BL0-BL2 params
        adv = QGroupBox("Advanced — BL0/BL2 Correction")
        adv.setCheckable(True)
        adv.setChecked(False)
        af = QFormLayout()
        self._use_snippets = QCheckBox("Use Snippet Mean")
        af.addRow(self._use_snippets)
        self._use_svp = QCheckBox("Use SVP Refraction")
        self._use_svp.setChecked(True)
        af.addRow(self._use_svp)
        self._use_insonified = QCheckBox("Ifremer Insonified Area")
        self._use_insonified.setChecked(True)
        af.addRow(self._use_insonified)
        self._remove_cal = QCheckBox("Remove BSCorr Calibration")
        self._remove_cal.setChecked(True)
        af.addRow(self._remove_cal)
        adv.setLayout(af)
        layout.addWidget(adv)

        # Output BSAR (optional)
        out_grp = QGroupBox("Optional Output")
        of = QFormLayout()
        self._out_bsar = QLineEdit()
        self._out_bsar.setPlaceholderText("Optional: output .bsar.nc for inspection")
        out_bsar_btn = QPushButton("Browse...")
        out_bsar_btn.clicked.connect(self._browse_out_bsar)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_bsar)
        out_row.addWidget(out_bsar_btn)
        of.addRow("Output BSAR:", out_row)
        out_grp.setLayout(of)
        layout.addWidget(out_grp)

        layout.addStretch()

        # Buttons
        bb = QDialogButtonBox()
        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(lambda: self.done(2))
        bb.addButton(save_btn, QDialogButtonBox.ActionRole)
        run_btn = QPushButton("Run")
        run_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; }")
        run_btn.clicked.connect(self.accept)
        bb.addButton(run_btn, QDialogButtonBox.AcceptRole)
        bb.addButton(QPushButton("Cancel"), QDialogButtonBox.RejectRole)
        layout.addWidget(bb)

    def _browse_dtm(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Select DTM file", "", "NC files (*.nc);;All (*.*)")
        if f:
            self._dtm_edit.setText(f)

    def _browse_out_bsar(self) -> None:
        f, _ = QFileDialog.getSaveFileName(self, "Output BSAR file", "", "NC files (*.nc)")
        if f:
            self._out_bsar.setText(f)

    @property
    def bathy_nc(self) -> str:
        return self._dtm_edit.text()

    def get_params(self) -> dict:
        return {
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
