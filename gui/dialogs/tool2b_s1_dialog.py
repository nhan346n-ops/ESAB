"""Tool 2B Step 2a: Statistical BSAR dialog.

Independent window for BSAR (Backscatter Angular Response) model generation.
Backend: avg_backscatter_model.json → stats_computer.compute_mean_model_process
"""
from typing import Optional, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QComboBox, QPushButton, QCheckBox, QLabel,
    QGroupBox, QDialogButtonBox, QListWidget,
    QLineEdit, QFileDialog, QWidget,
)

from ..utils.config import SOUNDER_TYPES, INTEGRATION_METHODS, LINEAR_SCALES


class Tool2BS1Dialog(QDialog):
    """Dialog for Tool 2B Step 2a: Statistical BSAR model generation."""

    def __init__(self, selected_files: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selected_files = selected_files
        self._mask_file = ""
        self._setup_ui()
        self.setWindowTitle("Tool 2B ①: Statistical BSAR Model")
        self.resize(520, 520)

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

        self._integration = QComboBox()
        self._integration.addItems(INTEGRATION_METHODS)
        self._integration.setCurrentText("MEAN")
        bf.addRow("Integration Method:", self._integration)

        self._scale = QComboBox()
        self._scale.addItems(LINEAR_SCALES)
        self._scale.setCurrentText("AMPLITUDE")
        bf.addRow("Linear Scale:", self._scale)
        basic.setLayout(bf)
        layout.addWidget(basic)

        # Advanced BL0-BL2
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
        self._remove_comp = QCheckBox("Remove Kongsberg Compensation")
        self._remove_comp.setChecked(True)
        af.addRow(self._remove_comp)
        self._remove_cal = QCheckBox("Remove BSCorr Calibration")
        self._remove_cal.setChecked(True)
        af.addRow(self._remove_cal)
        adv.setLayout(af)
        layout.addWidget(adv)

        # Spatial mask
        mask_grp = QGroupBox("Spatial Mask (optional)")
        mg = QFormLayout()
        mask_row = QHBoxLayout()
        self._mask_edit = QLineEdit()
        self._mask_edit.setPlaceholderText("KML/SHP mask file (draw on Map View)")
        mask_row.addWidget(self._mask_edit)
        mask_btn = QPushButton("Browse...")
        mask_btn.clicked.connect(self._browse_mask)
        mask_row.addWidget(mask_btn)
        mg.addRow("Mask File:", mask_row)
        mask_grp.setLayout(mg)
        layout.addWidget(mask_grp)

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

    def _browse_mask(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Select Mask file", "", "KML/SHP (*.kml *.shp);;All (*.*)")
        if f:
            self._mask_edit.setText(f)

    @property
    def bathy_nc(self) -> str:
        return self._dtm_edit.text()

    def get_params(self) -> dict:
        p = {
            "sounder_type": self._sounder.currentText(),
            "integration_method": self._integration.currentText(),
            "linear_scale": self._scale.currentText(),
            "use_snippets": self._use_snippets.isChecked(),
            "use_svp": self._use_svp.isChecked(),
            "use_insonified_area": self._use_insonified.isChecked(),
            "remove_compensation": self._remove_comp.isChecked(),
            "remove_calibration": self._remove_cal.isChecked(),
        }
        mask = self._mask_edit.text()
        if mask:
            p["mask_files"] = [mask]
        return p
