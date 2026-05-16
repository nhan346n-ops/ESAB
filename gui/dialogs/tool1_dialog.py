"""Tool 1: Export Reference DTM dialog.

Parameters: sounder type, projection, resolution, gap fill,
elevation filter, BL1/BL2 correction flags.
Generates JSON config and launches pyat subprocess.
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
    """Dialog for Tool 1: Export Reference DTM & Uncorrected BS Preview."""

    def __init__(
        self,
        selected_files: List[str],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._selected_files = selected_files
        self._setup_ui()
        self.setWindowTitle("Tool 1: Export Reference DTM")
        self.resize(550, 600)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Selected files display
        layout.addWidget(QLabel(f"Selected XSF Files: {len(self._selected_files)}"))
        self._file_list = QListWidget()
        for f in self._selected_files[:10]:  # Show first 10
            self._file_list.addItem(f)
        if len(self._selected_files) > 10:
            self._file_list.addItem(f"... and {len(self._selected_files) - 10} more")
        self._file_list.setMaximumHeight(80)
        layout.addWidget(self._file_list)

        # Basic parameters
        basic_group = QGroupBox("Basic Parameters")
        basic_form = QFormLayout()

        self._sounder_type = QComboBox()
        self._sounder_type.addItems(SOUNDER_TYPES)
        self._sounder_type.setCurrentText("AUTO")
        basic_form.addRow("Sounder Type:", self._sounder_type)

        self._projection = QComboBox()
        self._projection.addItems(PROJECTIONS)
        self._projection.setCurrentText("Auto Detect")
        basic_form.addRow("Projection:", self._projection)

        self._resolution = QComboBox()
        self._resolution.addItems(RESOLUTIONS)
        self._resolution.setCurrentText("2.0")
        self._resolution.setEditable(True)
        basic_form.addRow("Resolution (m):", self._resolution)

        self._gap_fill = QComboBox()
        self._gap_fill.addItems(GAP_FILL_METHODS)
        self._gap_fill.setCurrentText("None")
        basic_form.addRow("Gap Filling:", self._gap_fill)

        self._elev_min = QDoubleSpinBox()
        self._elev_min.setRange(-12000, 12000)
        self._elev_min.setValue(-12000)
        self._elev_min.setSpecialValueText("No limit")
        self._elev_min.setDecimals(1)
        basic_form.addRow("Elevation Min:", self._elev_min)

        self._elev_max = QDoubleSpinBox()
        self._elev_max.setRange(-12000, 12000)
        self._elev_max.setValue(12000)
        self._elev_max.setSpecialValueText("No limit")
        self._elev_max.setDecimals(1)
        basic_form.addRow("Elevation Max:", self._elev_max)

        basic_group.setLayout(basic_form)
        layout.addWidget(basic_group)

        # Advanced BL1/BL2 parameters (collapsible)
        adv_group = QGroupBox("Advanced — BL1/BL2 Correction Parameters")
        adv_group.setCheckable(True)
        adv_group.setChecked(False)
        adv_form = QFormLayout()

        self._use_snippets = QCheckBox("Use Snippet Mean (amplitude domain)")
        self._use_snippets.setToolTip(
            "Recompute mean backscatter from seabed image snippets "
            "instead of using pre-computed detection values."
        )
        adv_form.addRow(self._use_snippets)

        self._use_svp = QCheckBox("Use SVP Refraction Correction")
        self._use_svp.setChecked(True)
        self._use_svp.setToolTip(
            "Apply Snell's law refraction for incidence angle calculation "
            "using embedded sound velocity profiles."
        )
        adv_form.addRow(self._use_svp)

        self._use_insonified_area = QCheckBox("Apply Ifremer Insonified Area Estimation")
        self._use_insonified_area.setChecked(True)
        self._use_insonified_area.setToolTip(
            "Replace Kongsberg manufacturer insonified area estimate "
            "with Ifremer seafloor slope-aware computation."
        )
        adv_form.addRow(self._use_insonified_area)

        self._remove_compensation = QCheckBox("Remove Kongsberg Compensation (Lambert + Specular)")
        self._remove_compensation.setChecked(True)
        self._remove_compensation.setToolTip(
            "Remove Kongsberg real-time Lambert TVG and specular reflection "
            "compensation. Only applicable to Kongsberg .all/.km_app files."
        )
        adv_form.addRow(self._remove_compensation)

        self._remove_calibration = QCheckBox("Remove BSCorr Calibration")
        self._remove_calibration.setChecked(True)
        self._remove_calibration.setToolTip(
            "Remove Kongsberg BSCorr calibration data from kmall files."
        )
        adv_form.addRow(self._remove_calibration)

        adv_group.setLayout(adv_form)
        layout.addWidget(adv_group)

        # Output directory
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("Output Directory:"))
        self._output_dir = QLineEdit()
        self._output_dir.setReadOnly(True)
        out_layout.addWidget(self._output_dir)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output)
        out_layout.addWidget(browse_btn)
        layout.addLayout(out_layout)

        layout.addStretch()

        # Buttons
        btn_box = QDialogButtonBox()
        save_btn = QPushButton("Save Config Only")
        save_btn.clicked.connect(self._save_config)
        btn_box.addButton(save_btn, QDialogButtonBox.ActionRole)

        run_btn = QPushButton("Run")
        run_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; }")
        run_btn.clicked.connect(self._run)
        btn_box.addButton(run_btn, QDialogButtonBox.AcceptRole)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_box.addButton(cancel_btn, QDialogButtonBox.RejectRole)

        layout.addWidget(btn_box)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self._output_dir.setText(folder)

    def _save_config(self) -> None:
        """Save config only (emit accepted with run=False)."""
        self.done(2)  # Custom code: 2 = save only

    def _run(self) -> None:
        """Accept and run."""
        self.accept()

    # --- Getters for form values ---

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
        """True if user clicked Run, False if Save Only."""
        return self.result() == QDialog.Accepted
