"""Tool 2B Step 2b: Static Angular Renormalization dialog.

Apply BSAR model to XSF files for BL4 normalization.
Backend: bs_angular_renormalization.json -> angular_renormalization.xsf_constant_process
"""
from typing import Optional, List

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QCheckBox, QLabel,
    QGroupBox, QDialogButtonBox, QListWidget,
    QLineEdit, QFileDialog, QDoubleSpinBox, QWidget,
    QMessageBox,
)


class Tool2BS2Dialog(QDialog):
    """Dialog for Tool 2B Step 2b: Apply BSAR Renormalization."""

    def __init__(self, selected_files: List[str], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._selected_files = selected_files
        self._setup_ui()
        self.setWindowTitle("Tool 2B \u2461: Apply BSAR Renormalization")
        self.resize(480, 420)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Selected files
        layout.addWidget(QLabel(f"Input XSF Files: {len(self._selected_files)} selected"))
        fl = QListWidget()
        for f in self._selected_files[:8]:
            fl.addItem(f)
        fl.setMaximumHeight(60)
        layout.addWidget(fl)

        # BSAR model (required)
        bsar_layout = QHBoxLayout()
        bsar_layout.addWidget(QLabel("BSAR Model (.bsar.nc):"))
        self._bsar_edit = QLineEdit()
        self._bsar_edit.setPlaceholderText("Select .bsar.nc from Tool 2B Step 2a")
        bsar_layout.addWidget(self._bsar_edit)
        bsar_btn = QPushButton("Browse...")
        bsar_btn.clicked.connect(self._browse_bsar)
        bsar_layout.addWidget(bsar_btn)
        layout.addLayout(bsar_layout)

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

        # Parameters
        params = QGroupBox("Normalization Parameters")
        pf = QFormLayout()

        # Reference level with Evaluate button
        ref_layout = QHBoxLayout()
        self._ref_level = QDoubleSpinBox()
        self._ref_level.setRange(-100, 0)
        self._ref_level.setValue(-20)
        self._ref_level.setSuffix(" dB")
        self._ref_level.setDecimals(1)
        ref_layout.addWidget(self._ref_level)

        eval_btn = QPushButton("Evaluate")
        eval_btn.setToolTip(
            "Auto-estimate reference level from BSAR model.\n"
            "Calls evaluate_mean_bs_level.json internally."
        )
        eval_btn.clicked.connect(self._on_evaluate)
        ref_layout.addWidget(eval_btn)

        pf.addRow("Reference Level:", ref_layout)

        self._apply_comp = QCheckBox("Apply Incidence Angle Compensation")
        self._apply_comp.setChecked(True)
        self._apply_comp.setToolTip(
            "Remove incidence angular dependency. When unchecked, only "
            "transmission angle residual correction is applied."
        )
        pf.addRow(self._apply_comp)

        self._use_snippets = QCheckBox("Use Snippet Mean")
        pf.addRow(self._use_snippets)

        params.setLayout(pf)
        layout.addWidget(params)

        layout.addStretch()

        # Buttons
        bb = QDialogButtonBox()
        bb.addButton(QPushButton("Save Config"), QDialogButtonBox.ActionRole).clicked.connect(lambda: self.done(2))
        run_btn = QPushButton("Run")
        run_btn.setStyleSheet("QPushButton { background-color: #0e639c; color: white; }")
        run_btn.clicked.connect(self.accept)
        bb.addButton(run_btn, QDialogButtonBox.AcceptRole)
        bb.addButton(QPushButton("Cancel"), QDialogButtonBox.RejectRole)
        layout.addWidget(bb)

    def _browse_bsar(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Select BSAR file", "", "NC files (*.bsar.nc *.nc);;All (*.*)")
        if f:
            self._bsar_edit.setText(f)

    def _browse_dtm(self) -> None:
        f, _ = QFileDialog.getOpenFileName(self, "Select DTM file", "", "NC files (*.nc);;All (*.*)")
        if f:
            self._dtm_edit.setText(f)

    def _on_evaluate(self) -> None:
        """Estimate reference level from BSAR model.

        Calls evaluate_mean_bs_level.json via subprocess.
        For now, shows a placeholder message since actual execution
        requires pyat subprocess integration.
        """
        QMessageBox.information(
            self, "Evaluate Reference Level",
            "Auto-evaluation will call:\n\n"
            "  python -m pyat <config_with_evaluate_function>\n\n"
            "Using evaluate_mean_bs_level.json template.\n"
            "Full subprocess integration pending QProcess implementation.\n\n"
            "For now, use default value -20 dB or enter manually."
        )

    @property
    def bsar_nc(self) -> str:
        return self._bsar_edit.text()

    @property
    def bathy_nc(self) -> str:
        return self._dtm_edit.text()

    def get_params(self) -> dict:
        return {
            "reference_level": self._ref_level.value(),
            "apply_compensation": self._apply_comp.isChecked(),
            "use_snippets": self._use_snippets.isChecked(),
        }
