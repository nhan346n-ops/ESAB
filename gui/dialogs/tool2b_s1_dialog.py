"""\u9759\u6001\u89d2\u5ea6\u91cd\u89c4\u8303\u5316\u591a\u9875\u5411\u5bfc\u3002

\u4f7f\u7528\u5df2\u8ba1\u7b97\u7684 BSAR \u6a21\u578b\u5e94\u7528\u5230 XSF \u6587\u4ef6\uff0c\u8f93\u51fa\u5df2\u6821\u6b63\u7684
XSF \u6587\u4ef6\uff08``_bs_renorm`` \u540e\u7f00\uff09\uff0c\u5e26\u6709 ``backscatterCorrection = ON`` \u6807\u8bb0\u3002
\u540e\u7aef\uff1asonar/bs/bs_angular_renormalization.json \u2192 xsf_constant_process

\u9875\u9762\uff1a
  1. \u8f93\u5165/\u8f93\u51fa  \u2014 BSAR \u6a21\u578b\u3001\u53c2\u8003 DTM\u3001\u8f93\u51fa\u76ee\u5f55
  2. \u53c2\u6570      \u2014 \u53c2\u8003\u7ea7\u522b\u3001\u8865\u507f\u3001\u7c92\u5ea6\u3001\u58f0\u7eb3\u7c7b\u578b\u3001\u79ef\u5206\u3001\u5c3a\u5ea6
  3. \u9ad8\u7ea7      \u2014 SVP\u3001\u7167\u5c04\u533a\u57df\u3001\u8865\u507f\u3001\u6821\u51c6
  4. \u603b\u7ed3      \u2014 \u68c0\u67e5\u5e76\u8fd0\u884c
"""
import os
from typing import Optional, List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QCheckBox, QDoubleSpinBox,
    QPushButton, QListWidget, QFileDialog, QGroupBox,
    QTextEdit, QWidget,
)

from ..utils.config import SOUNDER_TYPES, INTEGRATION_METHODS, LINEAR_SCALES


# ── Page 1: Input / Output ──────────────────────────────────────────

class _InputOutputPage(QWizardPage):
    def __init__(self, selected_files: List[str],
                 parent: Optional[QWizard] = None):
        super().__init__(parent)
        self.setTitle("Step 1: Input & Output")
        self.setSubTitle("Select the BSAR model, optional DTM, and output folder.")

        layout = QVBoxLayout(self)

        # Input files
        layout.addWidget(QLabel(f"XSF Files to process: {len(selected_files)}"))
        fl = QListWidget()
        for f in selected_files[:8]:
            fl.addItem(f)
        fl.setMaximumHeight(60)
        layout.addWidget(fl)

        # BSAR model (required)
        bsar_row = QHBoxLayout()
        bsar_row.addWidget(QLabel("BSAR Model (.bsar.nc):"))
        self._bsar_edit = QLineEdit()
        self._bsar_edit.setPlaceholderText("Required \u2014 pre-computed BSAR model")
        bsar_row.addWidget(self._bsar_edit)
        bsar_btn = QPushButton("Browse\u2026")
        bsar_btn.clicked.connect(self._browse_bsar)
        bsar_row.addWidget(bsar_btn)
        layout.addLayout(bsar_row)
        self.registerField("bsar*", self._bsar_edit)

        # Reference DTM (optional)
        dtm_row = QHBoxLayout()
        dtm_row.addWidget(QLabel("Reference DTM (_bathy.nc):"))
        self._dtm_edit = QLineEdit()
        self._dtm_edit.setPlaceholderText("Optional \u2014 for incidence-angle correction")
        dtm_row.addWidget(self._dtm_edit)
        dtm_btn = QPushButton("Browse\u2026")
        dtm_btn.clicked.connect(self._browse_dtm)
        dtm_row.addWidget(dtm_btn)
        layout.addLayout(dtm_row)

        # Output
        out_grp = QGroupBox("Output")
        of = QFormLayout(out_grp)
        dir_row = QHBoxLayout()
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("Leave empty to use input file directory")
        dir_row.addWidget(self._out_dir)
        browse_out = QPushButton("Browse\u2026")
        browse_out.clicked.connect(self._browse_out)
        dir_row.addWidget(browse_out)
        of.addRow("Directory:", dir_row)
        self._overwrite = QCheckBox("Overwrite existing output files")
        of.addRow(self._overwrite)
        layout.addWidget(out_grp)

    def _browse_bsar(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "Select BSAR model", "",
            "BSAR files (*.bsar.nc);;NetCDF files (*.nc);;All (*.*)")
        if f:
            self._bsar_edit.setText(f)

    def _browse_dtm(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "Select DTM file", "",
            "DTM files (*.dtm.nc *.nc);;All (*.*)")
        if f:
            self._dtm_edit.setText(f)

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d:
            self._out_dir.setText(d)

    def getBsarNc(self) -> str:
        return self._bsar_edit.text()

    def getBathyNc(self) -> str:
        return self._dtm_edit.text()

    def getOutputDir(self) -> str:
        return self._out_dir.text()

    def getOverwrite(self) -> bool:
        return self._overwrite.isChecked()


# ── Page 2: Parameters ──────────────────────────────────────────────

class _ParametersPage(QWizardPage):
    def __init__(self, parent: Optional[QWizard] = None):
        super().__init__(parent)
        self.setTitle("Step 2: Parameters")
        self.setSubTitle("Reference level, compensation and processing options.")

        layout = QFormLayout(self)

        # Reference level
        ref_row = QHBoxLayout()
        self._ref_level = QDoubleSpinBox()
        self._ref_level.setRange(-100, 0)
        self._ref_level.setValue(-20)
        self._ref_level.setSuffix(" dB")
        self._ref_level.setDecimals(1)
        ref_row.addWidget(self._ref_level)
        eval_btn = QPushButton("Evaluate")
        eval_btn.setToolTip("Auto-estimate reference level.")
        eval_btn.clicked.connect(self._on_evaluate)
        ref_row.addWidget(eval_btn)
        layout.addRow("Reference level:", ref_row)

        self._apply_comp = QCheckBox("Apply incidence angle compensation")
        self._apply_comp.setChecked(True)
        layout.addRow(self._apply_comp)

        self._snippets = QCheckBox("Use snippet mean")
        layout.addRow(self._snippets)

        # Separator
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("The following options are read from the BSAR model "
                             "and shown here for reference:"))

        self._sounder = QComboBox()
        self._sounder.addItems(SOUNDER_TYPES)
        self._sounder.setCurrentText("AUTO")
        self._sounder.setEnabled(False)
        layout.addRow("Sounder type:", self._sounder)

        self._integration = QComboBox()
        self._integration.addItems(INTEGRATION_METHODS)
        self._integration.setCurrentText("MEAN")
        self._integration.setEnabled(False)
        layout.addRow("Integration method:", self._integration)

        self._scale = QComboBox()
        self._scale.addItems(LINEAR_SCALES)
        self._scale.setCurrentText("AMPLITUDE")
        self._scale.setEnabled(False)
        layout.addRow("Linear scale:", self._scale)

    def _on_evaluate(self) -> None:
        wiz = self.wizard()
        if not wiz:
            return
        bsar_path = wiz._page1.getBsarNc()
        if not bsar_path or not os.path.isfile(bsar_path):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Missing BSAR Model",
                "Please select a BSAR model file in Step 1 first.")
            return

        from PySide6.QtWidgets import QMessageBox, QApplication
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            import sys
            _src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "src")
            if _src not in sys.path:
                sys.path.insert(0, _src)

            from pyat.sonarscope.function.evaluate_bs_reference_level import BSReferenceLevelEvaluator
            evaluator = BSReferenceLevelEvaluator(mean_model_file=bsar_path)
            result = evaluator()

            if result is not None and "result" in result:
                ref_level = result["result"].get("reference_level")
                if ref_level is not None:
                    self._ref_level.setValue(round(float(ref_level), 1))
                    QMessageBox.information(
                        self, "Evaluation Complete",
                        f"Estimated reference level: {ref_level:.1f} dB\n"
                        f"(weighted mean of all modes in the BSAR model)")
        except Exception as e:
            QMessageBox.warning(
                self, "Evaluation Failed",
                f"Could not evaluate reference level:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def getParams(self) -> dict:
        return {
            "reference_level": self._ref_level.value(),
            "apply_compensation": self._apply_comp.isChecked(),
            "use_snippets": self._snippets.isChecked(),
        }


# ── Page 3: Advanced Options ────────────────────────────────────────

class _AdvancedPage(QWizardPage):
    def __init__(self, parent: Optional[QWizard] = None):
        super().__init__(parent)
        self.setTitle("Step 3: Advanced Options")
        self.setSubTitle("BL0/BL2 correction flags (read from BSAR model, shown for reference).")

        layout = QFormLayout(self)

        self._use_svp = QCheckBox("Use embedded sound velocity profiles")
        self._use_svp.setChecked(True)
        self._use_svp.setEnabled(False)
        layout.addRow(self._use_svp)

        self._use_ia = QCheckBox("Recompute insonified area from seafloor incidence")
        self._use_ia.setChecked(True)
        self._use_ia.setEnabled(False)
        layout.addRow(self._use_ia)

        self._remove_comp = QCheckBox("Remove angular compensation")
        self._remove_comp.setChecked(True)
        self._remove_comp.setEnabled(False)
        layout.addRow(self._remove_comp)

        self._remove_cal = QCheckBox("Remove calibration (BScorr from kmall)")
        self._remove_cal.setChecked(True)
        self._remove_cal.setEnabled(False)
        layout.addRow(self._remove_cal)

    def getParams(self) -> dict:
        return {}


# ── Page 4: Summary ─────────────────────────────────────────────────

class _SummaryPage(QWizardPage):
    def __init__(self, parent: Optional[QWizard] = None):
        super().__init__(parent)
        self.setTitle("Step 4: Summary")
        self.setSubTitle("Review settings and run.")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Configuration summary:"))
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet(
            "QTextEdit{background:#1e1e1e;color:#d4d4d4;font-family:Consolas}")
        layout.addWidget(self._text)

    def setSummary(self, text: str) -> None:
        self._text.setText(text)


# ── Main Wizard ─────────────────────────────────────────────────────

class Tool2BS1Dialog(QWizard):
    """Multi-page wizard for 静态角度重规范化.

    Applies a pre-computed BSAR model to XSF files and produces
    corrected XSF files with ``backscatterCorrection = ON``.
    """

    def __init__(self, selected_files: List[str],
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("\u9759\u6001\u89d2\u5ea6\u91cd\u89c4\u8303\u5316")
        self.setMinimumSize(540, 490)
        self.setWizardStyle(QWizard.ModernStyle)

        self._selected_files = selected_files

        self._page1 = _InputOutputPage(selected_files, self)
        self._page2 = _ParametersPage(self)
        self._page3 = _AdvancedPage(self)
        self._page4 = _SummaryPage(self)

        self.addPage(self._page1)
        self.addPage(self._page2)
        self.addPage(self._page3)
        self.addPage(self._page4)

        self.currentIdChanged.connect(self._on_page_changed)
        self._on_page_changed(0)

    @property
    def bsar_nc(self) -> str:
        return self._page1.getBsarNc()

    @property
    def bathy_nc(self) -> str:
        return self._page1.getBathyNc()

    def getOutputDir(self) -> str:
        return self._page1.getOutputDir()

    def getOverwrite(self) -> bool:
        return self._page1.getOverwrite()

    def get_params(self) -> dict:
        return self._page2.getParams()

    def _on_page_changed(self, page_id: int) -> None:
        if page_id == 3:
            self._page4.setSummary(self._build_summary())

    def _build_summary(self) -> str:
        p1 = self._page1
        p2 = self._page2.getParams()
        lines = [
            "=== Input / Output ===",
            f"  XSF files:          {len(self._selected_files)}",
            f"  BSAR model:         {p1.getBsarNc()}",
            f"  Reference DTM:      {p1.getBathyNc() or '(not set)'}",
            f"  Output directory:   {p1.getOutputDir() or '(same as input)'}",
            f"  Overwrite:          {p1.getOverwrite()}",
            "",
            "=== Parameters ===",
            f"  Reference level:           {p2['reference_level']} dB",
            f"  Incidence compensation:    {p2['apply_compensation']}",
            f"  Use snippet mean:          {p2['use_snippets']}",
        ]
        return "\n".join(lines)