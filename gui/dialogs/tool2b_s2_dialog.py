"""Multi-page wizard for 统计角响应（BSAR） model computation.

Computes a BSAR (Backscatter Angular Response) model from XSF input files.
Backend: sonar/bs/avg_backscatter_model.json → stats_computer.compute_mean_model_process

Pages:
  1. Input/Output  — input XSF files, output .bsar.nc path, optional reference DTM
  2. Parameters    — sounder type, integration method, linear scale
  3. Advanced      — SVP, insonified area, compensation, calibration
  4. Summary       — review & execute
"""
from typing import Optional, List

from PySide6.QtWidgets import (
    QWizard, QWizardPage, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QCheckBox,
    QPushButton, QListWidget, QFileDialog, QGroupBox,
    QTextEdit, QWidget,
)

from ..utils.config import SOUNDER_TYPES, INTEGRATION_METHODS, LINEAR_SCALES


# ── Page 1: Input / Output ──────────────────────────────────────────

class _InputOutputPage(QWizardPage):
    def __init__(self, selected_files: List[str],
                 parent: Optional[QWizard] = None):
        super().__init__(parent)
        self.setTitle("\u6b65\u9aa4 1\uff1a\u8f93\u5165 & \u8f93\u51fa")
        self.setSubTitle("\u9009\u62e9 XSF \u6587\u4ef6\u5e76\u8bbe\u7f6e\u8f93\u51fa BSAR \u6587\u4ef6\u8def\u5f84\u3002")

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel(f"\u8f93\u5165 XSF \u6587\u4ef6: {len(selected_files)}"))
        fl = QListWidget()
        for f in selected_files[:8]:
            fl.addItem(f)
        fl.setMaximumHeight(60)
        layout.addWidget(fl)

        out_grp = QGroupBox("\u8f93\u51fa BSAR \u6a21\u578b")
        of = QFormLayout(out_grp)

        bsar_row = QHBoxLayout()
        self._out_bsar = QLineEdit()
        self._out_bsar.setPlaceholderText("\u5fc5\u586b \u2014 \u751f\u6210\u7684 .bsar.nc \u6587\u4ef6\u8def\u5f84")
        bsar_row.addWidget(self._out_bsar)
        bsar_btn = QPushButton("\u6d4f\u89c8\u2026")
        bsar_btn.clicked.connect(self._browse_out_bsar)
        bsar_row.addWidget(bsar_btn)
        of.addRow("BSAR \u6587\u4ef6:", bsar_row)
        self.registerField("out_bsar*", self._out_bsar)

        layout.addWidget(out_grp)

        dtm_grp = QGroupBox("\u53c2\u8003 DTM\uff08\u53ef\u9009\uff09")
        dtm_form = QFormLayout(dtm_grp)
        dtm_row = QHBoxLayout()
        self._dtm_edit = QLineEdit()
        self._dtm_edit.setPlaceholderText("\u53ef\u9009 \u2014 _bathy.nc \u7528\u4e8e\u5165\u5c04\u89d2\u6821\u6b63")
        dtm_row.addWidget(self._dtm_edit)
        dtm_btn = QPushButton("\u6d4f\u89c8\u2026")
        dtm_btn.clicked.connect(self._browse_dtm)
        dtm_row.addWidget(dtm_btn)
        dtm_form.addRow("DTM file:", dtm_row)
        layout.addWidget(dtm_grp)

    def _browse_out_bsar(self) -> None:
        f, _ = QFileDialog.getSaveFileName(
            self, "另存 BSAR 模型为", "bsar_model.bsar.nc",
            "BSAR files (*.bsar.nc);;NetCDF files (*.nc);;All (*.*)")
        if f:
            if not f.endswith(".bsar.nc") and not f.endswith(".nc"):
                f += ".bsar.nc"
            self._out_bsar.setText(f)

    def _browse_dtm(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "选择 DTM 文件", "",
            "DTM files (*.dtm.nc *.nc);;All (*.*)")
        if f:
            self._dtm_edit.setText(f)

    def getOutputBsar(self) -> str:
        return self._out_bsar.text()

    def getBathyNc(self) -> str:
        return self._dtm_edit.text()


# ── Page 2: Parameters ──────────────────────────────────────────────

class _ParametersPage(QWizardPage):
    def __init__(self, parent: Optional[QWizard] = None):
        super().__init__(parent)
        self.setTitle("\u6b65\u9aa4 2\uff1a\u53c2\u6570")
        self.setSubTitle("\u58f0\u7eb3\u7c7b\u578b\u3001\u79ef\u5206\u65b9\u6cd5\u548c\u7ebf\u6027\u5c3a\u5ea6\u3002")

        layout = QFormLayout(self)

        self._sounder = QComboBox()
        self._sounder.addItems(SOUNDER_TYPES)
        self._sounder.setCurrentText("AUTO")
        layout.addRow("\u58f0\u7eb3\u7c7b\u578b:", self._sounder)

        self._integration = QComboBox()
        self._integration.addItems(INTEGRATION_METHODS)
        self._integration.setCurrentText("MEAN")
        layout.addRow("\u79ef\u5206\u65b9\u6cd5:", self._integration)

        self._scale = QComboBox()
        self._scale.addItems(LINEAR_SCALES)
        self._scale.setCurrentText("AMPLITUDE")
        layout.addRow("\u7ebf\u6027\u5c3a\u5ea6:", self._scale)

        self._snippets = QCheckBox("\u4f7f\u7528\u7c92\u5ea6\u5747\u503c")
        self._snippets.setToolTip(
            "\u4ece\u7c92\u5ea6\u91cd\u65b0\u8ba1\u7b97\u540e\u5411\u6563\u5c04\uff0c\u800c\u975e\u4f7f\u7528\u68c0\u6d4b\u503c\u3002")
        layout.addRow(self._snippets)

    def getParams(self) -> dict:
        return {
            "sounder_type": self._sounder.currentText(),
            "integration_method": self._integration.currentText(),
            "linear_scale": self._scale.currentText(),
            "use_snippets": self._snippets.isChecked(),
        }


# ── Page 3: Advanced Options ────────────────────────────────────────

class _AdvancedPage(QWizardPage):
    def __init__(self, parent: Optional[QWizard] = None):
        super().__init__(parent)
        self.setTitle("\u6b65\u9aa4 3\uff1a\u9ad8\u7ea7\u9009\u9879")
        self.setSubTitle("BL0/BL2 \u6821\u6b63\u6807\u5fd7\u3002")

        layout = QFormLayout(self)

        self._use_svp = QCheckBox("\u4f7f\u7528\u5d4c\u5165\u5f0f\u58f0\u901f\u5206\u5e03")
        self._use_svp.setChecked(True)
        layout.addRow(self._use_svp)

        self._use_ia = QCheckBox("\u6839\u636e\u6d77\u5e95\u5165\u5c04\u89d2\u91cd\u65b0\u8ba1\u7b97\u7167\u5c04\u533a\u57df")
        self._use_ia.setChecked(True)
        layout.addRow(self._use_ia)

        self._remove_comp = QCheckBox("\u79fb\u9664\u89d2\u5ea6\u8865\u507f")
        self._remove_comp.setChecked(True)
        layout.addRow(self._remove_comp)

        self._remove_cal = QCheckBox("\u79fb\u9664\u6821\u51c6 (BScorr \u4ece kmall)")
        self._remove_cal.setChecked(True)
        layout.addRow(self._remove_cal)

    def getParams(self) -> dict:
        return {
            "use_svp": self._use_svp.isChecked(),
            "use_insonified_area": self._use_ia.isChecked(),
            "remove_compensation": self._remove_comp.isChecked(),
            "remove_calibration": self._remove_cal.isChecked(),
        }


# ── Page 4: Summary ─────────────────────────────────────────────────

class _SummaryPage(QWizardPage):
    def __init__(self, parent: Optional[QWizard] = None):
        super().__init__(parent)
        self.setTitle("\u6b65\u9aa4 4\uff1a\u603b\u7ed3")
        self.setSubTitle("\u68c0\u67e5\u8bbe\u7f6e\u5e76\u8fd0\u884c\u3002")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("\u914d\u7f6e\u603b\u7ed3:"))
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        layout.addWidget(self._text)

    def setSummary(self, text: str) -> None:
        self._text.setText(text)


# ── Main Wizard ─────────────────────────────────────────────────────

class Tool2BS2Dialog(QWizard):
    """Multi-page wizard for \u7edf\u8ba1\u89d2\u54cd\u5e94\uff08BSAR\uff09 computation."""

    def __init__(self, selected_files: List[str],
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("\u7edf\u8ba1\u89d2\u54cd\u5e94\uff08BSAR\uff09")
        self.setButtonText(QWizard.NextButton, "下一步 >")
        self.setButtonText(QWizard.BackButton, "< 上一步")
        self.setButtonText(QWizard.CancelButton, "取消")
        self.setButtonText(QWizard.FinishButton, "完成")
        self.setMinimumSize(540, 480)
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
        return ""

    @property
    def bathy_nc(self) -> str:
        return self._page1.getBathyNc()

    def getOutputBsar(self) -> str:
        return self._page1.getOutputBsar()

    def get_params(self) -> dict:
        p2 = self._page2.getParams()
        p3 = self._page3.getParams()
        return {**p2, **p3}

    def _on_page_changed(self, page_id: int) -> None:
        if page_id == 3:
            self._page4.setSummary(self._build_summary())

    def _build_summary(self) -> str:
        p1 = self._page1
        p2 = self._page2.getParams()
        p3 = self._page3.getParams()
        lines = [
            "=== 输入 / 输出 ===",
            f"  XSF 文件:        {len(self._selected_files)}",
            f"  输出 BSAR:       {p1.getOutputBsar()}",
            f"  参考 DTM:        {p1.getBathyNc() or '(not set)'}",
            "",
            "=== 参数 ===",
            f"  测深仪类型:            {p2['sounder_type']}",
            f"  积分方法:      {p2['integration_method']}",
            f"  线性比例:            {p2['linear_scale']}",
            f"  使用片段平均值:        {p2['use_snippets']}",
            "",
            "=== 高级 ===",
            f"  使用内嵌 SVP:            {p3['use_svp']}",
            f"  重新计算声照面积:        {p3['use_insonified_area']}",
            f"  移除补偿:                {p3['remove_compensation']}",
            f"  移除校准:                {p3['remove_calibration']}",
        ]
        return "\n".join(lines)
