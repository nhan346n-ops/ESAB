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
        self.setTitle("第1步：输入与输出")
        self.setSubTitle("选择 BSAR 模型、可选的 DTM，以及输出文件夹。")

        layout = QVBoxLayout(self)

        # Input files
        layout.addWidget(QLabel(f"待处理 XSF 文件: {len(selected_files)}"))
        fl = QListWidget()
        for f in selected_files[:8]:
            fl.addItem(f)
        fl.setMaximumHeight(60)
        layout.addWidget(fl)

        # BSAR model (required)
        bsar_row = QHBoxLayout()
        bsar_row.addWidget(QLabel("角度响应模型 (.bsar.nc):"))
        self._bsar_edit = QLineEdit()
        self._bsar_edit.setPlaceholderText("必填 — 预先计算的 BSAR 模型")
        bsar_row.addWidget(self._bsar_edit)
        bsar_btn = QPushButton("浏览...")
        bsar_btn.clicked.connect(self._browse_bsar)
        bsar_row.addWidget(bsar_btn)
        layout.addLayout(bsar_row)
        self.registerField("bsar*", self._bsar_edit)

        # Reference DTM (optional)
        dtm_row = QHBoxLayout()
        dtm_row.addWidget(QLabel("参考 DTM (_bathy.nc):"))
        self._dtm_edit = QLineEdit()
        self._dtm_edit.setPlaceholderText("可选 — 用于入射角校正")
        dtm_row.addWidget(self._dtm_edit)
        dtm_btn = QPushButton("浏览...")
        dtm_btn.clicked.connect(self._browse_dtm)
        dtm_row.addWidget(dtm_btn)
        layout.addLayout(dtm_row)

        # Output
        out_grp = QGroupBox("输出")
        of = QFormLayout(out_grp)
        dir_row = QHBoxLayout()
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("留空则使用输入文件所在的目录")
        dir_row.addWidget(self._out_dir)
        browse_out = QPushButton("浏览...")
        browse_out.clicked.connect(self._browse_out)
        dir_row.addWidget(browse_out)
        of.addRow("目录:", dir_row)
        self._overwrite = QCheckBox("覆盖已存在的输出文件")
        of.addRow(self._overwrite)
        layout.addWidget(out_grp)

    def _browse_bsar(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "选择 BSAR 模型", "",
            "BSAR files (*.bsar.nc);;NetCDF files (*.nc);;All (*.*)")
        if f:
            self._bsar_edit.setText(f)

    def _browse_dtm(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "选择 DTM 文件", "",
            "DTM files (*.dtm.nc *.nc);;All (*.*)")
        if f:
            self._dtm_edit.setText(f)

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "输出目录")
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
        self.setTitle("第2步：参数")
        self.setSubTitle("参考基准面、补偿及处理选项。")

        layout = QFormLayout(self)

        # Reference level
        ref_row = QHBoxLayout()
        self._ref_level = QDoubleSpinBox()
        self._ref_level.setRange(-100, 0)
        self._ref_level.setValue(-20)
        self._ref_level.setSuffix(" dB")
        self._ref_level.setDecimals(1)
        ref_row.addWidget(self._ref_level)
        eval_btn = QPushButton("评估")
        eval_btn.setToolTip("自动评估参考基准面。")
        eval_btn.clicked.connect(self._on_evaluate)
        ref_row.addWidget(eval_btn)
        layout.addRow("参考基准面:", ref_row)

        self._apply_comp = QCheckBox("应用入射角补偿")
        self._apply_comp.setChecked(True)
        layout.addRow(self._apply_comp)

        self._snippets = QCheckBox("使用片段平均值")
        layout.addRow(self._snippets)

        # Separator
        layout.addRow(QLabel(""))
        layout.addRow(QLabel("以下选项从角度响应模型中读取 "
                             "如果不可用或需要覆盖，请勾选复选框并手动设置:"))

        self._sounder = QComboBox()
        self._sounder.addItems(SOUNDER_TYPES)
        self._sounder.setCurrentText("AUTO")
        self._sounder.setEnabled(False)
        layout.addRow("测深仪类型:", self._sounder)

        self._integration = QComboBox()
        self._integration.addItems(INTEGRATION_METHODS)
        self._integration.setCurrentText("MEAN")
        self._integration.setEnabled(False)
        layout.addRow("积分方法:", self._integration)

        self._scale = QComboBox()
        self._scale.addItems(LINEAR_SCALES)
        self._scale.setCurrentText("AMPLITUDE")
        self._scale.setEnabled(False)
        layout.addRow("线性比例:", self._scale)

    def _on_evaluate(self) -> None:
        wiz = self.wizard()
        if not wiz:
            return
        bsar_path = wiz._page1.getBsarNc()
        if not bsar_path or not os.path.isfile(bsar_path):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "缺失 BSAR 模型",
                "请先在第一步中选择一个 BSAR 模型文件。")
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
                        self, "评估完成",
                        f"预估参考基准面: {ref_level:.1f} dB\n"
                        f"（基于 BSAR 模型中所有模式的加权平均值）")
        except Exception as e:
            QMessageBox.warning(
                self, "评估失败",
                f"无法评估参考基准面:\n{e}")
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
        self.setTitle("第3步：高级选项")
        self.setSubTitle("BL0/BL2 校正标志（从 BSAR 模型读取，仅供参考）。")

        layout = QFormLayout(self)

        self._use_svp = QCheckBox("使用内嵌声速剖面")
        self._use_svp.setChecked(True)
        self._use_svp.setEnabled(False)
        layout.addRow(self._use_svp)

        self._use_ia = QCheckBox("通过海底入射角重新计算声照面积")
        self._use_ia.setChecked(True)
        self._use_ia.setEnabled(False)
        layout.addRow(self._use_ia)

        self._remove_comp = QCheckBox("移除角度补偿")
        self._remove_comp.setChecked(True)
        self._remove_comp.setEnabled(False)
        layout.addRow(self._remove_comp)

        self._remove_cal = QCheckBox("移除校准 (从 kmall 中移除 BScorr)")
        self._remove_cal.setChecked(True)
        self._remove_cal.setEnabled(False)
        layout.addRow(self._remove_cal)

    def getParams(self) -> dict:
        return {}


# ── Page 4: Summary ─────────────────────────────────────────────────

class _SummaryPage(QWizardPage):
    def __init__(self, parent: Optional[QWizard] = None):
        super().__init__(parent)
        self.setTitle("第4步：总结")
        self.setSubTitle("确认设置并运行。")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("配置总结:"))
        self._text = QTextEdit()
        self._text.setReadOnly(True)
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
        self.setButtonText(QWizard.NextButton, "下一步 >")
        self.setButtonText(QWizard.BackButton, "< 上一步")
        self.setButtonText(QWizard.CancelButton, "取消")
        self.setButtonText(QWizard.FinishButton, "完成")
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
            "=== 输入 / 输出 ===",
            f"  XSF 文件:           {len(self._selected_files)}",
            f"  BSAR model:         {p1.getBsarNc()}",
            f"  参考 DTM:           {p1.getBathyNc() or '(not set)'}",
            f"  Output directory:   {p1.getOutputDir() or '(same as input)'}",
            f"  Overwrite:          {p1.getOverwrite()}",
            "",
            "=== 参数 ===",
            f"  参考基准面:           {p2['reference_level']} dB",
            f"  Incidence compensation:    {p2['apply_compensation']}",
            f"  使用片段平均值:          {p2['use_snippets']}",
        ]
        return "\n".join(lines)