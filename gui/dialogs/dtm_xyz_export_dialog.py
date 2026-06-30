"""DTM → XYZ export dialog."""
import os
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QCheckBox, QGroupBox,
    QPushButton, QFileDialog, QDialogButtonBox,
    QComboBox, QWidget, QListWidget,
)


class DtmXyzExportDialog(QDialog):
    def __init__(self, filepath: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._filepath = filepath
        self._setup_ui()
        self.setWindowTitle("导出 DTM 到 XYZ")
        self.resize(500, 400)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Input file
        layout.addWidget(QLabel("输入 DTM 文件:"))
        fl = QListWidget()
        fl.addItem(self._filepath)
        fl.setMaximumHeight(40)
        layout.addWidget(fl)

        # Output group
        out_grp = QGroupBox("输出设置")
        of = QFormLayout(out_grp)

        # Output directory
        dir_row = QHBoxLayout()
        self._out_dir = QLineEdit(os.path.dirname(self._filepath))
        dir_row.addWidget(self._out_dir)
        browse_btn = QPushButton("浏览…")
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        of.addRow("输出目录:", dir_row)

        # File name prefix
        base = os.path.splitext(os.path.basename(self._filepath))[0]
        if base.endswith(".dtm"):
            base = base[:-4]
        self._name_edit = QLineEdit(f"{base}_export")
        of.addRow("文件名:", self._name_edit)

        self._overwrite = QCheckBox("覆盖已存在的文件")
        of.addRow(self._overwrite)
        layout.addWidget(out_grp)

        # Format options group
        fmt_grp = QGroupBox("格式设置")
        ff = QFormLayout(fmt_grp)

        # Column Separator
        self._col_sep = QComboBox()
        self._col_sep.addItems(["分号", "逗号", "空格", "制表符 (Tab)", "其他"])
        self._col_sep.setCurrentText("分号")
        self._col_sep.currentTextChanged.connect(self._on_sep_changed)
        ff.addRow("列分隔符:", self._col_sep)

        # 其他 Separator (enabled only when '其他' selected)
        self._other_sep = QLineEdit(";")
        self._other_sep.setEnabled(False)
        ff.addRow("其他分隔符:", self._other_sep)

        # Decimal Separator
        self._dec_sep = QComboBox()
        self._dec_sep.addItems(["Dot", "逗号"])
        self._dec_sep.setCurrentText("Dot")
        ff.addRow("小数点分隔符:", self._dec_sep)

        # Column Order
        self._col_order = QComboBox()
        self._col_order.addItems(["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"])
        self._col_order.setCurrentText("XYZ")
        ff.addRow("列顺序:", self._col_order)

        # Export Missing Values
        self._export_missing = QCheckBox("导出缺失/无效值")
        self._export_missing.setChecked(False)
        ff.addRow(self._export_missing)

        layout.addWidget(fmt_grp)
        layout.addStretch()

        # Buttons
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self._out_dir.setText(d)

    def _on_sep_changed(self, text: str) -> None:
        self._other_sep.setEnabled(text == "其他")

    def getOutputDir(self) -> str:
        return self._out_dir.text()

    def getFileName(self) -> str:
        return self._name_edit.text()

    def getOverwrite(self) -> bool:
        return self._overwrite.isChecked()

    def getColumnSeparator(self) -> str:
        return self._col_sep.currentText()

    def getOtherSeparator(self) -> str:
        return self._other_sep.text()

    def getDecimalSeparator(self) -> str:
        return self._dec_sep.currentText().lower()

    def getColumnOrder(self) -> str:
        return self._col_order.currentText()

    def getExportMissing(self) -> bool:
        return self._export_missing.isChecked()
