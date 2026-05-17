"""DTM → GeoTIFF export dialog."""
import os
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QCheckBox, QGroupBox,
    QPushButton, QFileDialog, QDialogButtonBox,
    QSpinBox, QWidget, QListWidget,
)


DTM_LAYERS = [
    ("elevation", "\u6c34\u6df1"),
    ("backscatter", "\u540e\u5411\u6563\u5c04\u5f3a\u5ea6"),
    ("interpolation_flag", "\u5185\u63d2\u6807\u5fd7"),
    ("value_count", "\u56de\u58f0\u6570"),
]


class DtmExportDialog(QDialog):
    def __init__(self, filepath: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._filepath = filepath
        self._setup_ui()
        self.setWindowTitle("\u5bfc\u51fa GeoTIFF")
        self.resize(500, 420)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Input file
        layout.addWidget(QLabel("\u8f93\u5165 DTM \u6587\u4ef6:"))
        fl = QListWidget()
        fl.addItem(self._filepath)
        fl.setMaximumHeight(40)
        layout.addWidget(fl)

        # Output group
        out_grp = QGroupBox("\u8f93\u51fa\u8bbe\u7f6e")
        of = QFormLayout(out_grp)

        # Output directory
        dir_row = QHBoxLayout()
        self._out_dir = QLineEdit(os.path.dirname(self._filepath))
        dir_row.addWidget(self._out_dir)
        browse_btn = QPushButton("\u6d4f\u89c8\u2026")
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        of.addRow("\u76ee\u5f55:", dir_row)

        # File name prefix
        base = os.path.splitext(os.path.basename(self._filepath))[0]
        self._name_edit = QLineEdit(base)
        of.addRow("\u6587\u4ef6\u540d:", self._name_edit)

        self._overwrite = QCheckBox("\u8986\u76d6\u5df2\u6709\u6587\u4ef6")
        of.addRow(self._overwrite)
        layout.addWidget(out_grp)

        # Layers group
        layer_grp = QGroupBox("\u5bfc\u51fa\u56fe\u5c42")
        lg = QVBoxLayout(layer_grp)
        self._layer_cbs = {}
        for key, label in DTM_LAYERS:
            cb = QCheckBox(label)
            if key in ("backscatter", "elevation"):
                cb.setChecked(True)
            self._layer_cbs[key] = cb
            lg.addWidget(cb)
        layout.addWidget(layer_grp)

        # GDAL options group
        gdal_grp = QGroupBox("GTiff \u9a71\u52a8\u53c2\u6570")
        gf = QFormLayout(gdal_grp)

        fill_row = QHBoxLayout()
        self._nan_fill = QCheckBox("\u4f7f\u7528 NaN \u4f5c\u4e3a\u7f3a\u5931\u503c")
        fill_row.addWidget(self._nan_fill)
        fill_row.addWidget(QLabel("\u6216\u81ea\u5b9a\u4e49:"))
        self._fill_value = QSpinBox()
        self._fill_value.setRange(-9999, 99999)
        self._fill_value.setValue(32767)
        fill_row.addWidget(self._fill_value)
        gf.addRow("\u7f3a\u5931\u503c:", fill_row)

        self._compress = QCheckBox("\u5e94\u7528\u538b\u7f29 (DEFLATE)")
        self._compress.setChecked(True)
        gf.addRow(self._compress)
        layout.addWidget(gdal_grp)

        layout.addStretch()

        # Buttons
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "\u9009\u62e9\u8f93\u51fa\u76ee\u5f55")
        if d:
            self._out_dir.setText(d)

    def getOutputDir(self) -> str:
        return self._out_dir.text()

    def getFileName(self) -> str:
        return self._name_edit.text()

    def getLayers(self) -> dict:
        return {key: cb.isChecked() for key, cb in self._layer_cbs.items()}

    def getFillValue(self) -> Optional[float]:
        return float(self._fill_value.value()) if not self._nan_fill.isChecked() else None

    def getNanFill(self) -> bool:
        return self._nan_fill.isChecked()

    def getCompression(self) -> bool:
        return self._compress.isChecked()

    def getOverwrite(self) -> bool:
        return self._overwrite.isChecked()
