"""
ESAB (Extended Specular + Angular Blending) 声学底质反演模型

完整复刻自 pyat-main（Python Acoustic ToolBox），基于论文
"Extended Specular+Angular Blending (ESAB) model for seafloor backscatter
angular response characterization" (Frontiers in Remote Sensing, 2025).

物理模型架构 (Eq 1-20):
  - 阻抗比与 Rayleigh 反射系数 (Eq 1-3)
  - 双高斯 Facet 散射 (Eq 4-6): 大尺度 δ₁ + 小尺度 δ₂
  - 粗糙度频谱指数 γ = 10/3 (Eq 5)
  - 频率缩放 (Eq 10-11): 归一化粗糙度 s₁ → 有效 δ₁(f)
  - Bragg 微观粗糙度散射 (Eq 13-14)
  - 界面 Sigmoid 过渡 (Eq 17-18)
  - 体积散射 (Eq 15, 19)
  - 总散射方程 (Eq 20)

反演引擎:
  - 模拟退火 (SA) 全局优化
  - PINN (Physics-Informed Neural Network) 快速推理
  - PINN+SA 混合流水线 (95% PINN + 5% SA 精修)

目录结构:
  ESAB/
  ├── __init__.py          # 本文件 — 包入口与主类导出
  ├── signal_utils.py      # 信号单位转换工具 (dB/能量/振幅)
  ├── gsab_model.py        # GSAB 平滑拟合模型
  ├── esab_model.py        # ESAB 业务适配器层
  └── esab/                # ESAB 核心包
      ├── __init__.py
      ├── esab_core.py           # 前向物理模型 (numpy)
      ├── bragg_model.py         # Bragg 散射
      ├── volume_scattering.py   # 体积散射
      ├── simulated_annealing.py # 模拟退火反演
      ├── esab_inversion.py      # 单频反演入口
      ├── parameter_manager.py   # 参数数据类
      ├── geoacoustic_model.py   # 地声参数恢复
      ├── pinn_esab_forward.py   # PyTorch 可微前向模型
      ├── pinn_accelerated_sa.py # PINN 加速 SA 流水线
      ├── pinn_sa_hybrid.py      # PINN+SA 混合流水线
      └── pinn/                  # PINN 子包
          ├── __init__.py
          ├── pinn_model.py      # 神经网络架构 (CNN/Transformer)
          ├── pinn_trainer.py    # PINN 训练器
          ├── pinn_inference.py  # 批量推理引擎
          ├── pinn_finetune.py   # 真实数据微调
          └── data_generator.py  # 合成数据生成器
"""

from .gsab_model import GsabDataModel, GsabDataCoefficients
from .esab_model import EsabDataModel
from .esab.esab_core import compute_esab_bs
from .esab.esab_inversion import invert_esab_single_freq
from .esab.parameter_manager import ESABParameters
from .esab.geoacoustic_model import GeoacousticModel

__all__ = [
    "EsabDataModel",
    "GsabDataModel",
    "GsabDataCoefficients",
    "compute_esab_bs",
    "invert_esab_single_freq",
    "ESABParameters",
    "GeoacousticModel",
]
