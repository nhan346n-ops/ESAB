"""
ESAB 数据的业务适配器层。

负责接收平滑后的 GSAB 模型数据，并执行 ESAB 物理反演，
提供物理分量分解与地声参数推断。

从 pyat-main 独立提取，import 已改为本地文件。
"""

import numpy as np
from typing import Dict, Any

from gsab_model import GsabDataModel
from esab.esab_inversion import invert_esab_single_freq
from esab.esab_core import compute_esab_bs
from esab.bragg_model import compute_bragg_scattering
from esab.volume_scattering import compute_volume_scattering, compute_volume_sigmoid
from esab.geoacoustic_model import GeoacousticModel
from esab.esab_core import find_crossing_angle


class EsabDataModel:
    """
    ESAB 数据的业务适配器层。
    负责接收平滑后的 GSAB 模型数据，并执行 ESAB 物理反演。
    """
    def __init__(self, gsab_model: GsabDataModel, freq_hz: float = 300000.0, gamma: float = 10/3):
        self.gsab_model = gsab_model
        self.freq_hz = freq_hz
        self.gamma = gamma
        self.f0_hz = 150000.0  # 参考频率 (论文定义)

        # 提取高密度的连续参考曲线，通常 ESAB 需要 0-70 度的输入
        self.angles_continuous = np.arange(0, 71, 1.0)

        # 参数结果缓存
        self.esab_results = None
        self.geo_props = None

        # 生成的曲线数据缓存
        self.esab_curve = None

    def _delta1_eff(self, s1_deg: float) -> float:
        """
        从频率归一化的 s1 计算在当前工作频率下的有效粗糙度 (Eq 11).
        delta1_eff = s1 * (f/f0)^(D2/2),  D2 = (gamma-2)/2
        """
        D2 = (self.gamma - 2) / 2
        return s1_deg * (self.freq_hz / self.f0_hz) ** (D2 / 2)

    def fit_esab(self) -> Dict[str, Any]:
        """
        基于 GSAB 平滑后的背向散射曲线，运行 ESAB 物理反演。
        返回包含 s1 (频率归一化粗糙度) 的结果。
        """
        # 1. 确保 GSAB 已经拟合
        if not hasattr(self.gsab_model.coeffs, 'a') or self.gsab_model.coeffs.a is None:
            self.gsab_model.fit_gsab()

        linear_coeffs = self.gsab_model.coeffs.linear_coeffs()
        bs_smoothed = self.gsab_model.gsab_func(
            self.angles_continuous,
            linear_coeffs.a, linear_coeffs.b, linear_coeffs.c,
            linear_coeffs.d, linear_coeffs.e, linear_coeffs.f
        )

        # 2. 调用底层的单频 ESAB 反演引擎
        self.esab_results = invert_esab_single_freq(
            self.angles_continuous, bs_smoothed, self.freq_hz,
            gamma=self.gamma
        )

        # 3. 提取结果，计算附加地声参数
        geo = GeoacousticModel()
        self.geo_props = geo.recover_properties(self.esab_results["z"])

        # 4. 缓存总 ESAB 前向曲线
        s1 = self.esab_results["s1"]
        d2 = self.esab_results["delta2"]
        self.esab_curve = compute_esab_bs(
            self.angles_continuous, self.freq_hz,
            self.esab_results["z"], s1,
            s1, d2,
            self.esab_results["mu_db"],
            gamma=self.gamma, f0_hz=self.f0_hz
        )

        return self.esab_results

    def get_components(self):
        """
        获取解剖后的物理分量 (Facet1, Facet2, Bragg, Volume) 供可视化。
        必须在 fit_esab() 之后调用。
        """
        if self.esab_results is None:
            raise ValueError("必须先调用 fit_esab()")

        theta_rad = np.deg2rad(self.angles_continuous)

        z = self.esab_results["z"]
        s1 = self.esab_results["s1"]        # 频率归一化粗糙度
        d2_deg = self.esab_results["delta2"]
        mu_db = self.esab_results["mu_db"]

        # 通过频率缩放得到当前频率下的有效粗糙度
        d1_eff = self._delta1_eff(s1)
        d1_sq = np.deg2rad(d1_eff)**2
        d2_sq = np.deg2rad(d2_deg)**2

        c_ratio = 0.7030 + 0.2055 * z
        sin_theta2 = c_ratio * np.sin(theta_rad)
        cos_theta2 = np.sqrt(1 - sin_theta2**2 + 0j)

        numerator = z * np.cos(theta_rad) - cos_theta2
        denominator = z * np.cos(theta_rad) + cos_theta2
        V = numerator / denominator
        V2_abs = np.abs(V)**2
        V_0 = (z - 1.0) / (z + 1.0)
        V2_0 = V_0**2

        coherence_loss = np.exp(-1)

        def facet(d_sq):
            return (V2_0 * np.exp(- (np.tan(theta_rad)**2) / (2 * d_sq))) / \
                   (8 * np.pi * d_sq * (np.cos(theta_rad)**4)) * coherence_loss

        f1 = facet(d1_sq)
        f2 = facet(d2_sq)

        # Bragg
        bragg = compute_bragg_scattering(theta_rad, 10/3, V2_abs, d1_sq)

        # Volume
        vol = compute_volume_scattering(theta_rad, self.freq_hz, self.f0_hz, mu_db, V, c_ratio, cos_theta2)
        vol_sig = compute_volume_sigmoid(theta_rad)

        theta_x = find_crossing_angle(d1_sq)
        sigmoid_I = 1 / (1 + np.exp(-180/np.pi * (theta_rad - theta_x)))

        # 加权分量 (dB)
        comp_f1 = 10 * np.log10((1 - sigmoid_I) * f1 + 1e-15)
        comp_f2 = 10 * np.log10(sigmoid_I * f2 + 1e-15)
        comp_bragg = 10 * np.log10(sigmoid_I * bragg + 1e-15)
        comp_vol = 10 * np.log10(vol * vol_sig + 1e-15)

        return {
            "angles": self.angles_continuous,
            "facet1": comp_f1,
            "facet2": comp_f2,
            "bragg": comp_bragg,
            "volume": comp_vol
        }
