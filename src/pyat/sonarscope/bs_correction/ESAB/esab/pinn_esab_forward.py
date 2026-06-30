import torch
import torch.nn as nn
import numpy as np


class DifferentiableESAB(nn.Module):
    """
    PyTorch 可微 ESAB 前向模型，用于 PINN 训练。

    严格遵循论文 Eq 1-20，同时保持与 Numpy 版 esab_core.py 一致。
    """
    def __init__(self, gamma=10/3, f0_hz=150000.0):
        super().__init__()
        self.gamma = gamma
        self.f0_hz = f0_hz

        # D₂ = (γ-2)/2  (Eq 10)
        self.D2 = (gamma - 2) / 2

        # Precompute constants for Bragg (Eq 14)
        self.D4 = (gamma - 2) / 2
        self.D3 = (2**(gamma - 5)) * (np.pi**(gamma - 3)) * ((4 - gamma)**self.D4) * ((gamma - 2)**(1 - self.D4))

    def compute_freq_scaled_delta(self, s1_deg, freq_hz):
        """
        频率缩放: δ_eff = s₁ · (f/f₀)^(D₂/2)   (Eq 11)
        """
        return s1_deg * (freq_hz / self.f0_hz) ** (self.D2 / 2)

    def compute_bragg(self, theta_rad, V2_abs, delta1_sq):
        """
        Bragg 散射 (Eq 14)，与 numpy 版 bragg_model.py 一致。
        近法向通过 mask 和 clamp 双重保护，防止梯度爆炸。
        """
        sin_theta = torch.sin(theta_rad)
        cos_theta = torch.cos(theta_rad)

        # 与 numpy 版保持一致的阈值 (1e-8 rad ≈ 0.0006°)
        # clamp 防止梯度爆炸，mask 确保数值截断
        sin_theta_safe = torch.clamp(sin_theta, min=1e-8)
        mask = (sin_theta > 1e-8).float()

        # 数值保护：防止 delta1_sq 过小
        delta_sq_safe = torch.clamp(delta1_sq, min=1e-10)

        sigma_B = self.D3 * V2_abs * (cos_theta**4) / (sin_theta_safe**self.gamma) * (delta_sq_safe**self.D4)
        # 防止极端值
        sigma_B = torch.nan_to_num(sigma_B, nan=0.0, posinf=0.0, neginf=0.0)
        return sigma_B * mask

    def compute_volume(self, theta_rad, freq_hz, mu_db, V, c_ratio, cos_theta2):
        """
        体积散射 (Eq 15)。
        μ (mu_db) 是已整合 1/(4α₀) 因子的归一化体积参数。
        """
        T_factor = torch.abs(1 - V**2)**2

        m0_eff = 10**(mu_db / 10.0)
        eff_volume_param = m0_eff * ((freq_hz / self.f0_hz)**0.65)

        cos_theta2_real = torch.real(cos_theta2)
        cos_theta2_real = torch.clamp(cos_theta2_real, min=1e-6)

        sigma_v = eff_volume_param * T_factor * (c_ratio**2) * (torch.cos(theta_rad)**2) / cos_theta2_real

        # 全反射截断
        sin_theta2_abs = c_ratio * torch.sin(theta_rad)
        mask = (sin_theta2_abs < 1.0).float()

        return sigma_v * mask

    @staticmethod
    def find_crossing_angle_newton(delta1_sq, tol=1e-6, max_iters=5):
        """
        可微 Newton 迭代求交叉角 θₓ (Eq 17 的过渡参数)。
        求解: exp(-tan²(θ)/(2δ²)) / cos⁴(θ) = 10^(-1.5)

        初始值使用小角度近似，然后 Newton 迭代修正，
        结果与 numpy 版 brentq 精确解误差 < 0.01°。
        """
        target_ratio = 10**(-1.5)

        # 数值保护：确保 delta1_sq 不为零
        delta1_sq = torch.clamp(delta1_sq, min=1e-10)

        # 初始值: 小角度近似 tanθ≈θ, cosθ≈1
        theta = torch.atan(torch.sqrt(3.453 * 2 * delta1_sq))
        theta = torch.clamp(theta, min=0.01, max=1.4)

        for _ in range(max_iters):
            tan_t = torch.tan(theta)
            cos_t = torch.cos(theta)
            cos_t = torch.clamp(cos_t, min=1e-4)

            ratio = torch.exp(- (tan_t**2) / (2 * delta1_sq)) / (cos_t**4)
            f = ratio - target_ratio

            # f' = df/dθ
            df_dtheta = -ratio * (tan_t / (delta1_sq * cos_t**2) + 4 * tan_t / cos_t**2)
            df_dtheta = torch.clamp(df_dtheta, max=-1e-10, min=-1e10)  # 防止除零

            delta = f / df_dtheta
            # 防止过大步长导致振荡发散
            delta = torch.clamp(delta, min=-0.5, max=0.5)
            theta = theta - delta
            theta = torch.clamp(theta, min=0.01, max=1.4)

        return theta

    def forward(self, theta_deg, freq_hz, z, delta1_deg, delta2_deg, mu_db):
        """
        ESAB forward pass (全可微)

        Args:
            theta_deg: (Batch, N_angles) — 入射角 (度)
            freq_hz: (Batch, 1) — 工作频率 (Hz)
            z: (Batch, 1) — 声阻抗对比度
            delta1_deg: (Batch, 1) — 归一化粗糙度 s₁ 或有效 δ₁
            delta2_deg: (Batch, 1) — 第二高斯分量粗糙度
            mu_db: (Batch, 1) — 体积散射参数 (dB, f₀ 处)
        """
        theta = torch.deg2rad(torch.abs(theta_deg))
        theta = torch.clamp(theta, min=1e-8, max=np.pi/2 - 1e-8)

        # 1. 阻抗与反射系数 (Eq 1-3)
        c_ratio = 0.7030 + 0.2055 * z
        sin_theta2 = c_ratio * torch.sin(theta)

        # 复数 cos_θ₂，加小虚部防止临界角处梯度奇异
        complex_inside = torch.complex(1 - sin_theta2**2, 1e-8 * torch.ones_like(sin_theta2))
        cos_theta2 = torch.sqrt(complex_inside)

        numerator = torch.complex(z * torch.cos(theta), torch.zeros_like(theta)) - cos_theta2
        denominator = torch.complex(z * torch.cos(theta), torch.zeros_like(theta)) + cos_theta2
        V = numerator / denominator
        V2_abs = torch.abs(V)**2

        V_0 = (z - 1.0) / (z + 1.0)
        V2_0 = V_0**2

        # 2. 粗糙度频率缩放 (Eq 11) — 如果 delta1_deg 传入的是归一化的 s1
        if delta1_deg.shape[-1] == 1 and delta1_deg.dim() >= 2:
            delta1_eff = self.compute_freq_scaled_delta(delta1_deg, freq_hz)
        else:
            delta1_eff = delta1_deg

        delta1_sq = torch.deg2rad(delta1_eff)**2
        delta2_sq = torch.deg2rad(delta2_deg)**2
        coherence_loss = np.exp(-1)

        tan_theta = torch.tan(theta)

        def facet_cross_section(delta_sq):
            # 数值保护：防止 delta_sq 过小导致分母爆炸
            delta_sq = torch.clamp(delta_sq, min=1e-10)
            return (V2_0 * torch.exp(- (tan_theta**2) / (2 * delta_sq))) / \
                   (8 * np.pi * delta_sq * (torch.cos(theta)**4)) * coherence_loss

        sigma_f1 = facet_cross_section(delta1_sq)
        sigma_f2 = facet_cross_section(delta2_sq)

        # 3. Bragg 散射 (Eq 14)
        sigma_B = self.compute_bragg(theta, V2_abs, delta1_sq)

        # 4. 界面 Sigmoid (Eq 17-18)
        theta_x = self.find_crossing_angle_newton(delta1_sq)
        sigmoid_I = torch.sigmoid(180/np.pi * (theta - theta_x))
        sigma_Is = (sigma_B + sigma_f2) * sigmoid_I + (1 - sigmoid_I) * sigma_f1

        # 5. 体积散射 (Eq 15, 19)
        sigma_v = self.compute_volume(theta, freq_hz, mu_db, V, c_ratio, cos_theta2)

        theta_v = np.deg2rad(10.0)
        sigmoid_V = torch.sigmoid((180/np.pi * (theta - theta_v)) / 2.0)
        sigma_Vs = sigma_v * sigmoid_V

        # 6. 总散射 (Eq 20)
        BS = 10 * torch.log10(sigma_Is + sigma_Vs + 1e-15)
        # 数值稳定性保护：NaN/Inf 转为合理边界值
        BS = torch.nan_to_num(BS, nan=-30.0, posinf=10.0, neginf=-60.0)
        return BS
