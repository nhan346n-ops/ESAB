import torch
import numpy as np
import scipy.interpolate as interp
from torch.utils.data import Dataset
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pinn_esab_forward import DifferentiableESAB


class ESABDataset(Dataset):
    """
    PINN 合成数据集生成器。

    生成物理参数 (z, mu_db, s1, delta2) 及其对应的 BS 角响应曲线，
    用于训练 PINN 从 BS 曲线反演海底物理参数。

    - s1: 归一化粗糙度 (频率无关, 参考频率 150kHz)
    - 曲线生成使用 s1 经频率缩放后的有效粗糙度 δ₁(f)
    """
    def __init__(self, num_samples=100000, n_angles=71, gamma=10/3, f0_hz=150000.0):
        self.num_samples = num_samples
        self.n_angles = n_angles
        self.theta_deg = torch.linspace(0, 70, n_angles)
        self.gamma = gamma
        self.f0_hz = f0_hz

        self.model = DifferentiableESAB(gamma=gamma, f0_hz=f0_hz)

        # Pre-generate parameters
        self._generate_params()
        # Generate pure theoretical curves
        self._generate_curves()

    def _generate_params(self):
        # 1. z: 1.0 to 14.0 (阻抗对比度, 频率无关)
        self.z = torch.rand(self.num_samples, 1) * 13.0 + 1.0

        # 2. mu_db: -15.0 to 15.0 (体积散射, dB, 参考频率 f0)
        self.mu_db = torch.rand(self.num_samples, 1) * 30.0 - 15.0

        # 3. s1: 0.5 to 55.0 (归一化粗糙度, 参考频率 f0 处)
        #    这是频率无关的 intrinsic roughness 参数
        #    注意: SA 搜索范围是 [0.5, 70.0]，为保留 ScaledSigmoid 边界余量设为 [0.5, 55.0]
        self.s1 = torch.rand(self.num_samples, 1) * 54.5 + 0.5

        # 4. delta2: > delta1_eff (当前频率有效值) 到 35.0
        #    先生成任意频率，然后确保 delta2 > delta1_eff(freq)
        self.freq_hz = torch.rand(self.num_samples, 1) * 415000.0 + 35000.0

        # 计算当前频率下的有效 delta1
        D2 = (self.gamma - 2) / 2
        freq_scale = (self.freq_hz / self.f0_hz) ** (D2 / 2)
        delta1_eff = self.s1 * freq_scale

        # 确保 delta2 > delta1_eff (物理约束)
        margin = torch.rand(self.num_samples, 1) * 20.0 + 0.1
        self.delta2 = delta1_eff + margin
        self.delta2 = torch.clamp(self.delta2, max=75.0)

    def _generate_curves(self):
        theta_batch = self.theta_deg.unsqueeze(0).expand(self.num_samples, -1)

        with torch.no_grad():
            # 传入 s1 (归一化粗糙度) 而非 delta1_eff
            # DifferentiableESAB.forward 内部会自动做频率缩放
            self.bs_pure = self.model(
                theta_batch,
                self.freq_hz,
                self.z,
                self.s1,
                self.delta2,
                self.mu_db
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        bs_clean = self.bs_pure[idx]

        # 添加噪声: 每次调用随机生成不同噪声模式
        # 1. 角度依赖的噪声 (近法向噪声小，远角噪声大)
        theta_local = self.theta_deg / 70.0  # 0~1 归一化
        noise_std_profile = 0.5 + 4.0 * (theta_local**2)  # 0°: 0.5dB, 70°: ~4.5dB
        base_noise_scale = torch.rand(1).item() * 1.5 + 0.5  # 每样本随机缩放因子
        speckle_noise = torch.randn(self.n_angles) * noise_std_profile * base_noise_scale

        # 2. TVG 残留/基线漂移 (低频样条)
        n_knots = 5
        knots_x = np.linspace(0, self.n_angles - 1, n_knots)
        knots_y = np.random.randn(n_knots) * 3.0
        spline = interp.CubicSpline(knots_x, knots_y)
        drift_noise = torch.tensor(spline(np.arange(self.n_angles)), dtype=torch.float32)

        # 3. 功率域散斑 (乘性噪声，物理更真实)
        bs_linear = 10.0**(bs_clean / 10.0)
        speckle_power = torch.randn(self.n_angles) * bs_linear * 0.08  # 8% 标准差
        bs_power_noisy = bs_linear + speckle_power
        speckle_noise_db = 10.0 * torch.log10(bs_power_noisy / (bs_linear + 1e-15) + 1e-15)

        bs_noisy = bs_clean + speckle_noise + drift_noise + speckle_noise_db

        # 归一化输入
        bs_norm = (bs_noisy + 20.0) / 20.0
        theta_norm = self.theta_deg / 70.0

        x = torch.stack([bs_norm, theta_norm], dim=0)  # (2, 71)

        # 目标参数归一化到 [0, 1]
        # s1 是频率归一化的 intrinsic 参数
        y_z = (self.z[idx] - 1.0) / 13.0
        y_mu = (self.mu_db[idx] + 15.0) / 30.0
        y_s1 = (self.s1[idx] - 0.5) / 54.5   # s1 范围 [0.5, 55.0]
        y_d2 = (self.delta2[idx] - 1.0) / 74.0  # delta2 范围 [1, 75]

        y = torch.cat([y_z, y_mu, y_s1, y_d2], dim=0)

        # 频率条件 (归一化)
        freq_norm = (self.freq_hz[idx] - 35000.0) / 415000.0

        return x, freq_norm, y


if __name__ == "__main__":
    # Test generator
    print("Initializing Dataset...")
    dataset = ESABDataset(num_samples=1000)
    print(f"Dataset created with {len(dataset)} samples.")
    x, freq, y = dataset[0]
    print(f"Input shape: {x.shape}")
    print(f"Freq shape: {freq.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target values: {y}")
    # Re-scale for verification
    z = y[0].item() * 13.0 + 1.0
    mu = y[1].item() * 30.0 - 15.0
    s1 = y[2].item() * 24.0 + 1.0
    d2 = y[3].item() * 34.0 + 1.0
    print(f"Re-scaled: z={z:.2f}, mu={mu:.2f}dB, s1={s1:.2f}deg, delta2={d2:.2f}deg")
