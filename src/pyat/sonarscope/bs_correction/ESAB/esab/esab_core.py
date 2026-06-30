import numpy as np
from scipy.optimize import brentq
from .bragg_model import compute_bragg_scattering
from .volume_scattering import compute_volume_scattering, compute_volume_sigmoid

def find_crossing_angle(delta1_sq):
    """
    通过求根算法 (brentq) 精确计算交叉角 theta_x，
    满足 sigma_f1 相比 nadir 处下降 15 dB。
    即求解: exp(-tan^2(theta)/(2*delta1_sq)) / cos^4(theta) - 10^(-1.5) = 0
    """
    target_ratio = 10**(-1.5) # -15 dB
    
    def func(theta):
        tan_t = np.tan(theta)
        cos_t = np.cos(theta)
        # 防止越界
        if cos_t < 1e-4:
            return -target_ratio
        ratio = np.exp(- (tan_t**2) / (2 * delta1_sq)) / (cos_t**4)
        return ratio - target_ratio
        
    # 在 0 到接近 90 度之间寻找根
    try:
        theta_x = brentq(func, 0.0, np.deg2rad(89.0))
    except ValueError:
        # 如果 brentq 失败（罕见，例如极度平滑的极限），使用近似解
        theta_x = np.arctan(np.sqrt(3.453 * 2 * delta1_sq))
    return theta_x

def compute_esab_bs(theta_deg, freq_hz, z, s1_deg, delta1_deg, delta2_deg, mu_db, gamma=10/3, f0_hz=150000.0):
    """
    精确重构 ESAB 前向物理模型。

    Parameters 论文对应:
    - z: 声阻抗对比度 (Eq 1)
    - s1_deg: 归一化粗糙度斜率 (在参考频率 f0 处的值, Eq 11)
    - delta1_deg: 当前频率的有效粗糙度 (若 s1_deg 非 None 则从 s1 通过频率缩放计算)
    - delta2_deg: 第二高斯分量粗糙度
    - mu_db: 体积散射参数 (参考频率处, dB)
    - gamma: 粗糙度频谱指数 (默认 10/3, Eq 5)
    - f0_hz: 参考频率 (默认 150 kHz)

    频率缩放 (Eq 10-11):
    δ²(f) = s² · (f/f₀)^D₂,  D₂ = (γ-2)/2
    δ(f) = s · (f/f₀)^(D₂/2)
    """
    theta_deg = np.atleast_1d(theta_deg).astype(float)
    theta = np.deg2rad(np.abs(theta_deg))
    theta = np.clip(theta, 1e-8, np.pi/2 - 1e-8)

    # 1. 阻抗与声速
    c_ratio = 0.7030 + 0.2055 * z
    sin_theta2 = c_ratio * np.sin(theta)
    cos_theta2 = np.sqrt(1 - sin_theta2**2 + 0j)

    numerator = z * np.cos(theta) - cos_theta2
    denominator = z * np.cos(theta) + cos_theta2
    V = numerator / denominator
    V2_abs = np.abs(V)**2

    # Facet 的法向反射率 V(0)^2
    V_0 = (z - 1.0) / (z + 1.0)
    V2_0 = V_0**2

    # 2. 粗糙度频率缩放 (论文 Eq 10-11)
    # D₂ = (γ-2)/2, δ(f) = s · (f/f₀)^(D₂/2)
    D2 = (gamma - 2) / 2
    if s1_deg is not None:
        # 从频率归一化的 s1 计算当前频率的有效粗糙度
        freq_scale = (freq_hz / f0_hz) ** (D2 / 2)
        delta1_eff = s1_deg * freq_scale
    else:
        delta1_eff = delta1_deg

    # 将角度转为弧度平方 (公式中使用 δ²)
    delta1_sq = np.deg2rad(delta1_eff)**2
    delta2_sq = np.deg2rad(delta2_deg)**2
    coherence_loss = np.exp(-1)
    
    def facet_cross_section(delta_sq):
        tan_theta = np.tan(theta)
        return (V2_0 * np.exp(- (tan_theta**2) / (2 * delta_sq))) / (8 * np.pi * delta_sq * (np.cos(theta)**4)) * coherence_loss
        
    sigma_f1 = facet_cross_section(delta1_sq)
    sigma_f2 = facet_cross_section(delta2_sq)
    
    # 3. Bragg
    sigma_B = compute_bragg_scattering(theta, gamma, V2_abs, delta1_sq)
    
    # 4. Interface Sigmoid
    theta_x = find_crossing_angle(delta1_sq)
    sigmoid_I = 1 / (1 + np.exp(-180/np.pi * (theta - theta_x)))
    sigma_Is = (sigma_B + sigma_f2) * sigmoid_I + (1 - sigmoid_I) * sigma_f1
    
    # 5. Volume Scattering
    sigma_v = compute_volume_scattering(theta, freq_hz, f0_hz, mu_db, V, c_ratio, cos_theta2)
    sigmoid_V = compute_volume_sigmoid(theta)
    sigma_Vs = sigma_v * sigmoid_V
    
    # 6. 总散射
    BS = 10 * np.log10(sigma_Is + sigma_Vs + 1e-15)
    return BS
