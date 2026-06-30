import numpy as np

def compute_volume_scattering(theta_rad, freq_hz, f0_hz, mu_db, V, c_ratio, cos_theta2):
    """
    严格按照论文 Equation 15 计算体积散射。
    
    参数:
    - theta_rad: 入射角 (弧度)
    - freq_hz: 当前工作频率 (Hz)
    - f0_hz: 参考频率 (通常 150000 Hz)
    - mu_db: 参考频率处的体积散射参数 (dB)
    - V: 反射系数复数
    - c_ratio: 声速比 c2/c1
    - cos_theta2: 透射角余弦复数
    
    返回:
    - sigma_v: 体积散射截面 (线性)
    """
    # 1. 传输系数 (Transmission coefficient)
    # T_factor = |1 - V^2|^2
    T_factor = np.abs(1 - V**2)**2
    
    # 2. 频率依赖性 (Frequency dependence)
    # 论文指出 beta 与 f 成正比，mv 与 f^1.65 成正比
    # 所以 mv / 4*beta 整体与 f^0.65 成正比
    # mu_db 是综合参数 m0 的 dB 表达
    m0_eff = 10**(mu_db / 10.0)
    eff_volume_param = m0_eff * ((freq_hz / f0_hz)**0.65)
    
    # 3. 计算 sigma_v
    cos_theta2_real = np.real(cos_theta2)
    cos_theta2_real[cos_theta2_real < 1e-6] = 1e-6 # 防止除零
    
    sigma_v = eff_volume_param * T_factor * (c_ratio**2) * (np.cos(theta_rad)**2) / cos_theta2_real
    
    # 4. 全反射截断
    # 当全反射发生时，sin_theta2 >= 1.0, cos_theta2 变为纯虚数，没有能量透射进入海底
    sin_theta2_abs = c_ratio * np.sin(theta_rad)
    sigma_v[sin_theta2_abs >= 1.0] = 0.0
    
    return sigma_v

def compute_volume_sigmoid(theta_rad):
    """
    计算体积散射的平滑过渡 Sigmoid 函数 (Equation 19 部分)。
    """
    theta_v = np.deg2rad(10.0)
    # 抑制近法向的体积散射
    sigmoid_V = 1 / (1 + np.exp(-180/np.pi * (theta_rad - theta_v) / 2.0))
    return sigmoid_V
