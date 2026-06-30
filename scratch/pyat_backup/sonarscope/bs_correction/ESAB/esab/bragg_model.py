import numpy as np

def compute_bragg_scattering(theta_rad, gamma, V2_abs, delta1_sq):
    """
    精确计算布拉格微观粗糙度散射 (Equation 13/14)
    避免近法向 (theta -> 0) 时的数学奇异性 (人工截断)，通过严格极限处理。
    """
    D4 = (gamma - 2) / 2
    D3 = (2**(gamma - 5)) * (np.pi**(gamma - 3)) * ((4 - gamma)**D4) * ((gamma - 2)**(1 - D4))
    
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    
    # 初始化 sigma_B，当 sin_theta 接近 0 时，因为最终乘以 sigmoid_I 会趋于 0，所以这里直接将其设为 0。
    # 避免了之前的 np.clip(sin_theta, 5度) 的人工经验截断
    sigma_B = np.zeros_like(theta_rad)
    
    # 仅在 sin_theta 足够大时进行标准计算，防止除以极小值产生 inf
    valid_idx = sin_theta > 1e-8
    
    sigma_B[valid_idx] = (
        D3 * V2_abs[valid_idx] * (cos_theta[valid_idx]**4) / 
        (sin_theta[valid_idx]**gamma) * (delta1_sq**D4)
    )
    
    return sigma_B
