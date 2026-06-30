import numpy as np
from .simulated_annealing import run_sa
from .esab_core import compute_esab_bs


def invert_esab_single_freq(theta_deg, bs_measured, freq_hz, gamma=10/3,
                            initial_guess=None, n_cycles=15, iters_per_cycle=10000,
                            step_scale=1.0):
    """
    对单个频率进行 ESAB 反演，提取 z, mu, s1, delta2。

    Parameters
    ----------
    theta_deg : np.ndarray
        入射角 (度)
    bs_measured : np.ndarray
        实测后向散射强度 (dB)
    freq_hz : float
        工作频率 (Hz)
    gamma : float
        粗糙度频谱指数 (默认 10/3)
    initial_guess : list or None
        初始猜测 [z, mu_db, s1, delta2]
    n_cycles : int
        SA 冷却周期数 (有 PINN 初值时可用 3-5)
    iters_per_cycle : int
        SA 每周期迭代数 (有 PINN 初值时可用 3000-5000)
    step_scale : float
        SA 扰动步长缩放 (有 PINN 初值时可用 0.5-1.0)

    Returns
    -------
    dict with keys:
        z: 声阻抗对比度 (频率无关)
        mu_db: 体积散射参数 (dB, 归一化到 f0=150kHz)
        s1: 归一化粗糙度斜率 (度, 归一化到 f0=150kHz)
        delta2: 第二高斯分量粗糙度 (度, 当前频率有效值)
        rmse: 反演均方根误差 (dB)
    """
    theta_rad = np.deg2rad(np.abs(theta_deg))

    best_state, best_energy = run_sa(
        theta_rad, bs_measured, freq_hz, gamma, compute_esab_bs,
        initial_guess, n_cycles, iters_per_cycle, step_scale
    )

    z, mu_db, s1, delta2 = best_state

    return {
        "z": z,
        "mu_db": mu_db,
        "s1": s1,
        "delta1": s1,  # 向后兼容 — delta1 返回 s1 的值
        "delta2": delta2,
        "rmse": best_energy
    }
