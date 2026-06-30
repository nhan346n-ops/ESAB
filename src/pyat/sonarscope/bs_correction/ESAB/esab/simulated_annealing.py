import numpy as np


def esab_cost_function(params, theta_rad, bs_measured, freq_hz, gamma, compute_bs_func):
    """
    加权 RMS 成本函数 (论文中强调 0-20 度权重更大)。

    params: [z, mu_db, s1, delta2]
    - s1: 归一化粗糙度斜率 (在参考频率 f0=150kHz 处的值, Eq 11)
      有效粗糙度 delta1(f) = s1 * (f/f0)^(D2/2), D2 = (gamma-2)/2
    """
    z, mu_db, s1, delta2 = params

    # 物理约束惩罚
    if z < 1.0 or z > 14.0:
        return 1e6
    if mu_db < -15.0 or mu_db > 15.0:
        return 1e6
    if s1 < 0.5 or s1 > 70.0:
        return 1e6
    if delta2 < 0.5 or delta2 > 80.0:
        return 1e6
    # delta2 必须大于当前频率下的有效 delta1
    D2 = (gamma - 2) / 2
    delta1_eff = s1 * (freq_hz / 150000.0) ** (D2 / 2)
    if delta2 <= delta1_eff:
        return 1e6

    # 计算预测值 — s1 在 compute_esab_bs 中通过频率缩放得到 delta1(f)
    bs_pred = compute_bs_func(np.rad2deg(theta_rad), freq_hz, z=z, s1_deg=s1,
                              delta1_deg=s1, delta2_deg=delta2, mu_db=mu_db, gamma=gamma)

    # 加权误差
    theta_deg = np.rad2deg(theta_rad)
    weights = np.where(theta_deg <= 20.0, 2.0, 1.0)
    errors = (bs_pred - bs_measured)**2
    weighted_mse = np.sum(weights * errors) / np.sum(weights)

    return np.sqrt(weighted_mse)


def run_sa(theta_rad, bs_measured, freq_hz, gamma, compute_bs_func, initial_guess=None, n_cycles=15, iters_per_cycle=10000, step_scale=1.0):
    """
    严格按照论文实现的 Simulated Annealing。
    T0 = 0.1, Nc = n_cycles, iters_per_cycle configurable.
    T = T0 / log2(1 + nc)

    Parameters
    ----------
    n_cycles : int
        冷却周期数 (默认15，有PINN初值时可用3-5)
    iters_per_cycle : int
        每周期迭代数 (默认10000，有PINN初值时可用3000-5000)
    step_scale : float
        扰动步长缩放 (有PINN初值时可用0.5-1.0)
    """
    T0 = 0.1
    Nc = n_cycles

    # 参数: [z, mu_db, s1, delta2]
    # s1 是归一化到参考频率 f0=150kHz 的粗糙度斜率
    if initial_guess is None:
        current_state = np.array([2.5, -5.0, 5.0, 10.0])
    else:
        current_state = np.array(initial_guess)
        # 边界保护
        current_state = np.clip(current_state, [1.0, -15.0, 0.5, 1.0], [14.0, 15.0, 70.0, 80.0])

    current_energy = esab_cost_function(current_state, theta_rad, bs_measured,
                                         freq_hz, gamma, compute_bs_func)

    best_state = current_state.copy()
    best_energy = current_energy

    # 扰动步长 (step_scale 缩小初值附近的搜索范围)
    step_sizes = np.array([0.5, 1.0, 2.0, 2.0]) * step_scale

    for nc in range(1, Nc + 1):
        T = T0 / np.log2(1 + nc)

        for _ in range(iters_per_cycle):
            # 随机选择一个参数进行高斯扰动
            param_idx = np.random.randint(0, 4)
            new_state = current_state.copy()
            new_state[param_idx] += np.random.normal(0, step_sizes[param_idx])

            # 计算新能量
            new_energy = esab_cost_function(new_state, theta_rad, bs_measured,
                                             freq_hz, gamma, compute_bs_func)

            # Metropolis 准则
            delta_E = new_energy - current_energy
            if delta_E < 0:
                accept = True
            else:
                p_accept = np.exp(-delta_E / T) if delta_E / T < 500 else 0.0
                accept = np.random.rand() < p_accept

            if accept:
                current_state = new_state
                current_energy = new_energy

                # 更新全局最优
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = current_state.copy()

    return best_state, best_energy
