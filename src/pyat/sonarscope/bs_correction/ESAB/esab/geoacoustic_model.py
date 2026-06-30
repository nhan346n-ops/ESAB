import numpy as np

class GeoacousticModel:
    """
    根据 Hamilton (1980) 和 Hamilton and Bachman (1982) 的经验公式，
    从阻抗对比度 (z) 恢复沉积物的密度、声速和衰减系数。
    """
    def __init__(self, c1=1500.0, rho1=1025.0):
        """
        c1: 海水声速 (m/s)
        rho1: 海水密度 (kg/m^3)
        """
        self.c1 = c1
        self.rho1 = rho1

    def recover_properties(self, z):
        """
        根据声阻抗比 z 恢复底质物理参数。
        
        Returns:
            dict: 包含密度 rho2, 声速 c2, 衰减常数 beta (dB/m/kHz)
        """
        # 1. 声速比近似公式
        c_ratio = 0.7030 + 0.2055 * z
        c2 = c_ratio * self.c1
        
        # 2. 密度恢复
        # z = (rho2 * c2) / (rho1 * c1) = (rho2 / rho1) * c_ratio
        rho2 = (z / c_ratio) * self.rho1
        
        # 3. 衰减系数近似 (Hamilton 1980: beta 随频率线性增加)
        # 这里给出一个典型的经验估计，粗略与密度和声速相关。实际更精确的值需要查表。
        # 普遍规律：泥 (低 z) 衰减较小，砂 (中等 z) 衰减较大。
        # 这里返回一个随经验分布的 dB/m/kHz 系数。
        if z < 1.8:
            beta_k = 0.05 # Mud
        elif z < 3.0:
            beta_k = 0.4  # Sand
        else:
            beta_k = 0.1  # Rock (通常很小)
            
        return {
            "rho2": rho2,
            "c2": c2,
            "c_ratio": c_ratio,
            "beta_k": beta_k
        }

if __name__ == "__main__":
    gm = GeoacousticModel()
    print("MUD (z=1.7):", gm.recover_properties(1.7))
    print("SAND (z=2.5):", gm.recover_properties(2.5))
    print("ROCK (z=6.0):", gm.recover_properties(6.0))
