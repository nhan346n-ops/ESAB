from dataclasses import dataclass

@dataclass
class ESABParameters:
    """
    ESAB 模型的五个核心物理参数（与 Frontiers in Remote Sensing 2025 论文一致）。
    """
    z: float       # 声阻抗对比度 (Impedance ratio)
    mu_db: float   # 归一化频率处的体积散射参数，单位 dB (Volume scattering index)
    s1_deg: float  # 150kHz 处的表面粗糙度斜率 (Facet-slope std dev 1 at 150kHz)
    delta1: float  # 实际频率下的粗糙度斜率 1 (Facet-slope std dev 1 at f)
    delta2: float  # 实际频率下的粗糙度斜率 2 (Facet-slope std dev 2 at f)

    def __str__(self):
        return (f"z={self.z:.2f}, mu={self.mu_db:.2f}dB, "
                f"s1={self.s1_deg:.2f}°, delta1={self.delta1:.2f}°, delta2={self.delta2:.2f}°")
