# PyAT 深海之眼 v0.3-production：声学物理约束驱动的多模态海底底质语义分割

> **目标**：在声学物理约束（频率条件嵌入、BL3 处理级别、Hamilton 先验、尺度解耦）与数据驱动（Swin MAE + Mask2Former）深度融合框架下，实现跨频率、跨分辨率的像素级海底底质分类，达到海洋地质学顶刊学术严谨度标准。
> **版本**: v0.3-production
> **创建**: 2026-05-23 | **更新**: 2026-05-23

---

## 1. TL;DR

- **核心思想**：在 **BL3 级** DTM 数据（仅辐射校正、照射面积校正，保留原始角度依赖）上，以声呐工作频率为条件嵌入、以局部入射角图为几何约束、以 BSAR 曲线 Hamilton-物理拟合参数为先验知识，驱动 Swin-MAE 预训练 + Mask2Former 多模态底质语义分割。
- **输入**：
  - 2D（3 通道，**BL3 级**）：`DTM.backscatter`（dB，保留角度依赖）、`DTM.elevation`（m）、**`DTM.incidence_angle`（°）**
  - 标量条件：**声呐频率 Token**（Hz，频率条件嵌入） + **GSD Token**（m，地面采样距离）
  - 1D：BSAR → **Hamilton 先验约束 Jackson 拟合** / **Lambert-Gaussian 鲁棒退避** 物理参数
- **输出**：底质分类 GeoTIFF + 贝叶斯置信度 map + 分类统计报告
- **与现有管线对接**：工具 1（Sounder→DTM）输出 **BL3** DTM → 工具 4（本方案）；入射角图层需新增后端计算。
- **学术定位**：IEEE TGRS / Marine Geology 级别，核心创新为"频率条件嵌入 + BL3 物理约束 + GSD 尺度解耦"三位一体的声学物理融合范式。

---

## 2. 方案总览

```
┌────────────────────────────────────────────────────────────────────┐
│                    阶段 1: 预训练                                    │
│                                                                     │
│  多源 DTM（1-115m）→ 保持原生 GSD + GSD Token                      │
│  + 频率条件嵌入（12kHz ~ 300kHz）[v0.3 修正]                       │
│  + 入射角通道（BL3 级输入独有）[v0.3 严格界定]                       │
│       ↓                                                             │
│  Masked Autoencoder (MAE) — 随机遮挡 75% patch                      │
│  输入: [BS, elev, θ] + freq_embed + gsd_embed                      │
│       ↓                                                             │
│  预训练 Swin-Base 权重（跨频率、跨分辨率）                            │
└────────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────────┐
│                    阶段 2: 微调                                      │
│                                                                     │
│  BL3 DTM 图块（有底质标签）                                         │
│  + BSAR 曲线 → Hamilton 先验约束 Jackson 拟合 [v0.3 抗异谱风险]     │
│    → 失败时 Lambert-Gaussian 鲁棒退避                               │
│  + 频率/ GSD 条件嵌入 → 尺度不变特征                                │
│  + Nadir 角度掩膜数据增强                                           │
│       ↓                                                             │
│  Cross-Attention: 物理参数 token(Q) + 2D patch(K/V) = 底质分类     │
│       ↓                                                             │
│  像素级底质分类（6 类）→ 置信度 map → 底质图                        │
└────────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────────┐
│                    阶段 3: 部署                                      │
│                                                                     │
│  ONNX Runtime → 滑动窗口推理（保持原生 DTM GSD）                    │
│  输入检测: 自动识别 BL3/BL4/未知 级别并提示                          │
│  新增工具 4: 底质分类 → GUI 集成                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. 算法架构

### 3.1 预训练阶段 — 物理条件驱动多通道 Swin-MAE

| 组件 | 选择 | 物理/工程理由 |
|------|------|---------|
| 编码器 | **Swin Transformer V2** (Swin-B, window=7×7, 4 stages) | Hierarchical 结构兼容多尺度底质纹理；shifted window 提供跨窗口上下文 |
| 预训练策略 | **Masked Autoencoder (MAE)** | 无标注预训练；迫使编码器从局部 patch 重建全局场景 |
| 遮挡比例 | 75% random patches | 高遮挡率迫使学习语义而非像素邻接关系 |
| 解码器 | 轻量 Swin Decoder（4 层, embed_dim=256） | 仅预训练阶段 |
| **2D 输入通道** | **3 通道**: backscatter(dB, **BL3**) + elevation(m) + **incidence_angle(°)** | BL3 保留了角度依赖；三个通道一起输入使模型能解耦声呐方程 $BS(\theta, z, \text{底质})$ |
| **标量条件嵌入** | **频率 Token + GSD Token** [v0.3 新增] | 见下方核心修正 ① 和 ④ |
| 输入尺寸 | **可变**（保持原生 GSD，空间覆盖约 1km²） | 见核心修正 ④ |

---

#### ▸ v0.3 核心修正 ①：频率条件嵌入（替代 v0.2 的 dB 跨频率换算）

**v0.2 谬误诊断**：

原方案企图用 Francois-Garrison 吸收公式 $\Delta BS = -2[\alpha(f) - \alpha(f_{\text{ref}})]\cdot z$ 将 12kHz 与 300kHz 的 dB 值进行绝对校准。这在物理上是**根本性错误**的：

| 频率 | 散射机制 | 穿透深度 | 携带的信息 |
|------|---------|---------|-----------|
| 12kHz（EM122） | **体积散射主导** | 数米~数十米 | 沉积层结构、层理、体积不均匀体 |
| 200-300kHz（EM2040） | **界面散射主导** | <0.1m | 海底表面粗糙度、孔隙度 |

两种频率下测得的 $BS$ 差异不仅来自水柱吸收——**海底本身的散射机制截然不同**。12kHz 的回声包含沉积层内部信息，300kHz 仅反映表层。水柱吸收修正（Francois-Garrison）只纠正了传播路径上的衰减差异，无法修正散射机制的物理差异。强行换算会制造"假纹理"——把 12kHz 的体积散射信号错误地映射为 300kHz 的界面散射信号。

**v0.3 修正方案 — 频率条件嵌入（Frequency Conditional Embedding）**：

```
声呐工作频率 f (Hz)
     ↓
f_norm = log10(f) - log10(12e3) / (log10(300e3) - log10(12e3))   [0, 1] 归一化
     ↓
nn.Embedding(64) 或 nn.Linear(1 → 64) [可学习]
     ↓
FreqToken (64-dim) → + Swin Patch Embedding / 作为 class token
     ↓
模型自动学习不同频率下的底质响应差异（界面散射 vs 体积散射）
```

- 不假设 12kHz 和 300kHz 的 dB 可以互相换算——让模型从数据中自适应学习不同频率的底质响应分布
- 64 维嵌入向量可以编码频率的连续变化（整个 12kHz-300kHz 连续谱），而非离散的声呐型号分类
- 在推理时，从 XSF 文件的 `transmit_frequency_start` 全局属性读取频率，归一化后作为条件

**实现**：

```python
class FrequencyConditionalSwin(SwinTransformer):
    def __init__(self, freq_dim=64):
        super().__init__()
        self.freq_embed = nn.Linear(1, freq_dim)
        # freq_embed 加到 patch_embed 输出或 position encoding

    def forward(self, x, freq_hz):
        # freq_hz: [B, 1] normalized to [0, 1]
        freq_token = self.freq_embed(freq_hz)      # [B, 64]
        x = self.patch_embed(x)                     # [B, N, C]
        x = x + freq_token.unsqueeze(1)             # broadcast over patches
        x = x + self.pos_embed
        ...
```

**可行性**：⭐⭐⭐⭐⭐ | 零额外依赖，PyTorch 原生 nn.Linear + nn.Embedding。

---

#### ▸ v0.3 核心修正 ②：严格定义输入处理级别 — 必须为 BL3

**v0.2 缺陷**：未定义 `DTM.backscatter` 的校正级别。如果输入的是已消除角度依赖的 **BL4**（`backscatterCorrection = ON`，即工具 2B S1 或工具 2A 的产出），再输入 `incidence_angle` 通道将**毫无意义**——因为角度依赖已经被移除，入射角通道与后向散射值之间不存在物理映射关系，模型学不到任何角度效应。

**PyAT 处理级别定义**：

| 级别 | 名称 | 操作 | 前端标记 |
|------|------|------|---------|
| **BL0** | 原始/检测值 | 无操作 | — |
| **BL1** | 物理修正 | 去除 BSCorr | — |
| **BL2** | 物理/几何修正 | BL1 + 去除角增益 + 重算照射面积 | DTM（工具 1 输出） |
| **BL3** | 辐射校正 | BL2 + 无源级/吸收标准化 (即保留原始角度衰减特征) | **本方案输入** |
| **BL4** | 角度重规范化 | BL3 + 应用角度校正 → `backscatterCorrection=ON` | 绿色标记 XSF（工具 2B S1/2A 输出） |

**严格约束**：本方案的 2D 输入必须是 **BL3 级** DTM，即：
- 已完成辐射校正（`remove_compensation=True`，`remove_calibration=True`）
- **未**经历角度重规范化（`backscatterCorrection ≠ ON`）
- 保留了 `BS(θ)` 的原始角度依赖

**实现**：工具 4 对话框在加载 DTM 时自动检测其来源。如果是工具 2B S1/2A 的输出（绿色标记 XSF），则提示用户需要对应的 BL3 级 DTM 作为 2D 输入。

---

#### ▸ v0.3 核心修正 ④：尺度解耦（Scale-Aware Input）+ GSD Token

**v0.2 缺陷**：强制所有输入到 5m 分辨率、224×224 像素，同时摧毁了两种极端：

```
12kHz 深水 DTM（原生 50m）→ 上采样到 5m → 每像素无信息，只有插值假纹理 ❌
300kHz 浅水 DTM（原生 1m） → 下采样到 5m → 丢失沙波/沙纹细节的微观结构 ❌
```

**v0.3 修正 — 尺度解耦**：

1. **保持原生 GSD**：输入 patch 时不做统一重采样。允许 DTM 以原始最佳分辨率输入。
2. **GSD 条件嵌入（Ground Sample Distance Token）**：

```python
GSD_norm = log10(GSD_原生) - log10(0.5m) / (log10(100m) - log10(0.5m))   # [0, 1]
GSD_token = nn.Linear(1, 64)(GSD_norm)   # 可学习的 GSD 嵌入

# 与频率 Token 拼接为联合条件
cond_token = GSD_token + freq_token       # element-wise or concat
x = x + cond_token.unsqueeze(1)
```

3. **空间覆盖恒定**：不同 GSD 的 patch 覆盖相同的物理面积（约 1km²），因此像素数可变（如 50m GSD → 20×20 px, 1m GSD → 1000×1000 px）。Swin Transformer 的窗口机制天然支持可变输入尺寸。

**物理意义**：模型通过 GSD Token 知道"当前输入的每个像素代表多少米的物理空间"，从而在同一套权重下同时理解：
- 200m 尺度的深海沉积扇形态（12kHz，低 GSD）
- 2m 尺度的近岸沙波迁移（300kHz，高 GSD）

---

### 3.2 微调阶段 — 多模态语义分割

| 组件 | v0.2 方案 | **v0.3 方案** | 变更理由 |
|------|-----------|-----------|---------|
| 骨干网络 | Swin-B（冻结前 3 层） | Swin-B（冻结前 3 层） | 不变 |
| 条件嵌入 | 无 | **频率 Token + GSD Token + 物理参数 Token** | 三条件联合驱动 |
| BSAR 编码 | Jackson/Lambert 自由拟合 | **Hamilton 先验约束 Jackson + Lambert-Gaussian 退避** | 见下方核心修正 ③ |
| 融合 | Cross-Attention | Cross-Attention | 不变 |
| 分割头 | Mask2Former | Mask2Former | 不变 |
| 类别 | 6 类 | 6 类 | 不变 |

#### ▸ v0.3 核心修正 ③：Hamilton 先验约束 + Lambert-Gaussian 鲁棒退避

**v0.2 缺陷**：无约束的 scipy 曲线拟合在高噪声环境下极易陷入**"异物同谱（Equifinality）"**——不同的底质类型（如软泥和含砾石粗沙）拟合出几乎相同的 Jackson 参数。这是声学反演的经典病态问题：90 个数据点 + 高噪声 → 解空间存在多个局部最优解。

**v0.3 修正 — 双重防线**：

**第一道防线：Hamilton 地质声学先验边界（Geoacoustic Priors）**

基于 Hamilton (1970, 1980) 和 APL-UW (2023) 建立的全球海底沉积层声学数据库，为 Jackson 模型参数设置物理边界：

```python
# Hamilton 地质声学先验边界（面向 Jackson 模型参数）
HAMILTON_PRIORS = {
    "Mud":    {"density_ratio": (1.2, 1.5), "velocity_ratio": (0.98, 1.02), "attenuation": (0.1, 0.5)},
    "Silt":   {"density_ratio": (1.4, 1.7), "velocity_ratio": (0.99, 1.04), "attenuation": (0.3, 0.8)},
    "Sand":   {"density_ratio": (1.7, 2.0), "velocity_ratio": (1.05, 1.15), "attenuation": (0.5, 1.5)},
    "Gravel": {"density_ratio": (1.8, 2.2), "velocity_ratio": (1.10, 1.20), "attenuation": (0.8, 2.0)},
    "Rock":   {"density_ratio": (2.0, 2.8), "velocity_ratio": (1.15, 1.30), "attenuation": (0.5, 1.0)},
}
```

在 `scipy.optimize.curve_fit` 中通过 `bounds` 参数将这些先验范围作为拟合上下界，确保拟合结果不违反已知的地质声学极限。

**第二道防线：Lambert-Gaussian 鲁棒退避机制**

当 Jackson 模型拟合质量不达标时，自动降级到更鲁棒的 Lambert-Gaussian 经验模型：

```python
def fit_bsar_curve(angle, mean_bs, value_count):
    # Stage 1: 尝试 Jackson 模型（带 Hamilton 先验边界）
    try:
        popt, R2 = fit_jackson_with_priors(angle, mean_bs, valid_mask=value_count > 0)
        if R2 > 0.85 and all_params_physical(popt, HAMILTON_PRIORS):
            return {"model": "jackson", "params": popt, "R2": R2}
    except (RuntimeError, ValueError):
        pass
    
    # Stage 2: Lambert-Gaussian 经验模型退避
    # BS(θ) = BS₀ + 10·k·log₁₀(cosθ) + ε_GP(θ)
    #   其中 ε_GP 为高斯过程残差，捕捉非兰伯特分量
    popt_lg, R2_lg = fit_lambert_gaussian(angle, mean_bs)
    return {"model": "lambert_gaussian", "params": popt_lg, "R2": R2_lg}
```

Lambert-Gaussian 模型只有 2-3 个自由参数（$BS_0$、$k$、$\sigma_{GP}$），比 Jackson 模型（5-6 参数）更鲁棒，适合低信噪比场景。

**退避触发条件**：
- R² < 0.85
- 拟合参数超出 Hamilton 先验边界
- `value_count` 在关键角度区（10°~60°）有超过 30% 的 NaN
- 收敛失败（奇异矩阵、迭代超限）

---

### 3.3 推理管线（v0.3）

```
用户选择 XSF 文件（或 DTM）
      ↓
工具 4 对话框：
  ├── DTM 级别检测: BL3 → 继续 | BL4 → 警告"请使用角度归一化前的 DTM"
  ├── 读取代 frequency_start → 构建频率 Token
  └── 读取 DTM 原生 GSD → 构建 GSD Token
      ↓
  ├── 入射角通道: DTM incidence_angle 图层（优先）或 在线计算
  ├── BSAR: .bsar.nc → Hamilton-Jackson/Lambert-Gaussian → 物理参数 Token
  └── 滑动窗口: 保持原生 GSD，patch 中心间距 = min(224×GSD, 500m)
      ↓
ONNX Runtime 推理
      ↓
输出: classified_{name}.tif / confidence_{name}.tif / stats_{name}.json
      ↓
地图视图自动加载
```

---

## 4. 训练数据方案

### 4.1 自监督预训练（无标注）

| 来源 | 频率范围 | GSD 范围 | 可用性 |
|------|---------|---------|:--:|
| PyAT 用户自有 XSF（工具 1 产出，BL3） | 12~300kHz | 1~30m | ✅ |
| EMODnet 2024 DTM | 12~100kHz 混合 | 115m | ✅ |
| NOAA NCEI multibeam | 12~200kHz 混合 | 30~100m | ✅ |

**预训练数据加载（v0.3）**：

```python
class SeaLearnPretrainDataset(Dataset):
    def __getitem__(self, idx):
        dtm, meta = self.load_dtm(self.file_list[idx])
        
        # v0.3: 保持原生 GSD，不重采样
        bs_bl3 = dtm["backscatter"]    # BL3 级
        elev   = dtm["elevation"]
        theta  = dtm.get("incidence_angle", compute_theta_nadirs(elev))
        
        # v0.3: 频谱条件 → 频率 Token
        freq_hz = meta["transmit_frequency_start"]  # from XSF global attr
        freq_norm = (np.log10(freq_hz) - np.log10(12e3)) / (np.log10(300e3) - np.log10(12e3))
        
        # v0.3: GSD → GSD Token
        gsd = meta["spatial_resolution"] / np.cos(np.radians(mean_lat))  # 估计地面采样距离
        gsd_norm = (np.log10(gsd) - np.log10(0.5)) / (np.log10(100) - np.log10(0.5))
        
        # 3 通道 + 2 条件嵌入
        patch_3ch = np.stack([bs_bl3, elev, theta], axis=-1)
        return patch_3ch, freq_norm, gsd_norm
```

### 4.2 微调（需要底质标签）

同 v0.2，EMODnet Seabed Substrate + BALM + 弱监督伪标签。

### 4.3 数据增强（v0.3 扩展）

| 增强 | 方法 | 目的 |
|------|------|------|
| 几何增强 | 翻转、旋转（90°倍数）、随机缩放 | 空间不变性 |
| 强度增强 | 对比度 ±20%、亮度 ±10%、σ=0.5dB 高斯噪声 | 信噪比鲁棒 |
| 混合增强 | CutMix + MixUp | 类别不平衡 |
| Nadir 角度掩膜 | 随机 Mask 0°~10° 入射角像素 | 防镜面反射伪影 |
| **频率抖动** | 频率 Token ±5% 随机偏移（模拟频率误差） | 频率泛化鲁棒性 [v0.3 新增] |
| **GSD 抖动** | GSD Token ±10% 随机偏移 | 分辨率泛化鲁棒性 [v0.3 新增] |
| 间隙填充强制 | 使用 fill_holes DTM | 防梯度爆炸 |

---

## 5. 项目集成方案

### 5.1 目录结构（v0.3 修正）

```
src/pyat/sealearn/
├── __init__.py
├── pretrain/
│   ├── mae.py                       # MAE（3 通道 + 频率/GSD 条件嵌入）
│   ├── swin_cond_encoder.py         # 条件 Swin（新增 freq_embed + gsd_embed）
│   └── dataset.py                   # 数据加载器（保持原生 GSD + 条件提取）
├── finetune/
│   ├── swin_seg_head.py             # Mask2Former 分割头（带条件）
│   ├── bsar_fitter.py               # Hamilton-Jackson + Lambert-Gaussian [v0.3 重写]
│   ├── hamilton_priors.json         # Hamilton 地质声学先验数据库 [v0.3 新增]
│   ├── physics_token_encoder.py     # 物理参数 → embedding
│   ├── cross_attention.py
│   └── trainer.py
├── inference/
│   ├── onnx_export.py
│   ├── predictor.py                 # 推理引擎（读 DTM 级别检测）
│   └── sliding.py                   # 保持原生 GSD 的滑动窗口推理
├── utils/
│   ├── dtm_loader.py                # BL3 级别检测 + 3 通道读取
│   ├── bsar_loader.py               # BSAR → 物理参数
│   ├── level_detector.py            # 自动识别 DTM 处理级别 [v0.3 新增]
│   ├── augmentations.py             # 含 Nadir 掩膜 + 频率/GSD 抖动
│   ├── incidence_computer.py        # 在线入射角估算
│   └── metrics.py
└── configs/
    ├── mae_pretrain.yaml            # 含 freq_embed_dim + gsd_embed_dim
    └── seg_finetune.yaml
```

### 5.2 后端新增

- `SounderToDtmExporter` 新增可选 `incidence_angle` 图层（同 v0.2）
- XSF 文件读取时暴露 `transmit_frequency_start / stop` 和 `sounder_type` 给 metadata → 供频率 Token 构建

### 5.3 GUI 草图（v0.3）

```
┌─────────────────────────────────────────────────────────────┐
│  工具 4: 海底底质语义分割（Sea-Learn v0.3-production）       │
│                                                              │
│  输入 DTM:                              处理级别检测: BL3 ✅ │
│  ┌────────────────────────────────────────────────────┐      │
│  │ 后向散射 20260522_bathy.dtm.nc           浏览       │      │
│  │ 入射角   20260522_bathy.dtm.nc (可选)     浏览       │      │
│  │ BSAR     bsar_model.bsar.nc              浏览       │      │
│  └────────────────────────────────────────────────────┘      │
│  声呐频率: 200.0 kHz (自动从 XSF 读取)          [自动]      │
│  原生 GSD: 5.0 m                                         │
│                                                              │
│  BSAR 拟合: ◉ 自动 (Hamilton-Jackson → Lambert-Gaussian)     │
│             ○ 强制 Jackson   ○ 强制 Lambert-Gaussian         │
│                                                              │
│  输出: ☑ GeoTIFF  ☑ 置信度  ☑ JSON 统计                     │
│  ┌── 运行 ──┐  ┌ 取消 ┐                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 训练计划

| 阶段 | GPU | VRAM | 时长 |
|------|-----|------|------|
| MAE 预训练（3 通道 + 条件嵌入） | 1× RTX 4090 | 24GB | 72h / 100 epochs |
| 分割微调 | 1× RTX 4090 | 24GB | 10h / 50 epochs |
| ONNX 推理 | CPU | — | 3-5s per patch |

| 指标 | 预训练 | 微调 |
|------|:---:|:---:|
| MSE (3ch) | < 0.05 | — |
| mIoU | — | > 0.75 |
| **跨频率 mIoU** [v0.3] | — | > 0.68（12kHz 测试集）|
| **Nadir mIoU** | — | > 0.65 |
| **跨 GSD mIoU** [v0.3] | — | > 0.70（<2m 和 >30m 数据）|

---

## 7. 评价体系（v0.3）

| 指标 | 说明 |
|------|------|
| mIoU | 标准语义分割 |
| **跨频率 mIoU** | 分别评估 12kHz、100kHz、200kHz+ 测试集 — 验证频率条件嵌入有效性 |
| **跨 GSD mIoU** | 分别评估 <2m、2~10m、>30m GSD 测试集 — 验证 GSD Token 有效性 |
| **Nadir mIoU** | 仅评估 0°~10° 入射角区 — 验证入射角通道 + BL3 约束 |
| Jackson vs LG 退避 mIoU | 仅评估 R²<0.85 图块的 mIoU — 验证 Lambert-Gaussian 退避效果 |
| Equifinality 偏差 | 评估同一 BSAR 曲线多次拟合的参数标准差 — 验证 Hamilton 先验约束效果 |
| Bayesian Confidence | MC Dropout |

---

## 8. 参考文献

| 年份 | 工作 | 与本方案关系 |
|------|------|------|
| 1970 | **Hamilton, E. L.** "Geoacoustic models of the sea floor" | Hamilton 地质声学先验数据库 |
| 1980 | **Hamilton, E. L.** "Geoacoustic modeling of the sea floor" | 沉积层声学参数综合 |
| 1986 | **Jackson et al.** "Tests of models for backscatter" | BSAR 物理模型 |
| 2009 | **APL-UW TR 0703** "High-Frequency Seafloor Acoustics" | 高频海底散射标准参考 |
| 2022 | **He et al.** "Masked Autoencoders Are Scalable Vision Learners" | MAE 预训练 |
| 2023 | **Liu et al.** "Swin Transformer V2" | 骨干网络 |
| 2024 | **Jakubik et al.** "Prithvi V2" (NASA) | 地学基础模型参考 |
| 2024 | **IFREMER** "BSAR v0.3" | 本项目的物理数据源 |

---

## 9. 公开数据集

| 数据集 | 内容 | 用途 |
|--------|------|------|
| **BALM** | 500+ BSAR 曲线含底质标签 | 微调 + Hamilton 先验验证 |
| **EMODnet Seabed Substrate 1:250k** | 欧洲底质图 | 弱监督标签 |
| **EMODnet 2024 DTM** | 115m 泛欧 DTM | 预训练（需频率标记） |
| **NOAA NCEI multibeam** | 全球多波束 archive | 预训练（需频率标记） |
| **Hamilton (1970/1980)** | 全球 500+ 站位沉积声学参数 | 先验约束数据库 |

---

## 10. 风险与缓解（v0.3）

| 风险 | v0.3 缓解措施 |
|------|---------|
| 标注数据不足 | 物理参数聚类→弱监督伪标签 + Hamilton 先验减少所需标注量 |
| 频率差异导致泛化差 | **频率条件嵌入**（§3.1）— 模型自适应学习不同频率的散射物理 |
| BL4 数据误输入 | **级别检测器**（§5.1 `level_detector.py`）自动判断并提示用户 |
| Jackson 拟合异谱同形 | **Hamilton 先验边界**（§3.2）限制参数空间 + **Lambert-Gaussian 退避** |
| 分辨率差异导致纹理错乱 | **GSD Token**（§3.1）— 模型感知物理采样尺度 |
| Nadir 镜面伪影 | 入射角通道 + Nadir 掩膜增强 |
| 测线间隙梯度爆炸 | 强制输入 fill_holes DTM |
| 计算资源不足 | Swin-B→Swin-T (28M)，频率/GSD Token 尺寸 64→32 |

---

## 11. v0.2 → v0.3 变更总结

| # | 修正项 | v0.2（错误/不足） | v0.3（正确） | 影响 |
|---|--------|------------------|-------------|------|
| ① | 频率归一化 | Francois-Garrison 跨频率 dB 换算 | **频率条件嵌入**（模型自适应学习） | 避免物理谬误 |
| ② | 输入级别定义 | 未定义 `DTM.backscatter` 处理级别 | **严格 BL3** + 级别检测器 | 避免入射角通道无效化 |
| ③ | Jackson 拟合 | 自由拟合，无先验约束 | **Hamilton 先验边界** + **Lambert-Gaussian 退避** | 抗异谱同形 |
| ④ | 分辨率处理 | 强制 5m/224px | **保持原生 GSD** + **GSD Token** | 跨分辨率兼容 |

---

## 12. 最终交付物

1. 预训练权重 `swin_mae_pretrain_v3.pth`（3 通道 + 条件嵌入）
2. ONNX 模型 `seabed_seg_v3.onnx`
3. 模块 `src/pyat/sealearn/`（含 `swin_cond_encoder.py`、`bsar_fitter.py`、`hamilton_priors.json`、`level_detector.py`）
4. 入射角图层后端扩展 `sounder_to_dtm.py`
5. 频率/GSD 条件嵌入代码 `freq_gsd_embed.py`
6. 工具 4 GUI + 技术报告 `docs/sealearn_v3.md`（目标期刊格式：IEEE TGRS / Marine Geology）
