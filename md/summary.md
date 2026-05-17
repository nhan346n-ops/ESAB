# PyAT 多波束后向散射全流程——前端功能与后端算法大全

> **适用版本**: PyAT v0.1.40+ / GUI Phase 4  
> **文档生成日期**: 2026-05-18  
> **数据库**: SONAR-netCDF4 (.xsf.nc) · DTM NetCDF (.dtm.nc) · BSAR NetCDF (.bsar.nc)

---

## 目录

1. [第〇章 多波束声呐后向散射物理基础](#0-多波束声呐后向散射物理基础)
2. [第一章 整体处理流程概览](#1-整体处理流程概览)
3. [第二章 声纳至 DTM (Sounder to DTM)——工具 1](#2-工具-1-声纳至-dtm)
4. [第三章 统计角响应 BSAR 模型计算——工具 2B Step 2](#3-工具-2b-step-2-统计角响应-bsar-模型计算)
5. [第四章 静态角度重规范化——工具 2B Step 1](#4-工具-2b-step-1-静态角度重规范化)
6. [第五章 滑动角度重规范化——工具 2A](#5-工具-2a-滑动角度重规范化)
7. [第六章 BSAR 角度响应曲线可视化——BSAR Viewer](#6-bsar-角度响应曲线可视化bsar-viewer)
8. [第七章 DTM 导入 / 图层可视化 / GeoTIFF 导出](#7-dtm-导入与-geotiff-导出)
9. [第八章 物理 / 数学公式专业详解](#8-物理数学公式专业详解)
10. [附录 A: 前端-后端文件映射表](#附录-a-前端-后端文件映射表)
11. [附录 B: SONAR-netCDF4 关键变量](#附录-b-sonar-netcdf4-关键变量)
12. [附录 C: BSAR NetCDF 文件结构](#附录-c-bsar-netcdf-文件结构)
13. [附录 D: 前端 UI 完成清单](#附录-d-前端-ui-完成清单)

---

## 0. 多波束声呐后向散射物理基础

### 0.1 什么是多波束后向散射

多波束回声测深仪（Multibeam Echosounder, MBES）通过发射扇形声脉冲，阵列接收海底反向散射能量。除了测量水深（Bathymetry），每个波束还记录反向散射信号的强度——这就是后向散射（Backscatter）。

**后向散射的物理定义**: 海底反向散射强度 $BS$ 是描述海底单位面积反向散射能力的声学量。它反映**海底物质的声学阻抗对比度**——即海底对入射声波的反射效率。

### 0.2 声呐方程（SONAR Equation）

完整的主动声呐方程为:

$$\boxed{EL = SL - 2TL + BS + 10\log_{10}(A)}$$

或等价地:

$$\boxed{BS = EL - SL + 2TL - 10\log_{10}(A_{\text{insonified}})}$$

其中各符号含义：

| 符号 | 名称 | 物理含义 | 典型单位 |
|---|---|---|---|
| $EL$ | Echo Level（回声级） | 接收换能器测得的回波信号强度 | dB re 1 μPa |
| $SL$ | Source Level（声源级） | 发射换能器在水中的声压级，距声源 1 米处 | dB re 1 μPa @ 1 m |
| $TL$ | Transmission Loss（传输损失） | 声波在水中传播时的能量衰减，含球面扩展损失和吸收损失 | dB |
| $2TL$ | 双向传输损失 | 声波从发射到海底再返回的全程损失（往返） | dB |
| $A$ | Insonified Area（照射面积） | 海底被声呐波束照射的有效面积 | m² |
| $BS$ | 后向散射强度 | 海底单位面积的后向散射能力，**底质分类的核心特征量** | dB re 1 m² |

**声呐方程的物理意义**: 该方程将"接收到的信号强度"（EL）分解为"发射"（SL）、"传播"（2TL）、"海底反射"（BS × A）三部分，是水下声学中最基础的关系式。任何后向散射处理都需要基于此方程进行"逆运算"——从 EL 反推 BS。

### 0.3 传输损失

传输损失 $TL$ 包括两个分量:

$$TL = TL_{\text{spreading}} + TL_{\text{absorption}}$$

**球面扩展损失**（Spreading Loss）：

$$TL_{\text{spreading}}(R) = 20 \cdot \log_{10}(R)$$

其中 $R$ 是斜距（声源到海底的距离，单位为米）。该式假定了理想球面波扩展，声能按球面面积 $4\pi R^2$ 扩散。每增加一倍距离，损失约 6 dB。

**海水吸收损失**（Absorption Loss）：

$$TL_{\text{absorption}}(R, f) = \alpha(f) \cdot R$$

其中 $\alpha(f)$ 是频率相关的海水吸收系数（dB/m），取决于声波频率 $f$、水温、盐度和 pH。对于典型的 300 kHz 多波束声呐（如 EM2040），$\alpha \approx 0.06$ dB/m。

**双向传输损失** $2TL$ 考虑了往返路径：

$$2TL = 40\log_{10}(R) + 2\alpha R$$

### 0.4 照射面积与后向散射强度的归一化

海底照射面积 $A_{\text{insonified}}$ 取决于波束宽度、斜距和海底入射角：

$$A_{\text{insonified}} = R \cdot \Delta\phi \cdot \frac{c\tau}{2\sin\theta}$$

或等价地（IFREMER 算法）：

$$A_{\text{insonified}} = R^2 \cdot \Delta\phi \cdot \frac{1}{\cos\theta}$$

其中：

- $R$ = 斜距（m）
- $\Delta\phi$ = 波束宽度（弧度）
- $c$ = 声速（约 1500 m/s）
- $\tau$ = 脉冲长度（s）
- $\theta$ = 海底入射角

**物理意义**: 后向散射强度 $BS$ 是归一化到单位面积的值。照射面积 $A$ 的计算直接影响 $BS$ 的绝对水平。如果 $A$ 计算偏大，$BS$ 会偏低（因为同样的反射能量平摊到了更大的面积上）。

### 0.5 后向散射强度与海底底质

不同海底类型的后向散射强度范围：

| 海底类型 | BS 近似范围 (dB) | 角度依赖性 | 物理原因 |
|---|---|---|---|
| **软泥/粘土** | -35 至 -25 | 弱（曲线平坦） | 低声阻抗对比度，声波被吸收 |
| **粉砂** | -30 至 -20 | 中等 | 声阻抗中等 |
| **细沙** | -25 至 -15 | 中等-显著 | 沙粒散射 |
| **粗沙/砾石** | -20 至 -10 | 显著 | 粗糙表面引起的 Bragg 散射 |
| **岩石/基岩** | -15 至 -5 | 非常显著 | 高声阻抗对比度，强反射 |
| **贝壳层** | -10 至 0 | 极其显著 | 复杂几何形状，强散射 |

角度依赖性的物理机制：

1. **镜面反射**（Specular Reflection）：$\theta \approx 0°$（天底），声波垂直入射时，反射以相干镜面分量为主，强度最高。粗糙海底的镜面反射分量降低。

2. **漫散射**（Diffuse Scattering）：$\theta > 0°$（斜入射），反射以非相干散射分量为主，强度随角度逐渐降低。粗糙度越大，漫散射越强，角度依赖性越大。

3. **体积散射**（Volume Scattering）：$\theta > 0°$ 且海底有沉积层时，声波穿透海底沉积层后，在体积内产生后向散射。软泥的 BSAR 曲线在较大角度下降缓慢。

---

## 1. 整体处理流程概览

```text
原始 XSF 文件 (.xsf.nc)  — 含原始后向散射 + 厂家校正值
    │
    ├── 阶段 1: 工具 1 (声纳至 DTM)
    │    → 剥离厂家补偿 / 校准值
    │    → 计算无补偿的后向散射 BS_raw
    │    → 网格化到 DTM (.dtm.nc)
    │
    ├── 阶段 2a: 统计角响应 (BSAR) — 工具 2B Step 2
    │    → 从 XSF 计算角度响应曲线 (MAV)
    │    → 统计入射角分区的后向散射均值
    │    → 输出 .bsar.nc 模型文件
    │
    ├── 阶段 2b: 静态角度重规范化 — 工具 2B Step 1
    │    → 输入 BSAR 模型 + XSF
    │    → 应用角度校正量
    │    → 输出规范化后的 XSF (processing_flag = ON)
    │
    ├── 阶段 2c: 滑动角度重规范化 — 工具 2A
    │    → 按 ping 逐窗口滑动计算 BSAR
    │    → 窗口内自建模型 → 窗口内应用
    │    → 适合非均匀底质场景
    │
    └── 可视化与导出:
         ├── DTM 导入 (后向散射 / 水深叠加到地图)
         ├── BSAR Viewer (角度响应曲线分析)
         ├── GeoTIFF 导出 (供 GIS 外部使用)
         └── 导航线 (船舶轨迹可视化)
```

---

## 2. 工具 1: 声纳至 DTM

### 2.1 功能定位

**声纳至 DTM** 是多波束后向散射处理的**第一步**。它从 XSF 原始文件中提取无补偿的后向散射值，按地理坐标网格化，输出数字地形模型（DTM），作为后续 BSAR 计算的基础参考数据。

### 2.2 GUI 入口

- **项目浏览器** → 工具箱 → "工具 1: 导出参考 DTM"
- **右键菜单** → "工具 1: 导出参考 DTM"
- **前端 UI 文件**: `gui/dialogs/sounder_to_dtm_wizard.py`（6 页多页向导）

### 2.3 向导页面详解

**Page 1: 输入/输出**
- **输入 XSF 文件**: 可多选、添加、移除、清空
- **合并/分离选项**: 勾选"分离 DTM"时为每个输入文件单独生成 DTM，否则合并为一个网格
- **输出目录**: 文件输出位置
- **文件前缀**: 输出文件名前缀（默认 `bathy`）
- **覆盖选项**: 允许覆盖已存在文件

**Page 2: 投影与分辨率**
- **投影定义**: 预设投影（Auto Detect / UTM / Mercator）或自定义 PROJ.4 定义串
- **空间分辨率**: 支持度（经纬度网格）和米（投影网格），带自动估算按钮
- **自动分辨率**: 根据数据地理范围估算最佳分辨率（约 500 网格的分辨率）
- **展开至整数**: 将网格边界扩展至整数个分辨率单元，确保网格整齐

**Page 3: 高级网格化**
- **有效测深**: 仅使用标记为有效的测深数据
- **空间抗锯齿**: 在单元格重心上使用双线性插值减少混叠
- **空洞填充**: 对空单元格插值填补，需 GDAL netCDF 插件支持
- **掩模尺寸**: 膨胀掩模直径（空洞填充时用）
- **质量指标**: 计算网格质量指标（TIFF）

**Page 4: 高度/测深过滤**
- **最低高程**: 低于此高程的测深被忽略
- **最高高程**: 高于此高程的测深被忽略
- **最小测深数/单元格**: 低于此数量的单元格标记为空

**Page 5: 元数据**
- Title / Institution / Source / References / Comment 等 DTM 元数据字段

**Page 6: 汇总与执行**
- 展示所有参数摘要，确认后执行

### 2.4 后端算法

| 项目 | 值 |
|---|---|
| **GWS 配置文件** | `src/gws/conf/dtm/convert/sounder_to_dtm.json` |
| **后端 Python 函数** | `pyat.dtm.convert.sounder_to_dtm.SounderToDtm` |
| **输入格式** | SONAR-netCDF4 (.xsf.nc) |
| **输出格式** | DTM NetCDF (.dtm.nc) — 多图层网格文件 |
| **输出图层** | elevation / backscatter / elevation_min / elevation_max / value_count / filtered_sounding / etc. |
| **核心算法** | XSF 声线跟踪 → 声线照射面积计算 → 网格化（地理栅格化）|

### 2.5 算法详解——步骤 1: 声线跟踪与海底定位

多波束声呐的每个波束记录的是**斜距**（slant range）和**波束角**（beam angle），而非直接的海底水深。需要通过**声线跟踪**（Ray Tracing / Ray Bending）将每个波束转换为海底三维坐标 $(x,y,z)$。

声线跟踪的原理是 Snell 折射定律：

$$\frac{\sin\alpha_1}{c_1} = \frac{\sin\alpha_2}{c_2} = \text{常数}$$

其中 $\alpha$ 是声线与垂直方向的夹角，$c$ 是声速。海水中的声速剖面 $c(z)$ 是深度 $z$ 的函数。声线在声速梯度中弯曲（类似于光线在大气中的折射），因此声线跟踪是迭代过程，在每一深度层按声速计算声线方向和传播时间。

### 2.6 算法详解——步骤 2: 后向散射值计算

XSF 文件中的后向散射值可能已经嵌入了厂家角增益补偿。工具 1 在执行时：

1. **如果 `remove_compensation = True`**: 剥离厂家内置的 TVG（时变增益）和角增益补偿，恢复物理原始 BS 值
2. **如果 `remove_calibration = True`**: 剥离 kmall 文件中的 BSCorr 校准值
3. **使用声速剖面**（`use_svp = True`）修正声线弯曲对入射角的影响
4. **使用粒度均值**（`use_snippets = True`）从原始粒度区间重新计算后向散射（而非使用检测值）
5. **重算照射面积**（`use_insonified_area = True`）使用 IFREMER 算法计算正确的照射面积

每个波束的后向散射值按上述选项计算后，得到一个 "无补偿" 的后向散射值 $BS_{\text{raw}}$。

### 2.7 算法详解——步骤 3: 网格化（Gridding / Rasterization）

网格化将散点数据（每个波束是一个散点）转换为规则的二维网格。流程：

1. **确定网格范围**: 从导航边界计算最小/最大经纬度
2. **创建空网格**: 根据分辨率和范围创建 `nlat × nlon` 的空白数组
3. **投影散点**: 将每个波束的 BS 值分配到所在网格单元
4. **单元聚合**: 对每个单元内的所有 BS 值计算统计量——均值（mean）、最小值（min）、最大值（max）、计数（count）
5. **空洞处理**: 对没有数据的空网格单元插值填充（可选）

网格化后的 DTM 含有多个图层（NetCDF variables），每个图层存储不同的统计量。

### 2.8 前端参数 → 后端 JSON 参数映射

| 前端参数 | JSON 参数 | 类型 | 说明 |
|---|---|---|---|
| 投影 (Projection) | `target_spatial_reference` | string | PROJ.4 定义字符串，如 `+proj=longlat +ellps=WGS84` |
| 空间分辨率 (Resolution) | `target_resolution` | float | 单元大小（度/米） |
| 空洞填充 (Gap Filling) | `gap_filling` | bool | 是否插值填补空单元 |
| 掩模尺寸 (Mask Size) | `mask_size` | int | 膨胀掩模直径（单元数） |
| 有效测深 (Valid sounds) | `valid_sounds_only` | bool | 仅用有效标记的测深 |
| 空间抗锯齿 (Spatial AA) | `spatial_antialiasing` | bool | 单元格重心双线性插值 |
| 最低/最高海拔 | `min_elevation` / `max_elevation` | float | 高程过滤范围 |
| 最少测深数 (Min sounds) | `min_sounds` | int | 每单元最小测深数 |
| 输出图层 | `layers` | list | 如 `["elevation", "backscatter"]` |
| 元数据字段 | `title`/`institution`/`comment`/... | string | DTM NetCDF 全局属性 |

---

## 3. 工具 2B Step 2: 统计角响应 BSAR 模型计算

### 3.1 功能定位

**统计角响应（BSAR）** 是多波束后向散射处理的**核心步骤**。它从输入 XSF 文件统计海底后向散射强度随入射角变化的规律，生成 **BSAR 模型文件**（.bsar.nc）。该模型文件是后续角度重规范化的基础。

### 3.2 GUI 入口

- **右键菜单** → "统计角响应（BSAR）"
- **前端 UI 文件**: `gui/dialogs/tool2b_s2_dialog.py`（4 页向导）

### 3.3 向导页面详解

**Page 1: 输入/输出**
- **输入 XSF 文件**: 只读显示已选文件
- **输出 BSAR 路径**: 必填，选择保存位置，默认 `bsar_model.bsar.nc`
- **参考 DTM**: 可选，用于精确入射角计算
- **输入 BSAR 模型**（`i_meanmodel`）: 可选，外部 BSAR 模型参与校准
- **空间掩模**: 可选，KML/SHP 地理掩模

**Page 2: 参数**
- **声纳类型** (Sounder Type): EM2040_ALL / EM122_ALL / AUTO 等
- **积分方法** (Integration Method): MEAN（均值）或 MEDIAN（中位数）
- **线性标度** (Linear Scale): AMPLITUDE（振幅）或 ENERGY（能量）
- **使用粒度均值** (Use snippet mean): 从原始粒度重算 BS

**Page 3: 高级选项** (BL0 / BL1 / BL2 级别)
- **使用声速剖面** (use_svp): 用 XSF 中嵌入的 SVP 修正声线弯曲
- **重算照射区面积** (use_insonified_area): IFREMER 算法计算
- **移除厂家角补偿** (remove_compensation): 去除 Kongsberg 内置角增益
- **移除厂家校准** (remove_calibration): 去除 kmall BSCorr 校准

**Page 4: 汇总**
- 显示全部参数，确认后执行

### 3.4 后端算法详解

| 项目 | 值 |
|---|---|
| **GWS 配置文件** | `src/gws/conf/sonar/bs/avg_backscatter_model.json` |
| **后端 Python 函数** | `pyat.sonarscope.bs_correction.stats_computer.compute_mean_model_process` |
| **核心类** | `MeanBSComputer` — 后向散射均值统计计算机 |
| **输出** | `.bsar.nc` 文件（BSAR 模型） |
| **处理级别** | BL0 → BL1 → BL2 → 统计 |
| **入射角范围** | `IncidenceAngleBins(min=-0.5°, max=89.5°, resolution=1°)` = 90 bins |
| **发射角范围** | `TransmissionAngleBins(min=-80.5°, max=+80.5°, resolution=1°)` = 161 bins |

### 3.5 处理级别详解: BL0, BL1, BL2

后向散射处理的三个预处理级别决定了输入数据的"净化"程度:

| 级别 | 名称 | 操作 | 影响 |
|---|---|---|---|
| **BL0** | 原始/检测值 | 无操作——直接使用 XSF 中的 detection-level BS 值 | 保留了厂家所有内置补偿 |
| **BL1** | 物理修正 | 移除厂家的发射功率校正（BSCorr） | BL1 从原始波形中重算 BS，但保留角增益 |
| **BL2** | 物理/几何修正 | BL1 + 移除角增益补偿 + 重算照射面积 | BL2 是完全的物理原始 BS，无任何厂家校正 |

工具 2B 通常在 BL2 级别进行处理（`remove_compensation=True`, `remove_calibration=True`）。

### 3.6 算法详解——步骤 1: 后向散射与入射角计算

对每个输入文件的每个检测点（detection），计算:

**后向散射值** $BS_{\text{detection}}$（dB）：通过 `BSComputer.compute_bs()` 从 XSF ping 数据和 ping 检测数据计算。该函数:

1. 读取 ping 信号数据集（PingSignal model）——含每个 ping 的发射参数
2. 读取 ping 检测数据集（PingDetectionSignal model）——含每个检测的接收参数
3. 使用 `GenericCorrectionComputer` 计算无补偿 BS
4. 使用 `KongsbergCorrectionComputer` 计算 Kongsberg 厂商校正值
5. 最终 $BS_{\text{detection}} = BS_{\text{generic}}$ （if `remove_compensation=True`）或 $BS_{\text{detection}} = BS_{\text{kongsberg}}$ （if `False`）
6. 可选的: 使用 `Snippets` 重算均值 BS（从原始时间序列积分而非检测值）

**海底入射角** $\theta$（°）：通过声线跟踪（Ray Bending）或 DTM 参考计算。如果提供了 `i_dtm`，使用 `DtmAnglesComputer` 从 DTM 的当地坡度推算精确入射角。

### 3.7 算法详解——步骤 2: 照射面积修正

照射面积 $A_{\text{insonified}}$ 的两种算法对比：

| 算法 | 类 | 假设 | 公式 |
|---|---|---|---|
| **Kongsberg** | `KongsbergCorrectionComputer.compute_insonified_area_db()` | 平坦海底 | $A_K = R \cdot \Delta\phi \cdot \frac{c\tau}{2\sin\theta}$ |
| **IFREMER** | `GenericCorrectionComputer.compute_insonified_area_db()` | 局部倾斜海底，考虑海底坡度 | $A_I = R^2 \cdot \Delta\phi \cdot \frac{1}{\cos\theta_{\text{local}}}$ |

其中：

- $R$ = 斜距（m）
- $\Delta\phi$ = 波束宽度（弧度）
- $\theta_{\text{local}}$ = 海底局部入射角（°），考虑海底坡度
- $c\tau / 2$ = 脉冲足迹沿声线方向的范围（$c$ = 声速，$\tau$ = 脉冲长度）

**物理差异**: IFREMER 算法考虑了海底局部坡度（从 DTM 或声线跟踪计算），因此在大坡度地形上比 Kongsberg 的平坦海底假设更准确。在高坡区，Kongsberg 会低估有效照射面积，导致 BS 偏高。

### 3.8 算法详解——步骤 3: 角度分区统计 (Binning)

将后向散射样本按入射角分区的过程:

1. **定义入射角区间**: 从 `IncidenceAngleBins` 取得 90 个区间，每个宽 1°，中心角 0.0°, 1.0°, ..., 89.0°

2. **分区统计**: 使用 `scipy.stats.binned_statistic`:

   $$\boxed{BS_{k}^{\text{mean}} = \frac{1}{|\{i: \theta_i \in [\theta_k, \theta_{k+1}]\}|} \sum_{i: \theta_i \in [\theta_k, \theta_{k+1}]} BS_i}$$

   其中 $\theta_k$ 是第 $k$ 个区间的左边界角，区间 $k$ 的宽度 $\Delta\theta = 1°$。

   **积分方法**：
   - `MEAN`: 直接计算区内 BS 的算术均值（对数域平均）
   - `MEDIAN`: 区内中位数（稳健，抵抗野值）

3. **线性标度处理**: 均值计算前可转换线性标度:
   - `AMPLITUDE`: BS 保持在对数域（dB），直接取平均。$BS_{\text{linear}} = 10^{BS/10}$ 后平均再转换回 dB
   - `ENERGY`: BS 转换为能量域（dB → 平方根），平均后转回 dB

   公式:
   
   $$\boxed{BS_k^{\text{avg,dB}} = 10 \cdot \log_{10}\left(\frac{1}{N_k} \sum_{i \in \text{bin}_k} 10^{BS_i / 10}\right)}$$

   这确保了线性域的平均后再转回 dB，避免了纯 dB 值直接平均的误差。因为 dB 是对数单位，直接平均会低估 BS。

4. **后处理**: 滤波（spline smoothing）消除噪声，得到平滑的角度响应曲线

### 3.9 算法详解——步骤 4: 曲线存储

统计完成后的曲线数据存储为 xarray Dataset:

**入射角曲线** (`/mode_name/by_incidence_angle/`):
```
Variables:
  mean_bs (angle: 90) — filtered mean BS (dB) — bs_value 取自 bin 的均值
  raw_mean_bs (angle: 90) — raw mean BS (dB) — bs 的原始均值（滤波前）
  value_count (angle: 90) — value count — bs 样本数

Coordinates:
  angle (angle: 90) — incidence angle bin centers (°)
```

**发射角残留曲线** (`/mode_name/by_transmission_angle/`):
```
Variables:
  mean_bs (rx_antenna: 2, tx_beam: 1, angle: 161) — mean BS (dB)
  mean_residual_bs (rx_antenna: 2, tx_beam: 1, angle: 161) — filtered residual
  raw_mean_residual_bs (rx_antenna: 2, tx_beam: 1, angle: 161) — raw residual
  value_count (rx_antenna: 2, tx_beam: 1, angle: 161) — count

Coordinates:
  rx_antenna (rx_antenna: 2) — receiver antenna index
  tx_beam (tx_beam: 1) — transmitter beam index
  angle (angle: 161) — transmission angle bin centers (°) [-80, ..., +80]
```

### 3.10 算法详解——步骤 5: 发射角残留分析

发射角残留是**系统诊断**的关键工具:

$$\boxed{BS_{\text{residual}}(\theta_{\text{tx}}) = BS_{\text{measured}}(\theta_{\text{tx}}) - BS_{\text{incidence\_model}}(\theta_{\text{inc}})}$$

即实测后向散射减去入射角模型期望值后的剩余。如果声呐系统在跨扇区（不同发射角）性能一致，残留曲线应接近**平坦的零线**。如果残留曲线不平坦：

- 正残留 → 该扇区后向散射偏强（系统偏差或校准误差）
- 负残留 → 该扇区后向散射偏弱
- 跨天线（rx_antenna）差异 → 接收天线之间不平衡

### 3.11 前端参数 → 后端 JSON 参数映射

| 前端参数 | JSON 参数 | 类型 | 说明 |
|---|---|---|---|
| 声纳类型 | `sounder_type` | string | EM2040_ALL / EM122_ALL / AUTO 等 |
| 积分方法 | `integration_method` | string | MEAN 或 MEDIAN |
| 线性标度 | `linear_scale` | string | AMPLITUDE 或 ENERGY |
| 使用粒度均值 | `use_snippets` | bool | True → 从粒度重算 BS（非检测值） |
| 使用 SVP | `use_svp` | bool | 用文件中的声速剖面 |
| 重算照射区 | `use_insonified_area` | bool | IFREMER vs Kongsberg 算法 |
| 移除角补偿 | `remove_compensation` | bool | 去除厂家角增益校正 |
| 移除校准 | `remove_calibration` | bool | 去除 BSCorr 校准 |
| 参考 DTM | `i_dtm` | string(path) | DTM 路径（可选） |
| 输入 BSAR 模型 | `i_meanmodel` | string(path) | BSAR 校准参考（可选） |
| 空间掩模 | `mask` | list(paths) | KML/SHP 掩模文件 |

---

## 4. 工具 2B Step 1: 静态角度重规范化

### 4.1 功能定位

**静态角度重规范化** 使用预计算的 BSAR 模型，将测得的后向散射归一化到恒定参考水平，消除角度依赖性。输出的 XSF 文件获得 `backscatterCorrection = ON` 处理标签。

### 4.2 GUI 入口

- **右键菜单** → "静态角度重规范化"
- **前端 UI 文件**: `gui/dialogs/tool2b_s1_dialog.py`（3 页向导）

### 4.3 向导页面详解

**Page 1: 输入/输出**
- **XSF 文件**: 只读显示已选文件
- **BSAR 模型**: 必填，预计算的 .bsar.nc 模型文件
- **参考 DTM**: 可选，用于精确入射角
- **输出目录**: 可选，留空则使用输入文件同目录
- **覆盖选项**: 覆盖已有输出文件

**Page 2: 参数**
- **参考水平** (reference_level): 归一化目标值 (dB)，默认 -20 dB
- **入射角补偿** (apply_compensation): 勾选 = 全补偿，不勾选 = 仅发射角残留
- **使用粒度均值** (use_snippets): 从粒度重算 BS

**Page 3: 汇总**

### 4.4 后端算法详解

| 项目 | 值 |
|---|---|
| **GWS 配置文件** | `src/gws/conf/sonar/bs/bs_angular_renormalization.json` |
| **后端 Python 函数** | `pyat.sonarscope.bs_correction.angular_renormalization.xsf_constant_process` |
| **核心类** | `AngleNormalizer` + `ConstantModel`（常数模型） |
| **输入** | XSF 文件 + BSAR 模型（.bsar.nc） |
| **输出** | 规范化后的 XSF 文件（`_bs_renorm` 后缀，.xsf.nc） |
| **处理标签** | `backscatterCorrection = ON`（写入 XSF 全局属性） |

### 4.5 ConstantModel（常数模型）——归一化的数学核心

`ConstantModel` 实现的是最简单的物理模型——假设所有入射角上后向散射都等于同一个参考水平。模型从 BSAR 数据中提取每个声呐模式的角度响应曲线，构建 Look-Up Table（LUT），内存在内存中的**补偿量查表**。

**LUT 构建**:

$$\boxed{\Delta BS_{\text{inc}}(\theta) = BS_{\text{ref}} - BS_{\text{avg}}(\theta)}$$

$$\boxed{\Delta BS_{\text{tx}}(\theta) = BS_{\text{ref}} - BS_{\text{residual}}(\theta)}$$

其中：

- $BS_{\text{ref}}$ = 参考水平（dB）— 用户选择的目标值
- $BS_{\text{avg}}(\theta)$ = BSAR 模型中入射角 $\theta$ 处的平均后向散射（dB）
- $BS_{\text{residual}}(\theta)$ = BSAR 模型中发射角 $\theta$ 处的残留后向散射（dB）
- $\Delta BS_{\text{inc}}$ = 入射角补偿量（dB） — 需要在 $\theta$ 处添加的值
- $\Delta BS_{\text{tx}}$ = 发射角残留补偿量（dB）

**物理含义**:
- $BS_{\text{avg}}(\theta)$ 是**海底的期望BS**——即该底质在该角度上的"典型"回波强度
- $BS_{\text{ref}} - BS_{\text{avg}}(\theta)$ 是**补偿量**——添加这个 dB 值后，所有角度上的 BS 都相同
- 补偿后的 BS 在物理上代表"归一化到参考水平的后向散射"，消除了底质类型和角度的依赖

### 4.6 AngleNormalizer.apply_on_file —— 逐点补偿流程

这是整个角度重规范化的核心算法（约 160 行代码），在 `pyat.sonarscope.bs_correction.angular_renormalization.py` 中的 `AngleNormalizer.apply_on_file()` 方法中实现。

```
输入: XSF 文件 (已复制到临时文件) + BSAR 模型 + DTM 参考
  ↓
1. 读取 XSF 文件元数据:
  - 声呐模式信息 (mode_computer)
  - Ping 模型 (PingSignal: 发射参数)
  - Ping 检测模型 (PingDetectionSignal: 波束指向角)
  ↓
2. 计算每个检测点的:
  - 后向散射 BS (dB) — BSComputer.compute_bs()
  - 海底入射角 θ_inc (°) — 声线跟踪 / DTM 推算
  - 发射角 θ_tx (°) — 波束指向角 reference_to_platform
  ↓
3. 对每个声呐模式 (mode) × 接收天线 (rx_antenna) × 发射扇区 (tx_beam):
  ↓
  a) 从 ConstantModel LUT 中提取:
     - 入射角补偿量 ΔBS_inc(θ_inc) — 通过 np.interp 线性插值 LUT
     - 发射角残留补偿量 ΔBS_tx(θ_tx) — 通过 np.interp 线性插值 LUT
  ↓
  b) 全补偿模式 (apply_compensation=True):
     BS_corrected = BS_measured + ΔBS_inc(θ_inc) + ΔBS_tx(θ_tx)
     仅发射角模式 (apply_compensation=False):
     BS_corrected = BS_measured + ΔBS_tx(θ_tx)
  ↓
  c) 将补偿后的 BS 写回 XSF 的 detection_backscatter_R 变量
  ↓
4. 设置处理标签:
  - xsf.update_processing_status({backscatterCorrection: ON})
  - xsf.append_history_line("Backscatter angular renormalization (ref: BSAR_file) with PyAT")
  ↓
5. 将临时文件重命名为正式输出文件
```

**补偿示例**:

假设在入射角 $\theta = 45°$ 处:
- 实测 BS = -25 dB
- BSAR 模型在此角度的均值 = -28 dB
- 参考水平 = -20 dB
- 入射角补偿量 = -20 - (-28) = +8 dB
- 发射角残留补偿量（如有）= +1 dB
- 补偿后 BS = -25 + 8 + 1 = -16 dB

补偿后的 -16 dB 是在一个参考水平上的归一化 BS 值，与入射角无关。

### 4.7 输出的 XSF 文件

输出文件具有以下特征：

- **文件名**: `原文件名_bs_renorm.xsf.nc`
- **位置**: 原文件同目录（或用户选的输出目录）
- **处理标签**: `backscatterCorrection = ON` （在 XSF 全局属性中）
- **内容**: 与输入 XSF 相同的数据内容，仅 detection_backscatter_R 变量中的 BS 值已替换为补偿后的值

处理过的 XSF 文件在项目浏览器中会显示为**绿色**（表示 "已处理"），用于后续的工具 3（Backscatter Mosaic）。

### 4.8 前端参数 → 后端 JSON 参数映射

| 前端参数 | JSON 参数 | 类型 | 说明 |
|---|---|---|---|
| BSAR 模型路径 | `mean_model_file` | string(path) | 统计角响应模型文件 |
| 参考 DTM | `i_dtm` | string(path) | DTM 路径（可选） |
| 参考水平 | `reference_level` | float | 归一化目标 (dB, 默认 -20) |
| 入射角补偿 | `apply_compensation` | bool | True = 全补偿，False = 仅发射角 |
| 使用粒度均值 | `use_snippets` | bool | 从粒度重算 BS |
| 覆盖已有文件 | `overwrite` | bool | 允许覆盖 |

---

## 5. 工具 2A: 滑动角度重规范化

### 5.1 功能定位

**滑动角度重规范化** 按 ping 逐窗口滑动计算角度响应模型，适合非均匀底质区域（底质类型在测线上频繁变化）。每个 ping 窗口内自建模型并进行补偿，输出规范化后的 XSF 文件。

### 5.2 GUI 入口

- **右键菜单** → "滑动角度重规范化"
- **前端 UI 文件**: `gui/dialogs/tool2a_dialog.py`

### 5.3 后端算法详解

| 项目 | 值 |
|---|---|
| **GWS 配置文件** | `src/gws/conf/sonar/bs/bs_sliding_angular_renormalization.json` |
| **后端 Python 函数** | `pyat.sonarscope.bs_correction.sliding_angular_renormalization.xsf_sliding_process` |
| **核心类** | `SlidingModel` — 按 ping 窗口逐窗口建模 |
| **处理级别** | BL2（默认） |
| **窗口大小** | 默认 10 ping 窗口（前端可调 `sliding_window`） |

### 5.4 与静态方法的深度对比

| 特性 | 静态角度重规范化 | 滑动角度重规范化 |
|---|---|---|
| **模型来源** | 预先计算的 BSAR (.bsar.nc) | 每个 sliding window 内实时计算 |
| **模型规模** | 整个测区一个模型 | 每个窗口一个模型 |
| **计算复杂度** | O(N) — 只查表 | O(N·W) — 每个窗口 W 都要构建模型 |
| **适用底质** | 均匀底质（大范围相同底质类型） | 非均匀底质（底质类型频繁变化） |
| **输出文件类型** | XSF (.xsf.nc) | XSF (.xsf.nc) |
| **输出文件名** | `原文件_bs_renorm.xsf.nc` | `原文件_bs_sliding.xsf.nc` |
| **参考角区间** | 无 | 用户可选 `ref_angle_min` / `ref_angle_max` |
| **处理标签** | `backscatterCorrection = ON` | 同左 |
| **内存需求** | 低（仅 LUT 内存） | 高（每个窗口需要 N×M 的模型存储） |

### 5.5 SlidingModel 与 ConstantModel 的区别

`ConstantModel` (静态)：
```
BSAR 文件 → ConstantModel(BS_ref) → ΔBS(θ) = BS_ref - BS_model(θ) → LUT 查表
```

`SlidingModel` (滑动)：
```
窗口内 pings → 自建 BSAR 模型 → ΔBS_local(p, θ) → 窗口内应用
下一个窗口 → 重新自建模型 → ΔBS_local(p+W, θ) → 窗口内应用
```

### 5.6 算法流程

```
输入: XSF 文件 + 参考 DTM (可选)
  ↓
1. 计算整个文件的 BS 和入射角 (全部 pings)
  ↓
2. 滑动窗口循环:
   窗口 w: 从 ping p 到 ping p+W
  ↓
   2a) 在窗口内: 自建 BSAR 模型
     - 分区统计 (binning)
     - 发射角残留计算
  ↓
   2b) 在窗口内: 应用 SlidingModel 进行补偿
     - BS_corrected = BS_measured + ΔBS_local(θ)
  ↓
3. 输出规范化后的 XSF
```

### 5.7 前端参数 → 后端 JSON 参数映射

| 前端参数 | JSON 参数 | 说明 |
|---|---|---|
| 滑动窗口 (Sliding Window) | `sliding_window` | 每个窗口的 ping 数量 |
| 参考角最小/最大 | `ref_angle_min` / `ref_angle_max` | 用这个角度区间内的 BS 值作为参考 |

---

## 6. BSAR 角度响应曲线可视化 (BSAR Viewer)

### 6.1 功能定位

**BSAR Viewer** 是内置于 GUI 中的角度响应曲线交互式可视化工具。它读取 .bsar.nc 文件，图形化展示每个声呐模式的角度响应曲线，支持多曲线叠加分析、单个曲线筛选等功能。

### 6.2 GUI 入口

- **中心标签页** → "BSAR 视图"
- **前端 UI 文件**: `gui/views/bsar_viewer.py`
- **渲染引擎**: PyQtGraph（基于 OpenGL 的快速渲染）

### 6.3 数据读取详解

```
.bsar.nc 文件
  ↓
netCDF4 Dataset (open_nc_file with locale encoding)
  ↓
读取全局属性: sounder_type, integration_method, use_snippets, linear_scale…
  ↓
遍历 Mode 组: 每个声呐模式 {mode_name}
  ↓
  子组 /mode_name/by_incidence_angle/
    → 提取变量: angle (°), mean_bs (dB)
    → 变量优先级: mean_bs > raw_mean_bs > mean_residual_bs > raw_mean_residual_bs
  ↓
  子组 /mode_name/by_transmission_angle/
    → 提取变量: angle (°, 3D), mean_bs (dB, 3D)
    → 3D → 1D 转换: np.nanmean over (rx_antenna, tx_beam) axes
  ↓
曲线存储为 {name: {angle, mean_bs, type}}
```

### 6.4 曲线类型说明

| 曲线类型 | 显示名称 | X 轴 | Y 轴 | 物理含义 |
|---|---|---|---|---|
| **入射角响应** | `模式名/入射角` | 入射角 (°) | 后向散射强度 (dB) | **底质类型特征**——不同底质的 BSAR 形态不同 |
| **发射角残留** | `模式名/发射角` | 发射角 (°) | 残留后向散射 (dB) | **系统诊断**——扇区间不平衡度 |

### 6.5 X 轴镜像显示

入射角曲线从原始的 0°-89°（绝对值）镜像为 -89° 到 +89°（带符号）:

```
原始数据:  [0°, 1°, 2°, ..., 89°]
镜像显示: [-89°, ..., -1°, 0°, 1°, ..., 89°]

左舷 (port)  → 负角度
天底 (nadir)  → 0°
右舷 (starboard) → 正角度
```

这种对称表示是多波束标准的角度响应显示方法——虽然在物理上左舷和右舷通常存在细微差异（船舶姿态、底质差异），但 BSAR 模型基于对称假设，将实测值对称投影到两侧。

### 6.6 与多波束标准的符合性

| 标准要求 | 实现 | 验证 |
|---|---|---|
| X 轴 = 入射角 (°)，0° 中心 | `IncidenceAngleBins(-0.5°, 89.5°, 1° 分辨率)` | 90 bins, 中心角 0-89°, 镜像为 -89° 至 +89° ✓ |
| Y 轴 = 后向散射 (dB) | `linear_to_db()` 在 `stats_computer.py` 中调用 | dB 单位，负值 ✓ |
| 左舷/右舷对称 | 镜像投影 | ✓ |
| 多模式叠加 | 不同颜色 + 图例 + 下拉筛选 | ✓ |
| 数据点可视化 | 散点图符号 (symbol='o', size=3) + 连线 | ✓ |

### 6.7 BSAR 曲线的典型诊断用法

1. **数据质量检查**: 查看 `value_count`——低样本数的角度区间统计不可靠
2. **底质分类**: 不同的 BSAR 形态和水平指示不同海底类型
3. **系统一致性**: 发射角残留曲线应平坦——若不平坦，表明发射/接收系统存在跨扇区偏差
4. **天线平衡**: 发射角曲线的 rx_antenna 维度（共 2 个天线）应匹配——若不匹配，表明接收天线之间不平衡
5. **模型验证**: 在运行静态角度重规范化前，先确认 BSAR 模型是可靠的——没有奇异值、缺失数据

---

## 7. DTM 导入与 GeoTIFF 导出

### 7.1 DTM 导入

#### 7.1.1 GUI 入口

- **菜单栏** → 文件 → 导入 DTM (.dtm.nc)
- 支持多选文件

#### 7.1.2 渲染引擎

| 项目 | 值 |
|---|---|
| **前端文件** | `gui/views/dtm_renderer.py` |
| **渲染引擎** | numpy → RGBA → PIL Image.save (PNG) |
| **目标分辨率** | 最长轴 ≤ 4096 px（`_TARGET_PX`） |
| **透明 NaN** | `rgba[np.isnan(data), 3] = 0.0` |
| **高程映射** | matplotlib colormap（灰度/terrain 等） |
| **地理定位** | Leaflet `L.imageOverlay(data_url, bounds)` |

#### 7.1.3 渲染流程

```
DTM 文件 (.dtm.nc)
  ↓
读取图层: 用 netCDF4 读取 lon / lat / backscatter / elevation
  ↓
数据清洗: 掩码数组 → float, NaN 替换无穷值
  ↓
色阶计算: 全分辨率数据的 2%/98% 百分位 → vmin/vmax
  ↓
数据翻转: np.flipud(data) — 将 CF 惯例 (行 0 = 南) 翻转为 PNG 惯例 (行 0 = 北)
  ↓
降采样: 如果长轴 > 4096 px，等步长降采样
  ↓
RGBA 转换: (data - vmin) / (vmax - vmin) → matplotlib colormap → RGBA uint8
  ↓
透明处理: rgba[data 为 NaN] 的 α 通道设为 0
  ↓
PNG 编码: PIL Image.fromarray(rgba, 'RGBA').save(compress_level=1)
  ↓
Base64 编码 → data URL → 发送给 QWebEngineView → Leaflet 渲染
```

### 7.2 GeoTIFF 导出

#### 7.2.1 GUI 入口

- **右键 DTM 文件** → 导出 GeoTIFF
- 打开 `DtmExportDialog` 对话框，配置导出参数

#### 7.2.2 后端算法

| 项目 | 值 |
|---|---|
| **GWS 配置文件** | `src/gws/conf/dtm/export/dtm_to_tiff.json` |
| **后端 Python 函数** | `pyat.dtm.export.dtm_to_tiff.Dtm2Tiff` |
| **核心引擎** | GDAL Warp — `NETCDF:"文件路径":图层名 → GTiff` |
| **图层后缀** | 自动添加 `_backscatter` / `_elevation` 等后缀 |

#### 7.2.3 GDAL Warp 技术原理

```
输入: NETCDF:"文件路径":图层名
  ↓
GDAL 打开 NetCDF 子数据集（通过 GDAL netCDF driver）
  ↓
读取子数据集的地理参考信息 (CRS, geotransform)
  ↓
gdal.Warp:
  - 格式转换 (NetCDF → GeoTIFF)
  - 重投影 (如需要)
  - NODATA 值设定
  - 压缩选项 (COMPRESS=DEFLATE)
  ↓
输出: GeoTIFF (.tif) 文件
```

#### 7.2.4 对话框参数详解

| 控件 | 说明 |
|---|---|
| **输入文件** | 只读显示当前 DTM 路径 |
| **输出目录** | 默认 = 输入文件目录，可选 |
| **文件名** | 不含扩展名，后端为每层加后缀 `_layername` |
| **图层选择** | elevation / backscatter / interpolation_flag / value_count |
| **缺失值** | 自定数值（默认 32767）或 NaN |
| **压缩** | COMPRESS=DEFLATE（无损压缩，GeoTIFF 常用） |
| **覆盖** | 允许覆盖已存在的 GeoTIFF |

---

## 8. 物理 / 数学公式专业详解

### 8.1 声呐方程（完整的后向散射模型）

$$
\boxed{BS = EL - SL + 2TL - 10\log_{10}(A_{\text{insonified}})}
$$

**各参数的物理机制详细说明**:

1. **回声级 EL (Echo Level)**:
   - $EL = 20\log_{10}(P_{\text{received}} / P_{\text{ref}})$
   - $P_{\text{received}}$ = 接收换能器输出的电压经过接收增益后对应的声压
   - $P_{\text{ref}}$ = 参考声压 (1 μPa)
   - 这个值是从 XSF 文件中每个检测点的原始采样数据中提取的

2. **声源级 SL (Source Level)**:
   - $SL = 20\log_{10}(P_{\text{tx}} / P_{\text{ref}})$ at 1 m
   - 发射换能器的等效声压级，在距声源 1 米处测定
   - 典型的多波束声呐 SL 值: 210-230 dB (re 1 μPa @ 1m)
   - 来自声呐制造商的校准数据

3. **双向传输损失 $2TL$**:
   $$2TL = 40\log_{10}(R) + 2\alpha(f) \cdot R$$
   - 第一项 $40\log_{10}(R)$ 是球面扩展损失（往返）
   - 第二项 $2\alpha(f)R$ 是双向海水吸收损失
   - $\alpha(f)$ 取决于声波频率 $f$，通过 Francois-Garrison 公式计算:
   $$\alpha(f) = \frac{A_1 f_1 f^2}{f_1^2 + f^2} + \frac{A_2 P_2 f_2 f^2}{f_2^2 + f^2} + A_3 P_3 f^2$$
   其中 $f$ 是声波频率 (kHz)，系数 $A_i, f_i, P_i$ 取决于水温、盐度、深度和 pH

4. **照射面积 $A_{\text{insonified}}$**（见 8.3 节）

### 8.2 声速剖面和声线弯曲

声速 $c(z)$ 随深度变化（温度梯度、盐度变化、压力增加）导致声线弯曲。声线跟踪是将声线按等时间步长离散化迭代求解的过程:

每个步长 $\Delta t$ 中:
- 根据 Snell 定律: $\frac{\sin\alpha(z)}{c(z)} = \frac{\sin\alpha_0}{c_0}$ = 常数
- 水平位移: $\Delta x = \frac{c(z)}{\cos\alpha(z)} \cdot \Delta t$
- 深度变化: $\Delta z = c(z) \cdot \sin\alpha(z) \cdot \Delta t$

从声线跟踪最后一步得到海底点的三维坐标 $(x, y, z)$ 和入射角 $\theta$。

### 8.3 照射面积计算——IFREMER 算法详者

IFREMER 算法的物理模型和公式推导:

**步骤 1**: 计算海底局部法向量 $\mathbf{n}$。

如果提供了参考 DTM，$\mathbf{n}$ 可以从 DTM 的局部梯度计算:

$$\mathbf{n} = \frac{(-\frac{\partial z}{\partial x}_{\text{DTM}}, -\frac{\partial z}{\partial y}_{\text{DTM}}, 1)}{\sqrt{(\frac{\partial z}{\partial x}_{\text{DTM}})^2 + (\frac{\partial z}{\partial y}_{\text{DTM}})^2 + 1}}$$

如果无 DTM，从声线跟踪推算 $\mathbf{n}$。

**步骤 2**: 计算局部入射角 $\theta_{\text{local}}$:

$$\boxed{\cos\theta_{\text{local}} = \mathbf{n} \cdot \frac{\mathbf{r}}{|\mathbf{r}|}}$$

其中 $\mathbf{r}$ 是从发射点到海底点的向量（声线方向）。当海底是倾斜的时，$\theta_{\text{local}}$ 不同于声波到达角 $\theta_{\text{arrival}}$。

**步骤 3**: 照射面积公式:

$$\boxed{A_{\text{insonified}}^{\text{IFREMER}} = R^2 \cdot \Delta\phi \cdot \frac{1}{\cos\theta_{\text{local}}}}$$

其中 $\Delta\phi$ 是波束宽度（弧度），$R$ 是斜距。

**为什么 IFREMER 比 Kongsberg 更准确？**

Kongsberg 假设的海底是平坦的（$\theta_{\text{local}} = \theta_{\text{arrival}}$），但在现实海洋中，海底有复杂的微观地形。在坡度 $\neq 0$ 的地方，Kongsberg 会**低估**真实照射面积——因为平坦海底假设忽略了坡面上面积增加的效果。在高坡区（例如海山侧翼），这种低估可能导致 BS 偏高 3-5 dB——**这个偏高会扭曲后续的底质分类**。

IFREMER 算法通过引入 DTM 坡度修正，弥补了这 3-5 dB 的偏差，使 BS 值更接近物理真实值。

### 8.4 角度分区统计（Binning）——步进详解

Binning 过程的详细步骤:

```
Given: N samples S = {(BS_i, θ_i)}_{i=1}^{N}
Parameters: θ_min = -0.5°, θ_max = 89.5°, Δ = 1° → K = 90 bins

For k = 0, 1, 2, ..., K-1:
  θ_k^left   = θ_min + k·Δ            # bin 左边界
  θ_k^right  = θ_min + (k+1)·Δ          # bin 右边界
  θ_k^center = θ_min + (k+0.5)·Δ        # bin 中心

  # 收集 bin k 内的样本
  S_k = {i : θ_k^left ≤ θ_i < θ_k^right}
  N_k = |S_k|

  # 统计 bin k 的均值
  BS_k_mean (dB) = (1/N_k) · Σ_{i∈S_k} BS_i

  # 可选: 线性标度转换
  BS_k_mean (energy-averaged) = 10·log₁₀((1/N_k)· Σ_{i∈S_k} 10^{BS_i/10})
```

**为什么需要 linear_scale 选项？**

直接对数域平均（`linear_scale=AMPLITUDE`）和对数域转换后平均（`linear_scale=ENERGY`）的物理差异:

- **AMPLITUDE**: $\overline{BS}_{\text{dB}} = \frac{1}{N}\sum_{i=1}^{N} BS_i$ ——这是简单算术平均，噪声抑制不强
- **ENERGY**: $\overline{BS}_{\text{dB}} = 10\log_{10}\left(\frac{1}{N}\sum_{i=1}^{N} 10^{BS_i/10}\right)$ ——这是能量域平均，**对大值样本更敏感**

在物理上，能量域平均是正确的（后向散射本质上是能量反射）。能量域平均后，高 BS 值（近天底、岩石底）比低 BS 值（远舷、泥底）在平均值中的权重更大，更符合物理真实。

### 8.5 积分方法的选择: MEAN vs MEDIAN

**MEAN（均值）**: 
$$\overline{BS}_{\text{mean}} = \frac{1}{N}\sum_{i=1}^{N} BS_i$$

- 优点: 计算简单，数学性质好
- 缺点: 对野值（outliers）敏感，少量极端值可以显著拉偏均值
- 适用: 数据质量好的场景

**MEDIAN（中位数）**:
$$\overline{BS}_{\text{median}} = BS_{(N/2)} \text{ (排序后的中间值)}$$

- 优点: 稳健，不受野值影响
- 缺点: 统计效率低（相同 N 下标准差是均值的 1.25 倍）
- 适用: 有野值、数据质量波动大的场景

在多波束后向散射处理中，MEAN 更常用（数据量足够大时，中心极限定理保证均值分布接近正态）。MEDIAN 用于对噪声敏感的精细分析。

### 8.6 角度归一化的数学本质

角度归一化的目标可以表达为:

$$\boxed{\frac{\partial BS_{\text{corrected}}}{\partial\theta} = 0 \quad \forall\theta}$$

即补偿后的 BS 在所有入射角上的梯度为零。这是通过在原始 BS 上添加角度相关的补偿量 $\Delta BS(\theta)$ 来实现的:

$$BS_{\text{corrected}} = BS_{\text{measured}} + \Delta BS(\theta)$$

其中 $\Delta BS(\theta) = BS_{\text{ref}} - BS_{\text{model}}(\theta)$，因此:

$$BS_{\text{corrected}} = BS_{\text{measured}} + BS_{\text{ref}} - BS_{\text{model}}(\theta) = BS_{\text{ref}} + (BS_{\text{measured}} - BS_{\text{model}}(\theta))$$

**物理洞察**: $(BS_{\text{measured}} - BS_{\text{model}}(\theta))$ 是"测得的偏离模型多少 dB"。补偿后的 BS = 参考水平 + 偏离量。这保留了**原始 BS 的变异信息**（即底质类型差异），但消除了**角度引起的 BS 变异**。

### 8.7 角度响应补偿量的计算

在 `ConstantModel` 类中的数学推导:

**步骤 1**: 构建 BSAR 模型的 Incidence LUT

```
从 BSAR 文件中提取:
  BS_avg(θ) = 入射角 θ 处的平均 BS (dB) — 90 个点的数组

对每个声呐模式:
  ΔBS_inc(θ) = BS_ref - BS_avg(θ)

  结果: 入射角补偿 LUT — 90 个点的数组
```

**步骤 2**: 构建 BSAR 模型的 Transmission LUT

```
从 BSAR 文件中提取:
  BS_residual(θ_tx) = 发射角 θ_tx 处的平均残留 BS (dB) — 161 个点

对每个声呐模式 × 接收天线 × 发射扇区:
  ΔBS_tx(θ_tx) = BS_ref - BS_residual(θ_tx)

  结果: 发射角残留补偿 LUT — 161 个点（每个 (rx, tx) 组合）
```

**步骤 3**: 实时补偿时查 LUT

```
对给定的检测 (θ_inc, θ_tx):
  1. 从 LUT 内插 BS_inc_correction = np.interp(θ_inc, LUT_angles, LUT_ΔBS_inc)
     使用 np.interp 在 LUT 点之间线性插值
  2. 从 LUT 内插 BS_tx_correction = np.interp(θ_tx, LUT_angles, LUT_ΔBS_tx)
  3. BS_corrected = BS_measured + BS_inc_correction + BS_tx_correction
```

---

## 附录 A: 前端-后端文件映射表

| 前端文件 | 后端配置文件 | 后端 Python 函数 | 核心类 |
|---|---|---|---|
| `gui/dialogs/sounder_to_dtm_wizard.py` | `dtm/convert/sounder_to_dtm.json` | `pyat.dtm.convert.sounder_to_dtm.SounderToDtm` | `SounderToDtm` |
| `gui/dialogs/tool2b_s2_dialog.py` | `sonar/bs/avg_backscatter_model.json` | `stats_computer.compute_mean_model_process` | `MeanBSComputer` → `MeanBSModel` |
| `gui/dialogs/tool2b_s1_dialog.py` | `sonar/bs/bs_angular_renormalization.json` | `angular_renormalization.xsf_constant_process` | `AngleNormalizer` → `ConstantModel` |
| `gui/dialogs/tool2a_dialog.py` | `sonar/bs/bs_sliding_angular_renormalization.json` | `sliding_angular_renormalization.xsf_sliding_process` | `SlidingModel` |
| `gui/views/bsar_viewer.py` | — | `mean_bs_model.MeanBSModel.read_from_netcdf` (数据读取) | 自定义 PyQtGraph 渲染 |
| `gui/views/dtm_renderer.py` | — | 自定义 `_make_rgba()` → `_make_png()` | numpy → PIL RGBA 编码 |
| `gui/dialogs/dtm_export_dialog.py` | `dtm/export/dtm_to_tiff.json` | `pyat.dtm.export.dtm_to_tiff.Dtm2Tiff` | `Dtm2Tiff` (GDAL Warp) |

## 附录 B: SONAR-netCDF4 (.xsf.nc) 关键变量

| 变量 | 级别 | 形状 | 说明 |
|---|---|---|---|
| `platform_latitude` | Ping | (nping,) | 船舶北纬坐标 |
| `platform_longitude` | Ping | (nping,) | 船舶东经坐标 |
| `detection_latitude` | Detection | (nping, nbeam) | 每个波束的海底检测纬度 |
| `detection_longitude` | Detection | (nping, nbeam) | 每个波束的海底检测经度 |
| `detection_backscatter_R` | Detection | (nping, nbeam) | 后向散射值（dB），处理前的值 |
| `beam_pointing_angle_ref_platform` | Detection | (nping, nbeam) | 波束指向角（相对于平台） |
| `processing_status` | Global attr | — | 含 `backscatterCorrection` 标志 |
| `sounder_type` | Global attr | — | 声纳型号 (如 EM2040_ALL) |

## 附录 C: BSAR NetCDF (.bsar.nc) 文件结构

```
根级 (root)
  ├── 全局属性:
  │   ├── title: "Mean backscatter angular response"
  │   ├── bs_angular_response_version: "0.3"
  │   ├── sounder_type: "EM2040_ALL"
  │   ├── use_snippets: bool
  │   ├── use_svp: bool
  │   ├── use_insonified_area: bool
  │   ├── remove_calibration: bool
  │   ├── remove_compensation: bool
  │   ├── integration_method: "MEAN" 或 "MEDIAN"
  │   └── linear_scale: "AMPLITUDE" 或 "ENERGY"
  │
  └── Mode 组: 每个声呐模式 {mode_name}
      ├── mode_serialized: 模式标识的 JSON 序列化字符串
      ├── /by_incidence_angle/
      │   ├── angle: (90,) inc_angle_bin_centers (°) [0, 1, 2, ..., 89]
      │   ├── mean_bs: (90,) BS_mean_filtered (dB)
      │   ├── raw_mean_bs: (90,) BS_mean_raw (dB)
      │   └── value_count: (90,) count_per_bin (int)
      │
      └── /by_transmission_angle/
          ├── rx_antenna: (2,) receiver antenna index
          ├── tx_beam: (n,) transmitter beam index
          ├── angle: (161,) tx_angle bin centers (°) [-80, ..., 0, ..., +80]
          ├── mean_bs: (2, n, 161) mean BS per rx/tx/angle (dB)
          ├── mean_residual_bs: (2, n, 161) filtered mean residual BS (dB)
          ├── raw_mean_residual_bs: (2, n, 161) raw mean residual BS (dB)
          └── value_count: (2, n, 161) count per bin (int, 3D)
```

## 附录 D: 前端 UI 完成清单

### 已实现的核心功能

| 功能 | 前端 | 后端对接 | 状态 |
|---|---|---|---|
| 项目浏览器（文件树） | `project_explorer.py` | — | ✅ |
| 右键多选保留 | `_SelectionPreservingTreeWidget` | — | ✅ |
| 工具箱按钮 | `_setup_ui` tab 2 | — | ✅ |
| 工具 1: 声纳至 DTM 向导 | `sounder_to_dtm_wizard.py` (6 页) | `SounderToDtm` | ✅ |
| 工具 2A: 滑动角度重规范化 | `tool2a_dialog.py` | `xsf_sliding_process` | ✅ |
| 工具 2B Step 1: 静态角度重规范化 | `tool2b_s1_dialog.py` (3 页) | `xsf_constant_process` | ✅ |
| 工具 2B Step 2: 统计角响应 BSAR | `tool2b_s2_dialog.py` (4 页) | `compute_mean_model_process` | ✅ |
| BSAR Viewer 曲线可视化 | `bsar_viewer.py` | `MeanBSModel.read_from_netcdf` | ✅ |
| DTM 导入 + 地图显示 | `dtm_renderer.py` | Leaflet image overlay | ✅ |
| DTM 图层切换 | `showDtmLayer()` JS | Leaflet layer control | ✅ |
| GeoTIFF 导出 | `dtm_export_dialog.py` | `Dtm2Tiff` (GDAL Warp) | ✅ |
| 导航线显示 / 高亮 | `map_view.py` | `_extract_nav_coords` | ✅ |
| 导航线颜色 (红色) + 图层上下顺序 | `_redraw_all()` + `_leaflet_html()` | Leaflet polyline | ✅ |
| 复选框显隐导航线 | `_on_item_changed` | `show_file_track` | ✅ |
| 主题切换 (深色/浅色) | `app.py` theme toggle | — | ✅ |
| 响应式控制台输出 | `console_view.py` | `process_manager` QProcess | ✅ |
| 中文界面 | 所有 UI 文件 | — | ✅ |

---

**文档结束**

此文档基于 PyAT v0.1.40 源代码和 GUI 前端代码的全面分析编写，覆盖了从物理原理到前端实现的完整链路。
