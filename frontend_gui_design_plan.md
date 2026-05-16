# GLOBE/pyat 多波束声呐后向散射数据处理与可视化前端设计规划

## 1. 需求与深度适配分析

基于本项目 `pyat-main` 的代码架构与 JSON 参数驱动模式，结合文献 *"Acoustic backscatter processing in GLOBE"* 及 `D:\globe-main` 的源码结构，本前端 GUI 规划不仅需要实现可视化，更核心的是要 **作为 `pyat` 核心引擎的图形化编排器与交互式监视器**。

### 1.1 前后端协同核心逻辑
- **JSON 驱动架构**：`pyat-main` 通过 `src/pyat/__main__.py` 解析参数文件（如 `bs_angular_renormalization.json` 等）执行任务。前端的核心工作是将用户的 UI 交互转化为合法的 JSON 配置，并派发子进程 (`subprocess`) 执行，同时解析控制台输出提取进度。
- **项目树结构 (Project Explorer)**：前端需提供一个数据项目树，加载已有 XSF 文件列表，双击可展开查看元数据（声纳型号、波束数、导航范围、处理状态）。统一管理 `XSF Data` → `Reference DTM` → `BSAR Models (.nc)` → `Corrected XSF (绿色标签)` → `Products (DTM/TIFF)` 的文件层级流转。
- **交互式编辑器提取**：针对算法中的 `bsar_editor` 和 `gsab_editor`，前端需原生实现二维折线图的交互调参模块。

### 1.2 XSF 处理状态追踪机制

pyat 后端在每次 BS 处理完成后，通过 XSF 文件（SONAR-netCDF4 格式）的属性字段记录处理状态：

| 状态属性 | 含义 | 取值 |
|---|---|---|
| `ATT_PROCESSING_STATUS_BACKSCATTER_CORRECTION` | 是否已完成 BL4 角归一化 | `FLAG_ON` / `FLAG_OFF` |

前端应通过 `xsf_driver.py` 或直接读取 NetCDF 全局属性来查询每个文件的状态，并在 Project Explorer 中反映：

- **未处理**（`FLAG_OFF` 或属性不存在）→ 文件节点显示灰色图标
- **已处理**（`FLAG_ON`）→ 文件节点显示绿色图标
- **处理中**（任务运行中）→ 显示加载动画

这避免了后端需要维护一个全局状态机（pyat 本身是无状态的批处理引擎），同时让用户一眼看清每个文件处于哪个阶段。

---

## 2. 前端界面布局设计

采用 **经典的科学计算 IDE 布局（类似 Eclipse RCP / QGIS）**，共分为四大区域：

1. **左侧：Project Explorer (项目资源管理器)**
   - **树状数据结构**：按文件类型分组存放导入的 `.all/.s7k` 原始文件、生成的 `.xsf` 文件、参考的 DTM 模型文件、BSAR 统计模型文件 (`.nc`) 以及处理生成的镶嵌产品 (`.nc` / `.tiff`)。
   - **文件状态指示**：每个 XSF 文件节点根据 §1.2 的处理状态标记显示不同图标颜色（未处理/已处理/处理中），直观反映当前进度。
   - **Toolbox 工具面板**：作为 Project Explorer 的第二个标签页，按处理阶段分组列出可用工具，用户双击工具打开参数配置对话框——参考 GLOBE 的 `PyToolbox` 菜单模式而非强制向导。
2. **左下角：Properties Panel (动态参数配置表单)**
   - 根据选中的处理节点或文件，动态解析并映射 `src/gws/conf/sonar/bs/` 目录下的 JSON 模板结构，自动渲染出输入表单（如：声速、补偿选项、平滑滤波窗口等）。
3. **中央/右侧主视图区 (Main Canvas - 多标签页)**
   - **Map View (高分辨率地理视窗)**：底图、导航线、后向散射镶嵌图的 2D/2.5D 高性能渲染区。
   - **Editor View (模型编辑视图)**：展示 Angular Response 曲线、GSAB 拟合参数曲线等，复刻底层 Bokeh/hvplot 的交互能力。
4. **底部：Console & Job Manager (控制台与任务管理)**
   - 多任务进度条、后端 Python 执行的 stdout/stderr 实时打印，支持日志追踪。

---

## 3. UI 工具与交互流程设计

本设计对应 GLOBE 的 `PyToolbox` 菜单中的 BS 处理工具，每个**工具**独立可调用，用户按数据依赖关系依次执行。BL0-BL2 校正参数已嵌入各工具内部，不暴露为独立步骤。

### 前置准备：XSF 文件加载与元数据读取

> 用户已完成原始声纳数据到 XSF 格式的转换。所有工具操作面向已有 XSF 文件。

- **XSF 文件读取**：`xsf_driver.py` + `xsf_reader.py`。Project Explorer 中列出 XSF 文件，双击展开可查看元数据（声纳型号、波束数、导航范围、处理状态标记）。
- **XSF 版本检查**：如版本 < 0.5，工具对话框自动提示"请先执行 XSF Upgrade"（`xsf_upgrader_from_mbg.py`）。

### 完整用户工作流

```
Project Explorer: 加载 XSF 文件列表 (查看元数据/状态)
    │
    ├─ [Tool 1] Export Reference DTM & Uncorrected BS Preview
    │   │ 批量选中 XSF → 右键 "Export Reference DTM"
    │   │ 内部流程: BL1(Lambert+镜面反射校正) → BL2(Ifremer入射面积重估计)
    │   │           → Grid Bathymetry DTM → Grid Uncorrected Backscatter
    │   │ 参数: 声纳类型 / 地图投影 / DTM分辨率 / 缝隙填充 / 高程过滤
    │   │       BL1/BL2 校正参数 (use_snippets, use_svp, use_insonified_area,
    │   │       remove_compensation, remove_calibration)
    │   │ 产出: _bathy.nc (参考DTM, 含ELEVATION+BACKSCATTER图层)
    │   │       未校正后向散射镶嵌图 (Map View 预览, 未做角度响应补偿)
    │   │ 注意: 原始 XSF 文件保持不变, Tool 1 在内部临时应用 BL1/BL2 后即释放
    │   │
    │   ├─ [Tool 2A] Sliding Angular Renormalization (推荐)
    │   │   单独工具窗口, 独立引导页
    │   │   输入: 原始 XSF + _bathy.nc
    │   │   内部自动: BL0(BS读取) → BL1 → BL2 → 滑动窗口统计 → BL4归一化
    │   │   参数: 声纳类型 / 滑动窗口(分钟) / 参考角度范围
    │   │         use_snippets, use_svp, use_insonified_area, remove_calibration
    │   │   产出: _bs_sliding.xsf.nc (绿色图标, backscatterCorrection=FLAG_ON)
    │   │
    │   └─ [Tool 2B] Static Angular Renormalization
    │       单独工具窗口, 独立引导页, 分两步:
    │       ① Statistical BSAR: 原始XSF + _bathy.nc → .bsar.nc
    │          参数: use_snippets, use_svp, use_insonified_area,
    │                remove_compensation, remove_calibration,
    │                integration_method(MEAN/MEDIAN),
    │                linear_scale(AMPLITUDE/ENERGY), 可选掩膜KML
    │       ② Apply BSAR: 原始XSF + .bsar.nc + _bathy.nc → _bs_renorm.xsf.nc
    │          参数: reference_level(-20dB) + Evaluate按钮, apply_compensation
    │
    └─ [Tool 3] Grid Corrected Backscatter Mosaic & Export
        输入: 带处理标签的 XSF (绿色图标, 批量选中)
        参数: 地图投影 / DTM分辨率 / 缝隙填充 / 高程过滤
              (与 Tool 1 完全相同, 保证两次镶嵌图可比)
        内部: Grid Backscatter → 写入同一 DTM.nc 的 BACKSCATTER 图层
        产出: DTM.nc (含 ELEVATION + BACKSCATTER 双图层)
              DTM 分辨率 = 后向散射图分辨率 (同一格网)
        导出: GeoTIFF / COG / MBTiles 一键生成
        可视化: Color Ramp 调色板 + dB 截断滑块 + 数据探针
```

---

### Tool 1: Export Reference DTM & Uncorrected BS Preview

> **对应 GLOBE**：Convert → Grid DTM (Bathymetry)。第一步：从 XSF 水深数据生成参考 DTM，**同时内部应用 BL1+BL2 校正**后输出未做角度响应补偿的后向散射预览图。

- **后端定位**：`sonarscope/bs_correction/bs_computer.py`（内部 BL1+BL2，临时处理不写回原文件）→ `dtm/dtm_gridder.py`（水深层网格化 + 后向散射层网格化）、`dtm/dtm_driver.py`。
- **输入**：用户在 Project Explorer 中**批量选中 XSF 文件**，右键 → "Export Reference DTM"。
- **参数表单**：
  - **声纳类型**：[AUTO / EM2040_ALL / EM710_ALL / ...]（`sounder_type`，默认 AUTO）
  - **地图投影**：[自动检测 / UTM / 墨卡托 / 自定义 EPSG]（GDAL 处理）
  - **格网分辨率** (Cell Size)：0.5m / 1m / 2m / 5m
  - **缝隙填充**：[无 / 双线性插值 / IDW 反距离权重]（`dtm/transform/interpolation/gap_filling.py`）
  - **高程范围过滤**：[min, max]（可选）
  - **BL1/BL2 校正参数（展开面板，后端 `configuration.py` 驱动）**：
    - `use_snippets`：振幅域均值重算 / 直接读取检测值
    - `use_svp`：启用 SVP 折射入射角修正
    - `use_insonified_area`：Ifremer 入射面积重估计（替换 Kongsberg 厂商估计）
    - `remove_compensation`：移除 Kongsberg Lambert + 镜面反射补偿
    - `remove_calibration`：移除 BSCorr 校准
- **内部处理流程**：
  1. 读取 XSF → 应用 BL1 (Kongsberg 补偿移除) → 应用 BL2 (Ifremer 入射面积重估计)
  2. 从校正后数据 Grid 水深 DTM → 写入 `_bathy.nc` 的 ELEVATION 图层
  3. 从校正后数据 Grid 后向散射 → 写入 `_bathy.nc` 的 BACKSCATTER 图层
  4. 原始 XSF 文件**保持不变**（Tool 1 仅做临时校正，不写回）
- **产出文件**：
  - `_bathy.nc` — NetCDF DTM（含 ELEVATION + BACKSCATTER 双图层，直接作后续工具的 `i_dtm`）
  - 可选导出：GeoTIFF / COG / ASCII / MBTiles
- **未校正后向散射预览**：Map View 中以半透明叠加层显示 Tool 1 产出的后向散射镶嵌图，帮助用户在角度响应补偿前直观评估数据质量。**注意：此图未做 BL4 归一化，不是最终产品。**
- **Map View 联动**：DTM 高程图层 + 未校正后向散射图同时叠加，用户可目视确认覆盖范围与测线匹配。

---

### Tool 2A: Sliding Angular Renormalization（推荐）

> **对应 GLOBE**：PyToolbox → "Sliding Angular Renormalization"。**独立工具窗口，独立引导页**——与 Tool 2B 分开。

- **后端定位**：`bs_sliding_angular_renormalization.json` → `sliding_angular_renormalization.xsf_sliding_process`（885 行）。
- **工作原理**：以每个 ping 为中心取滑动时间窗口，窗口内自行统计角响应 → BL4 归一化。**内部自动完成 BL0→BL2 全链路**，无需预计算 BSAR。
- **输入**：原始 XSF 文件 + Tool 1 产出的 `_bathy.nc`。
- **参数表单**：
  - **声纳类型**：[AUTO / EM2040_ALL / ...]（默认 AUTO）
  - **滑动窗口**：默认 10 分钟（`sliding_window`）
  - **参考角度范围**：`ref_angle_min`(30°) / `ref_angle_max`(60°)
  - **BL0-BL2 校正参数（展开面板）**：`use_snippets` / `use_svp` / `use_insonified_area` / `remove_calibration`
  - `i_dtm`：自动填入 Tool 1 的 `_bathy.nc`
  - **可选**：`o_bsar` — 输出滑动窗口 BSAR 模型（供检视/归档）
- **产出**：`_bs_sliding.xsf.nc`（绿色图标），自动出现在 Project Explorer 列表中，后台项目目录同步保存

---

### Tool 2B: Static Angular Renormalization

> **对应 GLOBE**：PyToolbox → "Statistical Angular Response (BSAR)" + "Backscatter Angular Renormalization"。**独立工具窗口，独立引导页**，分两步引导。

##### Step 2a: Statistical BSAR — 统计模型生成

- **后端定位**：`avg_backscatter_model.json` → `stats_computer.compute_mean_model_process`（595 行）。
- **工作原理**：遍历所有 ping 的 BS 值，按入射角 1° 分 bin 统计均值/中位数，内部自动完成 BL0→BL2 校正 → 生成 BSAR 模型。
- **输入**：原始 XSF 文件 + Tool 1 产出的 `_bathy.nc`。
- **参数表单**：
  - **声纳类型**：[AUTO / EM2040_ALL / ...]（默认 AUTO）
  - **BL0-BL2 校正参数（展开面板）**：`use_snippets` / `use_svp` / `use_insonified_area` / `remove_compensation` / `remove_calibration`
  - **统计参数**：`integration_method`(MEAN/MEDIAN) / `linear_scale`(AMPLITUDE/ENERGY)
  - **空间掩膜**（可选）：Map View 多边形 → KML 导出 → `mask` 参数
  - `i_dtm`：自动填入 Tool 1 的 `_bathy.nc`
- **产出**：`.bsar.nc`（NetCDF4），可用 BSAR 查看器检视
- **BSAR 查看器**（PyQtGraph）：入射角响应曲线 + 透射角残余曲线，样条滤波滑块，GSAB 拟合

##### Step 2b: Apply BSAR — 应用模型归一化

- **后端定位**：`bs_angular_renormalization.json` → `angular_renormalization.xsf_constant_process`。
- **输入**：原始 XSF 文件 + Step 2a 的 `.bsar.nc` + Tool 1 的 `_bathy.nc`。
- **参数表单**：
  - `mean_model_file`：自动填入 Project Explorer 中选中的 `.bsar.nc`
  - `reference_level`(dB)：默认 -20，旁设 "Evaluate" 按钮调用 `evaluate_mean_bs_level.json` 自动估算
  - `apply_compensation`：是否应用入射角补偿
  - `use_snippets` / `i_dtm`（自动填入）
- **产出**：`_bs_renorm.xsf.nc`（绿色图标），自动出现在 Project Explorer 列表中，后台项目目录同步保存

---

### Tool 3: Grid Corrected Backscatter Mosaic & Export

> **对应 GLOBE**：Convert → Grid DTM (Backscatter) → Export to GeoTIFF。与 Tool 1 **参数完全相同**（投影/分辨率/缝隙填充），保证两次镶嵌图在相同格网下严格可比。

- **后端定位**：`dtm/dtm_gridder.py`（后向散射层网格化）+ `dtm/export/dtm_to_tiff.py`、`dtm_to_cog.py`、`dtm_to_mbtiles.py`。
- **输入**：Tool 2A 或 Tool 2B 产出的带处理标签 XSF 文件（Project Explorer 中绿色图标）。支持**批量选中**。
- **参数表单**（与 Tool 1 **完全相同**——用户可复用 Tool 1 的配置）：
  - 地图投影 / 格网分辨率 / 缝隙填充 / 高程范围过滤
- **内部处理**：
  - 读取已补偿 XSF 的后向散射变量 → 按指定分辨率网格化
  - 将后向散射镶嵌结果写入同一 DTM.nc 的 BACKSCATTER 图层
  - **DTM 分辨率 = 后向散射图分辨率**（同一格网，`dtm_gridder.py` 内部统一控制）
- **产出**：
  - DTM.nc（含 ELEVATION + BACKSCATTER 双图层）
  - **一键导出**：GeoTIFF / COG / MBTiles
- **Map View 可视化**：
  - 瓦片金字塔加载镶嵌图 + 底图叠加
  - 浮动工具条：Color Ramp 调色板、dB 范围截断滑块（min/max）、像素数据探针（鼠标悬停回显 dB 值）

> **后端算法保护原则**：所有质量提升选项（缝隙填充、投影转换、分辨率设定）均通过调用 pyat 现有后端函数实现（`gap_filling.py`、`dtm_gridder.py` 的 GDAL 投影参数、`signal.py` 的 dB 转换公式等）。**不修改本项目任何后端物理算法公式。**

---

## 4. 技术栈深度选型建议

鉴于底层存在大量的 C 扩展 (Numba)、GDAL 和数据科学库（Xarray/NetCDF4），为了避免跨语言的内存数据拷贝和打包困难，**强烈推荐采用 Python 原生的现代桌面端方案**：

- **基础框架**：`PySide6` (Qt for Python)，支持原生的跨平台桌面 UI，能直接复刻 Eclipse RCP 的多停靠面板体系 (QDockWidget)。
- **科学图表与编辑器**：`PyQtGraph` (用于极致性能的 1D 角响应曲线编辑器、散点剖面编辑器)，由于底层基于 OpenGL，性能远超基于 Web 的 Echarts 或 Bokeh。
- **地理与高分辨率影像视图**：
  - **推荐方案**：嵌入 `QWebEngineView` 运行 **MapboxGL JS** 或 **Leaflet + GeoTIFF overlay** 插件。BS 镶嵌图本质是 2D 栅格数据，2D Web 地图库完全满足需求且生态成熟。GeoTIFF 通过 GDAL 预切金字塔瓦片或直接以 COG (Cloud Optimized GeoTIFF) 格式加载
  - **备选方案**：`QWebEngineView` + `Cesium.js`（如需 3D 地形叠加，但 Cesium 引入的复杂度可能超出当前需求）
  - **不推荐**：`QGraphicsView` 自建瓦片渲染引擎（开发成本极高，已有成熟 Web 方案可直接复用）

---

## 5. 开发实施路线图 (Roadmap)

### 第一阶段 (Phase 1)：外壳、数据准备与参考 DTM

- 搭建 PySide6 主界面外壳（Project Explorer 树 + Toolbox 面板 + QDockWidget 布局）
- 实现 XSF 文件读取与元数据展示（导航信息、处理状态标记读取、XSF 版本检查）
- 实现 **Tool 1: Grid Bathymetry DTM** — 从 XSF 水深数据生成 `_bathy.nc` + 未校正后向散射预览图，含地图投影/分辨率/缝隙填充参数
- 编写 **JSON Task Builder** 模块：UI 表单 → JSON 配置 → `subprocess` 调用 `python -m pyat`
- 封装子进程管理（启动/取消/超时），实现 Console 面板的 stdout/stderr 实时回显（JSON-Lines 协议）
- 实现任务生命周期管理：错误恢复、重试机制、历史配方保存

### 第二阶段 (Phase 2)：角度响应补偿与空间可视化

- 集成 Map View（`QWebEngineView` + 地图库）
- 实现 XSF 航迹线渲染 + DTM 图层叠加显示
- 实现 **Tool 2A: Sliding Renormalization**（独立窗口，推荐默认工具 — 内部完成 BL0→BL4）
- 实现 **Tool 2B Step 2a: Statistical BSAR**（独立窗口 — 含空间掩膜 KML 绘制 + 全部 BL0-BL2 参数）
- 开发基于 PyQtGraph 的 BSAR 曲线查看器（Phase 3A：查看+滑块）

### 第三阶段 (Phase 3)：静态归一化与状态联动

- 实现 **Tool 2B Step 2b: Static Angular Renormalization**（独立窗口 — Evaluate 按钮 + 动态表单）
- XSF 处理状态标记回读与 Project Explorer 图标联动（绿色=已处理）
- BSAR 辅助工具（CSV 导入/合并/拆分/模式摘要）

### 第四阶段 (Phase 4)：镶嵌输出与发布级可视化

- 实现 **Tool 3: Grid Backscatter Mosaic** — 从已处理 XSF 生成后向散射 DTM
- 高分辨率 GeoTIFF 镶嵌图的瓦片化加载 + 底图叠加
- 色彩映射表 (Color Map) 动态调节 + dB 范围截断滑块 + 像素级数据探针
- 一键导出 GeoTIFF / COG / MBTiles

---

## 6. 遗漏补充说明与风险对策

### 6.1 错误恢复与重试机制

pyat 通过 `subprocess` 调用，每个处理任务是独立进程。前端需建立完整的任务生命周期管理：

```
任务状态机：
  QUEUED → RUNNING → COMPLETED
                   → FAILED → (用户触发) → RETRY → QUEUED

任务上下文保存：
  - 每个任务提交时，将生成的 JSON 配置文件保存到项目工作目录
    (.pyat_gui/jobs/<timestamp>_<step_name>.json)
  - 任务失败后，用户可以从 Console 面板右键选择 "Retry with same config"
    或 "Edit config and retry" 重新打开参数对话框
  - 任务成功完成的配置文件保留作为"历史配方"，可从 Toolbox 面板的
    "Recent Jobs" 下拉列表快速复用

错误日志：
  - stderr 输出全部捕获并写入 .pyat_gui/logs/<timestamp>_<step_name>.log
  - Console 面板支持按任务过滤日志（点击进度条跳转到对应日志段）
  - Python traceback 自动高亮显示
```

### 6.2 Tool 2a (Statistical BSAR) JSON 配置模板（已存在）

`avg_backscatter_model.json`（155 行）已完整定义 BSAR 统计模型生成的参数，包括 BL0-BL2 校正参数（`use_snippets`、`use_svp`、`use_insonified_area`、`remove_compensation`、`remove_calibration`）、统计参数（`integration_method`、`linear_scale`）、掩膜（`mask`）和 DTM 输入（`i_dtm`）。**无需额外创建**。

### 6.3 子进程进度报告协议

pyat 内部使用 `pygws.service.progress_monitor.DefaultMonitor` 进行进度追踪（`monitor.begin_task(..., 100)`）。当通过 `subprocess` 在独立进程中运行时，需要约定一种 stdout 格式来传递进度信息到前端：

```
建议协议：JSON-Lines (每行一个 JSON 对象)

{"type": "progress", "percent": 45, "message": "Processing ping 4500/10000"}
{"type": "log", "level": "INFO", "message": "Detected Kongsberg EM2040 format"}
{"type": "log", "level": "ERROR", "message": "Failed to open DTM file"}
{"type": "complete", "output_file": "/path/to/output.xsf.nc"}
{"type": "error", "code": 1, "message": "ValueError: ..."}

实现方式：
  - 在 pyat 的 application_utils.py 中注入一个 StdoutProgressMonitor 子类，
    重写 begin_task/work/progress/done 方法，每个状态变化输出一行 JSON
  - 前端读取 subprocess.stdout 逐行解析 JSON-Lines，更新进度条和日志面板
  - stderr 全部作为 ERROR 级别的 log 事件
```

### 6.4 Tool 1 与 Tool 3 的 DTM 角色区分

**背景**：`dtm_gridder.py` 同时服务于两个不同的处理目标，容易混淆：

| 工具 | 网格化对象 | XSF 变量层 | 产出用途 |
|---|---|---|---|
| **Tool 1** | 水深 (Elevation) | `/Sonar/Beam_group1/Bathymetry/` 中的深度变量 | 参考 DTM，输入到 Tool 2 的 `i_dtm` |
| **Tool 3** | 后向散射 (Backscatter) | `DETECTION_BACKSCATTER_R` 变量 | 最终镶嵌产品，导出 GeoTIFF/COG/MBTiles |

**实施建议**：
- Toolbox 中分别列出 "Grid Bathymetry DTM"（Tool 1）和 "Grid Backscatter Mosaic"（Tool 3），使用不同图标避免混淆
- Tool 1 在 Phase 1 中实现（与 XSF 读取同期），确保 Phase 2 的角度响应补偿工具有 DTM 可用
- Tool 3 在 Phase 4 中实现，此时 BL4 已完成，后向散射数据已就绪

### 6.5 BSAR 交互编辑分期策略

**问题**：Tool 2 的 BSAR 曲线"交互编辑"（拖拽控制点、框选排除异常点）工作量大，且 `mean_bs_model.py` 仅提供 NetCDF 读写接口，无编辑 API。

**分期策略**：

```
Phase 3A（Roadmap 第三阶段范围内）：
  ┌────────────────────────────────────────────────┐
  │ 实现 BSAR 曲线查看器（PyQtGraph PlotWidget）    │
  │  - 加载 .bsar.nc，绘制双曲线（入射角+透射角） │
  │  - 缩放、平移、鼠标悬停数据探针                 │
  │  - 样条滤波参数滑块（调用 apply_spline_filtering│
  │    后重新绘制，保存时覆盖 .bsar.nc）            │
  │  - GSAB 模型 6 参数显示                         │
  └────────────────────────────────────────────────┘

Phase 3B（独立迭代，不阻塞 Phase 4）：
  ┌────────────────────────────────────────────────┐
  │ 实现曲线交互编辑功能                            │
  │  - 控制点拖拽（前端存储编辑后的曲线数组）       │
  │  - 异常点框选（前端维护选区掩膜）               │
  │  - 撤销/重做栈                                  │
  │  - 编辑结果通过 mean_bs_model.save_to_netcdf()  │
  │    写回 .bsar.nc（需要从 PyQtGraph 曲线数据     │
  │    重建 BackscatterCurve 对象）                 │
  └────────────────────────────────────────────────┘
```
