# PyAT (Python Acoustic ToolBox) - 多束回波声呐数据处理桌面系统

PyAT (Python Acoustic ToolBox) 是一个基于 Python 编写的声学数据处理工具箱，集成了法国海洋巡航船队（French Oceanographic Fleet）的核心声学算法。项目提供了一套现代化的 **PySide6 桌面客户端界面**，其**主要功能是多束回波声呐的后向散射强度（Backscatter）处理与底质特征校正**，同时集成了水柱（Water Column）声图可视化与数字地形模型（DTM）网格化生成等能力。

---

## 🛰️ 核心功能：后向散射强度处理与校正 (Backscatter Processing & Correction)

后向散射强度是多波束声呐探测海底地质属性、栖息地分类（Habitat Mapping）最关键的物理指标。本系统通过以下核心模块，实现了全流程的声呐图像去噪、平准、建模与校正：

### 1. 滑动角重归一化 (Sliding Angular Renormalization - Tool 2A)
- **解决痛点**：多波束测线在航行中由于水体多变、吃水变化、传感器运动或底质的突然改变，声呐图像在沿航迹方向常会出现明暗交替的“条带状”伪影。
- **物理算法**：系统提供动态滑动窗口（Sliding Window，如 10 ~ 100 pings 连续统计）机制。在滑动窗口内统计回波在不同波束入射角（Incident Angle）下的响应分布，估算并反演局部平均角度响应曲线，进行实时滑动去伪影校正，重建无条带的高保真声呐马赛克图像。

### 2. 平均回波特征建模 (Mean BS Model Computation - Tool 2B Step 2)
- **解决痛点**：底质的散射强度随入射角（通常在 $0^\circ \sim 80^\circ$ 之间）变化而呈现非线性剧烈波动（即角度响应效应）。为了获取底质真实物理属性，必须精准建模这种响应曲线。
- **物理算法**：遍历测线回波，可结合高程地形 DTM 计算出每个回波点在海底格网处的真实物理入射角。系统提供均值（MEAN）等多种统计聚合方法，支持在振幅（AMPLITUDE）或对数分贝（DB）线性尺度下进行全航迹建模，计算并输出特定的 `.bsar.nc` 后向散射特征模型文件。

### 3. 标准/静态角重归一化 (Static/Standard Angular Renormalization - Tool 2B Step 1)
- **物理算法**：利用预先计算或导入的全局标准 BSAR 角度响应模型（`.bsar.nc` 格式），消除全航迹声呐数据因入射角几何效应带来的图像边缘过暗/中间过亮现象。
- **增益对齐**：支持用户自定义参考分贝等级（Reference Level，如 `-20.0 dB`）进行全局增益对齐，输出已校正的 XSF 数据（生成 `_bs_renorm` 测线），为最终的声学地质马赛克拼图打下基础。

### 4. BSAR 声学底质分类辅助工具集 (BSAR Toolbox)
提供工程化管理多波束散射模型的图形化辅助工具：
- **CSV ➜ BSAR 导入**：支持将文本格式（如科研分析导出的入射角-分贝关系）的散射特征曲线重新导入并打包为 NetCDF 格式的标准模型文件。
- **合并角度响应模型**：支持将多段航次、多个测区分别生成的本地角度响应模型融合成全局统一模型，实现大面积马赛克拼接时的无缝过渡。
- **按模式/深度拆分**：支持根据水深带、频段或声呐工作模式，对大模型进行切分与归档。

---

## 🌟 辅助功能模块

### 1. 水柱数据处理与可视化 (Water Column - WC)
支持对三维水柱声呐回波数据（XSF/NC格式）进行极坐标和三维空间切片网格化处理：
- **极坐标声图 (Polar Echograms)**：利用船体横摇（Roll）姿态数据进行反向旋转校正，呈现完全直立正视的物理扇面，并支持通过海底线滤波（Bottom Filter）彻底切除海底下方噪点与亮带。
- **水平切片 / 沿航迹切片 / 垂直积分**：支持从不同维度提取水柱强度变化，支持在 PyQtGraph 中以 1:1 物理纵横比无畸变缩放展示。

### 2. 数字地形模型生成与交互视图 (DTM Gridder & Viewer)
- **地形网格化**：通过 DtmGridder 将声呐测深点云构建为 2D 高程格网（支持 WGS84 及 Mercator/UTM 投影）。支持空白区插值填充（Gap Filling）。
- **交互地图与实时查询 (Hover Query)**：在 Leaflet 地图上叠加地形/后向散射渲染图层。鼠标移动到图层上时，地图正下方会高频、微秒级显示鼠标指针所在格网的经纬度、投影坐标以及对应的后向散射值（dB）或水深值（m）。

### 3. DTM 多格式数据导出
支持通过图形化向导界面配置参数，将 DTM 格网导出为：
- **GeoTIFF / COG (.tif)**：支持选择多图层导出，包含 LZW/Deflate 压缩、无效值替换及 GDAL 空间参考绑定。
- **XYZ 文本点云 (.xyz / .emo)**：支持自定义列分隔符（分号、逗号、Tab等）、小数点符号（点或逗号）以及列字段的输出顺序（如 XYZ, YXZ, ZYX等）。

---

## 🛠 运行环境配置与安装

本项目的运行环境完全基于 [requirements_pyat_runtime.yml](file:///D:/Software/pyat-main/requirements/requirements_pyat_runtime.yml) 依赖定义。为了保证其他人能正常、无误地运行本系统，建议使用 **Anaconda / Miniconda** 在 **Python 3.10**（推荐版本，保证 C++ 加速库原生兼容）下进行部署。

### 1. 私有依赖库 (Private Packages) 准备
项目在本地 `private_packages/` 目录下携带了所需的全部闭源及内部依赖包（Wheel 文件）：
- `sonarnative-2.0.4-cp310-cp310-win_amd64.whl` (C++ 声呐滤波与物理校正加速库)
- `pygws-0.3.0-py3-none-any.whl` (Globe 服务核心框架)
- `pynvi-0.15-py3-none-any.whl` (导航解算驱动)
- `pytechsas-0.15.0-py3-none-any.whl` (多源传感器融合包)
- `sonar_netcdf-0.2.12-py3-none-any.whl` (声呐 netCDF 读写封装)
- `heightmap_interpolation-1.0.6-py3-none-any.whl` (高程格网插值模块)
- `pytide-0.0.7-py3-none-any.whl` (潮汐解算模块)
- `rsocket-0.4.15-py3-none-any.whl` (服务通信总线)

> [!WARNING]
> **关于 `sonarnative` 与 Python 版本兼容性说明：**
> 由于 `sonarnative` 加速库是为 **CPython 3.10** 编译的，在 Python 3.10 环境下其旁瓣滤波等底层能力可全速开启。若您在更高版本（如 Python 3.13）中运行，该链接库将因二进制不兼容而无法导入。系统已做好了自动 Fallback 保护：会自动回退到等效的纯 Python 物理实现（如 `python_dtm2ascii_export.py`），核心 DTM 导出/计算依然 100% 正常。

---

### 2. 依赖环境安装方法

请选择以下两种方式之一进行环境安装：

#### 方法 A：使用 .yml 配置文件一键安装（适用于可连接 Ifremer 内部 GitLab 仓库的网络环境）
在项目根目录下执行以下命令，Conda 会自动拉取开源依赖并解析 `pip` 中的 GitLab 仓库链接安装私有库：
```bash
conda env create -f requirements/requirements_pyat_runtime.yml -n pyat_runtime
conda activate pyat_runtime
```

#### 方法 B：离线/隔离环境分步安装（普通互联网环境，推荐）
由于 Ifremer 的 GitLab 私有仓库一般无法直接连接，请使用本方法配合本地的 `private_packages/` 离线安装：

1. **创建基础 Conda 环境并指定安装依赖（配置 `conda-forge` 渠道）**：
   ```bash
   conda create -n pyat_runtime python=3.10 -y
   conda activate pyat_runtime
   
   # 安装 GDAL 科学计算底座及 netCDF 核心包（包含关键的 libgdal-netcdf 驱动插件）
   conda install -c conda-forge gdal libgdal-netcdf netcdf4 hdf5 libssh2 -y
   
   # 安装数据科学与空间地理包
   conda install -c conda-forge dask distributed cftime xarray rioxarray pandas geopandas pyproj fiona pyogrio scikit-image scikit-learn scipy numba seaborn requests -y
   ```
2. **安装辅助依赖与 PySide6 界面框架**：
   ```bash
   pip install PySide6 opencv-python httpx==0.25.2 pydantic==2.5.* result>=0.15.0 dataclasses-json geopy haversine scikit-spatial progress mhkit
   ```
3. **离线部署本地 `private_packages/` 的所有私有 Wheels**：
   ```bash
   pip install --no-index --find-links=private_packages private_packages/*.whl
   ```

---

### 3. 以编辑模式（Editable Mode）挂载本系统
在项目根目录（`D:\Software\pyat-main`）下执行以下命令，使得 `pyat` 后端可全局导入：
```bash
pip install -e . --no-cache-dir
```

---

## 🚀 启动与使用指南

### 1. 启动桌面客户端 (GUI)
激活环境后，直接在项目根目录下运行：
```bash
python run_gui.py
```
这会拉起 PyQt 图形化桌面界面，您可以通过左侧的项目管理器加载 XSF 测线，并执行后向散射滑动去条带校正、水柱多维展示或 DTM 生成。

### 2. 命令行批量任务执行 (CLI)
如果您想绕过图形界面，可以通过传递一个 JSON 任务配置参数文件来调用 `pyat` 模块：
```bash
python -m pyat path/to/task_config.json
```
JSON 参数定义格式详见各处理模块说明文档。
