# 重构决策文档：VectorNet 双版本仓库体检 & 重构计划

> 生成日期：2026-05-11
> 仓库根路径：`zxzx/`
> 体检范围：`Vector（驾驶轨迹预测）/yet-another-vectornet/`（原始版）+ `Vector/yet-another-vectornet/`（增强版）

---

## 1. 当前仓库目录树（3 层，含角色标注）

```
zxzx/
│
├── Vector（驾驶轨迹预测）/                          ← ★ 项目根目录
│   └── yet-another-vectornet/                       ← ★ 原始版 PyG VectorNet
│       ├── train.py                                  ← [入口] 多GPU训练入口
│       ├── single_gpu_train.py                       ← [入口] 单GPU训练入口
│       ├── test_and_generate_H5.py                   ← [入口] 测试推理 & 生成Argoverse提交H5
│       ├── compute_feature_module.py                 ← [核心可复用] 数据预处理入口脚本
│       ├── dataset.py                                ← [核心可复用] GraphData / GraphDataset (PyG)
│       ├── requirements.txt                          ← [配置] 原始版依赖（含PyG 1.5.0系列）
│       ├── README.md                                 ← [文档] 原始版说明
│       │
│       ├── modeling/                                 ← ★ 模型核心层
│       │   ├── vectornet.py                          ← [核心可复用] HGNN 主模型 (组合SubGraph+SelfAttn+MLP)
│       │   ├── subgraph.py                           ← [核心可复用] SubGraph + GraphLayerProp (MessagePassing)
│       │   ├── selfatten.py                          ← [核心可复用] SelfAttentionLayer + masked_softmax
│       │   └── predmlp.py                            ← [核心可复用] TrajPredMLP 预测头
│       │
│       ├── utils/                                    ← ★ 工具层
│       │   ├── config.py                             ← [核心可复用] 全局配置常量
│       │   ├── feature_utils.py                      ← [核心可复用] 特征提取 & 编码核心流水线
│       │   ├── agent_utils.py                        ← [核心可复用] Agent轨迹特征提取
│       │   ├── object_utils.py                       ← [核心可复用] 周围目标特征（速度/静止/padding）
│       │   ├── lane_utils.py                         ← [核心可复用] 车道特征（含幻觉车道生成）
│       │   ├── viz_utils.py                          ← [核心可复用] 可视化（polyline重建/预测对比）
│       │   ├── eval.py                               ← [核心可复用] 评估指标（minADE/minFDE/MissRate）
│       │   └── mv.py                                 ← [废弃] 文件移动工具（一次性脚本）
│       │
│       ├── mydata/                                   ← [数据] Argoverse CSV
│       │   ├── train/  (14个CSV: 11800.csv~893.csv)
│       │   ├── val/    (14个CSV)
│       │   └── test/   (14个CSV)
│       │
│       └── interm_data/                              ← [中间产物] compute_feature_module.py 生成
│           ├── train_intermediate/  (*.pkl)
│           ├── val_intermediate/    (*.pkl)
│           ├── test_intermediate/   (*.pkl)
│           └── *-norm_center_dict.pkl
│
├── Vector/                                           ← ★ 衍生项目根
│   └── yet-another-vectornet/                        ← ★ 增强版 纯PyTorch 多模态
│       ├── enhanced_train.py                         ← [入口] ★ 增强版主训练入口
│       ├── simple_train.py                           ← [重复] 简化版训练（与原始版功能重叠）
│       ├── standalone_train.py                       ← [废弃/教学] 纯Python无框架版
│       ├── demo_app.py                               ← [新增] Flask Web演示 & REST API
│       ├── predict.py                                ← [新增] 独立预测脚本
│       ├── evaluate.py                               ← [重复] 独立评估脚本（与原始版utils/eval.py功能重叠）
│       ├── project_report.py                         ← [废弃] 一次性报告生成脚本
│       ├── templates/index.html                      ← [新增] Web前端（Canvas交互）
│       ├── requirements.txt                          ← [强耦合] 从原始版复制，含虚依赖（PyG等未实际使用）
│       ├── Dockerfile                                ← [新增] 容器化部署
│       ├── .gitignore                                ← [配置]
│       ├── best_multimodal_model.pth                  ← [产物] 最佳模型权重
│       ├── final_multimodal_model.pth                 ← [产物] 最终模型权重
│       ├── demo_pred_*.png                           ← [产物] 演示可视化输出
│       ├── 技术报告.txt                               ← [产物] project_report.py输出
│       └── mydata/                                   ← [重复] 与原始版完全相同的42个CSV副本
│           ├── train/
│           ├── val/
│           └── test/
│
├── argoverse-api-master/                             ← [依赖] Argoverse 1.1 API源码
│   └── argoverse-api-master/
│       ├── argoverse/
│       │   ├── data_loading/   (argoverse_forecasting_loader.py)
│       │   ├── map_representation/  (map_api.py, lane_segment.py)
│       │   ├── evaluation/     (eval_forecasting.py)
│       │   └── utils/          (geometry, se2, se3)
│       ├── setup.py
│       └── argoverse/config/   (argoverse-v1.1.yaml)
│
└── docs/                                             ← ★ 本文档所在
    └── REFACTOR_PLAN.md
```

### 标注汇总

| 角色 | 文件列表 |
|------|---------|
| **[入口]** | `train.py`, `single_gpu_train.py`, `test_and_generate_H5.py`, `enhanced_train.py` |
| **[核心可复用]** | `modeling/*`, `utils/*`, `dataset.py`, `compute_feature_module.py` |
| **[重复]** | `simple_train.py`(与原始版训练重叠), `evaluate.py`(与 `utils/eval.py` 重叠), `mydata/`(42个CSV完全重复) |
| **[废弃/一次性]** | `mv.py`, `standalone_train.py`(纯Python教学版), `project_report.py`(一次性脚本) |
| **[强耦合/虚依赖]** | `Vector/yet-another-vectornet/requirements.txt`(含未使用的PyG等) |
| **[新增可保留]** | `demo_app.py`, `templates/index.html`, `Dockerfile`, `predict.py` |

---

## 2. 合并路线选择

### ⭐ 选择：路线 A —— 以原始 PyG 版为主干，把增强版多模态思想"移植"为可选模块

### 理由

#### 为什么不选 路线 B（以增强版为主干）？

| 维度 | 路线 B 致命缺陷 |
|------|----------------|
| **图拓扑丢失** | 增强版 `SubGraph` 本质是 3 层 `nn.Linear`，不是图网络。完全丢失了 polyline 内部节点间的消息传递和图拓扑结构信息，这恰恰是 VectorNet 论文的核心创新点。 |
| **无车道/周围车辆** | 增强版只输入 agent 的 20 帧轨迹（`(B,20,8)`），完全丢弃了车道线特征和周围车辆特征。这不是 VectorNet，是时序轨迹预测。原始版通过 `compute_feature_module.py` 完整提取了 agent + 周围车辆 + 车道 的 polyline 图。 |
| **性能天花板低** | 没有车道上下文和交互建模，多模态的 K=3 条轨迹本质上是同一个编码器 + 不同 MLP 头的过参数化技巧，不是真正的多模态推理。 |
| **PyG 无法复现** | 要在增强版上"重新实现 PyG 图结构"，等价于从零重写整个 `dataset.py` + `subgraph.py` + `selfatten.py`，代价远大于把增强版的多模态头移植到原始版。 |
| **与 Argoverse 断联** | 增强版不依赖 Argoverse API，这意味着无法使用完整的 Argoverse 数据集（20 万+场景），也无法生成官方提交格式的 H5 文件。 |

#### 路线 A 的优势

| 维度 | 路线 A |
|------|--------|
| **保留 GraphNN 核心** | 原始版的 `GraphLayerProp(MessagePassing)` + `max_pool` + `SelfAttentionLayer` 是 VectorNet 论文的正确实现 |
| **保留数据流水线** | `compute_feature_module.py` → `.pkl` → `GraphDataset` 已经跑通，支持完整 Argoverse 数据 |
| **多模态可插拔** | 增强版的 `MultiModalVectorNet` 的核心思想（K 个轨迹预测头 + mode_selector + 不确定性）可以作为原始版 HGNN 的一个可选预测头模块 |
| **改动最小** | 只需 `modeling/` 下新增一个文件 + 修改 `vectornet.py` 的一个子类，不破坏现有流程 |

#### 执行思路

```text
原始版 HGNN（保留）                     增强版多模态思想
     │                                       │
     │  subgraph.py (PyG MessagePassing)     │
     │  selfatten.py (masked_softmax)        │
     │  predmlp.py (MLP 预测头)             │
     │                                       │
     └──── 新增 MultiModalHead ──────────────┘
              │
              ├─ mode_selector: Linear -> K个logits
              ├─ traj_predictors: K个MLP头
              ├─ uncertainty_predictor: softplus
              └─ MultiModalLoss (best-of-K MSE + CE)

最终统一下：
  - train.py 增加 --multimodal 开关
  - 默认行为不变（单模态，max_n_guesses=1）
  - 可选开启多模态（K=3, 输出 trajectories+probs+uncertainty）
```

---

## 3. Top 20 问题清单

### P0 — 阻塞运行

| # | 问题 | 位置 | 影响 | 根因 |
|---|------|------|------|------|
| P0-1 | **增强版 requirements.txt 含虚依赖** | `Vector/yet-another-vectornet/requirements.txt` | `pip install -r` 失败或安装不必要的 PyG/argoverse，且实际代码不依赖它们 | 从原始版直接复制，未清理 |
| P0-2 | **两个版本各有独立 mydata/** | `Vector（驾驶轨迹预测）/yet-another-vectornet/mydata/` 和 `Vector/yet-another-vectornet/mydata/` | 42个CSV完全重复，数据修改不同步，磁盘浪费 | 两个版本独立复制数据 |
| P0-3 | **路径硬编码，跨版本不通用** | `config.py` 中 `DATA_DIR = './mydata'`，`enhanced_train.py` 中 `DATA_DIR = './mydata'` | 迁移到新结构后所有硬编码路径失效 | 缺乏统一路径解析 |
| P0-4 | **依赖版本锁定过老** | `requirements.txt` 中 `torch==1.4.0`, `torch-geometric==1.5.0`, CUDA 版本不明确 | 新环境(如CUDA 11/12, torch 2.x)可能安装失败 | 未维护依赖矩阵 |

### P1 — 影响质量

| # | 问题 | 位置 | 影响 | 根因 |
|---|------|------|------|------|
| P1-1 | **增强版无车道/周围车辆特征** | `enhanced_train.py:TrajectoryDataset` | 只有agent轨迹，模型退化为时序预测，非图预测 | 简化设计，未使用原始版的特征提取 |
| P1-2 | **增强版 SubGraph 是假的图网络** | `enhanced_train.py:SubGraph` → 只是3层 Linear+LayerNorm | 没有消息传递、没有节点间拓扑，名不副实 | 不用PyG的简化实现 |
| P1-3 | **demo_app.py 重新定义了一套模型类** | `demo_app.py:L26-L177` 与 `enhanced_train.py:L68-L182` 完全重复 | 三份相同模型定义（enhanced/simple/demo），任何修改需同步3处 | 没有提取共享 model 模块 |
| P1-4 | **simple_train.py 和 enhanced_train.py 功能高度重叠** | `Vector/yet-another-vectornet/simple_train.py` | 两个训练入口，数据加载逻辑各写一遍 | 缺乏统一的训练框架 |
| P1-5 | **evaluate.py 与原始版 utils/eval.py 功能重叠** | `Vector/yet-another-vectornet/evaluate.py` vs 原始版 `utils/eval.py` | 两套评估，一个算ADE/FDE/MSE，另一个算minADE/minFDE/MissRate，口径不一致 | 各自独立开发 |
| P1-6 | **数据量极小（42场景）对模型无明显区分度** | 两个版本的 `mydata/` | 模型评估无统计意义，1400个batch就能过拟合 | 精简数据集，完整Argoverse有20万+场景 |
| P1-7 | **原始版 DataLoader 导入不统一** | `train.py` 用 `torch_geometric.loader`，`single_gpu_train.py` 用 `torch_geometric.data` | 不同PyG版本行为不同，可能导致静默错误 | 历史遗留 |
| P1-8 | **原始版 valid_lens 变量名拼写不一致** | `vectornet.py:L53` `valid_lens` vs 存储的 `valid_len` | 可运行但容易混淆 | 代码未review |

### P2 — 影响维护

| # | 问题 | 位置 | 影响 | 根因 |
|---|------|------|------|------|
| P2-1 | **两个版本目录名几乎相同** | `Vector（驾驶轨迹预测）/yet-another-vectornet/` vs `Vector/yet-another-vectornet/` | 极易混淆，`（驾驶轨迹预测）` 含中文括号在部分shell中需转义 | 复制目录未改结构 |
| P2-2 | **utils/mv.py 是一次性脚本遗留在代码库** | `utils/mv.py` | 无文档、无人知晓用途，污染代码库 | 开发时临时创建未清理 |
| P2-3 | **modeling/ 下的文件引入了大量无关 import** | `subgraph.py`, `selfatten.py`, `predmlp.py` 每个文件都 import 了 `numpy`, `pandas`, `matplotlib`, `viz_utils` | 模块耦合度过高，无法独立测试单个建模层 | 复制粘贴 import 块 |
| P2-4 | **test_and_generate_H5.py 内有未被调用的死代码块** | `test_and_generate_H5.py:L142-L153` 被注释掉的 `forward_and_save` 函数 | 维护困惑 | 重构遗留 |
| P2-5 | **两个版本使用不同的 numpy→tensor→GPU 范式** | 原始版用 `.pkl`→GraphData→DataLoader；增强版用 `.csv`→Dataset→collate_fn | 维护者需理解两套完全不同的数据管线 | 架构分裂 |
| P2-6 | **缺乏统一配置文件** | 增强版参数硬编码为模块级常量（L30-L41），原始版在 config.py | 调整超参需改代码而非改配置文件 | 开发习惯 |
| P2-7 | **缺少 smoke_test / CI 脚本** | 整个 repo 无 | 重构后无法快速验证破坏性变更 | 未建立工程化流程 |
| P2-8 | **中文命名目录/文件** | `Vector（驾驶轨迹预测）/`, `技术报告.txt` | 跨平台兼容性风险，部分CI/CD不支持中文路径 | 开发习惯 |

---

## 4. 分 6 阶段执行计划

> **核心原则**：每阶段结束后代码必须可运行；commit 粒度按阶段划分。

---

### 阶段 1：统一目录结构 & 清理废弃物

**目标**：建立清晰的 monorepo 结构，消除中英文命名和重复数据。

**操作**：

| 动作 | 文件/目录 | 说明 |
|------|-----------|------|
| **新建** | `project/` 顶层工作目录 | 合并后的唯一代码根 |
| **迁移** | `Vector（驾驶轨迹预测）/yet-another-vectornet/*` → `project/` | 原始版所有代码搬到 `project/` |
| **迁移** | `argoverse-api-master/` → `project/argoverse-api/` | 第三方依赖统一放子目录或 setup.py 声明 |
| **合并数据** | 删除 `Vector/yet-another-vectornet/mydata/`，`project/mydata/` 作为唯一数据源 | 不再有重复 CSV |
| **删除** | `Vector（驾驶轨迹预测）/` 和 `Vector/` 空目录 | 中文括号目录彻底移除 |
| **删除** | `utils/mv.py` | 一次性废弃脚本 |
| **删除** | `project_report.py`, `技术报告.txt` | 一次性产物 |
| **新建** | `project/requirements.txt` | 分类写明 PyG 依赖（原始版所需）和 Web 依赖（demo_app 所需），带注释 |
| **新建** | `project/smoke_test.py` | 最小冒烟测试：读1个CSV → 编码特征 → 构造1个GraphData → forward → 检查输出 shape |

**验证方式**：

```powershell
cd project
python smoke_test.py
# 预期输出: "Smoke test PASSED: GraphData created, HGNN forward shape=(1,60)"
```

**阶段结束状态**：单一 `project/` 目录，原始代码可运行，无中文路径。

---

### 阶段 2：提取共享模型模块 & 消除代码重复

**目标**：把增强版的模型定义从 `enhanced_train.py` / `demo_app.py` 中提取为独立模块，消除三处重复定义。

**操作**：

| 动作 | 文件 | 说明 |
|------|------|------|
| **新建** | `project/modeling/multimodal.py` | 从 `enhanced_train.py` 提取：`SubGraph(nn.Linear版)` → 重命名 `MLPSubGraph`；`AttentionPooling`；`MultiModalVectorNet`；`LinearPredictor`；`LSTMPredictor` |
| **新建** | `project/modeling/losses.py` | 从 `enhanced_train.py` 提取 `MultiModalLoss` |
| **修改** | `enhanced_train.py` | 删除类定义，改为 `from modeling.multimodal import ...` |
| **修改** | `demo_app.py` | 删除类定义，改为 `from modeling.multimodal import MultiModalVectorNet` |
| **修改** | `simple_train.py` | 删除 `main()` 内部类定义，改为 `from modeling.multimodal import MLPSubGraph` |

**验证方式**：

```powershell
cd project
python smoke_test.py                          # 确认原始版HGNN仍然正常
python -c "from modeling.multimodal import MultiModalVectorNet; m=MultiModalVectorNet(8,60,3); print(m)"  # 确认模块可导入
python enhanced_train.py --epochs 1 --dry-run  # 增强版跑1个epoch确认无导入错误
```

**阶段结束状态**：模型定义统一在 `modeling/`，所有训练入口通过 import 引用，不再有重复类定义。

---

### 阶段 3：提取共享数据模块

**目标**：统一数据加载层，让原始版和增强版共享数据读取逻辑。

**操作**：

| 动作 | 文件 | 说明 |
|------|------|------|
| **新建** | `project/data/__init__.py` | 数据层包初始化 |
| **迁移** | `dataset.py` → `project/data/graph_dataset.py` | GraphData/GraphDataset 独立为子模块 |
| **新建** | `project/data/csv_reader.py` | 从 `enhanced_train.py` / `simple_train.py` 提取 `read_csv_file()` / `get_agent_trajectory()` |
| **新建** | `project/data/traj_dataset.py` | 从 `enhanced_train.py` 提取 `TrajectoryDataset` + `collate_fn` |
| **修改** | `train.py`, `single_gpu_train.py`, `test_and_generate_H5.py` | `from dataset import ...` → `from data.graph_dataset import ...` |
| **修改** | `enhanced_train.py`, `simple_train.py`, `predict.py`, `evaluate.py` | `read_csv_file` 等 → `from data.csv_reader import ...` |

**验证方式**：

```powershell
cd project
python -c "from data.graph_dataset import GraphDataset; ds=GraphDataset('interm_data/train_intermediate'); print(len(ds))"
python -c "from data.traj_dataset import TrajectoryDataset; ds=TrajectoryDataset('mydata','train'); print(len(ds))"
```

**阶段结束状态**：所有数据相关代码集中在 `data/` 包，不再散落在各训练脚本中。

---

### 阶段 4：统一配置 & 消除硬编码

**目标**：用单一 YAML/JSON 配置文件替换所有硬编码参数。

**操作**：

| 动作 | 文件 | 说明 |
|------|------|------|
| **新建** | `project/config/default.yaml` | 统一配置：`data_dir`, `obs_len`, `pred_len`, `feature_dim`, `hidden_dim`, `k_modes`, `batch_size`, `epochs`, `lr`, `lane_radius`, `obj_radius` |
| **新建** | `project/config/__init__.py` | `load_config()` 函数，支持命令行覆盖 |
| **修改** | `utils/config.py` | 保留常量作为 fallback，新增 `from config import load_config` |
| **修改** | `enhanced_train.py` | 删除模块级常量（L30-L41），改为 `cfg = load_config()` |
| **修改** | 所有入口文件 | 参数来源于 config 而非硬编码 |

**验证方式**：

```powershell
cd project
python -c "from config import load_config; cfg=load_config(); print(cfg['obs_len'])"
python enhanced_train.py --config config/default.yaml --epochs 1
```

**阶段结束状态**：超参通过 YAML 管理，改参数不需要改代码。

---

### 阶段 5：多模态模块接入原始版 HGNN

**目标**：这就是合并的核心步骤——让原始版 HGNN 支持可选的多模态输出。

**操作**：

| 动作 | 文件 | 说明 |
|------|------|------|
| **新建** | `project/modeling/multimodal_head.py` | `MultiModalHead(nn.Module)`：接收 `(B, 64)` encoded 向量 → mode_selector + K个TrajPredMLP + uncertainty_predictor → `{trajectories, mode_probs, uncertainty}` |
| **修改** | `project/modeling/vectornet.py` | `HGNN.__init__` 增加 `multimodal=False, k_modes=3` 参数；`forward` 末尾增加分支：`if self.multimodal: return self.mm_head(encoded); else: return self.traj_pred_mlp(encoded)` |
| **修改** | `project/train.py` | 添加 `--multimodal` 命令行开关，`if multimodal: criterion = MultiModalLoss(k_modes=3)` |
| **修改** | `project/utils/eval.py` | 增加多模态评估分支（分别计算每个 mode 的 minADE/minFDE） |

**验证方式**：

```powershell
cd project
# 单模态（回退测试，确保不破坏原有功能）
python train.py --epochs 2 --batch_size 256

# 多模态（新功能）
python train.py --epochs 2 --batch_size 256 --multimodal --k_modes 3

# smoke test 自动检测两种模式
python smoke_test.py --multimodal
```

**阶段结束状态**：HGNN 原生支持单/多模态切换，通过参数控制，不破坏原有功能。

---

### 阶段 6：清理、文档 & 集成验证

**目标**：构建标准化项目结构，添加 smoke_test 和 CI 基本脚本。

**操作**：

| 动作 | 文件 | 说明 |
|------|------|------|
| **更新** | `project/README.md` | 统一的项目说明：架构图、快速开始、两种训练模式说明 |
| **更新** | `project/requirements.txt` | 最终清理：分成 `[base]`, `[pyg]`, `[web]` 三组可选依赖 |
| **新建** | `project/scripts/run_all_tests.ps1` | Windows PowerShell 冒烟测试脚本 |
| **新建** | `project/scripts/run_all_tests.sh` | Linux/macOS 冒烟测试脚本 |
| **修改** | `smoke_test.py` | 扩展为覆盖：CSV读取→pkl生成→GraphDataset加载→单模态forward→多模态forward→评估 |
| **删除** | `simple_train.py` | 功能已被 `enhanced_train.py --simple` 或 `train.py` 完全覆盖 |
| **删除** | `standalone_train.py` | 纯Python教学版无生产价值，如需要可归档到 `archive/` |
| **移动** | `evaluate.py` → `scripts/evaluate.py` | 作为辅助脚本而非核心入口 |
| **移动** | `predict.py` → `scripts/predict.py` | 同上 |
| **新建** | `.gitignore` | Python标准忽略规则 + `*.pkl` + `*.pth` |

**验证方式**：

```powershell
cd project
python scripts/run_all_tests.ps1
# 预期输出:
#   [1/5] CSV data integrity check ... PASSED
#   [2/5] Feature encoding smoke test ... PASSED
#   [3/5] Single-modal HGNN forward ... PASSED
#   [4/5] Multi-modal HGNN forward ... PASSED
#   [5/5] Full train 1-epoch pipeline ... PASSED
#   =========================================
#   ALL TESTS PASSED
```

**阶段结束状态**：干净的项目结构，可一键验证完整性。

---

## 附录 A：最终目录结构（目标状态）

```
project/
├── config/
│   ├── __init__.py           # load_config()
│   └── default.yaml           # 统一配置
├── data/
│   ├── __init__.py
│   ├── graph_dataset.py       # GraphData, GraphDataset (PyG)
│   ├── traj_dataset.py        # TrajectoryDataset (pure torch)
│   └── csv_reader.py          # read_csv_file, get_agent_trajectory
├── modeling/
│   ├── __init__.py
│   ├── vectornet.py           # HGNN (含多模态开关)
│   ├── subgraph.py            # SubGraph + GraphLayerProp (PyG MessagePassing)
│   ├── selfatten.py           # SelfAttentionLayer + masked_softmax
│   ├── predmlp.py             # TrajPredMLP
│   ├── multimodal.py          # MLPSubGraph, AttentionPooling, MultiModalVectorNet
│   ├── multimodal_head.py     # MultiModalHead (接入HGNN)
│   └── losses.py              # MultiModalLoss
├── utils/
│   ├── config.py              # 保留常量
│   ├── feature_utils.py       # 特征提取流水线
│   ├── agent_utils.py
│   ├── object_utils.py
│   ├── lane_utils.py
│   ├── viz_utils.py
│   └── eval.py                # 评估指标（含多模态分支）
├── scripts/
│   ├── evaluate.py            # 独立评估
│   ├── predict.py             # 独立预测
│   ├── run_all_tests.ps1
│   └── run_all_tests.sh
├── train.py                   # 统一训练入口
├── compute_feature_module.py  # 数据预处理
├── demo_app.py               # Web演示
├── smoke_test.py              # 冒烟测试
├── templates/index.html       # Web前端
├── Dockerfile
├── requirements.txt           # 分组依赖
├── README.md
├── mydata/                    # 唯一数据源
│   ├── train/
│   ├── val/
│   └── test/
└── interm_data/               # 中间产物（.gitignore忽略）
```

---

## 附录 B：合并前后对比

| 维度 | 合并前 | 合并后 |
|------|--------|--------|
| 目录 | 2个版本，中文路径 | 1个 `project/`，纯英文 |
| 模型定义 | 3处重复 | `modeling/` 统一 |
| 数据文件 | 2份相同的42个CSV | 1份 |
| 配置文件 | 硬编码散落各处 | `config/default.yaml` 集中管理 |
| 多模态 | 仅增强版有，但不支持图结构 | HGNN原生支持，可选开关 |
| 评估 | 两套口径不一致 | 统一在 `utils/eval.py` |
| 依赖声明 | 增强版含虚依赖 | 分组可选依赖 |
| 可验证性 | 无自动化测试 | `smoke_test.py` + 阶段脚本 |
