# ChronoMotion

基于 VectorNet 架构的自动驾驶轨迹预测系统。支持多模态轨迹预测（K=3 条候选轨迹），提供训练、评估、预测和 Web 可视化 Demo。

## 项目结构

```
zxzx/
├── Vector/
│   └── yet-another-vectornet/    # 增强版——纯 PyTorch 实现
│       ├── enhanced_train.py      # 训练脚本
│       ├── predict.py             # 单文件预测脚本
│       ├── evaluate.py            # 批量评估脚本
│       ├── model.py               # 模型定义（MLPEncoder + MultiModalVectorNet）
│       ├── check.py               # 冒烟测试（验证模型/配置/设备）
│       ├── config.json            # 训练配置文件
│       ├── demo_app.py            # Web 可视化 Demo
│       ├── templates/index.html   # Web 前端界面
│       ├── Dockerfile             # Docker 容器化部署
│       └── requirements.txt       # Python 依赖
└── docs/                          # 项目文档
    ├── PROJECT_GUIDE.md           # 增强版操作指南
    └── REFACTOR_PLAN.md           # 版本对比与融合计划
```

## 快速开始

### 1. 创建虚拟环境

```powershell
python -m venv TraeAI-4
.\TraeAI-4\Scripts\Activate.ps1
```

### 2. 安装依赖

```powershell
cd Vector\yet-another-vectornet
pip install -r requirements.txt
```

### 3. GPU 支持（推荐）

如果你有 NVIDIA GPU（如 RTX 3060），使用 GPU 版 PyTorch 可以加速训练。

```powershell
# 1. 安装 Visual C++ 运行时（如遇到 DLL 加载错误）
winget install Microsoft.VCRedist.2015+.x64

# 2. 安装 CUDA 版 PyTorch
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 验证 GPU 可用
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

### 4. 冒烟测试

```powershell
python check.py
```

输出应显示模型参数数量（~198K）、输入/输出形状，并打印 `ALL CHECKS PASSED`。

## 数据准备

### 数据格式

CSV 文件，每行包含 agent 的一条轨迹记录：
```
TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,HEADING,VX,VY,SCENARIO_ID,CITY_NAME,INTERSECTION
```

### 目录结构

```
yet-another-vectornet/mydata/
├── train/    # 训练集（205,472 条）
├── val/      # 验证集（39,472 条）
└── test/     # 测试集（39,472 条）
```

**注意**：数据集文件不随代码仓库一同提交，请自行将数据放入 `mydata/` 目录。

## 训练

### 配置文件（config.json）

```json
{
    "data_dir": "./mydata",
    "obs_len": 20,
    "pred_len": 30,
    "feature_dim": 8,
    "hidden_dim": 128,
    "k_modes": 3,
    "out_dim": 60,
    "batch_size": 64,
    "epochs": 20,
    "lr": 0.001,
    "num_workers": 0,
    "decay_lr_factor": 0.3,
    "decay_lr_every": 10,
    "seed": 13,
    "weight_decay": 0.01,
    "grad_clip": 1.0
}
```

| 参数 | 说明 |
|------|------|
| `obs_len` | 观测帧数（20 帧 = 2 秒） |
| `pred_len` | 预测帧数（30 帧 = 3 秒） |
| `k_modes` | 多模态预测数量（=3） |
| `batch_size` | 批次大小（64，适配 RTX 3060 6GB） |
| `epochs` | 训练轮数 |

### 开始训练

```powershell
python enhanced_train.py
```

### 预期性能

| 设备 | batch_size | 速度 | 20 epoch 时间 |
|------|-----------|------|--------------|
| RTX 3060 6GB | 64 | ~1.15 it/s | ~14 小时 |
| 内存预加载模式 | 64 | ~15-25 it/s | ~40-60 分钟 |

当前瓶颈在于从 20 万个小 CSV 文件读取数据（磁盘 `fopen()` 开销），而非 GPU 计算能力。

模型权重保存为 `best_multimodal_model.pth`，包含 `model_state_dict`、`optimizer_state_dict` 和 `config`，支持断点续训。

## 推理与评估

### 单文件预测

```powershell
python predict.py
```
输入文件名（如 `893.csv`），输出预测轨迹 JSON。

### 批量评估

```powershell
python evaluate.py
```
计算验证集上的 ADE / FDE 指标。

## Web 可视化 Demo

```powershell
python demo_app.py
```

浏览器打开 `http://localhost:5000`，拖拽 CSV 文件即可查看预测结果可视化。

## Docker 部署

```powershell
cd Vector\yet-another-vectornet
docker build -t chronomotion .
docker run -p 5000:5000 chronomotion
```

## 版本说明

本仓库包含两个版本的实现：

| | 增强版（当前使用） | 原始 PyG 版 |
|---|---|---|
| **代码** | `Vector/yet-another-vectornet/` | `Vector（驾驶轨迹预测）/yet-another-vectornet/` |
| **架构** | MLP Encoder + AttentionPooling + MLP Predictor | 真正的 VectorNet 分层 GNN（MessagePassing） |
| **输入** | Agent 轨迹 (20, 8) | Agent + 周围车辆 + 车道拓扑 |
| **优势** | 训练快、轻量、易部署 | 更准确、符合论文设计 |
| **劣势** | 预测精度有限 | 训练慢（2h/epoch）、需要 PyG 安装 |

如需使用原始版，请参考 [Google Drive](https://drive.google.com/drive/folders/1XJ2Oz4Qc2UstnfRw3DNvQThuEVvM6tUL) 上的预处理数据和 `docs/REFACTOR_PLAN.md`。

## 常见问题

**Q: 训练很慢怎么办？**
A: 瓶颈是磁盘 I/O（20 万个小 CSV 文件）。可以做内存预加载——先将所有 CSV 读入一个 `.pt` 文件，训练时直接加载到内存，速度可提升 10-20 倍。

**Q: GPU 不可用？**
A: 确保安装了 Visual C++ Redistributable（`winget install Microsoft.VCRedist.2015+.x64`），然后安装 CUDA 版 PyTorch。重启终端后验证。

**Q: 预测精度不够？**
A: 增强版使用简化的 MLP 预测器，精度有限。如需更高精度，可切换到原始 VectorNet GNN 版本或对增强版做进一步优化。
