# 轻量化自动驾驶轨迹预测系统

根据车辆过去 2 秒的行驶轨迹，预测未来 3 秒的 3 条可能路径。
基于 VectorNet 框架简化实现——MLP 编码器 + 注意力池化 + 多模态输出，支持断点续训。

## 项目文件说明

```
zxzx/
├── README.md                       # 本文档
├── .gitignore                      # Git 忽略规则（mydata/ 不提交到 GitHub）
├── docs/
│   ├── 断点续训修改指南.md           # 断点续训功能的技术说明
│   └── 项目研究内容与关键问题.docx    # 项目报告文档（含结果图表）
└── Vector/
    └── yet-another-vectornet/      # 核心代码
        ├── config.json             # 训练参数配置
        ├── requirements.txt        # Python 依赖包
        ├── check.py                # 冒烟测试（先跑这个！）
        ├── enhanced_train.py       # 训练脚本（支持断点续训）
        ├── evaluate.py             # 评估脚本（计算 ADE/FDE 指标）
        ├── predict.py              # 单文件/批量预测脚本
        ├── model.py                # 模型结构定义（MultiModalVectorNet）
        ├── demo_app.py             # Web 可视化 Demo
        ├── templates/
        │   └── index.html          # Web 界面
        ├── Dockerfile              # Docker 部署文件
        └── mydata/                 # ⚠️ 数据文件夹（需自行创建）
            ├── train/              #   训练集 CSV 文件（205,472 个）
            ├── val/                #   验证集 CSV 文件（ 39,472 个）
            └── test/               #   测试集 CSV 文件（ 39,472 个）
```

## 第一步：环境准备（Windows + NVIDIA 显卡）

以下操作全部在 PowerShell 中执行。

### 1.1 安装 Visual C++ 运行时（必须！）

如果之前没装过，先装这个，否则 PyTorch GPU 版会报 DLL 错误：

```powershell
winget install Microsoft.VCRedist.2015+.x64
```

装完之后**重启终端**。

### 1.2 创建虚拟环境

在项目根目录（`zxzx/`）下执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 报「无法加载文件，因为在此系统上禁止运行脚本」：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

然后再执行激活命令。激活成功后，终端前面会出现 `(.venv)` 字样。

### 1.3 安装依赖

```powershell
cd Vector\yet-another-vectornet
pip install -r requirements.txt
```

### 1.4 安装 GPU 版 PyTorch（有 NVIDIA 显卡才做）

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

验证 GPU 是否可用：

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

如果输出 `CUDA: True` 并显示你的显卡型号，说明 GPU 环境配置成功。

### 1.5 冒烟测试

```powershell
python check.py
```

输出示例：

```
==================================================
Smoke Test
==================================================

[1/4] Config loaded: feature_dim=8, hidden_dim=128, k_modes=3
[2/4] Model created: 198,516 params
[3/4] Input shape: torch.Size([2, 20, 8])
[4/4] Output: trajectories=torch.Size([2, 3, 60]), mode_probs=torch.Size([2, 3]), uncertainty=torch.Size([2, 3])

==================================================
ALL CHECKS PASSED
==================================================
```

看到 `ALL CHECKS PASSED` 就说明代码没问题，可以继续了。

## 第二步：准备数据

### 2.1 数据说明

项目使用 CSV 格式的轨迹数据，每行记录一个目标（AGENT）或周围车辆（OTHERS）在某一时刻的状态：

```
TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,HEADING,VX,VY,SCENARIO_ID,CITY_NAME,INTERSECTION
```

其中 `OBJECT_TYPE` 列：
- `AGENT` — 预测目标车辆
- `OTHERS` — 周围其他车辆
- `AV` — 自动驾驶车辆（忽略）

每条样本包含 20 个历史时刻（2 秒，10Hz）+ 30 个未来时刻（3 秒）。

### 2.2 数据目录结构

在 `Vector/yet-another-vectornet/` 目录下新建 `mydata/` 文件夹：

```
mydata/
├── train/          # 放训练集 CSV 文件（205,472 个文件）
├── val/            # 放验证集 CSV 文件（ 39,472 个文件）
└── test/           # 放测试集 CSV 文件（ 39,472 个文件）
```

> ⚠️ **mydata/ 不会提交到 GitHub**（已在 .gitignore 中排除），每个队友需自行准备数据。

文件命名规则：`{编号}.csv`，例如 `893.csv`、`16177.csv`。

### 2.3 验证数据是否正确

在 `yet-another-vectornet/` 下执行：

```powershell
python -c "import os; [print(f'{d}: {len(os.listdir(\"mydata/\"+d))} files') for d in ['train','val','test']]"
```

预期输出：

```
train: 205472 files
val: 39472 files
test: 39472 files
```

## 第三步：训练模型

### 3.1 训练参数说明

`config.json` 里可以调整的参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 64 | 每批处理的样本数（6GB 显存不要超过 64） |
| `epochs` | 20 | 训练轮数（可在命令行用 `--epochs` 覆盖） |
| `lr` | 0.001 | 学习率 |
| `obs_len` | 20 | 观测帧数（= 2 秒） |
| `pred_len` | 30 | 预测帧数（= 3 秒） |
| `k_modes` | 3 | 预测多少条候选轨迹 |
| `hidden_dim` | 128 | 隐藏层维度 |
| `weight_decay` | 0.01 | 权重衰减 |

### 3.2 开始训练

```powershell
python enhanced_train.py
```

训练日志会实时显示进度条和损失值（tqdm 进度条）。每轮结束后自动保存最优模型到 `best_multimodal_model.pth`。

### 3.3 断点续训（训练中断后恢复）

脚本已内置断点续训功能，无需额外参数。如果 `best_multimodal_model.pth` 存在：
- 自动加载模型权重和优化器状态
- 从上次中断的轮次继续训练
- 恢复学习率调度器状态

```
# 首次训练
python enhanced_train.py          # 从 epoch 0 开始

# 中断后继续（Ctrl+C 或电脑休眠后）
python enhanced_train.py          # 自动检测 .pth 并从断点恢复
```

输出示例：

```
Resumed from epoch 15, best val_loss=1.6594, best ADE=2.4777
```

### 3.4 预计训练时间

| 显卡 | batch_size | 每轮时间 | 20 轮总时间 |
|------|-----------|---------|------------|
| RTX 3060 6GB | 64 | ~48 分钟 | ~16 小时 |
| CPU 模式 | 64 | ~3 小时 | ~60 小时 |

> 训练慢主要是因为要从 20 万个小 CSV 文件读取数据（磁盘 I/O 瓶颈），GPU 利用率约 8%。内存预加载可加速 10 倍以上。

## 第四步：使用模型

### 4.1 预测单个 CSV 文件

```powershell
# 用最佳模型预测
python predict.py --model best_multimodal_model.pth --input ../../csv数据/893.csv --device cuda

# 或用纯权重文件
python predict.py --model final_multimodal_model.pth --input ../../csv数据/893.csv --device cuda
```

### 4.2 批量预测文件夹

```powershell
python predict.py --model best_multimodal_model.pth --input ../../csv数据 --output ./output --device cuda
```

### 4.3 评估模型性能

```powershell
# 在测试集上评估
python evaluate.py --model best_multimodal_model.pth --split test --device cuda

# 在验证集上评估
python evaluate.py --model best_multimodal_model.pth --split val --device cuda
```

会在 `evaluation_results/` 目录下生成评估报告（JSON + 文本 + 图表）。

### 4.4 Web 可视化 Demo

```powershell
python demo_app.py
```

浏览器打开 `http://localhost:5000`，可以拖拽 CSV 文件查看预测轨迹可视化。

## 模型信息

| 项目 | 数值 |
|------|------|
| 参数量 | 198,516 |
| 输入 | 8 维特征 × 20 帧 |
| 输出 | 3 条候选轨迹 × 30 帧（60 维） |
| 架构 | MLP Encoder → Attention Pooling → Multi-Modal Predictor |
| 训练轮数 | 20 epochs |
| 最佳轮次 | Epoch 18 |
| 验证损失 | 1.66 |

### 性能指标（测试集 5,000 样本）

| 指标 | 数值 | 说明 |
|------|------|------|
| ADE | 1.30 ± 1.11 m | 平均位移误差（30 帧平均） |
| FDE | 2.81 ± 2.73 m | 终点位移误差（最后一帧） |

## 常见问题

**Q: `ModuleNotFoundError: No module named 'xxx'`**
A: 确认已激活虚拟环境（终端前面有 `(.venv)`），然后 `pip install -r requirements.txt`。

**Q: `CUDA: False`**
A: 
1. 确认装了 Visual C++ 运行时（`winget install Microsoft.VCRedist.2015+.x64`）
2. 确认装了 GPU 版 PyTorch（`pip install torch --index-url https://download.pytorch.org/whl/cu124`）
3. 重启终端再试

**Q: 训练到一半中断了怎么办？**
A: 直接重新运行 `python enhanced_train.py`，脚本会自动加载 `best_multimodal_model.pth` 从上次的进度继续训练，无需任何额外操作。

**Q: 如何增加训练轮数？**
A: 修改 `config.json` 中的 `epochs` 值，或在命令行指定：`python enhanced_train.py --epochs 50`。已有模型会自动从当前进度继续。

**Q: CSV 文件太多，训练太慢怎么办？**
A: 可将所有 CSV 预加载为 `.pt` 二进制文件，训练时从内存直接读取，速度能快 10 倍以上。GPU 利用率可从此的 8% 提升至接近 100%。

**Q: GitHub 上没有 mydata 文件夹？**
A: 数据文件太大（28 万个 CSV），已通过 `.gitignore` 排除。每个队友需自行把数据放到 `Vector/yet-another-vectornet/mydata/train/`、`mydata/val/`、`mydata/test/` 目录下。

**Q: best_multimodal_model.pth 和 final_multimodal_model.pth 有什么区别？**
A: `best_multimodal_model.pth` 包含模型权重 + 优化器状态 + 配置，用于断点续训；`final_multimodal_model.pth` 仅含纯权重，体积更小（790KB），发给队友做预测用。

**Q: 如何让模型更准确？**
A:
1. 增加训练轮数（修改 `epochs` 并续训）
2. 引入地图车道线特征（当前版本未使用地图信息）
3. 增强周围车辆交互建模（当前仅使用最近一辆车）
4. 增大隐藏维度（128 → 256）或加深编码器
