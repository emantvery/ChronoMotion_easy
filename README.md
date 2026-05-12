# ChronoMotion

自动驾驶轨迹预测系统——根据车辆过去 2 秒的行驶轨迹，预测未来 3 秒的 3 条可能路径。

## 项目文件说明

```
zxzx/
├── README.md                       # 本文档
├── .gitignore                      # Git 忽略规则（mydata/ 不提交到 GitHub）
└── Vector/
    └── yet-another-vectornet/      # 核心代码
        ├── config.json             # 训练参数配置
        ├── requirements.txt        # Python 依赖包
        ├── check.py                # 冒烟测试（先跑这个！）
        ├── enhanced_train.py       # 训练脚本
        ├── evaluate.py             # 评估脚本（计算 ADE/FDE 指标）
        ├── predict.py              # 单文件预测脚本
        ├── model.py                # 模型结构定义
        ├── demo_app.py             # Web 可视化 Demo
        ├── templates/
        │   └── index.html          # Web 界面
        ├── Dockerfile              # Docker 部署文件
        └── mydata/                 # ⚠️ 数据文件夹（需自行创建，见下方说明）
            ├── train/              #   训练集 CSV 文件
            ├── val/                #   验证集 CSV 文件
            └── test/               #   测试集 CSV 文件
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
python -m venv TraeAI-4
.\TraeAI-4\Scripts\Activate.ps1
```

如果 PowerShell 报「无法加载文件，因为在此系统上禁止运行脚本」：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

然后再执行激活命令。

激活成功后，终端前面会出现 `(TraeAI-4)` 字样。

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

项目使用 CSV 格式的轨迹数据，每行记录一个目标（agent）或周围车辆（others）在某一时刻的状态：

```
TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,HEADING,VX,VY,SCENARIO_ID,CITY_NAME,INTERSECTION
```

其中 `OBJECT_TYPE` 列：

- `AGENT` — 预测目标车辆
- `OTHERS` — 周围其他车辆
- `AV` — 自动驾驶车辆（忽略）

### 2.2 数据目录结构（需要你手动创建）

在 `Vector/yet-another-vectornet/` 目录下新建 `mydata/` 文件夹，结构如下：

```
mydata/
├── train/          # 放训练集 CSV 文件（约 205,472 个文件）
├── val/            # 放验证集 CSV 文件（约 39,472 个文件）
└── test/           # 放测试集 CSV 文件（约 39,472 个文件）
```

> ⚠️ **mydata/ 不会提交到 GitHub**（已在 .gitignore 中排除），每个队友需要自行准备数据。

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
| `epochs` | 20 | 训练轮数 |
| `lr` | 0.001 | 学习率 |
| `obs_len` | 20 | 观测帧数（= 2 秒） |
| `pred_len` | 30 | 预测帧数（= 3 秒） |
| `k_modes` | 3 | 预测多少条候选轨迹 |

### 3.2 开始训练

```powershell
python enhanced_train.py
```

训练日志会实时显示进度条和损失值。每轮训练结束后会自动保存最优模型权重到 `best_multimodal_model.pth`，如果训练中断，下次运行可以从断点继续。

### 3.3 预计训练时间

| 显卡 | batch_size | 每轮时间 | 20 轮总时间 |
|------|-----------|---------|------------|
| RTX 3060 6GB | 64 | ~45 分钟 | ~14 小时 |

训练慢主要是因为要从 20 万个小 CSV 文件读取数据（磁盘 I/O 瓶颈），GPU 本身压力不大。

## 第四步：使用模型

### 4.1 单文件预测

```powershell
python predict.py
```

按提示输入 CSV 文件名（如 `893.csv`），脚本会输出预测轨迹。

### 4.2 批量评估

```powershell
python evaluate.py
```

会在整个测试集上计算 ADE（平均位移误差）和 FDE（终点位移误差）指标。

### 4.3 Web 可视化 Demo

```powershell
python demo_app.py
```

浏览器打开 `http://localhost:5000`，可以拖拽 CSV 文件上去，看到预测轨迹的可视化结果。

## 常见问题

**Q: `ModuleNotFoundError: No module named 'xxx'`**
A: 确认已激活虚拟环境（终端前面有 `(TraeAI-4)`），然后 `pip install -r requirements.txt`。

**Q: `CUDA: False`**
A: 
1. 确认装了 Visual C++ 运行时（`winget install Microsoft.VCRedist.2015+.x64`）
2. 确认装了 GPU 版 PyTorch（`pip install torch --index-url https://download.pytorch.org/whl/cu124`）
3. 重启终端再试

**Q: 训练到一半中断了怎么办？**
A: 直接重新运行 `python enhanced_train.py`，脚本会自动加载 `best_multimodal_model.pth` 从上次的进度继续训练。

**Q: CSV 文件太多，训练太慢怎么办？**
A: 可以做内存预加载——先把所有 CSV 读成一个大 `.pt` 文件，之后训练直接从内存读，速度能快 10-20 倍。需要的话找项目负责人要预处理脚本。

**Q: GitHub 上没有 mydata 文件夹？**
A: 这是故意的。数据文件太大（28 万个 CSV），已通过 `.gitignore` 排除。每个队友需要自行把数据放到 `mydata/train/`、`mydata/val/`、`mydata/test/` 目录下。
