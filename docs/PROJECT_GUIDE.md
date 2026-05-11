# 增强版操作说明：VectorNet 轨迹预测

> 工作目录：`Vector\yet-another-vectornet\`
> 看完就能跑。

---

## 1. 这个项目做什么？

给模型看一辆车过去 2 秒的行驶轨迹（20 帧），预测接下来 3 秒它会怎么走（30 帧）。

模型会给出 **3 条候选路线**，每条附一个概率，还会告诉你预测有多不确定。

---

## 2. 装环境

```bash
pip install torch numpy matplotlib tqdm
```

如果需要 Web 界面加一个：

```bash
pip install flask
```

> ⚠️ 项目里的 `requirements.txt` 是错的，不要用。

---

## 3. 数据在哪？

```
mydata/
├── train/   14 个 CSV（训练）
├── val/     14 个 CSV（验证）
└── test/    14 个 CSV（测试）
```

每个 CSV 长这样：

| TIMESTAMP | TRACK_ID | OBJECT_TYPE | X | Y | CITY_NAME |
|-----------|----------|-------------|---|---|-----------|
| ... | ... | AGENT | 3874.3 | 2262.9 | PIT |
| ... | ... | OTHERS | ... | ... | PIT |

不需要你做任何预处理，脚本直接读。

---

## 4. 训练

```bash
cd "Vector\yet-another-vectornet"
python enhanced_train.py
```

输出大致：

```
设备: cuda
多模态预测模式数: 3
训练样本: 14  验证样本: 14

Epoch   1/100 | Train Loss: 0.0234 | Val Loss: 0.0189 | ADE: 2.34 | FDE: 4.12
Epoch  10/100 | Train Loss: 0.0012 | Val Loss: 0.0009 | ADE: 0.52 | FDE: 0.98
  [NEW BEST] ...
训练完成!
```

结束后得到：
- `best_multimodal_model.pth` — 最佳模型
- `final_multimodal_model.pth` — 最终模型
- `demo_pred_1.png` ~ `demo_pred_3.png` — 预测效果图

**改参数**：打开 `enhanced_train.py`，开头几行就是：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| BATCH_SIZE | 8 | 显存不够改小 |
| EPOCHS | 100 | 训练轮数 |
| LR | 0.001 | 学习率 |
| K_MODES | 3 | 候选路线条数 |

---

## 5. 预测

```bash
# 单个文件
python predict.py --input mydata/test/893.csv --model best_multimodal_model.pth --output ./result

# 整个文件夹
python predict.py --input mydata/test/ --model best_multimodal_model.pth --output ./result_all
```

输出 `prediction.png`（图）和 `prediction.json`（数据）。

不想训练的话，项目里已经有现成的模型文件：`best_multimodal_model.pth`、`final_multimodal_model.pth`，直接预测即可。

---

## 6. 评估

```bash
python evaluate.py --model best_multimodal_model.pth --data ./mydata --split test --output ./eval_result
```

生成：文字报告 + JSON 结果 + 统计图。看 ADE 和 FDE，越小越好。

---

## 7. Web 界面（可选）

```bash
python demo_app.py
```

浏览器打开 `http://localhost:5000`，可以在 Canvas 上拖拽画轨迹、点按钮看预测结果。有中英文切换。

---

## 8. 常见问题

### Q1：`No module named 'torch'`
```bash
pip install torch
```

### Q2：显存不够
打开 `enhanced_train.py`，把 `BATCH_SIZE = 8` 改成 `BATCH_SIZE = 2` 或 `4`。

### Q3：数据只有 14 个，能练出效果吗？
14 个场景能验证代码跑通，但数值结果没有泛化意义。要出有效结果需要更多数据。

### Q4：`demo_app.py` 启动报 `No module named 'flask'`
```bash
pip install flask
```

### Q5：收到报错 `FileNotFoundError: mydata/...`
确认你在 `Vector\yet-another-vectornet\` 目录下运行。脚本里路径是 `./mydata`，依赖当前工作目录。

---

## 9. 当前已知需要修正的点

以下几个问题是代码里真实存在的，按优先级排列：

### 🔴 需要立刻改（P0）

**① `requirements.txt` 是错的，装了会失败**

文件里列了 `torch-geometric`、`argoverse` 等，但增强版根本不依赖它们。

**修法**：删掉原文件，新建一个只写：

```
torch
numpy
matplotlib
tqdm
flask
```

---

### 🟡 建议尽快改（P1）

**② `demo_app.py` 和 `enhanced_train.py` 里各写了一遍同样的模型代码**

两个文件都定义了完全相同的 `SubGraph`、`AttentionPooling`、`MultiModalVectorNet`。改一处另一处不会同步。

**修法**：把模型定义从 `enhanced_train.py` 里提取出来，新建一个 `model.py`。然后 `demo_app.py` 和 `enhanced_train.py` 都用 `from model import MultiModalVectorNet`。

---

**③ `enhanced_train.py` 训练过程里评估指标算错了**

代码 L336：`best_traj = outputs['trajectories'][:, 0, :]` — 总是取第 0 条轨迹算 ADE，不是真正的最好轨迹。而 `evaluate.py` 是正确实现了 best-of-K。

**修法**：把 L336 改成和 `evaluate.py` 一样的逻辑（遍历 K 条轨迹取 ADE 最小的那个）。改完后再训练，日志里的 ADE 应该会变小。

---

**④ 模型名叫 SubGraph 但它不是图网络**

代码里 `class SubGraph(nn.Module)` 其实是 3 层 `nn.Linear`，没有任何图操作。

**修法**：把类名改成 `MLPEncoder` 或 `TemporalEncoder`，避免误导。

---

**⑤ `predict.py` 和 `evaluate.py` 里各写了一遍 `read_csv_trajectory()` 和 `compute_features()`**

两份一模一样的代码。

**修法**：统一放到 `enhanced_train.py` 里（那里已经有 `read_csv_file` 和特征构造），然后 import。

---

### 🟢 有空再改（P2）

**⑥ 所有参数硬编码在脚本开头**

改参数要改代码，不方便。

**修法**：加一个 `config.json` 或支持命令行参数覆盖。

---

**⑦ `simple_train.py` 和 `standalone_train.py` 跟 `enhanced_train.py` 功能重叠**

三个训练脚本，容易搞混。

**修法**：保留 `enhanced_train.py` 作为唯一入口，另外两个删掉或移到 `archive/`。

---

## 10. 推荐修正顺序

```
第1步：改 requirements.txt（5分钟）           → 新来的人不会再装错包
第2步：提取共享 model.py（20分钟）            → 改模型只需要改一处
第3步：修复训练评估指标（5分钟）              → 训练的 ADE 数值更准确
第4步：合并 predict/evaluate 重复代码（15分钟）→ 减少维护负担
第5步：重命名 SubGraph（2分钟）               → 名字和实现一致
```

前 3 步做完，这个项目就很干净了。

---

## 文件速查

```
Vector/yet-another-vectornet/
├── enhanced_train.py    ← ★ 训练入口
├── predict.py           ← 预测
├── evaluate.py          ← 评估
├── demo_app.py          ← Web 界面
├── simple_train.py      ← 不用管
├── standalone_train.py  ← 不用管
├── mydata/              ← 数据
│   ├── train/ (14 CSV)
│   ├── val/   (14 CSV)
│   └── test/  (14 CSV)
├── best_multimodal_model.pth
├── final_multimodal_model.pth
└── templates/index.html
```
