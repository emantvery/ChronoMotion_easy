# ChronoMotion — 轨迹预测

给模型看一辆车过去 2 秒的轨迹，让它猜接下来 3 秒怎么走。

---

## 傻瓜式操作（一步一步跟着做）

### 1. 拉代码

打开终端（PowerShell 或 CMD），复制粘贴下面这行，回车：

```bash
git clone https://github.com/emantvery/ChronoMotion_easy.git
```

等它跑完。

### 2. 进入项目目录

```bash
cd ChronoMotion_easy\Vector\yet-another-vectornet
```

### 3. 放数据

项目不带完整数据集（太大），需要你自己下载 Argoverse 1.1 Motion Forecasting：

> 官网：https://www.argoverse.org/av1.html（注册后下载）
>
> 下载 **Argoverse Motion Forecasting v1.1** 下的 Training、Validation、Testing 三个包。

解压后会得到 `train/`、`val/`、`test/` 三个文件夹（里面全是 CSV），**直接覆盖**项目里 `mydata/` 下面同名文件夹：

```bash
# 假设你解压到了 D:\argoverse_data\
# 把解压出来的 CSV 复制进 mydata/
cp D:\argoverse_data\train\* mydata\train\
cp D:\argoverse_data\val\*   mydata\val\
cp D:\argoverse_data\test\*  mydata\test\
```

> 项目里原来有 14 个示例 CSV，覆盖掉就行。

### 4. 装依赖

```bash
pip install torch numpy matplotlib tqdm
```

> 想玩网页版再多装一个：
> ```bash
> pip install flask
> ```

### 5. 验证环境

```bash
python check.py
```

看到最后一行是 **`ALL CHECKS PASSED`** 就说明装好了。

> 如果报 `No module named 'torch'`，回到第 4 步重装。

### 6. 开始训练

```bash
python enhanced_train.py
```

等它跑完。跑完之后目录下会生成 `best_multimodal_model.pth` 和几张预测效果图。

想调参数？打开 `config.json`，改完保存再跑：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| batch_size | 8 | 显存不够改 2 |
| epochs | 100 | 训练轮数 |
| lr | 0.001 | 学习率 |
| k_modes | 3 | 输出几条候选路线 |

也可以命令行临时覆盖：

```bash
python enhanced_train.py --epochs 10 --batch_size 4
```

### 7. 预测

```bash
python predict.py --input mydata/test/893.csv --model best_multimodal_model.pth --output ./result
```

打开 `result/prediction.png` 看结果。

### 8. 评估

```bash
python evaluate.py --model best_multimodal_model.pth --data ./mydata --split test --output ./eval_result
```

生成报告 + 图表。

### 9. （可选）网页版

```bash
python demo_app.py
```

浏览器打开 `http://localhost:5000`，用鼠标画轨迹，点 Predict 看预测。

---

## 每个文件干什么的

```
Vector/yet-another-vectornet/
│
├── model.py                 ← ★ 模型定义文件
│     MLPEncoder             — 3层MLP编码器（把8维特征变成128维）
│     AttentionPooling       — 注意力池化（给20帧加权求出一个向量）
│     MultiModalVectorNet    — 主模型（编码→注意力→K条轨迹+概率+不确定性）
│     MultiModalLoss         — 损失函数（best-of-K MSE + 交叉熵）
│     LinearPredictor        — 线性基线模型（对比用）
│     LSTMPredictor          — LSTM基线模型（对比用）
│
├── enhanced_train.py        ← ★ 训练入口
│     read_csv_file()        — 读CSV文件
│     get_agent_trajectory() — 提取目标车轨迹
│     get_others_trajectories() — 提取周围车辆轨迹
│     TrajectoryDataset      — 数据集类（把CSV转成(B,20,8)的tensor）
│     train_epoch()          — 训练一轮
│     evaluate()             — 验证一轮（best-of-K ADE/FDE）
│     TrajectoryVisualizer   — 可视化类
│     main()                 — 主函数
│
├── predict.py               ← 预测工具（命令行）
├── evaluate.py              ← 评估工具（命令行）
├── demo_app.py              ← Web演示（Flask + Canvas）
│
├── config.json              ← 所有超参数（改这个不用改代码）
├── requirements.txt         ← 依赖包列表（pip install -r 一键装）
├── check.py                 ← 冒烟测试（跑一遍验证环境是否正常）
├── .gitignore               ← Git忽略规则（不提交模型权重/缓存）
├── Dockerfile               ← Docker镜像配置
│
├── mydata/                  ← ★ 数据（放下载好的Argoverse CSV）
│   ├── train/
│   ├── val/
│   └── test/
│
└── templates/
    └── index.html           ← Web前端页面
```

---

## 常见报错

| 报错 | 怎么办 |
|------|--------|
| `No module named 'torch'` | 回到第 4 步，`pip install torch` |
| `No module named 'flask'` | `pip install flask` |
| `FileNotFoundError: mydata/...` | 确认你当前目录是 `ChronoMotion_easy\Vector\yet-another-vectornet` |
| 训练跑不动/显存不够 | 打开 `config.json`，把 `batch_size` 改成 2 |
| 数据太少/跑不出来 | 确认你下了完整数据集（几万个CSV），不是项目自带的14个示例 |
