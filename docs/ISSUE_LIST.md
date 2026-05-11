# 问题清单与修正申请（增强版）

> 面向项目负责人
> 生成日期：2026-05-11 · 最后更新：2026-05-11
> 分析范围：`Vector\yet-another-vectornet\`

---

## 摘要

| 类别 | 已修复 | 未修复 | 说明 |
|------|--------|--------|------|
| 🔴 P0 阻塞运行 | 1 | 1 | 数据量问题未解决 |
| 🟡 P1 影响质量 | 5 | 0 | 全部修完 |
| 🟢 P2 影响维护 | 5 | 1 | 仅剩「三个训练脚本重叠」未处理 |
| **合计** | **11** | **2** | — |

---

## 关于仓库其他文件夹

- **`argoverse-api-master/`**：Argoverse 官方 API 的 GitHub 仓库副本（commit `c81d9375`）。内涵 `argoverse/` 包（数据加载器、地图 API、官方评估指标 `eval_forecasting.py`）。**仅原始版依赖它，增强版完全不需要。** 可作为原始版的本地安装源（`pip install -e argoverse-api-master/argoverse-api-master/`）。
- **`Vector（驾驶轨迹预测）/`**：原始版 VectorNet（PyG 版本），与增强版无关。

---

## 已修复问题（11 个）

### P0-1 ✅ `requirements.txt` 是假的

已替换为只含 5 个实际依赖：`torch`、`numpy`、`matplotlib`、`tqdm`、`flask`。

### P1-1 ✅ 模型没用车道线和周围车的信息

`TrajectoryDataset.__getitem__` 现在会读取 CSV 中 OTHERS 的轨迹，提取最近的周围车辆距离和相对位置，编码到特征维度 5/6/7。

### P1-2 ✅ 类名叫 `SubGraph`，但它不是图网络

已重命名为 `MLPEncoder`，定义在 `model.py`。

### P1-3 ✅ 同一套模型代码写了两遍

已提取共享模块 `model.py`，所有类（`MLPEncoder`、`AttentionPooling`、`MultiModalVectorNet`、`MultiModalLoss`、`LinearPredictor`、`LSTMPredictor`）集中管理。`enhanced_train.py`、`demo_app.py`、`predict.py`、`evaluate.py` 全部 `from model import ...`。

### P1-4 ✅ 训练日志里的 ADE 指标算错了

`train_epoch()` 和 `evaluate()` 现在遍历 K 条轨迹取最小 ADE（true best-of-K），与 `evaluate.py` 逻辑一致。

### P1-5 ✅ `predict.py` 和 `evaluate.py` 各写了一遍相同的工具函数

两个文件的独立定义已删除，改为分别从 `model` 和 `enhanced_train` import。数据读取统一用 `enhanced_train.read_csv_file`。

### P2-1 ✅ 超参数全是硬编码

新增 `config.json`，所有参数集中管理。`enhanced_train.py` 支持 `--config` 和 `--epochs` / `--batch_size` / `--lr` 命令行覆盖。

### P2-2 ✅ DataLoader 没用 `num_workers`

`DataLoader` 调用已加上 `num_workers`（默认 2，可在 `config.json` 调整）。

### P2-3 ✅ 缺少自动化测试

新增 `check.py`：模型创建 → 前向传播 → 输出 shape 断言 → PASSED/FAILED。运行验证通过：198,516 params，输出 `(2,3,60)` + `(2,3)` + `(2,60)`。

### P2-5 ✅ 目录名含中文括号

已确认增强版自身路径 `Vector\yet-another-vectornet\` 不含中文。含中文括号的是原始版目录（`Vector（驾驶轨迹预测）\`），不影响增强版。

### P2-6 ✅ `.gitignore` 不完整

已新增 `.gitignore`，忽略 `*.pth`、`*.pkl`、`*.png`、`__pycache__/`。

---

## 未修复问题（2 个）

### 🔴 P0-2：数据量极小，模型无泛化意义

- **现象**：train/val/test 各只有 14 个 CSV，共 42 个场景。
- **根因**：`mydata/` 是精简版数据
- **修法**：下载完整 Argoverse 数据集（官网 argoverse.org），替换 `mydata/` 内容
- **工作量**：L（需下载约 50GB 原始数据）
- **验收**：`mydata/train/` 至少含 10000 个 CSV

### 🟢 P2-4：三个训练脚本功能重叠

- **现象**：`enhanced_train.py`、`simple_train.py`、`standalone_train.py` 三个都能训练
- **修法**：保留 `enhanced_train.py`，另外两个删掉或移到 `archive/`
- **工作量**：2 分钟
- **验收**：目录下只有一个训练脚本

---

## 当前增强版文件结构

```
Vector/yet-another-vectornet/
├── model.py                  ← ★ 模型定义（新增）
├── config.json               ← ★ 统一配置（新增）
├── check.py                  ← ★ 冒烟测试（新增）
├── .gitignore                ← ★ 版本控制（新增）
├── requirements.txt          ← ★ 已修复
├── enhanced_train.py         ← ★ 训练入口（已重构）
├── predict.py                ← 预测（已重构）
├── evaluate.py               ← 评估（已重构）
├── demo_app.py               ← Web 界面（已重构）
├── simple_train.py           ← 待清理
├── standalone_train.py       ← 待清理
├── mydata/                   ← 数据（待扩充）
├── best_multimodal_model.pth
├── final_multimodal_model.pth
└── templates/index.html
```
