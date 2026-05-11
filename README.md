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

### 3. 装依赖

```bash
pip install torch numpy matplotlib tqdm
```

> 如果装了之后还想玩网页版，再多装一个：
> ```bash
> pip install flask
> ```

### 4. 验证环境

```bash
python check.py
```

看到屏幕最后一行是 **`ALL CHECKS PASSED`** 就说明装好了。

> 如果报 `No module named 'torch'`，回到第 3 步重装。

### 5. 开始训练

```bash
python enhanced_train.py
```

等它跑完（100 轮，大概几分钟）。跑完之后目录下会生成 `best_multimodal_model.pth` 和几张预测效果图。

### 6. 预测新数据

```bash
python predict.py --input mydata/test/893.csv --model best_multimodal_model.pth --output ./result
```

打开 `result/prediction.png` 看结果。

### 7. （可选）打开网页版

```bash
python demo_app.py
```

浏览器打开 `http://localhost:5000`，在画布上用鼠标画轨迹，点 Predict 看预测。

---

## 常见报错

| 报错 | 怎么办 |
|------|--------|
| `No module named 'torch'` | 回到第 3 步，重装依赖 |
| `No module named 'flask'` | `pip install flask` |
| `FileNotFoundError: mydata/...` | 确认当前目录是 `ChronoMotion_easy\Vector\yet-another-vectornet` |
| 训练跑不动/显存不够 | 打开 `config.json`，把 `batch_size` 从 8 改成 2 |

---

## 如果不想训练，只想看效果

项目里已经有两个练好的模型，直接用：

```bash
python predict.py --input mydata/test/893.csv --model best_multimodal_model.pth --output ./result
```
