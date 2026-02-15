# Week 08 作业：区间估计与重采样

> "没有置信区间的报告不是科学，是赌博。"
> — 老潘

---

## 作业说明

本周作业分为三层：**基础作业**（必须完成）、**进阶作业**（选做）、**挑战作业**（加分）。每层都有明确的输入输出格式、评分点和常见错误提示。

**提交格式**：将所有代码和结果整理到 `week08_assignment.ipynb` 或 `week08_assignment.py` 中，并在 `week08_report.md` 中回答文字题。

**数据集**：使用 `seaborn.load_dataset("penguins")`

---

## 基础作业（必须完成）

### 1. 计算置信区间

**任务**：手动计算 Adelie 企鹅喙长（`bill_length_mm`）的 95% 置信区间，使用两种方法：t 分布公式法和 Bootstrap Percentile 法。

**输入**：
- 数据：`penguins[penguins["species"] == "Adelie"]["bill_length_mm"].dropna()`
- 置信水平：95%

**输出格式**：
```
均值: XX.XX
标准误: X.XXX

方法 1 (t 分布): 95% CI [XX.XX, XX.XX]
方法 2 (Percentile Bootstrap): 95% CI [XX.XX, XX.XX]
```

**提示**：
- t 分布法使用 `scipy.stats.t.interval()`
- Bootstrap 使用 `np.random.choice(..., replace=True)` 重采样 10000 次
- 两种方法的 CI 应该很接近（因为数据近似正态）

**常见错误**：
- 忘记处理缺失值（`dropna()`）
- Bootstrap 时忘记设置 `replace=True`
- 混淆标准差（SD）和标准误（SE）

---

### 2. Bootstrap 均值估计

**任务**：对 Gentoo 企鹅的喙长执行 Bootstrap 分析，估计均值及其 95% CI，并可视化 Bootstrap 分布。

**输入**：
- 数据：`penguins[penguins["species"] == "Gentoo"]["bill_length_mm"].dropna()`
- 重采样次数：10000

**输出**：
1. Bootstrap 均值、标准误、95% CI（Percentile 和 BCa 两种方法）
2. Bootstrap 均值的直方图，标注原始均值和 95% CI 边界

**代码框架**（仅作参考，你需要补全）：
```python
from scipy.stats import bootstrap
import numpy as np

def mean_func(x):
    return np.mean(x)

# 使用 scipy.stats.bootstrap 计算 BCa CI
res = bootstrap((data,), mean_func, ...)
```

**可视化要求**：
- x 轴：Bootstrap 均值
- y 轴：频数
- 用红色竖线标注 95% CI 的上下界
- 用虚线标注原始均值

**常见错误**：
- Bootstrap 次数太少（< 1000），导致 CI 不稳定
- 可视化时不设置随机种子，导致每次结果不同
- 混淆 Percentile 和 BCa 方法的输出

---

### 3. CI 解释练习

**任务**：判断以下关于置信区间的解释是否正确。如果错误，说明为什么，并给出正确解释。

**陈述 A**："95% CI [38.0, 39.5] 意味着真实均值有 95% 的概率落在 38.0 到 39.5 之间。"

**陈述 B**："如果我们从总体中重复抽样 100 次，每次计算 95% CI，大约有 95 个区间会包含真实均值。"

**陈述 C**："95% CI 不包含 0，说明 p < 0.05，差异显著。"

**输出格式**：
```
陈述 A: [正确/错误]
理由: ...
正确解释: ...

陈述 B: [正确/错误]
理由: ...

陈述 C: [正确/错误]
理由: ...
```

**提示**：
- 回顾 WEEK 08 第 2 节关于 CI 含义的讨论
- 区分"参数是固定的，区间是随机的"与"参数是随机的"

---

### 4. 置换检验实践

**任务**：用置换检验检验 Adelie 和 Chinstrap 企鹅的喙长是否有显著差异。

**输入**：
- A 组：`penguins[penguins["species"] == "Adelie"]["bill_length_mm"].dropna()`
- B 组：`penguins[penguins["species"] == "Chinstrap"]["bill_length_mm"].dropna()`
- 置换次数：10000

**输出格式**：
```
观测差异（B - A）: X.XXX
置换检验 p 值（双尾）: X.XXXX
结论: [显著/不显著]
```

**提示**：
- 合并两组数据后用 `np.random.permutation()` 打乱标签
- 双尾 p 值：计算 `|置换差异| >= |观测差异|` 的比例
- 可以同时用 `scipy.stats.ttest_ind()` 验证结果

**常见错误**：
- 忘记取绝对值计算双尾 p 值
- 置换次数太少导致 p 值不稳定
- 混淆单尾和双尾检验

---

## 进阶作业（选做）

### 5. BCa vs Percentile：比较 Bootstrap CI 方法

**任务**：用偏态分布（指数分布）数据比较 Percentile Bootstrap 和 BCa Bootstrap 的差异。

**输入**：
```python
np.random.seed(42)
skewed_data = np.random.exponential(scale=2.0, size=100)
```

**输出**：
1. 用两种方法计算中位数的 95% CI
2. 可视化 Bootstrap 分布，标注两种方法的 CI
3. 解释：为什么对于偏态数据的中位数，BCa 和 Percentile 会不同？

**提示**：
- 中位数的 Bootstrap 分布通常也是偏态的
- BCa 会校正偏态和加速，因此 CI 可能不对称
- 可以用 `scipy.stats.bootstrap(..., method='percentile')` 和 `method='BCa'`

**常见错误**：
- 使用均值而不是中位数（偏态不明显）
- 只看数字不看分布（可视化很重要）

---

### 6. Bootstrap 失效情况识别

**任务**：识别并解释 Bootstrap 会失效的场景。给出一个具体例子（模拟数据），展示 Bootstrap CI 不准确的情况。

**提示场景**：
- **样本量太小**（n < 20）
- **时间序列数据**（重采样破坏时间结构）
- **极值敏感统计量**（如最大值、最小值）

**输出格式**：
```
场景: [描述]
示例代码: [简短代码]
Bootstrap CI: [结果]
为什么失效: [解释]
更好的方法: [建议]
```

**示例**（时间序列）：
```python
# 生成有趋势的时间序列
t = np.arange(100)
ts = 0.5 * t + np.random.normal(0, 1, 100)

# Bootstrap 会破坏趋势，导致 CI 不准确
```

---

## 挑战作业（加分）

### 7. 完整分析报告：从数据到 CI 报告

**任务**：写一个完整的数据分析报告，比较三种企鹅（Adelie、Gentoo、Chinstrap）在至少两个数值型特征上的差异。报告必须包含：

**要求**：
1. 描述统计表（均值、SD、样本量）
2. 可视化（箱线图或小提琴图）
3. 组间差异的点估计 + 95% CI（Bootstrap）
4. 置换检验 p 值
5. 效应量（Cohen's d）
6. 结论（用"带不确定性量化"的语言）

**输出**：一个完整的 Markdown 报告（`week08_challenge_report.md`）

**评分标准**：
- CI 的计算和解释是否正确（30%）
- 可视化是否清晰、信息丰富（20%）
- 结论是否量化不确定性（不只是"显著/不显著"）（20%）
- 代码是否可复现（固定随机种子、清晰注释）（20%）
- 报告写作是否专业、易读（10%）

**提示**：
- 参考本周 StatLab 进度中的报告格式
- 可以复用第 4 题的置换检验代码
- Cohen's d 公式：`(mean1 - mean2) / pooled_sd`

---

## AI 协作练习（可选）

### 题目 8：审查 AI 生成的统计结论

假设你让 AI 分析 Adelie 和 Gentoo 企鹅的喙长差异，它给你生成了下面这份结论。请审查它，找出其中的问题。

**AI 生成的结论**：

> "分析结果显示，Adelie 和 Gentoo 企鹅的喙长存在显著差异（p = 0.0001）。Adelie 的平均喙长为 38.8 mm，Gentoo 的平均喙长为 47.5 mm。这表明 Gentoo 企鹅的喙明显更长，建议在生态研究中重点关注 Gentoo 企鹅的喙特征。"

---

**审查清单**：

- [ ] **置信区间**：AI 报告了置信区间吗？
  - AI 只报告了点估计（均值），没有报告 CI
  - 没有量化"差异有多确定"
  - 风险：读者不知道这个差异有多稳定

- [ ] **效应量**：AI 报告了效应量吗？
  - AI 只报告了 p 值，没有标准化效应量
  - 差异 8.7 mm 在生态学上是否有意义？
  - 应该报告 Cohen's d 或其他效应量指标

- [ ] **前提假设检查**：AI 检查了前提假设吗？
  - 报告中没有提到正态性、方差齐性检查
  - 如果假设不满足，t 检验的 p 值可能不准确

- [ ] **样本量与缺失值**：AI 说明了样本量和数据质量吗？
  - 没有说明样本量（N 是多少？）
  - 没有说明缺失值处理（是否 dropna？）

- [ ] **因果推断**：AI 把"相关"说成"因果"了吗？
  - "建议重点关注 Gentoo 企鹅" —— 这是因果建议
  - 但观察性研究不能证明因果关系
  - 可能存在混淆变量（如栖息地、食物来源等）

**你的任务**：

1. 写一份审查报告（200-300 字），列出你发现的问题
2. 给出修订建议（应该补充什么信息、如何改写结论）
3. （可选）用 Python 计算缺失的 CI 和效应量，验证你的结论

**提交物**：
- 审查报告（Markdown 格式）
- 你发现的 3-5 个问题列表
- 修订后的报告（可选，如果做了代码验证）

**提示**：
- 使用本周学到的知识：点估计 vs 区间估计、CI 的正确解释、Bootstrap
- 参考正文第 1-2 节关于 CI 含义的讨论
- 参考 StatLab 进度中的报告格式
- 如果你遇到困难，可以参考 `starter_code/solution.py`

**评分标准**（如果提交）：
- 识别问题的准确性（40%）
- 审查报告的说服力（30%）
- 修订建议的合理性（30%）

---

## 提交清单

**基础作业**（必须完成）：
- [ ] 题目 1：计算置信区间（代码 + 输出 + 解释）
- [ ] 题目 2：Bootstrap 均值估计（代码 + 可视化 + 结果）
- [ ] 题目 3：CI 解释练习（文字回答）
- [ ] 题目 4：置换检验实践（代码 + 结果 + 结论）

**进阶作业**（选做）：
- [ ] 题目 5：BCa vs Percentile（代码 + 可视化 + 解释）
- [ ] 题目 6：Bootstrap 失效情况识别（示例代码 + 解释）

**挑战作业**（加分）：
- [ ] 题目 7：完整分析报告（Markdown 报告 + 代码）

**AI 协作练习**（可选）：
- [ ] 题目 8：AI 报告审查报告

---

## 参考资源

如果你遇到困难，可以参考：
- `starter_code/solution.py`（示例实现）
- 本周 CHAPTER.md 的第 1-5 节
- `scipy.stats.bootstrap` 文档
- `scipy.stats.t.interval` 文档

---

祝你好运！记住：**点估计只是一个猜测，区间估计才是诚实的科学。**
