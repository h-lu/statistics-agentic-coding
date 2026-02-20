# Week 10 作业：准确率陷阱——分类模型与评估

> "所有模型都是错的，但有些是有用的。问题在于：你怎么知道'有用'？"
> — George Box

## 作业概述

本周作业将带你走完一个完整的**分类建模与评估流程**：从数据探索到模型训练，从"只看准确率"到"多维度评估"，从"简单训练-测试"到"Pipeline + 交叉验证防数据泄漏"。

你将使用一份**电商客户流失数据**，回答业务问题：**哪些客户可能流失？模型预测有多可靠？**

---

## 基础层（必做）

### 任务 1：训练你的第一个逻辑回归模型

小北上周学会了回归分析，这周他遇到了一个新问题：老板问的不是"客户会花多少钱"，而是"这个客户会不会流失"。

**你的任务**：
1. 加载 `data/customer_churn.csv`（或使用 `starter_code/week_10.py` 中的示例数据）
2. 探索数据：查看目标变量 `is_churned` 的分布（流失/不流失各占多少比例）
3. 选择至少 3 个特征作为预测变量（建议：`purchase_count`, `avg_spend`, `days_since_last_purchase`）
4. 划分训练集和测试集（test_size=0.3, random_state=42）
5. 训练逻辑回归模型
6. 在测试集上预测并计算准确率

**输入示例**：
```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载数据
df = pd.read_csv("data/customer_churn.csv")

# 特征和目标
X = df[['purchase_count', 'avg_spend', 'days_since_last_purchase']]
y = df['is_churned']

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测并计算准确率
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
```

**输出示例**：
```
准确率: 0.8500
目标变量分布:
不流失: 800 (80%)
流失: 200 (20%)
```

**提交物**：
- 代码文件（如 `week10_basic.py`）
- 输出准确率和目标变量分布

**评分点**：
- [ ] 正确划分训练集和测试集
- [ ] 使用逻辑回归模型（不是线性回归）
- [ ] 准确率计算正确
- [ ] 报告了目标变量分布（为后续思考做铺垫）

---

### 任务 2：解读混淆矩阵与评估指标

小北看到准确率 85%，很高兴。但老潘问了一句："你的数据里 80% 没流失，如果你的模型永远预测'不流失'，准确率也是 80%。你的模型比'永远猜不流失'好多少？"

**你的任务**：
1. 计算并打印混淆矩阵
2. 计算精确率（Precision）、召回率（Recall）、F1 分数
3. 回答：**在你的业务场景中，应该优先关注哪个指标？为什么？**
4. 写一段 2-3 句话的分析，解释混淆矩阵中 TP、TN、FP、 FN 的含义

**输入示例**：
```python
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)

# 评估指标
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1 分数: {f1:.4f}")
```

**输出示例**：
```
混淆矩阵:
[[230   10]
 [ 35   25]]

精确率: 0.7143
召回率: 0.4167
F1 分数: 0.5263
```

**分析示例**（你需要用自己的话写）：

> 注：sklearn 混淆矩阵格式为 `[[TN, FP], [FN, TP]]`，即：
> - 第一行：实际负类（不流失）的预测结果 [正确, 误判]
> - 第二行：实际正类（流失）的预测结果 [漏判, 正确]

```
混淆矩阵解读：
- TN=230: 正确预测不流失的客户
- FP=10: 误判为流失的不流失客户（"误判"）
- FN=35: 漏判的流失客户（"漏判"）
- TP=25: 正确抓到的流失客户

在流失预测场景中，我认为应该优先关注召回率。因为：
1. 漏掉一个流失客户（FN）的代价大于误判一个不流失客户（FP）
2. 我们希望尽可能抓到所有可能流失的客户，即使会误判一些
```

**提交物**：
- 代码
- 混淆矩阵和评估指标的输出
- 一段文字分析（2-3 句话）解释你选择的指标及其原因

**评分点**：
- [ ] 混淆矩阵计算正确
- [ ] 精确率、召回率、F1 计算正确
- [ ] 正确解读了 TP/TN/FP/FN 的含义
- [ ] 给出了合理的指标选择理由（需要结合业务场景）

**常见错误**：
- 混淆精确率和召回率
  - 精确率 = 0.7 表示"预测为流失的客户中，70% 真的流失了"
  - 召回率 = 0.7 表示"所有流失的客户中，70% 被正确预测了"
- 只给出数字，没有解释为什么选择某个指标
- 忽略了类别不平衡对准确率的影响

---

## 进阶层（推荐完成）

### 任务 3：处理类别不平衡——你能比"永远猜多数类"更好吗？

阿码看了你的结果，说："等等，数据里 80% 不流失，20% 流失。如果我的模型永远预测'不流失'，准确率也有 80%。你的模型 85%，真的比'傻瓜基线'好吗？"

**你的任务**：
1. 训练一个"傻瓜基线"模型（DummyClassifier，strategy='most_frequent'）
2. 比较你的模型和傻瓜基线的准确率、精确率、召回率、F1
3. 尝试至少一种处理类别不平衡的方法（选一个）：
   - 方法 A：调整类别权重（LogisticRegression 中设置 `class_weight='balanced'`）
   - 方法 B：调整决策阈值（从 0.5 改为 0.3 或 0.4）
4. 回答：**调整后，哪些指标改善了？哪些变差了？为什么？**

**输入示例**：
```python
from sklearn.dummy import DummyClassifier

# 傻瓜基线
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)

print("傻瓜基线准确率:", accuracy_score(y_test, dummy_pred))

# 方法 A：调整类别权重
model_balanced = LogisticRegression(class_weight='balanced', random_state=42)
model_balanced.fit(X_train, y_train)
y_pred_balanced = model_balanced.predict(X_test)

# 比较指标
print("\n原始模型:")
print(f"召回率: {recall_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")

print("\n平衡权重模型:")
print(f"召回率: {recall_score(y_test, y_pred_balanced):.4f}")
print(f"F1: {f1_score(y_test, y_pred_balanced):.4f}")
```

**输出示例**：
```
傻瓜基线准确率: 0.8000

原始模型:
召回率: 0.4167
F1: 0.5263

平衡权重模型:
召回率: 0.7500
F1: 0.6316
```

**提交物**：
- 代码
- 傻瓜基线和你模型的指标对比
- 一段分析：调整后哪些指标改善/变差，为什么

**评分点**：
- [ ] 正确训练并评估了傻瓜基线
- [ ] 尝试了至少一种处理类别不平衡的方法
- [ ] 正确解读了调整前后的指标变化
- [ ] 理解精确率-召回率的权衡关系（一个改善，另一个可能变差）

---

### 任务 4：ROC 曲线与 AUC——全面评估分类器

小北发现每次都是阈值 0.5，但他很好奇：如果阈值变了会怎样？模型会不会更好或者更差？

**你的任务**：
1. 绘制 ROC 曲线（测试集）
2. 计算 AUC
3. 在一张图上画出：你的模型 vs 傻瓜基线的 ROC 曲线
4. 回答：**AUC 和准确率有什么区别？什么时候 AUC 更有用？**

**输入示例**：
```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 预测概率
y_prob = model.predict_proba(X_test)[:, 1]

# ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

# 画图
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'My Model (AUC = {auc:.4f})')

# 傻瓜基线的 ROC（对角线）
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.5)')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"AUC: {auc:.4f}")
```

**输出示例**：
```
AUC: 0.7850
```

**分析要点**（用自己的话回答）：
- AUC 衡量的是"排序能力"：正类样本是否比负类样本有更高的预测概率
- 准确率依赖于单一阈值（0.5），AUC 不依赖阈值
- 当数据类别不平衡、或者需要排序（如"优先联系哪些客户"）时，AUC 更有用

**提交物**：
- 代码
- ROC 曲线图
- 一段分析（2-3 句话）解释 AUC 和准确率的区别

**评分点**：
- [ ] 正确计算了 ROC 曲线和 AUC
- [ ] 正确绘制了 ROC 曲线（包括坐标轴标签、图例）
- [ ] 理解 AUC 和准确率的区别
- [ ] 能解释 AUC 的含义（排序能力，不依赖阈值）

---

## 挑战层（可选）

### 任务 5：Pipeline + 交叉验证——防止数据泄漏

老潘看了你的代码，眉头皱了起来："你在哪里做的缺失值填充？在划分训练集和测试集之前？那你就有数据泄漏了。"

数据泄漏是分类评估中最隐蔽的坑：测试集的信息悄悄进入了训练集，导致评估指标虚高。

**你的任务**：
1. 识别代码中的数据泄漏风险（如果有）
2. 使用 Pipeline 把预处理和模型绑在一起
3. 用交叉验证评估模型（scoring='roc_auc'）
4. 对比：有 Pipeline vs 没有 Pipeline 的交叉验证 AUC
5. 写一段分析：**为什么 Pipeline 能防止数据泄漏？**

**输入示例**：
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# 创建 Pipeline（防止数据泄漏）
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # 填充缺失值
    ('scaler', StandardScaler()),  # 标准化
    ('model', LogisticRegression())  # 模型
])

# 交叉验证（Pipeline 确保每一折的预处理独立）
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')

print(f"交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"各折 AUC: {cv_scores}")
```

**输出示例**：
```
交叉验证 AUC: 0.7820 (+/- 0.0234)
各折 AUC: [0.7650 0.8012 0.7734 0.7890 0.7814]
```

**分析要点**：
- Pipeline 确保预处理（如均值计算）只在训练集上进行
- 交叉验证时，每一折的预处理都是独立的，验证集信息不会泄漏
- 没有 Pipeline 的交叉验证，是在"作弊"——测试集信息参与了预处理

**提交物**：
- 代码
- 交叉验证 AUC 输出
- 一段分析（3-4 句话）解释 Pipeline 如何防止数据泄漏

**评分点**：
- [ ] 正确使用了 Pipeline（包含至少一个预处理步骤）
- [ ] 使用 cross_val_score 进行交叉验证
- [ ] 正确解读了交叉验证结果（均值和标准差）
- [ ] 清晰解释了数据泄漏的概念和 Pipeline 的防御作用

**常见错误**：
- 在 cross_val_score 之前做预处理（这是数据泄漏！）
- 混淆了 `fit()` 和 `transform()` 的作用
- 交叉验证时没有用 Pipeline，导致每一折的预处理不独立

---

### 任务 6（加分）：写一份"带评估指标选择理由"的分类报告

把本周学到的东西整合成一份完整的分类评估报告，就像你在 StatLab 中做的那样。

**你的任务**：
1. 选择一个阈值（可以是 0.5，也可以是你认为更合适的阈值）
2. 在报告中解释：**为什么选择这个阈值？优先关注哪些指标？**
3. 包含以下内容：
   - 评估指标表（准确率、精确率、召回率、F1、AUC）
   - 混淆矩阵（带 TP/TN/FP/FN 标注）
   - ROC 曲线图
   - 指标选择理由（2-3 句话）
   - 模型局限性（2-3 句话）

**报告模板**（Markdown 格式）：
```markdown
## 客户流失预测模型评估报告

### 数据概览
- 样本量：1000
- 流失率：20%（类别不平衡）
- 特征：购买次数、平均消费金额、距上次购买天数

### 评估指标
| 指标 | 值 |
|------|-----|
| 准确率 | 0.8500 |
| 精确率 | 0.7143 |
| 召回率 | 0.4167 |
| F1 分数 | 0.5263 |
| AUC | 0.7850 |

### 混淆矩阵
| | 预测不流失 | 预测流失 |
|---|-----------|---------|
| 实际不流失 | 230 (TN) | 10 (FP) |
| 实际流失 | 35 (FN) | 25 (TP) |

### 指标选择理由
本数据存在类别不平衡（流失率 20%），准确率可能误导。在流失预测场景中，我们优先关注召回率（抓到所有可能流失的客户），同时兼顾精确率（减少误判）。因此，F1 分数是综合评估指标。

### 模型局限性
1. 模型召回率较低（41.67%），可能漏掉部分流失客户
2. 未考虑非线性关系和特征交互
3. 需要在新数据上验证稳定性
```

**提交物**：
- 一份 Markdown 格式的分类报告（`classification_report.md`）
- 生成报告的代码

**评分点**：
- [ ] 报告结构完整（包含所有必要部分）
- [ ] 指标选择理由清晰、合理
- [ ] 正确解读了混淆矩阵和评估指标
- [ ] 诚实地列出了模型的局限性

---

## AI 协作练习（可选）

下面这段分析是某个 AI 工具生成的分类模型结论。请审查它：

> "基于逻辑回归模型，我们准确预测了 85% 的客户流失情况。模型表现优秀，可以投入生产使用。特征 importance 显示购买次数是最重要的预测因素。"

**审查清单**：
- [ ] 它有没有检查类别不平衡？（准确率 85% 可能没有意义）
- [ ] 它有没有说明选择了哪个评估指标？为什么？
- [ ] 它有没有报告混淆矩阵（TP/FP/FN/TN）？
- [ ] 它有没有检查数据泄漏？
- [ ] 它有没有说明模型的局限性？

**你的修订版**（2-3 句话）：
```
（用你自己的话重写，修正上述问题）
例如："模型准确率 85%，但由于数据存在类别不平衡（80% 不流失），我们更关注召回率（42%）和 F1（0.53）。混淆矩阵显示模型漏判了 35 个流失客户（FN），误判了 10 个不流失客户（FP）。AUC 为 0.79，表明模型有一定的排序能力。建议后续调整阈值或使用类别权重来提升召回率。"
```

**提交物**：
- 审查清单（勾选哪些问题存在）
- 你的修订版（2-3 句话）

---

## 提交检查清单

在提交作业前，请确认：

- [ ] 代码可以运行（或注明哪些部分是伪代码）
- [ ] 输出结果包含在提交中（或截图）
- [ ] 分析部分用你自己的话写（不是复制粘贴）
- [ ] 如果遇到困难，参考了 `starter_code/solution.py`，请说明参考了哪些部分

---

## 提示与帮助

如果你在完成作业时遇到困难：
1. 回顾 CHAPTER.md 中的示例代码
2. 参考本周的 StatLab 示例（`examples/10_statlab_classification.py`）
3. 查阅 scikit-learn 官方文档（不要害怕查文档！）

**记住**：作业的目的是巩固理解，不是完美复制代码。即使遇到困难，也要尝试用自己的话解释问题和思路。

---

祝你本周学习愉快！记住老潘的话：**"没有 Pipeline 的评估不是评估，是自欺欺人。"**
