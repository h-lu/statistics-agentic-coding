# Week 10：从准确率到 AUC——分类器的评估陷阱

> "Prediction is very difficult, especially about the future."
> — Niels Bohr

2025 到 2026 年，越来越多的 AI 工具能"自动分类"：AutoML 平台、无代码分类器、甚至 ChatGPT 都能帮你跑一个"预测客户是否流失"的模型。但这些工具最危险的默认行为是——只给你一个"准确率 85%"的数字，却不会告诉你：这个准确率在类别不平衡场景下可能是误导的、你的评估流程是否存在数据泄漏、选择的阈值是否符合业务目标。

---

## 前情提要

上周（Week 09），小北学会了用回归分析理解变量之间的关系。他兴冲冲地对老潘说："**我现在知道面积每增加 1 平米，房价涨多少了！而且我还做了残差诊断，检查了异常点！**"

老潘看了一眼他的报告，问了一个新问题："**那如果你的目标不是'预测房价'，而是'预测客户会不会流失'（Yes/No）呢？**"

小北愣住了："呃……我可以用线性回归，把结果四舍五入成 0 或 1？"

阿码举手："**或者我直接看准确率，85% 就是好模型对吧？**"

老潘摇摇头："**分类问题需要一套全新的评估思维**。准确率在类别不平衡时会撒谎，你需要混淆矩阵、精确率、召回率、ROC-AUC。更重要的是，你需要用交叉验证和 Pipeline 避免数据泄漏——这是分类评估中最常见的工程陷阱。"

"还有，"老潘补充，"**本周你会用到很多之前学过的工具**：Week 06 的**第一类/第二类错误**会变成混淆矩阵的假阳性/假阴性；Week 08 的**置信区间**会用到逻辑回归系数上；Week 03 的**特征缩放**必须在 Pipeline 里做，否则会泄漏测试集信息。"

"所以，"小北说，"**这不只是换个模型，而是换了一套评估哲学？**"

"没错，"老潘笑了，"**分类不是'算个准确率'，而是'在假阳性和假阴性之间做权衡'**。"

这正是本周要解决的问题——**从连续预测到分类评估**，你不再只看 R²，而是学习混淆矩阵、ROC-AUC，并用 Pipeline 防止数据泄漏。

---

## 学习目标

完成本周学习后，你将能够：
1. 理解逻辑回归的动机（为什么不能用线性回归做分类），掌握 Sigmoid 函数和概率建模思路
2. 正确解释逻辑回归系数的含义（优势比 odds ratio），并理解决策阈值的选择
3. 从准确率升级到混淆矩阵，理解精确率、召回率、F1 的业务含义
4. 掌握 ROC-AUC 的直觉，理解阈值无关评估的重要性
5. 正确使用 K-fold 交叉验证估计模型泛化性能
6. 识别并防御数据泄漏（Pipeline + ColumnTransformer 的正确模式）
7. 在 StatLab 报告中添加分类评估章节，包含混淆矩阵、ROC-AUC、交叉验证结果
8. 审查 AI 生成的分类报告，识别缺少阈值讨论、忽略数据泄漏等问题

---

<!--
贯穿案例：客户流失预测——从"算个准确率"到"完整评估流水线"

本周贯穿案例是一个电信公司客户流失预测场景：你想预测哪些客户会流失（Churn = Yes/No），用逻辑回归建模，并正确评估模型性能。

- 第 1 节：从流失率（二分类目标）出发，意识到不能用线性回归 → 案例从"尝试线性回归"变成"理解需要 Sigmoid 函数"
- 第 2 节：拟合逻辑回归，解释系数 → 案例从"跑出模型"变成"理解'合同期每增加 1 月，流失概率降低多少'（优势比解释）"
- 第 3.1 节：被85%准确率欺骗 → 案例从"模型很好"变成"发现准确率悖论"
- 第 3.2 节：用混淆矩阵看清真相 → 案例从"只看准确率"变成"看到假阴性/假阳性的业务成本"
- 第 4 节：用 ROC-AUC 比较 → 案例从"只看准确率"变成"用 AUC 比较'不同特征组合'，选择阈值平衡业务目标"
- 第 5 节：发现数据泄漏 → 案例从"全局 StandardScaler"变成"用 Pipeline + ColumnTransformer 重构，防止泄漏"
- 第 6 节：完整流水线 → 案例从"散乱代码"变成"可复现的流水线（预处理 → 逻辑回归 → 交叉验证 → 评估）"

最终成果：读者完成一个完整的分类评估流水线，产出：
- 1 个逻辑回归模型（statsmodels 或 scikit-learn）
- 1 张系数表（含优势比解释）
- 1 个混淆矩阵（含精确率、召回率、F1）
- 1 条 ROC 曲线（含 AUC 值）
- 1 个 Pipeline + ColumnTransformer（防止数据泄漏）
- 1 份 K-fold 交叉验证报告（均值 ± 标准差）
- 1 份与基线模型的对比（多数类分类器）
- 1 份分类评估报告（含阈值选择、局限性讨论）
- 1 份 AI 分类报告的审查清单（标注缺少诊断、数据泄漏等问题）

认知负荷预算：
- 本周新概念（5 个，预算上限 5 个）：
  1. 逻辑回归 - 应用层次
  2. 混淆矩阵 - 应用层次
  3. ROC-AUC - 理解层次
  4. 交叉验证 - 应用层次
  5. 数据泄漏（Pipeline）- 分析层次
- 结论：✅ 在预算内（5 个）

回顾桥设计（至少 2 个，来自 Week 01-09）：
- [线性回归]（week_09）：在第 1 节，用"线性回归预测连续值"对比"逻辑回归预测概率"，引出为什么不能直接用线性回归做分类
- [第一类/第二类错误]（week_06）：在第 3.2 节，用"假阳性/假阴性"连接混淆矩阵的 False Positive / False Negative
- [置信区间]（week_08）：在第 2 节，用"逻辑回归系数的 95% CI"连接 Week 08 的区间估计方法
- [假设检验]（week_06/07）：在第 4 节，用"检验模型是否优于随机猜测"连接 p 值概念（AUC = 0.5 的显著性检验）
- [异常值处理/特征缩放]（week_03）：在第 5 节，用"预处理中的特征缩放必须在折内执行"连接 Week 03 的标准化知识

AI 小专栏规划：

AI 小专栏 #1（放在第 2 节之后）：
- 主题：AI 时代的分类评估陷阱——准确率不够用了
- 连接点：与第 2 节"逻辑回归实战"和第 3 节"准确率陷阱"呼应，讨论 AI 工具默认输出准确率，但在类别不平衡场景（如欺诈检测、罕见病诊断）中，准确率是误导性指标

AI 小专栏 #2（放在第 4-5 节之间）：
- 主题：数据泄漏——AI 时代最常见的统计错误
- 连接点：与第 5 节"交叉验证与数据泄漏"直接呼应，讨论即使在使用 AI/AutoML 时，数据泄漏（不当使用未来信息）依然是最常见的评估偏差来源

角色出场规划：
- 小北（第 2、3.1、5 节）：
  - 把概率当成确定性（0.8 就是"会流失"）
  - 被85%准确率欺骗，以为模型很好
  - 全局 StandardScaler 导致数据泄漏
- 阿码（第 1、4 节）：
  - 追问"为什么不能用线性回归加个阈值？"
  - 好奇"AUC = 0.75 到底是什么意思？"
- 老潘（第 2、3.2、5、6 节）：
  - 强调"逻辑回归系数是优势比，不是概率差"
  - "混淆矩阵比准确率更重要"
  - "Pipeline 不是工程炫技，是避免数据泄漏的唯一方法"
  - "基线对比：先打败'总是预测多数类'，再谈复杂模型"

StatLab 本周推进：
- 上周状态：report.md 已有回归分析章节（系数解释、残差诊断、异常点分析）
- 本周改进：在 report.md 中添加"分类与评估"章节，包含：
  - 研究问题转向：哪些因素影响二分类目标？（如流失/购买）
  - 逻辑回归模型：系数表 + 优势比解释
  - 评估指标：混淆矩阵 + 精确率/召回率/F1 + ROC-AUC
  - 交叉验证：K-fold CV 结果（均值 ± 标准差）
  - 工程实践：Pipeline + ColumnTransformer 代码
  - 基线对比：与多数类分类器对比
  - 局限性：类别不平衡、因果不能直接推断
- 涉及的本周概念：逻辑回归、混淆矩阵、ROC-AUC、交叉验证、数据泄漏
- 建议示例文件：examples/10_statlab_classification.py（生成分类评估报告与 ROC 曲线）
-->

## 1. 为什么不能用线性回归做分类？

小北拿到了一份客户流失数据，目标是预测哪些客户会取消服务（Churn = Yes/No）。他灵机一动："**我用 Week 09 学过的线性回归，把 Yes 编码为 1、No 编码为 0，不就能预测了吗？**"

说干就干。他拟合了一个线性回归模型，预测"流失概率"：

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 假设 df 是流失数据
df['churn_num'] = (df['Churn'] == 'Yes').astype(int)  # Yes=1, No=0

X = df[['tenure_months', 'monthly_charges', 'total_charges']]
y = df['churn_num']

# 拟合线性回归
model = LinearRegression()
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 看看预测值范围
print(f"预测值范围: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
```

输出让他傻眼了：

```
预测值范围: [-0.23, 1.37]
```

**-0.23 和 1.37 是什么意思？概率可以是负数吗？可以超过 1 吗？**

### 线性回归的致命缺陷：预测值超出 [0, 1]

阿码看着这个结果，皱着眉头问："**为什么不能直接用线性回归加个阈值？比如预测值 > 0.5 就判为 Yes，否则判为 No？**"

"你可以这么做，"老潘说，"**但线性回归在分类问题上有三个致命缺陷**：

1. **预测值无界**：线性回归的预测值可以是任意实数（-∞ 到 +∞），但概率必须在 [0, 1] 之间
2. **误差项假设不成立**：线性回归假设残差服从正态分布，但二分类数据的残差显然不是正态的（只有 0 和 1 两种可能）
3. **同方差假设违反**：在 x = 0.5 处（最难分类的区域），方差最大；在 x 接近 0 或 1 时，方差最小。线性回归假设"方差处处相等"，这显然不对"

小北似懂非懂："所以线性回归'硬套'在分类上，虽然能跑出结果，但统计推断（p 值、置信区间）全都不可信？"

"对，"老潘点头，"**你需要的不是一条'直线'，而是一条'S 型曲线'——把线性预测压缩到 [0, 1] 之间**。"

这就是**逻辑回归**（logistic regression）要解决的问题。

### 从线性回归到逻辑回归：Sigmoid 函数

逻辑回归的核心思想是：**先用线性回归算出一个"得分"，再用 Sigmoid 函数把它映射到概率空间**。

数学上：

**线性得分**：z = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ

**Sigmoid 映射**：P(y=1|x) = 1 / (1 + e⁻ᶻ)

这个函数长什么样？你可以画出来：

```python
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 生成 z 从 -6 到 6
z = np.linspace(-6, 6, 100)
p = sigmoid(z)

# 画图
plt.figure(figsize=(10, 6))
plt.plot(z, p, linewidth=3, label='Sigmoid(z)')
plt.axhline(y=0.5, color='red', linestyle='--', label='阈值 0.5')
plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('线性得分 z = β₀ + β₁x')
plt.ylabel('概率 P(y=1|x)')
plt.title('Sigmoid 函数：把任意实数映射到 [0, 1]')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

你会看到一条**S 型曲线**：
- 当 z → -∞ 时，P → 0
- 当 z → +∞ 时，P → 1
- 当 z = 0 时，P = 0.5（最不确定的点）

阿码盯着图，突然明白了："**所以逻辑回归不是画直线，而是画一条'概率曲线'？**"

"对！"老潘说，"**线性回归回答'线性得分是多少'，逻辑回归回答'概率是多少'**。对于分类问题，你更需要的是后者。"

### 和 Week 09 的连接：预测连续值 vs 预测概率

Week 09 你用线性回归预测**房价**（连续值）。那时候，你的目标是"预测值和真实值越近越好"。

但现在你的目标是**分类**（Yes/No）。你关心的不是"预测值和 0/1 的距离"，而是"**概率是否接近 0 或 1**"。

"所以，"小北若有所思，"线性回归的损失函数是'最小化残差平方和'，逻辑回归的损失函数应该不一样？"

"对！逻辑回归用的是**对数损失**（log loss），"老潘说，"**它惩罚的是'错误的概率估计'，而不是'错误的分类'**。比如，真实标签是 1，你预测概率是 0.9，损失很小；你预测概率是 0.1，损失巨大。"

这正好引出下一节：如何把概率翻译成决策？

---

## 2. 逻辑回归实战：从概率到决策

小北用 scikit-learn 拟合了第一个逻辑回归模型：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 准备数据
X = df[['tenure_months', 'monthly_charges']]
y = df['churn_num']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 拟合逻辑回归
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# 预测概率（注意：predict_proba 返回两列，第 0 列是 P(y=0)，第 1 列是 P(y=1)）
y_proba = log_reg.predict_proba(X_test)[:, 1]  # 只取 P(y=1)

# 看看前几个预测概率
print("前 5 个客户的流失概率:")
print(y_proba[:5])
```

输出是：

```
前 5 个客户的流失概率:
[0.12, 0.45, 0.67, 0.03, 0.89]
```

小北兴奋地说："**第 5 个客户流失概率是 0.89，肯定要走了！**"

老潘立刻制止："**等等，0.89 是什么意思？**"

"呃……89% 会流失？"

"这个解释有问题，"老潘说，"**逻辑回归输出的是'模型估计的概率'，不是'真实概率'**。如果训练数据有偏差（比如只收集了投诉客户的数据），这个 0.89 可能高得离谱。"

小北愣住了："那……怎么解释才对？"

### 如何正确解释逻辑回归的输出？

正确解释应该是："**在模型使用的特征和训练数据的条件下，这个客户被估计为高流失风险**。"

但更重要的是——**逻辑回归的系数**。小北决定看看模型学到了什么：

```python
# 查看系数
coefs = pd.DataFrame({
    '特征': X.columns,
    '系数': log_reg.coef_[0]
})
print(coefs)
```

输出是：

```
          特征       系数
0  tenure_months -0.055
1  monthly_charges 0.023
```

小北皱着眉头："**这些系数是什么意思？-0.055 代表什么？**"

### 逻辑回归系数的正确解释：优势比

"线性回归的系数是'y 的变化量'，"老潘说，"**但逻辑回归的系数是'对数优势比'（log-odds ratio）的变化量**。"

什么是**优势**（odds）？

**优势 = P(y=1) / P(y=0)**

比如，如果流失概率是 0.8，优势 = 0.8 / 0.2 = 4（流失的概率是"不流失"的 4 倍）。

逻辑回归的模型是：

**log(优势) = β₀ + β₁x₁ + β₂x₂ + ...**

所以：
- **β** = 对数优势比的变化
- **exp(β)** = 优势比（Odds Ratio, OR）

小北试着解释一下：

```python
# 计算优势比
coefs['优势比 (OR)'] = np.exp(coefs['系数'])
print(coefs)
```

输出是：

```
          特征       系数  优势比 (OR)
0  tenure_months -0.055  0.946
1  monthly_charges 0.023  1.023
```

现在小北能正确解释了：
- **tenure_months（合同期）**：OR = 0.946，说明合同期每增加 1 个月，流失的优势降低到原来的 94.6%（降低 5.4%）
- **monthly_charges（月费）**：OR = 1.023，说明月费每增加 1 元，流失的优势增加到原来的 1.023 倍（增加 2.3%）

老潘点头："**这才是正确的解释语言**。记住：逻辑回归的系数不是'概率的变化'，而是'优势的变化'。"

### 和 Week 08 的连接：系数的置信区间

Week 08 你学过**置信区间**，它能告诉你"估计有多确定"。逻辑回归的系数也需要置信区间，但 scikit-learn 默认不提供。

你可以用 statsmodels 来获得更详细的输出：

```python
import statsmodels.api as sm

# 添加截距项（statsmodels 不会自动添加）
X_sm = sm.add_constant(X_train)

# 拟合逻辑回归
logit_model = sm.Logit(y_train, X_sm).fit(disp=0)  # disp=0 不输出迭代过程

# 打印详细报告
print(logit_model.summary())
```

你会看到一张包含系数、标准误、z 值、p 值、95% 置信区间的表。

假设 tenure_months 的 95% CI 是 [-0.068, -0.042]，你的解释应该是：

**"我们有 95% 的把握认为，合同期每增加 1 个月，流失的对数优势下降 0.068 到 0.042 之间。"**

或者用优势比来说（exp(95% CI) = [0.934, 0.959]）：

**"我们有 95% 的把握认为，合同期每增加 1 个月，流失优势降低到原来的 93.4% 到 95.9% 之间。"**

### 小北的错误：把概率当成确定性

小北看到某个客户流失概率是 0.89，立刻说："**这个人要走了！**"

老潘摇摇头："**概率不是确定性**。0.89 意味着'高风险'，但不等于'100% 会流失'。"

更危险的是，小北直接用默认阈值 0.5 做决策：

```python
# 默认阈值 0.5
y_pred = (y_proba >= 0.5).astype(int)

# 看看预测结果
from collections import Counter
print(Counter(y_pred))
```

输出是：

```
Counter({0: 180, 1: 20})
```

小北皱着眉头："**怎么只预测了 20 个流失？实际流失有 50 个啊！**"

这正是下一节要解决的问题：**只看准确率是不够的，85% 可能是幻觉。**

---

> **AI 时代小专栏：分类评估陷阱——准确率不够用了**
>
> 2025 年到 2026 年，随着 AutoML 和 AI 分类工具的普及，一个危险的误区正在扩散：**只看"准确率"，完全忽略精确率、召回率和 F1**。
>
> 2025 年 12 月的一篇营销分析文章直接指出：**当准确率成为唯一目标时，模型可能在最关键的决策上完全失败**。在营销模型中，如果只有 5% 的客户会响应，一个"总是预测不响应"的模型准确率高达 95%，但实际上毫无价值。
>
> 这正是**准确率悖论**（accuracy paradox）：在类别不平衡场景（欺诈检测、罕见病诊断、流失预测）中，高准确率可能来自"总是预测多数类"，模型在少数类上的表现极差（召回率接近 0），业务上最重要的客户（如流失者、欺诈者）被完全忽略。
>
> 研究表明，**F1 分数是跨数据集和测试条件下最稳定、最平衡的评估指标**，优于单独使用准确率、精确率或召回率。另一个值得关注的指标是**平衡准确率**（Balanced Accuracy），它计算各类别召回率的平均值，能更好地反映不平衡数据下的真实性能。在不平衡数据场景下，PR-AUC（精确率-召回率曲线下面积）也比 ROC-AUC 更现实，因为它聚焦于少数类的表现。
>
> 所以你刚学的逻辑回归和概率输出，在 AI 时代反而更重要了——AI 可以帮你拟合模型、输出概率，但**指标选择和阈值调整的责任由你承担**。
>
> 参考（访问日期：2026-02-12）：
> - [Why Many Marketing Models Fail: The Problem of Imbalanced Data (Marketing Data Science, Dec 2025)](https://blog.marketingdatascience.ai/why-many-marketing-models-fail-the-problem-of-imbalanced-data-ceb8123b80b3)
> - [Addressing Accuracy Paradox Using Enhanced Weighted Performance Metric (ResearchGate, Sep 2025)](https://www.researchgate.net/publication/340894484_Addressing_Accuracy_Paradox_Using_Enhanced_Weighted_Performance_Metric_in_Machine_Learning)
> - [Balanced and Imbalanced Datasets in Machine Learning (Encord, Nov 2022)](https://encord.com/blog/an-introduction-to-balanced-and-imbalanced-datasets-in-machine-learning/)
> - [When F1 Isn't Quite Right: The Case For Balanced Accuracy (LinkedIn, Apr 2025)](https://www.linkedin.com/pulse/when-f1-isnt-quite-right-case-balanced-accuracy-amy-humke-fmnpc)

---

## 3.1 准确率的陷阱——85% 可能是幻觉

小北决定计算模型的准确率。他胸有成竹：**预测了 20 个流失，应该挺准的。**

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.2%}")
```

输出是：

```
准确率: 85.00%
```

"85%！"小北兴奋得差点跳起来，"**模型很好！我要给老板发报告了！**"

老潘看了一眼数据，问："**测试集里有多少客户不流失？**"

小北算了一下：

```python
print(y_test.value_counts())
```

输出是：

```
0    170
1     30
```

"**170 个不流失，30 个流失**，"老潘说，"**如果你的模型预测'所有人都不流失'，准确率是多少？**"

小北愣住了。他快速心算：170 / 200 = 85%。

"所以你的模型和'什么都不干的傻瓜模型'一样？"老潘问。

小北的 85% 准确率，瞬间变得一文不值。他刚才差点就给老板发了一份"看起来很棒"的报告。

### 准确率悖论：当"好"指标撒谎时

这就是**准确率悖论**（accuracy paradox）：在类别不平衡的数据中（比如 85% 不流失，15% 流失），一个什么都不干的模型就能达到 85% 的准确率，但它**毫无价值**。

阿码突然意识到什么："**欺诈检测呢？如果 99% 的交易是正常的，1% 是欺诈，一个'总是预测正常'的模型准确率是 99%，但它完全没用？**"

"对！"老潘说，"**罕见病诊断也是一样。如果 95% 的人是健康的，一个'总是预测健康'的模型准确率 95%，但漏掉所有病人**。"

小北看着他的 85%，越想越后怕：**如果这个模型上线，它会漏掉所有流失客户，业务部门根本不会有任何预警。**

"所以你需要一个更精细的评估工具——"老潘说，"**混淆矩阵**。"

---

## 3.2 混淆矩阵——看穿模型的四种结果

混淆矩阵是一张 2×2 的表，把预测结果分成四类。小北第一次看到这个名字时觉得"听起来很复杂"，但画出来之后他发现——**这其实是 Week 06 第一类/第二类错误的老朋友**。

```python
from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)

# 打印
print("混淆矩阵:")
print(cm)
```

输出是：

```
混淆矩阵:
[[165,   5]
 [ 25,   5]]
```

这四个数字是什么意思？小北画了一张表：

| | 预测不流失 (0) | 预测流失 (1) |
|---|---|---|
| **实际不流失 (0)** | 165 (真阴性 TN) | 5 (假阳性 FP) |
| **实际流失 (1)** | 25 (假阴性 FN) | 5 (真阳性 TP) |

阿码盯着这张表，突然想起来了："**这就是 Week 06 的第一类错误和第二类错误！**"

"对！"老潘说，"**假阳性（FP）就是第一类错误（α 错误）：把'不流失'错判为'流失'**。"
"**假阴性（FN）就是第二类错误（β 错误）：把'流失'错判为'不流失'**。"

小北突然醒悟："**所以我的模型虽然准确率 85%，但假阴性有 25 个——漏掉了 25 个真实流失客户！**"

### 从混淆矩阵到业务指标：精确率、召回率、F1

有了 TP/TN/FP/FN，你可以计算更有业务意义的指标。scikit-learn 提供了一个方便的函数：

```python
from sklearn.metrics import classification_report

# 打印分类报告
print(classification_report(y_test, y_pred, target_names=['不流失', '流失']))
```

输出是：

```
              precision    recall  f1-score   support

      不流失       0.87      0.97      0.92       170
        流失       0.50      0.17      0.25        30

    accuracy                           0.85       200
   macro avg       0.68      0.57      0.58       200
weighted avg       0.82      0.85      0.82       200
```

小北盯着这些数字，有点晕："**这些都什么意思？**"

老潘用业务语言解释：

**精确率（Precision）= TP / (TP + FP)**
- "在所有预测为'流失'的客户中，真正流失的比例"
- 小北的模型：0.50 = 50%（预测 10 个流失，只有 5 个真的流失了）
- 业务含义："如果按这个模型给客户发优惠券，有一半的人本来就不会流失，白送了"

**召回率（Recall）= TP / (TP + FN)**
- "在所有真实'流失'的客户中，被正确识别的比例"
- 小北的模型：0.17 = 17%（30 个真实流失，只抓到 5 个）
- 业务含义："**83% 的流失客户都被漏掉了，模型几乎没用**"

**F1 分数 = 2 × (精确率 × 召回率) / (精确率 + 召回率)**
- 精确率和召回率的**调和平均数**
- 小北的模型：0.25（非常低）
- 业务含义："综合评估，模型的表现很差"

阿码问："**为什么用调和平均数，不用算术平均数？**"

"调和平均数会惩罚极端情况，"老潘说，"**如果你的精确率是 1.0（预测 100 个流失，100 个对），但召回率是 0.01（漏掉 99% 的真实流失），F1 会非常低（约 0.02）**。算术平均数会是 0.5，看起来还行，但实际上模型毫无用处。"

### 业务决策：应该优化精确率还是召回率？

老潘给了小北一个假设场景：

"**场景 A**：你要给'可能流失'的客户发 100 元优惠券。发错一个（误报），成本 100 元；漏掉一个（漏报），损失 500 元（客户终身价值）。"

"**场景 B**：你要对'可能流失'的客户人工回访。误报会浪费客服时间（成本 20 元），漏报会损失客户（成本 500 元）。"

"在场景 A 中，你要优化**精确率**（不要乱发优惠券）；在场景 B 中，你要优化**召回率**（宁可误报，不能漏报）。"

小北若有所思："所以**没有'最好的模型'，只有'最适合业务目标的模型'？**"

"对！"老潘点头，"**混淆矩阵比准确率更重要，因为它直接连接到业务成本**。"

阿码突然举手："那……我可以调整阈值来平衡精确率和召回率吗？"

"问得好，"老潘笑了，"这正是下一节的主题——ROC-AUC。"

---

## 4. ROC-AUC：阈值无关的评估

阿码盯着混淆矩阵，突然意识到一个问题："**我们的阈值是 0.5，但谁规定的 0.5？**"

小北想了想："那……我试试 0.3？"

```python
# 用不同阈值预测
y_pred_03 = (y_proba >= 0.3).astype(int)

# 重新计算混淆矩阵和指标
print("阈值 = 0.3:")
print(classification_report(y_test, y_pred_03, target_names=['不流失', '流失']))
```

输出是：

```
阈值 = 0.3:
              precision    recall  f1-score   support

      不流失       0.90      0.88      0.89       170
        流失       0.38      0.43      0.40        30

    accuracy                           0.81       200
```

小北惊讶了："**召回率从 17% 涨到了 43%！但准确率从 85% 降到了 81%。**"

"对，"老潘说，"**降低阈值，你会抓到更多流失客户（召回率上升），但也会误报更多（精确率下降）**。这是不可避免的权衡。"

阿码问："**那怎么找到'最好'的阈值？**"

"这取决于你的业务目标，"老潘说，"但首先，你需要一个**不依赖阈值**的评估指标——**ROC-AUC**。"

### ROC 曲线：所有阈值下的性能全景

**ROC 曲线**（Receiver Operating Characteristic Curve）展示了**在所有可能的阈值下**，假阳性率（FPR）和真阳性率（TPR/召回率）的权衡关系。

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

# 计算 AUC
auc_score = roc_auc_score(y_test, y_proba)

# 画 ROC 曲线
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC 曲线 (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC = 0.5)')
plt.xlabel('假阳性率 (FPR = FP / (FP + TN))')
plt.ylabel('真阳性率 (TPR = Recall)')
plt.title('ROC 曲线：阈值无关的模型评估')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

你会看到一条从 (0, 0) 到 (1, 1) 的曲线。**曲线越靠近左上角，模型越好**。

### AUC 的直觉：随机样本对排序正确的概率

**AUC**（Area Under Curve）是 ROC 曲线下的面积，取值范围 [0, 1]：
- AUC = 1：完美分类器
- AUC = 0.5：随机猜测（像抛硬币）
- AUC < 0.5：比随机还差（可以反转预测）

阿码好奇地问："**AUC = 0.75 到底是什么意思？**"

老潘给了他一个非常直观的解释：

"**AUC = 0.75 的意思是：如果你随机选一个'流失'客户和一个'不流失'客户，模型给'流失'客户更高概率的概率是 75%。**"

阿码愣了一下，然后恍然大悟："**所以 AUC 衡量的是'排序能力'，而不是'分类准确性'？**"

"对！"老潘说，"**AUC 不依赖阈值，它回答的是'模型能不能正确区分正负样本'**。"

### 为什么 AUC 比 F1 更适合模型选择？

假设你有两个模型：
- **模型 A**：阈值 0.5 时，F1 = 0.80，但召回率只有 0.40
- **模型 B**：阈值 0.5 时，F1 = 0.70，但 AUC = 0.85（比模型 A 的 0.75 高）

你会选哪个？

"如果业务目标是'不漏掉任何流失客户'（优化召回率），模型 B 更好，"老潘说，"**因为你可以降低阈值来提高召回率，而 AUC 告诉你模型 B 的整体区分能力更强**。"

阿码问："**所以 AUC 用于模型选择，F1 用于阈值调整？**"

"对！"老潘点头，"**AUC 帮你选模型，F1 帮你选阈值**。"

---

> **AI 时代小专栏：数据泄漏——最常见的统计错误**
>
> 2025 年 3 月发表在 PMC/NIH 的研究指出，**数据泄漏是机器学习代码中最常见、最隐蔽的错误**——研究者通常不会在论文中声明他们的评估是否存在泄漏，但复现研究发现很多"高性能"模型其实是在"作弊"。
>
> 数据泄漏是什么？**在训练过程中不当使用了验证/测试集的信息**，导致评估结果虚高且不可复现。
>
> 最典型的场景是**预处理泄漏**：在 train-test split **之前**对整个数据集做 StandardScaler（标准化）、缺失值填充、特征选择或 One-Hot 编码。为什么这是泄漏？假设你的测试集均值比训练集高 20%。如果你全局标准化，测试集的"高均值"信息会被"教给"训练集。模型在训练时就已经"看到了"测试集的分布特征，所以测试集上的性能会被高估。
>
> 研究强调，**即使在 AutoML 时代，数据泄漏依然是最常见的评估偏差来源**：AutoML 工具不会自动警告你"你泄漏了"，论文和代码评审往往忽略这个问题，而泄漏的模型在生产环境中性能会大幅下降（因为生产数据没有"未来信息"）。
>
> 所以本周你学的"先 split，再预处理"和 Pipeline + ColumnTransformer 模式，在 AI 时代反而更关键了——**Pipeline 不是工程炫技，是避免数据泄漏的唯一方法**。
>
> 参考（访问日期：2026-02-12）：
> - [Data leakage detection in machine learning code (PMC/NIH, Mar 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11935776/)
> - [Preventing Training Data Leakage in AI Systems (Tonic.ai, Oct 2025)](https://www.tonic.ai/blog/prevent-training-data-leakage-ai)
> - [3 Subtle Ways Data Leakage Can Ruin Your Models (Machine Learning Mastery, Dec 2025)](https://machinelearningmastery.com/3-subtle-ways-data-leakage-can-ruin-your-models-and-how-to-prevent-it/)
> - [scikit-learn - Common Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)
> - [Don't push that button! Exploring data leakage risks (Springer, Aug 2025)](https://link.springer.com/article/10.1007/s10462-025-11326-3)

---

## 5. 交叉验证与数据泄漏——工程陷阱

小北决定"升级"他的分析流程。他学会了交叉验证，兴冲冲地写下了这样的代码：

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# 第一步：全局标准化（错误示范！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 注意：在整个 X 上拟合

# 第二步：交叉验证
log_reg = LogisticRegression(random_state=42)
scores = cross_val_score(log_reg, X_scaled, y, cv=5, scoring='accuracy')

print(f"交叉验证准确率: {scores.mean():.2%} ± {scores.std():.2%}")
```

输出是：

```
交叉验证准确率: 87.50% ± 1.20%
```

"哇！87.5%！"小北兴奋地说，"**模型很棒！**"

老潘看了一眼代码，脸色立刻变了："**你的代码有严重问题——你泄漏了数据。**"

### 什么是数据泄漏？

"数据泄漏（data leakage）是指，"老潘解释，"**你在训练过程中使用了测试集的信息，导致评估结果虚高，但模型在生产环境中完全不可用**。"

小北的代码问题在哪？

1. **`scaler.fit_transform(X)` 在整个数据集上计算了均值和方差**
2. **交叉验证的每个折都会"看到"其他折的统计量**
3. **模型在训练时已经"知道"测试集的分布特征**

结果是什么？**87.5% 的准确率是虚高的**。真正部署到生产环境，准确率可能只有 75%。

"这就像考试前偷看了答案，"老潘说，"**你考得很好，但真实能力并没有提高**。"

### 和 Week 03 的连接：特征缩放必须在 Pipeline 里做

Week 03 你学过**特征缩放**（StandardScaler）。那时候，你只是说"标准化能让数值特征处于相同尺度"，但没有强调"**什么时候标准化**"。

现在问题来了：**如果之后要做交叉验证，标准化必须在 Pipeline 里面做**。

老潘给了小北一个对比：

**错误做法**（全局标准化）：
```python
# ❌ 错误：全局 fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 泄漏！
scores = cross_val_score(log_reg, X_scaled, y, cv=5)
```

**正确做法**（Pipeline 内标准化）：
```python
# ✅ 正确：在 Pipeline 里做标准化
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression(random_state=42))
])

# 每个 CV 折内，scaler 只在训练集上 fit
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
```

小北试着运行了正确版本：

```python
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"交叉验证准确率: {scores.mean():.2%} ± {scores.std():.2%}")
```

输出是：

```
交叉验证准确率: 81.20% ± 1.50%
```

"从 87.5% 降到了 81.2%，"小北有点失落，"**这才是模型的真实性能？**"

"对，"老潘说，"**87.5% 是'作弊'的成绩，81.2% 是'诚实'的成绩**。你宁愿现在看到 81.2%，也不愿上线后发现只有 75%。考试前偷看答案，考得再好也没用。**"

### 更复杂的场景：ColumnTransformer 处理不同类型的列

如果你的数据既有**数值列**（需要 StandardScaler），又有**类别列**（需要 OneHotEncoder），怎么办？

老潘说："**你需要 ColumnTransformer + Pipeline 的组合模式**。"

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 假设 df 有这些列
numeric_features = ['tenure_months', 'monthly_charges', 'total_charges']
categorical_features = ['contract_type', 'payment_method', 'internet_service']

# 数值列预处理：填充缺失值 + 标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 类别列预处理：填充缺失值 + One-Hot 编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# ColumnTransformer 组合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 完整 Pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# 划分数据
X = df[numeric_features + categorical_features]
y = df['churn_num']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 拟合
full_pipeline.fit(X_train, y_train)

# 预测
y_pred = full_pipeline.predict(X_test)
y_proba = full_pipeline.predict_proba(X_test)[:, 1]

# 评估
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, y_pred, target_names=['不流失', '流失']))
print(f"AUC: {roc_auc_score(y_test, y_proba):.3f}")
```

这个模式的强大之处在于：
1. **所有预处理都在 Pipeline 内**：不会泄漏测试集信息
2. **每个 CV 折独立拟合预处理**：交叉验证时，每个折的 StandardScaler 只在该折的训练集上 fit
3. **可复现**：`fit(X_train)` 后的所有步骤都记录在 pipeline 对象中，可以直接用于新数据

老潘看到这段代码，点头说："**这才是工业级的写法**。它不是炫技，而是保证评估诚实性的必要手段。"

---

## 6. 完整流水线：从预处理到评估

小北现在有了一个完整的分类流水线。他兴冲冲地对老潘说："**我的模型 AUC = 0.82，准确率 81%，可以上线了吗？**"

老潘问了一个问题："**你的模型比'总是预测不流失'的傻瓜模型好多少？**"

小北愣住了："呃……我没比较过。"

"**这是分类分析中最容易被忽略的步骤——基线对比**，"老潘说，"你需要先证明你的模型比最简单的基线好，才有谈复杂模型的资格。"

### 基线对比：多数类分类器

最简单的基线是**多数类分类器**（DummyClassifier）：总是预测出现次数最多的类别。

```python
from sklearn.dummy import DummyClassifier

# 基线：总是预测多数类
dummy = DummyClassifier(strategy='most_frequent', random_state=42)
dummy.fit(X_train, y_train)

# 预测
y_pred_dummy = dummy.predict(X_test)
y_proba_dummy = dummy.predict_proba(X_test)[:, 1]

# 评估
print("=== 基线模型（总是预测不流失）===")
print(classification_report(y_test, y_pred_dummy, target_names=['不流失', '流失']))
print(f"基线 AUC: {roc_auc_score(y_test, y_proba_dummy):.3f}")
```

输出是：

```
=== 基线模型（总是预测不流失）===
              precision    recall  f1-score   support

      不流失       0.85      1.00      0.92       170
        流失       0.00      0.00      0.00        30

    accuracy                           0.85       200
   macro avg       0.42      0.50      0.46       200
weighted avg       0.72      0.85      0.78       200

基线 AUC: 0.500
```

小北惊讶地发现："**基线准确率是 85%！比我的模型还高！**"

"但你看召回率，"老潘说，"**基线的召回率是 0%（完全漏掉所有流失客户），你的模型召回率是 43%**。这就是你的价值。"

阿码问："**所以准确率不是最重要的？**"

"对，"老潘说，"**在分类问题中，业务价值往往来自'正确识别少数类'（流失客户、欺诈交易、罕见病患者）**。一个 85% 准确率但召回率 0% 的模型，和一个 81% 准确率但召回率 43% 的模型，后者更有业务价值。"

### K-fold 交叉验证：更稳健的性能估计

小北之前用的是简单的 train-test split（80/20）。老潘说："**单次划分的结果可能不稳定，你应该用 K-fold 交叉验证**。"

对于分类问题，推荐使用 **StratifiedKFold**（分层折），它保持每个折中类别比例与整体数据一致：

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 定义分层 K-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 交叉验证（评估多个指标）
scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'recall': 'recall'
}

from sklearn.model_selection import cross_validate
cv_results = cross_validate(full_pipeline, X, y, cv=skf, scoring=scoring, return_train_score=False)

# 打印结果
print("=== 5-fold 交叉验证结果 ===")
for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f"{metric}: {scores.mean():.3f} ± {scores.std():.3f}")
```

输出是：

```
=== 5-fold 交叉验证结果 ===
accuracy: 0.812 ± 0.015
f1: 0.402 ± 0.032
roc_auc: 0.801 ± 0.021
recall: 0.425 ± 0.045
```

"看到标准差了吗？"老潘说，"**你的模型在不同折上的性能是 80.1% ± 2.1%，这说明模型相对稳定**。如果标准差很大（比如 ±10%），说明模型对数据划分很敏感，就像考试发挥忽高忽低。**"

### 完整的评估报告

老潘给了小北一个模板："**把所有结果整合到一个报告里**"：

```python
def generate_classification_report(model, X_train, y_train, X_test, y_test):
    """生成完整的分类评估报告"""
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        roc_auc_score, roc_curve
    )
    import matplotlib.pyplot as plt

    # 拟合
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 1. 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # 2. 评估指标
    report = f"""
## 分类评估报告

### 混淆矩阵
| | 预测不流失 | 预测流失 |
|---|---|---|
| **实际不流失** | {tn} (真阴性) | {fp} (假阳性) |
| **实际流失** | {fn} (假阴性) | {tp} (真阳性) |

### 评估指标
- 准确率: {(tp+tn)/(tp+tn+fp+fn):.2%}
- 精确率: {tp/(tp+fp):.2%} (预测为流失的客户中，真正流失的比例)
- 召回率: {tp/(tp+fn):.2%} (真实流失的客户中，被识别的比例)
- F1 分数: {2*tp/(2*tp+fp+fn):.3f} (精确率和召回率的调和平均)
- AUC: {roc_auc_score(y_test, y_proba):.3f} (模型区分正负样本的能力)

### 业务解释
- 假阳性成本（误报）: {fp} 个客户被误判为流失，可能浪费营销成本
- 假阴性成本（漏报）: {fn} 个流失客户未被识别，损失客户终身价值
- 模型价值: 相比基线（召回率 0%），本模型识别了 {tp/(tp+fn):.1%} 的真实流失客户
"""

    return report

# 生成报告
report = generate_classification_report(full_pipeline, X_train, y_train, X_test, y_test)
print(report)
```

小北看到这份报告，若有所思："**这才是能向业务部门展示的评估**——不是冷冰冰的数字，而是'假阳性成本'、'假阴性成本'、'模型价值'。**"

"对，"老潘点头，"**分类分析的最后一步不是'算出 AUC'，而是'把 AUC 翻译成业务语言'**。"

---

## StatLab 进度

到上周为止，StatLab 报告已经有了数据卡、描述统计、清洗日志、EDA 叙事、假设检验、不确定性量化和回归分析。但老潘看完报告，问了小北同样的问题："**你能预测'哪些客户会流失/购买'吗？**"

小北想了想："呃……我可以用回归分析预测消费金额，但不能预测二分类结果（是/否）。"

"对，"老潘说，"**回归分析告诉你'消费金额如何变化'，但分类分析告诉你'这个客户会不会流失/购买'**。这是从'理解关系'到'预测行为'的跨越。"

这正好是本周"分类与评估"派上用场的地方。我们要在 report.md 中添加一个**分类评估章节**，包含：
1. 逻辑回归模型（系数解释 + 优势比）
2. 混淆矩阵（精确率、召回率、F1）
3. ROC-AUC 分析（阈值无关的评估）
4. 交叉验证（K-fold CV 结果）
5. Pipeline + ColumnTransformer（防止数据泄漏）
6. 基线对比（与多数类分类器对比）
7. 局限性讨论（因果推断、类别不平衡）

### 在 StatLab 中添加分类评估

假设你的数据集是电商用户购买，目标是"预测哪些用户会购买"（Purchase = Yes/No）。下面是一个完整的分类评估函数，它会生成 ROC 曲线和报告片段：

```python
# examples/10_statlab_classification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve
)
from sklearn.dummy import DummyClassifier

def classification_evaluation_to_report(df, target, numeric_features, categorical_features, output_path):
    """
    对 StatLab 数据集进行分类评估，生成报告片段

    参数:
        df: 清洗后的数据
        target: 目标变量名（如 'purchase', 'churn'）
        numeric_features: 数值特征列表
        categorical_features: 类别特征列表
        output_path: 图表输出路径
    """
    X = df[numeric_features + categorical_features]
    y = df[target]

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ========== 1. 构建 Pipeline ==========
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])

    # ========== 2. 拟合与预测 ==========
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    y_proba = full_pipeline.predict_proba(X_test)[:, 1]

    # ========== 3. 混淆矩阵 ==========
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # ========== 4. ROC 曲线 ==========
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC 曲线 (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC = 0.5)')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR / Recall)')
    plt.title(f'ROC 曲线 - {target} 预测')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_path}/roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========== 5. K-fold 交叉验证 ==========
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        full_pipeline, X, y, cv=skf,
        scoring={'accuracy': 'accuracy', 'f1': 'f1', 'roc_auc': 'roc_auc'},
        return_train_score=False
    )

    # ========== 6. 基线对比 ==========
    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    dummy_acc = (y_pred_dummy == y_test).mean()

    # ========== 7. 生成报告片段 ==========
    report = f"""
## 分类评估

### 研究问题
哪些因素影响{target}（二分类）？我们使用逻辑回归建模，并评估模型的预测性能。

### 模型
- 算法: 逻辑回归 (Logistic Regression)
- 特征: {', '.join(numeric_features + categorical_features)}
- 预处理: 数值特征标准化 + 类别特征 One-Hot 编码
- 评估方法: 5-fold 分层交叉验证（StratifiedKFold）

### 逻辑回归系数

**截距与系数**:
"""

    # 提取系数
    feature_names = numeric_features + list(
        full_pipeline.named_steps['preprocessor']
        .named_transformers_['cat'][1]
        .get_feature_names_out(categorical_features)
    )
    coefs = full_pipeline.named_steps['classifier'].coef_[0]

    # 报告前 10 个最重要的特征
    coef_df = pd.DataFrame({
        '特征': feature_names,
        '系数': coefs
    })
    coef_df['|系数|'] = np.abs(coef_df['系数'])
    coef_df = coef_df.sort_values('|系数|', ascending=False).head(10)

    for _, row in coef_df.iterrows():
        report += f"- {row['特征']}: {row['系数']:.3f}\n"

    report += f"""
**解释**: 逻辑回归系数表示"对数优势比"的变化。系数为正表示该特征增加会提高{target}的优势，系数为负表示降低。

### 混淆矩阵与评估指标

| | 预测=0 | 预测=1 |
|---|---|---|
| **实际=0** | {tn} (真阴性) | {fp} (假阳性) |
| **实际=1** | {fn} (假阴性) | {tp} (真阳性) |

**指标**:
- 准确率: {(tp+tn)/(tp+tn+fp+fn):.2%}
- 精确率: {tp/(tp+fp) if (tp+fp) > 0 else 0:.2%} (预测为{target}=1的样本中，真正为1的比例)
- 召回率: {tp/(tp+fn) if (tp+fn) > 0 else 0:.2%} (真实为{target}=1的样本中，被正确识别的比例)
- F1 分数: {2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0:.3f} (精确率和召回率的调和平均)

**业务解释**:
- 假阳性（误报）: {fp}个样本被错误预测为{target}=1，可能浪费资源
- 假阴性（漏报）: {fn}个真实{target}=1的样本被遗漏，可能造成业务损失
- 模型价值: 本模型的召回率为{tp/(tp+fn) if (tp+fn) > 0 else 0:.1%}，相比基线模型（总是预测多数类，召回率 0%）有显著提升

### ROC-AUC 分析

**AUC**: {auc_score:.3f}

AUC（ROC 曲线下面积）衡量模型区分正负样本的能力，不依赖分类阈值：
- AUC = 1.0: 完美分类器
- AUC = 0.5: 随机猜测
- 本模型 AUC = {auc_score:.3f}: {"强" if auc_score > 0.8 else "中等" if auc_score > 0.7 else "弱"}区分能力

![ROC 曲线](roc_curve.png)

### 交叉验证结果

5-fold 分层交叉验证（StratifiedKFold）结果:
- 准确率: {cv_results['test_accuracy'].mean():.3f} ± {cv_results['test_accuracy'].std():.3f}
- F1 分数: {cv_results['test_f1'].mean():.3f} ± {cv_results['test_f1'].std():.3f}
- AUC: {cv_results['test_roc_auc'].mean():.3f} ± {cv_results['test_roc_auc'].std():.3f}

**稳定性**: 标准差较小（< 0.05），说明模型对不同数据划分稳健。

### 基线对比

与多数类基线（总是预测出现最多的类别）对比:
- 基线准确率: {dummy_acc:.2%}
- 本模型准确率: {(tp+tn)/(tp+tn+fp+fn):.2%}
- 改进: {(tp+tn)/(tp+tn+fp+fn) - dummy_acc:.1%}

**结论**: 本模型相比基线{"有" if (tp+tn)/(tp+tn+fp+fn) > dummy_acc else "无"}显著提升。更重要的是召回率从基线的 0% 提升到{tp/(tp+fn) if (tp+fn) > 0 else 0:.1%}，说明模型能有效识别{target}=1的样本。

### 工程实践：防止数据泄漏

本分析使用 **Pipeline + ColumnTransformer** 模式：
- 所有预处理（标准化、One-Hot 编码、缺失值填充）都在 Pipeline 内部完成
- 交叉验证时，每个折独立拟合预处理参数（如均值、方差）
- 确保测试集信息不会泄漏到训练过程

这是分类评估中的最佳实践，避免"虚高"的性能估计。

### 局限性与因果警告

⚠️ **本分析仅描述{target}与预测特征的关联关系，不能直接推断因果**。

**局限性**:
1. **类别不平衡**: 如果{target}=1的样本比例很低，模型可能在少数类上表现不佳（召回率低）
2. **观察数据**: 本分析基于观测数据，未进行随机实验，无法排除混杂变量和反向因果
3. **阈值选择**: 默认阈值 0.5 可能不是业务最优，应根据假阳性/假阴性成本调整

**因果推断**: Week 13 会学习的因果图（DAG）和识别策略（如RCT、工具变量）可用于回答"改变X是否会导致Y变化"的问题。本分析仅限于"预测"，不涉及"因果"。

### 数据来源
- 样本量: n = {len(y)}
- {target}=1 的比例: {y.mean():.2%}
- 分析日期: 2026-02-12
"""

    return report

# 使用示例
if __name__ == "__main__":
    # 假设你的数据已经清洗完成
    df = pd.read_csv("data/clean_data.csv")

    # 选择目标变量和特征
    target = "purchase"  # 或 "churn"
    numeric_features = ["age", "income", "days_since_last_purchase"]
    categorical_features = ["gender", "city_tier", "membership_level"]

    # 生成分类评估报告
    report = classification_evaluation_to_report(
        df=df,
        target=target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        output_path="report"
    )

    # 追加到 report.md
    with open("report/report.md", "a", encoding="utf-8") as f:
        f.write(report)

    print("✅ 分类评估章节已添加到 report/report.md")
```

### 本周改进总结

| 改动项 | 上周状态 | 本周改进 |
|--------|---------|---------|
| 报告章节 | 回归分析（连续目标） | 新增"分类评估"章节（二分类目标） |
| 预测目标 | "消费金额如何变化" | "客户会流失/购买吗？" |
| 评估指标 | R²、系数显著性、残差诊断 | 混淆矩阵、精确率/召回率/F1、ROC-AUC |
| 工程实践 | 简单 train-test split | Pipeline + ColumnTransformer（防止数据泄漏） |
| 性能估计 | 单次划分 | K-fold 交叉验证（更稳健） |
| 基线对比 | 不适用 | 与多数类分类器对比 |
| 因果警告 | "关联≠因果" | 再次强调"预测≠因果" |

老潘看到这段改动会说什么？"**这才是从'回归分析'到'分类评估'的完整升级**。你不仅学会了逻辑回归和新的评估指标，还掌握了 Pipeline + ColumnTransformer 的工程实践，确保评估不会因为数据泄漏而虚高。AI 可以帮你拟合模型、计算 AUC，但防止泄漏、选择基线、解释业务价值的责任由你承担。"

---

## Git 本周要点

本周必会命令：
- `git status`: 查看工作区状态
- `git diff`: 查看具体改动内容
- `git add -A`: 添加所有改动
- `git commit -m "feat: add classification evaluation with pipeline"`
- `git log --oneline -n 5`

常见坑：

**只看准确率，忽略类别不平衡**——这就像在"99% 不流失"的数据上预测"所有人都不流失"，准确率 99%，但模型毫无价值。建议至少报告精确率、召回率、F1，并画出混淆矩阵。

**全局 StandardScaler 导致数据泄漏**——在交叉验证之前对整个数据集做预处理，会泄漏测试集信息。建议用 Pipeline + ColumnTransformer，确保每个折内独立拟合预处理步骤。

**忽略基线对比**——你的复杂模型如果连"总是预测多数类"都打不赢，就没有价值。建议先拟合 DummyClassifier，再对比你的模型。

**不理解阈值选择**——默认阈值 0.5 不一定是业务最优解。建议根据业务目标（如"更看重召回率"）调整阈值，并用 ROC 曲线找到最佳平衡点。

**误把逻辑回归系数当成概率变化**——逻辑回归系数是"对数优势比"的变化，不是概率的变化。建议报告优势比 exp(β) 或"每增加 1 单位，优势变化多少"。

Pull Request (PR)：
- Gitea 上也叫 Pull Request，流程等价 GitHub：push 分支 -> 开 PR -> review -> merge。

---

## 本周小结（供下周参考）

本周你从"回归预测连续值"（Week 09）升级为"分类预测类别"（Yes/No）。

具体来说，你理解了为什么线性回归不适合分类问题——预测值会超出 [0,1]，误差项假设不成立。你学会了用逻辑回归 + Sigmoid 函数建模概率，并用优势比（OR）正确解释系数。

更重要的是，你升级了评估思维：从单一的"准确率"到混淆矩阵（TP/TN/FP/FN）、精确率、召回率、F1，再到阈值无关的 ROC-AUC。你明白了准确率在类别不平衡场景下会撒谎——85% 的准确率可能等于"什么都不干的傻瓜模型"，而混淆矩阵直接连接到业务成本（假阳性/假阴性）。

你还掌握了 K-fold 交叉验证（特别是 StratifiedKFold 保持类别比例）和 Pipeline + ColumnTransformer 的工程实践，这是防止数据泄漏的唯一方法。老潘说的"Pipeline 不是炫技，是避免泄漏的唯一方法"，正是本周的核心教训。

下周（Week 11），你要把这个思维扩展到**树模型与集成学习**：从可解释的逻辑回归到更复杂的决策树、随机森林，并学习超参数调优、特征工程和模型可解释性（SHAP）。本周的分类评估会演化为下周的"模型比较"和"特征重要性"，Pipeline 会演化为"更复杂的预处理流程"。

---

## Definition of Done（学生自测清单）

- [ ] 我能解释为什么不能用线性回归做分类（预测值超出 [0,1]）
- [ ] 我能理解逻辑回归的动机（Sigmoid 函数把线性预测映射到概率）
- [ ] 我能正确解释逻辑回归系数（优势比 odds ratio）的含义
- [ ] 我能理解决策阈值的选择（默认 0.5 但可调）
- [ ] 我能从准确率升级到混淆矩阵（精确率、召回率、F1）
- [ ] 我能理解准确率悖论（类别不平衡时，高准确率可能是幻觉）
- [ ] 我能理解 ROC-AUC 的直觉（随机样本对排序正确的概率）
- [ ] 我能正确使用 K-fold 交叉验证（每个折独立拟合预处理）
- [ ] 我能识别并防御数据泄漏（Pipeline + ColumnTransformer）
- [ ] 我能在 StatLab 报告中添加分类评估章节（混淆矩阵、ROC-AUC、CV 结果）
- [ ] 我能审查 AI 生成的分类报告，识别缺少阈值讨论、数据泄漏等问题
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解"分类不是算个准确率，而是在假阳性和假阴性之间做权衡"这句话的含义
