# Week 09：模型不是魔法——回归分析与诊断

> "All models are wrong, but some are useful."
> — George Box

---

2026 年，你可以在几秒钟内把数据丢给 AI，得到一份"看起来很专业"的回归报告：R²、系数、p 值，甚至"结论"一应俱全。但这里有一个被很多人忽略的问题：**拟合好不等于模型对，R² 高不等于假设满足**。

小北上周学会了 Bootstrap 和置信区间，兴冲冲地拿着一份回归结果去找老潘："R² = 0.85，模型很棒！"

老潘看完残差图，只说了一句话："你的模型违反了等方差假设，这个 R² 不可信。"

小北愣住了。"R² 还有真假？"

"R² 是真的，但它基于的假设是假的。"老潘继续说，"你有没有检查残差？有没有验证线性假设？有没有考虑异常值的影响？"

这正是本周的核心问题：**回归分析与模型诊断**。你将学习简单线性回归和多元回归的核心原理，更重要的是，你将学会如何检查模型的假设是否满足——残差图、QQ图、Cook's 距离、VIF（多重共线性）等诊断工具。

Andrew Gelman 说过："模型的检查比模型的拟合更重要。"在 AI 时代，AI 可以帮你拟合模型，但只有你能判断"这个模型是否可信"。

---

## 前情提要

上一周你学会了区间估计与重采样：从"只给点估计"到"点估计 + 95% CI"，从"依赖分布假设的理论公式"到"让数据说话的 Bootstrap"。

阿码拿着上周的置信区间报告问："我有了均值差异的 CI，也知道 p 值和效应量。接下来能做什么？"

老潘点头："你已经在回答'有没有差异'这个问题。接下来要问的是：'这个差异是怎么来的？哪些因素在影响它？'"

"这就是 Week 09 要讲的：回归分析与模型诊断。"

---

## 学习目标

完成本周学习后，你将能够：

1. 理解回归分析的本质——"用自变量预测因变量"的统计模型
2. 解释回归系数的含义，并用回归分析回答"X 对 Y 有什么影响"
3. 检查回归模型的假设（LINE：线性、独立性、正态性、等方差）
4. 诊断模型问题：残差图、QQ图、Cook's 距离、VIF（多重共线性）
5. 在 StatLab 报告中写出"带诊断检查"的回归结论

---

<!--
贯穿案例：从"画散点图"到"回归模型 + 诊断报告"

案例演进路线：
- 第 1 节（回归是什么）→ 从"散点图 + 相关系数"到"回归线 + 预测"
- 第 2 节（简单线性回归）→ 从"描述关系"到"量化关系：斜率、截距、R²"
- 第 3 节（回归假设）→ 从"只看 R²"到"检查 LINE 假设"
- 第 4 节（模型诊断）→ 从"拟合模型"到"残差诊断 + 异常点检查"
- 第 5 节（多元回归与问题）→ 从"简单回归"到"多元回归 + 多重共线性 + 交互项"

最终成果：读者能拟合回归模型、解释系数、检查假设、诊断问题，并能写出"带诊断检查"的回归报告

数据集：复用电商数据，聚焦"广告投入 vs 销售额"的简单回归，以及"多渠道广告投入 + 其他因素"的多元回归

---

认知负荷预算：
- 本周新概念（5 个，预算上限 5 个）：
  1. 简单线性回归（simple linear regression）：y = a + bx + ε
  2. 回归系数解释（regression coefficient interpretation）：斜率表示 x 每增加 1 单位，y 的变化
  3. R² 和拟合优度（R² and goodness of fit）：模型解释的方差比例
  4. 回归假设（LINE 假设）：线性、独立性、正态性、等方差
  5. 模型诊断（model diagnostics）：残差分析、QQ图、Cook's 距离、VIF
- 结论：✅ 在预算内

回顾桥设计（至少 2 个，来自 week_05-08）：
- [相关分析]（来自 week_04）：在第 1 节，通过"相关 vs 回归"再次使用——相关描述关系强度，回归量化关系方向和大小
- [假设检验]（来自 week_06）：在第 2 节，通过"回归系数的显著性检验"再次使用
- [置信区间]（来自 week_08）：在第 2 节，通过"回归系数的 CI"再次使用
- [ANOVA]（来自 week_07）：在第 2 节，通过"回归与 ANOVA 的关系（F 检验）"再次使用
- [异常值检测]（来自 week_03）：在第 4 节，通过"Cook's 距离识别高影响点"再次使用
- [Bootstrap]（来自 week_08）：在第 5 节，通过"Bootstrap 稳健回归"再次使用

AI 小专栏规划：
- 第 1 个侧栏（第 1-2 节之后）：
  - 主题："AI 为什么会'过度拟合'？"
  - 连接点：刚学完 R² 和拟合优度，讨论 AI/ML 模型常见的"高 R² 但泛化差"问题
  - 建议搜索词："AI overfitting high R² 2026", "machine learning overfitting detection 2026", "R² vs generalization 2026"

- 第 2 个侧栏（第 3-4 节之后）：
  - 主题："AI 时代的模型诊断——从残差图到 SHAP"
  - 连接点：刚学完模型诊断（残差图、Cook's 距离），讨论现代可解释 AI（XAI）如何扩展传统诊断工具
  - 建议搜索词："model diagnostics AI 2026", "SHAP vs residual analysis 2026", "interpretable machine learning diagnostics 2026"

角色出场规划：
- 小北（第 2 节）：误以为"高 R² 就是好模型"，引出 R² 的局限性和假设检查的重要性
- 阿码（第 3 节）：追问"回归系数的 p 值和假设检验的 p 值有什么区别？"，引出回归的检验框架
- 老潘（第 4 节）：看到"没有残差图的回归报告"后点评"这不是分析，是自欺欺人"

StatLab 本周推进：
- 上周状态：数据卡 + 描述统计 + 可视化 + 清洗日志 + 相关分析 + 分组比较 + 假设清单 + 多组比较 + 区间估计 + Bootstrap + 置换检验
- 本周改进：添加回归分析模块，包括模型拟合、系数解释、假设检查、诊断图表
- 涉及的本周概念：简单线性回归、回归系数解释、R² 和拟合优度、回归假设、模型诊断
- 建议示例文件：examples/09_statlab_regression.py（本周报告生成入口脚本）
-->

## 1. 回归不是魔法——从"画散点图"到"预测"

老潘在看一份报告，眉头皱了起来。小北凑过去看，发现是一张广告投入和销售额的散点图。

"相关性 0.75，"报告上写着，"说明两者高度相关。"

"那我问你，"老潘头也不抬，"如果明年广告预算增加 10 万，销售额会涨多少？"

小北愣住了。"大概……会涨？"

"大概不够。"老潘把报告放下，"你的相关系数告诉决策者'有关系'，但决策者要的是'有多大关系'。相关回答不了这个问题。"

---

<!--
**Bloom 层次**：理解
**学习目标**：理解回归分析的本质，区分"相关"和"回归"
**贯穿案例推进**：从"散点图 + 相关系数"到"回归线 + 预测"
**建议示例文件**：01_correlation_vs_regression.py
**叙事入口**：从"小北有散点图但无法量化关系"开头
**角色出场**：小北误以为"高相关就是好模型"
**回顾桥**：[相关分析]（week_04）：通过"相关 vs 回归"再次使用——相关描述关系强度，回归量化关系方向和大小
-->

### 相关 vs 回归：描述 vs 预测

你在 Week 04 学过**相关分析**（correlation analysis）：Pearson 相关系数 r 描述两个变量的**线性关系强度**。r 在 -1 到 1 之间，绝对值越接近 1 表示关系越强。

但相关有个局限：它不告诉你"y 会随着 x 变化多少"。你说广告投入和销售额"强相关"，但决策者问的是："我多投 10 万广告，能多卖多少？"相关回答不了这个问题。

**回归分析**（regression analysis）更进一步。它不仅告诉你"有没有关系"，还告诉你"y 如何随 x 变化"。

回归方程是这样的：y = a + bx + ε

- a 是**截距**（intercept）：x = 0 时 y 的预测值
- b 是**斜率**（slope）：x 每增加 1 单位，y 的变化量
- ε 是**残差**（residual）：预测值与真实值的差距

阿码举手："那我是不是以后都应该用回归，不用相关了？"

"不。"老潘说，"相关和回归回答不同问题。相关告诉你'有多强的关系'，回归告诉你'什么关系'。两者配合使用才是正道。"

### 最小二乘法：为什么回归线是这样的？

你可能会问：**为什么回归线是这条线，而不是那条线？**

答案是：**最小二乘法**（Ordinary Least Squares, OLS）。它选择一条线，使得**残差平方和最小**（Sum of Squared Residuals, SSR）。换句话说：让预测值和真实值的距离（平方）总和最小。

老潘说："想象你在散点图上画一条线。移动这条线，直到所有点到线的距离（平方和）最小。这就是 OLS 找的回归线。"

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 示例数据：广告投入 vs 销售额
np.random.seed(42)
ad_spend = np.random.normal(loc=50, scale=15, size=100)
sales = 10 + 0.5 * ad_spend + np.random.normal(loc=0, scale=5, size=100)

# 画散点图 + 回归线
plt.figure(figsize=(8, 6))
sns.regplot(x=ad_spend, y=sales, line_kws={'color': 'red'})
plt.xlabel('广告投入（万元）')
plt.ylabel('销售额（万元）')
plt.title('广告投入 vs 销售额')
plt.show()
```

运行这段代码，你会看到一条红色的回归线穿过散点图。这条线的公式大致是：sales = 10 + 0.5 × ad_spend + ε。斜率 0.5 表示：广告投入每增加 1 万元，销售额平均增加 0.5 万元。

但 OLS 有个"脾气"：它对异常值敏感。如果有一个极端点（比如广告投入 200 万，销售额 10 万），整条回归线会被拽偏。这就是为什么我们需要模型诊断——这周后面会讲。

### 回归的三大用途

你现在知道了回归是什么。那它能做什么？老潘总结了三个主要用途：

1. **描述**：x 和 y 之间是什么关系？（正相关？负相关？无关？）
2. **预测**：给定 x，预测 y 是多少？
3. **推断**：x 对 y 有没有显著影响？（系数的 p 值、置信区间）

小北若有所思："所以我上周算的相关系数，只是第一步？"

"对。"老潘说，"相关告诉你'有关系'，回归告诉你'什么关系'。而且，回归还能告诉你'这个关系是否显著'——这就要看 p 值了。"

这正好引出下一节的内容：如何解读回归系数和 R²？什么算"显著"？

> **AI 时代小专栏：AI 为什么会'过度拟合'？**

> 2026 年，很多 AI 工具可以帮你自动拟合模型，输出高 R² 值。但这里有一个被广泛忽略的问题：**R² 高不等于模型好，过度拟合（Overfitting）是 AI/ML 模型的常见陷阱**。
>
> **过度拟合是什么？**
>
> 过度拟合是指模型在训练数据上表现很好（高 R²），但在新数据上表现很差。模型"记住了"训练数据的噪声，而不是学习真实的模式。
>
> **为什么 AI 会过度拟合？**
>
> 1. **模型太复杂**：比如用 10 次多项式拟合 20 个数据点；
> 2. **特征太多**：比如用 100 个特征预测 50 个样本；
> 3. **缺乏正则化**：没有对模型复杂度进行惩罚。
>
> **2026 年的研究指出**：
>
> - R² 本身不是预测准确性的好指标——仅依赖训练集 R² 不足以判断泛化能力。R² 总是随特征增加而增加，即使是无意义的特征——这是过度拟合的经典来源。
> - 交叉验证是防止过度拟合的核心方法：将数据分成 k 份，在 k-1 份上训练，在 1 份上验证。
> - 最佳实践：CV 调参后，在全部数据上重新训练最终模型。
>
> **对你的启示**：
>
> 你本周学的 R² 只是第一步。AI 可以帮你拟合模型，但只有你能判断"这个模型是否可信"。高 R² 不等于好模型——你需要检查假设、诊断残差、验证泛化能力（交叉验证在 Week 10 会讲）。
>
> 参考（访问日期：2026-02-16）：
> - [R-Squared: Coefficient of Determination - Arize AI](https://arize.com/blog-course/r-squared-understanding-the-coefficient-of-determination/)
> - [Cross Validation - Scikit-learn Official Docs](https://scikit-learn.org/stable/modules/cross_validation.html)
> - [Cross Validation in Machine Learning - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/cross-validation-machine-learning/) (Dec 17, 2025)
> - [Prevent Overfitting with Automated ML - Azure](https://docs.azure.cn/en-us/machine-learning/concept-manage-ml-pitfalls?view=azureml-api-2) (Jan 4, 2026)

---

## 2. 简单线性回归——解读系数与 R²

小北看完上一节，说："我懂了，回归是'用 x 预测 y'。那怎么解读回归方程？"

老潘在黑板上写了一个方程：`sales = 10 + 0.5 × ad_spend`

"你来解读一下。"

---

<!--
**Bloom 层次**：应用
**学习目标**：能解读回归系数和 R² 的含义，能判断模型拟合好坏
**贯穿案例推进**：从"描述关系"到"量化关系：斜率、截距、R²"
**建议示例文件**：02_interpret_coefficients.py
**叙事入口**：从"小北不会解读回归方程"开头
**角色出场**：小北误以为"高 R² 就是好模型"
**回顾桥**：[置信区间]（week_08）：通过"回归系数的 CI"再次使用；[假设检验]（week_06）：通过"回归系数的显著性检验"再次使用；[ANOVA]（week_07）：通过"回归与 ANOVA 的关系（F 检验）"再次使用
-->

### 回归方程的解读

假设你拟合了广告投入（ad_spend）和销售额（sales）的回归模型：

```
sales = 10 + 0.5 × ad_spend
         ↑    ↑
       截距  斜率
```

**解读规则**：

- **截距（10）**：当 ad_spend = 0 时，sales 的预测值是 10 万元。但要注意：截距有时没有实际意义（比如"广告投入为 0"可能不在数据范围内，甚至是反事实的）。
- **斜率（0.5）**：ad_spend 每增加 1 万元，sales 平均增加 0.5 万元。斜率 > 0 表示正相关，< 0 表示负相关，= 0 表示无关系。

小北若有所思："所以斜率告诉我'影响有多大'？"

"对。而且斜率还有一个 p 值，告诉你这个影响是否显著。"老潘继续说，"这和你 Week 06 学的假设检验是同一回事。"

### 回归系数的显著性检验

你在 Week 06 学过假设检验。回归分析中也有类似的检验：

- **零假设（H0）**：斜率 = 0（x 对 y 没有影响）
- **备择假设（H1）**：斜率 ≠ 0（x 对 y 有影响）

如果 p < 0.05，你拒绝 H0，认为"x 对 y 有显著影响"。

阿码举手："这个 p 值和 Week 06 的 t 检验有什么区别？"

"本质上是一样的。"老潘说，"回归系数的显著性检验就是 t 检验——检验'斜率是否显著不为 0'。不信你看代码。"

```python
import statsmodels.api as sm

# 准备数据
X = ad_spend
y = sales

# 添加截距项（statsmodels 不会自动加）
X_with_const = sm.add_constant(X)

# 拟合 OLS 模型
model = sm.OLS(y, X_with_const).fit()

# 输出结果
print(model.summary())
```

运行这段代码，你会看到一张长长的表格。其中的关键信息：

| 指标 | 含义 |
|------|------|
| **coef（const）** | 截距的估计值 |
| **coef（ad_spend）** | 斜率的估计值 |
| **P>\|t\|（ad_spend）** | 斜率的 p 值，< 0.05 表示显著 |
| **\[0.025   0.975\]** | 回归系数的 95% 置信区间 |
| **R-squared** | 模型解释的方差比例（拟合优度） |
| **F-statistic** | 整体模型的 F 检验（类似 ANOVA） |
| **Prob (F-statistic)** | F 检验的 p 值，< 0.05 表示整体模型显著 |

注意：F 检验和 t 检验在简单回归中是等价的（p 值相同），但在多元回归中，F 检验回答"整体模型是否显著"，t 检验回答"某个系数是否显著"。

### R²：拟合优度

**R²**（R-squared）是回归分析中最常见的指标。R² = 0.85 表示：模型解释了销售额 85% 的变异。R² 在 0 到 1 之间，越接近 1 表示拟合越好。

但 R² 有三个陷阱：

1. **R² 高不等于假设满足**：一个高 R² 的模型可能违反等方差、正态性等假设
2. **R² 会随特征增加而增加**：加入任何特征（哪怕是噪声）都会让 R² 变大——这就是为什么下一节要讲"调整 R²"
3. **R² 无法判断因果关系**：高 R² 不等于 x 导致 y

小北问："那怎么看 R² 才对？"

老潘给了经验法则：
- **R² < 0.3**：模型解释力很弱，可能需要更多特征或换模型
- **0.3 ≤ R² < 0.7**：模型有一定解释力，但还有改进空间
- **R² ≥ 0.7**：模型解释力较强（但也要检查假设）

"记住：R² 只是参考，不是金标准。"老潘说，"更重要的是：残差诊断。高 R² 但残差图有模式，模型仍然是错的。"

阿码追问："那你怎么知道'残差图有模式'？"

这正是下一节要讲的内容：回归假设和模型诊断。

---

## 3. 回归假设——模型是有前提的

小北兴冲冲地拿着回归结果来找老潘："R² = 0.75，斜率的 p < 0.001，模型很完美！"

老潘看完残差图，只说了一句话："你的模型违反了等方差假设，这个结论不可信。"

小北愣住了。"回归还有假设？"

---

<!--
**Bloom 层次**：分析
**学习目标**：理解回归的 LINE 假设，能检查假设是否满足
**贯穿案例推进**：从"只看 R²"到"检查 LINE 假设"
**建议示例文件**：03_regression_assumptions.py
**叙事入口**：从"小北以为回归没有假设"开头
**角色出场**：阿码追问"回归系数的 p 值和假设检验的 p 值有什么区别？"，引出回归的检验框架
**回顾桥**：[假设检验框架]（week_06）：通过"回归也是假设检验"再次使用；[检验前提假设]（week_06）：通过"回归也需要检查前提假设"再次使用
-->

### LINE：回归的四大假设

老潘在黑板上写了四个字母：LINE。"回归分析有四个核心假设，记住这个词就好。"

| 假设 | 英文 | 含义 | 违反后果 |
|------|------|------|----------|
| **L**inear | 线性 | y 和 x 之间是线性关系 | 系数有偏、预测不准 |
| **I**ndependence | 独立性 | 残差之间相互独立 | p 值不可信 |
| **N**ormal | 正态性 | 残差服从正态分布 | p 值、CI 不准确 |
| **E**qual variance | 等方差 | 残差的方差恒定（同方差） | p 值、CI 不准确 |

阿码问："如果假设不满足怎么办？"

"先别急着换方法。"老潘说，"第一步永远是：**知道你的假设是否满足**。如果残差图告诉你有问题，再考虑处理。"

小北追问："那有哪些处理方法？"

"常见的有几种，"老潘数了数，"数据转换（对 y 取对数或做 Box-Cox 转换），或者用稳健标准误让 p 值更可信。更高级的还有稳健回归和非参数回归，但这些是后面的话题——先把诊断学会，再说治疗。"

### 假设 1：线性（Linear）

**如何检查？** 画散点图 + 回归直线，或者画**残差 vs 拟合值图**（Residuals vs Fitted）。

**理想情况**：散点图中的点大致分布在直线周围；残差图中的残差在 0 上下随机分布，没有明显模式。

**违反的情况**：散点图显示曲线关系（需要多项式回归或数据转换）；残差图显示 U 型或倒 U 型（需要加入二次项）。

老潘说："线性假设不是说'数据必须完美地落在一条直线上'，而是说'直线是一个合理的近似'。如果你的残差图有明显的曲线，那就要处理了。"

### 假设 2：独立性（Independence）

**如何检查？** 看数据收集方式（是否独立抽样），画**残差 vs 观测顺序图**，或者看 `model.summary()` 中的 Durbin-Watson 统计量。

**违反的情况**：数据有时间序列结构（今天的销售额和昨天相关）；数据有聚类结构（同一城市、同一学校的观测相关）。

"独立性假设更多是数据收集的问题，不是模型能解决的。"老潘说，"如果你有时间序列或聚类数据，可能需要更高级的模型（时间序列模型、混合效应模型）。"

### 假设 3：正态性（Normal）

**如何检查？** 画 **QQ 图**（Quantile-Quantile plot），或者做 Shapiro-Wilk 检验。

**理想情况**：QQ 图中的点大致落在对角线上；残差的直方图大致对称、钟形。

**违反的情况**：QQ 图偏离对角线（S 型或反 S 型）；残差有偏态（左偏或右偏）。

"好消息是：回归对正态性假设的偏离比较稳健。"老潘说，"大样本时，轻微的偏态问题不大。但严重违反就要处理：对 y 取对数、Box-Cox 转换，或者用 Bootstrap 稳健标准误。"

### 假设 4：等方差（Equal variance，Homoscedasticity）

**如何检查？** 画**残差 vs 拟合值图**。

**理想情况**：残差的宽度在整个拟合值范围内大致恒定——像一个水平的"带子"。

**违反的情况（异方差，Heteroscedasticity）**：残差的宽度随拟合值变化（比如拟合值越大，残差越宽）——像一个"漏斗"或"喇叭"。

"异方差是个大问题。"老潘说，"它会让 p 值和置信区间不可信。解决方法：对 y 取对数、使用稳健标准误（HC3），或者加权最小二乘法（WLS）。"

### 残差图实战

让我们画出残差图和 QQ 图，检查这些假设：

```python
# 画残差图
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 残差 vs 拟合值图
plt.figure(figsize=(8, 6))
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title('Residuals vs Fitted')
plt.show()

# QQ 图
plt.figure(figsize=(8, 6))
stats.probplot(model.resid, dist="norm", plot=plt)
plt.title('Normal Q-Q Plot')
plt.show()
```

老潘的经验法则：**先画图，再做检验**。图比检验更直观，因为检验可能会过于敏感（大样本时）或不够敏感（小样本时）。小偏差可以容忍，严重违反要处理。

阿码问："那怎么算'严重'？"

"没有硬性规则。"老潘说，"但我的经验是：如果残差图有明显的模式（U 型、漏斗型），或者 QQ 图严重偏离对角线，那就该处理了。"

> **AI 时代小专栏：AI 时代的模型诊断——从残差图到 SHAP**

> 2026 年，随着 AI 和机器学习模型的复杂化，传统的回归诊断（残差图、QQ 图）已经不够用了。**可解释 AI（Explainable AI, XAI）** 成为新的标准——Google、Microsoft 等公司的 AI 伦理指南都要求高风险领域的 AI 模型必须提供可解释性。
>
> **传统诊断 vs XAI**
>
> | 维度 | 传统诊断（回归） | XAI（机器学习） |
> |------|------------------|-----------------|
> | 诊断对象 | 线性回归 | 神经网络、随机森林等复杂模型 |
> | 诊断工具 | 残差图、QQ 图、Cook's 距离 | SHAP、LIME、Permutation Importance |
> | 诊断目标 | 检查假设是否满足 | 理解模型如何做预测 |
>
> **SHAP（SHapley Additive exPlanations）**
>
> SHAP 是 2026 年最流行的可解释 AI 工具之一，在 GitHub 上已有超过 25k stars：
> - 它基于博弈论（Shapley value）
> - 它可以告诉你：每个特征对预测的贡献有多大
> - 它适用于任何模型（线性回归、神经网络、随机森林）
> - 有两种解释层次：局部解释（单个预测）和全局解释（整体模型行为）
>
> **2024-2026 年的发展趋势**：
>
> - **GPU 加速**：NVIDIA 推出的 GPU 加速 SHAP，让大规模数据的解释速度提升 10-100 倍
> - **临床应用**：医疗领域的预测模型开始大规模采用 SHAP 进行可解释性分析
> - **深度学习集成**：PyTorch 和 TensorFlow 都有了 SHAP 的原生支持
> - 但 SHAP 不能替代传统诊断——它告诉你"特征的重要性"，但不告诉你"模型假设是否满足"
>
> **对你的启示**：
>
> 你本周学的残差图、QQ 图、Cook's 距离，是 XAI 的基础。在 AI 时代，**传统诊断 + XAI = 完整的模型评估**。如果你将来要用神经网络或随机森林，SHAP 是必须掌握的工具。但记住：SHAP 不替代假设检查，它是补充。
>
> 参考（访问日期：2026-02-16）：
> - [机器学习中的可解释性：SHAP值及其应用 - 腾讯云](https://cloud.tencent.com/developer/article/2560431) (Aug 26, 2025)
> - [使用GPU加速SHAP解释机器学习模型 - NVIDIA开发者](https://developer.nvidia.cn/blog/explain-your-machine-learning-model-predictions-with-gpu-accelerated-shap/) (Oct 5, 2022)
> - [SHAP可视化方法，临床预测模型解释新框架](https://www.medsta.cn/archives/hi-xin-kuang-jia) (Aug 4, 2025)

---

## 4. 模型诊断——当模型告诉你"有问题"时

小北学会了画残差图，他发现自己的残差图有一个点离 0 很远。

"这个点是异常值吗？"小北问，"我应该删掉它吗？"

老潘摇头："别急着删。我们先来看看模型诊断的四个工具——每个工具回答不同的问题。"

**诊断工具速览**：

| 工具 | 检查什么 | 问题形式 |
|------|---------|---------|
| 残差图 + LOWESS | 线性、等方差 | "残差有没有模式？" |
| QQ 图 | 正态性 | "残差是不是正态分布？" |
| Cook's 距离 | 高影响点 | "哪个点对模型影响最大？" |
| Breusch-Pagan | 同方差（统计检验） | "方差是不是恒定的？" |

老潘说："这四个工具不需要全部用上，但**残差图是底线**——没有残差图的回归分析，就像没有心电图的健康体检。"

现在我们逐个来看。

---

<!--
**Bloom 层次**：分析
**学习目标**：能用残差图、QQ图、Cook's 距离诊断模型问题
**贯穿案例推进**：从"拟合模型"到"残差诊断 + 异常点检查"
**建议示例文件**：04_model_diagnostics.py
**叙事入口**：从"小北发现残差图有异常点"开头
**角色出场**：老潘提醒"不要急着删异常值"
**回顾桥**：[异常值检测]（week_03）：通过"Cook's 距离识别高影响点"再次使用
-->

### 诊断工具 1：残差图 + LOWESS 线

你在上一节已经学过：残差图可以检查线性、等方差假设。这里补充一个技巧：**LOWESS 线**（Locally Weighted Scatterplot Smoothing）——它是残差的平滑曲线。

如果 LOWESS 线不是水平直线，说明违反了线性或等方差假设。

```python
# 残差图 + LOWESS 线
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={'color': 'red', 'lw': 2})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title('Residuals vs Fitted (with LOWESS)')
plt.show()
```

**如何解读？**
- LOWESS 线水平且靠近 0：假设满足
- LOWESS 线有趋势（向上或向下）：违反线性假设
- LOWESS 线呈 U 型或倒 U 型：违反线性假设（需要加入二次项）
- 残差的宽度变化：违反等方差假设

老潘说："LOWESS 线就像'温度计'——它能告诉你模型哪里出了问题。如果线是水平的，假设满足；如果线有趋势或弯曲，就要处理了。"

### 诊断工具 2：QQ 图

QQ 图可以检查正态性假设。**如何解读？**
- 点大致落在对角线上：正态性满足
- 点偏离对角线（两端偏离）：残差有长尾
- 点呈 S 型或反 S 型：残差有偏态

```python
from scipy import stats
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
stats.probplot(model.resid, dist="norm", plot=plt)
plt.title('Normal Q-Q Plot')
plt.show()
```

**小样本 vs 大样本**：小样本时，QQ 图可能不太可靠，要结合 Shapiro-Wilk 检验；大样本时，QQ 图更可靠，但 Shapiro-Wilk 可能过于敏感（任何轻微偏离都会显著）。

### 诊断工具 3：Cook's 距离——识别高影响点

**Cook's 距离**（Cook's Distance）衡量每个观测对回归系数的影响。Cook's 距离大的点：删除它会显著改变回归系数。经验法则：Cook's 距离 > 4/n（n 是样本量）的点要关注。

但要注意：**高影响点 ≠ 异常值**。这两个概念经常被混淆，但它们衡量的是不同的东西。

**举一个具体的例子**：

假设你在分析"工作年限 vs 薪资"的回归，数据点是：
- 大部分人：工作 2-10 年，薪资 10-30 万
- 小张：工作 3 年，薪资 100 万（可能是高管亲戚）
- 老王：工作 40 年，薪资 25 万

| 观测 | 特征 | 是异常值吗？ | 是高影响点吗？ |
|------|------|-------------|---------------|
| 小张 | y 值极端高 | ✅ 是（残差大） | ✅ 是（会拉高斜率） |
| 老王 | x 值极端高 | ❌ 不是（残差正常） | ✅ 是（杠杆高） |

小张是异常值，也是高影响点——删除他，斜率会明显下降。老王不是异常值（他的薪资符合预期），但他是高影响点——因为他的 x 值（40年）远离其他人的 x 值（2-10年），所以他对回归线的"拉力"很大。

**简单记忆**：
- **异常值**：y 值离群（在纵轴方向偏离）
- **高杠杆点**：x 值离群（在横轴方向偏离）
- **高影响点**：可能是异常值、可能是高杠杆点、可能是两者的结合——关键看"删除它会不会改变回归线"

```python
# 计算 Cook's 距离
influence = model.get_influence()
cooks_d = influence.cooks_distance[0]

# 画 Cook's 距离图
plt.figure(figsize=(8, 6))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=',')
plt.axhline(y=4/len(cooks_d), color='red', linestyle='--', label='Threshold (4/n)')
plt.xlabel('Observation index')
plt.ylabel("Cook's distance")
plt.title("Cook's Distance")
plt.legend()
plt.show()
```

**如何处理高影响点？** 老潘给了决策流程：
1. **检查是否录入错误**：如果是，修正或删除
2. **检查是否真实但极端**：如果是，保留并报告
3. **删除后重新拟合**：比较删除前后的系数，看结论是否改变
4. **稳健回归**：使用对异常值不敏感的方法（如 RLM）

阿码问："如果我删掉这个点，R² 从 0.75 变成 0.85，我该不该删？"

"不该。"老潘说，"如果你为了提高 R² 而删点，这是数据造假。你应该报告：有一个高影响点，删除后结论是 X，保留后结论是 Y。"

**三个概念的总结**：

| 概念 | 定义 | 识别方法 | 处理方式 |
|------|------|---------|---------|
| **异常值** | y 值远离其他点 | 残差大（|残差| > 2×SD） | 检查是否录入错误 |
| **高杠杆点** | x 值远离其他点 | 杠杆值高（h > 2×(k+1)/n） | 关注但不一定删除 |
| **高影响点** | 删除它会改变系数 | Cook's D > 4/n | 检查原因，报告敏感性 |

简单记忆：异常值在纵轴方向"跳"，高杠杆点在横轴方向"跳"，高影响点是"能左右回归线"的点。

### 诊断工具 4：同方差检验（Breusch-Pagan 检验）

**Breusch-Pagan 检验**（BP test）用于检验等方差假设：
- **零假设（H0）**：等方差（残差方差恒定）
- **备择假设（H1）**：异方差（残差方差变化）

```python
from statsmodels.stats.diagnostic import het_breuschpagan

# Breusch-Pagan 检验
bp_test = het_breuschpagan(model.resid, model.model.exog)
bp_statistic, bp_pvalue, _, _ = bp_test

print(f"BP 统计量: {bp_statistic:.4f}")
print(f"BP p 值: {bp_pvalue:.4f}")
```

**如何解读？**
- p < 0.05：拒绝 H0，存在异方差
- p ≥ 0.05：不能拒绝 H0，等方差假设满足

**如果存在异方差怎么办？** 使用稳健标准误（HC3）、对 y 取对数，或者加权最小二乘法（WLS）。

### 诊断总结

老潘总结道："模型诊断不是'找茬'，而是'了解你的模型在哪些地方可靠、哪些地方不可靠'。残差图告诉你：模型在哪些地方'没讲真话'。Cook's 距离告诉你：哪些点'绑架'了你的模型。BP 检验告诉你：p 值是否可信。"

"如果诊断有问题，不要慌。"老潘继续说，"先尝试简单的修复（对 y 取对数、删除高影响点），如果还不行，再考虑复杂的方法（稳健回归、Bootstrap）。记住：完美的模型不存在，你要做的是诚实地报告模型的局限。"

这正是下一节要讲的内容：多元回归——当影响因素不止一个时，问题会变得更复杂。

---

## 5. 多元回归——当影响因素不止一个时

小北本周学了不少东西，他总结道："所以现在我能拟合一个简单的回归模型（一个 x 预测 y），检查假设，诊断问题。但如果我有多个 x 怎么办？"

老潘笑了："这正是多元回归要解决的问题。"

---

<!--
**Bloom 层次**：应用
**学习目标**：理解多元回归的原理，能识别和解决多重共线性问题
**贯穿案例推进**：从"简单回归"到"多元回归 + 多重共线性 + 交互项"
**建议示例文件**：05_multiple_regression.py
**叙事入口**：从"小北问多个 x 怎么办"开头
**角色出场**：阿码追问"多元回归的系数和简单回归的系数有什么区别？"
**回顾桥**：[Bootstrap]（week_08）：通过"Bootstrap 稳健回归"再次使用；[方差分解]（week_07）：通过"多元回归的方差分解"再次使用
-->

### 从简单回归到多元回归

**简单线性回归**：y = a + bx + ε（一个自变量 x）

**多元线性回归**：y = a + b₁x₁ + b₂x₂ + ... + bₖxₖ + ε（多个自变量 x₁, x₂, ..., xₖ）

阿码问："多元回归的系数和简单回归的系数有什么区别？"

"好问题。"老潘说，"多元回归的系数是'控制其他变量后'的影响。这和简单回归的系数很不一样。"

举个例子：
- 简单回归：广告投入对销售额的影响（b = 0.5）
- 多元回归：控制价格、促销后，广告投入对销售额的影响（b = 0.3）

"为什么系数变了？"小北问。

"因为广告投入和价格相关。"老潘说，"简单回归中，广告投入的系数'吸收'了价格的影响。多元回归中，价格的影响被单独分出来，广告投入的系数变得更'纯粹'。这就是'控制其他变量后'的含义。"

### 多重共线性：当自变量之间"打架"时

**多重共线性**（Multicollinearity）是指自变量之间高度相关。这会导致：
- 系数不稳定（小样本变化导致系数大变化）
- p 值不可靠（即使 x 重要，p 值也可能 > 0.05）
- R² 很高，但单个系数都不显著

**如何检测？**
1. **相关矩阵**：看自变量之间的相关系数（> 0.7 或 < -0.7 要关注）
2. **VIF**（Variance Inflation Factor）：VIF > 10 表示严重多重共线性

```python
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 假设你有三个自变量：ad_spend, price, promotion
X = pd.DataFrame({'ad_spend': ad_spend, 'price': price, 'promotion': promotion})

# 相关系数矩阵
corr_matrix = X.corr()
print(corr_matrix)

# 计算 VIF
X_with_const = sm.add_constant(X)
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
vif["variable"] = X_with_const.columns
print(vif)
```

**如何解决多重共线性？**
1. **删除高度相关的变量**：如果两个变量相关系数 > 0.9，删除一个
2. **正则化**（L1/L2）：Week 11 会讲
3. **降维方法**（如 PCA）：后续专题会讲，核心思想是把多个相关变量压缩成少数几个不相关的"成分"

### 交互项：当影响依赖于另一个变量时

**交互项**（Interaction）是指一个变量对 y 的影响依赖于另一个变量。举个例子：广告投入对销售额的影响，可能依赖于价格——如果价格很高，广告的效果可能很差。

```python
# 创建交互项
X['ad_spend_x_price'] = X['ad_spend'] * X['price']

# 拟合带交互项的模型
X_with_const = sm.add_constant(X)
model_interaction = sm.OLS(y, X_with_const).fit()
print(model_interaction.summary())
```

**如何解读交互项系数？**
- 如果交互项系数为负：ad_spend 的效果随 price 增加而减弱
- 如果交互项系数为正：ad_spend 的效果随 price 增加而增强

阿码问："我什么时候应该加交互项？"

老潘给了两个标准：
1. **理论支持**：你有理由相信两个变量会互相影响
2. **图示支持**：画分组散点图，看不同组别的回归线是否平行

"交互项是个强大的工具，但不要滥用。"老潘说，"只有在理论或数据支持时才加，否则会增加模型复杂度，导致过度拟合。"

### 模型选择：哪个模型更好？

当你有多个候选模型时，如何选择？

| 指标 | 含义 | 优点 | 缺点 |
|------|------|------|------|
| **R²** | 模型解释的方差比例 | 直观 | 会随特征增加而增加 |
| **调整 R²** | 惩罚模型复杂度的 R² | 考虑了特征数量 | 仍可能过度拟合 |
| **AIC** | 赤池信息准则 | 越小越好，惩罚复杂度 | 绝对值无意义 |
| **BIC** | 贝叶斯信息准则 | 比 AIC 更惩罚复杂度 | 更保守 |

老潘的经验法则：
- **预测优先**：用交叉验证（Week 10 会讲）
- **解释优先**：选调整 R² 高、系数显著的模型
- **简洁原则**：在拟合相近时，选更简单的模型（更少特征）

"记住：没有'最好的模型'，只有'最适合你的问题的模型'。"老潘说，"如果你的目标是预测，用交叉验证；如果你的目标是解释，选简单的模型。"

小北若有所思："所以多元回归不是'越多变量越好'？"

"对。"老潘说，"多元回归是'权衡的艺术'。你要在拟合度、复杂度、可解释性之间找平衡。这需要经验，也需要诊断。"

---

## StatLab 进度

到目前为止，StatLab 已经有了一个"不确定性量化"的报告：点估计 + 置信区间 + Bootstrap + 置换检验。但这里有一个"看不见的坑"：我们在报告里只做了"比较差异"，但没有回答"哪些因素在影响这个差异"。

这正是本周"回归分析与模型诊断"派上用场的地方。让我们来写一个函数，做回归分析并输出诊断报告：

```python
# examples/09_statlab_regression.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

def regression_with_diagnostics(y, X, var_names=None, confidence=0.95):
    """
    拟合回归模型并输出诊断报告

    参数:
    - y: 因变量（array-like）
    - X: 自变量矩阵（array-like, 每列是一个自变量）
    - var_names: 自变量名称列表（可选）
    - confidence: 置信水平（默认 0.95）

    返回:
    - dict: 包含模型、诊断结果、图表数据的字典
    """
    # 转换为 DataFrame
    if var_names is None:
        var_names = [f"x{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=var_names)

    # 添加截距
    X_with_const = sm.add_constant(X_df)

    # 拟合 OLS 模型
    model = sm.OLS(y, X_with_const).fit()

    # 1. 提取模型结果
    results = {
        "model": model,
        "summary": model.summary(),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_pvalue": model.f_pvalue,
    }

    # 2. 置信区间
    ci = model.conf_int(alpha=1-confidence)
    ci.columns = ['lower', 'upper']
    results["confidence_intervals"] = ci

    # 3. 残差分析
    residuals = model.resid
    fitted = model.fittedvalues

    # 3.1 正态性检验（Shapiro-Wilk）
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    results["normality_test"] = {
        "statistic": shapiro_stat,
        "p_value": shapiro_p,
        "is_normal": shapiro_p > 0.05
    }

    # 3.2 同方差检验（Breusch-Pagan）
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
    results["homoscedasticity_test"] = {
        "statistic": bp_stat,
        "p_value": bp_p,
        "is_homoscedastic": bp_p > 0.05
    }

    # 4. 高影响点（Cook's 距离）
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(cooks_d)
    high_influence = np.where(cooks_d > threshold)[0]

    results["high_influence_points"] = {
        "cook_distances": cooks_d,
        "threshold": threshold,
        "high_influence_indices": high_influence.tolist(),
        "n_high_influence": len(high_influence)
    }

    # 5. 多重共线性（VIF）——如果是多元回归
    if X.shape[1] > 1:
        vif_data = []
        for i in range(1, X_with_const.shape[1]):  # 跳过截距
            vif = variance_inflation_factor(X_with_const.values, i)
            vif_data.append({
                "variable": var_names[i-1],
                "vif": vif,
                "is_high_collinear": vif > 10
            })
        results["multicollinearity"] = vif_data

    # 6. 图表数据
    results["plots"] = {
        "residuals_vs_fitted": {"x": fitted, "y": residuals},
        "qq_plot": {"residuals": residuals},
        "cook_distance": {"index": np.arange(len(cooks_d)), "cook_d": cooks_d, "threshold": threshold},
    }

    return results

def plot_diagnostics(diag_results, figsize=(15, 10)):
    """
    画诊断图表
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. 残差 vs 拟合值
    plot_data = diag_results["plots"]["residuals_vs_fitted"]
    axes[0, 0].scatter(plot_data["x"], plot_data["y"], alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')

    # 2. QQ 图
    residuals = diag_results["plots"]["qq_plot"]["residuals"]
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q Plot')

    # 3. Cook's 距离
    cook_data = diag_results["plots"]["cook_distance"]
    axes[1, 0].stem(cook_data["index"], cook_data["cook_d"], markerfmt=',')
    axes[1, 0].axhline(y=cook_data["threshold"], color='red', linestyle='--', label='Threshold')
    axes[1, 0].set_xlabel('Observation index')
    axes[1, 0].set_ylabel("Cook's distance")
    axes[1, 0].set_title("Cook's Distance")
    axes[1, 0].legend()

    # 4. 残差直方图
    axes[1, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Histogram of Residuals')

    plt.tight_layout()
    return fig

def format_regression_report(diag_results, confidence=0.95):
    """
    格式化回归结果为 Markdown 报告
    """
    model = diag_results["model"]

    md = ["## 回归分析\n\n"]

    # 1. 模型摘要
    md.append("### 模型拟合摘要\n\n")
    md.append(f"| 指标 | 值 |\n")
    md.append(f"|------|-----|\n")
    md.append(f"| R² | {diag_results['r_squared']:.4f} |\n")
    md.append(f"| 调整 R² | {diag_results['adj_r_squared']:.4f} |\n")
    md.append(f"| F 统计量 | {diag_results['f_statistic']:.4f} |\n")
    md.append(f"| F 检验 p 值 | {diag_results['f_pvalue']:.4f} |\n\n")

    # 2. 回归系数
    md.append("### 回归系数\n\n")
    md.append(f"| 变量 | 系数 | 标准误 | t 值 | p 值 | {int(confidence*100)}% CI |\n")
    md.append(f"|------|------|--------|------|------|-------|\n")

    ci = diag_results["confidence_intervals"]
    for idx, row in model.params.items():
        se = model.bse[idx]
        t = model.tvalues[idx]
        p = model.pvalues[idx]
        lower = ci.loc[idx, 'lower']
        upper = ci.loc[idx, 'upper']
        md.append(f"| {idx} | {row:.4f} | {se:.4f} | {t:.4f} | {p:.4f} | [{lower:.4f}, {upper:.4f}] |\n")
    md.append("\n")

    # 3. 假设检验
    md.append("### 假设检验\n\n")

    # 正态性
    norm = diag_results["normality_test"]
    md.append(f"**正态性检验（Shapiro-Wilk）**：\n")
    md.append(f"- 统计量：{norm['statistic']:.4f}\n")
    md.append(f"- p 值：{norm['p_value']:.4f}\n")
    if norm['is_normal']:
        md.append(f"- 结论：p > 0.05，不能拒绝正态性假设 ✅\n\n")
    else:
        md.append(f"- 结论：p < 0.05，拒绝正态性假设 ⚠️\n\n")

    # 同方差
    homo = diag_results["homoscedasticity_test"]
    md.append(f"**同方差检验（Breusch-Pagan）**：\n")
    md.append(f"- 统计量：{homo['statistic']:.4f}\n")
    md.append(f"- p 值：{homo['p_value']:.4f}\n")
    if homo['is_homoscedastic']:
        md.append(f"- 结论：p > 0.05，不能拒绝同方差假设 ✅\n\n")
    else:
        md.append(f"- 结论：p < 0.05，拒绝同方差假设 ⚠️\n\n")

    # 4. 高影响点
    md.append("### 高影响点（Cook's 距离）\n\n")
    inf = diag_results["high_influence_points"]
    md.append(f"- Cook's 距离阈值：{inf['threshold']:.4f}\n")
    md.append(f"- 高影响点数量：{inf['n_high_influence']}\n")
    if inf['n_high_influence'] > 0:
        md.append(f"- 高影响点索引：{inf['high_influence_indices']}\n")
        md.append(f"- ⚠️ 建议：检查这些数据点是否为录入错误或极端值\n\n")
    else:
        md.append(f"- ✅ 未发现高影响点\n\n")

    # 5. 多重共线性（如果有）
    if "multicollinearity" in diag_results:
        md.append("### 多重共线性（VIF）\n\n")
        md.append(f"| 变量 | VIF | 诊断 |\n")
        md.append(f"|------|-----|------|\n")
        for vif in diag_results["multicollinearity"]:
            diagnosis = "⚠️ VIF > 10" if vif["is_high_collinear"] else "✅"
            md.append(f"| {vif['variable']} | {vif['vif']:.2f} | {diagnosis} |\n")
        md.append("\n")

    return "".join(md)

# 使用示例
import seaborn as sns

# 加载数据
penguins = sns.load_dataset("penguins")
penguins = penguins.dropna()

# 简单回归：喙长度预测喙深度
X_simple = penguins[["bill_length_mm"]].values
y = penguins["bill_depth_mm"].values

# 拟合模型
diag_simple = regression_with_diagnostics(y, X_simple, var_names=["喙长度"])

# 生成报告
report_simple = format_regression_report(diag_simple)
print(report_simple)

# 画诊断图
fig = plot_diagnostics(diag_simple)
plt.show()

# 多元回归：喙长度 + 翼展预测喙深度
X_multiple = penguins[["bill_length_mm", "flipper_length_mm"]].values
diag_multiple = regression_with_diagnostics(y, X_multiple, var_names=["喙长度", "翼展"])

# 生成报告
report_multiple = format_regression_report(diag_multiple)
print(report_multiple)

# 写入文件
from pathlib import Path
Path("output/regression_report.md").parent.mkdir(parents=True, exist_ok=True)
Path("output/regression_report.md").write_text(report_multiple)
print("\n报告已保存到 output/regression_report.md")
```

现在 `report.md` 会多出一个"回归分析"章节，包括：
- 模型拟合摘要（R²、调整 R²、F 检验）
- 回归系数表格（系数、标准误、t 值、p 值、置信区间）
- 假设检验（正态性、同方差）
- 高影响点诊断（Cook's 距离）
- 多重共线性诊断（VIF）
- 诊断图表（残差图、QQ 图、Cook's 距离图、残差直方图）

### 与本周知识的连接

**简单线性回归** → 我们不仅描述了两个变量的关系，还量化了"每单位 x 变化带来的 y 变化"。

**回归系数解释** → 我们学会了解读斜率和截距，理解了"控制其他变量后"的含义。

**回归假设** → 我们学会了检查 LINE 假设，而不是盲目相信高 R²。

**模型诊断** → 我们用残差图、QQ 图、Cook's 距离、VIF 等工具诊断模型问题，而不是"拟合完就完事"。

### 与上周的对比

| 上周 | 本周 |
|------|------|
| 比较差异（p 值 + CI） | 解释影响（回归系数 + 假设检查） |
| 检验"有没有差异" | 检验"x 如何影响 y" |
| 量化不确定性（CI） | 量化关系（系数）+ 验证假设（诊断） |

老潘看到这段改动会说什么？"这才是分析。你不仅告诉了读者'有什么关系'，还验证了'这个模型是否可信'。没有诊断的回归不是分析，是自欺欺人。"

小北问："自欺欺人？"

"对。"老潘说，"如果你只报告 R² 和系数，但不检查假设，你可能在误导自己。残差图告诉你：模型在哪些地方'没讲真话'。"

阿码若有所思："所以 AI 工具如果只给我 R² 和系数，不给我诊断……"

"那你就要补上。"老潘说，"AI 可以帮你拟合模型，但只有你能判断'这个模型是否可信'。"

---

## Git 本周要点

本周必会命令：
- `git status`（查看未跟踪的新文件：回归分析脚本、诊断图表）
- `git diff`（查看对 StatLab 报告生成脚本的修改）
- `git add -A`（添加所有变更）
- `git commit -m "draft: add regression analysis and diagnostics"`（提交回归分析）

常见坑：
- 只报告 R² 和系数，不做假设检查。这是 2026 年的"不合格回归分析"；
- 看到 Cook's 距离大的点就删除，不检查是否为录入错误；
- 多元回归不检查 VIF，导致多重共线性问题；
- 混淆"相关"和"回归"，误以为高相关就是因果关系。

老潘的建议：**残差图是底线**。AI 可以拟合模型，但只有你能判断"这个模型是否满足假设"。

---

## Definition of Done（学生自测清单）

本周结束后，你应该能够：

- [ ] 解释"相关"和"回归"的区别，说明它们分别回答什么问题
- [ ] 正确解读回归系数（斜率和截距），理解"控制其他变量后"的含义
- [ ] 理解 R² 的含义和局限，知道"高 R² 不等于好模型"
- [ ] 列出并检查回归的 LINE 假设（线性、独立性、正态性、等方差）
- [ ] 用残差图、QQ 图、Cook's 距离诊断模型问题
- [ ] 检测多重共线性（VIF），并知道如何解决
- [ ] 写出"带诊断检查"的回归报告

---

## 本周小结（供下周参考）

老潘最后给了一个比喻："回归分析就像看病。R² 是你的体检报告——'各项指标都正常'。残差图是你的 CT 片——能看到体检报告上看不到的东西。Cook's 距离是你的关键指标——哪些病人需要特别关注。"

"只看体检报告不看 CT 片，不是好医生。只看 R² 不看残差图，不是好分析师。"

这周你学会了**回归分析与模型诊断**：从"画散点图看相关性"到"拟合回归模型量化关系"，从"只看 R²"到"检查 LINE 假设"。

你理解了**回归分析的本质**：它不仅告诉你"x 和 y 有没有关系"，还告诉你"y 如何随 x 变化"。你学会了解读回归系数：斜率表示"x 每增加 1 单位，y 的变化量"，截距表示"x = 0 时 y 的预测值"。

你掌握了 **LINE 假设**：线性、独立性、正态性、等方差。你学会了用残差图、QQ 图、Breusch-Pagan 检验来检查这些假设是否满足——假设不满足时，p 值和置信区间都不可信。

更重要的是，你学会了**模型诊断**：用 Cook's 距离识别高影响点，用 VIF 检测多重共线性，用调整 R²、AIC、BIC 比较模型。你知道"高 R² 不等于好模型"，残差图比 R² 更重要。

最后，你现在能**写出"带诊断检查"的回归报告**。你知道"没有诊断的回归不是分析，是自欺欺人"，学会了用残差图、QQ 图、Cook's 距离等工具来验证模型的可信性。这正是 AI 时代分析者的核心能力：AI 可以拟合模型，但只有你能判断"这个模型是否可信"。

老潘的总结很简洁："模型是错的，但有些是有用的。关键是要知道：你的模型在哪些地方是错的。"

下周，我们将继续深入**预测建模**：分类模型与评估。你本周学的回归诊断、假设检查、模型评估指标，在下周都会用到。下周的核心问题是："准确率高就是好模型吗？"
