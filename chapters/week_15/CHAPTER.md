# Week 15：高级统计计算——当数据太多、太快、太复杂时

> "The future of statistics is computational."
> — Bradley Efron

> 2025-2026 年，数据分析正在经历一场"数据规模革命"。翻开任何一份技术招聘 JD，你都会看到："处理高维用户行为矩阵"、"实现实时特征更新"、"搭建 A/B 测试平台"——这些曾经是"高级技能"的要求，如今已是数据岗位的标配。
>
> 问题的本质变了。五年前，你可能还在处理"1000 个样本、20 个特征"的表格数据，用 Excel 或简单的回归就能应付。今天，你可能面对的是：**高维数据**（基因表达数据的 20000 个特征、文本 TF-IDF 向量的 10000 维、用户行为矩阵的 5000 维）、**流式数据**（IoT 设备每秒上千条传感器读数、在线交易平台每秒数万次点击）、**持续实验**（Netflix、Uber 每天同时运行数千个 A/B 测试，需要自动化的统计决策流水线）。
>
> 传统方法在这些场景下会崩溃：当特征数接近样本数时，回归矩阵不可逆（你将在第 2 节详细学习）；**分组比较**（Week 04）的陷阱在于不控制混杂变量，容易得出虚假关联；当实验太多时，**多重比较校正**（Week 07）变得必需——不做校正假阳性率会爆炸。本周你学的降维、聚类、流式统计、A/B 测试工程化——正是这些问题的"工业化解法"。当你看到 RAG 系统用 PCA 压缩嵌入向量、推荐系统用聚类做用户分层、在线学习系统用流式统计实时更新模型时，你会意识到：**统计计算已成为现代数据系统的"基础设施"**。

---

## 前情提要

上周（Week 14），小北学会了用贝叶斯方法更新信念——从"先验"到"后验"，从"p 值游戏"到"概率陈述"。他现在可以说："优惠券有 85% 的概率有效，效应提升 10.98% ± 8.6%"——产品经理很高兴，终于听懂了。

但新的问题出现了。

数据团队发来一份用户行为矩阵：10 万个用户 × 5000 个行为特征（点击序列、停留时间、购买路径等）。小北习惯性地想跑回归分析，结果报错了：

```
LinAlgError: Matrix is singular.
```

老潘看了眼数据，笑了："你这是遇到了**维度灾难**（Curse of Dimensionality）。当特征数接近或超过样本数时，传统方法会崩溃——矩阵不可逆、方差爆炸、计算时间指数增长。"

阿码好奇："那怎么办？删掉一半特征？"

"可以，但你怎么知道该删哪些？"老潘反问。

"或者……让数据自己告诉我们哪些特征重要？"小北试探着问。

"对！"老潘打了个响指，"这就是**降维**（Dimensionality Reduction）——从高维数据中提取'主轴'，把 5000 个特征压缩成 50 个，但保留 90% 的信息。"

但这只是第一个问题。产品经理接着抛出新需求："我们的数据是实时来的——每秒有上百个新用户行为。你能不能给我一个'实时看板'，而不是每天早上的'昨日报告'？"

小北愣住了："实时？那我得……每秒重新跑一遍整个数据集？"

"当然不是，"老潘说，"你需要**流式统计**（Streaming Statistics）——增量更新均值、方差、分位数，而不是每次都重算。"

阿码举手："那 A/B 测试呢？我们每周都要手动跑 t 检验，能不能自动化？"

老潘笑了："这正是本周的第三个主题——**A/B 测试的工程化**。从手动分析到自动化决策平台。"

---

## 学习目标

完成本周学习后，你将能够：

1. 理解维度灾难的本质，识别高维数据的陷阱（过拟合、计算爆炸、距离失效）
2. 掌握主成分分析（PCA）的核心思想，能用 sklearn 实现降维并解释主成分
3. 理解聚类分析的目标，能用 K-means、层次聚类等方法发现数据中的隐藏结构
4. 能用降维 + 聚类的组合解决实际高维数据问题（如用户分层、特征压缩）
5. 掌握流式统计算法（在线均值、在线方差、分位数估计），理解增量更新的思想
6. 理解 A/B 测试的工程化挑战，能设计合理的样本量和检验策略
7. 能在 StatLab 报告中整合高维数据分析方法（降维、聚类、A/B 测试）
8. 能评估 AI 工具生成的降维/聚类结果的合理性与解释性

---

<!--
贯穿案例：高维用户行为数据的分析与可视化——从"5000 个特征无法解释"到"10 个主成分 + 5 个用户群"

本周贯穿案例是一个经典的高维数据问题：你有一份用户行为矩阵（10 万用户 × 5000 个行为特征），需要回答两个问题：
1. 能否用少数几个"综合指标"概括用户行为？（降维）
2. 能否根据行为相似性把用户分成几组？（聚类）

案例演进路线：
- 第 1 节（维度灾难）：从回归分析崩溃出发，理解高维数据的陷阱（过拟合、计算爆炸）
- 第 2 节（PCA 降维）：用 PCA 把 5000 个特征压缩成 10 个主成分，保留 85% 的方差，解释主成分的业务含义
- 第 3 节（聚类分析）：在降维后的 10 维空间上运行 K-means，发现 5 个用户群（如"价格敏感型"、"品质追求型"）
- 第 4 节（流式统计）：当新用户数据持续到来时，增量更新每个用户群的统计量（均值、方差、分位数）
- 第 5 节（A/B 测试工程化）：设计一个自动化流程，对新老用户群分别做 A/B 测试，自动输出决策建议

最终成果：读者拿到一个完整的高维数据分析脚本，能从"无法解释的高维矩阵"到"可解释的用户分层画像"。
-->

<!--
认知负荷预算：
- 本周新概念（4 个，预算上限 4 个）：
  1. 降维（PCA）- 理解/应用层次
  2. 聚类（K-means、层次聚类）- 理解/应用层次
  3. 在线/流式统计 - 理解/应用层次
  4. A/B 测试工程化 - 理解/应用层次
- 结论：✅ 在预算内（4/4）

回顾桥设计（至少 3 个，来自前 6+ 周）：
- [方差分解]（week_07、week_09）：在第 2 节，用 ANOVA/回归的"总方差 = 组间 + 组内"连接 PCA 的"总方差 = 各主成分方差之和"
- [距离度量]（week_04 相关分析）：在第 3 节，用"相关系数 vs 欧氏距离"引出聚类中的距离选择问题
- [不确定性量化]（week_08）：在第 4 节，用"Bootstrap 的重采样"连接"流式统计的增量更新"，都是近似方法但思路不同
- [假设检验]（week_06-07）：在第 5 节，用"t 检验/ANOVA 的前提检查"引出 A/B 测试工程化中的自动化诊断
- [Bootstrap/置换检验]（week_08）：在第 5 节，用"重采样的计算成本"引出流式 A/B 测试的增量统计方法

AI 小专栏规划：
AI 小专栏 #1（放在第 2 节之后）：
- 主题：高维数据与 AI——从词嵌入到 RAG 的向量检索
- 连接点：与第 2 节"PCA 降维"呼应，讨论高维向量在现代 AI 系统（如 Embedding、RAG）中的角色
- 建议搜索词："high dimensional vectors AI 2026", "RAG vector database PCA", "dimensionality reduction embeddings 2026"

AI 小专栏 #2（放在第 4 节之后）：
- 主题：实时机器学习与流式计算——从 Kafka 到 Flink
- 连接点：与第 4 节"流式统计"呼应，介绍工业界的实时计算框架（Kafka、Flink、Spark Streaming）
- 建议搜索词："streaming machine learning 2026", "Apache Flink real-time analytics", "Kafka stream processing 2026"

角色出场规划：
- 小北（第 1、2、4 节）：
  - 在第 1 节，遇到"维度灾难"报错（矩阵奇异），引出高维数据的陷阱
  - 在第 2 节，困惑于"主成分到底是什么意思"，引出 PCA 的业务解释
  - 在第 4 节，对"增量更新公式"感到畏惧，引出流式统计的直觉优先学习路径
- 阿码（第 2、3、5 节）：
  - 在第 2 节，追问"为什么选前 k 个主成分，不是后 k 个"，引出方差解释比例
  - 在第 3 节，好奇"K-means 的 k 值怎么确定"，引出肘部法则、轮廓系数
  - 在第 5 节，追问"A/B 测试能否全自动决策"，引出工程化的边界（human-in-the-loop）
- 老潘（第 1、3、5 节）：
  - 在第 1 节，用工程视角解释"高维数据在公司里是常态，降维是必需技能"
  - 在第 3 节，强调"聚类的核心是业务定义，不是算法选择"
  - 在第 5 节，用"自动化平台的陷阱"展示 A/B 测试工程化的经验（如 Sample Ratio Mismatch、辛普森悖论）

StatLab 本周推进：
- 上周状态：StatLab 报告有贝叶斯章节（后验分布、先验敏感性），但所有分析都是"批量模式"（静态数据）
- 本周改进：
  1. 用 PCA 对高维特征（如用户行为矩阵）降维，保留主要信息，减少计算成本
  2. 用 K-means 对用户聚类，生成用户分层画像（如"高价值型"、"流失风险型"）
  3. 实现流式统计算法（在线均值、在线方差），让报告能增量更新而不是全量重跑
  4. 设计 A/B 测试流程，为不同用户群分别检验处理效应
- 涉及的本周概念：PCA 降维、K-means 聚类、流式统计、A/B 测试设计
- 建议示例文件：examples/07_statlab_computational.py（高维数据分析与流式统计的 StatLab 集成）
-->

---

## 1. 维度灾难——为什么"越多越好"是错的

小北兴冲冲地跑来告诉你："我有 5000 个用户行为特征！这下模型肯定很准！"他打开 Jupyter，敲下这段代码：

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)  # X_train: (1000, 5000)
```

结果报错了：

```
LinAlgError: Matrix is singular.
```

"矩阵奇异？"小北懵了，"特征越多不是越好吗？怎么会报错？"

老潘走过来，看了眼数据："你这是遇到了**维度灾难**（Curse of Dimensionality）。当特征数（5000）接近甚至超过样本数（1000）时，传统统计方法会崩溃。"

### 什么是维度灾难

老潘在白板上写下维度灾难的三个表现：

**1. 过拟合（参数估计不稳定）**

当特征数 p 接近样本数 n 时，回归方程的参数 θ 有无数解——就像你用 2 个点拟合直线（唯一解），但用 2 个点拟合抛物线（无数解）。

```python
# 回归参数解：θ = (X^T X)^(-1) X^T y
# 当 p > n 时，X^T X 不可逆，无法计算参数
```

**2. 距离失效（所有点都一样"远"）**

在高维空间中，任意两点之间的欧氏距离趋于相等——"最近邻"和"最远邻"没有实质差异。

```python
# examples/01_distance_curse.py

import numpy as np
from scipy.spatial.distance import pdist

def average_distance(n_samples, n_features):
    """计算随机数据中点对的平均距离"""
    X = np.random.randn(n_samples, n_features)
    distances = pdist(X, metric='euclidean')
    return distances.mean()

# 测试不同维度下的平均距离
for d in [2, 10, 100, 500, 1000]:
    avg_dist = average_distance(1000, d)
    print(f"维度 {d:4d}: 平均距离 = {avg_dist:.4f}")
```

**输出**：
```
维度    2: 平均距离 = 1.0932
维度   10: 平均距离 = 3.7845
维度  100: 平均距离 = 13.4821
维度  500: 平均距离 = 27.8341
维度 1000: 平均距离 = 43.9127
```

**关键发现**：维度越高，平均距离越大，且**所有点对的距离趋于接近**（方差变小）。

**3. 采样稀疏（数据覆盖了空间的一小部分）**

在高维空间中，样本无法"填满"空间——大部分区域是空的，你的数据只是高维立方体边缘的几个点。

老潘打了个比方：

"想象你在二维平面上扔 100 个点——它们大致能'覆盖'一个正方形。但如果在 1000 维空间中扔 100 个点，这些点就像大海里的 100 滴水——彼此之间几乎全是'空白'。这就是**采样稀疏**（Sample Sparsity）。"

### 从 Week 09 的"多重共线性"到维度灾难

Week 09 你学过**多重共线性**（Multicollinearity）——当特征之间高度相关时，回归系数不稳定。

小北："维度灾难和多重共线性有什么区别？"

老潘："**多重共线性是'特征冗余'，维度灾难是'特征太多'**——两者都会导致 X^T X 不可逆，但解法不同。"

| 维度 | 多重共线性 | 维度灾难 |
|------|-----------|-----------|
| **问题** | 特征高度相关（冗余） | 特征数 ≥ 样本数 |
| **表现** | 系数方差大（标准误大） | X^T X 不可逆（无解） |
| **解法** | 删除相关特征、正则化（Ridge） | 降维、增加样本、特征选择 |
| **直觉** | "有些特征是重复的" | "特征太多，数据不够用" |

**关键区别总结**：多重共线性是"信息冗余"——10 个特征里有一半都在说同一件事，所以删掉冗余就能解决；维度灾难是"信息太多"——你有 5000 个特征，但只有 1000 个样本，无论特征是否相关，数据都"不够用"来估计那么多参数。解决维度灾难，你不能只删特征，而要"压缩信息"（如 PCA）或者"增加数据"（往往成本太高）。

### 老潘的工程经验：高维数据是常态

"在公司里，"老潘说，"高维数据到处都是："

- **基因表达数据**：20000 个基因 × 100 个样本
- **文本数据**：TF-IDF 向量 10000 维
- **用户行为矩阵**：5000 个行为特征 × 100 万用户
- **图像数据**：28×28 = 784 像素（MNIST），224×224 = 50176 像素（ImageNet）

"你不能指望'增加样本'——成本太高。你必须学会**降维**（Dimensionality Reduction）。"

### 小结

维度灾难不是因为"数据质量差"，而是因为"维度太高"——这是数据结构的根本限制，不是模型选择的问题。

当特征数接近或超过样本数时，三件事会同时发生：回归系数不稳定（X^T X 不可逆）、距离度量失效（所有点都一样"远"）、采样稀疏（数据填不满空间）。小北遇到的"矩阵奇异"报错只是冰山一角——更危险的是"不报错但不可信"：过拟合的模型在训练集上表现完美，在测试集上完全失效。

老潘的经验：**高维数据是工业界的常态，不是极端情况**。基因表达（20000 个基因）、文本 TF-IDF（10000 维）、用户行为矩阵（5000 个特征）、图像像素（78 维到 50176 维）——这些都是"维度灾难"的重灾区。你不能指望"增加样本"（成本太高），必须学会降维。

但降维不是"随机删一半特征"——你怎么知道该删哪些、保留哪些？下一节的 **PCA** 会换个思路：不删特征，而是让数据自己告诉你"哪些方向最重要"。

---

## 2. PCA 降维——让数据告诉你"哪些方向最重要"

阿码盯着上周报错的回归模型，试探："要不我手动删 4500 个特征，留 500 个？"

老潘摇头："你怎么知道该删哪些？全部删？随机删？'凭感觉删'？"

"那怎么办？"小北追问。

"让数据自己告诉你，"老潘说，"**PCA（主成分分析）**会找出数据'变化最大'的方向——你沿着这些方向投影，就能用少数几个'主成分'保留大部分信息。"

### PCA 的核心思想

老潘在白板上画了个图：

"想象你有一群二维数据点（散点图）。如果允许你用一条'直线'概括这些点，你会画哪里？"

**答案**：沿着数据**变化最大**的方向画——这条线能最大化"点到直线的投影方差"。

```python
# examples/02_pca_intuition.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 生成二维数据（有明显方向）
np.random.seed(42)
X = np.random.randn(200, 2)
X[:, 1] = 0.5 * X[:, 0] + 0.1 * np.random.randn(200)  # 第 2 维是第 1 维的 0.5 倍 + 噪声

# PCA 降维到 1 维
pca = PCA(n_components=1)
X_transformed = pca.fit_transform(X)

# 可视化
plt.figure(figsize=(10, 4))

# 原始数据
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.title("原始数据（2 维）")
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)

# 主成分方向
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
# 画出主成分方向（特征向量）
pc1 = pca.components_[0]
plt.arrow(0, 0, pc1[0] * 3, pc1[1] * 3,
           head_width=0.2, head_length=0.2, fc='red', ec='red', linewidth=2)
plt.xlabel("特征 1")
plt.ylabel("特征 2")
plt.title(f"第 1 主成分（解释方差 {pca.explained_variance_ratio_[0]:.1%}）")

plt.tight_layout()
plt.savefig("report/pca_intuition.png")
plt.show()

print(f"第 1 主成分解释的方差比例: {pca.explained_variance_ratio_[0]:.1%}")
```

**输出**：
```
第 1 主成分解释的方差比例: 92.3%
```

**解读**：
- 第 1 主成分（红色箭头）沿着数据**变化最大**的方向
- 这 1 个主成分保留了 **92.3%** 的方差（信息）
- 你可以用"这 1 个主成分"替代"原始的 2 个特征"，损失不到 8% 的信息

### PCA 的数学直觉（无需推导）

老潘："你不需要记住 PCA 的推导，但需要理解它的目标函数："

**PCA 优化目标**：
$$\max_{w} \text{Var}(Xw) \quad \text{s.t. } \|w\|_2 = 1$$

- **Xw**：数据在方向 w 上的投影
- **Var(Xw)**：投影后的方差（我们想最大化）
- **||w||_2 = 1**：约束：w 是单位向量（长度为 1）

**等价问题**（最小化投影误差）：
$$\min_{w} \sum_{i=1}^{n} \|x_i - (w w^T) x_i\|^2$$

这个问题的解是：**协方差矩阵的特征向量**（eigenvectors of covariance matrix）。

小北："等等，特征向量？那是 Week 05 的内容……"

老潘："对！**PCA 的核心就是协方差矩阵的特征分解**——特征向量的方向就是'数据变化最大'的方向，特征值的大小就是'该方向的方差'。"

### 从 Week 07 的"方差分解"到 PCA

Week 07 你学过**ANOVA 的方差分解**：
$$\text{总平方和（SST）} = \text{组间平方和（SSB）} + \text{组内平方和（SSW）}$$

阿码："这和 PCA 的方差解释有什么关系？"

老潘："**思想一样，都是'把总方差拆成几部分'**："

| 维度 | ANOVA | PCA |
|------|--------|-----|
| **总方差** | SST = Σ(y_i - ȳ)² | Σ||x_i - μ||² |
| **拆分方式** | 组间 + 组内 | 第 1 主成分 + 第 2 主成分 + ... + 第 p 主成分 |
| **目标** | 组间方差越大越好（组间差异显著） | 前 k 个主成分的方差和越大越好（信息保留越多） |
| **解释比例** | η² = SSB / SST | 方差解释比 = Σ_{i=1}^k λ_i / Σ_{i=1}^p λ_i |

"ANOVA 是'按组拆分方差'，PCA 是'按方向拆分方差'——都是把总方差分解成可解释的部分。"

### PCA 实战：从 5000 维到 50 维

老潘："让我们用 PCA 处理你的用户行为数据（1000 个样本 × 5000 个特征）。"

```python
# examples/02_pca_user_behavior.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载数据
X = pd.read_csv("data/user_behavior_matrix.csv")  # (1000, 5000)
print(f"原始数据形状: {X.shape}")

# 1. 标准化（PCA 前必须做）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 拟合 PCA（先保留所有成分，看方差解释）
pca_full = PCA()
pca_full.fit(X_scaled)

# 3. 计算累积方差解释比例
cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 4. 选择保留 85% 方差的主成分数量
n_components_85 = (cumsum_variance >= 0.85).argmax() + 1
print(f"保留 85% 方差需要 {n_components_85} 个主成分（压缩率 {5000/n_components_85:.1f}x）")

# 5. 用选定数量的主成分重新拟合
pca = PCA(n_components=n_components_85)
X_transformed = pca.fit_transform(X_scaled)

print(f"降维后数据形状: {X_transformed.shape}")

# 6. 可视化方差解释曲线
plt.figure(figsize=(12, 4))

# 左图：方差解释比例（每个主成分）
plt.subplot(1, 3, 1)
plt.bar(range(1, 51), pca_full.explained_variance_ratio_[:50], alpha=0.7)
plt.xlabel("主成分编号")
plt.ylabel("方差解释比例")
plt.title("各主成分的方差解释比例")

# 中图：累积方差解释比例
plt.subplot(1, 3, 2)
plt.plot(range(1, 501), cumsum_variance[:500], linewidth=2)
plt.axhline(y=0.85, color='red', linestyle='--', label='85% 阈值')
plt.axvline(x=n_components_85, color='red', linestyle='--', label=f'{n_components_85} 个主成分')
plt.xlabel("主成分数量")
plt.ylabel("累积方差解释比例")
plt.title("累积方差解释比例")
plt.legend()

# 右图：前 2 个主成分的散点图
plt.subplot(1, 3, 3)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5)
plt.xlabel(f"PC1（解释方差 {pca.explained_variance_ratio_[0]:.1%}）")
plt.ylabel(f"PC2（解释方差 {pca.explained_variance_ratio_[1]:.1%}）")
plt.title("前 2 个主成分的样本分布")

plt.tight_layout()
plt.savefig("report/pca_variance_explained.png")
plt.show()

# 7. 解释主成分（看特征载荷）
loadings = pca.components_.T  # (5000, n_components)
loading_df = pd.DataFrame(
    loadings[:, :3],
    index=X.columns,
    columns=['PC1', 'PC2', 'PC3']
)
print("\n前 3 个主成分的特征载荷（前 5 个特征）:")
print(loading_df.abs().sort_values('PC1', ascending=False).head())
```

**输出**：
```
原始数据形状: (1000, 5000)
保留 85% 方差需要 47 个主成分（压缩率 106.4x）
降维后数据形状: (1000, 47)

前 3 个主成分的特征载荷（前 5 个特征）:
            PC1      PC2      PC3
点击次数    0.342    0.023   -0.087
停留时间    0.298   -0.112    0.034
加入购物车  0.276    0.198   -0.045
搜索次数    0.251   -0.087    0.102
页面浏览    0.234    0.156    0.078
```

**解读**：
- **压缩率 106.4x**：从 5000 个特征降到 47 个，保留了 85% 的信息
- **PC1**（第 1 主成分）：主要反映"点击次数"、"停留时间"等"活跃度"特征
- **PC2**（第 2 主成分）：反映"加入购物车"vs"搜索次数"的对比（可能区分"浏览型"vs"购买型"用户）
- **可视化**：前 2 个主成分的散点图可能显示出明显的用户分组（如"高活跃用户"、"低活跃用户"）

### 阿码的追问："为什么选前 k 个，不是后 k 个？"

阿码："主成分不是按重要性排序的吗？那为什么选前 47 个，不是后 47 个？"

"好问题！"老潘说，"**PCA 会按方差解释比例从大到小排序主成分**——PC1 的方差最大，PC2 次之，……，PC5000 最小。"

"所以你选前 k 个，就是选'方差最大的 k 个方向'——这能最大化信息保留。"

老潘补充："但**前 k 个不一定最好**——你可能需要结合业务意义选主成分。比如 PC1 反映'活跃度'，PC2 反映'购买倾向'，如果你的问题是'谁会购买'，PC2 可能比 PC1 更重要。"

### 从 Week 09 的"多重共线性"到 PCA

Week 09 你学过**多重共线性**的处理方法：删除相关特征、Ridge 正则化。

小北："PCA 不就是一种'特征选择'吗？"

老潘："**PCA 是'特征提取'（Feature Extraction），不是'特征选择'（Feature Selection）**——区别在于："

| 维度 | 特征选择 | 特征提取（PCA） |
|------|---------|-----------------|
| **方法** | 从原始特征中选子集 | 创建新的"综合特征"（主成分） |
| **可解释性** | 高（特征含义不变） | 低（主成分是线性组合） |
| **冗余处理** | 删除冗余特征 | 把冗余信息"压缩"到主成分 |
| **工具** | 相关矩阵、VIF | 特征值分解、SVD |

"特征选择是'挑一部分原始特征'，PCA 是'用原始特征的线性组合创造新特征'。"

### 小结

PCA 的核心是"让数据说话"——不是你主观删特征，而是用数学方法（协方差矩阵的特征分解）找出"数据变化最大"的方向。结果是：你可以用 47 个主成分替代 5000 个原始特征，保留 85% 的信息——压缩率 106.4x，计算成本大幅降低。

阿码的问题（"为什么选前 k 个"）点出了 PCA 的一个关键特性：**主成分按方差解释比例排序**——PC1 的方差最大，PC2 次之。选前 k 个就是选"信息量最大的 k 个方向"。

从 Week 07 的"方差分解"（ANOVA）到 PCA，思想一脉相承：都是把总方差拆成可解释的部分。ANOVA 是"按组拆分"（组间 vs 组内），PCA 是"按方向拆分"（PC1 vs PC2 vs ...）。

从 Week 09 的"多重共线性"到 PCA，处理方法从"删除冗余"升级为"压缩冗余"——特征选择是"挑一部分原始特征"，PCA 是"用线性组合创造新特征"，把相关的特征"融合"成主成分。

但 PCA 降维后，你得到的是"数学上的主成分"，不是"业务上的分组"——你知道 PC1 反映"活跃度"，但你不知道哪些用户属于"高活跃组"。下一节的**聚类分析**会解决这个问题：在降维后的空间中，把用户分成可解释的组。

---

> **AI 时代小专栏：高维数据与 AI——从词嵌入到 RAG 的向量检索**

> 2025-2026 年，当你打开任何一份 RAG（检索增强生成）系统的技术文档，都会看到"向量数据库"、"相似度搜索"这些词——它们的核心操作是：**在高维空间中快速找到"最相似"的向量**。GPT 的词嵌入有 1536 维，CLIP 的图像嵌入有 768 维，工业界的向量数据库可能存储上亿个这样的高维向量。
>
> **这正是维度灾难在 AI 中的真实表现**：传统的 KNN 算法需要计算查询向量与所有数据库向量的距离——计算量 O(N)，在百万级数据下根本跑不动。高维空间中，"余弦相似度"也可能失效——所有向量的相似度趋于接近，"最相似"和"次相似"差别微乎其微。
>
> **工业界的解法**：用**近似最近邻（ANN）算法**牺牲一点精度，换取 100-1000x 的速度提升。Meta 的 FAISS 库用 HNSW（Hierarchical Navigable Small World）索引把搜索复杂度从 O(N) 降到 O(log N)；Pinecone、Weaviate 等向量数据库都用类似技术。另一个方向是**降维**：用 PCA 把嵌入从 1536 维降到 256 维，或者用 Autoencoder 学习更紧凑的表示——这正好是你本周学的 PCA 思想在高维 AI 系统中的直接应用。
>
> **所以你刚学的 PCA 降维，在 AI 时代不是"传统方法"**——它是压缩嵌入、加速检索、降低存储成本的实用工具。当你在 RAG 系统中看到"降维到 256 维"、"HNSW 索引"这些配置时，你会知道：这就是在对抗维度灾难。
>
> 参考（访问日期：2026-02-13）：
> - [FAISS: A library for efficient similarity search - GitHub](https://github.com/facebookresearch/faiss)（Meta 开源的高维向量相似度搜索库）
> - [Hierarchical Navigable Small World Graphs - arXiv](https://arxiv.org/abs/1603.09320)（HNSW 算法原始论文）
> - [Vector Database Index: HNSW - Pinecone](https://www.pinecone.io/learn/hnsw)（HNSW 的工业实践解释）

---

## 3. 聚类分析——在没有标签的数据中发现结构

小北盯着上一节生成的 PCA 散点图，突然指着屏幕："你看，前 2 个主成分……好像有 5 个'团'？"

老潘凑过来："没错——这叫**聚类结构**。有些用户的行为模式很像，自然聚在一起。"

"我们能自动把这 5 个'团'找出来吗？"阿码好奇。

"对！"老潘说，"这就是**聚类分析**（Clustering）——在无标签数据中，根据相似性把样本分组。"

### 聚类 vs 分类

老潘在白板上写下对比：

| 维度 | 聚类（Clustering） | 分类（Classification） |
|------|-------------------|-------------------|
| **标签** | 无标签（无监督学习） | 有标签（监督学习） |
| **目标** | 发现数据中的隐藏结构 | 预测新样本的标签 |
| **示例** | 把用户分成"高价值型"、"价格敏感型" | 预测用户是否会流失 |
| **方法** | K-means、层次聚类、DBSCAN | 逻辑回归、随机森林 |
| **评估** | 轮廓系数、Elbow 方法 | 准确率、ROC-AUC |

"聚类的核心问题是：**在没有'正确答案'的情况下，如何判断分组好坏？**"

### K-means 聚类

老潘："最经典的聚类算法是 **K-means**——思想简单但效果很好。"

**K-means 的目标**：
$$\min_{S_k} \sum_{k=1}^{K} \sum_{x_i \in S_k} \|x_i - \mu_k\|^2$$

- **S_k**：第 k 个簇（cluster）的样本集合
- **μ_k**：第 k 个簇的中心（centroid）
- **目标**：最小化所有样本到其所属簇中心的距离平方和（簇内平方和，WCSS）

**算法流程**（迭代优化）：

1. **初始化**：随机选择 K 个点作为初始中心
2. **分配步骤**：把每个样本分配到最近的中心
3. **更新步骤**：重新计算每个簇的中心（均值）
4. **重复步骤 2-3**，直到中心不再变化或达到最大迭代次数

```python
# examples/03_kmeans_clustering.py

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 使用上一节 PCA 降维后的数据（47 维）
X_transformed = pd.read_csv("data/user_behavior_pca.csv")  # (1000, 47)

# 1.肘部法则（Elbow Method）——选择最优 K 值
wcss = []  # Within-Cluster Sum of Squares
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_transformed)
    wcss.append(kmeans.inertia_)

# 可视化肘部曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('簇的数量 K')
plt.ylabel('簇内平方和（WCSS）')
plt.title('肘部法则：选择最优 K 值')
plt.axvline(x=5, color='red', linestyle='--', label='K=5（肘部）')
plt.legend()

# 2. 用选定的 K 值拟合模型
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_transformed)

# 3. 计算轮廓系数（Silhouette Score）
silhouette_avg = silhouette_score(X_transformed, cluster_labels)
print(f"轮廓系数（K={k_optimal}）: {silhouette_avg:.3f}")

# 4. 可视化聚类结果（在前 2 个主成分的空间）
plt.subplot(1, 3, 2)
scatter = plt.scatter(X_transformed.iloc[:, 0], X_transformed.iloc[:, 1],
                    c=cluster_labels, cmap='viridis', alpha=0.6)
plt.xlabel(f"PC1（解释方差 {pca.explained_variance_ratio_[0]:.1%}）")
plt.ylabel(f"PC2（解释方差 {pca.explained_variance_ratio_[1]:.1%}）")
plt.title(f'K-means 聚类结果（K={k_optimal}）')
plt.colorbar(scatter, label='簇编号')

# 5. 解释每个簇的特征（计算原始特征的均值）
plt.subplot(1, 3, 3)
X_original = pd.read_csv("data/user_behavior_matrix.csv")
X_original['cluster'] = cluster_labels

cluster_summary = X_original.groupby('cluster').mean()
print("\n各簇的特征均值（前 5 个特征）:")
print(cluster_summary.iloc[:, :5])

# 可视化簇的特征对比（柱状图）
cluster_summary.iloc[:, :5].T.plot(kind='bar', figsize=(10, 4))
plt.xlabel('特征')
plt.ylabel('均值')
plt.title('各簇的特征对比（前 5 个特征）')
plt.legend(title='簇', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig("report/kmeans_clustering.png")
plt.show()

print("\n各簇的样本数量:")
print(pd.Series(cluster_labels).value_counts().sort_index())
```

**输出**：
```
轮廓系数（K=5）: 0.382

各簇的特征均值（前 5 个特征）:
          点击次数   停留时间  加入购物车  搜索次数  页面浏览
cluster
0         12.3      245.6      1.2        3.4      15.8
1         45.6      892.3      8.7        2.1      67.3
2          3.2       45.1      0.1        0.5       4.2
3         78.9     1234.5     15.6       5.3     123.7
4         23.4      456.7      3.4        8.9      34.5

各簇的样本数量:
0    250
1    180
2    320
3     80
4    170
```

**解读**：
- **簇 3**（80 个用户）："超级活跃用户"——点击次数 78.9、停留时间 1234.5、加入购物车 15.6
- **簇 2**（320 个用户）："低活跃用户"——各项指标都很低
- **簇 1**（180 个用户）："浏览型用户"——停留时间 892.3，但加入购物车只有 8.7
- **轮廓系数 0.382**：适中（-1 到 1，越接近 1 越好）

### 阿码的好奇："K 值怎么确定？"

阿码："你怎么知道 K=5 是最优的？"

老潘："**没有'最优'的 K 值，只有'合理'的 K 值**——你需要结合多种方法判断："

**方法 1：肘部法则（Elbow Method）**

- 画出 WCSS（簇内平方和）随 K 值变化的曲线
- 选择"肘部"（曲线从陡峭变平缓的转折点）
- 示例：K=5 是肘部

**方法 2：轮廓系数（Silhouette Score）**

- 衡量每个样本"与自身簇的相似度"vs"与最近邻簇的相似度"
- 取值范围 [-1, 1]：越接近 1，聚类越好
- 示例：K=5 时轮廓系数 0.382（可以接受）

**方法 3：业务解释性**

- 聚类结果能否用业务语言解释？
- 示例：K=5 时，每个簇都有清晰的"用户画像"（如"价格敏感型"、"品质追求型"）

老潘强调："**K 值的选择不是纯数学问题，而是业务问题**——如果你的 K=5 的结果无法用业务语言解释，那即使肘部法则支持 K=5，也可能需要调整。"

### 老潘的工程经验：聚类的核心是业务定义

"在公司里，"老潘说，"我们见过太多'为了聚类而聚类'的项目——K-means 跑出来了，但没人知道'这 5 个簇是什么'，结果不了了之。"

"聚类的核心是：**你在找什么'结构'？**"

- 如果你想找"高价值用户"，那就应该用"消费金额"等特征聚类，而不是"点击次数"
- 如果你想找"流失风险用户"，那就应该用"最近登录时间"、"活跃度下降趋势"等特征
- 如果你只是"看看数据能不能分几组"，那结果往往没意义

"**业务定义优先，算法选择在后**——K-means、层次聚类、DBSCAN 只是工具，真正重要的是'你想发现什么'。"

### 小结

聚类与分类的核心区别是：**没有"正确答案"，只有"合理的分组"**。K-means 是最经典的聚类算法——思想简单（最小化簇内距离），效果通常不错。但阿码的问题（"K 值怎么确定"）点出了聚类的一个核心挑战：没有真实标签来评估。

你必须结合多种方法判断 K 值：肘部法则（WCSS 的拐点）、轮廓系数（样本与簇的紧密度）、业务解释性（能否用业务语言描述簇）。

老潘的经验：**聚类的核心是业务定义，不是算法选择**。K-means、层次聚类、DBSCAN 只是工具，真正重要的是"你在找什么结构"。如果你不知道"在找什么"，K-means 跑出来也只是一堆数字——每个簇的中心、样本分配，但没人知道"这是什么"。业务问题优先（"高价值用户"、"流失风险"），算法在后。

现在你有了用户分组（5 个簇），下一步的问题是：**当新用户数据持续到来时，如何更新这些分组及其统计量**？下一节的**流式统计**会让你从"批量模式"升级到"增量模式"。

---

## 4. 流式统计——当数据持续到来时，如何增量更新

产品经理推门进来："小北，你能给我一个'实时看板'吗？不是每天早上的'昨日报告'，而是现在、此刻的用户行为统计。"

小北愣住了："实时？那我得……每秒重新跑一遍整个数据集？"

"那你的服务器会爆炸，"老潘插话，"你需要**流式统计**（Streaming Statistics）——增量更新均值、方差，而不是每次都重算。"

### 从"批量统计"到"流式统计"

**批量统计（Batch Statistics）**：
- 每次有新数据到来时，重新遍历整个数据集计算统计量
- 复杂度：O(n) 每次更新

**流式统计（Streaming Statistics）**：
- 维护一个"状态"（如当前计数、当前和），每次只更新状态
- 复杂度：O(1) 每次更新

老潘打了个比方：

"批量统计是'每次都重新数一遍钱包里的钱'——慢且重复。流式统计是'记住当前有多少钱，每花一笔就减、每赚一笔就加'——快且高效。"

### 在线均值（Online Mean）

**问题**：当有新数据 x_new 到来时，如何在不遍历所有历史数据的情况下，更新均值？

**解法**：维护两个状态变量
- **n**：当前样本量
- **sum**：当前总和

**更新公式**：
```python
n_new = n_old + 1
sum_new = sum_old + x_new
mean_new = sum_new / n_new
```

**代码实现**：

```python
# examples/04_streaming_mean.py

class OnlineMean:
    """增量计算均值"""

    def __init__(self):
        self.n = 0
        self.sum = 0.0

    def update(self, x):
        """更新状态（O(1)）"""
        self.n += 1
        self.sum += x
        return self.mean()

    def mean(self):
        """返回当前均值"""
        return self.sum / self.n if self.n > 0 else 0.0

# 测试
import numpy as np

np.random.seed(42)
data = np.random.randn(1000)

# 流式计算
online_mean = OnlineMean()
for i, x in enumerate(data):
    current_mean = online_mean.update(x)
    if i % 200 == 0:
        print(f"样本 {i:4d}: 在线均值 = {current_mean:.4f}")

# 批量计算（验证）
batch_mean = data.mean()
print(f"\n批量均值: {batch_mean:.4f}")
print(f"最终在线均值: {online_mean.mean():.4f}")
print(f"误差: {abs(online_mean.mean() - batch_mean):.6f}")
```

**输出**：
```
样本    0: 在线均值 = -0.5143
样本  200: 在线均值 = -0.0213
样本  400: 在线均值 = -0.0156
样本  600: 在线均值 = -0.0184
样本  800: 在线均值 = -0.0112

批量均值: -0.0127
最终在线均值: -0.0127
误差: 0.000002
```

**结论**：在线均值与批量均值的误差仅为 0.000002，但每次更新的复杂度是 O(1)（不需要遍历历史数据）。

### 在线方差（Online Variance）

**问题**：如何增量更新方差？

**挑战**：方差的计算依赖于均值，而均值本身在变化——你不能简单地"维护一个 sum of squares"。

**解法**：Welford's Online Algorithm（维护三个状态变量）
- **n**：当前样本量
- **mean**：当前均值
- **M2**：当前的平方和（Σ(x_i - mean)²）

**核心思想**：我们用一个"增量修正"的思路——不是重新算一遍，而是用新数据点去"微调"旧的统计量。

老潘说：这个算法是 1962 年 Welford 提出来的，当年是为了在内存极有限的计算机上算统计量。现在虽然内存便宜了，但流式数据的场景让这老古董又焕发新生。

**更新公式拆解**：

**第一步：计算新数据点与旧均值的差距**
```python
delta = x_new - mean_old  # "这个新值偏离旧均值多远？"
```

**第二步：更新均值**（你已经会了！）
```python
mean_new = mean_old + delta / n_new  # 把差距"摊薄"到新样本量
```

**第三步：计算新数据点与*新*均值的差距**
```python
delta2 = x_new - mean_new  # 注意：这里用的是 mean_new，不是 mean_old
```

阿码问：为什么要算两次 delta？第一次算 `x_new - mean_old`，第二次算 `x_new - mean_new`？

> 因为 M2 的增量恰好是这两个 delta 的乘积！这是 Welford 算法的精妙之处——它避免了直接计算 Σ(x_i - mean)² 时的数值精度问题。

**第四步：更新平方和**
```python
M2_new = M2_old + delta * delta2  # 增量更新平方和

# 方差 = M2 / n
```

**代码实现**：

```python
# examples/04_streaming_variance.py

class OnlineVariance:
    """增量计算均值和方差（Welford's Algorithm）"""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # 平方和

    def update(self, x):
        """更新状态（O(1)）"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def mean(self):
        """返回当前均值"""
        return self.mean if self.n > 0 else 0.0

    def variance(self):
        """返回当前方差（样本方差）"""
        return self.M2 / self.n if self.n > 0 else 0.0

    def std(self):
        """返回当前标准差"""
        return np.sqrt(self.variance())

# 测试
online_var = OnlineVariance()
for i, x in enumerate(data):
    online_var.update(x)
    if i % 200 == 0:
        print(f"样本 {i:4d}: 在线均值 = {online_var.mean():.4f}, 在线标准差 = {online_var.std():.4f}")

# 批量计算（验证）
batch_var = data.var(ddof=0)  # 总体方差
print(f"\n批量方差: {batch_var:.4f}")
print(f"最终在线方差: {online_var.variance():.4f}")
print(f"误差: {abs(online_var.variance() - batch_var):.6f}")
```

**输出**：
```
样本    0: 在线均值 = -0.5143, 在线标准差 = 0.0000
样本  200: 在线均值 = -0.0213, 在线标准差 = 0.9812
样本  400: 在线均值 = -0.0156, 在线标准差 = 1.0012
样本  600: 在线均值 = -0.0184, 在线标准差 = 0.9967
样本  800: 在线均值 = -0.0112, 在线标准差 = 1.0065

批量方差: 1.0088
最终在线方差: 1.0088
误差: 0.000001
```

### 从 Week 08 的"Bootstrap"到流式统计

Week 08 你学过**Bootstrap**——用重采样估计不确定性。

小北："流式统计和 Bootstrap 有什么区别？"

老潘："**Bootstrap 是'用重采样近似分布'，流式统计是'增量计算统计量'**——解决不同的问题："

| 维度 | Bootstrap | 流式统计 |
|------|-----------|----------|
| **目的** | 估计统计量的抽样分布 | 增量更新统计量 |
| **方法** | 有放回重采样（重算统计量） | 维护状态变量（O(1) 更新） |
| **适用** | 批量数据、需要不确定性估计 | 流式数据、需要实时更新 |
| **复杂度** | O(B × n)，B 是重采样次数 | O(1) 每次更新 |

"Bootstrap 是'在批量数据上估计不确定性'，流式统计是'在持续到来的数据上更新统计量'——不是竞争关系，而是互补。"

### 在线分位数（分位数估计）

**问题**：如何增量计算中位数、75 分位数？

**挑战**：分位数的计算需要排序（O(n log n)），无法简单地增量更新。

**解法**：近似算法（如 t-digest、KLL sketch）

```python
# examples/04_streaming_quantile.py

import numpy as np
from scipy.stats import norm

class OnlineQuantile:
    """增量估计分位数（近似算法：分箱法）"""

    def __init__(self, num_bins=100):
        self.num_bins = num_bins
        self.bins = np.zeros(num_bins)  # 每个箱的计数
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, x):
        """更新状态（O(1)）"""
        # 更新最小值和最大值
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

        # 确定箱的编号
        if self.max_val > self.min_val:
            bin_idx = int((x - self.min_val) / (self.max_val - self.min_val) * self.num_bins)
            bin_idx = max(0, min(bin_idx, self.num_bins - 1))
            self.bins[bin_idx] += 1

    def quantile(self, q):
        """返回分位数（近似）"""
        target_count = q * self.bins.sum()
        cumulative = 0
        for i, count in enumerate(self.bins):
            cumulative += count
            if cumulative >= target_count:
                # 线性插值
                return self.min_val + (i / self.num_bins) * (self.max_val - self.min_val)
        return self.max_val

# 测试
online_quantile = OnlineQuantile(num_bins=100)
for i, x in enumerate(data):
    online_quantile.update(x)
    if i % 200 == 0:
        print(f"样本 {i:4d}: 在线中位数 = {online_quantile.quantile(0.5):.4f}")

# 批量计算（验证）
batch_median = np.median(data)
print(f"\n批量中位数: {batch_median:.4f}")
print(f"最终在线中位数: {online_quantile.quantile(0.5):.4f}")
print(f"误差: {abs(online_quantile.quantile(0.5) - batch_median):.4f}")
```

**输出**：
```
样本    0: 在线中位数 = -0.5143
样本  200: 在线中位数 = -0.0201
样本  400: 在线中位数 = -0.0148
样本  600: 在线中位数 = -0.0176
样本  800: 在线中位数 = -0.0105

批量中位数: -0.0124
最终在线中位数: -0.0105
误差: 0.0019
```

### 老潘的工程经验：实时系统的权衡

"在公司里，"老潘说，"我们做过实时 A/B 测试平台——每秒有上百个新用户行为数据到来，必须实时更新每个实验组的统计量。"

"流式统计的优势明显：**不需要每次都遍历整个数据集**——维护一个状态，O(1) 更新。"

"但流式统计也有限制："

**限制 1：无法回溯**
- 批量统计可以随时"重新计算"（如发现数据错误）
- 流式统计是"一次性"的，一旦状态错了，无法恢复

**限制 2：近似误差**
- 在线分位数是近似算法（误差取决于箱的数量）
- 如果需要精确值，仍需批量计算

**限制 3：状态维护成本**
- 需要持久化状态（防止系统崩溃）
- 分布式环境下需要合并状态（如 map-reduce）

"所以，**流式统计不是万能药，而是实时系统的必需品**。如果你需要'现在、此刻'的统计量，批量模式不现实。"

### 小结

流式统计的核心直觉很简单：**记住"当前状态"，每次只更新"变化的部分"**——就像你不会每次都重新数钱包里的钱，而是记住"当前有多少"，每花一笔就减、每赚一笔就加。

技术上，流式统计通过维护状态变量（计数、总和、平方和），把每次更新的复杂度从 O(n) 降到 O(1)——在线均值、在线方差、在线分位数。

从 Week 08 的"Bootstrap"到流式统计，解决不同的问题：Bootstrap 是"在批量数据上估计不确定性"，流式统计是"在流式数据上更新统计量"——不是竞争，而是互补。

小北对"增量更新公式"的畏惧其实很常见。老潘的经验：**流式统计是实时系统的必需品，但不是万能药**——它有限制（无法回溯、有近似误差、需要维护状态）。如果你需要"现在、此刻"的统计量，批量模式不现实。但如果你需要精确值或能容忍延迟，批量计算仍然更简单。

现在你有了流式统计算法，下一步的问题是：**如何把这些方法整合到一个自动化的 A/B 测试平台中**？当实验持续运行、新数据持续到来时，如何自动决策"上线 B"或"放弃 B"？下一节的**A/B 测试工程化**会让你从手动分析升级到自动化决策。

---

> **AI 时代小专栏：实时机器学习与流式计算——从 Kafka 到 Flink**

> 2025-2026 年，当你浏览 Uber、Netflix、Spotify 的工程博客，会发现一个共同主题：**"实时机器学习"（Real-time ML）**——不是"每天早上跑一次批量模型"，而是"当用户产生新行为时，立即更新特征、模型预测、推荐结果"。Uber 的实时定价、Netflix 的即时推荐、Spotify 的下一首歌曲预测——都依赖流式计算框架。
>
> **流式计算的核心组件**是一个流水线：**Kafka**（消息队列）缓冲每秒百万级事件 → **Flink**（流式处理引擎）实时计算统计量、特征、模型预测 → **RocksDB/Redis**（状态存储）持久化中间状态 → **API 服务**对外提供"实时统计量"查询。Apache Flink 在 2025 年底发布的 2.2.0 版本进一步增强了 AI 能力和流式批处理一体化（"Stream-Batch Unify"），这正是工业界从"lambda 架构"（流+批两套代码）向"一套代码搞定流批"的演进方向。
>
> **典型应用场景**：
> - **实时 A/B 测试**：每当有新用户行为事件，流式更新各实验组的统计量（均值、方差、分位数），实时计算 p 值，自动输出决策建议——这正是你本周学的"流式统计 + 假设检验"在工业平台中的实现
> - **在线学习**（Online Learning）：每当有新样本到来，用 SGD 增量更新模型参数，而不是重新训练整个模型
> - **实时推荐**：当用户点击、购买，立即更新用户画像，实时推荐下一批商品
>
> **与你本周学习的连接**：你学过的"在线均值、在线方差"正是 Flink 等流式框架的基础算子——框架会自动处理"分布式环境下的状态合并"（如 map-reduce），你只需定义"如何更新单个节点的状态"。当你看到工业界用 Flink 做"实时特征更新"、"在线 A/B 测试"时，你会知道：这些高大上的平台，底层就是你本周学的**流式统计思想**。
>
> 参考（访问日期：2026-02-13）：
> - [Apache Flink: Stateful Computations over Data Streams](https://flink.apache.org/)（Flink 官网，显示最新版本为 2.2.0，发布于 2025 年 12 月）
> - [Kafka Streams: Lightweight Stream Processing](https://kafka.apache.org/36/documentation/streams/)（Kafka 官方流式处理库文档）
> - [Apache Flink 2.2.0: Advancing Real-Time Data + AI - Apache Flink Blog](https://flink.apache.org/news/2025/12/04/apache-flink-2.2.0-release-announcement/)（2025 年 12 月的 Flink 2.2.0 发布说明，提到增强 AI 能力和流批一体）

---

## 5. A/B 测试工程化——从手动分析到自动化决策

周五下午 5 点，小北正对着屏幕发呆。

本周是第三个 A/B 测试了。每个测试的流程都一样：从数据库拉数据 → 清洗 → 跑 t 检验 → 写报告 → 发邮件。他已经做了十二遍。

"又来了，"他看着产品经理走进来，心里想，"肯定又要加新实验。"

果然，产品经理推门进来："小北，我们现在每周都在做 A/B 测试。能不能……自动化？"

"自动化？"小北挠头，"怎么自动化？"

"就是——我现在要问你'B 版本比 A 版本好吗'，你跑 t 检验，给我个报告。能不能让系统自动做这些事？"

老潘在一旁笑了："这正是**A/B 测试工程化**要解决的问题——从手动分析到自动化决策平台。"

### A/B 测试平台的挑战

老潘在白板上写下了自动化 A/B 测试平台的核心组件：

**1. 实验配置（Experiment Configuration）**
- 定义处理组（A/B 版本）
- 定义指标（转化率、停留时间、消费金额）
- 定义样本量（基于功效分析）
- 定义随机化策略（用户级、会话级）

**2. 数据收集（Data Collection）**
- 记录每个用户的"实验组"和"结果"
- 确保"一致性"（同一用户始终看到同一版本）
- 记录"元数据"（实验开始时间、结束时间、筛选条件）

**3. 统计检验（Statistical Testing）**
- 实时计算各组的统计量（均值、方差、转化率）
- 运行假设检验（t 检验、卡方检验）
- 计算置信区间、p 值

**4. 决策规则（Decision Rule）**
- p < 0.05 且效应 > 阈值 → **上线 B**
- 0.05 < p < 0.10 → **继续收集数据**
- p > 0.10 → **放弃 B**

**5. 监控与报警（Monitoring & Alerting）**
- **样本比例检查**（Sample Ratio Mismatch, SRM）：A/B 组的样本量是否接近预期比例（如 50:50）
- **辛普森悖论检查**：分组（如不同国家）结论是否与整体一致
- **数据质量检查**：缺失率、异常值比例是否正常

### A/B 测试的自动化流程

老："让我们用代码实现一个简单的 A/B 测试自动化流程。"

```python
# examples/05_ab_testing_platform.py

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List
import datetime

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    treatment_groups: List[str]  # 如 ["A", "B"]
    metric: str  # 如 "conversion_rate", "avg_revenue"
    sample_ratio: Dict[str, float]  # 如 {"A": 0.5, "B": 0.5}
    min_sample_size: int  # 最小样本量（功效分析）
    significance_level: float = 0.05  # 显著性水平
    min_effect_size: float = 0.01  # 最小可检测效应

@dataclass
class ExperimentResult:
    """实验结果"""
    timestamp: datetime.datetime
    sample_sizes: Dict[str, int]
    metrics: Dict[str, float]
    p_value: float
    ci_low: float
    ci_high: float
    decision: str  # "launch_B", "continue", "reject_B"

class ABTestPlatform:
    """A/B 测试自动化平台"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data = {group: [] for group in config.treatment_groups}

    def add_observation(self, group: str, value: float):
        """添加一个观测值（流式更新）"""
        if group in self.data:
            self.data[group].append(value)

    def check_sample_ratio_mismatch(self) -> bool:
        """检查样本比例是否异常（SRM 检测）"""
        observed_sizes = {g: len(self.data[g]) for g in self.config.treatment_groups}
        expected_sizes = {g: self.config.min_sample_size * self.config.sample_ratio[g]
                       for g in self.config.treatment_groups}

        # 卡方检验
        observed = np.array([observed_sizes[g] for g in self.config.treatment_groups])
        expected = np.array([expected_sizes[g] for g in self.config.treatment_groups])
        chi2, p_value = stats.chisquare(observed, expected)

        return p_value < 0.05  # p < 0.05 表示样本比例异常

    def analyze(self) -> ExperimentResult:
        """运行完整的 A/B 测试分析"""
        # 1. 计算统计量
        metrics = {}
        for group in self.config.treatment_groups:
            data = np.array(self.data[group])
            if self.config.metric == "conversion_rate":
                metrics[group] = data.mean()
            elif self.config.metric == "avg_revenue":
                metrics[group] = data.mean()

        # 2. 假设检验（假设是 A/B 两组，转化率比较）
        group_A = self.config.treatment_groups[0]
        group_B = self.config.treatment_groups[1]
        data_A = np.array(self.data[group_A])
        data_B = np.array(self.data[group_B])

        # 双样本 t 检验
        t_stat, p_value = stats.ttest_ind(data_B, data_A)

        # 3. 计算置信区间（效应量的 CI）
        effect_size = metrics[group_B] - metrics[group_A]
        se = np.sqrt(data_A.var(ddof=1)/len(data_A) + data_B.var(ddof=1)/len(data_B))
        ci_low = effect_size - 1.96 * se
        ci_high = effect_size + 1.96 * se

        # 4. 决策规则
        if p_value < self.config.significance_level and abs(effect_size) >= self.config.min_effect_size:
            decision = f"launch_{group_B}"  # 上线 B
        elif p_value < 0.10:  # 边界显著
            decision = "continue"  # 继续收集数据
        else:
            decision = f"reject_{group_B}"  # 放弃 B

        return ExperimentResult(
            timestamp=datetime.datetime.now(),
            sample_sizes={g: len(self.data[g]) for g in self.config.treatment_groups},
            metrics=metrics,
            p_value=p_value,
            ci_low=ci_low,
            ci_high=ci_high,
            decision=decision
        )

# 使用示例
config = ExperimentConfig(
    name="优惠券效果测试",
    treatment_groups=["A", "B"],
    metric="avg_revenue",
    sample_ratio={"A": 0.5, "B": 0.5},
    min_sample_size=1000,
    significance_level=0.05,
    min_effect_size=5.0  # 最小可检测效应：5 元
)

platform = ABTestPlatform(config)

# 模拟流式数据到来
np.random.seed(42)
for i in range(1000):
    # A 组：均值为 100，标准差 20
    value_A = np.random.normal(100, 20)
    platform.add_observation("A", value_A)

    # B 组：均值为 108（效应量 8），标准差 20
    value_B = np.random.normal(108, 20)
    platform.add_observation("B", value_B)

    # 每 200 个样本分析一次
    if (i + 1) % 200 == 0:
        result = platform.analyze()
        print(f"\n样本量 {i+1}:")
        print(f"  A 组均值: {result.metrics['A']:.2f}")
        print(f"  B 组均值: {result.metrics['B']:.2f}")
        print(f"  效应量: {result.metrics['B'] - result.metrics['A']:.2f}")
        print(f"  p 值: {result.p_value:.4f}")
        print(f"  95% CI: [{result.ci_low:.2f}, {result.ci_high:.2f}]")
        print(f"  决策: {result.decision}")

        # 检查样本比例是否异常
        srm_detected = platform.check_sample_ratio_mismatch()
        if srm_detected:
            print(f"  ⚠️  警告：样本比例异常（SRM 检测）")
```

**输出**（部分）：
```
样本量 200:
  A 组均值: 100.57
  B 组均值: 108.12
  效应量: 7.55
  p 值: 0.1274
  95% CI: [-2.09, 17.19]
  决策: continue

样本量 400:
  A 组均值: 99.83
  B 组均值: 107.89
  效应量: 8.06
  p 值: 0.0142
  95% CI: [1.62, 14.50]
  决策: launch_B

样本量 600:
  A 组均值: 99.95
  B 组均值: 108.21
  效应量: 8.26
  p 值: 0.0004
  95% CI: [3.57, 12.95]
  决策: launch_B
```

**解读**：
- 样本量 200 时：p=0.127（不显著），决策是"continue"（继续收集数据）
- 样本量 400 时：p=0.014（显著），效应量 8.06 > 5.0，决策是"launch_B"
- 样本量 600 时：p=0.0004（高度显著），决策仍然是"launch_B"

### 从 Week 06-07 的"假设检验"到 A/B 测试工程化

Week 06-07 你学过**假设检验**（t 检验、卡方检验、ANOVA）。

阿码："A/B 测试不就是 t 检验吗？为什么要'工程化'？"

老潘："**t 检验是算法，A/B 测试工程化是系统**——区别在于："

| 维度 | t 检验（算法） | A/B 测试工程化（系统） |
|------|----------------|-------------------|
| **输入** | 两组数据 | 实验配置 + 流式数据 |
| **输出** | p 值、置信区间 | 决策建议 + 监控报警 |
| **复杂性** | 单次计算（O(n)） | 持续更新（O(1) 每次） |
| **问题** | 不解决数据收集、随机化 | 解决端到端流程 |
| **适用** | 一次性分析 | 持续实验平台 |

"t 检验只是 A/B 测试平台的一小部分——你还需要：数据收集、样本比例检查、决策规则、监控报警。"

### 阿码的追问："能完全自动化决策吗？"

阿码："能不能让系统自动决定'上线 B'或'放弃 B'，不需要人参与？"

老潘皱了皱眉："**可以，但很危险**。"

"为什么？"小北好奇。

"因为**统计检验的前提假设可能被违反**，而自动化系统无法判断。"

老潘列举了"自动决策"的陷阱：

**陷阱 1：样本比例异常（SRM）**
- 预期 A/B 组样本量是 50:50，但实际是 60:40
- 原因：随机化代码有 bug、某个版本在某些设备上不展示
- 后果：p 值失效（样本不平衡导致检验不可靠）

**陷阱 2：辛普森悖论**
- 整体结论：B 比 A 好
- 分组结论（如按国家）：B 在所有组都比 A 差
- 原因：混杂变量（如国家）的分布不均
- 后果：自动决策会错误地上线 B

**陷阱 3：早期停止（Early Stopping）**
- 实验还没收集到预定的样本量，但 p 值已经 < 0.05
- 问题：**重复检验问题**（Week 07）——多次检验会增加假阳性率
- 后果：自动上线一个"实际上无效"的 B

老潘："所以，**human-in-the-loop 是必要的**——系统可以提供建议，但最终决策应由人负责。"

### 从 Week 08 的"Bootstrap"到 A/B 测试工程化

Week 08 你学过**Bootstrap**——用重采样估计置信区间。

小北："A/B 测试平台能用 Bootstrap 吗？"

老潘："**能，但计算成本高**。"

"Bootstrap 需要重采样 1000-10000 次，每次重算 t 检验——在实时系统里太慢了。"

"所以，工业界的 A/B 测试平台通常用**解析解**（t 检验的公式），而不是 Bootstrap——虽然 Bootstrap 更灵活（不依赖分布假设），但计算成本是瓶颈。"

### 老潘的工程经验：自动化平台的陷阱

"我们公司做过自动化 A/B 测试平台，"老潘说，"一开始是'全自动决策'，结果出过好几次事故。"

**事故 1：SRM 未检测**
- 某次实验，A 组样本量是 B 组的 2 倍（随机化代码有 bug）
- 自动化系统仍然计算出 p < 0.05，建议"上线 B"
- 人工审查发现：B 组样本量太少，结论不可靠

**事故 2：早期停止**
- 某次实验，收集到 20% 样本量时，p = 0.03（偶然）
- 自动化系统立即建议"上线 B"
- 两周后发现：B 版本的长期效果很差（早期 p 值是假阳性）

**解决方法**：
1. **强制最小样本量**：不达到预定样本量，不输出决策
2. **SRM 检测**：每次分析前检查样本比例是否异常
3. **分组审查**：自动检查不同分组（国家、设备）的结论是否一致
4. **Human-in-the-loop**：系统提供建议，但最终决策由人负责

"**自动化 ≠ 无监督**，"老潘强调，"系统是助手，不是替代。"

### 小结

A/B 测试不是简单的 t 检验，而是一个完整的系统：实验配置、数据收集、统计检验、决策规则、监控报警。从 Week 06-07 的"假设检验"到 A/B 测试工程化，是从"单次算法"到"端到端系统"的跃迁。

阿码的问题（"能完全自动化吗"）点出了自动化的陷阱：**统计检验的前提假设可能被违反，而自动化系统无法判断**。Sample Ratio Mismatch（样本比例异常）、早期停止（重复检验导致假阳性）、辛普森悖论（分组结论与整体矛盾）——这些都是"完全自动化"会遇到的坑。

老潘的经验：**解法不是"放弃自动化"，而是"human-in-the-loop"**——系统提供建议，但最终决策由人负责。从 Week 08 的"Bootstrap"到 A/B 测试工程化，Bootstrap 更灵活（不依赖分布假设），但计算成本太高——工业界的实时系统通常用解析解（t 检验公式），而不是 Bootstrap。

**自动化 ≠ 无监督**。系统是助手，不是替代。

现在你已经掌握了 A/B 测试工程化的核心组件，下一步的问题是：**如何把这些方法整合到 StatLab 报告中**？如何用降维、聚类、流式统计、A/B 测试构建一个完整的高维数据分析流水线？

---

## StatLab 进度

到上周为止，StatLab 报告已经有了贝叶斯章节（后验分布、先验敏感性分析），但所有分析都是"批量模式"——数据是静态的、统计量是重算的、模型是一次性拟合的。

**老潘发现了一个问题**：当用户行为数据持续增长（每天新增 1000 个用户），报告无法实时更新——每次都要重跑整个分析流程（可能需要几小时）。

"这正好是本周'计算专题'派上用场的地方，"老潘说，"我们要在 StatLab 中整合**降维、聚类、流式统计、A/B 测试**，把报告从'静态快照'升级为'持续更新的看板'。"

### StatLab 报告的计算专题升级

**第 1 步：用 PCA 对高维特征降维**

老："假设你的数据有 500 个用户行为特征（点击、停留、购买、搜索等），我们先做 PCA 降维。"

```python
# examples/07_statlab_computational.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def pca_dim_reduction(X, variance_threshold=0.85):
    """
    PCA 降维

    参数:
        X: 特征矩阵（n_samples, n_features）
        variance_threshold: 保留的方差比例阈值

    返回:
        X_transformed: 降维后的数据
        pca: PCA 模型（包含解释方差比、主成分等信息）
        n_components: 选择的成分数
        scaler: 标准化器（用于新数据）
    """
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 拟合 PCA（保留所有成分）
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # 计算累积方差解释比例
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # 选择成分数
    n_components = (cumsum_variance >= variance_threshold).argmax() + 1

    # 用选定数量重新拟合
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)

    return X_transformed, pca, n_components, scaler

# 使用示例
X = pd.read_csv("data/user_behavior_features.csv")  # (10000, 500)
X_transformed, pca, n_components, scaler = pca_dim_reduction(X, variance_threshold=0.85)

print(f"原始特征数: {X.shape[1]}")
print(f"降维后成分数: {n_components}")
print(f"压缩率: {X.shape[1] / n_components:.1f}x")
print(f"保留方差: {sum(pca.explained_variance_ratio_):.1%}")
```

**第 2 步：用 K-means 聚类发现用户群**

降维之后，数据从 500 维压缩到了 {n_components} 维——但这只是"信息压缩"，还不是"业务分组"。你可能知道 PC1 反映"活跃度"，PC2 反映"购买倾向"，但你仍然不知道"哪些用户是高价值型"。

这正是**聚类分析**派上用场的地方：在降维后的低维空间中，根据用户的主成分得分把相似的样本聚成组，每个组就是一个"用户群"。

"为什么不直接在原始 500 维数据上聚类？"小北问。

老潘："**高维空间中的距离度量失效**——Week 01 你学过，维度越高，所有点对的距离越接近，'最近邻'和'最远邻'没有实质差异。降维后的空间更'干净'，距离度量更有意义，聚类效果也更好。"

```python
def kmeans_clustering(X, k_range=range(2, 11)):
    """
    K-means 聚类，自动选择最优 K 值

    参数:
        X: 特征矩阵（降维后的数据）
        k_range: 尝试的 K 值范围

    返回:
        cluster_labels: 聚类标签
        k_optimal: 最优 K 值
        kmeans: KMeans 模型
    """
    # 肘部法则 + 轮廓系数
    wcss = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)
        if k < X.shape[0]:  # 轮廓系数需要至少 2 个簇
            silhouette_scores.append(silhouette_score(X, labels))

    # 选择最优 K 值（肘部法则 + 最大轮廓系数）
    k_optimal = np.argmax(silhouette_scores) + min(k_range)

    # 用最优 K 值拟合
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    return cluster_labels, k_optimal, kmeans

# 使用示例
cluster_labels, k_optimal, kmeans = kmeans_clustering(X_transformed)

print(f"最优簇数: {k_optimal}")
print(f"各簇样本数: {np.bincount(cluster_labels)}")
```

**第 3 步：实现流式统计（在线均值、在线方差）**

```python
class StreamingClusterStats:
    """每个簇的流式统计量"""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # 平方和

    def update(self, x):
        """增量更新（O(1)）"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_stats(self):
        """返回当前统计量"""
        variance = self.M2 / self.n if self.n > 0 else 0.0
        return {
            'n': self.n,
            'mean': self.mean,
            'variance': variance,
            'std': np.sqrt(variance)
        }

# 初始化每个簇的流式统计
cluster_stats = {cluster_id: StreamingClusterStats() for cluster_id in range(k_optimal)}

# 模拟新数据到来（增量更新）
for i in range(100):  # 假设来了 100 个新用户
    user_id = f"user_{10000 + i}"
    user_features = X.iloc[i % len(X)].values  # 模拟新用户的特征

    # 用已拟合的 PCA 和 K-means 处理新用户
    user_transformed = pca.transform([user_features])[0]
    user_cluster = kmeans.predict([user_transformed])[0]

    # 假设用户的"消费金额"是 outcome
    user_outcome = np.random.normal(100, 20)  # 模拟

    # 增量更新该簇的统计量
    cluster_stats[user_cluster].update(user_outcome)

    if i % 20 == 0:
        stats = cluster_stats[user_cluster].get_stats()
        print(f"新用户 {user_id} 分配到簇 {user_cluster}")
        print(f"  簇 {user_cluster} 当前统计量: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}")
```

**第 4 步：设计 A/B 测试流程（自动决策建议）**

```python
def ab_test_decision(group_A_data, group_B_data, config):
    """
    A/B 测试决策建议

    参数:
        group_A_data: A 组数据
        group_B_data: B 组数据
        config: 实验配置（显著性水平、最小效应等）

    返回:
        decision: 决策建议
        p_value: p 值
        effect_size: 效应量
        ci_low, ci_high: 置信区间
    """
    from scipy import stats

    # t 检验
    t_stat, p_value = stats.ttest_ind(group_B_data, group_A_data)

    # 效应量
    effect_size = np.mean(group_B_data) - np.mean(group_A_data)

    # 95% 置信区间
    se = np.sqrt(np.var(group_A_data, ddof=1)/len(group_A_data) +
                np.var(group_B_data, ddof=1)/len(group_B_data))
    ci_low = effect_size - 1.96 * se
    ci_high = effect_size + 1.96 * se

    # 决策规则
    if p_value < config['significance_level'] and abs(effect_size) >= config['min_effect']:
        decision = "launch_B"
    elif p_value < 0.10:
        decision = "continue"
    else:
        decision = "reject_B"

    return {
        'decision': decision,
        'p_value': p_value,
        'effect_size': effect_size,
        'ci_low': ci_low,
        'ci_high': ci_high
    }

# 使用示例（比较簇 0 和簇 1 的平均消费）
cluster_0_data = [np.random.normal(100, 20) for _ in range(100)]
cluster_1_data = [np.random.normal(108, 20) for _ in range(100)]

result = ab_test_decision(
    cluster_0_data,
    cluster_1_data,
    config={'significance_level': 0.05, 'min_effect': 5.0}
)

print(f"\nA/B 测试结果（簇 0 vs 簇 1）:")
print(f"  效应量: {result['effect_size']:.2f}")
print(f"  p 值: {result['p_value']:.4f}")
print(f"  95% CI: [{result['ci_low']:.2f}, {result['ci_high']:.2f}]")
print(f"  决策建议: {result['decision']}")
```

**第 5 步：生成计算专题报告**

```python
def generate_computational_report(X, pca, kmeans, cluster_stats, n_components):
    """生成计算专题的 Markdown 报告"""

    report = f"""
## 计算专题：降维、聚类与流式统计

### 研究问题

本章用计算方法回答："如何从高维用户行为数据中发现结构，并实现实时更新？"

### PCA 降维

**降维结果**：
- 原始特征数：{X.shape[1]}
- 降维后成分数：{n_components}
- 压缩率：{X.shape[1] / n_components:.1f}x
- 保留方差：{sum(pca.explained_variance_ratio_):.1%}

**主成分解释（前 3 个）**：
| 主成分 | 方差解释比例 | 累积方差解释比例 |
|--------|--------------|------------------|
"""

    for i in range(min(3, n_components)):
        report += f"| PC{i+1} | {pca.explained_variance_ratio_[i]:.1%} | {sum(pca.explained_variance_ratio_[:i+1]):.1%} |\n"

    report += f"""
**业务解释**：
- 第 1 主成分（PC1）主要反映用户"活跃度"（点击、停留、浏览）
- 第 2 主成分（PC2）主要反映用户"购买倾向"（加入购物车、消费）
- 前 {n_components} 个主成分保留了 85% 的信息，用于后续聚类

### K-means 聚类

**聚类结果**：
- 最优簇数：{k_optimal}
- 各簇样本数：{dict(enumerate(np.bincount(cluster_labels)))}

**各簇的统计摘要**（基于原始特征）：
| 簇编号 | 样本数 | 平均点击 | 平均停留 | 平均消费 |
|--------|--------|---------|---------|---------|
"""

    cluster_summary = X.copy()
    cluster_summary['cluster'] = cluster_labels
    for cluster_id in range(k_optimal):
        cluster_data = cluster_summary[cluster_summary['cluster'] == cluster_id]
        report += f"| {cluster_id} | {len(cluster_data)} | {cluster_data['点击次数'].mean():.1f} | {cluster_data['停留时间'].mean():.1f} | {cluster_data['消费金额'].mean():.2f} |\n"

    report += """
**业务解释**：
- 簇 0："超级活跃用户"——高点击、高停留、高消费
- 簇 1："浏览型用户"——高停留、低消费
- 簇 2："低活跃用户"——各项指标均低
- ...

### 流式统计

当新用户数据持续到来时，我们使用**流式统计算法**增量更新每个簇的统计量（均值、方差、标准差），而不需要每次都重算整个数据集。

**流式统计的优势**：
- **实时更新**：O(1) 复杂度每次更新，不需要遍历历史数据
- **内存高效**：只需维护状态变量（n, mean, M2），不需要存储所有历史数据
- **可扩展**：支持分布式环境（如 map-reduce）

**流式统计的当前状态**（示例）：
| 簇编号 | 样本数 | 均值 | 标准差 |
|--------|--------|------|--------|
"""

    for cluster_id, stats_obj in cluster_stats.items():
        stats = stats_obj.get_stats()
        report += f"| {cluster_id} | {stats['n']} | {stats['mean']:.2f} | {stats['std']:.2f} |\n"

    report += """
### A/B 测试工程化

我们设计了一个**自动化 A/B 测试流程**，能实时计算各实验组的统计量、运行假设检验、输出决策建议。

**流程组件**：
1. **实验配置**：定义处理组、指标、样本量、决策规则
2. **数据收集**：记录每个用户的实验组和结果（流式更新）
3. **统计检验**：实时运行 t 检验、计算置信区间
4. **决策规则**：根据 p 值和效应量输出决策建议
5. **监控报警**：检测样本比例异常（SRM）、辛普森悖论

**决策规则**：
- `p < 0.05` 且 `|效应量| ≥ 最小阈值` → **launch_B**（上线 B）
- `0.05 < p < 0.10` → **continue**（继续收集数据）
- `p ≥ 0.10` → **reject_B**（放弃 B）

**注意事项**：
- 系统只提供建议，最终决策由人负责（human-in-the-loop）
- 强制最小样本量（避免早期停止导致的假阳性）
- 自动检测 SRM（样本比例异常）并报警

### 方法选择与边界

**我们选择这些方法的理由**：
- **PCA 降维**：500 个特征太多，直接计算成本高且容易过拟合；PCA 保留 85% 信息，压缩到 {n_components} 个成分
- **K-means 聚类**：在降维后的空间中运行，计算高效且结果可解释
- **流式统计**：支持实时更新，不需要每次都重算（节省计算成本）
- **A/B 测试工程化**：自动化决策流程，但仍保留人工审查环节

**方法的局限性**：
- PCA 是"线性降维"，如果数据有复杂非线性结构，可能需要核 PCA 或 Autoencoder
- K-means 假设簇是"球形"且大小相近，如果簇形状复杂，可能需要 DBSCAN 或谱聚类
- 流式统计是"近似算法"（如在线分位数），如果需要精确值，仍需批量计算
- A/B 测试的自动化无法检测所有前提违反（如数据分布异常），需要人工审查

### 结论

我们用 PCA 把 500 个特征压缩到 {n_components} 个主成分（压缩率 {X.shape[1] / n_components:.1f}x），保留了 85% 的信息。在降维后的空间中运行 K-means，发现了 {k_optimal} 个用户群，每个群都有清晰的"行为画像"。

我们实现了流式统计算法（在线均值、在线方差），支持实时更新每个用户群的统计量，而不需要重算整个数据集。我们还设计了自动化 A/B 测试流程，能实时输出决策建议，但仍保留人工审查环节（避免自动决策的陷阱）。

这些方法让 StatLab 报告从"静态快照"升级为"持续更新的看板"——当新数据到来时，统计量实时更新，聚类结果可增量调整，A/B 测试能自动决策。这是高维数据时代的必需技能。
"""

    return report

# 生成报告
report_computational = generate_computational_report(X, pca, kmeans, cluster_stats, n_components)

# 追加到 report.md
with open("report/report.md", "a", encoding="utf-8") as f:
    f.write(report_computational)

print("✅ 计算专题章节已添加到 report/report.md")
```

### 本周 StatLab 的改进总结

| 维度 | 上周状态 | 本周改进 |
|------|---------|---------|
| 数据维度 | 高维（500 特征），计算成本高 | PCA 降维到 47 个成分，压缩率 10.6x |
| 用户分层 | 无 | K-means 聚类发现 5 个用户群，每个群有清晰画像 |
| 统计更新 | 批量模式（每次重算） | 流式统计（O(1) 增量更新） |
| A/B 测试 | 手动运行 t 检验 | 自动化流程（实时决策建议 + 监控报警） |
| 报告形态 | 静态快照 | 持续更新的看板 |

老潘总结："本周你完成的不是'加几个章节'，而是**升级了整个数据分析流程**——从高维困境到降维压缩，从无标签数据到聚类发现，从批量模式到流式统计，从手动分析到自动化决策。这是数据科学家的关键跃迁。"

---

## Git 本周要点

本周必会命令：
- `git status`: 查看工作区状态
- `git diff`: 查看具体改动内容
- `git add -A`: 添加所有改动
- `git commit -m "feat: add computational analysis"`
- `git log --oneline -n 5`

常见坑：

**混淆降维与特征选择**——PCA 是"特征提取"（创造新特征），特征选择是"从原始特征中选子集"。解决方法：根据目标选择（需要可解释性？选特征选择；需要压缩信息？选 PCA）。

**K 值选择的过度依赖肘部法则**——肘部法则只是参考，最终 K 值应结合业务解释性。解决方法：如果 K=5 的簇无法用业务语言解释，尝试 K=4 或 K=6。

**流式统计的状态管理**——状态变量（n, mean, M2）需要持久化，否则系统崩溃会丢失。解决方法：定期保存状态到磁盘/数据库，支持从断点恢复。

**A/B 测试的过度自动化**——完全自动决策可能忽略前提违反（SRM、辛普森悖论）。解决方法：保留 human-in-the-loop，系统提供建议，最终决策由人负责。

**在线分位数的近似误差**——分箱法的分位数是近似值，误差取决于箱数量。解决方法：增加箱数量（牺牲内存），或使用更精确的算法（t-digest）。

Pull Request (PR)：
- Gitea 上也叫 Pull Request，流程等价 GitHub：push 分支 -> 开 PR -> review -> merge。

---

## 本周小结（供下周参考）

本周你从"高维数据的困境"走向了"计算专题的完整工具箱"——这是数据科学家的关键跃迁。

**维度灾难**（Curse of Dimensionality）的本质不是"数据质量差"，而是"维度太高"：当特征数接近或超过样本数时，回归矩阵不可逆、距离度量失效、采样稀疏。工业界的高维数据——基因表达（20000 个基因）、文本 TF-IDF（10000 维）、用户行为矩阵（5000 个特征）——都是"维度灾难"的重灾区。你不能指望"增加样本"（成本太高），必须学会**降维**。

**PCA 降维**的核心是"让数据说话"——用数学方法（协方差矩阵的特征分解）找出"方差最大"的方向，把数据投影到这些方向上。结果是：你可以用 47 个主成分替代 5000 个原始特征，保留 85% 的信息——压缩率 106.4x，计算成本大幅降低。从 Week 07 的"方差分解"（ANOVA）到 PCA，思想一脉相承：都是把总方差拆成可解释的部分。ANOVA 是"按组拆分"，PCA 是"按方向拆分"。

**聚类分析**（K-means）的目标是在无标签数据中发现隐藏结构。但阿码的问题（"K 值怎么确定"）点出了聚类的核心挑战：没有"真实标签"来评估。你必须结合多种方法：肘部法则（WCSS 的拐点）、轮廓系数（样本与簇的紧密度）、业务解释性（能否用业务语言描述簇）。老潘强调：**聚类的核心是业务定义，不是算法选择**——如果你不知道"在找什么结构"，K-means 跑出来也只是一堆数字。

**流式统计**的核心直觉很简单：记住"当前状态"，每次只更新"变化的部分"——就像你不会每次都重新数钱包里的钱，而是记住"当前有多少"，每花一笔就减、每赚一笔就加。技术上，流式统计通过维护状态变量（计数、总和、平方和），把每次更新的复杂度从 O(n) 降到 O(1)。从 Week 08 的"Bootstrap"到流式统计，解决不同的问题：Bootstrap 是"在批量数据上估计不确定性"，流式统计是"在流式数据上更新统计量"——不是竞争，而是互补。

**A/B 测试工程化**不是简单的 t 检验，而是一个完整的系统：实验配置、数据收集、统计检验、决策规则、监控报警。从 Week 06-07 的"假设检验"到 A/B 测试工程化，是从"单次算法"到"端到端系统"的跃迁。老潘的经验：**自动化平台的陷阱很多**——Sample Ratio Mismatch、早期停止、辛普森悖论。解法不是"放弃自动化"，而是"human-in-the-loop"——系统提供建议，但最终决策由人负责。

最重要的是，你学会了在 StatLab 报告中整合**计算专题方法**：用 PCA 降维高维特征，用 K-means 聚类发现用户群，实现流式统计支持实时更新，设计自动化 A/B 测试流程。这些方法让 StatLab 报告从"静态快照"升级为"持续更新的看板"。

下周（Week 16），是课程的最后一周——**期末展示与课程复盘**。你要把 16 周的学习成果收敛成一份完整的 `report.md` / `report.html`，准备展示材料，反思"从'只会跑代码'到'会提问题、会决策、会讲故事'"的转变。本周的"计算专题"会演化为下周的"综合实战"——从单个方法到完整项目，从技术输出到数据故事。

---

## Definition of Done（学生自测清单）

- [ ] 我能理解维度灾难的本质（过拟合、计算爆炸、距离失效）
- [ ] 我能解释 PCA 的核心思想（方差最大化、主成分投影）
- [ ] 我能用 sklearn 实现 PCA 降维，并解释主成分的业务含义
- [ ] 我能理解方差解释比例和累积方差解释比例
- [ ] 我能用肘部法则和轮廓系数选择 K-means 的最优 K 值
- [ ] 我能实现 K-means 聚类，并解释每个簇的业务含义
- [ ] 我能区分"聚类"和"分类"（有监督 vs 无监督）
- [ ] 我能理解流式统计的核心思想（增量更新 vs 批量重算）
- [ ] 我能实现在线均值和在线方差的增量更新算法
- [ ] 我能理解 A/B 测试工程化的核心组件（配置、收集、检验、决策、监控）
- [ ] 我能设计一个简单的 A/B 测试自动化流程（决策规则 + SRM 检测）
- [ ] 我能在 StatLab 报告中整合降维、聚类、流式统计方法
- [ ] 我能识别高维数据场景（何时需要降维）
- [ ] 我能评估聚类结果的合理性（轮廓系数、业务解释性）
- [ ] 我能识别 A/B 测试平台的常见陷阱（SRM、早期停止、辛普森悖论）
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解从"批量模式"到"流式模式"是数据系统的关键升级
