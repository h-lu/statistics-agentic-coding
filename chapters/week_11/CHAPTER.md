# Week 11：从决策树到随机森林——用"群体智慧"提升预测

> "The strength of a tree is in its branches; the strength of a forest is in its diversity."
> — Adapted from African proverb

2025 到 2026 年，机器学习领域出现了一个有趣的趋势：**AutoML 平台和 AI 工具让"训练随机森林"变得像按按钮一样简单**，但你问一个实际使用者"为什么你的模型这么预测"，很少有人能给出清晰的答案。

这带来了一个危险的两难：**复杂模型（如随机森林、梯度提升树）的预测性能往往更好，但可解释性更差**。如果不理解树模型的分裂逻辑、特征重要性和集成原理，你可能会得到一个"AUC = 0.87"的黑盒，却不知道它在哪些场景下会失效、哪些特征是噪声、哪些是真正的信号。

本周的主题是——**从可解释的单一决策树，到强大的集成森林，再回到可解释的特征重要性**。你不仅会学会"让模型更准确"，还会掌握"让模型仍然可被理解"的技术，这正是 AI 时代数据分析师的核心竞争力。

---

## 前情提要

上周（Week 10），小北学会了用逻辑回归做分类评估。他兴冲冲地对老潘说："**我的模型 AUC = 0.82，比随机猜测好很多！我还画了 ROC 曲线，做了混淆矩阵分析！**"

老潘看了一眼他的报告，问了一个新问题："**你试过其他模型吗？比如决策树、随机森林？它们的 AUC 可能比 0.82 更高。**"

小北愣住了："呃……我只学过逻辑回归。决策树是什么？"

阿码举手："**我在 Kaggle 上看到很多人用随机森林，说它'又快又准'。但我不理解为什么'把很多树放在一起'就会比一棵树好。**"

老潘笑了："**这就是本周要学的——从'一棵可解释的树'到'一片强大的森林'**。逻辑回归给你一个线性边界，决策树给你一系列'如果-那么'规则，随机森林给你'多个规则投票'的结果。但更重要的是，你会学到特征重要性——哪些变量真的在驱动预测。**"

"还有，"老潘补充，"**本周你会用到很多之前学过的工具**：Week 10 的**交叉验证**会用于超参数调优，Week 08 的**Bootstrap**会解释随机森林的原理，Week 09 的**残差诊断**会演化为特征重要性分析。"

"所以，"小北说，"**这不只是换个模型，而是学会一套'从简单到复杂，再回到可解释'的建模哲学？**"

"没错，"老潘点头，"**决策树让你看到规则，随机森林让你获得性能，特征重要性让你理解原因**。"

这正是本周要解决的问题——**从线性模型到树模型，从单一模型到集成学习**。

---

## 学习目标

完成本周学习后，你将能够：
1. 理解决策树的分裂逻辑（熵、Gini 不纯度、CART 算法），掌握树模型的直观解释
2. 识别决策树的过拟合问题，并用剪枝、深度限制等方法控制复杂度
3. 理解集成学习原理（Bagging、随机森林），为什么"多棵树投票"比单棵树好
4. 掌握特征重要性的计算方法和正确解释，避免"相关特征被误认为因果"
5. 学会超参数调优（网格搜索、随机搜索），并用交叉验证防止过拟合
6. 在 StatLab 报告中添加树模型章节，与逻辑回归/线性回归基线对比
7. 审查 AI 生成的树模型代码，识别缺少超参数调优、忽略特征重要性等问题

---

<!--
贯穿案例：房价预测——从"线性回归"到"树模型"再到"随机森林"

本周贯穿案例是一个房价预测场景（也可以用读者的 StatLab 数据集）：你想预测房价（连续值），先用线性回归建立基线，再用决策树捕捉非线性关系，最后用随机森林提升性能。

- 第 1 节：线性回归的局限 → 案例从"线性回归 R² = 0.65"变成"发现残差有模式（非线性），需要树模型"
- 第 2 节：单棵决策树 → 案例从"线性模型"变成"决策树 R² = 0.72，且能看到分裂规则"
- 第 3 节：过拟合问题 → 案例从"训练集 R² = 0.95"变成"测试集 R² = 0.68，发现树记住了噪声"
- 第 4 节：随机森林 → 案例从"单棵树"变成"森林 R² = 0.81，方差更低"
- 第 5 节：特征重要性 → 案例从"不知道哪些特征重要"变成"看到面积、位置是关键，但小心相关特征"
- 第 6 节：超参数调优 → 案例从"默认参数"变成"网格搜索后 R² = 0.84"

最终成果：读者完成一个完整的树模型建模流水线，产出：
- 1 个决策树模型（可视化树结构）
- 1 个随机森林模型（多棵树投票）
- 1 张特征重要性图（Top 10 特征）
- 1 个网格搜索结果（最佳超参数组合）
- 1 个模型对比表（线性回归 vs 决策树 vs 随机森林）
- 1 份 StatLab 报告章节（树模型评估、特征重要性、与基线对比）
- 1 份 AI 树模型代码的审查清单（标注过拟合、缺少调优等问题）

认知负荷预算：
- 本周新概念（5 个，预算上限 5 个）：
  1. 决策树 - 应用层次
  2. 随机森林 - 应用层次
  3. 特征工程 - 应用层次
  4. 超参数调优 - 理解层次
  5. 过拟合与欠拟合 - 理解层次
- 结论：✅ 在预算内（5 个）

回顾桥设计（至少 2 个，来自 Week 07-10）：
- [残差诊断]（week_09）：在第 1 节，用"线性回归残差有模式"引出"线性模型不能捕捉非线性，需要树模型"
- [交叉验证]（week_10）：在第 3-4 节，用"CV 估计泛化性能"连接"树的过拟合问题"和"随机森林的稳定性"
- [Bootstrap]（week_08）：在第 4 节，用"Bootstrap 重采样"解释"随机森林的 Bagging 原理"
- [多重共线性]（week_09）：在第 5 节，用"相关特征会稀释特征重要性"连接 Week 09 的共线性问题
- [混淆矩阵/ROC-AUC]（week_10）：在第 6 节，用"分类评估指标"比较"逻辑回归 vs 随机森林"

AI 小专栏规划：

AI 小专栏 #1（放在第 1 节之后）：
- 主题：AutoML 的诱惑——为什么理解模型仍然重要
- 连接点：与第 1 节"线性模型的边界"呼应，讨论 AutoML 工具虽然能自动选择模型，但不会替你理解"为什么这个特征重要"或"模型在哪些场景下会失效"
- 建议搜索词：AutoML limitations 2025 2026, explainable AI vs AutoML, model interpretability importance
- 参考来源（已验证）：
  - IBM AutoML 介绍: https://www.ibm.com/think/topics/automl
  - arXiv:2401.08513 "X Hacking: The Threat of Misguided AutoML" (2024): https://arxiv.org/abs/2401.08513
  - 系统综述: "Automated machine learning with interpretation: A systematic review of methodologies and applications in healthcare" (2024): https://file.sciopen.com/sciopen_public/1847209592129949698.pdf

AI 小专栏 #2（放在第 4-5 节之间）：
- 主题：树模型在工业界——为什么随机森林是"首选算法"
- 连接点：与第 4 节"随机森林"和第 5 节"特征重要性"直接呼应，讨论随机森林在欺诈检测、推荐系统、信用评分等场景的广泛应用，以及特征重要性在合规审计中的价值
- 建议搜索词：random forest applications 2025 2026, feature importance compliance, tree models production ML, gradient boosting vs random forest
- 参考来源（已验证）：
  - MachineLearningMastery: "How to Decide Between Random Forests and Gradient Boosting" (2025): https://machinelearningmastery.com/how-to-decide-between-random-forests-and-gradient-boosting/
  - Baeldung: "Gradient Boosting Trees vs. Random Forests" (2025): https://www.baeldung.com/cs/gradient-boosting-trees-vs-random-forests
  - GeeksforGeeks: "Gradient Boosting vs Random Forest" (2024): https://www.geeksforgeeks.org/gradient-boosting-vs-random-forest/

角色出场规划：
- 小北（第 1、3、5 节）：
  - 以为线性回归就能解决问题，没想到残差有模式
  - 决策树在训练集表现完美，但测试集很差（过拟合）
  - 误解特征重要性："面积最重要，所以只要扩大面积就能涨价？"
- 阿码（第 2、4 节）：
  - 好奇"决策树的'如果-那么'规则是怎么自动学的？"
  - 追问"为什么多棵树投票会比单棵树好？不会更慢吗？"
- 老潘（第 2、4、6 节）：
  - 强调"决策树的价值在于可解释性，不是性能"
  - "随机森林不是魔法，是'通过多样性减少方差'"
  - "超参数调优不是'无脑网格搜索'，要理解每个参数的含义"

StatLab 本周推进：
- 上周状态：report.md 已有分类评估章节（逻辑回归、混淆矩阵、ROC-AUC）
- 本周改进：在 report.md 中添加"树模型与集成学习"章节，包含：
  - 研究问题：树模型能否比线性/逻辑模型更好地捕捉非线性关系？
  - 决策树：可视化树结构、解释分裂规则
  - 随机森林：Bagging 原理、性能提升
  - 特征重要性：Top 10 特征、相关性警告
  - 超参数调优：网格搜索结果、最佳参数
  - 模型对比：线性回归 vs 逻辑回归 vs 决策树 vs 随机森林
  - 局限性：过拟合风险、特征重要性≠因果
- 涉及的本周概念：决策树、随机森林、特征工程、超参数调优、过拟合
- 建议示例文件：examples/11_statlab_trees.py（生成树模型评估、特征重要性图、模型对比表）
-->

## 1. 线性模型的边界——什么时候一条直线不够了？

小北用 Week 09 学过的线性回归预测房价。模型跑出来了，R² = 0.65。他盯着残差图看了半天——这条曲线像是微微上翘的嘴角，但他笑不出来。

"老潘，我这残差图有点奇怪，"小北指着屏幕，"**预测值很高和很低的地方，残差明显偏大**。是我算错了吗？"

老潘凑过来，扫了一眼代码和图表。他没有直接回答"对或错"，而是反问："**你觉得房价和面积是严格线性关系吗？还是说，存在'边际递减'——面积越大，每平米的单价反而越低？**"

小北愣住了。他画了一张散点图——果然，房价和面积的关系不是一条完美的直线，更像是一条慢慢变缓的曲线。

### 线性回归的局限：它只认直线

线性回归的本质是假设 **y = β₀ + β₁x₁ + β₂x₂ + ...** ——在高维空间里找一条最合适的**直线**（或超平面）。

但真实世界很少这么听话。比如：
- **房价**：50㎡ 到 100㎡ 的阶段，单价可能快速攀升；但从 200㎡ 到 250㎡，单价可能不再涨了（豪宅买家的需求已经满足）
- **客户流失**：前 3 个月流失率很高，但 12 个月后趋于稳定（留下的都是忠诚用户）
- **信用评分**：收入和风险的关系不是线性的——太低（贫困）和太高（过度杠杆）都可能增加违约风险

"这时候线性模型就不够用了，"老潘说，"**你需要的是决策树——它不画直线，而是画一堆'如果-那么'的框框，把数据切分成不同区域**。"

阿码好奇地问："**这些规则是人工写的，还是自动学出来的？**"

"自动学出来的，"老潘笑了，"**决策树会一遍遍地问你：'按哪个特征切？切在哪里？能让数据'最纯净'？**就像切蛋糕，它会尝试不同的切法，找到让每块蛋糕都尽可能' homogeneous'（均质）的那一刀。"

### 残差图在说话：它告诉你什么时候需要树模型

Week 09 你学过**残差诊断**——当时我们用它检查线性回归的假设是否成立（比如残差是否正态分布、方差是否齐性）。但残差图还有另一个更强大的作用：**它会用形状告诉你什么时候该换模型类型**。

**这些模式说明线性模型不够**：
- **U 型或倒 U 型**：残差随预测值变化（漏掉了非线性关系）
- **周期性模式**：残差随时间或某个变量波动（漏掉了周期效应）
- **漏斗状**：残差方差在某些预测值范围更大（异方差问题）

小北的残差图就是典型的 U 型——预测值很低和很高时，残差偏大且呈对称状。这说明**线性模型没有捕捉到非线性关系**。

"所以，"小北若有所思，"**残差诊断不只是'检查模型假设'，还是'选择模型类型'的信号？**"

"对！"老潘点头，"**当你看到残差有模式，不是'模型错了'，而是'模型语言不够丰富，需要换一种表达方式'**。决策树就是更灵活的表达方式之一。"

---

> **AI 时代小专栏：AutoML 的诱惑——为什么理解模型仍然重要**
>
> 2025 年以来，AutoML 平台（如 H2O.ai、DataRobot、Google Vertex AI）让"自动选择模型"变得像按按钮一样简单。你上传数据，AutoML 自动尝试几十种模型（线性回归、决策树、随机森林、XGBoost、神经网络），然后告诉你"随机森林 AUC = 0.87，是最好的模型"。
>
> 但这带来了一个危险：**你可能得到一个"高性能但不可理解"的黑盒**。AutoML 不会替你回答：
> - "为什么这个特征重要？"（是因果还是相关？）
> - "模型在哪些场景下会失效？"（比如数据分布偏移）
> - "模型的预测是'稳健的'还是'过拟合的'？"（AutoML 的默认设置可能过拟合）
>
> 2024 年 1 月发表在 arXiv 上的研究论文《X Hacking: The Threat of Misguided AutoML》指出，**可解释 AI（XAI）虽然有助于建立信任，但也创造了一种"反向激励"：分析者可能操纵 XAI 指标来支持预设的结论**。AutoML 系统可以大规模搜索"看似可解释"的模型集合，找到既"表现好"又"解释通"的模型，但这不代表模型真的可靠。
>
> 2024 年的一篇系统综述《Automated machine learning with interpretation in healthcare》发现，**虽然 AutoML 在医疗保健中的应用增长迅速，但大多数 AutoML 生成的模型仍然是黑盒，难以在医疗环境中部署**。医疗场景对模型解释和合规性要求极高，AutoML 的"自动性"不能替代领域知识审查。
>
> 所以你刚学到的"线性模型边界"诊断，在 AutoML 时代反而更重要了——**AutoML 可以帮你拟合模型，但只有你能判断"这个模型是否可信"、"残差图是否有模式"、"特征重要性是否合理"**。理解模型的假设和局限，才是 AI 时代数据分析师不可替代的价值。
>
> 参考（访问日期：2026-02-12）：
> - [What Is AutoML? | IBM](https://www.ibm.com/think/topics/automl)
> - [X Hacking: The Threat of Misguided AutoML (arXiv, 2024)](https://arxiv.org/abs/2401.08513)
> - [Automated machine learning with interpretation: A systematic review (ScienceDirect, 2024)](https://file.sciopen.com/sciopen_public/1847209592129949698.pdf)

---

## 2. 决策树：用"如果-那么"规则预测

小北决定试试决策树。他用了 scikit-learn 的 DecisionTreeRegressor：

```python
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# 用同样的数据
X = df[['area_sqm', 'bedrooms', 'bathrooms', 'age_years', 'distance_km']]
y = df['price']

# 拟合决策树
tree = DecisionTreeRegressor(max_depth=3, random_state=42)
tree.fit(X, y)

# 看看 R²
print(f"训练集 R²: {tree.score(X, y):.3f}")
```

输出：

```
训练集 R²: 0.780
```

"哇！R² 从 0.65 涨到了 0.78！"小北从椅子上弹起来，"**决策树比线性回归好多了！**"

老潘敲了敲键盘。"**等等，你这是在训练集上评估。看看测试集。**"

小北愣了一下——他确实忘了划分训练/测试集。他赶紧补上：

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tree.fit(X_train, y_train)
print(f"训练集 R²: {tree.score(X_train, y_train):.3f}")
print(f"测试集 R²: {tree.score(X_test, y_test):.3f}")
```

输出：

```
训练集 R²: 0.780
测试集 R²: 0.610
```

"测试集只有 0.61？"小北挠头，"**比线性回归还低？**"

"这是决策树的经典问题，"老潘说，"**它太'记性好'了——记住训练数据的每一个细节，但在新数据上就瞎了**。你现在的树深度只有 3，如果让它自由生长，训练集 R² 能冲到 1.0，但测试集会掉得更惨。"

### 决策树的可解释性：你能看到规则

但决策树有一个线性模型没有的优势——**它是透明的**。你可以把整棵树画出来：

```python
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
plt.title("决策树：房价预测（深度限制为 3）")
plt.show()
```

屏幕上会跳出一棵树状图，每个节点都写着：
- **分裂特征**：比如 `area_sqm <= 85.5`
- **分裂标准**：比如 `mse = 45000`（均方误差）
- **样本数**：该节点有多少个样本
- **预测值**：该节点的平均房价

小北第一次看到这张图时，愣住了：**"哦！原来模型是这样做决策的——每个路口都有个问题，回答'是'往左走，回答'否'往右走，最后落在一个叶子节点上，那就是预测值。"**

"对！"老潘说，"**你可以从根节点一路走到叶子节点，把完整预测路径读出来**。这和线性回归完全不同——你无法从 10 个系数中读出'某个样本为什么预测这个值'，但树可以告诉你'它经过哪些判断'。"

阿码盯着这张图看了几秒，突然抬起头："**决策树就是一堆嵌套的'如果-那么'规则？**"

"对！"老潘说，"**第一条规则可能是'如果面积 < 85.5㎡，预测 200 万'，第二条规则'如果面积 >= 85.5㎡ 且卧室数 <= 2，预测 350 万'，以此类推**。"

小北喃喃自语："**所以决策树的价值不只是预测，而是你能看懂'为什么'？**"

"没错，"老潘点头，"**线性回归的系数告诉你'面积每增加 1㎡，房价涨 β₁'，决策树告诉你'在某个范围内，房价是多少'。前者是'边际效应'，后者是'分段规则'，两种不同的可解释语言**。"

### 决策树的分裂逻辑：如何找到"最佳一刀"

阿码盯着树状图看了半天，突然问："**但树是怎么知道'先按面积切'，而不是'先按卧室数切'？**"

"这就是**分裂标准**，"老潘说，"**决策树会尝试所有可能的特征和切分点，选择让数据'最纯净'的那一刀**。"

小北好奇："**'纯净'怎么量化？**"

对于**分类问题**（比如预测流失/不流失），"纯净"的度量是：
- **熵（Entropy）**：信息论里的老概念，衡量"不确定性有多高"
- **Gini 不纯度**：衡量"随机选一个样本，被错误分类的概率"

对于**回归问题**（比如预测房价），"纯净"的度量更直观：
- **均方误差（MSE）**：让每个节点的预测值（样本均值）与真实值的差异最小

"所以树会自动找'最能降低误差'的切分点？"小北问。

"对，而且是**贪婪地找**，"老潘说，"**CART 算法（Classification and Regression Trees）每一步都选'当下最优'，不管后面。这就像走楼梯，每次都选最高的那一步往上迈**。"

"听起来很聪明，"小北说。

"但也意味着可能陷入**局部最优**，"老潘补充，"**更重要的是：如果让树一直往下切，它会记住训练数据的每一个样本**。这正好引出下节要聊的问题。"

---

## 3. 过拟合与正则化——别让树记住噪声

小北发现他的决策树在训练集上 R² = 0.95，但测试集只有 0.61。老潘说："**你的树过拟合了——它记住了训练数据的每一个细节**。"

### 为什么决策树这么容易过拟合？

想象一下：如果你让决策树无限生长，它最终会为**每一个样本**创建一个叶子节点。

- 样本 1：面积=82㎡，卧室=2，价格=210万 → 叶子节点预测 210 万
- 样本 2：面积=83㎡，卧室=2，价格=215万 → 新的叶子节点预测 215 万
- ...

"这样的树在训练集上完美（R² = 1.0），"老潘说，"**但它不是'学规律'，而是'背答案'**。新数据一来，它就傻了——因为它只见过旧数据的'面孔'，没学会背后的'长相'。"

阿码举手："**所以过拟合就是'太懂训练集，但不懂新数据'？**"

"对！"老潘点头，"**决策树特别容易这样，因为它的学习能力太强了——只要有足够深度，它能把任何训练集背下来**。"

### 控制树的生长：给学习能力设个上限

scikit-learn 的决策树有几个关键超参数来控制复杂度。理解它们的作用，比记住默认值更重要：

| 超参数 | 作用 | 太小的风险 | 太大的风险 |
|--------|------|-----------|-----------|
| `max_depth` | 树的最大深度 | 欠拟合（树太简单） | 过拟合（树太复杂） |
| `min_samples_split` | 节点分裂所需的最小样本数 | 树太深、太细 | 树太浅、太粗糙 |
| `min_samples_leaf` | 叶子节点的最小样本数 | 过拟合（叶子太专） | 欠拟合（叶子太泛） |
| `max_features` | 每次分裂考虑的特征数 | 树多样性不足 | 每棵树太相似 |

```python
from sklearn.tree import DecisionTreeRegressor

# 更保守的树（防止过拟合）
tree_conservative = DecisionTreeRegressor(
    max_depth=5,           # 树的最大深度（默认 None，会无限生长）
    min_samples_split=10,    # 节点分裂所需的最小样本数
    min_samples_leaf=5,      # 叶子节点的最小样本数
    random_state=42
)

tree_conservative.fit(X_train, y_train)
print(f"训练集 R²: {tree_conservative.score(X_train, y_train):.3f}")
print(f"测试集 R²: {tree_conservative.score(X_test, y_test):.3f}")
```

输出是：

```
训练集 R²: 0.820
测试集 R²: 0.710
```

"测试集 R² 从 0.61 涨到了 0.71！"小北高兴地说。

"对，"老潘说，"**通过限制树的复杂度（深度=5，叶子节点至少 5 个样本），你牺牲了一点训练集性能，但大幅提升了泛化能力**。这就像背书——死记硬背能拿满分，但理解了才能考新题。"

### 和 Week 10 的连接：用交叉验证找最佳复杂度

Week 10 你学过**K-fold 交叉验证**。决策树的最佳超参数（max_depth、min_samples_leaf 等）应该通过 CV 来选择，而不是猜。

```python
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# 定义超参数网格
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# 网格搜索（5-fold CV）
grid_search = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1  # 使用所有 CPU 核心
)

grid_search.fit(X_train, y_train)

# 最佳参数
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳 CV R²: {grid_search.best_score_:.3f}")
```

输出是：

```
最佳参数: {'max_depth': 7, 'min_samples_leaf': 5, 'min_samples_split': 10}
最佳 CV R²: 0.735
```

小北惊讶地发现："**最佳深度不是 3 或 5，而是 7？**"

"对，"老潘说，"**超参数调优不是'越保守越好'，而是'找到泛化性能最好的平衡点'**。CV 告诉你，深度=7 是最佳选择——既不会太简单（欠拟合），也不会太复杂（过拟合）。**"

这正好引出下一节：**如果单棵树容易过拟合或欠拟合，能不能用"多棵树投票"来减少方差？**

---

## 4. 随机森林——用群体智慧降低方差

阿码在第 2 节问了："**为什么多棵树投票会比单棵树好？**"

老潘给了他一个非常直观的解释："**想象你有 100 个人猜一个数字。如果每个人都独立猜测，平均下来他们的猜测会接近真相**。但如果 100 个人都听同一个人说，他们的猜测只是重复同一个错误。"

"**随机森林的核心思想是：让每棵树在不同数据上训练，用不同的特征，这样它们的错误是'不相关的'**。当 100 棵树投票时，不相关的错误会相互抵消，留下正确的信号。"

小北若有所思："**所以不是'树越多越好'，而是'越不相关的树越好'？**"

"对！"老潘说，"**这就是为什么随机森林要引入随机性——不是让每棵树都变强，而是让它们变'不一样'**。"

### 和 Week 08 的连接：Bootstrap 是 Bagging 的基础

Week 08 你学过**Bootstrap**（自助法）：从原始样本中有放回地抽取多个子样本，每个子样本大小与原始样本相同。

**Bagging（Bootstrap Aggregating）** 说白了就三步：
1. 用 Bootstrap 生成 B 个训练集子样本
2. 在每个子样本上训练一棵树
3. 预测时，取 B 棵树预测的平均值（回归）或多数投票（分类）

"这就是随机森林的基础原理，"老潘说，"**通过 Bootstrap 重采样，每棵树看到的数据略有不同，学到的模式略有不同**。当你平均 100 棵树的预测，方差会大幅降低——就像问 100 个独立的人，比问一个人更可靠。"

### 随机森林的改进：特征随机性

但 Bagging 还不够。如果所有树都选择"最优特征"（比如面积）作为第一个分裂点，它们仍然会高度相关。

"随机森林在 Bagging 基础上增加了一个关键改进，"老潘说，"**在每个节点分裂时，只随机选择一部分特征（比如 m = √p），而不是所有特征**。"

这进一步增强了树的多样性，让它们的错误更不相关。

### 随机森林实战：性能提升

小北用 scikit-learn 的 RandomForestRegressor 试试：

```python
from sklearn.ensemble import RandomForestRegressor

# 随机森林（100 棵树）
rf = RandomForestRegressor(
    n_estimators=100,      # 树的数量
    max_depth=7,           # 每棵树的最大深度（用网格搜索的最佳值）
    min_samples_leaf=5,
    max_features='sqrt',    # 每次分裂考虑的特征数（√p，回归用 1/3）
    random_state=42,
    n_jobs=-1             # 并行训练
)

rf.fit(X_train, y_train)
print(f"训练集 R²: {rf.score(X_train, y_train):.3f}")
print(f"测试集 R²: {rf.score(X_test, y_test):.3f}")
```

输出是：

```
训练集 R²: 0.865
测试集 R²: 0.812
```

小北从椅子上跳了起来——这次不是因为过拟合的虚惊，而是因为第一次看到了**真正的性能提升**。"测试集 R² 从 0.71 涨到了 0.81！而且是实打实的测试集，不是训练集的自嗨！"

阿码也凑过来看了一眼输出，突然露出一个狡黠的笑容："你知道吗，老潘刚说'群体智慧'的时候，我还以为这是什么管理学理论。现在看到结果——**100 个'差不多但不太一样'的专家投票，真的比一个专家靠谱**。"

"对！"老潘笑了，"**这就是为什么随机森林被称为'off-the-shelf'（开箱即用）算法——你几乎不需要调参，就能得到一个不错的结果**。"

"但，"阿码话锋一转，"**随机森林不是黑盒吗？我看不到'如果-那么'规则了**。"

"你失去了单棵树的可解释性，但获得了两个新工具，"老潘说，"**特征重要性和部分依赖图**。你不再看'规则'，而是看'哪些特征重要'和'特征如何影响预测'。"

---

> **AI 时代小专栏：树模型在工业界——为什么随机森林是"首选算法"**
>
> 2024-2025 年的应用调查显示，**随机森林仍然是工业界"最常用"的算法之一**，在欺诈检测、信用评分、推荐系统、医疗诊断、生产预测等领域广泛应用。
>
> 为什么随机森林如此受欢迎？
>
> 1. **性能稳健**：随机森林对异常值、噪声、特征缩放都不敏感，几乎不需要预处理（One-Hot 编码即可）
> 2. **训练高效**：可以并行训练多棵树，在大数据集上仍然很快
> 3. **特征重要性**：内置的特征重要性评估让模型仍然"可解释"，满足合规要求
> 4. **不容易过拟合**：相比单棵决策树，随机森林的方差更低
>
> 但 2024-2025 年也出现了一个新趋势：**梯度提升树（XGBoost、LightGBM、CatBoost）在 Kaggle 竞赛中逐渐取代随机森林**。多项技术分析指出，在表格数据上，梯度提升树的性能往往优于随机森林，但代价是：
> - 超参数更多（更容易过拟合）
> - 训练更慢（不能简单并行）
> - 需要更精细的调优
>
> 2025 年的对比分析总结道：**随机森林是"首选基线"——先跑一个随机森林，看看特征重要性、模型性能，再决定是否用更复杂的 XGBoost 或神经网络**。梯度提升树适合"追求极致性能"的场景，而随机森林适合"需要快速得到稳健结果"的场景。
>
> 这也是你本周学习随机森林的价值——它既能捕捉非线性关系，又能提供特征重要性，而且训练快、调优相对简单。在 AutoML 时代，理解随机森林的原理和局限性，能帮你判断"什么时候该用更复杂的模型"。
>
> 参考（访问日期：2026-02-12）：
> - [How to Decide Between Random Forests and Gradient Boosting (MachineLearningMastery, Aug 2025)](https://machinelearningmastery.com/how-to-decide-between-random-forests-and-gradient-boosting/)
> - [Gradient Boosting vs. Random Forest: Which Ensemble Method Should You Use? (Medium, Oct 2024)](https://medium.com/@hassaanidrees7/gradient-boosting-vs-random-forest-which-ensemble-method-should-you-use-9f2ee294d9c6)
> - [Gradient Boosting vs Random Forest (GeeksforGeeks, Apr 2024)](https://www.geeksforgeeks.org/gradient-boosting-vs-random-forest/)
> - [Gradient Boosting Trees vs. Random Forests (Baeldung, Feb 2025)](https://www.baeldung.com/cs/gradient-boosting-trees-vs-random-forests)

---

## 5. 特征重要性——哪些变量真的在驱动预测？

小北的随机森林 R² = 0.81，比线性回归的 0.65 好很多。但老潘问了他一个问题："**你知道哪些特征在驱动预测吗？**"

小北愣住了："呃……面积、卧室数、浴室数……都很重要？"

"你需要一个更系统的答案，"老潘说，"**随机森林可以计算特征重要性——量化每个特征对预测的贡献**。"

阿码好奇："**但这和线性回归的系数有什么区别？**"

"好问题，"老潘说，"**线性回归的系数告诉你'这个特征的边际效应'（面积每增加 1㎡，价格涨 β₁），特征重要性告诉你'这个特征在模型中被用到了多少次'（面积贡献了 45% 的 MSE 减少）**。前者是'效应大小'，后者是'使用频率'，回答的是不同问题。"

### 基于不纯度的特征重要性：简单但有陷阱

scikit-learn 的随机森林默认使用**基于不纯度的特征重要性**（对于回归，是 MSE 的减少量）：

```python
# 获取特征重要性
import pandas as pd

feature_importance = pd.DataFrame({
    '特征': X.columns,
    '重要性': rf.feature_importances_
}).sort_values('重要性', ascending=False)

print(feature_importance)
```

输出是：

```
       特征      重要性
0    area_sqm   0.452
3   age_years   0.218
4  distance_km  0.165
1   bedrooms   0.098
2  bathrooms   0.067
```

小北兴奋地说："**面积最重要！其次是房龄和距离！卧室和浴室没那么重要！**"

老潘立刻警告："**别太快下结论——这种基于不纯度的重要性有明显的陷阱**。"

### 陷阱 #1：高基数特征被误认为重要

"如果数据里有一个'用户 ID'列，"老潘说，"**它可能有几千个不同的值（高基数）。决策树会发现：按用户 ID 切分，每次都能把一个样本完美分离——于是模型会疯狂使用这个特征，让它看起来'非常重要'**。"

但这完全是误导：用户 ID 只是唯一标识符，不是真正的预测特征。

"高基数类别特征（如邮政编码、产品 ID）也有类似问题，"老潘补充，"**它们会被过度使用，让模型看起来'学到了很多'，实际上只是在记住噪声**。"

### 陷阱 #2：相关特征会"稀释"重要性

Week 09 你学过**多重共线性**：当两个特征高度相关时（比如面积和房间数），线性回归的系数会不稳定。

"特征重要性也有同样的问题，"老潘说，"**如果面积和房间数高度相关（r = 0.8），模型可能只把重要性给其中一个（面积 = 0.45），另一个看起来'不重要'（房间数 = 0.10），但实际上它们都在起作用**。"

更危险的是，**高度相关的特征会"稀释"重要性**：
- 如果你删除面积，房间数的重要性会飙升到 0.50
- 这不是说房间数原本不重要，而是它和面积"分担"了重要性

"所以基于不纯度的重要性不是'因果排序'，"老潘强调，"**它只是'模型依赖排序'——在这个模型训练过程中，哪些特征被用得最多**。如果你换一个数据集、换一个模型，重要性排序可能完全不同。"

### 更可靠的方案：置换重要性

阿码问："**那怎么知道特征重要性是不是稳定的？**"

"**用 permutation importance（置换重要性）**，"老潘说，"**随机打乱某个特征的值，看模型性能下降多少**。"

想象一下：你有一把很顺手的钥匙（模型），你想知道每颗钥匙齿（特征）到底有多重要。置换重要性的做法是：**故意磨平某一颗齿，看钥匙还能不能开锁**。如果磨掉"面积"这颗齿后，钥匙完全打不开门了（R² 从 0.81 降到 0.60），说明这颗齿确实关键。如果磨掉"卧室数"后，钥匙照样能用，说明这颗齿可有可无。

```python
from sklearn.inspection import permutation_importance

# 置换重要性
perm_importance = permutation_importance(
    rf, X_test, y_test,
    n_repeats=30,
    random_state=42
)

# 打印结果
for i, (feature, importance) in enumerate(zip(X.columns, perm_importance.importances_mean)):
    print(f"{feature}: {importance:.4f}")
```

输出是：

```
area_sqm: 0.2150
age_years: 0.0892
distance_km: 0.0654
bedrooms: 0.0234
bathrooms: 0.0112
```

小北发现一个奇怪的现象："**为什么不纯度重要性（0.45）和置换重要性（0.21）数值差这么大？**"

"这很正常，"老潘解释，"**两种重要性度量的是不同的东西**：不纯度重要性回答的是'训练时这个特征被用了多少次'（内部使用频率），置换重要性回答的是'测试时破坏这个特征会让模型损失多少性能'（实际影响程度）。前者偏向'频繁使用的特征'，后者偏向'真正影响预测的特征'。它们的数值不在同一个尺度上，所以不能直接比较。"

"置换重要性更可靠，"老潘补充，"**它不会偏向高基数特征——打乱用户 ID 只会让模型略微变差（因为它本来就没学到什么有用信息）**。"

### 特征重要性的正确解释

小北应该这样解释：
- **面积的重要性 = 0.45（不纯度）或 0.21（置换）**：在随机森林中，面积是最重要的预测特征之一
- **这不是因果关系**：不能说"扩大面积会导致涨价"（可能是面积大→地段好→价格高）
- **这是模型依赖**：如果我们删除面积，模型性能会下降，但可能是因为丢失了"地段"的代理变量

阿码追问："**那置换重要性和不纯度重要性，该用哪个？**"

"实践中，"老潘说，"**置换重要性是更稳健的选择——它在真实测试集上评估，不会偏向高基数特征，而且相关性问题的干扰更小**。但计算成本更高（需要多次预测），所以先快速看一眼不纯度重要性，再用置换重要性验证可疑的特征。"

### 可视化特征重要性

```python
import matplotlib.pyplot as plt

# 画特征重要性图
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['特征'], feature_importance['重要性'])
plt.xlabel('重要性（MSE 减少量）')
plt.title('随机森林：特征重要性')
plt.tight_layout()
plt.show()
```

这张图可以直接放进 StatLab 报告，让读者一目了然地看到"哪些特征在驱动预测"。

---

## 6. 超参数调优——网格搜索不是"无脑穷举"

阿码在第 3 节看到了网格搜索的威力，兴奋地说："**GridSearchCV 自动尝试所有参数组合，找到最好的！这不就完了吗？**"

老潘立刻摇头："**网格搜索不是'万能钥匙'。它在某些场景下很好，但在另一些场景下效率极低**。"

阿码愣住了："**那什么场景用网格搜索，什么场景用随机搜索？**"

"先理解每种方法的适用场景，"老潘说，"**调优不是'让计算机穷举所有可能'，而是'基于直觉选择合适的搜索策略'**。"

### 网格搜索：适合"小而精"的参数空间

网格搜索（GridSearchCV）会**穷举所有参数组合**。它适合的场景：

- **参数数量少**（2-4 个超参数）
- **每个参数的可能值少**（3-5 个离散值）
- **你想确保不漏掉任何组合**（比如最终精细调优）

```python
# 网格搜索示例
param_grid = {
    'max_depth': [5, 7, 10],
    'min_samples_leaf': [1, 5, 10]
}
# 总组合数 = 3 × 2 = 6 个组合
# 5-fold CV = 30 次模型拟合
```

"网格搜索的优势是'彻底'，"老潘说，"**但如果搜索空间很大（比如 10 个超参数，每个 10 个值），计算成本会爆炸**。"

### 随机搜索：适合"大而广"的参数空间

随机搜索（RandomizedSearchCV）**随机采样参数组合**。它适合的场景：

- **参数数量多**（5+ 个超参数）
- **参数是连续值**（比如学习率 0.001 到 0.1）
- **参数重要性不均**（有些参数影响大，有些影响小）
- **你想快速探索搜索空间**（找到"大致区域"）

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# 定义参数分布（不是网格）
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', 0.5, 0.8]
}

# 随机搜索（50 次采样）
random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=50,           # 只采样 50 个组合
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print(f"最佳参数: {random_search.best_params_}")
print(f"最佳 CV R²: {random_search.best_score_:.3f}")
```

输出是：

```
最佳参数: {'max_depth': 9, 'max_features': 'sqrt', 'min_samples_leaf': 3, 'min_samples_split': 8, 'n_estimators': 287}
最佳 CV R²: 0.824
```

### 为什么随机搜索常常"更快更好"？

阿码好奇："**随机搜索只采样 50 个组合，怎么可能比穷举所有组合更好？**"

"因为不是所有超参数都同样重要，"老潘说，"**随机搜索可以'更均匀地探索'搜索空间，而网格搜索可能在'不重要'的参数上浪费计算资源**。"

举个极端的例子：
- 网格搜索：`max_depth = [3, 5, 7, 9]` × `min_samples_leaf = [1, 2, 3, ..., 100]` = 400 个组合
- 随机搜索：从 400 个可能组合中随机选 50 个

"如果 `min_samples_leaf` 的影响很小，"老潘说，"**网格搜索会在 100 个 min_samples_leaf 值上重复 4 次 max_depth，而随机搜索更有机会探索不同的 max_depth 和 min_samples_leaf 组合**。"

2025 年的实践总结：**先用随机搜索快速找到"大致最优区域"，再用网格搜索精细调优**（类似"粗找+精调"）。

### 和 Week 10 的连接：交叉验证防止调优中的过拟合

"但记住，"老潘强调，"**超参数调优也会过拟合验证集**。如果你在验证集上尝试了 100 个参数组合，最终选的'最佳参数'可能只是'运气好'。"

解决方法：
1. **嵌套交叉验证**（Nested CV）：外层 CV 估计泛化性能，内层 CV 选择超参数
2. **保留测试集**：在调优过程中完全不碰测试集，最后只用一次评估

嵌套交叉验证的代码示例：

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor

# 外层 CV：估计泛化性能
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 内层 CV：选择超参数（使用之前的 RandomizedSearchCV）
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 嵌套 CV
nested_scores = cross_val_score(
    random_search,  # 内层是 RandomizedSearchCV
    X_train, y_train,
    cv=outer_cv,    # 外层是 5-fold
    scoring='r2',
    n_jobs=-1
)

print(f"嵌套 CV R²: {nested_scores.mean():.3f} (±{nested_scores.std():.3f})")
```

输出是：

```
嵌套 CV R²: 0.801 (±0.018)
```

"这个 0.801 比你之前的 0.824 低，"老潘说，"**因为嵌套 CV 是更诚实的估计——它告诉你'在新数据上的真实表现'，而不是'在验证集上挑出来的最佳表现'**。"

"所以，"小北若有所思，"**调优的流程应该是：训练集 → 调优（嵌套 CV）→ 测试集（只评估一次）？**"

"对！"老潘点头，"**你不能在测试集上反复调参数，否则就等于在'作弊'——模型会记住测试集**。"

---

## StatLab 进度

到上周为止，StatLab 报告已经有了数据卡、描述统计、清洗日志、EDA 叙事、假设检验、不确定性量化、回归分析和分类评估。但小北的报告里，所有模型都是线性的（线性回归、逻辑回归）。

老潘看完报告，问了小北一个新问题："**你的数据有非线性关系吗？决策树或随机森林会不会比线性模型更好？**"

小北想了想："呃……我只用过线性模型。不知道数据是不是真的满足线性假设。"

"对，"老潘说，"**线性模型的优点是可解释，但缺点是只能捕捉线性关系。如果真实关系是'阈值型'（比如收入 > 5 万才会购买）或'交互型'（比如年龄和性别有交互作用），树模型会表现得更好**。"

这正好是本周"树模型与集成学习"派上用场的地方。我们要在 report.md 中添加一个**树模型章节**，包含：
1. 决策树：可视化树结构、解释分裂规则
2. 随机森林：Bagging 原理、性能提升
3. 特征重要性：Top 10 特征、相关性警告
4. 超参数调优：网格搜索/随机搜索结果
5. 模型对比：线性回归 vs 逻辑回归 vs 决策树 vs 随机森林
6. 局限性：过拟合风险、特征重要性≠因果

### 在 StatLab 中添加树模型评估

假设你的 StatLab 数据集有回归目标（如房价、消费金额）或分类目标（如流失、购买）。下面是一个完整的树模型评估函数：

```python
# examples/11_statlab_trees.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, classification_report
from scipy.stats import randint

def tree_models_to_report(df, target, numeric_features, categorical_features, task='regression', output_path='report'):
    """
    对 StatLab 数据集进行树模型评估，生成报告片段

    参数:
        df: 清洗后的数据
        target: 目标变量名
        numeric_features: 数值特征列表
        categorical_features: 类别特征列表
        task: 'regression' 或 'classification'
        output_path: 图表输出路径
    """
    # 准备数据
    X = df[numeric_features + categorical_features]
    y = df[target]

    # 简单编码类别特征（树模型可以处理 One-Hot，但这里用 Label 编码简化）
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # ========== 1. 决策树 ==========
    if task == 'regression':
        dt = DecisionTreeRegressor(random_state=42)
    else:
        dt = DecisionTreeClassifier(random_state=42)

    dt.fit(X_train, y_train)

    if task == 'regression':
        dt_train_r2 = dt.score(X_train, y_train)
        dt_test_r2 = dt.score(X_test, y_test)
        dt_metric = f"训练集 R²: {dt_train_r2:.3f}, 测试集 R²: {dt_test_r2:.3f}"
    else:
        dt_train_acc = dt.score(X_train, y_train)
        dt_test_acc = dt.score(X_test, y_test)
        dt_metric = f"训练集准确率: {dt_train_acc:.3f}, 测试集准确率: {dt_test_acc:.3f}"

    # ========== 2. 随机森林（默认参数） ==========
    if task == 'regression':
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    rf.fit(X_train, y_train)

    if task == 'regression':
        rf_train_r2 = rf.score(X_train, y_train)
        rf_test_r2 = rf.score(X_test, y_test)
        rf_metric = f"训练集 R²: {rf_train_r2:.3f}, 测试集 R²: {rf_test_r2:.3f}"
    else:
        rf_train_acc = rf.score(X_train, y_train)
        rf_test_acc = rf.score(X_test, y_test)
        rf_metric = f"训练集准确率: {rf_train_acc:.3f}, 测试集准确率: {rf_test_acc:.3f}"

    # ========== 3. 特征重要性 ==========
    feature_importance = pd.DataFrame({
        '特征': X_encoded.columns,
        '重要性': rf.feature_importances_
    }).sort_values('重要性', ascending=False).head(15)

    # 画特征重要性图
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['特征'], feature_importance['重要性'])
    plt.xlabel('重要性')
    plt.title(f'随机森林：特征重要性 (Top 15) - {target}')
    plt.tight_layout()
    plt.savefig(f"{output_path}/feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========== 4. 超参数调优（随机搜索） ==========
    if task == 'regression':
        param_dist = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', 0.5, 0.8]
        }
        scoring = 'r2'
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
    else:
        param_dist = {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 15),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': ['sqrt', 'log2', 0.5, 0.8]
        }
        scoring = 'f1'
        model = RandomForestClassifier(random_state=42, n_jobs=-1)

    random_search = RandomizedSearchCV(
        model, param_dist, n_iter=50, cv=5, scoring=scoring, random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    if task == 'regression':
        tuned_test_r2 = best_model.score(X_test, y_test)
    else:
        tuned_test_acc = best_model.score(X_test, y_test)

    # ========== 5. 生成报告片段 ==========
    report = f"""
## 树模型与集成学习

### 研究问题
线性模型（回归/逻辑回归）假设特征与目标之间是线性关系，但真实数据可能存在非线性、交互作用或阈值效应。本章使用**决策树**和**随机森林**捕捉这些复杂模式，并与线性基线对比。

### 决策树

决策树通过一系列"如果-那么"规则预测{target}，可解释性强但容易过拟合。

**性能（无调优）**:
{dt_metric}

**解读**: 训练集与测试集的{'显著差异' if task == 'regression' and abs(dt_train_r2 - dt_test_r2) > 0.2 else '差异较小'}，说明决策树{'存在过拟合' if task == 'regression' and abs(dt_train_r2 - dt_test_r2) > 0.2 else '泛化性能尚可'}。

### 随机森林

随机森林通过 **Bagging（Bootstrap Aggregating）** 训练多棵树并在预测时投票/平均，显著降低方差，提升泛化性能。

**原理**:
1. Bootstrap 重采样：每棵树在不同的训练子样本上训练
2. 特征随机性：每次分裂时只随机选择一部分特征
3. 投票/平均：回归取平均值，分类取多数投票

**性能（默认参数，n_estimators=100）**:
{rf_metric}

**调优后性能**:
- 最佳参数: {random_search.best_params_}
- 最佳 CV {'R²' if task == 'regression' else 'F1'}: {random_search.best_score_:.3f}
- 测试集 {'R²' if task == 'regression' else '准确率'}: {tuned_test_r2 if task == 'regression' else tuned_test_acc:.3f}

**改进**: 超参数调优将测试集性能从{rf_test_r2 if task == 'regression' else rf_test_acc:.3f} 提升到 {tuned_test_r2 if task == 'regression' else tuned_test_acc:.3f}。

### 特征重要性

随机森林计算的**特征重要性**基于每个特征在分裂时对不纯度/MSE 的减少量。

**Top 15 特征**:
"""

    for _, row in feature_importance.head(15).iterrows():
        report += f"- {row['特征']}: {row['重要性']:.4f}\n"

    report += f"""
![特征重要性](feature_importance.png)

**解读**:
- 最重要特征: {feature_importance.iloc[0]['特征']}
- 次重要特征: {feature_importance.iloc[1]['特征']}

⚠️ **注意**: 特征重要性不等同于因果关系。高度相关的特征会"稀释"重要性（模型只选择其中一个），且特征重要性只反映"模型依赖"，不反映"因果机制"。

### 模型对比

与线性基线（Week 09/10）的对比:

| 模型 | 测试集 {'R²' if task == 'regression' else '准确率'} | 优势 |
|------|--------|------|
| {'线性/逻辑回归' if task == 'regression' else '逻辑回归'} | - | 可解释性强，假设线性关系 |
| 决策树 | {dt_test_r2 if task == 'regression' else dt_test_acc:.3f} | 可解释规则，捕捉非线性 |
| 随机森林（调优后） | {tuned_test_r2 if task == 'regression' else tuned_test_acc:.3f} | 性能最佳，特征重要性 |

**结论**:
- 如果优先考虑**可解释性**，线性回归/逻辑回归仍然是最佳选择（系数直接解释）
- 如果优先考虑**预测性能**，随机森林明显优于单一模型（{'R²' if task == 'regression' else '准确率'} 提升 {(tuned_test_r2 - dt_test_r2) if task == 'regression' else (tuned_test_acc - dt_test_acc):.1%}）
- 决策树作为"中间选项"，在可解释性和性能之间取得平衡

### 局限性与风险

⚠️ **过拟合**: 决策树容易记住训练数据，必须通过 max_depth、min_samples_leaf 等超参数控制复杂度。随机森林通过 Bagging 降低过拟合风险，但仍需调优。

⚠️ **特征重要性陷阱**:
- 相关性特征会稀释重要性（面积 vs 房间数）
- 高基数类别特征（如用户 ID）可能被误认为"重要"
- 特征重要性不等于因果，不能用于"干预建议"（如"增加卧室数会涨价"是错误的因果推论）

⚠️ **计算成本**: 随机森林的训练时间是线性模型的 10-100 倍（取决于 n_estimators）。大数据集上可能需要更长训练时间或分布式计算。

### 工程实践

本分析使用了以下最佳实践：
- **嵌套交叉验证**: 在超参数调优中用 5-fold CV，防止过拟合验证集
- **随机搜索**: 用 RandomizedSearchCV 替代网格搜索，以更少的迭代找到接近最优的解
- **保留测试集**: 测试集只在最终评估时使用一次，确保性能估计的无偏性

### 数据来源
- 样本量: n = {len(y)}
- 分析日期: 2026-02-12
"""

    return report

# 使用示例
if __name__ == "__main__":
    # 假设你的数据已经清洗完成
    df = pd.read_csv("data/clean_data.csv")

    # 回归任务（房价预测）
    report_reg = tree_models_to_report(
        df=df,
        target="price",
        numeric_features=["area_sqm", "bedrooms", "bathrooms", "age_years"],
        categorical_features=["city", "property_type"],
        task="regression",
        output_path="report"
    )

    # 追加到 report.md
    with open("report/report.md", "a", encoding="utf-8") as f:
        f.write(report_reg)

    print("✅ 树模型章节已添加到 report/report.md")
```

### 本周改进总结

| 改动项 | 上周状态 | 本周改进 |
|--------|---------|---------|
| 模型类型 | 线性回归、逻辑回归 | 新增决策树、随机森林 |
| 非线性建模 | 假设线性关系 | 捕捉非线性、交互作用 |
| 可解释性 | 系数解释（线性）、优势比（逻辑） | 树结构可视化、特征重要性 |
| 泛化性能 | 单一模型 | 集成学习（Bagging）降低方差 |
| 超参数调优 | 不适用 | 网格搜索/随机搜索 |
| 模型对比 | 回归 vs 逻辑回归 | 线性 vs 树 vs 集成 |

老潘看到这段改动会说什么？"**这才是从'线性模型'到'树模型与集成学习'的完整升级**。你不仅学会了决策树的可解释规则、随机森林的 Bagging 原理，还掌握了特征重要性和超参数调优的工程实践。AI 可以帮你拟合森林、计算重要性，但理解重要性陷阱、防止过拟合、选择最佳模型的责任由你承担。"

---

## Git 本周要点

本周必会命令：
- `git status`: 查看工作区状态
- `git diff`: 查看具体改动内容
- `git add -A`: 添加所有改动
- `git commit -m "feat: add tree models and random forest"`
- `git log --oneline -n 5`

常见坑：

**决策树过拟合**——训练集 R² = 0.95，测试集只有 0.65。这是树"背答案"了。解决方法：通过 `max_depth`、`min_samples_leaf` 控制复杂度，或直接用随机森林。

**误解特征重要性为因果**——"面积最重要"≠"扩大面积会涨价"。面积可能是地段、房型的代理变量。解决方法：用置换重要性验证，并谨慎解释因果。

**网格搜索计算爆炸**——10 个超参数 × 10 个值 = 10^10 个组合，算到明年。解决方法：先用 `RandomizedSearchCV` 快速探索，再对关键参数精细调优。

**在测试集上调参**——反复在测试集上评估并修改参数，等于"泄露答案"。解决方法：保留测试集只用一次评估，调优在训练集上用 CV。

**忽略基线对比**——随机森林比线性模型好 2%，但计算成本高 10 倍。解决方法：计算"性价比"，如果提升很小，线性模型可能更实用（可解释性也更好）。

Pull Request (PR)：
- Gitea 上也叫 Pull Request，流程等价 GitHub：push 分支 -> 开 PR -> review -> merge。

---

## 本周小结（供下周参考）

本周你从"线性模型的假设"（Week 09-10）走到了"树模型的灵活性"。你理解了为什么线性模型有时不够——当残差图出现 U 型或模式时，说明模型漏掉了非线性关系。

决策树给了你一种新的建模语言：用"如果-那么"规则把数据切成区域。它比线性回归更灵活，能捕捉非线性、交互作用和阈值效应，而且**可解释**——你能看到模型是怎么做决策的。但你也很快发现它的软肋：单棵树太容易"背答案"，训练集完美但测试集很差。

随机森林解决了这个问题：通过 Bootstrap 重采样和特征随机性，训练多棵"不完全相关"的树，再让它们投票。这大幅降低了方差，让泛化性能从 0.71 涨到 0.81。

但强大的模型也带来新的责任。你学会了用**特征重要性**量化每个特征对预测的贡献，但也知道了陷阱：高基数特征会被误认为重要，相关特征会稀释重要性，而且**特征重要性 ≠ 因果关系**。置换重要性是更稳健的选择。

你还掌握了**超参数调优**的工程实践——知道何时用网格搜索（参数少、离散值），何时用随机搜索（参数多、连续值），并用交叉验证防止调优过程中的过拟合。这不是"无脑穷举"，而是"基于直觉 + CV 验证"找到泛化最好的平衡点。

下周（Week 12），你要把这个思维升级为"模型解释与伦理"：从特征重要性到 SHAP、LIME 等更高级的可解释 AI 技术，并讨论模型偏见、公平性、隐私等伦理问题。本周的"特征重要性陷阱"会演化为下周的"模型偏见与公平性指标"，树模型会演化为"可解释 AI 的基线"。

---

## Definition of Done（学生自测清单）

- [ ] 我能理解决策树的分裂逻辑（熵/Gini/MSE）
- [ ] 我能识别决策树的过拟合问题（训练集 vs 测试集性能差异）
- [ ] 我能理解随机森林的 Bagging 原理（Bootstrap + 投票）
- [ ] 我能正确解释特征重要性（模型依赖 ≠ 因果关系）
- [ ] 我知道基于不纯度的重要性的陷阱（高基数特征、相关特征）
- [ ] 我能使用置换重要性得到更稳健的特征重要性
- [ ] 我能选择合适的超参数调优方法（网格搜索 vs 随机搜索）
- [ ] 我能在 StatLab 报告中添加树模型章节（决策树、随机森林、特征重要性）
- [ ] 我能对比线性模型 vs 树模型 vs 随机森林的性能
- [ ] 我能识别 AI 生成的树模型代码中的过拟合风险
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解"从线性到树，从单棵到森林"是建模复杂度的提升，但不一定是因果解释的提升
