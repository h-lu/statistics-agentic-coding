# Week 11：树模型与基线对比——更复杂不等于更好

> "All models are wrong, but some are useful."
> — George Box

小北上周学会了一件事：如何用逻辑回归预测客户流失。模型 AUC 0.87，他很高兴。

然后他看到了一篇 2025-2026 年 Kaggle 竞赛获胜方案，用的是随机森林 + XGBoost + 深度学习的三层集成，AUC 0.91。

"我也要用这个！"小北眼睛一亮。

阿码在旁边泼了冷水："你确定 0.91 比 0.87 好 4% 值得多花三个月调参吗？而且你能向业务方解释你的集成模型是怎么做决策的吗？"

记住 Week 10 我们学的**分类 vs 回归**的根本区别：回归预测连续值（如"销售额是多少"），分类预测离散类别（如"是否会流失"）。决策树和随机森林既能做分类，也能做回归，但它们的思维方式是"用 if-else 规则把数据分箱"，而不是像回归那样找"最优拟合线"。

老潘听完笑了："我见过一个团队花了半年优化一个模型，最后发现它比'总是预测不流失'好了不到 2%。问题是他们一开始就没和基线比。"

这正是本周要解决的核心问题：**树模型与基线对比**。在 AI 时代，AutoML 工具可以让你在几分钟内尝试几十种模型，但工业界的最佳实践——从 Google 的 ML 开发指南到 Kaggle 获胜者的经验——都强调同一个原则：**从简单模型开始，与基线对比**。

你会学习决策树——它像一系列 if-else 规则，直观但容易过拟合；然后学习随机森林——用"群体智慧"降低过拟合；更重要的是，你会学会**任何"更复杂"的模型都必须与基线对比**，以及如何判断"这种提升是否值得牺牲可解释性"。

AI 可以帮你训练 20 个模型，但只有你能回答"选哪个"和"为什么"。

---

## 前情提要

上周你完成了从回归到分类的跨越：从"预测销售额是多少"到"预测是否会流失"，从"MSE 和 R²"到"准确率陷阱和 ROC-AUC"。你还学会了用 Pipeline 防止数据泄漏——训练时泄露一点信息，测试时就会付出代价。

老潘当时说了一句话："分类评估不只是算几个数字。更重要的是：指标选对了吗？评估诚实了吗？模型可以解释吗？"

小北当时没太在意。现在他开始理解了：**更复杂的模型不等于更好的模型**。

这周，你要做的不是"追求最高 AUC"，而是学会"权衡提升量、复杂度与可解释性"。

---

## 学习目标

完成本周学习后，你将能够：

1. 理解决策树的工作原理，解读树结构（根节点、分裂规则、叶节点）
2. 理解决策树如何"递归地"分裂数据，以及过拟合的风险
3. 理解随机森林的"群体智慧"原理（Bagging、特征随机性）
4. 学会与基线对比：傻瓜基线、逻辑回归基线、单特征树基线
5. 在 StatLab 报告中写出"带基线对比和模型选择理由"的建模结论

---

<!--
贯穿案例：从"单棵决策树"到"随机森林 vs 基线对比"

案例演进路线：
- 第 1 节（从线性到树形）→ 从"逻辑回归的线性决策边界"到"决策树的阶梯式决策边界"
- 第 2 节（决策树原理）→ 从"黑盒模型"到"可视化树结构：根节点、分裂规则、叶节点"
- 第 3 节（树的优势与陷阱）→ 从"决策树的直观性"到"过拟合与不稳定：小变化导致大不同"
- 第 4 节（随机森林）→ 从"单棵树的脆弱"到"随机森林：群体智慧降低方差"
- 第 5 节（基线对比）→ 从"只看 AUC"到"与基线对比：复杂度 vs 提升量的权衡"

最终成果：读者能训练决策树和随机森林、理解树结构的可解释性、用随机森林降低过拟合、与基线对比评估模型选择是否合理

数据集建议：
- 复用电商流失预测数据
- 特征：购买次数、平均消费金额、注册时长、最近一次购买距今天数、会员等级等
- 保持与 Week 10 一致的数据集，便于直接对比模型

---

认知负荷预算：
- 本周新概念（5 个，预算上限 5 个）：
  1. 决策树（Decision Tree）：递归分裂、树结构可视化
  2. 信息增益/基尼不纯度（Information Gain/Gini Impurity）：树如何选择分裂特征
  3. 过拟合与剪枝（Overfitting and Pruning）：树的深度控制
  4. 随机森林（Random Forest）：Bagging + 特征随机性
  5. 基线对比（Baseline Comparison）：傻瓜基线、逻辑回归基线
- 结论：✅ 在预算内

回顾桥设计（至少 2 个，来自 week_05-10）：
- [过拟合]（来自 week_09）：在第 3 节，通过"决策树容易过拟合，和回归一样需要诊断"再次使用
- [ROC-AUC]（来自 week_10）：在第 4-5 节，通过"对比不同模型的 AUC，但要看提升量是否显著"再次使用
- [Pipeline]（来自 week_10）：在第 4-5 节，通过"随机森林也用 Pipeline 防止数据泄漏"再次使用
- [Bootstrap]（来自 week_08）：在第 4 节，通过"Bagging 本质上是 Bootstrap"再次连接
- [混淆矩阵与评估指标]（来自 week_10）：在第 5 节，通过"对比不同模型的混淆矩阵和指标"再次使用

AI 小专栏规划：
- 第 1 个侧栏（第 1-2 节之后）：
  - 主题："AI 时代的模型选择——为什么基线对比更重要了"
  - 连接点：刚学完决策树的原理，讨论 AI 工具如何让"尝试复杂模型"变得容易，但也更容易忽略基线对比
  - 建议搜索词："baseline comparison machine learning 2026", "model selection best practices", "when to use simple models vs complex models"

- 第 2 个侧栏（第 3-4 节之后）：
  - 主题："AutoML 与自动化基线对比"
  - 连接点：刚学完随机森林的原理，讨论 AutoML 工具（如 AutoGluon、H2O.ai）如何自动化模型选择和基线对比
  - 建议搜索词："AutoML baseline comparison 2026", "AutoGluon vs manual model selection", "automated machine learning best practices"

角色出场规划：
- 小北（第 2 节）：看到决策树可视化后惊叹"这个树好直观！"，但随后发现树太深、过拟合了
- 阿码（第 3 节）：追问"决策树为什么不选这个特征分裂？"，引出信息增益/基尼不纯度的概念
- 老潘（第 5 节）：看到"没有基线对比的模型报告"后点评"这不是分析，是炫技"

StatLab 本周推进：
- 上周状态：数据卡 + 描述统计 + 可视化 + 清洗日志 + 相关分析 + 分组比较 + 假设清单 + 多组比较 + 区间估计 + Bootstrap + 置换检验 + 回归分析 + 模型诊断 + 分类评估（逻辑回归、混淆矩阵、ROC-AUC、Pipeline 防泄漏）
- 本周改进：添加树模型模块，包括决策树可视化、随机森林训练、基线对比（傻瓜基线、逻辑回归基线、单特征树基线）、模型选择理由
- 涉及的本周概念：决策树、过拟合与剪枝、随机森林、基线对比
- 建议示例文件：examples/11_statlab_tree_models.py（本周报告生成入口脚本）
-->

## 1. 从"直线"到"阶梯"——决策边界形状的变化

小北上周用逻辑回归预测流失，AUC 0.87。成绩不错，但他总觉得哪里不对劲。

"我看了流失客户的名单，"小北说，"有些客户购买次数很少，但平均消费很高，也流失了。逻辑回归能捕捉到这种模式吗？"

阿码接过话题："逻辑回归的决策边界是一条直线，对吧？它在特征空间里画一条线，一边是'会流失'，另一边是'不流失'。但如果真正的模式是'购买次数 < 3 且平均消费 < 100' 的客户才容易流失，一条直线能抓住这个吗？"

老潘点头。"这就是逻辑回归的局限：**它假设特征与对数几率之间是线性关系**。如果真实关系更像'AND 逻辑'——两个条件都要满足——线性模型会很吃力。"

"那怎么办？"小北问。

"用**决策树**。"老潘说，"它不需要假设线性关系，决策边界是'阶梯式'的，可以逼近任何复杂的形状。"

### 线性 vs 树形决策边界

想象你在用两个特征预测流失：`purchase_count`（购买次数）和 `avg_spend`（平均消费）。

**逻辑回归**会找到一条直线，把平面分成两部分。决策边界可能是这样的：

```text
流失概率 > 0.5 的区域：
  当 -2.5 + 0.3 × purchase_count + 0.01 × avg_spend > 0
```

这是一条斜线，把平面切成两半。

**决策树**会递归地问问题：

```
1. 如果 purchase_count < 3：
     如果 avg_spend < 100：预测流失（高概率）
     否则：预测不流失
2. 否则（purchase_count >= 3）：
     预测不流失
```

决策边界是"阶梯式"的：先看 purchase_count 是否小于 3，如果是再看 avg_spend 是否小于 100。这更像你在现实中做决策的方式——**一步一步来，每一步问一个具体问题**。

这正好呼应了 Week 09 学的**简单线性回归**：线性回归假设 y 和 x 之间是线性关系。如果真实关系是非线性的（比如 y = x²），你可以加多项式特征，但决策树提供了一种更灵活的方式——不需要显式转换特征，树可以自动捕捉非线性关系。

阿码问："决策树这么灵活，为什么上周还要学逻辑回归？"

"灵活是双刃剑。"老潘说，"越灵活，越容易过拟合。而且决策树有个特点：**它的决策边界总是与坐标轴平行**。如果真正的边界是一条斜线，决策树需要很多次分裂才能逼近，而逻辑回归一条线就能搞定。"

"还有个区别。"老潘继续说，"逻辑回归给出的是'光滑的概率估计'，决策树给出的是'某个叶节点的历史比例'。前者更稳定，后者更直观。"

小北若有所思："所以决策树像是一系列 if-else 规则？"

"对。"老潘说，"决策树的优势就是**可解释性**——你可以把树画出来，看每个节点在用什么特征、用什么阈值分裂。业务方可能听不懂'逻辑回归系数 0.3 意味着什么'，但能听懂'购买次数 < 3 且平均消费 < 100 的客户流失概率 72%'。"

现在的问题是：决策树是怎么"长"出来的？它怎么知道先问 purchase_count，而不是先问 avg_spend？

> **AI 时代小专栏：AI 时代的模型选择——为什么基线对比更重要了**

> 2026 年，很多 AI 工具（如 AutoML、超参数优化服务）可以让你在几分钟内尝试几十种模型：逻辑回归、决策树、随机森林、XGBoost、LightGBM、神经网络……然后告诉你"这个模型的 AUC 最高"。
>
> **但这有一个陷阱**：如果这些工具没有与基线对比，你可能得到"虚假的提升"。
>
> **什么是基线对比？**
>
> 基线（Baseline）是一个简单的参考模型，用来评估更复杂模型的"净提升量"。常见基线包括：
> - **傻瓜基线（Dummy Classifier）**：总是预测多数类，或随机预测
> - **逻辑回归基线**：最简单的线性分类器
> - **单特征树基线**：只用最重要的特征做一棵浅树
>
> **为什么基线对比在 AI 时代更重要？**
>
> - **模型数量爆炸**：过去你可能只试 2-3 个模型，现在 AI 工具让你试 20+ 个模型。如果没有基线，你可能"选中"了运气好的那个，而不是真正更好的那个
> - **过拟合风险增加**：更复杂的模型（如深度树、神经网络）更容易过拟合。如果没有基线对比，你可能把"过拟合的提升"当成"真正的改进"
> - **复杂度 vs 提升量**：工业界不只看"模型有多准"，还看"部署成本、维护成本、解释难度"。如果复杂模型只比简单模型好了 0.5% AUC，但成本高 10 倍，你真的会选它吗？
>
> **行业最佳实践**：
>
> - Kaggle 竞赛中，获胜者的第一件事是建立一个"强基线"（如简单的逻辑回归或随机森林）
> - Google 的 ML 开发指南强调："从简单模型开始，再逐步增加复杂度"
> - AutoML 工具（如 AutoGluon、H2O.ai）会自动包含基线模型，并报告相对提升量
>
> **对你的启示**：
>
> 你本周学的决策树和随机森林只是第一步。更重要的是：**永远先训练一个基线模型，再训练复杂模型，然后问"提升量有多大？这个提升量值得牺牲可解释性吗？"**
>
> 参考（访问日期：2026-02-18）：
> - [Scikit-learn: DummyClassifier](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
> - [Google ML Guidelines: Start with Simple Models](https://developers.google.com/machine-learning/guides/rules-of-ml#simple_is_better)
> - [AutoGluon Documentation: AutoML with Automatic Baseline Comparison](https://auto.gluon.ai/stable/index.html)

---

## 2. 决策树是如何做决策的？——从根节点到叶节点

决策树的训练过程和逻辑回归很不一样。逻辑回归用梯度下降找"最优系数"，决策树用"贪婪分裂"——每个节点只选"当前最优"的分裂，一步步往前走。

老潘打开了一张决策树的可视化图。"看，这就是一棵训练好的树。"

### 决策树的结构

```
                        purchase_count < 3.5？
                       /                    \
                     是/                      \否
                    /                          \
        avg_spend < 87.3？              churn_prob: 0.08
        /              \
      是/                \否
      /                    \
  churn_prob: 0.72    days_since_last < 45.2？
                      /                  \
                    是/                    \否
                    /                        \
              churn_prob: 0.41          churn_prob: 0.89
```

小北第一次看到这个图时，反应和你可能一样："这看起来像流程图！"

"对，这就是决策树的优势。"老潘说，"**树结构直观**。你可以从根节点一路走到叶节点，看清楚每一步的决策规则。"

**树结构的术语**：

| 术语 | 含义 | 示例 |
|------|------|------|
| **根节点（Root Node）** | 树的起点，包含所有样本 | `purchase_count < 3.5?` |
| **内部节点（Internal Node）** | 中间的分裂节点 | `avg_spend < 87.3?` |
| **叶节点（Leaf Node）** | 终端节点，输出预测值 | `churn_prob: 0.72` |
| **分裂规则（Split Rule）** | 如何分裂样本 | `特征 < 阈值` |
| **深度（Depth）** | 从根到叶的层数 | 上图最大深度为 3 |

小北指着图说："这个叶节点'churn_prob: 0.72' 意味着掉进这个叶节点的客户，72% 都流失了？"

"对。"老潘说，"这是训练数据中的历史比例。决策树不会像逻辑回归那样给你'精确的概率估计'，它会告诉你'掉进这个篮子的样本，72% 是正类'。"

"这其实很好理解。"老潘继续说，"你可以把叶节点想象成'客户细分'——每个叶节点代表一类相似客户，流失率就是这类客户的历史比例。"

### 树如何选择分裂特征？

决策树在每个节点选择"最好的"分裂特征和阈值，选择标准是让分裂后的数据"更纯净"（同类的样本更聚集）。

**两个常见标准**：

1. **基尼不纯度（Gini Impurity）**（CART 算法，scikit-learn 默认）

$$
Gini = 1 - \sum_{i=1}^{k} p_i^2
$$

其中 $p_i$ 是第 $i$ 类的比例。基尼不纯度越小，数据越纯净。

直觉：如果一个节点里只有一类样本（如全是流失客户），基尼 = 0；如果两类样本各占一半，基尼 = 0.5。

**示例**：假设一个节点有 100 个样本，其中 60 个正类、40 个负类。
- 基尼 = 1 - (0.6)² - (0.4)² = 1 - 0.36 - 0.16 = 0.48
- 如果全是正类（100 个正、0 个负），基尼 = 1 - 1² - 0² = 0（完全纯净）
- 如果各占一半（50 个正、50 个负），基尼 = 1 - 0.5² - 0.5² = 0.5（最不纯）

2. **信息增益（Information Gain）**（ID3/C4.5 算法，基于熵）

$$
Entropy = -\sum_{i=1}^{k} p_i \log_2(p_i)
$$

信息增益 = 父节点熵 - 分裂后子节点加权熵。

阿码问："这两个标准有什么区别？"

"在实践中，它们通常给出相似的结果。"老潘说，"基尼计算更快（不需要 log），所以 scikit-learn 默认用它。你可以把它理解为'衡量纯度'——纯度越高，同类样本越聚集。"

"但这里有个关键点：决策树在每个节点只选'当前最优'的分裂，不管后面会怎样。这叫**贪婪算法**。"老潘继续说，"它可能找到一个'局部最优'，但不是'全局最优'。不过在实践中，贪婪决策树的效果通常已经足够好了。"

### 决策树实战

让我们训练一个决策树预测客户流失：

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 训练决策树（限制深度防止过拟合）
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# 预测
y_pred = tree.predict(X_test)
y_prob = tree.predict_proba(X_test)[:, 1]

# 评估
from sklearn.metrics import accuracy_score, roc_auc_score
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f"准确率: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")

# 可视化树
plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=X_train.columns, class_names=['Not Churn', 'Churn'],
          filled=True, rounded=True, fontsize=10)
plt.show()
```

小北盯着代码看了半天，问了一个好问题："`max_depth=3` 是什么意思？为什么不让树自由生长？"

"如果你不限制深度，决策树会一直分裂，直到每个叶节点纯净（只有一类样本）。"老潘说，"听起来很好？但问题是：**它会记住训练数据中的每个样本**。"

"这叫**过拟合**（Overfitting）。"老潘继续说，"训练集准确率 100%，但测试集准确率可能只有 70%。树太'聪明'了，把噪声也学进去了。"

这正好呼应了 Week 09 学的**简单线性回归**和**过拟合**：回归模型如果特征太多、太复杂，也会过拟合。决策树更容易过拟合，因为它可以无限生长，直到"记住"每个样本。这就像是 Week 09 中讨论的**R² 与拟合优度**问题——当你加入过多特征时，R² 会虚高，但模型的泛化能力反而下降。

小北问："那怎么控制？"

"有两种办法：**提前叫停**（预剪枝）或**事后修剪**（后剪枝）。"

---

## 3. 决策树的陷阱——过拟合与不稳定

小北上周犯了个典型错误：他训练决策树时，没有设置任何限制参数，让树"自由生长"。

结果：训练集准确率 100%，测试集准确率只有 68%。

老潘看完代码笑了："你得到一棵完美的记忆树，但不是一棵有用的预测树。"

"为什么会这样？"小北很困惑。

"想象一下。"老潘说，"如果你要记住 100 个人的名字，你可以编 100 条规则——'如果是张三，身高 175.3cm，体重 68.7kg，就说这是张三'。这样你在训练集上能 100% 答对，但遇到一个新样本——身高 175.4cm，体重 68.8kg——你就傻眼了。"

"**决策树会做同样的事**。如果你不限制它，它会一直分裂，直到每个叶节点纯净。每个叶节点可能只有 1-2 个样本，树完美记住了训练数据，但对新数据毫无泛化能力。"

阿码问："这和 Week 09 学的回归过拟合是一回事吗？"

"原理一样，形式不同。"老潘说，"回归过拟合是'曲线太扭，把每个点都连上了'。决策树过拟合是'树太深，把每个样本都装进独立的叶节点了'。两者都是在'死记硬背'而不是'学习规律'。"

### 判断是否过拟合

阿码问："怎么判断我的树有没有过拟合？"

"看训练集和测试集的表现差距。"老潘说，"如果训练集 AUC 0.99，测试集 AUC 0.75，差距很大，说明过拟合了。"

**过拟合的典型表现**：
- 训练集准确率接近 100%，测试集准确率远低于训练集
- 树很深（如 `max_depth=10` 甚至更深），叶节点很多
- 树的规则过于具体（如"购买次数 = 3.7 且平均消费 = 87.3"）——这些精确的数字通常是"记忆噪声"的标志

老潘打了个比方："想象你在背单词。过拟合的决策树就像那种'死记硬背'——它把每个单词的位置都记住了，但你换个顺序，它就傻了。好的模型是'理解了规律'，不管题目怎么变，都能应对。"

"另一个类比：**过拟合的树像是一个过度敏感的保安**。"老潘继续说，"他记住每个员工的准确特征——身高 175.3cm、穿蓝色衬衫、姓张、早上 8:02 到——这些细节太精确了。第二天有个人身高 175.4cm、穿浅蓝色衬衫、早上 8:03 到，保安就认不出来了。"

"好的模型应该记住'规律'而不是'噪声'。规律是'穿工服、有工牌的人可以进'，噪声是'身高 175.3cm'这种无关细节。"

小北问："怎么判断什么是规律，什么是噪声？"

"这就是机器学习最难的问题。"老潘说，"但有个经验法则：**如果一个规则太精确（如阈值精确到小数点后一位），通常是在记噪声**。如果规则简单粗暴（如'购买次数 < 3'），更可能是规律。"

"那怎么防止？"小北继续问。

### 预剪枝：提前停止生长

**预剪枝**是在训练过程中提前停止树的生长。常见参数：

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `max_depth` | 树的最大深度 | 3-5（小数据集），5-10（大数据集） |
| `min_samples_split` | 节点分裂所需的最小样本数 | 10-50 |
| `min_samples_leaf` | 叶节点的最小样本数 | 5-20 |
| `max_leaf_nodes` | 最大叶节点数 | 10-30 |

```python
# 预剪枝示例
tree_pruned = DecisionTreeClassifier(
    max_depth=5,              # 限制深度
    min_samples_split=20,     # 每个节点至少 20 个样本才分裂
    min_samples_leaf=10,      # 每个叶节点至少 10 个样本
    random_state=42
)
tree_pruned.fit(X_train, y_train)

# 对比
train_auc = roc_auc_score(y_train, tree_pruned.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(y_test, tree_pruned.predict_proba(X_test)[:, 1])
print(f"训练集 AUC: {train_auc:.4f}")
print(f"测试集 AUC: {test_auc:.4f}")
```

### 后剪枝：生长后再修剪

**后剪枝**是让树先充分生长，然后剪掉不重要的分支。scikit-learn 使用的是**成本复杂度剪枝（CCP）**：

```python
# 计算剪枝路径
path = tree.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas  # 不同的剪枝强度

# 选择最佳 alpha（通过交叉验证）
from sklearn.model_selection import cross_val_score

scores = []
for alpha in alphas:
    tree_pruned = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    score = cross_val_score(tree_pruned, X_train, y_train, cv=5, scoring='roc_auc').mean()
    scores.append(score)

# 选择 alpha
best_alpha = alphas[scores.index(max(scores))]
tree_final = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
tree_final.fit(X_train, y_train)
```

阿码问："预剪枝和后剪枝，哪个更好？"

"实践中预剪枝更常用。"老潘说，"计算更快，效果通常也不错。后剪枝理论上更优，但需要更多计算资源。"

"但决策树还有另一个问题，比过拟合更隐蔽。"老潘继续说，"**它不稳定**。"

### 决策树的另一个问题：不稳定

老潘打开了一个新的 Jupyter Notebook。"看，这是同一个数据集，我只改了 5 个样本的值。"

他运行了两次决策树训练，两个结果的可视化完全不同。

"这就是**不稳定（High Variance）**。"老潘说，"决策树对数据的小变化非常敏感。因为树的分裂是贪婪的——每个节点只选'当前最优'的分裂，根部的一个小变化会层层传导，导致整棵树结构完全不同。"

"这很危险。"老潘继续说，"你在训练集上得到一棵树，AUC 0.85。但如果你重新采样再训练一次，可能得到 AUC 0.78 或 0.82 的树。你不稳定，业务方就不信任。"

小北问："为什么会这样？"

"想象你在玩'二十个问题'猜谜游戏。"老潘说，"第一个问题是关键。如果第一个问题问'是动物吗？'，整个游戏会往一个方向走。如果问'是活的吗？'，游戏会往另一个方向走。"

"决策树的根节点就是第一个问题。如果数据略有变化，根节点可能选不同的分裂特征——比如从'购买次数 < 3'变成'平均消费 < 100'——后面的所有分裂都会跟着变，整棵树就完全不同了。"

"这就像**蝴蝶效应**：根部的小变化，导致叶子处的大不同。"

阿码问："怎么解决这个问题？"

"用**随机森林（Random Forest）**。"老潘说，"训练很多棵决策树，每棵树看到的数据略有不同，然后让它们'投票'。单棵树不稳定，但群体智慧稳定。"

"这就像民主选举——一个人可能判断失误，但一千个人的投票通常更可靠。"

---

## 4. 随机森林——用"群体智慧"降低方差

小北现在理解了决策树的两个致命问题：**容易过拟合**和**不稳定**。

"有没有办法同时解决这两个问题？"小北问。

"有。"老潘说，"**随机森林**。"

"原理是什么？"

"**群体智慧**。"老潘说，"一个人的判断可能出错，但一百个人的投票通常比一个人可靠。随机森林就是训练很多棵决策树，每棵树看到的数据略有不同，然后让它们投票。"

### Bagging：从 Bootstrap 到模型聚合

"这和 Week 08 学的 Bootstrap 有关吗？"阿码突然问。

"好问题！"老潘说，"Bagging 本质上就是 Bootstrap 的应用——只是 Bootstrap 原本用于估计统计量的分布，Bagging 用于降低模型的方差。"

**Bagging（Bootstrap Aggregating）**是 Leo Breiman 在 1996 年提出的方法，核心思想：

1. **Bootstrap**：从原始数据中有放回地抽取多个样本集（每个样本集大小和原始数据相同）
2. **训练**：在每个样本集上训练一个模型（如决策树）
3. **聚合（Aggregating）**：对于分类，用多数投票；对于回归，用平均

"等等，"小北问，"有放回抽样？那同一个样本会不会在一个样本集里出现多次？"

"会。"老潘说，"Bootstrap 样本中，大约 63.2% 的原始样本会至少出现一次，36.8% 的样本不会出现（这些叫 OOB，Out-of-Bag 样本）。每个 Bootstrap 样本都略有不同，这恰好是我们想要的——让每棵树看到不同的数据。"

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging + 决策树
bagging = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=100,      # 训练 100 棵树
    max_samples=0.8,       # 每棵树看 80% 的样本（Bootstrap）
    max_features=0.8,      # 每棵树看 80% 的特征
    random_state=42
)
bagging.fit(X_train, y_train)

# 预测
y_pred = bagging.predict(X_test)
y_prob = bagging.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"Bagging AUC: {auc:.4f}")
```

小北看着代码，问了一个细节："`max_features` 是什么意思？"

"这是随机森林的第二个'随机性'来源。"老潘说，"Bagging 只让每棵树看到不同的样本，随机森林还让每棵树在每个分裂时只考虑一部分特征。这进一步降低了树之间的相关性——树越不相似，投票越有意义。"

### 随机森林：Bagging + 特征随机性

**随机森林（Random Forest）**是 Leo Breiman 在 2001 年提出的，是对 Bagging 的改进：

1. **Bootstrap 样本**：每棵树看到不同的样本
2. **特征随机性**：在每个节点**分裂时**，从所有特征中随机选取一部分（如 `sqrt(n_features)`）作为候选分裂特征，然后从中选择最优分裂。这确保每棵树的不同节点可能看到不同的特征子集。
3. **聚合**：多棵树投票

阿码问："为什么要随机选特征？不是应该让每棵树都看所有特征吗？"

"反直觉的是，**限制特征反而效果更好**。"老潘说，"这就像问路——如果你问 10 个人同一条路，他们都会告诉你同一个答案。但如果你让 10 个人从不同角度观察，他们的投票才更有参考价值。"

"如果每棵树都看所有特征，它们会选择'最强'的特征分裂，树之间会很相似——就像 10 个人用完全相同的思路想问题。随机选特征强制它们找不同的分裂方式，树更多样化，投票的效果更好。"

```python
from sklearn.ensemble import RandomForestClassifier

# 随机森林
rf = RandomForestClassifier(
    n_estimators=100,         # 100 棵树
    max_depth=5,              # 限制深度（防止过拟合）
    max_features='sqrt',      # 每次分裂考虑 sqrt(n_features) 个特征
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1                 # 并行训练
)
rf.fit(X_train, y_train)

# 注意：使用 Pipeline 可以防止数据泄漏（data leakage）。数据泄漏是指测试集的信息泄露到训练中，这会导致评估指标虚高。我们在 Week 10 学了**数据泄漏与防御**，知道预处理（如标准化、特征缩放）必须只在训练集上 fit，然后在测试集上 transform。随机 forest 本身不需要预处理，但如果在它前面有特征工程步骤，一定要用 Pipeline！

# 预测
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"随机森林 AUC: {auc:.4f}")
```

### 随机森林的优势与代价

"听起来随机森林完美无缺？"小北问。

"没有什么是完美的。"老潘说，"**随机森林降低了方差，但也付出了代价**。"

老潘打了个比方："随机森林像是一个专家委员会——决策更稳定，但你很难解释'为什么委员会这么决定'。决策树像是一个专家——可能不稳定，但你可以问他'你是怎么想的'，他能告诉你推理过程。"

**优势**：
- **预测力强**：通常比单棵决策树好很多
- **不易过拟合**：多棵树投票，降低了方差
- **稳定性好**：数据的小变化不会导致预测的大变化
- **自动特征选择**：可以输出特征重要性
- **鲁棒性强**：对异常值和噪声不敏感

**代价**：
- **可解释性下降**：你不能像单棵树那样画出整个森林——100 棵树的森林是"黑盒"
- **训练慢**：需要训练多棵树（但可以并行）
- **预测慢**：需要对多棵树求和/投票
- **模型大**：100 棵树可能占很多内存
- **调参复杂**：有更多超参数需要调整

小北问："那什么时候用决策树，什么时候用随机森林？"

"看你最关心什么。"老潘说，"如果需要向业务方解释规则，用决策树（或只训练一棵浅树）。如果追求预测力，用随机森林。但记住：**永远先和基线比**。"

> **AI 时代小专栏：AutoML 与自动化基线对比**

> 2026 年，AutoML（Automated Machine Learning）工具已经可以自动化整个建模流程：数据预处理、特征工程、模型选择、超参数调优……
>
> **常见的 AutoML 工具**：
> - **AutoGluon**（Amazon）：自动训练多个模型（逻辑回归、决策树、随机森林、XGBoost、神经网络等），并选择最好的
> - **H2O.ai**：企业级 AutoML 平台
> - **TPOT**：基于遗传算法的 AutoML
> - **Auto-sklearn**：基于 scikit-learn 的 AutoML
>
> **AutoML 的优势**：
> - **省时**：几分钟内尝试几十种模型
> - **基线对比自动化**：大多数 AutoML 工具会自动训练基线模型（如逻辑回归、Dummy Classifier），并报告相对提升量
> - **超参数优化**：自动搜索最佳超参数
>
> **AutoML 的陷阱**：
> - **"选中"了运气好的模型**：如果 AutoML 尝试了 20 个模型，最好的那个可能只是运气好（多重比较问题，见 Week 07）
> - **过拟合验证集**：如果 AutoML 用验证集选择模型，它可能过拟合验证集（需要用交叉验证）
> - **无法替代业务理解**：AutoML 可以找到"预测力强"的模型，但无法告诉你"为什么这个模型适合这个业务场景"
>
> **行业实践**：
>
> - AutoGluon 的默认策略是：先训练一个"强基线"（如随机森林），再尝试更复杂的模型（如 XGBoost、神经网络）
> - H2O.ai 会报告每个模型的训练时间、预测时间、内存占用，让你评估"提升量是否值得"
> - Google 的 Vertex AI AutoML 会自动进行基线对比，并生成"模型选择报告"
>
> **对你的启示**：
>
> AutoML 可以加速你的工作，但它不能替代你的判断。本周学的**基线对比**、**过拟合检查**、**复杂度 vs 提升量权衡**，这些都需要你自己来评估。AutoML 可以训练模型，但只有你能回答"这个模型比基线好多少"以及"这种提升是否值得"。
>
> 参考（访问日期：2026-02-18）：
> - [AutoGluon Documentation: AutoML with Automatic Baseline Comparison](https://auto.gluon.ai/stable/index.html)
> - [H2O.ai AutoML Documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
> - [Scikit-learn: DummyClassifier](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)

---

## 5. 更复杂一定更好吗？——与基线对比

小北训练了三个模型：逻辑回归（AUC 0.87）、决策树（AUC 0.85）、随机森林（AUC 0.89）。

"随机森林最好！"小北兴奋地说，"我应该用它。"

老潘问了一个让小北愣住的问题："**和基线比了吗？**"

"基线？"小北困惑了。

"最简单的模型。"老潘说，"比如'总是预测多数类'。如果傻瓜基线 AUC 0.50，随机森林 AUC 0.89，提升了 39 个点，这很厉害。但如果随机森林比逻辑回归只好了 2 个点，而训练时间慢了 50 倍，你觉得值得吗？"

小北一时答不上来。

"这就是工业界和 Kaggle 的区别。"老潘继续说，"Kaggle 只看 AUC，工业界要看 AUC、训练时间、预测延迟、部署成本、可解释性。**没有基线对比的模型选择，不是分析，是炫技**。"

### 什么是基线（Baseline）？

**基线**是一个简单的参考模型，用来评估更复杂模型的"净提升量"。常见基线包括：

| 基线类型 | 描述 | 适用场景 |
|---------|------|---------|
| **傻瓜基线（Dummy）** | 总是预测多数类，或随机预测 | 检查模型是否比"瞎猜"好 |
| **逻辑回归基线** | 最简单的线性分类器 | 作为"简单模型"的参考 |
| **单特征树基线** | 只用最重要的特征做一棵浅树 | 检查复杂模型的额外特征是否有价值 |

### 傻瓜基线

```python
from sklearn.dummy import DummyClassifier

# 总是预测多数类
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_prob_dummy = dummy.predict_proba(X_test)[:, 1]
auc_dummy = roc_auc_score(y_test, y_prob_dummy)
print(f"傻瓜基线 AUC: {auc_dummy:.4f}")
```

**输出示例**：
```
傻瓜基线 AUC: 0.5000
```

AUC = 0.5 表示随机猜测（因为傻瓜基线总是预测多数类，没有区分能力）。

### 逻辑回归基线

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 逻辑回归基线
log_reg_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])
log_reg_pipe.fit(X_train, y_train)
y_prob_lr = log_reg_pipe.predict_proba(X_test)[:, 1]
auc_lr = roc_auc_score(y_test, y_prob_lr)
print(f"逻辑回归基线 AUC: {auc_lr:.4f}")
```

### 单特征树基线

```python
# 找出最重要的特征（用逻辑回归系数或随机森林特征重要性）
# 假设我们发现 'days_since_last_purchase' 最重要
single_feature = 'days_since_last_purchase'

# 单特征树
single_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
single_tree.fit(X_train[[single_feature]], y_train)
y_prob_single = single_tree.predict_proba(X_test[[single_feature]])[:, 1]
auc_single = roc_auc_score(y_test, y_prob_single)
print(f"单特征树基线 AUC: {auc_single:.4f}")
```

### 模型对比表

| 模型 | AUC | 训练时间 | 预测时间 | 可解释性 |
|------|-----|---------|---------|---------|
| 傻瓜基线 | 0.5000 | < 1ms | < 1ms | N/A |
| 逻辑回归 | 0.8700 | ~10ms | ~1ms | 高 |
| 决策树（max_depth=3） | 0.8500 | ~5ms | ~1ms | 高 |
| 随机森林（100 棵树） | 0.8900 | ~500ms | ~10ms | 低 |

### 可视化对比：ROC 曲线

光看数字不够直观。让我们画出所有模型的 ROC 曲线，一眼看出谁更好：

```python
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))

# 画出每个模型的 ROC 曲线
colors = ['gray', 'blue', 'green', 'darkgreen']
models_list = [
    ('Dummy', dummy, 'gray'),
    ('Logistic Regression', log_reg_pipe, 'blue'),
    ('Decision Tree', tree_pruned, 'green'),
    ('Random Forest', rf, 'darkgreen')
]

for name, model, color in models_list:
    RocCurveDisplay.from_estimator(
        model, X_test, y_test, ax=ax, name=name, color=color
    )

ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison')
ax.legend(loc='lower right')
plt.show()
```

小北看完图说："随机森林的曲线确实在最上面，但逻辑回归也差不多啊。"

"对。"老潘说，"**曲线挤在一起，说明差距不大**。"

如果两条 ROC 曲线**大面积重叠**（像两条线缠绕在一起），说明在不同阈值下，两个模型的区分能力相近，提升不明显。如果一条曲线**明显压在另一条上方**（不重叠），说明提升显著。

### 特征重要性：模型在"看"什么？

随机森林虽然不如决策树直观，但可以告诉我们"哪些特征最重要"：

```python
import pandas as pd

# 获取特征重要性（随机森林）
importances = rf.feature_importances_
feature_names = X_train.columns

# 排序并可视化
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=True)

# 水平条形图
feat_imp.tail(10).plot(
    x='feature', y='importance', kind='barh',
    title='Top 10 Feature Importance (Random Forest)',
    legend=False, figsize=(8, 5)
)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
```

阿码看着图说："`days_since_last_purchase` 最重要？这和业务直觉一致——很久没买的客户更容易流失。"

"对。"老潘说，"**特征重要性不只是'模型怎么看'，也是'验证业务直觉'**。如果最重要的特征和业务理解相反，你要停下来检查数据是否有问题。"

小北看了一眼表："随机森林比逻辑回归好了 0.02 AUC，但训练时间慢了 50 倍。这值得吗？"

"看你最关心什么。"老潘说。

**决策框架**：

| 场景 | 优先级 | 推荐模型 |
|------|--------|---------|
| **需要向业务方解释** | 可解释性 | 逻辑回归 或 决策树 |
| **追求最高预测力** | AUC | 随机森林 |
| **实时预测（低延迟）** | 预测时间 | 逻辑回归 |
| **训练资源有限** | 训练时间 | 逻辑回归 或 决策树 |
| **边缘设备部署** | 模型大小 | 逻辑回归 |

### 提升量是否显著？

阿码问："0.02 AUC 的提升量，是真实的还是运气？"

"好问题。"老潘说，"你可以用统计检验判断两个模型的 AUC 是否有显著差异。这和 Week 08 学的置换检验思路一样——我们想知道'观察到的差异是否可能由随机性产生'。"

**快速判断法**：用交叉验证的标准差判断提升是否可靠。

```python
from sklearn.model_selection import cross_val_score

# 用 5 折交叉验证估计 AUC 及其不确定性
lr_cv = cross_val_score(log_reg_pipe, X, y, cv=5, scoring='roc_auc')
rf_cv = cross_val_score(rf, X, y, cv=5, scoring='roc_auc')

print(f"逻辑回归: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}")
print(f"随机森林: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print(f"提升量: {(rf_cv.mean() - lr_cv.mean()):.4f}")

# 简单判断：如果均值差 > 2×max(标准差)，提升较可靠
se_diff = np.sqrt(lr_cv.std()**2 + rf_cv.std()**2)
is_significant = (rf_cv.mean() - lr_cv.mean()) > 2 * se_diff
print(f"提升显著: {'是' if is_significant else '否'}（均值差 > 2×SE）")
```

> **注意**：这是一个粗略的经验法则，仅供参考。正式的统计检验应使用 Bootstrap + 置信区间（见 ASSIGNMENT.md 任务 6），或使用 DeLong 检验比较两个 AUC。

小北问："这个方法够准确吗？"

"对大多数场景够用。"老潘说，"交叉验证本身已经多次重采样，标准差能反映 AUC 的波动范围。如果两个模型的置信区间不重叠，提升大概率是真实的。"

**更严谨的方法**（可选）：如果你需要正式的统计检验，可以用 Bootstrap + Mann-Whitney U 检验，详见 `examples/11_bootstrap_test.py`。

小北追问："为什么不能直接比测试集上的 AUC？"

"因为**单个测试集分数有波动**。"老潘说，"交叉验证让你看到'如果数据划分不同，AUC 会怎么变'。如果两个模型的 AUC 分布完全重叠，那它们本质上差不多；如果分布明显分离，提升才是真实的。"

### 模型选择理由

老潘总结道："你在报告里不应该只写'随机森林 AUC 0.89，所以选它'。**那不是分析，是陈述事实**。你应该写："

> **模型选择理由**：
>
> - **傻瓜基线 AUC 0.50**：模型比瞎猜好（✓）
> - **逻辑回归 AUC 0.87**：作为简单模型，已经不错
> - **随机森林 AUC 0.89**：比逻辑回归提升 0.02 AUC（p < 0.05，显著）
> - **权衡**：随机森林的预测力提升 2.3%（0.02/0.87），但训练时间慢 50 倍、可解释性下降
> - **结论**：如果业务最关心预测力，选随机森林；如果需要向业务方解释规则，选逻辑回归

"这才是一个完整的分析。"老潘说，"你不仅告诉了读者'哪个模型最好'，还解释了'比基线好多少'、'提升量是否显著'、'复杂度是否值得'。"

阿码若有所思："所以 AutoML 工具如果只给我'最好'的模型，不告诉我基线……"

"那你就要补上。"老潘说，"AI 可以训练模型，但只有你能判断'这个模型值得吗'。"

现在的问题是：**如何把这些实践整合到一个可复现的建模报告中？** 这正是 StatLab 本周要解决的问题。

---

## StatLab 进度

到目前为止，StatLab 已经有了逻辑回归分类模块和评估报告。但这里有一个"看不见的坑"：我们只报告了一个模型的 AUC，没有与基线对比，不知道这个"提升量"是真实的还是运气，也不知道更复杂的模型是否值得。

这正是本周"树模型与基线对比"派上用场的地方。**本周的 StatLab 进展，是将"单一模型评估"升级为"多模型对比与基线评估"——从"只看 AUC"到"权衡提升量、复杂度与可解释性"。**

### 第一步：定义模型对比函数

我们需要一个函数，同时训练多个模型（傻瓜基线、逻辑回归、决策树、随机森林），并输出完整的对比结果。这个函数复用 Week 10 的 Pipeline 结构，防止数据泄漏：

```python
# examples/11_statlab_tree_models.py
def baseline_comparison(X, y, numeric_features, categorical_features,
                       test_size=0.3, random_state=42):
    """
    训练多个模型并与基线对比
    """
    # 1. 预处理器（复用 Week 10 的 Pipeline）
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 2. 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
```

这个函数和 Week 10 的逻辑回归训练很相似，区别在于我们要定义多个模型。

### 第二步：定义四个模型

```python
    # 3. 定义模型
    models = {
        'dummy': Pipeline([
            ('preprocessor', preprocessor),
            ('model', DummyClassifier(strategy='most_frequent'))
        ]),
        'logistic_regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        'decision_tree': Pipeline([
            ('preprocessor', preprocessor),
            ('model', DecisionTreeClassifier(
                max_depth=5,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=random_state
            ))
        ]),
        'random_forest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                max_features='sqrt',
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=random_state,
                n_jobs=-1
            ))
        ])
    }
```

老潘看到这段代码会说："**每个模型都用同一个 Pipeline**。这很重要——所有模型都在同样的预处理条件下训练，对比才公平。"

### 第三步：训练和评估

```python
    # 4. 训练和评估
    results = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            'pipeline': pipeline,
            'metrics': {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_prob)
            },
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        # 交叉验证 AUC（5 折）
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
        results[name]['cv_scores'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
```

这里有个关键点：**交叉验证用完整数据集 (X, y)**，而测试集评估用分割后的 (X_test, y_test)。这是两个不同目的：
- 交叉验证：评估模型在不同数据分割下的稳定性
- 测试集：评估模型在"未见过的数据"上的表现

### 第四步：提取特征重要性

```python
    # 5. 获取特征重要性（随机森林）
    rf_pipeline = models['random_forest']
    # 获取特征名称（数值型 + One-Hot 后的分类型）
    feature_names = numeric_features.copy()
    cat_encoder = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    for cat_feat in categorical_features:
        categories = cat_encoder.categories_[categorical_features.index(cat_feat)]
        feature_names.extend([f"{cat_feat}_{cat}" for cat in categories])

    # 获取特征重要性
    importances = rf_pipeline.named_steps['model'].feature_importances_
    results['feature_importance'] = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return results
```

### 第五步：格式化报告

```python
def format_model_comparison_report(results):
    """格式化模型对比结果为 Markdown 报告"""
    md = ["## 模型对比与选择\n\n"]

    # 1. 模型对比表
    md.append("### 评估指标对比\n\n")
    md.append(f"| 模型 | 准确率 | 精确率 | 召回率 | F1 | AUC | 交叉验证 AUC |\n")
    md.append(f"|------|--------|--------|--------|-----|-----|-------------|\n")

    for name, res in results.items():
        if name == 'feature_importance' or name.startswith('X_') or name.startswith('y_'):
            continue
        metrics = res['metrics']
        cv = res['cv_scores']
        md.append(f"| {name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
                  f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['auc']:.4f} | "
                  f"{cv['mean']:.4f} (+/- {cv['std']:.4f}) |\n")

    md.append("\n")

    # 2. 基线对比
    md.append("### 基线对比\n\n")
    dummy_auc = results['dummy']['metrics']['auc']
    lr_auc = results['logistic_regression']['metrics']['auc']
    rf_auc = results['random_forest']['metrics']['auc']

    md.append(f"- **傻瓜基线 AUC**: {dummy_auc:.4f}\n")
    md.append(f"- **逻辑回归基线 AUC**: {lr_auc:.4f}\n")
    md.append(f"- **随机森林 AUC**: {rf_auc:.4f}\n")
    md.append(f"- **提升量**:\n")
    md.append(f"  - 逻辑回归 vs 傻瓜基线: +{(lr_auc - dummy_auc):.4f}\n")
    md.append(f"  - 随机森林 vs 逻辑回归: +{(rf_auc - lr_auc):.4f} ({(rf_auc - lr_auc)/lr_auc*100:.1f}%)\n\n")

    # 3. 特征重要性（Top 10）
    if 'feature_importance' in results:
        md.append("### 特征重要性（随机森林，Top 10）\n\n")
        top_features = results['feature_importance'].head(10)
        md.append(f"| 排名 | 特征 | 重要性 |\n")
        md.append(f"|------|------|--------|\n")
        for idx, row in top_features.iterrows():
            md.append(f"| {top_features.index.get_loc(idx) + 1} | {row['feature']} | {row['importance']:.4f} |\n")
        md.append("\n")

    # 4. 模型选择理由（根据提升量自动生成建议）
    md.append("### 模型选择理由\n\n")
    md.append("**基线对比结论**：\n\n")
    md.append(f"- 所有模型的 AUC 都显著高于傻瓜基线（{dummy_auc:.4f}），说明模型比瞎猜好\n\n")

    md.append("**复杂度 vs 提升量权衡**：\n\n")
    improvement = (rf_auc - lr_auc) / lr_auc * 100
    if improvement < 2:
        md.append(f"- 随机森林比逻辑回归提升 {improvement:.1f}%，提升量较小\n")
        md.append("- 如果业务最关心预测力，选随机森林；如果需要向业务方解释规则，选逻辑回归\n\n")
    elif improvement < 5:
        md.append(f"- 随机森林比逻辑回归提升 {improvement:.1f}%，提升量中等\n")
        md.append("- 如果预测力是关键，选随机森林；如果需要可解释性，选逻辑回归\n\n")
    else:
        md.append(f"- 随机森林比逻辑回归提升 {improvement:.1f}%，提升量显著\n")
        md.append("- 建议选择随机森林\n\n")

    md.append("**可解释性考虑**：\n\n")
    md.append("- **逻辑回归**：可解释性高，可以解读每个特征的系数方向和强度\n")
    md.append("- **决策树**：可解释性高，可以画出树结构，直观展示决策规则\n")
    md.append("- **随机森林**：可解释性较低，但可以输出特征重要性\n\n")

    return "".join(md)
```

### 第六步：可视化对比

```python
def plot_model_comparison(results, figsize=(14, 5)):
    """画模型对比图表"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. AUC 对比柱状图
    models = ['dummy', 'logistic_regression', 'decision_tree', 'random_forest']
    model_names = ['Dummy\nBaseline', 'Logistic\nRegression', 'Decision\nTree', 'Random\nForest']
    aucs = [results[m]['metrics']['auc'] for m in models]

    bars = axes[0].bar(model_names, aucs, color=['gray', 'blue', 'green', 'darkgreen'])
    axes[0].set_ylabel('AUC')
    axes[0].set_title('Model Comparison (Test Set AUC)')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color='red', linestyle='--', label='Random Guess')

    # 2. 交叉验证 AUC 分布（箱线图）
    cv_data = [results[m]['cv_scores']['scores'] for m in models]
    bp = axes[1].boxplot(cv_data, labels=model_names, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['gray', 'blue', 'green', 'darkgreen']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1].set_ylabel('AUC')
    axes[1].set_title('Cross-Validation AUC Distribution (5-fold)')
    axes[1].axhline(y=0.5, color='red', linestyle='--', label='Random Guess')

    plt.tight_layout()
    return fig
```

### 使用示例

```python
# 加载数据（用泰坦尼克数据集作为示例）
import seaborn as sns
titanic = sns.load_dataset("titanic")

# 准备特征和目标
feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
X = titanic[feature_cols].copy()
y = titanic['survived']

# 定义特征类型
numeric_features = ['age', 'sibsp', 'parch', 'fare']
categorical_features = ['pclass', 'sex', 'embarked']

# 训练并对比模型
results = baseline_comparison(X, y, numeric_features, categorical_features)

# 生成报告
report = format_model_comparison_report(results)
print(report)

# 画图
fig = plot_model_comparison(results)
plt.show()

# 写入文件
from pathlib import Path
Path("output/model_comparison_report.md").parent.mkdir(parents=True, exist_ok=True)
Path("output/model_comparison_report.md").write_text(report)
print("\n报告已保存到 output/model_comparison_report.md")
```

现在 `report.md` 会多出一个"模型对比与选择"章节，包括：
- 评估指标对比（准确率、精确率、召回率、F1、AUC）
- 基线对比（傻瓜基线、逻辑回归、随机森林）
- 特征重要性（随机森林）
- 模型选择理由（权衡提升量、复杂度、可解释性）
- 模型对比图表

### 与本周知识的连接

**决策树** → 你学会了从根节点到叶节点的递归分裂，理解了决策树的可解释性和可视化优势。决策树可以画出完整的决策规则，这在需要向业务方解释模型时非常有用。

**过拟合与剪枝** → 你学会了决策树容易过拟合（树太深会记住训练数据），并通过预剪枝（限制深度、最小样本数）和后剪枝（CCP）来控制。这呼应了 Week 09 学的过拟合：复杂模型容易过拟合，需要通过诊断和控制来平衡拟合与泛化。

**随机森林** → 你学会了用 Bagging（Bootstrap + Aggregating）和特征随机性来降低决策树的方差，用"群体智慧"提升预测力。Bagging 本质上是 Week 08 学的 Bootstrap 方法的应用——只是这次 Bootstrap 用于训练多个模型，而不是估计统计量的分布。

**基线对比** → 你学会了与傻瓜基线、逻辑回归基线、单特征树基线对比，评估"更复杂的模型是否值得"。这让你在报告里能解释"为什么选择这个模型"，而不是只给一个 AUC 数字。

### 与上周的对比

| 上周 | 本周 |
|------|------|
| 逻辑回归（线性模型） | 决策树（树形模型） |
| 单一模型评估 | 多模型对比与基线评估 |
| 只看 AUC | 权衡提升量、复杂度、可解释性 |
| 数据泄漏防护（Pipeline） | 过拟合防护（剪枝、Bagging） |
| 分类可解释性（系数、混淆矩阵） | 树可解释性（树结构可视化）、特征重要性 |

老潘看到这段改动会说什么？"这才是完整的建模分析。你不仅告诉了读者'哪个模型最好'，还解释了'比基线好多少'、'提升量是否显著'、'复杂度是否值得'。"

小北问："基线对比真的那么重要吗？"

"比你想的更重要。"老潘说，"我见过一个项目，团队花了三个月调优一个神经网络，最后发现它比逻辑回归只好了 0.5% AUC，但部署成本高了 100 倍。如果他们一开始就做了基线对比，就不会走这个弯路。"

阿码若有所思："所以 AutoML 工具如果只给我'最好'的模型，不告诉我基线……"

"那你就要补上。"老潘说，"**AI 可以训练模型，但只有你能判断'这个模型值得吗'**。"

---

## Git 本周要点

本周必会命令：
- `git status`（查看未跟踪的新文件：树模型脚本、模型对比图表）
- `git diff`（查看对 StatLab 报告生成脚本的修改）
- `git add -A`（添加所有变更）
- `git commit -m "draft: add tree models and baseline comparison"`（提交树模型与基线对比）

常见坑：
- 不做基线对比，直接选择"最高 AUC"的模型；
- 决策树不限制深度，导致严重过拟合；
- 忽略可解释性，只看预测力；
- 误认为"随机森林总是最好的"，不考虑业务场景；
- 混淆 Bagging 和随机森林（Bagging 只在样本层面随机，随机森林在样本和特征两个层面都随机）。

老潘的建议：**永远先训练基线模型**。没有基线对比的模型选择，不是分析，是炫技。

---

## Definition of Done（学生自测清单）

本周结束后，你应该能够：

- [ ] 理解决策树的工作原理（根节点、分裂规则、叶节点）
- [ ] 解读决策树可视化，解释树的决策规则
- [ ] 理解决策树如何选择分裂特征（信息增益、基尼不纯度）
- [ ] 识别决策树的过拟合，用预剪枝/后剪枝控制
- [ ] 理解随机森林的原理（Bagging + 特征随机性）
- [ ] 训练决策树和随机森林，并与基线对比
- [ ] 权衡提升量、复杂度、可解释性，做出模型选择
- [ ] 在 StatLab 报告中写出"带基线对比和模型选择理由"的建模结论

---

## 本周小结（供下周参考）

老潘最后给了一个比喻："逻辑回归像是一条直线，简单但可能不够灵活。决策树像是一系列 if-else 规则，灵活但容易过拟合。随机森林像是一群专家投票，稳定但不那么直观。"

"**没有'最好的'模型，只有'最适合'的模型**。"

这周你学会了**树模型与基线对比**：从"线性决策边界"到"阶梯式决策边界"，从"单棵树的不稳定"到"随机森林的群体智慧"。

你理解了**决策树的工作原理**：根节点、内部节点、叶节点，以及树如何递归地分裂数据。你学会了**可视化决策树**，直观地展示决策规则——这在需要向业务方解释模型时非常有用。

你掌握了**过拟合与剪枝**：决策树如果不加限制，会一直分裂直到记住训练数据。预剪枝（限制深度、最小样本数）和后剪枝（CCP）是控制过拟合的两种方法。这呼应了 Week 09 学的过拟合诊断：复杂模型容易过拟合，需要通过诊断和控制来平衡。

更重要的是，你学会了**随机森林**：用 Bagging（Bootstrap + Aggregating）和特征随机性来降低决策树的方差。Bagging 本质上是 Week 08 学的 Bootstrap 方法的应用——只是这次 Bootstrap 用于训练多个模型，而不是估计统计量的分布。

最后，你学会了**基线对比**：与傻瓜基线、逻辑回归基线、单特征树基线对比，评估"更复杂的模型是否值得"。你知道"没有基线对比的模型选择不是分析，是炫技"。

老潘的总结很简洁："**AUC 只是一个数字。更重要的是：这个数字比基线高多少？提升量是否显著？复杂度是否值得？**"

### 三个模型的核心对比

| 维度 | 逻辑回归 | 决策树 | 随机森林 |
|------|---------|--------|----------|
| **决策边界** | 线性（直线） | 阶梯式（与坐标轴平行） | 复杂的阶梯式组合 |
| **可解释性** | 高（系数方向和强度） | 高（树结构可视化） | 低（100+棵树的黑盒） |
| **过拟合风险** | 低（线性约束强） | 高（容易记住数据） | 中（投票降低方差） |
| **稳定性** | 高（小数据变化影响小） | 低（小变化可能导致大不同） | 高（群体智慧） |
| **训练速度** | 快 | 快 | 慢（需训练多棵树） |
| **预测速度** | 快 | 快 | 慢（需遍历多棵树） |
| **何时使用** | 需要解释规则、特征少 | 需要 if-else 规则、展示决策过程 | 追求预测力、不急需解释 |
| **如何调参** | 正则化强度（C） | max_depth、min_samples_* | n_estimators、max_depth、max_features |

下周，我们将继续深入**预测建模**：从"模型可解释性"到"模型伦理与公平性"。你本周学的基线对比、模型选择权衡、复杂度 vs 提升量，在下周都会用到。

下周的核心问题是："模型不仅能预测，还能被理解吗？它会放大偏见吗？"
