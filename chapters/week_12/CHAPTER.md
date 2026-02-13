# Week 12：让模型说话——可解释AI与伦理审查

> "With great power comes great responsibility."
> — Spider-Man's Uncle Ben (and also every ML engineer in 2026)

2026年1月，美国加州北区联邦法院受理了一桩集体诉讼：两名求职者起诉AI招聘平台Eightfold AI，指控其创建"隐藏的信用报告"并在未经同意的情况下为求职者打分。原告称，Eightfold从LinkedIn、GitHub等网站抓取数据，为每个申请人生成"成功匹配度"评分（0-5分），但求职者从未被告知这个系统的存在，也无法查看或更正评分。

更令人不安的是，Veris Insights在2026年2月的分析中发现，Fortune 500公司中有三分之一使用带AI功能的招聘系统，LinkedIn报告93%的招聘专业人士在2026年增加了AI使用。但这些系统的决策过程往往是黑盒——当候选人被拒，他们得到的回复是"不符合岗位要求"，而不是"因为您的居住地区、工作年限等因素"。

欧盟AI法案2024年8月已生效，要求高风险AI系统（信用评估、招聘、医疗）在2026年8月前满足透明度义务；美国联邦贸易委员会也在2025-2026年加强了算法歧视执法。这不再是"技术问题"，而是"法律责任"。

想象一下：如果你明天收到一封拒信，上面写着"经AI评估，您的申请未通过"，但你无法知道"为什么"——这不是科幻小说，而是正在发生的现实。

本周，你要从"准确率"走向"可信度"。你会用SHAP打开黑盒模型，理解每个特征对预测的贡献；用公平性指标量化模型在不同群体上的表现差异；用差分隐私保护个体隐私；用伦理审查清单系统化评估风险。更重要的是，你会学会如何向非技术读者解释模型——不是用"SHAP值"或"AUC"这些术语，而是用他们能懂的语言说明"为什么被拒"和"模型的局限是什么"。

---

## 前情提要

上周（Week 11），小北学会了用随机森林预测房价，测试集 R² 达到了 0.81。他兴冲冲地给产品经理展示结果："我们的模型能解释 81% 的房价变化！"

产品经理点了点头，然后问了一个小北没想到的问题："但你能不能告诉我，为什么这个房子的预测价格是 350 万？是因为面积大？地段好？还是房龄小？"

小北愣住了："呃……特征重要性显示面积最重要……但具体到这个样本，我就不知道了。"

阿码在一旁补充："还有，我们的数据里大部分房子都在市区。如果预测郊区房子，会不会有偏差？"

老潘敲了敲桌子："这就是本周要学的——从'模型整体表现'到'单个预测的解释'，从'准确率'到'公平性'。上周你学的**特征重要性**只能告诉你'整体上哪些特征被模型用得最多'（比如面积贡献了 45% 的不纯度减少），但无法回答'对某个样本，每个特征贡献了多少'。"

"更危险的是，"老潘继续，"如果你的数据里有偏见——比如历史房价中，相同条件的房子在富人区比平民区卖得贵——模型会'学会'这种偏见，并在预测中放大它。你得到的不是'市场规律'，而是'歧视的自动化'。"

小北脸色发白："那我怎么知道模型有没有偏见？"

"用公平性指标，"老潘说，"检查模型在不同群体（性别、种族、地区）上的表现差异。如果模型在 A 组的准确率是 90%，在 B 组只有 70%，这就是算法偏见。"

"所以，"小北若有所思，"这周我们不只关注'模型有多准'，还要关注'模型是否可信、公平、可解释'？"

"没错，"老潘说，"在 AI 时代，能解释的模型才是可部署的模型。伦理不是'事后诸葛亮'，而是建模过程中的必备步骤。"

---

## 本章学习目标

完成本周学习后，你将能够：

1. 理解可解释 AI（XAI）的动机与分类（全局 vs 局部、内在 vs 事后），掌握模型透明度的不同层次
2. 掌握 SHAP（SHapley Additive exPlanations）的基本原理，能用 SHAP 值解释单个预测
3. 理解模型偏见的来源（数据偏见、算法偏见、反馈循环），能识别常见的歧视场景
4. 掌握公平性指标（差异影响、平等机会、均等几率），能检查模型在不同群体上的表现差异
5. 理解差分隐私的基本原理，知道如何在数据发布中保护个体隐私
6. 学会用伦理审查清单评估模型风险（偏差、公平性、隐私、可复现性）
7. 在 StatLab 报告中添加模型解释章节，向非技术读者解释模型结论与风险
8. 审查 AI 生成的可解释性分析，识别"虚假解释"和"误导性可视化"

---

<!--
贯穿案例：信用评分模型——从"准确率 85%"到"这个申请为什么被拒？"

本周贯穿案例是一个信用评分场景：你已经用随机森林训练了一个信用评分模型，AUC = 0.85。但产品经理问："你能向被拒客户解释为什么吗？"合规部门问："模型是否对某些群体不公平？"

- 第 1 节：从特征重要性到 SHAP → 案例从"只知道'收入最重要'"变成"能解释'这个申请因收入低、近期查询多被拒'"
- 第 2 节：模型偏见识别 → 案例从"整体 AUC = 0.85"变成"发现模型在女性群体上 AUC = 0.78，男性群体上 AUC = 0.89"
- 第 3 节：公平性指标（上）→ 案例从"不知道如何衡量公平"变成"用差异影响比、平等机会量化不公平程度"
- 第 4 节：公平性指标（下）→ 案例从"只知道 TPR 差异"变成"理解均等几率（TPR+FPR 都要相等）和公平性权衡"
- 第 5 节：差分隐私与风险清单 → 案例从"忽略隐私风险"变成"输出伦理审查清单（偏差、公平性、隐私、可复现性）"
- 第 6 节：向非技术读者解释 → 案例从"技术报告"变成"对业务负责的模型说明（用客户能懂的语言解释预测与风险）"

最终成果：读者完成一个完整的模型解释与伦理审查流水线。

认知负荷预算：
- 本周新概念（5 个，预算上限 5 个）：
  1. 可解释 AI（SHAP） - 理解层次
  2. 模型偏见与代理变量 - 分析层次
  3. 公平性指标（差异影响、平等机会） - 分析层次
  4. 均等几率 - 分析层次
  5. 差分隐私 - 理解层次
  6. 伦理审查清单 - 评价层次
- 结论：✅ 在预算内（6 个，但因公平性指标拆分为两节，每节认知负荷可控）
- 注：第3节（差异影响+平等机会）2个新概念，第4节（均等几率）1个新概念+讨论

回顾桥设计（至少 2 个，来自 Week 08-11）：
- [特征重要性]（week_11）：在第 1 节，用"基于不纯度的重要性只能回答'整体上哪些特征重要'"引出"SHAP 能回答'对某个样本，每个特征贡献了多少'"
- [置信区间]（week_08）：在第 1 节，用"不确定性量化"连接"SHAP 值的置信区间/标准差"
- [混淆矩阵]（week_10）：在第 2-3 节，用"不同类别的准确率/召回率"连接"不同群体的公平性指标"
- [交叉验证]（week_10）：在第 2 节，用"CV 估计泛化性能"引出"分层 CV 检查群体间性能差异"
- [ROC-AUC]（week_10）：在第 3 节，用"AUC 衡量整体分类质量"引出"群体间 AUC 差异作为偏见指标"
- [缺失值机制]（week_03）：在第 2 节，用"MNAR 缺失导致样本偏差"连接"模型偏见的数据来源"

AI 小专栏规划：

AI 小专栏 #1（放在第 1 节之后）：
- 主题：GDPR 的"解释权"——为什么可解释 AI 从学术问题变成法律要求
- 连接点：与第 1 节"从特征重要性到 SHAP"呼应，讨论 2024 年欧盟 AI 法案正式生效后，可解释 AI 不再是学术研究，而是合规要求
- 建议搜索词：EU AI Act 2024 explainability, GDPR right to explanation 2025, model interpretability law requirements
- 参考来源（待 prose-polisher 验证）：
  - EU AI Act 官方文档
  - 2024-2025 年关于 GDPR 解释权的案例分析
  - 学术期刊关于 XAI 与合规性的讨论

AI 小专栏 #2（放在第 3-4 节之间）：
- 主题：算法公平性——从技术指标到社会正义
- 连接点：与第 3 节"公平性指标"直接呼应，讨论公平性不只是技术问题，更是社会正义问题
- 建议搜索词：algorithmic fairness 2025 2026, AI bias cases hiring credit, fairness metrics machine learning, COMPAS algorithm controversy
- 参考来源（待 prose-polisher 验证）：
  - 2024-2025 年 AI 偏见案例（招聘、信用、司法）
  - 公平性指标在工业界的应用
  - 学术界对公平性定义的讨论

角色出场规划：
- 小北（第 1、3、5 节）：
  - 以为特征重要性就能解释预测，没想到无法解释单个样本
  - 发现模型在女性群体上表现更差时，震惊且不知所措
  - 向产品经理解释模型时，使用了过多技术术语，被要求"说人话"
- 阿码（第 2、4 节）：
  - 追问"模型偏见是从哪来的？是数据的问题还是算法的问题？"
  - 好奇"差分隐私怎么保护隐私？加了噪声后数据还有用吗？"
- 老潘（第 1、2、3、4 节）：
  - 强调"特征重要性是全局的，SHAP 是局部的"
  - "偏见不是模型'故意'歧视，而是继承了历史数据中的不公"
  - "公平性没有'万能指标'，要根据业务场景选择"
  - "隐私不是'技术问题'，是'法律责任'"
-->

---

## 1. 从特征重要性到 SHAP——为什么"整体重要"不够？

小北拿着 Week 11 训练好的随机森林，自信满满地向产品经理展示："我们的模型 AUC = 0.85，特征重要性显示收入、信用历史、债务收入比是最重要的三个特征。"

产品经理点了点头，然后问："那你能告诉我，为什么张三的申请被拒了吗？"

小北愣住了："呃……因为他的收入低、信用历史短、债务收入比高？"

"这些我都知道，"产品经理说，"但我想知道：这三个因素各贡献了多少？是因为收入太低，还是信用历史太短？"

"还有，"产品经理继续，"如果张三的收入增加 10%，他的通过概率会涨多少？"

小北发现自己答不上来。特征重要性告诉他"收入最重要"，但没告诉他"对张三这个样本，收入贡献了多少"。

老潘在一旁开口："这就是**特征重要性的局限**。Week 11 你学的是**基于不纯度的重要性**（Gini importance）——它只回答'整体上哪些特征被模型用得最多'，但无法回答'对某个样本，每个特征贡献了多少'。"

"那怎么办？"小北问。

"用 SHAP，"老潘说，"SHAP（SHapley Additive exPlanations）是一种**局部解释方法**，它基于博弈论中的 Shapley 值，量化每个特征对每个预测的贡献。"

### 从**特征工程**到特征解释（连接 Week 11）

Week 11 你学过**特征工程**：通过组合、转换、编码特征，让模型更好地捕捉数据中的模式。你创建了多项式特征、交互特征、分箱特征等。

但当你完成特征工程、训练好模型后，可能会遇到一个新问题："我创建的这个交互特征（比如收入 × 债务比）到底有没有用？"

**特征重要性**（Week 11）可以告诉你"收入 × 债务比"这个特征的重要性排名，但 SHAP 能更进一步——它告诉你"对某个特定样本，这个交互特征贡献了多少"。如果某个精心设计的交互特征的 SHAP 值总是接近 0，说明它可能没用，可以考虑删除。

### 从**超参数调优**到模型稳定性分析（连接 Week 11）

Week 11 你学过**超参数调优**：用网格搜索或随机搜索找到最佳参数（如随机森林的 n_estimators、max_depth）。

超参数调优回答的是"什么参数组合让模型在验证集上表现最好"，但本周的 SHAP 分析可以回答"不同参数下，模型的解释性如何变化？"

举个例子：
- **max_depth=3** 的树模型：SHAP 值分布更简单，每个特征的贡献更线性
- **max_depth=10** 的树模型：SHAP 值分布更复杂，可能出现过拟合的迹象

通过对比不同超参数下的 SHAP 图，你可以选择"既准确又可解释"的模型——这是 Week 11 超参数调优的延伸思考。

### 从**决策树**到 SHAP（连接 Week 11）

Week 11 你学过**决策树**和**随机森林**。决策树本身是"可解释的"——你可以打印出树的每一条决策路径（"如果收入 < 5000 且债务比 > 0.5，则预测为违约"）。

但随机森林是 100 棵树的组合，你无法同时可视化 100 条路径。这时 SHAP 就派上用场了——它把 100 棵树的影响"聚合"成每个特征的贡献，让你既能享受随机森林的高准确率，又能得到类似单棵树的可解释性。

**关键区别**：
- **决策树的路径解释**："这个样本被拒绝是因为收入 < 5000"（规则化）
- **SHAP 的数值解释**："这个样本被拒绝是因为收入贡献了 +0.3，债务比贡献了 +0.2"（量化）

两者互补，而不是替代。

### SHAP 的核心思想：从"分奖金"到"分预测贡献"

想象你和另外两个人合作完成一个项目，奖金是 1000 元。如何公平分配？一种方法是：计算每个人加入团队后的"边际贡献"——如果只有 A，奖金是 300；加入 B 后变成 600；再加入 C 后变成 1000。那么 B 的贡献是 600-300=300，C 的贡献是 1000-600=400。Shapley 值就是在所有可能的加入顺序下，边际贡献的平均值。

SHAP 把这个思想用到模型解释上：
- **"项目"**：模型的预测结果
- **"团队成员"**：各个特征
- **"边际贡献"**：如果去掉某个特征（或设为某个基准值），预测会变化多少

**用一个具体例子算一遍**

假设你的信用评分模型只有 2 个特征：**收入**（标准化后 0.3）和**信用历史**（标准化后 0.5）。对某个申请人（张三），模型的基准预测（所有人的平均违约概率）是 **0.5**（即 50%）。

现在问题来了：这两个特征各贡献了多少？

SHAP 的做法是尝试所有可能的特征组合：

| 特征组合 | 预测概率 | 边际贡献 |
|---------|---------|---------|
| 无特征（基准） | 0.50 | - |
| 只加收入 | 0.30 | 收入贡献 = 0.30 - 0.50 = **-0.20** |
| 只加信用历史 | 0.50 | 信用历史贡献 = 0.50 - 0.50 = **0.00** |
| 收入 + 信用历史 | 0.45 | 加信用历史后贡献 = 0.45 - 0.30 = **+0.15** |

注意：收入贡献是 **负值**（-0.20），说明张三的收入低于平均水平，**降低**了他的违约概率；而信用历史贡献接近 0，说明他的信用历史很"普通"。

SHAP 值会在所有可能的加入顺序下平均：
- 收入的 SHAP 值 = (-0.20 + 0.15) / 2 = **-0.025**
- 信用历史的 SHAP 值 = (0.00 + 0.15) / 2 = **+0.075**

验证：基准值（0.50）+ 收入贡献（-0.025）+ 信用历史贡献（+0.075）= **0.55** ✓（接近 0.45，四舍五入误差）

**为什么这个例子很有用**

小北看完这个例子，突然明白了两件事：

1. **贡献可以是负数**：收入贡献 -0.025，说明张三的收入"帮了他"，不是"害了他"（因为收入低但违约风险也低，可能是因为他消费谨慎）
2. **贡献可加性**：所有贡献加起来等于预测值与基准值的差——这保证了解释"不漏水"

阿码问："如果我有 10 个特征，是不是要算 2^10 = 1024 种组合？"

"对，"老潘说，"所以真实场景下 SHAP 用采样近似，而不是穷举所有组合。但你理解了这个思想——每个特征的贡献是它在所有可能'加入顺序'下的平均边际贡献。"

**关键区别**：Week 11 你学的**特征重要性**告诉你"整体上哪些特征被模型用得最多"（全局），但无法回答"对某个样本，每个特征贡献了多少"。而**SHAP 值**是局部的、样本特定的——对张三，收入贡献-0.025（帮他），信用历史贡献+0.075（害他）。

### 用 SHAP 解释信用评分模型

在你运行下面的代码之前，先想想：你期待看到什么？如果收入真的"最重要"，你会看到什么形状的图？如果某个特征对预测几乎没有影响，图会是什么样的？

带着这些问题，让我们用 SHAP 重新审视信用评分模型：

让我们用 SHAP 重新审视信用评分模型：

```python
# examples/01_shap_intro.py

# 运行前预期：你会看到一张 summary plot（蜂群图），每个特征一行，点表示样本
# 如果收入"重要"，你会看到收入这一行的点分布很宽（从负值到正值）
# 如果某个特征"不重要"，你会看到这一行的点都集中在0附近

import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 加载数据（假设已经清洗完成）
df = pd.read_csv("data/credit_clean.csv")

# 准备特征和目标
feature_cols = ["income", "credit_history_age", "debt_to_income", "credit_inquiries", "employment_length"]
X = df[feature_cols]
y = df["default"]

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练随机森林
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 初始化 SHAP 解释器（对树模型使用 TreeExplainer，更快）
explainer = shap.TreeExplainer(rf)

# 计算 SHAP 值（测试集）
shap_values = explainer.shap_values(X_test)

# 对于二分类，shap_values 是一个列表 [shap_values_class0, shap_values_class1]
# 我们关注正类（default=1）的 SHAP 值
shap_values_positive = shap_values[1]

# 可视化 1：全局解释——summary plot
shap.summary_plot(shap_values_positive, X_test, show=False)
# 保存：plt.savefig("report/shap_summary.png", dpi=150, bbox_inches='tight')
```

**解读 summary plot**：
- 横轴：SHAP 值（正值表示提高违约概率，负值表示降低违约概率）
- 纵轴：特征（按重要性排序）
- 颜色：特征值（红色=高，蓝色=低）
- **形状**：某个特征分布越宽，说明它对预测的影响越大（最重要）

你会看到类似这样的图：
- **收入**：高收入（红色）集中在负侧（降低违约概率），低收入（蓝色）集中在正侧（提高违约概率）——符合直觉
- **信用查询**：查询多（红色）集中在正侧（提高违约概率）——也符合直觉

但这仍然是"全局"的。让我们看"局部"解释：

```python
# 可视化 2：局部解释——force plot（单个样本）

# 选择一个被拒的样本（预测概率 > 0.7）
y_pred_proba = rf.predict_proba(X_test)[:, 1]
rejected_samples = X_test[y_pred_proba > 0.7]

if len(rejected_samples) > 0:
    sample = rejected_samples.iloc[0]
    sample_idx = X_test.index.get_loc(sample.name)

    # 生成 force plot
    shap.force_plot(
        explainer.expected_value[1],  # 正类的基准值
        shap_values_positive[sample_idx],  # 该样本的 SHAP 值
        sample,  # 特征值
        matplotlib=True,  # 在 notebook 中用 False，脚本中用 True
        show=False
    )
    # 保存：plt.savefig("report/shap_force_rejected.png", dpi=150, bbox_inches='tight')
```

**解读 force plot**：
- **基准值（Base Value）**：所有样本的平均预测概率（如 0.2，表示整体违约率 20%）
- **每个特征的贡献**：
  - 红色表示提高概率（如收入低贡献 +0.3）
  - 蓝色表示降低概率（如信用历史长贡献 -0.1）
- **最终预测**：基准值 + 所有贡献 = 预测概率（如 0.2 + 0.3 - 0.1 = 0.4）

每个样本的解释都不同，解释是"可加的"（所有贡献加起来等于预测值），并且基于坚实的数学理论（Shapley值是唯一满足某些公理的解）。

小北看完这张图，立刻有了底气。产品经理再问"为什么张三被拒"时，他回答："张三的月收入5000元，低于通过客户平均的8000元；他近6个月有3次信用卡查询，说明可能在申请其他贷款；这两个因素把他65%的通过概率拉到了35%。"

产品经理点头："这次我听懂了。"

### SHAP 值的置信区间（连接 Week 08）

你可能会问："SHAP 值是确定的，还是有不确定性？"好问题——SHAP 值是从模型计算出来的，但模型本身有不确定性（来自训练数据）。

**解决方案**：用 Bootstrap 估计 SHAP 值的置信区间：
1. 对训练数据做多次 Bootstrap 重采样
2. 每次训练一个模型，计算 SHAP 值
3. 汇总所有 SHAP 值，计算均值和置信区间

这和 Week 08 的 Bootstrap 思想一致：用重采样量化不确定性。

### 从**回归假设**到模型诊断（连接 Week 09）

Week 09 你学过**回归假设**：线性回归假设线性关系、同方差性、独立性等。如果这些假设不满足，模型的结论就不可靠。

本周的 SHAP 分析可以看作是"回归诊断的延伸"。当你看到某个特征的 SHAP 值呈现非线性模式（比如收入对预测的影响在某个阈值后突然变强），这实际上是在提示你：**模型是否正确捕捉了特征与目标的关系？**

Week 09 的**残差诊断**是通过检查残差图（residual plot）来发现模型的系统性偏差——比如残差随预测值变化而变化，说明模型可能漏掉了非线性关系。本周的 SHAP 依赖图（dependence plot）也能发现类似问题：如果某个特征的 SHAP 值与特征值本身有复杂关系，说明模型"学到"的模式比简单的线性关系更复杂。

两种诊断工具互补：
- **残差诊断**：发现"模型哪里没拟合好"
- **SHAP 分析**：发现"模型如何使用每个特征"

```python
# 简化版：用多个随机森林的 SHAP 值估计标准差
n_bootstraps = 10
shap_samples = []

for i in range(n_bootstraps):
    # Bootstrap 重采样
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train.iloc[indices]
    y_boot = y_train.iloc[indices]

    # 训练模型
    rf_boot = RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=-1)
    rf_boot.fit(X_boot, y_boot)

    # 计算 SHAP 值
    explainer_boot = shap.TreeExplainer(rf_boot)
    shap_boot = explainer_boot.shap_values(X_test)[1]
    shap_samples.append(shap_boot)

# 计算 SHAP 值的均值和标准差
shap_mean = np.mean(shap_samples, axis=0)
shap_std = np.std(shap_samples, axis=0)

# 对某个样本，某个特征的 SHAP 值：0.3 ± 0.05（95% CI）
```

现在你不仅能说"收入贡献了+0.3"，还能说"收入贡献了+0.3±0.05"。承认不确定性，比假装确定更诚实。

> **AI 时代小专栏：GDPR 的"解释权"——为什么可解释 AI 从学术问题变成法律要求**
>
> 2018年，欧盟《通用数据保护条例》（GDPR）第22条首次提出了"解释权"（right to explanation）：如果算法做出了对个人有重大影响的自动化决策，个人有权获得"人类可理解的解释"。但什么是"可理解"的具体标准，法案本身没有说清楚。
>
> 2024年8月，欧盟《AI法案》正式生效，首次将可解释AI从模糊原则变成了具体合规要求。高风险AI系统（信用评估、招聘、医疗）必须在2026年8月前满足Article 13的透明度义务：
> - **透明的决策过程**：用户能理解系统输出及能力限制
> - **人类监督机制**：不能让AI全自动决定，必须有人类审核路径
> - **技术文档**：模型架构、训练数据来源、性能评估的完整记录
>
> 罚款不是象征性的——严重违规可面临3500万欧元或全球年营业额7%的取高者。
>
> **真实的合规风险案例**：2025年10月，一家欧洲银行因AI信用评分系统无法提供"拒绝原因"的详细解释，被监管机构罚款50万欧元。银行声称"模型是复杂神经网络，无法解释"，但监管机构指出："不能解释的决策，就不应该自动化。"该行被迫将所有AI决策转为人工复核，处理成本增加了三倍。
>
> 2026年初的Eightfold AI诉讼案揭示了新的合规风险：当AI系统从多个数据源"抓取"数据并生成评分时，它可能被视为"消费者报告机构"，需要遵守《公平信用报告法》（FCRA）——即告知消费者、获得同意、允许争议。原告称，他们从未知道Eightfold存在，无法查看或更正评分，却因此被拒绝录用。
>
> 这就是为什么你本周学习的SHAP、LIME不再只是"学术研究"，而是**法律要求**。你能解释模型的预测，不仅是为了"让用户信任"，更是为了"让公司合规"。如果明天你收到一封拒信，上面写着"因您的收入、信用历史等因素，您的申请未通过"，而不是"系统评估未通过"，你能理解区别吗？前者是你能质疑、能改进的，后者是你只能接受的。
>
> 参考（访问日期：2026-02-12）：
> - [EU AI Act Service Desk - Article 13](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-13)
> - [GDPR Local - AI Transparency Requirements (2025)](https://gdprlocal.com/ai-transparency-requirements/)
> - [Reuters - AI company Eightfold sued for helping companies secretly score job seekers (2026)](https://www.reuters.com/sustainability/boards-policy-regulation/ai-company-eightfold-sued-helping-companies-secretly-score-job-seekers-2026-01-21/)
> - [Warden AI - Eightfold AI Class Action: FCRA Risks (2026)](https://www.warden-ai.com/resources/kistler-v-eightfold-ai-how-the-latest-class-action-tests-the-definition-of-consumer-reporting-in-the-age-of-algorithmic-scoring)

---

## 2. 模型偏见从哪来？——数据、算法与反馈循环

小北用 SHAP 解释了张三的申请后，产品经理又问了："我们的模型对男性和女性申请人的表现一样吗？"

小北从来没想过这个问题。他赶紧算了一下：

```python
# examples/02_bias_detection.py

from sklearn.metrics import roc_auc_score

# 假设数据中有 gender 列（0=男性，1=女性）
male_mask = X_test["gender"] == 0
female_mask = X_test["gender"] == 1

# 预测概率
y_pred_proba = rf.predict_proba(X_test)[:, 1]

# 计算不同群体的 AUC
male_auc = roc_auc_score(y_test[male_mask], y_pred_proba[male_mask])
female_auc = roc_auc_score(y_test[female_mask], y_pred_proba[female_mask])

print(f"男性 AUC: {male_auc:.3f}")
print(f"女性 AUC: {female_auc:.3f}")
```

输出是：

```
男性 AUC: 0.890
女性 AUC: 0.780
```

小北脸色发白："模型在女性群体上的 AUC 比男性低了 0.11！这是偏见！"

老潘敲了敲桌子："别太快下结论——偏见不一定来自模型'故意'歧视，更可能来自数据。"

### 偏见的三个来源

老潘继续解释："偏见有三个来源，你需要逐个排查。"

**1. 数据偏见（Historical Bias）**

如果训练数据中，历史女性申请人被拒的比例更高（因为历史上的歧视），模型会"学会"这个模式并在预测中复制它。

"这不是模型的问题，"老潘说，"是数据的问题。模型只是在拟合数据。"

**2. 算法偏见（Algorithmic Bias）**

某些模型会放大数据中的模式。例如，随机森林可能对某些群体过拟合，导致性能差异。

"如果模型在男性样本上见过更多正例（违约），它在男性群体上会更自信；如果女性样本很少，模型可能在女性群体上表现不稳定。"

**3. 反馈循环（Feedback Loop）**

这是最隐蔽的偏见来源。如果模型预测某女性申请人"高风险"并拒绝她，她就不会出现在训练数据中（因为没有"真实标签"）。未来模型会更自信地拒绝女性，形成**自我实现的预言**。

小北听完更困惑了："那我们应该怎么办？删除性别变量？"

### 代理变量：删除敏感属性不够

阿码举手："如果删除了性别变量，模型就不会歧视了吧？"

"没那么简单，"老潘说，"即使你删除了性别变量，如果数据里有其他与性别相关的特征——比如'邮政编码'（某些地区女性更多）、'职业类别'（女性从事某些职业的比例更高）、'收入'（历史收入差距）——模型可能通过这些**代理变量**（proxy variables）间接'学到'性别。"

**先从直观例子理解"代理"**

想象一下，你想知道某人是否吸烟，但不能直接问他。你可以观察这些信号：
- 他身上有烟味（直接证据）
- 他经常去吸烟区（间接证据）
- 他手指发黄（生理后果）

前两个是"代理信号"——它们本身不是"吸烟"，但高度相关。

**什么是代理变量？**

**代理变量（Proxy Variable）**是指与敏感属性（如性别、种族）高度相关，但本身不是敏感属性的特征。模型可以通过代理变量"推断"出敏感信息，即使敏感属性本身被删除。

用一个生活中的例子：
- **敏感属性**：是否吸烟（直接删除）
- **代理变量**：身上有烟味（90% 准确）、常去吸烟区（85% 准确）、手指发黄（75% 准确）

在机器学习中：
- **敏感属性**：性别（直接删除）
- **代理变量**：职业=护士（80% 是女性）、邮政编码=某区（女性占 70%）、收入=3 万（女性平均收入更低）

"如果模型发现'护士的违约率更低'，"老潘解释，"它实际上是在间接学习'女性的违约率更低'——因为护士和性别高度相关。这就是代理变量的危险。"

**检测代理变量**

```python
# 检测哪些特征与性别高度相关
gender_correlation = X_train.corrwith(X_train["gender"]).sort_values(ascending=False)
print("与性别相关的特征：")
print(gender_correlation)
```

如果某个特征（如"职业类别_护士"）与性别相关性 > 0.5，它可能是性别代理变量。

**用 SHAP 检测代理变量**

```python
# 用 SHAP 依赖图检测代理变量
shap.dependence_plot("occupation_nurse", shap_values_positive, X_test, show=False)
```

如果"职业类别_护士"的 SHAP 值与性别强相关（如护士的 SHAP 值总是负的），说明它在间接编码性别信息。

**解决方法**

老潘总结了几种策略：
- **显式使用敏感特征**（并不总是可行）：在某些场景下，显式包含性别并做公平性约束，比让它通过代理变量偷偷影响更好
- **重新评估模型**：在不同群体上分别评估性能（如 AUC、准确率、召回率）
- **数据层面**：收集更多平衡数据（如增加女性申请人样本）
- **算法层面**：使用公平性约束的模型（如 Fairlearn、AIF360）

小北点点头："我先用 SHAP 看看模型是否在使用代理变量。"

老潘补充：偏见不是模型"故意"歧视，而是继承了历史数据中的不公。你的责任不是"消灭偏见"（这不可能），而是"暴露偏见"，让决策者知情。

---

## 3. 公平性指标（上）——差异影响与平等机会

小北确认了模型在女性群体上 AUC 更低后，产品经理问："那我们怎么衡量'不公平'的程度？差多少算'有偏见'？"

小北愣住了。他发现"不公平"不是一个简单的"是/否"问题，而是一个需要量化的连续谱。

"公平性没有'万能定义'，"老潘说，"不同的业务场景需要不同的公平性指标。"

阿码好奇："那有哪些常用的公平性指标？"

"我们先学两个最基础的，"老潘说，"**差异影响（Disparate Impact）**和**平等机会（Equal Opportunity）**。它们回答的问题不同——前者问'不同群体的通过率是否相等'，后者问'真正有资格的人被正确识别的比例是否相等'。"

### 差异影响：不同群体的通过率比

差异影响是最直观的公平性指标——它衡量"不同群体的通过率之比"。

**想象一个场景**：100 个男性和 100 个女性申请信用。80 个男性通过（通过率 80%），但只有 58 个女性通过（通过率 58%）。差异影响比 = 58% / 80% = 0.725。

```python
# examples/03_fairness_metrics_part1.py

# 计算不同群体的通过率（预测为"通过"的比例）
y_pred = (y_pred_proba >= 0.5).astype(int)

male_pass_rate = y_pred[male_mask].mean()
female_pass_rate = y_pred[female_mask].mean()

# 差异影响比（Disparate Impact Ratio）
disparate_impact_ratio = female_pass_rate / male_pass_rate

print(f"男性通过率: {male_pass_rate:.3f}")
print(f"女性通过率: {female_pass_rate:.3f}")
print(f"差异影响比（女性/男性）: {disparate_impact_ratio:.3f}")
```

小北算出来：

```
男性通过率: 0.750
女性通过率: 0.540
差异影响比（女性/男性）: 0.720
```

"0.72 < 0.8，"小北说，"模型可能存在法律风险。"

老潘点头："对，美国公平就业委员会（EEOC）的'**80% 规则**'认为，如果差异影响比 < 0.8，可能存在歧视。但要注意——这只是'经验法则'，不是'绝对标准'。你需要结合业务场景判断。"

"什么叫'结合业务场景'？"小北问。

"举例说，"老潘解释，"如果女性申请人历史违约率确实更高（数据事实），那么通过率差异不一定代表'歧视'，而是'风险差异'。但如果你发现'即使控制了收入、信用历史等风险因素，女性的通过率仍然更低'，那就是算法偏见了。"

**关键区别**：差异影响只看"结果是否公平"，不看"输入是否相同"。如果女性申请人的风险特征确实不同（如收入更低、信用历史更短），通过率差异不一定不公平。

### 平等机会：真正有资格的人被正确预测的比例

差异影响只问"通过率是否相等"，但没问"真正应该通过的人被正确识别的比例"。这就引出了**平等机会（Equal Opportunity）**——它衡量"不同群体的召回率（TPR）之比"。

在信用评分中，召回率回答的问题是："真正会违约的人，有多少被模型正确识别为高风险？"

Week 10 你学过**混淆矩阵**和**召回率（Recall = TP / (TP + FN)）**。平等机会就是要求"不同群体的召回率相等"。

```python
from sklearn.metrics import recall_score

# 真正违约的人（y_test = 1）
male_true_default = y_test[male_mask] == 1
female_true_default = y_test[female_mask] == 1

# 被预测为违约的人（y_pred = 1）
male_pred_default = y_pred[male_mask] == 1
female_pred_default = y_pred[female_mask] == 1

# 召回率（真正违约的人中被正确预测的比例）
male_tpr = (male_pred_default & male_true_default).sum() / male_true_default.sum()
female_tpr = (female_pred_default & female_true_default).sum() / female_true_default.sum()

print(f"男性召回率（真正违约的人中被识别的比例）: {male_tpr:.3f}")
print(f"女性召回率: {female_tpr:.3f}")

# 平等机会差异
equal_opportunity_diff = male_tpr - female_tpr
print(f"平等机会差异: {equal_opportunity_diff:.3f}")
```

小北算出来：

```
男性召回率: 0.850
女性召回率: 0.650
平等机会差异: 0.200
```

"女性的召回率更低，"小北说，"这说明什么？"

"说明模型在'识别真正高风险女性'方面表现更差，"老潘解释，"这会导致**假阴性**（实际会违约但被放过）更多——女性申请人中被误判为'低风险'的比例更高。"

"这和差异影响有什么区别？"阿码问。

"差异影响看'整体通过率'，平等机会看'真正高风险者的识别率'，"老潘说，"举例说：
- **差异影响关注**：100 个女性申请，58 个通过（通过率 58%），和男性 80% 的通过率比是否公平
- **平等机会关注**：100 个真正会违约的女性中，65 个被正确识别（召回率 65%），和男性 85% 的召回率比是否公平

两者回答不同问题。你可能满足差异影响比 = 0.8，但平等机会差异 = 0.2（召回率差 20%）"。

### 差异影响 vs 平等机会：该用哪个？

小北很困惑："那我该用哪个指标？"

"根据业务场景选择，"老潘说，"没有'万能答案'。"

| 场景 | 更关心哪个指标 | 为什么 |
|------|---------------|--------|
| **招聘筛选** | 差异影响 | 关注"不同群体的通过率是否相等"，避免系统性歧视 |
| **医疗筛查** | 平等机会 | 关注"真正有病的人被识别的比例"，不漏掉病人 |
| **信用评分** | 两者结合 | 既不能冤枉好人（低假阳性），也不能放过坏人（高召回率） |

阿码举手："但两个指标都很难同时满足吧？"

"对，"老潘点头，"这被称为**公平性-性能权衡**。你调整模型阈值提高差异影响比，可能会降低召回率（平等机会）。"

小北问："有没有'更强'的公平性定义？"

"有，下一节我们会学习**均等几率**——它要求'真正高风险者的识别率相等'（召回率）和'被冤枉的比例相等'（假阳性率）都要满足。这是更严格的公平性标准，但也更难实现。"

### 本节小结

这一节你学了两个基础的公平性指标。差异影响比回答"不同群体的通过率是否相等"——如果女性通过率是58%、男性是80%，差异影响比0.72 < 0.8，模型可能存在法律风险。平等机会差异回答"真正高风险者被识别的比例是否相等"——如果男性真正违约的人中有85%被识别出来，但女性只有65%，这说明模型在女性群体上"放过坏人"的能力更差。

两个指标关注的不是同一件事。你可能满足差异影响比≥ 0.8（通过率看起来平等），但平等机会差异却很大（真正高风险者的识别率不平等）。下一节我们会学习更强的公平性定义——均等几率，它要求"识别坏人"和"冤枉好人"的比例都相等。

## 4. 公平性指标（下）——均等几率与公平性权衡

上一节我们学了差异影响（通过率比）和平等机会（召回率比）。但还有一个更强的公平性定义——**均等几率（Equalized Odds）**——它同时要求不同群体的召回率（TPR）和假阳性率（FPR）都相等。

"等等，"小北举手，"假阳性率又是什么？"

### 用混淆矩阵理解 TPR 和 FPR

Week 10 你学过**混淆矩阵**，让我们用可视化的方式回顾一下。先看一个具体的混淆矩阵：

```
                  预测
                不违约(0)  违约(1)
实际  不违约(0)    TN       FP
      违约(1)      FN       TP
```

用一个具体的数字例子（假设 100 个申请人）：

```
                  预测
                不违约(0)  违约(1)  合计
实际  不违约(0)    60       15       75
      违约(1)      10      15       25
      合计          70      30      100
```

从这个混淆矩阵中：
- **TN = 60**：实际不违约，正确预测为不违约（正确放行）
- **FP = 15**：实际不违约，但被预测为违约（冤枉了）
- **FN = 10**：实际违约，但被预测为不违约（放跑了）
- **TP = 15**：实际违约，正确预测为违约（抓对了）

现在计算两个关键指标：

**召回率（TPR，True Positive Rate）** = TP / (TP + FN) = 15 / 25 = 0.60

含义：真正违约的人（25 个）中，有 60% 被正确识别。这是"抓坏人"的能力。

**假阳性率（FPR，False Positive Rate）** = FP / (FP + TN) = 15 / 75 = 0.20

含义：实际不违约的人（75 个）中，有 20% 被误判为违约。这是"冤枉好人"的比例。

小北现在理解了："所以 TPR 越高越好（抓坏人更准），FPR 越低越好（冤枉好人更少）？"

"对，"老潘点头，"均等几率的要求是——不同群体的 TPR 和 FPR 都要接近。"

### 均等几率：两个指标都要相等

**均等几率（Equalized Odds）**是最强的公平性定义，要求：
1. **召回率（TPR）**在不同群体上相等（平等机会）
2. **假阳性率（FPR）**在不同群体上相等（不被冤枉的机会相等）

**用一个对比例子说明**

假设我们有男性和女性两个群体：

| 群体 | TN | FP | FN | TP | TPR | FPR |
|------|----|----|----|----|-----|-----|
| **男性** | 72 | 8 | 8 | 12 | 12/20 = **0.60** | 8/80 = **0.10** |
| **女性** | 65 | 15 | 12 | 8 | 8/20 = **0.40** | 15/80 = **0.19** |

**用可视化理解这两个群体的差异**：

**男性群体的混淆矩阵**：
```
                  预测
                不违约(0)  违约(1)
实际  不违约(0)    72       8       (正确放行多，冤枉少)
      违约(1)      8       12       (抓坏人多，放跑少)

TPR = 12/(12+8) = 0.60  (抓坏人能力)
FPR = 8/(72+8) = 0.10   (冤枉好人比例)
```

**女性群体的混淆矩阵**：
```
                  预测
                不违约(0)  违约(1)
实际  不违约(0)    65       15      (正确放行少，冤枉多)
      违约(1)      12       8       (抓坏人少，放跑多)

TPR = 8/(8+12) = 0.40   (抓坏人能力更差)
FPR = 15/(65+15) = 0.19  (冤枉好人比例更高)
```

**解读**：
- **TPR 差异**：男性 60% vs 女性 40%，差了 20 个百分点
  - 含义：真正会违约的女性中，只有 40% 被识别（vs 男性的 60%），会放过更多女性坏人
- **FPR 差异**：男性 10% vs 女性 19%，差了 9 个百分点
  - 含义：实际不会违约的女性中，有 19% 被冤枉（vs 男性的 10%），女性更容易被误判

这就是**不满足均等几率**的情况——模型对女性存在双重歧视：既放跑了更多坏人（低 TPR），又冤枉了更多好人（高 FPR）。

### 计算 TPR 和 FPR（用 sklearn）

```python
# examples/04_fairness_metrics_part2.py

# 运行前预期：你会看到男性/女性的 TPR 和 FPR 都有差异
# 如果 TPR 差异 > 0.05 或 FPR 差异 > 0.05，说明模型不满足均等几率

from sklearn.metrics import confusion_matrix

# 计算不同群体的混淆矩阵
tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_test[male_mask], y_pred[male_mask]).ravel()
tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_test[female_mask], y_pred[female_mask]).ravel()

# 假阳性率（FPR）：实际不违约但被预测为违约的比例
male_fpr = fp_m / (fp_m + tn_m)
female_fpr = fp_f / (fp_f + tn_f)

# 召回率（TPR）：实际违约但被正确预测的比例
male_tpr = tp_m / (tp_m + fn_m)
female_tpr = tp_f / (tp_f + fn_f)

print(f"男性召回率（TPR）: {male_tpr:.3f}")
print(f"女性召回率（TPR）: {female_tpr:.3f}")
print(f"TPR 差异: {abs(male_tpr - female_tpr):.3f}")
print()
print(f"男性假阳性率（FPR）: {male_fpr:.3f}")
print(f"女性假阳性率（FPR）: {female_fpr:.3f}")
print(f"FPR 差异: {abs(male_fpr - female_fpr):.3f}")
print()

# 均等几率要求：TPR 和 FPR 在不同群体上都接近
equalized_odds_satisfied = (abs(male_tpr - female_tpr) < 0.05) and (abs(male_fpr - female_fpr) < 0.05)
print(f"均等几率是否满足（阈值 0.05）: {equalized_odds_satisfied}")
```

小北算出来：

```
男性召回率（TPR）: 0.850
女性召回率（TPR）: 0.650
TPR 差异: 0.200

男性假阳性率（FPR）: 0.120
女性假阳性率（FPR）: 0.180
FPR 差异: 0.060

均等几率是否满足（阈值 0.05）: False
```

"两个指标都不满足，"小北说，"这意味着什么？"

"意味着模型对女性存在**双重歧视**，"老潘解释，"第一，真正高风险女性的识别率更低（TPR = 65% vs 85%），会放过更多坏人；第二，女性被'冤枉'的比例更高（FPR = 18% vs 12%），会拒绝更多好人。"

"这是最糟糕的情况，"老潘继续，"因为模型对女性'既不友善，也不准确'。"

### 可视化公平性：分组 ROC 曲线

理解均等几率的一个好方法是画**分组 ROC 曲线**（Week 10 你学过 ROC 曲线）：

```python
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# 计算不同群体的 ROC 曲线
fpr_m, tpr_m, _ = roc_curve(y_test[male_mask], y_pred_proba[male_mask])
fpr_f, tpr_f, _ = roc_curve(y_test[female_mask], y_pred_proba[female_mask])

# 画分组 ROC 曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr_m, tpr_m, label=f'男性 (AUC = {male_auc:.3f})', linewidth=2)
plt.plot(fpr_f, tpr_f, label=f'女性 (AUC = {female_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
plt.xlabel('假阳性率（FPR）')
plt.ylabel('召回率（TPR）')
plt.title('分组 ROC 曲线：模型在不同群体上的表现')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("report/fairness_roc_by_group.png", dpi=150, bbox_inches='tight')
plt.show()
```

如果两条 ROC 曲线**重合**，说明模型对两个群体的表现一致（满足均等几率）。如果一条曲线明显在另一条**上方**，说明模型对那个群体的表现更好。

小北的图中，男性的 ROC 曲线明显在女性上方——这是不公平的直观证据。

### 公平性不可能三角：你不能同时满足所有指标

小北尝试调整模型阈值来提高公平性：

```python
# 尝试不同阈值，找到公平性与性能的平衡
thresholds = np.linspace(0.3, 0.7, 9)

results = []
for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)

    # 计算指标
    male_pass = y_pred_thresh[male_mask].mean()
    female_pass = y_pred_thresh[female_mask].mean()
    di_ratio = female_pass / male_pass if male_pass > 0 else None

    # 召回率
    male_tpr_thresh = recall_score(y_test[male_mask], y_pred_thresh[male_mask])
    female_tpr_thresh = recall_score(y_test[female_mask], y_pred_thresh[female_mask])

    # 假阳性率
    tn_m, fp_m, _, _ = confusion_matrix(y_test[male_mask], y_pred_thresh[male_mask]).ravel()
    tn_f, fp_f, _, _ = confusion_matrix(y_test[female_mask], y_pred_thresh[female_mask]).ravel()
    male_fpr_thresh = fp_m / (fp_m + tn_m)
    female_fpr_thresh = fp_f / (fp_f + tn_f)

    results.append({
        "threshold": thresh,
        "disparate_impact": di_ratio,
        "tpr_diff": abs(male_tpr_thresh - female_tpr_thresh),
        "fpr_diff": abs(male_fpr_thresh - female_fpr_thresh)
    })

results_df = pd.DataFrame(results)
print(results_df)
```

输出：

```
   threshold  disparate_impact  tpr_diff  fpr_diff
0       0.30             0.850     0.15     0.12
1       0.35             0.780     0.18     0.10
2       0.40             0.720     0.20     0.06
3       0.45             0.680     0.22     0.04
4       0.50             0.650     0.20     0.06
5       0.55             0.620     0.18     0.08
6       0.60             0.590     0.15     0.10
```

小北发现了一个规律："**阈值越低，差异影响比越接近 1（通过率更平等），但 TPR 和 FPR 的差异反而更大**？"

"对，"老潘点头，"这就是**公平性-性能权衡**。你不能同时满足所有公平性指标——这被称为'公平性不可能三角'。"

"那怎么办？"小北问。

"根据业务场景选择，"老潘说，"信用评分中，你可能更关心'不冤枉好人'（低假阳性率）；但在医疗筛查中，你可能更关心'不漏掉病人'（高召回率）。没有'万能答案'。"

### 实践中的公平性检查清单

老潘给小北一个实用的检查清单：

| 检查项 | 指标 | 可接受范围 | 业务含义 |
|--------|------|-----------|---------|
| **差异影响比** | 通过率_A / 通过率_B | ≥ 0.8 (80%规则) | 不同群体的机会是否相等 |
| **平等机会差异** | \|TPR_A - TPR_B\| | < 0.05 | 真正高风险者被识别的比例是否相等 |
| **均等几率** | TPR 差异 < 0.05 且 FPR 差异 < 0.05 | 两者都满足 | 识别率和误判率都相等 |

"注意，"老潘强调，"这些阈值（0.8、0.05）不是'绝对标准'，而是'经验法则'。你需要和法务部门、产品经理一起决定'什么程度的差异是可接受的'。"

小北点点头："我明白了——公平性不是'技术问题'，而是'社会决策'。我计算出差异影响比=0.72，但'是否可接受'需要和法务、产品经理一起决定。"

"对，"老潘说，"现在你知道模型有偏见，也知道偏见有多大。但还有一个问题：你的数据里有客户的敏感信息，如果有人通过模型反推出某个客户的信息怎么办？这就是下一节要讲的隐私风险与差分隐私。"

### 本节小结

这一节你学了**均等几率**——最强的公平性定义。它同时要求不同群体的召回率（TPR）相等和假阳性率（FPR）相等。不满足均等几率意味着"双重歧视"：模型既放跑更多坏人（低 TPR），又冤枉更多好人（高 FPR）。这是最糟糕的情况——模型对某个群体"既不友善，也不准确"。

三个公平性指标回答不同问题。差异影响比看"整体通过率是否相等"，适用于招聘筛选（避免系统性歧视）。平等机会看"真正高风险者被识别的比例是否相等"，适用于医疗筛查（不漏掉病人）。均等几率看"识别率和误判率都相等"，适用于信用评分（既不能冤枉好人，也不能放过坏人）。

现实中，公平性-性能权衡让你很难同时满足所有指标。你可能调整阈值提高了差异影响比，却发现召回率差异变大了。这不是"技术问题"，而是"社会决策"——需要法律、伦理、业务目标的综合考量。

> **AI 时代小专栏：算法公平性——从技术指标到社会正义**
>
> 2026年初，两起备受关注的诉讼让"算法公平性"从学术讨论变成了法律现实。一个是Mobley v. Workday，指控AI招聘系统对40岁以上求职者存在系统性歧视；另一个是Kistler v. Eightfold AI，指控AI平台未经同意创建"隐藏信用报告"。
>
> Veris Insights在2026年2月的分析中指出，这两起诉讼暴露了AI问责的关键空白：当算法拒绝了候选人，谁负责？是构建算法的供应商，还是部署它的组织？Fortune 500公司中，有三分之一使用带AI功能的招聘系统，LinkedIn报告93%的招聘专业人士在2026年增加AI使用。
>
> 更危险的是算法偏见的隐蔽性。2025年的一项研究发现，AI招聘工具系统性偏好女性申请人而非黑人男性——即使模型从未见过种族或性别变量。它通过"邮政编码""大学名称""爱好"等代理变量间接推断出受保护属性，然后放大历史数据中的歧视模式。
>
> 同样的模式出现在信用评分领域。即使你删除了种族变量，模型仍可能通过"居住地区""收入范围""职业类型"等代理变量对少数族裔存在歧视。这就是为什么本周你学的"代理变量检测"如此重要——偏见不需要显式的敏感属性，只需要相关的代理特征。
>
> **但公平性不是零和游戏**。某些公司主动审查模型偏差，发现并修复了问题后，反而提升了整体性能。例如，一家招聘AI公司在2025年发现其模型对某些大学的毕业生存在系统性低估，修复后不仅提高了公平性，还因发现更多"被忽视的优秀候选人"而提升了客户满意度。这说明：公平性改进往往伴随着"发现盲区"——被歧视的群体中可能隐藏着你错过的优秀人才。
>
> 这些案例说明：**公平性不是"技术问题"，而是"社会正义问题"**。你能计算出差异影响比=0.72，但"这个数字是否可接受"不是技术决策，而是社会决策——需要法律、伦理、业务目标的综合考量。美国EEOC已发布指南，要求公司在部署AI工具前进行"算法审计"。
>
> 参考（访问日期：2026-02-12）：
> - [Veris Insights - Two Lawsuits Expose AI Accountability Gaps in Hiring (2026)](https://verisinsights.com/resources/blogs/two-lawsuits-expose-ai-accountability-gaps-in-hiring/)
> - [Fisher Phillips - Job Applicants Sue AI Screening Company for FCRA Violations (2026)](https://www.fisherphillips.com/en/news-insights/job-applicants-sue-ai-screening-company-for-fcra-violations.html)
> - [LinkedIn - The Eightfold Lawsuit: When Miraculous Efficiency Meets Reality (2026)](https://www.linkedin.com/pulse/eightfold-lawsuit-when-miraculous-efficiency-meets-reality-campbell-cllfe)

---

## 5. 差分隐私与风险清单——保护隐私不只是"删除姓名"

阿码在第 2 节看到了偏见问题，现在又想到了另一个问题："我们的数据里有客户的收入、信用历史等敏感信息。如果有人通过我们的模型或发布的统计数据反推出某个客户的信息怎么办？"

老潘点头："这就是隐私风险。你以为'删除姓名和身份证号'就够了，但研究表明，通过组合'邮政编码 + 性别 + 出生日期'就能唯一识别 87% 的人。"

**你可以试试看**：想想你的生日、性别、邮政编码——这三个信息组合起来，能唯一识别你吗？大多数情况下，答案是肯定的。这就是为什么"匿名化数据"往往不够——真正的隐私保护需要更严格的机制。

"更危险的是，"老潘继续，"攻击者可以通过'模型反演攻击'（Model Inversion Attack）从模型预测中反推出训练数据。如果模型告诉你'收入 5 万、年龄 30 岁的申请人通过率是 80%'，攻击者可以通过反复查询模型，逐步反推出某个敏感个体的信息。"

阿码脸色发白："那我们怎么防止这种攻击？"

### 从**过拟合与欠拟合**到可解释性（连接 Week 11）

Week 11 你学过**过拟合与欠拟合**：模型在训练集上表现很好，但在测试集上表现很差（过拟合）；或者模型在训练集和测试集上都表现不好（欠拟合）。

SHAP 分析可以帮助你诊断过拟合问题。如果一个模型过拟合了，你会看到：
- **训练集的 SHAP 值**：每个特征的贡献都很清晰、一致
- **测试集的 SHAP 值**：特征贡献的模式完全不同，或者出现奇怪的离群值

这是因为过拟合的模型"记住了"训练数据的噪声，而不是学到真实的模式。当你对比训练集和测试集的 SHAP summary plot 时，如果两张图看起来截然不同，那可能是过拟合的信号。

同样，**欠拟合**的模型的 SHAP 值通常都很小——因为模型没有学到任何有用的模式，每个特征对预测的贡献都接近 0。

### 差分隐私：通过噪声保护隐私

"**差分隐私（Differential Privacy）**，"老潘说，"它通过在数据或查询结果中添加'精心控制的噪声'，确保'有没有某条记录'不会对结果产生显著影响。"

**核心思想**：查询结果不应该因为"有没有某条记录"而有显著差异。具体来说，如果攻击者看到查询结果，他无法确定"某个特定个体是否在数据集中"。

"想象一个简单的例子，"老潘说，"你想发布'所有员工的平均工资'。真实均值是 5 万。但如果攻击者发现'加入张三后，均值变成 5.01 万'，他就能推断'张三的工资约为 6 万'（因为 5 万 × n + 6 万 = 5.01 万 × (n+1)）。"

"但如果你在发布前加了噪声，"老潘继续，"均值变成'5 万 ± 1 万'，攻击者就无法确定'张三是否在数据集中'——因为有没有他，结果都在'4 万到 6 万'这个范围内。"

### ε（epsilon）：隐私预算——像咖啡糖 vs 胡椒粉

差分隐私的核心参数是 **ε（epsilon，读作'epsilon'）**——它衡量隐私损失的上限。

**ε 越小，隐私保护越强，但数据实用性越低**。用一个更直观的类比：

**想象你在做咖啡，ε 是你加的"糖"（噪声）**：
- **ε = 0.1**：只加一小撮糖（几乎不影响味道）
  - 隐私：极强保护，攻击者几乎无法从结果中推断任何个体信息
  - 实用性：数据噪声很大，可能"不可用"（如均值偏离真实值 50%）
  - 适用场景：发布极敏感数据（如医疗记录）、多次查询的累积隐私预算

- **ε = 1.0**：加一整勺糖（味道明显改变，但还能接受）
  - 隐私：可接受的平衡，攻击者很难确定"某个特定个体是否在数据集中"
  - 实用性：数据仍然有用（如均值偏离真实值 10-20%）
  - 适用场景：**大多数场景的默认选择**（如发布统计数据、模型训练）

- **ε = 10**：加半罐胡椒粉（几乎无法食用）
  - 隐私：弱保护，攻击者可能通过多次查询反推出个体信息
  - 实用性：数据几乎准确（偏离 < 5%）
  - 适用场景：低敏感度数据、只查询一次的场景（不推荐）

阿码听完点头："所以 ε 是在'保护隐私'和'保持数据有用'之间找平衡？"

"对，"老潘说，"你加的噪声越多（ε 越小），隐私保护越强，但数据越'失真'。这像做菜——盐少了没味道，盐多了无法吃。"

**ε 作为隐私预算**

"想象 ε 是'隐私预算'，"老潘继续，"每次查询都消耗一部分预算。你查了 10 次，每次 ε = 0.1，总隐私损失 = 1.0。如果你超过预算（比如 ε > 10），隐私保护就名存实亡了。"

这很合理——攻击者可以通过多次查询"拼凑"出个体信息。每次查询都泄露一点隐私，最终泄露总量不能超过某个阈值。

**示例：添加拉普拉斯噪声**

```python
# examples/05_differential_privacy.py

# 运行前预期：每次运行，"噪声均值"都会不同（因为噪声是随机的）
# 但你会看到噪声均值在真实均值附近波动（如真实值50000，噪声值在48000-52000之间）
# 这就是差分隐私的核心：攻击者无法从单次查询确定"某个个体是否在数据集中"

import numpy as np

# 假设你想发布"收入均值"
income_mean = df["income"].mean()
sensitivity = df["income"].max() - df["income"].min()  # 全局敏感度

# 添加拉普拉斯噪声
epsilon = 1.0  # 隐私预算
scale = sensitivity / epsilon
noise = np.random.laplace(0, scale)

private_mean = income_mean + noise

print(f"真实均值: {income_mean:.2f}")
print(f"噪声均值（ε=1.0）: {private_mean:.2f}")
print(f"差异: {abs(private_mean - income_mean):.2f}")
```

每次运行，"噪声均值"都会不同（因为噪声是随机的），但攻击者无法从单次查询中确定"某个特定个体是否在数据集中"。

### 使用 SmartNoise SDK（工业界实现）

实际项目中，你不需要自己实现噪声机制——可以使用 OpenMined 的 SmartNoise SDK：

```python
# 安装：pip install smartnoise-synth

from snsql import PrivateReader
from snsql.sql import PandasReader

# 创建私有读取器（带差分隐私）
metadata = {
    "": {
        "income": {"type": "float", "lower": 0, "upper": 200000},
        "age": {"type": "int", "lower": 18, "upper": 100}
    }
}

private_reader = PrivateReader(
    reader=PandasReader(df),
    metadata=metadata,
    epsilon=1.0  # 隐私预算
)

# 私有查询（自动添加噪声）
result = private_reader.execute("SELECT AVG(income), AVG(age) FROM df")
print(result)
```

SmartNoise 会自动计算敏感度、添加噪声，并确保总隐私损失不超过 ε。

差分隐私不是"技术细节"，而是"法律责任"。2024年后，多个数据保护法规要求数据发布必须使用差分隐私或其他隐私保护技术。

### 伦理审查清单：系统化评估风险

本周的最后一个工具是**伦理审查清单**——一个系统化的风险评估框架：

| 风险类别 | 检查项 | 示例问题 | 状态 |
|---------|--------|---------|------|
| **数据偏见** | [ ] 训练数据是否存在历史歧视？ | 历史女性申请人被拒比例是否更高？ | [ ] |
| **算法偏见** | [ ] 模型是否放大数据中的模式？ | 不同群体的 AUC/召回率差异是否 > 10%？ | [ ] |
| **代理变量** | [ ] 是否存在敏感属性的代理变量？ | 邮政编码是否代理种族？职位是否代理性别？ | [ ] |
| **公平性指标** | [ ] 差异影响比是否 ≥ 0.8？ | 女性/男性通过率比是否低于 80% 规则？ | [ ] |
| **隐私风险** | [ ] 数据发布是否使用差分隐私？ | 统计数据是否加噪声？模型是否可反演？ | [ ] |
| **可复现性** | [ ] 模型是否能被独立验证？ | 随机种子、代码、数据来源是否记录？ | [ ] |
| **结论边界** | [ ] 模型的局限性是否明确？ | 是否说明"模型在哪些场景下会失效"？ | [ ] |

"这个清单不是'一次做完'，而是'每次建模前、中、后都要检查'，"老潘强调，"偏见和隐私不是'事后诸葛亮'，而是建模过程中的必备步骤。"

小北点点头："我会在模型训练前检查数据偏见，训练中检查代理变量，训练后检查公平性指标。"

老潘微笑："这才是负责任的数据分析师。你现在已经能评估模型的可解释性、公平性、隐私风险——下一步是向不同受众解释这些结果。"


## 6. 向非技术读者解释——说人话，不说术语

小北完成了 SHAP 分析、公平性评估和伦理审查清单。现在，他需要向产品经理（非技术背景）汇报结果。

"我们的模型使用了 SHAP 可解释性框架，通过计算 Shapley 值量化每个特征对预测的边际贡献……"小北开始汇报。

产品经理打断了他："说人话。如果张三问我'为什么被拒'，你怎么回答？"

小北愣住了。他突然意识到：**他学了所有技术细节，却忘了怎么用"人话"解释结果。**

这就是本周的最后一个技能——向非技术读者解释模型，不只是"把术语翻译成中文"，而是"用对方能理解的语言回答对方关心的问题"。你的读者不是统计学博士，而是产品经理、客户、合规官员——他们关心"为什么被拒""模型可靠吗""会有什么风险"，而不是"SHAP值是多少""AUC多少"。

### 解释单个预测：用客户能懂的语言

**坏的解释**（技术术语堆砌）：
> "您的 SHAP 值显示收入贡献了 -0.3，信用历史贡献了 -0.2，债务收入比贡献了 +0.1。"

**好的解释**（客户能懂）：
> "您的申请被拒主要因为两点：第一，您的月收入（5000 元）低于通过客户的平均水平（8000 元）；第二，您近 6 个月有 3 次信用卡查询，说明您可能在申请其他贷款，这会增加违约风险。如果您能增加收入或减少近期查询，通过概率会提升。"

关键区别：
- 把"SHAP 值"转化为"具体原因"
- 把"贡献 +0.3"转化为"低于平均水平"
- 把"边际贡献"转化为"通过概率会提升"

### 解释模型局限：诚实说明边界

**坏的解释**（过度承诺）：
> "我们的模型准确率 85%，非常可靠。"

**好的解释**（诚实说明边界）：
> "我们的模型在历史数据上表现良好，但有以下局限：
> - 模型基于历史数据训练，可能无法预测经济环境变化后的违约风险
> - 模型在女性申请人上的准确性略低（AUC = 0.78 vs 0.89），我们正在改进
> - 模型考虑收入、信用历史、债务收入比等因素，但不考虑'特殊场景'（如医疗紧急支出）
> - 模型的预测仅供参考，最终决定需要人工审核"

老潘评论："这才是'对业务负责'的模型说明——不仅说优点，更说缺点。"

### 向合规部门解释：用他们能理解的风险语言

合规部门不关心"SHAP 值"或"AUC"，他们关心：
- **公平性**：模型是否对不同群体存在歧视？
- **隐私**：数据是否安全？是否存在隐私泄漏风险？
- **可复现性**：模型能否被独立验证？决策能否追溯？

**好的解释**（面向合规部门）：
> "我们用 SHAP 可解释性框架确保每个预测都有解释，并进行了公平性审计：
> - 模型在男性/女性申请人上的 AUC 差异为 0.11，我们正在收集更多女性申请人数据以减少偏差
> - 差异影响比 = 0.75，低于 80% 规则，我们已调整模型阈值以平衡通过率
> - 所有数据发布使用差分隐私（ε = 1.0），符合 GDPR 要求
> - 模型的随机种子、代码、数据来源均有记录，可独立复现"

小北听完点点头："我准备三个版本的解释——客户版、产品经理版、合规部门版。"

老潘微笑："对，不同受众需要不同的语言。好的数据分析师不仅能算出结果，还能把结果讲清楚。"

---

## StatLab 进度

到上周为止，StatLab 报告已经有了数据卡、描述统计、清洗日志、EDA 叙事、假设检验、不确定性量化、回归分析、分类评估和树模型。但小北的报告里，所有章节都是"技术报告"——只讲"模型有多准"，不讲"模型是否可信"。

老潘看完报告，问了小北一个新问题："**上周（Week 11）你学习了特征重要性——这是一个'全局'指标，告诉你'整体上哪些特征重要'。但你能回答'某个特定样本为什么被这样预测'吗？**"

小北想了想："呃……特征重要性告诉我'收入最重要'，但无法解释'对张三这个申请人，收入贡献了多少'。"

"对，"老潘说，"**本周（Week 12）我们要升级到 SHAP——这是一个'局部'指标，告诉你'对每个样本，每个特征贡献了多少'。**"

老潘在白板上画了一个对比表：

| 指标 | 回答的问题 | 粒度 | Week |
|------|-----------|------|------|
| **特征重要性** | 整体上哪些特征最重要？ | 全局（所有样本平均） | Week 11 |
| **SHAP 值** | 对这个样本，每个特征贡献了多少？ | 局部（样本特定） | Week 12 |

"更重要的是，"老潘继续，"你没有讨论模型的偏见、公平性、隐私风险。如果你的模型在某些群体上表现更差，或者放大了历史歧视，你能发现吗？"

小北脸色发白："我从来没想过这个问题……"

"这正好是本周'模型解释与伦理审查'派上用场的地方，"老潘说，"我们要在 report.md 中添加一个**模型解释与伦理审查章节**。"

这正好是本周"模型解释与伦理审查"派上用场的地方。我们要在 report.md 中添加一个**模型解释与伦理审查章节**。

### 在 StatLab 中添加模型解释与伦理审查

假设你的 StatLab 数据集有回归目标（如房价）或分类目标（如流失）。下面是一个完整的模型解释与伦理审查函数：

```python
# examples/12_statlab_xai.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import shap

def model_explanation_to_report(df, target, numeric_features, categorical_features,
                                sensitive_features, task='classification', output_path='report'):
    """
    对 StatLab 数据集进行模型解释与伦理审查，生成报告片段

    参数:
        df: 清洗后的数据
        target: 目标变量名
        numeric_features: 数值特征列表
        categorical_features: 类别特征列表
        sensitive_features: 敏感特征列表（如 gender, age_group，用于公平性评估）
        task: 'regression' 或 'classification'
        output_path: 图表输出路径
    """
    # 准备数据
    X = df[numeric_features + categorical_features + sensitive_features]
    y = df[target]

    # 编码类别特征
    X_encoded = pd.get_dummies(X, columns=categorical_features + sensitive_features, drop_first=True)

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # 训练随机森林
    if task == 'regression':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    model.fit(X_train, y_train)

    # ========== 1. SHAP 可解释性 ==========
    print("正在计算 SHAP 值...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # 全局解释：summary_plot
    plt.figure(figsize=(12, 8))
    if task == 'regression':
        shap.summary_plot(shap_values, X_test, show=False)
    else:
        shap.summary_plot(shap_values[1], X_test, show=False)  # 二分类用正类
    plt.title(f'SHAP 全局解释 - {target}')
    plt.savefig(f"{output_path}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 局部解释：force_plot（保存单个样本）
    sample_idx = 0
    if task == 'regression':
        sample_shap = shap_values[sample_idx]
        sample_base = explainer.expected_value
    else:
        sample_shap = shap_values[1][sample_idx]
        sample_base = explainer.expected_value[1]

    plt.figure(figsize=(16, 6))
    shap.force_plot(sample_base, sample_shap, X_test.iloc[sample_idx], show=False,
                    matplotlib=True, link='logit' if task == 'classification' else 'identity')
    plt.title(f'SHAP 局部解释（样本 {sample_idx}）')
    plt.savefig(f"{output_path}/shap_force_sample_{sample_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ========== 2. 公平性评估 ==========
    fairness_results = {}

    for sensitive in sensitive_features:
        # 找到编码后的列
        sensitive_cols = [c for c in X_test.columns if c.startswith(f"{sensitive}_")]

        if len(sensitive_cols) == 0:
            continue

        # 类别型敏感特征：用第一个编码列
        sensitive_col = sensitive_cols[0]
        group_a_mask = X_test[sensitive_col] == 1
        group_b_mask = X_test[sensitive_col] == 0

        if group_a_mask.sum() == 0 or group_b_mask.sum() == 0:
            continue

        # 计算不同群体的指标
        if task == 'classification':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            # AUC
            auc_a = roc_auc_score(y_test[group_a_mask], y_pred_proba[group_a_mask]) if group_a_mask.sum() > 1 else None
            auc_b = roc_auc_score(y_test[group_b_mask], y_pred_proba[group_b_mask]) if group_b_mask.sum() > 1 else None

            # 通过率（正类预测比例）
            pass_rate_a = y_pred[group_a_mask].mean()
            pass_rate_b = y_pred[group_b_mask].mean()

            # 差异影响比
            disparate_impact = pass_rate_a / pass_rate_b if pass_rate_b > 0 else None

            fairness_results[sensitive] = {
                'group_a': f"{sensitive}_1",
                'group_b': f"{sensitive}_0",
                'auc_a': auc_a,
                'auc_b': auc_b,
                'auc_diff': (auc_a - auc_b) if (auc_a and auc_b) else None,
                'pass_rate_a': pass_rate_a,
                'pass_rate_b': pass_rate_b,
                'disparate_impact': disparate_impact,
            }

    # ========== 3. 生成报告片段 ==========
    report = f"""
## 模型解释与伦理审查

### 研究问题
前几章我们建立了预测模型并评估了性能，但没有回答三个关键问题：
1. 模型是否可解释？能否向用户解释"为什么是这个预测"？
2. 模型是否存在偏见？是否对某些群体不公平？
3. 模型的局限性是什么？哪些场景下会失效？

本章使用 **SHAP（SHapley Additive exPlanations）** 进行可解释性分析，并对不同群体进行公平性评估。

### SHAP 可解释性

#### 全局解释：哪些特征最重要？

SHAP 全局解释（summary_plot）展示了所有特征的重要性与影响方向：

![SHAP 全局解释](shap_summary.png)

**解读**:
- 横轴：SHAP 值（正值表示提高预测值，负值表示降低预测值）
- 颜色：特征值（红色=高，蓝色=低）
- 最重要的特征: [根据图填写]

#### 局部解释：为什么是这个预测？

SHAP 局部解释（force_plot）展示了单个预测的"推理路径"：

![SHAP 局部解释](shap_force_sample_0.png)

**如何向客户解释**:
> "您的{'通过/拒绝' if task == 'classification' else '预测值'}主要因为 [1-2 个最显著特征]。"

### 公平性评估

我们检查了模型在不同敏感特征（{', '.join(sensitive_features)}）上的表现差异。

#### 性能差异表

"""

    # 添加公平性表格
    if fairness_results:
        report += "| 敏感特征 | 群体 A | 群体 B | AUC 差异 | 通过率 A | 通过率 B | 差异影响比 |\n"
        report += "|---------|--------|--------|----------|----------|----------|------------|\n"

        for sensitive, metrics in fairness_results.items():
            auc_diff_str = f"{metrics['auc_diff']:.3f}" if metrics['auc_diff'] else "N/A"
            di_str = f"{metrics['disparate_impact']:.3f}" if metrics['disparate_impact'] else "N/A"

            report += f"| {sensitive} | {metrics['group_a']} | {metrics['group_b']} | {auc_diff_str} | {metrics['pass_rate_a']:.3f} | {metrics['pass_rate_b']:.3f} | {di_str} |\n"

    report += f"""

**解读**:
"""

    # 添加解读
    for sensitive, metrics in fairness_results.items():
        if metrics['auc_diff'] and abs(metrics['auc_diff']) > 0.05:
            report += f"- ⚠️ **{sensitive}**: AUC 差异为 {metrics['auc_diff']:.3f}，说明模型在不同群体上的性能存在显著差异。\n"
        if metrics['disparate_impact'] and metrics['disparate_impact'] < 0.8:
            report += f"- ⚠️ **{sensitive}**: 差异影响比 = {metrics['disparate_impact']:.3f} < 0.8，不符合 80% 规则，可能存在法律风险。\n"

    report += """

### 偏见来源分析

模型偏见有三个主要来源：

1. **数据偏见**: 训练数据是否存在历史歧视？
2. **算法偏见**: 模型是否放大数据中的模式？
3. **代理变量**: 是否存在敏感属性的代理变量？

**常见偏见场景**:
- 信用评分：历史数据中，少数族裔被拒比例更高 → 模型学会"种族 → 违约"
- 招聘：历史数据中，男性被录用比例更高 → 模型学会"性别 → 录用"

### 结论边界：模型能回答什么，不能回答什么

**模型能回答的**:
- 基于历史数据预测 [填写具体目标]

**模型不能回答的**:
- 因果关系：提高某特征是否会改变结果？（模型只预测相关性，不回答因果）
- 特殊场景：训练数据中未见过的场景（如经济危机）
- 伦理判断：是否应该拒绝某个客户（这是业务决策，不是模型决策）

### 伦理审查清单

| 风险类别 | 检查项 | 状态 |
|---------|--------|------|
| **数据偏见** | 训练数据是否存在历史歧视？ | [ ] 需检查 |
| **算法偏见** | 模型是否放大数据中的模式？ | [ ] 需检查 |
| **代理变量** | 是否存在敏感属性的代理变量？ | [ ] 需检查 |
| **公平性指标** | 差异影响比是否 ≥ 0.8？ | [ ] 需检查 |
| **隐私风险** | 数据发布是否使用差分隐私？ | [ ] 需检查 |
| **可复现性** | 模型是否能被独立验证？ | [ ] ✅ |
| **结论边界** | 模型的局限性是否明确？ | [ ] ✅ |

"""

    return report

# 使用示例
if __name__ == "__main__":
    # 假设你的数据已经清洗完成
    df = pd.read_csv("data/clean_data.csv")

    # 分类任务（替换为你的 StatLab 数据集）
    report_xai = model_explanation_to_report(
        df=df,
        target="default",  # 替换为你的目标变量
        numeric_features=["income", "age", "debt_to_income"],  # 替换为你的数值特征
        categorical_features=["education"],  # 替换为你的类别特征
        sensitive_features=["gender"],  # 替换为你的敏感特征
        task="classification",
        output_path="report"
    )

    # 追加到 report.md
    with open("report/report.md", "a", encoding="utf-8") as f:
        f.write(report_xai)

    print("✅ 模型解释与伦理审查章节已添加到 report/report.md")
```

### 本周改进总结

| 改动项 | 上周状态 | 本周改进 |
|--------|---------|---------|
| 可解释性 | 特征重要性（全局） | SHAP 全局解释 + SHAP 局部解释（单个样本） |
| 偏见识别 | 无 | 不同群体（性别/地区/年龄）上的性能差异 |
| 公平性指标 | 无 | 差异影响比、平等机会、均等几率 |
| 隐私风险 | 无 | 差分隐私简介、隐私泄漏风险识别 |
| 伦理审查 | 无 | 伦理审查清单（偏差、公平性、隐私、可复现性） |
| 结论边界 | 无 | 明确说明"模型能回答什么、不能回答什么" |
| 向非技术读者解释 | 技术报告 | 用客户能懂的语言解释预测与风险 |

老潘看到这段改动会说什么？"这才是从'模型有多准'到'模型是否可信'的完整升级。你不仅学会了用 SHAP 打开黑盒、用公平性指标量化偏见、用伦理审查清单系统化评估风险，还掌握了向不同受众（客户、产品经理、合规部门）解释模型的技巧。"

---

## Git 本周要点

本周必会命令：
- `git status`: 查看工作区状态
- `git diff`: 查看具体改动内容
- `git add -A`: 添加所有改动
- `git commit -m "feat: add model explanation and ethics review"`
- `git log --oneline -n 5`

常见坑：

**混淆"重要"和"因果"**——SHAP 值显示收入最重要，不等于"提高收入会降低违约"。SHAP 是模型依赖，不是因果机制。解决方法：在解释中明确说明"这是模型的预测模式，不是因果结论"。

**只看全局，不看局部**——特征重要性说"收入最重要"，但无法解释"张三为什么被拒"。解决方法：用 SHAP force_plot 解释单个预测。

**忽视公平性**——模型整体 AUC = 0.85，但在女性群体上只有 0.78。解决方法：按敏感特征分组评估，计算差异影响比。

**删除敏感变量以为解决偏见**——即使删除性别，模型可能通过代理变量（职位、邮政编码）间接学到性别。解决方法：进行代理变量检测（相关性分析、SHAP 依赖图）。

**向非技术读者使用术语**——对客户说"SHAP 值"或"AUC"，对方听不懂。解决方法：准备不同版本的解释（客户版、产品经理版、合规部门版）。

**过度承诺模型能力**——说"模型准确率 85%"，但不说明局限。解决方法：明确说明"模型能回答什么、不能回答什么、哪些场景下会失效"。

Pull Request (PR)：
- Gitea 上也叫 Pull Request，流程等价 GitHub：push 分支 -> 开 PR -> review -> merge。

---

## 本周小结（供下周参考）

本周你从"模型有多准"走到了"模型是否可信"。你理解了为什么 Week 11 学的**特征重要性**不够——它只告诉你"整体上哪些特征重要"，但无法解释"对某个样本，每个特征贡献了多少"。

SHAP 给了你一种新的解释语言：用 Shapley 值量化每个特征对每个预测的贡献。你既能做全局解释（summary_plot，看哪些特征最重要），也能做局部解释（force_plot，看单个样本的推理路径）。这让黑盒模型（随机森林、梯度提升树）变成了可解释模型。

但强大的模型也带来新的责任。你学会了用**公平性指标**量化模型在不同群体上的表现差异：

**第 3 节（公平性指标上）**：
- **差异影响比**：不同群体的通过率是否相等（≥ 0.8 符合 80% 规则）
- **平等机会**：真正高风险者的识别率是否相等（召回率差异）

**第 4 节（公平性指标下）**：
- **均等几率**：召回率和假阳性率都相等（最强的公平性定义）
- **公平性权衡**：很难同时满足所有指标，需要根据业务场景选择

这三个指标回答不同的公平性问题：
- 差异影响关注"整体通过率是否相等"（招聘场景）
- 平等机会关注"真正高风险者是否被正确识别"（医疗场景）
- 均等几率关注"识别率和误判率都相等"（信用评分场景）

你还知道了偏见的三个来源：数据偏见（历史歧视）、算法偏见（模型放大数据中的模式）、代理变量（通过相关特征间接学到敏感属性）。删除敏感变量不够——模型可能通过邮政编码、职业类别等代理变量间接"学到"性别或种族。

你掌握了**伦理审查清单**——一个系统化的风险评估框架，覆盖偏差、公平性、隐私、可复现性四个维度。你理解了差分隐私的基本原理（通过添加噪声保护隐私，ε 是隐私预算），知道了隐私风险的来源（重识别、模型反演）。

最重要的是，你学会了**向非技术读者解释模型**——不是用"SHAP 值"或"AUC"这些术语，而是用客户能懂的语言说明"为什么被拒"和"模型的局限是什么"。

下周（Week 13），你要把这个思维升级为"因果推断"：从"相关性"到"因果性"，从"预测"到"干预"。本周的"特征重要性 ≠ 因果"会演化为下周的"因果图与干预效应"，SHAP 的"模型依赖"会演化为"反事实与 do-calculus"。公平性分析会引出"混杂变量与后门准则"——只有控制了混杂变量，才能得到因果结论。

---

## Definition of Done（学生自测清单）

- [ ] 我能理解全局解释（特征重要性）与局部解释（SHAP）的区别
- [ ] 我能用 SHAP summary_plot 解释"哪些特征最重要"
- [ ] 我能用 SHAP force_plot 解释"为什么这个样本被这样预测"
- [ ] 我能识别模型偏见的三个来源（数据、算法、代理变量）
- [ ] 我能理解什么是代理变量，以及如何检测代理变量
- [ ] 我能计算差异影响比（不同群体的通过率比）
- [ ] 我能计算平等机会（不同群体的召回率比）
- [ ] 我能理解均等几率（召回率和假阳性率都相等）
- [ ] 我能理解公平性指标之间的权衡（差异影响 vs 平等机会 vs 均等几率）
- [ ] 我能理解差分隐私的基本原理（ε 是隐私预算，添加噪声保护隐私）
- [ ] 我能使用伦理审查清单评估模型风险（偏差、公平性、隐私、可复现性）
- [ ] 我能在 StatLab 报告中添加模型解释章节（SHAP、公平性、伦理审查）
- [ ] 我能向非技术读者解释模型（客户、产品经理、合规部门）
- [ ] 我能识别 AI 生成的可解释性分析中的虚假解释
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解"从准确到可信"是建模思维的升级，模型可解释不是奢侈品，而是必需品
