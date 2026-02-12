# Week 04：从"看数字"到"讲故事"——EDA 叙事与假设清单

> "EDA 不是正式分析的前奏，它就是分析的核心。"
> — John Tukey

2025 年底到 2026 年初，AI 数据分析工具迎来爆发式增长：ChatGPT 的代码解释器、Claude 的数据分析模式、以及各种"一键 EDA"工具，都能在几秒钟内生成相关性矩阵、分组统计和可视化图表。这听起来很美好，但隐藏着一个危险陷阱——AI 可以帮你算出相关系数，却不会替你问：这个相关在业务上意味着什么？有没有混杂变量？该不该分组看？当你拿到 AI 生成的"发现"时，你是否有能力判断哪些值得深挖、哪些只是噪音？本周我们要做的，正是培养这种判断力：从"跑代码出图"升级为"用数据讲故事"，从"计算相关性"升级为"提出可检验的假设"。

---

## 前情提要

过去三周，你已经为分析打下了坚实基础：Week 01 你学会了用数据卡记录数据来源和字段含义；Week 02 你掌握了描述统计和可视化，能用均值/中位数/箱线图描述数据特征；Week 03 你处理了缺失值和异常值，学会了记录每一个清洗决策。老潘看完你的清洗日志，点了点头："数据质量这块你过关了。现在的问题是——**你知道这些数据在说什么故事吗？**"

这不是修辞问题。一份干净的数据如果不会"说话"，就只是数字的堆砌。本周我们要学习 EDA（探索性数据分析）的核心技艺：**发现变量间的关系、比较不同群体的差异、并把这些发现转化为可检验的研究假设**。这是你从"数据处理员"升级为"数据分析师"的关键一步。

---

## 学习目标

完成本周学习后，你将能够：
1. 计算并解释 Pearson、Spearman 相关系数，理解"相关不等于因果"
2. 使用 groupby 和透视表进行分组比较，发现群体间的系统性差异
3. 识别多变量关系中的潜在混杂和交互效应
4. 从 EDA 发现中提炼出 3-5 个可检验的假设，写成假设清单
5. 将探索性发现整合进 StatLab 报告，形成完整的 EDA 叙事

---

<!--
贯穿案例：电商用户数据的探索性分析

本周贯穿案例是一个渐进式小项目：读者从清洗后的电商用户数据出发，逐节探索变量关系、发现群体差异、生成假设清单，最终产出一份"可检验假设列表"，为 Week 06-08 的统计推断做铺垫。

- 第 1 节：相关性分析 → 案例从"清洗后数据"变成"相关性矩阵 + 关键发现"
- 第 2 节：分组比较 → 案例从"整体统计"变成"分组洞察 + 群体画像"
- 第 3 节：多变量关系 → 案例从"两两关系"变成"变量网络 + 潜在混杂识别"
- 第 4 节：假设生成 → 案例从"零散发现"变成"可检验假设清单"
- 第 5 节：EDA 叙事 → 案例从"假设清单"变成"完整 EDA 报告章节"

最终成果：读者为选定的数据集生成一份"可检验假设列表"，包含：
- 3-5 个基于数据的假设（H0/H1 形式）
- 每个假设的数据支持（相关性/分组差异）
- 待检验的统计方法建议
- 潜在的混杂变量提醒

认知负荷预算：
- 本周新概念（4 个，预算上限 4 个）：
  1. 相关系数（correlation）- Pearson/Spearman
  2. 分组比较（group comparison）- groupby/透视表
  3. 多变量关系（multivariate relationships）- 混杂/交互
  4. 假设生成（hypothesis generation）- 可检验假设
- 结论：✅ 在预算内

回顾桥设计（至少 2 个）：
- [数据类型]（来自 week_01）：在第 1 节，通过"数值型 vs 分类型需要不同相关性方法"再次强调数据类型决定分析方法
- [描述统计]（来自 week_02）：在第 2 节，分组比较需要均值/标准差/中位数等描述统计作为基础
- [缺失值机制]（来自 week_03）：在第 1 节，相关性分析中的成对删除 vs 完全删除涉及缺失值处理
- [异常值]（来自 week_03）：在第 1 节，异常值对相关性的影响（Pearson 敏感 vs Spearman 稳健）
- [清洗决策日志]（来自 week_03）：在第 5 节，把假设清单作为清洗日志的"分析延伸"

AI 小专栏规划：

AI 小专栏 #1（放在第 1-2 节之间）：
- 主题：AI 辅助 EDA 工具的能力与局限
- 连接点：与第 1 节"相关性分析"呼应，讨论 AI 可以生成相关性矩阵但无法判断业务意义
- 建议搜索词：
  - "AI EDA tools 2025 2026 automated data analysis limitations"
  - "pandas-ai ChatGPT code interpreter data analysis pitfalls 2026"
  - "automated exploratory data analysis human oversight 2025"

AI 小专栏 #2（放在第 3-4 节之间）：
- 主题：AI 生成相关性分析的陷阱——伪相关与虚假关联
- 连接点：与第 3 节"多变量关系"呼应，讨论 AI 可能发现"冰淇淋销量与溺水事件"这类伪相关
- 建议搜索词：
  - "spurious correlation AI data analysis 2025 2026"
  - "AI hallucination statistics false correlations"
  - "automated correlation discovery causation fallacy 2026"

角色出场规划：
- 小北（第 1 节）：看到 AI 生成的相关性矩阵，直接下结论"消费金额和年龄正相关，说明年龄越大消费越多"，忽略了混杂变量（收入水平）
- 阿码（第 2、3 节）：追问"AI 能不能自动发现所有重要关联"、"分组比较能不能让 AI 自动选分组变量"
- 老潘（第 3、4、5 节）：强调"控制混杂变量"、"假设清单比代码更重要"、"EDA 报告要写给三个月后的自己看"

StatLab 本周推进：
- 上周状态：report.md 已有数据卡 + 描述统计 + 清洗日志，数据已完成清洗和预处理
- 本周改进：在 report.md 中添加"探索性数据分析"章节，包含相关性矩阵、分组比较结果、假设清单
- 涉及的本周概念：相关系数、分组比较、多变量关系、假设生成
- 建议示例文件：examples/04_eda_narrative.py（生成 EDA 报告章节的脚本）
-->

## 1. 这两个变量真的有关吗？

小北把清洗后的电商用户数据丢给 AI，几秒钟后拿到一张相关性矩阵热力图。他眼睛一亮，指着图上一个深色格子说："看！消费金额和年龄正相关，系数 0.45，说明年龄越大消费越多。"

老潘走过来，看了一眼："**等等，你确定是'年龄导致消费'，而不是'收入'在幕后操纵？**"

小北愣住了。他确实没多想——AI 给了他一个数字，他就直接解读成了因果关系。但相关性（correlation）只是告诉你两个变量"一起变化"，至于谁影响谁、有没有第三个变量在捣鬼，相关系数本身不会告诉你。

### Pearson vs Spearman：两种不同的"相关"

Week 01 我们学过数据类型的重要性，这里再次派上用场。还记得吗？你用 pandas DataFrame 加载数据时，首先要判断每列是数值型还是分类型——这个判断直接决定了你该用什么分析方法。Week 02 的描述统计（均值、标准差）也提醒你：极端值会扭曲统计量。就像你选择用均值还是中位数一样，这里你也要选择 Pearson 还是 Spearman——**数据类型和分布形状决定工具选择**。**数值型变量**的相关性有两种常用度量：

- **Pearson 相关系数**：衡量线性关系，对异常值敏感
- **Spearman 秩相关系数**：衡量单调关系，基于秩次，对异常值稳健

Week 03 你学会了识别异常值。如果数据里有极端值（比如几个年消费百万的 VIP 用户），Pearson 系数会被它们拽着跑，而 Spearman 不受影响——因为它看的是"排名"而不是原始数值。Week 02 的分布可视化（箱线图、直方图）也告诉你：在看相关性之前，先画个图看看数据长什么样——**可视化是检验假设的第一步**。

```python
import pandas as pd
import numpy as np
from scipy import stats

# 假设 df 是清洗后的电商用户数据
df = pd.DataFrame({
    'user_id': range(1, 101),
    'age': np.random.normal(35, 10, 100).astype(int).clip(18, 65),
    'monthly_income': np.random.lognormal(8.5, 0.5, 100).astype(int),
    'monthly_spend': np.random.lognormal(6.5, 0.8, 100).astype(int)
})

# 让消费和收入真正相关（模拟真实业务逻辑）
df['monthly_spend'] = (df['monthly_spend'] * 0.7 +
                       df['monthly_income'] * 0.3 * np.random.uniform(0.8, 1.2, 100)).astype(int)

# 计算 Pearson 相关系数
pearson_age_spend = df['age'].corr(df['monthly_spend'], method='pearson')
pearson_income_spend = df['monthly_income'].corr(df['monthly_spend'], method='pearson')

# 计算 Spearman 相关系数
spearman_age_spend = df['age'].corr(df['monthly_spend'], method='spearman')
spearman_income_spend = df['monthly_income'].corr(df['monthly_spend'], method='spearman')

print(f"Pearson: 年龄-消费 = {pearson_age_spend:.3f}, 收入-消费 = {pearson_income_spend:.3f}")
print(f"Spearman: 年龄-消费 = {spearman_age_spend:.3f}, 收入-消费 = {spearman_income_spend:.3f}")
```

运行后你可能会发现：收入与消费的相关性明显高于年龄与消费的相关性。这说明什么？**年龄和消费的相关性，可能只是"年龄大的人往往收入也高"这个链条的副产品。**

### 相关性矩阵：一眼看尽变量关系

当你有多个数值变量时，相关性矩阵是快速发现关联的利器：

```python
# 选择数值列计算相关性矩阵
numeric_cols = ['age', 'monthly_income', 'monthly_spend']
corr_matrix = df[numeric_cols].corr(method='pearson')

print("\nPearson 相关性矩阵：")
print(corr_matrix.round(3))

# 用热力图可视化（需要 matplotlib/seaborn）
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, square=True)
plt.title('变量相关性热力图 (Pearson)')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
plt.show()
```

小北盯着这张图，有点沮丧："那我刚才的结论是不是错了？"

"不完全是错，"老潘说，"你只是跳过了一步。**相关是起点，不是终点。** 看到 0.45 这个数字，你应该问：这是直接的因果，还是间接的链条？有没有我没看到的变量？"

### 缺失值的处理：成对删除 vs 完全删除

Week 03 你学会了处理缺失值。在相关性分析中，pandas 默认使用**成对删除**（pairwise deletion）——对于每一对变量，只删除这对变量中至少有一个缺失的行。这意味着不同相关系数可能基于不同的样本量计算。

```python
# 查看相关性计算使用的样本量
print(f"\n各列有效样本量：")
print(df[numeric_cols].count())

# 如果想确保所有相关系数基于相同样本，使用完全删除
df_complete = df[numeric_cols].dropna()
corr_matrix_complete = df_complete.corr(method='pearson')
print(f"\n完全删除后样本量：{len(df_complete)}")
```

如果你的数据缺失率很高，成对删除和完全删除可能给出截然不同的结果。这是你需要在报告里说明的——Week 03 的清洗决策日志在这里再次派上用场。

---

> **AI 时代小专栏：AI 辅助 EDA 工具的能力与局限**
>
> 2025 年的一项研究让不少人清醒过来：当研究者让 ChatGPT 在基础提示下完成统计推断任务时，它的准确率只有 32.5% 到 47.5%。即便使用高级提示，准确率也才达到 92.5%。这意味着什么？AI 可以帮你快速生成相关性矩阵、画出漂亮的热力图，但它可能选错方法、用错校正、甚至算错数字。
>
> 更隐蔽的问题是：AI 不会替你问"这个相关系数在业务上意味着什么"。它可以告诉你"年龄和消费相关系数是 0.45"，但不会提醒你"收入可能是混杂变量"。就像小北刚才的经历——AI 给了数字，但解读数字的责任还在你身上。
>
> 2025 年的另一项分析指出，使用 ChatGPT Code Interpreter 进行数据分析时，常见错误包括方法选择错误、编码错误（如 Yates 校正应用不当）、假设检验不完整等。而且，LLM 即使在统计计算不准确时也会生成输出——错误答案看起来自信且合理，这是最危险的地方。
>
> 所以本周你要学的，不是"怎么用 AI 生成相关性矩阵"，而是"**AI 生成之后，你怎么判断它靠不靠谱**"。这包括：检查样本量是否一致、判断相关系数大小是否被异常值扭曲、以及——最重要的——问自己"这个数字背后还有没有我没看到的故事"。
>
> 参考（访问日期：2026-02-11）：
> - [JMIR: ChatGPT Statistical Inference Accuracy](https://www.jmir.org/2025/1/e63550/)
> - [ChatGPT Code Interpreter Limitations](https://codingthesmartway.com/chatgpt-code-interpreter-the-limitations-you-must-be-aware-of/)
> - [Expert Beacon: ChatGPT Code Interpreter Analysis](https://expertbeacon.com/chatgpt-code-interpreter/)

---

## 2. 不同群体真的不一样吗？

相关性分析告诉你变量之间的关系，但它假设这种关系对所有样本都一样。现实中，**不同群体往往有不同的行为模式**。

阿码举手问："AI 能不能自动发现所有重要的分组差异？"

好问题。答案是：AI 可以帮你算，但"哪些分组值得看"需要你的业务判断。

### Groupby：分而治之的艺术

Week 02 你学会了描述统计——集中趋势（均值、中位数）和离散程度（标准差、IQR）。现在我们要把这些统计量**按组计算**，看看不同群体之间有没有系统性差异。还记得统计三问吗？描述（Description）只是第一问，现在我们要用分组比较来发现模式，为第二问（推断，Inference）做铺垫。

```python
# 假设我们有一个分类型变量：用户等级
df['user_level'] = pd.cut(df['monthly_spend'],
                          bins=[0, 500, 1500, 5000, float('inf')],
                          labels=['普通', '银卡', '金卡', '钻石'])

# 按用户等级分组，计算描述统计
group_stats = df.groupby('user_level').agg({
    'age': ['mean', 'std', 'median'],
    'monthly_income': ['mean', 'std', 'median'],
    'monthly_spend': ['mean', 'std', 'median', 'count']
}).round(2)

print("按用户等级的分组统计：")
print(group_stats)
```

这个表格回答了一个关键问题：**不同等级的用户，在年龄、收入、消费上有没有差异？** 如果钻石用户的平均年龄明显高于普通用户，这可能说明"年龄大的用户更有经济实力"——或者反过来，"平台需要更长时间才能培养出高价值用户"。

### 透视表：多维度的交叉分析

当有两个分组变量时，透视表（pivot table）比 groupby 更直观：

```python
# 假设还有一个分类型变量：城市级别
df['city_tier'] = np.random.choice(['一线', '二线', '三线'], size=len(df))

# 创建透视表：城市级别 × 用户等级 → 平均消费
pivot = pd.pivot_table(df,
                       values='monthly_spend',
                       index='city_tier',
                       columns='user_level',
                       aggfunc='mean',
                       margins=True,  # 添加总计行/列
                       margins_name='总计')

print("\n城市级别 × 用户等级的平均消费透视表：")
print(pivot.round(0))
```

透视表的魅力在于**同时看两个维度的交叉效应**。如果一线城市的钻石用户消费远高于三线城市钻石用户，这说明"城市级别"和"用户等级"有**交互效应**——不是简单叠加，而是互相放大。

阿码看着透视表，又忍不住问："那我发现了一线城市的金卡用户消费特别高，这说明什么？"

"这说明你有发现了一个**现象**，"老潘插话，"但现象不等于**解释**。"

可能的解释有很多：一线城市生活成本高，同样的"金卡"门槛对应更高的实际消费力；一线城市的金卡用户收入确实更高；或者平台在一线城市的运营策略更有效。EDA 的任务是发现现象、记录现象，并为后续检验铺路。至于哪个解释是对的，需要更严格的研究设计——这就是下一节"假设生成"要做的事。

---

## 3. 谁在幕后操纵这些关系？

小北又有了新的"发现"：女性用户的平均消费比男性高 15%。他准备把这个结论写进报告。

老潘看了一眼："**你控制收入了吗？**"

"什么？"

"如果女性用户平均收入本来就比男性高，那消费差异可能只是收入的副产品，而不是性别本身的影响。"

小北皱起眉头："你是说……我看到的'女性消费高'，其实是'收入高的人消费高'，而女性恰好收入高？"

"对。"老潘点头，"你看到的是性别和消费的**相关**，但真正的驱动力可能是收入。收入就是一个**混杂变量**（confounding variable）——它同时影响性别比例（在这个样本中）和消费水平，制造了一种'虚假关联'。"

小北倒吸一口凉气。他想起刚才第1节的教训：相关不等于因果。原来陷阱到处都是。

### 识别混杂：分层分析

要检验"性别→消费"的关系是否被"收入"混杂，最直接的方法是**分层分析**：在收入相同的群体中，比较男女消费差异。

老潘在白板上画了个例子："假设整体看女性消费比男性高 15%。但当你把人群按收入分成低、中、高三组，你会发现——在每一组内部，男女消费其实差不多。甚至在中收入组，男性消费还略高一点。"

"这怎么可能？"小北瞪大眼睛。

"这就是**辛普森悖论**（Simpson's Paradox）：整体趋势和分组趋势完全相反。就像你看到的——整体女性消费高 15%，但分层后差异消失甚至反转。"

```python
# 假设我们有一个性别变量
df['gender'] = np.random.choice(['男', '女'], size=len(df))

# 创建收入分层（把收入分成低、中、高三组）
df['income_tier'] = pd.qcut(df['monthly_income'], q=3, labels=['低收入', '中收入', '高收入'])

# 在每个收入层内，比较男女消费
stratified = df.groupby(['income_tier', 'gender'])['monthly_spend'].mean().unstack()

print("各收入层内的性别消费差异：")
print(stratified.round(0))

# 计算层内差异
stratified['差异(女-男)'] = stratified['女'] - stratified['男']
print("\n层内差异：")
print(stratified['差异(女-男)'].round(0))
```

如果整体上看"女 > 男"，但在每个收入层内差异变小甚至反转，说明收入是一个**混杂变量**。这就是辛普森悖论：整体趋势和分组趋势相反。

小北盯着屏幕上的数字，半天没说话。最后他喃喃道："幸好你提醒了我……不然我就要带着一个完全错误的结论去汇报了。"

"这就是 EDA 的价值，"老潘说，"不是算出数字就完事，而是不断追问'这个数字背后还有什么我没看到的故事'。"

### 多变量关系的可视化

当变量变多，可视化能帮你直观感受关系网络：

```python
# 散点图矩阵：一眼看尽多个变量两两关系
import seaborn as sns

# 选择关键变量
plot_cols = ['age', 'monthly_income', 'monthly_spend']

# 按性别分色的散点图矩阵
sns.pairplot(df[plot_cols + ['gender']], hue='gender', diag_kind='kde')
plt.suptitle('变量关系散点图矩阵（按性别分色）', y=1.02)
plt.savefig('pairplot.png', dpi=150)
plt.show()
```

这张图的价值在于**同时看多个关系**：对角线显示各变量的分布形状，上下三角显示两两变量的散点图，颜色区分不同群体。如果不同颜色的点呈现不同的分布模式，说明**群体间存在系统性差异**——这正是你需要在假设清单里记录的。

老潘强调："在真实项目中，**控制混杂变量**是因果推断的第一步。你不能只跑一个相关性就下结论，要问自己：还有什么变量可能解释这个关系？"

这周的 EDA 不是要做严格的因果分析，而是为后续检验**标记潜在的混杂变量**。当你写假设清单时，每个假设后面都应该有一列"潜在混杂"——提醒自己：这个关系可能是假的，需要进一步验证。

---

> **AI 时代小专栏：AI 生成相关性分析的陷阱——伪相关与虚假关联**
>
> 2025 年 3 月，北卡州立大学的研究团队发布了一项突破性成果：他们开发出无需识别虚假特征即可消除虚假相关的数据剪枝方法，并在 ICLR 2025 上发表。这听起来很技术，但它揭示的问题每个数据分析师都该警惕：AI 模型（包括大语言模型）经常捕捉到"看起来相关但实际无关"的模式。
>
> 经典的例子是"冰淇淋销量与溺水事件"——两者高度相关，但不是因为吃冰淇淋导致溺水，而是因为它们都受气温驱动。AI 可以发现这个相关性，甚至可能用很自信的语言描述它，但它不会自动问"有没有第三个变量在幕后操纵"。
>
> 2025 年的研究还总结了"Clever Hans 效应"的五大成因：数据伪影、缺乏因果监督、数据集不平衡、评估流程不足、以及可解释性工具缺乏。这些问题在自动化 EDA 工具中同样存在。Statsig 在 2025 年的分析中指出，相关性分析常见陷阱包括：相关≠因果、虚假相关、多重共线性掩盖真实驱动因素、以及隐藏变量造成误导性关联。
>
> 正如一项 2025 年的研究所说："Causal AI does not merely look at correlations, but it rather allows study of the causal relationships that are embedded in the data."（因果 AI 不只是看相关性，而是研究数据中嵌入的因果关系。）
>
> 这正是你本周学习混杂变量的意义：当 AI 给你一个"发现"时，你要有能力问——**这是真正的因果，还是又一个"冰淇淋与溺水"？**
>
> 参考（访问日期：2026-02-11）：
> - [NC State University: AI Spurious Correlations Research](https://news.ncsu.edu/2025/03/ai-spurious-correlations/)
> - [Frontiers in AI: Clever Hans Effect](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1692454/full)
> - [Statsig: AI Evaluation Correlation Techniques](https://www.statsig.com/perspectives/ai-eval-correlation-techniques)
> - [Risk Management Magazine: Causal AI](https://www.aifirm.it/wp-content/uploads/2025/04/RMM-2025-01-Excerpt-1.pdf)

---

## 4. 从发现到假设——你的检验清单

前三节你发现了很多"现象"：收入与消费正相关、钻石用户年龄偏大、一线城市金卡用户消费特别高、性别消费差异在控制收入后消失。但这些只是**描述性发现**。要让它们成为可检验的**研究假设**，你需要把它们写成 H0/H1 的形式。

小北试着写了第一个假设："收入和消费有关系。"

老潘看了一眼，摇头："太模糊了。什么关系？正相关还是负相关？多强的相关？可检验的假设必须具体。"

"那……'收入高的人消费高'？"

"好了一点，但还不够。统计假设有一套标准语言——H0 和 H1。"

### 假设的语法：H0 vs H1

统计假设检验有一套标准语言：

- **H0（零假设）**：默认状态，通常表示"没有效应/没有差异"
- **H1（备择假设）**：你想证明的，通常表示"有效应/有差异"

```python
# 假设清单结构示例
hypotheses = [
    {
        'id': 'H1',
        'description': '用户收入与月消费金额存在正相关关系',
        'H0': '收入与消费的 Pearson 相关系数 = 0',
        'H1': '收入与消费的 Pearson 相关系数 > 0',
        'data_support': 'Pearson r = 0.52, p < 0.001 (初步计算)',
        'proposed_test': 'Pearson 相关性检验',
        'confounders': '年龄、城市级别可能混杂',
        'priority': '高'
    },
    {
        'id': 'H2',
        'description': '不同城市级别用户的平均消费存在差异',
        'H0': '一线 = 二线 = 三线城市的平均消费',
        'H1': '至少有一组城市的平均消费不同',
        'data_support': '透视表显示一线城市均值 2850，三线 1920',
        'proposed_test': '单因素方差分析 (ANOVA)',
        'confounders': '收入分布、用户等级构成可能不同',
        'priority': '中'
    },
    {
        'id': 'H3',
        'description': '控制收入后，性别对消费无显著影响',
        'H0': '相同收入层内，男女消费差异 = 0',
        'H1': '相同收入层内，男女消费差异 ≠ 0',
        'data_support': '分层分析显示层内差异 < 5%',
        'proposed_test': '协方差分析 (ANCOVA) 或分层 t 检验',
        'confounders': '年龄、职业类型',
        'priority': '中'
    }
]

# 打印假设清单
for h in hypotheses:
    print(f"\n{'='*50}")
    print(f"假设 {h['id']} [{h['priority']}优先级]")
    print(f"{'='*50}")
    print(f"描述：{h['description']}")
    print(f"H0：{h['H0']}")
    print(f"H1：{h['H1']}")
    print(f"数据支持：{h['data_support']}")
    print(f"建议检验：{h['proposed_test']}")
    print(f"潜在混杂：{h['confounders']}")
```

### 好假设的标准

老潘看了一眼小北的假设清单，提了几点建议：

**可检验**：H0 和 H1 必须能用数据证伪。"用户喜欢我们的产品"不可检验；"用户满意度评分 > 4 分"可检验。

**有数据支持**：每个假设后面都应该有 EDA 发现作为依据，不能拍脑袋。

**考虑混杂**：标记潜在混杂变量，提醒自己结论可能是假的。

**区分优先级**：不是所有发现都值得检验。选择 3-5 个对业务最有价值的假设深入。

### 从假设到检验：Week 06-08 的预告

你现在写的只是"假设清单"，还没有真正做统计检验。Week 06-08 我们会学习：如何计算 p-value 和置信区间、如何控制多重比较问题、如何计算效应量、以及如何用 Bootstrap 估计不确定性。

这周的假设清单是为那几周准备的"待办事项"。**好的 EDA 不是一次性给出答案，而是提出好问题。**

但问题来了——你手里现在有一堆零散发现：几个相关系数、几张透视表、一份假设清单。如果直接把这些塞进报告，读者（包括三个月后的你自己）会看到一团乱麻。你需要一条**叙事线**，把这些碎片串成一个连贯的故事。

---

## 5. 让数据讲一个可信的故事

老潘说："**EDA 报告要写给三个月后的自己看。** 到时候你早就忘了当时为什么这么分析，所以报告本身要说清楚'我看到了什么、为什么会这样看、下一步该验证什么'。"

### EDA 叙事的结构

一份好的 EDA 报告章节应该像一条连贯的叙事线：从"我想知道什么"出发，经过"数据长什么样"，到"我发现了什么关系"，再到"这些关系可能受什么影响"，最后落在"我需要验证什么"。

```python
def generate_eda_narrative(df, corr_matrix, hypotheses, output_path='eda_narrative.md'):
    """生成 EDA 叙事章节，用于写入 report.md。"""

    # ========== 第 1 部分：章节标题与概述 ==========
    narrative = []
    narrative.append("# 探索性数据分析\n")
    narrative.append("> 本章记录从数据中发现的关系、差异与假设，为后续统计推断提供基础。\n")

    # ========== 第 2 部分：探索目标 ==========
    narrative.append("## 探索目标\n")
    narrative.append("本次 EDA 旨在回答以下问题：\n")
    narrative.append("1. 哪些用户特征与消费行为最相关？\n")
    narrative.append("2. 不同用户群体是否存在系统性消费差异？\n")
    narrative.append("3. 这些差异是否受其他变量（如收入）混杂？\n")
    narrative.append("4. 哪些发现值得进一步统计检验？\n")

    # ========== 第 3 部分：数据概览 ==========
    narrative.append("\n## 数据概览\n")
    narrative.append(f"- 样本量：{len(df)} 名用户\n")
    narrative.append(f"- 分析变量：年龄、月收入、月消费、用户等级、城市级别、性别\n")
    narrative.append(f"- 数据时间范围：2024年1月-12月（详见数据卡）\n")
    narrative.append(f"- 清洗说明：缺失值已按 Week 03 决策日志处理\n")

    # ========== 第 4 部分：双变量关系 ==========
    narrative.append("\n## 变量关系发现\n")
    narrative.append("### 相关性矩阵\n")
    narrative.append("| 变量对 | Pearson r | Spearman ρ | 解读 |\n")
    narrative.append("|--------|-----------|------------|------|\n")

    # 提取关键相关性
    income_spend_r = corr_matrix.loc['monthly_income', 'monthly_spend']
    age_spend_r = corr_matrix.loc['age', 'monthly_spend']

    narrative.append(f"| 收入-消费 | {income_spend_r:.3f} | {income_spend_r:.3f} | 中度正相关，收入是消费的重要预测因子 |\n")
    narrative.append(f"| 年龄-消费 | {age_spend_r:.3f} | {age_spend_r:.3f} | 弱相关，可能受收入混杂 |\n")

    narrative.append("\n**关键发现**：收入与消费的相关性（r≈0.5）明显高于年龄与消费的相关性（r≈0.2），" +
                    "提示年龄对消费的影响可能是通过收入间接实现的。\n")

    # ========== 第 5 部分：分组洞察 ==========
    narrative.append("\n## 分组比较发现\n")
    narrative.append("### 用户等级差异\n")
    narrative.append("钻石用户的平均消费（约 4500 元）是普通用户（约 800 元）的 5-6 倍，" +
                    "但钻石用户的平均年龄也偏大（42 岁 vs 31 岁），提示年龄-等级-消费三者存在关联网络。\n")

    narrative.append("\n### 城市级别差异\n")
    narrative.append("一线城市用户的平均消费高于二三线城市，但这一差异在控制收入后缩小，" +
                    "提示城市级别的影响可能主要由收入分布差异解释。\n")

    # ========== 第 6 部分：混杂变量识别 ==========
    narrative.append("\n## 潜在混杂变量\n")
    narrative.append("通过分层分析，识别以下潜在混杂：\n")
    narrative.append("- **收入**：可能混杂年龄-消费、性别-消费关系\n")
    narrative.append("- **年龄**：可能混杂用户等级-消费关系\n")
    narrative.append("- **城市级别**：可能混杂性别-消费关系（一线城市女性用户比例较高）\n")

    # ========== 第 7 部分：假设清单 ==========
    narrative.append("\n## 可检验假设清单\n")
    narrative.append("基于上述发现，提出以下假设供后续统计检验：\n")

    for h in hypotheses:
        narrative.append(f"\n### 假设 {h['id']} [{h['priority']}优先级]\n")
        narrative.append(f"**描述**：{h['description']}\n")
        narrative.append(f"**H0**：{h['H0']}\n")
        narrative.append(f"**H1**：{h['H1']}\n")
        narrative.append(f"**数据支持**：{h['data_support']}\n")
        narrative.append(f"**建议检验**：{h['proposed_test']}\n")
        narrative.append(f"**潜在混杂**：{h['confounders']}\n")

    # ========== 第 8 部分：局限与下一步 ==========
    narrative.append("\n## 局限与下一步\n")
    narrative.append("**数据局限**：\n")
    narrative.append("- 样本为平台现有用户，可能存在选择偏差\n")
    narrative.append("- 收入数据为用户自报，可能存在测量误差\n")
    narrative.append("- 横截面数据无法确定因果方向\n")

    narrative.append("\n**下一步工作**：\n")
    narrative.append("- Week 06-08 将对假设 H1-H3 进行统计检验\n")
    narrative.append("- 考虑收集纵向数据以支持因果推断\n")
    narrative.append("- 探索更多潜在混杂变量（如职业、教育水平）\n")

    # ========== 第 9 部分：写入文件 ==========
    content = '\n'.join(narrative)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"EDA 叙事已写入 {output_path}")
    return content

# 使用示例
# generate_eda_narrative(df, corr_matrix, hypotheses, 'report.md')
```

### 回顾桥：从数据卡到假设清单

Week 01 的数据卡告诉你"数据从哪来"，Week 03 的清洗日志告诉你"数据怎么变"，这周的假设清单告诉你"数据在说什么、还需要验证什么"。三者合在一起，才是完整的"数据说明书"。

老潘的总结："**数据卡是地基，清洗日志是施工记录，假设清单是设计图纸。** 没有图纸，你只是在盲目挖掘；没有施工记录，你不知道自己挖到了哪；没有地基，一切都是空中楼阁。"

---

## StatLab 进度

到目前为止，StatLab 的报告已经有了数据卡、描述统计和清洗日志。但这还不足以支撑后续的分析决策——你需要知道"哪些变量值得深入研究"、"哪些关系可能只是假象"。

本周的改进是把"探索性数据分析"章节写进 report.md。这不是简单的图表堆砌，而是一个**有逻辑的叙事**：从变量关系到分组差异，从混杂识别到假设生成。

```python
# examples/04_statlab_eda.py
import pandas as pd
import numpy as np
from datetime import datetime

def add_eda_section_to_report(report_path, df, hypotheses):
    """在 report.md 中添加 EDA 章节。"""

    # 计算相关性矩阵
    numeric_cols = ['age', 'monthly_income', 'monthly_spend']
    corr_matrix = df[numeric_cols].corr(method='pearson')

    # 生成 EDA 章节
    eda_section = f"""

## 探索性数据分析

> 本章记录从数据中发现的关系、差异与假设，为后续统计推断提供基础。
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

### 变量关系概览

| 变量对 | Pearson r | 强度 | 方向 |
|--------|-----------|------|------|
"""

    # 填充相关性表格
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            r = corr_matrix.loc[col1, col2]
            strength = "强" if abs(r) > 0.5 else "中" if abs(r) > 0.3 else "弱"
            direction = "正" if r > 0 else "负"
            eda_section += f"| {col1} - {col2} | {r:.3f} | {strength} | {direction} |\n"

    eda_section += """
### 关键发现

1. **收入-消费关系**：中度正相关（r≈0.5），是消费的最强预测因子
2. **年龄-消费关系**：弱相关（r≈0.2），可能受收入混杂
3. **年龄-收入关系**：中度正相关（r≈0.4），提示收入随年龄增长

### 可检验假设清单

"""

    # 添加假设清单
    for h in hypotheses:
        eda_section += f"""
**假设 {h['id']}** [{h['priority']}优先级]
- 描述：{h['description']}
- H0：{h['H0']}
- H1：{h['H1']}
- 建议检验：{h['proposed_test']}
- 潜在混杂：{h['confounders']}
"""

    eda_section += """
### 分析局限

- 横截面数据，无法确定因果方向
- 收入为用户自报，可能存在测量误差
- 未控制的潜在混杂变量：职业、教育水平、家庭状况

---
"""

    # 追加到报告
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(eda_section)

    print(f"EDA 章节已追加到 {report_path}")

# 使用示例
if __name__ == "__main__":
    # 假设 df 和 hypotheses 已经准备好
    # add_eda_section_to_report('report.md', df, hypotheses)
    pass
```

现在你的 report.md 有了四个层次：
1. **数据卡**：数据从哪来、字段什么意思
2. **描述统计**：数据长什么样、分布如何
3. **清洗日志**：数据怎么处理的、为什么这样处理
4. **EDA 叙事**：数据在说什么故事、还需要验证什么假设

老潘看到这份报告会说什么？"这才是一份能对外交付的分析。不仅告诉读者'我发现了什么'，还告诉他们'我是怎么发现的、还有哪些不确定性'。"

---

## Git 本周要点

本周必会命令：
- `git status`：查看工作区状态
- `git diff`：查看具体改动内容
- `git add -A`：添加所有改动
- `git commit -m "feat: add EDA narrative and hypothesis list"`：提交改动
- `git log --oneline -n 5`：查看最近 5 条提交

常见坑：
- 只保存图表不保存生成代码：图表无法复现，建议用脚本生成并版本控制
- 假设清单没有数据支持：每个假设都应该能追溯到具体的 EDA 发现
- 忘记记录分析决策：和清洗日志一样，EDA 决策也需要记录理由

---

## 本周小结（供下周参考）

本周你做了四件事：学会计算和解读 Pearson/Spearman 相关系数，理解"相关不等于因果"的核心原则；使用 groupby 和透视表进行分组比较，发现不同用户群体的系统性差异；识别多变量关系中的潜在混杂变量，学会用分层分析检验关联的稳健性；最重要的是，你把所有发现整合成了一份"可检验假设清单"——从"我看到了什么"升级为"我需要验证什么"。

这就是 EDA 的核心：**不是一次性给出答案，而是提出好问题、标记不确定性、为后续推断铺路**。这周的假设清单不是终点——它是 Week 06-08 统计检验的"待办事项"。

---

## Definition of Done（学生自测清单）

- [ ] 我能计算并解释 Pearson 和 Spearman 相关系数，知道何时用哪个
- [ ] 我能使用 groupby 和透视表进行分组比较，发现群体差异
- [ ] 我能识别多变量关系中的潜在混杂变量，理解"控制变量"的必要性
- [ ] 我能从 EDA 发现中提炼 3-5 个可检验的假设，写成 H0/H1 形式
- [ ] 我能解释"相关不等于因果"，并举出至少一个伪相关的例子
- [ ] 我为数据集生成了一份"可检验假设清单"，包含数据支持和检验方法建议
- [ ] 我在 report.md 中添加了"探索性数据分析"章节
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解"EDA 不是正式分析的前奏，它就是分析的核心"
