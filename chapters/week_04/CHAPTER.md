# Week 04：从"看数字"到"讲故事"——EDA 叙事与假设清单

> "The greatest value of a picture is when it forces us to notice what we never expected to see."
> — John Tukey

2026 年的数据分析工具已经能让任何人在几秒钟内生成一堆图表：相关性热力图、散点图矩阵、分组统计表。AI 工具甚至能自动写出"数据洞察报告"：收入和年龄高度相关、A 组比 B 组均值高 20%、时间序列呈上升趋势。

但这些"自动洞察"有个致命问题：它们在告诉你"有什么"，却没告诉你"为什么"。更重要的是，它们没告诉你"接下来该问什么"。

John Tukey 在 1977 年提出的"探索性数据分析"（EDA）理念，在 AI 时代反而更重要了。因为 AI 可以比你更快地算出一个相关系数，却不会替你问：这个相关是真实的关联，还是混杂变量的假象？这个差异是稳定的模式，还是偶然的波动？

本周我们不追求"跑完所有统计检验"，而是学会"让数据讲故事"。从相关分析和分组比较开始，学会用多变量可视化的方式发现模式，然后把有趣的观察写成"可检验假设清单"——为 Week 06-08 的统计推断做准备。

---

## 前情提要

上一周你学会了"不要因为数据看起来有问题就直接动手"。你知道了缺失值有 MCAR、MAR、MNAR 三种机制，异常值可能是错误、发现或边界情况。你在 StatLab 报告中加入了清洗日志，让每一个数据决策都变得可审计。

小北拿着上周的报告问："我知道数据有什么问题了，也写清楚了怎么处理的。但现在我该怎么发现变量之间的关系？"

阿码接话："让 AI 生成所有散点图和相关性矩阵，不就行了吗？"

这正是本周要解决的问题：**EDA 不是"自动生成所有图表"，而是学会提出好问题，并把观察变成可检验的假设。**

---

## 学习目标

完成本周学习后，你将能够：
1. 计算并解释 Pearson、Spearman、Kendall 相关系数，知道何时用哪个
2. 用 groupby 和透视表做分组比较，发现组间差异
3. 用散点图矩阵和热力图探索多变量关系
4. 识别时间序列中的趋势和季节性模式
5. 将 EDA 观察转化为"可检验假设清单"，写入 StatLab 报告

---

<!--
贯穿案例：电商用户行为分析

案例演进路线：
- 第 1 节（相关分析入门）→ 计算用户属性与行为指标的相关性，发现潜在关联
- 第 2 节（分组比较）→ 按用户来源/地区分组，发现组间差异
- 第 3 节（多变量可视化）→ 用散点图矩阵和热力图发现隐藏模式
- 第 4 节（时间序列初步）→ 识别流量/销售的趋势与季节性
- 第 5 节（假设清单生成）→ 将观察转化为 3-5 个可检验假设

最终成果：读者拥有一份"假设清单"，每个假设都有数据支持和后续检验计划

数据集：引入电商用户行为数据（包含用户属性：年龄/地区/来源；行为指标：访问次数/停留时长/购买金额/复购率）

---

认知负荷预算：
- 本周新概念（4 个，预算上限 4 个）：
  1. 相关分析（Pearson/Spearman/Kendall）
  2. 分组比较（groupby/透视表）
  3. 多变量可视化（散点图矩阵/热力图）
  4. EDA 假设生成
- 结论：✅ 在预算内

回顾桥设计（至少 2 个，来自 week_02-03）：
- [分布形状]（来自 week_02）：在第 1 节，通过"相关系数对偏态的敏感性"再次使用
- [箱线图]（来自 week_02）：在第 2 节，通过"分组箱线图展示组间差异"再次使用
- [诚实可视化]（来自 week_02）：在第 3 节，通过"热力图的诚实性讨论"再次使用
- [异常值检测]（来自 week_03）：在第 1 节，通过"异常值对相关系数的影响"再次使用

AI 小专栏规划：
- 第 1 个侧栏（第 2 节之后）：
  - 主题："AI 能替你做 EDA 吗？"
  - 连接点：刚学完分组比较，讨论 AI 自动 EDA 的边界
  - 建议搜索词："AI automated EDA tools 2026", "pandas profiling alternatives 2026", "AutoEDA comparison 2026"

- 第 2 个侧栏（第 4 节之后）：
  - 主题："时间序列：预测还是理解？"
  - 连接点：刚学完时间序列初步，讨论 AI 预测的局限
  - 建议搜索词："time series forecasting vs understanding 2026", "AI time series limitations 2026", "interpretability time series models 2025"

角色出场规划：
- 小北（第 1 节）：把所有相关系数都当成"因果关系"，引出相关≠因果的讨论
- 阿码（第 3 节）：问"能不能让 AI 自动生成所有散点图？"，引出人工设计 EDA 的重要性
- 老潘（第 5 节）：看到"无计划的图表堆砌"后点评"在公司里这样写报告会被退回来"，引出假设清单的重要性

StatLab 本周推进：
- 上周状态：数据卡 + 描述统计 + 可视化 + 清洗日志
- 本周改进：相关分析 + 分组比较 + 假设清单（为 Week 06-08 做铺垫）
- 涉及的本周概念：相关分析、分组比较、多变量可视化、EDA 假设生成
- 建议示例文件：examples/04_statlab_eda.py（本周报告生成入口脚本）
-->

## 1. 两个变量一起看，会发生什么？

小北上周学会了"看一个变量的分布"：画直方图、算均值和中位数。这周她拿到一份电商数据，兴冲冲地算出"平均停留时长是 180 秒"，然后在报告里写："用户平均停留 3 分钟。"

老潘看了一眼，问："然后呢？停留时长和什么有关？年龄？购买金额？还是来源渠道？"

小北愣住了："呃……我还没算。我以为……把每个变量的统计都算一遍就行？"

老潘摇头："单变量统计只是'描述'。真正的问题是'关系'：谁和谁一起涨？谁和谁没关系？你报告里没写这些，决策者看了只会问'所以呢？'"

阿码插嘴："让 AI 把所有相关性都算一遍不就行了？"

小北眼睛一亮："对啊！AI 可以秒算所有相关系数！"

老潘笑了："那 AI 会给你一个 10×10 的相关矩阵。然后呢？你会怎么解读？"

阿码和小北面面相觑。

技术上是可行的。但这里有个更大的问题：**相关系数不是"按下按钮就能理解"的魔法，它需要你先搞清楚"我在问什么问题"。**

---

### 从单变量到双变量：散点图的语言

上周你学过**分布形状**：直方图告诉你一个变量的"长什么样"。现在我们要问：两个变量之间有"关系"吗？

最直观的方法是**散点图**（scatter plot）：X 轴放一个变量，Y 轴放另一个变量，每个点是一个观测值。

```python
# examples/01_correlation_intro.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 创建模拟数据：电商用户行为
np.random.seed(42)
n = 200

# 年龄与购买金额（正相关，但有噪声）
age = np.random.randint(18, 70, n)
purchase_amount = 50 + age * 2 + np.random.normal(0, 30, n)

# 停留时长与购买金额（弱正相关）
time_on_site = np.random.exponential(180, n)  # 平均 180 秒
purchase_amount_from_time = 20 + time_on_site * 0.5 + np.random.normal(0, 40, n)

df = pd.DataFrame({
    "age": age,
    "time_on_site": time_on_site,
    "purchase_amount": purchase_amount
})

print("数据概览：")
print(df.describe().round(1))
print()

# 散点图：年龄 vs 购买金额
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(df["age"], df["purchase_amount"], alpha=0.6)
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Purchase Amount ($)")
axes[0].set_title("Age vs Purchase Amount")

# 散点图：停留时长 vs 购买金额
axes[1].scatter(df["time_on_site"], df["purchase_amount"], alpha=0.6)
axes[1].set_xlabel("Time on Site (seconds)")
axes[1].set_ylabel("Purchase Amount ($)")
axes[1].set_title("Time on Site vs Purchase Amount")

plt.tight_layout()
plt.savefig("output/scatter_age_purchase.png", dpi=100)
print("图表已保存到 output/scatter_age_purchase.png")
```

运行后你会发现：左图（年龄 vs 购买金额）显示出明显的上升趋势——年龄越大，购买金额越高。右图（停留时长 vs 购买金额）也有上升趋势，但更松散。

![](images/scatter_age_purchase.png)
*图：散点图能直观显示两个变量之间的关系。左图显示强正相关，右图显示弱正相关*

小北说："哦！所以散点图能让我'看见'关系，而不是只看一个变量的均值。"

没错。散点图是 EDA 的第一语言：它告诉你两个变量是"一起涨"（正相关）、"一个涨一个跌"（负相关），还是"没啥关系"（无相关）。

### 相关系数：用一个数字总结关系

散点图很直观，但当你有 10 个变量时，两两画散点图会变成 45 张图（10×9/2）。这时候你需要一个**总结数字**：**相关系数**（correlation coefficient）。

最常见的是 **Pearson 相关系数**（Pearson's r），衡量线性相关的强度和方向：

```python
# Pearson 相关系数
corr_matrix = df[["age", "time_on_site", "purchase_amount"]].corr(method="pearson")
print("Pearson 相关系数矩阵：")
print(corr_matrix.round(2))
```

Pearson 相关系数的范围是 -1 到 1：
- **r = 1**：完全正相关（点在一条上升直线上）
- **r = -1**：完全负相关（点在一条下降直线上）
- **r = 0**：无线性相关（点散乱，无趋势）

阿码问："那 r = 0.5 算强还是弱？"

好问题。这取决于领域。在社会科学中，r = 0.5 已经算强相关；在物理学中，r < 0.9 可能都嫌弱。这就像评价"这个山高不高"——在平原地区，200 米就算"山"了；在青藏高原，200 米只是个土坡。

但 Pearson 相关系数有个前提：**它假设数据近似正态分布，且关系是线性的**。上周你学过**偏态**：如果数据有长尾，Pearson 相关系数会被极端值拉偏。

这时候你需要 **Spearman 相关系数**（Spearman's ρ）：它不测量"数值本身的相关"，而是测量"排名的相关"——对极端值更稳健。

```python
# Spearman 相关系数（对偏态稳健）
corr_spearman = df[["age", "time_on_site", "purchase_amount"]].corr(method="spearman")
print("\nSpearman 相关系数矩阵：")
print(corr_spearman.round(2))
```

还有一种 **Kendall 相关系数**（Kendall's τ），它更适用于小样本或有很多"并列排名"的数据。

小北问："那我用哪个？"

老潘的经验法则：
1. **先用散点图看一眼**：如果关系大致是线性的，用 Pearson
2. **如果有明显极端值或偏态**：用 Spearman
3. **如果是小样本（< 20）**：用 Kendall

但最重要的是：**相关系数只是一个总结数字，它不能替代散点图**。著名的"安斯库姆四重奏"（Anscombe's quartet）展示了四组数据，Pearson 相关系数都是 0.82，但散点图完全不同——有的是线性关系，有的是曲线，有的是离群点主导，有的是完美的拟合加一个离群点。

小北听完有点晕："所以相关系数会骗人？"

不，相关系数不会骗人，但它会**沉默**。r = 0.82 只告诉你"有线性关系"，不会告诉你"这是什么样的关系"。散点图才是那个"讲故事的人"。

### 异常值如何欺骗相关系数？

上周你学过**异常值检测**：异常值会把均值拉偏。同样，单个极端值也能把相关系数"拽"到一个误导性的数字。

```python
# 异常值对相关系数的影响
df_clean = df.copy()
df_with_outlier = df.copy()

# 添加一个极端值：年龄=20，购买金额=1000（异常高）
df_with_outlier = pd.concat([
    df_with_outlier,
    pd.DataFrame({"age": [20], "time_on_site": [150], "purchase_amount": [1000]})
], ignore_index=True)

# 计算相关系数
corr_clean = df_clean["age"].corr(df_clean["purchase_amount"])
corr_with_outlier = df_with_outlier["age"].corr(df_with_outlier["purchase_amount"])

print(f"不含异常值：Pearson r = {corr_clean:.2f}")
print(f"含异常值：Pearson r = {corr_with_outlier:.2f}")
```

运行后你会发现：加入一个极端值后，相关系数可能从 0.6 掉到 0.2——一个点就改变了结论。

小北瞪大了眼睛："一个点就能把 0.6 拖到 0.2？那相关系数也太脆弱了吧？"

老潘笑了："这不叫脆弱，这叫'诚实'。相关系数告诉你：'如果有这个点，关系是这样的；如果没有，关系是那样的'。它不会替你删点，但会提醒你'这个点影响很大'。"

老潘接着说："我当年刚入行时，做过一个分析，发现'网站停留时间和转化率负相关'。我差点就写结论说'优化网站要让用户快点走'。幸好画了散点图，才发现有个用户停留了 8 小时（估计是忘了关标签页），然后没购买。删掉这个点后，相关系数直接从 -0.3 变成 +0.4。"

这就是为什么老潘强调：**先看散点图，再看相关系数**。散点图能让你立刻发现"这个点是不是有问题"，而相关系数会把它藏在平均值里。

小北恍然大悟："所以相关系数是'总结'，散点图是'证据'？"

没错。散点图让你看见关系的形状，相关系数给它一个数字标签。

但你还没回答老潘的问题：停留时长和购买金额相关，但不同用户来源的差距有多大？这就需要**分组比较**。

---

## 2. 这个差异是真的吗？

阿码拿到数据后，立刻让 AI 生成了一张热力图，显示所有变量之间的相关性。他兴冲冲地跑来找你："看！年龄和购买金额的相关性是 0.65！"

老潘看了一眼，问："那不同来源渠道的用户呢？直接访问的和搜索进来的，有差异吗？"

阿码愣住了："呃……我没按渠道分组。"

这正是 EDA 的核心：**不分组看相关性，可能会错过组间的重要差异**。

---

### 分组比较：从"全局"到"局部"

上周你学过**箱线图**：它能显示中位数、四分位数和异常值。现在我们要做的是**分组箱线图**（grouped boxplot）：按类别变量（如来源渠道）分组，看数值变量（如购买金额）的分布差异。

```python
# examples/02_grouped_comparison.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 创建带分组信息的数据
np.random.seed(42)
n = 300

# 三个来源渠道：direct, search, social
source = np.random.choice(["direct", "search", "social"], n)

# 不同渠道的购买金额分布不同
purchase_by_source = {
    "direct": np.random.normal(100, 20, n),
    "search": np.random.normal(120, 25, n),
    "social": np.random.normal(80, 15, n)
}

purchase_amount = np.concatenate([
    purchase_by_source[s] for s in ["direct", "search", "social"]
])

# 重新分配（简化处理）
source_list = []
for s in ["direct", "search", "social"]:
    source_list.extend([s] * n)
source = np.array(source_list[:len(purchase_amount)])

df = pd.DataFrame({
    "source": source,
    "purchase_amount": purchase_amount
})

print("各渠道购买金额统计：")
print(df.groupby("source")["purchase_amount"].describe().round(1))
print()

# 分组箱线图
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="source", y="purchase_amount")
plt.xlabel("Source Channel")
plt.ylabel("Purchase Amount ($)")
plt.title("Purchase Amount by Source Channel")
plt.savefig("output/boxplot_grouped_source.png", dpi=100)
print("图表已保存到 output/boxplot_grouped_source.png")
```

运行后你会发现：search 渠道的购买金额（中位数约 120）明显高于 social 渠道（中位数约 80），direct 渠道在中间。

![](images/boxplot_grouped_source.png)
*图：分组箱线图能直观显示不同组之间的分布差异。search 渠道的购买金额显著高于 social 渠道*

小北说："哦！所以不分组看'平均购买金额'，会掩盖渠道之间的差异？"

没错。全局的"平均值"可能是三个渠道的"平均值"，但这对决策没有帮助——你想知道的是"哪个渠道更值得投入"，而不是"大家平均怎么样"。

### groupby 与透视表：pandas 的分组工具

可视化能让你"看见"差异，但**统计表**能让你"量化"差异。pandas 的 `groupby` 和 `pivot_table` 是 EDA 的核心工具。

```python
# groupby：按渠道分组统计
group_stats = df.groupby("source")["purchase_amount"].agg([
    ("count", "count"),
    ("mean", "mean"),
    ("median", "median"),
    ("std", "std")
]).round(1)

print("按渠道分组的统计：")
print(group_stats)
print()

# 透视表：多维度分组（假设再加一个维度：是否复购）
# 创建一个模拟的复购标签
df["is_returning"] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])

pivot = df.pivot_table(
    values="purchase_amount",
    index="source",
    columns="is_returning",
    aggfunc="mean"
).round(1)

print("透视表：按渠道和复购状态的交叉平均：")
print(pivot)
```

`groupby` 让你按一个维度分组统计，`pivot_table` 让你按两个维度交叉统计（如"渠道 × 是否复购"）。

老潘看到这些表，会说什么？"在公司里，我们不会只放数字。我们会问：这个差异是稳定的吗？还是偶然波动？"

这正是 Week 06-08 要学的内容：如何用统计检验判断"这个差异是真的，还是运气"。但现在，你已经学会了：**先用 EDA 发现差异，再问"这个差异值得检验吗"。**

### 分组陷阱：辛普森悖论

阿码可能会问："如果我按 A 分组有差异，按 B 分组也有差异，但合并起来差异消失了，怎么办？"

这不是假设场景，而是真实存在的**辛普森悖论**（Simpson's Paradox）：分组结论和整体结论相反。

举例：
- 搜索渠道的整体转化率比社交渠道低
- 但如果你按"新用户 vs 老用户"分组，搜索渠道在每个分组里都更高

为什么？因为搜索渠道的新用户比例更高，而新用户转化率本来就更低。**混杂变量**（用户类型）掩盖了真实的渠道效果。

小北皱着眉头："那我怎么知道有没有辛普森悖论？感觉要检查无穷无尽的分组组合……"

好问题。答案不是"算出一个统计量"，而是：**多维度分组看一眼**。不要只看全局的"渠道差异"，还要看"渠道 × 用户类型"的交叉。这不是要你穷尽所有组合——而是养成一个习惯：每次看到一个"有趣的差异"，先问自己"这个差异在所有分组里都成立吗？"

老潘补充："在公司里，我们吃过这个亏。有一年我们以为'移动端用户转化率更高'，投入了大量资源优化移动端体验。后来才发现，移动端用户大多是老客户，转化率本来就高。按'新老客户'分组后，移动端的优势消失了。"

这正是下一节要讲的：**多变量可视化**能帮你发现这些隐藏的模式。

> **AI 时代小专栏：AI 能替你做 EDA 吗？**
>
> 你刚学完分组比较，可能已经在想：AI 能不能自动生成所有 groupby 统计、所有散点图、所有箱线图？
>
> 2026 年确实有一批 AutoEDA 工具在做这件事：ydata-profiling（原 pandas-profiling）、Sweetviz、dtale、Lux 等。它们很快——几秒钟就能生成一份包含 50 张图表的"数据报告"。但这里有个关键区别：**AI 可以生成"所有图表"，但不会替你回答"哪些图表重要"。**
>
> 举例：AI 会自动生成"用户年龄的直方图"和"用户 ID 的直方图"。前者有意义，后者完全没意义（ID 是随机编号）。AI 不知道"哪个变量是业务相关的"，它只知道"这是个变量"。
>
> 同理，AI 会自动计算所有变量两两之间的相关性，但不会替你问：这个相关是真实的关联，还是混杂变量的假象？AI 会告诉你"A 渠道和 B 渠道的均值差 20 元"，但不会提醒你：A 渠道的用户群体本来就不同——年龄分布、地域分布可能完全不同。
>
> 更微妙的是，AI 工具对数据质量很敏感。2025-2026 年的研究显示，AutoEDA 工具在面对**缺失值多、数据类型混乱、异常值未处理**的数据时，生成的"洞察"可能会误导。工具不会告诉你"这个相关系数只基于 30 个非缺失值"，它会自信地报告 r = 0.85。
>
> 这正好呼应了本周前几节的内容：**你需要先理解数据的"健康状态"（Week 02-03），再让 AI 帮你"扫描"**。AI 是 X 光机，你是医生——X 光机能拍出所有片子，但不会替你诊断"这个阴影是炎症还是肿瘤"。
>
> AutoEDA 的正确用法是：**让 AI 生成候选图表，你负责筛选和解释**。AI 是扫描仪，你是编辑。
>
> 参考（访问日期：2026-02-15）：
> - https://www.findanomaly.ai/ai-data-analysis-tools-2026（2026 年 AI 数据分析工具综述）
> - https://towardsdatascience.com/comparing-five-most-popular-eda-tools-dccdef05aa4c/（五种最流行的 EDA 工具对比）
> - https://www.splunk.com/en_us/blog/learn/data-analysis-tools.html（数据分析工具指南）
> - https://www.montecarlodata.com/blog/data-ai-predictions/（数据与 AI 预测）

## 3. 怎么一眼看懂所有变量之间的关系？

现在你学会了：散点图看两个变量，分组箱线图看一个类别变量和一个数值变量。但如果你有 10 个变量，两两画散点图会变成 45 张图——这不可读。

这时候你需要**多变量可视化**：用一张图展示多个变量之间的关系。

---

### 散点图矩阵：把所有两两关系放一张图

**散点图矩阵**（Scatter Plot Matrix, SPLOM）是 EDA 的利器：它是一个 n×n 的网格，每个格子是两个变量的散点图（对角线是单变量分布）。

```python
# examples/03_multivariate_viz.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 创建多变量数据
np.random.seed(42)
n = 200

df = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "income": np.random.lognormal(10, 0.5, n),
    "purchase_amount": 50 + np.random.randint(18, 70, n) * 2 + np.random.normal(0, 20, n),
    "time_on_site": np.random.exponential(180, n)
})

# 选择要可视化的变量
vars_to_plot = ["age", "income", "purchase_amount", "time_on_site"]

# 散点图矩阵
sns.pairplot(df[vars_to_plot], diag_kind="hist", plot_kws={"alpha": 0.6})
plt.suptitle("Scatter Plot Matrix: All Variables", y=1.02)
plt.savefig("output/scatter_matrix_all.png", dpi=100, bbox_inches="tight")
print("图表已保存到 output/scatter_matrix_all.png")
```

运行后你会得到一个 4×4 的网格：对角线是每个变量的直方图（上周你学过的**分布形状**），非对角线是两两散点图。

![](images/scatter_matrix_all.png)
*图：散点图矩阵让你一眼看到所有变量之间的关系。对角线是单变量分布，非对角线是两两散点图*

小北说："哦！这张图能让我快速发现哪些变量之间有关系，哪些没啥关系。"

没错。但散点图矩阵也有问题：如果变量很多（比如 20 个），图表会变得太密、太复杂。这时候你需要**相关热力图**。

### 相关热力图：用颜色编码相关强度

**相关热力图**（Correlation Heatmap）是把相关矩阵可视化的方法：颜色越深/越暖（如红色），相关性越强；颜色越浅/越冷（如蓝色），相关性越弱。

```python
# 相关热力图
corr_matrix = df[vars_to_plot].corr(method="pearson")

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/heatmap_correlation.png", dpi=100)
print("图表已保存到 output/heatmap_correlation.png")
```

运行后你会发现：`income` 和 `purchase_amount` 的格子是深红色（强正相关），`age` 和 `time_on_site` 的格子是浅蓝色（弱负相关）。

![](images/heatmap_correlation.png)
*图：相关热力图用颜色编码相关强度。红色=正相关，蓝色=负相关，颜色越深相关越强*

阿码问："那热力图和散点图矩阵，我该用哪个？"

老潘的经验法则：
1. **探索阶段**：用散点图矩阵——能看见关系的形状（线性？非线性？有异常值？）
2. **汇报阶段**：用相关热力图——更简洁，适合放在报告中
3. **变量很多时**：只用热力图——散点图矩阵会太密

但两者都只是工具，真正的问题是：**你看到的相关，是真实的关联，还是混杂变量的假象？**

### 诚实可视化：热力图会不会误导？

上周你学过**诚实可视化**：Y 轴应该从 0 开始（除非有充分理由并说明）。热力图也有类似的陷阱：

1. **颜色映射的主观性**：有些人用红蓝，有些人用紫绿，读者可能误解颜色的含义
2. **相关系数的截断**：有些热力图把颜色范围设成 [-0.5, 0.5]，让 r = 0.6 看起来"非常强"——这和截断 Y 轴是同样的误导
3. **掩盖非线性关系**：热力图只显示线性相关（Pearson r），如果两个变量是 U 型关系（如"最佳温度"曲线），相关系数接近 0，热力图会显示"无相关"，但实际上有很强的非线性关系

阿码插嘴："等等，如果两个变量是 U 型关系，散点图矩阵能看出来吗？"

好问题。能！散点图矩阵中，U 型关系会像微笑或皱眉的形状——一眼就能看出"这不是线性相关，但确实有关系"。热力图却会把它标成浅蓝色（接近 0），让你误以为"没啥关系"。

小北笑了："所以热力图是'近视眼'？它只看得清线性关系，非线性的就模糊了？"

这个比喻不错。热力图擅长发现"线性相关"，但会错过"非线性模式"。老潘的建议是：**热力图是"扫描仪"，不是"结论"**。它帮你快速发现候选关系，但每个有趣的格子，都要回到散点图验证形状。

老潘补充："在公司里，我们有个不成文的规定：任何要在报告里展示的相关性，必须有人工审核过散点图。因为去年有个分析师被热力图'骗'过：他看到'用户登录次数和满意度'相关系数是 0.02，就写了'两者无关'。后来画散点图才发现是 U 型——登录太少（忘事儿）和太多（遇到问题）的用户满意度都低，中间的登录次数才是最佳区间。"

阿码听完若有所思："所以可视化也是会骗人的……那我要怎么知道'看到的模式是真的'，还是'只是我这样切片时刚好出现的'？"

好问题。这正是下一节要讲的：**时间维度**。如果一个模式只在本周数据中出现，过去 12 周都没有，那它可能只是随机波动。如果一个模式随时间稳定存在，那你才有底气说"这值得深入验证"。

多变量可视化能让你发现模式，但还有一个重要的维度没讨论：**时间**。如果你的数据有时间戳，你需要问：这个模式是稳定的，还是随时间变化的？

---

## 4. 这个模式是稳定的，还是随时间变化的？

阿码在分析电商数据时，发现一个有趣的模式：周末的平均购买金额比工作日高 20%。他兴冲冲地写了结论："周末用户更愿意花钱。"

老潘看了一眼，问："这个模式是稳定的吗？你看了过去 12 周的数据吗？"

阿码愣住了："呃……我只看了本周的数据。"

这正是 EDA 中常见的问题：**用一小段时间的数据下结论，可能会忽略长期趋势或季节性**。

---

### 时间序列初探：趋势与季节性

如果你的数据有时间戳（如日期、小时、周次），你需要问两个问题：

1. **趋势**（Trend）：长期来看，数据是上升、下降，还是持平？
2. **季节性**（Seasonality）：是否有固定的周期性模式（如周末高、工作日低；冬季高、夏季低）？

```python
# examples/04_time_series_intro.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建模拟时间序列数据（12 个月，每日数据）
np.random.seed(42)
dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")

# 基础趋势：线性上升
trend = np.linspace(100, 200, len(dates))

# 季节性：周末高、工作日低
seasonality = np.where(dates.dayofweek >= 5, 30, 0)  # 周六/周日加 30

# 随机噪声
noise = np.random.normal(0, 15, len(dates))

daily_sales = trend + seasonality + noise

df_ts = pd.DataFrame({
    "date": dates,
    "sales": daily_sales
})

print("时间序列数据概览：")
print(df_ts.head(10))
print()

# 按周聚合，减少噪声
df_ts["week"] = df_ts["date"].dt.isocalendar().week
weekly_sales = df_ts.groupby("week")["sales"].mean().reset_index()

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 上图：每日数据（有噪声）
axes[0].plot(df_ts["date"], df_ts["sales"], alpha=0.6, linewidth=0.8)
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Daily Sales ($)")
axes[0].set_title("Daily Sales (Noisy)")
axes[0].grid(True, alpha=0.3)

# 下图：周平均数据（更清晰）
axes[1].plot(weekly_sales["week"], weekly_sales["sales"], marker="o", linewidth=2)
axes[1].set_xlabel("Week")
axes[1].set_ylabel("Weekly Average Sales ($)")
axes[1].set_title("Weekly Average Sales (Trend + Seasonality Clearer)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/time_series_trend_seasonality.png", dpi=100)
print("图表已保存到 output/time_series_trend_seasonality.png")
```

运行后你会发现：上图（每日数据）有很多噪声，下图的周平均数据更清晰地显示了**趋势**（从 1 月到 12 月上升）和**季节性**（某些周有规律地波动）。

![](images/time_series_trend_seasonality.png)
*图：时间序列可视化。上图显示每日数据（噪声大），下图显示周平均数据（趋势和季节性更清晰）*

小北说："哦！所以时间序列分析的第一步是'去噪声'？"

对。聚合（如按周/月平均）能减少随机波动，让趋势和季节性更清晰。但要注意：**聚合会掩盖细节**，比如你无法再看到"周五晚上 vs 周六早上"的差异。

### 时间序列的陷阱：把相关当成因果

阿码发现一个模式："冰淇淋销量和溺水事故数高度相关（r = 0.85）！"

小北说："那我们应该限制冰淇淋销售来减少溺水？"

老潘笑了："这就是经典的'相关 ≠ 因果'例子。冰淇淋和溺水都和第三个变量有关：**温度**。夏天温度高，冰淇淋卖得好，游泳的人也多，溺水事故自然增加。"

这个"第三个变量"就是**混杂变量**（confounder）。时间序列分析中特别容易出现这种假象：两个变量都随时间上升（如 GDP 和房价），看起来相关，但实际上都是长期趋势的结果。

阿码问："那我怎么知道有没有混杂变量？"

好问题。答案不是"算出一个统计量"，而是：**画图、思考业务逻辑**。如果你怀疑温度是混杂变量，就把数据按温度分组，看"相同温度下，冰淇淋销量和溺水是否还相关"。

这正是下一节要讲的：**把 EDA 观察转化为假设，然后用统计检验验证**。

> **AI 时代小专栏：时间序列——预测还是理解？**
>
> 你刚学完了时间序列的基本概念：趋势、季节性、噪声。你可能在想：AI 能不能直接给我一个"预测模型"，自动预测下周/下月的销售额？
>
> 2026 年确实有一批 AI 工具在做时间序列预测：Prophet、基于深度学习的模型、各种 AutoML 平台。它们很快——几秒钟就能拟合一个模型并给出预测区间。但这里有个关键区别：**AI 可以帮你"预测"，但不会替你"理解预测背后的原因"。**
>
> 举例：AI 可以告诉你"下周销售额会上升 15%"，但不会回答"为什么"——是因为促销活动？季节性？还是竞争对手断货了？如果你不知道原因，预测再准也无法指导行动。
>
> 这正是 2025-2026 年时间序列研究的热点：**可解释性与准确性的权衡**。ACM Computing Surveys 2025 年的一篇综述指出，金融时间序列中的深度学习模型虽然预测精度高，但往往作为"黑箱"运行，限制了科学家理解潜在时间机制的能力。
>
> 更前沿的是 2026 年 2 月提出的"Agentic Forecasting"理念：超越以模型为中心的方法，转向更自适应的、基于代理的预测系统。这类系统不只输出一个数字，而是能告诉你"这个预测基于什么假设、哪些因素最敏感、在什么场景下会失效"。
>
> 阿码可能会问："那大模型呢？GPT-4 能不能直接分析时间序列？"
>
> 好问题。2026 年的研究确实在探索**基础模型与时间序列分析的交叉**，但目前的共识是：大模型擅长解释趋势和生成分析报告，但不擅长直接预测数值。真正的流程是：EDA 理解模式 → AI 建模预测 → AI 解释结果 → 你结合业务逻辑决策。
>
> 时间序列分析的正确路径是：**先用 EDA 理解趋势和季节性，再用 AI 建模预测，最后把预测结果与业务逻辑对照**。AI 是计算器，你是解释者。
>
> 参考（访问日期：2026-02-15）：
> - https://dl.acm.org/doi/full/10.1145/3729531（ACM：金融时间序列中的可解释 AI 综述，2025）
> - https://arxiv.org/html/2602.01776v1（arXiv：超越以模型为中心的预测——Agentic 时间序列预测，2026）
> - https://journal.hep.com.cn/fcs/EN/10.1007/s11704-025-50947-3（时间序列预测深度学习综述）
> - https://ai4ts.github.io/aaai2026（AAAI'26 Workshop：AI for Time Series Analysis）

## 5. 你的观察能变成可检验的问题吗？

小北在完成 EDA 后，兴冲冲地写了一份报告：包含 15 张图表、30 个统计数字。老潘看了一眼，问："你的结论是什么？"

小北愣住了："结论……就是这些图表？"

老潘摇头："图表不是结论。**假设才是结论。**"

---

### 从观察到假设：EDA 的出口

你前四节做了很多事：算相关系数、画分组箱线图、看散点图矩阵、分析时间序列。但这些只是"观察"，不是"结论"。

**结论**是你对观察的解释，而**假设**（hypothesis）是你准备检验的猜测。

举例：
- **观察**：搜索渠道的平均购买金额（120 元）高于社交渠道（80 元）
- **结论**：搜索渠道的用户更有购买力
- **假设**：如果随机抽取 100 个搜索渠道用户和 100 个社交渠道用户，搜索渠道的均值会显著高于社交渠道（Week 06 会用 t 检验验证）

**假设**必须具备两个特征：
1. **可检验**（testable）：能用数据验证对错
2. **具体**（specific）：明确"谁比谁高"、"趋势是什么方向"

小北问："那我怎么写出好的假设？"

### 假设清单模板

老潘的经验法则：每个假设都要回答三个问题：
1. **你观察到了什么？**（基于 EDA）
2. **你认为为什么会这样？**（业务逻辑）
3. **你打算怎么验证？**（统计方法）

```python
# examples/05_hypothesis_list.py
import pandas as pd

# 假设清单模板
hypothesis_list = []

def add_hypothesis(observation, explanation, test_method, priority="high"):
    """添加一个假设到清单"""
    hypothesis_list.append({
        "observation": observation,
        "explanation": explanation,
        "test_method": test_method,
        "priority": priority
    })

# 示例：基于前几节的观察
add_hypothesis(
    observation="搜索渠道的平均购买金额（$120）显著高于社交渠道（$80）",
    explanation="搜索渠道的用户有明确的购买意图，而社交渠道用户更多是浏览",
    test_method="双样本 t 检验（Week 06）",
    priority="high"
)

add_hypothesis(
    observation="年龄与购买金额呈正相关（Pearson r = 0.65）",
    explanation="年龄大的用户购买力更强，可能因为收入更高",
    test_method="回归分析（Week 09）",
    priority="medium"
)

add_hypothesis(
    observation="周末的平均购买金额比工作日高 20%",
    explanation="周末用户有更多时间浏览和购买",
    test_method="按星期分组的方差分析（Week 07）",
    priority="high"
)

add_hypothesis(
    observation="停留时长与购买金额的相关性较弱（Pearson r = 0.3）",
    explanation="停留时长可能包含浏览行为（如比较价格），不直接转化为购买",
    test_method="相关系数的置信区间（Week 08）",
    priority="low"
)

# 输出假设清单
hypothesis_df = pd.DataFrame(hypothesis_list)
print("可检验假设清单：")
print(hypothesis_df.to_string(index=False))
```

运行后你会得到一份**假设清单**：每个假设都有观察、解释和检验方法。

### 假设优先级：先验证哪个？

阿码问："我有 20 个观察，都要验证吗？"

老潘的经验是：**按"业务影响力"和"数据可检验性"排序**：
1. **高优先级**：直接影响决策的假设（如"哪个渠道更值得投入"）
2. **中优先级**：有趣但不紧急的假设（如"年龄与购买的关系"）
3. **低优先级**：探索性假设（如"停留时长是否真的不重要"）

不要试图验证所有假设。EDA 的目标不是"穷尽所有可能性"，而是**找到 3-5 个最值得深入的问题**。

小北松了一口气："我还以为要验证所有 20 个呢……那不是要做到明年？"

老潘笑了："我见过更夸张的。有个新来的分析师，EDA 后列了 50 个假设，然后问我'什么时候能全部验证完'。我告诉他：'你列的不是待办事项，是遗愿清单。我们的目标是'找到 3 个能改变决策的洞察'，不是'写一本关于这个数据集的百科全书'。'"

小北说："哦！所以 EDA 不是'终点'，而是'起点'？"

没错。EDA 是"发现问题的过程"，统计推断（Week 06-08）是"回答问题的工具"。你本周生成的假设清单，就是下周开始的"待办事项"。

现在你已经完成了 EDA 的全流程：从单变量分布到双变量相关，从分组比较多变量可视化，从时间序列到假设生成。让我们把这些知识整合进 StatLab 报告，让 EDA 变成一份可审计的"假设清单"。

## StatLab 进度

### 从"只看描述"到"提出可检验假设"

上周的 StatLab 报告包含了数据卡、描述统计、可视化和清洗日志。这些告诉你"数据长什么样、有什么问题"。这周我们要加入**相关分析、分组比较和假设清单**，让报告变成"可检验的故事"。

现在 `report.md` 会多出一个部分：
- 相关性分析：哪些变量相关？
- 分组比较：不同组之间有什么差异？
- 可检验假设清单：准备验证什么？

```python
# examples/04_statlab_eda.py
import pandas as pd
import seaborn as sns
from pathlib import Path

penguins = sns.load_dataset("penguins")

# 1. 相关性分析（数值变量）
numeric_cols = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
corr_matrix = penguins[numeric_cols].corr()

def generate_correlation_section(corr_df: pd.DataFrame) -> str:
    """生成相关性分析的 Markdown 片段"""
    md = ["## 相关性分析\n\n"]
    md.append("以下展示数值型变量两两之间的 Pearson 相关系数。\n\n")

    # 找出最强的相关
    corr_unstacked = corr_df.abs().unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1]  # 排除自相关
    max_corr = corr_unstacked.idxmax()
    max_corr_value = corr_unstacked.max()

    md.append(f"**最强相关**：{max_corr[0]} 与 {max_corr[1]}（r = {corr_df.loc[max_corr[0], max_corr[1]]:.2f}）\n\n")

    return "".join(md)

# 2. 分组比较（按物种分组）
def generate_group_comparison_section(df: pd.DataFrame) -> str:
    """生成分组比较的 Markdown 片段"""
    md = ["## 分组比较\n\n"]
    md.append("以下按物种（species）分组，比较数值型变量的分布。\n\n")

    # 计算各物种的描述统计
    group_stats = df.groupby("species")[numeric_cols].describe().T
    md.append("### 各物种的描述统计\n\n")
    md.append(group_stats.to_html())
    md.append("\n\n")

    return "".join(md)

# 3. 假设清单生成
class HypothesisList:
    """假设清单：记录可检验的假设"""

    def __init__(self):
        self.hypotheses = []

    def add(self, observation: str, explanation: str, test_method: str, priority: str = "medium"):
        """添加一个假设"""
        self.hypotheses.append({
            "observation": observation,
            "explanation": explanation,
            "test_method": test_method,
            "priority": priority
        })

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return pd.DataFrame(self.hypotheses)

def generate_hypothesis_section(hypotheses: HypothesisList) -> str:
    """生成假设清单的 Markdown 片段"""
    df = hypotheses.to_dataframe()
    if df.empty:
        return "## 可检验假设清单\n\n本数据集暂无待检验假设。\n"

    md = ["## 可检验假设清单\n\n"]
    md.append("以下假设基于 EDA 观察，将在后续章节用统计方法验证。\n\n")

    for priority in ["high", "medium", "low"]:
        priority_hyps = df[df["priority"] == priority]
        if not priority_hyps.empty:
            md.append(f"### {priority.capitalize()} 优先级\n\n")
            for _, row in priority_hyps.iterrows():
                md.append(f"**观察**：{row['observation']}\n\n")
                md.append(f"**解释**：{row['explanation']}\n\n")
                md.append(f"**检验方法**：{row['test_method']}\n\n")

    return "".join(md)

# 使用示例
hypotheses = HypothesisList()

# 添加假设（基于 EDA 观察）
hypotheses.add(
    observation="Gentoo 企鹅的 flipper_length_mm（翅长）均值显著高于其他物种",
    explanation="Gentoo 体型更大，翅长可能是物种特征",
    test_method="方差分析 ANOVA（Week 07）",
    priority="high"
)

hypotheses.add(
    observation="bill_length_mm 与 flipper_length_mm 呈正相关（r ≈ 0.65）",
    explanation="体型更大的企鹅嘴峰更长",
    test_method="相关系数的置信区间（Week 08）",
    priority="medium"
)

# 输出各部分
corr_md = generate_correlation_section(corr_matrix)
group_md = generate_group_comparison_section(penguins)
hyp_md = generate_hypothesis_section(hypotheses)

print("=== StatLab EDA 报告片段 ===\n")
print(corr_md)
print(group_md)
print(hyp_md)

# 写入文件
Path("output/eda_sections.md").parent.mkdir(parents=True, exist_ok=True)
Path("output/eda_sections.md").write_text(corr_md + group_md + hyp_md)
print("\n报告片段已保存到 output/eda_sections.md")
```

### 与本周知识的连接

**相关分析** → 我们计算了所有数值型变量两两之间的 Pearson 相关系数，并用 Markdown 格式化输出。

**分组比较** → 我们按物种分组计算描述统计，为 Week 07 的方差分析做准备。

**假设清单** → 我们把 EDA 观察转化为可检验假设，每个假设都有观察、解释和检验方法。

### 与上周的对比

| 上周 | 本周 |
|------|------|
| 数据卡 + 描述统计 + 可视化 + 清洗日志 | 以上全部 + **相关性分析 + 分组比较 + 假设清单** |
| 知道"数据长什么样、有什么问题" | 知道"变量之间有什么关系、差异是什么" |
| 无法回答"接下来要检验什么" | 生成"可检验假设清单"，为 Week 06-08 做铺垫 |

老潘看到这份新报告，会说："现在你不仅告诉别人'数据是什么'，还告诉别人'你接下来要验证什么'。这就是从'看数据'到'讲故事'的关键一步。"

阿码问："那如果我下周验证某个假设，结果不显著怎么办？"

老潘笑了："那也是发现。'没有显著差异'本身就是一个结论——说明你以为的'模式'可能只是随机波动。EDA 是'提出问题'，统计检验是'回答问题'。'回答'可能是'是的，这个差异真实存在'，也可能是'不，这只是噪声'。两者都有价值。"

### 下周预告

本周的报告包含了"可检验假设清单"。下周开始，我们将学习**概率与抽样分布**（Week 05），然后进入**统计推断**（Week 06-08）：用假设检验、置信区间和效应量来验证这些假设。

## Git 本周要点

本周必会命令：
- `git status`（查看未跟踪的新文件：EDA 脚本、假设清单文件）
- `git diff`（查看对 StatLab 报告生成脚本的修改）
- `git add -A`（添加所有变更，包括新生成的 EDA 输出）
- `git commit -m "draft: add EDA sections and hypothesis list"`（提交 EDA 分析）

常见坑：
- 假设清单写成纯文本（无法复现）。应该用代码生成，这样每次运行都能得到最新版本
- 相关性矩阵太大，不适合直接塞进报告。应该只展示"最强的相关"或"与目标变量相关的"

老潘的建议：把假设清单保存为独立的 Markdown 文件（`hypotheses.md`），然后在 `report.md` 中引用。这样假设清单可以单独维护，Week 06-08 验证时可以逐条打勾。

---

## 本周小结（供下周参考）

这周你学会了"让数据讲故事"。你知道了相关系数不是"按下按钮就能理解"的魔法，它需要你先用散点图看关系的形状，再用 Pearson（线性）、Spearman（排名）或 Kendall（小样本）量化强度。更重要的是，你学会了**异常值可以把相关系数"拽"到误导性的数字**——所以散点图永远是第一步。

你也学会了**分组比较**：不分组看"全局统计"可能会掩盖组间的重要差异。老潘的经验法则是："先看全局，再按类别分组"。如果结论反转了（辛普森悖论），说明有混杂变量在起作用。

多变量可视化（散点图矩阵、相关热力图）能让你快速扫描所有变量之间的关系，但它们只是"扫描仪"，不是"结论机"。**热力图告诉你"值得深入看哪对关系"，散点图告诉你"这对关系长什么样"**。

时间序列分析的第一步是"去噪声"：聚合数据能让趋势和季节性更清晰。但要记住：**时间序列中的"相关"特别容易出现假象**——两个变量都随时间上升（如 GDP 和房价），看起来相关，但实际上只是长期趋势的结果。

最重要的技能是**把观察转化为假设**。你本周生成的假设清单，就是下周开始的"待办事项"。老潘的总结很简短："图表不是结论。假设才是结论。"

下周，我们会从"探索数据"进入"量化不确定性"（Week 05-08 的统计推断阶段）。我们会先建立概率直觉，再学习假设检验、置信区间和效应量，把你本周生成的假设一个个验证。

## Definition of Done（学生自测清单）

- [ ] 我能解释 Pearson、Spearman、Kendall 相关系数的区别和适用场景
- [ ] 我能判断散点图中关系的形状（线性/非线性/无相关）
- [ ] 我能用 groupby 和透视表做分组比较
- [ ] 我能用散点图矩阵和相关热力图探索多变量关系
- [ ] 我能识别时间序列中的趋势和季节性
- [ ] 我能把 EDA 观察转化为"可检验假设清单"
- [ ] 我能在 StatLab 报告中加入相关分析和假设清单
- [ ] 我知道为什么不能"只看相关系数，不看散点图"
