# Week 02：一页分布报告 —— 从"一个均值"到"看见数据的形状"

> "Without data, you're just another person with an opinion."
> — W. Edwards Deming

你把一份用户数据丢给 AI，十秒钟后它吐回来一份报告："平均月消费 320 元，标准差 150 元，数据呈现右偏分布。"看起来很专业，对吧？

但这里面藏着一个你可能没注意到的问题：**AI 给你的只是一个"看起来对"的摘要，但它不会告诉你——320 元的均值被几个超级大户拽高了；中位数其实只有 180 元；所谓"右偏"意味着有 5% 的用户消费超过 1000 元，他们才是你真正需要单独研究的群体。**

GitHub 的数据显示，截至 2025 年 7 月，Copilot 已经有超过 2000 万用户——但同一时期的研究也发现，AI 生成的统计分析经常出现错误的假设检验、误导性的图表编码，甚至把"相关"直接写成"因果"。AI 确实能加速计算，但它不会替你思考：这个均值能代表典型用户吗？标准差大意味着什么？为什么有时候中位数比均值更诚实？

本周我们要做的，是"慢下来"——从只报一个均值，到真正看见数据的形状。描述统计不是"背公式"，而是建立数据直觉的第一步：数据大概落在什么范围？典型值在哪？波动有多大？有没有奇怪的地方？有了这些直觉，你才能判断 AI 给你的分析是否靠谱。

---

## 前情提要

上周你给自己的数据办了一张"身份证"（数据卡），知道每列是什么类型、能回答什么类型的问题。但这只是起点——老潘看过你的初稿后，只问了一句："**均值之后呢？**"

这是个好问题。如果你只给老板或客户报一个"平均留存率 40%"，他们可能会问："这是典型用户的情况吗？有没有极端情况？不同用户群体差异大吗？"如果你回答不上来，说明你还没有真正"看见"数据。

本周我们要做的，是把你对数据的认识从"一个数字"升级为"一张图 + 几个关键指标"。这不是炫技，而是诚实地回答："数据大概长什么样？典型值在哪？波动有多大？有没有奇怪的地方？"

AI 可以在几秒钟内为你生成一堆图表和统计量，但它不会替你回答"这些图说明什么"——这是你的工作，也是本周的核心。

---

## 学习目标

完成本周学习后，你将能够：
1. 选择合适的集中趋势指标（均值 vs 中位数 vs 众数），并解释为何选择它
2. 用标准差、IQR 等指标刻画数据的波动大小
3. 用直方图、密度图、箱线图展示数据的分布形状
4. 识别并避免常见的可视化误导（截断 Y 轴、面积误导等）
5. 为你的数据集生成一页"分布报告"（统计摘要 + 诚实图表）

---

<!--
贯穿案例：一页分布报告

本周贯穿案例是一个渐进式小项目：读者逐周把同一份数据的描述统计报告打磨成"可对外展示"的一页纸。

- 第 1 节：集中趋势 → 案例从"只知道一个均值"变成"理解何时该用均值/中位数/众数"
- 第 2 节：离散程度 → 案例从"只看中心"变成"同时理解中心和波动"
- 第 3 节：分布可视化 → 案例从"只有数字"变成"有图有真相"（直方图/密度图/箱线图）
- 第 4 节：可视化诚实性 → 案例从"随便画图"变成"能解释为什么这张图诚实"
- 第 5 节：一页报告 → 案例从"零散的分析"变成"一页可展示的报告"

最终成果：读者为选定的数据集生成一份"一页分布报告"（report.md 的第二版），包含：
- 数据卡（上周的）
- 核心指标的集中趋势和离散程度摘要
- 2-3 张"诚实"的分布图
- 每张图的简短解释（为什么选它、说明了什么）

认知负荷预算：
- 本周新概念（4 个，预算上限 4 个）：
  1. 集中趋势（central tendency）
  2. 离散程度（dispersion）
  3. 分布可视化（distribution plots）
  4. 可视化诚实性（honest visualization）
- 结论：✅ 在预算内

回顾桥设计（至少 2 个）：
- [数据类型]（来自 week_01）：在第 1 节，通过"数值型 vs 分类型"再次强调不同类型用不同指标
- [数据卡]（来自 week_01）：在第 5 节，把一页报告作为数据卡的"内容扩展"
- [DataFrame]（来自 week_01）：在代码示例中自然使用 pandas，不重复讲解基础

AI 小专栏规划：
合并为一个 AI 专栏放在章末（小结之前）：
- 主题：AI 时代如何审查自动生成的统计图表
- 内容要点：
  1. AI 生成图表的"一刀切"问题（默认用均值、截断 Y 轴）
  2. 误导性图表在 AI 输出中的普遍性
  3. 实用的审查清单
- 建议搜索词：
  - "AI automated data visualization limitations 2025 2026"
  - "misleading data visualization AI generated 2025"
  - "human-in-the-loop visualization 2026"

角色出场规划：
- 小北（第 1 节）：只看均值就下结论，被引导思考"这个均值代表性如何"
- 阿码（第 4 节）：问"AI 生成的图会不会错？"
- 老潘（第 3 节结尾）：追问"Y 轴从 0 开始了吗？"

StatLab 本周推进：
- 上周状态：只有数据卡（Data Card），描述了数据来源、字段、缺失情况
- 本周改进：在 report.md 中补充描述统计章节，包含均值/中位数、标准差/IQR，以及 2-3 张分布图
- 涉及的本周概念：集中趋势、离散程度、分布可视化、可视化诚实性
- 建议示例文件：examples/02_descriptive_report.py（生成描述统计报告的脚本）
-->

## 1. 别让平均数骗了你

小北拿到一份"用户月消费"数据，兴冲冲地算了个平均数，然后在汇报里写："**我们的用户平均月消费 320 元**"。

老潘看了一眼，只问了一句："那中位数呢？"

"中位数……"小北愣住了，"为什么要算中位数？均值不就够了吗？"

这是个极其真实的问题。小北算出的"320 元"在数学上是对的，但可能极具误导性——因为只要有一两个高消费的极端值，均值就会被拽着跑。比如这份数据里如果有 10 个普通用户（月消费 100-200 元）和 1 个超级大户（月消费 5000 元），均值会被拉到 600 多，完全失去"典型用户"的意义。

这就是**集中趋势**（central tendency）要回答的问题：数据的"典型水平"在哪里？但你有一个选择：是用均值、中位数，还是众数？

上周你已经知道，**数值型数据**才适合算均值。但数值型内部还有一个重要区别：这列数据有没有极端值？如果有，中位数往往比均值更"诚实"——中位数是排序后中间位置的数，不会被极端值拽着跑。

```python
import pandas as pd

# 读取数据
df = pd.read_csv("data/users.csv")

# 计算集中趋势的三个指标
mean_value = df["monthly_spend"].mean()
median_value = df["monthly_spend"].median()
mode_value = df["monthly_spend"].mode()[0]  # mode 返回 Series，取第一个

print(f"均值: {mean_value:.2f}")
print(f"中位数: {median_value:.2f}")
print(f"众数: {mode_value:.2f}")
```

小北跑了一下，发现均值是 320，中位数是 180，众数是 150。

"差这么多……"他盯着屏幕，"那我汇报的时候该用哪个？"

好问题。答案是：**取决于你想回答什么问题**。

- 均值适合给你一个整体规模感："所有用户的消费加起来，平均每人贡献了多少。"
- 中位数更像"典型个体"的位置："如果随机抓一个用户，他的消费大概率在中位数附近。"
- 众数适合分类型数据："哪个城市/哪个商品/哪个等级的用户最多？"

阿码在旁边突然冒出一个问题："等等，我试过把性别（编码成 0 和 1）算个均值，结果出来个 0.52。这能说明什么吗？"

这是个很"阿码式"的问题——他总是喜欢试探边界，偶尔还能踩到些有趣的坑。

你当然可以算，pandas 不会拦着你。但上周我们已经学过，性别是**分类型数据**，不是数值型。你算出来的"平均性别 0.52"在数学上没错，但在语义上是空的——它不代表任何真实的东西，就像问"这组苹果和橙子的平均水果是什么"一样。

所以这里有个"哦！"时刻：**工具不会替你思考语义**。pandas 会算你让它算的一切，但判断"这个指标有意义吗"——那是你的工作。

**先问数据类型，再选指标。** 数值型且没有极端值，用均值没问题；数值型但有长尾或异常点，用中位数更稳妥；分类型数据，用众数。

小北还是有点纠结："那我每次都要算三个吗？"

不是。你先画个分布图，看一眼数据长什么样——下一节我们就来聊这个。

---

> **AI 时代小专栏：数据质量是 AI 分析的底线**
>
> 你可能觉得：既然 AI 能自动生成描述统计、画分布图，那我学这些还有什么意义？
>
> 2025-2026 年的研究给出了一个清晰的答案：**数据质量决定了 AI 输出的质量**。MicroStrategy 的分析指出，2026 年 AI 成功的关键在于高质量数据——"垃圾进，垃圾出"这条老规则在 AI 时代不仅没过时，反而更致命。
>
> Duke University 图书馆 2026 年 1 月的文章指出：LLM 的幻觉问题在 2026 年依然存在，而幻觉的一个重要来源就是**稀疏、矛盾或低质量的数据**。当你让 AI 做描述统计时，它不会替你检查：缺失值是不是随机缺失？异常值是录入错误还是真实情况？分类型变量有没有被错误编码？
>
> 更有意思的是，GPTZero 在 2025 年底发现，ICLR 2026 的研究投稿中有超过 50 处幻觉引用——每处都被 3-5 位审稿人遗漏。这说明什么？**AI 生成的分析看起来"很专业"，但专业不等于正确**。
>
> 所以本周你学的"先看数据类型、再选指标""画分布图、看异常值"不是"过时的基本功"，而是 AI 时代的**质检流程**：在你把数据丢给 AI 之前，先搞清楚数据长什么样；在 AI 吐回分析结果之后，你有能力判断"这个均值靠谱吗""这张图有没有误导"。
>
> 记住：**AI 是你的副驾驶，但你是那个握方向盘的人**。
>
> 参考（访问日期：2026-02-15）：
> - [Why Data Quality is Key to AI Success in 2026](https://www.strategysoftware.com/blog/why-data-quality-is-key-to-ai-success-in-2026)
> - [It's 2026. Why Are LLMs Still Hallucinating?](https://blogs.library.duke.edu/blog/2026/01/05/its-2026-why-are-llms-still-hallucinating/)
> - [GPTZero Finds Over 50 Hallucinations in ICLR 2026 Submissions](https://gptzero.me/news/iclr-2026/)

---

## 2. 波动不只是"噪音"

小北盯着屏幕上那三个数字——均值 320、中位数 180、众数 150——觉得自己已经"看透"了这份数据。他满意地靠在椅背上，端起咖啡抿了一口，准备点击"发送"。

老潘恰好路过，瞥了一眼屏幕，问了一个让小北差点呛到的问题："波动多大？"

"波动……"小北想了想，在 Excel 里找到标准差公式，填进去。"标准差是 150。"

"标准差是个东西，"老潘说，"但 150 到底算大还是算小？"

这是个好问题。**离散程度**（dispersion）要回答的是：数据有多"散"？波动有多大？但和集中趋势一样，你也有选择：标准差、IQR（四分位距）、方差、极差——每个指标回答的是不同的问题。

先用一个例子感受一下：两组数据的均值都是 50，但第一组是 [48, 49, 50, 51, 52]，第二组是 [0, 25, 50, 75, 100]。均值相同，但第一组"稳定"，第二组"抖动"——这就是离散程度的差别。

最常用的两个指标是**标准差**（standard deviation）和**四分位距 IQR**（interquartile range）：

```python
import pandas as pd

df = pd.read_csv("data/users.csv")

# 标准差（Standard Deviation）
std = df["monthly_spend"].std()
print(f"标准差: {std:.2f}")

# 四分位距 IQR（Interquartile Range）
q1 = df["monthly_spend"].quantile(0.25)
q3 = df["monthly_spend"].quantile(0.75)
iqr = q3 - q1
print(f"IQR: {iqr:.2f} (Q1: {q1:.2f}, Q3: {q3:.2f})")
```

阿码看着输出，问了一个很犀利的问题："那标准差和 IQR，我该用哪个？"

答案是：**取决于你的数据有没有极端值**。

标准差对极端值非常敏感——有一个异常点，标准差就会被拽大。而 IQR 是中间 50% 数据的范围，极端值几乎不影响它。这里有一个"哦！"时刻：**如果你发现标准差远大于 IQR，说明数据里有极端值在"捣乱"**——这时候 IQR 更诚实。

小北追问："那如果我的数据是正态分布呢？"

好问题。正态分布时，标准差有简洁的解释：约 68% 的数据落在均值 ±1 个标准差内，95% 落在 ±2 个标准差内。但这个解释只在数据接近正态分布时才成立——如果分布有长尾，这个规则就不靠谱了。

上周我们学过，**分类型数据不能算标准差**——你可以说"性别"的众数是什么，但说"性别的标准差是 0.5"没有意义。分类型数据的离散程度要用熵或者基尼系数，那是更后面的内容了。

现在你同时有了"中心"和"波动"两个视角。但数字还是太抽象——你可能算出标准差是 150，但直觉上不知道"150 到底算大还是算小"。

小北想了想："那如果标准差是 0 呢？"

"那说明你们用户都是克隆人，消费一模一样。"老潘难得开了个玩笑，"快去画图吧。下一节，我们画图。"

---

## 3. 数据长什么样？先画分布

小北算出了均值、中位数、标准差，在报告里写了一堆数字。老潘看完只说了一句话："**画张图。**"

"画什么？"小北愣住了。

"画数据长什么样。"

这就是**分布可视化**（distribution plots）的核心：数字可以告诉你均值是多少、标准差是多大，但只有图能让你"看见"数据的形状。是单峰还是双峰？是对称还是偏斜？有没有异常值？这些信息藏在分布里，不在摘要数字里。

等等，"看见"数据是什么意思？你可能会说：我有一千行数据，我怎么"看见"？答案就是：**把分布画出来**。

最常用的三种分布图是：**直方图**（histogram）、**密度图**（density plot）、**箱线图**（boxplot）。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据（Week 01 你已经熟悉了 DataFrame）
df = pd.read_csv("data/users.csv")

# 直方图（Histogram）
plt.figure(figsize=(10, 4))
sns.histplot(df["monthly_spend"], kde=True, bins=30)
plt.title("月消费的分布（直方图）")
plt.xlabel("月消费（元）")
plt.ylabel("频数")
plt.show()

# 箱线图（Boxplot）
plt.figure(figsize=(6, 4))
sns.boxplot(y=df["monthly_spend"])
plt.title("月消费的分布（箱线图）")
plt.ylabel("月消费（元）")
plt.show()
```

阿码看着这两张图，突然问了一个问题："那我什么时候用直方图，什么时候用箱线图？"

好问题。它们回答的是不同的问题：

- **直方图**让你看到"整体形状"：单峰还是多峰？对称还是偏斜？数据集中在哪个区间？如果你加上 `kde=True`，还能看到一条平滑的密度曲线，帮助你判断"这像不像正态分布"。

- **箱线图**让你看到"异常值"：箱子的上下边是 Q1 和 Q3（中间 50% 数据的范围），中间的线是中位数，"须"延伸到 1.5 倍 IQR，超出须的点就是异常值。箱线图不适合判断"形状"，但非常适合发现极端点。

小北盯着屏幕上的箱线图，发现有 5 个点在须的外面。"这些是异常值吗？我要删掉它们吗？"

别急着删。这里有一个很多人都会踩的坑：**异常值不一定是错的**。异常值可能有三类：

1. **录入错误**：比如身高写成了 250 厘米——这是修数据
2. **极端但真实**：比如真的有用户月消费 10000 元——这是修解释
3. **某个群体的特征**：比如 VIP 用户的消费本来就高——这是修分组策略

Week 03 我们会专门聊异常值处理。现在你只需要记住：**箱线图把异常值摆到你面前，但怎么处理，是业务决策，不是统计操作**。

换句话说，箱线图不仅是"看见异常值"的工具，更是"理解异常值"的起点——这些点到底是怎么回事？是数据质量问题，还是你忽略的一个细分市场？

阿码追问："如果我有多个组（比如不同城市的用户），怎么画？"

用分组箱线图，这个很常见——而且经常能发现"意外的故事"：

```python
# 分组箱线图
plt.figure(figsize=(10, 4))
sns.boxplot(x=df["city"], y=df["monthly_spend"])
plt.title("不同城市的月消费分布")
plt.xlabel("城市")
plt.ylabel("月消费（元）")
plt.xticks(rotation=45)
plt.show()
```

这样你不仅能看到每个城市的"典型消费"（中位数），还能看到不同城市的波动和异常点差异。上周我们学过，**分类型数据不能算均值**，但可以用箱线图来比较不同类别组的分布——这就是可视化的力量。

小北满意地看着自己的图。老潘走过来，指着屏幕问了一句："**Y 轴从 0 开始了吗？**"

小北一愣："这……重要吗？"

"重要。"老潘说，"图不只是画出来，还要诚实地传达信息。下一节，我们聊聊图怎么说谎。"

---

## 4. 图会说谎，但你可以不

小北画了一张柱状图，展示"两个版本的转化率对比"：A 版本 5.2%，B 版本 5.8%。为了让差异看起来更明显，他把 Y 轴的起点设成了 5%——看起来 B 版本比 A 版本高了一大截，简直像是翻倍了似的。

老板看完很高兴："B 版本效果这么好？全面上线！"

小北心里美滋滋的，觉得今晚可以加个鸡腿。

老潘看了一眼，直接把图打回来了。小北的鸡腿梦碎了。

"在公司里，截断 Y 轴的图表会被打回来的。"

"为什么？"小北委屈，"数据没错啊……"

"数据没错，但你制造了错觉。"

这就是**可视化诚实性**（honest visualization）要解决的问题：图不仅是"画出来"，更要"诚实地传达信息"。常见的误导手法有几种，你需要识别并避免。

### 截断 Y 轴（Truncated Y-axis）

这是最常见的误导。比如上面那个例子：Y 轴从 5% 开始而不是 0%，让 0.6% 的差异看起来像 60%。

```python
# ❌ 误导图：Y 轴从 5% 开始
import matplotlib.pyplot as plt
versions = ["A", "B"]
rates = [5.2, 5.8]

plt.figure(figsize=(6, 4))
plt.bar(versions, rates)
plt.ylim(5, 6)  # 截断 Y 轴
plt.ylabel("转化率 (%)")
plt.title("误导图：Y 轴从 5% 开始")
plt.show()

# ✅ 诚实图：Y 轴从 0 开始
plt.figure(figsize=(6, 4))
plt.bar(versions, rates)
plt.ylim(0, max(rates) * 1.2)  # Y 轴从 0 开始
plt.ylabel("转化率 (%)")
plt.title("诚实图：Y 轴从 0 开始")
plt.show()
```

但注意——这个规则不是绝对的。如果你的数据本身就不接近 0（比如"人类身高 160-180cm"），Y 轴从 0 开始会压缩变化。关键原则是：**不要让截断制造错觉**。如果你必须截断，要在标题或注释中明确说明。

### 面积误导（Area Misleading）

饼图和气泡图经常掉进这个坑。如果你把"饼图的面积"当作数值本身，很容易犯错——因为人类对"面积"的感知不如对"长度"敏感。

```python
# ❌ 误导：用饼图展示差异很小的两个值
import matplotlib.pyplot as plt
sizes = [51, 49]
labels = ["A", "B"]

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct="%1.1f%%")
plt.title("误导图：差异只有 2%，但看起来很大")
plt.show()

# ✅ 诚实：用柱状图
plt.figure(figsize=(6, 4))
plt.bar(labels, sizes)
plt.ylim(0, max(sizes) * 1.2)
plt.ylabel("数值")
plt.title("诚实图：柱状图让差异更清晰")
plt.show()
```

老潘说得很直白："在公司里，你画的图会被很多人看到。如果你的图制造了错觉，结论就会被打折扣。没人会信任一个'会骗人的分析师'。"

阿码在旁边突然举手："那如果我让 AI 生成图表呢？**AI 生成的图会不会错？**"

这是个好问题。答案是：**会，而且它犯错的方式可能更隐蔽**——因为 AI 生成的图表通常很"漂亮"，你更容易放松警惕。

老潘点点头："AI 不会故意骗你，但它可能为了'让图表好看'而截断 Y 轴，或者默认用均值而不是中位数。你对你输出的所有内容负责——包括让 AI 帮你生成的部分。"

记住：**你对你画的图（或让 AI 画的图）负责**。

---

## 5. 一页报告：把数据的故事讲清楚

老潘走进会议室，只说了一句话："老板给你一分钟时间。"

小北慌了："一分钟？我光讲数据卡就要半分钟……"

"所以你得把故事压缩到一页纸上。"老潘在白板上画了个框，"30 秒让老板听懂：数据长什么样、典型值在哪、有什么需要注意的。"

这就是本周的贯穿案例：**一页分布报告**。你不再是"报一堆数字"，而是把本周学的东西——集中趋势、离散程度、分布可视化——整合成一张可对外展示的"数据快照"。

一页报告不是"偷懒"，而是聚焦。它回答三个问题：
1. **数据长什么样？**（核心指标摘要）
2. **分布是什么形状？**（2-3 张诚实的图）
3. **有什么需要注意的？**（极端值、长尾、分组差异）

上周你写的数据卡是"数据的地基"。这周你要在它基础上扩展一个"描述统计"小节——不是堆砌数字，而是选择最关键的指标和图表，让读者在 30 秒内建立对数据的直觉。

### 第一步：生成统计摘要

首先，我们需要一个函数来计算核心指标：

```python
# examples/05_one_page_report.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_descriptive_summary(df: pd.DataFrame) -> dict:
    """生成描述统计摘要（用于一页报告）"""
    numeric_cols = df.select_dtypes(include=["number"]).columns

    summary = {}
    for col in numeric_cols:
        summary[col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "q1": df[col].quantile(0.25),
            "q3": df[col].quantile(0.75),
            "iqr": df[col].quantile(0.75) - df[col].quantile(0.25),
        }
    return summary
```

这个函数遍历所有数值型列，计算均值、中位数、标准差和 IQR。注意我们没有计算所有可能的统计量——只选最关键的，这是"一页报告"的精髓。

### 第二步：生成可视化

接下来，为每个数值列生成两张图：直方图（看整体形状）和箱线图（看异常值）：

```python
def create_distribution_plots(df: pd.DataFrame, output_dir: str = "figures") -> list:
    """生成分布图（返回文件路径列表）"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns
    plot_paths = []

    for col in numeric_cols[:3]:  # 示例：只画前 3 个数值列
        # 直方图 + 密度曲线
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        plt.title(f"{col} 的分布")
        plt.xlabel(col)
        plt.ylabel("频数")
        path = f"{output_dir}/dist_{col}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(path)

        # 箱线图
        plt.figure(figsize=(4, 4))
        sns.boxplot(y=df[col])
        plt.title(f"{col} 的箱线图")
        plt.ylabel(col)
        path = f"{output_dir}/boxplot_{col}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        plot_paths.append(path)

    return plot_paths
```

这里的关键是 `bbox_inches="tight"`——它确保图例和标签不会被截断。每张图保存后立刻 `plt.close()`，避免内存泄漏。

### 第三步：写入报告

最后，把统计摘要和图表整合进 `report.md`：

```python
def append_to_report(df: pd.DataFrame, report_path: str = "report.md") -> None:
    """在数据卡后追加描述统计章节"""
    summary = generate_descriptive_summary(df)
    plot_paths = create_distribution_plots(df)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n## 描述统计\n\n")
        f.write("### 核心指标摘要\n\n")

        # 生成 Markdown 表格
        f.write("| 变量 | 均值 | 中位数 | 标准差 | IQR |\n")
        f.write("|------|------|--------|--------|-----|\n")
        for col, stats in summary.items():
            f.write(f"| {col} | {stats['mean']:.2f} | {stats['median']:.2f} | "
                   f"{stats['std']:.2f} | {stats['iqr']:.2f} |\n")

        f.write("\n### 分布图\n\n")
        for path in plot_paths:
            basename = path.replace("figures/", "").replace(".png", "")
            f.write(f"**{basename}**\n\n")
            f.write(f"![{basename}]({path})\n\n")

            # 添加解释：根据均值 vs 中位数判断偏斜
            col = basename.split("_")[1]
            if col in summary:
                mean_val = summary[col]["mean"]
                median_val = summary[col]["median"]
                if mean_val > median_val * 1.2:
                    f.write(f"说明：{col} 的均值 ({mean_val:.2f}) 显著高于中位数 ({median_val:.2f})，"
                           f"表明分布有右偏，可能存在高值异常点。\n\n")
                elif mean_val < median_val * 0.8:
                    f.write(f"说明：{col} 的均值 ({mean_val:.2f}) 显著低于中位数 ({median_val:.2f})，"
                           f"表明分布有左偏。\n\n")
                else:
                    f.write(f"说明：{col} 的均值 ({mean_val:.2f}) 和中位数 ({median_val:.2f}) 接近，"
                           f"分布相对对称。\n\n")

# 使用示例
if __name__ == "__main__":
    df = pd.read_csv("data/users.csv")
    append_to_report(df)
    print("✅ 描述统计章节已追加到 report.md")
```

小北看完这段代码，问了最关键的问题："我该怎么解释这些图？"

记住：**解释比图表本身更重要**。每张图配一段话，说明三件事：
1. **你为什么选这张图？**（直方图看整体形状，箱线图看异常点）
2. **这张图说明了什么？**（均值和中位数的对比、IQR 的大小、异常点的位置）
3. **有什么需要注意？**（右偏意味着高值拉高均值，异常点可能是录入错误或 VIP 用户）

老潘说："一页报告不是终点，是起点。它让你在 30 秒内建立对数据的直觉——后面所有推断和建模，都要基于这个直觉。"

下周我们会聊数据清洗。到时候你会发现，这周画的分布图不是装饰——它们在提醒你：**该修数据，还是该修解释**。

---

## StatLab 进度

上周你写了第一版 `report.md`——一份数据卡，说明数据是谁、从哪来、有哪些字段。这周的 StatLab 要在这个基础上往前走一步：**补充描述统计章节，生成 2-3 张"诚实"的分布图**。

具体来说，你要在 `report.md` 中新增一个"## 描述统计"小节，包含：

1. **核心指标摘要表**：对数值型变量列出均值、中位数、标准差；对分类型变量列出众数和频数
2. **分布图**：至少 2 张（如直方图、箱线图），每张图配一段解释（为什么选它、说明了什么）
3. **诚实性说明**：如果某个变量的分布有长尾或异常点，说明你是如何处理可视化的（如"Y 轴从 0 开始"）

```python
# examples/02_statlab_update.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def add_descriptive_section(data_path: str, report_path: str) -> None:
    """在 report.md 中追加描述统计章节"""
    df = pd.read_csv(data_path)

    # 生成描述统计摘要
    numeric_cols = df.select_dtypes(include=['number']).columns
    summary = df[numeric_cols].agg(['mean', 'median', 'std']).T

    # 生成分布图
    for col in numeric_cols[:3]:  # 示例：只画前 3 个数值列
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"{col} 的分布")
        plt.xlabel(col)
        plt.ylabel("频数")
        plt.savefig(f"figures/dist_{col}.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 追加到 report.md
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n## 描述统计\n\n")
        f.write("### 核心指标摘要\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n### 分布图\n\n")
        for col in numeric_cols[:3]:
            f.write(f"**{col} 的分布**\n\n")
            f.write(f"![{col} 分布](figures/dist_{col}.png)\n\n")
            f.write(f"说明：{col} 的均值是 {summary.loc[col, 'mean']:.2f}，")
            f.write(f"中位数是 {summary.loc[col, 'median']:.2f}。")

if __name__ == "__main__":
    add_descriptive_section("data/users.csv", "report.md")
    print("✅ 描述统计章节已追加到 report.md")
```

老潘看到这段代码会说："别急着写复杂代码。先确保你的图能回答一个问题——**数据长什么样？** 一张好的直方图或箱线图，比一百行数字更有说服力。"

---

> **AI 时代小专栏：AI 时代如何审查自动生成的统计图表**
>
> 2025-2026 年，AI 生成图表的能力越来越强——上传 CSV，几秒钟就能得到一堆直方图、散点图，甚至 AI 会"自动帮你选择最合适的图表类型"。但这恰恰是危险所在。
>
> LinkedIn 的一篇分析指出，AI 在自动化图表生成时常犯"一刀切"的错误——它可能默认用均值总结收入数据，但你很清楚收入有长尾，中位数才诚实；它可能为了"让图表好看"而截断 Y 轴，让微小差异显得巨大。BBC 2025 年的研究更发现：**45% 的 AI 查询会产生错误答案**，统计分析领域也不例外。
>
> 更隐蔽的问题是，AI 生成的图表通常很"漂亮"，你更容易放松警惕。Medium 上的分析指出，截断 Y 轴是"最常见的误导性可视化实践"之一——从 70% 而不是 0% 开始 Y 轴，会让微小的性能差异显得巨大。AI 不会告诉你"这张图有误导性"，它只是在执行指令。
>
> 所以本周学的"可视化诚实性"不是"过时的设计规范"，而是 AI 时代必备的"审查能力"。无论图是你手画的还是 AI 生成的，你都要能回答：这张图诚实吗？有没有制造错觉？
>
> **实用的审查清单：**
> - Y 轴从 0 开始了吗？（如果截断，有明确说明吗？）
> - 坐标轴标签和单位清晰吗？
> - 如果用面积编码（饼图、气泡图），差异真的需要用面积来表达吗？
> - 标题是否准确描述了图表传达的信息？
> - AI 默认用的指标（通常是均值）适合你的业务场景吗？
>
> 记住：**你对你输出的所有内容负责——包括让 AI 帮你生成的部分**。AI 可以加速你画图的过程，但只有你能回答"这张图要传达什么"。
>
> 参考（访问日期：2026-02-15）：
> - [How AI Can Create Misleading and Distorted Graphs (And How to Spot Them)](https://www.linkedin.com/pulse/how-ai-can-create-misleading-distorted-graphs-spot-them-eugene-woo-ajtle)
> - [BBC Finds That 45% of AI Queries Produce Erroneous Answers](https://joshbersin.com/2025/10/bbc-finds-that-45-of-ai-queries-produce-erroneous-answers/)
> - [The Most Common Misleading Errors in Data Visualization](https://medium.com/@hamzamlwh/the-most-common-misleading-errors-in-data-visualization-aa30bd1c89d4)

---

## Git 本周要点

本周必会命令：
- `git status`：查看工作区状态
- `git diff`：查看具体改动内容
- `git add -A`：添加所有改动
- `git commit -m "draft: add descriptive statistics"`：提交改动
- `git log --oneline -n 5`：查看最近 5 条提交

常见坑：
- 把生成的图片也提交进仓库：建议用 `.gitignore` 排除 `figures/`，或单独管理图片文件
- 只提交代码不更新报告：`report.md` 和代码一样重要
- 不写提交信息：`git commit -m "add figures"` 比 `git commit` 不写信息好一百倍

---

## 本周小结（供下周参考）

本周你做了三件事：不只是算均值，还学会了看中位数和众数——你知道了"平均值可能是骗人的"；不只是看中心，还学会了看波动——标准差和 IQR 回答的是不同的问题；不只是报数字，还学会了画"诚实"的图——Y 轴从 0 开始、不截断、不误导。

这就是描述统计的核心：**让数据自己说话，而不是让一个数字代替它**。

小北问："下周还要学什么？"

下周是数据清洗与预处理。你会发现：当你真正"看见"数据分布后，缺失值和异常值处理就不再是"机械操作"，而是有据可依的决策。这周画的分布图不是装饰——它们在提醒你：**该修数据，还是该修解释**。

---

## Definition of Done（学生自测清单）

- [ ] 我能解释为什么有时候中位数比均值更合适
- [ ] 我能计算标准差和 IQR，并解释它们的区别
- [ ] 我能画直方图、密度图、箱线图，并说明每张图的用途
- [ ] 我能识别至少两种常见的可视化误导（如截断 Y 轴、面积误导）
- [ ] 我为自己的数据集生成了一页"分布报告"（包含指标摘要 + 2-3 张图 + 解释）
- [ ] 我的报告中的图表都是"诚实"的（Y 轴从 0 开始、没有误导性编码）
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解"描述统计不是终点，是后续分析的起点"
