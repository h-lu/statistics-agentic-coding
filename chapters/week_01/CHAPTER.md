# Week 01：数据思维入门 —— 从"看数字"到"问问题"

> "In God we trust; all others must bring data."
> — W. Edwards Deming

2026 年的数据分析现场已经和几年前大不相同了：你上传一个 CSV 文件，Claude 或 ChatGPT 就能在几秒钟内给你生成图表、计算统计量，甚至直接"下结论"。这很诱人，也很危险。

因为 AI 可以比你更快地算出一个 p 值，却不会替你问：**样本是谁？缺失意味着什么？我们在比较多少个指标？** 换个切分方式结论还成立吗？GitHub Copilot 在 2025 年已经有超过两千万用户，AI 目前正在编写接近一半的代码——但代码只是工具，问题才是灵魂。

更现实的问题是：当 AI 一键生成 EDA（探索性数据分析）报告时，你如何判断它做对了什么？2025 年的学术研究已经在反复提醒：ChatGPT 在卡方检验、单因素方差分析等基础统计方法上的表现并不稳定。如果你不理解背后的逻辑，就容易被误导。

所以本周我们从一个反直觉的起点开始：**慢下来，先问问题，再找答案。** 我们要学的是"数据卡"——不是炫酷的预测模型，而是给数据办一张"身份证"，把数据的来源、边界、缺失情况写清楚。这是任何可复现分析的真正地基。

---

## 本章学习目标

完成本周学习后，你将能够：
1. 用"统计三问"（描述/推断/预测）明确你的分析目标
2. 识别数据类型并理解它们如何决定分析方法的选择
3. 用 pandas 加载和初步探索数据集
4. 为你的数据集创建一份"数据卡"（Data Card）
5. 理解"先问问题，再找答案"的统计思维模式

---

<!--
贯穿案例：数据卡生成器
- 第 1 节：统计三问 → 案例从"拿到数据不知道做什么"变成"明确我要回答什么类型的问题"
- 第 2 节：数据类型 → 案例从"把所有列当成数字"变成"识别每列的统计类型"
- 第 3 节：pandas 入门 → 案例从"无法读数据"变成"加载成功并看到前几行"
- 第 4 节：数据卡 → 案例从"零散的数据信息"变成"一份完整的数据说明文档"
最终成果：读者为选定的数据集写出一份可读、可审计的数据卡（report.md 的第一版）

认知负荷预算：
- 本周新概念（4 个，预算上限 4 个）：
  1. 统计三问（description/inference/prediction）
  2. 数据类型（数值型/分类型，连续/离散）
  3. pandas DataFrame
  4. 数据卡（Data Card）
- 结论：✅ 在预算内

回顾桥设计：
- 本周是第一周，无需回顾桥

AI 小专栏规划：
AI 小专栏 #1（放在第 2 节之后）：
- 主题：AI 可以一键生成 EDA，为什么还要学基础？
- 连接点：呼应第 2 节的"数据类型识别"——AI 可能自动推断类型，但它不知道你的业务语义
- 建议搜索词：
  - "AI automated EDA tools 2025 2026"
  - "ChatGPT data analysis limitations statistics"
  - "human-in-the-loop data analysis 2025"

AI 小专栏 #2（放在第 4 节之后）：
- 主题：Data Card —— AI 时代的"数据身份证"
- 连接点：呼应第 4 节的"数据卡"实践——数据集越来越多来自各种来源，没有数据卡的分析是不可审计的
- 建议搜索词：
  - "dataset data cards machine learning 2025 2026"
  - "datasheet for datasets best practices"
  - "data documentation standards AI"

角色出场规划：
- 小北（第 1 节）：拿到数据就急着算均值，被引导思考"你想回答什么问题"
- 阿码（第 3 节）：尝试用 AI 生成 pandas 代码，但因为没有理解 DataFrame 的索引机制而出错
- 老潘（第 4 节）：指出"在公司里，没有数据卡的数据分析报告是不会被 review 的"
-->

## 1. 你拿到数据，第一件事该做什么？

小北拿到一份电商用户数据，第一反应是："老师，怎么算均值？"

这是个极其常见、但极其危险的问题。不是"均值不好"，而是——**在算均值之前，你还没想清楚自己在做什么。**

试着反问三个问题：
1. 你想描述这批用户的特点吗？
2. 你想从这批用户推断整体用户的行为吗？
3. 你想预测某个新用户会不会购买吗？

这就是**统计三问**：描述（Description）、推断（Inference）、预测（Prediction）。它们看起来很像，但在做分析之前，你得先选定一个——因为不同的问题需要不同的方法。

描述问题最简单：你只想说明"这批数据本身长什么样"，算算均值、画个直方图就够了，结论只适用于手头这批数据，不往外推。推断问题就复杂一步：你想从这批样本说"总体大概怎么样"，这时候就需要抽样分布、置信区间、假设检验，任何结论都要带上不确定性。预测问题走得更远：你不是在描述已知，而是在对未来或未见样本下注，需要建模、评估、检验泛化能力，结论本质上是在说"新情况来了，大概率会怎样"。

小北的反应很诚实："我就是想看看用户平均花多少钱……"好，那这是一个描述问题。但如果你想基于这个结论说"下个月也会差不多"，那就变成预测了——你得问自己：数据来自哪个月？季节性呢？促销活动呢？

**记住：先问问题，再找答案。** 问题决定了答案的边界。

现在你已经知道自己要回答什么类型的问题了。下一步自然会冒出来：我要用哪些列来回答？这些列可以被怎样"对待"？

---

## 2. 这列数据到底是什么"类型"？

阿码举着手，问了一个让他自己都意外尴尬的问题："老师……如果我把性别（编码成 0 和 1）拿去算均值，会怎样？"

你当然可以算——pandas 不会拦着你。但算出来的"平均性别 0.52"有什么意义？

小北在旁边补刀："意思是……我们班一半男一半女？"

"不，"你说，"意思是阿码搞错了数据类型。"

全班笑了。但这是个真实的问题——pandas 只是个工具，它不替你思考业务语义。你把**分类型数据**当成了**数值型数据**，算出来的数字在数学上是对的，但在语义上是空的。

数据类型是统计分析的"交通规则"。如果你选错了车道，后面的分析都会走偏：

| 维度 | 数值型 | 分类型 |
|------|--------|--------|
| **可以做什么** | 加减乘除、求均值、做回归 | 计数、求比例、分组比较 |
| **不能做什么** | 对没有序关系的类别做加减 | 对没有序关系的类别求均值 |
| **例子** | 年龄、收入、身高 | 性别、城市、是否购买 |

但这还不够细。数值型内部还有一个重要区别：**连续 vs 离散**。

连续型可以取任意值——比如身高可以是 170.2 厘米，也可以是 170.23 厘米，取决于你测量的精度。离散型只能取某些值——比如你本月的外卖订单数可以是 5 单、6 单，但不可能是 5.3 单；再比如 APP 的评分是 1 到 5 星，你给不了 3.7 星（虽然你心里可能是这么想的）。

小北问："这不就是个数学定义吗？为什么要分这么细？"

因为在统计推断中，**方法的选择依赖于数据类型**。连续变量常常假设正态分布；离散变量可能用泊松分布；分类型变量用卡方检验。选错了，结论就不可靠。

更现实的问题是：pandas 不会替你思考。`df['gender'].mean()` 会给你一个数字，但不会告诉你"这可能不对"。2025 年的研究发现，ChatGPT 在处理数据类型识别时也会犯错——它可能把邮政编码当成连续数值，或者把评分当成纯类别。

所以：**在看数据之前，先问每一列的"类型是什么"**，并把它们写下来。这不是形式主义，是在为后面所有分析铺路。

想一个问题：如果这份数据丢给 AI，它会不会替你做这件事？好问题——我们正好来聊聊。

> **AI 时代小专栏 #1：AI 可以一键生成 EDA，为什么还要学基础？**

2025-2026 年，数据科学家用 ChatGPT、Claude 或专门的 AI 工具一键生成 EDA 报告已经很常见了。AI 可以在几秒钟内为你计算均值、标准差、相关性，甚至画出漂亮的图表。但学术研究和实践都发现了一个问题：AI 不会替你思考这份数据的"业务语义"。

举例来说：
- AI 可能把"邮政编码"识别成连续数值，并计算均值和标准差——但这些数字没有数学意义
- AI 可能自动对缺失值做删除或填充，但不会告诉你缺失背后的机制是什么（MCAR/MAR/MNAR）
- 2025 年的多项研究显示，ChatGPT 在卡方检验、方差分析等基础统计方法上的表现并不稳定

更重要的是，"统计直觉"是你判断 AI 输出是否可靠的前提。如果你不理解数据类型、分布假设、效应量的含义，就无法审查 AI 给出的结论。

所以本周学的东西——统计三问、数据类型识别——不是"过时的手工工作"，而是你在 AI 时代必备的"审查能力"。AI 可以加速你的计算，但只有你能回答"这个问题是否有意义"。

**参考（访问日期：2026-02-11）：**
- [Top 7 AI Tools in 2026 - Medium](https://medium.com/ai-analytics-diaries/top-7-ai-tools-in-2026-9784e67de643)
- [How Data Scientists Are Using ChatGPT to Automate EDA Workflows](https://ai.plainenglish.io/how-data-scientists-are-using-chatgpt-to-automate-eda-workflows-8bd9258256c0)
- [Agentic EDA with AI Foundry: Automating Exploratory Data Analysis](https://pub.towardsai.net/agentic-eda-automating-exploratory-data-analysis-for-data-science-workflow-b874dec24d7a)

现在你知道每列的数据类型了，下一步自然就是：怎么在 Python 里和这些数据打交道？

---

## 3. 用 pandas 说话：DataFrame 是什么？

现在我们终于可以写代码了。但在那之前，先承认一个事实：**用 Excel 打开超过 10 万行的数据文件，基本等于自找麻烦**。

小北试过一次。他打开一个 50 万行的 CSV，等了三分钟，Excel 直接弹窗："文件未完全加载"。那一刻他意识到：该换工具了。

pandas 的 DataFrame 是 Python 数据分析的"通用语言"。你可以把它想象成一张在内存中的表格——可以行索引、列索引、筛选、分组、合并，但比 Excel 快得多，也更可复现。

```python
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("data/users.csv")

# 查看数据的基本形状（行数、列数）
print(df.shape)

# 查看前几行
print(df.head())

# 查看列名和数据类型
print(df.dtypes)

# 查看基本的统计摘要
print(df.describe())
```

阿码看完 `df.head()` 的输出，想起之前让 AI 生成过一些 pandas 代码，就试着写了 `df[50]`，想直接看看第 50 行长什么样。结果报错了——`KeyError: 50`。

他盯着屏幕有点懵："但 AI 的代码里明明有 `df['列名']` 这种写法啊……"

老潘在旁边看了一眼："你刚才选的是列，不是行。"

"啊？"阿码盯着屏幕。

这就是 DataFrame 的一个关键概念：**索引**。`df[50]` 会被 pandas 理解为"选择列名为 50 的列"，而不是第 50 行。正确的写法是：

```python
# 用 iloc 按位置选择（从 0 开始）
df.iloc[50]  # 第 51 行
df.iloc[0:5]  # 前 5 行

# 用 loc 按标签选择
df.loc[0]  # 索引标签为 0 的行
```

阿码问："AI 给我生成的代码直接用了 `df[列名]`，为什么那样是对的？"

好问题。`df['列名']` 是选择列的标准写法，这和行选择不同。DataFrame 的设计逻辑是：
- `df['列名']` → 选择列
- `df.loc[行标签]` 或 `df.iloc[行位置]` → 选择行
- `df.loc[行标签, '列名']` → 同时选择行和列

这个区别看似琐碎，但当你让 AI 生成代码时，理解它可以帮你更快地定位错误。GitHub Copilot 现在有超过两千万用户，但它不会替你理解为什么 `iloc` 和 `loc` 的区别很重要——特别是在你做数据清洗、重置索引之后。

小北有点崩溃："那我每次都要记这些吗？我连下周自己会不会记得都怀疑……"

老潘笑了，说了一句很哲学的话："忘了很正常。所以我们要写文档。"

这话说得有点玄，但小北很快就会理解——当他三天后打开自己写的代码，看着 `df['user_type']` 这列，完全不记得它是什么意思的时候。

---

## 4. 数据卡：给你的数据办张"身份证"

三个月后，你打开自己写过的代码，看着 `df['user_type']` 这列，完全不记得它是什么意思了。是"用户等级"？是"新老用户标识"？还是"购买渠道"？

这不是假设——这是真实发生的事情。数据分析最大的敌人不是复杂的模型，而是**遗忘**。

**数据卡**（Data Card）就是给数据办一张"身份证"。它是一份简短的文档，回答：
1. **数据来源**：这份数据从哪来？采集时间是什么？
2. **字段字典**：每一列是什么意思？单位是什么？
3. **样本规模**：有多少行？代表什么群体？
4. **时间范围**：数据覆盖的时间段是什么？有没有季节性？
5. **缺失概览**：哪些列缺失值多？可能的原因是什么？
6. **使用限制**：这份数据能回答什么问题？不能回答什么？

老潘说得很直接："在公司里，没有数据卡的数据分析报告是不会被 review 的。因为没人知道你的结论建立在什么数据地基上。"

我们来写一个最简单的数据卡：

```python
import pandas as pd

def create_data_card(df: pd.DataFrame, data_source: str, description: str) -> str:
    """生成数据卡的 Markdown 文本"""
    card = f"""# 数据卡（Data Card）

## 数据来源
{data_source}

## 数据描述
{description}

## 样本规模
- 行数：{df.shape[0]:,}
- 列数：{df.shape[1]}

## 字段列表
"""
    # 添加字段信息
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_rate = df[col].isna().mean() * 100
        card += f"- **{col}** ({dtype}) - 缺失率: {missing_rate:.1f}%\n"

    card += f"""
## 缺失概览
"""
    missing_summary = df.isna().sum()
    for col, count in missing_summary[missing_summary > 0].items():
        rate = count / len(df) * 100
        card += f"- {col}: {count} ({rate:.1f}%)\n"

    return card

# 使用示例
df = pd.read_csv("data/users.csv")
card = create_data_card(
    df,
    data_source="公司内部用户行为数据库，导出时间：2026-01-15",
    description="2025 年全年活跃用户的交易记录，包含注册、浏览、购买三个环节的数据。"
)

# 保存到 report.md
with open("report.md", "w", encoding="utf-8") as f:
    f.write(card)
```

小北问："这不就是个格式化的数据摘要吗？为什么要搞得这么正式？"

因为**可复现性**。三个月后，如果你想把这份分析交给同事，或者重新跑一遍脚本，数据卡会告诉你：
- 数据的边界是什么（时间范围、样本来源）
- 哪些列的缺失值很多（分析时要小心）
- 每一列的业务含义是什么（不会误用字段）

更重要的是，2025-2026 年，**数据卡已经成为 AI 和数据科学社区的标准实践**。Google 的 Data Cards Playbook、NIST 的 AI 数据集文档标准、NeurIPS 的数据集提交要求，都在强调同一个事情：没有清晰文档的数据集，其结论无法被审计和复现。

老潘会这么说："你写的不是数据卡，是你给自己和别人的'免责声明'——告诉大家这份数据能干什么、不能干什么。"

这听起来很"工程化"，但在这个 AI 可以生成海量分析结论的时代，"免责声明"恰恰是最稀缺的东西。因为你不仅要让人相信你的结论，还得让他们知道——什么情况下你的结论会失效。

> **AI 时代小专栏 #2：Data Card —— AI 时代的"数据身份证"**

2025-2026 年，数据集的来源越来越多样：爬虫抓取的公开数据、第三方 API 购买的数据、企业内部数据库的导出文件。如果这些数据没有清晰的文档，任何基于它们的分析结论都无法被审计和复现。

"数据卡"（Data Card）或"数据表"（Datasheet for Datasets）的概念正在成为标准：
- **Google 的 Data Cards Playbook**（2022）为团队提供了创建透明数据文档的框架
- **NIST AI 标准**（2025）正在制定 AI 数据集和模型的文档规范
- **NeurIPS**（2025）提高了数据集提交的标准，要求提交的必须是"良好文档化、可复现、可访问"的数据
- **Open Trusted Data Initiative**（2025）要求所有有用的数据集必须包含来源、许可、目标用途、已知限制等元数据

为什么这很重要？因为 AI 模型的训练数据越来越不透明，而模型的偏见、公平性问题往往追溯到数据本身。如果你不知道数据是从哪来的、覆盖了哪些群体、缺失值的机制是什么，就无法评估模型的适用范围和风险。

所以本周你学的"数据卡"不只是"文档作业"，而是 AI 时代任何可信赖数据分析的前提。当你让 AI 帮你分析数据时，第一步应该是：先写数据卡，再问 AI 问题。

**参考（访问日期：2026-02-11）：**
- [The Data Cards Playbook - Google Research](https://research.google/blog/the-data-cards-playbook-a-toolkit-for-transparency-in-dataset-documentation/)
- [NIST AI Standards: Documentation of AI Datasets and AI Models](https://www.nist.gov/document/extended-outline-proposed-zero-draft-standard-documentation-ai-datasets-and-ai-models)
- [Open Trusted Data Initiative - Dataset Specification](https://the-ai-alliance.github.io/open-trusted-data-initiative/dataset-requirements/)

好了，概念都讲完了。但我们不是来"背定义"的——我们是来"做一个东西"的。这个东西就是 StatLab。

---

## StatLab 进度

StatLab 是什么？它是贯穿 16 周的"超级线"：你要从本周开始，逐周把同一个分析项目打磨成**可复现、可审计、可对外展示**的统计分析报告。

本周 StatLab 的目标非常明确：**写出第一版 `report.md`**。

这不是随便写写的 Markdown 文件，而是你整个项目的"数据地基"。它包含：

```python
# examples/01_init_statlab.py
import pandas as pd

def generate_data_card(data_path: str, metadata: dict) -> str:
    """为 StatLab 生成初始数据卡"""
    df = pd.read_csv(data_path)

    card = f"""# {metadata['title']}

## 数据来源
- **来源**：{metadata['source']}
- **采集时间**：{metadata['collection_date']}
- **数据联系人**：{metadata.get('contact', 'N/A')}

## 数据描述
{metadata['description']}

## 统计三问
本周的分析目标属于：**{metadata['analysis_type']}**
- 描述（Description）：说明数据本身的特点
- 推断（Inference）：从样本推断总体
- 预测（Prediction）：对未来或未见样本做出判断

## 样本规模
- **行数**：{df.shape[0]:,}
- **列数**：{df.shape[1]}

## 时间范围
{metadata.get('time_range', '未指定')}

## 字段字典
| 字段名 | 数据类型 | 业务含义 | 缺失率 |
|--------|----------|----------|--------|
"""
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_rate = df[col].isna().mean() * 100
        meaning = metadata['field_meanings'].get(col, '待补充')
        card += f"| {col} | {dtype} | {meaning} | {missing_rate:.1f}% |\n"

    card += f"""
## 缺失概览
"""
    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    for col, count in missing_summary.items():
        rate = count / len(df) * 100
        card += f"- **{col}**：{count} ({rate:.1f}%)\n"

    if len(missing_summary) == 0:
        card += "- 无缺失值\n"

    card += """
## 使用限制与注意事项
"""
    card += metadata.get('limitations', '待补充')

    return card

# 使用示例
if __name__ == "__main__":
    # 准备元数据（你需要根据实际数据集填写）
    metadata = {
        'title': '电商用户行为分析数据卡',
        'source': '公司内部数据库，SQL 导出',
        'collection_date': '2026-01-15',
        'contact': 'data-team@example.com',
        'description': '2025 年全年活跃用户的注册、浏览、购买行为数据。',
        'analysis_type': '描述（Description）',  # 或推断/预测
        'time_range': '2025-01-01 至 2025-12-31',
        'field_meanings': {
            'user_id': '用户唯一标识',
            'register_date': '注册日期',
            'age': '用户年龄（岁）',
            'gender': '性别（0=未知，1=男，2=女）',
            'city': '所在城市',
            'total_spend': '年度消费金额（元）',
            'purchase_count': '年度购买次数',
            # ... 其他字段
        },
        'limitations': '本数据仅包含活跃用户，不包含注册后未购买的用户；因此不适用于"新用户转化率"等分析。'
    }

    # 生成数据卡
    card = generate_data_card("data/users.csv", metadata)

    # 保存到 report.md
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(card)

    print("✅ 数据卡已生成到 report.md")
```

老潘看到这个脚本会说什么？"不错。但你记住——数据卡不是写一次就完的。每做一步清洗、每加一个特征，都要回来更新这个文档。不然三个月后你自己都不知道这列是从哪来的。"

本周 StatLab 的交付物就是这份 `report.md`。它可能不完美，但它是你所有后续分析的起点。没有它，任何结论都缺乏"审计地基"。

---

## Git 本周要点

本周必会命令：
- `git init`：初始化仓库
- `git status`：查看工作区状态
- `git add -A`：添加所有改动
- `git commit -m "draft: initial data card"`：提交改动
- `git log --oneline -n 5`：查看最近 5 条提交

常见坑：
- 不写提交信息：`git commit -m "draft"` 比 `git commit` 不写信息好一百倍
- 只提交代码不提交数据说明：`report.md` 和代码一样重要
- 把大文件（如原始 CSV）也提交进仓库：建议用 `.gitignore` 排除数据文件，只提交脚本和报告

Pull Request (PR)：
- Gitea 上也叫 Pull Request，流程等价 GitHub：push 分支 → 开 PR → review → merge

---

## 本周小结（供下周参考）

本周你只学了四件事，但它们会贯穿整门课：先问"你在做什么"（统计三问），再看"数据长什么样"（数据类型），然后用 pandas 把数据"请进内存"，最后给数据办一张"身份证"（数据卡）。

小北说："听起来好多……"

其实不多。下周我们就在这份有"身份证"的数据上画图、算均值、看分布。那时候你会发现——数据卡不是额外的负担，是让你以后少走弯路的提前投资。每次清洗、每加一个特征，记得回来更新它。

---

## Definition of Done（学生自测清单）

- [ ] 我能用自己的话解释"统计三问"（描述/推断/预测）
- [ ] 我能识别一个数据集中各列的数据类型（数值/分类，连续/离散）
- [ ] 我能用 pandas 读取 CSV 并查看基本形状和前几行
- [ ] 我能解释什么是"数据卡"以及它为什么重要
- [ ] 我为自己的数据集写了一份完整的数据卡（包含：数据来源、字段字典、样本规模、时间范围、缺失概览）
- [ ] 我把这份数据卡保存成了 `report.md`
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解"先问问题，再找答案"的统计思维
