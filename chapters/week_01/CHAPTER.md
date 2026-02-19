# Week 01：拿到数据先别急着跑模型 —— 统计三问与数据卡

> "数据不是答案，数据是问题的起点。"
> —— W. Edwards Deming

2026 年，"让 AI 先跑一遍 EDA" 已经成了很多人的起手式：上传 CSV，几秒钟就能得到一堆图表、相关性矩阵，甚至"初步结论"。Stack Overflow 2025 年开发者调查显示，84% 的开发者已经在使用或计划使用 AI 工具，GitHub Copilot 的用户数量在一年内增长了 400%，超过 1500 万开发者用它来辅助编码。这很诱人，也很危险。

因为 AI 可以比你更快地算出一个均值、画出一张分布图，却不会替你问：这份数据到底在讲谁？每个字段是什么意思？缺失值是怎么产生的？样本有没有代表性？如果你连"数据的地基"都没看清楚，后面的任何结论——无论 p 值多小、模型多复杂——都可能建在沙滩上。

<!-- 参考（访问日期：2026-02-15）：
- Stack Overflow Developer Survey 2025 - AI Tools: https://survey.stackoverflow.co/2025/ai
- GitHub Copilot Statistics 2026 (Panto AI): https://panto.ai/insights/github-copilot-statistics-2026
- State of Code Developer Survey 2026 (Sonar): https://www.sonarsource.com/developer-survey
-->

本周我们不急着跑模型。从"统计三问"和"数据卡"开始，先把数据的边界写清楚：来源是什么、字段怎么定义的、有多大、缺多少。这不是"准备工作"，这是分析本身。

---

## 本章学习目标

完成本周学习后，你将能够：
1. 用"统计三问"（描述/推断/预测）识别自己的分析目标
2. 用 pandas 读取数据并判断数据类型（数值型/分类型，连续/离散）
3. 编写一份**数据卡**（data card）：数据来源、字段解释、规模概览、缺失概览
4. 建立最小可用的 StatLab 报告（`report.md`）

---
<!--
贯穿案例：数据卡生成器（Data Card Generator）

案例演进路线：
- 第 1 节：统计三问 → 让读者识别"我要回答的问题类型"（描述/推断/预测）
- 第 2 节：数据类型判断 → 识别每列是数值还是类别，是连续还是离散
- 第 3 节：pandas 读数据与类型推断 → 把数据加载进 Python 并检查 dtype
- 第 4 节：数据卡生成 → 编写函数，自动生成数据卡（字段字典、规模、缺失概览）
- 第 5 节：写入 report.md → 把数据卡写入 Markdown 文件，建立 StatLab 基础

最终成果：读者拥有一份可读的 `data_card.md`，清楚回答"这份数据在讲谁、有哪些字段、有多大、缺多少"

数据集建议：使用公开数据集（如 Titanic、Penguins、或某城市的房价/空气质量数据），让案例有意义

---

认知负荷预算：
- 本周新概念（3 个，预算上限 4 个）：
  1. 统计三问（description/inference/prediction）
  2. 数据类型（数值型/分类型，连续/离散）
  3. 数据卡（data card）
- 结论：✅ 在预算内

回顾桥设计：Week 01 豁免（前情提要）

AI 小专栏规划：
- 第 1 个侧栏（第 2 节之后）：
  - 主题："AI 能替你做 EDA 吗？"
  - 连接点：刚学完"数据类型"，讨论 AI 自动类型推断的边界
  - 建议搜索词："AI automated EDA tools 2026", "pandas AI tools 2026", "data profiling automation 2025"

- 第 2 个侧栏（第 4 节之后）：
  - 主题："数据卡：AI 时代的'数据身份证'"
  - 连接点：刚生成数据卡，讨论工业界/开源项目的数据卡实践
  - 建议搜索词："datasheet for datasets 2026", "data card documentation best practices 2025", "model card data card 2026"

角色出场规划：
- 小北（第 1 节）：拿到数据就想"能不能直接训练模型"，引出"统计三问"的必要性
- 阿码（第 3 节）：用 pandas 读数据时遇到类型推断问题（如 zipcode 被当成整数），引出"类型判断很重要"
- 老潘（第 5 节）：看到生成的 report.md，点评"可复现性"的价值，铺垫 StatLab 超级线

StatLab 本周推进：
- 上周状态：无（第一周）
- 本周改进：选择数据集，生成最小可用 report.md（数据来源、字段字典、规模、缺失概览）
- 涉及的本周概念：统计三问（识别研究问题类型）、数据类型（字段字典）
- 建议示例文件：examples/99_statlab.py（本周报告生成入口脚本）
-->

## 1. 你到底想回答什么问题？

小北第一次拿到 Palmer Penguins 数据集，第一反应是："我们能不能直接训练一个模型，预测企鹅的物种？"

这很正常。在你还没见过这份数据的时候，"预测"听起来是最酷的事。但老潘会立刻问你一句："那你到底想回答什么问题？"

这听起来像废话，但它决定了你接下来要走的路。

---

### 从"想做模型"到"想回答问题"

统计学家 Andrew Gelman 说过："统计学不是计算，是提问。"在你写任何代码之前，先搞清楚你的分析属于哪一类：

**1. 描述（Description）**：数据长什么样？

典型问题：
- 这三种企鹅的嘴峰长度平均是多少？
- 哪个岛屿的企鹅最多？
- 数据里有没有缺失值？有多少？

这类问题不需要"推断到总体"，你只是在"描述手头的样本"。比如你算出 Adelie 企鹅的平均嘴峰长度是 38.8 mm——这就是一个描述性统计。

**2. 推断（Inference）**：从这个样本，能对总体说什么？

典型问题：
- Adelie 企鹅平均嘴峰长度 38.8 mm，Chinstrap 41.2 mm——这个 2.4 mm 的差异是"真的"，还是只是我们这 344 只企鹅的运气？
- 我们能不能说"雄性企鹅的体重比雌性大"这个结论对整个南极企鹅种群都成立？

这类问题需要引入"不确定性"：你看到的差异，可能只是运气。我们会在 Week 06-08 学习假设检验和置信区间来回答这类问题。

**3. 预测（Prediction）**：给定一个新样本，能猜出它的 Y 吗？

典型问题：
- 给定一只新企鹅的嘴峰长度、嘴峰深度、鳍肢长度，能不能猜出它的物种？
- 给定一套房子的面积、房龄、位置，能不能预测它的价格？

这类问题的目标是"猜得准"，你甚至不需要理解变量之间的关系（虽然理解会帮到你）。我们会在 Week 09-12 学习回归和分类。

```
                    🎯 你想回答什么问题？
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              "需要超出样本吗？"
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   不需要         判断        预测
   (长啥样)      (真差异?)    (新样本Y)
        │           │           │
        ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ 📊 描述 │ │ 🔬 推断 │ │ 🔮 预测 │
   └────┬────┘ └────┬────┘ └────┬────┘
        │           │           │
        ▼           ▼           ▼
   • 均值      • 假设检验    • 回归/分类
   • 中位数    • 置信区间    • 交叉验证
   • 可视化    • Week 06-08  • Week 09-12
```

```python
# examples/01_three_questions.py
import seaborn as sns

penguins = sns.load_dataset("penguins")

# 1. 描述：数据长什么样？
print("描述：三种企鹅的平均嘴峰长度")
print(penguins.groupby("species")["bill_length_mm"].mean())

# 2. 推断：差异是真差异吗？（Week 06 会学）
# from scipy import stats
# stats.ttest_ind(...)

# 3. 预测：给定特征，猜物种（Week 09-10 会学）
# from sklearn.ensemble import RandomForestClassifier
# ...
```

运行上面的代码，你会看到三个物种的平均嘴峰长度被打印出来。这就是一个"描述"——你在回答"数据长什么样"，而不是"这个差异在总体中是否成立"或"能不能预测新企鹅"。

### 为什么这很重要？

小北想做的"预测模型"当然是可行的，但她跳过了一步：先描述，再预测。如果你连"三种企鹅的嘴峰长度大概是多少"都不知道，又怎么能判断你的模型预测得对不对？

阿码举手："那我是不是每次分析都要回答这三类问题？"

也不是。更准确的说法是：**明确你在回答哪一类问题**。不要算了一个均值（描述），就立刻下结论说"总体就是这样"（推断），也不要跑了一个回归（预测），就以为你理解了变量之间的关系（推断）。

所以本周的任务是：先用"描述"把数据看清楚，把数据卡写好。预测和推断，我们在后面几周会慢慢来。

---

> **AI 时代小专栏：AI 能替你做 EDA 吗？**
>
> 2026 年的 AI 工具确实能自动生成描述性统计和可视化。它们可以在几秒钟内计算出均值、中位数、标准差，甚至画出相关性热图和分布图。一些工具如 Ataccama ONE 和 Atlan 等数据平台已经集成了自动化数据画像（profiling）功能。
>
> 但这里有三个 AI 替不了你的判断：
>
> **第一，业务背景**。AI 可以告诉你 "bill_length_mm 的均值是 43.9"，但不会告诉你"嘴峰长度是区分物种的关键特征之一"。这个知识需要你去查企鹅的生物学文献，或者请教领域专家。
>
> **第二，异常值的解释**。AI 可以标出"这个值离群"，但不会告诉你"这可能是一次测量错误"还是"这是一个真实但罕见的极端样本"。判断需要人。
>
> **第三，分析目标**。AI 可以给出所有可能的统计量，但不会帮你筛选"哪些和我真正的问题相关"。如果你想做推断，AI 可能给你一堆预测准确率——方向错了。
>
> 所以你刚学的"统计三问"，在 AI 时代其实更重要了：AI 是执行者，你是提问者。你不告诉它你想回答什么类型的问题，它就会什么都给你算一遍。
>
> <!-- TODO: 需联网搜索 "AI data analysis tools 2026" 补充真实参考链接 -->

## 2. 这列数字到底算什么类型？

阿码用 pandas 读取了一份包含美国邮政编码（zipcode）的数据，发现 pandas 自动把它识别成了 `int64` 类型。他挺满意："挺好，整数类型，节省内存。"

但这里有个陷阱：**数值型（numeric）不等于可以用来算均值**。

---

### 数值型 vs 分类型：能算什么，不代表该算什么

数据类型可以分为两大类：

**1. 数值型数据（Numeric）**
- **连续（Continuous）**：可以取任意值，如身高、体重、温度
- **离散（Discrete）**：只能取整数，如孩子数量、访问次数

**2. 分类型数据（Categorical）**
- **名义型（Nominal）**：没有顺序，如物种、岛屿、性别
- **有序型（Ordinal）**：有顺序但间隔不等，如满意度评分（低/中/高）

```
                        数据类型
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
    ┌──────────────┐               ┌──────────────┐
    │  🔢 数值型   │               │  📝 分类型   │
    └──────┬───────┘               └──────┬───────┘
           │                              │
     ┌─────┴─────┐                  ┌─────┴─────┐
     ▼           ▼                  ▼           ▼
  连续型       离散型            名义型       有序型
(Continuous) (Discrete)       (Nominal)    (Ordinal)
     │           │                  │           │
     ▼           ▼                  ▼           ▼
 • 身高      • 孩子数量         • 物种      • 满意度
 • 体重      • 访问次数         • 性别      (低/中/高)
 • 温度      • 只能取整数       • 邮政编码   • 有顺序
 • 任意值                       • 无顺序     • 间隔不等
```

想象一下衣服尺码：S、M、L、XL。你能算出"平均尺码"吗？(S + M + L) / 3 = ? 没有意义对吧？但如果编码成 1、2、3、4，计算机可不知道它们不能相加——它会很乐意地告诉你"平均尺码是 2.5"。

 zipcode 看起来像数字（90210），但它实际上是**名义型分类数据**。算它的均值（"平均邮政编码"）就像算"平均电话号码"——再精确的数字也没用。两个 zipcode 相减更没有意义：90210 减去 10001 等于 80209？不，这只是两个地区的代号，不能加减乘除。

```python
# examples/02_data_types.py
import seaborn as sns
import pandas as pd

penguins = sns.load_dataset("penguins")

# 看 pandas 自动推断的类型
print("Pandas 自动推断的类型：")
print(penguins.dtypes)
print()

# species 看起来是字符串，但应该是分类型
print("species 的唯一值：", penguins["species"].unique())
print("island 的唯一值：", penguins["island"].unique())
print()

# bill_length_mm 是连续数值型
print("bill_length_mm 的描述统计：")
print(penguins["bill_length_mm"].describe())
```

运行后你会发现，pandas 把 `species` 和 `island` 推断成了 `object` 类型（Python 字符串）。但从统计学的角度，它们是**分类型数据**。

这里有个容易踩的坑：如果你把它们转成数值（比如 Adelie=0, Chinstrap=1, Gentoo=2），pandas 会很开心地让你算"平均物种"——结果是 0.92。这个数字看起来很精确，但没有任何意义。类型不对，统计量就是笑话。

### 为什么类型判断很重要？

小北可能会问："我反正就是跑代码，类型对不对有什么关系？"

关系很大。**类型决定了你能做什么分析**：

| 分析类型 | 适用的数据类型 | 示例 |
|---------|--------------|------|
| 计算均值、标准差 | 连续数值型 | 嘴峰长度、体重 |
| 计算频数、比例 | 分类型 | 物种分布、岛屿分布 |
| 分组比较 | 分类型（组） + 连续型（指标） | 不同物种的嘴峰长度 |
| 相关性分析 | 两个连续数值型 | 嘴峰长度 vs 体重 |
| 卡方检验 | 两个分类型 | 物种 vs 岛屿 |

如果你把 `species` 当成数值型算均值，会得到一个无意义的数字；如果你把 `bill_length_mm` 当成分类型，就无法使用很多强大的统计工具。

阿码眼睛一亮，突然问："那如果我把 species 转成整数（Adelie=0, Chinstrap=1, Gentoo=2），然后直接跑个随机森林，预测准确率会不会更高？"

这是个好问题——但也暴露了他还没真正理解"类型"的语义。随机森林确实可以接受数值输入，但当你把物种编码成 0、1、2 时，模型会误以为 Gentoo（2）"比" Chinstrap（1）"大"，而 Chinstrap 又"比" Adelie（0）"大"——这个顺序关系完全是虚构的。更关键的是，如果下周来了新数据，出现第四种物种（比如 Chinstrap penguin 的亚种），你的编码体系就崩了。

类型转换不只是"让代码能跑"，而是让代码的**语义正确**。你可以在 `read_csv` 时指定 `dtype` 参数，或者在读取后用 `astype()` 转换类型。下一节我们会看到具体怎么做。

---

### 小步快跑：检查你的数据类型

在写任何分析代码之前，先做三件事：

1. **打印 `dtypes`**：看看每列被推断成什么类型
2. **打印 `head()`**：看看前几行的实际值
3. **打印 `unique()`（对分类型）**：看看有哪些类别，有没有拼写错误（比如 "Adelie" 和 "adelie" 同时出现）

这三步能帮你避开 80% 的类型相关错误。阿码试过一次：他把 `species` 当字符串处理，结果写了一堆 `if row["species"] == "Adelie"` 这样的代码，后来才发现用 `groupby("species")` 一行就能搞定。

现在你学会了判断类型。但类型判断只是纸上谈兵——下一节我们把数据真正加载进 Python，看看 pandas 实际怎么推断类型，以及当它猜错时该怎么办。

## 3. 用 pandas 把数据请进 Python

小北第一次尝试用 pandas 读取数据，写了这样一行代码：

```python
df = pd.read_csv("data.csv")
```

运行后立刻报错：`FileNotFoundError: data.csv not found`。她盯着屏幕，一脸茫然："文件明明就在桌面上啊！"

---

### 第一次读数据：避开的坑

这是新手最容易遇到的第一个坑：**路径问题**。pandas 默认从"当前工作目录"找文件，而你的工作目录可能不是你想象的那样。

```python
# examples/03_pandas_basics.py
import pandas as pd
import seaborn as sns

# 方法 1：使用 seaborn 内置数据集（最简单）
penguins = sns.load_dataset("penguins")
print("方法 1：seaborn 内置数据集")
print(f"形状：{penguins.shape}")  # (344, 7)
print()

# 方法 2：从本地文件读取（注意路径）
# 假设你的文件在 data/penguins.csv
# df = pd.read_csv("data/penguins.csv")  # 相对路径
# df = pd.read_csv("/Users/yourname/project/data/penguins.csv")  # 绝对路径
# df = pd.read_csv("../data/penguins.csv")  # 上级目录

# 检查数据的基本信息
print("前 3 行数据：")
print(penguins.head(3))
print()

print("数据类型：")
print(penguins.dtypes)
print()

print("数据规模：")
print(f"行数：{penguins.shape[0]}，列数：{penguins.shape[1]}")
print()

print("缺失值统计：")
print(penguins.isna().sum())
```

小北跑通后问："为什么要检查 `dtypes` 和 `shape`？我直接开始分析不就行了吗？"

老潘的回答是："你总得知道你手里有多少行数据、有哪些列，才能判断后面的结论可信度。如果只有 30 行企鹅数据，你算出的均值就很不稳定；如果某一列缺失值超过 50%，你算任何统计量都要打个问号。"

### 类型转换：告诉 pandas "这列是什么"

pandas 的类型推断很聪明，但不是万能的。有时候你需要手动指定类型：

```python
# 指定列的数据类型
penguins = penguins.astype({
    "species": "category",
    "island": "category",
    "sex": "category"
})

print("转换后的类型：")
print(penguins.dtypes)
```

把分类型数据转成 `category` 类型有两个好处：
1. **节省内存**：存储重复字符串时更高效
2. **语义明确**：你告诉 pandas（和你自己），这列不是用来算算术的

### 常见错误与恢复

**错误 1：路径问题**

```python
# ❌ 错误：路径写死
df = pd.read_csv("/Users/xiaobei/Desktop/data.csv")

# ✅ 正确：使用相对路径
df = pd.read_csv("data/data.csv")
```

**错误 2：编码问题**

如果你的文件包含中文，可能会遇到 `UnicodeDecodeError`：

```python
# 尝试不同编码
df = pd.read_csv("data.csv", encoding="utf-8")  # 默认
# df = pd.read_csv("data.csv", encoding="gbk")   # 中文 Windows
# df = pd.read_csv("data.csv", encoding="gb18030")  # 更广泛的中文支持
```

**错误 3：日期没被解析**

如果你的数据有日期列，pandas 可能把它当成字符串：

```python
# df = pd.read_csv("data.csv", parse_dates=["date_column"])
```

现在数据已经在 Python 里了，你知道它的规模、类型和缺失情况。但你可能会问：**然后呢？我总不能每次都重新跑一遍这些检查命令吧？**

这就需要把信息写成一份人类可读的"数据卡"。它是你分析的起点，也是别人审核你工作的第一站。

## 4. 把数据的"身份证"写清楚 —— 数据卡生成

老潘看到小北的代码后，问了一句："这份数据从哪来的？每个字段是什么意思？"

小北愣了一下："这个……我从一个教程里下载的，字段应该是……企鹅的属性？"

这就是问题所在。**代码跑通 ≠ 分析可信**。如果你连数据的背景都说不清楚，后面的任何结论都可能被人质疑。

---

### 什么是数据卡？

数据卡（data card）是一份数据的"身份证"，它用人类可读的方式回答：

1. **数据来源**：数据从哪来？谁收集的？什么时候？
2. **字段字典**：每列是什么意思？单位是什么？
3. **规模概览**：有多少行？多少列？
4. **缺失概览**：哪些列有缺失？缺失率多少？

Google 和 MIT 的研究者提出了"Datasheets for Datasets"框架，建议每个数据集都配上一份标准化的文档。到 2026 年，这个实践已经被广泛应用于医疗 AI 和计算机视觉领域。

### 编写数据卡生成函数

让我们写一个函数，自动生成数据卡：

```python
# examples/04_data_card.py
import pandas as pd
import seaborn as sns
from typing import Dict, Any

def generate_data_card(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """
    生成数据卡（Markdown 格式）

    Parameters
    ----------
    df : pd.DataFrame
        数据集
    metadata : dict
        数据的元信息，包含 source、description、collection_date 等

    Returns
    -------
    str
        Markdown 格式的数据卡
    """
    lines = []
    lines.append("# 数据卡（Data Card）\n")

    # 1. 数据来源
    lines.append("## 数据来源\n")
    for key, value in metadata.items():
        lines.append(f"- **{key}**：{value}")
    lines.append("\n")

    # 2. 字段字典
    lines.append("## 字段字典\n")
    lines.append("| 字段名 | 数据类型 | 描述 | 缺失率 |")
    lines.append("|--------|---------|------|--------|")

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_rate = (df[col].isna().sum() / len(df) * 100).round(1)
        lines.append(f"| {col} | {dtype} | （待补充） | {missing_rate}% |")
    lines.append("\n")

    # 3. 规模概览
    lines.append("## 规模概览\n")
    lines.append(f"- **行数**：{len(df)}")
    lines.append(f"- **列数**：{len(df.columns)}")
    lines.append("\n")

    # 4. 缺失概览
    lines.append("## 缺失概览\n")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, count in missing.items():
            rate = (count / len(df) * 100).round(1)
            lines.append(f"- **{col}**：{count} ({rate}%)")
    else:
        lines.append("- 无缺失值")
    lines.append("\n")

    return "\n".join(lines)


# 使用示例
penguins = sns.load_dataset("penguins")

metadata = {
    "数据集名称": "Palmer Penguins",
    "来源": "seaborn 内置数据集",
    "原始来源": "Palmer Station, Antarctica LTER",
    "描述": "南极 Palmer Station 的三种企鹅（Adelie, Chinstrap, Gentoo）的形态测量数据",
    "收集时间": "2007-2009 年",
    "单位说明": "长度单位为毫米（mm），重量单位为克（g）"
}

data_card = generate_data_card(penguins, metadata)
print(data_card)

# 写入文件
with open("data_card.md", "w", encoding="utf-8") as f:
    f.write(data_card)
```

运行后，你会得到一份 `data_card.md` 文件，内容大致如下：

```markdown
# 数据卡（Data Card）

## 数据来源
- **数据集名称**：Palmer Penguins
- **来源**：seaborn 内置数据集
- **原始来源**：Palmer Station, Antarctica LTER
- **描述**：南极 Palmer Station 的三种企鹅（Adelie, Chinstrap, Gentoo）的形态测量数据
- **收集时间**：2007-2009 年
- **单位说明**：长度单位为毫米（mm），重量单位为克（g）

## 字段字典
| 字段名 | 数据类型 | 描述 | 缺失率 |
|--------|---------|------|--------|
| species | object | （待补充） | 0.0% |
| island | object | （待补充） | 0.0% |
| bill_length_mm | float64 | （待补充） | 2.4% |
| bill_depth_mm | float64 | （待补充） | 2.4% |
| flipper_length_mm | float64 | （待补充） | 2.4% |
| body_mass_g | float64 | （待补充） | 2.4% |
| sex | object | （待补充） | 2.8% |

## 规模概览
- **行数**：344
- **列数**：7

## 缺失概览
- **sex**：10 (2.8%)
- **bill_length_mm**：8 (2.4%)
- **bill_depth_mm**：8 (2.4%)
- **flipper_length_mm**：8 (2.4%)
- **body_mass_g**：8 (2.4%)
```

### 数据卡的价值

小北看完这份数据卡，愣了一下："原来这么多列有缺失值，而且缺失率还不一样！"

这就是数据卡的价值：**让问题暴露出来**。

---

### 实战故事：没有数据卡的代价

老潘讲了一个他年轻时的故事：

"有一年，我接手了一个离职同事留下的分析项目。代码能跑，但没有任何文档。我花了三天时间才搞清楚：数据是从哪个 API 拉的、每个字段的含义是什么、为什么有一半的 `user_id` 是空的。后来发现，那个同事把测试数据混进了生产数据——但因为没有数据卡，没人知道这回事。那份分析结论被业务方拿去做了决策，后来才发现全是错的。"

"从那以后，我给自己定了一个规矩：**任何数据集，先写数据卡，再开始分析**。宁可多花 10 分钟写文档，也不要花 3 天去猜前人的意图。"

阿码听完说："这个故事让我想起了代码注释……但数据卡好像比注释更重要？"

"对，因为**数据会比你活得更久**。三个月后你可能不记得 `bill_depth_mm` 是什么意思，但数据卡会替你记住。"如果缺失值被藏在数据里，你可能算出一个均值，却不知道它只基于 95% 的数据；如果数据卡把缺失率写得清清楚楚，你在读任何结论时都会问"这个统计量是基于多少数据算出来的？"

阿码问："字段字典里的'描述'为什么是（待补充）？我能不能让 AI 帮我填？"

1. AI 不知道你的**业务背景**——它可能把 `bill_length_mm` 解释成"鸟嘴长度"，但准确的术语是"嘴峰长度"（从嘴基部到顶端的距离）
2. AI 不知道你的**分析目标**——如果是为了比较物种差异，你需要强调"这是区分物种的关键特征之一"

你需要自己补充（或者让 AI 生成候选后人工核实）：

| 字段名 | 描述 |
|--------|------|
| species | 企鹅物种：Adelie（阿德利）、Chinstrap（颈带）、Gentoo（巴布亚） |
| island | 采集岛屿：Biscoe、Dream、Torgersen |
| bill_length_mm | 嘴峰长度（从嘴基部到顶端的距离） |
| bill_depth_mm | 嘴峰深度（嘴巴最薄处的厚度） |
| flipper_length_mm | 鳍肢长度 |
| body_mass_g | 体重 |
| sex | 性别：Male（雄）、Female（雌） |

补充完字段字典，你的数据卡就完整了。现在你回答了第 1 节的"统计三问"中的第一问：**描述**——你知道了数据长什么样、字段是什么意思、缺失情况如何。

---

> **AI 时代小专栏：数据卡：AI 时代的"数据身份证"**
>
> 2026 年，"Datasheets for Datasets"已经从学术论文的标准实践变成了工业界的共识。Google 发布了《Data Cards Playbook》工具包，帮助团队系统化地记录数据集的透明度信息。
>
> 在 AI 时代，数据卡的重要性不减反增：
>
> **第一，模型审计需要**。如果你训练了一个预测模型，别人（或者你自己三个月后）会问："你的训练数据是什么？有没有代表性？缺失值怎么处理的？"数据卡就是这些问题的答案。
>
> **第二，伦理审查需要**。2026 年的医学 AI 论文中，数据集文档（datasheet）已经是标准配置。你需要说明数据的来源、收集过程、潜在偏见，否则审稿人会问"你的模型对哪些人群有效？对哪些可能失效？"
>
> **第三，可复现性需要**。阿码把代码发给小北，但小北跑出来结果不一样。为什么？因为数据集版本不同、预处理方式不同。数据卡把这些信息写清楚，才能真正"可复现"。
>
> 所以你刚学的"生成数据卡"，在 AI 时代不是"额外的 paperwork"，而是"数据项目的基础设施"。Open Trusted Data Initiative 等组织正在推动统一的数据卡片标准，包括来源、许可证、用途、限制和风险等元数据。
>
> 参考（访问日期：2026-02-15）：
> - [Datasheets for Datasets](https://cacm.acm.org/research/datasheets-for-datasets/)
> - [The Data Cards Playbook](https://research.google/blog/the-data-cards-playbook-a-toolkit-for-transparency-in-dataset-documentation/)
> - [Dataset Specification - Open Trusted Data Initiative](https://the-ai-alliance.github.io/open-trusted-data-initiative/dataset-requirements/)

## 5. 从脚本到可复现报告 —— StatLab 起步

小北下周打开电脑，想重新跑一次上周的分析，结果发现："我当时是在 Jupyter 里跑的，还是存成了 .py 文件？数据放在哪了？我是不是改过什么参数？"

这就是没有"可复现性"的典型场景。老潘看到后会摇摇头："能跑的烂代码，比不能跑的好代码有用。但能复现的结论，才有长期价值。"

---

### StatLab：一条可复现的报告流水线

从这周开始，你要建立一条**可复现分析报告流水线**——我们叫它 StatLab。

它不是一堆散落的 Jupyter notebook，也不是手工复制粘贴出来的 Word 文档，而是一个可以从原始数据一键生成 `report.md`（可选导出 `report.html`）的脚本。

每周你都在上周的基础上增量修改，到 Week 16，你手里会有一份**可审计、可展示、可复现**的完整报告。

### 本周里程碑：最小可用报告

本周的 StatLab 目标很简单：**生成一份包含数据卡的 `report.md`**。

```python
# examples/99_statlab.py
import pandas as pd
import seaborn as sns
from pathlib import Path

def generate_data_card(df: pd.DataFrame, metadata: dict) -> str:
    """生成数据卡（同第 4 节）"""
    # ...（代码同上，省略）


def generate_report(data_df: pd.DataFrame, output_path: str = "report.md"):
    """
    生成 StatLab 报告

    Parameters
    ----------
    data_df : pd.DataFrame
        数据集
    output_path : str
        输出文件路径
    """
    metadata = {
        "数据集名称": "Palmer Penguins",
        "来源": "seaborn 内置数据集",
        "原始来源": "Palmer Station, Antarctica LTER",
        "描述": "南极 Palmer Station 的三种企鹅（Adelie, Chinstrap, Gentoo）的形态测量数据",
        "收集时间": "2007-2009 年",
        "单位说明": "长度单位为毫米（mm），重量单位为克（g）"
    }

    # 1. 生成数据卡
    data_card = generate_data_card(data_df, metadata)

    # 2. 组装报告
    report = f"""# StatLab 分析报告

> 本报告由 StatLab 流水线自动生成
> 生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d')}

{data_card}

---

## 下一步

- [ ] 补充字段字典的业务含义
- [ ] 补充描述统计（Week 02）
- [ ] 生成可视化图表（Week 02）
"""

    # 3. 写入文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"报告已生成：{output_file.absolute()}")
    return output_file


# 运行
if __name__ == "__main__":
    penguins = sns.load_dataset("penguins")
    generate_report(penguins, "output/report.md")
```

运行后，你会得到 `output/report.md`，它是一份完整的、可读的 Markdown 报告。

### 为什么要这样做？

小北可能会问："我直接在 Jupyter 里看不好吗？为什么要生成 .md 文件？"

老潘会这样回答：

**第一，可审计。** 别人（或者三个月后的你）打开报告，一眼就能看到数据来源、字段解释、缺失情况。如果是 Jupyter，你得翻很多 cell 才能找到这些信息。老潘当年就吃过这个亏：三个月后被问到"你当时的数据是什么版本"，他翻遍了 notebook 也找不到数据来源。

**第二，可版本控制。** `.md` 文件是纯文本，可以用 git 追踪改动。你知道哪一周改了什么，为什么要改。如果是手工复制的 Word 文档，版本控制几乎不可能。

**第三，可导出。** Markdown 可以用工具（如 pandoc）一键转换成 HTML、PDF，甚至 PPT。你只需要维护一份源文件，就能生成多种输出格式。

**第四，可复现。** 只要数据和代码都在，任何人运行 `python examples/99_statlab.py` 都能得到完全相同的报告。这就是"可复现分析"的起点。

### 本周小结（供 Week 02 参考）

这周你做的不是"准备工作"，而是建立了整个分析的地基。你学会了先问清楚"我想回答什么问题"——是描述数据的样子、推断总体的规律，还是预测新样本？这个问题决定了后续每一步怎么走。你也学会了判断数据类型： zipcode 看起来像数字，但它和物种一样是分类型，不能算均值；bill_length_mm 是数值型，可以算均值、画分布图。你还用 pandas 把数据真正加载进 Python，检查了它的规模、类型和缺失情况。

最后一项技能是数据卡——一份人类可读的"数据身份证"。它回答了"数据从哪来、字段是什么意思、有多少缺失"，这些信息不是"额外工作"，而是任何统计分析的起点。StatLab 报告流水线也从这周起步：从脚本到可复现的 `report.md`，任何人运行 `examples/99_statlab.py` 都能得到完全相同的报告。

下周，我们会在这份数据卡的基础上加入描述统计和可视化，把"数据长什么样"这个问题回答得更清楚。

---

## StatLab 进度

### 本周改进：从零到可复现报告

本周是 StatLab 的起点。在之前的"没有流水线"状态下，你可能习惯在 Jupyter 里做分析——能跑，但不方便审计和复现。

我们做了三件事：

**1. 选择数据集**：我们用 Palmer Penguins 作为贯穿案例。它比 Titanic 更适合教学（无敏感内容），字段简单但足够展示所有概念，并且自带少量缺失值。

**2. 生成数据卡**：我们把第 4 节学的"数据卡生成器"集成进了报告流水线。现在 `report.md` 的开头会自动包含：
- 数据来源和背景
- 字段字典（包括缺失率）
- 规模概览
- 缺失概览

**3. 建立脚本入口**：`examples/99_statlab.py` 是这周的 StatLab 入口脚本。运行它就能生成 `output/report.md`。任何人拿到代码和数据，运行这个脚本，都能得到完全相同的报告。

### 与本周知识的连接

**统计三问** → 在数据卡中补充字段字典时，你要思考"这个字段将来可能用来回答哪类问题"。比如 `species` 是分类型，适合做分组比较；`bill_length_mm` 是连续数值型，适合计算均值和画分布图。

**数据类型判断** → 我们在字段字典中记录了每列的数据类型，为下周的"描述统计"打基础——不同类型的字段要用不同的统计量和图表。

**pandas 基础** → `generate_data_card` 函数用到了 `dtypes`、`isna().sum()`、`shape` 等操作，这些都是本周学的基础技能。

### 下周预告

本周的 `report.md` 只有一份数据卡。下周我们会：
- 加入描述统计（均值、中位数、标准差、分位数）
- 生成 2-3 张诚实的数据可视化
- 更新 `generate_report` 函数，让这些新内容自动进入报告

老潘看到这份基础的 `report.md`，会说："路还很长，但至少你迈出了第一步——先看清楚数据是什么，再谈任何结论。"

---

## Git 本周要点

本周必会命令：
- `git init`（如果还没有仓库）
- `git status`
- `git add -A`
- `git commit -m "feat: week_01 data card generator"`

常见坑：
- 只在 Jupyter 里写代码，不存成 `.py` 文件：下周无法复现
- 路径写死（如 `/Users/xxx/data.csv`）：换个电脑就跑不了

---


## Definition of Done（学生自测清单）

- [ ] 我能用自己的话解释"描述/推断/预测"的区别
- [ ] 我能判断一列数据是"数值型"还是"分类型"
- [ ] 我能用 pandas 读取数据并打印 shape 和 dtypes
- [ ] 我能编写函数生成数据卡（字段字典、规模、缺失概览）
- [ ] 我能把数据卡写入 report.md，且能重新运行生成相同内容
- [ ] 我知道为什么不能"拿到数据就训练模型"
