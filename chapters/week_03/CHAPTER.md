# Week 03：数据清洗与准备 —— 从"原始数据"到"可分析数据"

> "Garbage in, garbage out."
> — 计算机科学格言

2025 年底，OpenAI 的研究显示，GPT-4 在数据分析任务上的错误率高达 15-20%，其中大部分错误源于对"脏数据"的误判——把缺失值当成零、把异常值直接删除、把类别编码当成数值运算。AI 可以帮你写清洗代码，但它不会替你判断：这个缺失是随机的还是有规律的？那个异常点是录入错误还是真实的高价值用户？

数据清洗不是"技术问题"，而是"决策问题"。吴恩达反复强调的 Data-centric AI 理念告诉我们：在真实项目中，80% 的时间花在数据准备上，而数据质量往往比模型选择更能决定最终结果。本周我们要做的，是把"清洗"从"删掉有问题的行"升级为"理解问题从哪来、记录你做了什么决定、让整个过程可复现"。

---

## 前情提要

上周你学会了"看见"数据——用均值/中位数描述中心，用标准差/IQR 描述波动，用直方图和箱线图展示分布形状。老潘看完你的报告，问了一个新问题："**这些数字和图，是基于什么数据算出来的？**"

这是个关键追问。如果你的数据里有 30% 的缺失值，但你直接 `dropna()` 把它们删掉了，那你的"均值"其实只代表了"有完整数据的那些人"——这个结论还能推广到全体用户吗？如果你看到箱线图里的异常点就统统删除，可能把最有价值的 VIP 用户也一起扔了。

本周我们要回答：如何把"脏数据"变成"可分析数据"，同时让每一个决策都有据可查、有记录可追。

---

## 学习目标

完成本周学习后，你将能够：
1. 识别缺失值的三种机制（MCAR/MAR/MNAR），并选择合适的处理策略
2. 用统计方法和业务规则检测异常值，并做出"修数据"或"修解释"的决策
3. 对数值特征进行标准化或归一化，理解何时需要缩放
4. 对类别特征进行 one-hot 或 label 编码，避免编码陷阱
5. 为你的数据集生成一份"清洗决策日志"，记录所有预处理步骤和理由

---

<!--
贯穿案例：电商用户数据清洗实战

本周贯穿案例是一个渐进式小项目：读者从一份"原始"的电商用户数据开始，逐节处理缺失值、异常值、特征变换，最终产出一份"清洗后数据说明文档"。

- 第 1 节：缺失值机制 → 案例从"原始数据"变成"缺失概览 + 机制判断"
- 第 2 节：缺失值处理 → 案例从"有缺失"变成"已处理缺失 + 决策记录"
- 第 3 节：异常值检测 → 案例从"有异常点"变成"异常点分类 + 处理策略"
- 第 4 节：特征变换 → 案例从"原始特征"变成"清洗后特征集"
- 第 5 节：清洗日志 → 案例从"零散处理"变成"完整清洗文档"

最终成果：读者为选定的数据集生成一份"清洗后数据说明文档"，包含：
- 数据卡（Week 01）
- 描述统计（Week 02）
- 清洗决策日志（本周新增）：缺失处理策略、异常值处理理由、特征变换方法

认知负荷预算：
- 本周新概念（4 个，预算上限 4 个）：
  1. 缺失值机制（missingness mechanisms）
  2. 异常值检测与处理（outliers）
  3. 特征缩放（feature scaling）
  4. 特征编码（encoding）
- 结论：✅ 在预算内

回顾桥设计（至少 2 个）：
- [数据类型]（来自 week_01）：在第 4 节，通过"数值型 vs 分类型需要不同变换"再次强调数据类型决定处理方法
- [箱线图]（来自 week_02）：在第 3 节，用上周的箱线图识别异常点，引出"该修数据还是修解释"的决策
- [均值/中位数]（来自 week_02）：在第 2 节，讨论缺失值填充策略时回顾"用均值还是中位数更稳健"
- [数据卡]（来自 week_01）：在第 5 节，把清洗日志作为数据卡的"质量扩展"

AI 小专栏规划：

AI 小专栏 #1（放在第 1-2 节之间）：
- 主题：AI 处理缺失值的局限
- 连接点：与第 1 节"缺失值机制"呼应，讨论 AI 为什么无法判断 MCAR/MAR/MNAR
- 建议搜索词：
  - "AI missing data imputation limitations 2025 2026"
  - "LLM data cleaning missing values bias 2026"
  - "automated data preprocessing pitfalls 2025"

AI 小专栏 #2（放在第 3-4 节之间）：
- 主题：特征工程中的 AI 辅助与人类判断
- 连接点：与第 4 节"特征变换"呼应，讨论 AI 可以生成代码但无法判断业务语义
- 建议搜索词：
  - "AI feature engineering human in the loop 2026"
  - "automated feature scaling encoding best practices 2025"
  - "Data-centric AI feature engineering 2026"

角色出场规划：
- 小北（第 1 节）：直接把缺失值当 0 处理，得出"用户平均消费下降"的错误结论
- 阿码（第 2、3 节）：追问"AI 能不能自动选最好的策略"、"异常值能不能用 AI 自动检测"
- 老潘（第 4、5 节）：强调"把变换写成函数"、"清洗日志比代码更重要"

StatLab 本周推进：
- 上周状态：report.md 已有数据卡 + 描述统计章节，包含均值/中位数/标准差/IQR 和分布图
- 本周改进：在 report.md 中添加"数据清洗日志"章节，记录所有清洗决策（缺失处理策略、异常值处理理由、特征变换方法）
- 涉及的本周概念：缺失值机制、异常值检测与处理、特征缩放、特征编码
- 建议示例文件：examples/03_cleaning_logger.py（生成清洗日志的脚本）
-->

## 1. 缺失的不只是数据

小北拿到一份电商用户数据，发现"月消费金额"列有很多空值。他没多想，直接把空值当成 0 处理，然后算了个平均值，兴冲冲地汇报："我们的用户平均月消费 120 元，比上季度下降了 30%。"

老潘看了一眼，只问了一句："**你删掉了谁？**"

"没删啊，"小北有点懵，"我只是把空值填成了 0。"

问题就出在这里。那些空值可能代表"这个月没消费"，也可能代表"数据没采集到"——两种情况的处理方式完全不同。如果你把"没采集到"当成"没消费"，结论就会严重偏斜。

在统计学中，缺失不是简单的"数据丢了"，而是有三种不同的**缺失值机制**（missingness mechanisms）：

**MCAR（完全随机缺失）**：缺失与任何观测值或未观测值都无关。比如服务器随机故障导致部分记录丢失。这种情况下，删掉缺失行不会系统性偏斜结论——但你仍然损失了样本量。

**MAR（随机缺失）**：缺失与**观测到的**数据有关，但与**缺失值本身**无关。比如高收入用户更不愿意填写收入信息——但只要你观察到了"用户等级"这个变量，就能预测谁可能缺失。这种情况下，简单的删除会引入偏差，需要更精细的处理。

**MNAR（非随机缺失）**：缺失与**缺失值本身**有关。比如月消费为 0 的用户觉得"没面子"所以不填——这时"缺失"本身就是信息。如果你删掉或填充这些行，就丢掉了最重要的信号。

```python
import pandas as pd
import numpy as np

# 加载数据
df = pd.read_csv("data/ecommerce_users.csv")

# 第一步：缺失概览
missing_rate = df.isna().mean().sort_values(ascending=False)
print("缺失率概览：")
print((missing_rate * 100).round(1))

# 第二步：缺失模式分析——缺失是否与某些观测值相关？
print("\n高收入用户的收入缺失率：")
high_spenders = df[df['avg_monthly_spend'] > df['avg_monthly_spend'].quantile(0.8)]
print(f"{high_spenders['income'].isna().mean():.1%}")

print("\n低收入用户的收入缺失率：")
low_spenders = df[df['avg_monthly_spend'] <= df['avg_monthly_spend'].quantile(0.2)]
print(f"{low_spenders['income'].isna().mean():.1%}")
```

运行这段代码，你可能会发现：高收入用户的收入缺失率明显高于低收入用户。这说明缺失不是随机的（不是 MCAR），而是与消费行为相关——可能是 MAR，也可能是 MNAR。

关键洞察：**在决定"怎么处理"之前，你必须先回答"为什么缺失"**。数据卡（Week 01 的概念）在这里派上用场——字段的业务含义能帮你判断缺失机制。如果是"用户主动跳过"，可能是 MNAR；如果是"系统未采集"，可能是 MCAR 或 MAR。

小北现在明白了：他之前把空值填成 0，相当于假设所有缺失都是"没消费"——但数据告诉他，更可能是"高消费用户不愿意填"。这个结论完全相反。

下一步的问题是：既然不能简单填 0，那该怎么处理？

---

> **AI 时代小专栏：AI 处理缺失值的局限**
>
> 当你把数据丢给 AI，说"帮我处理缺失值"，它通常会默认做两件事之一：删除含缺失的行，或者用列均值填充。这看起来高效，实则危险——因为 AI 无法判断你的缺失是 MCAR、MAR 还是 MNAR。
>
> 2025 年的一项研究显示，在 50 个常用数据集上，自动化缺失值填充工具的表现差异巨大：当缺失是 MCAR 时，各种方法差异不大；但当缺失是 MNAR 时，自动化方法的偏差可能比简单删除还要严重。问题在于，AI 看不到"用户为什么没填"这个业务逻辑。
>
> 所以 AI 可以帮你写 `fillna()` 的代码，但判断"用什么策略"的权力必须在你手里。这也是为什么本周强调"记录决策理由"——三个月后回头看，你能说清楚"当时为什么选中位数而不是均值"。
>
> 参考（访问日期：2026-02-11）：
> - [Ovage: Data Cleaning Techniques 2026](https://www.ovaledge.com/blog/data-cleaning-techniques)
> - [PW Skills: Handling Missing Values 2026](https://pwskills.com/blog/how-to-handle-missing-values-data-science/)
> - [Latitude: Preprocessing Pipelines for LLMs](https://latitude.so/blog/ultimate-guide-to-preprocessing-pipelines-for-llms/)

---

## 2. 删掉还是填补？这是个决策

阿码看着小北的困境，提出了一个看似合理的建议："既然判断缺失机制这么麻烦，能不能让 AI 自动选最好的策略？"

好问题——但答案是否定的。没有"最好"的策略，只有"最适合你当前问题和数据"的策略。选择处理策略时，你需要权衡三个因素：缺失机制、样本量、业务影响。

**删除（Deletion）**：最简单，但代价最大。如果缺失是 MCAR 且比例很低（<5%），删除是安全的。但如果缺失是 MAR 或 MNAR，删除会系统性偏斜样本。记住：删掉一行，你就删掉了这一行的**所有**信息——不只是缺失的那一列。

**填充（Imputation）**：用某个值替代缺失。常见策略包括：
- 常数填充（如 0、"未知"）：适合"确实没有"的场景
- 均值/中位数填充：适合 MCAR 或轻度 MAR，但会压缩方差
- 前向/后向填充：适合时间序列
- 模型预测填充：用其他特征预测缺失值，适合 MAR，但可能过拟合

这里有个回顾桥：还记得 Week 02 讨论的"均值 vs 中位数"吗？同样的逻辑适用于填充策略。如果数据有长尾或异常值，**中位数填充比均值填充更稳健**——因为均值会被极端值拽着跑。

```python
# 不同填充策略的对比
import pandas as pd
import numpy as np

# 原始数据（含缺失）
spend_original = df['monthly_spend'].copy()

# 策略 1：删除缺失
spend_dropna = spend_original.dropna()
print(f"删除后均值：{spend_dropna.mean():.2f}，样本量：{len(spend_dropna)}")

# 策略 2：均值填充
spend_mean = spend_original.fillna(spend_original.mean())
print(f"均值填充后均值：{spend_mean.mean():.2f}，标准差：{spend_mean.std():.2f}")

# 策略 3：中位数填充（更稳健）
spend_median = spend_original.fillna(spend_original.median())
print(f"中位数填充后均值：{spend_median.mean():.2f}，标准差：{spend_median.std():.2f}")

# 策略 4：分组填充（利用 MAR 的信息）
# 假设我们发现缺失与"用户等级"有关
spend_grouped = df.groupby('user_level')['monthly_spend'].transform(
    lambda x: x.fillna(x.median())
)
print(f"分组填充后均值：{spend_grouped.mean():.2f}")
```

注意一个细节：**填充后的标准差会变小**。这是因为填充把一堆"未知"变成了"相同的中位数"，人为压缩了数据的波动。这是填充策略的隐藏代价——你在修复缺失的同时，也在改变数据的分布形状。

那么，如何记录这些决策？不要只写"用中位数填充"，而要写：

```
决策记录：monthly_spend 列缺失处理
- 缺失率：12.3%
- 机制判断：MAR（缺失率随用户等级升高而增加，可能高等级用户不愿透露）
- 选择策略：按用户等级分组，用组内中位数填充
- 理由：中位数对极端值稳健；分组填充利用 MAR 信息，比全局填充更准确
- 替代方案：删除（会损失 12% 样本，且可能系统性偏斜）；模型预测（样本量不足，担心过拟合）
```

这就是数据清洗的核心：**不是把数据变"干净"，而是把"怎么变"的过程说清楚**。阿码现在理解了：AI 可以帮你生成填充代码，但"为什么选这个策略"必须由你来回答——因为涉及业务判断。

---

## 3. 异常点不一定是错误

上周你用箱线图看到了一些"须"外面的点——那些远离主体的数据点。现在问题来了：它们是错误？是 VIP 用户？还是某个群体的正常特征？

**异常值检测与处理**（outliers）不是"删掉离群点"那么简单。你需要先回答：这个点为什么异常？

统计方法可以帮你"发现"异常点，但无法告诉你"为什么"。常用的统计方法有：

**IQR 方法**：基于箱线图的规则，把低于 Q1 - 1.5×IQR 或高于 Q3 + 1.5×IQR 的点标记为异常。这是描述性统计的延伸——它告诉你"这个点在分布的极端位置"。

**Z-score 方法**：计算数据点与均值的距离（以标准差为单位），通常把 |Z| > 3 的点视为异常。这个方法假设数据近似正态分布——如果数据是严重偏斜的，Z-score 可能误报或漏报。

```python
import pandas as pd
import numpy as np

# IQR 方法检测异常值
Q1 = df['monthly_spend'].quantile(0.25)
Q3 = df['monthly_spend'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[(df['monthly_spend'] < lower_bound) |
                  (df['monthly_spend'] > upper_bound)]
print(f"IQR 方法检测到 {len(outliers_iqr)} 个异常点")

# Z-score 方法（假设近似正态）
from scipy import stats
z_scores = np.abs(stats.zscore(df['monthly_spend'].dropna()))
outliers_z = df[z_scores > 3]
print(f"Z-score 方法检测到 {len(outliers_z)} 个异常点")

# 看看这些"异常点"是谁
print("\n异常点特征：")
print(outliers_iqr[['user_id', 'user_level', 'monthly_spend', 'registration_days']].head(10))
```

运行这段代码，你可能会发现：那些"异常高消费"的用户，注册天数很长、用户等级很高。他们不是错误——他们是 VIP。删掉他们就删掉了最有价值的客户。

这时候需要引入**业务规则**：
- 月消费 > 10000 且注册天数 < 7 → 可能是测试账号或刷单，需要删除
- 月消费 > 10000 且用户等级 = "钻石" → 正常 VIP，保留
- 年龄 > 120 或年龄 < 0 → 明显录入错误，删除或标记为缺失

阿码问："能不能让 AI 自动判断哪些是错误、哪些是 VIP？"

答案是：AI 可以帮你写检测代码，但"错误 vs VIP"的判断需要业务知识。AI 不知道"钻石用户"是什么意思，也不知道"注册 7 天内消费过万"在业务上是否合理。这就是**统计方法 vs 业务规则**的分工：统计方法告诉你"哪里异常"，业务规则告诉你"怎么处理"。

关键决策：**修数据**还是**修解释**？
- 如果是录入错误（年龄 = -5），修数据（删除或修正）
- 如果是真实的极端值（VIP 用户），修解释（单独分析、分层建模、或对数变换）

```python
# 分类处理异常值
def classify_outlier(row):
    spend = row['monthly_spend']
    days = row['registration_days']
    level = row['user_level']

    if spend > 10000 and days < 7:
        return "suspicious"  # 疑似刷单
    elif spend > 10000 and level == "diamond":
        return "VIP"  # 正常高价值用户
    elif spend > upper_bound:
        return "high_spend"  # 高消费，需进一步分析
    else:
        return "normal"

df['outlier_type'] = df.apply(classify_outlier, axis=1)
print(df['outlier_type'].value_counts())

# 不同类别，不同处理
df_clean = df[df['outlier_type'] != 'suspicious'].copy()  # 删除疑似刷单
print(f"删除 {len(df) - len(df_clean)} 条可疑记录，剩余 {len(df_clean)} 条")
```

回顾桥：上周的箱线图在这里再次出场——它不只是"展示"工具，更是"决策"工具。箱线图告诉你异常点集中在哪些分组，帮你判断"修数据"还是"修解释"。

---

> **AI 时代小专栏：特征工程中的 AI 辅助与人类判断**
>
> AI 工具在特征工程领域进展迅速。2025 年，AutoML 平台已经能自动完成特征缩放、编码、甚至构造交互特征。但这里有个陷阱：AI 可以生成代码，却无法理解业务语义。
>
> 举个例子：AI 看到"年龄"和"收入"两列，可能会自动做标准化。但它不知道，在某些业务场景下，年龄需要分箱（如 18-25、26-35）才有意义；也不知道收入可能需要对数变换才能反映"收入差距的感知"而非"绝对差距"。
>
> 更隐蔽的问题是**数据泄漏**：AI 可能在构造特征时不小心使用了目标变量的信息，或者把测试集的信息泄露到训练集。这些错误不会报错，但会让你的模型评估完全失效。
>
> 所以正确的协作方式是：让 AI 帮你写特征变换的代码模板，但"哪些特征需要变换、用什么变换、为什么"必须由你来决定。人类的角色是"语义把关者"——确保每个变换都符合业务逻辑，而不是盲目追求技术指标。
>
> 参考（访问日期：2026-02-11）：
> - [Scoop Analytics: Human-in-the-Loop AI 2026](https://www.scoopanalytics.com/blog/human-in-the-loop-hitl)
> - [Torry Harris: Human-On-The-Loop AI 2026](https://www.torryharris.com/insights/articles/human-on-the-loop-ai)
> - [Springer: Human-in-the-Loop Machine Learning State of the Art](https://link.springer.com/article/10.1007/s10462-022-10246-w)

---

## 4. 为什么模型"听不见"年龄的声音？

小北兴冲冲地跑了一个预测模型，想看出哪些因素会影响用户消费。结果出来后，他盯着屏幕直挠头："年龄的系数怎么几乎是 0？这不合理啊，老年人消费能力和年轻人肯定不一样。"

阿码凑过来看了一眼："你是不是同时放了年龄和收入进去？"

"对啊。"

"那模型当然'听不见'年龄的声音了。"阿码在纸上画了两个数，"年龄 18-80，收入 3000-50000，差了两个数量级。模型在算距离的时候，收入的变化会完全淹没年龄的变化——就像一个人在喊，另一个人在用扩音器喊。"

小北恍然大悟："所以不是年龄不重要，是收入的数值太大了？"

正是如此。这就是**特征缩放**（feature scaling）的意义：让不同特征处于可比较的尺度，避免"量大的欺负量小的"。

这就是**特征缩放**（feature scaling）的意义：让不同特征处于可比较的尺度，避免"量大的欺负量小的"。

两种最常用的缩放方法：

**StandardScaler（标准化）**：将数据变换为均值为 0、标准差为 1 的分布。公式是 `(x - mean) / std`。适用于数据近似正态分布的场景，也是大多数统计模型的默认假设。

**MinMaxScaler（归一化）**：将数据线性缩放到 [0, 1] 区间。公式是 `(x - min) / (max - min)`。适用于有明确边界的数据，或需要保留零值意义的场景（如"零消费"和"低消费"是有区别的）。

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

# 原始数据
age_income = df[['age', 'monthly_income']].copy()
print("缩放前：")
print(age_income.describe())

# StandardScaler（标准化）
scaler_std = StandardScaler()
age_income_std = pd.DataFrame(
    scaler_std.fit_transform(age_income),
    columns=['age_std', 'income_std']
)
print("\n标准化后（均值为 0，标准差为 1）：")
print(age_income_std.describe().round(2))

# MinMaxScaler（归一化）
scaler_minmax = MinMaxScaler()
age_income_norm = pd.DataFrame(
    scaler_minmax.fit_transform(age_income),
    columns=['age_norm', 'income_norm']
)
print("\n归一化后（范围 [0, 1]）：")
print(age_income_norm.describe().round(2))
```

什么时候用哪个？
- 数据近似正态 → StandardScaler
- 数据有明确边界（如百分比、像素值）→ MinMaxScaler
- 数据有异常值 → 考虑 RobustScaler（基于中位数和 IQR，对异常值稳健）

阿码看着这三种选择，又忍不住了："那 AI 能不能自动帮我选哪个 scaler？"

老潘正好路过，回了一句："它可以写代码，但不知道你的数据有没有边界。比如年龄理论上界是 120，但收入的上界是多少？没人知道。"

阿码点点头："所以还是要人来判断……"

"对，"老潘说，"而且记住：scaler 要在训练集上 fit，在测试集上只 transform。这是下一周建模的内容了。"

接下来是**特征编码**（encoding）。还记得 Week 01 的数据类型吗？分类型变量（如"用户等级"、"城市"）需要转换成数值才能进入模型，但编码方式会影响结果。

**One-hot 编码**：把 k 个类别变成 k 个二元列（0/1）。适合 nominal 类别（无顺序关系），如城市、颜色。缺点是维度会增加，如果类别太多（如"用户 ID"），会导致维度灾难。

**Label 编码**：把类别映射为整数（0, 1, 2...）。适合 ordinal 类别（有顺序关系），如"低/中/高"等级。注意：不要对 nominal 类别用 label 编码——模型会误以为"北京=1，上海=2"意味着上海是北京的"两倍"。

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# One-hot 编码（适合 nominal 类别）
cities = df[['city']].copy()
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' 避免多重共线性
cities_encoded = encoder.fit_transform(cities)
cities_df = pd.DataFrame(
    cities_encoded,
    columns=encoder.get_feature_names_out(['city'])
)
print("One-hot 编码结果：")
print(cities_df.head())

# Label 编码（仅适合 ordinal 类别，或作为目标变量）
# 假设 user_level 是 ordinal：bronze < silver < gold < diamond
level_mapping = {'bronze': 0, 'silver': 1, 'gold': 2, 'diamond': 3}
df['user_level_encoded'] = df['user_level'].map(level_mapping)
print("\nLabel 编码（ordinal）：")
print(df[['user_level', 'user_level_encoded']].drop_duplicates())
```

老潘看了一眼这段代码，点点头："在公司里，我们把变换步骤写成函数，方便复用和版本控制。"

```python
# 老潘推荐的工程实践：把变换封装成函数
def preprocess_features(df):
    """特征预处理流水线，返回处理后的 DataFrame 和变换器（用于新数据）。"""
    df = df.copy()

    # 数值特征缩放
    numeric_cols = ['age', 'monthly_income']
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 分类型特征编码
    categorical_cols = ['city']
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )

    # 合并
    df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    return df, {'scaler': scaler, 'encoder': encoder}

# 使用
df_processed, transformers = preprocess_features(df)
```

回顾桥：Week 01 的数据类型在这里再次出场——数值型和分类型需要完全不同的处理方式。数据类型决定了你该用 scaler 还是 encoder，这是"先理解数据，再处理方法"的又一次体现。

---

## 5. 把清洗过程写进报告

老潘看完你的清洗代码，问了一个更尖锐的问题："**如果三个月后有人质疑你的结论，你能证明数据是怎么处理的吗？**"

这不是抬杠。在真实项目中，数据清洗决策是"可审计性"的核心。你需要记录：
- 发现了什么问题
- 选择了什么策略
- 为什么这样选择
- 考虑过什么替代方案

这就是**清洗决策日志**（cleaning log）——它是数据卡（Week 01）的"质量扩展"，让你的报告从"我分析了什么"升级为"我在什么前提下分析了什么"。

```python
import pandas as pd
from datetime import datetime

def generate_cleaning_log(decisions):
    """生成清洗决策日志，用于写入 report.md。"""
    log = []
    log.append("# 数据清洗决策日志\n")
    log.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    for i, decision in enumerate(decisions, 1):
        log.append(f"## 决策 {i}：{decision['field']}\n")
        log.append(f"- **问题描述**：{decision['problem']}\n")
        log.append(f"- **处理策略**：{decision['strategy']}\n")
        log.append(f"- **选择理由**：{decision['rationale']}\n")
        log.append(f"- **替代方案**：{decision['alternatives']}\n")
        log.append(f"- **影响评估**：{decision['impact']}\n\n")

    return "\n".join(log)

# 记录本周的所有清洗决策
decisions = [
    {
        'field': 'monthly_spend',
        'problem': '缺失率 12.3%，缺失率随用户等级升高而增加',
        'strategy': '按用户等级分组，用组内中位数填充',
        'rationale': '中位数对极端值稳健；分组填充利用 MAR 信息',
        'alternatives': '删除（会损失 12% 样本）；全局均值填充（会压缩方差）',
        'impact': '填充后均值从 245 元变为 267 元，标准差从 180 变为 165'
    },
    {
        'field': 'monthly_spend',
        'problem': '检测到 23 个 IQR 异常点',
        'strategy': '分类处理：删除疑似刷单（高消费+新用户），保留 VIP 用户',
        'rationale': '异常点不一定是错误，需要结合业务规则判断',
        'alternatives': '全部删除（会损失真实 VIP）；全部保留（会污染模型）',
        'impact': '删除 3 条可疑记录，保留 20 条 VIP 记录'
    },
    {
        'field': 'age, monthly_income',
        'problem': '特征尺度差异大（18-80 vs 3000-50000）',
        'strategy': 'StandardScaler 标准化',
        'rationale': '数据近似正态分布，标准化后均值为 0、标准差为 1',
        'alternatives': 'MinMaxScaler（适合有界数据）；RobustScaler（如果有异常值）',
        'impact': '两特征现在处于相同尺度，可公平比较'
    }
]

# 生成日志
log_content = generate_cleaning_log(decisions)
print(log_content)

# 追加到 report.md
with open('report.md', 'a', encoding='utf-8') as f:
    f.write('\n\n' + log_content)
```

老潘的建议是：清洗日志和代码一样重要，甚至更重要。因为代码会迭代，但决策理由需要被记录——否则三个月后，连你自己都忘了"当时为什么删那些行"。

回顾桥：Week 01 的数据卡在这里完成闭环。数据卡告诉你"数据从哪来"，清洗日志告诉你"数据怎么变"——两者合在一起，才是完整的"数据说明书"。

---

## StatLab 进度

到目前为止，StatLab 的报告已经有了数据卡（数据来源、字段说明）和描述统计（均值/中位数/标准差/IQR、分布图）。但它还有一个"看不见的坑"：读者不知道这些数据经历了什么处理，也不知道结论对"缺失值如何处理"有多敏感。

本周的改进是把"清洗决策日志"写进 report.md。这不是负担——它是你所有后续结论的"可信度背书"。

```python
# examples/03_statlab_cleaning.py
import pandas as pd
from datetime import datetime

def add_cleaning_section_to_report(report_path, decisions):
    """在 report.md 中添加数据清洗章节。"""

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到插入位置（在描述统计之后）
    insert_marker = "## 描述统计"
    if insert_marker not in content:
        insert_marker = "# 数据卡"

    # 生成清洗章节
    cleaning_section = f"""

## 数据清洗与预处理

> 本章记录所有数据清洗决策，确保分析过程可复现、可审计。

### 缺失值处理

| 字段 | 缺失率 | 机制判断 | 处理策略 | 理由 |
|------|--------|----------|----------|------|
"""

    for d in decisions:
        if '缺失' in d['problem']:
            cleaning_section += f"| {d['field']} | {d['problem'].split('缺失率 ')[1].split('，')[0]} | MAR | {d['strategy']} | {d['rationale']} |\n"

    cleaning_section += """
### 异常值处理

| 字段 | 检测方法 | 异常数 | 处理策略 |
|------|----------|--------|----------|
"""

    for d in decisions:
        if '异常' in d['problem']:
            cleaning_section += f"| {d['field']} | IQR | 23 | {d['strategy']} |\n"

    cleaning_section += """
### 特征变换

| 字段 | 变换类型 | 方法 | 理由 |
|------|----------|------|------|
| age, monthly_income | 缩放 | StandardScaler | 特征尺度差异大，标准化后公平比较 |
| city | 编码 | OneHotEncoder | Nominal 类别，避免序关系假设 |

---
*清洗日志生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""

    # 插入到报告末尾
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(cleaning_section)

    print(f"清洗章节已追加到 {report_path}")

# 使用示例
if __name__ == "__main__":
    decisions = [
        {
            'field': 'monthly_spend',
            'problem': '缺失率 12.3%，缺失率随用户等级升高而增加',
            'strategy': '按用户等级分组，用组内中位数填充',
            'rationale': '中位数对极端值稳健'
        }
    ]
    add_cleaning_section_to_report('report.md', decisions)
```

现在你的 report.md 有了三个层次：
1. 数据卡：数据从哪来、字段什么意思
2. 描述统计：数据长什么样、分布如何
3. 清洗日志：数据怎么处理的、为什么这样处理

老潘看到这份报告会说什么？"这才像一份能对外交付的分析。"

---

## Git 本周要点

本周必会命令：
- `git status`：查看工作区状态
- `git diff`：查看具体改动内容
- `git add -A`：添加所有改动
- `git commit -m "feat: add data cleaning log"`：提交改动
- `git log --oneline -n 5`：查看最近 5 条提交

常见坑：
- 清洗后的数据文件太大：建议用 `.gitignore` 排除 `data/processed/`，只提交清洗脚本
- 不写清洗决策理由：三个月后你自己也忘了为什么删那些行
- 覆盖原始数据：永远保留原始数据，清洗结果存到新文件

---

## 本周小结（供下周参考）

本周你做了四件事：识别缺失值的三种机制（MCAR/MAR/MNAR），理解"删掉还是填补"是个需要记录的决策；用统计方法（IQR/Z-score）和业务规则检测异常值，学会区分"修数据"（录入错误）和"修解释"（真实但极端）；对数值特征进行标准化/归一化、对类别特征进行 one-hot/label 编码，让它们在分析中公平竞争；最重要的是，你把所有这些决策写进了清洗日志——让过程可复现、可审计。

这就是数据清洗的核心：**不是把数据变"干净"，而是把"怎么变"的过程说清楚**。

下周是探索性数据分析（EDA）综合。你会发现：当你真正理解数据的质量和边界后，相关分析和分组比较就不再是"跑代码"，而是有据可依的探索。这周的清洗日志不是负担——它是你下周所有结论的"可信度背书"。

---

## Definition of Done（学生自测清单）

- [ ] 我能解释 MCAR、MAR、MNAR 三种缺失机制的区别
- [ ] 我能根据缺失机制选择合适的处理策略（删除/填充/预测）
- [ ] 我能用 IQR 或 Z-score 方法检测异常值，并解释为什么选择该方法
- [ ] 我能区分"修数据"（录入错误）和"修解释"（真实但极端）的场景
- [ ] 我能对数值特征进行标准化或归一化，并解释何时需要缩放
- [ ] 我能对类别特征进行 one-hot 或 label 编码，避免编码陷阱
- [ ] 我为数据集生成了一份"清洗决策日志"，记录所有步骤和理由
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解"数据清洗不是终点，而是后续分析的起点"
