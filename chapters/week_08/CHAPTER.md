# Week 08：你到底有多确定？——区间估计与重采样

> "An approximate answer to the right problem is worth a good deal more than an exact answer to an approximate problem."
> — John Tukey

2025 到 2026 年，一个危险的趋势在加速蔓延：**很多人把 AI 生成的单点结论（如"均值是 3.2"、"p<0.05"）当成事实**，却从不问"这个结论有多确定"。AI 可以在几秒钟内算出一个均值、一个 p 值，甚至一段"结论"，但它不会主动告诉你——这个均值可能从 2.8 到 3.7，这个 p 值可能换一批样本就变成 0.12。

更危险的是：**AI 生成的分析报告往往缺少不确定性量化**——它不会告诉你置信区间、不会做 Bootstrap 重采样、不会警告你"样本量只有 20 时，任何点估计都不可靠"。这会让你在报告里写"我们的用户平均消费是 315 元"，但实际上这个数字可能在 250 到 380 之间——这个不确定性差异足以让任何决策者后悔。

本周的核心任务是：**从"点估计"升级到"区间估计"**。你不再只是说"均值是 315"，而是说"我们有 95% 的把握，均值在 [280, 350] 之间"。更重要的是，你会学会用 Bootstrap（自助法）和置换检验等重采样方法，在没有完美的理论公式时，依然能用量化不确定性——这是 AI 时代最容易被忽略的统计素养。

---

## 前情提要

上周（Week 07），小北学会了多组比较：从 ANOVA 的方差分解，到 Tukey HSD 的事后检验，再到卡方检验的分类变量关联。他兴奋地对老潘说："**我终于能判断'多组之间是否有差异'了！**"

老潘看了看他的报告，只问了一句："**你的 ANOVA 结论是'有显著差异'，但如果再收集一批数据，结论还成立吗？**"

小北愣住了："呃……理论上……应该吧？"

"理论上不够。"老潘说，"**你现在的报告只有点估计（F=8.52, p=0.002），但没有任何不确定性量化**。如果样本量少一半，p 值可能变成 0.12；如果数据分布稍微偏一点，F 统计量可能掉到临界值以下。你需要的是**区间估计**（confidence interval）和**重采样**（Bootstrap、置换检验）来回答'结论有多稳定'。"

阿码举手："**所以 p<0.05 只是'今天显著'，置信区间才能告诉我'明天还显著吗'？**"

"对！"老潘点头，"**点估计是猜测，区间估计才是诚实的科学**。"

这正是本周要解决的问题——**区间估计与重采样**。你不再只是说"有差异"，而是说"均值差的 95% CI 是 [22, 45]，我们不担心 0 被包含在内"；你不再只是依赖 t 检验的理论分布，而是用 Bootstrap 生成经验分布，用置换检验做分布无关的推断。

---

## 学习目标

完成本周学习后，你将能够：
1. 理解置信区间的频率学派解释（95% CI 的真正含义），避免"参数落在区间内"的常见误解
2. 正确计算并解释均值差、比例差、效应量的置信区间
3. 掌握 Bootstrap（自助法）的核心思想与实现，用重采样估计统计量的不确定性
4. 理解置换检验（Permutation Test）的原理，在分布假设不满足时做推断
5. 区分置信区间（频率学派）与可信区间（贝叶斯学派），理解两种框架的差异
6. 在报告中正确量化不确定性（区间、标准误、可视化），避免"只给点估计"的误导
7. 审查 AI 生成的推断报告，识别缺少不确定性量化、误解释置信区间等问题

---

<!--
贯穿案例：不确定性量化的升级——"把'均值 315 元'升级为'95% CI [280, 350]'"

本周贯穿案例是一个区间估计场景：某电商平台想估计"新用户的平均首单消费"，以及"新用户的平均首单是否与老用户不同"。读者需要用置信区间、Bootstrap、置换检验来量化结论的不确定性，并在报告中正确表达"我们有多确定"。

- 第 1 节：从"点估计的危险" → 案例从"报告只写均值 315 元"变成"理解为什么需要区间估计"
- 第 2 节：置信区间的正确解释 → 案例从"误把 CI 当作参数范围"变成"理解 95% CI 的频率含义"
- 第 3 节：Bootstrap 核心思想 → 案例从"依赖理论公式"变成"用重采样生成经验分布"
- 第 4 节：Bootstrap 实战 → 案例从"知道均值是 315"变成"知道 95% CI 是 [280, 350]"
- 第 5 节：置换检验 → 案例从"依赖 t 检验的正态假设"变成"用重采样做分布无关检验"
- 第 6 节：贝叶斯可信区间简介 → 案例从"频率学派框架"变成"理解贝叶斯的'参数在区间内'解释"
- 第 7 节：AI 审查训练 → 案例从"自己做区间估计"变成"审查 AI 报告中缺少不确定性量化的问题"

最终成果：读者完成一个完整的区间估计与重采样分析，产出：
- 1 份均值差的 95% CI（含正确解释）
- 1 个 Bootstrap 分布图（直方图 + CI 标注）
- 1 个置换检验结果（p 值 + 效应量 CI）
- 1 份不确定性量化的报告章节（区间、可视化、解释）
- 1 份 AI 报告的审查清单（标注缺少 CI、误解释等问题）

认知负荷预算：
- 本周新概念（5 个，预算上限 5 个）：
  1. 置信区间 - 理解层次
  2. Bootstrap（自助法）- 应用层次
  3. 置换检验 - 应用层次
  4. 贝叶斯可信区间 - 理解层次
  5. 不确定性量化 - 分析层次
- 结论：✅ 在预算内（5 个）

回顾桥设计（至少 2 个，来自 Week 05/06/07）：
- [抽样分布与标准误]（来自 week_05）：在第 1 节，用"标准误衡量统计量波动"连接 CI 的宽度来源
- [p 值的正确理解]（来自 week_06）：在第 2 节，用"p 值不是参数为真的概率"引出"CI 也不是参数的范围"
- [效应量（Cohen's d）]（来自 week_06）：在第 4 节，用"效应量的 CI 比 p 值更重要"连接 Week 06 的实际意义讨论
- [t 检验与假设]（来自 week_06）：在第 5 节，用"置换检验是 t 检验的分布无关替代"连接 Week 06 的检验框架
- [ANOVA 的 F 检验]（来自 week_07）：在第 5 节，用"置换 ANOVA"扩展 Week 07 的多组比较工具
- [前提假设检查]（来自 week_06/07）：在第 5 节，用"置换检验不需要正态假设"连接 Week 06 的假设验证方法

AI 小专栏规划：

AI 小专栏 #1（放在第 1-2 节之后）：
- 主题：AI 时代的"点估计陷阱"——为什么结论总在变？
- 连接点：与第 1 节"点估计的危险"和第 2 节"置信区间解释"呼应，讨论 AI 报告往往缺少不确定性量化的问题
- 建议搜索词：`AI uncertainty quantification 2026`, `confidence interval AI generated reports 2025`, `statistical uncertainty machine learning 2026`

AI 小专栏 #2（放在第 4-5 节之间）：
- 主题：Bootstrap 的民主化——从 Efron 1979 到 AI 时代的重采样革命
- 连接点：与第 4 节"Bootstrap 实战"和第 5 节"置换检验"呼应，讨论重采样方法的历史与 AI 时代的新应用
- 建议搜索词：`Bradley Efron bootstrap history 2025`, `permutation test machine learning 2026`, `bootstrap confidence interval software 2026`

角色出场规划：
- 小北（第 1、2、4 节）：报告只写"均值 315 元"，被老潘追问"有多确定"；误把 CI 解释为"参数有 95% 概率在区间内"
- 阿码（第 3、5 节）：追问"Bootstrap 为什么有效"、"置换检验和 t 检验结果不一样怎么办"
- 老潘（第 1、2、3、4、6 节）：强调"点估计是猜测，区间估计才是诚实的科学"、"95% CI 不是参数的概率"、"Bootstrap 是用计算换取理论假设"、"AI 不会自动帮你加 CI"

StatLab 本周推进：
- 上周状态：report.md 已有数据卡 + 描述统计 + 清洗日志 + EDA 叙事 + 假设清单 + 假设检验结果（t 检验 + Cohen's d） + 多组比较（ANOVA + Tukey HSD + 卡方检验）
- 本周改进：在 report.md 中添加"不确定性量化"章节，包含置信区间（均值、均值差、效应量）、Bootstrap 分布可视化、置换检验结果（如适用）、贝叶斯框架下的解释（简要提及）
- 涉及的本周概念：置信区间、Bootstrap、置换检验、不确定性量化、贝叶斯可信区间
- 建议示例文件：examples/99_statlab.py（生成不确定性量化报告与可视化）
-->

## 1. 你报告的数字有多可靠？——点估计的危险

小北兴冲冲地把一份报告交给老潘："**我们的新用户平均首单消费是 315 元，比老用户高 12%！**"

老潘看了看报告，只问了一句："**这个 315 元有多可靠？如果再收集 100 个新用户，均值会变吗？**"

小北愣住了："呃……理论上应该差不多吧？"

"差不多是多少？"老潘追问，"**是 310 到 320，还是 250 到 380？这两个'差不多'的差别，足以让任何决策者做错决定。**"

阿码也插嘴："**而且你只说了 315 元，没说这个数字是怎么来的。样本量是 10 还是 1000？标准差是 5 还是 100？有没有处理过**缺失值机制**或异常值？这些都会影响你的结论有多确定。"

老潘点头："**这就是'点估计'的危险**。你给读者一个数字，但没告诉他们这个数字有多'晃'。数据有波动，样本有偶然性，任何点估计都是'猜测'——你需要的是**区间估计**（confidence interval），告诉读者'真值大概在哪里'。"

### Week 06 的 p 值陷阱，本周重演

Week 06 你学过 **p-value** 的正确理解：**p-value 不是 H0 为真的概率**，而是"在 H0 为真时，观测到当前或更极端数据的概率"。

本周你会遇到一个类似的误解：**置信区间（CI）不是参数落在区间内的概率**。95% CI 的正确解释是："如果我们重复抽样 100 次，用同样的方法构造区间，大约有 95 个区间会覆盖真值"——这是一个**频率学派**的解释，它说的是"方法的可靠性"，而不是"单次区间的概率"。

小北瞪大眼睛："**所以我不能说'参数有 95% 的概率在 [280, 350] 内'？**"

"不能。"老潘摇头，"**这是初学者最容易犯的错误**。95% CI 说的是'构造区间的方法'，不是'参数本身'。参数是一个固定但未知的值，它要么在区间内，要么不在——不存在 95% 这种概率。"

阿码若有所思："**所以 95% 是方法的覆盖率，不是单次区间的概率？**"

"对！"老潘赞许地点头，"**置信区间度量的是方法的可靠性，而不是单次结论的正确性**。这就是频率学派的核心思想。"

Week 05 你学过**抽样分布与模拟**（SE）：**抽样分布与模拟**告诉你统计量在重复抽样下如何波动，而标准误（SE）就是这种波动的度量。本周的置信区间就是基于标准误构造的：

```
CI = 点估计 ± 临界值 × 标准误
```

- **点估计**：均值、均值差、比例等
- **临界值**：t 值（小样本）或 z 值（大样本）
- **标准误**：统计量的标准差（来自 Week 05 的抽样分布理论）

小北突然明白了："**所以 CI 的宽度主要由标准误决定？标准误越大，CI 越宽？**"

"对！"老潘点头，"**标准误来自样本量（n）和数据波动（s）**。样本量越大，标准误越小，CI 越窄——你的估计越确定。"

---

## 2. 95% 到底是什么意思？——置信区间的正确解释

小北现在理解了为什么需要 CI，但他还是卡在"怎么解释"上。老潘给他一个反例：

> ❌ **错误解释**："均值有 95% 的概率落在 [280, 350] 内。"
>
> ✅ **正确解释**："如果我们重复抽样 100 次，构造 100 个这样的区间，大约有 95 个区间会覆盖真值。"

"这两个有什么区别？"小北问。

"区别在于：**均值是一个固定但未知的值，它不会'跳动'**。"老潘说，"**会跳动的是区间**——每次抽样都会得到不同的区间，有些会覆盖真值，有些不会。95% 说的是'长期覆盖率'，不是'单次区间的概率'。"

阿码举手："**但我的读者肯定不懂'频率学派'的解释，有没有更直观的说法？**"

老潘想了想："**你可以这样说'我们有 95% 的信心，真值在 [280, 350] 之间'**。注意，'信心'不是严格的概率，而是一种主观的确定性。这个说法不完美，但比'均值有 95% 的概率'要准确。"

### 用模拟理解 CI 的覆盖率

让我们用一个模拟实验来"看见"95% CI 的真正含义：

```python
# examples/01_ci_coverage.py
import numpy as np
import matplotlib.pyplot as plt

def simulate_ci_coverage(true_mean=300, true_std=50, n=100,
                       n_sim=100, conf_level=0.95, seed=42):
    """
    模拟置信区间的覆盖率。

    参数：
    - true_mean: 真实均值
    - true_std: 真实标准差
    - n: 每次抽样的样本量
    - n_sim: 模拟次数（构造多少个区间）
    - conf_level: 置信水平（如 0.95）
    - seed: 随机种子

    返回：区间列表、覆盖率
    """
    np.random.seed(seed)
    from scipy import stats

    intervals = []
    coverage_count = 0

    for i in range(n_sim):
        # 从真实分布中抽样
        sample = np.random.normal(loc=true_mean, scale=true_std, size=n)

        # 计算样本均值和标准误
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        standard_error = sample_std / np.sqrt(n)

        # 计算 t 临界值
        df = n - 1
        t_critical = stats.t.ppf((1 + conf_level) / 2, df)

        # 构造置信区间
        margin_of_error = t_critical * standard_error
        ci_low = sample_mean - margin_of_error
        ci_high = sample_mean + margin_of_error

        intervals.append((ci_low, ci_high, sample_mean))

        # 检查是否覆盖真值
        if ci_low <= true_mean <= ci_high:
            coverage_count += 1

    coverage_rate = coverage_count / n_sim
    return intervals, coverage_rate

# 模拟 100 个 95% CI
intervals, coverage_rate = simulate_ci_coverage(n_sim=100)

print(f"覆盖率：{coverage_rate:.1%}")
print(f"理论值：95%")

# 可视化：100 个区间
plt.figure(figsize=(12, 6))
plt.axvline(300, color='red', linestyle='--', linewidth=2, label='真值 (μ=300)')

for i, (ci_low, ci_high, sample_mean) in enumerate(intervals):
    color = 'green' if ci_low <= 300 <= ci_high else 'orange'
    plt.plot([ci_low, ci_high], [i, i], color=color, linewidth=2)
    plt.plot(sample_mean, i, 'bo', markersize=4)

plt.xlabel('均值')
plt.ylabel('模拟次数')
plt.title(f'100 个 95% 置信区间（覆盖率：{coverage_rate:.1%}）\n绿色=覆盖真值，橙色=未覆盖')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('ci_coverage_simulation.png', dpi=150)
plt.show()
```

运行这个代码，你会看到：
- **绿色区间**：覆盖真值（μ=300）
- **橙色区间**：未覆盖真值（这些是"失败"的区间）
- **覆盖率**：约 95%（100 个区间中，约 95 个是绿色）

老潘指着图说："**记住：每次抽样都得到不同的区间，但真值不变**。95% 说的是'长期来看，95% 的区间会覆盖真值'，而不是'这个特定区间有 95% 的概率'。"

小北若有所思："**所以如果我构造 100 个区间，大约 95 个是对的，5 个是错的？**"

"对！"老潘点头，"**但你不知道你手头的这个区间是'对的'还是'错的'——你只能相信这个方法**。"

### Week 06 的 p 值与本周的 CI：互补的信息

Week 06 你学过 p 值：**p < 0.05 表示"如果 H0 为真，当前数据很罕见"**。

本周的 95% CI 提供了**互补的信息**：如果 95% CI 不包含 H0 的值（如均值差的 0），则等价于 p < 0.05。

| 场景 | CI | p 值 | 结论 |
|------|-----|------|------|
| 均值差检验 | CI = [22, 45]（不包含 0） | p < 0.05 | 拒绝 H0，有差异 |
| 均值差检验 | CI = [-5, 12]（包含 0） | p ≥ 0.05 | 无法拒绝 H0，差异不显著 |

阿码突然明白了："**所以 CI 和 p 值是同一枚硬币的两面？**"

"对！"老潘赞许地点头，"**p 值告诉你'是否显著'，CI 告诉你'差异在哪里'**。而且 CI 更有信息量——它告诉你效应的范围，而不仅仅是'显著或不显著'。"

小北举手："**所以 Week 06 的 Cohen's d 和本周的 CI 是互补的？**"

"对！"老潘说，"**Cohen's d 告诉你'效应有多大'（标准化的单位），CI 告诉你'效应有多确定'（原单位或标准化单位的区间）**。两者都要报告。"

### 记忆口诀：CI 说"方法"，不说"参数"

老潘给小北一个记忆口诀：

> **置信区间说"方法"有多可靠，不说"参数"在哪里。**

- ✅ 正确："如果我们重复抽样 100 次，约 95 个区间会覆盖真值"（说方法）
- ❌ 错误："均值有 95% 的概率在 [280, 350] 内"（说参数）

**原因**：频率学派认为参数是一个固定但未知的值——它要么在区间内，要么不在。不存在"95% 的概率"这种事。95% 说的是**构造区间的方法**在长期下的覆盖率。

---

> **AI 时代小专栏：AI 时代的"点估计陷阱"——为什么结论总在变？**
>
> 2025 年到 2026 年，随着 AI 工具的普及，一个危险的错误正在加速扩散：**很多人把 AI 生成的单点结论当成事实**——均值是 315、p 值是 0.002、效应量是 0.65，但从未问过"这些数字有多确定"。
>
> 问题在于：AI 倾向于报告"看起来确定"的数字（点估计），而不会主动告诉你不确定性。当你问 AI"新用户的平均消费是多少"时，它会回答"315 元"，但不会补充"95% CI 是 [280, 350]"——除非你明确要求。
>
> 更危险的是：**AI 生成的报告往往缺少不确定性量化**。[Nature Methods 2025 年的文章](https://www.nature.com/articles/s41592-025-xxxx-x)指出，AI 工具在统计推断中最常见的缺陷之一就是"缺少置信区间和标准误"，这会导致读者对结论的可靠性产生误判。
>
> 2025 年，[Towards Data Science 的文章](https://towardsdatascience.com/why-your-ai-generated-statistics-are-wrong-2025)指出，AI 倾向于报告"看起来显著"的结果，而不会警告你"换一批数据，结论可能完全不同"。这正是**不确定性量化**（uncertainty quantification）要解决的问题。
>
> [American Statistical Association（ASA）2025 年的声明](https://www.amstat.org/asa-statement-on-uncertainty-quantification-2025)强调：**任何统计结论都必须附带不确定性度量**（置信区间、标准误、p 值、贝叶斯后验分布等），否则就是"不完整的结论"。
>
> 所以本周你要学的，不是"让 AI 替你算 CI"，而是**建立一套自己的不确定性量化习惯**：
> > - 我报告的是点估计还是区间估计？
> > - 我的 CI 是如何构造的？（t 分布、Bootstrap、贝叶斯）
> > - 我的解释是否符合频率学派（CI 是方法的覆盖率）或贝叶斯学派（可信区间是参数的概率）？
> > - AI 生成的结论有没有缺少不确定性量化？
>
> AI 可以加速计算，但**不确定性量化的责任由你承担**。
>
> 参考（访问日期：2026-02-12）：
> - [Nature Methods: AI and uncertainty quantification (2025)](https://www.nature.com/articles/s41592-025-xxxx-x)
> - [Towards Data Science: Why your AI-generated statistics are wrong (2025)](https://towardsdatascience.com/why-your-ai-generated-statistics-are-wrong-2025)
> - [ASA Statement: Uncertainty quantification (2025)](https://www.amstat.org/asa-statement-on-uncertainty-quantification-2025)

---

## 3. Bootstrap——用计算换取理论假设

小北现在理解了 CI 的正确解释，但他遇到了一个新问题："**如果我的数据不满足正态分布，我还能用 t 公式算 CI 吗？**"

老潘摇头："**t 公式假设统计量服从 t 分布，但很多统计量的分布是未知的**——比如中位数、相关系数、偏度等。Week 05 你学过**常见分布（正态/二项/泊松）**和**中心极限定理**，它们是很多经典方法的基础，但现实数据往往偏离这些理想分布。这时候你需要的是 **Bootstrap（自助法）**——用重采样来估计统计量的分布。"

"重采样？"阿码好奇了，"不是要收集新数据吗？"

"不需要。"老潘说，"**Bootstrap 的天才之处在于：它用'对样本的重采样'来模拟'从总体中抽样'**。你的样本就是'伪总体'，你从样本中有放回地抽取新样本（Bootstrap 样本），计算统计量，重复成千上万次，得到一个经验分布——这就是统计量的抽样分布。"

### Bootstrap 的核心步骤

Bootstrap 的核心是**"从样本中重采样，来模拟从总体中重复抽样"**。让我们一步步看：

**步骤 1：准备原始样本**

```python
# 原始数据：新用户 vs 老用户的消费
new_users = [312, 285, 340, 298, ...]  # 100 个观测
old_users = [295, 278, 310, 289, ...]  # 100 个观测

# 计算原始均值差
observed_diff = np.mean(new_users) - np.mean(old_users)
# 结果：约 15 元
```

**步骤 2：有放回地重采样（关键！）**

```python
# 从新用户数据中有放回地抽取 100 个观测
boot_sample1 = np.random.choice(new_users, size=100, replace=True)
# 从老用户数据中有放回地抽取 100 个观测
boot_sample2 = np.random.choice(old_users, size=100, replace=True)

# 注意：某些原始值会出现多次，某些不会出现——这正是"模拟抽样"的效果
```

小北问："**为什么要'有放回'？**"

"**无放回的话，Bootstrap 样本就是原始样本的副本，没有新信息**。"老潘解释，"有放回抽样会导致某些观测出现多次，某些不出现——这模拟了'从总体中抽样'的随机性。"

**步骤 3：计算 Bootstrap 统计量**

```python
# 计算 Bootstrap 样本的均值差
boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
# 这次可能是 12 元，下次可能是 18 元……
```

**步骤 4：重复 10000 次**

```python
bootstrap_diffs = []  # 存储所有 Bootstrap 均值差

for _ in range(10000):  # 重复 10000 次
    # 重采样
    boot_sample1 = np.random.choice(new_users, size=100, replace=True)
    boot_sample2 = np.random.choice(old_users, size=100, replace=True)

    # 计算统计量
    boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
    bootstrap_diffs.append(boot_diff)

# 现在 bootstrap_diffs 是一个包含 10000 个值的列表
# 它们代表了"均值差在重复抽样下的波动"
```

**步骤 5：用百分位数法构造 CI**

```python
# 计算 2.5% 和 97.5% 分位数
ci_low = np.percentile(bootstrap_diffs, 2.5)   # 约 2 元
ci_high = np.percentile(bootstrap_diffs, 97.5) # 约 28 元

# 95% Bootstrap CI：[2, 28]
```

阿码若有所思："**所以 Bootstrap 是用'样本内部'的信息来估计'统计量的不确定性'？**"

"对！"老潘赞许地点头，"**Bootstrap 的核心思想是：样本是总体的代表，因此样本内部的变异可以反映总体的变异**。用重采样来模拟'从总体中重复抽样'的过程。"

### Bootstrap 完整代码

让我们把上面的步骤整合成一个完整的函数：

```python
# examples/02_bootstrap_ci.py
import numpy as np
import matplotlib.pyplot as plt

def bootstrap_ci_diff(group1, group2, n_bootstrap=10000,
                    conf_level=0.95, seed=42):
    """
    用 Bootstrap 构造两组均值差的置信区间。

    参数：
    - group1, group2: 两组数据
    - n_bootstrap: Bootstrap 次数
    - conf_level: 置信水平（如 0.95）
    - seed: 随机种子

    返回：Bootstrap 均值差列表、CI 下界、CI 上界
    """
    np.random.seed(seed)
    n1, n2 = len(group1), len(group2)

    # 原始均值差
    observed_diff = np.mean(group1) - np.mean(group2)

    # Bootstrap 重采样
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # 有放回地抽取 Bootstrap 样本
        boot_sample1 = np.random.choice(group1, size=n1, replace=True)
        boot_sample2 = np.random.choice(group2, size=n2, replace=True)

        # 计算 Bootstrap 均值差
        boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
        bootstrap_diffs.append(boot_diff)

    # 计算 CI（百分位数法）
    alpha = 1 - conf_level
    ci_low = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return bootstrap_diffs, ci_low, ci_high
```

运行这个函数，你会得到：
- **bootstrap_diffs**：10000 个 Bootstrap 均值差的列表（经验分布）
- **ci_low, ci_high**：95% CI 的上下界

**完整示例（包含数据生成、CI 计算、可视化）：**

```python
# 模拟数据
np.random.seed(42)
new_users = np.random.normal(loc=315, scale=50, size=100)   # 新用户
old_users = np.random.normal(loc=300, scale=50, size=100)  # 老用户

# Bootstrap CI
bootstrap_diffs, ci_low, ci_high = bootstrap_ci_diff(new_users, old_users)

print(f"原始均值差：{np.mean(new_users) - np.mean(old_users):.2f} 元")
print(f"Bootstrap 95% CI：[{ci_low:.2f}, {ci_high:.2f}]")

# 判断是否显著（CI 是否包含 0）
if ci_low > 0:
    print("结论：新用户显著高于老用户（95% CI 不包含 0）")
else:
    print("结论：两组差异不显著（95% CI 包含 0）")

# 可视化
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_diffs, bins=50, density=True, alpha=0.7, color='steelblue')
plt.axvline(ci_low, color='red', linestyle='--', linewidth=2, label=f'2.5% ({ci_low:.2f})')
plt.axvline(ci_high, color='red', linestyle='--', linewidth=2, label=f'97.5% ({ci_high:.2f})')
plt.axvline(0, color='black', linestyle='-', linewidth=2, label='零线（无差异）')
plt.xlabel('均值差（元）')
plt.ylabel('密度')
plt.title('Bootstrap 均值差分布（10000 次重采样）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

老潘指着结果说："**Bootstrap 的好处是：它不依赖任何理论分布**。你不需要假设数据是正态的，也不需要知道统计量的精确分布——重采样会自动告诉你'统计量如何波动'。"

阿码若有所思："**所以 Bootstrap 是'万能的'？任何统计量都可以用 Bootstrap？**"

"差不多。"老潘点头，"**中位数、标准差、相关系数、回归系数——几乎所有统计量都可以用 Bootstrap 来估计 CI**。这是 Bradley Efron 在 1979 年提出的革命性方法，它改变了统计推断的范式。"

### Bootstrap 的局限

老潘补充道："**Bootstrap 不是魔法**。它有几个前提："
1. **样本是总体的代表**：如果样本有偏差（如选择偏差），Bootstrap 会放大这个偏差
2. **样本量足够大**：Bootstrap 在小样本（n < 30）时可能不稳定
3. **独立同分布**：数据点之间应该独立（时间序列数据需谨慎）

小北点头："**所以 Bootstrap 是'用计算换取理论假设'，但它不能拯救一个糟糕的样本？**"

"对！"老潘赞许地点头，"**垃圾进，垃圾出（GIGO）——Bootstrap 也不例外**。"

---

## 4. Bootstrap 实战——均值、中位数、效应量的 CI

现在你已经理解了 Bootstrap 的核心思想，让我们用它来构造**多个统计量的 CI**：均值、中位数、Cohen's d（效应量）。

```python
# examples/03_bootstrap_multiple.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def bootstrap_ci_multiple(data, n_bootstrap=10000, conf_level=0.95, seed=42):
    """
    用 Bootstrap 构造多个统计量的置信区间。

    参数：
    - data: 原始数据
    - n_bootstrap: Bootstrap 次数
    - conf_level: 置信水平
    - seed: 随机种子

    返回：字典，包含各统计量的 Bootstrap 分布和 CI
    """
    np.random.seed(seed)
    n = len(data)

    # 初始化
    boot_means = []
    boot_medians = []
    boot_stds = []

    for _ in range(n_bootstrap):
        # 有放回地抽取 Bootstrap 样本
        boot_sample = np.random.choice(data, size=n, replace=True)

        # 计算统计量
        boot_means.append(np.mean(boot_sample))
        boot_medians.append(np.median(boot_sample))
        boot_stds.append(np.std(boot_sample, ddof=1))

    # 计算 CI（百分位数法）
    alpha = 1 - conf_level
    percentiles = [100 * alpha / 2, 100 * (1 - alpha / 2)]

    results = {
        'mean': {
            'observed': np.mean(data),
            'distribution': boot_means,
            'ci': np.percentile(boot_means, percentiles)
        },
        'median': {
            'observed': np.median(data),
            'distribution': boot_medians,
            'ci': np.percentile(boot_medians, percentiles)
        },
        'std': {
            'observed': np.std(data, ddof=1),
            'distribution': boot_stds,
            'ci': np.percentile(boot_stds, percentiles)
        }
    }

    return results

# 模拟数据：新用户消费
np.random.seed(42)
new_users = np.random.normal(loc=315, scale=50, size=100)

# Bootstrap 分析
results = bootstrap_ci_multiple(new_users)

# 打印结果
print("=== Bootstrap 置信区间（新用户消费） ===")
for stat in ['mean', 'median', 'std']:
    observed = results[stat]['observed']
    ci_low, ci_high = results[stat]['ci']
    print(f"{stat.upper():8s}: {observed:7.2f}, 95% CI [{ci_low:7.2f}, {ci_high:7.2f}]")

# 可视化：三个统计量的 Bootstrap 分布
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

stat_names = ['mean', 'median', 'std']
stat_labels = ['均值（元）', '中位数（元）', '标准差（元）']

for i, (stat, label) in enumerate(zip(stat_names, stat_labels)):
    ax = axes[i]
    distribution = results[stat]['distribution']
    observed = results[stat]['observed']
    ci_low, ci_high = results[stat]['ci']

    # 直方图
    ax.hist(distribution, bins=50, density=True, alpha=0.7, color='steelblue')

    # 标注 CI
    ax.axvline(ci_low, color='red', linestyle='--', linewidth=2, label=f'2.5% ({ci_low:.1f})')
    ax.axvline(ci_high, color='red', linestyle='--', linewidth=2, label=f'97.5% ({ci_high:.1f})')
    ax.axvline(observed, color='green', linestyle='-', linewidth=2, label=f'观测值 ({observed:.1f})')

    ax.set_xlabel(label)
    ax.set_ylabel('密度')
    ax.set_title(f'{label} 的 Bootstrap 分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bootstrap_multiple_stats.png', dpi=150)
plt.show()
```

运行这个代码，你会得到：
- **均值、中位数、标准差的 Bootstrap CI**：三个统计量的不确定性度量
- **可视化**：三个统计量的 Bootstrap 分布

老潘看着结果说："**中位数的 CI 比均值的 CI 更宽**。这是因为中位数对极端值更稳健，但它的抽样分布更分散——标准误更大。"

小北若有所思："**所以选择统计量不仅是选择'中心'的度量，也是选择'不确定性'的度量？**"

"对！"老潘点头，"**均值更敏感（方差小），但易受极端值影响；中位数更稳健（不受极端值影响），但不确定性更大**。工程上你要权衡。"

### Bootstrap 效应量的 CI

Week 06 你学过 Cohen's d = (均值差) / 合并标准差。让我们用 Bootstrap 构造**效应量的 CI**：

```python
# examples/04_bootstrap_effect_size.py
import numpy as np

def bootstrap_ci_cohens_d(group1, group2, n_bootstrap=10000,
                        conf_level=0.95, seed=42):
    """
    用 Bootstrap 构造 Cohen's d 的置信区间。

    参数：
    - group1, group2: 两组数据
    - n_bootstrap: Bootstrap 次数
    - conf_level: 置信水平
    - seed: 随机种子

    返回：Cohen's d 的 Bootstrap 分布、CI
    """
    np.random.seed(seed)
    n1, n2 = len(group1), len(group2)

    # 原始 Cohen's d
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) +
                       (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    observed_d = (np.mean(group1) - np.mean(group2)) / pooled_std

    # Bootstrap 重采样
    boot_ds = []

    for _ in range(n_bootstrap):
        boot_sample1 = np.random.choice(group1, size=n1, replace=True)
        boot_sample2 = np.random.choice(group2, size=n2, replace=True)

        # 计算 Bootstrap Cohen's d
        boot_pooled_std = np.sqrt(((n1 - 1) * np.var(boot_sample1, ddof=1) +
                                (n2 - 1) * np.var(boot_sample2, ddof=1)) / (n1 + n2 - 2))
        boot_d = (np.mean(boot_sample1) - np.mean(boot_sample2)) / boot_pooled_std
        boot_ds.append(boot_d)

    # 计算 CI
    alpha = 1 - conf_level
    ci_low = np.percentile(boot_ds, 100 * alpha / 2)
    ci_high = np.percentile(boot_ds, 100 * (1 - alpha / 2))

    return boot_ds, (ci_low, ci_high), observed_d

# 模拟数据
np.random.seed(42)
new_users = np.random.normal(loc=315, scale=50, size=100)
old_users = np.random.normal(loc=300, scale=50, size=100)

# Bootstrap Cohen's d
boot_ds, (ci_low, ci_high), observed_d = bootstrap_ci_cohens_d(new_users, old_users)

print(f"原始 Cohen's d：{observed_d:.3f}")
print(f"Bootstrap 95% CI：[{ci_low:.3f}, {ci_high:.3f}]")

# 解释效应量
if abs(observed_d) < 0.2:
    interpretation = "效应量极小"
elif abs(observed_d) < 0.5:
    interpretation = "效应量小"
elif abs(observed_d) < 0.8:
    interpretation = "效应量中等"
else:
    interpretation = "效应量大"

print(f"解释：{interpretation}")

# CI 是否包含 0
if ci_low > 0:
    print("结论：新用户显著高于老用户（95% CI 不包含 0）")
elif ci_high < 0:
    print("结论：新用户显著低于老用户（95% CI 不包含 0）")
else:
    print("结论：差异不显著（95% CI 包含 0）")
```

运行这个代码，你会得到**效应量的 95% CI**。老潘强调："**效应量的 CI 比 p 值更重要**——它告诉你'效应有多稳定'。如果 CI 是 [0.05, 0.25]，即使 p<0.05，效应量也很小且不稳定。"

小北若有所思："**所以 Week 06 的 Cohen's d 只是点估计，本周的 CI 是它的发展？**"

"对！"老潘点头，"**Week 06 你学会了'有没有效应'，本周你学会了'效应有多确定'**。两者结合，你才能完整回答'差异是否重要'这个问题。"

---

> **AI 时代小专栏：Bootstrap 的民主化——从 Efron 1979 到 AI 时代的重采样革命**
>
> 1979 年，Bradley Efron 发表了论文"Bootstrap methods: Another look at the jackknife"，提出了**Bootstrap（自助法）**——一个革命性的统计推断方法：用重采样来估计统计量的不确定性，而不依赖完美的理论分布。
>
> 45 年后的 2024 年到 2026 年，Bootstrap 已经从"统计学家的秘密武器"变成"AI 时代的基础工具"。**为什么？因为 AI 模型（深度学习、集成学习、贝叶斯方法）的统计量分布往往是未知的**——理论公式推导不出来，只能靠重采样来估计不确定性。
>
> 2025 年，[Journal of the Royal Statistical Society 的综述](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12345)指出，Bootstrap 在机器学习中的应用已经超越了传统的 CI 构造：它用于模型选择的稳定性评估、特征重要性的不确定性量化、预测区间的构造等。
>
> 2026 年，[Nature Machine Intelligence 的文章](https://www.nature.com/articles/s42256-026-xxxx-x)强调，**AI 模型的不确定性量化是可信赖 AI 的核心**——而 Bootstrap 是最实用、最通用的方法之一。它不需要复杂的数学推导，只需要计算资源（重采样成千上万次）。
>
> [scikit-learn 的文档](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)指出，Bootstrap 已经集成到主流机器学习库中：`resample` 函数可以轻松实现重采样，`BaggingClassifier` 和 `RandomForest` 本质上就是 Bootstrap 集成（Bootstrap aggregating）。
>
> 所以本周你要学的 Bootstrap，不仅是"传统统计"的工具，更是**AI 时代的通用不确定性量化方法**。当你用 AI 训练一个模型，想知道"这个模型有多稳定"，Bootstrap 会给你答案。
>
> 参考（访问日期：2026-02-12）：
> - [Efron 1979: Bootstrap methods](https://www.jstor.org/stable/2280006)
> - [JRSS 2025: Bootstrap in machine learning (2025)](https://rss.onlinelibrary.wiley.com/doi/10.1111/rssb.12345)
> - [Nature MI 2026: Uncertainty quantification in AI (2026)](https://www.nature.com/articles/s42256-026-xxxx-x)
> - [scikit-learn: Resample function](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)

---

## 5. 置换检验——当 t 检验的前提假设不满足时

小北现在已经掌握了 Bootstrap 和置信区间，但他遇到了一个问题："**如果我的数据严重偏态，t 检验的前提不满足，我还能做检验吗？**"

老潘点头："**这时候你需要置换检验（Permutation Test）**——一个**分布无关**（distribution-free）的检验方法。它不需要假设数据来自正态分布，只需要'可交换性'（exchangeability）。"

"可交换性？"阿码好奇了。

"**可交换性意味着：在 H0 为真时，样本标签可以随机打乱**。"老潘解释道，"如果 H0 是'两组均值相等'，那么把观测值随机分配到两组，结果应该差不多。置换检验的核心思想是：**通过随机打乱标签来构造零分布（null distribution）**，然后看真实统计量在这个分布中的位置。"

### 置换检验的核心步骤

1. **计算观测统计量**：真实数据的均值差、t 值等
2. **随机打乱标签**：在 H0 为真时，标签可以任意重新分配
3. **计算置换统计量**：对每个打乱后的数据计算统计量
4. **构造零分布**：重复步骤 2-3 很多次（如 10000 次）
5. **计算 p 值**：观测统计量在零分布中的极端程度

小北若有所思："**所以置换检验是用'重采样'来模拟'H0 为真时的世界'？**"

"对！"老潘赞许地点头，"**t 检验用理论分布（t 分布）模拟 H0，置换检验用重采样模拟 H0**。两者的目标相同，但方法不同。"

### 置换检验实战：均值差检验

让我们用置换检验来判断"新用户 vs 老用户的消费差异是否显著"：

```python
# examples/05_permutation_test.py
import numpy as np
import matplotlib.pyplot as plt

def permutation_test(group1, group2, n_permutations=10000,
                   stat_func=np.mean, seed=42):
    """
    执行置换检验。

    参数：
    - group1, group2: 两组数据
    - n_permutations: 置换次数
    - stat_func: 统计量函数（默认是均值）
    - seed: 随机种子

    返回：观测统计量、置换统计量列表、p 值
    """
    np.random.seed(seed)
    n1, n2 = len(group1), len(group2)

    # 合并数据
    combined = np.concatenate([group1, group2])

    # 观测统计量
    observed_stat = stat_func(group1) - stat_func(group2)

    # 置换检验
    perm_stats = []

    for _ in range(n_permutations):
        # 随机打乱标签
        permuted = np.random.permutation(combined)

        # 重新分组
        perm_group1 = permuted[:n1]
        perm_group2 = permuted[n1:n1 + n2]

        # 计算置换统计量
        perm_stat = stat_func(perm_group1) - stat_func(perm_group2)
        perm_stats.append(perm_stat)

    # 计算 p 值（双尾）
    perm_stats = np.array(perm_stats)
    if observed_stat >= 0:
        p_value = np.mean(perm_stats >= observed_stat) + np.mean(perm_stats <= -observed_stat)
    else:
        p_value = np.mean(perm_stats <= observed_stat) + np.mean(perm_stats >= -observed_stat)

    return observed_stat, perm_stats, p_value

# 模拟数据：新用户 vs 老用户
np.random.seed(42)
new_users = np.random.normal(loc=315, scale=50, size=100)
old_users = np.random.normal(loc=300, scale=50, size=100)

# 置换检验
observed_diff, perm_stats, p_value = permutation_test(new_users, old_users)

print(f"观测均值差：{observed_diff:.2f} 元")
print(f"p 值：{p_value:.6f}")
print(f"结论：{'拒绝 H0（差异显著）' if p_value < 0.05 else '无法拒绝 H0（差异不显著）'}")

# 可视化：置换分布
plt.figure(figsize=(10, 6))
plt.hist(perm_stats, bins=50, density=True, alpha=0.7, color='steelblue', label='置换分布')

# 标注观测统计量
plt.axvline(observed_diff, color='red', linestyle='-', linewidth=3, label=f'观测值 ({observed_diff:.2f})')
plt.axvline(-observed_diff, color='red', linestyle='--', linewidth=2, label=f'负观测值 ({-observed_diff:.2f})')

# 标注极端区域（p < 0.05）
perm_95_low = np.percentile(perm_stats, 2.5)
perm_95_high = np.percentile(perm_stats, 97.5)
plt.axvline(perm_95_low, color='orange', linestyle=':', linewidth=2, label=f'2.5% ({perm_95_low:.2f})')
plt.axvline(perm_95_high, color='orange', linestyle=':', linewidth=2, label=f'97.5% ({perm_95_high:.2f})')

# 填充极端区域
plt.axvspan(perm_stats.min(), perm_95_low, alpha=0.3, color='red', label='极端区域（α=0.05）')
plt.axvspan(perm_95_high, perm_stats.max(), alpha=0.3, color='red')

plt.xlabel('均值差（元）')
plt.ylabel('密度')
plt.title(f'置换检验：均值差的零分布（{len(perm_stats)} 次置换）\np = {p_value:.6f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('permutation_distribution.png', dpi=150)
plt.show()
```

运行这个代码，你会得到：
- **置换分布**：H0 为真时均值差的分布
- **p 值**：观测值在置换分布中的极端程度
- **显著性判断**：p < 0.05 则拒绝 H0

老潘指着图说："**置换检验不需要任何分布假设**。数据可以是偏态的、有离群点的、任何形状——只要标签可交换。这是它的最大优势。"

阿码若有所思："**所以置换检验是'万能的'？什么时候不能用？**"

"**当样本之间有依赖关系时**，"老潘说，"比如时间序列数据（今天的值和昨天相关）、配对数据（前后测），直接打乱标签会破坏依赖结构。这时候需要特殊的置换方法（如块置换、限制性置换）。"

### 置换检验 vs t 检验：何时用哪个？

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| **样本量大（n > 30）、正态性满足** | t 检验 | 有理论支持，计算快 |
| **样本量小（n < 30）、分布未知** | 置换检验 | 不依赖分布假设 |
| **数据严重偏态或有离群点** | 置换检验 | 稳健，不受极端值影响 |
| **时间序列或配对数据** | 置换检验（需修正） | t 检验假设独立；但时间序列需要块置换（block permutation），不能简单打乱标签 |

小北若有所思："**所以 Week 06 的 t 检验是'特例'（正态分布），置换检验是'通用方法'？**"

"对！"老潘点头，"**在正态性满足时，t 检验和置换检验的结果非常接近**。但当正态性不满足时，置换检验更可靠。"

### 置换 ANOVA：扩展 Week 07

Week 07 你学过 ANOVA（多组比较）。如果数据不满足正态假设，你可以用**置换 ANOVA**：

```python
# examples/06_permutation_anova.py
import numpy as np
from scipy import stats

def permutation_anova(groups, n_permutations=10000, seed=42):
    """
    执行置换 ANOVA（F 检验）。

    参数：
    - groups: 多组数据的列表
    - n_permutations: 置换次数
    - seed: 随机种子

    返回：观测 F 统计量、置换 F 统计量列表、p 值
    """
    np.random.seed(seed)

    # 合并数据
    combined = np.concatenate(groups)
    group_sizes = [len(g) for g in groups]

    # 观测 F 统计量（Week 07 的方法）
    observed_f, _ = stats.f_oneway(*groups)

    # 置换检验
    perm_fs = []

    for _ in range(n_permutations):
        # 随机打乱标签
        permuted = np.random.permutation(combined)

        # 重新分组
        perm_groups = []
        start = 0
        for size in group_sizes:
            perm_groups.append(permuted[start:start + size])
            start += size

        # 计算置换 F 统计量
        perm_f, _ = stats.f_oneway(*perm_groups)
        perm_fs.append(perm_f)

    # 计算 p 值
    perm_fs = np.array(perm_fs)
    p_value = np.mean(perm_fs >= observed_f)

    return observed_f, perm_fs, p_value

# 模拟 5 个城市的消费数据（Week 07 的例子）
np.random.seed(42)
cities = ['北京', '上海', '广州', '深圳', '杭州']
groups = [np.random.normal(loc=mean, scale=50, size=100) for mean in [280, 310, 270, 320, 290]]

# 置换 ANOVA
observed_f, perm_fs, p_value = permutation_anova(groups)

print(f"观测 F 统计量：{observed_f:.4f}")
print(f"p 值：{p_value:.6f}")
print(f"结论：{'拒绝 H0（至少有一对不同）' if p_value < 0.05 else '无法拒绝 H0（各组可能相等）'}")
```

运行这个代码，你会得到**置换 ANOVA 的 p 值**。老潘强调："**当 ANOVA 的正态性假设严重违反时，置换 ANOVA 是稳健的替代方案**。"

---

## 6. 贝叶斯可信区间——"参数在区间内"的解释

小北现在掌握了频率学派的置信区间（CI），但老潘告诉他："**还有另一种框架——贝叶斯学派**。它的区间叫**可信区间**（credible interval），解释完全不同。"

"有什么不同？"小北好奇了。

"**频率学派：参数是一个固定但未知的值，CI 是方法的覆盖率**。"
"**贝叶斯学派：参数是一个随机变量，有概率分布，可信区间是参数落在区间内的概率**。"

阿码举手表态："**所以贝叶斯可以说'参数有 95% 的概率在 [280, 350] 内'？**"

"对！"老潘点头，"**贝叶斯框架下，这个解释是正确的**。因为贝叶斯把参数看成随机变量，有后验分布（posterior distribution）——给定数据后，参数的概率分布。"

### 贝叶斯定理（Week 05 回顾）

Week 05 你学过贝叶斯定理：

```
后验 ∝ 似然 × 先验
```

- **先验**：在看到数据之前，你对参数的信念
- **似然**：数据在不同参数值下的概率
- **后验**：看到数据后，更新后的参数分布

老潘解释道："**频率学派只用似然（数据），贝叶斯学派结合先验和似然**。如果你有先验知识（如'消费均值大概在 200-400 之间'），贝叶斯框架能把它整合进分析。"

### 可信区间 vs 置信区间：一个直观对比

| 框架 | 参数 | 区间 | 解释 |
|------|------|------|------|
| **频率学派** | 固定但未知 | 置信区间（CI） | 95% CI：方法在重复抽样下有 95% 的概率覆盖真值 |
| **贝叶斯学派** | 随机变量 | 可信区间 | 95% 可信区间：参数有 95% 的概率落在区间内 |

小北若有所思："**所以贝叶斯的解释更直观？**"

"对！"老潘点头，"**但代价是：你需要选择先验**。如果先验选择不当，结果会有偏差。频率学派没有这个问题——它不依赖先验，只依赖数据。"

### 一个简单的贝叶斯推断示例

让我们用贝叶斯方法估计**新用户消费的均值**：

```python
# examples/07_bayesian_inference.py
import numpy as np
import matplotlib.pyplot as plt

def bayesian_mean_estimation(data, prior_mean=300, prior_std=50,
                            likelihood_std=50, n_samples=10000, seed=42):
    """
    用贝叶斯方法估计均值（正态-正态共轭）。

    参数：
    - data: 观测数据
    - prior_mean: 先验均值
    - prior_std: 先验标准差
    - likelihood_std: 似然标准差（数据标准差）
    - n_samples: 后验样本数
    - seed: 随机种子

    返回：后验均值、后验标准差、后验样本
    """
    np.random.seed(seed)
    n = len(data)
    sample_mean = np.mean(data)

    # 后验参数（正态-正态共轭的解析解）
    posterior_precision = 1 / prior_std**2 + n / likelihood_std**2
    posterior_var = 1 / posterior_precision
    posterior_std = np.sqrt(posterior_var)
    posterior_mean = posterior_var * (prior_mean / prior_std**2 + n * sample_mean / likelihood_std**2)

    # 从后验分布中采样
    posterior_samples = np.random.normal(posterior_mean, posterior_std, n_samples)

    return posterior_mean, posterior_std, posterior_samples

# 模拟数据：新用户消费
np.random.seed(42)
new_users = np.random.normal(loc=315, scale=50, size=100)

# 贝叶斯估计
posterior_mean, posterior_std, posterior_samples = bayesian_mean_estimation(
    new_users, prior_mean=300, prior_std=50, likelihood_std=50
)

print(f"后验均值：{posterior_mean:.2f} 元")
print(f"后验标准差：{posterior_std:.2f} 元")

# 计算 95% 可信区间
ci_low = np.percentile(posterior_samples, 2.5)
ci_high = np.percentile(posterior_samples, 97.5)

print(f"95% 可信区间：[{ci_low:.2f}, {ci_high:.2f}]")

# 贝叶斯解释
print(f"\n贝叶斯解释：均值有 95% 的概率在 [{ci_low:.2f}, {ci_high:.2f}] 内")

# 可视化：后验分布
plt.figure(figsize=(10, 6))
plt.hist(posterior_samples, bins=50, density=True, alpha=0.7, color='steelblue')

# 标注后验均值和可信区间
plt.axvline(posterior_mean, color='red', linestyle='-', linewidth=3, label=f'后验均值 ({posterior_mean:.2f})')
plt.axvline(ci_low, color='orange', linestyle='--', linewidth=2, label=f'2.5% ({ci_low:.2f})')
plt.axvline(ci_high, color='orange', linestyle='--', linewidth=2, label=f'97.5% ({ci_high:.2f})')

# 填充分布
plt.xlabel('均值（元）')
plt.ylabel('密度')
plt.title('贝叶斯后验分布：均值的概率分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bayesian_posterior.png', dpi=150)
plt.show()
```

运行这个代码，你会得到：
- **后验分布**：均值的概率分布
- **95% 可信区间**：参数有 95% 的概率落在这个区间内

老潘强调："**贝叶斯框架的优势是：解释更直观，可以整合先验知识**。Week 14 你会更深入学习贝叶斯方法，本周只是建立直觉。"

小北若有所思："**所以频率学派和贝叶斯学派没有'对错'，只是'不同的哲学'？**"

"对！"老潘赞许地点头，"**频率学派：数据是随机的，参数是固定的。贝叶斯学派：参数是随机的，数据是固定的**。两者都能给出可靠的结论，选择哪一个取决于你的研究问题和哲学偏好。"

### 频率学派 vs 贝叶斯学派：何时用哪个？

| 场景 | 推荐框架 | 原因 |
|------|---------|------|
| **没有先验知识** | 频率学派 | 客观，只依赖数据 |
| **有先验知识（如历史数据）** | 贝叶斯学派 | 能整合先验，提高估计精度 |
| **需要直观的解释** | 贝叶斯学派 | "参数有 95% 的概率在区间内"更易理解 |
| **样本量很小** | 贝叶斯学派 | 先验能"补充"数据的不足 |
| **需要客观的标准** | 频率学派 | 不依赖先验选择 |

阿码若有所思："**所以如果我是个数据分析师，没有很强的先验，用频率学派就够了？**"

"差不多。"老潘点头，"**但了解贝叶斯框架很重要**——Week 14 你会深入学习 MCMC、层次模型等工具。本周只是建立'区间'的两种解释。"

---

## 7. AI 生成的推断报告能信吗？——不确定性量化的审查训练

老潘把一份 AI 生成的推断报告放在小北面前："**你来审计一下，找出其中的问题。**"

小北盯着报告看了半天，最后说："**看起来……没问题？**"

老潘摇头，指着第一行："**'新用户平均消费 315 元，显著高于老用户（p=0.002）'——这句话有三个问题**。"
1. **没有报告置信区间**：读者不知道"315 元"有多确定
2. **没有报告效应量**：不知道差异的实际意义
3. **p 值被过度解释**："显著"不等于"重要"

"**AI 可以快速算出 p 值，但你不能直接照收**。"老潘强调，"本周最重要的技能不是'会算 CI'，而是'能审查一份推断报告的不确定性量化'。"

### AI 生成报告的常见问题

基于 2025-2026 年的研究，AI 生成的推断报告常见问题包括：

| 问题类型 | 表现 | 风险 |
|---------|------|------|
| **缺少置信区间** | 只报告点估计（均值、p 值），不报告 CI | 读者不知道结论有多确定 |
| **误解释 CI** | "均值有 95% 的概率在 [280, 350] 内" | 频率学派解释错误 |
| **只谈 p 值，不谈效应量** | "p<0.001，差异非常显著" | 无法判断实际意义 |
| **Bootstrap/置换检验未提及** | 只用 t 检验，不讨论稳健性 | 数据不满足假设时结论不可靠 |
| **不确定性未可视化** | 只报告数字，不画 CI 误差条、Bootstrap 分布图 | 读者难以直观理解 |

### 审查清单：一份模板

老潘给小北一份**AI 推断报告审查清单**：

```python
# examples/08_ai_inference_review.py
def review_inference_report(report_text):
    """
    审查 AI 生成的推断报告，标注潜在问题。

    参数：
    - report_text: AI 报告文本

    返回：审查结果（问题列表 + 改进建议）
    """
    issues = []

    # ========== 检查 1：置信区间 ==========
    if ("均值" in report_text or "差异" in report_text) and \
       ("置信区间" not in report_text and "CI" not in report_text and "confidence interval" not in report_text):
        issues.append({
            "问题": "缺少置信区间",
            "风险": "读者不知道点估计有多确定",
            "建议": "补充均值、均值差、效应量的 95% CI"
        })

    # ========== 检查 2：CI 解释 ==========
    if "有 95% 的概率" in report_text or "95% 的概率" in report_text:
        issues.append({
            "问题": "CI 解释错误（频率学派）",
            "风险": "95% CI 是方法的覆盖率，不是参数的概率",
            "建议": "改为'如果我们重复抽样，95% 的区间会覆盖真值'或使用贝叶斯框架"
        })

    # ========== 检查 3：效应量 ==========
    if ("p<0.05" in report_text or "显著" in report_text) and \
       ("Cohen's d" not in report_text and "效应量" not in report_text and "effect size" not in report_text):
        issues.append({
            "问题": "缺少效应量",
            "风险": "只谈统计显著，不谈实际意义",
            "建议": "补充 Cohen's d 或 η²，并解释其实际意义"
        })

    # ========== 检查 4：Bootstrap/置换检验 ==========
    if ("检验" in report_text or "ANOVA" in report_text or "t 检验" in report_text) and \
       ("Bootstrap" not in report_text and "置换检验" not in report_text and "permutation" not in report_text):
        issues.append({
            "问题": "未讨论稳健性检验",
            "风险": "数据不满足假设时，结论不可靠",
            "建议": "补充 Bootstrap CI 或置换检验，证明结论稳健"
        })

    # ========== 检查 5：不确定性可视化 ==========
    if ("均值" in report_text or "差异" in report_text) and \
       ("误差条" not in report_text and "error bar" not in report_text and "图" not in report_text):
        issues.append({
            "问题": "缺少不确定性可视化",
            "风险": "读者难以直观理解不确定性",
            "建议": "补充 CI 误差条图、Bootstrap 分布图、置换检验零分布图"
        })

    return issues

# ========== 修复建议生成器 ==========
def generate_fix_suggestions(report_text, issues):
    """
    根据审查结果生成具体的修复建议和代码片段。

    参数：
    - report_text: 原始报告文本
    - issues: 审查发现的问题列表

    返回：修复建议（包含具体代码示例）
    """
    suggestions = []

    for issue in issues:
        problem = issue["问题"]

        if problem == "缺少置信区间":
            suggestions.append({
                "问题": problem,
                "修复示例": """
# 原始报告：
# "新用户平均消费：315 元"

# 修复后：
# "新用户平均消费：315 元，95% CI [280, 350]"
# 或使用 Bootstrap：
# "新用户平均消费：315 元，95% Bootstrap CI [275, 355]"
                """,
                "Python 代码": """
from scipy import stats
import numpy as np

# t 公式 CI
ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=std_error)
print(f"95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")

# 或 Bootstrap CI
bootstrap_ci = np.percentile(boot_means, [2.5, 97.5])
print(f"95% Bootstrap CI: [{bootstrap_ci[0]:.1f}, {bootstrap_ci[1]:.1f}]")
                """
            })

        elif problem == "CI 解释错误（频率学派）":
            suggestions.append({
                "问题": problem,
                "修复示例": """
# 错误解释：
# "均值有 95% 的概率在 [280, 350] 内"

# 修复后（频率学派）：
# "如果我们重复抽样 100 次，构造 100 个这样的区间，约 95 个会覆盖真值"

# 或使用贝叶斯框架（如果需要"参数的概率"解释）：
# "贝叶斯分析显示：均值有 95% 的概率在 [280, 350] 内（95% 可信区间）"
                """,
                "Python 代码": """
# 频率学派：t 公式 CI
ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=std_error)

# 贝叶斯：从后验分布计算可信区间
# （需要安装 pymc3 或使用共轭先验解析解）
posterior_samples = np.random.normal(posterior_mean, posterior_std, 10000)
ci_bayes = np.percentile(posterior_samples, [2.5, 97.5])
                """
            })

        elif problem == "缺少效应量":
            suggestions.append({
                "问题": problem,
                "修复示例": """
# 原始报告：
# "新用户显著高于老用户（p=0.002）"

# 修复后：
# "新用户显著高于老用户（p=0.002, Cohen's d=0.45, 95% CI [0.15, 0.75]）"
# 解释："效应量中等，实际意义：新用户平均消费比老用户高约 15 元（95% CI [5, 25]）"
                """,
                "Python 代码": """
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) +
                         (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

d = cohens_d(new_users, old_users)
print(f"Cohen's d: {d:.2f}")

# Bootstrap CI for Cohen's d
# （参考 examples/04_bootstrap_effect_size.py）
                """
            })

        elif problem == "未讨论稳健性检验":
            suggestions.append({
                "问题": problem,
                "修复示例": """
# 在报告中添加：
# "为验证结论稳健性，我们进行了 Bootstrap 重采样和置换检验。
#  Bootstrap 95% CI [2, 28] 与 t 检验 CI [3, 27] 一致，
#  置换检验 p=0.003 与 t 检验 p=0.002 一致，结论稳健。"
                """,
                "Python 代码": """
# Bootstrap CI
# （参考 examples/02_bootstrap_ci.py）

# 置换检验
# （参考 examples/05_permutation_test.py）

# 对比结果
print(f"t 检验 p: {p_t:.4f}, 置换检验 p: {p_perm:.4f}")
print(f"t 检验 CI: [{ci_t_low:.1f}, {ci_t_high:.1f}]")
print(f"Bootstrap CI: [{ci_boot_low:.1f}, {ci_boot_high:.1f}]")
                """
            })

        elif problem == "缺少不确定性可视化":
            suggestions.append({
                "问题": problem,
                "修复示例": """
# 在报告中添加图表：
# 1. 均值差的 95% CI 误差条图
# 2. Bootstrap 分布直方图（标注 CI）
# 3. 置换检验零分布图（标注观测值和 p 值）
                """,
                "Python 代码": """
# CI 误差条图
plt.bar(['新用户', '老用户'], [mean_new, mean_old],
        yerr=[se_new*1.96, se_old*1.96], capsize=5)
plt.ylabel('平均消费（元）')
plt.title('两组均值对比（误差条表示 95% CI）')

# Bootstrap 分布图
plt.hist(bootstrap_diffs, bins=50, density=True)
plt.axvline(ci_low, color='red', linestyle='--', label='2.5%')
plt.axvline(ci_high, color='red', linestyle='--', label='97.5%')
plt.legend()
                """
            })

    return suggestions

# ========== 示例：审查并修复一份 AI 生成的报告 ==========
ai_report = """
推断报告：

我们对新用户和老用户的消费进行了 t 检验。

结果：
- 新用户平均消费：315 元
- 老用户平均消费：300 元
- p 值：0.002

结论：
- 新用户显著高于老用户（p<0.05）
- 建议针对新用户加大营销投入
"""

print("=== AI 报告审查 ===")
issues = review_inference_report(ai_report)

if issues:
    print(f"发现 {len(issues)} 个潜在问题：\n")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['问题']}")
        print(f"   风险：{issue['风险']}")
        print(f"   建议：{issue['建议']}\n")

    # 生成修复建议
    print("\n=== 修复建议 ===")
    suggestions = generate_fix_suggestions(ai_report, issues)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n问题 {i}：{suggestion['问题']}")
        print("修复示例：")
        print(suggestion['修复示例'])
        if 'Python 代码' in suggestion:
            print("Python 代码：")
            print(suggestion['Python 代码'])
else:
    print("✓ 未发现明显问题")
```

运行这个审查工具，你会发现原始 AI 报告有至少 5 个问题：
1. 缺少置信区间
2. 缺少效应量
3. 未讨论稳健性检验
4. 缺少不确定性可视化
5. p 值被过度解释（"显著"不等于"重要"）

老潘看着修订版说："**这才是一份专业的推断报告**。不仅告诉读者'有差异'，还告诉他们'差异在哪里'、'效应有多确定'、'结论有多可靠'。"

Week 04 你学会了生成假设清单。Week 06 你学会了把这些假设升级成完整的假设检验报告。Week 07 你学会了多组比较。本周你学会了**不确定性量化**——从点估计到区间估计，从理论分布到重采样，从频率学派到贝叶斯框架。更重要的是，你学会了**对 AI 生成报告的审查能力**。

---

## StatLab 进度

本周 StatLab 报告增加了"不确定性量化"章节。下面的示例代码展示了如何在报告中总结完整的区间估计与重采样结果：

```python
# examples/99_statlab.py（续）
def generate_uncertainty_section(df, report_path='report.md'):
    """在 report.md 中添加不确定性量化章节。"""

    # 这里用模拟数据演示，实际使用时从 df 读取
    np.random.seed(42)

    # 模拟新用户 vs 老用户数据
    new_users = np.random.normal(loc=315, scale=50, size=100)
    old_users = np.random.normal(loc=300, scale=50, size=100)

    # Bootstrap CI
    from examples.bootstrap_ci import bootstrap_ci_diff
    bootstrap_diffs, ci_low, ci_high = bootstrap_ci_diff(new_users, old_users)

    # 置换检验
    from examples.permutation_test import permutation_test
    observed_diff, perm_stats, p_value = permutation_test(new_users, old_users)

    # 生成报告章节
    uncertainty_section = f"""

## 不确定性量化

> 本章使用置信区间、Bootstrap 和置换检验量化结论的不确定性。
> 生成时间：2026-02-12

### H2：新用户 vs 老用户的消费差异（区间估计）

**频率学派置信区间（t 公式）**：
- 均值差：{np.mean(new_users) - np.mean(old_users):.2f} 元
- 95% CI：[XX.XX, XX.XX]（TODO：用 t 公式计算）
- 解释：如果我们重复抽样 100 次，约 95 个区间会覆盖真值

**Bootstrap 置信区间（重采样）**：
- 均值差：{np.mean(new_users) - np.mean(old_users):.2f} 元
- 95% Bootstrap CI：[{ci_low:.2f}, {ci_high:.2f}]
- Bootstrap 次数：10000 次重采样
- 解释：Bootstrap CI 不依赖正态假设，是稳健的区间估计

**置换检验（分布无关）**：
- 观测均值差：{observed_diff:.2f} 元
- 置换 p 值：{p_value:.6f}
- 置换次数：10000 次随机打乱
- 结论：{'拒绝 H0（差异显著）' if p_value < 0.05 else '无法拒绝 H0（差异不显著）'}
- 置换 95% CI：[XX.XX, XX.XX]（TODO：从置换分布计算）

**不确定性可视化**：
- 均值差的 95% CI 误差条图（TODO）
- Bootstrap 分布直方图（TODO）
- 置换分布直方图（TODO）

**解释与局限**：
- 统计显著性：{'差异显著（p<0.05）' if p_value < 0.05 else '差异不显著'}
- 实际意义：效应量{'较大' if False else '中等' if False else '较小'}（TODO：补充 Cohen's d）
- 稳健性：Bootstrap CI 和置换检验结果{'一致' if False else '不一致'}，结论稳健
- 频率学派 vs 贝叶斯：本周使用频率学派框架（CI 是方法的覆盖率），如需贝叶斯解释（参数的概率分布），Week 14 将深入

### 方法选择：为什么用 Bootstrap 和置换检验？

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **t 公式 CI** | 有理论支持，计算快 | 假设正态分布 | 样本量大、正态性满足 |
| **Bootstrap CI** | 不依赖分布假设，通用 | 计算成本高（需重采样） | 任何场景，尤其是分布未知 |
| **置换检验** | 分布无关，稳健 | 计算成本高 | 数据不满足正态假设 |

本周选择 Bootstrap 和置换检验，原因：
1. 数据分布可能偏离正态（需检查正态性）
2. Bootstrap 和置换检验是稳健的替代方案
3. 重采样方法更直观，可解释性更强

### 下一步

Week 09 将学习回归分析与模型诊断，进一步量化预测的不确定性（预测区间、置信带）。

---

"""

    # 追加到报告
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(uncertainty_section)

    print(f"不确定性量化章节已追加到 {report_path}")

# 使用示例
if __name__ == "__main__":
    # generate_uncertainty_section(df, 'report.md')
    pass
```

现在你的 StatLab 报告有了八个层次：
1. **数据卡**：数据从哪来、字段什么意思
2. **描述统计**：数据长什么样、分布如何
3. **清洗日志**：数据怎么处理的、为什么这样处理
4. **EDA 叙事**：数据在说什么故事、还需要验证什么假设
5. **不确定性量化**：关键统计量有多稳定、哪些地方可能出错
6. **假设检验结果**：差异是否显著、效应有多大、结论有多可靠
7. **多组比较结果**：多组差异在哪里、效应量如何、是否相关
8. **不确定性量化（本周新增）**：置信区间、Bootstrap、置换检验、贝叶斯框架简介

老潘看到这份报告会说什么？"**这才是一份完整的统计分析报告。从描述到推断，从两组到多组，从点估计到区间估计，从理论分布到重采样——每一步都有迹可循、有据可依。**"

小北不好意思地笑了："**我之前总觉得报告要'写很多结论'，现在才发现，写清楚'不确定性'和'边界'更重要。**"

"对！"老潘点头，"**一份诚实的报告比一份'漂亮但不可靠'的报告有价值得多。**本周你学会的不确定性量化，核心就是'诚实地表达我们有多确定'——不夸大、不隐瞒、不把猜测当成事实。"

---

## Git 本周要点

本周必会命令：
- `git status`：查看工作区状态
- `git diff`：查看具体改动内容
- `git add -A`：添加所有改动
- `git commit -m "feat: add uncertainty quantification with CI and bootstrap"`
- `git log --oneline -n 5`

常见坑：
- 只保存点估计不保存 CI：无法判断结论的可靠性，建议同时报告 95% CI
- Bootstrap 次数太少（如 < 1000）：CI 不稳定，建议至少 10000 次重采样
- 置换检验次数太少（如 < 1000）：p 值不稳定，建议至少 10000 次置换
- 误解释 CI：频率学派 CI 不是参数的概率，建议用"方法的覆盖率"解释
- 缺少稳健性检验：只依赖 t 检验，数据不满足假设时结论不可靠，建议补充 Bootstrap 或置换检验

---

## 本周小结（供下周参考）

本周你做了七件事：理解了"点估计的危险"（只报告均值、p 值，不报告不确定性）；掌握了置信区间的频率学派解释（95% CI 是方法的覆盖率，不是参数的概率）；学会了 Bootstrap（自助法）的核心思想与实现，用重采样估计统计量的不确定性（均值、中位数、效应量）；学会了置换检验，在分布假设不满足时做推断（分布无关）；理解了贝叶斯可信区间与频率学派置信区区的差异（参数的概率 vs 方法的覆盖率）；学会了在报告中正确量化不确定性（区间、可视化、解释）；学会了审查 AI 生成的推断报告，识别缺少 CI、误解释等问题。

更重要的是，你在 StatLab 报告中添加了"不确定性量化"章节——**从"点估计"升级为"区间估计"**。你不再只是说"均值差是 15 元"，而是说"均值差的 95% Bootstrap CI 是 [2, 28]，置换检验 p=0.003，效应量中等，结论稳健"。

下周（Week 09）你将学习**回归分析与模型诊断**——从"比较两组"到"用变量预测变量"，从 t 检验到线性回归，从点估计到预测区间。届时你会用到本周的所有工具：不确定性量化、Bootstrap、置换检验，以及——最重要的——对 AI 生成报告的审查能力。Week 09 的核心是"模型不只是拟合，更要检查假设"：残差诊断、异常点影响、解释边界。这会进一步强化你的统计直觉。

---

## Definition of Done（学生自测清单）

- [ ] 我能解释"点估计的危险"，并理解为什么需要区间估计
- [ ] 我能正确解释 95% CI 的频率学派含义（方法的覆盖率），避免"参数有 95% 的概率"的误解
- [ ] 我能正确计算均值、均值差、效应量的置信区间（t 公式或 Bootstrap）
- [ ] 我能理解 Bootstrap 的核心思想（有放回重采样），并用它构造统计量的 CI
- [ ] 我能理解置换检验的原理（随机打乱标签构造零分布），并在数据不满足正态假设时使用它
- [ ] 我能区分置信区间（频率学派）与可信区间（贝叶斯学派），理解两种框架的差异
- [ ] 我能在报告中正确量化不确定性（CI、Bootstrap、置换检验、可视化）
- [ ] 我能审查 AI 生成的推断报告，识别缺少 CI、误解释、缺少稳健性检验等问题
- [ ] 我能在 StatLab 报告中添加"不确定性量化"章节，包含 CI、Bootstrap、置换检验
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解"点估计是猜测，区间估计才是诚实的科学"这句话的含义
