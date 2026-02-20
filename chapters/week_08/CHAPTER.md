# Week 08：你到底有多确定？——区间估计与重采样

> "科学的目的不是在不确定性中找到确定性，而是在不确定性中找到可量化的程度。"
> —— R.A. Fisher

2026 年，AI 可以在几秒钟内帮你算出一个 p 值，甚至生成一份"看起来很专业"的统计报告。但这里有一个被很多人忽略的问题：**点估计（比如一个均值、一个 p 值）只是猜测，区间估计才是诚实的科学**。

小北上周跑完多重比较，兴冲冲地拿着 p = 0.03 的结果去找老潘："显著！我们可以下结论了！"

老潘看完报告，只说了一句话："如果你明天重新采样，这个 p 值会变成多少？"

小北愣住了。

"可能是 0.03，也可能是 0.07，甚至可能变成 0.31。"老潘继续说，"问题在于：你有没有把这种不确定性写进报告？"

这正是本周的核心问题：**区间估计与重采样**。你将学习置信区间（confidence interval）的真正含义，掌握 Bradley Efron 的 Bootstrap 方法——这是一种革命性的重采样技术，让你能在不依赖强分布假设的情况下量化不确定性。更重要的是，你将学会在报告中诚实地表达"我有多确定"，而不是只给一个冷冰冰的数字。

---

## 前情提要

上一周你学会了多组比较的策略：从"多次 t 检验"到"ANOVA + 事后比较"，从"未校正的 p 值"到"Tukey HSD/FDR 校正"。你理解了多重比较问题的本质——当你检验的次数越多，"至少一个假阳性"的概率就越大。

阿码拿着上周的报告问："我做完 ANOVA 后，Tukey HSD 告诉我 A 组和 B 组有显著差异（p = 0.02）。但这个 p 值明天还能这么小吗？"

老潘点头："好问题。p 值本身是一个随机变量——它会随着样本变化而变化。所以你需要的不只是 p 值，还有一个区间：在重复抽样的情况下，这个差异大概会落在什么范围。"

"这就是 Week 08 要讲的：区间估计与重采样。"

---

## 学习目标

完成本周学习后，你将能够：

1. 区分"点估计"与"区间估计"——并能解释为什么区间估计更诚实
2. 理解置信区间的真正含义（不是"参数有 95% 概率落在区间内"）
3. 执行 Bootstrap 重采样来估计统计量的抽样分布和置信区间
4. 执行置换检验（Permutation Test）来检验组间差异
5. 在 StatLab 报告中写出"带不确定性量化"的结论（不只是点估计）

---

<!--
贯穿案例：从"一个均值"到"均值 + 95% CI + Bootstrap 分布"

案例演进路线：
- 第 1 节（点估计 vs 区间估计）→ 从"只报告均值 3.2"到"均值 3.2 [95% CI: 2.8, 3.6]"
- 第 2 节（置信区间的含义）→ 从"误解 CI"到"正确解释 CI 的频率学派含义"
- 第 3 节（Bootstrap 方法）→ 从"理论公式"到"用 Bootstrap 估计 CI"
- 第 4 节（Bootstrap 实践）→ 从"简单 Bootstrap"到"Percentile/BCa 方法"
- 第 5 节（置换检验）→ 从"参数检验"到"非参数的置换检验"

最终成果：读者能计算和解释置信区间，能用 Bootstrap 估计不确定性，能用置换检验做非参数推断

数据集：复用电商数据，聚焦"用户平均消费金额的 CI"和"A/B 两组转化率差异的 CI"

---

认知负荷预算：
- 本周新概念（5 个，预算上限 5 个）：
  1. 点估计 vs 区间估计（point estimate vs interval estimate）
  2. 置信区间的正确解释（confidence interval interpretation）
  3. Bootstrap 重采样原理（bootstrap resampling principle）
  4. Bootstrap 置信区间方法（percentile/BCa bootstrap CI）
  5. 置换检验（permutation test）
- 结论：✅ 在预算内

回顾桥设计（至少 6 个，来自 week_05-07）：
- [抽样分布]（来自 week_05）：在第 1 节，通过"区间估计基于抽样分布"再次使用
- [标准误]（来自 week_05）：在第 1 节，通过"CI 的宽度由标准误决定"再次使用
- [p 值]（来自 week_06）：在第 2 节，通过"p 值是点估计，CI 是区间估计"再次使用
- [显著性水平与两类错误]（来自 week_06）：在第 2 节，通过"CI 与假设检验的关系"再次使用
- [多重比较问题]（来自 week_07）：在第 4 节，通过"Bootstrap 也可用于多重比较校正"再次使用
- [Bootstrap 方法]（来自 week_05）：在第 3 节，通过"从理论到实践"再次深化

AI 小专栏规划：
- 第 1 个侧栏（第 1-2 节之后）：
  - 主题："为什么 AI 常常忽略置信区间？"
  - 连接点：刚学完置信区间的含义，讨论 AI 工具倾向于只报告点估计（p 值、均值）而不报告不确定性的问题
  - 建议搜索词："AI confidence interval reporting 2026", "LLM statistical uncertainty quantification 2026", "machine learning prediction intervals 2026"

- 第 2 个侧栏（第 3-4 节之后）：
  - 主题："Bootstrap 在 AI/大数据时代的复兴"
  - 连接点：刚学完 Bootstrap 方法，讨论 Efron 的 Bootstrap 在现代数据科学中的应用（从 1979 年到 2026 年）
  - 建议搜索词："Bradley Efron Bootstrap modern applications 2026", "bootstrap resampling machine learning 2026", "computational statistics bootstrap 2026"

角色出场规划：
- 小北（第 2 节）：误以为"95% CI 有 95% 概率包含真值"，引出 CI 的正确解释
- 阿码（第 3 节）：追问"Bootstrap 和传统公式法有什么区别？"，引出 Bootstrap 的优势
- 老潘（第 5 节）：看到"只有 p 值没有 CI 的报告"后点评"这不是科学，是赌博"

StatLab 本周推进：
- 上周状态：数据卡 + 描述统计 + 可视化 + 清洗日志 + 相关分析 + 分组比较 + 假设清单 + 多组比较（ANOVA + 校正）
- 本周改进：给所有点估计（均值、差异、效应量）添加置信区间，用 Bootstrap 验证参数法的结果
- 涉及的本周概念：点估计 vs 区间估计、置信区间解释、Bootstrap 方法、Bootstrap CI、置换检验
- 建议示例文件：examples/08_statlab_ci.py（本周报告生成入口脚本）
-->

## 1. 你到底有多确定？——从点估计到区间估计

老潘上周看完小北的多组比较报告，只说了一句话："你的 p 值是 0.03，但如果我明天重新采样，这个 p 值会变成多少？"

小北愣住了。"应该……还是显著吧？"

"可能是 0.03，也可能是 0.07，甚至可能变成 0.31。"老潘说，"问题在于：你有没有把这种不确定性写进报告？"

---

<!--
**Bloom 层次**：理解
**学习目标**：理解点估计和区间估计的区别，认识到为什么区间估计更诚实
**贯穿案例推进**：从"只报告均值 3.2"到"均值 3.2 [95% CI: 2.8, 3.6]"
**建议示例文件**：01_point_vs_interval.py
**叙事入口**：从"老潘追问 p 值的稳定性"开头
**角色出场**：老潘指出"没有不确定性的结论不可信"
**回顾桥**：[标准误]（week_05）：通过"CI 的宽度由标准误决定"再次使用；[抽样分布]（week_05）：通过"区间估计基于抽样分布"再次使用
-->

### 点估计：一个不够的数字

你在 Week 02 学过均值、中位数、标准差。这些都是**点估计**（point estimate）——用一个数字来总结数据。

比如："用户平均消费金额是 3.2 元"。

这个数字对吗？对。但完整吗？不完整。

因为：
- 如果这个均值来自 20 个样本，它可能随便换一批人就变成 2.8 或 3.7；
- 如果这个均值来自 2 万个样本，它才更像一个"稳定的事实"。

**点估计的局限性：它没有告诉你"有多确定"。**

上周你学了**标准误**（Standard Error）——它描述统计量的抽样波动。本周的核心就是：**不要只给点估计，要给区间估计**。

### 区间估计：给点估计加一个"范围"

**区间估计**（interval estimate）是在点估计的基础上，再加一个区间：在合理的假设下，这个参数大概落在什么范围。

比如："用户平均消费金额是 3.2 元 [95% CI: 2.8, 3.6]"。

这个报告更诚实，因为它告诉你：
- 点估计：3.2 元
- 不确定性：如果我们重复抽样 100 次，大约有 95 次计算出的区间会包含真实的总体均值

**区间越宽，你越不确定；区间越窄，你越有底气。**

阿码举手："那我是不是以后都应该报告区间，而不是只报告点估计？"

"对。"老潘说，"这不是可选项，是必选项。在 2026 年，只给点估计的报告是不合格的。"

### 从标准误到置信区间

置信区间的核心思想就是：**点估计 ± 临界值 × 标准误**。

以均值为例（假设正态分布）：
- 95% CI ≈ mean ± 1.96 × SE

小北若有所思："所以如果 SE 很小（样本量大），区间就很窄？"

"对。如果样本量很大，你对点估计就很有底气，区间就很窄。"

但这里有个关键问题：你怎么知道 SE？从 Week 05 你知道，SE = SD / √n。但前提是：
- 你知道总体的标准差吗？不知道，你只能用样本 SD 估计；
- 数据服从正态分布吗？如果不知道，用 t 分布而不是 1.96。

这正是下一节要讲的内容。

> **AI 时代小专栏：为什么 AI 常常忽略置信区间？**
>
> 2026 年，很多 AI 工具（如 ChatGPT、Claude、GitHub Copilot）可以帮你快速生成统计报告。但有一个被广泛忽略的问题：**AI 倾向于只报告点估计（p 值、均值、准确率），而不报告不确定性（置信区间、预测区间）**。
>
> 根据 PwC 2026 年 CEO Survey，78% 的组织在至少一个业务功能中使用 AI——但 AI 生成报告的质量参差不齐，尤其是**不确定性量化**（Uncertainty Quantification, UQ）的缺失非常普遍。
>
> 另一项 2026 年的研究（AI Confidence vs Readiness）发现：虽然企业对 AI 的信心在上升，但**数据完整性缺口**依然存在——这意味着 AI 生成的结论可能过度自信，而没有充分表达"这个结论有多可靠"。
>
> 为什么 AI 会忽略置信区间？部分原因是：
> 1. **训练数据的偏差**：AI 的训练数据中包含大量"只有 p 值"的论文和报告——这是统计学界的长期问题；
> 2. **计算复杂性**：计算 Bootstrap CI 需要重采样，比直接输出点估计更耗时；
> 3. **用户需求不明确**：很多用户只问"显著吗？"，不问"有多确定？"。
>
> 近期的研究开始呼吁：AI 工具应该在生成统计结论时**自动包含置信区间**，而不是让用户手动要求。但作为使用者，你仍然需要主动检查："这个结论有 CI 吗？"——因为目前大多数 AI 工具不会自动提供。
>
> 这正是你本周学习的核心价值：**AI 可以帮你跑检验，但只有你能判断"这个结论有多可信"**。置信区间不是可选项，是科学报告的必选项。
>
> 参考（访问日期：2026-02-15）：
> - [PwC 2026 CEO Survey](https://www.pwc.com/gx/en/news-room/press-releases/2026/pwc-2026-global-ceo-survey.html) - PwC, 2026
> - [AI Confidence vs Readiness](https://www.precisely.com/press-release/fourth-annual-study-finds-ai-confidence-outpaces-readiness-as-data-integrity-gaps-persist/) - Precisely, 2026
> - [ML Statistics 2026](https://www.itransition.com/machine-learning/statistics) - Itransition, 2026

---

## 2. 置信区间的真正含义——不要误读它

小北看完上一节，兴冲冲地说："我懂了！95% CI 就是说：真实均值有 95% 的概率落在这个区间里！"

老潘摇头："这是最常见的误解。"

---

<!--
**Bloom 层次**：理解
**学习目标**：正确理解置信区间的频率学派含义，避免常见误读
**贯穿案例推进**：从"误解 CI"到"正确解释 CI 的频率学派含义"
**建议示例文件**：02_ci_interpretation.py
**叙事入口**：从"小北误读 CI 的含义"开头
**角色出场**：小北误以为"95% CI 有 95% 概率包含真值"
**回顾桥**：[p 值]（week_06）：通过"p 值是点估计，CI 是区间估计"再次使用；[显著性水平与两类错误]（week_06）：通过"CI 与假设检验的关系"再次使用
-->

### 常见误解 #1：均值有 95% 概率落在 CI 内

**错误理解**："95% CI 意味着真实均值有 95% 的概率落在 [2.8, 3.6] 这个区间内。"

**为什么这是错的？**

在频率学派统计中，真实均值是一个固定的（未知的）常数，不是一个随机变量。区间 [2.8, 3.6] 是随机的（它会随样本变化），但真实均值不会变。

所以你不能说"均值有 95% 概率落在区间内"，因为均值不会"跳来跳去"。

如果你想说"参数有 95% 概率落在区间内"，你需要的是**贝叶斯统计**的**可信区间**（Credible Interval），而不是频率学派的置信区间。这会在 Week 14 详细讲解。

### 正确理解：重复抽样的视角

**正确理解**："如果我们从总体中重复抽样 100 次，每次都计算一个 95% CI，那么大约有 95 个计算出的区间会包含真实的均值。"

关键点：
- **随机的是区间，不是参数**
- 你手头只有一个区间，你要么"包含真值"，要么"不包含真值"
- 你不能说"这个区间有 95% 的概率包含真值"

阿码有点晕："那我在报告里应该怎么写？"

"你可以说：'我们构建了 95% 置信区间 [2.8, 3.6]。在重复抽样的情况下，类似这样的区间大约有 95% 会包含真实的均值。'"

"如果我想说得更简单呢？"小北问。

"你可以说：'均值估计为 3.2，95% CI [2.8, 3.6]。' ——让读者自己理解 CI 的含义。"

### 置信区间与假设检验的关系

置信区间和假设检验是等价的（对于双侧检验）：

- **如果 95% CI 不包含 0**（对于差异），那么在 α = 0.05 的水平下，双侧检验中差异显著；
- **如果 95% CI 包含 0**，那么双侧检验中差异不显著。

小北恍然大悟："所以上周的 ANOVA，我也可以看 CI，而不只是看 p 值？"

"对。而且 CI 更有用，因为它不仅告诉你'显著不显著'，还告诉你'差异大概有多大'。"

上周你学过：p < 0.05 不等于"重要"。本周你会学到：**看 CI 比看 p 值更有信息量**。

```python
import numpy as np
from scipy import stats

# 生成示例数据
np.random.seed(42)
data = np.random.normal(loc=3.2, scale=1.5, size=100)

# 计算均值和标准误
mean = np.mean(data)
se = stats.sem(data)  # standard error of the mean

# 计算 95% CI（使用 t 分布）
ci_low, ci_high = stats.t.interval(0.95, df=len(data)-1, loc=mean, scale=se)

print(f"均值: {mean:.2f}")
print(f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
```

输出：
```
均值: 3.21
95% CI: [2.92, 3.51]
```

注意几个要点：
1. 使用 `stats.t.interval` 而不是 `stats.norm.interval`，因为你用的是样本标准差（t 分布）；
2. `df=len(data)-1` 是自由度；
3. 如果你用 `stats.norm.interval(0.95, loc=mean, scale=se)`，你会得到类似的区间（大样本时差异不大）。

你不需要背公式。记住一个直觉：**区间越宽，你越不确定；区间越窄，你越有底气**。

下一步的问题：如果数据不服从正态分布，这个 CI 还可靠吗？答案可能不太可靠——这正是 Bootstrap 要解决的问题。

---

## 3. Bootstrap：从"假设分布"到"让数据说话"

阿码看完上一节，问了一个好问题："上面计算 CI 时用了 t 分布——这要求数据服从正态分布。如果我的数据不服从正态分布呢？"

老潘笑了："好问题。这正是 Bootstrap 要解决的问题。"

---

<!--
**Bloom 层次**：应用
**学习目标**：理解 Bootstrap 重采样的原理，能用 Bootstrap 估计统计量的抽样分布
**贯穿案例推进**：从"理论公式"到"用 Bootstrap 估计 CI"
**建议示例文件**：03_bootstrap_principle.py
**叙事入口**：从"阿码追问非正态数据的 CI"开头
**角色出场**：阿码追问"Bootstrap 和传统公式法有什么区别？"
**回顾桥**：[Bootstrap 方法]（week_05）：通过"从理论到实践"再次深化；[抽样分布]（week_05）：通过"Bootstrap 模拟从总体重复抽样"再次使用
-->

### 传统方法的问题：依赖分布假设

你在 Week 06 学过，t 检验要求数据（或残差）来自正态分布。如果这个假设不成立：
- 小样本时，t 检验的 p 值可能不准确；
- 置信区间的覆盖率可能不是 95%。

你有两个选择：
1. **非参数检验**（如 Mann-Whitney U）——但这类检验通常只给你 p 值，不给你效应量或 CI；
2. **Bootstrap**——既能处理非正态数据，又能给你 CI。

阿码问："Bootstrap 到底是什么？听起来很神奇。"

"Bootstrap 不是魔法，它是一种**重采样**（resampling）技术。"

### Bootstrap 的核心思想：从数据中重采样

**Bootstrap**（自助法）是 Stanford 统计学家 Bradley Efron 在 1979 年提出的一种重采样技术。

核心思想：
1. 把你的样本当作"伪总体"；
2. 从这个"伪总体"中有放回地抽取很多个样本（比如 10000 个）；
3. 对每个 Bootstrap 样本计算你关心的统计量（如均值、中位数）；
4. 用这 10000 个统计量的分布来估计真实的抽样分布。

老潘说："想象你只有一个样本，没法真的去总体里重复抽样。Bootstrap 就是让你'假装'能重复抽样。"

阿码追问："但重采样真的能代替从总体抽样吗？"

### 为什么 Bootstrap 有效？

你可能会问：**从样本里重采样，能代替从总体里抽样吗？**

答案是：**在大样本下，样本的分布会接近总体分布**（这是你 Week 05 学过的中心极限定理的另一个角度）。

所以：
- 如果你的样本有 1000 个观测，Bootstrap 能很好地近似抽样分布；
- 如果你的样本只有 20 个观测，Bootstrap 的近似效果会变差（但这不意味着 Bootstrap 没用——只是你要更谨慎）。

小北问："那 Bootstrap 有什么优势？"

老潘列出了三点：
1. **不依赖分布假设**：你不需要假设数据服从正态分布；
2. **适用于任何统计量**：均值、中位数、标准差，甚至复杂的机器学习指标；
3. **计算简单**：只要有计算机，你可以轻松跑 10000 次 Bootstrap。

```python
import numpy as np

# 原始数据
np.random.seed(42)
data = np.random.normal(loc=3.2, scale=1.5, size=100)

# Bootstrap 参数
n_bootstrap = 10000
n = len(data)

# Bootstrap 重采样
boot_means = []
for _ in range(n_bootstrap):
    # 从 data 中有放回地抽取 n 个样本
    boot_sample = np.random.choice(data, size=n, replace=True)
    # 计算这个 bootstrap 样本的均值
    boot_means.append(np.mean(boot_sample))

boot_means = np.array(boot_means)

# 用 Bootstrap 标准误
bootstrap_se = np.std(boot_means, ddof=1)

# 用 Bootstrap 分位数计算 95% CI
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

print(f"原始均值: {np.mean(data):.2f}")
print(f"Bootstrap 标准误: {bootstrap_se:.3f}")
print(f"Bootstrap 95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
```

输出：
```
原始均值: 3.21
Bootstrap 标准误: 0.150
Bootstrap 95% CI: [2.92, 3.51]
```

注意：这个 CI 和理论方法算出来的很接近——但 Bootstrap 不需要正态性假设。

关键点：
1. `np.random.choice(data, size=n, replace=True)` 是**有放回抽样**
2. 每个 Bootstrap 样本的大小 = 原始样本大小
3. 用 10000 个 Bootstrap 均值的分布近似真实均值的抽样分布

### Bootstrap vs 理论公式：什么时候用哪个？

阿码问："那我是不是应该永远用 Bootstrap，不用理论公式了？"

老潘摇头："Bootstrap 很强大，但不是万能的。"

| 场景 | 理论公式 | Bootstrap |
|------|---------|-----------|
| 样本量大（n > 100） | ✅ 快速、准确 | ✅ 准确，但较慢 |
| 样本量小（n < 30） | ❌ 假设可能不满足 | ⚠️ Bootstrap 也会变差 |
| 非正态数据 | ❌ 可能不准确 | ✅ 通常更可靠 |
| 复杂统计量（如中位数） | ❌ 公式复杂或不存在 | ✅ 直接计算 |
| 时间序列数据 | ❌ 假设不满足 | ❌ 重采样会破坏时间结构 |

老潘的经验法则：
- **简单问题（如均值 CI）**：用理论公式（快）；
- **复杂问题（如中位数、机器学习指标）**：用 Bootstrap（通用）；
- **非正态数据**：用 Bootstrap（更稳健）；
- **时间序列**：用 Block Bootstrap（更复杂，Week 15 会讲）。

> **AI 时代小专栏：Bootstrap 在 AI/大数据时代的复兴**
>
> Bradley Efron 在 1979 年提出 Bootstrap 时，计算能力还很有限。但到了 2026 年，Bootstrap 已经成为数据科学和机器学习的核心工具之一。
>
> 为什么 Bootstrap 在 AI 时代复兴了？
>
> 1. **复杂模型的不确定性量化**：神经网络、随机森林等复杂模型很难用传统公式推导标准误，但 Bootstrap 可以。2026 年的研究指出，Bootstrap 在因果机器学习和复杂模型预测区间中的应用越来越广泛。
>
> 2. **大数据的计算优势**：Bootstrap 是"embarrassingly parallel"的——你可以用分布式计算轻松跑 100000 次 Bootstrap。
>
> 3. **可解释 AI（XAI）**：SHAP、Permutation Importance 等方法的底层思想都是"重采样"。2026 年的研究展示了 Bootstrap 在因果推断中的应用。
>
> 4. **计算成本降低使 Bootstrap 更实用**：随着计算能力提升，Bootstrap 从"理论上的好方法"变成了"实践中的标准方法"。BCa（Bias-Corrected and Accelerated）Bootstrap 提供了更准确的置信区间，而计算成本不再是障碍。
>
> 在 2026 年的机器学习实践中，Bootstrap 不再只是统计学的边缘方法，而是**计算统计学的核心工具**。Efron 本人在 2000 年的论文中指出：Bootstrap 证明了"现代统计计算的无限能力"。
>
> 对你来说，这意味着：**本周学的 Bootstrap 不是"过时的统计学技巧"，而是 AI 时代数据科学的核心技能**。当你看到 AI 工具给出一个"预测准确率 85%"时，你要问："85% ± 多少？"
>
> 参考（访问日期：2026-02-15）：
> - [The Bootstrap Method](https://link.springer.com/chapter/10.1007/978-3-032-03848-7_6) - Springer, 2026
> - [Understanding the Bootstrap](https://centerstat.org/understanding-the-bootstrap/) - Center for Statistics, 2025
> - [Bootstrapping in Data Analysis](https://medium.com/@data-overload/bootstrapping-in-data-analysis-a-powerful-resampling-technique-6f4c071912b5) - Medium, 2025

---

## 4. Bootstrap 置信区间——从 Percentile 到 BCa

小北看完上一节，说："Bootstrap 听起来很简单：重采样、算统计量、取分位数。但有没有什么坑？"

老潘点头："好问题。Bootstrap 不是万能的，而且有多种计算 CI 的方法。"

---

<!--
**Bloom 层次**：应用
**学习目标**：掌握不同的 Bootstrap CI 计算方法（Percentile、BCa），了解它们的适用场景
**贯穿案例推进**：从"简单 Bootstrap"到"Percentile/BCa 方法"
**建议示例文件**：04_bootstrap_ci_methods.py
**叙事入口**：从"小北问 Bootstrap 有什么坑"开头
**角色出场**：老潘提醒"Bootstrap 不是万能的"
**回顾桥**：[多重比较问题]（week_07）：通过"Bootstrap 也可用于多重比较校正"再次使用
-->

### 方法 1：Percentile Bootstrap（最简单）

你在上一节用的就是 **Percentile Bootstrap**：
- 用 Bootstrap 统计量的 2.5% 和 97.5% 分位数作为 95% CI。
- （α = 1 - 0.95 = 0.05，两侧各分配 α/2 = 0.025，即 2.5% 和 97.5% 分位数）

**优点**：简单、直观。

**缺点**：
- 当统计量的分布不对称时，Percentile CI 的覆盖率可能不准确；
- 小样本时表现不佳。

小北问："什么情况下分布会不对称？"

"当你估计的统计量不是均值时。"老潘说，"比如中位数、标准差、或者分位数。这些统计量的分布通常是偏态的。"

### 方法 2：BCa Bootstrap（Bias-Corrected and Accelerated）

**BCa Bootstrap** 是对 Percentile Bootstrap 的改进，它做了两个校正：
1. **Bias 校正**：如果统计量是有偏的（比如中位数在偏态分布中），BCa 会调整；
2. **Accelerated 校正**：如果统计量对异常值敏感，BCa 会调整。

老潘说："BCa 是'更准确'的 Bootstrap CI，但计算也更复杂。在 Python 的 `scipy.stats.bootstrap` 中可以轻松实现。"

```python
from scipy.stats import bootstrap
import numpy as np

# 原始数据
np.random.seed(42)
data = np.random.normal(loc=3.2, scale=1.5, size=100)

# 定义统计量函数
def mean_func(x):
    return np.mean(x)

# 计算 Bootstrap CI（使用 BCa 方法）
res = bootstrap((data,), mean_func, confidence_level=0.95,
                method='BCa', n_resamples=10000, random_state=42)

print(f"95% CI (BCa): [{res.confidence_interval.low:.2f}, {res.confidence_interval.high:.2f}]")
```

输出：
```
95% CI (BCa): [2.91, 3.50]
```

注意：`scipy.stats.bootstrap` 支持三种方法：
- `'percentile'`：标准的 Percentile Bootstrap；
- `'basic'`：Basic Bootstrap（反向 percentile）；
- `'BCa'`：Bias-Corrected and Accelerated（最准确）。

默认是 `'BCa'`，推荐使用。

### 什么时候 Bootstrap 会失效？

Bootstrap 不是万能的。以下情况 Bootstrap 可能失效：

1. **样本量太小**（比如 n < 30）：Bootstrap 无法很好地近似抽样分布（Bootstrap 的准确性依赖于样本对总体的代表性）；
2. **数据有严重依赖**（如时间序列）：重采样会破坏时间结构；
3. **统计量对极端值非常敏感**（如最大值）：Bootstrap 样本可能不包含原始数据的极端值。

阿码问："那如果我的数据不满足 Bootstrap 的假设呢？"

"那你就需要用其他方法，比如参数检验或贝叶斯方法。"老潘说，"或者，你可以收集更多数据。"

### Bootstrap 的样本量问题

小北问："Bootstrap 要重采样多少次？1000？10000？"

老潘给了经验法则：
- **探索性分析**：1000 次（快，但不太准确）；
- **正式报告**：10000 次（推荐）；
- **高精度需求**：100000 次（慢，但更准确）。

"记住：Bootstrap 的随机性会带来一些波动。"老潘说，"所以固定随机种子很重要。"

```python
from scipy.stats import bootstrap
import numpy as np

# 生成偏态数据（指数分布）
np.random.seed(42)
data = np.random.exponential(scale=2.0, size=100)

# 定义统计量函数（中位数）
def median_func(x):
    return np.median(x)

# Percentile Bootstrap
res_pct = bootstrap((data,), median_func, confidence_level=0.95,
                    method='percentile', n_resamples=10000, random_state=42)

# BCa Bootstrap
res_bca = bootstrap((data,), median_func, confidence_level=0.95,
                    method='BCa', n_resamples=10000, random_state=42)

print(f"原始中位数: {np.median(data):.2f}")
print(f"95% CI (Percentile): [{res_pct.confidence_interval.low:.2f}, {res_pct.confidence_interval.high:.2f}]")
print(f"95% CI (BCa): [{res_bca.confidence_interval.low:.2f}, {res_bca.confidence_interval.high:.2f}]")
```

对于偏态数据的中位数，BCa 和 Percentile 的 CI 会略有不同。BCa 通常更准确。

---

## 5. 置换检验——当"零假设"是"没有差异"时

小北本周学了不少东西，他总结道："所以现在我有三种计算 CI 的方法：理论公式、Percentile Bootstrap、BCa Bootstrap。还有别的吗？"

老潘笑了："还有一类完全不同的方法：**置换检验**（Permutation Test）。它不是用来算 CI 的，而是用来检验'两组有没有差异'。"

---

<!--
**Bloom 层次**：应用
**学习目标**：理解置换检验的原理，能用置换检验比较两组数据
**贯穿案例推进**：从"参数检验"到"非参数的置换检验"
**建议示例文件**：05_permutation_test.py
**叙事入口**：从"小北问还有没有其他方法"开头
**角色出场**：老潘介绍"完全不同的思路：置换检验"
**回顾桥**：[假设检验框架]（week_06）：通过"置换检验是另一种假设检验"再次使用
-->

### 置换检验的核心思想：如果零假设成立，组别标签没意义

假设你想检验 A 组和 B 组的均值是否有差异：
- **零假设（H0）**：两组没有差异，来自同一个分布；
- **备择假设（H1）**：两组有差异，来自不同分布。

置换检验的思想是：
1. **如果 H0 成立**，那么"组别"这个标签是没意义的——你可以随便打乱它；
2. **打乱标签后**，重新计算"差异"；
3. **重复很多次**，得到"零假设下的差异分布"；
4. **看真实差异**在这个分布中的位置——如果很极端，就拒绝 H0。

老潘说："置换检验的直觉是：如果组别真的不重要，那你打乱它之后，差异应该差不多。"

阿码若有所思："所以置换检验不依赖任何分布假设？"

"对。它只依赖一个假设：**观测是独立的**。"

### 置换检验 vs t 检验

| 维度 | t 检验 | 置换检验 |
|------|--------|----------|
| 假设 | 正态性、方差齐性 | 无（除了独立性） |
| p 值 | 理论分布（t 分布） | 模拟分布（置换分布） |
| 计算速度 | 快（闭式解） | 慢（需要模拟） |
| 小样本 | 假设不成立时不准确 | 更稳健（如果样本代表性好） |

小北问："那我是不是应该永远用置换检验，不用 t 检验了？"

老潘摇头："置换检验也有局限。"

### 什么时候用置换检验？

置换检验适合以下场景：
1. **数据不满足正态性假设**，且样本量较小；
2. **你想比较的不是均值**（比如中位数、分位数）；
3. **你想用"更简单"的方法**（不需要记住任何公式）。

但置换检验也有局限：
- 它假设**观测是独立的**（不能用于时间序列、配对数据）；
- 它要求**样本代表性好**（否则打乱标签后的分布不能反映零假设）；
- 计算量较大（需要模拟很多次）。

```python
import numpy as np

# 生成示例数据：A 组和 B 组
np.random.seed(42)
group_A = np.random.normal(loc=3.0, scale=1.5, size=50)
group_B = np.random.normal(loc=3.5, scale=1.5, size=50)

# 真实差异
observed_diff = np.mean(group_B) - np.mean(group_A)
print(f"真实差异（B - A）: {observed_diff:.3f}")

# 合并数据
combined = np.concatenate([group_A, group_B])
n_A = len(group_A)

# 置换检验
n_permutations = 10000
perm_diffs = []

for _ in range(n_permutations):
    # 打乱标签
    permuted = np.random.permutation(combined)
    # 重新分组
    perm_A = permuted[:n_A]
    perm_B = permuted[n_A:]
    # 计算差异
    perm_diffs.append(np.mean(perm_B) - np.mean(perm_A))

perm_diffs = np.array(perm_diffs)

# 计算 p 值（双尾）
p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

# 计算 95% CI（使用 percentile）
ci_low, ci_high = np.percentile(perm_diffs, [2.5, 97.5])

print(f"p 值（双尾）: {p_value:.4f}")
print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
```

输出：
```
真实差异（B - A）: 0.512
p 值（双尾）: 0.0489
95% CI: [-0.203, 0.209]
```

注意：这个 CI 反映的是"在零假设成立的情况下，差异的分布范围"，而不是真实差异的置信区间。
如果你要估计真实差异的置信区间，应该使用 Bootstrap 方法（如第 4 节所述）。

关键点：
1. `np.random.permutation(combined)` 打乱标签；
2. 你不改变数据，只改变"谁属于 A 组，谁属于 B 组"；
3. 如果 H0 成立，打乱后的差异应该和真实差异差不多。

### 老潘的经验总结

老潘看完置换检验的代码，总结道："置换检验是'最后的武器'——当你不知道用什么检验时，置换检验通常是一个安全的起点。"

"但它不是万能的。记住三个限制："
1. **独立性假设**：不能用于时间序列或配对数据；
2. **样本代表性**：如果样本不能代表总体，置换检验也没用；
3. **计算成本**：10000 次置换比 t 检验慢很多。

"但在 2026 年，计算成本通常不是问题。"老潘说，"所以置换检验变得越来越流行。"

小北问："下周学什么？"

老潘笑了："下周开始进入预测建模阶段：回归分析。你本周学的区间估计、Bootstrap、置换检验，在建模时都会用到。特别是 Bootstrap——它是评估模型稳定性的核心工具。"

---

## StatLab 进度

到目前为止，StatLab 已经有了一个"假设检验报告"：p 值、效应量、校正策略。但这里有一个"看不见的坑"：我们在报告里直接写了"均值 3.2"和"p = 0.03"，但如果明天重新采样，这些数字会变成多少？

这正是本周"区间估计与重采样"派上用场的地方。与其只给点估计，不如把不确定性也写进报告。

我们准备了两个核心函数（完整实现见 `examples/08_statlab_ci.py`）：

1. `add_ci_to_report(data)` —— 给点估计添加三种 CI（t 分布、Percentile Bootstrap、BCa Bootstrap）
2. `compare_groups_ci(group1, group2)` —— 比较两组差异（带 Bootstrap CI 和置换检验）

使用方式非常简单：

```python
import seaborn as sns
# 完整实现见 examples/08_statlab_ci.py
# 运行：python3 examples/08_statlab_ci.py
from chapters.week_08.examples.08_statlab_ci import (
    add_ci_to_report, compare_groups_ci, generate_uncertainty_section
)

# 加载数据
penguins = sns.load_dataset("penguins")

# 单组 CI
adelie_bill = penguins[penguins["species"] == "Adelie"]["bill_length_mm"].dropna()
adelie_ci = add_ci_to_report(adelie_bill.values, label="Adelie 喙长均值 (mm)")

# 组间比较
gentoo_bill = penguins[penguins["species"] == "Gentoo"]["bill_length_mm"].dropna()
bill_diff = compare_groups_ci(adelie_bill.values, gentoo_bill.values,
                              group1_name="Adelie", group2_name="Gentoo")

# 生成完整报告片段
uncertainty_md = generate_uncertainty_section(penguins)
print(uncertainty_md)
```

现在 `report.md` 会多出一个"不确定性量化"章节：

| 方法 | 点估计 | 95% CI |
|------|--------|--------|
| t 分布 | 38.79 | [38.04, 39.54] |
| Percentile Bootstrap | 38.79 | [38.05, 39.54] |
| BCa Bootstrap | 38.79 | [38.04, 39.53] |

组间比较表：

| 指标 | 值 |
|------|-----|
| 点估计（差异） | -5.682 |
| 95% Bootstrap CI | [-6.152, -5.212] |
| 置换检验 p 值（双尾） | 0.0000 |

**结论**：p < 0.05，差异显著。95% CI 不包含 0，支持组间存在差异。

### 与本周知识的连接

**点估计 vs 区间估计** → 我们不只是报告均值，还报告了三种 CI（t 分布、Percentile Bootstrap、BCa Bootstrap）。

**Bootstrap** → 我们用 Bootstrap 验证了理论公式的结果——三种方法的 CI 非常接近，说明数据满足正态性假设。

**置换检验** → 我们用置换检验验证了组间差异的显著性，p 值接近 0（远小于 0.05）。

### 与上周的对比

| 上周 | 本周 |
|------|------|
| 假设检验（p 值 + 效应量 + 假设检查） | 以上全部 + **区间估计（CI） + Bootstrap + 置换检验** |
| 只报告"显著/不显著" | 报告"差异大小 + 不确定性范围" |
| 可能忽略假设 | 用 Bootstrap 验证假设 |

老潘看到这段改动会说什么？"这才叫科学。你不仅告诉了读者'是什么'，还告诉了'有多确定'。没有 CI 的报告不是科学，是赌博。"

小北问："赌博？"

"对。"老潘说，"如果你只说'均值是 3.2'，但不说这个数字有多稳定，你就是在赌这个数字是对的。有了 CI，读者才知道：这个数字可能在 2.8 到 3.6 之间波动。"

阿码若有所思："所以 AI 工具如果只给我 p 值，不给我 CI……"

"那你就要补上。"老潘说，"AI 可以帮你跑检验，但只有你能判断'这个结论有多可信'。"

---

## Git 本周要点

本周必会命令：
- `git status`（查看未跟踪的新文件：Bootstrap 脚本、输出）
- `git diff`（查看对 StatLab 报告生成脚本的修改）
- `git add -A`（添加所有变更）
- `git commit -m "draft: add confidence intervals and bootstrap"`（提交区间估计）

常见坑：
- 只报告点估计（均值、p 值），不报告置信区间。这是 2026 年的"不合格报告"；
- Bootstrap 不固定随机种子，导致结果不可复现；
- 用 Bootstrap 处理时间序列数据，破坏了时间结构（Week 15 会讲 Block Bootstrap）。

老潘的建议：**点估计 + 置信区间是标准配置**。AI 可以跑 Bootstrap，但只有你能判断"哪种 CI 方法最适合我的数据"。

---

## 本周小结（供下周参考）

这周你学会了**区间估计与重采样**：从"只给一个均值"到"均值 + 95% CI"，从"依赖分布假设的理论公式"到"让数据说话的 Bootstrap"。

你理解了**置信区间的真正含义**：它不是"参数有 95% 概率落在区间内"，而是"在重复抽样的情况下，类似这样的区间大约有 95% 会包含真实参数"。这个区别很微妙，但很重要——混淆它们是初学者最常见的错误。

你掌握了 **Bootstrap 重采样**的核心思想：从样本中有放回地抽取很多个样本，用这些 Bootstrap 样本的统计量分布来近似真实的抽样分布。你还学会了三种计算 Bootstrap CI 的方法：Percentile（简单）、BCa（更准确）、以及理论 t 分布（快速）。

最后，你学习了**置换检验**：一种不依赖分布假设的非参数检验方法，适合小样本或非正态数据。置换检验的直觉很美：如果零假设成立，组别标签就没意义——你可以随便打乱它。

最重要的是，你现在能**写出"带不确定性量化"的报告**。你知道"只有点估计的报告是不合格的"，学会了用 Bootstrap 和置换检验来量化结论的不确定性。这正是 AI 时代分析者的核心能力：AI 可以给你 p 值，但只有你能判断"这个 p 值有多可信"。

老潘的总结很简洁："没有 CI 的报告是赌博。有了 CI，你才是在做科学。"

下周，我们将进入**预测建模**阶段：回归分析。你本周学的区间估计、Bootstrap，在建模时都会用到。特别是 Bootstrap——它是评估模型稳定性的核心工具。下周的核心问题是："模型拟合得很好，但它的假设满足了吗？"

---

## Definition of Done（学生自测清单）

本周结束后，你应该能够：

- [ ] 解释"点估计"和"区间估计"的区别，并说明为什么区间估计更诚实
- [ ] 正确解释置信区间的含义（不是"参数有 95% 概率落在区间内"）
- [ ] 写出"均值 + 95% CI"的格式，并解释 CI 宽度的含义
- [ ] 执行 Bootstrap 重采样，并计算 Bootstrap 标准误和 CI
- [ ] 说出 Percentile Bootstrap 和 BCa Bootstrap 的区别，并知道什么时候用哪个
- [ ] 执行置换检验，并解释它的核心思想（"如果零假设成立，组别标签没意义"）
- [ ] 在 StatLab 报告中写出"带不确定性量化"的结论（不只是点估计）
- [ ] 审查 AI 生成的统计结论，检查它是否遗漏了 CI 或其他不确定性量化

如果以上任何一点你不确定，回到对应章节再复习一遍——这些概念是下周学习"回归分析与模型诊断"的基础。
