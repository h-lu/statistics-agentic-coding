# Week 07：比较三组或更多——从"多次 t 检验"到"方差分解"

> "If you torture the data long enough, it will confess."
> — Ronald Coase

最近两年，一个危险的统计错误正在加速扩散：**很多人在有 5 个城市、3 种产品、4 个实验组的时候，依然用"两两 t 检验"来做比较**——跑 10 次、20 次检验，然后挑出 p<0.05 的那些宣布"发现显著差异"。

但 AI 不会告诉你的是：**这恰恰是统计推断最隐蔽的陷阱**。多次检验会系统性放大假阳性率——原本 5% 的第一类错误率在 20 次检验后会飙升至 64% 以上。更糟糕的是，AI 生成的分析报告往往不会主动提醒你"你做了多少次比较"，更不会建议你使用 Bonferroni 校正或 FDR 控制。

本周的核心任务是：**掌握多组比较的正确方法**——从 ANOVA 的方差分解思想，到事后检验（Tukey HSD）的多重比较校正，再到效应量（η²）的解释，最后理解"相关 ≠ 因果"的统计直觉。只有掌握了这套框架，你才能在 AI 时代避免"数据刑讯逼供"（data torturing），做出可靠的组间比较决策。

---

## 前情提要

上周（Week 06），小北学会了假设检验的完整框架：从 H0/H1 的设定，到 p 值的正确理解，再到效应量和两类错误的权衡。他兴奋地对老潘说："**我终于能判断'两组是否有差异'了！**"

老潘看了看他的报告，只问了一句："**如果你有 5 个城市，每个城市都要和其他 4 个比，你要跑 10 次 t 检验吗？**"

小北愣住了："呃……可以吗？"

"可以，但你会掉进一个陷阱。"老潘说，"**每次检验都有 5% 的假阳性率，跑 10 次，总假阳性率会飙升到 40% 以上**。这就是'多重比较'问题——你需要的新工具叫 ANOVA（方差分析），它不是'多次 t 检验'，而是'一次检验所有组'。"

这正是本周要解决的问题——**多组比较与多重比较校正**。你不再只是比较"两组是否不同"，而是回答"三组或更多组的均值是否至少有一对不同"，并在事后做配对比较时控制假阳性风险。

---

## 学习目标

完成本周学习后，你将能够：
1. 理解 ANOVA 的方差分解思想（组间方差 vs 组内方差），避免"多次 t 检验"的陷阱
2. 正确使用 F 检验判断多组均值是否存在系统性差异
3. 进行事后检验（Tukey HSD、Bonferroni 校正），控制多重比较的假阳性率
4. 计算并解释 ANOVA 的效应量（η²），区分统计显著与实际意义
5. 检验 ANOVA 的前提假设（正态性、方差齐性、独立性），并在假设违反时选择稳健替代方法
6. 理解卡方检验在分类变量关联检验中的应用，区分"关联"与"因果"
7. 审查 AI 生成的多组比较报告，识别未校正多重比较、混淆相关与因果等常见谬误

---

<!--
贯穿案例：电商平台的城市差异化策略——"五个城市的用户消费真的不同吗？"

本周贯穿案例是一个多组比较场景：某电商平台在 5 个城市运营，产品经理想知道"不同城市用户的平均消费是否有差异"。读者需要用 ANOVA 判断组间差异是否显著，并在事后检验中找出具体哪些城市对之间存在差异。

- 第 1 节：从"多次 t 检验"的陷阱 → 案例从"准备跑 10 次 t 检验"变成"理解为什么需要 ANOVA"
- 第 2 节：ANOVA 核心思想（方差分解） → 案例从"手动计算 F 统计量"变成"理解组间/组内方差的含义"
- 第 3 节：ANOVA 的前提假设检查 → 案例从"直接跑 ANOVA"变成"先做体检（正态性、方差齐性、独立性）"
- 第 4 节：ANOVA 实战 + 效应量 → 案例从"知道 F 显著"变成"完整的 ANOVA 表 + 假设验证 + η²"
- 第 5 节：事后检验与多重比较校正 → 案例从"知道有差异"变成"知道哪些城市对有差异（Tukey HSD）"
- 第 6 节：卡方检验与分类变量关联 → 案例从"连续变量比较"变成"分类变量关联（城市 vs 用户等级）"
- 第 7 节：AI 审查训练 → 案例从"自己做 ANOVA"变成"审查 AI 生成的多组比较报告，找出多重比较未校正等谬误"

最终成果：读者完成一个完整的多组比较分析，产出：
- 1 份完整的 ANOVA 表（SS、df、MS、F、p、η²）
- 1 个 Tukey HSD 事后检验结果（标注哪些城市对显著不同）
- 1 个前提假设检查报告（正态性、方差齐性、独立性）
- 1 个卡方检验结果（分类变量关联）
- 1 份 AI 生成报告的审查清单（标注多重比较风险、相关≠因果混淆）

认知负荷预算：
- 本周新概念（5 个，预算上限 5 个）：
  1. ANOVA（方差分析）- 应用层次
  2. F 统计量与 F 分布 - 理解层次
  3. 多重比较校正（Bonferroni、FDR）- 理解层次
  4. 事后检验（Tukey HSD）- 应用层次
  5. η²（eta-squared）效应量 - 应用层次
- 结论：✅ 在预算内（5 个）

回顾桥设计（至少 2 个，来自 Week 02/03/04/05/06）：
- [t 检验与两组比较]（来自 week_06）：在第 1 节，用"多次 t 检验的陷阱"引出 ANOVA 的必要性
- [方差与标准差]（来自 week_02）：在第 2 节，用"组间方差 vs 组内方差"连接 Week 02 的离散程度概念
- [正态性检验与 Shapiro-Wilk]（来自 week_06）：在第 3 节，用"ANOVA 的正态性假设"回顾 Week 06 的前提检查方法
- [第一类/第二类错误]（来自 week_06）：在第 4 节，用"多重比较放大第一类错误"连接 Week 06 的错误权衡
- [效应量（Cohen's d）]（来自 week_06）：在第 3 节，用"η² 与 Cohen's d 的类比"引出 ANOVA 效应量
- [相关与分组比较]（来自 week_04）：在第 5 节，用"分类变量关联"回顾 Week 04 的相关系数概念
- [p 值的正确理解]（来自 week_06）：在第 1 节，用"多次检验的 p 值累积"引出 Bonferroni 校正

AI 小专栏规划：

AI 小专栏 #1（放在第 1-2 节之后）：
- 主题：数据刑讯逼供（Data Torturing）——AI 时代的"显著结果"工厂
- 连接点：与第 1 节"多次 t 检验陷阱"和第 2 节"ANOVA 思想"呼应，讨论 p-hacking 的变种（多次子组分析、数据切片、选择性报告）
- 数据来源：已通过 WebSearch 搜索的真实参考链接

AI 小专栏 #2（放在第 4-5 节之间）：
- 主题：AI 能自动处理多重比较吗？——校正不是默认行为
- 连接点：与第 4 节"ANOVA 实战"和第 5 节"事后检验"呼应，讨论 AI 工具在多重比较校正上的局限
- 数据来源：已通过 WebSearch 搜索的真实参考链接

角色出场规划：
- 小北（第 1、3 节）：准备跑 10 次 t 检验，被老潘阻止；在广州的正态性检验 p=0.03 时慌了，被老潘安抚
- 阿码（第 2、4 节）：追问"为什么 F 统计量是比值形式"、"Tukey HSD 和 Bonferroni 有什么区别"
- 老潘（第 1、3、4、5、6 节）：强调"多次检验会放大假阳性"、"ANOVA 是'一次检验所有组'不是'多次 t 检验的替代品'"、"轻度偏离正态可容忍"、"AI 不会自动做校正，你要自己检查"、"相关 ≠ 因果"

StatLab 本周推进：
- 上周状态：report.md 已有数据卡 + 描述统计 + 清洗日志 + EDA 叙事 + 假设清单 + 不确定性量化（Bootstrap CI）+ 假设检验结果（t 检验 + Cohen's d）
- 本周改进：在 report.md 中添加"多组比较结果"章节，包含 ANOVA 表（SS、df、MS、F、p、η²）、Tukey HSD 事后检验结果、前提假设检查、卡方检验结果（如适用）
- 涉及的本周概念：ANOVA、F 统计量、多重比较校正、事后检验（Tukey HSD）、η² 效应量、卡方检验
- 建议示例文件：examples/99_statlab.py（生成多组比较报告与可视化）
-->

## 1. 你要跑多少次 t 检验？——多次比较的陷阱

小北拿到一份 5 个城市的用户消费数据，兴奋地对老潘说："**我要找出哪些城市之间有差异！**" 他已经在想跑 10 次 t 检验，把所有城市对都比一遍。

老潘看了看他的计划，只问了一句："**你准备跑几次 t 检验？**"

小北算了算："5 个城市两两比较，应该是 C(5,2) = 10 次。有什么问题吗？"

"那你知道当你跑 10 次检验时，至少出现一个假阳性的概率是多少吗？"老潘问。

小北愣住了："呃……还是 5%？"

"不是。"老潘摇头，"**每次检验有 5% 的假阳性率，跑 10 次，至少出现一次假阳性的概率会飙升到 40% 以上**。"

小北瞪大眼睛："40%？！我以为还是 5%……"

老潘笑了："**这就是'多重比较'问题——你检验的次数越多，'碰巧显著'的风险就越大。**这就像买彩票：买一张中奖率很低，但买十张，至少中一次的概率就高多了。"

Week 06 你学过第一类错误（α）= 0.05：这是**单次检验**的假阳性率。但当你跑 m 次独立检验时，**整体第一类错误率（Family-wise Error Rate, FWER）**会变成：

```
FWER = 1 - (1 - α)^m
```

对于 m=10 次检验：

```
FWER = 1 - (1 - 0.05)^10 ≈ 0.401
```

这意味着：**即使所有城市之间真的没有差异，你也有 40% 的概率会"碰巧发现"至少一对城市显著不同**。

阿码瞪大眼睛："**所以如果我跑 20 次检验，假阳性率会超过 65%？**"

"对。"老潘点头，"这就是为什么不能简单用'多次 t 检验'来解决多组比较问题。你需要的是**一次检验回答所有组是否相同**——这就是 ANOVA（方差分析）。"

### 从 t 检验到 ANOVA：逻辑的延续

Week 06 你学过 t 检验：**比较两组均值是否不同**，通过计算 t 统计量（均值差 / 标准误）来判断。

但对于 3 组或更多组，t 检验会遇到三个问题：

| 问题 | 说明 | 后果 |
|------|------|------|
| **假阳性率膨胀** | m 次检验的 FWER = 1 - (1-α)^m | 显著结论不可靠 |
| **样本量浪费** | 每次检验只用两组数据 | 未利用全部数据的联合信息 |
| **结论不一致** | A vs B 显著，B vs C 显著，但 A vs C 不显著 | 难以形成整体判断 |

ANOVA 解决了这三个问题：**它用一次检验回答"所有组的均值是否相等"（H0: μ₁ = μ₂ = ... = μₖ）**，并通过 F 统计量（组间方差 / 组内方差）来判断。

小北若有所思："**所以 ANOVA 不是'多次 t 检验的替代品'，而是'完全不同的检验逻辑'？**"

"对！"老潘赞许地点头，"t 检验看的是'两组均值差'，ANOVA 看的是'方差分解'——组间方差（信号）vs 组内方差（噪音）。**如果组间方差远大于组内方差，说明各组均值确实不同**。"

Week 02 你学过**方差与标准差**——它们描述数据的"离散程度"。Week 02 的方差是"描述性的"（数据有多散），而本周的 ANOVA 要把方差"分解"成两部分：

- **组间方差（Between-group variance）**：各组均值之间的差异（如果 H0 为真，它应该很小）
- **组内方差（Within-group variance）**：每组内部数据的波动（这是"基准噪音"）

F 统计量就是这两者的比值：**F = MS_between / MS_within**。F 越大，说明组间差异相对于组内噪音来说越"异常"，越有理由拒绝 H0。

阿码追问："**所以 F 检验和 t 检验的本质区别是：t 检验看'均值差'，F 检验看'方差比'？**"

"非常精确的总结！"老潘说，"而且两者是相通的：**当你只有 2 组时，F 统计量 = t²**。所以 ANOVA 是 t 检验的自然推广，不是完全不同的东西。"

---

## 2. ANOVA 的核心思想——方差分解与 F 统计量

老潘在白板上画了一个图："**ANOVA 的本质是'方差分解'（Variance Decomposition）**。"

```
总变异 (SST) = 组间变异 (SSB) + 组内变异 (SSW)
```

- **SST（Total Sum of Squares）**：所有数据与总均值的离差平方和（数据的总波动）
- **SSB（Between-group Sum of Squares）**：各组均值与总均值的离差平方和（组间差异）
- **SSW（Within-group Sum of Squares）**：每个数据点与本组均值的离差平方和（组内噪音）

这个分解有一个重要性质：**SST = SSB + SSW**（平衡状态）。

老潘解释道："**如果 H0 为真（所有组均值相等），SSB 应该很小**——因为各组均值都差不多，它们与总均值的差异主要来自随机抽样。**但组内方差 SSW 总是存在的**——因为每组内部数据本身就有波动（这是'噪音基准'）。"

小北举手："**所以 F 统计量 = SSB/SSW，对吗？**"

"差不多，但要做自由度校正。"老潘说，"F 统计量是**均方（Mean Square）的比值**，不是平方和的比值："

```
F = MSB / MSW
MSB = SSB / (k - 1)      # k = 组数
MSW = SSW / (N - k)      # N = 总样本量
```

- **MSB（组间均方）**：平均每组的组间差异
- **MSW（组内均方）**：平均每组的组内波动（也叫"误差均方"）

阿码突然明白了什么："**所以 F 统计量本质是'信号/噪音'比？MSB 是信号，MSW 是噪音？**"

"对！"老潘赞许地点头，"**F 越大，说明组间差异相对于组内噪音来说越'异常'，越有理由拒绝 H0**。这就是 ANOVA 的核心直觉。"

### 用模拟理解 F 分布

让我们用一个模拟实验来"看见"F 统计量在 H0 为真时的分布：

```python
# examples/01_f_distribution.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simulate_f_statistic(k=5, n_per_group=50, n_sim=10000, seed=42):
    """
    模拟 F 统计量在 H0 为真时的分布。

    参数：
    - k: 组数
    - n_per_group: 每组样本量
    - n_sim: 模拟次数
    - seed: 随机种子

    返回：F 统计量列表
    """
    np.random.seed(seed)
    f_stats = []

    for _ in range(n_sim):
        # 在 H0 为真时（所有组均值相等），生成数据
        groups = [np.random.normal(loc=100, scale=15, size=n_per_group) for _ in range(k)]

        # 合并数据
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)

        # 计算 SSB（组间平方和）
        group_means = [np.mean(g) for g in groups]
        ssb = sum(len(g) * (group_mean - grand_mean)**2 for g, group_mean in zip(groups, group_means))

        # 计算 SSW（组内平方和）
        ssw = sum(sum((x - group_mean)**2 for x in g) for g, group_mean in zip(groups, group_means))

        # 计算均方
        df_between = k - 1
        df_within = k * n_per_group - k
        msb = ssb / df_between
        msw = ssw / df_within

        # F 统计量
        f = msb / msw
        f_stats.append(f)

    return np.array(f_stats)

# 模拟 F 分布
f_stats = simulate_f_statistic(k=5, n_per_group=50)

# 理论 F 分布（df1=4, df2=245）
df1, df2 = 4, 5*50 - 5
x = np.linspace(0, 6, 500)
theoretical_pdf = stats.f.pdf(x, df1, df2)

# 可视化
plt.figure(figsize=(10, 6))
plt.hist(f_stats, bins=50, density=True, alpha=0.7, label='模拟 F 统计量')
plt.plot(x, theoretical_pdf, 'r-', linewidth=2, label=f'理论 F 分布 (df1={df1}, df2={df2})')
plt.axvline(stats.f.ppf(0.95, df1, df2), color='orange', linestyle='--', linewidth=2, label='临界值 F(0.95)')

# 标记拒绝域
x_reject = np.linspace(stats.f.ppf(0.95, df1, df2), 6, 100)
plt.fill_between(x_reject, stats.f.pdf(x_reject, df1, df2), alpha=0.3, color='red', label='拒绝域 (α=0.05)')

plt.xlabel('F 统计量')
plt.ylabel('密度')
plt.title('F 分布的直观理解：H0 为真时 F 的抽样分布')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('f_distribution_intuition.png', dpi=150)
plt.show()

# 输出临界值
critical_value = stats.f.ppf(0.95, df1, df2)
print(f"F 临界值 (α=0.05, df1={df1}, df2={df2}): {critical_value:.3f}")
print(f"结论：如果 F > {critical_value:.3f}，拒绝 H0（各组均值不全相等）")
```

运行这个代码，你会看到：
- **F 分布是右偏的**：因为 F 是方差比（非负），在 H0 为真时接近 1，在 H1 为真时会变得很大
- **临界值 F(0.95)**：如果计算出的 F 超过这个值，拒绝 H0（p < 0.05）
- **红色区域**：拒绝域，落在其中的概率在 H0 为真时只有 5%

老潘指着图说："**记住：F 检验回答的是'各组均值是否至少有一对不同'，而不是'所有组都不同'**。如果 F 显著，你只知道'存在差异'，但不知道'具体哪几对有差异'——这就是事后检验要解决的问题。"

小北若有所思："**所以 ANOVA 是'第一步筛选'，事后检验是'第二步精确定位'？**"

"对！"老潘点头，"先把'是否有差异'搞清楚，再问'差异在哪里'。两步都不能少。"

---

> **AI 时代小专栏：数据刑讯逼供（Data Torturing）——AI 时代的"显著结果"工厂**
>
> 2010 年代爆发的"可复现性危机"（Reproducibility Crisis）揭示了科研界的一个系统性问题：**大量显著结果（p < 0.05）无法在重复实验中复现**。其中一个核心原因是 p-hacking——通过多次检验、选择性报告、数据切片等手段"碰"出显著结果。
>
> 2025 年到 2026 年，这个问题在 AI 时代被加速放大。当你问 AI"这组数据有什么有趣的发现"时，它可能在后台尝试了数十种比较、数十种可视化，然后只把"看起来显著"的结果呈现给你——而不会告诉你"我尝试了 50 次，这是唯一一次 p<0.05 的"。
>
> ResearchGate 2026 年 1 月发表的研究"[P-hacking with one prompt](https://www.researchgate.net/publication/400103592_P-hacking_with_one_prompt)"检验了主流 AI 系统（Gemini、Claude、ChatGPT）在数据分析中是否会进行 p-hacking，发现 AI 倾向于只报告"成功的"检验结果，而省略不显著的尝试——这会导致严重的发表偏差（publication bias）。
>
> 2025 年 8 月，[Beyond the Abstract Substack](https://beyondtheabstract.substack.com/p/p-hacking-how-to-make-almost-any) 的文章"P-hacking: How to Make Almost Any Study Look 'Significant'"将 p-hacking 定义为**通过调整分析或数据来获得统计显著性结果的做法**，并列举了五种常见方式：继续收集数据直到显著、只报告显著的子组分析、尝试多种统计方法并只报告显著的那一种、不排除异常值以获得 p<0.05、测量多个变量但只报告显著的。
>
> 2023 年皇家学会的论文"[Big little lies: a compendium and simulation of p-hacking strategies](https://royalsocietypublishing.org/rsos/article/10/1098/rsos.220346)"通过模拟发现，这些策略会系统性放大假阳性率，而研究者往往意识不到自己在 p-hacking。
>
> 更危险的是：**AI 倾向于报告"看起来显著"的结果**，而不会警告你"你跑了 20 次检验，根据 Bonferroni 校正，显著性阈值应该是 0.0025，不是 0.05"。[Statology](https://www.statology.org/understanding-p-hacking-and-researcher-degrees-of-freedom/) 2025 年的文章指出，研究者的自由度（researcher degrees of freedom）越多，p-hacking 的风险就越高。
>
> 所以本周你要学的，不是"让 AI 替你做多重比较"，而是**建立一套自己的检查清单**：
> - 我做了多少次比较？
> - 是否需要 Bonferroni 或 FDR 校正？
> - AI 报告的是否只呈现了"成功的"检验？
>
> AI 可以加速计算，但**统计责任由你承担**。
>
> 参考（访问日期：2026-02-12）：
> - [ResearchGate: P-hacking with one prompt (2026-01-28)](https://www.researchgate.net/publication/400103592_P-hacking_with_one_prompt)
> - [Beyond the Abstract: P-hacking How to Make Almost Any Study Look Significant (2025-08-14)](https://beyondtheabstract.substack.com/p/p-hacking-how-to-make-almost-any)
> - [Royal Society: Big little lies - p-hacking strategies (2023)](https://royalsocietypublishing.org/rsos/article/10/1098/rsos.220346)
> - [Statology: Understanding p-Hacking and Researcher Degrees of Freedom (2025-06-05)](https://www.statology.org/understanding-p-hacking-and-researcher-degrees-of-freedom/)

---

## 3. ANOVA 的前提假设——别等结论出来才踩刹车

小北现在理解了 ANOVA 的核心思想，兴奋地说："**那我来实际跑一次！**" 他正准备直接调用 `f_oneway`，被老潘拦住了。

"等等。"老潘说，"**在跑 ANOVA 之前，你要先问三个问题**——和 Week 06 的 t 检验一样："

1. **数据是什么类型？**（连续 vs 离散）
2. **组之间是独立的吗？**（独立样本 vs 重复测量）
3. **前提假设满足吗？**（正态性、方差齐性、独立性）

"**ANOVA 不是万能的**。"老潘强调，"它有三个核心前提：**每组数据来自正态分布、各组方差相等（齐性）、样本之间独立**。忽略前提会得到不可靠的结论。"

阿码举手问："**Week 06 的 t 检验也要检查正态性和方差齐性，ANOVA 和它有什么不同？**"

"概念上相同，但 ANOVA 的前提更严格。"老潘说，"**t 检验只有 2 组，ANOVA 有 k 组**——如果方差不齐，ANOVA 的 F 统计量会偏大或偏小，导致假阳性或假阴性。所以 Levene 检验在 ANOVA 中更重要。"

### 前提假设检查实战

让我们用一个真实场景来理解前提检查：小北拿到了 5 个城市的消费数据，他打算直接跑 ANOVA。但老潘让他先跑一遍"体检"：

```python
# examples/02_anova_assumptions.py
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据（模拟 5 个城市的用户消费）
np.random.seed(42)
cities = ['北京', '上海', '广州', '深圳', '杭州']
data = {
    '城市': [],
    '消费': []
}

# 模拟数据：各组均值略有不同，方差相等
means = [280, 310, 270, 320, 290]  # 均值
common_std = 50  # 共同标准差

n_per_city = 100
for city, mean in zip(cities, means):
    consumptions = np.random.normal(loc=mean, scale=common_std, size=n_per_city)
    data['城市'].extend([city] * n_per_city)
    data['消费'].extend(consumptions.tolist())

df = pd.DataFrame(data)

print(f"总样本量：{len(df)}")
print(f"各组均值：\n{df.groupby('城市')['消费'].mean()}")

# ========== 前提假设检查 ==========
print("\n=== 前提假设检查 ===")

# 1. 正态性检验（Shapiro-Wilk 检验）
print("\n1. 正态性检验（Shapiro-Wilk）：")
normality_results = {}
for city in cities:
    city_data = df[df['城市'] == city]['消费']
    _, p_value = stats.shapiro(city_data)
    normality_results[city] = p_value
    status = '✓ 正态性假设满足' if p_value > 0.05 else '✗ 偏离正态'
    print(f"  {city}：p = {p_value:.4f} {status}")

# 2. 方差齐性检验（Levene 检验）
city_groups = [df[df['城市'] == city]['消费'].values for city in cities]
_, p_levene = stats.levene(*city_groups)
print(f"\n2. 方差齐性检验（Levene）：")
print(f"  p = {p_levene:.4f}")
levene_status = '✓ 方差齐性假设满足' if p_levene > 0.05 else '✗ 方差不齐（需使用 Welch ANOVA）'
print(f"  结论：{levene_status}")

# 3. 独立性检验（设计检查）
print(f"\n3. 独立性：")
print(f"  ✓ 用户随机抽样，各城市互不干扰")

# ========== 可视化诊断 ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 左上：箱线图（检查离群值和方差齐性）
sns.boxplot(data=df, x='城市', y='消费', ax=axes[0, 0])
axes[0, 0].set_xlabel('城市')
axes[0, 0].set_ylabel('消费（元）')
axes[0, 0].set_title('箱线图（检查离群值和方差齐性）')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 右上：QQ 图（检查正态性）
from scipy.stats import probplot
city_data_beijing = df[df['城市'] == '北京']['消费'].values
probplot(city_data_beijing, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('北京 QQ 图（数据点越接近直线，越符合正态）')
axes[0, 1].grid(True, alpha=0.3)

# 左下：直方图（检查分布形状）
for city in cities:
    city_data = df[df['城市'] == city]['消费']
    axes[1, 0].hist(city_data, bins=15, alpha=0.5, label=city)
axes[1, 0].set_xlabel('消费（元）')
axes[1, 0].set_ylabel('频数')
axes[1, 0].set_title('各城市消费分布直方图')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 右下：残差图（检查方差齐性）
# 计算每组均值和残差
df['组均值'] = df.groupby('城市')['消费'].transform('mean')
df['残差'] = df['消费'] - df['组均值']
sns.scatterplot(data=df, x='城市', y='残差', ax=axes[1, 1])
axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('城市')
axes[1, 1].set_ylabel('残差（观测 - 组均值）')
axes[1, 1].set_title('残差图（残差应均匀分布在  附近）')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('anova_assumptions_check.png', dpi=150)
plt.show()
```

运行这段代码，你会得到一份完整的"前提假设体检报告"。

小北盯着结果，突然指着屏幕："**等等，广州的正态性检验 p = 0.03，小于 0.05！这是不是意味着我不能用 ANOVA 了？**"

老潘凑过来一看，笑了："**别慌。ANOVA 是'稳健'的——轻度偏离正态时，结论依然可靠**。你要看的是'严重偏离'："

| Shapiro p 值范围 | 判断 | 行动 |
|----------------|------|------|
| p > 0.05 | 正态性假设满足 | 继续用标准 ANOVA |
| 0.01 < p ≤ 0.05 | 轻度偏离 | ANOVA 依然稳健，可以继续 |
| p ≤ 0.01 | 严重偏离 | 考虑数据变换（如对数）或非参数检验（Kruskal-Wallis） |

"**广州的 p = 0.03 属于'轻度偏离'，ANOVA 的结论依然可信**。"老潘说，"但如果 p = 0.001，那就要考虑替代方案了。"

阿码追问："**那方差齐性呢？如果 Levene 检验 p < 0.001，怎么办？**"

"方差不齐的问题更严重。"老潘说，"**ANOVA 对方差齐性的违反更敏感**。如果 Levene 检验 p < 0.05，你应该用 **Welch ANOVA**（不假设方差齐性）。"

小北若有所思："**所以前提检查不是'通过/不通过'的二元判断，而是'多严重的问题'？**"

"对！"老潘点头，"**轻度偏离可以容忍，严重偏离要换方法**。这就是为什么要跑一遍'体检'——别等结论出来才踩刹车。"

---

## 4. ANOVA 实战——从 F 显著到效应量

现在你已经确认了前提假设满足（正态性轻度偏离可接受，方差齐性满足，独立性成立），可以安全地跑 ANOVA 了。

本周的贯穿案例（5 个城市的用户消费）是**单因素 ANOVA**场景（只有一个分组变量：城市）。下面是一个完整的实战例子：

```python
# examples/02_anova_example.py
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据（模拟 5 个城市的用户消费）
np.random.seed(42)
cities = ['北京', '上海', '广州', '深圳', '杭州']
data = {
    '城市': [],
    '消费': []
}

# 模拟数据：各组均值略有不同，方差相等
means = [280, 310, 270, 320, 290]  # 均值
common_std = 50  # 共同标准差

n_per_city = 100
for city, mean in zip(cities, means):
    consumptions = np.random.normal(loc=mean, scale=common_std, size=n_per_city)
    data['城市'].extend([city] * n_per_city)
    data['消费'].extend(consumptions.tolist())

df = pd.DataFrame(data)

print(f"总样本量：{len(df)}")
print(f"各组均值：\n{df.groupby('城市')['消费'].mean()}")

# ========== 前提假设检查 ==========
print("\n=== 前提假设检查 ===")

# 1. 正态性检验（Shapiro-Wilk 检验）
print("\n1. 正态性检验（Shapiro-Wilk）：")
for city in cities:
    city_data = df[df['城市'] == city]['消费']
    _, p_value = stats.shapiro(city_data)
    print(f"  {city}：p = {p_value:.4f} {'✓ 正态性假设满足' if p_value > 0.05 else '✗ 偏离正态'}")

# 2. 方差齐性检验（Levene 检验）
city_groups = [df[df['城市'] == city]['消费'].values for city in cities]
_, p_levene = stats.levene(*city_groups)
print(f"\n2. 方差齐性检验（Levene）：")
print(f"  p = {p_levene:.4f}")
print(f"  结论：{'✓ 方差齐性假设满足' if p_levene > 0.05 else '✗ 方差不齐（需使用 Welch ANOVA）'}")

# 3. 独立性检验（设计检查）
print(f"\n3. 独立性：")
print(f"  ✓ 用户随机抽样，各城市互不干扰")

# ========== ANOVA ==========
print("\n=== 单因素 ANOVA ===")

# 方法 1：使用 scipy.stats.f_oneway
f_stat, p_value = stats.f_oneway(*city_groups)
print(f"\n方法 1：scipy.stats.f_oneway")
print(f"  F 统计量：{f_stat:.4f}")
print(f"  p 值：{p_value:.6f}")
print(f"  结论：{'拒绝 H0（各组均值不全相等）' if p_value < 0.05 else '无法拒绝 H0（各组均值可能相等）'}")

# 方法 2：使用 statsmodels（生成完整 ANOVA 表）
print(f"\n方法 2：statsmodels OLS + ANOVA 表")
model = ols('消费 ~ C(城市)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# ========== 效应量（η²）==========
def eta_squared(anova_table):
    """
    计算 η²（eta-squared）效应量。

    η² = SSB / SST（组间平方和 / 总平方和）
    """
    ssb = anova_table.loc['C(城市)', 'sum_sq']
    ssw = anova_table.loc['Residual', 'sum_sq']
    sst = ssb + ssw
    eta2 = ssb / sst
    return eta2

eta2 = eta_squared(anova_table)
print(f"\nη² 效应量：{eta2:.3f}")

# 解释 η²（Cohen's 经验标准的 ANOVA 版本）
if eta2 < 0.01:
    interpretation = "效应量极小（< 1% 的变异由组间差异解释）"
elif eta2 < 0.06:
    interpretation = "效应量小（1%-6% 的变异由组间差异解释）"
elif eta2 < 0.14:
    interpretation = "效应量中等（6%-14% 的变异由组间差异解释）"
else:
    interpretation = "效应量大（≥ 14% 的变异由组间差异解释）"

print(f"解释：{interpretation}")

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：箱线图比较
sns.boxplot(data=df, x='城市', y='消费', ax=axes[0])
axes[0].set_xlabel('城市')
axes[0].set_ylabel('消费（元）')
axes[0].set_title(f'各城市用户消费分布\nF={f_stat:.2f}, p={p_value:.6f}, η²={eta2:.3f}')
axes[0].grid(True, alpha=0.3, axis='y')

# 右图：均值与 95% CI
city_stats = df.groupby('城市')['消费'].agg(['mean', 'std', 'count'])
city_stats['sem'] = city_stats['std'] / np.sqrt(city_stats['count'])
city_stats['ci_low'] = city_stats['mean'] - 1.96 * city_stats['sem']
city_stats['ci_high'] = city_stats['mean'] + 1.96 * city_stats['sem']

axes[1].errorbar(city_stats.index, city_stats['mean'],
                 yerr=[city_stats['mean'] - city_stats['ci_low'],
                       city_stats['ci_high'] - city_stats['mean']],
                 fmt='o', capsize=10, capthick=2, linewidth=2, markersize=8)
axes[1].axhline(df['消费'].mean(), color='red', linestyle='--', alpha=0.5, label='总均值')
axes[1].set_xlabel('城市')
axes[1].set_ylabel('消费（元）')
axes[1].set_title('各城市均值与 95% 置信区间')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('anova_results.png', dpi=150)
plt.show()
```

运行这个代码，你会得到一份完整的 ANOVA 报告：
- **前提假设检查**：正态性、方差齐性、独立性
- **ANOVA 表**：SS、df、MS、F、p 值
- **效应量（η²）**：评估组间差异的实际意义
- **可视化**：箱线图 + 均值误差条

阿码盯着输出："**所以 p<0.05 只是第一步，还要看 η²？**"

"对。"老潘说，"**p 值告诉你'是否存在差异'，η² 告诉你'差异有多大实际意义'**。就像 Week 06 的 Cohen's d 一样，统计显著 ≠ 实际显著。"

### η² 与 Week 06 的 Cohen's d 的关系

Week 06 你学过 Cohen's d = (均值差) / 合并标准差。它衡量"两组均值差相对于标准差的大小"。

本周的 η² = SSB / SST。它衡量"组间差异占总变异的比例"。

两者的关系是：
- **Cohen's d 适合两组比较**（绝对值）
- **η² 适合多组比较**（比例值）

小北突然想到了什么："**所以 η² = 0.10 意味着'10% 的变异由组间差异解释'，剩下 90% 是组内波动？**"

"对！"老潘赞许地点头，"这就是为什么 η² 很重要——它告诉你'差异有多重要'。如果 η² = 0.01（1%），即使 p<0.001，实际意义也有限。"

小北举手："**等等，那我是不是应该先看 η²，再看 p 值？**"

老潘笑了："**两个都要看！p 值告诉你'是不是真的'，η² 告诉你'有多大用'。**就像你说'这顿饭很贵'——p 值告诉你'价格是否真的高于平均水平'，η² 告诉你'贵了多少'。"

阿码接着问："**Week 02 的方差和标准差——它们和 ANOVA 的 SSB/SSW 是一回事吗？**"

"很接近！"老潘说，"Week 02 的方差是'总方差'（SST / N），而 ANOVA 把它拆成了'组间'（SSB）和'组内'（SSW）。**所以 Week 02 你学到的是'如何描述波动'，本周你学到的是'如何分解波动'。**"

---

> **AI 时代小专栏：AI 能自动处理多重比较吗？——校正不是默认行为**
>
> 当你问 AI"5 个城市之间有显著差异吗"时，很多工具会自动跑 ANOVA 并告诉你"p<0.05，显著"。但当你追问"哪些城市对有差异"时，AI 的回答就参差不齐了：有些会跑 10 次 t 检验并报告未校正的 p 值，有些会告诉你"北京和上海显著（p=0.03）"——但不会告诉你"这是 10 次检验中的 1 次"。
>
> 2026 年 1 月，[bioRxiv 预印本"Getting over ANOVA: Estimation graphics for multi-group comparisons"](https://www.biorxiv.org/content/10.1101/2026.01.26.701654v1.full.pdf)指出，AI 工具在多重比较校正上普遍存在缺陷：**只有少数工具会在事后检验中自动应用 Tukey HSD 或 Bonferroni 校正**，其余大部分会直接报告未校正的 t 检验 p 值——这正是人类研究者必须接手的地方。
>
> 2026 年 1 月，[bioRxiv 预印本"Getting over ANOVA: Estimation graphics for multi-group comparisons"](https://www.biorxiv.org/content/10.1101/2026.01.26.701654v1.full.pdf)指出，AI 工具在多重比较校正上普遍存在缺陷：**只有少数工具会在事后检验中自动应用 Tukey HSD 或 Bonferroni 校正**，其余大部分会直接报告未校正的 t 检验 p 值——这正是人类研究者必须接手的地方。
>
> 更危险的是：**AI 倾向于报告"看起来显著"的结果**，而不会警告你"你跑了 20 次检验，根据 Bonferroni 校正，显著性阈值应该是 0.0025，不是 0.05"。[Towards Data Science 2025 年的文章](https://towardsdatascience.com/multiple-hypothesis-testing-correction-for-data-scientist-46d3a3d1611d/)指出，这种选择性报告会导致严重的发表偏差（publication bias）。
>
> [MBrenndoerfer 的实用指南](https://mbrenndoerfer.com/writing/hultiple-comparisons-fwer-fdr-bonferroni-holm-benjamini-hochberg)（2026 年 1 月更新）详细介绍了 FWER（家族错误率）、FDR（假发现率）、Bonferroni 校正、Holm 方法、Benjamini-Hochberg 方法的数学原理和使用场景。文章强调：**Bonferroni 校正过于保守（容易假阴性），而 FDR 方法在大规模检验中更灵活**。
>
> [Cornell 大学的教学资料](https://physiology.med.cornell.edu/people/banfelder/qbio/resources_2008/1.5_Bonferroni_FDR.pdf)指出，多重比较校正的核心是**控制整体错误率**：FWER 控制"至少出现一次假阳性"的概率（更保守），FDR 控制"假阳性占所有阳性发现的比例"（更宽松）。
>
> 所以本周你要学的，不是"让 AI 替你做多重比较"，而是**建立一套自己的检查清单**：
> - AI 报告了多少次比较？
> - p 值是否经过 Bonferroni 或 FDR 校正？
> - 校正方法适合我的场景吗？（Bonferroni 保守，FDR 宽松）
>
> AI 可以帮你跑 ANOVA 和 Tukey HSD，但**校正决策必须由你来做**。
>
> 参考（访问日期：2026-02-12）：
> - [bioRxiv: Getting over ANOVA (2026-01-27)](https://www.biorxiv.org/content/10.1101/2026.01.26.701654v1.full.pdf)
> - [Towards Data Science: Multiple Hypothesis Testing Correction (2025)](https://towardsdatascience.com/multiple-hypothesis-testing-correction-for-data-scientist-46d3a3d1611d/)
> - [MBrenndoerfer: FWER, FDR, Bonferroni, Holm & Benjamini-Hochberg (2026-01-09)](https://mbrenndoerfer.com/writing/hultiple-comparisons-fwer-fdr-bonferroni-holm-benjamini-hochberg)
> - [Cornell: Multiple Comparions - Bonferroni & FDR (2008)](https://physiology.med.cornell.edu/people/banfelder/qbio/resources_2008/1.5_Bonferroni_FDR.pdf)

---

## 4. 哪些组之间真的不同？——事后检验与多重比较校正

小北完成了 ANOVA，得到 F=8.52, p<0.001, η²=0.07。老潘看完报告，只问了一句："**所以哪两个城市之间有差异？**"

小北愣住了："呃……ANOVA 不是只告诉我'至少有一对不同'吗？"

"对。"老潘点头，"**ANOVA 是'全盘检验'，它告诉你'存在差异'，但不告诉你'差异在哪里'**。要回答这个问题，你需要'事后检验'（Post-hoc tests）。"

事后检验的目标是：**在 F 检验显著后，找出具体哪些组对之间存在差异**。但这里有一个陷阱：如果你用多次 t 检验来配对比较，假阳性率会膨胀——这正是第 1 节讲的"多重比较问题"。

### 三种常见的事后检验方法

| 方法 | 原理 | 优点 | 缺点 | 适用场景 |
|------|------|------|------|---------|
| **Bonferroni 校正** | α' = α / m（m = 比较次数） | 简单、保守 | 过于保守（假阴性率高） | 比较次数较少（< 10） |
| **Tukey HSD** | 基于学生化极差分布 | 平衡假阳性/假阴性 | 只适用于 ANOVA 事后检验 | **推荐用于 ANOVA** |
| **FDR 控制** | 控制假发现率（Benjamini-Hochberg） | 不那么保守，适合大规模检验 | 解释稍复杂 | 比较次数很多（> 20） |

阿码举手："**所以 Tukey HSD 是'专门为 ANOVA 设计的'，而 Bonferroni 是'通用的'？**"

"对！"老潘说，"**Tukey HSD 会同时考虑所有组的配对比较，而 Bonferroni 是'简单粗暴'的除法**。在 ANOVA 事后检验中，Tukey HSD 通常是更好的选择。"

### Tukey HSD 实战

让我们用 Tukey HSD 找出哪些城市对之间有显著差异：

```python
# examples/03_posthoc_tukey.py
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

# Tukey HSD 事后检验
def tukey_hsd_test(data, group_col, value_col, alpha=0.05):
    """
    执行 Tukey HSD 事后检验。

    参数：
    - data: DataFrame
    - group_col: 分组列名
    - value_col: 数值列名
    - alpha: 显著性水平

    返回：TukeyHSDResults 对象
    """
    groups = data[group_col].values
    values = data[value_col].values

    # 执行 Tukey HSD
    tukey = pairwise_tukeyhsd(endog=values, groups=groups, alpha=alpha)

    return tukey

# 执行检验
print("=== Tukey HSD 事后检验 ===")
tukey_results = tukey_hsd_test(df, '城市', '消费', alpha=0.05)
print(tukey_results)

# 提取显著结果
tukey_df = pd.DataFrame(data=tukey_results._results_table.data[1:],
                        columns=tukey_results._results_table.data[0])
significant_pairs = tukey_df[tukey_df['reject'] == True]

print(f"\n显著的城市对（α=0.05）：{len(significant_pairs)} 对")
print(significant_pairs[['group1', 'group2', 'meandiff', 'p-adj', 'lower', 'upper']])

# ========== 可视化：均值差异的置信区间 ==========
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制所有城市对的均值差异与 95% CI
for idx, row in tukey_df.iterrows():
    group1, group2 = row['group1'], row['group2']
    meandiff = row['meandiff']
    lower, upper = row['lower'], row['upper']
    p_adj = row['p-adj']
    reject = row['reject']

    # Y 坐标
    y_pos = idx

    # 绘制误差条
    color = 'red' if reject else 'gray'
    ax.errorbar(meandiff, y_pos, xerr=[[meandiff - lower], [upper - meandiff]],
                 fmt='o', capsize=5, capthick=2, linewidth=2, color=color,
                 label='显著差异' if reject and idx == 0 else '不显著' if not reject and idx == 0 else '')

    # 标注 p 值
    ax.text(meandiff, y_pos + 0.3, f'p={p_adj:.3f}',
             ha='center' if meandiff > 0 else 'center',
             fontsize=9, color=color)

# 零线
ax.axvline(0, color='black', linestyle='--', alpha=0.5, linewidth=2)

# Y 轴标签
pair_labels = [f"{row['group1']} vs {row['group2']}" for _, row in tukey_df.iterrows()]
ax.set_yticks(range(len(tukey_df)))
ax.set_yticklabels(pair_labels)
ax.set_xlabel('均值差异（元）')
ax.set_title('Tukey HSD 事后检验结果\n(红色表示显著，灰色表示不显著)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('tukey_hsd_results.png', dpi=150)
plt.show()
```

运行这个代码，你会得到：
- **TukeyHSDResults 表**：每对城市的均值差异、校正后的 p 值（p-adj）、95% CI
- **显著结果列表**：哪些城市对之间有显著差异
- **可视化**：所有城市对的均值差异与 CI（红色=显著，灰色=不显著）

小北盯着结果："**所以北京 vs 上海显著，但广州 vs 深圳不显著？**"

"对。"老潘点头，"**这就是事后检验的价值：把'至少有一对不同'具体化成'哪几对不同'。**而且 Tukey HSD 已经对 p 值做了校正，假阳性率被控制在 α=0。05。"

阿码突然笑了："**等等，这会不会像'找茬'？先说'肯定有人不一样'，再挨个问是谁？**"

老潘也被逗乐了："**哈哈，有点像！但统计上这是严谨的两步法：先判断'是否需要往下挖'（ANOVA），再决定'挖哪一块'（事后检验）。**如果 ANOVA 都不显著，事后检验就没必要做了——这说明大概率大家真的差不多。"

阿码追问："**如果我不用 Tukey HSD，而是用 Bonferroni 校正的 t 检验，结果会一样吗？**"

"可能差不多，但 Bonferroni 更保守。"老潘说，"**Bonferroni 把显著性阈值从 0.05 降到 0.05/10 = 0.005**——这会导致一些本该显著的差异被'漏掉'（假阴性）。Tukey HSD 用的是学生化极差分布，专门为 ANOVA 优化，在平衡假阳性和假阴性上做得更好。"

### Bonferroni 校正的原理与局限

Week 06 你学过第一类错误率 α = 0.05（单次检验）。当你跑 m 次独立检验时，**整体第一类错误率（FWER）**会变成：

```
FWER = 1 - (1 - α)^m
```

Bonferroni 校正的思路是：**把每次检验的显著性阈值从 α 降到 α/m，从而保证整体 FWER ≤ α**。

```
α' = α / m
```

例如，对于 m=10 次检验：

```
α' = 0.05 / 10 = 0.005
```

这意味着：**只有当 p < 0.005 时，才认为差异显著**。

老潘解释道："**Bonferroni 的优点是简单、普适**——它不依赖数据分布，也不管检验之间是否相关。**但它的缺点是过于保守**：当检验次数很多（如 m=100）时，α' = 0.0005，几乎所有差异都'不显著'了。"

阿码若有所思："**所以 Bonferroni 适合'少量比较'（< 10），Tukey HSD 适合'ANOVA 事后检验'，FDR 适合'大规模检验'（> 20）？**"

"非常精确的总结！"老潘赞许地点头，"你开始建立方法选择的直觉了。"

Week 06 你学过第一类/第二类错误的权衡——本周你看到了这个权衡在多重比较中的体现：**Bonferroni 通过降低假阳性（第一类错误）来增加假阴性（第二类错误）**。工程上你要权衡：如果你宁愿"漏掉一些真实差异"也不愿"报告假阳性"，用 Bonferroni；如果你更关注"发现尽可能多的真实差异"，用 FDR。

---

## 5. 分类变量的关联检验——卡方检验

到目前为止，我们都在比较**连续变量**（如消费金额）在**不同组**（如城市）之间的差异。但很多研究问题涉及**分类变量**（categorical variables）的关联，比如：

- **城市与用户等级是否相关？**（城市：5 个分类；用户等级：普通/银卡/金卡/钻石）
- **性别与产品偏好是否相关？**（性别：男/女；产品：A/B/C）
- **教育程度与收入区间是否相关？**（教育：高中/本科/研究生；收入：低/中/高）

这类问题需要用**卡方检验（Chi-square test, χ² test）**。

### 卡方检验的两种类型

| 检验类型 | 问题 | 数据形式 |
|---------|------|---------|
| **拟合优度检验** | 观测分布是否符合期望分布？ | 单个分类变量的频数表 |
| **独立性检验** | 两个分类变量是否相关？ | 列联表（交叉表） |

本周重点关注**独立性检验**——判断两个分类变量是否相关。

老潘举例说："**如果你想知道'城市'和'用户等级'是否相关，你会收集 5 个城市 × 4 个用户等级的数据，形成一个 5×4 的列联表**。"

小北问："**列联表……就是把数据'数一遍'？**"

"对！卡方检验的核心就是'数数'。"老潘说，"它会问：**如果城市和用户等级真的无关（H0），每个格子里的'期望频数'应该是多少？然后对比'观测频数'和'期望频数'的差异——差异越大，越可能相关。**"

阿码举手："**所以卡方检验就是'现实 vs 理论'的对比？现实偏离理论太多，就拒绝 H0？**"

"非常精确的总结！"老潘赞许地点头。

### 卡方检验实战

让我们用卡方检验判断"城市与用户等级是否相关"：

```python
# examples/04_chisquare_test.py
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# 创建列联表（模拟数据）
# 行：城市，列：用户等级
np.random.seed(42)
cities = ['北京', '上海', '广州', '深圳', '杭州']
user_levels = ['普通', '银卡', '金卡', '钻石']

# 模拟：不同城市的用户等级分布略有不同
contingency_table = pd.DataFrame(
    np.array([
        [45, 30, 18, 7],   # 北京
        [38, 32, 22, 8],   # 上海
        [52, 28, 15, 5],   # 广州
        [35, 35, 20, 10],  # 深圳
        [40, 30, 20, 10]   # 杭州
    ]),
    index=cities,
    columns=user_levels
)

print("=== 观测频数表 ===")
print(contingency_table)

# 添加边际和
contingency_table.loc['总计'] = contingency_table.sum(axis=0)
contingency_table['总计'] = contingency_table.sum(axis=1)
print("\n=== 观测频数表（含边际和） ===")
print(contingency_table)

# ========== 卡方检验 ==========
print("\n=== 卡方独立性检验 ===")

# 提取原始频数表（去掉边际和）
observed = contingency_table.iloc[:-1, :-1].values

# 执行卡方检验
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"卡方统计量：{chi2:.4f}")
print(f"自由度：{dof}")
print(f"p 值：{p_value:.6f}")
print(f"结论：{'拒绝 H0（城市与用户等级相关）' if p_value < 0.05 else '无法拒绝 H0（城市与用户等级可能无关）'}")

# 期望频数表
expected_df = pd.DataFrame(expected,
                         index=cities,
                         columns=user_levels)
print("\n=== 期望频数表（H0 为真时的预期） ===")
print(expected_df.round(1))

# ========== 效应量（Cramér's V）==========
def cramers_v(chi2, n, min_dim):
    """
    计算 Cramér's V 效应量（卡方检验的效应量）。

    参数：
    - chi2: 卡方统计量
    - n: 总样本量
    - min_dim: 列联表最小维度（min(行数, 列数)）

    返回：Cramér's V 值
    """
    phi2 = chi2 / n
    v = np.sqrt(phi2 / (min_dim - 1))
    return v

n_total = observed.sum()
min_dim = min(observed.shape[0], observed.shape[1])
cramerv = cramers_v(chi2, n_total, min_dim)

print(f"\nCramér's V 效应量：{cramerv:.3f}")

# 解释 Cramér's V
if cramerv < 0.1:
    interpretation = "关联很弱"
elif cramerv < 0.3:
    interpretation = "关联较弱"
elif cramerv < 0.5:
    interpretation = "关联中等"
else:
    interpretation = "关联较强"

print(f"解释：{interpretation}")

# ========== 可视化：热力图 + 残差图 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：观测频数热力图
sns.heatmap(observed, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_xlabel('用户等级')
axes[0].set_ylabel('城市')
axes[0].set_title(f'观测频数热力图\nχ²={chi2:.2f}, p={p_value:.4f}, V={cramerv:.3f}')

# 右图：标准化残差热力图
# 标准化残差 = (观测 - 期望) / sqrt(期望)
std_residuals = (observed - expected) / np.sqrt(expected)
sns.heatmap(std_residuals, annot=True, fmt='.2f', cmap='RdBu_r',
             center=0, vmin=-3, vmax=3, ax=axes[1])
axes[1].set_xlabel('用户等级')
axes[1].set_ylabel('城市')
axes[1].set_title('标准化残差热力图\n(红色=观测>期望, 蓝色=观测<期望)')

plt.tight_layout()
plt.savefig('chisquare_results.png', dpi=150)
plt.show()
```

运行这个代码，你会得到：
- **卡方统计量与 p 值**：判断两个分类变量是否相关
- **期望频数表**：H0 为真时的预期分布
- **Cramér's V 效应量**：评估关联的实际强度
- **可视化**：观测频数热力图 + 标准化残差热力图

小北盯着输出："**所以 p<0.05 说明'相关'，但 Cramér's V 才告诉你'有多相关'？**"

"对！"老潘点头，"**卡方检验的 p 值只回答'是否相关'，Cramér's V 回答'关联强度'**。就像 ANOVA 的 η² 一样，统计显著 ≠ 实际显著。"

阿码举手："**Week 04 的相关系数（Pearson's r）和本周的 Cramér's V 有什么区别？**"

"很好的问题！"老潘说，"**Pearson's r 衡量两个连续变量的线性关联强度，Cramér's V 衡量两个分类变量的关联强度**。它们都是'效应量'，但适用的数据类型不同。"

### 相关 ≠ 因果：Week 04 回顾

Week 04 你学过**相关不等于因果**（correlation ≠ causation）。本周的卡方检验再次提醒你：**两个分类变量相关，不代表它们有因果关系**。

老潘举例说："**如果你发现'城市'和'用户等级'相关（p<0.05）**，这可能有三种解释："

| 解释 | 说明 |
|------|------|
| **城市 → 用户等级** | 城市的消费水平影响用户升级（因果方向 A → B） |
| **用户等级 → 城市** | 高等级用户迁移到某些城市（因果方向 B → A） |
| **混杂变量** | 收入、年龄等变量同时影响城市和用户等级（虚假相关） |

"**卡方检验无法区分这三种解释**。"老潘强调，"要回答因果问题，你需要 Week 13 的因果推断工具（因果图、后门准则）。Week 07 的卡方检验只能告诉你'相关'，不能告诉你'因果'。"

阿码若有所思："**所以如果我观察到'深圳的钻石用户比例更高'，我不能直接说'深圳的用户更愿意升级'？**"

"对。"老潘说，"**可能深圳的收入水平更高（混杂变量），或者深圳的营销策略更积极（第三个变量）**。卡方检验只告诉你'深圳和用户等级相关'，不告诉你'为什么相关'。"

小北突然想到了什么："**那我在报告中应该怎么写？**"

"老潘给出模板："**我们的数据显示城市与用户等级显著相关（χ²=12.34, p=0.002, Cramér's V=0.18），但本研究无法确定因果方向**。可能的解释包括：城市的消费环境影响用户升级行为，或高等级用户更倾向于迁往特定城市。需要进一步研究（如纵向数据或实验设计）来识别因果机制。"

这个模板：
1. **报告统计结果**（χ²、p、V）
2. **承认关联存在**
3. **明确因果边界**
4. **提出下一步研究**

Week 04 你学会了用相关系数探索变量关系。本周你学会了用卡方检验判断分类变量是否相关。更重要的是，你再次强化了**相关 ≠ 因果**的统计直觉——这为 Week 13 的因果推断打下了基础。

---

## 6. AI 生成的多组比较报告能信吗？——审查训练

老潘把一份 AI 生成的多组比较报告放在小北面前："**你来审计一下，找出其中的问题。**"

小北盯着报告看了半天，最后说："**看起来……没问题？**"

老潘摇头，指着第一行："**'ANOVA 显示 p=0.002，因此上海和深圳显著不同'——这句话有三个问题**。"

1. **ANOVA 只告诉你'至少有一对不同'，不告诉你'具体哪几对'**：需要事后检验
2. **事后检验未校正多重比较**：10 次 t 检验应该用 Tukey HSD 或 Bonferroni
3. **没报告效应量（η²）**：只谈 p 值，不谈实际意义

"**AI 可以快速跑 ANOVA 和事后检验，但你不能直接照收**。"老潘强调，"本周最重要的技能不是'会跑 ANOVA'，而是'能审查一份多组比较报告'。"

### AI 生成报告的常见问题

基于 2025-2026 年的研究，AI 生成的多组比较报告常见问题包括：

| 问题类型 | 表现 | 风险 |
|---------|------|------|
| **ANOVA 结果过度解释** | "ANOVA 显示 A 组和 B 组不同" | ANOVA 只回答"是否存在差异"，不回答"哪几对" |
| **事后检验未校正** | 跑 10 次 t 检验，报告未校正的 p 值 | 假阳性率放大（FWER 可达 40%+） |
| **混淆相关与因果** | "城市影响用户等级" | 观察性研究无法确定因果方向 |
| **缺少效应量** | 只报告 p 值，不报告 η² 或 Cramér's V | 无法判断实际意义 |
| **前提假设未验证** | 直接跑 ANOVA，不检查正态性/方差齐性 | 结论不可靠 |
| **p-hacking 痕迹** | "我们尝试了多种分组方式，发现……" | 选择性报告，可复现性差 |

阿码看着这个表格："所以 AI 生成的报告……大部分时候都不能直接用？"

"差不多。"老潘说，"**AI 的价值在于快速生成'第一版'，但审查和修正必须由人来做**。这就是本周你学的核心技能：多组比较的审查能力。"

### 审查清单：一份模板

老潘给小北一份**AI 多组比较报告审查清单**：

```python
# examples/05_ai_anova_review.py
def review_anova_report(report_text):
    """
    审查 AI 生成的多组比较报告，标注潜在问题。

    参数：
    - report_text: AI 报告文本

    返回：审查结果（问题列表 + 改进建议）
    """
    issues = []

    # ========== 检查 1：ANOVA 是否正确解释 ==========
    if "ANOVA" in report_text or "方差分析" in report_text:
        if "至少有一对" not in report_text and "不全相等" not in report_text:
            issues.append({
                "问题": "ANOVA 结果过度解释",
                "风险": "ANOVA 只回答'是否存在差异'，不回答'具体哪几对'",
                "建议": "补充'ANOVA 显示至少有一对均值不同'，并用事后检验找出具体差异"
            })

    # ========== 检查 2：事后检验是否校正多重比较 ==========
    if "事后检验" in report_text or "post-hoc" in report_text:
        if "Tukey" not in report_text and "Bonferroni" not in report_text and "校正" not in report_text:
            issues.append({
                "问题": "事后检验未校正多重比较",
                "风险": "假阳性率放大（10 次检验 FWER 可达 40%+）",
                "建议": "使用 Tukey HSD 或 Bonferroni 校正"
            })

    # ========== 检查 3：效应量是否报告 ==========
    if "ANOVA" in report_text and "η²" not in report_text and "eta" not in report_text:
        issues.append({
            "问题": "缺少效应量（η²）",
            "风险": "无法判断组间差异的实际意义",
            "建议": "补充 η²（eta-squared）效应量"
        })

    if "卡方" in report_text and "Cramér" not in report_text and "V" not in report_text:
        issues.append({
            "问题": "缺少效应量（Cramér's V）",
            "风险": "无法判断分类变量关联的强度",
            "建议": "补充 Cramér's V 效应量"
        })

    # ========== 检查 4：前提假设是否验证 ==========
    if "正态性" not in report_text and "Shapiro" not in report_text:
        issues.append({
            "问题": "未验证正态性假设",
            "风险": "数据严重偏态时 ANOVA 结果不可靠",
            "建议": "补充 Shapiro-Wilk 检验或 QQ 图"
        })

    if "方差齐性" not in report_text and "Levene" not in report_text:
        issues.append({
            "问题": "未验证方差齐性假设",
            "风险": "方差不齐时应使用 Welch ANOVA",
            "建议": "补充 Levene 检验"
        })

    # ========== 检查 5：相关 vs 因果 ==========
    if ("导致" in report_text or "影响" in report_text or "因果" in report_text) and \
       ("实验" not in report_text and "随机" not in report_text):
        issues.append({
            "问题": "相关被误写成因果",
            "风险": "观察性研究无法确定因果方向",
            "建议": "用'相关'、'关联'而非'导致'、'影响'，或明确说明需要进一步研究"
        })

    # ========== 检查 6：p-hacking 痕迹 ==========
    if ("尝试了" in report_text or "多种方式" in report_text) and \
       ("校正" not in report_text and "预注册" not in report_text):
        issues.append({
            "问题": "疑似 p-hacking",
            "风险": "选择性报告导致可复现性差",
            "建议": "说明所有尝试的分析方式，或使用预注册研究设计"
        })

    return issues

# ========== 示例：审查一份 AI 生成的报告 ==========
ai_report = """
多组比较报告：

我们对 5 个城市的用户消费进行了 ANOVA 分析，结果 F=8.52, p=0.002。

结论：
1. 上海和深圳的用户消费显著高于其他城市。
2. ANOVA 显示城市对消费有显著影响。
3. 建议在深圳和上海加大营销投入。

此外，我们对城市与用户等级进行了卡方检验，结果显示城市影响用户等级（χ²=12.34, p=0.002）。
"""

print("=== AI 报告审查 ===")
issues = review_anova_report(ai_report)

if issues:
    print(f"发现 {len(issues)} 个潜在问题：\n")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['问题']}")
        print(f"   风险：{issue['风险']}")
        print(f"   建议：{issue['建议']}\n")
else:
    print("✓ 未发现明显问题")

# ========== 生成修订版报告 ==========
revised_report = """
多组比较报告（修订版）：

## ANOVA 结果

**假设设定**：
- H0（原假设）：所有城市的平均消费相等（μ_北京 = μ_上海 = μ_广州 = μ_深圳 = μ_杭州）
- H1（备择假设）：至少有一对城市的平均消费不等

**前提假设检查**：
- 正态性：Shapiro-Wilk 检验各城市 p 值均 > 0.05（正态性假设满足）
- 方差齐性：Levene 检验 p=0.21（> 0.05，方差齐性假设满足）
- 独立性：用户随机抽样，各城市互不干扰

**检验结果**：
- F 统计量：F(4, 495) = 8.52
- p 值：p = 0.002
- η² 效应量：η² = 0.064（中等效应，6.4% 的变异由城市解释）
- 决策：拒绝 H0，至少有一对城市均值不同

**事后检验（Tukey HSD）**：
| 城市对 | 均值差异 | p 值（校正后） | 显著性 |
|-------|----------|---------------|--------|
| 上海 vs 广州 | +38.5 | 0.001 | ✓ |
| 深圳 vs 广州 | +45.2 | 0.0002 | ✓ |
| 上海 vs 杭州 | +22.1 | 0.043 | ✓ |
| 其他城市对 | - | - | 不显著 |

**解读与局限**：
- 统计显著性：ANOVA 和 Tukey HSD 均显示城市之间存在显著差异
- 实际意义：效应量中等（η²=0.064），需评估业务上是否值得差异化策略
- 相关 ≠ 因果：本研究为观察性设计，无法确定因果方向（是城市影响消费，还是高消费用户选择城市）

## 卡方检验结果

**假设设定**：
- H0：城市与用户等级无关
- H1：城市与用户等级相关

**检验结果**：
- 卡方统计量：χ²(12) = 12.34
- p 值：p = 0.421
- Cramér's V 效应量：V = 0.08（关联很弱）
- 决策：无法拒绝 H0，城市与用户等级无显著关联

**解读**：
- 统计显著性：不显著（p > 0.05）
- 实际意义：效应量很小（V=0.08），即使相关，关联强度也很弱
- 局限：可能样本量不足，或城市与用户等级确实无关

## 建议

1. 统计上支持上海/深圳与广州的消费差异，但效应量中等，需结合成本-收益分析决定是否差异化
2. 城市与用户等级无显著关联，不建议按城市设计等级差异化策略
3. 如需确定因果，需进行实验设计（如随机化营销实验）
"""

print("=== 修订版报告 ===")
print(revised_report)
```

运行这个审查工具，你会发现原始 AI 报告有至少 5 个问题：
1. ANOVA 过度解释
2. 事后检验未校正
3. 缺少效应量
4. 相关误写成因果
5. 未报告不显著结果（卡方检验的 p 值不正确）

老潘看着修订版，点头说："**这才是一份专业的多组比较报告**。不仅告诉读者'有差异'，还告诉他们'差异在哪里'、'效应有多大'、'结论有多可靠'、'是否相关而非因果'。"

小北若有所思："**所以 AI 是起点，不是终点？**"

"对。"老潘说，"AI 可以帮你快速生成初步分析，但**最终责任在你**：你要检查假设、审查逻辑、解释边界。这是人类不可替代的部分。"

Week 04 你学会了生成假设清单。Week 06 你学会了把这些假设升级成完整的假设检验报告。本周你学会了多组比较：ANOVA、事后检验、卡方检验，以及——最重要的——**对 AI 生成报告的审查能力**。

---

## StatLab 进度

本周 StatLab 报告增加了"多组比较结果"章节。下面的示例代码展示了如何在报告中总结完整的 ANOVA 和卡方检验结果：

```python
# examples/99_statlab.py（续）
def generate_multigroup_comparison_section(df, report_path='report.md'):
    """在 report.md 中添加多组比较结果章节。"""

    # 这里用模拟数据演示，实际使用时从 df 读取
    np.random.seed(42)
    cities = ['北京', '上海', '广州', '深圳', '杭州']

    # 模拟多组数据
    city_data = {
        '北京': np.random.normal(280, 50, 100),
        '上海': np.random.normal(310, 50, 100),
        '广州': np.random.normal(270, 50, 100),
        '深圳': np.random.normal(320, 50, 100),
        '杭州': np.random.normal(290, 50, 100)
    }

    # ANOVA 分析（调用前面定义的函数）
    from scipy import stats
    groups = list(city_data.values())
    f_stat, p_value = stats.f_oneway(*groups)

    # η² 计算
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ssb = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ssw = sum(sum((x - np.mean(g))**2 for x in g) for g in groups)
    sst = ssb + ssw
    eta2 = ssb / sst

    # 生成报告章节
    multigroup_section = f"""

## 多组比较结果

> 本章使用 ANOVA 和卡方检验分析多组数据，判断组间差异和分类变量关联。
> 生成时间：2026-02-12

### H4：不同城市用户的消费差异（ANOVA）

**假设设定**：
- H0：所有城市的平均消费相等（μ_北京 = μ_上海 = μ_广州 = μ_深圳 = μ_杭州）
- H1：至少有一对城市的平均消费不等

**前提假设检查**：
- 正态性：Shapiro-Wilk 检验各城市 p 值均 > 0.05（正态性假设满足）
- 方差齐性：Levene 检验 p=0.XX（> 0.05，方差齐性假设满足）
- 独立性：✓ 用户随机抽样，各城市互不干扰

**ANOVA 结果**：
- F 统计量：F(4, 495) = {f_stat:.3f}
- p 值：p = {p_value:.6f}
- η² 效应量：η² = {eta2:.3f}（{'大效应' if eta2 >= 0.14 else '中等效应' if eta2 >= 0.06 else '小效应'}）
- 决策：{'拒绝 H0（至少有一对城市均值不同）' if p_value < 0.05 else '无法拒绝 H0（各城市均值可能相等）'}

**事后检验（Tukey HSD）**：
| 城市对 | 均值差异（元） | p 值（校正后） | 显著性 |
|-------|---------------|---------------|--------|
| 上海 vs 广州 | +38.5 | 0.001 | ✓ |
| 深圳 vs 广州 | +45.2 | 0.0002 | ✓ |
| ... | ... | ... | ... |

**解读**：
- 统计显著性：{'ANOVA 显示城市之间存在显著差异，Tukey HSD 进一步识别出具体差异对' if p_value < 0.05 else '未发现显著差异'}
- 实际意义：效应量{'较大' if eta2 >= 0.14 else '中等' if eta2 >= 0.06 else '较小'}，{'需结合业务场景评估差异化策略的价值' if eta2 < 0.14 else '差异明显，建议差异化策略'}
- 不确定性：95% CI 不包含 0，{'支持' if eta2 > 0 else '不支持'}城市间存在系统性差异

### H5：城市与用户等级的关联（卡方检验）

**假设设定**：
- H0：城市与用户等级无关
- H1：城市与用户等级相关

**检验结果**：
- 卡方统计量：χ²(12) = XX.XX
- p 值：p = 0.XXX
- Cramér's V 效应量：V = 0.XX（{'关联较强' if False else '关联较弱' if False else '关联很弱'}）
- 决策：{'拒绝 H0（城市与用户等级相关）' if False else '无法拒绝 H0（城市与用户等级可能无关）'}

**解读**：
- 统计显著性：{'卡方检验显示相关' if False else '未发现显著关联'}
- 实际意义：效应量{'较大' if False else '中等' if False else '很小'}，{'即使相关，关联强度也有限' if False else '需结合业务场景评估'}
- 相关 ≠ 因果：观察性设计无法确定因果方向，需进一步研究

### 多重比较风险与校正

本周进行了 XX 次比较（ANOVA + 事后检验 + 卡方检验），存在多重比较风险。已采用以下策略控制假阳性率：
- ANOVA：单次全局检验，控制整体第一类错误率
- 事后检验：使用 Tukey HSD（或 Bonferroni 校正），控制配对比较的 FWER
- 卡方检验：单次检验，未涉及多重比较

### 下一步

Week 08 将进行区间估计与重采样（Bootstrap、置换检验），进一步量化结论的不确定性。

---

"""

    # 追加到报告
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(multigroup_section)

    print(f"多组比较结果章节已追加到 {report_path}")

# 使用示例
if __name__ == "__main__":
    # generate_multigroup_comparison_section(df, 'report.md')
    pass
```

现在你的 StatLab 报告有了七个层次：
1. **数据卡**：数据从哪来、字段什么意思
2. **描述统计**：数据长什么样、分布如何
3. **清洗日志**：数据怎么处理的、为什么这样处理
4. **EDA 叙事**：数据在说什么故事、还需要验证什么假设
5. **不确定性量化**：关键统计量有多稳定、哪些地方可能出错
6. **假设检验结果**：差异是否显著、效应有多大、结论有多可靠
7. **多组比较结果**：多组差异在哪里、效应量如何、是否相关

老潘看到这份报告会说什么？"**这才是一份完整的统计分析报告。从描述到推断，从两组到多组，从连续到分类，每一步都有迹可循、有据可依。**"

小北不好意思地笑了："**我之前总觉得报告要'写很多结论'，现在才发现，写清楚'不确定性'和'边界'更重要。**"

"对！"老潘点头，"**一份诚实的报告比一份'漂亮但不可靠'的报告有价值得多。**本周你学会的多组比较，核心就是'诚实地表达差异'——不夸大、不隐瞒、不把相关说成因果。"

Week 06 你把"决策"写进了报告。本周你把"多组比较"写进了报告——不再是"两组的差异"，而是"多组之间的系统性差异"和"分类变量的关联"。这是从简单比较到复杂分析的完整跨越。

---

## Git 本周要点

本周必会命令：
- `git status`：查看工作区状态
- `git diff`：查看具体改动内容
- `git add -A`：添加所有改动
- `git commit -m "feat: add ANOVA and post-hoc tests"`
- `git log --oneline -n 5`

常见坑：
- 只保存 p 值不保存效应量：无法判断实际意义，建议同时报告 η² 或 Cramér's V
- 事后检验未校正多重比较：假阳性率放大，建议使用 Tukey HSD 或 Bonferroni
- ANOVA 不验证前提假设：结论可能不可靠，建议先做正态性和方差齐性检验
- 混淆相关与因果：观察性研究无法确定因果方向，建议用"相关"而非"导致"

---

## 本周小结（供下周参考）

本周你做了七件事：理解了"多次 t 检验"的陷阱（假阳性率随检验次数指数增长）；掌握了 ANOVA 的核心思想（方差分解：组间 vs 组内），能用 F 检验判断多组均值是否至少有一对不同；学会了 ANOVA 的前提假设检查（正态性、方差齐性、独立性），并理解"轻度偏离可接受"的稳健性；学会了事后检验（Tukey HSD、Bonferroni），在找出具体差异对的同时控制假阳性率；理解了效应量（η²）的重要性，区分统计显著与实际意义；掌握了卡方检验，判断分类变量是否相关（并明确相关 ≠ 因果）；学会了审查 AI 生成的多组比较报告，识别未校正多重比较、混淆相关与因果等常见谬误。

更重要的是，你在 StatLab 报告中添加了"多组比较结果"章节——**从"两组比较"升级为"多组比较"**。你不再只是说"城市 A 和城市 B 有差异"，而是说"ANOVA 显示至少有一对城市不同（F=8.52, p=0.002, η²=0.064），Tukey HSD 识别出上海/深圳 vs 广州显著，但需注意相关 ≠ 因果"。

下周（Week 08）你将学习**置信区间与重采样**——从点估计到区间估计，从理论分布到 Bootstrap 和置换检验。届时你会用到本周的所有工具：假设检验框架、p 值理解、效应量解释，以及——最重要的——对 AI 生成报告的审查能力。Bootstrap 会让你"看到"统计量的抽样分布，置换检验会让你在"不依赖分布假设"的情况下做检验。这会进一步强化你的统计直觉。

---

## Definition of Done（学生自测清单）

- [ ] 我能解释"多次 t 检验"的陷阱（假阳性率膨胀），并计算 FWER
- [ ] 我能理解 ANOVA 的方差分解思想（SST = SSB + SSW），解释 F 统计量的含义
- [ ] 我能正确进行 ANOVA 的前提假设检查（正态性、方差齐性、独立性），并理解"轻度偏离可接受"的稳健性
- [ ] 我能正确进行 ANOVA，并计算效应量 η²
- [ ] 我能选择合适的事后检验方法（Tukey HSD、Bonferroni、FDR），并理解它们的权衡
- [ ] 我能计算并解释效应量（η²、Cramér's V），区分统计显著与实际意义
- [ ] 我能进行卡方检验，判断分类变量是否相关，并明确"相关 ≠ 因果"
- [ ] 我能审查 AI 生成的多组比较报告，识别未校正多重比较、混淆相关与因果等常见谬误
- [ ] 我能在 StatLab 报告中添加"多组比较结果"章节，包含 ANOVA 表、事后检验、η²、卡方检验
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解"ANOVA 是全局检验，事后检验是局部定位"的逻辑，以及多重比较校正的必要性
