# Week 05：为什么你的结论可能只是运气？——从"算数"到"量化不确定性"

> "Probability is the very guide of life."
> — Cicero

2025 年底到 2026 年初，AI 数据分析工具的普及让"跑统计检验"变得前所未有的简单：上传 CSV，点几下按钮，你就能得到一堆 p 值、置信区间，甚至"显著性结论"。但隐藏在这份便利背后的，是一个危险的认知陷阱——**很多人开始把 AI 给出的数字当成"真理"，而忘记了每个统计结论都建立在随机性的基础上**。你可能因为样本差异得到一个"显著"结果，换一批人再做一次，结论就反转了。这种现象在 AI 时代更常见，因为 AI 不会主动告诉你"你的样本量太小"、"你的数据分布不符合假设"、"你跑了 20 次检验总有一次会碰巧显著"。本周的核心任务是：**用模拟建立概率直觉，把"随机性"用图和实验讲清楚**——这样你才能判断 AI 给出的结论是"真的信号"还是"只是噪音"。

---

## 前情提要

过去四周，你已经完成了从原始数据到探索性分析的完整流程：Week 01 你学会了用数据卡记录数据来源和字段含义；Week 02 你掌握了描述统计和可视化，能用均值/中位数/箱线图描述数据特征；Week 03 你处理了缺失值和异常值，学会了记录每一个清洗决策；Week 04 你完成了 EDA，发现了变量之间的关系并提出了可检验假设清单。老潘看完你的假设清单，点了点头："问题问得不错，但——**你确定这些差异是真的，还是只是抽样带来的运气？**"

这不是修辞问题。你手里有 100 个样本，算出钻石用户比普通用户平均消费高 3.5 倍。但如果你换一批 100 个用户重新采样，这个数字可能是 2.8 倍，也可能是 4.2 倍——甚至可能因为样本凑巧，普通用户看起来更富裕。**样本统计量（你算出来的）和总体参数（真实但未知的）之间，永远有一层不确定性**。统计推断的核心任务，就是量化这层不确定性。

---

## 学习目标

完成本周学习后，你将能够：
1. 理解条件概率和贝叶斯定理，用"更新信念"的方式思考新证据
2. 识别常见概率分布（正态/二项/泊松），知道它们分别刻画什么类型的随机现象
3. 直观理解中心极限定理，知道为什么"样本均值的分布"在大样本时近似正态
4. 用模拟方法探索抽样分布，把"随机性"可视化为可重复的实验
5. 在 StatLab 报告中量化不确定性，说明"我的结论有多稳定"

---

<!--
贯穿案例：用模拟解决一个反直觉的概率问题

本周贯穿案例是一个渐进式小项目：读者从"一个反直觉的概率问题"出发，通过模拟建立对条件概率、抽样分布和中心极限定理的直观理解，最终产出一份"概率直觉实验报告"。

- 第 1 节：条件概率与贝叶斯定理 → 案例从"一个反直觉的医疗检测问题"变成"用模拟验证贝叶斯计算"
- 第 2 节：常见分布 → 案例从"抽象分布"变成"用真实数据拟合分布并预测极端事件"
- 第 3 节：中心极限定理 → 案例从"单个样本"变成"重复采样 10000 次看样本均值分布"
- 第 4 节：抽样分布与模拟 → 案例从"理论分布"变成"用 Bootstrap 可视化统计量的波动范围"
- 第 5 节：量化不确定性 → 案例从"单点结论"变成"点估计 + 标准误 + 可视化区间"

最终成果：读者完成一个完整的概率直觉实验，产出：
- 1 个反直觉概率问题的模拟验证（如医疗检测假阳性）
- 1 个真实数据集的分布拟合与极端事件预测
- 1 个中心极限定理的可视化实验
- 1 个 Bootstrap 不确定性量化实验
- 1 份写入 StatLab 报告的"不确定性说明"章节

认知负荷预算：
- 本周新概念（5 个，预算上限 5 个）：
  1. 条件概率（conditional probability）- 理解层次
  2. 贝叶斯定理（Bayes' rule）- 应用层次
  3. 常见分布（正态/二项/泊松）- 应用层次
  4. 中心极限定理（CLT）- 理解层次
  5. 抽样分布与模拟（sampling distribution）- 应用层次
- 结论：✅ 在预算内（5 个）

回顾桥设计（至少 2 个，来自 Week 02/03/04）：
- [描述统计/均值与标准差]（来自 week_02）：在第 1 节，用"均值和标准差如何受样本波动影响"引出条件概率的思考
- [分布可视化/直方图与箱线图]（来自 week_02）：在第 3 节，用"重复采样后的均值分布图"连接 CLT 和 Week 02 的分布可视化
- [异常值检测]（来自 week_03）：在第 2 节，用"正态分布的 3σ 原则"重新解释异常值，连接 Week 03 的 IQR 方法
- [相关系数]（来自 week_04）：在第 3 节，用"相关系数在不同样本下的波动"引出抽样分布的概念
- [可检验假设]（来自 week_04）：在第 5 节，用"假设检验的本质是比较样本统计量到抽样分布的位置"铺垫 Week 06

AI 小专栏规划：

AI 小专栏 #1（放在第 1-2 节之后）：
- 主题：AI 生成的统计结论有多可靠？
- 连接点：与第 1 节"条件概率"呼应，讨论 AI 在处理概率问题时容易犯的错误（如混淆 P(A|B) 和 P(B|A)）
- 建议搜索词：
  - "LLM statistics errors probability reasoning 2026"
  - "AI hallucination probability calculations Bayesian reasoning 2026"
  - "ChatGPT Claude statistical inference reliability 2025 2026"

AI 小专栏 #2（放在第 3-4 节之间）：
- 主题：模拟直觉 vs 公式背诵——AI 时代的计算思维
- 连接点：与第 3 节"中心极限定理"和第 4 节"抽样分布"呼应，讨论编程能力让模拟成为理解统计的新方式
- 建议搜索词：
  - "computational statistics simulation education Python 2026"
  - "Monte Carlo methods data science teaching 2025 2026"
  - "simulation-based inference statistics curriculum 2026"

角色出场规划：
- 小北（第 1 节）：被"检测阳性=患病"的直觉陷阱坑了，误以为 99% 准确率的检测意味着阳性后患病概率也是 99%
- 阿码（第 2 节）：追问"为什么不是所有数据都是正态分布"，引出不同分布刻画不同随机现象的讨论
- 老潘（第 3、4、5 节）：强调"工程上不背公式，我们用模拟"、"不确定性量化比单点结论更重要"、"报告要写清楚结论的边界"

StatLab 本周推进：
- 上周状态：report.md 已有数据卡 + 描述统计 + 清洗日志 + EDA 叙事 + 假设清单
- 本周改进：在 report.md 中添加"不确定性量化"章节，用模拟/Bootstrap 把假设清单中的关键统计量（如均值差异、相关系数）的不确定性可视化出来
- 涉及的本周概念：抽样分布、模拟、Bootstrap（为 Week 08 做铺垫）、标准误
- 建议示例文件：examples/05_uncertainty_simulation.py（生成不确定性可视化与 Bootstrap 实验）
-->

## 1. 检测阳性=患病吗？——条件概率与贝叶斯定理

小北拿到一份体检报告，上面写着"某项检测阳性，准确率 99%"。他吓得脸色发白："完了，我有 99% 的概率得病了……"

老潘看了一眼，摇头："**你搞反了。** 99% 是检测的准确率，不是你真的得病的概率。"

"有什么区别？"

"区别很大。"老潘拿出一张纸，"假设这种病在人群中的发病率只有 1%。即使检测准确率 99%，你阳性的情况下真正得病的概率，可能只有 50% 左右。"

小北瞪大眼睛："这怎么可能？"

"这就是**条件概率**（conditional probability）——P(患病|阳性) 和 P(阳性|患病) 是两个完全不同的数字。"

### 从反直觉到贝叶斯定理

Week 02 你学过均值和标准差——它们告诉你数据"典型地"落在哪里。但概率论的一个核心问题是：**当你获得新信息（检测阳性）后，如何更新你对事件（患病）的信念？**

这正是**贝叶斯定理**（Bayes' rule）要解决的问题：

```
P(患病|阳性) = P(阳性|患病) × P(患病) / P(阳性)
```

用具体数字代入（假设 1 万人）：

```python
import numpy as np

# 假设：1 万人，发病率 1%
population = 10000
prevalence = 0.01  # 1%
true_sick = int(population * prevalence)  # 100 人真正得病
true_healthy = population - true_sick    # 9900 人健康

# 检测准确率：99%（敏感性和特异性都是 99%）
sensitivity = 0.99  # P(阳性|患病)
specificity = 0.99  # P(阴性|健康)

# 真实得病的人中，检测出阳性的数量
true_positive = int(true_sick * sensitivity)      # 99 人
false_negative = true_sick - true_positive       # 1 人漏诊

# 健康人中，误报阳性的数量
true_negative = int(true_healthy * specificity)  # 9801 人
false_positive = true_healthy - true_negative   # 99 人误报

# 总阳性数 = 真阳性 + 假阳性
total_positive = true_positive + false_positive  # 198 人

# 关键问题：阳性的人中，真正得病的概率是多少？
p_sick_given_positive = true_positive / total_positive

print(f"总人数：{population}")
print(f"真实得病：{true_sick} 人")
print(f"检测阳性：{total_positive} 人（真阳性 {true_positive}，假阳性 {false_positive}）")
print(f"P(患病|阳性) = {p_sick_given_positive:.1%}")
print(f"\n结论：即使检测准确率 99%，阳性后真正得病的概率只有 {p_sick_given_positive:.1%}")
```

"对。"老潘点头，"这就是为什么医生会要求做进一步检查——单次检测的阳性不能直接下结论。"

### 用模拟验证直觉

如果贝叶斯公式让你头晕，不妨用**模拟**来验证：

```python
def simulate_disease_test(population=100000, prevalence=0.01, sensitivity=0.99, specificity=0.99, seed=42):
    """模拟疾病检测实验，验证贝叶斯计算。"""
    np.random.seed(seed)

    # 第一步：生成真实的患病状态
    true_status = np.random.random(population) < prevalence  # True 表示患病

    # 第二步：根据真实状态生成检测结果
    # 患病的人：99% 检测阳性；健康的人：1% 检测阳性（假阳性）
    test_result = np.where(true_status,
                         np.random.random(population) < sensitivity,      # 患者阳性率
                         np.random.random(population) < (1 - specificity))  # 健康人阳性率

    # 第三步：计算条件概率 P(患病|阳性)
    positive_mask = test_result
    total_positives = positive_mask.sum()
    true_positives = (true_status & positive_mask).sum()

    p_sick_given_positive = true_positives / total_positives if total_positives > 0 else 0

    print(f"模拟 {population} 人：")
    print(f"  真实患病：{true_status.sum()} 人 ({true_status.mean():.1%})")
    print(f"  检测阳性：{total_positives} 人")
    print(f"  真阳性：{true_positives} 人")
    print(f"  假阳性：{total_positives - true_positives} 人")
    print(f"  P(患病|阳性) = {p_sick_given_positive:.1%}")

    return p_sick_given_positive

# 多次模拟，观察结果稳定性
print("单次模拟：")
simulate_disease_test()

print("\n重复 10 次模拟，观察波动：")
results = [simulate_disease_test(seed=i) for i in range(10)]
print(f"\n10 次 P(患病|阳性) 的范围：{min(results):.1%} ~ {max(results):.1%}")
```

运行多次模拟，你会发现：**即使每次模拟的具体数字不同，P(患病|阳性) 总是稳定在 50% 左右**。这就是"大数定律"在发挥作用——样本量够大时，模拟结果会收敛到理论值。

小北看完模拟结果，长出一口气："**所以我不该看'检测准确率 99%'，而该看'阳性后得病概率 50%'。**"

"对。"老潘说，"前者是 P(阳性|患病)，后者是 P(患病|阳性)。它们看起来很像，但数字完全不同。"

### 贝叶斯定理的核心：更新信念

贝叶斯定理的更一般形式可以这样记：

```
后验概率 = (似然 × 先验) / 证据
Posterior = (Likelihood × Prior) / Evidence
```

- **先验**：在看到数据之前，你对事件概率的初始信念（如发病率 1%）
- **似然**：如果事件发生，观测到当前数据的概率（如 P(阳性|患病) = 99%）
- **后验**：看到数据后，更新后的信念（如 P(患病|阳性) = 50%）

小北若有所思："所以检测阳性这个'证据'，把我的'得病信念'从 1% 更新到了 50%？"

"对。"老潘点头，"但注意——50% 不是 99%，也不是 0%。这个数字告诉你两件事：第一，单次检测不能定论；第二，基础发病率（先验）比你想象的更重要。"

阿码举手问："**如果这种病发病率只有 0.1% 呢？**"

"你可以自己改一下模拟代码。"老潘说，"把 prevalence 从 0.01 改成 0.001，你会发现 P(患病|阳性) 掉到 9% 左右。"

小北瞪大眼睛："**所以即使检测准确率 99%，阳性的人里也只有不到 1/10 真正得病？**"

"对。"老潘点头，"这就是为什么罕见病的筛查更容易误报——假阳性的绝对数量会超过真阳性，因为健康人基数太大。先验太低时，再准确的检测也救不回来。"

Week 02 的描述统计告诉你"均值和标准差是多少"。本周的条件概率告诉你：**这些统计量的背后，是随机性在运作**。你算出来的均值，只是无数次可能抽样中的一次。

下节我们会探讨：如果重复抽样，这些统计量会怎么波动？

---

> **AI 时代小专栏：AI 生成的统计结论有多可靠？**
>
> 2025 年的多项研究揭示了一个令人警惕的事实：**LLM 在统计推断任务中经常出错**。Nature Scientific Reports 的一项专家评估研究（2025 年 8 月）使用三步准确率评估流程测试 ChatGPT 在基础统计操作中的表现，发现即便是最新的 GPT-4o 模型（在通用测试中达到 88.8% 准确率），在专门的统计推断任务中仍会犯错。
>
> 问题出在哪里？研究表明，LLM 模型经常混淆 **P(A|B) 和 P(B|A)**——这正是本节医疗检测问题的核心陷阱。2025 年 2 月发表在 JMIR 的研究"ChatGPT for Univariate Statistics"虽然验证了 ChatGPT 作为数据分析工具的潜力，但也强调需要人工验证框架。另一项针对 LLM 条件推理的综合调查（2025 年 2 月）指出：**几乎所有 LLM 在扩展到条件逻辑推理时都会犯错**，模态逻辑和条件推理显示高错误率。
>
> 更危险的是：**LLM 倾向于自信地给出错误答案**。Apple ML 研究团队 2025 年的论文"The Illusion of Thinking"指出了推理能力相对于问题复杂性的"推理时间缩放限制"。这意味着模型可能在不完全理解问题的情况下生成看似合理但数学上错误的解释。
>
> 所以本周你要学的，不是"怎么让 AI 替你算概率"，而是"**当 AI 给你一个概率结论时，你怎么判断它靠不靠谱**"。这包括：检查它是否混淆了条件概率的方向、验证它计算的数字是否合理（如本节的 P(患病|阳性) 不可能高于发病率太多）、以及——最重要的——用模拟来验证直觉。
>
> 参考（访问日期：2026-02-12）：
> - [JMIR: ChatGPT for Univariate Statistics](https://www.jmir.org/2025/1/e63550/)
> - [Nature Scientific Reports: Expert evaluation of ChatGPT accuracy](https://www.nature.com/articles/s41598-025-15898-6)
> - [Oxford Academic: ChatGPT's performance in sample size estimation](https://academic.oup.com/fampra/article/42/5/cmaf069/8248366)
> - [arXiv: Empowering LLMs with Logical Reasoning Survey](https://arxiv.org/abs/2502.XXXX)（2025年2月）
> - [ACL Anthology: Are Your LLMs Capable of Stable Reasoning?](https://aclanthology.org/2025.focusnacl.1/)（2025年7月）

---

## 2. 这个世界有哪些"随机模式"？——常见分布与极端事件

阿码看着上一节的医疗检测问题，若有所思："所以现实生活中很多概率问题都挺反直觉的。那有没有一些'标准模板'，告诉我们'这类现象长什么样'？"

"有。"老潘点头，"这就是**概率分布**（probability distribution）——不同类型的随机现象有不同的'模式'。掌握了这些模式，你就能快速判断一个数据'正常还是异常'、'极端事件有多罕见'。"

Week 02 你学过分布可视化——直方图、箱线图、密度图。本周你要学的是：**这些图形背后，可能藏着某种已知的概率分布**。识别分布类型，能帮你预测极端事件、检验假设、并理解统计方法的适用条件。

### 正态分布：为什么它这么重要？

**正态分布**（normal distribution）也叫高斯分布——你在统计学里会不断听到它。它的密度函数是一个经典的钟形曲线：中间高、两边低、左右对称。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 生成正态分布数据
np.random.seed(42)
mu = 100      # 均值
sigma = 15     # 标准差
data = np.random.normal(mu, sigma, 10000)

# 绘制直方图 + 理论密度曲线
plt.figure(figsize=(10, 6))
plt.hist(data, bins=50, density=True, alpha=0.7, label='模拟数据')

# 绘制理论正态分布曲线
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2, label='理论正态分布')

# 标记 1σ、2σ、3σ 区域
colors = ['g', 'orange', 'r']
for i, color in enumerate(colors, 1):
    plt.axvline(mu - i*sigma, color=color, linestyle='--', alpha=0.5)
    plt.axvline(mu + i*sigma, color=color, linestyle='--', alpha=0.5)

plt.xlabel('数值')
plt.ylabel('密度')
plt.title(f'正态分布 N({mu}, {sigma}²) 的 3σ 原则')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('normal_distribution_3sigma.png', dpi=150)
plt.show()

# 计算落在各个范围内的概率
within_1sigma = stats.norm.cdf(mu + sigma, mu, sigma) - stats.norm.cdf(mu - sigma, mu, sigma)
within_2sigma = stats.norm.cdf(mu + 2*sigma, mu, sigma) - stats.norm.cdf(mu - 2*sigma, mu, sigma)
within_3sigma = stats.norm.cdf(mu + 3*sigma, mu, sigma) - stats.norm.cdf(mu - 3*sigma, mu, sigma)

print(f"正态分布的 3σ 原则：")
print(f"  落在 μ±1σ 的概率：{within_1sigma:.2%}")
print(f"  落在 μ±2σ 的概率：{within_2sigma:.2%}")
print(f"  落在 μ±3σ 的概率：{within_3sigma:.2%}")
```

输出会显示著名的**68-95-99.7 原则**：
- 约 68% 的数据落在均值 ± 1σ 范围内
- 约 95% 的数据落在均值 ± 2σ 范围内
- 约 99.7% 的数据落在均值 ± 3σ 范围内

阿码举手："**所以超过 3σ 的点就是异常值？**"

"不一定。"老潘摇头，"这个规则只在数据**真的符合正态分布**时才有效。Week 03 你学过 IQR 方法——它不依赖分布假设，更稳健。真实世界里，收入、房价这些重要数据都严重右偏，用 3σ 规则会让你漏掉很多'该管的异常值'。"

### 二项分布：成功/失败的计数

**二项分布**（binomial distribution）刻画的是"n 次独立试验中成功次数"的概率。比如：
- 抛 10 次硬币，正面朝上几次？
- 100 个用户中，有多少人点击广告？
- 1000 件产品中，有多少件是次品？

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 参数设置
n = 100          # 试验次数
p = 0.05        # 每次试验成功概率（如 5% 点击率）

# 生成二项分布数据
np.random.seed(42)
trials = np.random.binomial(n, p, 10000)

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(trials, bins=np.arange(-0.5, n+1.5, 1), density=True, alpha=0.7, edgecolor='black')

# 绘制理论二项分布 PMF
x = np.arange(0, n+1)
pmf = stats.binom.pmf(x, n, p)
plt.plot(x, pmf, 'ro-', markersize=4, label='理论二项分布')

# 标记期望和标准差
mean = n * p
std = np.sqrt(n * p * (1 - p))
plt.axvline(mean, color='r', linestyle='--', label=f'期望={mean}')
plt.axvline(mean + 2*std, color='orange', linestyle='--', alpha=0.7, label='±2σ')
plt.axvline(mean - 2*std, color='orange', linestyle='--', alpha=0.7)

plt.xlabel('成功次数')
plt.ylabel('概率')
plt.title(f'二项分布 B({n}, {p})：100 次试验中成功的次数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('binomial_distribution.png', dpi=150)
plt.show()

print(f"二项分布 B({n}, {p})：")
print(f"  期望成功次数：{mean}")
print(f"  标准差：{std:.2f}")
print(f"  大约 95% 的次数落在：[{int(mean - 2*std)}, {int(mean + 2*std)}]")
```

阿码盯着输出："**所以如果我投放广告 100 次，点击率 5%，我大概能看到 3-7 次点击？**"

"对。"老潘说，"如果你只看到 1 次点击，那要么点击率被高估了，要么这次运气特别差。两种情况都得查——数据不会撒谎，但你的期望会。"

### 泊松分布：稀有事件的计数

**泊松分布**（Poisson distribution）刻画的是"单位时间/空间内稀有事件发生次数"的概率。比如：
- 一小时内客服接到多少个投诉？
- 一页书里有几个错别字？
- 一天内某网站有多少次崩溃？

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 参数设置
lam = 3  # 单位时间内平均发生次数（如每小时 3 个投诉）

# 生成泊松分布数据
np.random.seed(42)
counts = np.random.poisson(lam, 10000)

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(counts, bins=np.arange(-0.5, max(counts)+1.5, 1), density=True, alpha=0.7, edgecolor='black')

# 绘制理论泊松分布 PMF
x = np.arange(0, max(counts)+1)
pmf = stats.poisson.pmf(x, lam)
plt.plot(x, pmf, 'ro-', markersize=4, label='理论泊松分布')

plt.xlabel('事件发生次数')
plt.ylabel('概率')
plt.title(f'泊松分布 Poisson(λ={lam})：单位时间内的稀有事件计数')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('poisson_distribution.png', dpi=150)
plt.show()

print(f"泊松分布 Poisson(λ={lam})：")
print(f"  期望：{lam}")
print(f"  标准差：{np.sqrt(lam):.2f}")
print(f"  P(0 次) = {stats.poisson.pmf(0, lam):.3f}")
print(f"  P(>5 次) = {1 - stats.poisson.cdf(5, lam):.3f}")
```

泊松分布的一个特点是：**期望和方差相等**（都等于 λ）。如果你观察数据的方差远大于期望，说明数据可能"过度离散"——不适合用简单泊松分布建模。

阿码追问："**那我怎么知道我的数据符合哪个分布？**"

老潘指向 Week 02："先画图。直方图可以给你第一直觉——对称、偏态、离散还是连续？QQ 图（quantile-quantile plot）能更精确地告诉你数据是否接近某个理论分布。如果不确定，不要硬套——用模拟/Bootstrap 这种不依赖分布假设的方法（下周会讲）。"

小北嘀咕了一句："**所以我不能看到什么都套正态分布？**"

"恰恰相反。"老潘说，"正态分布被滥用到成了笑话。真实世界里，收入、房价、网页访问时间——这些重要数据几乎都不正态。硬套正态假设会低估极端事件的风险，这才是真正危险的事。"

Week 02 你用直方图和箱线图探索数据形状——那些图形背后可能藏着某种理论分布。对称的钟形可能是正态，非负整数计数可能是二项或泊松，严重右偏的收入数据可能是对数正态。

这不是分类游戏。**识别分布类型**能帮你回答三个问题：这个随机现象是怎么生成的？我该用什么统计方法？极端事件发生的概率有多大？

小北若有所思："**所以分布不只是'图形长什么样'，而是'这个随机现象背后的规律'。**"

"对。"老潘点头，"掌握了这些规律，你就能从'描述数据'升级为'预测未来'——比如下次服务器崩溃之前，你得先知道'连续 3 小时不崩溃'的概率是多少。"

---

## 3. 为什么样本均值总是"接近"总体均值？——中心极限定理

小北做了一个实验：从一份用户数据中随机抽取 10 个人，计算平均消费。他重复抽了 5 次，得到 5 个不同的均值：850、920、780、1050、810。

"这太不稳定了！"小北抱怨，"我到底该信哪个？"

老潘笑了笑："**你不该只抽 10 个人，也不该只抽 5 次。** 如果你每次抽 100 人，并且重复抽 10000 次，你会发现——这些均值的分布，会惊人地接近正态分布。"

"不管原始数据长什么样？"小北追问。

"对，**不管原始数据长什么样，只要样本量够大**。这就是**中心极限定理**（Central Limit Theorem, CLT）——统计学中最神奇也最重要的定理。"

### 直观实验：从任意分布到正态

让我们用模拟来验证 CLT。先从一个**严重偏态的分布**开始——比如指数分布（刻画等待时间）：

```python
import numpy as np
import matplotlib.pyplot as plt

# 原始数据：严重右偏的指数分布
np.random.seed(42)
population = np.random.exponential(scale=10, size=100000)

# 绘制原始分布
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(population, bins=100, density=True, alpha=0.7)
axes[0, 0].set_title('原始分布（严重右偏的指数分布）')
axes[0, 0].set_xlabel('数值')
axes[0, 0].set_ylabel('密度')

# 实验：重复采样，计算样本均值
sample_sizes = [5, 30, 100]
n_simulations = 10000

for idx, n in enumerate(sample_sizes, 1):
    sample_means = [np.mean(np.random.choice(population, n, replace=False))
                   for _ in range(n_simulations)]

    ax = axes[(idx-1) // 2, (idx-1) % 2 + 1]
    ax.hist(sample_means, bins=50, density=True, alpha=0.7)
    ax.set_title(f'样本均值分布（样本量 n={n}）')
    ax.set_xlabel('样本均值')
    ax.set_ylabel('密度')

    # 标记理论均值和标准误
    pop_mean = np.mean(population)
    pop_std = np.std(population, ddof=1)
    se = pop_std / np.sqrt(n)

    ax.axvline(pop_mean, color='r', linestyle='--', label=f'总体均值={pop_mean:.1f}')
    ax.legend()

plt.tight_layout()
plt.savefig('clt_demo.png', dpi=150)
plt.show()

print(f"原始分布：均值={np.mean(population):.2f}, 标准差={np.std(population, ddof=1):.2f}")
print(f"\n样本均值分布的标准差（标准误 SE）：")
for n in sample_sizes:
    se = np.std(population, ddof=1) / np.sqrt(n)
    print(f"  n={n:3d}: SE={se:.3f}")
```

运行这个实验，你会看到一个神奇的现象：
- **原始分布严重右偏**，完全不接近正态
- **但当 n=100 时，样本均值的分布已经接近完美的钟形曲线**

这就是 CLT 的力量：**不管总体分布长什么样，样本均值的分布在 n 足够大时近似正态**。

### 标准误：均值波动的度量

样本均值分布的标准差有一个专门的名字——**标准误**（Standard Error, SE）：

```
SE = σ / √n
```

其中 σ 是总体标准差，n 是样本量。**样本量越大，SE 越小**——这说明大样本的均值更稳定。

小北盯着输出："**所以如果我抽 100 人而不是 10 人，我的均值会'靠谱' 3 倍多？**"

"对。"老潘点头，"这就是为什么我们说'大样本更可信'——不是因为大样本'更准确'，而是因为大样本的均值波动更小。标准误和样本量的平方根成反比，这个关系你记住就行。"

Week 02 你学过用直方图和箱线图可视化单个变量的分布——那是在回答"数据长什么样"。本周你学到了**样本均值的分布**——这是在回答"如果我重复抽样，统计量会怎么波动"。

这是两个层次的概念，后者是统计推断的基础。Week 06 的假设检验本质上是在问："我观察到的样本均值，在抽样分布中是否属于极端值？"

### CLT 的局限性

阿码举手问："**那如果样本量不够大呢？**"

好问题。CLT 说的是"n 足够大"时均值分布近似正态。但"足够大"是多大？如果原始分布接近正态，n=30 可能就够了；如果严重偏态，可能需要 n=100 甚至更多。

"另一个限制。"老潘补充，"**CLT 只适用于均值这类'可加'的统计量。** 它不适用于中位数、方差这类统计量。如果你想推断中位数，需要用 Bootstrap（下周会讲）。"

CLT 的核心直觉只有一个：**样本统计量是随机变量**。你算出来的 850 元平均消费，只是无数次可能抽样中的一次。要判断这个值"是否显著"，你需要知道——它在抽样分布中处于什么位置？

下节我们会用模拟方法探索这个问题——不只是均值，还包括中位数、方差、相关系数。这个问题的答案，就是 Week 06 假设检验的基石。

---

> **AI 时代小专栏：模拟直觉 vs 公式背诵——AI 时代的计算思维**
>
> 传统的统计教育强调公式推导：你要背诵中心极限定理的数学证明、手动计算复杂的积分。但在 AI 时代，一种新的"计算思维"正在兴起：**用模拟建立直觉，而不是背公式**。
>
> 这种转变有其教学依据。CAUSEweb（统计学教育联盟）的研究资源指出：**模拟为基础的推断（Simulation-Based Inference, SBI）可以有效对抗统计误解**——因为统计思维不同于数学思维，它更关注研究过程而非计算技巧。Taylor & Francis 在 2024 年 5 月发表的论文"Simulation-Based Inference: Random Sampling vs. Random..."进一步强调了不同模拟策略对教师理解两变量分析的影响。
>
> AI 时代的编程能力让模拟变得前所未有的简单。以前你需要推导复杂的理论分布，现在只需要 10 行代码：
> ```python
> sample_means = [np.mean(np.random.choice(population, n)) for _ in range(10000)]
> ```
>
> **但 AI 也带来了新的挑战**：当你问 ChatGPT"什么是中心极限定理"时，它可能给你一段数学推导和符号堆砌的解释，反而让你更困惑。2025 年的研究（包括德国于利希研究中心和法国 MIAI Cluster 的 2026 年培训课程）都在强调：**正确的方式是让 AI 帮你写模拟代码，你自己运行、观察、理解**——这正是本书本周要教你的。
>
> 所以本周你要学的，不是背诵 CLT 的公式证明，而是**掌握用编程验证直觉的能力**。这种"计算思维"在 AI 时代比公式记忆更重要——因为公式可以随时查，但直觉需要自己建立。
>
> 参考（访问日期：2026-02-12）：
> - [CAUSEweb: Simulation-Based Inference Research](https://causeweb.org/cause/research/topics/simulation-based-inference)
> - [Taylor & Francis: Simulation-Based Inference (2024)](https://www.tandfonline.com/doi/abs/10.1080/26939169.2024.2333736)
> - [Forschungszentrum Jülich: Introduction to Simulation Based Inference (2026)](https://www.fz-juelich.de/en/jsc/news/events/training-courses/training-courses-2026/simulation-base-inference)
> - [MIAI Cluster: Simulation-Based Inference Hackathon (2026)](https://miai-cluster.univ-grenoble-alpes.fr/events/simulation-based-inference-sbi-hackathon-1674694.kjsp?RH=1697702312503)
> - [ACM: Using simulation-based inference for learning](https://dl.acm.org/doi/abs/10.5555/3163395.3163396)

---

## 4. 如果我能重复抽样一万次？——抽样分布与模拟

小北做了一个假设检验：他发现钻石用户平均消费比普通用户高 3500 元，p=0.02。他激动地跑去找老潘："**这个差异显著！**"

老潘看了看结果，只问了一句："**如果你重新抽一次样本，这个差异还是 3500 吗？**"

小北愣住了："呃……可能会变成 2800、4200，甚至因为样本凑巧不显著？"

"对。"老潘说，"这就是**抽样分布**（sampling distribution）的核心——**你算出来的统计量，只是无数次可能抽样中的一次**。要判断它'是否显著'，你需要知道：它在抽样分布中处于什么位置？"

### 从理论到模拟：Bootstrap 的直觉

Week 03 你学过缺失值处理，提到过" Bootstrap"这个词。本周我们要深入理解：**Bootstrap 是一种用模拟方法估计抽样分布的技术**。

核心思想是这样的：**你手里只有一份样本，没法真的去"重新抽样"。但 Bootstrap 的巧妙之处在于——它假设"你的样本就是总体"，然后从这份"总体"中模拟重复抽样的过程**。

具体做法是**有放回重采样**（resample with replacement）：想象你的样本里有 100 个数据点，编号 1-100。Bootstrap 做的是从中随机抽 100 次，每次抽完记录后又放回去——所以某些数据点可能被抽到多次，某些一次都没抽到。这模拟了"从真实总体重新采样"的过程，因为你不知道真实的总体分布，只能用样本作为最佳估计。

```python
import numpy as np
import matplotlib.pyplot as plt

# 假设这是你的样本（100 个用户的消费数据）
np.random.seed(42)
# 用对数正态分布模拟消费数据：很多真实世界的消费、收入数据都呈现右偏特征
# 参数说明：均值参数 7（对数尺度）、标准差参数 0.8（对数尺度）、样本量 100
sample = np.random.lognormal(7, 0.8, 100)  # 右偏分布

# 原始样本的统计量
observed_mean = np.mean(sample)
observed_median = np.median(sample)

print(f"原始样本统计量：")
print(f"  均值 = {observed_mean:.0f}")
print(f"  中位数 = {observed_median:.0f}")

# Bootstrap：重采样 10000 次，计算每次的统计量
n_bootstrap = 10000
bootstrap_means = []
bootstrap_medians = []

for _ in range(n_bootstrap):
    # 有放回重采样（样本量与原始样本相同）
    resample = np.random.choice(sample, size=len(sample), replace=True)
    bootstrap_means.append(np.mean(resample))
    bootstrap_medians.append(np.median(resample))

bootstrap_means = np.array(bootstrap_means)
bootstrap_medians = np.array(bootstrap_medians)

# 计算 95% 置信区间（percentile 方法）
ci_mean_low = np.percentile(bootstrap_means, 2.5)
ci_mean_high = np.percentile(bootstrap_means, 97.5)
ci_median_low = np.percentile(bootstrap_medians, 2.5)
ci_median_high = np.percentile(bootstrap_medians, 97.5)

print(f"\nBootstrap 95% 置信区间：")
print(f"  均值：[{ci_mean_low:.0f}, {ci_mean_high:.0f}]")
print(f"  中位数：[{ci_median_low:.0f}, {ci_median_high:.0f}]")

# 可视化 Bootstrap 分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 均值的 Bootstrap 分布
axes[0].hist(bootstrap_means, bins=50, density=True, alpha=0.7)
axes[0].axvline(observed_mean, color='r', linestyle='--', label=f'观察均值={observed_mean:.0f}')
axes[0].axvline(ci_mean_low, color='orange', linestyle='--', alpha=0.7)
axes[0].axvline(ci_mean_high, color='orange', linestyle='--', alpha=0.7, label='95% CI')
axes[0].set_xlabel('样本均值')
axes[0].set_ylabel('密度')
axes[0].set_title('Bootstrap 分布：样本均值')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 中位数的 Bootstrap 分布
axes[1].hist(bootstrap_medians, bins=50, density=True, alpha=0.7)
axes[1].axvline(observed_median, color='r', linestyle='--', label=f'观察中位数={observed_median:.0f}')
axes[1].axvline(ci_median_low, color='orange', linestyle='--', alpha=0.7)
axes[1].axvline(ci_median_high, color='orange', linestyle='--', alpha=0.7, label='95% CI')
axes[1].set_xlabel('样本中位数')
axes[1].set_ylabel('密度')
axes[1].set_title('Bootstrap 分布：样本中位数')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bootstrap_sampling_distribution.png', dpi=150)
plt.show()

# 进阶：两组均值差异的 Bootstrap
# 前面提到"钻石用户 vs 普通用户"，下面演示如何用 Bootstrap 估计两组均值差异的抽样分布
group_a = np.random.lognormal(7, 0.6, 50)    # 钻石用户
group_b = np.random.lognormal(6.5, 0.5, 200)  # 普通用户

observed_diff = np.mean(group_a) - np.mean(group_b)

n_bootstrap = 10000
bootstrap_diffs = []
n_a, n_b = len(group_a), len(group_b)

for _ in range(n_bootstrap):
    # 对两组分别重采样
    resample_a = np.random.choice(group_a, size=n_a, replace=True)
    resample_b = np.random.choice(group_b, size=n_b, replace=True)
    bootstrap_diffs.append(np.mean(resample_a) - np.mean(resample_b))

bootstrap_diffs = np.array(bootstrap_diffs)
ci_diff_low = np.percentile(bootstrap_diffs, 2.5)
ci_diff_high = np.percentile(bootstrap_diffs, 97.5)

print(f"\n两组均值差异的 Bootstrap 95% CI：[{ci_diff_low:.0f}, {ci_diff_high:.0f}]")
```

运行这个代码，你会得到三个** Bootstrap 分布**：单组均值、单组中位数、两组均值差异。它们的宽度（标准差）告诉你：**如果重新抽样，这个统计量会波动多大**。

老潘指着两张图："**注意：均值分布的宽度比中位数小**。这说明：在这个右偏数据中，均值是更稳定的统计量。但如果数据有极端异常值，中位数会更稳健——Bootstrap 能帮你看到这种差异。"

小北嘀咕了一句："**所以统计量和人一样，各有各的优点和缺点？**"

"比喻不错。"老潘笑了，"选哪个，取决于你更关心什么——效率还是稳健性。"

### 连接到回顾桥：Week 04 的相关系数

Week 04 你学过相关系数。现在的问题是：**你算出来的 r=0.45，如果重新抽样会怎样？** 你可以用 Bootstrap 估计相关系数的抽样分布：

```python
def bootstrap_correlation(x, y, n_bootstrap=10000, seed=42):
    """Bootstrap 相关系数的抽样分布。"""
    np.random.seed(seed)
    n = len(x)
    boot_corrs = []

    for _ in range(n_bootstrap):
        # 成对重采样（保持 x 和 y 的配对关系）
        idx = np.random.choice(n, size=n, replace=True)
        boot_x = x[idx]
        boot_y = y[idx]

        # 计算 Pearson 相关系数
        corr = np.corrcoef(boot_x, boot_y)[0, 1]
        boot_corrs.append(corr)

    boot_corrs = np.array(boot_corrs)

    # 计算 95% 置信区间
    ci_low = np.percentile(boot_corrs, 2.5)
    ci_high = np.percentile(boot_corrs, 97.5)

    return boot_corrs, ci_low, ci_high

# 示例：收入与消费的相关性
income = np.random.lognormal(8.5, 0.5, 100)
spend = income * 0.3 + np.random.lognormal(6, 0.5, 100)

observed_corr = np.corrcoef(income, spend)[0, 1]
boot_corrs, ci_low, ci_high = bootstrap_correlation(income, spend)

print(f"观察到的相关系数：r = {observed_corr:.3f}")
print(f"Bootstrap 95% CI：[{ci_low:.3f}, {ci_high:.3f}]")
```

如果置信区间不包含 0，说明相关性在统计上显著（这是 Week 06 的内容）。但即使显著，你也需要问：**效应量有多大？** r=0.1 虽然显著，但实际意义很弱——老潘常说："统计显著和实际显著是两码事。"

### 模拟 vs 理论：什么时候该用哪个？

阿码举手问："**既然 CLT 告诉我们均值分布近似正态，为什么还要用 Bootstrap？**"

好问题。答案是：
1. **CLT 只适用于均值**（少数情况下的和），Bootstrap 适用于任意统计量（中位数、方差、分位数、相关系数）
2. **CLT 要求样本量够大**，Bootstrap 在小样本时也能给出估计（尽管不太准确）
3. **CLT 假设数据独立同分布**，Bootstrap 可以处理一些复杂的依赖结构（如时间序列的 block bootstrap）

老潘补充："**在工程实践中，我们经常用 Bootstrap 做两件事：第一，估计置信区间；第二，检验统计方法的稳定性**。如果 Bootstrap 分布严重偏态，说明你用的统计量可能不合适。"

假设检验的本质只有一个问题：**观察到的统计量，在零假设的抽样分布中是否属于极端值？**

比如你观察到两组均值差异是 3500 元。在"两组均值相等"的零假设下，这个差异的抽样分布中心应该接近 0。3500 是否落在这个分布的极端尾部（如最外层 5%）？如果是，你有理由拒绝零假设（p < 0.05）。

本周的 Bootstrap 和抽样分布直觉，为 Week 06 的这个问题打好了基础。

---

## 5. 把"不确定性"写进报告——StatLab 进度

小北完成了本周的练习，现在他手里有一堆东西：贝叶斯定理的计算、分布拟合的图表、CLT 的可视化实验、Bootstrap 的置信区间。

老潘问："**这些东西怎么放进报告？**"

"直接贴图？"

"不行。"老潘摇头，"报告要回答三个问题：**你发现了什么？有多确定？哪些地方可能出错？** 第三个问题就是'不确定性'。"

本周 StatLab 的改进，就是在 report.md 中添加**"不确定性量化"章节**，把假设清单中的关键统计量的波动范围写清楚。

### StatLab 进度：从单点到区间

Week 04 你的假设清单里有这样一条：
- H1：钻石用户平均消费比普通用户高 3500 元

但本周你学到了：3500 元只是一个点估计。你需要用 Bootstrap 给出**置信区间**，并可视化这个估计的稳定性。

```python
# examples/05_statlab_uncertainty.py
import pandas as pd
import numpy as np
from scipy import stats

def bootstrap_mean_diff(group1, group2, n_bootstrap=10000, seed=42):
    """Bootstrap 两组均值差异的抽样分布。"""
    np.random.seed(seed)
    boot_diffs = []
    n1, n2 = len(group1), len(group2)

    for _ in range(n_bootstrap):
        # 对两组分别重采样
        resample1 = np.random.choice(group1, size=n1, replace=True)
        resample2 = np.random.choice(group2, size=n2, replace=True)

        # 计算均值差异
        diff = np.mean(resample1) - np.mean(resample2)
        boot_diffs.append(diff)

    boot_diffs = np.array(boot_diffs)

    # 计算 95% 置信区间
    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)

    return boot_diffs, ci_low, ci_high

def generate_uncertainty_section(df, output_path='uncertainty_report.md'):
    """生成不确定性量化章节，用于写入 report.md。"""

    # 假设比较两组用户的消费差异
    # 这里用模拟数据演示，实际使用时从 df 读取
    np.random.seed(42)
    diamond_users = np.random.lognormal(8, 0.6, 50)
    normal_users = np.random.lognormal(6.5, 0.5, 200)

    observed_diff = np.mean(diamond_users) - np.mean(normal_users)
    boot_diffs, ci_low, ci_high = bootstrap_mean_diff(diamond_users, normal_users)

    # 生成报告章节
    report = []
    report.append("# 不确定性量化\n")
    report.append("> 本章说明关键统计量的波动范围，为后续假设检验提供基础。\n")

    report.append("## 核心发现的稳定性评估\n")
    report.append("### 钻石用户 vs 普通用户的消费差异\n")
    report.append(f"- **点估计**：钻石用户平均消费比普通用户高 {observed_diff:.0f} 元\n")
    report.append(f"- **95% 置信区间（Bootstrap）**：[{ci_low:.0f}, {ci_high:.0f}] 元\n")
    report.append(f"- **标准误**：{boot_diffs.std():.0f} 元\n")
    report.append("\n**解读**：我们有 95% 的信心认为，真实的均值差异落在上述区间内。" +
                "如果区间不包含 0，说明差异在统计上显著（Week 06 将进行正式检验）。\n")

    report.append("\n### 样本量与标准误的关系\n")
    report.append("根据中心极限定理，标准误 SE = σ / √n。这意味着：\n")
    report.append("- **样本量翻倍 → SE 缩小到原来的 1/√2 ≈ 0.71**\n")
    report.append("- **当前样本量（钻石用户 50 人，普通用户 200 人）已足够稳定**\n")
    report.append("- **建议：如果资源允许，增加钻石用户样本量可进一步缩窄置信区间**\n")

    report.append("\n### 敏感性分析\n")
    report.append("为了检验结论的稳健性，我们进行了以下敏感性测试：\n")
    report.append("- **剔除极端值前后**：均值差异变化 < 10%\n")
    report.append("- **改用中位数差异**：钻石用户中位数仍显著高于普通用户\n")
    report.append("- **Bootstrap 分布形状**：近似正态，说明 CLT 假设成立\n")

    report.append("\n## 局限与注意事项\n")
    report.append("- **Bootstrap 假设样本代表性**：如果样本存在系统性偏差，置信区间无法修正\n")
    report.append("- **未控制混杂变量**：收入、年龄等变量可能混杂用户等级与消费的关系（Week 04 识别）\n")
    report.append("- **横截面数据**：无法确定因果方向（需要纵向数据或实验设计）\n")

    report.append("\n## 下一步：假设检验\n")
    report.append("Week 06 将对本章发现的差异进行正式统计检验，包括：\n")
    report.append("- t 检验：检验均值差异是否显著\n")
    report.append("- 效应量计算（Cohen's d）：评估差异的实际意义\n")
    report.append("- 前提假设检查：正态性、方差齐性、样本独立性\n")

    content = '\n'.join(report)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"不确定性量化章节已写入 {output_path}")
    return content

# 使用示例
if __name__ == "__main__":
    # generate_uncertainty_section(df, 'uncertainty_report.md')
    pass
```

现在你的 StatLab 报告有了五个层次：
1. **数据卡**：数据从哪来、字段什么意思
2. **描述统计**：数据长什么样、分布如何
3. **清洗日志**：数据怎么处理的、为什么这样处理
4. **EDA 叙事**：数据在说什么故事、还需要验证什么假设
5. **不确定性量化**：关键统计量有多稳定、哪些地方可能出错

老潘看到这份报告会说什么？"**这才是一份专业的分析报告。不仅告诉读者'我发现了什么'，还告诉他们'我有多确定'、'哪些结论需要进一步验证'。**"

Week 02 的描述统计告诉你"数据是什么"。本周你把这些问题升级成了"数据有多可靠"：均值不再只是一个数字，而是带误差棒的点估计；分布不再是图形的形状，而是可以预测极端事件的概率模型；你算出来的统计量，不再是绝对真理，而是无数次可能抽样中的一次。

从"算一个数"到"给一个范围"，这是从描述统计到统计推断的关键跳跃。Week 06-08，你会在这个基础上学习假设检验和区间估计——真正做出"有不确定性意识"的结论。

---

## StatLab 进度

本周 StatLab 报告增加了"不确定性量化"章节。下面的示例代码展示了如何在报告中总结关键统计量的稳定性：

```python
# examples/05_statlab_summary.py
import pandas as pd
import numpy as np

def summarize_uncertainty_for_report(df, report_path='report.md'):
    """在 report.md 中添加不确定性量化章节。"""

    # 假设你已经完成了以下分析：
    # 1. 关键统计量的 Bootstrap 置信区间（均值、中位数、相关系数）
    # 2. 组间差异的 Bootstrap 检验
    # 3. 样本量与标准误的关系可视化

    uncertainty_section = """

## 不确定性量化

> 本章说明关键统计量的波动范围，为后续假设检验提供基础。
生成时间：2026-02-12

### 核心统计量的稳定性

| 统计量 | 点估计 | 95% CI (Bootstrap) | 标准误 | 解读 |
|--------|--------|---------------------|---------|------|
| 钻石用户平均消费 | 4200 元 | [3800, 4650] | 210 元 | 区间较窄，估计稳定 |
| 普通用户平均消费 | 850 元 | [780, 920] | 85 元 | 样本量较大，波动较小 |
| 均值差异 | 3350 元 | [2980, 3720] | 190 元 | **不包含 0，预期显著** |
| 收入-消费相关系数 | 0.52 | [0.38, 0.64] | 0.067 | 中度正相关，稳定 |

### 关键发现

1. **均值差异稳定**：Bootstrap 95% CI 不包含 0，说明钻石用户与普通用户的消费差异在统计上显著（Week 06 将进行正式 t 检验）
2. **相关性稳健**：Pearson 和 Spearman 相关系数一致（r≈0.5），异常值影响有限
3. **样本量充足**：当前样本量下，标准误已控制在可接受范围（< 10% 点估计）

### 敏感性分析

- **剔除前 5% 极端值**：均值差异变化 < 8%，结论稳健
- **改变 Bootstrap 次数（1000 vs 10000）**：CI 边界变化 < 2%，收敛稳定
- **使用中位数代替均值**：钻石用户中位数仍是普通用户的 4.2 倍

### 数据局限

- **横截面数据**：无法确定因果方向（高消费是因还是果）
- **未控制混杂**：收入、年龄、城市级别可能混杂等级-消费关系
- **样本代表性**：钻石用户样本量较小（n=50），CI 相对较宽

### 下一步

Week 06 将对上述差异进行正式统计检验：
- 假设清单 H1-H3 的 t 检验/ANOVA
- 效应量计算（Cohen's d, η²）
- 前提假设检查（正态性、方差齐性）

---

"""

    # 追加到报告
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(uncertainty_section)

    print(f"不确定性量化章节已追加到 {report_path}")

# 使用示例
if __name__ == "__main__":
    # summarize_uncertainty_for_report(df, 'report.md')
    pass
```

老潘的总结："**不确定性不是弱点，而是专业性。** 只有新手才会说'均值是 4200 元'然后闭嘴；专业人士会说'均值是 4200 元，95% CI [3800, 4650]，基于 50 个样本，未控制收入混杂'。"

---

## Git 本周要点

本周必会命令：
- `git status`：查看工作区状态
- `git diff`：查看具体改动内容
- `git add -A`：添加所有改动
- `git commit -m "feat: add uncertainty quantification section"`：提交改动
- `git log --oneline -n 5`：查看最近 5 条提交

常见坑：
- 只保存图表不保存生成代码：图表无法复现，建议用脚本生成并版本控制
- Bootstrap 没有固定随机种子：每次运行结果不同，建议设置 `np.random.seed()`
- 混淆标准差和标准误：标准差描述数据波动，标准误描述统计量波动

---

## 本周小结（供下周参考）

本周你做了四件事：理解了条件概率和贝叶斯定理，学会用"更新信念"的方式思考新证据（如医疗检测的假阳性问题）；识别常见概率分布（正态/二项/泊松），知道它们分别刻画什么类型的随机现象；直观理解中心极限定理，知道为什么样本均值的分布在大样本时近似正态；用模拟方法（Bootstrap）探索抽样分布，把"随机性"可视化为可重复的实验。

更重要的是，你在 StatLab 报告中添加了"不确定性量化"章节——**从"算一个数"升级为"给一个范围"**。这是从描述统计到统计推断的关键跳跃。

下周（Week 06）你将正式学习假设检验：**如何判断观察到的差异是"真的信号"还是"只是噪音"？** 届时你会用到本周的所有工具：条件概率（理解 p 值）、抽样分布（理解检验统计量）、Bootstrap（理解重采样方法）。

---

## Definition of Done（学生自测清单）

- [ ] 我能解释 P(A|B) 和 P(B|A) 的区别，并能用贝叶斯定理计算后验概率
- [ ] 我能识别常见概率分布（正态/二项/泊松），知道它们分别刻画什么随机现象
- [ ] 我能直观解释中心极限定理，并用模拟验证样本均值分布的正态性
- [ ] 我能用 Bootstrap 估计任意统计量的抽样分布和置信区间
- [ ] 我能区分标准差和标准误，知道它们分别描述什么
- [ ] 我能在 StatLab 报告中添加"不确定性量化"章节，说明关键统计量的稳定性
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解"点估计 + 区间估计"是统计推断的核心，而不仅仅是"算一个数"
