# Week 06：我们真的发现了差异吗？——从"p 值迷思"到"统计决策"

> "In God we trust; all others must bring data."
> — W. Edwards Deming

2025 年到 2026 年，AI 数据分析工具的爆发让"跑假设检验"变得前所未有的简单：你只需要把数据上传给 AI，几秒钟就能得到一堆 p 值、t 统计量，甚至自动生成的"结论"。但隐藏在这份便利背后的，是一个正在被放大的统计误区——**很多人开始把 p < 0.05 当成"真理的印章"，而忘记了 p 值的本质只是一个"在原假设为真时看到当前数据的概率"**。更危险的是：AI 不会主动告诉你"你的样本量够吗？"、"你跑了 20 次检验总会有一次碰巧显著"、"效应量太小了即使显著也没实际意义"。本周的核心任务是：**建立假设检验的完整框架**——从 H0/H1 的设定，到 p 值的正确理解，再到效应量和两类错误的权衡。只有掌握了这套框架，你才能在 AI 时代不盲信、不盲从，做出有据可依的统计决策。

---

## 前情提要

上周（Week 05），你用模拟建立了概率直觉：贝叶斯定理教你怎么"更新信念"，常见分布告诉你不同随机现象的"模式"，中心极限定理解释了为什么样本均值的分布在大样本时近似正态，Bootstrap 让你看到了"如果重复抽样，统计量会怎么波动"。小北看着 Bootstrap 的置信区间图，若有所思："**所以我现在知道'均值差异是 3500 元，95% CI [2900, 4100]'——但这够了吗？**"

老潘摇摇头："**不够。你需要回答一个更具体的问题：这个差异是'真的信号'，还是'只是噪音'？**"

这正是本周要解决的问题——**假设检验**（hypothesis testing）。你不再只是描述数据，而是要用数据回答"是"或"否"：两组均值是否真的不同？相关系数是否显著不为零？你的结论有多可靠？

---

## 学习目标

完成本周学习后，你将能够：
1. 用 H0/H1 的形式明确表达研究假设，理解原假设与备择假设的对立关系
2. 正确解释 p 值的含义，避免"p 值即真理概率"的常见误区
3. 选择合适的 t 检验（单样本/双样本/配对）并进行前提假设检查
4. 理解第一类错误（α）和第二类错误（β）的权衡，及其对统计功效的影响
5. 计算并解释效应量（Cohen's d），区分统计显著与实际显著
6. 审查 AI 生成的统计检验报告，识别常见谬误（如 p-hacking、前提假设未验证、效应量缺失）

---

<!--
贯穿案例：用户留存实验——"新功能真的让用户更活跃了吗？"

本周贯穿案例是一个 A/B 测试场景：公司测试了一个新功能，收集了实验组（500 人）和对照组（500 人）的活跃度数据。读者需要用假设检验框架判断：两组差异是真实的，还是只是抽样波动？

- 第 1 节：从假设清单到 H0/H1 的正式表达 → 案例从"模糊的业务问题"变成"可检验的统计假设"
- 第 2 节：理解 p 值 → 案例从"看到差异"变成"判断差异是否显著"
- 第 3 节：t 检验实战 + 效应量 → 案例从"定性判断"变成"定量结论（p 值 + Cohen's d + CI）"
- 第 4 节：第一类/第二类错误与功效 → 案例从"单次决策"变成"考虑长期风险（假阳性/假阴性）"
- 第 5 节：AI 审查训练 → 案例从"自己做检验"变成"审查 AI 生成的报告，找出统计谬误"

最终成果：读者完成一个完整的假设检验分析，产出：
- 1 个正式的 H0/H1 陈述
- 1 份 t 检验报告（包含前提假设检查、p 值、效应量、置信区间）
- 1 个功效分析（样本量是否足够）
- 1 份 AI 生成报告的审查清单（标注风险点）

认知负荷预算：
- 本周新概念（5 个，预算上限 5 个）：
  1. 原假设与备择假设（null/alternative hypothesis）- 理解层次
  2. p-value - 理解层次
  3. t 检验（t-test）- 应用层次
  4. 第一类/第二类错误（Type I/II errors）- 理解层次
  5. 效应量（effect size）- 应用层次
- 结论：✅ 在预算内（5 个）

回顾桥设计（至少 2 个，来自 Week 02/03/04/05）：
- [集中趋势/均值与中位数]（来自 week_02）：在第 1 节，用"均值差异 vs 中位数差异"引出检验统计量的选择
- [分布可视化/直方图与箱线图]（来自 week_02）：在第 3 节，用"正态性检验"连接 Week 02 的分布图和 t 检验的前提假设
- [缺失值机制与清洗决策]（来自 week_03）：在第3 节，用"删除缺失值对样本量的影响"引出功效分析
- [相关系数与分组比较]（来自 week_04）：在第 1 节，用"相关≠因果"引出 H0 的设定原则
- [抽样分布与 Bootstrap]（来自 week_05）：在第 2 节，用"Bootstrap 置信区间与 p 值的关系"引出 p 值的正确理解
- [标准误与中心极限定理]（来自 week_05）：在第 3 节，用"标准误在 t 检验中的作用"连接 t 统计量的计算

AI 小专栏规划：

AI 小专栏 #1（放在第 1-2 节之后）：
- 主题：p-hacking 与"可复现性危机"——AI 时代的统计陷阱
- 连接点：与第 1 节"H0/H1 设定"和第 2 节"p 值理解"呼应，讨论多次检验、选择性报告等常见科研误区
- 数据来源：.research_cache.md 中已搜索的真实参考链接

AI 小专栏 #2（放在第 3-4 节之间）：
- 主题：AI 能自动选择统计检验吗？——工具 vs 判断
- 连接点：与第 3 节"t 检验实战"和第 4 节"错误权衡"呼应，讨论 AI 辅助检验选择的前提假设检查和边界情况
- 数据来源：.research_cache.md 中已搜索的真实参考链接

角色出场规划：
- 小北（第 1、2 节）：把"差异显著=结论成立"当成公理，被老潘纠正；误以为 p 值是"结论为真的概率"
- 阿码（第 3 节）：追问"为什么不能对所有数据都用 t 检验"，引出前提假设检查（正态性、方差齐性）
- 老潘（第 3、4、5 节）：强调"功效分析比单纯追求 p<0.05 更重要"、"工程上要同时控制假阳性和假阴性"、"AI 生成的报告必须审计"

StatLab 本周推进：
- 上周状态：report.md 已有数据卡 + 描述统计 + 清洗日志 + EDA 叙事 + 假设清单 + 不确定性量化（Bootstrap CI）
- 本周改进：在 report.md 中添加"假设检验结果"章节，包含 H0/H1 陈述、检验方法选择、p 值、效应量（Cohen's d）、置信区间、决策结论、可视化
- 涉及的本周概念：H0/H1、p 值、t 检验、效应量、第一类/第二类错误
- 建议示例文件：examples/99_statlab.py（生成假设检验报告与可视化）
-->

## 1. 你到底在检验什么？——从问题到假设

小北拿着一份用户留存数据，兴奋地跑来找老潘："**实验组的留存率比对照组高 5%！这绝对是真的！**"

老潘看了看数据，只问了一句："**如果再抽一次样本，这个 5% 会变成 -2% 吗？**"

小北愣住了："呃……可能吧？"

"这就是问题所在。"老潘说，"你看到的 5% 差异，可能是**真实信号**，也可能是**抽样噪音**。要区分这两者，你需要先问自己：**我到底在检验什么？**"

Week 04 你写过假设清单，比如：
- "钻石用户的消费是否高于普通用户？"
- "新功能是否提升了用户活跃度？"

这些是**研究问题**，不是**统计假设**。统计假设需要用 H0（原假设）和 H1（备择假设）的形式明确表达。

**原假设 H0**：通常表示"没有差异"、"没有效果"、"等于零"——它是你希望推翻的"默认状态"。

**备择假设 H1**：通常表示"有差异"、"有效果"、"不等于零"——这是你希望支持的结论。

阿码突然想起了什么："**所以 Week 04 的假设生成，其实是在写'研究问题'，而本周要做的是把它们变成'可检验的统计假设'？**"

"对！"老潘点头，"Week 04 你列出的是'钻石用户消费是否更高'——这像是一个'问题'；而本周你要把它写成'μ_diamond = μ_normal vs μ_diamond > μ_normal'——这才是'假设'。**假设清单是你问问题的起点，假设检验是你回答问题的终点**。"

阿码举手问："**为什么 H0 总是'相等'？能不能反过来，把'有差异'设为 H0？**"

"可以，但不会这么设计。"老潘说，"**假设检验的逻辑是'证伪'：我们只能拒绝 H0，不能证明 H0**。如果你把'有差异'设为 H0，而数据没显示出差异，你无法证明'真的没差异'——只能说不显著。"

这就是**原假设优先原则**：H0 通常是"保守的默认状态"，只有强有力的证据才能拒绝它。

```python
# 示例：将研究问题转化为 H0/H1
research_questions = [
    "钻石用户的消费是否高于普通用户？",
    "新功能是否提升了用户活跃度？",
    "广告点击率是否高于 5%？"
]

hypotheses = []

for i, question in enumerate(research_questions, 1):
    if "钻石用户" in question:
        h0 = "μ_diamond = μ_normal（两组均值相等）"
        h1 = "μ_diamond > μ_normal（钻石用户均值更高，单尾检验）"
    elif "活跃度" in question:
        h0 = "μ_experiment = μ_control（两组均值相等）"
        h1 = "μ_experiment > μ_control（实验组均值更高，单尾检验）"
    elif "点击率" in question:
        h0 = "p = 0.05（点击率等于 5%）"
        h1 = "p > 0.05（点击率高于 5%，单尾检验）"

    hypotheses.append({"问题": question, "H0": h0, "H1": h1})

# 打印假设列表
import pprint
pprint.pprint(hypotheses)
```

### 单尾 vs 双尾检验：你关心方向吗？

在设定 H1 时，你需要决定是**单尾检验**（one-tailed）还是**双尾检验**（two-tailed）：

| 场景 | H0 | H1（单尾） | H1（双尾） |
|------|----|-----------|-----------|
| 新功能是否提升活跃度 | μ_exp = μ_ctrl | μ_exp > μ_ctrl | μ_exp ≠ μ_ctrl |
| 两种教学方法是否有差异 | μ_A = μ_B | μ_A > μ_B | μ_A ≠ μ_B |

小北嘀咕了一句："**那我是不是应该总用单尾检验，因为更容易显著？**"

"恰恰相反。"老潘摇头，"如果你用单尾检验但结果与预期方向相反（比如新功能反而降低了活跃度），你无法检测到这种差异。**工程上要谨慎：除非你有强理由预判方向，否则用双尾检验。**"

阿码追问："那如果我用了双尾检验，结果 p=0.04，但我之前预测的是'提升'，我能不能说单尾的 p 就是 0.02？"

老潘笑了："这个问题问得好。答案是：**不能事后改方向**。你必须在做检验之前决定单尾还是双尾，否则就是在 p-hacking——这是后面会重点讲的陷阱。"

你可能会想：Week 02 用箱线图比较两组分布时，不是挺直观的吗？为什么要搞得这么复杂？问题是，**箱线图能让你"看见"差异，但无法告诉你这个差异是否可能只是随机噪音**。假设检验就是要回答这个问题：这个差异是"真的信号"，还是"运气造成的假象"？

老潘补充道："Week 02 你学过**离散程度**——方差、标准差告诉数据'有多散'。但 Week 02 的离散程度是'描述性的'，而本周你要用离散程度做'推断性的决策'：标准误（standard error）就是标准差除以根号 n——它告诉你'抽样分布有多散'，进而决定 t 统计量的大小。所以本周的假设检验，本质上是在问：**观察到的均值差异，相对于抽样分布的离散程度来说，是'异常的大'还是'正常的大'？**"

---

## 2. p 值到底是什么？——"在 H0 为真时看到当前数据的概率"

小北做了一次 t 检验，得到 p=0.03。他激动地写下结论："**原假设为真的概率是 3%。**"

老潘看了，直接划掉："**错。p 值不是 P(H0|data)，而是 P(data|H0)。**"

"有什么区别？"

"区别很大。"老潘说，"前者是'结论为真的概率'——这是贝叶斯统计的后验概率；后者是'在 H0 为真时，看到当前或更极端数据的概率'——这是频率学派的 p 值。"

小北瞪大眼睛："**所以 p=0.03 不是说'H0 只有 3% 可能是对的'？**"

"对。"老潘点头，"它说的是：**如果 H0 是真的（比如两组均值真的相等），你有 3% 的概率观察到当前差异或更大的差异**。这个概率越小，说明当前数据与 H0 越不相容——所以你倾向于拒绝 H0。**"

### 用模拟建立 p 值的直觉

光说抽象，我们用模拟来"看见"p 值：

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 模拟场景：对照组 vs 实验组
np.random.seed(42)
control = np.random.normal(loc=100, scale=15, size=50)      # 对照组：均值=100
treatment = np.random.normal(loc=108, scale=15, size=50)    # 实验组：均值=108

# 观察到的差异
observed_diff = np.mean(treatment) - np.mean(control)
print(f"观察到的均值差异：{observed_diff:.2f}")

# 方法 1：使用 scipy 进行 t 检验
t_stat, p_value_two_tailed = stats.ttest_ind(treatment, control)
print(f"t 统计量：{t_stat:.3f}")
print(f"双尾 p 值：{p_value_two_tailed:.4f}")

# 方法 2：用模拟理解 p 值（置换检验的思想）
# 在 H0 为真时（两组均值相等），随机打乱标签，观察差异分布
n_simulations = 10000
combined = np.concatenate([control, treatment])
n_ctrl, n_treat = len(control), len(treatment)

simulated_diffs = []
for _ in range(n_simulations):
    # 随机打乱数据
    shuffled = np.random.permutation(combined)
    # 分配到两组
    sim_ctrl = shuffled[:n_ctrl]
    sim_treat = shuffled[n_ctrl:]
    # 计算差异
    simulated_diffs.append(np.mean(sim_treat) - np.mean(sim_ctrl))

simulated_diffs = np.array(simulated_diffs)

# 计算 p 值：在 H0 下，观察到当前或更极端差异的概率
# 双尾检验：取绝对值比较
p_value_simulation = (np.abs(simulated_diffs) >= np.abs(observed_diff)).mean()

print(f"模拟得到的双尾 p 值：{p_value_simulation:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.hist(simulated_diffs, bins=50, density=True, alpha=0.7, label='H0 下的差异分布')
plt.axvline(observed_diff, color='r', linestyle='--', linewidth=2, label=f'观察差异={observed_diff:.2f}')
plt.axvline(-observed_diff, color='r', linestyle='--', linewidth=2, alpha=0.7)

# 标记极端区域
extreme_right = simulated_diffs >= observed_diff
extreme_left = simulated_diffs <= -observed_diff
plt.hist(simulated_diffs[extreme_right], bins=50, density=True, color='red', alpha=0.5)
plt.hist(simulated_diffs[extreme_left], bins=50, density=True, color='red', alpha=0.5)

plt.xlabel('均值差异')
plt.ylabel('密度')
plt.title('p 值的直观理解：在 H0 下观察到当前或更极端数据的概率')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('p_value_intuition.png', dpi=150)
plt.show()
```

运行这个代码，你会看到：
- **红色区域**代表在 H0 为真时，出现"当前差异或更大"的概率——这就是 p 值
- **红色区域越小（p 越小）**，说明当前数据与 H0 越不相容，你越有理由拒绝 H0

老潘指着图说："**记住：p 值回答的不是'H0 是真的概率'，而是'如果 H0 是真的，看到这种数据的概率有多小'。** 这个区别很微妙，但很重要。"

### 三个常见误区

阿码举手问："**那 p=0.04 和 p=0.06 有本质区别吗？**"

"从数学上没有。"老潘说，"但从 convention 上有——因为 0.05 是个阈值。**但这会导致一个荒谬的结果：p=0.049 就'显著'，p=0.051 就'不显著'，两者实际差异微乎其微。**"

这是 p 值的第一个误区：**把 0.05 当成魔法线**。

第二个误区：**p 值越小，效应越大**。实际上 p 值受样本量影响很大——大样本下，微小效应也能达到 p<0.01；小样本下，大效应也可能不显著。

第三个误区：**p>0.05 就证明 H0 是对的**。不显著只能说明"证据不足"，不能证明"没有差异"。

Week 05 你学过 Bootstrap 置信区间——**p 值和置信区间是对偶关系**：如果 95% CI 不包含 0（对于均值差异），对应的 p 值就会小于 0.05。所以老潘常说："**不要只报告 p 值，要同时给出置信区间和效应量**。"

小北突然想到了什么："**Week 05 学的条件概率——P(A|B) 表示'在 B 发生的条件下 A 发生的概率'——是不是和 p 值很像？**"

"非常敏锐的观察！"老潘点头，"p 值本质上就是条件概率：**P(当前或更极端数据 | H0 为真)**——在'原假设为真'这个条件下，你看到当前数据的概率。但 Week 05 的条件概率可以用来计算'后验概率'（贝叶斯），而频率学派的 p 值不会告诉你'H0 为真的概率'——这点一定要分清楚。"

小北若有所思："**所以 p 值只是一个指标，不是全部？**"

"对。"老潘点头，"它告诉你'数据与 H0 的不相容程度'，但不告诉你'效应有多大'、'实际意义如何'、'H0 是对的概率'。这些都需要你额外判断。"

"哦！"小北突然想到了什么，"所以上周用 Bootstrap 得到的 95% CI [2900, 4100]，因为不包含 0，所以对应的 p 值会小于 0.05？"

"没错。"老潘笑了，"你开始建立直觉了。"

---

> **AI 时代小专栏：p-hacking 与"可复现性危机"——AI 时代的统计陷阱**
>
> 2010 年代爆发的"可复现性危机"（Reproducibility Crisis）揭示了科研界的一个系统性问题：**大量显著结果（p < 0.05）无法在重复实验中复现**。其中一个核心原因是 p-hacking——通过多次检验、选择性报告、数据切片等手段"碰"出显著结果。
>
> 2025 年 5 月，Nature 发表的评论"P hacking — Five ways it could happen to you"将 p-hacking 定义为**通过调整分析或数据来获得统计显著性结果的做法**，并列举了五种常见方式：继续收集数据直到显著、只报告显著的子组分析、尝试多种统计方法并只报告显著的那一种、不排除异常值以获得 p<0.05、测量多个变量但只报告显著的。
>
> 更危险的是：**AI 生成的报告往往只呈现"成功的"检验结果**，而省略了"不显著"的尝试——这会导致严重的发表偏差。Royal Society 的研究"Big little lies: a compendium and simulation of p-hacking strategies"（2023 年）通过模拟发现，这些策略会系统性放大假阳性率，而研究者往往意识不到自己在 p-hacking。
>
> 有趣的是，2025 年 12 月 Clearer Thinking 发布的复现研究"Three Surprises From Attempting To Replicate Recent Studies in Top Psychology Journals"发现了一个意外结果：**"复现率高于专家预期，p-hacking 没有我们预期的那么常见！"** 这提示我们：可复现性问题不只有 p-hacking 一个原因，还涉及理论构建、测量误差、样本异质性等多重因素。
>
> AI 时代的新挑战是：**AI 可以自动尝试多种检验方法、多个子组、多个指标，直到找到"显著"的结果**。但 AI 不会主动告诉你"我跑了 20 次检验，这是第 18 次"，也不会提示你"这个差异在统计上显著，但效应量极小（Cohen's d = 0.08），实际意义有限"。
>
> 所以本周你要学的，不是"怎么用 AI 快速跑检验"，而是**怎么审查 AI 生成的统计报告**：检查它有没有说明做了多少次检验、有没有校正 p 值、有没有报告效应量和置信区间。这些细节，才是区分"玩具分析"和"可靠推断"的关键。
>
> 参考（访问日期：2026-02-12）：
> - [Nature: P hacking — Five ways it could happen to you (2025-05-08)](https://www.nature.com/articles/d41586-025-01246-1)
> - [Royal Society: Big little lies: a compendium and simulation of p-hacking strategies (2023)](https://royalsocietypublishing.org/rsos/article/10.1098/rsos.220346)
> - [Replication Index: P-Hacking Preregistered Studies Can Be Detected (2026-01-18)](https://replicationindex.com/2026/01/18/p-hacking-preregistered-insignificant-statistical-significance/)
> - [Clearer Thinking: Three Surprises From Attempting To Replicate Recent Studies (2025-12-04)](https://replications.clearerthinking.org/three-surprises-from-attempting-to-replicate-recent-studies-in-top-psychology-journals/)

---

## 3. t 检验实战——从差异到决策

小北现在有了 H0/H1 和 p 值的直觉，老潘说："**我们来实际跑一次 t 检验。**" 但在动手之前，你要先问三个问题：

1. **数据是什么类型？**（连续 vs 离散）
2. **组之间是独立的还是配对的？**（独立样本 vs 配对样本）
3. **前提假设满足吗？**（正态性、方差齐性）

"**t 检验不是万能的**。"老潘强调，"每个检验都有前提，忽略前提会得到不可靠的结论。"

阿码举手问："**为什么不能对所有数据都用 t 检验？**"

"好问题。"老潘说，"t 检验有三个前提：数据来自正态分布、方差齐性（两组方差相等）、样本独立。如果数据严重偏态或方差不齐，你得用其他方法——非参数检验或 Welch's t 检验。"

Week 02 你画过很多直方图和箱线图——那不只是为了"好看"，而是为了帮你判断这些前提是否满足。如果数据严重偏态（比如收入数据通常右偏），t 检验可能不合适，你应该考虑用秩检验（Mann-Whitney U test）或者对数据做变换。

老潘补充道："Week 05 你学过**常见分布**——正态分布、指数分布、泊松分布各有特点。**t 检验假设数据来自正态分布**，如果你的数据明显偏态（比如指数分布），t 检验的结果可能不可靠。这时候你得用 Week 05 学的分布知识来判断：这数据'看起来'像正态吗？如果不像，要么做变换，要么用非参数方法。"

### 三种 t 检验，用对了吗？

| 检验类型 | 场景 | Python 函数 |
|---------|------|------------|
| 单样本 t 检验 | 检验样本均值是否等于某个值 | `scipy.stats.ttest_1samp` |
| 独立样本 t 检验 | 检验两组独立样本的均值差异 | `scipy.stats.ttest_ind` |
| 配对样本 t 检验 | 检验同一组对象的前后差异 | `scipy.stats.ttest_rel` |

本周的贯穿案例（新功能 A/B 测试）是**独立样本 t 检验**场景。下面是一个完整的实战例子：

```python
# examples/02_ttest_example.py
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 加载数据（模拟 A/B 测试）
np.random.seed(42)
control = np.random.normal(loc=100, scale=15, size=500)       # 对照组
treatment = np.random.normal(loc=105, scale=15, size=500)     # 实验组

# 创建 DataFrame
df = pd.DataFrame({
    'group': ['control'] * 500 + ['treatment'] * 500,
    'activity': np.concatenate([control, treatment])
})

print(f"对照组均值：{np.mean(control):.2f}")
print(f"实验组均值：{np.mean(treatment):.2f}")
print(f"观察差异：{np.mean(treatment) - np.mean(control):.2f}")

# ========== 前提假设检查 ==========
print("\n=== 前提假设检查 ===")

# 1. 正态性检验（Shapiro-Wilk 检验）
# H0: 数据来自正态分布
_, p_norm_ctrl = stats.shapiro(control[:5000]) if len(control) > 5000 else stats.shapiro(control)
_, p_norm_treat = stats.shapiro(treatment[:5000]) if len(treatment) > 5000 else stats.shapiro(treatment)
print(f"对照组正态性检验 p 值：{p_norm_ctrl:.4f}")
print(f"实验组正态性检验 p 值：{p_norm_treat:.4f}")
print(f"结论：{'✓ 正态性假设满足' if p_norm_ctrl > 0.05 and p_norm_treat > 0.05 else '✗ 数据可能偏离正态（考虑非参数检验）'}")

# 2. 方差齐性检验（Levene 检验）
# H0: 两组方差相等
_, p_levene = stats.levene(control, treatment)
print(f"\n方差齐性检验 p 值：{p_levene:.4f}")
print(f"结论：{'✓ 方差齐性假设满足' if p_levene > 0.05 else '✗ 方差不齐（使用 Welch\'s t 检验）'}")

# ========== t 检验 ==========
print("\n=== t 检验 ===")

# 根据方差齐性结果选择检验方法
equal_var = p_levene > 0.05
t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=equal_var)

print(f"t 统计量：{t_stat:.4f}")
print(f"p 值（双尾）：{p_value:.6f}")
print(f"显著性水平 α=0.05：{'拒绝 H0（差异显著）' if p_value < 0.05 else '无法拒绝 H0（差异不显著）'}")

# ========== 效应量（Cohen's d）==========
def cohens_d(group1, group2):
    """计算 Cohen's d 效应量。"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    # 合并标准差（pooled standard deviation）
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return d

effect_size = cohens_d(treatment, control)
print(f"\nCohen's d 效应量：{effect_size:.3f}")

# 解释效应量（Cohen's 经验标准）
if abs(effect_size) < 0.2:
    interpretation = "效应量很小（small）"
elif abs(effect_size) < 0.5:
    interpretation = "效应量中等（medium）"
elif abs(effect_size) < 0.8:
    interpretation = "效应量较大（large）"
else:
    interpretation = "效应量非常大（very large）"

print(f"解释：{interpretation}")

# ========== 置信区间 ==========
# 95% 置信区间 for 均值差异
mean_diff = np.mean(treatment) - np.mean(control)
se_diff = np.sqrt(np.var(treatment, ddof=1)/len(treatment) + np.var(control, ddof=1)/len(control))
ci_low = mean_diff - 1.96 * se_diff
ci_high = mean_diff + 1.96 * se_diff

print(f"\n均值差异的 95% CI：[{ci_low:.2f}, {ci_high:.2f}]")
print(f"结论：{'CI 不包含 0，差异显著' if ci_low > 0 or ci_high < 0 else 'CI 包含 0，差异不显著'}")

# ========== 可视化 ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：两组分布对比
axes[0].hist(control, bins=30, alpha=0.7, label='对照组', density=True)
axes[0].hist(treatment, bins=30, alpha=0.7, label='实验组', density=True)
axes[0].axvline(np.mean(control), color='blue', linestyle='--', linewidth=2)
axes[0].axvline(np.mean(treatment), color='orange', linestyle='--', linewidth=2)
axes[0].set_xlabel('活跃度')
axes[0].set_ylabel('密度')
axes[0].set_title('两组分布对比')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 右图：差异的置信区间
axes[1].errorbar(0, mean_diff, yerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
                 fmt='o', capsize=10, capthick=2, linewidth=2)
axes[1].axhline(0, color='red', linestyle='--', alpha=0.7, label='零差异线')
axes[1].set_xlim(-0.5, 0.5)
axes[1].set_ylim(ci_low - 2, ci_high + 2)
axes[1].set_xticks([])
axes[1].set_ylabel('均值差异')
axes[1].set_title(f'均值差异的 95% CI\nd={effect_size:.3f}, p={p_value:.6f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('ttest_results.png', dpi=150)
plt.show()
```

运行这个代码，你会得到一份完整的 t 检验报告：
- **前提假设检查**：正态性、方差齐性
- **t 统计量和 p 值**：判断是否拒绝 H0
- **效应量（Cohen's d）**：评估实际意义
- **置信区间**：量化不确定性
- **可视化**：两组分布对比 + 差异的 CI

阿码盯着输出："**所以 p<0.05 只是第一步，还要看效应量？**"

"对。"老潘说，"**p 值告诉你'差异是否可能随机产生'，效应量告诉你'差异有多大'**。两者结合，才能做出可靠的决策。"

"哦！"阿码突然想到了什么，"所以如果我有 100 万个样本，可能 p<0.001，但 Cohen's d 只有 0.05——意思是'统计上显著，但实际意义极小'？"

"没错！"老潘赞许地点头，"你抓住了重点。**统计显著 ≠ 实际显著**。这就是为什么我们强调要同时看 p 值和效应量。"

Week 03 你学过缺失值处理——删除缺失值会损失样本量。本周你看到了样本量的另一面：**样本量影响功效**。如果你删除了太多数据，可能因为功效不足而无法检测出真实差异。这就是 Week 03 和本周的连接：数据清洗不是"越干净越好"，而是在"数据质量"和"统计功效"之间找平衡。

---

> **AI 时代小专栏：AI 能自动选择统计检验吗？——工具 vs 判断**
>
> 当你问 ChatGPT 或 Claude"我的数据该用什么检验"时，它们通常能给出不错的建议：连续数据用 t 检验、分类数据用卡方检验、非正态数据用秩和检验。但**AI 不会自动替你检查前提假设**——它不会读取你的数据、画正态 QQ 图、跑 Levene 检验，它只能基于你给的描述做"最佳猜测"。
>
> 2026 年 2 月，美国统计学会（ASA）旗下的 *Amstat News* 发表的文章"Statistics in the Age of AI"指出：**"选择压力偏向于问题表述而非计算，假设评估优于模型执行"**。这意味着在 AI 驱动的统计分析中，人类的价值正在从"会跑检验"转向"会评估假设"。
>
> Medium 2026 年 1 月的文章"Beyond Hype: What Actually Changed in AI in 2026"也提到：**AI 增强数据分析工作流程，通过发现统计错误、提出数据质量问题并建议不同的分析方法**——但这不等于 AI 能自动完成所有工作，前提假设检查仍然需要人类判断。
>
> 更危险的是：**AI 倾向于推荐"常用"的检验（如 t 检验、ANOVA），而忽略了前提假设的严重违反**（比如严重偏态的数据应该用非参数检验）。Patronus AI 2025 年 4 月的文章"Modeling Statistical Risk in AI Products"讨论了一个新的统计风险模型，量化 AI 错误如何影响关键业务指标——这提示我们，AI 在统计方法选择上的错误会直接导致业务损失。
>
> 所以本周你要学的，不是"让 AI 替你选检验"，而是**建立一套自己的检查清单**：
> - 数据是什么类型？
> - 组之间是独立的还是配对的？
> - 正态性假设满足吗？（用 Shapiro-Wilk 或 QQ 图）
> - 方差齐性假设满足吗？（用 Levene 检验）
> - 样本量够吗？（功效分析）
>
> AI 可以帮你写代码、画图、计算 p 值，但**前提假设检查和最终决策，必须由你来做**。
>
> 参考（访问日期：2026-02-12）：
> - [Amstat News: Statistics in the Age of AI (2026-02-02)](https://magazine.amstat.org/blog/2026/02/02/statistics-in-the-age-of-ai/)
> - [Medium: Beyond Hype - What Actually Changed in AI in 2026 (2026-01-31)](https://medium.com/beyond-the-hype-what-actually-changed-in-ai-in-2026-976fbb09f1080)
> - [Patronus AI: Modeling Statistical Risk in AI Products (2025-04-09)](https://www.patronus.ai/blog/modeling-statistical-risk-in-ai-products)

---

## 4. 你愿意承担哪种错误？——第一类/第二类错误与功效

小北完成了 t 检验，得到 p=0.04，拒绝 H0。老潘看完报告，只问了一句："**如果 H0 实际上是对的（两组均值真的相等），你拒绝它的概率是多少？**"

小北愣住了："呃……4%？"

"对。"老潘点头，"这就是**第一类错误（Type I Error）**，也叫假阳性。你在 5% 的显著性水平下做检验，长期来看会有 5% 的概率把'没有差异'误判为'有差异'。"

"那如果我降低 α 到 0.01，是不是就更保险了？"小北追问。

"更保险？不。"老潘说，"**降低 α 会减少第一类错误，但会增加第二类错误（Type II Error）**——也就是假阴性：明明有差异，你却没检测出来。这两个错误是跷跷板，你压缩一头，另一头就会翘起来。"

阿码突然笑了："所以这就是统计学的'薛定谔的猫'——你永远无法同时把两种错误都降到零？"

"比喻得不错。"老潘也笑了，"但工程上你不需要'零'，只需要'可接受'。关键是要知道**你在承担哪种风险，以及代价有多大**。"

### 错误矩阵：四种可能的结果

假设检验的决策有四种可能的结果：

| 真实情况 | 决策：拒绝 H0 | 决策：保留 H0 |
|---------|--------------|--------------|
| **H0 为真**（无差异） | **第一类错误（Type I）**<br>假阳性，概率=α | **正确决策**<br>真阴性，概率=1-α |
| **H0 为假**（有差异） | **正确决策**<br>真阳性，概率=1-β | **第二类错误（Type II）**<br>假阴性，概率=β |

- **α（显著性水平）**：第一类错误的概率，通常设为 0.05
- **β**：第二类错误的概率
- **功效（1-β）**：正确拒绝错误 H0 的概率（检测出真实差异的能力）

老潘指着矩阵说："**工程上你要同时权衡 α 和 β**：在某些场景（如医疗诊断），假阳性（误诊健康人有病）代价很高；在另一些场景（如质量检测），假阴性（漏掉有缺陷的产品）代价更高。"

阿码举了个例子："如果我在做 A/B 测试，假阳性意味着'上线一个没用的功能'，假阴性意味着'错过一个好功能'——哪个代价更大？"

"看情况。"老潘说，"**如果功能开发成本高，假阴性代价更大**（错过好功能太可惜）；**如果上线有风险（可能影响现有体验），假阳性代价更大**（上线烂功能破坏用户体验）。关键是要意识到：**你在做权衡，不是在追求'绝对正确'**。"

### 用模拟理解两类错误

让我们用一个模拟实验来直观看到两类错误的关系：

```python
# examples/03_type12_errors.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def simulate_type_errors(n_sim=10000, n_sample=50, true_diff=0, alpha=0.05, seed=42):
    """
    模拟两类错误。

    参数：
    - n_sim: 模拟次数
    - n_sample: 每组样本量
    - true_diff: 真实均值差异（H0 为真时=0，H0 为假时≠0）
    - alpha: 显著性水平
    - seed: 随机种子

    返回：第一类错误率、第二类错误率
    """
    np.random.seed(seed)

    type_i_errors = 0  # 第一类错误计数
    type_ii_errors = 0  # 第二类错误计数

    for _ in range(n_sim):
        # 生成数据（对照组均值=100，实验组均值=100+true_diff）
        control = np.random.normal(loc=100, scale=15, size=n_sample)
        treatment = np.random.normal(loc=100 + true_diff, scale=15, size=n_sample)

        # t 检验
        _, p_value = stats.ttest_ind(treatment, control)

        # 决策
        if p_value < alpha:
            # 拒绝 H0
            if true_diff == 0:
                # H0 为真，但拒绝了 → 第一类错误
                type_i_errors += 1
        else:
            # 保留 H0
            if true_diff != 0:
                # H0 为假，但保留了 → 第二类错误
                type_ii_errors += 1

    # 计算错误率
    type_i_rate = type_i_errors / n_sim if true_diff == 0 else 0
    type_ii_rate = type_ii_errors / n_sim if true_diff != 0 else 0
    power = 1 - type_ii_rate if true_diff != 0 else np.nan

    return type_i_rate, type_ii_rate, power

# ========== 实验 1：第一类错误率（H0 为真）==========
print("=== 实验 1：第一类错误率（H0 为真）===")
type_i_rates = []
for alpha in [0.01, 0.05, 0.10]:
    type_i_rate, _, _ = simulate_type_errors(true_diff=0, alpha=alpha)
    type_i_rates.append(type_i_rate)
    print(f"α={alpha:.2f}：第一类错误率 = {type_i_rate:.3f}")

# ========== 实验 2：功效分析（H0 为假）==========
print("\n=== 实验 2：功效分析（H0 为假，真实差异=8）===")
power_results = []
sample_sizes = [20, 50, 100, 200]

for n in sample_sizes:
    _, _, power = simulate_type_errors(n_sample=n, true_diff=8, alpha=0.05)
    power_results.append(power)
    print(f"样本量 n={n:3d}：功效 = {power:.3f}")

# 可视化功效曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：第一类错误率 vs α
axes[0].plot([0.01, 0.05, 0.10], type_i_rates, 'o-', linewidth=2, markersize=8)
axes[0].plot([0, 0.12], [0, 0.12], 'r--', alpha=0.5, label='理论线（α）')
axes[0].set_xlabel('显著性水平 α')
axes[0].set_ylabel('第一类错误率')
axes[0].set_title('第一类错误率 = α')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 0.12)
axes[0].set_ylim(0, 0.12)

# 右图：功效 vs 样本量
axes[1].plot(sample_sizes, power_results, 'o-', linewidth=2, markersize=8, color='green')
axes[1].axhline(0.8, color='red', linestyle='--', alpha=0.7, label='推荐功效 ≥ 80%')
axes[1].set_xlabel('样本量')
axes[1].set_ylabel('功效（1-β）')
axes[1].set_title('功效随样本量增加而提升')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('type12_errors_and_power.png', dpi=150)
plt.show()

# ========== 实验 3：权衡两类错误（固定样本量）==========
print("\n=== 实验 3：α 与 β 的权衡（样本量=50，真实差异=8）===")
alphas = np.linspace(0.01, 0.10, 10)
powers = []

for alpha in alphas:
    _, _, power = simulate_type_errors(n_sample=50, true_diff=8, alpha=alpha)
    powers.append(power)

plt.figure(figsize=(10, 6))
plt.plot(alphas, powers, 'o-', linewidth=2, markersize=8)
plt.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='推荐功效 ≥ 80%')
plt.axvline(0.05, color='orange', linestyle='--', alpha=0.7, label='常用 α=0.05')
plt.xlabel('第一类错误率 α')
plt.ylabel('功效（1-β）')
plt.title('α 与 β 的权衡关系（跷跷板效应）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('alpha_beta_tradeoff.png', dpi=150)
plt.show()
```

运行这个实验，你会看到三个关键结论：

1. **第一类错误率 = α**：当 H0 为真时，拒绝 H0 的概率恰好等于显著性水平
2. **功效随样本量增加而提升**：样本量越大，检测出真实差异的能力越强
3. **α 与 β 是跷跷板**：降低 α 会提高 β（降低功效），反之亦然

阿码盯着第三个图："**所以如果我把 α 从 0.05 降到 0.01，功效会从 0.6 掉到 0.3？**"

"对。"老潘说，"**这就是为什么不能盲目追求'超严格'的 α 水平**。在样本量有限的情况下，过低的 α 会导致你漏掉很多真实差异。"

### 功效分析：样本量够吗？

老潘指着第二个图（功效 vs 样本量）："**工程上我们通常希望功效 ≥ 80%**——这意味着如果真实差异存在，你有至少 80% 的概率检测出来。"

"怎么计算需要的样本量？"小北问。

"用功效分析。"老潘说，"**你需要指定四个参数：效应量、α、目标功效、标准差**，然后反推样本量。"

```python
from scipy import stats

def calculate_sample_size(effect_size, alpha=0.05, power=0.8, ratio=1):
    """
    计算独立样本 t 检验的所需样本量。

    参数：
    - effect_size: Cohen's d（标准化效应量）
    - alpha: 显著性水平（默认 0.05）
    - power: 目标功效（默认 0.8）
    - ratio: 两组样本量比例（默认 1:1）

    返回：每组需要的样本量
    """
    # Z 值
    z_alpha = stats.norm.ppf(1 - alpha/2)  # 双尾检验
    z_beta = stats.norm.ppf(power)

    # 样本量公式（简化版）
    n_per_group = 2 * ((z_alpha + z_beta)**2) / (effect_size**2)

    return int(np.ceil(n_per_group))

# 示例：检测中等效应（Cohen's d = 0.5）
n_needed = calculate_sample_size(effect_size=0.5, alpha=0.05, power=0.8)
print(f"检测中等效应（d=0.5），每组需要 {n_needed} 个样本")
print(f"总样本量：{n_needed * 2}")

# 不同效应量下的样本量需求
print("\n不同效应量下的样本量需求（α=0.05, 功效=0.8）：")
for d in [0.2, 0.5, 0.8]:
    n = calculate_sample_size(effect_size=d)
    print(f"  d={d:.1f}（{'小' if d<0.3 else '中' if d<0.7 else '大'}效应）：每组 {n} 个样本")
```

输出会显示：
- 小效应（d=0.2）：每组需要约 394 个样本
- 中等效应（d=0.5）：每组需要约 64 个样本
- 大效应（d=0.8）：每组需要约 26 个样本

老潘说："**这就是为什么功效分析很重要**。如果你只有 50 个样本，却试图检测小效应（d=0.2），功效会很低——即使真实差异存在，你也检测不出来。"

小北若有所思："所以……如果我为了'清洗数据'删除了太多样本，可能导致功效不足？"

"没错！"老潘点头，"Week 03 你学过缺失值处理，现在你看到了它的另一面：**数据清洗不是'越干净越好'，而是在'数据质量'和'统计功效'之间找平衡**。如果删除缺失值导致样本量大幅下降，你可能需要考虑其他策略（比如插补）或者接受功效不足的限制。"

阿码举手："**等等，Week 05 的'抽样分布'——我有点混淆。功效分析里的'抽样分布'和 Week 05 学的 Bootstrap 抽样分布是一回事吗？**"

"很接近！"老潘说，"Week 05 你用 Bootstrap 看到了：**如果重复抽样，统计量（比如均值）会形成一个分布**——这就是抽样分布。功效分析本质上是在问：**在这个抽样分布下，如果真实差异存在，我有多少概率检测到它？** 所以本周的'功效分析'，就是 Week 05 '抽样分布与模拟'的直接应用：你不是模拟一次抽样，而是模拟上千次，看看'拒绝 H0'的频率有多高。"

---

## 5. AI 生成的统计报告能信吗？——审查训练

老潘把一份 AI 生成的分析报告放在小北面前："**你来审计一下，找出其中的问题。**"

小北盯着报告看了半天，最后说："**看起来……没问题？**"

老潘摇头，指着第一行："**'t 检验显示 p=0.03，因此新功能显著提升了用户活跃度'——这句话有三个问题**。"

1. **没说明 H0/H1 是什么**："显著"是说"拒绝 H0"，但 H0 是什么没说清楚
2. **没报告效应量**：p=0.03 不能告诉你"提升了多少"
3. **没检查前提假设**：正态性、方差齐性都没验证

"**AI 可以快速生成报告，但你不能直接照收**。"老潘强调，"本周最重要的技能不是'会跑 t 检验'，而是'能审查一份检验报告'。"

### AI 生成报告的常见问题

基于 2024-2026 年的研究，AI 生成的统计分析报告常见问题包括：

| 问题类型 | 表现 | 风险 |
|---------|------|------|
| **p 值误解释** | "p=0.03 说明 H0 为真的概率是 3%" | 逻辑错误，误导决策 |
| **缺少效应量** | 只报告 p 值，不报告 Cohen's d 或 CI | 无法判断实际意义 |
| **前提假设未验证** | 直接跑 t 检验，不检查正态性/方差齐性 | 结论不可靠 |
| **多重比较未校正** | 跑 10 次检验，报告 p<0.05 的结果 | 假阳性风险放大 |
| **p-hacking 痕迹** | "我们尝试了多种分组方式，发现……" | 选择性报告，可复现性差 |
| **样本量不足** | 小样本检测小效应，报告 p>0.05 | 功效不足，假阴性风险高 |

阿码看着这个表格："所以 AI 生成的报告……大部分时候都不能直接用？"

"差不多。"老潘说，"**AI 的价值在于快速生成'第一版'，但审查和修正必须由人来做**。这就是本周你学的核心技能：统计审查能力。"

### 审查清单：一份模板

老潘给小北一份**AI 统计报告审查清单**：

```python
# examples/04_ai_report_checklist.py
def review_statistical_report(report_text):
    """
    审查 AI 生成的统计检验报告，标注潜在问题。

    参数：
    - report_text: AI 报告文本

    返回：审查结果（问题列表 + 改进建议）
    """
    issues = []

    # ========== 检查 1：H0/H1 是否明确 ==========
    if "H0" not in report_text and "原假设" not in report_text:
        issues.append({
            "问题": "未明确说明原假设 H0",
            "风险": "读者不清楚'显著'是拒绝什么假设",
            "建议": "补充 H0/H1 的正式陈述"
        })

    # ========== 检查 2：p 值解释是否正确 ==========
    if "H0 为真的概率" in report_text or "结论为真的概率" in report_text:
        issues.append({
            "问题": "p 值误解释",
            "风险": "严重逻辑错误（p ≠ P(H0|data）",
            "建议": "正确解释：在 H0 为真时，看到当前数据的概率"
        })

    # ========== 检查 3：效应量是否报告 ==========
    if "Cohen" not in report_text and "效应量" not in report_text:
        issues.append({
            "问题": "缺少效应量",
            "风险": "无法判断实际意义（p<0.05 可能是微小效应）",
            "建议": "补充 Cohen's d 或 η²"
        })

    # ========== 检查 4：置信区间是否报告 ==========
    if "CI" not in report_text and "置信区间" not in report_text:
        issues.append({
            "问题": "缺少置信区间",
            "风险": "无法量化不确定性",
            "建议": "补充 95% CI"
        })

    # ========== 检查 5：前提假设是否验证 ==========
    if "正态性" not in report_text and "Shapiro" not in report_text:
        issues.append({
            "问题": "未验证正态性假设",
            "风险": "数据严重偏态时 t 检验不可靠",
            "建议": "补充 Shapiro-Wilk 检验或 QQ 图"
        })

    if "方差齐性" not in report_text and "Levene" not in report_text:
        issues.append({
            "问题": "未验证方差齐性假设",
            "风险": "方差不齐时应使用 Welch's t 检验",
            "建议": "补充 Levene 检验"
        })

    # ========== 检查 6：多重比较是否校正 ==========
    if ("多次检验" in report_text or "多个指标" in report_text) and \
       ("校正" not in report_text and "Bonferroni" not in report_text):
        issues.append({
            "问题": "多重比较未校正",
            "风险": "假阳性风险放大（跑 20 次检验总会有 1 次碰巧显著）",
            "建议": "使用 Bonferroni 或 FDR 校正"
        })

    # ========== 检查 7：样本量/功效是否讨论 ==========
    if "功效" not in report_text and "样本量" not in report_text:
        issues.append({
            "问题": "未讨论样本量/功效",
            "风险": "小样本检测小效应时假阴性风险高",
            "建议": "补充功效分析或说明样本量限制"
        })

    # ========== 检查 8：相关 vs 因果 ==========
    if ("导致" in report_text or "因果" in report_text) and \
       ("实验" not in report_text and "随机" not in report_text):
        issues.append({
            "问题": "相关被误写成因果",
            "风险": "观察性研究无法确定因果方向",
            "建议": "用'相关'、'关联'而非'导致'、'因果'"
        })

    return issues

# ========== 示例：审查一份 AI 生成的报告 ==========
ai_report = """
统计检验报告：

我们对实验组和对照组进行了 t 检验，结果 t=2.15, p=0.03。

结论：
1. 新功能显著提升了用户活跃度（H0 为真的概率是 3%）。
2. 两组均值差异为 5.2 分。

建议：
- 上线新功能，因为效果显著。
"""

print("=== AI 报告审查 ===")
issues = review_statistical_report(ai_report)

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
统计检验报告（修订版）：

## 假设设定
- H0（原假设）：实验组与对照组的活跃度均值相等（μ_exp = μ_ctrl）
- H1（备择假设）：实验组活跃度高于对照组（μ_exp > μ_ctrl，单尾检验）

## 前提假设检查
- 正态性：Shapiro-Wilk 检验 p_ctrl=0.12, p_exp=0.08（> 0.05，正态性假设满足）
- 方差齐性：Levene 检验 p=0.21（> 0.05，方差齐性假设满足）

## 检验结果
- t 统计量：t(998) = 2.15
- p 值（单尾）：p = 0.016
- 均值差异：5.2 分，95% CI [1.8, 8.6]
- Cohen's d 效应量：d = 0.21（小效应）

## 解读与局限
- 统计显著性：在 α=0.05 水平下拒绝 H0，差异具有统计显著性
- 实际意义：效应量较小（d=0.21），需评估业务上是否值得上线
- 样本量：每组 n=500，功效≈0.68（对于小效应 d=0.21），存在假阴性风险
- 因果推断：本研究为 A/B 测试（随机分配），可支持因果结论

## 建议
- 统计上支持新功能有效，但效应较小，需结合成本-收益分析决定是否上线
- 如需更稳健的结论，可增加样本量以提高功效
"""

print("=== 修订版报告 ===")
print(revised_report)
```

运行这个审查工具，你会发现原始 AI 报告有至少 5 个问题：
1. 未说明 H0/H1
2. p 值误解释
3. 缺少效应量
4. 缺少置信区间
5. 未验证前提假设

老潘看着修订版，点头说："**这才是一份专业的检验报告**。不仅告诉读者'显著'，还告诉他们'在什么假设下显著'、'效应有多大'、'结论有多可靠'。"

小北若有所思："**所以 AI 是起点，不是终点？**"

"对。"老潘说，"AI 可以帮你快速生成初步分析，但**最终责任在你**：你要检查假设、审查逻辑、解释边界。这是人类不可替代的部分。"

Week 04 你学会了生成假设清单。本周你把这些假设升级成了完整的假设检验报告：H0/H1、p 值、效应量、置信区间、前提假设检查、功效分析。更重要的是，你学会了**审查 AI 生成的统计报告**——这是 AI 时代的关键技能。

---

## StatLab 进度

本周 StatLab 报告增加了"假设检验结果"章节。下面的示例代码展示了如何在报告中总结完整的检验结果：

```python
# examples/99_statlab.py
import pandas as pd
import numpy as np
from scipy import stats

def generate_hypothesis_test_section(df, report_path='report.md'):
    """在 report.md 中添加假设检验结果章节。"""

    # 假设你已经完成了：
    # 1. H0/H1 设定
    # 2. t 检验（含前提假设检查）
    # 3. 效应量计算（Cohen's d）
    # 4. 置信区间
    # 5. 功效分析

    # 这里用模拟数据演示，实际使用时从 df 读取
    np.random.seed(42)
    control = np.random.lognormal(6.5, 0.5, 200)       # 普通用户
    treatment = np.random.lognormal(7.0, 0.6, 50)      # 钻石用户

    # t 检验
    t_stat, p_value = stats.ttest_ind(treatment, control)

    # 效应量
    def cohens_d(g1, g2):
        n1, n2 = len(g1), len(g2)
        pooled_std = np.sqrt(((n1-1)*g1.var(ddof=1) + (n2-1)*g2.var(ddof=1)) / (n1+n2-2))
        return (g1.mean() - g2.mean()) / pooled_std

    effect_size = cohens_d(treatment, control)

    # 95% CI
    mean_diff = treatment.mean() - control.mean()
    se_diff = np.sqrt(treatment.var(ddof=1)/len(treatment) + control.var(ddof=1)/len(control))
    ci_low = mean_diff - 1.96 * se_diff
    ci_high = mean_diff + 1.96 * se_diff

    # 生成报告章节
    hypothesis_section = f"""

## 假设检验结果

> 本章对 Week 04 提出的可检验假设进行正式统计检验。
> 生成时间：2026-02-12

### H1：钻石用户与普通用户的消费差异

**假设设定**：
- H0（原假设）：钻石用户与普通用户的平均消费相等（μ_diamond = μ_normal）
- H1（备择假设）：钻石用户平均消费高于普通用户（μ_diamond > μ_normal，单尾检验）

**前提假设检查**：
- 正态性：Shapiro-Wilk 检验 p_diamond=0.XX, p_normal=0.XX（> 0.05，正态性假设满足）
- 方差齐性：Levene 检验 p=0.XX（> 0.05，方差齐性假设满足）
- 样本独立性：✓ 用户随机抽样，互不干扰

**检验结果**：
- t 统计量：t({len(treatment)+len(control)-2}) = {t_stat:.3f}
- p 值（单尾）：{p_value:.6f}
- 决策：{'拒绝 H0（差异显著）' if p_value < 0.05 else '无法拒绝 H0（差异不显著）'}

**效应量与置信区间**：
- 均值差异：{mean_diff:.0f} 元
- 95% 置信区间：[{ci_low:.0f}, {ci_high:.0f}] 元
- Cohen's d 效应量：{effect_size:.3f}（{'大效应' if abs(effect_size) >= 0.8 else '中等效应' if abs(effect_size) >= 0.5 else '小效应'}）

**解读**：
- 统计显著性：在 α=0.05 水平下，{'有充分证据拒绝 H0，认为钻石用户消费显著高于普通用户' if p_value < 0.05 else '证据不足，无法拒绝 H0'}
- 实际意义：效应量{'较大' if abs(effect_size) >= 0.8 else '中等' if abs(effect_size) >= 0.5 else '较小'}，需结合业务场景评估重要性
- 不确定性：95% CI 不包含 0，{'支持' if ci_low > 0 else '不支持'}差异为正的结论

**样本量与功效**：
- 当前样本量：钻石用户 n={len(treatment)}，普通用户 n={len(control)}
- 功效估算（基于当前效应量）：约 XX%（对于中等效应 d={abs(effect_size):.2f}）
- 局限：钻石用户样本量较小，如需检测更小效应需增加样本

### 其他假设（H2-H3）

| 假设 | 检验方法 | p 值 | 效应量 | 结论 |
|-------|---------|-------|--------|------|
| H2：不同城市用户活跃度有差异 | ANOVA | 0.XXX | η²=0.XX | {'显著' if p_value < 0.05 else '不显著'} |
| H3：收入与消费正相关 | Pearson 相关 | 0.XXX | r=0.XX | {'显著' if p_value < 0.05 else '不显著'} |

### 统计检验局限

- **多重比较风险**：本周进行了 3 次检验，存在假阳性放大的风险（Week 07 将讨论 Bonferroni 校正）
- **未控制混杂变量**：收入、年龄等变量可能混杂用户等级与消费的关系（Week 09 将用回归控制）
- **横截面数据**：无法确定因果方向（需要纵向数据或实验设计）

### 下一步

Week 07 将进行多组比较（ANOVA）和多重比较校正，进一步探索不同城市、不同年龄段之间的差异。

---

"""

    # 追加到报告
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(hypothesis_section)

    print(f"假设检验结果章节已追加到 {report_path}")

# 使用示例
if __name__ == "__main__":
    # generate_hypothesis_test_section(df, 'report.md')
    pass
```

现在你的 StatLab 报告有了六个层次：
1. **数据卡**：数据从哪来、字段什么意思
2. **描述统计**：数据长什么样、分布如何
3. **清洗日志**：数据怎么处理的、为什么这样处理
4. **EDA 叙事**：数据在说什么故事、还需要验证什么假设
5. **不确定性量化**：关键统计量有多稳定、哪些地方可能出错
6. **假设检验结果**：差异是否显著、效应有多大、结论有多可靠

老潘看到这份报告会说什么？"**这才是一份完整的统计分析报告。从数据到假设，从描述到推断，每一步都有迹可循、有据可依。**"

Week 05 你把"不确定性"写进了报告。本周你把"决策"写进了报告——不再是"数据是什么"，而是"数据告诉我们什么，以及我们有多确定"。这是从描述统计到统计推断的完整跨越。

---

## Git 本周要点

本周必会命令：
- `git status`：查看工作区状态
- `git diff`：查看具体改动内容
- `git add -A`：添加所有改动
- `git commit -m "feat: add hypothesis testing section"`
- `git log --oneline -n 5`

常见坑：
- 只保存 p 值不保存效应量：无法判断实际意义，建议同时报告 Cohen's d 和 CI
- t 检验不验证前提假设：结论可能不可靠，建议先做正态性和方差齐性检验
- 混淆第一类错误和第二类错误：决策逻辑混乱，建议用错误矩阵可视化

---

## 本周小结（供下周参考）

本周你做了五件事：学会了把研究问题转化为正式的 H0/H1；正确理解 p 值的含义，避免"p 值即真理概率"的误区；掌握了 t 检验的三种类型（单样本/独立样本/配对样本），并学会了前提假设检查（正态性、方差齐性）；理解了第一类错误（α）和第二类错误（β）的权衡，以及功效分析的重要性；学会了计算效应量（Cohen's d），区分统计显著与实际显著；掌握了 AI 生成统计报告的审查方法，能识别常见谬误（p-hacking、前提假设未验证、效应量缺失）。

更重要的是，你在 StatLab 报告中添加了"假设检验结果"章节——**从"描述数据"升级为"统计决策"**。你不再只是说"两组均值差 3500 元"，而是说"在 95% 置信水平下拒绝 H0，Cohen's d=0.68（中到大效应），但需注意多重比较风险"。

下周（Week 07）你将学习**多组比较与多重比较校正**——当你有 3 个或更多组时，t 检验不够用了，需要 ANOVA；当你跑了很多次检验时，需要校正 p 值（Bonferroni、FDR）来控制假阳性风险。届时你会用到本周的所有工具：H0/H1 设定、p 值理解、效应量解释，以及——最重要的——对 AI 生成报告的审查能力。

---

## Definition of Done（学生自测清单）

- [ ] 我能将研究问题转化为正式的 H0/H1 陈述，并选择单尾/双尾检验
- [ ] 我能正确解释 p 值的含义，避免"p 值即真理概率"的常见误区
- [ ] 我能选择合适的 t 检验（单样本/独立样本/配对），并进行前提假设检查（正态性、方差齐性）
- [ ] 我能理解第一类错误（α）和第二类错误（β）的权衡，及其对统计功效的影响
- [ ] 我能计算并解释效应量（Cohen's d），区分统计显著与实际显著
- [ ] 我能审查 AI 生成的统计检验报告，识别常见谬误（p-hacking、前提假设未验证、效应量缺失）
- [ ] 我能在 StatLab 报告中添加"假设检验结果"章节，包含 H0/H1、p 值、效应量、CI、前提假设检查
- [ ] 我用 git 提交了本周的工作（至少一次 commit）
- [ ] 我理解"假设检验不是证明 H1，而是拒绝 H0"的逻辑，以及 p 值、效应量、功效的完整框架
