# Week 05 阻塞项修复报告

## 修复时间
2026-02-12

## 修复的阻塞项（全部完成 ✅）

### 1. 重复台词（原第 200 行和第 202 行）
**问题**：小北的台词重复出现两次
**修复**：删除了重复的第 200 行台词，保留第 202 行更准确的表述
**修改位置**：CHAPTER.md 第 198-204 行
**修改后**：
```markdown
运行多次模拟，你会发现：**即使每次模拟的具体数字不同，P(患病|阳性) 总是稳定在 50% 左右**。这就是"大数定律"在发挥作用——样本量够大时，模拟结果会收敛到理论值。

小北看完模拟结果，长出一口气："**所以我不该看'检测准确率 99%'，而该看'阳性后得病概率 50%'。**"

"对。"老潘说，"前者是 P(阳性|患病)，后者是 P(患病|阳性)。它们看起来很像，但数字完全不同。"
```

### 2. lognormal 分布未解释（原第 477 行）
**问题**：代码中使用 `np.random.lognormal()` 但没有解释这是什么分布、为什么要用它
**修复**：添加详细注释，解释对数正态分布的特点及适用场景
**修改位置**：CHAPTER.md 第 585-589 行
**修改后**：
```python
# 假设这是你的样本（100 个用户的消费数据）
np.random.seed(42)
# 用对数正态分布模拟消费数据：很多真实世界的消费、收入数据都呈现右偏特征
# 参数说明：均值参数 7（对数尺度）、标准差参数 0.8（对数尺度）、样本量 100
sample = np.random.lognormal(7, 0.8, 100)  # 右偏分布
```

### 3. Bootstrap 代码与注释不符（原第 621 行之前）
**问题**：正文提到"两组均值差异"，但代码只演示单组统计量
**修复**：添加完整的两组均值差异 Bootstrap 示例代码
**修改位置**：CHAPTER.md 第 621-658 行
**修改后**：在单组 Bootstrap 代码后添加了：
```python
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

## 修复的建议项（全部完成 ✅）

### 1. Bootstrap "有放回重采样"解释优化
**问题**：原解释不够直观
**修复**：重写为更清晰的分步解释，强调 Bootstrap 的巧妙之处
**修改位置**：CHAPTER.md 第 577-579 行
**修改后**：
```markdown
核心思想是这样的：**你手里只有一份样本，没法真的去"重新抽样"。但 Bootstrap 的巧妙之处在于——它假设"你的样本就是总体"，然后从这份"总体"中模拟重复抽样的过程**。

具体做法是**有放回重采样**（resample with replacement）：想象你的样本里有 100 个数据点，编号 1-100。Bootstrap 做的是从中随机抽 100 次，每次抽完记录后又放回去——所以某些数据点可能被抽到多次，某些一次都没抽到。这模拟了"从真实总体重新采样"的过程，因为你不知道真实的总体分布，只能用样本作为最佳估计。
```

### 2. Week 03 IQR 方法连接更自然
**问题**：回顾桥连接 Week 03 的 IQR 方法略显生硬
**修复**：用角色对话自然引出，避免说教式回顾
**修改位置**：CHAPTER.md 第 315-317 行
**修改后**：
```markdown
阿码举手："**所以超过 3σ 的点就是异常值？**"

"不一定。"老潘摇头，"这个规则只在数据**真的符合正态分布**时才有效。Week 03 你学过 IQR 方法——它不依赖分布假设，更稳健。真实世界里，收入、房价这些重要数据都严重右偏，用 3σ 规则会让你漏掉很多'该管的异常值'。"
```

### 3. StatLab 进度段落去重
**问题**：第 847-849 行的 StatLab 进度段落与前面内容重复
**修复**：删除重复的说明性文字，精简为直接引入代码示例
**修改位置**：CHAPTER.md 第 870-874 行
**修改后**：
```markdown
## StatLab 进度

本周 StatLab 报告增加了"不确定性量化"章节。下面的示例代码展示了如何在报告中总结关键统计量的稳定性：
```

## 额外修复

### 术语表同步
**问题**：shared/glossary.yml 缺少 Week 05 的部分术语（正态分布、二项分布、泊松分布、抽样分布、标准误）
**修复**：在 shared/glossary.yml 的 Week 05 部分添加了 5 个缺失术语
**修改位置**：shared/glossary.yml 第 198-245 行
**添加的术语**：
- 正态分布 (normal distribution)
- 二项分布 (binomial distribution)
- 泊松分布 (Poisson distribution)
- 抽样分布 (sampling distribution)
- 标准误 (standard error)

## 验证结果

### 内容编辑验证 ✅
```bash
python3 scripts/validate_week.py --week week_05 --mode drafting
```
**结果**：通过 ✅

### Release 模式验证 ⚠️
```bash
python3 scripts/validate_week.py --week week_05 --mode release
```
**结果**：失败（但失败原因是缺失文件 QA_REPORT.md、ANCHORS.yml、starter_code/ 等，这些是其他 agent 的职责，不属于 prose-polisher 的修复范围）

### 测试状态 ⚠️
pytest 有 10 个失败，但这些都是**代码测试**，不是**内容编辑问题**。失败的测试涉及：
- `simulate_clt()` 函数签名不匹配
- 某些边界情况的测试断言

这些应该由 `example-writer` 和 `error-fixer` 处理，不属于 prose-polisher 的职责。

## 总结

**✅ 所有 QA 报告中的阻塞项已修复**
**✅ 所有建议项已处理**
**✅ 术语表已同步**

**遗留问题**（不属于 prose-polisher 职责）：
- QA_REPORT.md 文件缺失（应由 QA 生成或 lead agent 创建）
- ANCHORS.yml 文件缺失（应由 chapter-writer 创建）
- starter_code/ 目录缺失（应由 example-writer 创建）
- pytest 测试失败（应由 example-writer 和 error-fixer 修复）

## 修改文件清单

1. `/Users/wangxq/Documents/statistics-agentic-coding/chapters/week_05/CHAPTER.md` - 主要修改文件
2. `/Users/wangxq/Documents/statistics-agentic-coding/shared/glossary.yml` - 术语表同步
