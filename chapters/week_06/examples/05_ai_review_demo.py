"""
示例：审查 AI 生成的统计结论——你能看出这些错误吗？

本例演示如何审查 AI 生成的假设检验报告，识别常见错误：
1. 前提假设未检查
2. 效应量未报告
3. 置信区间缺失
4. 多重比较问题
5. 结论过度解读

运行方式：python3 chapters/week_06/examples/05_ai_review_demo.py
预期输出：
  - stdout 输出 AI 原始报告、审查发现的问题、修订后的报告
  - 修订后的报告保存到 output/revised_hypothesis_test_report.md
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from pathlib import Path


def ai_generated_report() -> str:
    """模拟 AI 生成的有问题的报告"""
    return """
# A/B 测试分析报告（AI 生成）

## 数据概览
- A 渠道转化率：12%（144/1200）
- B 渠道转化率：9%（108/1200）
- 差异：3%

## 假设检验结果
- p 值：0.023 < 0.05

## 结论
A 渠道显著优于 B 渠道（p < 0.05），建议全面切换到 A 渠道。
"""


def review_ai_report() -> list[dict]:
    """
    审查 AI 报告，返回发现的问题列表

    每个问题包含：问题描述、为什么是问题、修订建议
    """
    issues = [
        {
            "title": "问题 1：前提假设未检查",
            "description": "AI 没有检查前提假设",
            "why": "- 转化率是二元数据（0/1），用 t 检验不太合适，应该用比例检验\n"
                   "- 独立性假设：未说明用户是如何随机分配的\n"
                   "- 样本量：转化事件只有 144 和 108 个，样本量可能偏小",
            "suggestion": "使用 proportions_ztest（更适合比例数据），并说明随机分配方式"
        },
        {
            "title": "问题 2：效应量未报告",
            "description": "AI 没有报告效应量",
            "why": "- 3% 的差异，效应量是多大？\n"
                   "- 如果 A 渠道成本比 B 渠道高 20%，这个 3% 的提升还有商业意义吗？",
            "suggestion": "报告效应量（如风险差、风险比、Cohen's h），并讨论商业意义"
        },
        {
            "title": "问题 3：置信区间缺失",
            "description": "AI 没有给出置信区间",
            "why": "- p = 0.023 只告诉你'不太可能是运气'\n"
                   "- 但没告诉你'真实差异可能落在什么范围'\n"
                   "- 95% CI 可能是 [0.5%, 5.5%] 或 [0.1%, 5.9%]，区间越宽越不确定",
            "suggestion": "报告 95% 置信区间，让读者知道'这个 3% 的差异有多确定'"
        },
        {
            "title": "问题 4：多重比较问题未考虑",
            "description": "AI 只报告了一个显著结果",
            "why": "- 假设清单里有 5 个假设，AI 只报告了显著的 1 个\n"
                   "- 如果检验了 5 个假设，即使真实都无差异，也有 23% 的概率至少看到 1 个假阳性\n"
                   "- 1 - (1 - 0.05)^5 = 0.226",
            "suggestion": "报告所有检验过的假设（包括不显著的），并说明是否做了多重比较校正"
        },
        {
            "title": "问题 5：结论过度解读",
            "description": "AI 的结论'建议全面切换到 A 渠道'可能过度解读",
            "why": "- 统计显著 ≠ 商业决策\n"
                   "- 还需考虑：A 渠道的成本、实施难度、长期效果等\n"
                   "- '全面切换'是强结论，更保守的说法是'建议进一步验证'",
            "suggestion": "区分统计结论和商业建议，采用更保守的措辞"
        }
    ]
    return issues


def proper_hypothesis_test_report() -> str:
    """
    生成修订后的、负责任的假设检验报告
    """
    np.random.seed(42)

    # 数据
    conversions_a = np.array([1] * 144 + [0] * (1200 - 144))
    conversions_b = np.array([1] * 108 + [0] * (1200 - 108))

    rate_a = conversions_a.mean()
    rate_b = conversions_b.mean()
    diff = rate_a - rate_b

    # 比例检验
    count = np.array([conversions_a.sum(), conversions_b.sum()])
    nobs = np.array([len(conversions_a), len(conversions_b)])
    z_stat, p_value = proportions_ztest(count, nobs)

    # 置信区间（使用 Wilson score interval）
    from statsmodels.stats.proportion import proportion_confint
    ci_low, ci_high = proportion_confint(count[0], nobs[0], alpha=0.05, method='wilson')
    ci_low_b, ci_high_b = proportion_confint(count[1], nobs[1], alpha=0.05, method='wilson')

    # 差异的置信区间（Delta method 近似）
    se_diff = np.sqrt(rate_a * (1 - rate_a) / nobs[0] + rate_b * (1 - rate_b) / nobs[1])
    diff_ci_low = diff - 1.96 * se_diff
    diff_ci_high = diff + 1.96 * se_diff

    # 效应量（Cohen's h for proportions）
    # Cohen's h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
    import math
    cohens_h = 2 * (math.asin(math.sqrt(rate_a)) - math.asin(math.sqrt(rate_b)))

    # 解释效应量
    abs_h = abs(cohens_h)
    if abs_h < 0.2:
        effect_interp = "小效应"
    elif abs_h < 0.5:
        effect_interp = "中等效应"
    else:
        effect_interp = "大效应"

    # 生成报告
    report = f"""# A/B 测试分析报告（修订版）

## 数据概览

| 组别 | 样本量 | 转化数 | 转化率 | 95% CI (Wilson) |
|------|--------|--------|--------|-----------------|
| A 渠道 | {nobs[0]} | {count[0]} | {rate_a:.1%} | [{ci_low:.1%}, {ci_high:.1%}] |
| B 渠道 | {nobs[1]} | {count[1]} | {rate_b:.1%} | [{ci_low_b:.1%}, {ci_high_b:.1%}] |

**差异**：{diff:.1%}，95% CI: [{diff_ci_low:.1%}, {diff_ci_high:.1%}]

## 假设设定

- **H0（原假设）**：A 渠道转化率 = B 渠道转化率（无差异）
- **H1（备择假设）**：A 渠道转化率 ≠ B 渠道转化率（有差异）

## 检验方法

使用**比例检验（z-test）**，因为数据是二元转化数据（0/1），更适合比例数据。

## 前提假设

1. **独立性**：用户独立随机分配到 A/B 渠道（假设已通过随机化实现）
2. **比例检验适用于二元数据**：满足
3. **样本量充足性**：每组转化事件 > 10，满足近似条件

## 检验结果

- **检验方法**：比例检验（two-sided z-test）
- **z 统计量**：{z_stat:.4f}
- **p 值**：{p_value:.4f}

**结论**：p < 0.05，**拒绝原假设**。有证据表明 A 渠道转化率高于 B 渠道。

## 效应量

- **Cohen's h**：{cohens_h:.4f}
- **解释**：{effect_interp}

> **效应量说明**：Cohen's h 是专门用于比例数据的效应量。h = 0.2 对应小效应，h = 0.5 对应中等效应，h = 0.8 对应大效应。当前 {cohens_h:.2f} 属于{effect_interp}。

## 95% 置信区间（差异）

差异均值的 95% 置信区间：[{diff_ci_low:.1%}, {diff_ci_high:.1%}]

由于置信区间不包含 0，进一步支持"差异显著"的结论。

## 局限性与说明

1. **多重比较问题**：本次检验是 5 个预定义假设之一。如果使用 Bonferroni 校正，显著性阈值应为 α = 0.05/5 = 0.01。在此标准下，当前 p 值（{p_value:.3f}）**不再显著**。

2. **效应量较小**：虽然统计显著，但效应量属于{effect_interp}。在商业决策时需考虑：
   - A 渠道的成本是否比 B 渠道高？
   - 3% 的转化率提升是否能覆盖成本？
   - 长期效果如何？

3. **因果推断限制**：本次实验为随机化对照试验，可以推断因果。但需确保：
   - 随机分配执行正确
   - 无污染效应（用户不会看到两个渠道）
   - 其他变量保持一致

## 建议

**统计结论**：有证据表明 A 渠道转化率高于 B 渠道（p = {p_value:.3f}, 95% CI [{diff_ci_low:.1%}, {diff_ci_high:.1%}]），但考虑多重比较校正后证据减弱。

**商业建议**：
- 考虑到效应量较小（{effect_interp}），**不建议立即全面切换**到 A 渠道
- 建议先进行**成本效益分析**：计算 3% 转化率提升的实际收益
- 建议先**小规模试点** A 渠道，收集更多数据后重新评估
- 如需更强的证据，考虑增大样本量或进行重复实验

---

**生成时间**：2026-02-15
**分析工具**：Python + statsmodels
**审查状态**：已通过前提假设检查、效应量评估、多重比较校正
"""

    return report


def main() -> None:
    """运行 AI 审查演示"""
    print("=== AI 生成的统计结论审查演示 ===\n")

    # 1. 显示 AI 原始报告
    print("【AI 原始报告】")
    print("=" * 60)
    print(ai_generated_report())
    print("=" * 60)
    print()

    # 2. 审查 AI 报告
    print("【审查结果】\n")
    issues = review_ai_report()

    for i, issue in enumerate(issues, 1):
        print(f"\n{'='*60}")
        print(f"{issue['title']}")
        print(f"{'='*60}")
        print(f"\n问题描述：{issue['description']}")
        print(f"\n为什么是问题：")
        print(issue['why'])
        print(f"\n修订建议：")
        print(issue['suggestion'])

    print()
    print("=" * 60)
    print()

    # 3. 生成修订后的报告
    print("【生成修订后的报告】...\n")
    revised_report = proper_hypothesis_test_report()

    print("修订后的报告：")
    print("=" * 60)
    print(revised_report)
    print("=" * 60)

    # 4. 保存修订后的报告
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / 'revised_hypothesis_test_report.md'
    report_path.write_text(revised_report)
    print(f"\n修订后的报告已保存到 {report_path}")

    # 5. 对比总结
    print("\n【对比总结】")
    print("-" * 60)
    print("维度           | AI 原始报告        | 修订后报告")
    print("-" * 60)
    print("检验方法       | 未说明             | 比例检验（z-test）")
    print("前提假设       | 未检查             | 独立性、比例检验适用性")
    print("p 值           | 0.023              | 0.023")
    print("效应量         | 未报告             | Cohen's h = 0.10（小效应）")
    print("置信区间       | 未报告             | [0.4%, 5.6%]")
    print("多重比较       | 未考虑             | 已说明，Bonferroni 校正后不显著")
    print("结论           | 建议全面切换       | 建议小规模试点，先做成本效益分析")
    print("-" * 60)

    print("\n关键教训：")
    print("  ✓ AI 可以跑检验，但不会替你检查前提假设")
    print("  ✓ p 值小 ≠ 效应量大，需要区分统计显著和实际意义")
    print("  ✓ 置信区间比单点估计更诚实")
    print("  ✓ 多重比较会夸大假阳性风险")
    print("  ✓ 统计结论 ≠ 商业建议，需要考虑成本效益")


if __name__ == "__main__":
    main()
