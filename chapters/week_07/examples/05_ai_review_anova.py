"""
示例：审查 AI 生成的多组比较结论——你能看出这些错误吗？

本例演示如何审查 AI 工具生成的多组比较报告，识别常见错误：
1. 未说明是否做了多重比较校正
2. 没有报告效应量
3. 没有检查前提假设
4. 跳过 ANOVA，直接做两两比较

运行方式：python3 chapters/week_07/examples/05_ai_review_anova.py
预期输出：
  - stdout 输出审查报告
  - 修订后的报告保存到 output/revised_anova_report.md

反例：AI 报告了 10 次两两比较，p < 0.05 的就被标记为"显著"，
但实际上 FWER ≈ 40%，这些"显著"结果可能只是运气。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def generate_ai_report() -> str:
    """生成模拟的 AI 报告（包含错误）"""
    return """# AI 生成的报告

我们检验了 5 个渠道的转化率差异，结果如下：

## 描述统计

| 渠道 | 样本量 | 转化率 |
|------|--------|--------|
| A | 500 | 10.2% |
| B | 500 | 10.5% |
| C | 500 | 11.0% |
| D | 500 | 10.8% |
| E | 500 | 12.5% |

## 两两比较结果

- A vs E：p = 0.03 ✅ 显著
- B vs E：p = 0.04 ✅ 显著
- C vs E：p = 0.08 ❌ 不显著
- D vs E：p = 0.06 ❌ 不显著
- A vs B：p = 0.75 ❌ 不显著
- A vs C：p = 0.45 ❌ 不显著
- A vs D：p = 0.60 ❌ 不显著
- B vs C：p = 0.55 ❌ 不显著
- B vs D：p = 0.70 ❌ 不显著
- C vs D：p = 0.65 ❌ 不显著

## 结论

E 渠道显著优于 A 和 B 渠道（p < 0.05），建议全面切换到 E 渠道。
"""


def review_ai_report(ai_report: str) -> dict:
    """
    审查 AI 生成的报告，返回问题列表和修订建议

    参数:
        ai_report: AI 生成的原始报告

    返回:
        包含问题列表和修订建议的字典
    """
    issues = []

    # 问题 1：做了多少次比较？是否校正？
    issues.append({
        "category": "多重比较校正",
        "problem": "AI 报告了 10 次两两比较，但没有说明是否做了校正",
        "impact": f"10 次比较，每次 α = 0.05，FWER ≈ {1 - (1 - 0.05)**10:.1%}（至少一个假阳性的概率）",
        "recommendation": "说明是否做了多重比较校正（如 Tukey HSD 或 Bonferroni）"
    })

    # 问题 2：效应量报告了吗？
    issues.append({
        "category": "效应量",
        "problem": "AI 没有报告效应量",
        "impact": "A vs E：差异 2.3%，p = 0.03。但 Cohen's h 或风险差是多少？如果 E 渠道成本比 A 高 20%，这个 2.3% 的转化率提升还有商业意义吗？",
        "recommendation": "报告效应量（如风险差、Cohen's h），并讨论商业意义"
    })

    # 问题 3：前提假设检查了吗？
    issues.append({
        "category": "前提假设",
        "problem": "AI 没有检查前提假设",
        "impact": "转化率是二元数据，ANOVA 或 t 检验可能不合适，应该用比例检验或卡方检验。方差齐性？正态性？这些都没有报告",
        "recommendation": "检查前提假设，或使用非参数方法（如 Kruskal-Wallis）"
    })

    # 问题 4：ANOVA 用了吗？
    issues.append({
        "category": "分析流程",
        "problem": "AI 直接做了 10 次两两比较，没有先用 ANOVA 判断'是否有任何差异'",
        "impact": "标准流程：先 ANOVA（F 检验），如果显著再做事后比较（Tukey HSD）。AI 跳过了 ANOVA，直接做两两比较，假阳性风险更高",
        "recommendation": "先用 ANOVA 判断是否有任何差异，再做 Tukey HSD 事后比较"
    })

    return {
        "issues": issues,
        "ai_report": ai_report
    }


def generate_revised_report() -> str:
    """生成修订后的报告（正确版本）"""
    return """# 修订后的多组比较报告

## 分析概述

本报告比较 5 个渠道（A、B、C、D、E）的转化率差异，使用 ANOVA 和 Tukey HSD 事后比较，
控制 family-wise error rate ≤ 0.05。

## 描述统计

| 渠道 | 样本量 | 转化率 | 95% CI |
|------|--------|--------|--------|
| A | 500 | 10.2% | [7.6%, 13.2%] |
| B | 500 | 10.5% | [7.9%, 13.5%] |
| C | 500 | 11.0% | [8.3%, 14.1%] |
| D | 500 | 10.8% | [8.1%, 13.9%] |
| E | 500 | 12.5% | [9.6%, 15.8%] |

## ANOVA 结果

**原假设**：5 个渠道的转化率全部相等

- χ² 统计量：8.45
- p 值：0.076
- η²（效应量）：0.02（小效应）

**结论**：p ≥ 0.05，**无法拒绝原假设**。5 个渠道的转化率差异不具有统计显著性。

## Tukey HSD 事后比较（95% 置信区间）

| 比较 | 转化率差异 | 95% CI | 校正后 p 值 | 显著? |
|------|-----------|--------|-----------|------|
| A vs E | +2.3% | [-0.8%, 5.4%] | 0.18 | ❌ |
| B vs E | +2.0% | [-1.1%, 5.1%] | 0.28 | ❌ |
| C vs E | +1.5% | [-1.6%, 4.6%] | 0.52 | ❌ |
| D vs E | +1.7% | [-1.4%, 4.8%] | 0.38 | ❌ |
| 其他 | < 1.0% | - | > 0.50 | ❌ |

**注**：所有 p 值均已使用 Tukey HSD 方法校正，控制 family-wise error rate。

## 效应量

- A vs E：Cohen's h = 0.07（极小效应）
- B vs E：Cohen's h = 0.06（极小效应）

## 结论与建议

### 统计结论

- ANOVA 未发现 5 个渠道的转化率存在显著差异（p = 0.076）
- Tukey HSD 事后比较也未发现任何两两显著差异（校正后 p > 0.05）
- 观察到的差异较小（Cohen's h < 0.1），即使统计显著，实际意义有限

### 局限性

- 本研究检验了 10 对比较，使用了 Tukey HSD 校正
- 未校正的 p 值（如 A vs E 的 p = 0.03）不应单独使用
- 样本量（每组 n=500）对于检测小效应可能不足

### 商业建议

- E 渠道有潜力（转化率最高），但差异不显著
- 建议扩大样本量后重新评估
- 或考虑成本效益分析：如果 E 渠道成本不高，可以尝试逐步切换并监控效果
- **不建议**基于当前数据全面切换到 E 渠道
"""


def main() -> None:
    """运行 AI 审查演示"""
    print("=== AI 生成的多组比较报告审查 ===\n")

    # 1. 显示 AI 生成的报告
    ai_report = generate_ai_report()
    print("【AI 生成的报告】")
    print(ai_report)

    # 2. 小北的反应
    print("\n" + "=" * 60)
    print("小北看完报告，满意地说：")
    print('"很清楚啊！E 渠道最好，我们马上切换！"')
    print()

    # 3. 老潘的反应
    print("老潘摇摇头：")
    print('"先别急。这份报告有 4 个问题。"')
    print()

    # 4. 审查报告
    review_result = review_ai_report(ai_report)

    print("=" * 60)
    print("【老潘的审查清单】")
    print("=" * 60)
    print()

    for i, issue in enumerate(review_result["issues"], 1):
        print(f"### 问题 {i}：{issue['category']}")
        print(f"**问题**：{issue['problem']}")
        print(f"**影响**：{issue['impact']}")
        print(f"**修订建议**：{issue['recommendation']}")
        print()

    # 5. 修订后的报告
    print("=" * 60)
    print("【修订后的报告】")
    print("=" * 60)
    print()

    revised_report = generate_revised_report()
    print(revised_report)

    # 保存修订后的报告
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'revised_anova_report.md'
    output_path.write_text(revised_report)
    print(f"\n修订后的报告已保存到 {output_path}")

    # 6. 老潘的总结
    print("\n" + "=" * 60)
    print("【老潘的总结】")
    print("=" * 60)
    print()
    print("AI 给的第一份报告，看起来专业，实际上有 4 个坑。")
    print("你如果直接用这个结论做决策，就是在赌假阳性，不是做分析。")
    print()
    print("修订后的报告，才是一份负责任的统计报告。它告诉你：")
    print("  • 差异是否显著")
    print("  • 效应量有多大")
    print("  • 前提是什么")
    print("  • 局限在哪里")
    print()
    print("阿码若有所思：'所以 ANOVA + 事后比较 + 校正，是标准流程？'")
    print()
    print("老潘：'对。'AI 可以帮你跑检验、算 p 值、生成图表，")
    print("但只有你能：")
    print("  1. 决定是否需要校正（以及用哪种方法）")
    print("  2. 判断效应量是否有实际意义")
    print("  3. 检查前提假设是否满足")
    print("  4. 区分统计结论和商业建议")
    print("  5. 写出诚实、有局限性的报告")
    print()
    print("这正是 AI 时代分析者的核心价值：")
    print('**不是会跑 ANOVA，而是会设计多组比较的策略，并审查 AI 的结论**。')


if __name__ == "__main__":
    main()
