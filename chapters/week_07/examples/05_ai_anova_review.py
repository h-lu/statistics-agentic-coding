#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：AI 生成的多组比较报告审查工具

本例演示如何审查 AI 生成的 ANOVA/卡方检验报告，识别常见谬误。
内容：
1. 自动化审查清单（6 个检查项）
2. 潜在问题识别
3. 改进建议生成

运行方式：python3 chapters/week_07/examples/05_ai_anova_review.py
预期输出：
  - 发现的问题列表
  - 每个问题的风险说明
  - 改进建议

作者：Week 07 示例代码
"""
from __future__ import annotations

from typing import List, Dict


def review_anova_report(report_text: str) -> List[Dict[str, str]]:
    """
    审查 AI 生成的多组比较报告，标注潜在问题。

    参数：
        report_text: AI 报告文本

    返回：
        List[Dict]: 问题列表，每个问题包含问题、风险、建议
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


def print_review_results(issues: List[Dict[str, str]]) -> None:
    """打印审查结果"""
    if issues:
        print(f"发现 {len(issues)} 个潜在问题：\n")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue['问题']}")
            print(f"   风险：{issue['风险']}")
            print(f"   建议：{issue['建议']}\n")
    else:
        print("✓ 未发现明显问题")


def main() -> None:
    """主函数"""
    print("=" * 70)
    print("AI 生成的多组比较报告审查工具")
    print("=" * 70)

    # 示例：一份有问题的 AI 报告
    ai_report = """
多组比较报告：

我们对 5 个城市的用户消费进行了 ANOVA 分析，结果 F=8.52, p=0.002。

结论：
1. 上海和深圳的用户消费显著高于其他城市。
2. ANOVA 显示城市对消费有显著影响。
3. 建议在深圳和上海加大营销投入。

此外，我们对城市与用户等级进行了卡方检验，结果显示城市影响用户等级（χ²=12.34, p=0.002）。
"""

    # 改进版报告（参考）
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

**解读与局限**：
- 统计显著性：ANOVA 和 Tukey HSD 均显示城市之间存在显著差异
- 实际意义：效应量中等（η²=0.064），需评估业务上是否值得差异化策略
- 相关 ≠ 因果：本研究为观察性设计，无法确定因果方向

## 卡方检验结果

**假设设定**：
- H0：城市与用户等级无关
- H1：城市与用户等级相关

**检验结果**：
- 卡方统计量：χ²(12) = 12.34
- p 值：p = 0.421
- Cramér's V 效应量：V = 0.08（关联很弱）
- 决策：无法拒绝 H0，城市与用户等级无显著关联
"""

    print("\n=== 原始 AI 报告 ===")
    print(ai_report)

    print("\n=== AI 报告审查 ===")
    issues = review_anova_report(ai_report)
    print_review_results(issues)

    print("\n=== 修订版报告 ===")
    print(revised_report)

    print("\n" + "=" * 70)
    print("审查完成")
    print("=" * 70)
    print("\n提示：修订版报告解决了所有发现的问题")
    print("- 明确 ANOVA 只回答'是否存在差异'")
    print("- 使用 Tukey HSD 校正多重比较")
    print("- 报告 η² 和 Cramér's V 效应量")
    print("- 验证前提假设（正态性、方差齐性）")
    print("- 用'相关'而非'导致'，避免因果混淆")


if __name__ == "__main__":
    main()
