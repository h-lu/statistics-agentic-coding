#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：AI 生成统计报告的审查清单

本例演示如何审查 AI 生成的统计检验报告，识别常见谬误：
1. H0/H1 是否明确
2. p 值解释是否正确
3. 效应量是否报告
4. 置信区间是否报告
5. 前提假设是否验证
6. 多重比较是否校正
7. 样本量/功效是否讨论
8. 相关 vs 因果混淆

运行方式：python3 chapters/week_06/examples/04_ai_report_checklist.py
预期输出：终端显示审查结果、生成修订版报告模板

核心概念：
- AI 可以快速生成报告，但审查必须由人来做
- 常见问题包括：p 值误解释、缺少效应量、前提假设未验证
- 完整的统计报告应包含：H0/H1、p 值、效应量、CI、前提检查

作者：StatLab Week 06
"""
from __future__ import annotations

import re
from typing import List, Dict


# =============================================================================
# 审查函数
# =============================================================================

def review_statistical_report(report_text: str) -> List[Dict]:
    """
    审查 AI 生成的统计检验报告，标注潜在问题。

    检查项：
    1. H0/H1 是否明确
    2. p 值解释是否正确
    3. 效应量是否报告
    4. 置信区间是否报告
    5. 前提假设是否验证
    6. 多重比较是否校正
    7. 样本量/功效是否讨论
    8. 相关 vs 因果混淆

    参数：
        report_text: AI 报告文本

    返回：
        list: 问题列表，每个问题是包含'问题'、'风险'、'建议'的字典
    """
    issues = []

    # ========== 检查 1：H0/H1 是否明确 ==========
    if "H0" not in report_text and "原假设" not in report_text:
        issues.append({
            "问题": "未明确说明原假设 H0",
            "风险": "读者不清楚'显著'是拒绝什么假设",
            "建议": "补充 H0/H1 的正式陈述",
            "优先级": "高"
        })

    # ========== 检查 2：p 值解释是否正确 ==========
    if re.search(r"H0为真的概率|结论为真的概率|原假设成立的概率", report_text):
        issues.append({
            "问题": "p 值误解释",
            "风险": "严重逻辑错误（p ≠ P(H0|data）",
            "建议": "正确解释：在 H0 为真时，看到当前数据的概率",
            "优先级": "高"
        })

    # ========== 检查 3：效应量是否报告 ==========
    if "Cohen" not in report_text and "效应量" not in report_text and "d=" not in report_text:
        issues.append({
            "问题": "缺少效应量",
            "风险": "无法判断实际意义（p<0.05 可能是微小效应）",
            "建议": "补充 Cohen's d 或 η²",
            "优先级": "高"
        })

    # ========== 检查 4：置信区间是否报告 ==========
    if "CI" not in report_text and "置信区间" not in report_text:
        issues.append({
            "问题": "缺少置信区间",
            "风险": "无法量化不确定性",
            "建议": "补充 95% CI",
            "优先级": "中"
        })

    # ========== 检查 5：前提假设是否验证 ==========
    if "正态性" not in report_text and "Shapiro" not in report_text:
        issues.append({
            "问题": "未验证正态性假设",
            "风险": "数据严重偏态时 t 检验不可靠",
            "建议": "补充 Shapiro-Wilk 检验或 QQ 图",
            "优先级": "中"
        })

    if "方差齐性" not in report_text and "Levene" not in report_text:
        issues.append({
            "问题": "未验证方差齐性假设",
            "风险": "方差不齐时应使用 Welch's t 检验",
            "建议": "补充 Levene 检验",
            "优先级": "中"
        })

    # ========== 检查 6：多重比较是否校正 ==========
    if re.search(r"多次检验|多个指标|多个变量", report_text):
        if "校正" not in report_text and "Bonferroni" not in report_text and "FDR" not in report_text:
            issues.append({
                "问题": "多重比较未校正",
                "风险": "假阳性风险放大（跑 20 次检验总会有 1 次碰巧显著）",
                "建议": "使用 Bonferroni 或 FDR 校正",
                "优先级": "高"
            })

    # ========== 检查 7：样本量/功效是否讨论 ==========
    if "功效" not in report_text and "样本量" not in report_text and "n=" not in report_text:
        issues.append({
            "问题": "未讨论样本量/功效",
            "风险": "小样本检测小效应时假阴性风险高",
            "建议": "补充功效分析或说明样本量限制",
            "优先级": "中"
        })

    # ========== 检查 8：相关 vs 因果 ==========
    if re.search(r"导致|因果|引起", report_text):
        if "实验" not in report_text and "随机" not in report_text and "A/B" not in report_text:
            issues.append({
                "问题": "相关被误写成因果",
                "风险": "观察性研究无法确定因果方向",
                "建议": "用'相关'、'关联'而非'导致'、'因果'",
                "优先级": "高"
            })

    return issues


def generate_revised_report(
    original_report: str,
    issues: List[Dict],
    t_stat: float = 2.15,
    p_value: float = 0.032,
    mean_diff: float = 5.2,
    cohens_d: float = 0.21,
    ci_low: float = 1.8,
    ci_high: float = 8.6
) -> str:
    """
    基于审查结果生成修订版报告。

    参数：
        original_report: 原始报告
        issues: 问题列表
        t_stat: t 统计量
        p_value: p 值
        mean_diff: 均值差异
        cohens_d: Cohen's d 效应量
        ci_low: CI 下界
        ci_high: CI 上界

    返回：
        str: 修订后的报告
    """
    revised_report = f"""统计检验报告（修订版）

## 假设设定
- H0（原假设）：实验组与对照组的活跃度均值相等（μ_exp = μ_ctrl）
- H1（备择假设）：实验组活跃度高于对照组（μ_exp > μ_ctrl，单尾检验）

## 前提假设检查
- 正态性：Shapiro-Wilk 检验 p_ctrl=0.12, p_exp=0.08（> 0.05，正态性假设满足）
- 方差齐性：Levene 检验 p=0.21（> 0.05，方差齐性假设满足）
- 样本独立性：✓ 用户随机抽样，互不干扰

## 检验结果
- t 统计量：t(998) = {t_stat:.2f}
- p 值（单尾）：{p_value:.3f}
- 决策：{"拒绝 H0（差异显著）" if p_value < 0.05 else "无法拒绝 H0（差异不显著）"}

## 效应量与置信区间
- 均值差异：{mean_diff:.1f} 分
- 95% 置信区间：[{ci_low:.1f}, {ci_high:.1f}] 分
- Cohen's d 效应量：{cohens_d:.2f}（{"小效应" if abs(cohens_d) < 0.3 else "中效应" if abs(cohens_d) < 0.7 else "大效应"}）

## 解读与局限
- **统计显著性**：在 α=0.05 水平下拒绝 H0，差异具有统计显著性
- **实际意义**：效应量较小（d={cohens_d:.2f}），需评估业务上是否值得上线
- **不确定性**：95% CI 不包含 0，{"支持" if ci_low > 0 else "不支持"}差异为正的结论
- **样本量**：每组 n=500，功效≈68%（对于小效应 d={abs(cohens_d):.2f}），存在假阴性风险
- **因果推断**：本研究为 A/B 测试（随机分配），可支持因果结论

## 建议
- 统计上支持新功能有效，但效应较小，需结合成本-收益分析决定是否上线
- 如需更稳健的结论，可增加样本量以提高功效
- 注意：本检验未经多重比较校正（如果同时检验了其他指标，需考虑 Bonferroni 校正）

## 与原报告的主要差异
"""

    # 添加改进说明
    if any("p 值误解释" in issue["问题"] for issue in issues):
        revised_report += "- 修正了 p 值的错误解释\n"
    if any("缺少效应量" in issue["问题"] for issue in issues):
        revised_report += "- 补充了 Cohen's d 效应量计算\n"
    if any("未验证正态性假设" in issue["问题"] for issue in issues):
        revised_report += "- 补充了正态性和方差齐性检验\n"
    if any("缺少置信区间" in issue["问题"] for issue in issues):
        revised_report += "- 补充了 95% 置信区间\n"

    return revised_report


# =============================================================================
# 示例：审查 AI 报告
# =============================================================================

def example_ai_report_review():
    """演示审查 AI 生成的报告"""
    print("=" * 70)
    print("AI 报告审查示例")
    print("=" * 70)

    # 示例 AI 报告（含多种问题）
    ai_report = """
统计检验报告：

我们对实验组和对照组进行了 t 检验，结果 t=2.15, p=0.03。

结论：
1. 新功能显著提升了用户活跃度（H0 为真的概率是 3%）。
2. 两组均值差异为 5.2 分。

建议：
- 上线新功能，因为效果显著。

注：我们还检验了点击率、停留时长、转化率等 5 个指标，
其中点击率也显著（p=0.04）。
"""

    print("\n[原始 AI 报告]")
    print(ai_report)

    print("\n[审查结果]")
    issues = review_statistical_report(ai_report)

    if issues:
        # 按优先级排序
        priority_order = {'高': 0, '中': 1, '低': 2}
        issues_sorted = sorted(issues, key=lambda x: priority_order.get(x['优先级'], 3))

        print(f"发现 {len(issues)} 个潜在问题：\n")
        for i, issue in enumerate(issues_sorted, 1):
            print(f"{i}. 【{issue['优先级']}优先级】{issue['问题']}")
            print(f"   风险：{issue['风险']}")
            print(f"   建议：{issue['建议']}\n")
    else:
        print("✓ 未发现明显问题")

    # 生成修订版报告
    print("=" * 70)
    print("生成修订版报告...")
    print("=" * 70)

    revised = generate_revised_report(ai_report, issues)
    print(revised)


# =============================================================================
# 反例：AI 常犯的错误
# =============================================================================

def common_ai_mistakes():
    """展示 AI 在统计报告中常见的错误"""
    print("\n" + "=" * 70)
    print("AI 在统计报告中常见的错误模式")
    print("=" * 70)

    mistakes = [
        {
            "错误": "p-hacking：尝试多种方法只报告显著的结果",
            "示例": "我们尝试了 t 检验、Mann-Whitney、Bootstrap，发现 t 检验 p<0.05",
            "问题": "选择性报告，夸大显著性",
            "修正": "预先指定分析方法，报告所有尝试"
        },
        {
            "错误": "忽略前提假设：直接跑 t 检验",
            "示例": "t 检验显示 p=0.02，因此差异显著",
            "问题": "未检查正态性、方差齐性",
            "修正": "先做 Shapiro-Wilk 和 Levene 检验"
        },
        {
            "错误": "过度解读 p 值",
            "示例": "p=0.001 说明效果非常显著（非常重要）",
            "问题": "混淆统计显著和实际显著",
            "修正": "报告效应量（Cohen's d），评估业务意义"
        },
        {
            "错误": "因果推断过度",
            "示例": "收入与消费相关 r=0.6，说明收入导致消费增加",
            "问题": "观察性研究不能确定因果",
            "修正": "用'相关'而非'导致'，或说明是 A/B 测试"
        },
        {
            "错误": "忽略多重比较",
            "示例": "我们检验了 10 个指标，3 个显著（p<0.05）",
            "问题": "未校正 p 值，假阳性风险放大",
            "修正": "使用 Bonferroni 或 FDR 校正"
        }
    ]

    for i, m in enumerate(mistakes, 1):
        print(f"\n错误 {i}：{m['错误']}")
        print(f"  示例：{m['示例']}")
        print(f"  问题：{m['问题']}")
        print(f"  修正：{m['修正']}")


# =============================================================================
# 交互式审查工具
# =============================================================================

def interactive_review_tool():
    """交互式报告审查工具（示例）"""
    print("\n" + "=" * 70)
    print("交互式报告审查工具")
    print("=" * 70)
    print("\n使用方法：")
    print("1. 将你的 AI 报告保存到文件")
    print("2. 调用 review_statistical_report() 函数")
    print("3. 根据问题列表修订报告")
    print("\n示例代码：")
    print("""
# 读取 AI 报告
with open('ai_report.txt', 'r', encoding='utf-8') as f:
    report = f.read()

# 审查报告
issues = review_statistical_report(report)

# 打印问题
for issue in issues:
    print(f"问题：{issue['问题']}")
    print(f"建议：{issue['建议']}")
    print()
    """)


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数：运行 AI 报告审查示例"""
    # 示例：审查 AI 报告
    example_ai_report_review()

    # 常见错误模式
    common_ai_mistakes()

    # 交互式工具说明
    interactive_review_tool()

    print("\n" + "=" * 70)
    print("要点总结")
    print("=" * 70)
    print("1. AI 可以快速生成报告，但审查必须由人来做")
    print("2. 常见问题包括：p 值误解释、缺少效应量、前提假设未验证")
    print("3. 完整的统计报告应包含：H0/H1、p 值、效应量、CI、前提检查")
    print("4. 多重比较必须校正，否则假阳性风险放大")
    print("5. 统计显著 ≠ 实际显著，需要结合效应量和业务判断")
    print("6. 观察性研究不能确定因果，除非是随机对照实验")
    print("=" * 70)


if __name__ == "__main__":
    main()
