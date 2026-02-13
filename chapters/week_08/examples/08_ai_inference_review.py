"""
示例：AI 推断报告审查工具——识别缺少不确定性量化的问题。

本例演示如何审查 AI 生成的统计推断报告：
- 检查是否缺少置信区间
- 检查 CI 解释是否错误
- 检查是否缺少效应量
- 检查是否缺少稳健性检验（Bootstrap/置换检验）
- 检查是否缺少不确定性可视化

运行方式：python3 chapters/week_08/examples/08_ai_inference_review.py
预期输出：
  - stdout 输出审查结果（问题列表 + 改进建议）
"""
from __future__ import annotations

from typing import List, Dict


def review_inference_report(report_text: str) -> List[Dict[str, str]]:
    """
    审查 AI 生成的推断报告，标注潜在问题。

    参数：
        report_text: AI 报告文本

    返回：
        issues: 问题列表，每个元素包含 {问题, 风险, 建议}
    """
    issues = []

    # ========== 检查 1：置信区间 ==========
    if ("均值" in report_text or "差异" in report_text) and \
       ("置信区间" not in report_text and "CI" not in report_text and
        "confidence interval" not in report_text):
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
       ("Cohen's d" not in report_text and "效应量" not in report_text and
        "effect size" not in report_text):
        issues.append({
            "问题": "缺少效应量",
            "风险": "只谈统计显著，不谈实际意义",
            "建议": "补充 Cohen's d 或 η²，并解释其实际意义"
        })

    # ========== 检查 4：Bootstrap/置换检验 ==========
    if ("检验" in report_text or "ANOVA" in report_text or "t 检验" in report_text) and \
       ("Bootstrap" not in report_text and "置换检验" not in report_text and
        "permutation" not in report_text):
        issues.append({
            "问题": "未讨论稳健性检验",
            "风险": "数据不满足假设时，结论不可靠",
            "建议": "补充 Bootstrap CI 或置换检验，证明结论稳健"
        })

    # ========== 检查 5：不确定性可视化 ==========
    if ("均值" in report_text or "差异" in report_text) and \
       ("误差条" not in report_text and "error bar" not in report_text and
        "图" not in report_text):
        issues.append({
            "问题": "缺少不确定性可视化",
            "风险": "读者难以直观理解不确定性",
            "建议": "补充 CI 误差条图、Bootstrap 分布图、置换检验零分布图"
        })

    # ========== 检查 6：p 值过度解释 ==========
    if "非常显著" in report_text or "极其显著" in report_text:
        issues.append({
            "问题": "p 值过度解释",
            "风险": "'显著'不等于'重要'，需要效应量支持",
            "建议": "补充效应量及其 CI，讨论实际意义"
        })

    return issues


def generate_improved_report(original_report: str,
                             issues: List[Dict[str, str]]) -> str:
    """
    根据审查结果生成改进后的报告。

    参数：
        original_report: 原始 AI 报告
        issues: 问题列表

    返回：
        improved_report: 改进后的报告
    """
    improved = original_report

    # 添加不确定性量化章节
    uncertainty_section = "\n## 不确定性量化\n\n"

    # 检查是否需要添加 CI
    if any(issue["问题"] == "缺少置信区间" for issue in issues):
        uncertainty_section += "**95% 置信区间**：\n"
        uncertainty_section += "- 均值差：XX.XX 元，95% CI [XX.XX, XX.XX]\n"
        uncertainty_section += "- 解释：如果我们重复抽样 100 次，约 95 个区间会覆盖真值\n\n"

    # 检查是否需要添加效应量
    if any(issue["问题"] == "缺少效应量" for issue in issues):
        uncertainty_section += "**效应量**：\n"
        uncertainty_section += "- Cohen's d：X.XX（效应量小/中等/大）\n"
        uncertainty_section += "- 解释：差异的实际意义\n\n"

    # 检查是否需要添加稳健性检验
    if any(issue["问题"] == "未讨论稳健性检验" for issue in issues):
        uncertainty_section += "**稳健性检验**：\n"
        uncertainty_section += "- Bootstrap 95% CI：[XX.XX, XX.XX]（与 t 公式 CI 一致）\n"
        uncertainty_section += "- 置换检验 p 值：X.XXX（与 t 检验一致）\n"
        uncertainty_section += "- 结论：结果稳健\n\n"

    if len(uncertainty_section) > 30:  # 有内容才添加
        improved += uncertainty_section

    return improved


def main() -> None:
    """主函数：演示 AI 报告审查。"""
    print("=== AI 推断报告审查工具 ===\n")

    # 示例 1：一份有问题的 AI 报告
    bad_report = """
# 推断报告

我们对新用户和老用户的消费进行了 t 检验。

## 结果

- 新用户平均消费：315 元
- 老用户平均消费：300 元
- p 值：0.002

## 结论

- 新用户显著高于老用户（p<0.05）
- 建议针对新用户加大营销投入
"""

    print("=" * 60)
    print("原始 AI 报告：")
    print("=" * 60)
    print(bad_report)

    # 审查
    issues = review_inference_report(bad_report)

    print("=" * 60)
    print(f"审查结果：发现 {len(issues)} 个潜在问题\n")
    print("=" * 60)

    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['问题']}")
        print(f"   风险：{issue['风险']}")
        print(f"   建议：{issue['建议']}\n")

    # 生成改进版
    improved_report = generate_improved_report(bad_report, issues)

    print("=" * 60)
    print("改进后的报告框架：")
    print("=" * 60)
    print(improved_report)

    # 示例 2：一份合格的报告
    good_report = """
# 推断报告

我们对新用户和老用户的消费进行了差异分析。

## 结果

- 新用户平均消费：315 元
- 老用户平均消费：300 元
- 均值差：15 元
- 95% CI：[2.3, 27.7]
- Cohen's d：0.30（效应量小）
- Bootstrap 95% CI：[2.1, 28.2]
- 置换检验 p 值：0.003

## 结论

- 新用户显著高于老用户（p<0.05）
- 效应量小（Cohen's d = 0.30），实际意义有限
- Bootstrap CI 和置换检验结果一致，结论稳健
"""

    print("=" * 60)
    print("合格的 AI 报告示例：")
    print("=" * 60)
    print(good_report)

    # 审查合格报告
    good_issues = review_inference_report(good_report)

    print("=" * 60)
    if good_issues:
        print(f"审查结果：发现 {len(good_issues)} 个潜在问题")
    else:
        print("审查结果：✓ 未发现明显问题")
    print("=" * 60)

    # 审查清单总结
    print("\n" + "=" * 60)
    print("AI 推断报告审查清单总结：")
    print("=" * 60)
    print("1. 是否报告了置信区间？（均值、均值差、效应量的 95% CI）")
    print("2. CI 解释是否正确？（频率学派：方法的覆盖率）")
    print("3. 是否报告了效应量？（Cohen's d 及其解释）")
    print("4. 是否进行了稳健性检验？（Bootstrap 或置换检验）")
    print("5. 是否提供了不确定性可视化？（误差条、分布图）")
    print("6. p 值是否被过度解释？（'显著'不等于'重要'）")
    print("\n记住：不确定性量化的责任由你承担，AI 不会自动帮你做！")


if __name__ == "__main__":
    main()
