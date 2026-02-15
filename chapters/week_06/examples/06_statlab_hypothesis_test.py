"""
示例：StatLab 假设检验报告生成——从不确定性量化到假设检验。

本例是 StatLab 超级线在 Week 06 的入口脚本，在上周（不确定性量化）基础上，
加入假设检验章节：p 值、效应量、前提假设检查。

运行方式：python3 chapters/week_06/examples/06_statlab_hypothesis_test.py
预期输出：
  - stdout 输出假设检验报告片段
  - 报告片段保存到 output/hypothesis_test_sections.md
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from pathlib import Path


def two_group_test(df: pd.DataFrame, group_col: str, value_col: str,
                   alpha: float = 0.05) -> tuple[str | None, str | None]:
    """
    对两组数据进行 t 检验，返回完整的检验报告

    参数：
        df: 数据框
        group_col: 分组列名
        value_col: 数值列名
        alpha: 显著性水平

    返回：
        (markdown报告, 错误信息) - 成功时错误信息为 None
    """
    groups = df[group_col].unique()

    if len(groups) != 2:
        return None, f"当前只支持两组比较，检测到 {len(groups)} 组"

    group_a, group_b = groups[0], groups[1]
    data_a = df[df[group_col] == group_a][value_col].dropna()
    data_b = df[df[group_col] == group_b][value_col].dropna()

    # 描述统计
    desc_a = {"n": len(data_a), "mean": data_a.mean(), "std": data_a.std(ddof=1)}
    desc_b = {"n": len(data_b), "mean": data_b.mean(), "std": data_b.std(ddof=1)}

    # 前提假设检查
    # 1. 正态性（Shapiro-Wilk）
    _, p_norm_a = stats.shapiro(data_a)
    _, p_norm_b = stats.shapiro(data_b)

    # 2. 方差齐性（Levene）
    _, p_levene = stats.levene(data_a, data_b)

    # 选择检验方法
    if p_norm_a > alpha and p_norm_b > alpha and p_levene > alpha:
        # 假设满足：用标准 t 检验
        t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=True)
        test_method = "Student's t 检验（假设方差齐性）"
        is_nonparam = False
    elif p_norm_a > alpha and p_norm_b > alpha:
        # 正态但方差不齐：用 Welch's t 检验
        t_stat, p_value = stats.ttest_ind(data_a, data_b, equal_var=False)
        test_method = "Welch's t 检验（不假设方差齐性）"
        is_nonparam = False
    else:
        # 非正态：用 Mann-Whitney U 检验
        u_stat, p_value = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        t_stat = u_stat
        test_method = "Mann-Whitney U 检验（非参数）"
        is_nonparam = True

    # 计算效应量（Cohen's d）
    if not is_nonparam:
        pooled_std = np.sqrt(((len(data_a) - 1) * data_a.var(ddof=1) +
                             (len(data_b) - 1) * data_b.var(ddof=1)) /
                            (len(data_a) + len(data_b) - 2))
        cohens_d = (data_a.mean() - data_b.mean()) / pooled_std

        # 解释效应量
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_interp = "小效应"
        elif abs_d < 0.5:
            effect_interp = "小到中等效应"
        elif abs_d < 0.8:
            effect_interp = "中等效应"
        else:
            effect_interp = "大效应"

        effect_report = f"Cohen's d: {cohens_d:.4f}（{effect_interp}）"
    else:
        effect_report = "N/A（非参数检验，建议报告秩差异）"

    # 计算 95% 置信区间（Bootstrap）
    n_bootstrap = 1000
    np.random.seed(42)
    boot_diffs = []
    for _ in range(n_bootstrap):
        boot_a = np.random.choice(data_a, size=len(data_a), replace=True)
        boot_b = np.random.choice(data_b, size=len(data_b), replace=True)
        boot_diffs.append(boot_a.mean() - boot_b.mean())
    boot_diffs = np.array(boot_diffs)
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    # 生成报告
    md = f"### 假设检验：{value_col} 按 {group_col} 分组\n\n"
    md += f"**假设**：{group_a} 与 {group_b} 的 {value_col} 均值是否存在显著差异\n\n"

    md += f"**描述统计**\n\n"
    md += f"| 组别 | 样本量 | 均值 | 标准差 |\n"
    md += f"|------|--------|------|--------|\n"
    md += f"| {group_a} | {desc_a['n']} | {desc_a['mean']:.2f} | {desc_a['std']:.2f} |\n"
    md += f"| {group_b} | {desc_b['n']} | {desc_b['mean']:.2f} | {desc_b['std']:.2f} |\n\n"

    md += f"**前提假设检查**\n\n"
    md += f"- 正态性（Shapiro-Wilk）：\n"
    md += f"  - {group_a}: p = {p_norm_a:.4f} {'✅ 可视为正态' if p_norm_a > alpha else '❌ 非正态'}\n"
    md += f"  - {group_b}: p = {p_norm_b:.4f} {'✅ 可视为正态' if p_norm_b > alpha else '❌ 非正态'}\n"
    md += f"- 方差齐性（Levene）：p = {p_levene:.4f} {'✅ 方差齐性' if p_levene > alpha else '❌ 方差不齐'}\n\n"

    md += f"**检验结果**\n\n"
    md += f"- 检验方法：{test_method}\n"
    md += f"- 统计量：{t_stat:.4f}\n"
    md += f"- p 值：{p_value:.4f}\n"
    if p_value < alpha:
        md += f"- **结论**：p < {alpha}，**拒绝原假设**。两组均值存在显著差异。\n\n"
    else:
        md += f"- **结论**：p ≥ {alpha}，**无法拒绝原假设**。差异不具有统计显著性。\n\n"

    md += f"**效应量**\n\n"
    md += f"- {effect_report}\n\n"

    md += f"**95% 置信区间**（Bootstrap）\n\n"
    md += f"- 差异均值：{data_a.mean() - data_b.mean():.2f}\n"
    md += f"- 95% CI: [{ci_low:.2f}, {ci_high:.2f}]\n"
    if ci_low > 0:
        md += f"- ✅ 区间不包含 0，表明 {group_a} 的均值显著高于 {group_b}\n\n"
    elif ci_high < 0:
        md += f"- ✅ 区间不包含 0，表明 {group_b} 的均值显著高于 {group_a}\n\n"
    else:
        md += f"- ⚠️ 区间包含 0，表明差异可能不显著\n\n"

    return md, None


def generate_hypothesis_test_section(df: pd.DataFrame, tests: list[tuple[str, str]]) -> str:
    """
    生成假设检验的 Markdown 片段

    参数：
        df: 数据框
        tests: 检验列表，每个元素是 (group_col, value_col) 元组

    返回：
        Markdown 格式的假设检验报告
    """
    md = ["## 假设检验\n\n"]
    md.append("以下对上周假设清单中的关键假设进行检验。\n\n")

    for group_col, value_col in tests:
        test_md, error = two_group_test(df, group_col, value_col)
        if error:
            md.append(f"**{value_col} 按 {group_col}**\n\n")
            md.append(f"⚠️ {error}\n\n")
        else:
            md.append(test_md)

    return "".join(md)


def main() -> None:
    """运行 StatLab 假设检验报告生成"""
    print("=== StatLab 假设检验报告生成演示 ===\n")

    # 加载数据
    penguins = sns.load_dataset("penguins")
    print(f"加载数据：{len(penguins)} 条企鹅记录\n")

    # 定义要运行的检验
    tests_to_run = [
        ("sex", "bill_length_mm"),
        ("island", "bill_length_mm"),  # 这个会报错（>2 组）
    ]

    print("将要运行的检验：")
    for group_col, value_col in tests_to_run:
        print(f"  - {value_col} 按 {group_col} 分组")
    print()

    # 生成假设检验报告
    hypothesis_md = generate_hypothesis_test_section(penguins, tests_to_run)

    print("=== StatLab 假设检验报告片段 ===\n")
    print(hypothesis_md)

    # 写入文件
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'hypothesis_test_sections.md'
    output_path.write_text(hypothesis_md)
    print(f"报告片段已保存到 {output_path}")

    # 输出与上周的对比
    print("\n=== StatLab 进度对比 ===\n")
    print("| 上周（Week 05） | 本周（Week 06） |")
    print("|------------------|------------------|")
    print("| 不确定性量化 | 假设检验（p 值 + 效应量 + 假设检查） |")
    print("| 置信区间 | 前提假设检查 + 检验方法选择 |")
    print("| 无法回答'是否显著' | 能执行假设检验，审查 AI 结论 |")


if __name__ == "__main__":
    main()
