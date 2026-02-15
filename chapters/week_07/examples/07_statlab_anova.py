"""
示例：StatLab 多组比较报告生成——从"假设检验"到"ANOVA + 事后比较 + 校正"。

本例是 StatLab 超级线在 Week 07 的入口脚本，在上周（假设检验）基础上，
加入多组比较章节：ANOVA、Tukey HSD 事后比较、多重比较校正策略。

运行方式：python3 chapters/week_07/examples/07_statlab_anova.py
预期输出：
  - stdout 输出多组比较报告片段
  - 报告片段保存到 output/anova_sections.md

与上周对比：
  - 上周：两两 t 检验（可能忽略多重比较问题）
  - 本周：ANOVA + Tukey HSD 校正（明确说明校正策略）
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path


def calculate_eta_squared(df: pd.DataFrame, group_col: str, value_col: str) -> float:
    """
    计算 η²（eta squared）：ANOVA 的效应量

    η² = 组间平方和 / 总平方和
    表示组间变异占总变异的比例

    参数:
        df: 数据框
        group_col: 分组列名
        value_col: 数值列名

    返回:
        eta_squared: 效应量，范围 [0, 1]
    """
    groups = df[group_col].unique()
    all_data = df[value_col].dropna()
    grand_mean = all_data.mean()

    # 组间平方和
    ss_between = 0
    for g in groups:
        group_data = df[df[group_col] == g][value_col].dropna()
        ss_between += len(group_data) * (group_data.mean() - grand_mean) ** 2

    # 总平方和
    ss_total = ((all_data - grand_mean) ** 2).sum()

    return ss_between / ss_total


def one_way_anova(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    alpha: float = 0.05
) -> tuple[str | None, str | None]:
    """
    对多组数据进行单因素 ANOVA，返回完整的检验报告

    参数:
        df: 数据框
        group_col: 分组列名
        value_col: 数值列名
        alpha: 显著性水平

    返回:
        (markdown报告, 错误信息) - 成功时错误信息为 None
    """
    groups = df[group_col].unique()

    if len(groups) < 3:
        return None, f"ANOVA 需要至少 3 组，当前检测到 {len(groups)} 组。请使用两样本 t 检验。"

    # 描述统计
    desc = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max'])

    # 准备数据
    group_data = [df[df[group_col] == g][value_col].dropna() for g in groups]

    # 前提假设检查
    # 1. 正态性（Shapiro-Wilk）
    norm_results = {}
    for g in groups:
        data = df[df[group_col] == g][value_col].dropna()
        # 样本量大时用 Kolmogorov-Smirnov，否则用 Shapiro-Wilk
        if len(data) > 5000:
            _, p_value = stats.kstest(data, 'norm')
        else:
            _, p_value = stats.shapiro(data)
        norm_results[g] = p_value

    # 2. 方差齐性（Levene）
    _, p_levene = stats.levene(*group_data)

    # 选择检验方法
    all_normal = all(p > alpha for p in norm_results.values())

    if all_normal and p_levene > alpha:
        # 假设满足：用标准 ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        test_method = "单因素 ANOVA（假设满足）"
        is_parametric = True
    else:
        # 假设不满足：用 Kruskal-Wallis
        f_stat, p_value = stats.kruskal(*group_data)
        test_method = "Kruskal-Wallis 检验（非参数替代）"
        is_parametric = False

    # 计算 η²（效应量，仅参数检验）
    if is_parametric:
        eta_squared = calculate_eta_squared(df, group_col, value_col)

        # 解释效应量
        if eta_squared < 0.01:
            effect_interp = "极小效应"
        elif eta_squared < 0.06:
            effect_interp = "小效应"
        elif eta_squared < 0.14:
            effect_interp = "中等效应"
        else:
            effect_interp = "大效应"
    else:
        eta_squared = None
        effect_interp = "N/A（非参数检验）"

    # 生成报告
    md = f"### 多组比较：{value_col} 按 {group_col} 分组\n\n"
    md += f"**假设**：各组的 {value_col} 均值是否全部相等\n\n"

    md += f"**描述统计**\n\n"
    md += desc.round(2).to_markdown()
    md += "\n\n"

    md += f"**前提假设检查**\n\n"
    md += f"- 正态性检验：\n"
    for g, p in norm_results.items():
        status = "✅" if p > alpha else "❌"
        md += f"  - {g}: p = {p:.4f} {status}\n"
    md += f"- 方差齐性（Levene）：p = {p_levene:.4f} "
    md += f"{'✅' if p_levene > alpha else '❌'}\n"
    md += f"- **检验方法**：{test_method}\n\n"

    md += f"**ANOVA 结果**\n\n"
    md += f"- 统计量：{f_stat:.4f}\n"
    md += f"- p 值：{p_value:.4f}\n"
    if eta_squared is not None:
        md += f"- η²（效应量）：{eta_squared:.4f}（{effect_interp}）\n"

    if p_value < alpha:
        md += f"- **结论**：p < {alpha}，**拒绝原假设**。各组均值存在显著差异。\n\n"
    else:
        md += f"- **结论**：p ≥ {alpha}，**无法拒绝原假设**。各组均值差异不显著。\n\n"

    # 如果 ANOVA 显著且是参数检验，做 Tukey HSD 事后比较
    if p_value < alpha and is_parametric:
        md += f"**Tukey HSD 事后比较**（95% 置信区间）\n\n"
        md += f"*注：所有 p 值均已使用 Tukey HSD 方法校正，控制 family-wise error rate ≤ {alpha}*\n\n"

        tukey = pairwise_tukeyhsd(
            endog=df[value_col].dropna(),
            groups=df[group_col],
            alpha=0.05
        )

        tukey_results = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0]
        )

        # 只显示显著的比较
        significant = tukey_results[tukey_results['reject'] == True]

        if len(significant) > 0:
            md += significant[['group1', 'group2', 'meandiff', 'p-adj', 'lower', 'upper']].to_markdown()
            md += "\n\n"
        else:
            md += "未发现显著的两两比较差异（校正后 p < 0.05）。\n\n"
            md += "*这可能是因为：*\n"
            md += "* 1. ANOVA 的 F 检验在某些情况下比 Tukey HSD 更敏感\n"
            md += "* 2. '差异'是多组联合作用的结果，不是简单的两两差异\n"
            md += "* 3. 样本量不足，两两比较的功效较低\n\n"

    return md, None


def generate_anova_section(
    df: pd.DataFrame,
    tests: list[tuple[str, str]]
) -> str:
    """
    生成多组比较的 Markdown 片段

    参数:
        df: 数据框
        tests: 检验列表，每个元素是 (group_col, value_col) 元组

    返回:
        Markdown 格式的多组比较报告
    """
    md = ["## 多组比较\n\n"]
    md.append("以下对多组数据进行方差分析（ANOVA）和事后比较。\n\n")
    md.append("**校正策略说明**：")
    md.append("所有事后比较均使用 Tukey HSD 校正，控制 family-wise error rate ≤ 0.05。")
    md.append("这意味着当我们做多组两两比较时，自动调整显著性阈值以控制假阳性率。\n\n")

    for group_col, value_col in tests:
        anova_md, error = one_way_anova(df, group_col, value_col)
        if error:
            md.append(f"### {value_col} 按 {group_col}\n\n")
            md.append(f"⚠️ {error}\n\n")
        else:
            md.append(anova_md)

    return "".join(md)


def main() -> None:
    """运行 StatLab 多组比较报告生成"""
    print("=== StatLab 多组比较报告生成演示 ===\n")

    # 加载数据
    penguins = sns.load_dataset("penguins")
    print(f"加载数据：{len(penguins)} 条企鹅记录\n")

    # 定义要运行的检验
    tests_to_run = [
        ("species", "bill_length_mm"),  # 3 组：Adelie, Chinstrap, Gentoo
        ("island", "bill_length_mm"),   # 3 组：Biscoe, Dream, Torgersen
        ("species", "bill_depth_mm"),   # 3 组
        ("species", "flipper_length_mm"),  # 3 组
    ]

    print("将要运行的多组比较：")
    for group_col, value_col in tests_to_run:
        groups = penguins[group_col].unique()
        print(f"  - {value_col} 按 {group_col} 分组 ({len(groups)} 组)")
    print()

    # 生成多组比较报告
    anova_md = generate_anova_section(penguins, tests_to_run)

    print("=== StatLab 多组比较报告片段 ===\n")
    print(anova_md[:2000])  # 打印前 2000 字符
    print("...")
    print("\n[完整报告已保存到文件]")

    # 写入文件
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'anova_sections.md'
    output_path.write_text(anova_md)
    print(f"\n报告片段已保存到 {output_path}")

    # 输出与上周的对比
    print("\n=== StatLab 进度对比 ===\n")
    print("| 上周（Week 06） | 本周（Week 07） |")
    print("|------------------|------------------|")
    print("| 假设检验（p 值 + 效应量 + 假设检查） | 以上全部 + **多组比较（ANOVA + 事后比较 + 校正）** |")
    print("| 两两 t 检验 | 多组 ANOVA + Tukey HSD 校正 |")
    print("| 可能忽略多重比较问题 | 明确说明校正策略 |")
    print()

    # 老潘的点评
    print("=== 老潘的点评 ===\n")
    print('"现在你不仅告诉别人"3 个物种是否有差异"，')
    print('还告诉别人"哪一对有差异、效应量多大、是否做了校正"。')
    print('这就是从"赌 p 值"到"科学的多组比较"的关键一步。"')
    print()


if __name__ == "__main__":
    main()
