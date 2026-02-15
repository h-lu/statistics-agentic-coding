"""
示例：事后比较（Post-hoc Tests）——ANOVA 显著之后，找出"哪一对有差异"。

本例演示 Tukey HSD（Honestly Significant Difference）检验，用于在 ANOVA 显著后
进行两两比较。Tukey HSD 自动控制多重比较的假阳性率（FWER ≤ α）。

运行方式：python3 chapters/week_07/examples/03_posthoc_tests.py
预期输出：
  - stdout 输出 Tukey HSD 检验结果
  - 图表保存到 images/tukey_hsd_plot.png

反例：ANOVA 后用未校正的 t 检验做两两比较，会重新引入多重比较问题。
正确做法是使用 Tukey HSD 或 Bonferroni 校正。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


def simulate_multi_group_data(
    n_per_group: int,
    means: list[float],
    std: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    模拟多组正态数据

    参数:
        n_per_group: 每组样本量
        means: 各组真实均值
        std: 标准差（各组相同）
        seed: 随机种子

    返回:
        DataFrame: 包含 group 和 value 列
    """
    np.random.seed(seed)
    groups = [f"组{chr(65+i)}" for i in range(len(means))]  # A, B, C, D

    data_list = []
    for group, mean in zip(groups, means):
        values = np.random.normal(mean, std, n_per_group)
        data_list.append(pd.DataFrame({
            'group': [group] * n_per_group,
            'value': values
        }))

    return pd.concat(data_list, ignore_index=True)


def perform_anova(
    groups_data: list[np.ndarray],
    alpha: float = 0.05
) -> tuple[float, float, bool]:
    """
    执行单因素 ANOVA

    参数:
        groups_data: 各组数据列表
        alpha: 显著性水平

    返回:
        (f_stat, p_value, is_significant)
    """
    f_stat, p_value = stats.f_oneway(*groups_data)
    is_significant = p_value < alpha
    return f_stat, p_value, is_significant


def perform_tukey_hsd(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    执行 Tukey HSD 事后比较

    参数:
        df: 数据框
        group_col: 分组列名
        value_col: 数值列名
        alpha: 显著性水平

    返回:
        DataFrame: Tukey HSD 结果
    """
    tukey = pairwise_tukeyhsd(
        endog=df[value_col].dropna(),
        groups=df[group_col],
        alpha=alpha
    )

    # 提取结果到 DataFrame
    results_df = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    return results_df


def visualize_tukey_results(
    results_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    可视化 Tukey HSD 结果

    参数:
        results_df: Tukey HSD 结果
        output_path: 输出路径
    """
    font = setup_chinese_font()

    # 提取比较信息
    comparisons = results_df[['group1', 'group2', 'meandiff', 'p-adj', 'reject']].copy()
    comparisons['pair'] = comparisons['group1'] + ' vs ' + comparisons['group2']

    # 排序：均值差异从大到小
    comparisons = comparisons.sort_values('meandiff', ascending=True)

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 标记显著/不显著
    colors = ['#d62728' if r else '#7f7f7f' for r in comparisons['reject']]
    edgecolors = ['black' if r else '#999999' for r in comparisons['reject']]
    linewidths = [1.5 if r else 0.8 for r in comparisons['reject']]

    ax.barh(comparisons['pair'], comparisons['meandiff'], color=colors,
            edgecolor=edgecolors, linewidth=linewidths, alpha=0.85)
    ax.axvline(0, color='black', linestyle='-', linewidth=1.2)
    ax.set_xlabel('均值差异', fontsize=12)
    ax.set_ylabel('比较对', fontsize=12)
    ax.set_title('Tukey HSD 事后比较结果（红色=显著 p<0.05，灰色=不显著）', fontsize=14)

    # 添加 p 值标签
    for idx, row in comparisons.iterrows():
        x_pos = row['meandiff']
        y_pos = list(comparisons['pair']).index(row['pair'])
        offset = 0.02 if x_pos > 0 else -0.02
        ax.text(x_pos + offset, y_pos, f"p={row['p-adj']:.3f}",
                va='center', ha='left' if x_pos > 0 else 'right',
                fontsize=9, fontweight='bold' if row['reject'] else 'normal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def main() -> None:
    """运行事后比较演示"""
    font = setup_chinese_font()
    print(f"使用字体: {font}\n")

    print("=== Tukey HSD 事后比较演示 ===\n")

    # 1. 生成模拟数据：4 组，其中 D 组均值较高
    # A=50, B=50, C=50, D=53（小差异，可能检验不出）
    n_per_group = 100
    true_means = [50.0, 50.0, 50.0, 53.0]

    data = simulate_multi_group_data(n_per_group, true_means, std=10.0, seed=42)

    print("=== 描述统计 ===")
    print(data.groupby('group')['value'].agg(['count', 'mean', 'std']).round(2))
    print()

    # 2. 先执行 ANOVA
    groups_data = [group['value'].values for name, group in data.groupby('group')]
    f_stat, p_value, is_sig = perform_anova(groups_data, alpha=0.05)

    print("=== ANOVA 结果 ===")
    print(f"F 统计量: {f_stat:.4f}")
    print(f"p 值: {p_value:.4f}")

    if is_sig:
        print("结论: p < 0.05，拒绝原假设。至少有一对均值存在显著差异。")
        print("→ 需要做事后比较找出具体差异。\n")
    else:
        print("结论: p ≥ 0.05，无法拒绝原假设。")
        print("→ 即使做事后比较，也不太可能发现显著差异。\n")

    # 3. 执行 Tukey HSD
    tukey_results = perform_tukey_hsd(data, 'group', 'value', alpha=0.05)

    print("=== Tukey HSD 事后比较 ===")
    print(tukey_results.to_string(index=False))
    print()

    # 4. 提取显著比较
    significant_pairs = tukey_results[tukey_results['reject'] == True]

    print(f"=== 显著比较（校正后 p < 0.05） ===")
    if len(significant_pairs) > 0:
        for _, row in significant_pairs.iterrows():
            print(f"{row['group1']} vs {row['group2']}: "
                  f"差异={row['meandiff']:.2f}, "
                  f"p={row['p-adj']:.4f}, "
                  f"95% CI=[{row['lower']:.2f}, {row['upper']:.2f}]")
    else:
        print("未发现显著的两两比较差异。")
        print()
        print("阿码的困惑：ANOVA 说'有差异'，但 Tukey HSD 说'没有一对显著'？")
        print()
        print("这种情况很常见。可能的原因：")
        print("  1. ANOVA 的 F 检验在某些情况下比 Tukey HSD 更敏感")
        print("  2. '差异'是多组联合作用的结果，不是简单的两两差异")
        print("  3. 样本量不足，两两比较的功效较低")
        print()
        print("老潘的建议：")
        print('  "报告时诚实说明：ANOVA 边缘显著（p=xxx），但事后比较未发现具体的显著差异。"')

    print()

    # 5. 可视化
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    visualize_tukey_results(tukey_results, output_dir / 'tukey_hsd_plot.png')
    print("图表已保存到 images/tukey_hsd_plot.png")

    # 6. 对比未校正的 t 检验
    print("\n=== 坏例子：未校正的 t 检验 ===")
    print("如果用 6 次未校正的 t 检验比较所有对：")
    print(f"FWER = 1 - (1 - 0.05)^6 = {1 - (1 - 0.05)**6:.1%}")
    print()
    print("这意味着即使真实无差异，你也有 26% 的概率至少看到一个假阳性！")
    print("Tukey HSD 通过自动校正 p 值，控制 FWER ≤ 5%。")


if __name__ == "__main__":
    main()
