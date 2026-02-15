"""
示例：ANOVA 基础——从"多次 t 检验"到"一次 F 检验"。

本例演示单因素 ANOVA（方差分析）的原理和使用方法。ANOVA 通过比较"组间方差"和"组内方差"
来判断多组均值是否有差异，避免了多次 t 检验带来的多重比较问题。

运行方式：python3 chapters/week_07/examples/02_anova_basics.py
预期输出：
  - stdout 输出描述统计和 ANOVA 结果
  - 图表保存到 output/anova_barplot.png

坏例子：用 6 次 t 检验比较 4 组（A vs B, A vs C, A vs D, B vs C, B vs D, C vs D），
会导致 FWER ≈ 26.5%。正确做法是先用 ANOVA 判断"是否有任何差异"，再做事后比较。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from scipy import stats
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


def simulate_conversion_data(
    n_per_group: int,
    rates: list[float],
    seed: int = 42
) -> pd.DataFrame:
    """
    模拟多渠道转化率数据

    参数:
        n_per_group: 每组样本量
        rates: 各组真实转化率
        seed: 随机种子

    返回:
        DataFrame: 包含 channel 和 converted 列
    """
    np.random.seed(seed)
    channels = ['A', 'B', 'C', 'D']

    data_list = []
    for channel, rate in zip(channels, rates):
        conversions = np.random.binomial(1, rate, n_per_group)
        data_list.append(pd.DataFrame({
            'channel': [channel] * n_per_group,
            'converted': conversions
        }))

    return pd.concat(data_list, ignore_index=True)


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

    # 组间平方和：各组的均值与总均值之差的平方，加权组大小
    ss_between = 0
    for g in groups:
        group_data = df[df[group_col] == g][value_col].dropna()
        ss_between += len(group_data) * (group_data.mean() - grand_mean) ** 2

    # 总平方和：每个观测值与总均值之差的平方
    ss_total = ((all_data - grand_mean) ** 2).sum()

    return ss_between / ss_total


def perform_anova(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    alpha: float = 0.05
) -> dict:
    """
    执行单因素 ANOVA，返回完整结果

    参数:
        df: 数据框
        group_col: 分组列名
        value_col: 数值列名
        alpha: 显著性水平

    返回:
        包含 ANOVA 结果的字典
    """
    # 准备数据
    groups = df[group_col].unique()
    group_data = [df[df[group_col] == g][value_col].dropna() for g in groups]

    # 描述统计
    desc_stats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std'])

    # 前提假设检查
    # 1. 正态性（Shapiro-Wilk）
    normality_results = {}
    for g in groups:
        data = df[df[group_col] == g][value_col].dropna()
        _, p_value = stats.shapiro(data)
        normality_results[g] = p_value

    # 2. 方差齐性（Levene）
    _, p_levene = stats.levene(*group_data)

    # 选择检验方法
    all_normal = all(p > alpha for p in normality_results.values())

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

    # 计算效应量（仅参数检验）
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

    return {
        'desc_stats': desc_stats,
        'normality': normality_results,
        'levene_p': p_levene,
        'test_method': test_method,
        'is_parametric': is_parametric,
        'f_stat': f_stat,
        'p_value': p_value,
        'eta_squared': eta_squared,
        'effect_interp': effect_interp,
        'alpha': alpha
    }


def main() -> None:
    """运行 ANOVA 演示"""
    font = setup_chinese_font()
    print(f"使用字体: {font}\n")

    print("=== ANOVA 基础演示 ===\n")

    # 1. 生成模拟数据：4 个渠道的转化率
    # A: 10%, B: 10%, C: 10%, D: 12%（只有 D 不同）
    n_per_group = 500
    true_rates = [0.10, 0.10, 0.10, 0.12]

    data = simulate_conversion_data(n_per_group, true_rates)

    print("=== 描述统计 ===")
    print(data.groupby('channel')['converted'].agg(['mean', 'count']))
    print()

    # 2. 可视化
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x='channel', y='converted', errorbar='sd', ax=ax,
                color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('转化率', fontsize=12)
    ax.set_xlabel('渠道', fontsize=12)
    ax.set_title('4 个渠道的转化率比较（误差线：标准差）', fontsize=14)
    ax.set_ylim(0, 0.25)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'anova_barplot.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print("图表已保存到 images/anova_barplot.png\n")

    # 3. 执行 ANOVA
    result = perform_anova(data, 'channel', 'converted', alpha=0.05)

    print("=== 前提假设检查 ===")
    print("正态性（Shapiro-Wilk）：")
    for group, p_val in result['normality'].items():
        status = "✅ 可视为正态" if p_val > result['alpha'] else "❌ 非正态"
        print(f"  - {group}: p = {p_val:.4f} {status}")
    print(f"方差齐性（Levene）：p = {result['levene_p']:.4f} "
          f"{'✅ 方差齐性' if result['levene_p'] > result['alpha'] else '❌ 方差不齐'}")
    print()

    print("=== ANOVA 结果 ===")
    print(f"检验方法: {result['test_method']}")
    print(f"F 统计量: {result['f_stat']:.4f}")
    print(f"p 值: {result['p_value']:.4f}")
    if result['eta_squared'] is not None:
        print(f"η²（效应量）: {result['eta_squared']:.4f} ({result['effect_interp']})")
    print()

    # 4. 结论
    if result['p_value'] < result['alpha']:
        print(f"结论: p < {result['alpha']}，拒绝原假设。")
        print("至少有一对渠道的转化率存在显著差异。")
        print("\n注意：ANOVA 只说'有差异'，不说'哪一对有差异'。")
        print("下一节我们将学习 Tukey HSD 事后比较来找出具体差异。")
    else:
        print(f"结论: p ≥ {result['alpha']}，无法拒绝原假设。")
        print("各渠道转化率差异不具有统计显著性。")

    # 5. 小北的困惑
    print("\n=== 小北的困惑 ===")
    print("小北：ANOVA 是分析'方差'还是'均值'？")
    print("老潘：这是 ANOVA 最反直觉的地方——它用'方差'来检验'均值'。")
    print()
    print("F 统计量 = 组间方差 / 组内方差")
    print("  - 组间方差：各组均值之间的差异（如果 H0 为真，应该很小）")
    print("  - 组内方差：各组内部数据的波动（这是'噪音'，无法避免）")
    print()
    print(f"本例中 F = {result['f_stat']:.2f}")
    if result['f_stat'] > 1:
        print("F > 1，说明组间差异大于组内噪音，可能有显著差异。")
    else:
        print("F ≈ 1，说明组间差异 ≈ 组内差异，各组均值可能相等。")


if __name__ == "__main__":
    main()
