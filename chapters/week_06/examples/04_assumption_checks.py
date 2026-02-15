"""
示例：t 检验的前提假设检查——正态性、方差齐性。

本例演示如何检查 t 检验的前提假设：
1. 正态性检验（Q-Q 图 + Shapiro-Wilk 检验）
2. 方差齐性检验（Levene 检验）
3. 根据假设检查结果选择合适的检验方法

运行方式：python3 chapters/week_06/examples/04_assumption_checks.py
预期输出：
  - stdout 输出正态性检验结果、方差齐性检验结果
  - 图表保存到 output/qq_plot_comparison.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
    import matplotlib.font_manager as fm
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


def check_normality(data: np.ndarray, group_name: str, alpha: float = 0.05) -> tuple[bool, float]:
    """
    检验数据是否来自正态分布（Shapiro-Wilk 检验）

    H0: 数据来自正态分布
    H1: 数据不来自正态分布

    参数：
        data: 数据
        group_name: 组名（用于输出）
        alpha: 显著性水平

    返回：
        (是否正态, p值)
    """
    stat, p_value = stats.shapiro(data)
    print(f"{group_name}:")
    print(f"  Shapiro-Wilk 统计量: {stat:.4f}")
    print(f"  p 值: {p_value:.4f}")

    is_normal = p_value > alpha
    if is_normal:
        print(f"  结论: p > {alpha}，无法拒绝原假设。数据可视为正态分布。")
    else:
        print(f"  结论: p < {alpha}，拒绝原假设。数据不服从正态分布。")
    print()

    return is_normal, p_value


def check_homogeneity(group1: np.ndarray, group2: np.ndarray,
                     group1_name: str, group2_name: str,
                     alpha: float = 0.05) -> tuple[bool, float]:
    """
    检验两组方差是否相等（Levene 检验）

    H0: 两组方差相等
    H1: 两组方差不等

    参数：
        group1, group2: 两组数据
        group1_name, group2_name: 组名
        alpha: 显著性水平

    返回：
        (方差齐性是否成立, p值)
    """
    stat, p_value = stats.levene(group1, group2)

    print(f"Levene 检验: {group1_name} vs {group2_name}")
    print(f"  统计量: {stat:.4f}")
    print(f"  p 值: {p_value:.4f}")

    is_homogeneous = p_value > alpha
    if is_homogeneous:
        print(f"  结论: p > {alpha}，无法拒绝原假设。可假设方差齐性。")
    else:
        print(f"  结论: p < {alpha}，拒绝原假设。方差不齐。")
    print()

    return is_homogeneous, p_value


def mann_whitney_test(group1: np.ndarray, group2: np.ndarray,
                     group1_name: str, group2_name: str,
                     alpha: float = 0.05) -> tuple[float, float]:
    """
    非参数检验：比较两组分布是否相同（Mann-Whitney U 检验）

    H0: 两组分布相同
    H1: 两组分布不同

    参数：
        group1, group2: 两组数据
        group1_name, group2_name: 组名
        alpha: 显著性水平

    返回：
        (U统计量, p值)
    """
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

    print(f"Mann-Whitney U 检验: {group1_name} vs {group2_name}")
    print(f"  U 统计量: {u_stat:.4f}")
    print(f"  p 值: {p_value:.4f}")

    if p_value < alpha:
        print(f"  结论: p < {alpha}，拒绝原假设。两组分布存在显著差异。")
    else:
        print(f"  结论: p ≥ {alpha}，无法拒绝原假设。")
    print()

    return u_stat, p_value


def main() -> None:
    """运行前提假设检查演示"""
    setup_chinese_font()
    np.random.seed(42)

    print("=== t 检验前提假设检查演示 ===\n")

    # 生成两组数据
    # A 组：正态分布
    group_a = np.random.normal(loc=100, scale=15, size=200)

    # B 组：右偏分布（如收入数据）
    group_b = np.random.exponential(scale=30, size=200)

    print("数据生成：")
    print("  A 组：正态分布（均值=100，标准差=15，n=200）")
    print("  B 组：指数分布（尺度=30，n=200，右偏）\n")

    # 1. 创建 Q-Q 图
    print("正在生成 Q-Q 图...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    stats.probplot(group_a, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q 图：A 组（正态分布）", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    stats.probplot(group_b, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q 图：B 组（右偏分布）", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'qq_plot_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Q-Q 图已保存到 {output_dir}/qq_plot_comparison.png\n")

    # 2. 正态性检验
    print("=== 1. 正态性检验（Shapiro-Wilk）===\n")
    is_normal_a, p_norm_a = check_normality(group_a, "A 组（正态）")
    is_normal_b, p_norm_b = check_normality(group_b, "B 组（偏态）")

    # 3. 方差齐性检验
    print("=== 2. 方差齐性检验（Levene）===\n")
    equal_var, p_levene = check_homogeneity(group_a, group_b, "A 组", "B 组")

    # 4. 根据 Levene 检验结果选择 t 检验类型
    print("=== 3. t 检验（根据方差齐性选择）===\n")
    if equal_var:
        # 方差齐性：用标准 t 检验
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=True)
        test_method = "Student's t 检验（假设方差齐性）"
    else:
        # 方差不齐：用 Welch's t 检验
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
        test_method = "Welch's t 检验（不假设方差齐性）"

    print(f"检验方法：{test_method}")
    print(f"t 统计量: {t_stat:.4f}")
    print(f"p 值: {p_value:.4f}\n")

    # 5. 非参数检验（当数据非正态时）
    print("=== 4. 非参数检验（数据非正态时）===\n")
    mann_whitney_test(group_a, group_b, "A 组", "B 组")

    # 6. 总结建议
    print("=== 检验方法选择建议 ===\n")
    print("根据假设检查结果：")

    if is_normal_a and is_normal_b:
        print("  ✓ 正态性满足")
    else:
        print("  ✗ 正态性不满足（B 组右偏）")

    if equal_var:
        print("  ✓ 方差齐性满足")
    else:
        print("  ✗ 方差不齐")

    print()
    print("推荐方法：")
    if is_normal_a and is_normal_b:
        if equal_var:
            print("  → Student's t 检验（所有假设都满足）")
        else:
            print("  → Welch's t 检验（正态但方差不齐）")
    else:
        print("  → Mann-Whitney U 检验（非参数，不依赖正态性假设）")

    print()
    print("注意：")
    print("  - 轻微违反假设时，t 检验仍然比较稳健（robust）")
    print("  - 样本量大时（n > 30），中心极限定理会发挥作用")
    print("  - Q-Q 图和实际判断比单纯依赖 p 值更重要")


if __name__ == "__main__":
    main()
