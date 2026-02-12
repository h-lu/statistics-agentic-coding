#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：第一类错误、第二类错误与功效分析

本例演示假设检验中的两类错误及其权衡关系：
1. 第一类错误率（α）模拟：在 H0 为真时拒绝 H0 的概率
2. 功效（1-β）分析：在 H0 为假时正确拒绝 H0 的概率
3. α 与 β 的权衡关系（跷跷板效应）
4. 样本量对功效的影响

运行方式：python3 chapters/week_06/examples/03_type12_errors.py
预期输出：终端显示模拟结果、生成三张可视化图

核心概念：
- 第一类错误（α）：假阳性，H0 为真但拒绝 H0
- 第二类错误（β）：假阴性，H0 为假但保留 H0
- 功效（1-β）：正确检测出真实差异的能力
- α 与 β 是跷跷板：降低一个会提高另一个

作者：StatLab Week 06
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Tuple, Dict


# =============================================================================
# 两类错误模拟
# =============================================================================

def simulate_type_errors(
    n_sim: int = 10000,
    n_sample: int = 50,
    true_diff: float = 0,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    模拟两类错误。

    参数：
        n_sim: 模拟次数
        n_sample: 每组样本量
        true_diff: 真实均值差异（H0 为真时=0，H0 为假时≠0）
        alpha: 显著性水平
        seed: 随机种子

    返回：
        tuple: (第一类错误率, 第二类错误率, 功效)
    """
    np.random.seed(seed)

    type_i_errors = 0  # 第一类错误计数
    type_ii_errors = 0  # 第二类错误计数

    for _ in range(n_sim):
        # 生成数据（对照组均值=100，实验组均值=100+true_diff）
        control = np.random.normal(loc=100, scale=15, size=n_sample)
        treatment = np.random.normal(loc=100 + true_diff, scale=15, size=n_sample)

        # t 检验
        _, p_value = stats.ttest_ind(treatment, control)

        # 决策
        if p_value < alpha:
            # 拒绝 H0
            if true_diff == 0:
                # H0 为真，但拒绝了 → 第一类错误
                type_i_errors += 1
        else:
            # 保留 H0
            if true_diff != 0:
                # H0 为假，但保留了 → 第二类错误
                type_ii_errors += 1

    # 计算错误率
    if true_diff == 0:
        type_i_rate = type_i_errors / n_sim
        type_ii_rate = 0.0
        power = np.nan
    else:
        type_i_rate = 0.0
        type_ii_rate = type_ii_errors / n_sim
        power = 1 - type_ii_rate

    return type_i_rate, type_ii_rate, power


# =============================================================================
# 实验 1：第一类错误率
# =============================================================================

def experiment_1_type_i_error() -> Dict:
    """
    实验 1：验证第一类错误率 = α

    当 H0 为真时（true_diff=0），拒绝 H0 的概率应该恰好等于 α。
    这是对假设检验方法正确性的基本检验。
    """
    print("=" * 70)
    print("实验 1：第一类错误率（H0 为真）")
    print("=" * 70)
    print("\n在 H0 为真时（两组均值真的相等），拒绝 H0 的概率应该等于 α")

    alphas = [0.01, 0.05, 0.10]
    type_i_rates = []

    for alpha in alphas:
        type_i_rate, _, _ = simulate_type_errors(
            n_sim=10000,
            n_sample=50,
            true_diff=0,  # H0 为真
            alpha=alpha
        )
        type_i_rates.append(type_i_rate)
        print(f"  α={alpha:.2f}：第一类错误率 = {type_i_rate:.4f}（理论值 = {alpha:.2f}）")

    # 可视化
    plot_type_i_error(alphas, type_i_rates)

    return {'alphas': alphas, 'type_i_rates': type_i_rates}


def plot_type_i_error(alphas: list, type_i_rates: list) -> None:
    """可视化第一类错误率"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alphas, type_i_rates, 'o-', linewidth=2, markersize=10, label='模拟结果')
    ax.plot([0, 0.12], [0, 0.12], 'r--', alpha=0.7, linewidth=2, label='理论线（α）')
    ax.set_xlabel('显著性水平 α', fontsize=12)
    ax.set_ylabel('第一类错误率', fontsize=12)
    ax.set_title('第一类错误率 = α（验证假设检验的正确性）',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.12)
    ax.set_ylim(0, 0.12)

    # 保存图表
    output_path = Path('checkpoint/type_i_error_rate.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[可视化] 图表已保存：{output_path}")
    plt.close()


# =============================================================================
# 实验 2：功效分析
# =============================================================================

def experiment_2_power_analysis() -> Dict:
    """
    实验 2：功效随样本量增加而提升

    功效（power = 1-β）是在 H0 为假时正确拒绝 H0 的概率。
    样本量越大，功效越高（检测出真实差异的能力越强）。
    """
    print("\n" + "=" * 70)
    print("实验 2：功效分析（H0 为假，真实差异=8）")
    print("=" * 70)
    print("\n在 H0 为假时（两组均值真的不同），功效随样本量增加而提升")

    sample_sizes = [20, 50, 100, 200, 500]
    power_results = []
    true_diff = 8.0

    for n in sample_sizes:
        _, _, power = simulate_type_errors(
            n_sim=10000,
            n_sample=n,
            true_diff=true_diff,  # H0 为假
            alpha=0.05
        )
        power_results.append(power)
        print(f"  样本量 n={n:3d}：功效 = {power:.3f}（检测差异={true_diff}）")

    # 可视化
    plot_power_curve(sample_sizes, power_results)

    return {'sample_sizes': sample_sizes, 'powers': power_results}


def plot_power_curve(sample_sizes: list, powers: list) -> None:
    """可视化功效曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sample_sizes, powers, 'o-', linewidth=2, markersize=10,
            color='green', label='功效')
    ax.axhline(0.8, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label='推荐功效 ≥ 80%')
    ax.axhline(0.9, color='orange', linestyle='--', alpha=0.7, linewidth=2,
               label='优秀功效 ≥ 90%')

    # 标注 80% 功效对应的样本量
    for i, (n, p) in enumerate(zip(sample_sizes, powers)):
        if p >= 0.8:
            ax.annotate(f'n={n}\n功效={p:.2f}',
                       xy=(n, p), xytext=(10, 20),
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            break

    ax.set_xlabel('样本量', fontsize=12)
    ax.set_ylabel('功效（1-β）', fontsize=12)
    ax.set_title('功效随样本量增加而提升',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # 保存图表
    output_path = Path('checkpoint/power_vs_sample_size.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[可视化] 图表已保存：{output_path}")
    plt.close()


# =============================================================================
# 实验 3：α 与 β 的权衡
# =============================================================================

def experiment_3_alpha_beta_tradeoff() -> Dict:
    """
    实验 3：α 与 β 的权衡关系（跷跷板效应）

    降低 α 会提高 β（降低功效），反之亦然。
    这是假设检验中的基本权衡。
    """
    print("\n" + "=" * 70)
    print("实验 3：α 与 β 的权衡（样本量=50，真实差异=8）")
    print("=" * 70)
    print("\n降低 α 会提高 β（降低功效），这是跷跷板效应")

    alphas = np.linspace(0.01, 0.10, 10)
    powers = []
    true_diff = 8.0
    n_sample = 50

    for alpha in alphas:
        _, _, power = simulate_type_errors(
            n_sim=10000,
            n_sample=n_sample,
            true_diff=true_diff,
            alpha=alpha
        )
        powers.append(power)

    # 可视化
    plot_alpha_beta_tradeoff(alphas, powers)

    return {'alphas': alphas, 'powers': powers}


def plot_alpha_beta_tradeoff(alphas: np.ndarray, powers: list) -> None:
    """可视化 α 与 β 的权衡关系"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alphas, powers, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.axhline(0.8, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label='推荐功效 ≥ 80%')
    ax.axvline(0.05, color='orange', linestyle='--', alpha=0.7, linewidth=2,
               label='常用 α=0.05')

    # 标注平衡点
    idx = np.argmin(np.abs(np.array(powers) - 0.8))
    ax.annotate(f'α={alphas[idx]:.3f}\n功效={powers[idx]:.2f}',
               xy=(alphas[idx], powers[idx]), xytext=(-80, 20),
               textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax.set_xlabel('第一类错误率 α', fontsize=12)
    ax.set_ylabel('功效（1-β）', fontsize=12)
    ax.set_title('α 与 β 的权衡关系（跷跷板效应）',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    # 保存图表
    output_path = Path('checkpoint/alpha_beta_tradeoff.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[可视化] 图表已保存：{output_path}")
    plt.close()


# =============================================================================
# 样本量计算
# =============================================================================

def calculate_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0
) -> int:
    """
    计算独立样本 t 检验的所需样本量。

    参数：
        effect_size: Cohen's d（标准化效应量）
        alpha: 显著性水平
        power: 目标功效
        ratio: 两组样本量比例

    返回：
        int: 每组需要的样本量
    """
    # Z 值
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # 双尾检验
    z_beta = stats.norm.ppf(power)

    # 样本量公式（简化版）
    n_per_group = 2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)

    return int(np.ceil(n_per_group))


def demonstrate_sample_size_calculation() -> None:
    """演示不同效应量下的样本量需求"""
    print("\n" + "=" * 70)
    print("样本量计算：检测不同效应量需要多少样本？")
    print("=" * 70)

    # 不同效应量下的样本量需求
    print("\n不同效应量下的样本量需求（α=0.05, 功效=0.8）：")
    print(f"{'效应量':<10} {'定性':<15} {'每组样本量':<15}")
    print("-" * 40)

    for d in [0.2, 0.5, 0.8]:
        n = calculate_sample_size(effect_size=d)
        magnitude = '小' if d < 0.3 else '中' if d < 0.7 else '大'
        print(f"d={d:<9.1f} {magnitude:<15} {n:<15}")

    # 示例：检测中等效应
    n_needed = calculate_sample_size(effect_size=0.5)
    print(f"\n示例：检测中等效应（d=0.5），每组需要 {n_needed} 个样本")
    print(f"总样本量：{n_needed * 2}")


# =============================================================================
# 错误矩阵可视化
# =============================================================================

def plot_error_matrix() -> None:
    """可视化假设检验的错误矩阵"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # 创建 2x2 矩阵
    matrix_data = [
        ['第一类错误 (Type I)\n假阳性\n概率=α', '正确决策\n真阳性\n概率=1-α'],
        ['正确决策\n真阴性\n概率=1-β', '第二类错误 (Type II)\n假阴性\n概率=β']
    ]

    # 绘制矩阵
    table = ax.table(cellText=matrix_data,
                    rowLabels=['H0 为真（无差异）', 'H0 为假（有差异）'],
                    colLabels=['保留 H0', '拒绝 H0'],
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)

    # 着色
    colors = [['#ffcccc', '#ccffcc'], ['#ccffcc', '#ffcccc']]
    for i in range(2):
        for j in range(2):
            table[(i+1, j)].set_facecolor(colors[i][j])

    ax.axis('off')
    ax.set_title('假设检验的四种可能结果',
                fontsize=14, fontweight='bold', pad=20)

    # 保存图表
    output_path = Path('checkpoint/error_matrix.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[可视化] 错误矩阵已保存：{output_path}")
    plt.close()


# =============================================================================
# 反例：常见的错误理解
# =============================================================================

def common_misconceptions() -> None:
    """常见的两类错误理解误区"""
    print("\n" + "=" * 70)
    print("常见误区：小北的困惑")
    print("=" * 70)

    misconceptions = [
        {
            "问题": "小北：'我把 α 从 0.05 降到 0.01，检验就更严格了！'",
            "问题所在": "只考虑了第一类错误，忽略了第二类错误",
            "正确理解": "降低 α 会提高 β（降低功效），可能漏掉真实差异"
        },
        {
            "问题": "阿码：'p=0.06 > 0.05，所以 H0 是对的！'",
            "问题所在": "混淆了'无法拒绝 H0'和'证明 H0'",
            "正确理解": "p>0.05 只能说证据不足，不能证明没有差异"
        },
        {
            "问题": "小北：'功效 80% 太低了，我要 100%！'",
            "问题所在": "追求 100% 功效需要无限大样本量",
            "正确理解": "80% 是工程上常用的平衡点，90% 更好但成本更高"
        }
    ]

    for i, m in enumerate(misconceptions, 1):
        print(f"\n误区 {i}：{m['问题']}")
        print(f"  问题所在：{m['问题所在']}")
        print(f"  正确理解：{m['正确理解']}")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数：运行两类错误与功效分析示例"""
    # 实验 1：第一类错误率
    experiment_1_type_i_error()

    # 实验 2：功效分析
    experiment_2_power_analysis()

    # 实验 3：α 与 β 权衡
    experiment_3_alpha_beta_tradeoff()

    # 样本量计算
    demonstrate_sample_size_calculation()

    # 错误矩阵可视化
    plot_error_matrix()

    # 常见误区
    common_misconceptions()

    print("\n" + "=" * 70)
    print("要点总结")
    print("=" * 70)
    print("1. 第一类错误（α）= 假阳性率，第二类错误（β）= 假阴性率")
    print("2. 功效（1-β）= 正确检测出真实差异的能力")
    print("3. α 与 β 是跷跷板：降低一个会提高另一个")
    print("4. 功效随样本量增加而提升：大样本更容易检测出小效应")
    print("5. 工程上通常要求功效 ≥ 80%，但这需要足够的样本量")
    print("6. 不要盲目追求'超严格'的 α，要在两类错误间找平衡")
    print("=" * 70)


if __name__ == "__main__":
    main()
