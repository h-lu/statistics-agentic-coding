#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：p 值的直观理解——通过置换检验模拟

本例演示什么是 p 值：在原假设 H0 为真时，观察到当前或更极端数据的概率。
通过置换检验（permutation test）模拟 H0 下的差异分布，并与 scipy 的 ttest_ind 结果对比。

运行方式：python3 chapters/week_06/examples/01_p_value_simulation.py
预期输出：终端显示 p 值对比、生成 p_value_intuition.png 可视化图

核心概念：
- p 值不是"结论为真的概率"，而是"在 H0 为真时看到当前数据的概率"
- 置换检验通过随机打乱标签模拟 H0 下的抽样分布
- 红色区域面积 = p 值

作者：StatLab Week 06
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def simulate_p_value_intuition(
    n_control: int = 50,
    n_treatment: int = 50,
    true_diff: float = 8.0,
    seed: int = 42
) -> dict:
    """
    通过置换检验理解 p 值的直观含义。

    参数：
        n_control: 对照组样本量
        n_treatment: 实验组样本量
        true_diff: 真实均值差异（用于生成数据）
        seed: 随机种子

    返回：
        dict: 包含观察差异、t检验结果、模拟结果等
    """
    np.random.seed(seed)

    # 生成数据（模拟 A/B 测试）
    control = np.random.normal(loc=100, scale=15, size=n_control)
    treatment = np.random.normal(loc=100 + true_diff, scale=15, size=n_treatment)

    # 观察到的差异
    observed_diff = np.mean(treatment) - np.mean(control)

    print("=" * 70)
    print("p 值直观理解：置换检验模拟")
    print("=" * 70)
    print(f"\n[数据概览]")
    print(f"  对照组：n={len(control)}, 均值={np.mean(control):.2f}, 标准差={np.std(control, ddof=1):.2f}")
    print(f"  实验组：n={len(treatment)}, 均值={np.mean(treatment):.2f}, 标准差={np.std(treatment, ddof=1):.2f}")
    print(f"  观察差异：{observed_diff:.2f}")

    # 方法 1：使用 scipy 进行 t 检验
    print(f"\n[方法 1：scipy t 检验]")
    t_stat, p_value_scipy = stats.ttest_ind(treatment, control)
    print(f"  t 统计量：{t_stat:.4f}")
    print(f"  双尾 p 值：{p_value_scipy:.6f}")

    # 方法 2：用置换检验模拟理解 p 值
    print(f"\n[方法 2：置换检验模拟]")
    n_simulations = 10000

    # 合并数据
    combined = np.concatenate([control, treatment])
    n_ctrl, n_treat = len(control), len(treatment)

    # 置换检验：在 H0 为真时（两组均值相等），随机打乱标签
    simulated_diffs = []
    for i in range(n_simulations):
        # 随机打乱数据
        shuffled = np.random.permutation(combined)
        # 分配到两组
        sim_ctrl = shuffled[:n_ctrl]
        sim_treat = shuffled[n_ctrl:]
        # 计算差异
        simulated_diffs.append(np.mean(sim_treat) - np.mean(sim_ctrl))

        if (i + 1) % 2000 == 0:
            print(f"  进度：{i + 1}/{n_simulations}")

    simulated_diffs = np.array(simulated_diffs)

    # 计算 p 值：在 H0 下，观察到当前或更极端差异的概率
    # 双尾检验：取绝对值比较
    p_value_sim = (np.abs(simulated_diffs) >= np.abs(observed_diff)).mean()

    print(f"\n[结果对比]")
    print(f"  置换检验 p 值：{p_value_sim:.6f}")
    print(f"  scipy t 检验 p 值：{p_value_scipy:.6f}")
    print(f"  差异：{abs(p_value_sim - p_value_scipy):.6f}")

    # 可视化
    plot_p_value_intuition(simulated_diffs, observed_diff, p_value_sim)

    return {
        'observed_diff': observed_diff,
        'p_value_scipy': p_value_scipy,
        'p_value_sim': p_value_sim,
        't_stat': t_stat,
        'control_mean': float(np.mean(control)),
        'treatment_mean': float(np.mean(treatment))
    }


def plot_p_value_intuition(
    simulated_diffs: np.ndarray,
    observed_diff: float,
    p_value: float
) -> None:
    """
    可视化 p 值的直观理解。

    图表说明：
    - 蓝色直方图：H0 下的差异分布（通过置换检验模拟）
    - 红色虚线：观察到的差异
    - 红色阴影区域：p 值（极端区域）

    参数：
        simulated_diffs: 模拟的差异分布
        observed_diff: 观察到的差异
        p_value: 计算出的 p 值
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制 H0 下的差异分布
    ax.hist(simulated_diffs, bins=50, density=True, alpha=0.7,
            color='steelblue', label='H0 下的差异分布（置换检验）')

    # 标记观察差异（双尾）
    ax.axvline(observed_diff, color='red', linestyle='--', linewidth=2,
               label=f'观察差异={observed_diff:.2f}')
    ax.axvline(-observed_diff, color='red', linestyle='--', linewidth=2, alpha=0.7)

    # 标记极端区域（p 值）
    extreme_right = simulated_diffs >= observed_diff
    extreme_left = simulated_diffs <= -observed_diff

    if extreme_right.any():
        ax.hist(simulated_diffs[extreme_right], bins=50, density=True,
                color='red', alpha=0.5)
    if extreme_left.any():
        ax.hist(simulated_diffs[extreme_left], bins=50, density=True,
                color='red', alpha=0.5)

    # 添加 p 值注释
    ax.text(0.05, 0.95, f'p 值（红色区域面积）= {p_value:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('均值差异', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title('p 值的直观理解：在 H0 下观察到当前或更极端数据的概率',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 保存图表
    output_path = Path('checkpoint/p_value_intuition.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[可视化] 图表已保存：{output_path}")
    plt.close()


def demonstrate_p_value_misconceptions() -> None:
    """
    演示 p 值的常见误区。

    这个函数展示了对 p 值的错误理解，
    帮助读者避免常见陷阱。
    """
    print("\n" + "=" * 70)
    print("p 值常见误区辨析")
    print("=" * 70)

    misconceptions = [
        {
            "误区": "p 值是 H0 为真的概率",
            "正确理解": "p 值是 P(data|H0)，不是 P(H0|data)",
            "例子": "p=0.03 不是说 H0 只有 3% 可能是对的"
        },
        {
            "误区": "p=0.049 显著，p=0.051 不显著，有本质区别",
            "正确理解": "0.05 只是人为设定的阈值，两者差异微乎其微",
            "例子": "不应过度依赖 0.05 的魔法线"
        },
        {
            "误区": "p 值越小，效应越大",
            "正确理解": "p 值受样本量影响，大样本下微小效应也显著",
            "例子": "n=100000 时，d=0.01 也可能 p<0.001"
        },
        {
            "误区": "p>0.05 证明 H0 是对的",
            "正确理解": "不显著只能说明证据不足，不能证明没有差异",
            "例子": "可能功效不足，需要更大样本量"
        }
    ]

    for i, m in enumerate(misconceptions, 1):
        print(f"\n误区 {i}：{m['误区']}")
        print(f"  ✓ 正确理解：{m['正确理解']}")
        print(f"  例子：{m['例子']}")


# =============================================================================
# 反例：常见的 p 值误用
# =============================================================================

def bad_p_value_interpretation() -> None:
    """
    反例：常见的 p 值错误解释

    这个函数展示了小北常犯的错误，
    帮助读者避免类似问题。
    """
    print("\n" + "=" * 70)
    print("反例：小北的 p 值误用")
    print("=" * 70)

    # 模拟数据
    np.random.seed(42)
    group_a = np.random.normal(100, 15, 50)
    group_b = np.random.normal(105, 15, 50)

    t_stat, p_value = stats.ttest_ind(group_a, group_b)

    print("\n[实验结果]")
    print(f"  组 A 均值：{np.mean(group_a):.2f}")
    print(f"  组 B 均值：{np.mean(group_b):.2f}")
    print(f"  t 检验：t={t_stat:.3f}, p={p_value:.4f}")

    print("\n[小北的错误结论]")
    print(f'  ❌ "p={p_value:.4f} < 0.05，所以 H0 为真的概率只有 {p_value*100:.1f}%"')
    print(f'  ❌ "p 值很小，说明差异很大"')
    print(f'  ❌ "既然 p<0.05，结论就一定是对的"')

    print("\n[正确的理解]")
    print(f"  ✓ p={p_value:.4f} 表示：如果 H0 为真（两组均值相等），")
    print(f"    观察到当前或更极端差异的概率是 {p_value*100:.2f}%")
    print(f"  ✓ p 值不能告诉你效应大小，需要计算 Cohen's d")
    effect_size = (np.mean(group_b) - np.mean(group_a)) / np.sqrt(
        ((len(group_a)-1)*group_a.var(ddof=1) + (len(group_b)-1)*group_b.var(ddof=1)) /
        (len(group_a) + len(group_b) - 2)
    )
    print(f"  ✓ Cohen's d = {effect_size:.3f}（{'小' if abs(effect_size)<0.2 else '中' if abs(effect_size)<0.5 else '大'}效应）")
    print(f"  ✓ 统计显著 ≠ 实际显著，需要结合业务判断")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数：运行 p 值模拟示例"""
    # 1. p 值直观理解
    result = simulate_p_value_intuition()

    # 2. 常见误区辨析
    demonstrate_p_value_misconceptions()

    # 3. 反例：错误解释
    bad_p_value_interpretation()

    print("\n" + "=" * 70)
    print("要点总结")
    print("=" * 70)
    print("1. p 值 = P(在H0为真时看到当前数据|H0)，不是 P(H0|data)")
    print("2. 置换检验通过随机打乱标签模拟 H0 下的抽样分布")
    print("3. p 值受样本量影响，不能单独判断实际意义")
    print("4. p<0.05 不等于'结论为真'，需要结合效应量和置信区间")
    print("5. 避免 p-hacking：不要尝试多种方法只报告显著的结果")
    print("=" * 70)


if __name__ == "__main__":
    main()
