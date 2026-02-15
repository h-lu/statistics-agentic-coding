"""
示例：中心极限定理（CLT）模拟——验证"为什么样本均值的分布总是钟形"。

本例演示中心极限定理的神奇之处：
1. 总体是指数分布（高度右偏）
2. 随着样本量增加（n=1, 5, 30, 100），样本均值的分布逐渐变成钟形
3. 即使总体不是正态的，样本均值的分布也近似正态

运行方式：python3 chapters/week_05/examples/03_clt_simulation.py
预期输出：stdout 输出模拟结果 + 图表保存到 output/clt_simulation.png

核心概念：
- 中心极限定理（CLT）：无论总体分布是什么形状，样本量足够大时，样本均值的分布近似正态
- 总体分布 vs 抽样分布：原始数据的分布 vs 统计量的分布
- "够大"的样本量：通常 n≥30，但取决于总体分布的偏态程度

关键发现：
- 小北："如果总体是右偏的（如收入），样本均值的分布还是钟形吗？"
- 阿码："中心极限定理是不是就是'正态分布魔法'？"
- 老潘："平均值的魔法：极端值被'稀释'掉了"

反直觉的事实：
- 总体：指数分布（极度右偏）
- n=1：样本均值的分布 = 总体分布（右偏）
- n=5：开始有点对称
- n=30：已经很像钟形
- n=100：几乎完美拟合正态分布

这就是为什么正态分布在统计学中如此重要！
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.font_manager as fm


def setup_chinese_font() -> str:
    """配置中文字体"""
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


def main() -> None:
    """主函数：模拟中心极限定理"""
    setup_chinese_font()

    # 使用现代 NumPy 随机数生成器
    rng = np.random.default_rng(seed=42)

    print("=" * 60)
    print("中心极限定理（CLT）模拟")
    print("=" * 60)

    # 总体：指数分布（高度右偏）
    population_size = 100000
    population = rng.exponential(scale=1.0, size=population_size)

    print(f"\n总体分布：指数分布（高度右偏）")
    print(f"  - 总体均值：{population.mean():.3f}")
    print(f"  - 总体标准差：{population.std():.3f}")
    print(f"  - 总体偏度：{stats.skew(population):.3f}（>0 表示右偏）")

    # 模拟：从总体中重复抽样，计算样本均值
    sample_sizes = [1, 5, 30, 100]
    n_simulations = 1000

    print(f"\n开始模拟（重复 {n_simulations} 次）...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, n in enumerate(sample_sizes):
        sample_means = []
        for _ in range(n_simulations):
            sample = rng.choice(population, size=n, replace=False)
            sample_means.append(sample.mean())

        sample_means = np.array(sample_means)

        # 计算统计量
        mean_of_means = sample_means.mean()
        std_of_means = sample_means.std()
        theoretical_se = population.std() / np.sqrt(n)  # 标准误的理论值

        print(f"\nn={n}:")
        print(f"  - 样本均值分布的均值：{mean_of_means:.3f}")
        print(f"  - 样本均值分布的标准差（标准误）：{std_of_means:.3f}")
        print(f"  - 标准误理论值：{theoretical_se:.3f}")

        # 画分布
        axes[i].hist(sample_means, bins=30, edgecolor="black",
                     alpha=0.7, color='steelblue', density=True)

        # 叠加正态分布曲线（用于对比）
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        normal_curve = stats.norm.pdf(x, loc=mean_of_means, scale=std_of_means)
        axes[i].plot(x, normal_curve, "r--", linewidth=2, label="正态分布")

        axes[i].set_xlabel("样本均值")
        axes[i].set_ylabel("密度")
        axes[i].set_title(f"样本量 n={n} 的样本均值分布")
        axes[i].legend()

    plt.tight_layout()

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "clt_simulation.png", dpi=150,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    print(f"\n图表已保存到 {output_dir}/clt_simulation.png")

    # 额外可视化：总体分布 vs 样本均值分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：总体分布
    axes[0].hist(population, bins=50, edgecolor="black",
                 alpha=0.7, color='orange', density=True)
    axes[0].set_xlabel("值")
    axes[0].set_ylabel("密度")
    axes[0].set_title("总体分布（指数分布，高度右偏）")

    # 右图：n=30 的样本均值分布
    rng = np.random.default_rng(seed=42)
    sample_means_n30 = []
    for _ in range(n_simulations):
        sample = rng.choice(population, size=30, replace=False)
        sample_means_n30.append(sample.mean())
    sample_means_n30 = np.array(sample_means_n30)

    axes[1].hist(sample_means_n30, bins=30, edgecolor="black",
                 alpha=0.7, color='steelblue', density=True)
    x = np.linspace(sample_means_n30.min(), sample_means_n30.max(), 100)
    normal_curve = stats.norm.pdf(x, loc=sample_means_n30.mean(),
                                   scale=sample_means_n30.std())
    axes[1].plot(x, normal_curve, "r--", linewidth=2, label="正态分布")
    axes[1].set_xlabel("样本均值")
    axes[1].set_ylabel("密度")
    axes[1].set_title("样本均值分布（n=30，近似正态）")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "clt_population_vs_sampling.png", dpi=150,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    print(f"图表已保存到 {output_dir}/clt_population_vs_sampling.png")

    # 核心结论
    print("\n" + "=" * 60)
    print("核心结论")
    print("=" * 60)
    print("1. 中心极限定理：无论总体分布是什么形状，")
    print("   样本量足够大时，样本均值的分布近似正态")
    print("2. '够大'的样本量：")
    print("   - 总体对称时：n=5 可能就够了")
    print("   - 总体右偏时：n=30 通常够用")
    print("   - 总体极度偏态或有离群值：可能需要 n=50 或更大")
    print("3. 标准误 = 标准差 / sqrt(样本量)")
    print("4. 为什么正态分布重要：")
    print("   不是因为原始数据总是正态的")
    print("   而是因为统计量（如均值）的抽样分布近似正态")
    print("\n阿码：'所以只要样本量够大，我就能用正态分布的方法？'")
    print("老潘：'对。但先用直方图看总体分布——Week 02 的技能现在有新用途了！'")


if __name__ == "__main__":
    main()
