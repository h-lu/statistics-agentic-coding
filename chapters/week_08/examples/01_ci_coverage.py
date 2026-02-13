"""
示例：置信区间覆盖率模拟——理解 95% CI 的真正含义。

本例演示置信区间的频率学派解释：
- 重复抽样 100 次，构造 100 个 95% CI
- 约有 95 个区间会覆盖真值（绿色）
- 约有 5 个区间不会覆盖真值（橙色）

运行方式：python3 chapters/week_08/examples/01_ci_coverage.py
预期输出：
  - stdout 输出覆盖率（约 95%）
  - 生成 ci_coverage_simulation.png（100 个区间的可视化）
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def simulate_ci_coverage(true_mean: float = 300, true_std: float = 50,
                         n: int = 100, n_sim: int = 100,
                         conf_level: float = 0.95, seed: int = 42) -> tuple[list[tuple[float, float, float]], float]:
    """
    模拟置信区间的覆盖率。

    参数：
        true_mean: 真实均值
        true_std: 真实标准差
        n: 每次抽样的样本量
        n_sim: 模拟次数（构造多少个区间）
        conf_level: 置信水平（如 0.95）
        seed: 随机种子

    返回：
        intervals: 区间列表，每个元素是 (ci_low, ci_high, sample_mean)
        coverage_rate: 实际覆盖率
    """
    np.random.seed(seed)

    intervals = []
    coverage_count = 0

    for i in range(n_sim):
        # 从真实分布中抽样
        sample = np.random.normal(loc=true_mean, scale=true_std, size=n)

        # 计算样本均值和标准误
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        standard_error = sample_std / np.sqrt(n)

        # 计算 t 临界值
        df = n - 1
        t_critical = stats.t.ppf((1 + conf_level) / 2, df)

        # 构造置信区间
        margin_of_error = t_critical * standard_error
        ci_low = sample_mean - margin_of_error
        ci_high = sample_mean + margin_of_error

        intervals.append((ci_low, ci_high, sample_mean))

        # 检查是否覆盖真值
        if ci_low <= true_mean <= ci_high:
            coverage_count += 1

    coverage_rate = coverage_count / n_sim
    return intervals, coverage_rate


def plot_ci_coverage(intervals: list[tuple[float, float, float]],
                     true_mean: float,
                     coverage_rate: float,
                     output_path: str = 'ci_coverage_simulation.png') -> None:
    """
    可视化置信区间覆盖率。

    参数：
        intervals: 区间列表
        true_mean: 真实均值
        coverage_rate: 覆盖率
        output_path: 输出图片路径
    """
    plt.figure(figsize=(12, 6))
    plt.axvline(true_mean, color='red', linestyle='--', linewidth=2,
                label=f'真值 (μ={true_mean})')

    for i, (ci_low, ci_high, sample_mean) in enumerate(intervals):
        color = 'green' if ci_low <= true_mean <= ci_high else 'orange'
        plt.plot([ci_low, ci_high], [i, i], color=color, linewidth=2)
        plt.plot(sample_mean, i, 'bo', markersize=4)

    plt.xlabel('均值')
    plt.ylabel('模拟次数')
    plt.title(f'100 个 95% 置信区间（覆盖率：{coverage_rate:.1%}）\n'
              f'绿色=覆盖真值，橙色=未覆盖')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"图片已保存：{output_path}")


def main() -> None:
    """主函数：模拟并可视化置信区间覆盖率。"""
    print("=== 置信区间覆盖率模拟 ===\n")

    # 模拟 100 个 95% CI
    intervals, coverage_rate = simulate_ci_coverage(n_sim=100)

    print(f"理论覆盖率：95%")
    print(f"实际覆盖率：{coverage_rate:.1%}")
    print(f"覆盖区间数：{sum(1 for low, high, _ in intervals if low <= 300 <= high)}/100")

    # 解释
    print("\n频率学派解释：")
    print("  - 真值 μ=300 是固定不变的")
    print("  - 每次抽样得到不同的区间")
    print("  - 95% CI 说的是'长期覆盖率'，不是单次区间的概率")
    print("  - 你不知道当前区间是'对的'还是'错的'，只能相信这个方法")

    # 可视化
    plot_ci_coverage(intervals, true_mean=300, coverage_rate=coverage_rate)


if __name__ == "__main__":
    main()
