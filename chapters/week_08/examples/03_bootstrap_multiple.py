"""
示例：Bootstrap 多统计量置信区间——均值、中位数、标准差。

本例演示如何用 Bootstrap 同时构造多个统计量的 CI：
- 均值（Mean）
- 中位数（Median）
- 标准差（Std）

对比不同统计量的不确定性：均值 CI 更窄（更精确），中位数 CI 更宽（更稳健）。

运行方式：python3 chapters/week_08/examples/03_bootstrap_multiple.py
预期输出：
  - stdout 输出三个统计量的 Bootstrap CI
  - 生成 bootstrap_multiple_stats.png（三个子图对比）
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def bootstrap_ci_multiple(data: np.ndarray,
                          n_bootstrap: int = 10000,
                          conf_level: float = 0.95,
                          seed: int = 42) -> dict:
    """
    用 Bootstrap 构造多个统计量的置信区间。

    参数：
        data: 原始数据
        n_bootstrap: Bootstrap 次数
        conf_level: 置信水平
        seed: 随机种子

    返回：
        results: 字典，包含各统计量的 Bootstrap 分布和 CI
    """
    np.random.seed(seed)
    n = len(data)

    # 初始化
    boot_means = np.empty(n_bootstrap)
    boot_medians = np.empty(n_bootstrap)
    boot_stds = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # 有放回地抽取 Bootstrap 样本
        boot_sample = np.random.choice(data, size=n, replace=True)

        # 计算统计量
        boot_means[i] = np.mean(boot_sample)
        boot_medians[i] = np.median(boot_sample)
        boot_stds[i] = np.std(boot_sample, ddof=1)

    # 计算 CI（百分位数法）
    alpha = 1 - conf_level
    percentiles = [100 * alpha / 2, 100 * (1 - alpha / 2)]

    results = {
        'mean': {
            'observed': np.mean(data),
            'distribution': boot_means,
            'ci': np.percentile(boot_means, percentiles)
        },
        'median': {
            'observed': np.median(data),
            'distribution': boot_medians,
            'ci': np.percentile(boot_medians, percentiles)
        },
        'std': {
            'observed': np.std(data, ddof=1),
            'distribution': boot_stds,
            'ci': np.percentile(boot_stds, percentiles)
        }
    }

    return results


def plot_multiple_bootstrap(results: dict,
                            output_path: str = 'bootstrap_multiple_stats.png') -> None:
    """
    可视化三个统计量的 Bootstrap 分布。

    参数：
        results: bootstrap_ci_multiple 返回的结果
        output_path: 输出图片路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    stat_names = ['mean', 'median', 'std']
    stat_labels = ['均值（元）', '中位数（元）', '标准差（元）']

    for i, (stat, label) in enumerate(zip(stat_names, stat_labels)):
        ax = axes[i]
        distribution = results[stat]['distribution']
        observed = results[stat]['observed']
        ci_low, ci_high = results[stat]['ci']

        # 直方图
        ax.hist(distribution, bins=50, density=True, alpha=0.7, color='steelblue')

        # 标注 CI
        ax.axvline(ci_low, color='red', linestyle='--', linewidth=2, label=f'2.5% ({ci_low:.1f})')
        ax.axvline(ci_high, color='red', linestyle='--', linewidth=2, label=f'97.5% ({ci_high:.1f})')
        ax.axvline(observed, color='green', linestyle='-', linewidth=2, label=f'观测值 ({observed:.1f})')

        ax.set_xlabel(label)
        ax.set_ylabel('密度')
        ax.set_title(f'{label} 的 Bootstrap 分布')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"图片已保存：{output_path}")


def main() -> None:
    """主函数：Bootstrap 多统计量 CI。"""
    print("=== Bootstrap 多统计量置信区间 ===\n")

    # 模拟数据：新用户消费
    np.random.seed(42)
    new_users = np.random.normal(loc=315, scale=50, size=100)

    print("数据描述：")
    print(f"  样本量：n={len(new_users)}")
    print(f"  均值：{np.mean(new_users):.2f}")
    print(f"  中位数：{np.median(new_users):.2f}")
    print(f"  标准差：{np.std(new_users, ddof=1):.2f}")

    # Bootstrap 分析
    results = bootstrap_ci_multiple(new_users)

    # 打印结果
    print(f"\nBootstrap 95% CI（10000 次重采样）：")
    print(f"{'统计量':8s} {'观测值':>10s} {'95% CI 下界':>12s} {'95% CI 上界':>12s} {'CI 宽度':>10s}")
    print("-" * 60)

    for stat in ['mean', 'median', 'std']:
        observed = results[stat]['observed']
        ci_low, ci_high = results[stat]['ci']
        ci_width = ci_high - ci_low
        print(f"{stat.upper():8s} {observed:10.2f} {ci_low:12.2f} {ci_high:12.2f} {ci_width:10.2f}")

    # 对比分析
    mean_ci_width = results['mean']['ci'][1] - results['mean']['ci'][0]
    median_ci_width = results['median']['ci'][1] - results['median']['ci'][0]

    print(f"\n对比分析：")
    print(f"  均值 CI 宽度：{mean_ci_width:.2f}")
    print(f"  中位数 CI 宽度：{median_ci_width:.2f}")
    print(f"  中位数 CI 更宽约 {(median_ci_width / mean_ci_width - 1) * 100:.1f}%")
    print(f"\n解释：均值更精确（方差小），中位数更稳健（不受极端值影响）")

    # 可视化
    plot_multiple_bootstrap(results)


if __name__ == "__main__":
    main()
