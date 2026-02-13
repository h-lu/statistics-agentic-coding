"""
示例：贝叶斯均值估计——可信区间（Credible Interval）。

本例演示贝叶斯框架下的均值估计：
- 使用正态-正态共轭模型（Normal-Normal Conjugate）
- 后验分布：参数（均值）的概率分布
- 95% 可信区间：参数有 95% 的概率落在区间内

与频率学派 CI 的区别：
- 频率学派：参数固定，CI 是方法的覆盖率
- 贝叶斯学派：参数随机，可信区间是参数的概率

运行方式：python3 chapters/week_08/examples/07_bayesian_inference.py
预期输出：
  - stdout 输出后验均值、95% 可信区间、贝叶斯解释
  - 生成 bayesian_posterior.png（后验分布直方图）
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def bayesian_mean_estimation(data: np.ndarray,
                             prior_mean: float = 300,
                             prior_std: float = 50,
                             likelihood_std: float = 50,
                             n_samples: int = 10000,
                             seed: int = 42) -> tuple[float, float, np.ndarray]:
    """
    用贝叶斯方法估计均值（正态-正态共轭）。

    参数：
        data: 观测数据
        prior_mean: 先验均值
        prior_std: 先验标准差
        likelihood_std: 似然标准差（数据标准差）
        n_samples: 后验样本数
        seed: 随机种子

    返回：
        posterior_mean: 后验均值
        posterior_std: 后验标准差
        posterior_samples: 后验分布样本
    """
    np.random.seed(seed)
    n = len(data)
    sample_mean = np.mean(data)

    # 后验参数（正态-正态共轭的解析解）
    # 后验精度 = 先验精度 + 数据精度
    posterior_precision = 1 / prior_std**2 + n / likelihood_std**2
    posterior_var = 1 / posterior_precision
    posterior_std = np.sqrt(posterior_var)

    # 后验均值 = 加权平均（先验 vs 数据）
    posterior_mean = posterior_var * (prior_mean / prior_std**2 +
                                      n * sample_mean / likelihood_std**2)

    # 从后验分布中采样
    posterior_samples = np.random.normal(posterior_mean, posterior_std, n_samples)

    return posterior_mean, posterior_std, posterior_samples


def plot_posterior_distribution(posterior_samples: np.ndarray,
                                 posterior_mean: float,
                                 ci_low: float,
                                 ci_high: float,
                                 output_path: str = 'bayesian_posterior.png') -> None:
    """
    可视化贝叶斯后验分布。

    参数：
        posterior_samples: 后验分布样本
        posterior_mean: 后验均值
        ci_low: 95% 可信区间下界
        ci_high: 95% 可信区间上界
        output_path: 输出图片路径
    """
    plt.figure(figsize=(10, 6))
    plt.hist(posterior_samples, bins=50, density=True, alpha=0.7, color='steelblue')

    # 标注后验均值和可信区间
    plt.axvline(posterior_mean, color='red', linestyle='-',
                linewidth=3, label=f'后验均值 ({posterior_mean:.2f})')
    plt.axvline(ci_low, color='orange', linestyle='--',
                linewidth=2, label=f'2.5% ({ci_low:.2f})')
    plt.axvline(ci_high, color='orange', linestyle='--',
                linewidth=2, label=f'97.5% ({ci_high:.2f})')

    # 填充 95% 可信区间
    plt.axvspan(ci_low, ci_high, alpha=0.2, color='orange',
                label='95% 可信区间')

    plt.xlabel('均值（元）')
    plt.ylabel('密度')
    plt.title('贝叶斯后验分布：均值的概率分布\n（参数是随机变量）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"图片已保存：{output_path}")


def main() -> None:
    """主函数：贝叶斯均值估计。"""
    print("=== 贝叶斯均值估计（可信区间） ===\n")

    # 模拟数据：新用户消费
    np.random.seed(42)
    new_users = np.random.normal(loc=315, scale=50, size=100)

    print("数据描述：")
    print(f"  样本量：n={len(new_users)}")
    print(f"  样本均值：{np.mean(new_users):.2f}")
    print(f"  样本标准差：{np.std(new_users, ddof=1):.2f}")

    # 先验设定
    prior_mean = 300
    prior_std = 50

    print(f"\n先验设定：")
    print(f"  先验均值：{prior_mean}（基于历史数据，认为消费均值约 300 元）")
    print(f"  先验标准差：{prior_std}（先验较宽松，允许 200-400 之间）")

    # 贝叶斯估计
    posterior_mean, posterior_std, posterior_samples = bayesian_mean_estimation(
        new_users, prior_mean=prior_mean, prior_std=prior_std, likelihood_std=50
    )

    print(f"\n后验分布（{len(posterior_samples)} 个样本）：")
    print(f"  后验均值：{posterior_mean:.2f}")
    print(f"  后验标准差：{posterior_std:.2f}")

    # 计算 95% 可信区间
    ci_low = np.percentile(posterior_samples, 2.5)
    ci_high = np.percentile(posterior_samples, 97.5)

    print(f"  95% 可信区间：[{ci_low:.2f}, {ci_high:.2f}]")

    # 贝叶斯解释
    print(f"\n贝叶斯解释：")
    print(f"  均值有 95% 的概率在 [{ci_low:.2f}, {ci_high:.2f}] 内")
    print(f"  （这是参数的概率分布，不是方法的覆盖率）")

    # 对比频率学派
    print(f"\n频率学派 vs 贝叶斯学派：")
    print(f"  频率学派：")
    print(f"    - 参数是固定但未知的值")
    print(f"    - 95% CI：方法在重复抽样下有 95% 的概率覆盖真值")
    print(f"    - 不能说'参数有 95% 的概率在区间内'")
    print(f"  贝叶斯学派：")
    print(f"    - 参数是随机变量，有概率分布")
    print(f"    - 95% 可信区间：参数有 95% 的概率落在区间内")
    print(f"    - 解释更直观，但需要选择先验")

    # 后验均值 vs 先验均值 vs 样本均值
    sample_mean = np.mean(new_users)
    print(f"\n数据如何更新信念：")
    print(f"  先验均值：{prior_mean:.2f}（初始信念）")
    print(f"  样本均值：{sample_mean:.2f}（数据证据）")
    print(f"  后验均值：{posterior_mean:.2f}（更新后的信念）")

    # 计算权重
    prior_weight = (1 / prior_std**2) / (1 / prior_std**2 + len(new_users) / 50**2)
    data_weight = 1 - prior_weight
    print(f"  先验权重：{prior_weight * 100:.1f}%")
    print(f"  数据权重：{data_weight * 100:.1f}%")

    if data_weight > prior_weight:
        print(f"  → 数据主导后验（样本量足够大）")

    # 可视化
    plot_posterior_distribution(posterior_samples, posterior_mean, ci_low, ci_high)


if __name__ == "__main__":
    main()
