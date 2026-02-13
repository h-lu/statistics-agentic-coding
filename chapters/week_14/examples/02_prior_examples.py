"""
示例：先验类型示例——无信息、弱信息、强信息先验

运行方式：python3 chapters/week_14/examples/02_prior_examples.py
预期输出：三种先验的分布形状、均值和方差对比
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_14"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置中文字体（避免乱码）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False


def create_uninformative_prior() -> stats.beta:
    """
    创建无信息先验（Uniform Prior）

    Beta(1, 1) = 均匀分布
    任何转化率都"先验地"等可能
    """
    return stats.beta(1, 1)


def create_weakly_informative_prior() -> stats.beta:
    """
    创建弱信息先验

    Beta(2, 40)：均值约 4.7%，方差较大
    编码"通常转化率在 5% 上下"但不强
    """
    return stats.beta(2, 40)


def create_informative_prior(historical_mean: float = 0.052,
                           historical_std: float = 0.01) -> stats.beta:
    """
    创建强信息先验（基于历史数据）

    假设过去 50 次 A/B 测试的平均转化率是 5.2%，标准差 1%
    把这个历史分布编码为先验

    参数:
        historical_mean: 历史平均转化率
        historical_std: 历史转化率标准差

    返回:
        Beta 分布对象
    """
    # Beta 分布参数转换
    # 给定均值 mu 和方差 sigma²，求 alpha 和 beta：
    # alpha = mu * (mu * (1-mu) / sigma² - 1)
    # beta = (1-mu) * (mu * (1-mu) / sigma² - 1)
    common = historical_mean * (1 - historical_mean) / (historical_std ** 2) - 1
    alpha = historical_mean * common
    beta = (1 - historical_mean) * common

    return stats.beta(alpha, beta)


# 坏例子：不合理的极端先验
def bad_extreme_prior() -> stats.beta:
    """
    反例：极端乐观的先验

    Beta(10, 10)：均值 50%，这在转化率场景中不合理

    问题：
    1. 与历史经验严重不符
    2. 会让后验分布偏向不合理的值
    3. 缺乏领域知识支持
    """
    return stats.beta(10, 10)


def plot_prior_comparison():
    """绘制三种先验的对比图"""
    x = np.linspace(0, 0.15, 500)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 无信息先验
    prior_uninformative = create_uninformative_prior()
    axes[0].plot(x, prior_uninformative.pdf(x), 'b-', linewidth=2, label='无信息先验')
    axes[0].fill_between(x, prior_uninformative.pdf(x), alpha=0.3)
    axes[0].set_title('无信息先验 Beta(1,1)', fontsize=12)
    axes[0].set_xlabel('转化率')
    axes[0].set_ylabel('概率密度')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 弱信息先验
    prior_weak = create_weakly_informative_prior()
    axes[1].plot(x, prior_weak.pdf(x), 'g-', linewidth=2, label='弱信息先验')
    axes[1].fill_between(x, prior_weak.pdf(x), alpha=0.3)
    axes[1].set_title('弱信息先验 Beta(2,40)', fontsize=12)
    axes[1].set_xlabel('转化率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 强信息先验
    prior_strong = create_informative_prior()
    axes[2].plot(x, prior_strong.pdf(x), 'r-', linewidth=2, label='强信息先验')
    axes[2].fill_between(x, prior_strong.pdf(x), alpha=0.3)
    axes[2].set_title('强信息先验（历史数据）', fontsize=12)
    axes[2].set_xlabel('转化率')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'prior_comparison.png', dpi=100)
    print("✅ 先验对比图已保存: prior_comparison.png")


def compare_prior_statistics():
    """对比三种先验的统计特性"""
    priors = {
        '无信息先验': create_uninformative_prior(),
        '弱信息先验': create_weakly_informative_prior(),
        '强信息先验': create_informative_prior(),
        '极端乐观（反例）': bad_extreme_prior()
    }

    print("=" * 60)
    print("先验统计特性对比")
    print("=" * 60)
    print(f"{'先验类型':<15} {'均值':>10} {'标准差':>10} {'95% 区间':>20}")
    print("-" * 60)

    for name, prior in priors.items():
        mean = prior.mean()
        std = prior.std()
        ci_low, ci_high = prior.interval(0.95)

        print(f"{name:<15} {mean:>10.4f} {std:>10.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    print()


def main() -> None:
    # 对比先验统计特性
    compare_prior_statistics()

    # 绘制对比图
    print("\n绘制先验对比图...")
    plot_prior_comparison()

    # 解释每种先验的适用场景
    print("=" * 60)
    print("先验使用建议")
    print("=" * 60)
    print()
    print("【无信息先验】")
    print("  适用场景：对参数一无所知，或想保守")
    print("  优点：让数据自己说话")
    print("  缺点：小样本下方差大")
    print()
    print("【弱信息先验】")
    print("  适用场景：大部分场景（推荐）")
    print("  优点：编码基本常识，但不强")
    print("  示例：Beta(2, 40) 表示'通常转化率在 5% 上下'")
    print()
    print("【强信息先验】")
    print("  适用场景：有可靠的历史数据")
    print("  优点：小样本下也能稳健估计")
    print("  风险：如果历史数据有偏差，会误导后验")
    print()
    print("【极端先验（反例）】")
    print("  问题：与领域知识严重不符")
    print("  后果：让后验偏向不合理值")
    print("  解决：做先验敏感性分析，避免主观猜测")


if __name__ == "__main__":
    main()
