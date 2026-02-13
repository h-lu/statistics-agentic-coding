"""
示例：贝叶斯更新过程可视化——从先验到后验

运行方式：python3 chapters/week_14/examples/04_bayesian_update.py
预期输出：先验、似然、后验的对比图，展示"信念如何更新"
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_14"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False


def bayesian_update_visualization(
    conversions: int,
    exposures: int,
    alpha_prior: float = 2,
    beta_prior: float = 40,
    save_path: str = 'chapters/week_14/examples/bayesian_update.png'
) -> None:
    """
    可视化贝叶斯更新过程

    展示先验、似然（标准化）、后验三者的关系

    参数:
        conversions: 转化数
        exposures: 曝光数
        alpha_prior: 先验 Beta 分布的 alpha
        beta_prior: 先验 Beta 分布的 beta
        save_path: 图片保存路径
    """
    x = np.linspace(0, 0.15, 1000)

    # 先验分布
    prior = stats.beta(alpha_prior, beta_prior)

    # 后验分布（共轭先验的解析解）
    posterior = stats.beta(
        alpha_prior + conversions,
        beta_prior + exposures - conversions
    )

    # 似然函数（二项分布，标准化为概率密度）
    # 注意：似然不是概率分布，但我们可以标准化它以便可视化
    theta_mle = conversions / exposures
    likelihood_approx = stats.beta(conversions + 1, exposures - conversions + 1)

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制先验
    ax.plot(x, prior.pdf(x), 'b--', linewidth=2, label='先验 (Prior)', alpha=0.7)
    ax.fill_between(x, prior.pdf(x), alpha=0.1, color='blue')

    # 绘制似然（标准化）
    ax.plot(x, likelihood_approx.pdf(x), 'g-.', linewidth=2,
            label='似然 (Likelihood)', alpha=0.7)
    ax.fill_between(x, likelihood_approx.pdf(x), alpha=0.1, color='green')

    # 绘制后验
    ax.plot(x, posterior.pdf(x), 'r-', linewidth=3, label='后验 (Posterior)')
    ax.fill_between(x, posterior.pdf(x), alpha=0.2, color='red')

    # 标注关键值
    prior_mean = prior.mean()
    posterior_mean = posterior.mean()
    mle = conversions / exposures

    ax.axvline(prior_mean, color='blue', linestyle=':', alpha=0.5,
               label=f'先验均值: {prior_mean:.4f}')
    ax.axvline(mle, color='green', linestyle=':', alpha=0.5,
               label=f'MLE: {mle:.4f}')
    ax.axvline(posterior_mean, color='red', linestyle='-', alpha=0.8,
               linewidth=2,
               label=f'后验均值: {posterior_mean:.4f}')

    ax.set_xlabel('转化率 (Conversion Rate)', fontsize=12)
    ax.set_ylabel('概率密度 (Probability Density)', fontsize=12)
    ax.set_title(f'贝叶斯更新过程（数据：{conversions}/{exposures}）',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ 贝叶斯更新图已保存: {save_path}")


def sequential_update_demo():
    """
    演示序贯更新：数据逐步到来时，后验如何演化

    贝叶斯更新的一个关键特性：序贯性
    - 今天的数据 + 今天的先验 = 今天的后验
    - 明天的数据 + 今天的后验（作为明天的先验）= 明天的后验
    """
    # 模拟 5 天的 A/B 测试数据
    daily_data = [
        (10, 200),   # 第 1 天：10/200
        (25, 500),   # 第 2 天：25/500
        (40, 800),   # 第 3 天：40/800
        (50, 1000),  # 第 4 天：50/1000
        (58, 1000),  # 第 5 天：58/1000
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    x = np.linspace(0, 0.12, 500)

    # 初始先验
    alpha, beta = 2, 40  # 弱信息先验

    for i, (conv, exp) in enumerate(daily_data):
        prior = stats.beta(alpha, beta)

        # 计算后验
        posterior = stats.beta(alpha + conv, beta + exp - conv)

        # 绘制
        ax = axes[i]
        ax.plot(x, prior.pdf(x), 'b--', linewidth=2, label='先验', alpha=0.7)
        ax.plot(x, posterior.pdf(x), 'r-', linewidth=2, label='后验')
        ax.axvline(posterior.mean(), color='red', linestyle=':',
                  linewidth=2, alpha=0.8)
        ax.set_title(f'第 {i+1} 天\n({conv}/{exp})\n后验均值={posterior.mean():.4f}',
                   fontsize=10)
        ax.set_xlabel('转化率')
        ax.set_ylabel('概率密度' if i == 0 else '')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 后验成为下一次的先验（序贯更新）
        alpha = alpha + conv
        beta = beta + exp - conv

    plt.suptitle('序贯贝叶斯更新：后验 → 先验',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sequential_update.png', dpi=150)
    print("✅ 序贯更新图已保存: sequential_update.png")


def compare_influence_of_sample_size():
    """对比不同样本量下先验的影响"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    scenarios = [
        (5, 100, '小样本\n(5/100)'),
        (58, 1000, '中样本\n(58/1000)'),
        (580, 10000, '大样本\n(5800/10000)')
    ]

    x = np.linspace(0, 0.12, 500)

    for ax, (conv, exp, title) in zip(axes, scenarios):
        # 无信息先验
        prior_unif = stats.beta(1, 1)
        post_unif = stats.beta(1 + conv, 1 + exp - conv)

        # 强信息先验
        prior_strong = stats.beta(50, 1000)
        post_strong = stats.beta(50 + conv, 1000 + exp - conv)

        # MLE
        mle = conv / exp

        ax.plot(x, prior_unif.pdf(x), 'b--', linewidth=2,
               label='无信息先验', alpha=0.5)
        ax.plot(x, post_unif.pdf(x), 'b-', linewidth=2,
               label='无信息后验')

        ax.plot(x, prior_strong.pdf(x), 'g--', linewidth=2,
               label='强信息先验', alpha=0.5)
        ax.plot(x, post_strong.pdf(x), 'g-', linewidth=2,
               label='强信息后验')

        ax.axvline(mle, color='gray', linestyle=':', linewidth=2,
                  label=f'MLE={mle:.4f}', alpha=0.5)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('转化率')
        ax.set_ylabel('概率密度' if ax == axes[0] else '')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('样本量 vs 先验影响', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sample_size_vs_prior.png', dpi=150)
    print("✅ 样本量对比图已保存: sample_size_vs_prior.png")


def print_update_statistics():
    """打印贝叶斯更新的统计摘要"""
    print("=" * 60)
    print("贝叶斯更新统计摘要")
    print("=" * 60)

    # 初始先验
    alpha_prior, beta_prior = 2, 40
    prior = stats.beta(alpha_prior, beta_prior)

    print(f"\n【先验】Beta({alpha_prior}, {beta_prior})")
    print(f"  均值: {prior.mean():.4f}")
    print(f"  标准差: {prior.std():.4f}")
    print(f"  95% 区间: [{prior.interval(0.95)[0]:.4f}, {prior.interval(0.95)[1]:.4f}]")

    # 数据到来
    conversions, exposures = 58, 1000

    # 后验
    posterior = stats.beta(
        alpha_prior + conversions,
        beta_prior + exposures - conversions
    )

    print(f"\n【数据】{conversions}/{exposures} 转化")
    print(f"  MLE（最大似然估计）: {conversions/exposures:.4f}")

    print(f"\n【后验】Beta({alpha_prior + conversions}, {beta_prior + exposures - conversions})")
    print(f"  均值: {posterior.mean():.4f}")
    print(f"  标准差: {posterior.std():.4f}")
    print(f"  95% 可信区间: [{posterior.interval(0.95)[0]:.4f}, {posterior.interval(0.95)[1]:.4f}]")

    print(f"\n【更新过程】")
    print(f"  后验均值介于先验和 MLE 之间：")
    print(f"    {prior.mean():.4f} < {posterior.mean():.4f} < {conversions/exposures:.4f}")
    print(f"  不确定性降低：")
    print(f"    先验标准差 {prior.std():.4f} → 后验标准差 {posterior.std():.4f}")
    print(f"    减少了 {(1 - posterior.std()/prior.std())*100:.1f}%")


def main() -> None:
    # 打印统计摘要
    print_update_statistics()
    print()

    # 可视化单次更新
    print("\n生成贝叶斯更新过程图...")
    bayesian_update_visualization(58, 1000, alpha_prior=2, beta_prior=40)

    # 演示序贯更新
    print("\n生成序贯更新图...")
    sequential_update_demo()

    # 对比不同样本量
    print("\n生成样本量对比图...")
    compare_influence_of_sample_size()

    print("\n" + "=" * 60)
    print("贝叶斯更新的核心洞察")
    print("=" * 60)
    print("1. 后验 = 先验 × 似然（贝叶斯定理）")
    print("2. 后验均值介于先验均值和 MLE 之间（互相'妥协'）")
    print("3. 数据越多，先验影响越小（大数定律）")
    print("4. 序贯更新：今天的后验是明天的先验")


if __name__ == "__main__":
    main()
