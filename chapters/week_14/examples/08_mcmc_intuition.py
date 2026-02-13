"""
示例：MCMC 采样直觉——从"穷举"到"智能采样"

运行方式：python3 chapters/week_14/examples/08_mcmc_intuition.py
预期输出：MCMC 采样可视化、接受率、链混合情况
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_14"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def metropolis_hastings(
    log_target: callable,
    initial: float,
    n_samples: int,
    proposal_std: float = 1.0,
    random_seed: int = 42
) -> tuple[np.ndarray, float]:
    """
    Metropolis-Hastings 算法（MCMC 的基础）

    参数:
        log_target: 目标分布的对数密度函数
        initial: 初始值
        n_samples: 采样数量
        proposal_std: 提议分布的标准差
        random_seed: 随机种子

    返回:
        (样本数组, 接受率)
    """
    np.random.seed(random_seed)

    samples = np.zeros(n_samples)
    current = initial
    current_log_prob = log_target(current)
    n_accepted = 0

    for i in range(n_samples):
        # 1. 提议一个新值（从正态分布）
        proposal = current + np.random.normal(0, proposal_std)

        # 2. 计算新值的对数概率
        proposal_log_prob = log_target(proposal)

        # 3. 计算接受概率
        log_accept_ratio = proposal_log_prob - current_log_prob

        # 4. 接受或拒绝
        if np.log(np.random.random()) < log_accept_ratio:
            current = proposal
            current_log_prob = proposal_log_prob
            n_accepted += 1

        samples[i] = current

    accept_rate = n_accepted / n_samples
    return samples, accept_rate


def target_distribution_log_pdf(x: float) -> float:
    """
    目标分布：标准正态分布的对数密度

    用于演示 MCMC 采样
    """
    return -0.5 * x**2 - 0.5 * np.log(2 * np.pi)


# 坏例子：网格搜索（效率极低）
def bad_grid_search(
    target_pdf: callable,
    x_range: tuple[float, float],
    n_points: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """
    反例：网格搜索（穷举）

    问题：
    1. 高维问题中计算量爆炸（维度诅咒）
    2. 大部分时间浪费在低概率区域
    3. 无法处理复杂的后验分布

    参数:
        target_pdf: 目标分布的密度函数
        x_range: 搜索范围 (min, max)
        n_points: 网格点数

    返回:
        (x 值数组, 概率密度数组)
    """
    x = np.linspace(x_range[0], x_range[1], n_points)
    pdf = target_pdf(x)

    # 归一化（近似）
    pdf = pdf / pdf.sum() / (x[1] - x[0])

    return x, pdf


def visualize_mcmc_sampling(samples: np.ndarray,
                        accept_rate: float,
                        save_path: str = None):
    """
    可视化 MCMC 采样过程

    参数:
        samples: MCMC 采样结果
        accept_rate: 接受率
        save_path: 图片保存路径
    """
    fig = plt.figure(figsize=(14, 10))

    # 子图 1：采样轨迹
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(samples, alpha=0.5)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('参数值')
    ax1.set_title(f'MCMC 链轨迹（接受率={accept_rate:.1%}）')
    ax1.grid(True, alpha=0.3)

    # 子图 2：直方图 vs 真实分布
    ax2 = plt.subplot(3, 2, 2)
    ax2.hist(samples, bins=50, density=True, alpha=0.6, color='blue', label='MCMC 样本')

    # 真实分布（标准正态）
    from scipy import stats
    x = np.linspace(-4, 4, 200)
    ax2.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='真实分布')

    ax2.set_xlabel('参数值')
    ax2.set_ylabel('概率密度')
    ax2.set_title('MCMC 样本 vs 真实分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 子图 3：自相关
    ax3 = plt.subplot(3, 2, 3)
    max_lag = min(100, len(samples) // 4)
    lags = np.arange(max_lag)
    autocorr = [np.corrcoef(samples[:-lag], samples[lag:])[0, 1] if lag > 0 else 1
                for lag in lags]

    ax3.plot(lags, autocorr, 'o-')
    ax3.set_xlabel('滞后')
    ax3.set_ylabel('自相关系数')
    ax3.set_title('链内自相关')
    ax3.grid(True, alpha=0.3)

    # 子图 4：累积均值
    ax4 = plt.subplot(3, 2, 4)
    cumulative_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
    ax4.plot(cumulative_mean)
    ax4.axhline(0, color='r', linestyle='--', label='真实均值=0')
    ax4.set_xlabel('迭代次数')
    ax4.set_ylabel('累积均值')
    ax4.set_title('累积均值收敛')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 子图 5：步长分布
    ax5 = plt.subplot(3, 2, 5)
    steps = np.diff(samples)
    ax5.hist(steps, bins=50, alpha=0.6, color='green')
    ax5.set_xlabel('步长')
    ax5.set_ylabel('频数')
    ax5.set_title('MCMC 步长分布')
    ax5.grid(True, alpha=0.3)

    # 子图 6：前 100 步的轨迹（放大）
    ax6 = plt.subplot(3, 2, 6)
    n_show = min(100, len(samples))
    ax6.plot(samples[:n_show], 'o-', alpha=0.7)
    ax6.set_xlabel('迭代次数')
    ax6.set_ylabel('参数值')
    ax6.set_title(f'前 {n_show} 步轨迹')
    ax6.grid(True, alpha=0.3)

    plt.suptitle('MCMC 采样诊断', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ MCMC 诊断图已保存: {save_path}")


def compare_proposal_std():
    """对比不同提议标准差的效果"""
    proposal_stds = [0.1, 1.0, 10.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, prop_std in zip(axes, proposal_stds):
        samples, accept_rate = metropolis_hastings(
            target_distribution_log_pdf,
            initial=0,
            n_samples=5000,
            proposal_std=prop_std
        )

        # 绘制轨迹
        ax.plot(samples, alpha=0.5, linewidth=1)
        ax.set_title(f'提议 SD={prop_std}\n接受率={accept_rate:.1%}')
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('参数值')
        ax.grid(True, alpha=0.3)

        # 诊断
        if prop_std < 0.5:
            diagnostic = "步长太小 → 高接受率但探索慢"
        elif prop_std > 5:
            diagnostic = "步长太大 → 低接受率"
        else:
            diagnostic = "步长合适 → 平衡"

        ax.text(0.5, 0.05, diagnostic, transform=ax.transAxes,
               ha='center', va='bottom', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('提议标准差的影响', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'proposal_std_comparison.png', dpi=150)
    print("✅ 提议标准差对比图已保存: proposal_std_comparison.png")


def print_mcmc_intuition():
    """打印 MCMC 核心直觉"""
    print("=" * 60)
    print("MCMC 核心直觉")
    print("=" * 60)

    print("\n【为什么需要 MCMC？】")
    print("  问题：贝叶斯后验分布的分母（归一化常数）是高维积分")
    print("  解决：MCMC 不需要计算分母，直接采样后验分布")
    print("  核心：在高概率区域多采样，在低概率区域少采样")

    print("\n【Metropolis-Hastings 算法】")
    print("  1. 从当前位置出发，提议一个新值")
    print("  2. 如果新值概率更高 → 接受")
    print("  3. 如果新值概率更低 → 以一定概率接受")
    print("  4. 重复数千次，得到后验分布的样本")

    print("\n【网格搜索 vs MCMC】")
    print("  网格搜索：")
    print("    - 优点：简单直观")
    print("    - 缺点：维度灾难、计算量大、大部分点在低概率区域")
    print("  MCMC：")
    print("    - 优点：智能采样、适用于高维、自动集中在高概率区域")
    print("    - 缺点：需要检查收敛、链内自相关")

    print("\n【收敛诊断】")
    print("  Trace plot：链是否像'毛毛虫'（平稳、混合良好）")
    print("  R-hat：< 1.01 说明多条链收敛到同一分布")
    print("  ESS（有效样本量）：考虑自相关后的有效样本数，应 > 400")
    print("  自相关：高自相关 → 低效率 → 需要更多采样")


def main() -> None:
    print("=" * 60)
    print("MCMC 采样直觉演示")
    print("=" * 60)

    # 打印核心直觉
    print_mcmc_intuition()

    # 运行 Metropolis-Hastings
    print("\n" + "=" * 60)
    print("运行 Metropolis-Hastings 算法...")
    print("=" * 60)

    samples, accept_rate = metropolis_hastings(
        log_target=target_distribution_log_pdf,
        initial=0,
        n_samples=10000,
        proposal_std=1.0,
        random_seed=42
    )

    print(f"\n采样完成:")
    print(f"  样本数: {len(samples)}")
    print(f"  接受率: {accept_rate:.1%}")
    print(f"  样本均值: {samples.mean():.4f} (真实值: 0)")
    print(f"  样本标准差: {samples.std():.4f} (真实值: 1)")

    # 可视化
    print("\n生成 MCMC 诊断图...")
    visualize_mcmc_sampling(
        samples,
        accept_rate,
        'chapters/week_14/examples/mcmc_diagnostics.png'
    )

    # 对比不同提议标准差
    print("\n对比不同提议标准差...")
    compare_proposal_std()

    # 对比：网格搜索（坏例子）
    print("\n" + "=" * 60)
    print("【对比】网格搜索 vs MCMC")
    print("=" * 60)

    # 网格搜索
    from scipy import stats
    x_grid, pdf_grid = bad_grid_search(
        lambda x: stats.norm.pdf(x),
        (-4, 4),
        n_points=1000
    )
    print(f"\n网格搜索:")
    print(f"  评估点数: {len(x_grid)}")
    print(f"  计算: {len(x_grid)} 次密度评估")

    # MCMC
    n_mcmc_samples = 1000
    samples_mcmc, _ = metropolis_hastings(
        target_distribution_log_pdf,
        initial=0,
        n_samples=n_mcmc_samples,
        proposal_std=1.0
    )
    print(f"\nMCMC:")
    print(f"  样本数: {n_mcmc_samples}")
    print(f"  计算: {n_mcmc_samples * 2} 次密度评估（提议+当前）")

    print("\n结论：MCMC 用更少的计算量，获得了更集中在高概率区域的样本")


if __name__ == "__main__":
    main()
