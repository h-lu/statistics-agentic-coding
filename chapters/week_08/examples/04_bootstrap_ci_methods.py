"""
示例：Bootstrap 置信区间方法——从 Percentile 到 BCa。

本例演示不同的 Bootstrap CI 计算方法：
  - Percentile Bootstrap（最简单）
  - Basic Bootstrap（反向 percentile）
  - BCa Bootstrap（Bias-Corrected and Accelerated，最准确）

运行方式：python3 chapters/week_08/examples/04_bootstrap_ci_methods.py
预期输出：
  - stdout 输出不同 Bootstrap CI 方法的对比
  - 演示偏态数据下各种方法的差异
  - 保存图表到 images/04_bootstrap_ci_methods.png

核心概念：
  - Percentile：直接用分位数，简单但可能有偏差
  - Basic：反向 percentile，改进偏差
  - BCa：校正偏差和加速，最准确但计算复杂
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import bootstrap
from pathlib import Path


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
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


def demonstrate_percentile_bootstrap() -> None:
    """演示 Percentile Bootstrap 方法"""
    print("=" * 70)
    print("方法 1：Percentile Bootstrap（最简单）")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.normal(loc=3.2, scale=1.5, size=100)

    # Bootstrap
    n_bootstrap = 10000
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(boot_sample))

    boot_means = np.array(boot_means)

    # Percentile CI
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    print(f"\n数据：正态分布，n = 100")
    print(f"  均值 = {np.mean(data):.3f}")
    print(f"  Percentile 95% CI：[{ci_low:.3f}, {ci_high:.3f}]")

    print(f"\n方法：")
    print(f"  - 对 Bootstrap 统计量排序")
    print(f"  - 取 2.5% 和 97.5% 分位数")
    print(f"  - 简单、直观、易于理解")

    print(f"\n优点：")
    print(f"  - 计算简单")
    print(f"  - 直观易懂")
    print(f"  - 对于对称分布效果很好")

    print(f"\n缺点：")
    print(f"  - 当统计量分布不对称时，覆盖率可能不准确")
    print(f"  - 小样本时表现不佳")


def demonstrate_skewed_data_problem() -> None:
    """演示偏态数据的问题"""
    print("\n" + "=" * 70)
    print("偏态数据的挑战：为什么需要 BCa？")
    print("=" * 70)

    np.random.seed(42)
    # 偏态数据：指数分布
    data = np.random.exponential(scale=2.0, size=100)

    # 计算中位数（不是均值，中位数在偏态分布中更有偏差）
    n_bootstrap = 10000
    boot_medians = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_medians.append(np.median(boot_sample))

    boot_medians = np.array(boot_medians)

    # 检查偏度
    skewness = ((boot_medians - np.mean(boot_medians))**3).mean() / (np.std(boot_medians)**3)

    print(f"\n数据：指数分布（严重偏态），n = 100")
    print(f"  中位数 = {np.median(data):.3f}")
    print(f"  Bootstrap 中位数分布的偏度 = {skewness:.3f}")
    print(f"  （偏度 > 0 表示右偏）")

    print(f"\n问题：")
    print(f"  - 中位数的 Bootstrap 分布是偏态的")
    print(f"  - Percentile CI 假设分布对称，可能不准确")
    print(f"  - 需要 BCa 校正")


def demonstrate_bca_bootstrap() -> None:
    """演示 BCa Bootstrap 方法"""
    print("\n" + "=" * 70)
    print("方法 2：BCa Bootstrap（Bias-Corrected and Accelerated）")
    print("=" * 70)

    np.random.seed(42)
    # 偏态数据
    data = np.random.exponential(scale=2.0, size=100)

    def median_func(x):
        return np.median(x)

    # Percentile Bootstrap
    res_pct = bootstrap((data,), median_func, confidence_level=0.95,
                        method='percentile', n_resamples=10000,
                        random_state=42)

    # BCa Bootstrap
    res_bca = bootstrap((data,), median_func, confidence_level=0.95,
                        method='BCa', n_resamples=10000,
                        random_state=42)

    print(f"\n数据：指数分布（偏态），n = 100")
    print(f"  中位数 = {np.median(data):.3f}")
    print(f"\n对比：")
    print(f"  Percentile 95% CI：[{res_pct.confidence_interval.low:.3f}, "
          f"{res_pct.confidence_interval.high:.3f}]")
    print(f"  BCa 95% CI：[{res_bca.confidence_interval.low:.3f}, "
          f"{res_bca.confidence_interval.high:.3f}]")

    print(f"\nBCa 的两个校正：")
    print(f"  1. Bias 校正：调整统计量的偏差")
    print(f"     - 如果 Bootstrap 中位数系统性地低于原始中位数，校正")
    print(f"  2. Accelerated 校正：调整统计量对极值的敏感性")
    print(f"     - 如果统计量对异常值敏感，加速因子会调整区间")

    print(f"\n为什么 BCa 更准确？")
    print(f"  - 它考虑了统计量分布的偏态")
    print(f"  - 它根据数据特性自动调整区间")
    print(f"  - 对于小样本和偏态数据，BCa 通常比 Percentile 更可靠")


def compare_methods_on_skewed_data() -> None:
    """在偏态数据上对比所有方法"""
    print("\n" + "=" * 70)
    print("在偏态数据上对比所有 Bootstrap CI 方法")
    print("=" * 70)

    np.random.seed(42)
    # 严重偏态数据
    data = np.random.exponential(scale=2.0, size=100)

    def median_func(x):
        return np.median(x)

    # 三种方法
    methods = ['percentile', 'basic', 'BCa']
    results = {}

    for method in methods:
        res = bootstrap((data,), median_func, confidence_level=0.95,
                        method=method, n_resamples=10000,
                        random_state=42)
        results[method] = (res.confidence_interval.low, res.confidence_interval.high)

    print(f"\n数据：指数分布，n = 100，统计量 = 中位数")
    print(f"{'方法':<15} {'95% CI':<30} {'区间宽度':<10}")
    print("-" * 60)

    for method, (low, high) in results.items():
        width = high - low
        ci_str = f"[{low:.3f}, {high:.3f}]"
        print(f"{method:<15} {ci_str:<30} {width:<10.3f}")

    print(f"\n观察：")
    print(f"  - Percentile 和 Basic 可能有差异")
    print(f"  - BCa 通常给出更保守（更宽）的区间")
    print(f"  - 对于偏态数据，BCa 更可靠")


def demonstrate_when_bootstrap_fails() -> None:
    """演示 Bootstrap 什么时候失效"""
    print("\n" + "=" * 70)
    print("什么时候 Bootstrap 会失效？")
    print("=" * 70)

    print(f"\nBootstrap 不是万能的。以下情况 Bootstrap 可能失效：")

    print(f"\n1. 样本量太小（n < 20）")
    print(f"   - Bootstrap 无法很好地近似抽样分布")
    print(f"   - 小样本本身就有问题，Bootstrap 无法'创造'信息")

    print(f"\n2. 数据有严重依赖（如时间序列）")
    print(f"   - 重采样会破坏时间结构")
    print(f"   - 需要 Block Bootstrap（Week 15 会讲）")

    print(f"\n3. 统计量对极端值非常敏感（如最大值）")
    print(f"   - Bootstrap 样本可能不包含原始数据的极值")
    print(f"   - 最大值的 CI 可能不准确")

    # 演示最大值的问题
    np.random.seed(42)
    data = np.random.normal(loc=0, scale=1, size=100)
    true_max = data.max()

    boot_maxs = []
    for _ in range(10000):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_maxs.append(boot_sample.max())

    boot_maxs = np.array(boot_maxs)
    max_ci_low, max_ci_high = np.percentile(boot_maxs, [2.5, 97.5])

    print(f"\n示例：最大值的 Bootstrap CI")
    print(f"  原始数据最大值：{true_max:.3f}")
    print(f"  Bootstrap 95% CI：[{max_ci_low:.3f}, {max_ci_high:.3f}]")
    print(f"  问题：Bootstrap 样本的最大值 ≤ 原始数据最大值")
    print(f"  → CI 上界被低估（无法超过真实最大值）")

    print(f"\n阿码问：'那如果我的数据不满足 Bootstrap 的假设呢？'")
    print(f"\n老潘回答：")
    print(f"  '那你就需要用其他方法：")
    print(f"   - 参数检验（如果假设成立）")
    print(f"   - 贝叶斯方法（Week 14 会讲）")
    print(f"   - 或者，收集更多数据。'")


def plot_bootstrap_ci_comparison() -> None:
    """绘制不同 Bootstrap CI 方法的对比"""
    setup_chinese_font()

    np.random.seed(42)

    # 创建两种数据：正态和偏态
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 场景 1：正态数据，均值
    data1 = np.random.normal(loc=3.2, scale=1.5, size=100)
    methods = ['percentile', 'basic', 'BCa']
    cis1 = []

    for method in methods:
        res = bootstrap((data1,), np.mean, confidence_level=0.95,
                        method=method, n_resamples=10000, random_state=42)
        cis1.append((res.confidence_interval.low, res.confidence_interval.high))

    # 绘制正态数据的 CI
    ax1 = axes[0, 0]
    y_pos = np.arange(len(methods))
    for i, (method, (low, high)) in enumerate(zip(methods, cis1)):
        ax1.plot([low, high], [i, i], 'o-', linewidth=3, markersize=8,
                 color='#2E86AB')
        ax1.plot(np.mean(data1), i, 's', color='red', markersize=10,
                label='样本均值' if i == 0 else '')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods)
    ax1.set_xlabel('均值', fontsize=12)
    ax1.set_title('正态数据：均值的三种 Bootstrap CI\n(三种方法应该接近)', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='x')

    # 场景 2：偏态数据，中位数
    data2 = np.random.exponential(scale=2.0, size=100)
    cis2 = []

    for method in methods:
        res = bootstrap((data2,), np.median, confidence_level=0.95,
                        method=method, n_resamples=10000, random_state=42)
        cis2.append((res.confidence_interval.low, res.confidence_interval.high))

    # 绘制偏态数据的 CI
    ax2 = axes[0, 1]
    for i, (method, (low, high)) in enumerate(zip(methods, cis2)):
        ax2.plot([low, high], [i, i], 'o-', linewidth=3, markersize=8,
                 color='#A23B72')
        ax2.plot(np.median(data2), i, 's', color='red', markersize=10,
                label='样本中位数' if i == 0 else '')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods)
    ax2.set_xlabel('中位数', fontsize=12)
    ax2.set_title('偏态数据：中位数的三种 Bootstrap CI\n(BCa 与其他方法可能有差异)', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='x')

    # 场景 3：Bootstrap 分布可视化（正态数据）
    ax3 = axes[1, 0]
    boot_means = [np.mean(np.random.choice(data1, size=len(data1), replace=True))
                  for _ in range(5000)]
    ax3.hist(boot_means, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(data1), color='red', linestyle='--', linewidth=2)
    ax3.axvline(cis1[2][0], color='green', linestyle=':', linewidth=2)
    ax3.axvline(cis1[2][1], color='green', linestyle=':', linewidth=2)
    ax3.set_xlabel('Bootstrap 均值', fontsize=12)
    ax3.set_title('正态数据的 Bootstrap 分布\n(对称，三种方法接近)', fontsize=13)
    ax3.grid(True, alpha=0.3, axis='y')

    # 场景 4：Bootstrap 分布可视化（偏态数据）
    ax4 = axes[1, 1]
    boot_medians = [np.median(np.random.choice(data2, size=len(data2), replace=True))
                    for _ in range(5000)]
    ax4.hist(boot_medians, bins=50, color='#A23B72', alpha=0.7, edgecolor='black')
    ax4.axvline(np.median(data2), color='red', linestyle='--', linewidth=2)
    ax4.axvline(cis2[2][0], color='green', linestyle=':', linewidth=2)
    ax4.axvline(cis2[2][1], color='green', linestyle=':', linewidth=2)
    ax4.set_xlabel('Bootstrap 中位数', fontsize=12)
    ax4.set_title('偏态数据的 Bootstrap 分布\n(偏态，BCa 更准确)', fontsize=13)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '04_bootstrap_ci_methods.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("\n图表已保存到: images/04_bootstrap_ci_methods.png")


def demonstrate_sample_size_guidance() -> None:
    """演示 Bootstrap 的样本量指导"""
    print("\n" + "=" * 70)
    print("Bootstrap 的样本量选择")
    print("=" * 70)

    print(f"\n小北问：'Bootstrap 要重采样多少次？1000？10000？'")

    print(f"\n老潘的经验法则：")
    print(f"  - 探索性分析：1000 次")
    print(f"    → 快，但不太准确")
    print(f"  - 正式报告：10000 次（推荐）")
    print(f"    → 平衡速度和准确性")
    print(f"  - 高精度需求：100000 次")
    print(f"    → 慢，但更准确")

    print(f"\n注意：")
    print(f"  - Bootstrap 的随机性会带来波动")
    print(f"  - 固定随机种子很重要（保证可复现性）")
    print(f"  - 复杂统计量可能需要更多次")


def main() -> None:
    """主函数"""
    print("Bootstrap 置信区间方法对比\n")

    # Percentile 方法
    demonstrate_percentile_bootstrap()

    # 偏态数据的挑战
    demonstrate_skewed_data_problem()

    # BCa 方法
    demonstrate_bca_bootstrap()

    # 对比所有方法
    compare_methods_on_skewed_data()

    # Bootstrap 失效的情况
    demonstrate_when_bootstrap_fails()

    # 样本量指导
    demonstrate_sample_size_guidance()

    # 绘图
    plot_bootstrap_ci_comparison()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n关键要点：")
    print("  1. Percentile：简单直观，适合对称分布")
    print("  2. Basic：改进 Percentile 的偏差")
    print("  3. BCa：最准确，自动校正偏差和加速，适合偏态数据")
    print("\n建议：")
    print("  - 正态数据 + 大样本：用 Percentile（快）")
    print("  - 偏态数据 + 小样本：用 BCa（准确）")
    print("  - 不确定时：用 BCa（scipy 默认）")
    print("  - 正式报告：固定随机种子，使用 10000 次")
    print("\n记住：")
    print("  Bootstrap 不是万能的。小样本、时间序列、极值敏感的统计量")
    print("  都可能导致 Bootstrap 失效。")
    print()


if __name__ == "__main__":
    main()
