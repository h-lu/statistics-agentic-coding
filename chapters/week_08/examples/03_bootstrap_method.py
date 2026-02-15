"""
示例：Bootstrap 重采样——从"假设分布"到"让数据说话"。

本例演示 Bootstrap 方法的核心思想：从样本中有放回地抽取很多个样本，
用这些 Bootstrap 样本的统计量分布来近似真实的抽样分布。

运行方式：python3 chapters/week_08/examples/03_bootstrap_method.py
预期输出：
  - stdout 输出 Bootstrap 重采样的过程和结果
  - 对比理论方法与 Bootstrap 方法
  - 保存图表到 images/03_bootstrap_distribution.png

核心概念：
  - Bootstrap：从样本中重采样（有放回），模拟"从总体重复抽样"
  - 优势：不依赖分布假设，适用于任何统计量
  - 什么时候用：非正态数据、复杂统计量、验证理论结果
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
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


def demonstrate_bootstrap_idea() -> None:
    """演示 Bootstrap 的核心思想"""
    print("=" * 70)
    print("Bootstrap 的核心思想")
    print("=" * 70)

    np.random.seed(42)
    # 原始样本
    original_sample = np.array([3.2, 2.8, 3.5, 4.1, 2.9, 3.3, 3.0, 3.7, 2.6, 3.4])

    print(f"\n原始样本（n = {len(original_sample)}）：")
    print(f"  {original_sample}")
    print(f"  均值 = {np.mean(original_sample):.3f}")

    print(f"\nBootstrap 步骤：")
    print(f"  1. 把原始样本当作'伪总体'")
    print(f"  2. 从中有放回地抽取 n 个观测（Bootstrap 样本）")
    print(f"  3. 对每个 Bootstrap 样本计算统计量（如均值）")
    print(f"  4. 重复很多次，得到统计量的分布")

    print(f"\n示例：3 个 Bootstrap 样本")
    for i in range(3):
        boot_sample = np.random.choice(original_sample, size=len(original_sample),
                                       replace=True)
        boot_mean = np.mean(boot_sample)
        print(f"  样本 {i+1}: {boot_sample} → 均值 = {boot_mean:.3f}")

    print(f"\n观察：")
    print(f"  - 每个 Bootstrap 样本都来自原始样本")
    print(f"  - 因为有放回抽样，某些观测会出现多次，某些不出现")
    print(f"  - 每个样本的均值都不同，模拟了'抽样变异'")


def bad_example_small_sample_assumption() -> None:
    """❌ 坏例子：小样本时盲目相信理论公式"""
    print("\n" + "=" * 70)
    print("❌ 坏例子：小样本时盲目相信理论公式")
    print("=" * 70)

    np.random.seed(42)
    # 小样本，非正态数据
    data = np.random.exponential(scale=2.0, size=20)

    mean = np.mean(data)
    se = stats.sem(data)
    t_ci_low, t_ci_high = stats.t.interval(0.95, df=len(data)-1,
                                            loc=mean, scale=se)

    print(f"\n数据：指数分布，n = 20（非正态，小样本）")
    print(f"  均值 = {mean:.3f}")
    print(f"  95% CI（t 分布）：[{t_ci_low:.3f}, {t_ci_high:.3f}]")

    print(f"\n问题：")
    print(f"  - t 分布假设数据来自正态分布")
    print(f"  - 指数分布是严重偏态的，小样本时 t CI 可能不准确")
    print(f"  - 但你用了 t 公式，结果可能误导")


def good_example_bootstrap_ci() -> None:
    """✅ 好例子：用 Bootstrap 估计 CI"""
    print("\n" + "=" * 70)
    print("✅ 好例子：用 Bootstrap 估计 CI（不依赖分布假设）")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.exponential(scale=2.0, size=20)

    # Bootstrap
    n_bootstrap = 10000
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(boot_sample))

    boot_means = np.array(boot_means)
    boot_se = np.std(boot_means, ddof=1)
    boot_ci_low, boot_ci_high = np.percentile(boot_means, [2.5, 97.5])

    print(f"\n数据：指数分布，n = 20（非正态，小样本）")
    print(f"  Bootstrap 标准误：{boot_se:.3f}")
    print(f"  Bootstrap 95% CI：[{boot_ci_low:.3f}, {boot_ci_high:.3f}]")

    print(f"\n优势：")
    print(f"  - 不假设正态分布")
    print(f"  - 适用于任何统计量（均值、中位数、甚至复杂指标）")
    print(f"  - 小样本时也比盲目用 t 公式更可靠")


def compare_bootstrap_vs_theoretical() -> None:
    """对比 Bootstrap 与理论方法"""
    print("\n" + "=" * 70)
    print("Bootstrap vs 理论公式：什么时候用哪个？")
    print("=" * 70)

    np.random.seed(42)

    # 场景 1：大样本，正态数据 → 两种方法应该接近
    print(f"\n场景 1：大样本（n=500），正态数据")
    data1 = np.random.normal(loc=3.2, scale=1.5, size=500)

    # 理论方法
    mean1 = np.mean(data1)
    se1 = stats.sem(data1)
    t_ci_low1, t_ci_high1 = stats.t.interval(0.95, df=len(data1)-1,
                                               loc=mean1, scale=se1)

    # Bootstrap
    boot_means1 = [np.mean(np.random.choice(data1, size=len(data1), replace=True))
                   for _ in range(10000)]
    boot_ci_low1, boot_ci_high1 = np.percentile(boot_means1, [2.5, 97.5])

    print(f"  理论 t CI：[{t_ci_low1:.3f}, {t_ci_high1:.3f}]")
    print(f"  Bootstrap CI：[{boot_ci_low1:.3f}, {boot_ci_high1:.3f}]")
    print(f"  结论：两种方法接近，理论方法更快")

    # 场景 2：小样本，非正态数据 → Bootstrap 更可靠
    print(f"\n场景 2：小样本（n=30），偏态数据")
    data2 = np.random.exponential(scale=2.0, size=30)

    # 理论方法（可能不准确）
    mean2 = np.mean(data2)
    se2 = stats.sem(data2)
    t_ci_low2, t_ci_high2 = stats.t.interval(0.95, df=len(data2)-1,
                                               loc=mean2, scale=se2)

    # Bootstrap
    boot_means2 = [np.mean(np.random.choice(data2, size=len(data2), replace=True))
                   for _ in range(10000)]
    boot_ci_low2, boot_ci_high2 = np.percentile(boot_means2, [2.5, 97.5])

    print(f"  理论 t CI：[{t_ci_low2:.3f}, {t_ci_high2:.3f}]")
    print(f"  Bootstrap CI：[{boot_ci_low2:.3f}, {boot_ci_high2:.3f}]")
    print(f"  结论：Bootstrap 更可靠（不依赖正态性假设）")

    print(f"\n老潘的经验法则：")
    print(f"  - 简单问题（均值 CI）+ 大样本 → 用理论公式（快）")
    print(f"  - 复杂问题（中位数、机器学习指标）→ 用 Bootstrap（通用）")
    print(f"  - 非正态数据 → 用 Bootstrap（更稳健）")
    print(f"  - 时间序列 → 用 Block Bootstrap（Week 15 会讲）")


def demonstrate_bootstrap_for_median() -> None:
    """演示 Bootstrap 估计中位数的 CI"""
    print("\n" + "=" * 70)
    print("Bootstrap 的强大之处：适用于任何统计量")
    print("=" * 70)

    np.random.seed(42)
    # 偏态数据
    data = np.random.exponential(scale=2.0, size=100)

    # Bootstrap 中位数
    n_bootstrap = 10000
    boot_medians = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_medians.append(np.median(boot_sample))

    boot_medians = np.array(boot_medians)
    median_ci_low, median_ci_high = np.percentile(boot_medians, [2.5, 97.5])

    print(f"\n数据：指数分布（偏态），n = 100")
    print(f"  中位数 = {np.median(data):.3f}")
    print(f"  Bootstrap 95% CI：[{median_ci_low:.3f}, {median_ci_high:.3f}]")

    print(f"\n注意：")
    print(f"  - 中位数没有简单的公式计算 CI")
    print(f"  - Bootstrap 可以轻松处理")
    print(f"  - 这就是 Bootstrap 的强大之处")


def plot_bootstrap_distribution() -> None:
    """绘制 Bootstrap 分布的可视化"""
    setup_chinese_font()

    np.random.seed(42)
    # 原始数据
    data = np.random.normal(loc=3.2, scale=1.5, size=100)
    true_mean = np.mean(data)

    # Bootstrap
    n_bootstrap = 10000
    boot_means = []
    for _ in range(n_bootstrap):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(boot_sample))

    boot_means = np.array(boot_means)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1：Bootstrap 分布
    ax1.hist(boot_means, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(true_mean, color='red', linestyle='--', linewidth=2,
                label=f'原始样本均值 = {true_mean:.2f}')
    ax1.axvline(np.percentile(boot_means, 2.5), color='green',
                linestyle=':', linewidth=2, label='2.5% 分位数')
    ax1.axvline(np.percentile(boot_means, 97.5), color='green',
                linestyle=':', linewidth=2, label='97.5% 分位数')

    ax1.set_xlabel('Bootstrap 均值', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('Bootstrap 分布（10000 次重采样）\n'
                  '近似真实均值的抽样分布', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 子图 2：原始数据 vs Bootstrap 样本
    ax2.hist(data, bins=20, color='#A23B72', alpha=0.6,
             edgecolor='black', label='原始数据')
    ax2.hist(boot_means, bins=50, color='#2E86AB', alpha=0.5,
             edgecolor='black', label='Bootstrap 均值分布')

    ax2.set_xlabel('值', fontsize=12)
    ax2.set_ylabel('频数', fontsize=12)
    ax2.set_title('原始数据分布 vs Bootstrap 均值分布\n'
                  '(均值分布更窄，标准误更小)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '03_bootstrap_distribution.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("\n图表已保存到: images/03_bootstrap_distribution.png")


def demonstrate_why_bootstrap_works() -> None:
    """演示为什么 Bootstrap 有效"""
    print("\n" + "=" * 70)
    print("为什么 Bootstrap 有效？")
    print("=" * 70)

    print(f"\n直觉：")
    print(f"  - 大样本下，样本的分布会接近总体分布")
    print(f"  - 从样本中重采样 ≈ 从总体中抽样")
    print(f"  - Bootstrap 统计量的分布 ≈ 真实抽样分布")

    print(f"\n数学原理（简化）：")
    print(f"  - 根据大数定律，样本经验分布收敛于总体分布")
    print(f"  - Bootstrap 是'用经验分布代替未知总体分布'")
    print(f"  - 当样本量足够大，Bootstrap 近似会很好")

    print(f"\n限制：")
    print(f"  - 小样本（n < 20）：Bootstrap 近似效果差")
    print(f"  - 数据有依赖（时间序列）：重采样破坏依赖结构")
    print(f"  - 统计量对极值敏感（如最大值）：Bootstrap 可能不包含极值")


def main() -> None:
    """主函数"""
    print("Bootstrap 重采样方法\n")

    # Bootstrap 核心思想
    demonstrate_bootstrap_idea()

    # 坏例子：盲目相信理论公式
    bad_example_small_sample_assumption()

    # 好例子：用 Bootstrap
    good_example_bootstrap_ci()

    # 对比两种方法
    compare_bootstrap_vs_theoretical()

    # Bootstrap 估计中位数
    demonstrate_bootstrap_for_median()

    # 为什么 Bootstrap 有效
    demonstrate_why_bootstrap_works()

    # 绘图
    plot_bootstrap_distribution()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n关键要点：")
    print("  1. Bootstrap 核心思想：从样本中重采样，模拟重复抽样")
    print("  2. 优势：不依赖分布假设，适用于任何统计量")
    print("  3. 什么时候用：非正态数据、复杂统计量、验证理论结果")
    print("  4. 限制：小样本效果差、时间序列需要特殊处理")
    print("\n老潘的经验法则：")
    print("  - 简单 + 大样本 → 理论公式（快）")
    print("  - 复杂 + 非正态 → Bootstrap（通用）")
    print("  - 不确定时 → 两种方法都试，对比结果")
    print()


if __name__ == "__main__":
    main()
