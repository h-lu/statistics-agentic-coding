"""
示例：置信区间的正确解释——不要误读它。

本例演示置信区间的常见误解和正确理解。
最经典的误解："95% CI 意味着真实均值有 95% 概率落在区间内" —— 这是错的！

运行方式：python3 chapters/week_08/examples/02_ci_interpretation.py
预期输出：
  - stdout 输出 CI 的常见误解与正确理解
  - 通过模拟演示"重复抽样"的真正含义
  - 保存图表到 images/02_ci_interpretation.png

核心概念：
  - 错误理解：参数有 95% 概率落在 CI 内
  - 正确理解：重复抽样 100 次，约 95 个 CI 会包含真实参数
  - 关键区别：随机的是区间，不是参数
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


def bad_example_common_misunderstanding() -> None:
    """❌ 坏例子：常见误解"""
    print("=" * 70)
    print("❌ 常见误解 #1：'真实均值有 95% 概率落在 CI 内'")
    print("=" * 70)

    np.random.seed(42)
    data = np.random.normal(loc=3.2, scale=1.5, size=100)
    mean, ci_low, ci_high = calculate_ci(data)

    print(f"\n你的数据：均值 = {mean:.2f}, 95% CI = [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"\n错误解读：")
    print(f"  '真实均值 μ 有 95% 的概率落在 [{ci_low:.2f}, {ci_high:.2f}] 内'")
    print(f"\n为什么这是错的？")
    print(f"  - 真实均值 μ 是一个固定的（未知）常数")
    print(f"  - 区间 [{ci_low:.2f}, {ci_high:.2f}] 是随机的（会随样本变化）")
    print(f"  - μ 要么在区间内，要么不在，不存在'概率'之说")
    print(f"\n类比：")
    print(f"  你丢硬币，结果是正面或反面，不存在'50% 正面'（硬币已落地）")
    print(f"  但你说'我有 50% 的信心猜对'（这是你对方法的信心）")


def good_example_correct_interpretation() -> None:
    """✅ 好例子：正确理解"""
    print("\n" + "=" * 70)
    print("✅ 正确理解：'重复抽样的视角'")
    print("=" * 70)

    print(f"\n正确解读：")
    print(f"  '如果我们从总体中重复抽样 100 次，每次都计算 95% CI，")
    print(f"   那么大约有 95 个计算出的区间会包含真实的均值。'")
    print(f"\n关键区别：")
    print(f"  - 随机的是**区间**（每次抽样，区间都会变化）")
    print(f"  - 真实均值**不变**（它是固定的参数）")
    print(f"\n你手头只有一个区间，你要么'命中'，要么'未命中'")
    print(f"你无法知道这个特定区间是否包含真值，但你知道：")
    print(f"'这种方法有 95% 的概率产生包含真值的区间'")


def calculate_ci(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """计算均值及其置信区间"""
    mean = np.mean(data)
    se = stats.sem(data)
    df = len(data) - 1
    ci_low, ci_high = stats.t.interval(confidence, df=df, loc=mean, scale=se)
    return mean, ci_low, ci_high


def simulate_repeated_sampling() -> None:
    """模拟重复抽样，演示 CI 的频率学派含义"""
    print("\n" + "=" * 70)
    print("模拟演示：重复抽样 100 次，看看有多少 CI 包含真实均值")
    print("=" * 70)

    np.random.seed(42)
    true_mean = 3.2
    true_std = 1.5
    n_samples = 100
    sample_size = 50

    means = []
    ci_lows = []
    ci_highs = []
    covers = []

    for i in range(n_samples):
        data = np.random.normal(loc=true_mean, scale=true_std, size=sample_size)
        mean, ci_low, ci_high = calculate_ci(data)
        means.append(mean)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)
        covers.append(ci_low <= true_mean <= ci_high)

    coverage_rate = np.mean(covers)
    print(f"\n真实均值 μ = {true_mean}")
    print(f"重复抽样 {n_samples} 次，每次样本量 = {sample_size}")
    print(f"包含真实均值的 CI 数量：{sum(covers)} / {n_samples}")
    print(f"实际覆盖率：{coverage_rate * 100:.1f}%")
    print(f"理论覆盖率：95%")

    print(f"\n解释：")
    print(f"  - 每次抽样，我们都得到一个不同的区间")
    print(f"  - 大约 95% 的区间会包含真实均值")
    print(f"  - 你手头的某个区间，要么'命中'，要么'未命中'")
    print(f"  - 你无法知道你的区间是否命中，但你信任这个方法")

    return means, ci_lows, ci_highs, covers, true_mean


def plot_ci_simulation(means: list, ci_lows: list, ci_highs: list,
                       covers: list, true_mean: float) -> None:
    """绘制重复抽样的 CI 可视化"""
    setup_chinese_font()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 子图 1：前 20 个 CI
    n_plot = 20
    positions = range(1, n_plot + 1)

    for i in range(n_plot):
        color = '#2E86AB' if covers[i] else '#A23B72'
        alpha = 0.8 if covers[i] else 0.5
        ax1.plot([ci_lows[i], ci_highs[i]], [i, i], color=color,
                 linewidth=3, alpha=alpha)
        ax1.plot(means[i], i, 'o', color=color, markersize=8)

    # 真实均值线
    ax1.axvline(x=true_mean, color='gray', linestyle='--', linewidth=2,
                label=f'真实均值 μ = {true_mean}')

    ax1.set_xlabel('均值', fontsize=12)
    ax1.set_yticks(positions)
    ax1.set_yticklabels([f'样本 {i}' for i in positions])
    ax1.set_title('重复抽样 20 次：约 19 个区间应包含真实均值\n'
                  f'(蓝色=命中，红色=未命中)', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')

    # 子图 2：累积覆盖率
    cumulative_coverage = np.cumsum(covers) / np.arange(1, len(covers) + 1)
    ax2.plot(range(1, len(covers) + 1), cumulative_coverage * 100,
             color='#2E86AB', linewidth=2, label='实际覆盖率')
    ax2.axhline(y=95, color='red', linestyle='--', linewidth=2,
                label='理论覆盖率 95%')
    ax2.set_xlabel('抽样次数', fontsize=12)
    ax2.set_ylabel('累积覆盖率 (%)', fontsize=12)
    ax2.set_title('随着抽样次数增加，实际覆盖率趋近于 95%', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([80, 100])

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '02_ci_interpretation.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("\n图表已保存到: images/02_ci_interpretation.png")


def demonstrate_ci_vs_hypothesis_test() -> None:
    """演示 CI 与假设检验的关系"""
    print("\n" + "=" * 70)
    print("置信区间与假设检验的关系")
    print("=" * 70)

    np.random.seed(42)
    # 两组数据：A 组和 B 组
    group_a = np.random.normal(loc=3.0, scale=1.5, size=50)
    group_b = np.random.normal(loc=3.5, scale=1.5, size=50)

    # 计算 t 检验
    t_stat, p_value = stats.ttest_ind(group_b, group_a)

    # 计算差异的 CI
    diff_mean = np.mean(group_b) - np.mean(group_a)
    se_diff = np.sqrt(np.var(group_a, ddof=1)/len(group_a) +
                      np.var(group_b, ddof=1)/len(group_b))
    df = len(group_a) + len(group_b) - 2
    t_crit = stats.t.ppf(0.975, df=df)
    ci_low = diff_mean - t_crit * se_diff
    ci_high = diff_mean + t_crit * se_diff

    print(f"\n比较 B 组 vs A 组：")
    print(f"  点估计（差异）：{diff_mean:.3f}")
    print(f"  95% CI：[{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  t 检验：t = {t_stat:.3f}, p = {p_value:.4f}")

    print(f"\n关系：")
    if ci_low > 0:
        print(f"  - CI 不包含 0 → p < 0.05 → 差异显著")
    elif ci_high < 0:
        print(f"  - CI 不包含 0 → p < 0.05 → 差异显著")
    else:
        print(f"  - CI 包含 0 → p ≥ 0.05 → 差异不显著")

    print(f"\n✅ CI 比只报告 p 值更有信息量：")
    print(f"  - p 值只告诉你'显著不显著'")
    print(f"  - CI 还告诉你'差异大概有多大'")
    print(f"  - CI 的宽度反映了估计的精确度")


def bad_example_frequentist_confusion() -> None:
    """❌ 坏例子：混淆频率学派与贝叶斯学派"""
    print("\n" + "=" * 70)
    print("❌ 常见误解 #2：混淆频率学派 CI 与贝叶斯学派 CI")
    print("=" * 70)

    print(f"\n问题：能说'95% CI 表示我对参数有 95% 的信念'吗？")
    print(f"\n回答：不能。这是贝叶斯学派的'可信区间'（Credible Interval），")
    print(f"      不是频率学派的'置信区间'（Confidence Interval）。")
    print(f"\n区别：")
    print(f"  - 频率学派 CI：基于'重复抽样'的频率")
    print(f"  - 贝叶斯学派 CI：基于'对参数的主观信念'")
    print(f"\n如果你想表达'参数有 95% 概率在区间内'，你需要贝叶斯方法")
    print(f"（Week 14 会讲贝叶斯统计）")


def main() -> None:
    """主函数"""
    print("置信区间的正确解释\n")

    # 常见误解
    bad_example_common_misunderstanding()

    # 正确理解
    good_example_correct_interpretation()

    # 模拟演示
    means, ci_lows, ci_highs, covers, true_mean = simulate_repeated_sampling()

    # 绘图
    plot_ci_simulation(means, ci_lows, ci_highs, covers, true_mean)

    # CI 与假设检验的关系
    demonstrate_ci_vs_hypothesis_test()

    # 频率学派 vs 贝叶斯学派
    bad_example_frequentist_confusion()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n关键要点：")
    print("  1. ❌ 错误：'真实均值有 95% 概率落在 CI 内'")
    print("  2. ✅ 正确：'重复抽样 100 次，约 95 个 CI 会包含真实均值'")
    print("  3. 随机的是区间，不是参数")
    print("  4. CI 与假设检验等价：CI 包含 0 ↔ p ≥ 0.05")
    print("  5. 如果你想表达'信念'，需要贝叶斯方法（Week 14）")
    print("\n在报告中如何写：")
    print("  - ✅ '均值 3.2，95% CI [2.8, 3.6]'（让读者理解含义）")
    print("  - ✅ '在重复抽样的情况下，类似区间约 95% 会包含真值'")
    print("  - ❌ '均值有 95% 概率落在 [2.8, 3.6] 内'")
    print()


if __name__ == "__main__":
    main()
