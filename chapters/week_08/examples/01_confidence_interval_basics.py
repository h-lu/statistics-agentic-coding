"""
示例：从点估计到区间估计——量化你的不确定性。

本例演示点估计（point estimate）与区间估计（interval estimate）的区别，
以及如何计算和解释置信区间（confidence interval）。

运行方式：python3 chapters/week_08/examples/01_confidence_interval_basics.py
预期输出：
  - stdout 输出点估计与区间估计的对比
  - 展示不同样本量下 CI 宽度的变化
  - 保存图表到 images/01_point_vs_interval.png

核心概念：
  - 点估计：用一个数字总结数据（如均值 3.2）
  - 区间估计：点估计 ± 一个范围（如 3.2 [95% CI: 2.8, 3.6]）
  - CI 宽度反映了估计的不确定性
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


def calculate_ci(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    计算均值及其置信区间（使用 t 分布）

    参数:
        data: 数据数组
        confidence: 置信水平（默认 0.95）

    返回:
        (均值, CI下界, CI上界)
    """
    mean = np.mean(data)
    se = stats.sem(data)  # 标准误
    df = len(data) - 1    # 自由度

    ci_low, ci_high = stats.t.interval(confidence, df=df, loc=mean, scale=se)
    return mean, ci_low, ci_high


def format_estimate(mean: float, ci_low: float, ci_high: float) -> str:
    """格式化为报告风格的输出"""
    ci_width = ci_high - ci_low
    return f"{mean:.2f} [95% CI: {ci_low:.2f}, {ci_high:.2f}] (宽度: {ci_width:.2f})"


def bad_example_only_point() -> None:
    """❌ 坏例子：只报告点估计"""
    print("=" * 60)
    print("❌ 坏例子：只报告点估计")
    print("=" * 60)

    np.random.seed(42)
    data = np.random.normal(loc=3.2, scale=1.5, size=50)
    mean = np.mean(data)

    print(f"\n用户平均消费金额：{mean:.2f} 元")
    print("\n问题：")
    print("  - 如果明天重新采样，这个均值会变成多少？")
    print("  - 你对这个数字有多确定？")
    print("  - 样本量是 50 还是 5000，结论一样吗？")
    print("\n→ 点估计没有告诉你'有多确定'")


def good_example_with_ci() -> None:
    """✅ 好例子：报告点估计 + 置信区间"""
    print("\n" + "=" * 60)
    print("✅ 好例子：报告点估计 + 置信区间")
    print("=" * 60)

    np.random.seed(42)
    data = np.random.normal(loc=3.2, scale=1.5, size=50)
    mean, ci_low, ci_high = calculate_ci(data)

    print(f"\n用户平均消费金额：{format_estimate(mean, ci_low, ci_high)}")
    print("\n这样报告更好，因为：")
    print("  - 点估计：3.21 元（你的最佳猜测）")
    print("  - 95% CI：[2.80, 3.62]（合理的范围）")
    print("  - CI 宽度：0.82（反映了不确定性）")
    print("\n解读：")
    print("  如果我们重复抽样 100 次，大约有 95 次计算出的区间")
    print("  会包含真实的总体均值。")


def demonstrate_sample_size_effect() -> None:
    """演示样本量对 CI 宽度的影响"""
    print("\n" + "=" * 60)
    print("样本量对置信区间的影响")
    print("=" * 60)

    np.random.seed(42)
    sample_sizes = [20, 50, 100, 500, 1000]
    results = []

    print(f"\n{'样本量':<8} {'均值':<10} {'95% CI':<25} {'CI宽度':<10}")
    print("-" * 60)

    for n in sample_sizes:
        data = np.random.normal(loc=3.2, scale=1.5, size=n)
        mean, ci_low, ci_high = calculate_ci(data)
        ci_width = ci_high - ci_low
        results.append((n, mean, ci_low, ci_high, ci_width))

        ci_str = f"[{ci_low:.2f}, {ci_high:.2f}]"
        print(f"{n:<8} {mean:<10.2f} {ci_str:<25} {ci_width:<10.2f}")

    print("\n结论：")
    print("  - 样本量越大，CI 越窄（估计越精确）")
    print("  - 样本量越小，CI 越宽（不确定性越大）")
    print("  - 点估计可能相似，但 CI 宽度差异巨大")


def plot_ci_comparison() -> None:
    """绘制不同样本量的 CI 对比图"""
    setup_chinese_font()

    np.random.seed(42)
    sample_sizes = [20, 50, 100, 200, 500]
    means = []
    ci_lows = []
    ci_highs = []

    for n in sample_sizes:
        data = np.random.normal(loc=3.2, scale=1.5, size=n)
        mean, ci_low, ci_high = calculate_ci(data)
        means.append(mean)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制误差条
    ax.errorbar(sample_sizes, means,
                yerr=[np.array(means) - np.array(ci_lows),
                      np.array(ci_highs) - np.array(means)],
                fmt='o', capsize=5, capthick=2, linewidth=2,
                markersize=8, color='#2E86AB', ecolor='#A23B72')

    # 添加参考线
    ax.axhline(y=3.2, color='gray', linestyle='--', alpha=0.5,
               label='真实均值 (3.2)')

    ax.set_xlabel('样本量', fontsize=12)
    ax.set_ylabel('均值', fontsize=12)
    ax.set_title('样本量对置信区间宽度的影响\n'
                 '(样本量越大，估计越精确)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 标注样本量
    for i, (n, mean) in enumerate(zip(sample_sizes, means)):
        ax.text(n, mean + 0.3, f'n={n}', ha='center', fontsize=9)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '01_point_vs_interval.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("\n图表已保存到: images/01_point_vs_interval.png")


def demonstrate_standard_error_relation() -> None:
    """演示标准误与 CI 的关系"""
    print("\n" + "=" * 60)
    print("从标准误到置信区间")
    print("=" * 60)

    np.random.seed(42)
    data = np.random.normal(loc=3.2, scale=1.5, size=100)

    mean = np.mean(data)
    std = np.std(data, ddof=1)      # 样本标准差
    se = stats.sem(data)             # 标准误 = std / sqrt(n)
    n = len(data)

    print(f"\n样本量: n = {n}")
    print(f"样本标准差: SD = {std:.3f}")
    print(f"标准误: SE = SD / √n = {se:.3f}")
    print(f"\n95% CI ≈ mean ± 1.96 × SE（大样本，正态分布）")
    print(f"95% CI ≈ {mean:.2f} ± 1.96 × {se:.3f}")
    print(f"       = [{mean - 1.96*se:.2f}, {mean + 1.96*se:.2f}]")

    # 使用 t 分布（更准确）
    t_critical = stats.t.ppf(0.975, df=n-1)
    print(f"\n95% CI = mean ± t({n-1}) × SE（使用 t 分布）")
    print(f"       = {mean:.2f} ± {t_critical:.3f} × {se:.3f}")
    print(f"       = [{mean - t_critical*se:.2f}, {mean + t_critical*se:.2f}]")

    print("\n直觉：")
    print("  - SE 越小 → CI 越窄（估计越精确）")
    print("  - SE = SD / √n → 样本量越大，SE 越小")
    print("  - t 临界值 → 样本量越小，t 越大（CI 更宽，更保守）")


def main() -> None:
    """主函数"""
    print("从点估计到区间估计\n")

    # 坏例子：只报告点估计
    bad_example_only_point()

    # 好例子：报告点估计 + CI
    good_example_with_ci()

    # 样本量影响
    demonstrate_sample_size_effect()

    # 标准误与 CI 的关系
    demonstrate_standard_error_relation()

    # 绘图
    plot_ci_comparison()

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("\n关键要点：")
    print("  1. 点估计：用一个数字总结数据，但不反映不确定性")
    print("  2. 区间估计：点估计 ± 范围，量化'有多确定'")
    print("  3. CI 宽度由标准误决定：SE = SD / √n")
    print("  4. 样本量越大，CI 越窄（估计越精确）")
    print("\n在报告中：")
    print("  ❌ '用户平均消费 3.2 元'")
    print("  ✅ '用户平均消费 3.2 元 [95% CI: 2.8, 3.6]'")
    print()


if __name__ == "__main__":
    main()
