"""
示例：置换检验——当"零假设"是"没有差异"时。

本例演示置换检验（Permutation Test）的原理和实现：
  - 核心思想：如果零假设成立，组别标签没意义，可以随便打乱
  - 优势：不依赖分布假设，适用于小样本或非正态数据
  - 与 t 检验的对比

运行方式：python3 chapters/week_08/examples/05_permutation_test.py
预期输出：
  - stdout 输出置换检验的过程和结果
  - 对比置换检验与 t 检验
  - 保存图表到 images/05_permutation_test.png

核心概念：
  - 置换检验：打乱标签，模拟零假设下的差异分布
  - p 值：真实差异在零假设分布中的位置
  - 适用场景：小样本、非正态数据、想比较非均值统计量
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


def demonstrate_permutation_idea() -> None:
    """演示置换检验的核心思想"""
    print("=" * 70)
    print("置换检验的核心思想")
    print("=" * 70)

    np.random.seed(42)
    # 两组数据
    group_a = np.array([3.2, 2.8, 3.5, 4.1, 2.9])
    group_b = np.array([3.8, 4.2, 3.9, 4.5, 4.0])

    print(f"\n原始数据：")
    print(f"  A 组：{group_a}，均值 = {np.mean(group_a):.3f}")
    print(f"  B 组：{group_b}，均值 = {np.mean(group_b):.3f}")
    print(f"  真实差异（B - A）：{np.mean(group_b) - np.mean(group_a):.3f}")

    print(f"\n零假设（H0）：两组没有差异，来自同一个分布")
    print(f"备择假设（H1）：两组有差异，来自不同分布")

    print(f"\n置换检验的直觉：")
    print(f"  如果 H0 成立，那么'组别'这个标签是没意义的")
    print(f"  → 我们可以随便打乱标签")
    print(f"  → 打乱后重新计算差异")
    print(f"  → 重复很多次，得到'H0 下的差异分布'")
    print(f"  → 看真实差异在这个分布中的位置")

    print(f"\n示例：打乱标签 3 次")
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    for i in range(3):
        permuted = np.random.permutation(combined)
        perm_a = permuted[:n_a]
        perm_b = permuted[n_a:]
        perm_diff = np.mean(perm_b) - np.mean(perm_a)
        print(f"  置换 {i+1}：A 组均值 = {np.mean(perm_a):.3f}, "
              f"B 组均值 = {np.mean(perm_b):.3f}, "
              f"差异 = {perm_diff:.3f}")


def bad_example_t_test_small_sample() -> None:
    """❌ 坏例子：小样本非正态数据用 t 检验"""
    print("\n" + "=" * 70)
    print("❌ 坏例子：小样本非正态数据盲目用 t 检验")
    print("=" * 70)

    np.random.seed(42)
    # 小样本，非正态数据
    group_a = np.random.exponential(scale=2.0, size=15)
    group_b = np.random.exponential(scale=2.5, size=15)

    # t 检验
    t_stat, p_value = stats.ttest_ind(group_b, group_a)

    print(f"\n数据：指数分布，n = 15（小样本，非正态）")
    print(f"  A 组均值 = {np.mean(group_a):.3f}")
    print(f"  B 组均值 = {np.mean(group_b):.3f}")
    print(f"  差异 = {np.mean(group_b) - np.mean(group_a):.3f}")
    print(f"  t 检验：t = {t_stat:.3f}, p = {p_value:.4f}")

    print(f"\n问题：")
    print(f"  - t 检验假设数据来自正态分布")
    print(f"  - 指数分布是严重偏态的")
    print(f"  - 小样本时，t 检验的 p 值可能不准确")


def good_example_permutation_test() -> None:
    """✅ 好例子：用置换检验"""
    print("\n" + "=" * 70)
    print("✅ 好例子：用置换检验（不依赖分布假设）")
    print("=" * 70)

    np.random.seed(42)
    group_a = np.random.exponential(scale=2.0, size=15)
    group_b = np.random.exponential(scale=2.5, size=15)

    # 置换检验
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)
    observed_diff = np.mean(group_b) - np.mean(group_a)

    n_permutations = 10000
    perm_diffs = []

    for _ in range(n_permutations):
        permuted = np.random.permutation(combined)
        perm_a = permuted[:n_a]
        perm_b = permuted[n_a:]
        perm_diffs.append(np.mean(perm_b) - np.mean(perm_a))

    perm_diffs = np.array(perm_diffs)

    # 双尾 p 值
    p_value_two_tailed = np.mean(np.abs(perm_diffs) >= abs(observed_diff))

    print(f"\n数据：指数分布，n = 15（小样本，非正态）")
    print(f"  A 组均值 = {np.mean(group_a):.3f}")
    print(f"  B 组均值 = {np.mean(group_b):.3f}")
    print(f"  真实差异 = {observed_diff:.3f}")
    print(f"  置换检验 p 值（双尾）= {p_value_two_tailed:.4f}")

    print(f"\n置换检验的优势：")
    print(f"  - 不假设正态分布")
    print(f"  - 不假设方差齐性")
    print(f"  - 适用于小样本")
    print(f"  - 可以比较任何统计量（均值、中位数等）")


def compare_t_test_vs_permutation() -> None:
    """对比 t 检验与置换检验"""
    print("\n" + "=" * 70)
    print("t 检验 vs 置换检验")
    print("=" * 70)

    np.random.seed(42)

    # 场景 1：大样本，正态数据 → 两种方法应该接近
    print(f"\n场景 1：大样本（n=100），正态数据")
    group_a1 = np.random.normal(loc=3.0, scale=1.5, size=100)
    group_b1 = np.random.normal(loc=3.5, scale=1.5, size=100)

    t_stat1, p_ttest1 = stats.ttest_ind(group_b1, group_a1)

    combined = np.concatenate([group_a1, group_b1])
    n_a = len(group_a1)
    observed_diff1 = np.mean(group_b1) - np.mean(group_a1)
    perm_diffs1 = [np.mean(np.random.permutation(combined)[n_a:]) -
                   np.mean(np.random.permutation(combined)[:n_a])
                   for _ in range(10000)]
    p_perm1 = np.mean(np.abs(perm_diffs1) >= abs(observed_diff1))

    print(f"  t 检验：p = {p_ttest1:.4f}")
    print(f"  置换检验：p = {p_perm1:.4f}")
    print(f"  结论：两种方法接近")

    # 场景 2：小样本，非正态数据 → 置换检验更可靠
    print(f"\n场景 2：小样本（n=20），偏态数据")
    group_a2 = np.random.exponential(scale=2.0, size=20)
    group_b2 = np.random.exponential(scale=2.8, size=20)

    t_stat2, p_ttest2 = stats.ttest_ind(group_b2, group_a2)

    combined = np.concatenate([group_a2, group_b2])
    n_a = len(group_a2)
    observed_diff2 = np.mean(group_b2) - np.mean(group_a2)
    perm_diffs2 = [np.mean(np.random.permutation(combined)[n_a:]) -
                   np.mean(np.random.permutation(combined)[:n_a])
                   for _ in range(10000)]
    p_perm2 = np.mean(np.abs(perm_diffs2) >= abs(observed_diff2))

    print(f"  t 检验：p = {p_ttest2:.4f}")
    print(f"  置换检验：p = {p_perm2:.4f}")
    print(f"  结论：置换检验更可靠（不依赖正态性假设）")

    print(f"\n对比表：")
    print(f"{'维度':<15} {'t 检验':<20} {'置换检验':<20}")
    print("-" * 60)
    print(f"{'假设':<15} {'正态性、方差齐性':<20} {'无（除独立性）':<20}")
    print(f"{'p 值来源':<15} {'理论分布（t 分布）':<20} {'模拟分布':<20}")
    print(f"{'计算速度':<15} {'快（闭式解）':<20} {'慢（需模拟）':<20}")
    print(f"{'小样本':<15} {'假设不成立时不准':<20} {'更稳健':<20}")
    print(f"{'适用性':<15} {'主要比较均值':<20} {'任何统计量':<20}")


def demonstrate_permutation_for_median() -> None:
    """演示用置换检验比较中位数"""
    print("\n" + "=" * 70)
    print("置换检验的强大之处：可以比较任何统计量")
    print("=" * 70)

    np.random.seed(42)
    group_a = np.random.exponential(scale=2.0, size=50)
    group_b = np.random.exponential(scale=2.5, size=50)

    # 比较中位数
    observed_median_diff = np.median(group_b) - np.median(group_a)

    # 置换检验
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    perm_median_diffs = []
    for _ in range(10000):
        permuted = np.random.permutation(combined)
        perm_a = permuted[:n_a]
        perm_b = permuted[n_a:]
        perm_median_diffs.append(np.median(perm_b) - np.median(perm_a))

    perm_median_diffs = np.array(perm_median_diffs)
    p_value = np.mean(np.abs(perm_median_diffs) >= abs(observed_median_diff))

    print(f"\n比较中位数（不是均值）")
    print(f"  A 组中位数 = {np.median(group_a):.3f}")
    print(f"  B 组中位数 = {np.median(group_b):.3f}")
    print(f"  真实差异 = {observed_median_diff:.3f}")
    print(f"  置换检验 p 值 = {p_value:.4f}")

    print(f"\n注意：")
    print(f"  - 没有简单的'中位数 t 检验'公式")
    print(f"  - 置换检验可以轻松处理")
    print(f"  - 这就是置换检验的强大之处")


def plot_permutation_test() -> None:
    """绘制置换检验的可视化"""
    setup_chinese_font()

    np.random.seed(42)
    # 两组数据
    group_a = np.random.normal(loc=3.0, scale=1.5, size=50)
    group_b = np.random.normal(loc=3.8, scale=1.5, size=50)

    # 真实差异
    observed_diff = np.mean(group_b) - np.mean(group_a)

    # 置换检验
    combined = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    perm_diffs = []
    for _ in range(10000):
        permuted = np.random.permutation(combined)
        perm_a = permuted[:n_a]
        perm_b = permuted[n_a:]
        perm_diffs.append(np.mean(perm_b) - np.mean(perm_a))

    perm_diffs = np.array(perm_diffs)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1：原始数据分布
    ax1.hist(group_a, bins=20, alpha=0.6, color='#2E86AB',
             edgecolor='black', label='A 组')
    ax1.hist(group_b, bins=20, alpha=0.6, color='#A23B72',
             edgecolor='black', label='B 组')
    ax1.axvline(np.mean(group_a), color='#2E86AB', linestyle='--',
                linewidth=2, label=f'A 组均值 = {np.mean(group_a):.2f}')
    ax1.axvline(np.mean(group_b), color='#A23B72', linestyle='--',
                linewidth=2, label=f'B 组均值 = {np.mean(group_b):.2f}')

    ax1.set_xlabel('值', fontsize=12)
    ax1.set_ylabel('频数', fontsize=12)
    ax1.set_title('原始数据：A 组 vs B 组', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # 子图 2：置换分布
    ax2.hist(perm_diffs, bins=50, color='#2E86AB', alpha=0.7,
             edgecolor='black', label='零假设下的差异分布')
    ax2.axvline(observed_diff, color='red', linestyle='--', linewidth=3,
                label=f'真实差异 = {observed_diff:.3f}')
    ax2.axvline(-observed_diff, color='red', linestyle=':', linewidth=2)

    # 标记极端值
    extreme = np.sum(np.abs(perm_diffs) >= abs(observed_diff))
    p_value = extreme / len(perm_diffs)

    ax2.set_xlabel('均值差异（B - A）', fontsize=12)
    ax2.set_ylabel('频数', fontsize=12)
    ax2.set_title(f'置换检验：零假设下的差异分布\n'
                  f'p 值 = {p_value:.4f}（{extreme}/{len(perm_diffs)} '
                  f'次置换 ≥ |真实差异|）', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '05_permutation_test.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("\n图表已保存到: images/05_permutation_test.png")


def demonstrate_permutation_limitations() -> None:
    """演示置换检验的局限性"""
    print("\n" + "=" * 70)
    print("置换检验的局限性")
    print("=" * 70)

    print(f"\n置换检验不是万能的。以下情况需要谨慎：")

    print(f"\n1. 违反独立性假设")
    print(f"   - 置换检验假设观测是独立的")
    print(f"   - 不能用于时间序列（数据有时间依赖）")
    print(f"   - 不能用于配对数据（需要特殊的置换方式）")

    print(f"\n2. 样本代表性差")
    print(f"   - 如果样本不能代表总体")
    print(f"   - 打乱标签后的分布不能反映零假设")
    print(f"   - 置换检验也无能为力")

    print(f"\n3. 计算成本")
    print(f"   - 10000 次置换比 t 检验慢很多")
    print(f"   - 但在 2026 年，这通常不是问题")

    print(f"\n老潘的经验总结：")
    print(f'  "置换检验是最后的武器——当你不知道用什么检验时，')
    print(f'   置换检验通常是一个安全的起点。"')
    print(f'\n  "但它不是万能的。记住三个限制：独立性、样本代表性、计算成本。"')


def main() -> None:
    """主函数"""
    print("置换检验\n")

    # 置换检验的核心思想
    demonstrate_permutation_idea()

    # 坏例子：t 检验
    bad_example_t_test_small_sample()

    # 好例子：置换检验
    good_example_permutation_test()

    # 对比
    compare_t_test_vs_permutation()

    # 比较中位数
    demonstrate_permutation_for_median()

    # 局限性
    demonstrate_permutation_limitations()

    # 绘图
    plot_permutation_test()

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n关键要点：")
    print("  1. 核心思想：如果零假设成立，组别标签没意义")
    print("  2. 优势：不依赖分布假设，适用于小样本或非正态数据")
    print("  3. 可以比较任何统计量（均值、中位数等）")
    print("  4. 限制：假设独立性、样本代表性要好")
    print("\n老潘的建议：")
    print("  - 不知道用什么检验时 → 置换检验是安全的起点")
    print("  - 数据非正态 + 小样本 → 置换检验更可靠")
    print("  - 比较非均值统计量 → 置换检验几乎唯一选择")
    print("\n记住：")
    print("  置换检验是'最后的武器'，但不是万能的。")
    print("  独立性假设必须满足，样本必须具有代表性。")
    print()


if __name__ == "__main__":
    main()
