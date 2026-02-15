"""
示例：多重比较问题演示——从单个检验到多个检验的假阳性率累积。

本例演示多重比较问题的本质：当你同时检验多个假设时，"至少一个假阳性"的概率
会随着检验次数指数增长（Family-wise Error Rate，FWER）。

运行方式：python3 chapters/week_07/examples/01_multiple_comparison_demo.py
预期输出：
  - stdout 输出假阳性率统计
  - 图表保存到 output/multiple_comparisons_simulation.png

反例：如果不做校正，检验 20 次时假阳性率超过 60%，这就是"死鲑鱼实验"揭示的问题。
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


def calculate_fwer(alpha: float, n_tests: int) -> float:
    """
    计算 Family-wise Error Rate（至少一个假阳性的概率）

    参数:
        alpha: 单个检验的显著性水平
        n_tests: 检验次数

    返回:
        FWER: 1 - (1 - alpha)^n_tests
    """
    return 1 - (1 - alpha) ** n_tests


def simulate_multiple_comparisons(
    n_tests: int,
    n_simulations: int,
    alpha: float
) -> np.ndarray:
    """
    模拟多重比较：当真实无差异时，统计假阳性数量

    参数:
        n_tests: 检验次数
        n_simulations: 模拟次数
        alpha: 显著性水平

    返回:
        false_positive_counts: 每次模拟的假阳性数量数组
    """
    np.random.seed(42)
    false_positive_counts = []

    for _ in range(n_simulations):
        # 检验 n_tests 个假设（真实都无差异，p 值均匀分布）
        p_values = np.random.uniform(0, 1, n_tests)
        significant = p_values < alpha
        false_positive_counts.append(significant.sum())

    return np.array(false_positive_counts)


def main() -> None:
    """运行多重比较演示"""
    font = setup_chinese_font()
    print(f"使用字体: {font}\n")

    # 参数设置
    n_tests = 50
    n_simulations = 1000
    alpha = 0.05

    print("=== 多重比较问题演示 ===\n")
    print(f"检验次数: {n_tests}")
    print(f"模拟次数: {n_simulations}")
    print(f"显著性水平: {alpha}\n")

    # 1. 计算不同检验次数下的 FWER
    print("=== FWER 随检验次数的变化 ===\n")
    test_counts = [1, 5, 10, 20, 30, 50]
    print(f"{'检验次数':<10} {'FWER (至少一个假阳性的概率)':<35}")
    print("-" * 50)
    for n in test_counts:
        fwer = calculate_fwer(alpha, n)
        print(f"{n:<10} {fwer:>6.1%}")

    print()
    print("小北的发现：")
    print(f"  检验 5 次：{calculate_fwer(alpha, 5):.1%} 的概率至少看到一个假阳性")
    print(f"  检验 20 次：{calculate_fwer(alpha, 20):.1%} 的概率至少看到一个假阳性")
    print(f"  检验 50 次：{calculate_fwer(alpha, 50):.1%} 的概率至少看到一个假阳性")
    print()

    # 2. 模拟实验
    print("=== 模拟实验：50 次检验中的假阳性分布 ===\n")
    false_positive_counts = simulate_multiple_comparisons(
        n_tests, n_simulations, alpha
    )

    print(f"平均假阳性数: {false_positive_counts.mean():.2f} (理论值: {n_tests * alpha:.2f})")
    print(f"至少一个假阳性的概率: {(false_positive_counts > 0).mean():.2%}")
    print()

    # 3. 可视化
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(false_positive_counts, bins=np.arange(-0.5, n_tests + 1, 1),
            edgecolor='black', alpha=0.7, density=True, color='steelblue')
    ax.axvline(false_positive_counts.mean(), color='red', linestyle='--',
               linewidth=2, label=f'平均假阳性数: {false_positive_counts.mean():.1f}')
    ax.set_xlabel('假阳性数量（50 次检验中）', fontsize=12)
    ax.set_ylabel('频率', fontsize=12)
    ax.set_title('多重模拟：当真实无差异时，50 次检验中的假阳性分布', fontsize=14)
    ax.legend(fontsize=10)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'multiple_comparisons_simulation.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"图表已保存到 images/multiple_comparisons_simulation.png")

    # 4. 阿码的问题
    print("\n=== 阿码的追问 ===")
    print("阿码：如果我检验 100 次，假阳性率是多少？")
    print(f"答案：{calculate_fwer(alpha, 100):.1%}（几乎肯定会看到假阳性）")
    print()
    print("老潘的经验：")
    print('  "检验越多，越需要"买保险"——这就是多重比较校正。"')
    print()
    print("下周我们将学习 Bonferroni 和 FDR 校正方法。")


if __name__ == "__main__":
    main()
