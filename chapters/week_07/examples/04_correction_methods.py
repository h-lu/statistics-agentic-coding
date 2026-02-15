"""
示例：多重比较校正方法——Bonferroni vs FDR（Benjamini-Hochberg）。

本例演示不同校正方法的权衡：Bonferroni 最保守但最安全，FDR 更平衡但允许一些假阳性。
选择哪种方法取决于你的研究场景和风险容忍度。

运行方式：python3 chapters/week_07/examples/04_correction_methods.py
预期输出：
  - stdout 输出不同校正方法的拒绝数量对比
  - 图表保存到 images/correction_methods_comparison.png

反例：当检验次数很多（如 100 次）时，Bonferroni 校正后的阈值会变得极小（α/m），
几乎所有假设都无法拒绝，这时应该使用 FDR 方法。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
from statsmodels.stats.multitest import multipletests
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


def benjamini_hochberg_stepwise(
    p_values: np.ndarray,
    alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg (BH) FDR 校正（逐步算法）

    算法步骤：
    1. 将 p 值从小到大排序：p₁ ≤ p₂ ≤ ... ≤ pₘ
    2. 找到最大的 k，使得 pₖ ≤ (k/m) × α
    3. 拒绝前 k 个假设（p₁, ..., pₖ）

    参数:
        p_values: 原始 p 值数组
        alpha: 显著性水平

    返回:
        (rejected, adjusted_p): 拒绝标记和校正后的 p 值
    """
    m = len(p_values)

    # 从小到大排序，记录原始索引
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # 计算 BH 阈值线：(k/m) × α
    k_values = np.arange(1, m + 1)
    bh_thresholds = (k_values / m) * alpha

    # 找到最大的 k，使得 p_k <= (k/m) * alpha
    below_threshold = sorted_p <= bh_thresholds
    if below_threshold.any():
        k = np.where(below_threshold)[0].max() + 1
    else:
        k = 0

    # 标记拒绝的假设
    rejected = np.zeros(m, dtype=bool)
    if k > 0:
        rejected[sorted_indices[:k]] = True

    # 计算校正后的 p 值（简化方法）
    # 使用 statsmodels 的 multipletests 会更准确
    _, adjusted_p, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

    return rejected, adjusted_p


def simulate_hypothesis_tests(
    n_true: int,
    n_null: int,
    effect_size: float = 0.5,
    n_per_group: int = 100,
    seed: int = 42
) -> np.ndarray:
    """
    模拟假设检验的 p 值

    参数:
        n_true: 真实有差异的假设数量
        n_null: 真实无差异的假设数量
        effect_size: Cohen's d（效应量）
        n_per_group: 每组样本量
        seed: 随机种子

    返回:
        p_values: 所有假设的 p 值
    """
    np.random.seed(seed)
    total_tests = n_true + n_null
    p_values = []

    # 1. 真实有差异的假设（小 p 值）
    for _ in range(n_true):
        group1 = np.random.normal(0, 1, n_per_group)
        group2 = np.random.normal(effect_size, 1, n_per_group)
        _, p = stats.ttest_ind(group1, group2)
        p_values.append(p)

    # 2. 真实无差异的假设（均匀分布的 p 值）
    null_p_values = np.random.uniform(0, 1, n_null)
    p_values.extend(null_p_values)

    return np.array(p_values)


def visualize_correction_comparison(
    p_values: np.ndarray,
    alpha: float,
    output_path: Path
) -> None:
    """
    可视化不同校正方法的 p 值阈值

    参数:
        p_values: 原始 p 值
        alpha: 显著性水平
        output_path: 输出路径
    """
    font = setup_chinese_font()

    # 获取校正后的 p 值
    _, pvals_bonf, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
    _, pvals_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

    # 排序
    sorted_p = np.sort(p_values)
    sorted_p_bonf = np.sort(pvals_bonf)
    sorted_p_fdr = np.sort(pvals_fdr)

    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：p 值分布
    ax1 = axes[0]
    ax1.scatter(range(len(p_values)), sorted_p, alpha=0.6, s=50,
                label='原始 p 值', color='steelblue')
    ax1.axhline(alpha, color='red', linestyle='--', linewidth=2,
                label=f'未校正阈值 (α={alpha})')
    ax1.set_xlabel('假设（按 p 值排序）', fontsize=12)
    ax1.set_ylabel('p 值', fontsize=12)
    ax1.set_title('原始 p 值分布', fontsize=14)
    ax1.set_ylim(0, min(1, max(sorted_p) * 1.1))
    ax1.legend()

    # 右图：校正后对比
    ax2 = axes[1]
    ax2.scatter(range(len(p_values)), sorted_p_bonf, alpha=0.6, s=50,
                label='Bonferroni 校正', color='#d62728')
    ax2.scatter(range(len(p_values)), sorted_p_fdr, alpha=0.6, s=30,
                label='FDR (BH) 校正', color='#2ca02c', marker='s')
    ax2.axhline(alpha, color='red', linestyle='--', linewidth=2,
                label=f'显著性阈值 (α={alpha})')
    ax2.set_xlabel('假设（按校正后 p 值排序）', fontsize=12)
    ax2.set_ylabel('校正后 p 值', fontsize=12)
    ax2.set_title('校正方法对比', fontsize=14)
    ax2.set_ylim(0, min(1.1, max(sorted_p_bonf.max(), sorted_p_fdr.max()) * 1.1))
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def main() -> None:
    """运行校正方法演示"""
    font = setup_chinese_font()
    print(f"使用字体: {font}\n")

    print("=== 多重比较校正方法演示 ===\n")

    # 1. 生成模拟数据：20 个假设
    # 5 个真实显著（p 值小），15 个真实不显著（p 值均匀）
    np.random.seed(42)
    n_true_significant = 5
    n_true_null = 15
    p_values = simulate_hypothesis_tests(n_true_significant, n_true_null,
                                        effect_size=0.8, n_per_group=100)
    np.random.shuffle(p_values)

    alpha = 0.05
    m = len(p_values)

    print(f"模拟设置：")
    print(f"  总假设数: {m}")
    print(f"  真实显著数: {n_true_significant}")
    print(f"  真实不显著数: {n_true_null}")
    print(f"  显著性水平: {alpha}\n")

    # 2. 不同校正方法
    # 未校正
    uncorrected_rejected = p_values < alpha

    # Bonferroni 校正
    bonf_rejected, pvals_bonf, _, _ = multipletests(
        p_values, alpha=alpha, method='bonferroni'
    )

    # FDR (BH) 校正
    fdr_rejected, pvals_fdr, _, _ = multipletests(
        p_values, alpha=alpha, method='fdr_bh'
    )

    print("=== 校正方法比较 ===\n")
    print(f"{'方法':<20} {'拒绝数':<10} {'假阳性数':<15} {'假阴性数':<10}")
    print("-" * 60)

    # 假设前 n_true_significant 个是真实显著（简化计算）
    # 实际中我们不知道哪些是真实的，这里仅用于演示
    true_indices = set(range(n_true_significant))

    methods = [
        ("未校正", uncorrected_rejected),
        ("Bonferroni 校正", bonf_rejected),
        ("FDR (BH) 校正", fdr_rejected)
    ]

    for method_name, rejected in methods:
        n_rejected = rejected.sum()
        # 简化的假阳性/假阴性计算（仅用于演示）
        false_positive = sum([i for i, r in enumerate(rejected) if r and i not in true_indices])
        false_negative = sum([i for i, r in enumerate(rejected) if not r and i in true_indices])
        print(f"{method_name:<20} {n_rejected:<10} {false_positive:<15} {false_negative:<10}")

    print()
    print("解读：")
    print("  - 未校正：拒绝最多，但包含大量假阳性")
    print("  - Bonferroni：拒绝最少，很保守，可能漏掉真实显著")
    print("  - FDR (BH)：平衡发现率和假阳性率")
    print()

    # 3. 阿码的问题
    print("=== 阿码的追问 ===")
    print("阿码：如果我检验 100 个假设，Bonferroni 校正后的阈值是多少？")
    m_demo = 100
    bonf_threshold = alpha / m_demo
    print(f"答案：α/m = {alpha}/{m_demo} = {bonf_threshold:.5f}")
    print()
    print("老潘：'这时 Bonferroni 太保守了，几乎什么都不会显著。应该用 FDR。'")
    print()

    # 4. 决策树
    print("=== 选择校正方法的决策树 ===")
    print("""
你的场景是什么？

├─ 检验次数少（m < 10）
│  └─ 用 Bonferroni 或 Tukey HSD
│
├─ 检验次数中等（10 ≤ m ≤ 50）
│  ├─ 探索性研究（允许一些假阳性）→ 用 FDR (BH)
│  └─ 确认性研究（不能容忍假阳性）→ 用 Bonferroni
│
└─ 检验次数多（m > 50）
   └─ 必须用 FDR（Bonferroni 会太保守）
    """)

    # 5. 老潘的经验法则
    print("=== 老潘的经验法则 ===")
    print("• 期刊投稿：用保守的方法（Bonferroni 或 Tukey HSD）")
    print("• 内部决策：用 FDR，平衡发现率和假阳性率")
    print("• 探索性分析：报告未校正的 p 值，但明确标注'需后续验证'")
    print()

    # 6. 可视化
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    visualize_correction_comparison(p_values, alpha,
                                    output_dir / 'correction_methods_comparison.png')
    print("图表已保存到 images/correction_methods_comparison.png")


if __name__ == "__main__":
    main()
