"""
示例：先验敏感性分析

本例演示如何进行先验敏感性分析，评估贝叶斯结论对先验选择的依赖程度。
这是贝叶斯分析中"科学性"的关键：先验必须明确，且必须测试其影响。

运行方式：python3 chapters/week_14/examples/04_prior_sensitivity.py

预期输出：
- 不同先验下的后验分布对比表格
- 敏感性分析结论（稳健/敏感）
- 生成先验敏感性可视化图到 images/

核心思想：
- 如果结论对先验不敏感 → 结论稳健，可信
- 如果结论对先验敏感 → 需要更多数据，或说明结论依赖先验
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple


def setup_chinese_font() -> str:
    """配置中文字体"""
    import matplotlib.font_manager as fm
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK JP', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei', 'Droid Sans Fallback']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


@dataclass
class PosteriorResult:
    """后验分析结果"""
    prior_name: str
    alpha_prior: float
    beta_prior: float
    alpha_post: float
    beta_post: float
    posterior_mean: float
    ci_lower: float
    ci_upper: float
    ci_width: float


class PriorSensitivityAnalyzer:
    """
    先验敏感性分析器

    比较不同先验下的后验分布，评估结论对先验的敏感程度
    """

    def __init__(self, n: int, successes: int):
        """
        初始化分析器

        参数:
            n: 总样本数
            successes: 成功次数（如流失客户数）
        """
        self.n = n
        self.successes = successes
        self.failures = n - successes
        self.observed_rate = successes / n

    def analyze_prior(self, prior_name: str,
                     alpha_prior: float, beta_prior: float) -> PosteriorResult:
        """
        分析单个先验的后验分布

        参数:
            prior_name: 先验名称
            alpha_prior: Beta 分布 alpha 参数
            beta_prior: Beta 分布 beta 参数

        返回:
            PosteriorResult 对象
        """
        # Beta-Binomial 共轭后验
        alpha_post = alpha_prior + self.successes
        beta_post = beta_prior + self.failures

        posterior_mean = alpha_post / (alpha_post + beta_post)
        ci_lower, ci_upper = stats.beta.interval(0.95, alpha_post, beta_post)
        ci_width = ci_upper - ci_lower

        return PosteriorResult(
            prior_name=prior_name,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            alpha_post=alpha_post,
            beta_post=beta_post,
            posterior_mean=posterior_mean,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_width=ci_width
        )

    def analyze_multiple_priors(self,
                               priors: Dict[str, Tuple[float, float]]) -> List[PosteriorResult]:
        """
        分析多个先验

        参数:
            priors: 字典 {先验名称: (alpha, beta)}

        返回:
            PosteriorResult 列表
        """
        results = []
        for name, (alpha, beta) in priors.items():
            result = self.analyze_prior(name, alpha, beta)
            results.append(result)
        return results

    def assess_sensitivity(self, results: List[PosteriorResult],
                          threshold: float = 0.02) -> str:
        """
        评估敏感性

        参数:
            results: PosteriorResult 列表
            threshold: 判断敏感性的阈值（后验均值最大差异）

        返回:
            敏感性评估结论
        """
        means = [r.posterior_mean for r in results]
        mean_range = max(means) - min(means)

        if mean_range < threshold:
            return (f"✅ 结论对先验不敏感（差异 < {threshold:.1%}），"
                   f"当前数据（n={self.n}）足够强，能覆盖先验差异。")
        else:
            return (f"⚠️  结论对先验敏感（差异 = {mean_range:.1%}），"
                   f"建议收集更多数据以稳健估计。")

    def print_comparison_table(self, results: List[PosteriorResult]) -> None:
        """打印对比表格"""
        print("\n" + "=" * 70)
        print("先验敏感性分析结果")
        print("=" * 70)
        print(f"数据: {self.successes}/{self.n} = {self.observed_rate:.1%}")
        print("-" * 70)

        # 表头
        print(f"{'先验名称':<15} {'先验':<20} {'后验均值':<12} {'95% 可信区间':<20}")
        print("-" * 70)

        for r in results:
            prior_str = f"Beta({r.alpha_prior}, {r.beta_prior})"
            prior_mean = r.alpha_prior / (r.alpha_prior + r.beta_prior)
            ci_str = f"[{r.ci_lower:.1%}, {r.ci_upper:.1%}]"

            print(f"{r.prior_name:<15} {prior_str:<20} {r.posterior_mean:>10.1%}   {ci_str}")

        print("-" * 70)

        # 打印敏感性评估
        sensitivity = self.assess_sensitivity(results)
        print(f"\n{self.assess_sensitivity(results)}")

    def plot_sensitivity(self, results: List[PosteriorResult],
                        output_path: Path) -> None:
        """
        绘制敏感性分析图

        参数:
            results: PosteriorResult 列表
            output_path: 图片保存路径
        """
        setup_chinese_font()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：后验分布对比
        ax1 = axes[0]
        x = np.linspace(0.10, 0.25, 500)

        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

        for idx, r in enumerate(results):
            y = stats.beta.pdf(x, r.alpha_post, r.beta_post)
            ax1.plot(x, y, label=r.prior_name, color=colors[idx % len(colors)],
                    linewidth=2, alpha=0.8)
            ax1.axvline(r.posterior_mean, color=colors[idx % len(colors)],
                       linestyle='--', alpha=0.5, linewidth=1)

        # 标注观测值
        ax1.axvline(self.observed_rate, color='black', linestyle=':',
                   linewidth=2, label=f'观测值 {self.observed_rate:.1%}')

        ax1.set_xlabel('流失率 θ', fontsize=12)
        ax1.set_ylabel('概率密度', fontsize=12)
        ax1.set_title('不同先验下的后验分布', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)

        # 右图：后验均值对比（条形图）
        ax2 = axes[1]
        names = [r.prior_name for r in results]
        means = [r.posterior_mean * 100 for r in results]
        y_pos = np.arange(len(names))

        bars = ax2.barh(y_pos, means, color=colors[:len(results)], alpha=0.7, edgecolor='black')

        # 添加数值标签
        for bar, mean in zip(bars, means):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f'{mean:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

        # 标注观测值线
        ax2.axvline(self.observed_rate * 100, color='black', linestyle=':',
                   linewidth=2, label=f'观测值 {self.observed_rate:.1%}')

        ax2.set_xlabel('后验均值 (%)', fontsize=12)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(names)
        ax2.set_title('后验均值对比', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)

        plt.tight_layout()
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()


def large_data_example() -> None:
    """
    示例 1：数据量充足时，结论对先验不敏感
    """
    print("\n" + "=" * 60)
    print("示例 1：数据量充足（n=1000）")
    print("=" * 60)

    n = 1000
    churned = 180

    analyzer = PriorSensitivityAnalyzer(n, churned)

    priors = {
        '无信息': (1, 1),        # Beta(1,1) 均匀分布
        '弱信息': (5, 20),       # Beta(5,20) 均值 20%
        '市场部': (180, 820),    # 基于历史数据
        '产品部': (5, 15),       # 基于近期趋势
        '强信息': (150, 850),    # 强先验
    }

    results = analyzer.analyze_multiple_priors(priors)
    analyzer.print_comparison_table(results)

    # 绘图
    output_dir = Path(__file__).parent.parent / 'images'
    analyzer.plot_sensitivity(results, output_dir / '04_sensitivity_large_n.png')
    print(f"\n✅ 图片已保存: images/04_sensitivity_large_n.png")


def small_data_example() -> None:
    """
    示例 2：数据量不足时，结论对先验敏感
    """
    print("\n\n" + "=" * 60)
    print("示例 2：数据量不足（n=50）")
    print("=" * 60)

    n = 50
    churned = 10

    analyzer = PriorSensitivityAnalyzer(n, churned)

    priors = {
        '无信息': (1, 1),        # Beta(1,1) 均匀分布
        '弱信息': (5, 20),       # Beta(5,20) 均值 20%
        '强信息': (150, 850),    # 强先验（历史数据）
    }

    results = analyzer.analyze_multiple_priors(priors)
    analyzer.print_comparison_table(results)

    # 绘图
    output_dir = Path(__file__).parent.parent / 'images'
    analyzer.plot_sensitivity(results, output_dir / '04_sensitivity_small_n.png')
    print(f"\n✅ 图片已保存: images/04_sensitivity_small_n.png")

    print("\n🔍 关键洞察：")
    print(f"   数据少时（n={n}），不同先验的后验均值差异显著。")
    print(f"   强信息先验'主导'了后验，观测数据的影响力有限。")
    print(f"   这就是\"先验敏感\"——需要更多数据来收敛。")


def plot_data_vs_prior_sensitivity() -> None:
    """
    绘制数据量与先验敏感性的关系图
    """
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    # 模拟不同数据量下的后验均值差异
    sample_sizes = np.logspace(1, 3, 20)  # 10 到 1000
    observed_rate = 0.18

    # 两种极端先验
    prior_low = (1, 9)      # 均值 10%
    prior_high = (9, 1)     # 均值 90%

    ranges = []

    for n in sample_sizes:
        churned = int(n * observed_rate)

        # 低先验后验
        alpha_post_low = prior_low[0] + churned
        beta_post_low = prior_low[1] + (n - churned)
        mean_low = alpha_post_low / (alpha_post_low + beta_post_low)

        # 高先验后验
        alpha_post_high = prior_high[0] + churned
        beta_post_high = prior_high[1] + (n - churned)
        mean_high = alpha_post_high / (alpha_post_high + beta_post_high)

        ranges.append(abs(mean_high - mean_low))

    ax.plot(sample_sizes, np.array(ranges) * 100, 'o-', color='#e74c3c',
           linewidth=2, markersize=6)

    ax.set_xlabel('样本量 n', fontsize=12)
    ax.set_ylabel('后验均值差异 (%)', fontsize=12)
    ax.set_title('数据量 vs 先验敏感性', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # 添加说明
    insight = ("随着数据量增加，\n"
               "不同先验的后验会收敛。\n"
               "\"数据最终会战胜先验\"")
    ax.text(0.98, 0.95, insight, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    output_dir = Path(__file__).parent.parent / 'images'
    plt.tight_layout()
    plt.savefig(output_dir / '04_data_vs_sensitivity.png', dpi=150,
               bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n✅ 图片已保存: images/04_data_vs_sensitivity.png")


def main() -> None:
    """主函数：运行所有示例"""
    print("\n" + "=" * 60)
    print("先验敏感性分析示例")
    print("=" * 60)

    # 示例 1：大样本
    large_data_example()

    # 示例 2：小样本
    small_data_example()

    # 示例 3：数据量 vs 敏感性关系
    print("\n\n" + "=" * 60)
    print("示例 3：数据量与先验敏感性的关系")
    print("=" * 60)
    plot_data_vs_prior_sensitivity()


if __name__ == "__main__":
    main()
