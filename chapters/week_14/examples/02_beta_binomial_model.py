"""
示例：Beta-Binomial 共轭先验模型

本例演示贝叶斯统计中最重要的共轭先验对之一：Beta 先验 + Binomial 似然
后验仍是 Beta 分布，计算非常简洁。

运行方式：python3 chapters/week_14/examples/02_beta_binomial_model.py

预期输出：
- 打印不同先验下的后验分布参数
- 生成先验/后验分布对比图到 images/

关键概念：
- Beta 分布是 [0,1] 区间参数（如概率、比例）的共轭先验
- 共轭先验意味着后验与先验同族，计算方便
- 后验参数 = 先验参数 + 观测数据
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def setup_chinese_font() -> str:
    """配置中文字体"""
    import matplotlib.font_manager as fm
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


class BetaBinomialModel:
    """
    Beta-Binomial 共轭先验模型

    属性:
        alpha_prior: 先验 Beta 分布的 alpha 参数
        beta_prior: 先验 Beta 分布的 beta 参数
    """

    def __init__(self, alpha_prior: float, beta_prior: float):
        """
        初始化先验分布

        参数:
            alpha_prior: Beta(alpha, beta) 的 alpha 参数
            beta_prior: Beta(alpha, beta) 的 beta 参数
        """
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def update(self, successes: int, failures: int) -> tuple[float, float]:
        """
        用数据更新后验分布

        共轭先验的优美性质：后验参数 = 先验参数 + 数据

        参数:
            successes: 成功次数（如流失客户数）
            failures: 失败次数（如未流失客户数）

        返回:
            (alpha_post, beta_post): 后验 Beta 分布的参数
        """
        self.alpha_post = self.alpha_prior + successes
        self.beta_post = self.beta_prior + failures
        return self.alpha_post, self.beta_post

    def posterior_mean(self) -> float:
        """计算后验均值: alpha / (alpha + beta)"""
        if not hasattr(self, 'alpha_post'):
            raise ValueError("请先调用 update() 方法更新后验")
        return self.alpha_post / (self.alpha_post + self.beta_post)

    def posterior_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """
        计算后验可信区间

        参数:
            confidence: 置信水平（默认 0.95）

        返回:
            (ci_low, ci_high): 可信区间的下界和上界
        """
        if not hasattr(self, 'alpha_post'):
            raise ValueError("请先调用 update() 方法更新后验")
        return stats.beta.interval(confidence, self.alpha_post, self.beta_post)

    def prior_mean(self) -> float:
        """计算先验均值"""
        return self.alpha_prior / (self.alpha_prior + self.beta_prior)

    def prior_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """计算先验可信区间"""
        return stats.beta.interval(confidence, self.alpha_prior, self.beta_prior)


def plot_beta_distribution(alpha: float, beta: float,
                           ax: plt.Axes, label: str, color: str) -> None:
    """
    在指定轴上绘制 Beta 分布的 PDF

    参数:
        alpha, beta: Beta 分布参数
        ax: matplotlib 轴对象
        label: 图例标签
        color: 线条颜色
    """
    x = np.linspace(0, 1, 500)
    y = stats.beta.pdf(x, alpha, beta)
    ax.plot(x, y, label=label, color=color, linewidth=2)

    # 标注均值
    mean = alpha / (alpha + beta)
    ax.axvline(mean, color=color, linestyle='--', alpha=0.5)


def compare_priors() -> None:
    """
    比较不同先验选择对后验的影响

    场景：观察到 180/1000 的流失率
    """
    print("=" * 60)
    print("Beta-Binomial 模型：不同先验的影响")
    print("=" * 60)

    # 数据
    n = 1000
    churned = 180
    not_churned = n - churned
    observed_rate = churned / n

    print(f"\n观测数据：{churned}/{n} = {observed_rate:.1%}")

    # 三种先验
    priors = {
        '无信息先验 Beta(1,1)': (1, 1),
        '弱信息先验 Beta(15,85)': (15, 85),
        '强信息先验 Beta(150,850)': (150, 850),
    }

    results = {}

    for name, (alpha, beta) in priors.items():
        model = BetaBinomialModel(alpha, beta)
        model.update(churned, not_churned)

        prior_mean = model.prior_mean()
        post_mean = model.posterior_mean()
        post_ci = model.posterior_ci()

        results[name] = {
            'prior_mean': prior_mean,
            'post_mean': post_mean,
            'post_ci': post_ci,
            'model': model
        }

        print(f"\n{name}")
        print(f"  先验均值: {prior_mean:.1%}")
        print(f"  后验均值: {post_mean:.1%}")
        print(f"  95% 可信区间: [{post_ci[0]:.1%}, {post_ci[1]:.1%}]")

    # 绘图
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for idx, ((name, data), color) in enumerate(zip(results.items(), colors)):
        ax = axes[idx]
        model = data['model']

        # 绘制先验
        plot_beta_distribution(model.alpha_prior, model.beta_prior,
                              ax, '先验', colors[0])

        # 绘制后验
        plot_beta_distribution(model.alpha_post, model.beta_post,
                              ax, '后验', colors[1])

        # 标注观测值
        ax.axvline(observed_rate, color='black', linestyle=':',
                  linewidth=2, label=f'观测值 {observed_rate:.1%}')

        ax.set_title(name, fontsize=11)
        ax.set_xlabel('流失率 θ')
        ax.set_ylabel('概率密度')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        ax.legend(fontsize=9)

    plt.tight_layout()
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '02_beta_binomial_priors.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图片已保存到: {output_dir / '02_beta_binomial_priors.png'}")


def plot_posterior_comparison() -> None:
    """
    绘制后验分布对比图，直观展示不同先验的影响
    """
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0.10, 0.25, 500)

    # 数据
    n = 1000
    churned = 180

    # 三种后验
    posteriors = [
        ((1 + churned, 1 + n - churned), '无信息先验 Beta(1,1) → 后验', '#3498db'),
        ((15 + churned, 85 + n - churned), '弱信息先验 Beta(15,85) → 后验', '#2ecc71'),
        ((150 + churned, 850 + n - churned), '强信息先验 Beta(150,850) → 后验', '#e74c3c'),
    ]

    for (alpha, beta), label, color in posteriors:
        y = stats.beta.pdf(x, alpha, beta)
        ax.plot(x, y, label=label, color=color, linewidth=2)
        mean = alpha / (alpha + beta)
        ax.axvline(mean, color=color, linestyle='--', alpha=0.6, linewidth=1.5)

    # 标注观测值
    observed = churned / n
    ax.axvline(observed, color='black', linestyle=':', linewidth=2,
              label=f'观测值 {observed:.1%}')

    ax.set_xlabel('流失率 θ', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('不同先验下的后验分布对比 (n=1000, 流失=180)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)

    # 添加关键洞察文本
    insight = ("关键洞察：当数据量足够大时，\n"
               "不同先验的后验会收敛到观测值附近。\n"
               "这就是\"数据最终会战胜先验\"。")
    ax.text(0.98, 0.95, insight, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    output_dir = Path(__file__).parent.parent / 'images'
    plt.savefig(output_dir / '02_posterior_comparison.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"图片已保存到: {output_dir / '02_posterior_comparison.png'}")


def main() -> None:
    """主函数：运行所有示例"""
    print("\n" + "=" * 60)
    print("Beta-Binomial 共轭先验模型")
    print("=" * 60)

    # 示例 1：市场部先验（基于历史数据）
    print("\n### 示例 1：市场部的强先验")
    print("-" * 40)

    # 市场部先验：基于历史 1000 个客户，180 个流失
    alpha_mkt = 180
    beta_mkt = 820

    model = BetaBinomialModel(alpha_mkt, beta_mkt)
    print(f"市场部先验：Beta({alpha_mkt}, {beta_mkt})")
    print(f"  先验均值：{model.prior_mean():.1%}")
    print(f"  先验 95% CI：[{model.prior_ci()[0]:.1%}, {model.prior_ci()[1]:.1%}]")

    # 新数据：100 个客户中 22 个流失
    n_new = 100
    churned_new = 22
    print(f"\n新数据：{churned_new}/{n_new} = {churned_new/n_new:.1%}")

    model.update(churned_new, n_new - churned_new)
    print(f"\n后验：Beta({model.alpha_post}, {model.beta_post})")
    print(f"  后验均值：{model.posterior_mean():.1%}")
    print(f"  后验 95% CI：[{model.posterior_ci()[0]:.1%}, {model.posterior_ci()[1]:.1%}]")

    # 示例 2：产品部先验（弱先验）
    print("\n\n### 示例 2：产品部的弱先验")
    print("-" * 40)

    # 产品部先验：基于"感觉"，流失率可能更高
    alpha_prod = 5
    beta_prod = 15

    model2 = BetaBinomialModel(alpha_prod, beta_prod)
    print(f"产品部先验：Beta({alpha_prod}, {beta_prod})")
    print(f"  先验均值：{model2.prior_mean():.1%}")

    model2.update(churned_new, n_new - churned_new)
    print(f"\n后验：Beta({model2.alpha_post}, {model2.beta_post})")
    print(f"  后验均值：{model2.posterior_mean():.1%}")
    print(f"  后验 95% CI：[{model2.posterior_ci()[0]:.1%}, {model2.posterior_ci()[1]:.1%}]")

    print("\n### 对比")
    print(f"  市场部强先验 → 后验 {model.posterior_mean():.1%}（被历史数据\"拉住\"）")
    print(f"  产品部弱先验 → 后验 {model2.posterior_mean():.1%}（被新数据\"拉动\"）")

    # 示例 3：比较不同先验
    print("\n\n")
    compare_priors()

    # 示例 4：后验对比图
    print("\n\n")
    plot_posterior_comparison()


if __name__ == "__main__":
    main()
