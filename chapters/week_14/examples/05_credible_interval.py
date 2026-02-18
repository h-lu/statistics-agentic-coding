"""
示例：可信区间 vs 置信区间——两种不确定性的表达方式

本例演示贝叶斯可信区间（Credible Interval）与频率学派置信区间（Confidence Interval）
的区别。这是两种哲学对"不确定性"的不同回答。

运行方式：python3 chapters/week_14/examples/05_credible_interval.py

预期输出：
- 两种区间的数值对比
- 解释上的差异说明
- 生成可视化对比图到 images/

核心区别：
- 频率学派：参数是固定的，区间是随机的
  - 95% CI：如果重复抽样 100 次，约 95 个区间会包含真实参数
  - 不能说"参数有 95% 的概率在这个区间里"

- 贝叶斯学派：参数是随机的，区间是固定的
  - 95% 可信区间：给定数据，参数有 95% 的概率在这个区间里
  - 可以直接说"参数有 95% 的概率在这个区间里"
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


class IntervalComparison:
    """
    置信区间 vs 可信区间对比分析
    """

    def __init__(self, n: int, successes: int,
                 prior_alpha: float = 15, prior_beta: float = 85):
        """
        初始化

        参数:
            n: 总样本数
            successes: 成功次数
            prior_alpha, prior_beta: 贝叶斯先验参数
        """
        self.n = n
        self.successes = successes
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.observed_p = successes / n

    def frequentist_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """
        计算频率学派置信区间（正态近似）

        使用 Wald 方法：p̂ ± z × SE
        其中 SE = sqrt(p̂(1-p̂)/n)

        参数:
            confidence: 置信水平

        返回:
            (ci_lower, ci_upper)
        """
        z = stats.norm.ppf((1 + confidence) / 2)
        se = np.sqrt(self.observed_p * (1 - self.observed_p) / self.n)

        ci_lower = self.observed_p - z * se
        ci_upper = self.observed_p + z * se

        return max(0, ci_lower), min(1, ci_upper)

    def frequentist_ci_exact(self, confidence: float = 0.95) -> tuple[float, float]:
        """
        计算精确的频率学派置信区间（Clopper-Pearson）

        使用 Beta 分布分位数

        参数:
            confidence: 置信水平

        返回:
            (ci_lower, ci_upper)
        """
        alpha = 1 - confidence
        ci_lower = stats.beta.ppf(alpha/2, self.successes, self.n - self.successes + 1)
        ci_upper = stats.beta.ppf(1 - alpha/2, self.successes + 1, self.n - self.successes)

        return max(0, ci_lower), min(1, ci_upper)

    def bayesian_ci(self, confidence: float = 0.95) -> tuple[float, float]:
        """
        计算贝叶斯可信区间（Beta-Binomial 共轭后验）

        后验：Beta(alpha_prior + successes, beta_prior + failures)

        参数:
            confidence: 可信水平

        返回:
            (ci_lower, ci_upper)
        """
        alpha_post = self.prior_alpha + self.successes
        beta_post = self.prior_beta + (self.n - self.successes)

        ci_lower, ci_upper = stats.beta.interval(confidence, alpha_post, beta_post)

        return ci_lower, ci_upper

    def bayesian_posterior_mean(self) -> float:
        """计算贝叶斯后验均值"""
        alpha_post = self.prior_alpha + self.successes
        beta_post = self.prior_beta + (self.n - self.successes)
        return alpha_post / (alpha_post + beta_post)

    def print_comparison(self) -> None:
        """打印对比结果"""
        print("\n" + "=" * 70)
        print("置信区间 vs 可信区间对比")
        print("=" * 70)
        print(f"\n数据：{self.successes}/{self.n} = {self.observed_p:.1%}")
        print(f"贝叶斯先验：Beta({self.prior_alpha}, {self.prior_beta})")
        print("-" * 70)

        # 频率学派
        ci_freq_approx = self.frequentist_ci()
        ci_freq_exact = self.frequentist_ci_exact()

        print("\n### 频率学派（Frequentist）")
        print(f"  点估计：{self.observed_p:.4f}")
        print(f"  95% 置信区间（正态近似）：[{ci_freq_approx[0]:.4f}, {ci_freq_approx[1]:.4f}]")
        print(f"  95% 置信区间（精确）：      [{ci_freq_exact[0]:.4f}, {ci_freq_exact[1]:.4f}]")
        print(f"\n  解释：如果重复抽样 100 次，约 95 个区间会包含真实参数。")
        print(f"       ⚠️  不能说：\"参数有 95% 的概率在这个区间里\"")

        # 贝叶斯学派
        ci_bayes = self.bayesian_ci()
        post_mean = self.bayesian_posterior_mean()

        print("\n### 贝叶斯学派（Bayesian）")
        print(f"  后验均值：{post_mean:.4f}")
        print(f"  95% 可信区间：[{ci_bayes[0]:.4f}, {ci_bayes[1]:.4f}]")
        print(f"\n  解释：给定数据，参数有 95% 的概率在这个区间里。")
        print(f"       ✅ 可以说：\"参数有 95% 的概率在这个区间里\"")

        # 对比
        print("\n### 关键区别")
        print("-" * 70)
        print(f"{'维度':<20} {'频率学派':<25} {'贝叶斯学派':<25}")
        print("-" * 70)
        print(f"{'参数':<20} {'固定但未知':<25} {'随机变量(有分布)':<25}")
        print(f"{'区间':<20} {'随机(随样本变化)':<25} {'固定(给定数据)':<25}")
        print(f"{'解释':<20} {'长期频率':<25} {'信念程度':<25}")
        # 需要转义引号，使用变量
        freq_say = '"95%的区间包含参数"'
        bayes_say = '"参数在区间的概率95%"'
        print(f"{'可以说':<20} {freq_say:<25} {bayes_say:<25}")

    def plot_comparison(self, output_path: Path) -> None:
        """绘制对比图"""
        setup_chinese_font()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：频率学派视角
        ax1 = axes[0]

        # 模拟多次抽样，展示"区间是随机的"
        np.random.seed(42)
        n_sim = 50
        true_p = 0.17  # 假设真实参数

        ci_lower_list = []
        ci_upper_list = []
        contains_list = []

        for _ in range(n_sim):
            sim_successes = np.random.binomial(self.n, true_p)
            sim_p = sim_successes / self.n
            se = np.sqrt(sim_p * (1 - sim_p) / self.n)
            ci_l = max(0, sim_p - 1.96 * se)
            ci_u = min(1, sim_p + 1.96 * se)

            ci_lower_list.append(ci_l)
            ci_upper_list.append(ci_u)
            contains_list.append(ci_l <= true_p <= ci_u)

        y_pos = np.arange(n_sim)

        for i, (ci_l, ci_u, contains) in enumerate(zip(ci_lower_list, ci_upper_list, contains_list)):
            color = '#2ecc71' if contains else '#e74c3c'
            ax1.hlines(i, ci_l, ci_u, colors=color, linewidth=2, alpha=0.7)
            ax1.plot(ci_l, i, '>', color=color, markersize=5)
            ax1.plot(ci_u, i, '<', color=color, markersize=5)

        ax1.axvline(true_p, color='black', linestyle='--', linewidth=2,
                   label=f'真实参数 = {true_p:.2f}')
        ax1.set_xlabel('参数值', fontsize=12)
        ax1.set_yticks([])
        ax1.set_title('频率学派：区间是随机的', fontsize=13, fontweight='bold')
        ax1.legend()

        # 右图：贝叶斯学派视角
        ax2 = axes[1]

        # 绘制后验分布
        alpha_post = self.prior_alpha + self.successes
        beta_post = self.prior_beta + (self.n - self.successes)

        x = np.linspace(0.10, 0.25, 500)
        y = stats.beta.pdf(x, alpha_post, beta_post)

        ax2.plot(x, y, color='#3498db', linewidth=2.5, label='后验分布')

        # 标记 95% 可信区间
        ci_l, ci_u = stats.beta.interval(0.95, alpha_post, beta_post)
        ax2.fill_between(x, 0, y, where=(x >= ci_l) & (x <= ci_u),
                        color='#3498db', alpha=0.3, label='95% 可信区间')

        # 标记后验均值
        post_mean = alpha_post / (alpha_post + beta_post)
        ax2.axvline(post_mean, color='red', linestyle='--', linewidth=2,
                   label=f'后验均值 = {post_mean:.3f}')

        ax2.set_xlabel('流失率 θ', fontsize=12)
        ax2.set_ylabel('概率密度', fontsize=12)
        ax2.set_title('贝叶斯学派：参数是随机的', fontsize=13, fontweight='bold')
        ax2.legend()

        plt.tight_layout()
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

    def plot_interpretation_difference(self, output_path: Path) -> None:
        """绘制解释差异的可视化"""
        setup_chinese_font()
        fig, ax = plt.subplots(figsize=(12, 5))

        # 频率学派解释
        freq_text = (
            "频率学派：\n\n"
            "\"如果我们重复抽样 100 次，\n"
            "约 95 个区间会包含真实参数\"\n\n"
            "⚠️  问题是：我们只有一次抽样\n"
            "⚠️  不能说：\"参数有 95% 的概率在这个区间里\"\n"
            "⚠️  只能说：\"这个区间是 95% 置信水平下构造的\""
        )

        # 贝叶斯学派解释
        bayes_text = (
            "贝叶斯学派：\n\n"
            "\"给定数据，\n"
            "参数有 95% 的概率在这个区间里\"\n\n"
            "✅ 直接回答了我们要问的问题\n"
            "✅ 可以说：\"参数有 95% 的概率在 [a, b] 之间\"\n"
            "✅ 符合直觉和业务需求"
        )

        ax.text(0.25, 0.5, freq_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='center', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.8, edgecolor='#e74c3c', linewidth=2))

        ax.text(0.75, 0.5, bayes_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='center', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8, edgecolor='#2ecc71', linewidth=2))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('两种哲学对"不确定性"的不同回答', fontsize=14, fontweight='bold', y=0.95)

        plt.tight_layout()
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()


def main() -> None:
    """主函数：运行所有示例"""
    print("\n" + "=" * 60)
    print("可信区间 vs 置信区间对比示例")
    print("=" * 60)

    # 数据：1000 个客户中 180 个流失
    n = 1000
    churned = 180

    comparison = IntervalComparison(n, churned, prior_alpha=15, prior_beta=85)

    # 打印对比
    comparison.print_comparison()

    # 绘图
    output_dir = Path(__file__).parent.parent / 'images'

    print("\n### 生成图表")
    print("-" * 40)
    comparison.plot_comparison(output_dir / '05_interval_comparison.png')
    print("✅ 图片已保存: images/05_interval_comparison.png")

    comparison.plot_interpretation_difference(output_dir / '05_interpretation_diff.png')
    print("✅ 图片已保存: images/05_interpretation_diff.png")

    print("\n### 业务场景对比")
    print("-" * 40)
    print("\n业务方问：\"流失率超过 15% 的概率有多大？\"")
    print("\n频率学派：")
    print("  - 只能说：\"p < 0.05，拒绝原假设\"")
    print("  - 不能直接回答概率问题")
    print("\n贝叶斯学派：")
    print("  - 可以直接计算：P(θ > 15% | data)")
    alpha_post = 15 + 180
    beta_post = 85 + 820
    prob_gt_15 = 1 - stats.beta.cdf(0.15, alpha_post, beta_post)
    print(f"  - 答案：{prob_gt_15:.1%}")
    print("  - 这正是业务方想要的答案")


if __name__ == "__main__":
    main()
