"""
StatLab Week 14：贝叶斯分析模块

本模块整合本周所学的贝叶斯方法到 StatLab 分析流程中，提供：
1. 贝叶斯流失率估计（Beta-Binomial 模型）
2. 先验敏感性分析
3. 贝叶斯方式的不确定性量化
4. 生成贝叶斯分析报告片段

运行方式：
  python3 chapters/week_14/examples/statlab_week14.py

预期输出：
- 贝叶斯分析结果（打印到 stdout）
- 生成报告片段到 output/bayesian_report.md

注意：这是 StatLab 超级线的 Week 14 组件，应在上周基础上增量添加。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ===== 贝叶斯分析核心类 =====

@dataclass
class BayesianResult:
    """贝叶斯分析结果"""
    prior_name: str
    alpha_prior: float
    beta_prior: float
    prior_mean: float
    alpha_post: float
    beta_post: float
    posterior_mean: float
    ci_95_lower: float
    ci_95_upper: float


class BayesianChurnAnalyzer:
    """
    贝叶斯流失率分析器

    功能：
    1. 定义多个先验（用于敏感性分析）
    2. 计算后验分布
    3. 进行先验敏感性分析
    4. 生成贝叶斯报告
    """

    def __init__(self, n_total: int, n_churned: int):
        """
        初始化分析器

        参数:
            n_total: 总样本数
            n_churned: 流失数
        """
        self.n_total = n_total
        self.n_churned = n_churned
        self.n_retained = n_total - n_churned
        self.observed_rate = n_churned / n_total

    def define_priors(self) -> Dict[str, Tuple[float, float]]:
        """
        定义多个先验，用于敏感性分析

        返回:
            {先验名称: (alpha, beta)}
        """
        return {
            '无信息': (1, 1),           # Beta(1,1) 均匀分布
            '弱信息': (15, 85),         # Beta(15,85) 均值 15%
            '市场部': (180, 820),       # 基于历史数据：1000 个客户中 180 个流失
            '产品部': (5, 15),          # 基于近期趋势：流失率可能更高（25%）
        }

    def compute_posterior(self, alpha_prior: float,
                         beta_prior: float) -> BayesianResult:
        """
        计算单个先验的后验分布

        参数:
            alpha_prior: 先验 alpha 参数
            beta_prior: 先验 beta 参数

        返回:
            BayesianResult 对象
        """
        alpha_post = alpha_prior + self.n_churned
        beta_post = beta_prior + self.n_retained

        prior_mean = alpha_prior / (alpha_prior + beta_prior)
        posterior_mean = alpha_post / (alpha_post + beta_post)
        ci_lower, ci_upper = stats.beta.interval(0.95, alpha_post, beta_post)

        return BayesianResult(
            prior_name="",  # 由调用方设置
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            prior_mean=prior_mean,
            alpha_post=alpha_post,
            beta_post=beta_post,
            posterior_mean=posterior_mean,
            ci_95_lower=ci_lower,
            ci_95_upper=ci_upper
        )

    def prior_sensitivity_analysis(self) -> List[BayesianResult]:
        """
        进行先验敏感性分析

        返回:
            BayesianResult 列表
        """
        priors = self.define_priors()
        results = []

        for name, (alpha, beta) in priors.items():
            result = self.compute_posterior(alpha, beta)
            result.prior_name = name
            results.append(result)

        return results

    def assess_sensitivity(self, results: List[BayesianResult],
                          threshold: float = 0.02) -> Dict[str, any]:
        """
        评估先验敏感性

        参数:
            results: BayesianResult 列表
            threshold: 判断敏感性的阈值

        返回:
            包含敏感性评估的字典
        """
        means = [r.posterior_mean for r in results]
        mean_range = max(means) - min(means)

        return {
            'mean_range': mean_range,
            'is_sensitive': mean_range >= threshold,
            'interpretation': (
                f"后验均值对先验{'敏感' if mean_range >= threshold else '不敏感'}"
                f"（差异 = {mean_range:.1%}）"
            )
        }

    def probability_exceeds(self, threshold: float,
                          result: BayesianResult) -> float:
        """
        计算流失率超过阈值的后验概率

        参数:
            threshold: 阈值（如 0.15 表示 15%）
            result: BayesianResult 对象

        返回:
            P(θ > threshold | data)
        """
        return 1 - stats.beta.cdf(threshold, result.alpha_post, result.beta_post)


# ===== 报告生成器 =====

class BayesianReportGenerator:
    """
    贝叶斯分析报告生成器

    生成 Markdown 格式的贝叶斯分析报告片段
    """

    def __init__(self, analyzer: BayesianChurnAnalyzer):
        """
        初始化报告生成器

        参数:
            analyzer: BayesianChurnAnalyzer 实例
        """
        self.analyzer = analyzer

    def generate_markdown(self, output_path: Optional[Path] = None) -> str:
        """
        生成贝叶斯分析 Markdown 报告

        参数:
            output_path: 输出文件路径（可选）

        返回:
            Markdown 报告字符串
        """
        lines = []

        # 标题
        lines.append("## 贝叶斯分析：流失率估计")
        lines.append("")

        # 1. 数据概览
        lines.append("### 数据概览")
        lines.append("")
        lines.append(f"- 样本量: {self.analyzer.n_total}")
        lines.append(f"- 流失数: {self.analyzer.n_churned}")
        lines.append(f"- 观测流失率: {self.analyzer.observed_rate:.1%}")
        lines.append("")

        # 2. 先验假设
        lines.append("### 先验假设（不同部门）")
        lines.append("")
        lines.append("| 先验名称 | Beta 参数 | 均值 |")
        lines.append("|---------|----------|------|")

        priors = self.analyzer.define_priors()
        for name, (alpha, beta) in priors.items():
            mean = alpha / (alpha + beta)
            lines.append(f"| {name} | Beta({alpha}, {beta}) | {mean:.1%} |")
        lines.append("")

        # 3. 后验分布比较
        results = self.analyzer.prior_sensitivity_analysis()

        lines.append("### 后验分布比较")
        lines.append("")
        lines.append("| 先验 | 后验均值 | 95% 可信区间 |")
        lines.append("|------|---------|-------------|")

        for r in results:
            lines.append(f"| {r.prior_name} | {r.posterior_mean:.1%} | "
                        f"[{r.ci_95_lower:.1%}, {r.ci_95_upper:.1%}] |")
        lines.append("")

        # 4. 先验敏感性分析
        lines.append("### 先验敏感性分析")
        lines.append("")

        sensitivity = self.analyzer.assess_sensitivity(results)
        lines.append(f"**结论**: {sensitivity['interpretation']}")

        if not sensitivity['is_sensitive']:
            lines.append(f"当前数据（n={self.analyzer.n_total}）足够强，")
            lines.append("能覆盖不同先验之间的差异。结论稳健。")
        else:
            lines.append(f"建议收集更多数据以稳健估计。")
        lines.append("")

        # 5. 业务问题回答
        lines.append("### 业务问题回答")
        lines.append("")

        # 使用市场部先验（最常用）
        market_result = [r for r in results if r.prior_name == '市场部'][0]

        # P(θ > 15% | data)
        prob_gt_15 = self.analyzer.probability_exceeds(0.15, market_result)

        lines.append(f"**Q: 流失率超过 15% 的概率有多大？**")
        lines.append("")
        lines.append(f"A: {prob_gt_15:.1%}（基于市场部先验）")
        lines.append("")

        # P(θ > 20% | data)
        prob_gt_20 = self.analyzer.probability_exceeds(0.20, market_result)

        lines.append(f"**Q: 流失率超过 20% 的概率有多大？**")
        lines.append("")
        lines.append(f"A: {prob_gt_20:.1%}（基于市场部先验）")
        lines.append("")

        # 6. 与频率学派对比
        lines.append("### 与频率学派对比")
        lines.append("")

        # 频率学派置信区间
        se = np.sqrt(
            self.analyzer.observed_rate *
            (1 - self.analyzer.observed_rate) /
            self.analyzer.n_total
        )
        ci_freq_lower = self.analyzer.observed_rate - 1.96 * se
        ci_freq_upper = self.analyzer.observed_rate + 1.96 * se

        lines.append(f"- 频率学派点估计: {self.analyzer.observed_rate:.1%}")
        lines.append(f"- 频率学派 95% 置信区间: [{ci_freq_lower:.1%}, {ci_freq_upper:.1%}]")
        lines.append(f"- 贝叶斯后验均值（市场部先验）: {market_result.posterior_mean:.1%}")
        lines.append(f"- 贝叶斯 95% 可信区间（市场部先验）: "
                    f"[{market_result.ci_95_lower:.1%}, {market_result.ci_95_upper:.1%}]")
        lines.append("")

        lines.append("**解释差异**：")
        lines.append("- 频率学派：不能说\"参数有 95% 的概率在这个区间里\"")
        lines.append("- 贝叶斯学派：可以说\"参数有 95% 的概率在这个区间里\"")
        lines.append("")

        # 保存文件
        report = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding='utf-8')
            print(f"✅ 贝叶斯报告已保存到: {output_path}")

        return report


# ===== 主函数 =====

def main() -> None:
    """主函数：演示 StatLab 贝叶斯分析流程"""
    print("\n" + "=" * 60)
    print("StatLab Week 14：贝叶斯分析模块")
    print("=" * 60)

    # 示例数据：1000 个客户中 180 个流失
    n_total = 1000
    n_churned = 180

    print(f"\n数据：{n_churned}/{n_total} = {n_churned/n_total:.1%}")

    # 创建分析器
    analyzer = BayesianChurnAnalyzer(n_total, n_churned)

    # 先验敏感性分析
    print("\n" + "-" * 40)
    print("先验敏感性分析")
    print("-" * 40)

    results = analyzer.prior_sensitivity_analysis()

    print(f"\n{'先验':<12} {'先验均值':<12} {'后验均值':<12} {'95% CI'}")
    print("-" * 50)

    for r in results:
        ci_str = f"[{r.ci_95_lower:.1%}, {r.ci_95_upper:.1%}]"
        print(f"{r.prior_name:<12} {r.prior_mean:>10.1%}   "
             f"{r.posterior_mean:>10.1%}   {ci_str}")

    # 敏感性评估
    sensitivity = analyzer.assess_sensitivity(results)
    print(f"\n{sensitivity['interpretation']}")

    # 业务问题
    print("\n" + "-" * 40)
    print("业务问题回答（贝叶斯方式）")
    print("-" * 40)

    market_result = [r for r in results if r.prior_name == '市场部'][0]

    prob_gt_15 = analyzer.probability_exceeds(0.15, market_result)
    prob_gt_20 = analyzer.probability_exceeds(0.20, market_result)

    print(f"\nQ: 流失率超过 15% 的概率有多大？")
    print(f"A: {prob_gt_15:.1%}（基于市场部先验）")

    print(f"\nQ: 流失率超过 20% 的概率有多大？")
    print(f"A: {prob_gt_20:.1%}（基于市场部先验）")

    # 生成报告
    print("\n" + "-" * 40)
    print("生成贝叶斯分析报告")
    print("-" * 40)

    generator = BayesianReportGenerator(analyzer)
    output_path = Path(__file__).parent.parent.parent / 'output' / 'bayesian_report.md'
    generator.generate_markdown(output_path)

    print("\n" + "=" * 60)
    print("StatLab 贝叶斯分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
