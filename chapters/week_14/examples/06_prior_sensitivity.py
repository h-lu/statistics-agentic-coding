"""
示例：先验敏感性分析——测试结论的稳健性

运行方式：python3 chapters/week_14/examples/06_prior_sensitivity.py
预期输出：不同先验下的后验对比、敏感性分析报告
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def prior_sensitivity_analysis(
    conversions: int,
    exposures: int,
    prior_options: dict[str, tuple[float, float]],
    n_samples: int = 10000,
    random_seed: int = 42
) -> dict:
    """
    先验敏感性分析

    测试不同先验对后验的影响

    参数:
        conversions: 转化数
        exposures: 曝光数
        prior_options: 先验选项字典 {名称: (alpha, beta)}
        n_samples: 后验采样数量
        random_seed: 随机种子

    返回:
        包含每种先验下后验统计量的字典
    """
    np.random.seed(random_seed)
    results = {}

    mle = conversions / exposures  # 最大似然估计

    for prior_name, (alpha_prior, beta_prior) in prior_options.items():
        # 计算后验
        posterior = stats.beta(
            alpha_prior + conversions,
            beta_prior + exposures - conversions
        )

        # 采样
        samples = posterior.rvs(n_samples)

        results[prior_name] = {
            'prior': (alpha_prior, beta_prior),
            'posterior_mean': posterior.mean(),
            'posterior_std': posterior.std(),
            'ci_95': posterior.interval(0.95),
            'samples': samples,
            'distance_from_mle': abs(posterior.mean() - mle)
        }

    return results


def assess_robustness(results: dict, tolerance: float = 0.005) -> dict:
    """
    评估结论稳健性

    参数:
        results: prior_sensitivity_analysis 的返回结果
        tolerance: 容忍的差异阈值（如 0.5%）

    返回:
        包含稳健性评估的字典
    """
    # 提取所有后验均值
    posterior_means = [r['posterior_mean'] for r in results.values()]

    # 计算最大差异
    max_diff = max(posterior_means) - min(posterior_means)

    # 判断是否稳健
    robust = max_diff < tolerance

    # 找出最极端的先验
    distances = {name: r['distance_from_mle']
                for name, r in results.items()}
    most_influential = max(distances.items(), key=lambda x: x[1])

    return {
        'robust': robust,
        'max_diff': max_diff,
        'tolerance': tolerance,
        'most_influential_prior': most_influential[0],
        'most_influential_distance': most_influential[1]
    }


# 坏例子：只试一种先验就下结论
def bad_single_prior_analysis(conversions: int, exposures: int) -> str:
    """
    反例：只试一种先验就下结论

    问题：
    1. 无法判断结论是否依赖于先验选择
    2. 如果先验选择不当，结论可能误导
    3. 不符合贝叶斯分析的透明性原则
    """
    posterior = stats.beta(1 + conversions, 1 + exposures - conversions)
    return f"后验均值: {posterior.mean():.4f}"


def plot_sensitivity(results: dict, save_path: str = None):
    """
    绘制先验敏感性对比图

    参数:
        results: prior_sensitivity_analysis 的返回结果
        save_path: 图片保存路径
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1：后验分布对比
    ax1 = axes[0]
    x = np.linspace(0, 0.12, 500)

    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for i, (name, result) in enumerate(results.items()):
        posterior = stats.beta(
            result['prior'][0],
            result['prior'][1]
        )
        # 注意：这里画的是后验分布
        posterior_dist = stats.beta(
            result['prior'][0],
            result['prior'][1]
        )
        # 重新计算后验参数
        # 假设 conversions 和 exposures 是全局的（简化）
        # 实际应用中应该传入

    # 简化版：直接用采样结果画 KDE
    for i, (name, result) in enumerate(results.items()):
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(result['samples'])
        ax1.plot(x, kde(x), color=colors[i % len(colors)],
                 linewidth=2, label=name)

    ax1.set_xlabel('转化率')
    ax1.set_ylabel('概率密度')
    ax1.set_title('不同先验下的后验分布', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图 2：后验均值对比
    ax2 = axes[1]
    names = list(results.keys())
    means = [results[name]['posterior_mean'] for name in names]

    bars = ax2.bar(range(len(names)), means, color=colors[:len(names)])
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.set_ylabel('后验均值')
    ax2.set_title('后验均值对比', fontsize=12)
    ax2.grid(True, axis='y', alpha=0.3)

    # 添加数值标签
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"✅ 敏感性分析图已保存: {save_path}")


def generate_sensitivity_report(
    conversions: int,
    exposures: int,
    results: dict,
    assessment: dict
) -> str:
    """
    生成先验敏感性分析报告

    参数:
        conversions: 转化数
        exposures: 曝光数
        results: prior_sensitivity_analysis 的返回结果
        assessment: assess_robustness 的返回结果

    返回:
        Markdown 格式的报告字符串
    """
    mle = conversions / exposures

    report = f"""
# 先验敏感性分析报告

## 数据概况

- 观测数据: {conversions}/{exposures}
- 最大似然估计 (MLE): {mle:.4f}

## 测试的先验

"""

    for name, result in results.items():
        alpha, beta = result['prior']
        prior_mean = alpha / (alpha + beta)
        report += f"""
### {name}

- 先验参数: Beta({alpha}, {beta})
- 先验均值: {prior_mean:.4f}
- 后验均值: {result['posterior_mean']:.4f}
- 后验标准差: {result['posterior_std']:.4f}
- 95% 可信区间: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]
"""

    report += f"""

## 稳健性评估

- **后验均值最大差异**: {assessment['max_diff']:.4f}
- **稳健性判断**: {'✅ 结论稳健（差异 < 容忍阈值）' if assessment['robust'] else '⚠️ 结论对先验选择敏感'}
- **影响最大的先验**: {assessment['most_influential_prior']}
- **与 MLE 的最大偏差**: {assessment['most_influential_distance']:.4f}

## 建议

{'结论稳健，可以使用任何合理的先验。' if assessment['robust'] else '结论对先验选择敏感，建议：1) 在报告中说明先验选择依据；2) 收集更多数据以减少先验影响；3) 使用弱信息先验作为保守选择。'}
"""
    return report


def main() -> None:
    # 示例数据
    conversions, exposures = 58, 1000

    print("=" * 60)
    print("先验敏感性分析")
    print("=" * 60)
    print(f"\n数据: {conversions}/{exposures} 转化")
    print(f"MLE（最大似然估计）: {conversions/exposures:.4f}")
    print()

    # 定义先验选项
    prior_options = {
        '无信息先验': (1, 1),           # Beta(1,1) = 均匀
        '弱信息先验': (2, 40),          # 均值约 4.8%
        '强信息先验': (50, 1000),        # 均值约 4.8%
        'Jeffreys先验': (0.5, 0.5),     # 无信息先验的变体
        '极端乐观（反例）': (10, 10)     # 均值 50%（不合理）
    }

    # 运行敏感性分析
    results = prior_sensitivity_analysis(conversions, exposures, prior_options)

    # 打印结果
    print("\n" + "=" * 60)
    print("后验统计摘要")
    print("=" * 60)
    print(f"\n{'先验类型':<15} {'后验均值':>10} {'标准差':>10} {'95% CI':>20}")
    print("-" * 60)

    for name, result in results.items():
        mean = result['posterior_mean']
        std = result['posterior_std']
        ci_low, ci_high = result['ci_95']
        print(f"{name:<15} {mean:>10.4f} {std:>10.4f} [{ci_low:.4f}, {ci_high:.4f}]")

    # 评估稳健性
    assessment = assess_robustness(results, tolerance=0.005)

    print("\n" + "=" * 60)
    print("稳健性评估")
    print("=" * 60)
    print(f"\n后验均值最大差异: {assessment['max_diff']:.4f}")
    print(f"容忍阈值: {assessment['tolerance']:.4f}")
    print(f"稳健性: {'✅ 结论稳健' if assessment['robust'] else '⚠️ 结论对先验敏感'}")
    print(f"影响最大的先验: {assessment['most_influential_prior']}")

    # 绘图
    print("\n" + "=" * 60)
    print("生成敏感性分析图...")
    print("=" * 60)
    plot_sensitivity(results, 'chapters/week_14/examples/prior_sensitivity.png')

    # 生成报告
    report = generate_sensitivity_report(conversions, exposures, results, assessment)

    # 保存报告
    report_path = 'chapters/week_14/examples/sensitivity_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 敏感性分析报告已保存: {report_path}")

    # 对比：坏例子
    print("\n" + "=" * 60)
    print("【反例】只试一种先验")
    print("=" * 60)
    bad_result = bad_single_prior_analysis(conversions, exposures)
    print(f"❌ 错误做法：{bad_result}")
    print("   问题：不知道这个结论是否依赖于先验选择！")
    print("   正确做法：必须测试多种先验，评估稳健性")


if __name__ == "__main__":
    main()
