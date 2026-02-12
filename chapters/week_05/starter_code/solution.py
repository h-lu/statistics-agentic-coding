"""
Week 05: StatLab 不确定性量化模块

本 starter code 提供了 StatLab 报告中"不确定性量化"章节的基础框架。
学生需要补充完成标记为 # TODO 的部分。

概念覆盖：
- Bootstrap 均值置信区间
- Bootstrap 相关系数置信区间
- 报告生成
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# 设置随机种子确保可复现
np.random.seed(42)


def bootstrap_ci(
    data: np.ndarray,
    stat_func: Callable,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> Tuple[float, float, np.ndarray]:
    """
    计算 Bootstrap 置信区间

    Parameters
    ----------
    data : np.ndarray
        原始样本数据
    stat_func : Callable
        统计量函数（如 np.mean, np.median）
    n_bootstrap : int
        Bootstrap 重采样次数
    confidence : float
        置信水平

    Returns
    -------
    lower : float
        置信区间下界
    upper : float
        置信区间上界
    bootstrap_stats : np.ndarray
        所有 Bootstrap 统计量（用于绘图）
    """
    n = len(data)
    bootstrap_stats = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # TODO: 实现有放回重采样
        # 提示：使用 np.random.choice 的 replace 参数
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = stat_func(resample)

    # 计算置信区间
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return lower, upper, bootstrap_stats


def bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    计算 Pearson 相关系数的 Bootstrap 置信区间

    Parameters
    ----------
    x, y : np.ndarray
        两个变量的观测值
    n_bootstrap : int
        Bootstrap 重采样次数
    confidence : float
        置信水平

    Returns
    -------
    corr : float
        原始样本相关系数
    lower : float
        置信区间下界
    upper : float
        置信区间上界
    """
    from scipy.stats import pearsonr

    # 计算原始相关系数
    corr, _ = pearsonr(x, y)

    n = len(x)
    bootstrap_corrs = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        # TODO: 实现配对数据的 Bootstrap 重采样
        # 提示：需要同时重采样 x 和 y 的对应元素（按索引重采样）
        indices = np.random.choice(n, size=n, replace=True)
        x_resample = x[indices]
        y_resample = y[indices]
        bootstrap_corrs[i], _ = pearsonr(x_resample, y_resample)

    # 计算置信区间
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_corrs, 100 * alpha / 2)
    upper = np.percentile(bootstrap_corrs, 100 * (1 - alpha / 2))

    return corr, lower, upper


def plot_bootstrap_distribution(
    bootstrap_stats: np.ndarray,
    observed_stat: float,
    title: str = "Bootstrap 抽样分布",
    save_path: str = "bootstrap_dist.png",
) -> None:
    """
    绘制 Bootstrap 绽样分布图

    Parameters
    ----------
    bootstrap_stats : np.ndarray
        Bootstrap 重抽样得到的统计量
    observed_stat : float
        观测到的统计量（原始样本计算）
    title : str
        图表标题
    save_path : str
        保存路径
    """
    plt.figure(figsize=(10, 6))

    # 绘制直方图
    plt.hist(
        bootstrap_stats,
        bins=50,
        density=True,
        alpha=0.7,
        edgecolor="black",
    )

    # 标记观测值
    plt.axvline(observed_stat, color="red", linestyle="--", linewidth=2, label="观测值")

    # 计算并标记 95% 置信区间
    lower = np.percentile(bootstrap_stats, 2.5)
    upper = np.percentile(bootstrap_stats, 97.5)
    plt.axvline(lower, color="orange", linestyle="-", linewidth=2, label="95% CI 下界")
    plt.axvline(upper, color="orange", linestyle="-", linewidth=2, label="95% CI 上界")

    plt.xlabel("统计量值")
    plt.ylabel("密度")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"{title}:")
    print(f"  观测值: {observed_stat:.4f}")
    print(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
    print(f"  标准误: {bootstrap_stats.std():.4f}")


def generate_uncertainty_report(
    data: pd.DataFrame,
    numeric_cols: list,
    output_path: str = "uncertainty_report.md",
) -> str:
    """
    生成 StatLab 不确定性量化报告

    Parameters
    ----------
    data : pd.DataFrame
        分析数据集
    numeric_cols : list
        需要计算不确定性的数值列名列表
    output_path : str
        报告输出路径

    Returns
    -------
    report : str
        生成的报告内容
    """
    lines = ["## 不确定性量化\n"]
    lines.append("本节使用 Bootstrap 方法量化关键统计量的不确定性。\n")

    for col in numeric_cols:
        if col not in data.columns:
            continue

        values = data[col].dropna().values
        mean = values.mean()
        lower, upper, _ = bootstrap_ci(values, np.mean)

        lines.append(f"### {col}\n")
        lines.append(f"- **点估计**: {mean:.2f}\n")
        lines.append(f"- **95% 置信区间**: [{lower:.2f}, {upper:.2f}]\n")
        lines.append(f"- **标准误**: {(upper - lower) / 3.92:.2f}\n")  # 近似 SE
        lines.append("\n")

    report = "".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"不确定性报告已保存到: {output_path}")
    return report


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Week 05: StatLab 不确定性量化模块")
    print("=" * 60)
    print()

    # 示例 1: Bootstrap 均值置信区间
    print("示例 1: 均值的 Bootstrap 置信区间")
    print("-" * 40)

    # 模拟数据（替换为你的真实数据）
    sample_data = np.random.lognormal(8, 0.6, 100)

    lower, upper, bootstrap_means = bootstrap_ci(sample_data, np.mean)
    plot_bootstrap_distribution(
        bootstrap_means,
        sample_data.mean(),
        title="样本均值的 Bootstrap 分布",
        save_path="mean_bootstrap_dist.png",
    )

    print()

    # 示例 2: Bootstrap 相关系数置信区间
    print("示例 2: 相关系数的 Bootstrap 置信区间")
    print("-" * 40)

    # 模拟配对数据
    x = np.random.normal(0, 1, 100)
    y = x * 0.7 + np.random.normal(0, 0.5, 100)

    corr, lower, upper = bootstrap_correlation_ci(x, y)
    print(f"相关系数: {corr:.4f}")
    print(f"95% CI: [{lower:.4f}, {upper:.4f}]")
    print()

    # 示例 3: 生成不确定性报告
    print("示例 3: 生成 StatLab 不确定性报告")
    print("-" * 40)

    df = pd.DataFrame(
        {
            "revenue": sample_data,
            "sessions": np.random.poisson(50, 100),
        }
    )

    report = generate_uncertainty_report(df, ["revenue", "sessions"])
    print()
    print(report)

    print("=" * 60)
    print("提示：检查并完成代码中标记为 # TODO 的部分")
    print("=" * 60)
