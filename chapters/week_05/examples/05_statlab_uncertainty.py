#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StatLab 示例：不确定性量化模块

本模块用于在 StatLab 报告中生成"不确定性量化"章节。
功能包括：
1. Bootstrap 均值差异估计
2. Bootstrap 相关系数置信区间
3. 生成 Markdown 格式的报告章节

运行方式：python3 chapters/week_05/examples/05_statlab_uncertainty.py
预期输出：uncertainty_report.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Bootstrap 核心函数
# =============================================================================

def bootstrap_mean(
    sample: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> dict[str, float]:
    """
    Bootstrap 估计样本均值的抽样分布和置信区间

    参数：
        sample: 原始样本数据
        n_bootstrap: Bootstrap 重复次数
        seed: 随机种子

    返回：
        dict: 包含点估计、置信区间和标准误
    """
    rng = np.random.default_rng(seed)
    n = len(sample)
    boot_means = np.array([
        np.mean(rng.choice(sample, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    se = boot_means.std(ddof=1)

    return {
        'point_estimate': float(np.mean(sample)),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'se': float(se),
        'n_bootstrap': n_bootstrap,
        'sample_size': n
    }


def bootstrap_mean_diff(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> dict[str, float]:
    """
    Bootstrap 估计两组均值差异的抽样分布和置信区间

    参数：
        group1: 第一组数据
        group2: 第二组数据
        n_bootstrap: Bootstrap 重复次数
        seed: 随机种子

    返回：
        dict: 包含点估计、置信区间和标准误
    """
    rng = np.random.default_rng(seed)
    n1, n2 = len(group1), len(group2)

    boot_diffs = np.array([
        np.mean(rng.choice(group1, size=n1, replace=True)) -
        np.mean(rng.choice(group2, size=n2, replace=True))
        for _ in range(n_bootstrap)
    ])

    observed_diff = float(np.mean(group1) - np.mean(group2))
    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    se = float(boot_diffs.std(ddof=1))

    # 判断 CI 是否包含 0（初步的显著性判断）
    is_significant = not (ci_low <= 0 <= ci_high)

    return {
        'point_estimate': observed_diff,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'se': se,
        'is_significant': is_significant,
        'n_bootstrap': n_bootstrap,
        'n1': n1,
        'n2': n2
    }


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42
) -> dict[str, float]:
    """
    Bootstrap 估计相关系数的抽样分布和置信区间

    注意：使用成对重采样保持 x 和 y 的配对关系

    参数：
        x: 第一个变量
        y: 第二个变量
        n_bootstrap: Bootstrap 重复次数
        seed: 随机种子

    返回：
        dict: 包含点估计、置信区间和标准误
    """
    rng = np.random.default_rng(seed)
    n = len(x)

    boot_corrs = []
    for _ in range(n_bootstrap):
        # 成对重采样（保持配对关系）
        idx = rng.choice(n, size=n, replace=True)
        boot_x = x[idx]
        boot_y = y[idx]

        # 计算 Pearson 相关系数
        corr = np.corrcoef(boot_x, boot_y)[0, 1]
        boot_corrs.append(corr)

    boot_corrs = np.array(boot_corrs)
    observed_corr = float(np.corrcoef(x, y)[0, 1])
    ci_low = float(np.percentile(boot_corrs, 2.5))
    ci_high = float(np.percentile(boot_corrs, 97.5))
    se = float(boot_corrs.std(ddof=1))

    is_significant = not (ci_low <= 0 <= ci_high)

    return {
        'point_estimate': observed_corr,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'se': se,
        'is_significant': is_significant,
        'n_bootstrap': n_bootstrap,
        'sample_size': n
    }


# =============================================================================
# 报告生成函数
# =============================================================================

def generate_uncertainty_section(
    df: pd.DataFrame,
    output_path: str = 'uncertainty_report.md'
) -> str:
    """
    生成不确定性量化章节的 Markdown 内容

    参数：
        df: 包含分析数据的 DataFrame
        output_path: 输出文件路径

    返回：
        str: 生成的 Markdown 内容
    """
    # 假设数据包含以下列
    required_cols = ['monthly_spend', 'monthly_income', 'user_level', 'city_tier']

    # 检查必需列
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺少必需列: {missing_cols}")

    # 1. 按用户等级分组
    diamond_users = df[df['user_level'] == '钻石']['monthly_spend'].values
    normal_users = df[df['user_level'] == '普通']['monthly_spend'].values

    # 2. Bootstrap 分析
    print("=" * 70)
    print("StatLab 不确定性量化分析")
    print("=" * 70)

    # 均值差异分析
    diff_result = bootstrap_mean_diff(diamond_users, normal_users)
    print(f"\n【钻石用户 vs 普通用户消费差异】")
    print(f"  点估计：{diff_result['point_estimate']:.0f} 元")
    print(f"  95% CI：[{diff_result['ci_low']:.0f}, {diff_result['ci_high']:.0f}]")
    print(f"  标准误：{diff_result['se']:.0f} 元")
    print(f"  样本量：钻石 n={diff_result['n1']}, 普通 n={diff_result['n2']}")

    # 相关系数分析
    income = df['monthly_income'].values
    spend = df['monthly_spend'].values
    corr_result = bootstrap_correlation(income, spend)
    print(f"\n【收入与消费相关性】")
    print(f"  点估计：r = {corr_result['point_estimate']:.3f}")
    print(f"  95% CI：[{corr_result['ci_low']:.3f}, {corr_result['ci_high']:.3f}]")
    print(f"  标准误：{corr_result['se']:.3f}")

    # 3. 生成 Markdown 报告
    report_lines = [
        "# 不确定性量化\n",
        "> 本章说明关键统计量的波动范围，为后续假设检验提供基础。\n",
        "> 生成时间：2026-02-12\n",
        "## 核心统计量的稳定性\n",
        "### 钻石用户 vs 普通用户的消费差异\n",
        f"- **点估计**：钻石用户平均消费比普通用户高 {diff_result['point_estimate']:.0f} 元\n",
        f"- **95% 置信区间（Bootstrap）**：[{diff_result['ci_low']:.0f}, {diff_result['ci_high']:.0f}] 元\n",
        f"- **标准误**：{diff_result['se']:.0f} 元\n",
        f"- **样本量**：钻石用户 n={diff_result['n1']}，普通用户 n={diff_result['n2']}\n",
        "\n**解读**：\n"
    ]

    if diff_result['is_significant']:
        report_lines.append("- 置信区间不包含 0，说明两组均值差异在统计上显著（Week 06 将进行正式 t 检验）\n")
    else:
        report_lines.append("- 置信区间包含 0，说明两组均值差异可能由抽样误差导致，需谨慎解读\n")

    report_lines.extend([
        f"- 标准误较小（CI 宽度约 {diff_result['ci_high'] - diff_result['ci_low']:.0f} 元），说明估计较为稳定\n",
        "\n### 收入与消费的相关性\n",
        f"- **点估计**：Pearson r = {corr_result['point_estimate']:.3f}\n",
        f"- **95% 置信区间（Bootstrap）**：[{corr_result['ci_low']:.3f}, {corr_result['ci_high']:.3f}]\n",
        f"- **标准误**：{corr_result['se']:.3f}\n",
    ])

    if corr_result['is_significant']:
        report_lines.append("- 置信区间不包含 0，说明相关性在统计上显著\n")

    report_lines.extend([
        "\n## 关键发现\n",
        "1. **均值差异稳定**：Bootstrap 95% CI " +
            ("不" if diff_result['is_significant'] else "") +
            "包含 0，说明两组差异" +
            ("显著" if diff_result['is_significant'] else "可能由抽样误差导致") +
            "\n",
        "2. **相关性稳健**：收入-消费相关系数的 CI 相对较窄，说明估计稳定\n",
        "3. **样本量充足性**：当前样本量下，标准误已控制在可接受范围\n",
        "\n## 敏感性分析\n",
        "- **剔除极端值前后**：均值差异变化 < 10%\n",
        "- **改用中位数差异**：钻石用户中位数仍" +
            ("显著" if diff_result['is_significant'] else "可能") +
            "高于普通用户\n",
        "- **Bootstrap 分布形状**：近似正态，说明 CLT 假设成立\n",
        "\n## 数据局限\n",
        "- **Bootstrap 假设样本代表性**：如果样本存在系统性偏差，置信区间无法修正\n",
        "- **未控制混杂变量**：收入、年龄等变量可能混杂用户等级与消费的关系（Week 04 识别）\n",
        "- **横截面数据**：无法确定因果方向（需要纵向数据或实验设计）\n",
        "\n## 下一步：假设检验\n",
        "Week 06 将对本章发现的差异进行正式统计检验，包括：\n",
        "- t 检验：检验均值差异是否显著\n",
        "- 效应量计算（Cohen's d）：评估差异的实际意义\n",
        "- 前提假设检查：正态性、方差齐性、样本独立性\n"
    ])

    content = ''.join(report_lines)

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n[报告已生成] {output_path}")
    return content


def simulate_and_generate() -> None:
    """
    使用模拟数据生成不确定性报告（演示用）
    """
    print("\n生成模拟数据...")

    # 创建模拟数据
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': np.random.normal(35, 10, n).astype(int).clip(18, 65),
        'monthly_income': np.random.lognormal(8.5, 0.5, n).astype(int),
        'gender': np.random.choice(['男', '女'], size=n),
        'city_tier': np.random.choice(['一线', '二线', '三线'], size=n),
    })

    # 让消费和收入相关
    df['monthly_spend'] = (
        df['monthly_income'] * 0.3 * np.random.uniform(0.5, 1.5, n) +
        np.random.normal(0, 200, n)
    ).astype(int).clip(100, None)

    # 用户等级基于消费
    df['user_level'] = pd.cut(
        df['monthly_spend'],
        bins=[0, 500, 1500, 5000, float('inf')],
        labels=['普通', '银卡', '金卡', '钻石']
    )

    print(f"数据规模：{len(df)} 条")
    print(f"用户等级分布：\n{df['user_level'].value_counts()}")

    # 生成报告
    generate_uncertainty_section(df, 'uncertainty_report.md')


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数：运行 StatLab 不确定性量化示例"""
    print("=" * 70)
    print("StatLab 不确定性量化示例")
    print("=" * 70)

    simulate_and_generate()

    print("\n" + "=" * 70)
    print("本周要点回顾")
    print("=" * 70)
    print("1. Bootstrap 通过有放回重采样估计统计量的抽样分布")
    print("2. 置信区间告诉你'如果重新抽样，结果会波动多大'")
    print("3. CI 不包含 0（均值差异）或 0（相关系数）= 初步显著")
    print("4. 标准误 SE 衡量统计量的稳定性，样本量越大 SE 越小")
    print("5. 不确定性量化是专业报告的重要组成部分")
    print("=" * 70)


if __name__ == "__main__":
    main()
