#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：卡方检验（独立性检验）

本例演示如何使用卡方检验判断两个分类变量是否相关。
内容：
1. 创建列联表
2. 执行卡方检验
3. 计算效应量（Cramér's V）
4. 可视化（观测频数热力图 + 标准化残差图）

运行方式：python3 chapters/week_07/examples/04_chisquare_test.py
预期输出：
  - 观测频数表与期望频数表
  - 卡方统计量、p 值、自由度
  - Cramér's V 效应量
  - chisquare_results.png（可视化图）

作者：Week 07 示例代码
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def create_contingency_table(seed: int = 42) -> pd.DataFrame:
    """
    创建列联表（城市 vs 用户等级）。

    参数：
        seed: 随机种子

    返回：
        pd.DataFrame: 列联表
    """
    np.random.seed(seed)

    cities = ['北京', '上海', '广州', '深圳', '杭州']
    user_levels = ['普通', '银卡', '金卡', '钻石']

    # 模拟：不同城市的用户等级分布略有不同
    contingency_data = np.array([
        [45, 30, 18, 7],   # 北京
        [38, 32, 22, 8],   # 上海
        [52, 28, 15, 5],   # 广州
        [35, 35, 20, 10],  # 深圳
        [40, 30, 20, 10]   # 杭州
    ])

    contingency_table = pd.DataFrame(
        contingency_data,
        index=cities,
        columns=user_levels
    )

    return contingency_table


def print_contingency_table(table: pd.DataFrame) -> None:
    """打印列联表（含边际和）"""
    print("\n=== 观测频数表 ===")
    print(table)

    # 添加边际和
    table_with_margins = table.copy()
    table_with_margins.loc['总计'] = table_with_margins.sum(axis=0)
    table_with_margins['总计'] = table_with_margins.sum(axis=1)

    print("\n=== 观测频数表（含边际和） ===")
    print(table_with_margins)


def perform_chisquare_test(observed: np.ndarray) -> tuple:
    """
    执行卡方检验。

    参数：
        observed: 观测频数表（numpy 数组）

    返回：
        tuple: (chi2, p_value, dof, expected)
    """
    # 执行卡方检验
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    print(f"\n=== 卡方独立性检验 ===")
    print(f"卡方统计量：{chi2:.4f}")
    print(f"自由度：{dof}")
    print(f"p 值：{p_value:.6f}")

    decision = '拒绝 H0（城市与用户等级相关）' if p_value < 0.05 else '无法拒绝 H0（城市与用户等级可能无关）'
    print(f"结论：{decision}")

    return chi2, p_value, dof, expected


def print_expected_table(expected: np.ndarray, cities: list, user_levels: list) -> None:
    """打印期望频数表"""
    expected_df = pd.DataFrame(
        expected,
        index=cities,
        columns=user_levels
    )
    print("\n=== 期望频数表（H0 为真时的预期） ===")
    print(expected_df.round(1))


def calculate_cramers_v(chi2: float, n: int, min_dim: int) -> tuple:
    """
    计算 Cramér's V 效应量（卡方检验的效应量）。

    参数：
        chi2: 卡方统计量
        n: 总样本量
        min_dim: 列联表最小维度（min(行数, 列数)）

    返回：
        tuple: (v, interpretation)
    """
    phi2 = chi2 / n
    v = np.sqrt(phi2 / (min_dim - 1))

    # 解释
    if v < 0.1:
        interpretation = "关联很弱"
    elif v < 0.3:
        interpretation = "关联较弱"
    elif v < 0.5:
        interpretation = "关联中等"
    else:
        interpretation = "关联较强"

    return v, interpretation


def plot_chisquare_results(
    observed: np.ndarray,
    expected: np.ndarray,
    chi2: float,
    p_value: float,
    v: float,
    cities: list,
    user_levels: list
) -> Path:
    """
    绘制卡方检验结果可视化。

    参数：
        observed: 观测频数
        expected: 期望频数
        chi2: 卡方统计量
        p_value: p 值
        v: Cramér's V
        cities: 城市列表
        user_levels: 用户等级列表

    返回：
        Path: 保存的图片路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：观测频数热力图
    sns.heatmap(
        observed,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0],
        cbar_kws={'label': '频数'}
    )
    axes[0].set_xlabel('用户等级', fontsize=12)
    axes[0].set_ylabel('城市', fontsize=12)
    axes[0].set_title(
        f'观测频数热力图\nχ²={chi2:.2f}, p={p_value:.4f}, V={v:.3f}',
        fontsize=12
    )

    # 右图：标准化残差热力图
    # 标准化残差 = (观测 - 期望) / sqrt(期望)
    std_residuals = (observed - expected) / np.sqrt(expected)
    sns.heatmap(
        std_residuals,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-3,
        vmax=3,
        ax=axes[1],
        cbar_kws={'label': '标准化残差'}
    )
    axes[1].set_xlabel('用户等级', fontsize=12)
    axes[1].set_ylabel('城市', fontsize=12)
    axes[1].set_title(
        '标准化残差热力图\n(红色=观测>期望, 蓝色=观测<期望)',
        fontsize=12
    )

    plt.tight_layout()

    # 保存图片
    output_path = Path('chisquare_results.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

    return output_path


def main() -> None:
    """主函数"""
    print("=" * 70)
    print("卡方检验（独立性检验）")
    print("=" * 70)

    # 1. 创建列联表
    print("\n[1/5] 创建列联表...")
    contingency_table = create_contingency_table()
    print_contingency_table(contingency_table)

    # 2. 执行卡方检验
    print("\n[2/5] 执行卡方检验...")
    observed = contingency_table.values
    chi2, p_value, dof, expected = perform_chisquare_test(observed)

    # 3. 打印期望频数表
    print("\n[3/5] 期望频数...")
    cities = contingency_table.index.tolist()
    user_levels = contingency_table.columns.tolist()
    print_expected_table(expected, cities, user_levels)

    # 4. 计算效应量
    print("\n[4/5] 计算效应量...")
    n_total = observed.sum()
    min_dim = min(observed.shape[0], observed.shape[1])
    v, interpretation = calculate_cramers_v(chi2, n_total, min_dim)

    print(f"\nCramér's V 效应量：{v:.3f}")
    print(f"解释：{interpretation}")

    # 5. 可视化
    print("\n[5/5] 生成可视化...")
    output_path = plot_chisquare_results(
        observed, expected, chi2, p_value, v, cities, user_levels
    )
    print(f"图片已保存：{output_path}")

    # 6. 解读
    print("\n" + "=" * 70)
    print("结果解读")
    print("=" * 70)

    if p_value < 0.05:
        print(f"\n卡方检验显示城市与用户等级存在显著相关（χ²={chi2:.2f}, p={p_value:.4f}）")
        print(f"但 Cramér's V={v:.3f} 表明关联强度{interpretation}")
        print(f"\n注意：相关 ≠ 因果。观察性设计无法确定因果方向。")
    else:
        print(f"\n未发现城市与用户等级的显著相关（χ²={chi2:.2f}, p={p_value:.4f}）")
        print(f"Cramér's V={v:.3f} 表明关联强度{interpretation}")

    print("\n" + "=" * 70)
    print("分析完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
