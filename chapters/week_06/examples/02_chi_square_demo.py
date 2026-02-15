"""
示例：卡方检验——检验两个分类变量是否独立。

本例演示如何用卡方检验检验"用户类型（新/老）与渠道（A/B）是否独立"。
卡方检验适用于分类变量的关联性检验。

运行方式：python3 chapters/week_06/examples/02_chi_square_demo.py
预期输出：
  - stdout 输出列联表、期望频数、检验结果（卡方统计量、p 值、结论）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def main() -> None:
    """运行卡方检验演示"""
    print("=== 卡方检验演示：检验用户类型与渠道是否独立 ===\n")

    # 创建列联表
    # 假设你调查了 1000 个用户，记录他们的"类型"（新/老）和"渠道"（A/B）
    contingency_table = pd.DataFrame({
        "A渠道": [300, 200],  # 新用户 300，老用户 200
        "B渠道": [250, 250]   # 新用户 250，老用户 250
    }, index=["新用户", "老用户"])

    print("=== 观察频数（列联表）===")
    print(contingency_table)
    print(f"\n总计：{contingency_table.sum().sum()} 个用户\n")

    # 定义假设
    print("=== 假设设定 ===")
    print("H0（原假设）：用户类型与渠道独立（无关联）")
    print("H1（备择假设）：用户类型与渠道不独立（有关联）\n")

    # 卡方检验
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    print("=== 卡方检验结果 ===")
    print(f"卡方统计量：{chi2:.4f}")
    print(f"自由度：{dof}")
    print(f"p 值：{p_value:.4f}\n")

    print("=== 期望频数（如果独立的话）===")
    expected_df = pd.DataFrame(expected, index=contingency_table.index,
                               columns=contingency_table.columns)
    print(expected_df.round(1))
    print()

    # 判断
    alpha = 0.05
    if p_value < alpha:
        print(f"结论：p < {alpha:.2f}，拒绝原假设。")
        print(f"有证据表明用户类型与渠道不独立（有关联）。")
    else:
        print(f"结论：p ≥ {alpha:.2f}，无法拒绝原假设。")
        print(f"没有足够证据表明用户类型与渠道有关联。")

    # 解释
    print(f"\n=== 解释 ===")
    if p_value < alpha:
        # 计算残差（观察值 - 期望值）
        residuals = (contingency_table.values - expected) / np.sqrt(expected)
        residuals_df = pd.DataFrame(residuals, index=contingency_table.index,
                                    columns=contingency_table.columns)
        print("标准化残差（绝对值 > 2 表示该单元格对卡方贡献较大）：")
        print(residuals_df.round(2))

        # 找出贡献最大的单元格
        max_residual_idx = np.unravel_index(np.abs(residuals).argmax(), residuals.shape)
        max_cell = (contingency_table.index[max_residual_idx[0]],
                   contingency_table.columns[max_residual_idx[1]])
        print(f"\n最大偏差：{max_cell[0]} × {max_cell[1]}")
        print(f"观察值：{contingency_table.loc[max_cell]}，期望值：{expected[max_residual_idx]:.1f}")


if __name__ == "__main__":
    main()
