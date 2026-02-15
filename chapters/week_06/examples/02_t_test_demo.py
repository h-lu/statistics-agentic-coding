"""
示例：执行 t 检验和比例检验——从概念到代码。

本例演示如何用 Python 执行双样本 t 检验，包括：
1. 设置原假设和备择假设
2. 使用 scipy.stats.ttest_ind 进行 t 检验
3. 使用 statsmodels 的 proportions_ztest 进行比例检验（更适合转化率数据）

运行方式：python3 chapters/week_06/examples/02_t_test_demo.py
预期输出：
  - stdout 输出 A/B 渠道的转化率和差异
  - t 检验和比例检验的结果（统计量、p 值、结论）
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def main() -> None:
    """运行 t 检验和比例检验演示"""
    # 固定随机种子确保可复现
    np.random.seed(42)

    print("=== A/B 测试假设检验演示 ===\n")

    # 加载数据（模拟 A/B 测试的转化数据）
    # A 渠道：1200 个用户，144 个转化（12%）
    # B 渠道：1200 个用户，108 个转化（9%）
    conversions_a = np.array([1] * 144 + [0] * (1200 - 144))
    conversions_b = np.array([1] * 108 + [0] * (1200 - 108))

    # 观察
    rate_a = conversions_a.mean()
    rate_b = conversions_b.mean()
    observed_diff = rate_a - rate_b

    print("=== 数据概览 ===")
    print(f"A 渠道：{len(conversions_a)} 个用户，{conversions_a.sum()} 个转化（{rate_a:.2%}）")
    print(f"B 渠道：{len(conversions_b)} 个用户，{conversions_b.sum()} 个转化（{rate_b:.2%}）")
    print(f"观察到的差异：{observed_diff:.2%}\n")

    # 定义假设
    print("=== 假设设定 ===")
    print("H0（原假设）：A 渠道转化率 = B 渠道转化率（无差异）")
    print("H1（备择假设）：A 渠道转化率 ≠ B 渠道转化率（有差异）\n")

    # 方法 1：用 scipy.stats.ttest_ind（双样本 t 检验）
    print("=== 方法 1：双样本 t 检验 ===")
    t_stat, p_value = stats.ttest_ind(conversions_a, conversions_b)

    print(f"t 统计量：{t_stat:.4f}")
    print(f"p 值：{p_value:.4f}")

    # 判断
    alpha = 0.05
    if p_value < alpha:
        print(f"\n结论：p < {alpha}，拒绝原假设。有证据表明两组转化率存在显著差异。")
    else:
        print(f"\n结论：p ≥ {alpha}，无法拒绝原假设。")

    # 方法 2：用 proportions_ztest（比例检验，更适合转化率数据）
    print(f"\n=== 方法 2：比例检验（更适合转化率数据） ===")
    count = np.array([conversions_a.sum(), conversions_b.sum()])
    nobs = np.array([len(conversions_a), len(conversions_b)])

    z_stat, p_value_z = proportions_ztest(count, nobs)

    print(f"z 统计量：{z_stat:.4f}")
    print(f"p 值：{p_value_z:.4f}")

    if p_value_z < alpha:
        print(f"\n结论：p < {alpha}，拒绝原假设。有证据表明两组转化率存在显著差异。")
    else:
        print(f"\n结论：p ≥ {alpha}，无法拒绝原假设。")

    # 说明单尾 vs 双尾
    print(f"\n=== 补充说明 ===")
    print(f"当前使用的是双尾检验（H1: A ≠ B）")
    print(f"如果有明确的方向性假设（如 H1: A > B），可以使用单尾检验（p 值除以 2）")
    print(f"但必须提前声明，不能先看数据再决定用单尾还是双尾（那是 p-hacking）")


if __name__ == "__main__":
    main()
