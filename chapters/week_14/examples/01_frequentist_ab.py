"""
示例：频率学派 A/B 测试——scipy.stats 实现

运行方式：python3 chapters/week_14/examples/01_frequentist_ab.py
预期输出：转化率、提升幅度、z 统计量和 p 值
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def frequentist_ab_test(conversions_A: int, exposures_A: int,
                       conversions_B: int, exposures_B: int) -> dict:
    """
    频率学派 A/B 测试（双样本比例检验）

    使用 scipy.stats.proportions_ztest 进行 z 检验

    参数:
        conversions_A: A 版本转化数
        exposures_A: A 版本曝光数
        conversions_B: B 版本转化数
        exposures_B: B 版本曝光数

    返回:
        包含转化率、提升幅度、统计量和 p 值的字典
    """
    # 计算转化率
    p_A = conversions_A / exposures_A
    p_B = conversions_B / exposures_B

    # 双样本比例检验（z检验）
    count = np.array([conversions_B, conversions_A])
    nobs = np.array([exposures_B, exposures_A])

    z_stat, p_value = proportions_ztest(count, nobs)

    # 计算相对提升
    relative_lift = (p_B - p_A) / p_A * 100 if p_A > 0 else 0

    return {
        'conversion_rate_A': p_A,
        'conversion_rate_B': p_B,
        'relative_lift_percent': relative_lift,
        'z_statistic': z_stat,
        'p_value': p_value
    }


# 坏例子：只看 p 值的二元决策
def bad_frequentist_decision(p_value: float, alpha: float = 0.05) -> str:
    """
    反例：只用 p 值做二元决策

    问题：
    1. 损失了效应量信息
    2. 无法回答"B 有多大概率更好"
    3. 决策者需要的是概率陈述，不是"显著/不显著"
    """
    if p_value < alpha:
        return "拒绝原假设，B 显著更好"
    else:
        return "无法拒绝原假设，B 和 A 无显著差异"


def main() -> None:
    # 示例数据：A 版本 52/1000，B 版本 58/1000
    conversions_A = 52
    exposures_A = 1000
    conversions_B = 58
    exposures_B = 1000

    # 运行频率学派 A/B 测试
    result = frequentist_ab_test(conversions_A, exposures_A,
                               conversions_B, exposures_B)

    print("=" * 50)
    print("频率学派 A/B 测试结果")
    print("=" * 50)
    print(f"转化率 A: {result['conversion_rate_A']:.3f}")
    print(f"转化率 B: {result['conversion_rate_B']:.3f}")
    print(f"提升幅度: {result['relative_lift_percent']:.2f}%")
    print(f"z 统计量: {result['z_statistic']:.3f}")
    print(f"p 值: {result['p_value']:.3f}")
    print()

    # 频率学派的二元决策
    decision = bad_frequentist_decision(result['p_value'])
    print(f"频率学派决策: {decision}")
    print()

    # 解释：p 值无法回答决策者真正的问题
    print("=" * 50)
    print("频率学派的局限性")
    print("=" * 50)
    print("产品经理真正想知道的：")
    print("  - B 比 A 好的概率是多少？")
    print("  - 提升幅度的可能范围是多少？")
    print()
    print("频率学派只能回答：")
    print(f"  - p={result['p_value']:.3f} {'<' if result['p_value'] < 0.05 else '>='} 0.05")
    print("  - 如果真的没有差异，有 {0:.1%} 的概率看到这么大的差异".format(
        result['p_value']))
    print()
    print("注意：p 值不是 'B 比 A 好的概率'！")
    print("这正是贝叶斯方法要解决的问题。")


if __name__ == "__main__":
    main()
