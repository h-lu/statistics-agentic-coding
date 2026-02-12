"""
示例：回归系数的正确解释——简单回归 vs 多元回归

本例演示简单回归和多元回归的系数差异，以及如何正确解释
"在其他变量不变的情况下"。

运行方式：python3 chapters/week_09/examples/03_coefficient_interpretation.py
预期输出：
- 简单回归（面积 -> 房价）的系数
- 多元回归（面积 + 房龄 + 房间数 -> 房价）的系数
- 展示面积系数在简单回归和多元回归中的变化（遗漏变量偏差）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(42)


def generate_multi_feature_data(n_samples: int = 100) -> pd.DataFrame:
    """
    生成多特征房价数据（特征之间有相关）

    参数:
        n_samples: 样本数量

    返回:
        包含多个特征的 DataFrame
    """
    # 房间数：1-5个
    n_rooms = np.random.randint(1, 6, n_samples)

    # 面积：与房间数相关（大房子房间多）+ 随机性
    area_sqm = 30 + 20 * n_rooms + np.random.normal(0, 10, n_samples)
    area_sqm = np.maximum(area_sqm, 30)  # 最小30平米

    # 房龄：0-30年
    age_years = np.random.randint(0, 31, n_samples)

    # 房价：真实关系 + 噪音
    # price = 15 + 0.8*area - 0.5*age + 5*n_rooms + noise
    noise = np.random.normal(0, 12, n_samples)
    price_wan = (15 + 0.8 * area_sqm - 0.5 * age_years +
                 5 * n_rooms + noise)

    return pd.DataFrame({
        'area_sqm': area_sqm,
        'age_years': age_years,
        'n_rooms': n_rooms,
        'price_wan': price_wan
    })


def simple_regression(y: pd.Series, x: pd.Series, x_name: str) -> sm.regression.linear_model.RegressionResults:
    """
    简单回归（一个自变量）

    参数:
        y: 因变量
        x: 自变量
        x_name: 自变量名称（用于打印）

    返回:
        拟合的模型
    """
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model


def multiple_regression(y: pd.Series, X: pd.DataFrame) -> sm.regression.linear_model.RegressionResults:
    """
    多元回归

    参数:
        y: 因变量
        X: 自变量 DataFrame

    返回:
        拟合的模型
    """
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()
    return model


def print_coefficient_table(model: sm.regression.linear_model.RegressionResults,
                          model_name: str) -> None:
    """
    打印系数表（含置信区间）

    参数:
        model: 拟合的模型
        model_name: 模型名称
    """
    print(f"\n{'=' * 60}")
    print(f"{model_name}")
    print('=' * 60)

    # 获取系数和置信区间
    conf_int = model.conf_int(alpha=0.05)
    conf_int.columns = ['95% CI 低', '95% CI 高']

    # 合并结果
    results = pd.DataFrame({
        '系数': model.params,
        '标准误': model.bse,
        't值': model.tvalues,
        'p值': model.pvalues,
    })
    results = pd.concat([results, conf_int], axis=1)

    print(results.round(3))
    print(f"\nR² = {model.rsquared:.3f}")
    print(f"调整 R² = {model.rsquared_adj:.3f}")


def main() -> None:
    """主函数：演示系数解释"""
    print("=" * 60)
    print("示例3: 回归系数的正确解释")
    print("=" * 60)

    # 1. 生成数据
    df = generate_multi_feature_data(n_samples=100)
    print(f"\n📊 数据概览 (前5行):")
    print(df.head())

    # 2. 查看特征间相关性
    print(f"\n🔗 特征相关性矩阵:")
    print(df[['area_sqm', 'age_years', 'n_rooms']].corr().round(3))
    print("\n注意: 面积和房间数高度相关 (r = {:.3f})".format(
        df['area_sqm'].corr(df['n_rooms'])
    ))

    # 3. 简单回归：面积 -> 房价
    model_simple = simple_regression(df['price_wan'], df['area_sqm'], 'area_sqm')
    print_coefficient_table(model_simple, "简单回归: price ~ area")

    area_coef_simple = model_simple.params['area_sqm']
    area_ci_simple = model_simple.conf_int().loc['area_sqm']

    print(f"\n📖 解释 (简单回归):")
    print(f"  面积每增加 1 平米,房价平均上涨 {area_coef_simple:.2f} 万元")
    print(f"  95% CI: [{area_ci_simple[0]:.2f}, {area_ci_simple[1]:.2f}]")
    print(f"  ⚠️  但这个系数可能'抢了'房间数的功劳!")

    # 4. 多元回归：面积 + 房龄 + 房间数 -> 房价
    X_multi = df[['area_sqm', 'age_years', 'n_rooms']]
    model_multi = multiple_regression(df['price_wan'], X_multi)
    print_coefficient_table(model_multi, "多元回归: price ~ area + age + rooms")

    area_coef_multi = model_multi.params['area_sqm']
    area_ci_multi = model_multi.conf_int().loc['area_sqm']

    print(f"\n📖 解释 (多元回归):")
    print(f"  在房龄和房间数不变的情况下,")
    print(f"  面积每增加 1 平米,房价平均上涨 {area_coef_multi:.2f} 万元")
    print(f"  95% CI: [{area_ci_multi[0]:.2f}, {area_ci_multi[1]:.2f}]")

    # 5. 对比简单回归和多元回归
    print(f"\n🔄 对比: 面积系数的变化")
    print(f"  简单回归: {area_coef_simple:.3f}")
    print(f"  多元回归: {area_coef_multi:.3f}")
    print(f"  变化: {area_coef_multi - area_coef_simple:+.3f}")
    print(f"\n  解释: 简单回归中,面积系数'混杂'了房间数的影响")
    print(f"        多元回归中,各个变量'公平分配'了贡献")

    # 6. 完整解释示例
    print(f"\n📚 完整解释示例:")
    print(f"  根据多元回归模型:")
    print(f"  - 截距: {model_multi.params['const']:.2f} 万元")
    print(f"  - 面积: {area_coef_multi:.2f} 万元/平米 (在房龄和房间数不变时)")
    print(f"  - 房龄: {model_multi.params['age_years']:.2f} 万元/年 (在面积和房间数不变时)")
    print(f"  - 房间数: {model_multi.params['n_rooms']:.2f} 万元/个 (在面积和房龄不变时)")

    # 7. 预测示例
    print(f"\n🔮 预测示例:")
    new_house = pd.DataFrame({
        'const': [1],
        'area_sqm': [80],
        'age_years': [5],
        'n_rooms': [2]
    })
    pred_price = model_multi.predict(new_house[['const', 'area_sqm', 'age_years', 'n_rooms']])[0]
    print(f"  一套 80平米、5年房龄、2房的房子:")
    print(f"  预测价格: {pred_price:.2f} 万元")

    print("\n" + "=" * 60)
    print("✅ 示例3完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
