"""
Week 09 作业参考实现

本文件是 Week 09 "回归与模型诊断" 作业的参考实现。
当你在作业中遇到困难时，可以查看此文件，但建议先自己尝试完成。

⚠️  注意：这只是参考实现，可能不是唯一的正确答案。
     更重要的是理解背后的统计概念，而不是照抄代码。

作业要求概览:
1. 拟合简单回归和多元回归
2. 正确解释回归系数（含"在其他变量不变的情况下"）
3. 检查残差诊断（线性、正态性、等方差、独立性）
4. 计算多重共线性 (VIF)
5. 识别异常点 (Cook's 距离)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_09"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子
np.random.seed(42)


def generate_data():
    """
    作业数据生成函数

    生成房价数据，包含以下变量:
    - area_sqm: 面积（平米）
    - age_years: 房龄（年）
    - n_rooms: 房间数
    - price_wan: 房价（万元）
    """
    n = 100

    # 生成特征
    area = np.random.uniform(40, 150, n)
    age = np.random.randint(0, 31, n)
    n_rooms = np.random.randint(1, 6, n)

    # 生成目标变量（线性关系 + 噪音）
    price = (15 + 0.9 * area - 0.4 * age + 4 * n_rooms +
             np.random.normal(0, 12, n))

    df = pd.DataFrame({
        'area_sqm': area,
        'age_years': age,
        'n_rooms': n_rooms,
        'price_wan': price
    })

    return df


def part_1_simple_regression(df):
    """
    第 1 部分: 简单回归

    要求:
    1. 用面积预测房价
    2. 打印截距和斜率
    3. 解释斜率的含义
    """
    print("=" * 60)
    print("第 1 部分: 简单回归")
    print("=" * 60)

    # 1. 拟合简单回归
    X = sm.add_constant(df['area_sqm'])
    y = df['price_wan']
    model = sm.OLS(y, X).fit()

    # 2. 打印系数
    print(f"\n截距: {model.params['const']:.2f}")
    print(f"斜率: {model.params['area_sqm']:.2f}")
    print(f"R²: {model.rsquared:.3f}")

    # 3. 解释斜率
    print(f"\n解释: 面积每增加 1 平米,房价平均上涨 "
          f"{model.params['area_sqm']:.2f} 万元")

    # 4. 画图
    plt.figure(figsize=(10, 6))
    plt.scatter(df['area_sqm'], df['price_wan'], alpha=0.6)
    plt.plot(df['area_sqm'], model.fittedvalues,
             color='red', linewidth=2)
    plt.xlabel('面积 (平米)')
    plt.ylabel('房价 (万元)')
    plt.title('房价 vs 面积 - 简单回归')
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / 'simple_regression.png', dpi=150, bbox_inches='tight')
    print("\n✅ 图表已保存: simple_regression.png")
    plt.close()

    return model


def part_2_multiple_regression(df):
    """
    第 2 部分: 多元回归

    要求:
    1. 用面积、房龄、房间数预测房价
    2. 打印系数表
    3. 解释每个系数的含义（加"在其他变量不变的情况下"）
    """
    print("\n" + "=" * 60)
    print("第 2 部分: 多元回归")
    print("=" * 60)

    # 1. 拟合多元回归
    X = sm.add_constant(df[['area_sqm', 'age_years', 'n_rooms']])
    y = df['price_wan']
    model = sm.OLS(y, X).fit()

    # 2. 打印系数表
    print(f"\n{'系数':<15} {'估计值':>10} {'p值':>10}")
    print("-" * 40)
    for var in ['const', 'area_sqm', 'age_years', 'n_rooms']:
        print(f"{var:<15} {model.params[var]:>10.3f} {model.pvalues[var]:>10.4f}")

    # 3. 解释系数
    print(f"\n解释（在其他变量不变的情况下）:")
    print(f"  - 面积: 每增加 1 平米,房价涨 {model.params['area_sqm']:.2f} 万元")
    print(f"  - 房龄: 每增加 1 年,房价跌 {abs(model.params['age_years']):.2f} 万元")
    print(f"  - 房间数: 每增加 1 间,房价涨 {model.params['n_rooms']:.2f} 万元")

    return model


def part_3_residual_diagnostics(model, df):
    """
    第 3 部分: 残差诊断

    要求:
    1. 画残差 vs 拟合值图（检验线性和等方差）
    2. 画 QQ 图（检验正态性）
    3. 运行 Shapiro-Wilk 检验
    4. 计算 Durbin-Watson 统计量
    """
    print("\n" + "=" * 60)
    print("第 3 部分: 残差诊断")
    print("=" * 60)

    # 获取残差和拟合值
    residuals = model.resid
    fitted = model.fittedvalues

    # 1. 残差 vs 拟合值图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(fitted, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('拟合值')
    axes[0].set_ylabel('残差')
    axes[0].set_title('残差 vs 拟合值')
    axes[0].grid(True, alpha=0.3)

    # 2. QQ 图
    from scipy.stats import probplot
    probplot(residuals, plot=axes[1])
    axes[1].set_title('QQ 图')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'residual_diagnostics.png', dpi=150, bbox_inches='tight')
    print("✅ 诊断图已保存: residual_diagnostics.png")
    plt.close()

    # 3. Shapiro-Wilk 检验
    stat, p_value = shapiro(residuals)
    print(f"\nShapiro-Wilk 检验:")
    print(f"  统计量: {stat:.4f}")
    print(f"  p 值: {p_value:.4f}")
    if p_value > 0.05:
        print(f"  结论: 不能拒绝正态性假设 ✓")
    else:
        print(f"  结论: 拒绝正态性假设 ✗")

    # 4. Durbin-Watson 统计量
    dw = sm.stats.durbin_watson(residuals)
    print(f"\nDurbin-Watson 统计量: {dw:.2f}")
    if 1.5 < dw < 2.5:
        print(f"  结论: 独立性假设满足 ✓")
    else:
        print(f"  结论: 可能存在自相关 ✗")


def part_4_multicollinearity(df):
    """
    第 4 部分: 多重共线性

    要求:
    1. 计算每个变量的 VIF
    2. 判断是否存在严重共线性 (VIF >= 10)
    """
    print("\n" + "=" * 60)
    print("第 4 部分: 多重共线性")
    print("=" * 60)

    X = df[['area_sqm', 'age_years', 'n_rooms']]

    # 计算 VIF
    vif_data = pd.DataFrame()
    vif_data["变量"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                        for i in range(X.shape[1])]

    print("\nVIF 表:")
    print(vif_data)

    # 判断
    high_vif = vif_data[vif_data['VIF'] >= 10]
    if len(high_vif) == 0:
        print("\n结论: 所有 VIF < 10，无严重共线性问题 ✓")
    else:
        print(f"\n结论: 以下变量存在严重共线性 (VIF >= 10):")
        print(high_vif['变量'].tolist())


def part_5_cooks_distance(model, df):
    """
    第 5 部分: 异常点分析

    要求:
    1. 计算 Cook's 距离
    2. 标注 Cook's D > 1 的点
    3. 判断模型是否稳健
    """
    print("\n" + "=" * 60)
    print("第 5 部分: 异常点分析")
    print("=" * 60)

    # 1. 计算 Cook's 距离
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    # 2. 画图
    plt.figure(figsize=(10, 5))
    colors = ['red' if d >= 1 else 'steelblue' for d in cooks_d]
    plt.bar(range(len(cooks_d)), cooks_d, color=colors, alpha=0.7)
    plt.axhline(y=1, color='red', linestyle='--', linewidth=2, label='阈值 (D=1)')
    plt.xlabel('观测索引')
    plt.ylabel("Cook's 距离")
    plt.title("Cook's 距离")
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(OUTPUT_DIR / 'cooks_distance.png', dpi=150, bbox_inches='tight')
    print("✅ Cook's 距离图已保存: cooks_distance.png")
    plt.close()

    # 3. 统计
    n_high = (cooks_d >= 1).sum()
    print(f"\nCook's D >= 1 的点: {n_high} 个")

    if n_high > 0:
        high_idx = np.where(cooks_d >= 1)[0]
        print(f"  索引: {high_idx.tolist()}")

    # 简单判断（未进行删除对比）
    if n_high <= 3:
        print("\n结论: 强影响点较少，模型可能稳健 ✓")
    else:
        print("\n结论: 强影响点较多，建议进一步分析 ✗")


def main():
    """主函数：运行所有部分"""
    print("\n" + "=" * 60)
    print("Week 09 作业 - 回归与模型诊断")
    print("=" * 60)

    # 生成数据
    df = generate_data()
    print(f"\n数据概览 (n={len(df)}):")
    print(df.head())

    # 第 1 部分: 简单回归
    model_simple = part_1_simple_regression(df)

    # 第 2 部分: 多元回归
    model_multi = part_2_multiple_regression(df)

    # 第 3 部分: 残差诊断
    part_3_residual_diagnostics(model_multi, df)

    # 第 4 部分: 多重共线性
    part_4_multicollinearity(df)

    # 第 5 部分: 异常点
    part_5_cooks_distance(model_multi, df)

    print("\n" + "=" * 60)
    print("✅ 所有部分完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - simple_regression.png")
    print("  - residual_diagnostics.png")
    print("  - cooks_distance.png")


if __name__ == "__main__":
    main()
