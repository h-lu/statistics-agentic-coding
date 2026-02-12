"""
示例：从散点图到简单回归线——拟合第一条回归线

本例演示如何从房价数据的散点图出发，拟合第一条简单线性回归线，
并解释截距和斜率的含义。

运行方式：python3 chapters/week_09/examples/01_simple_regression.py
预期输出：
- 散点图 + 回归线（保存为 regression_line.png）
- 控制台输出截距、斜率、R² 等统计量
- 残差表的前几行
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 设置随机种子以保证可复现
np.random.seed(42)


def generate_house_price_data(n_samples: int = 100) -> pd.DataFrame:
    """
    生成模拟房价数据

    参数:
        n_samples: 样本数量

    返回:
        包含面积和房价的 DataFrame
    """
    # 生成面积：40-150平米之间的均匀分布
    area_sqm = np.random.uniform(40, 150, n_samples)

    # 生成房价：真实关系 + 随机噪音
    # 真实关系: price = 20 + 1.2 * area + noise
    noise = np.random.normal(0, 15, n_samples)  # 标准差15万的噪音
    price_wan = 20 + 1.2 * area_sqm + noise

    return pd.DataFrame({
        'area_sqm': area_sqm,
        'price_wan': price_wan
    })


def plot_scatter_with_regression(df: pd.DataFrame, model: LinearRegression) -> None:
    """
    画散点图和回归线

    参数:
        df: 包含 area_sqm 和 price_wan 的数据
        model: 已拟合的线性回归模型
    """
    plt.figure(figsize=(10, 6))

    # 画散点图
    sns.scatterplot(data=df, x='area_sqm', y='price_wan',
                   alpha=0.6, label='观测值')

    # 画回归线
    X = df[['area_sqm']]
    y_pred = model.predict(X)
    plt.plot(df['area_sqm'], y_pred, color='red',
             linewidth=2, label='回归线')

    plt.xlabel('面积 (平米)', fontsize=12)
    plt.ylabel('售价 (万元)', fontsize=12)
    plt.title(f'房价 vs 面积 (y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x)',
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('regression_line.png', dpi=150, bbox_inches='tight')
    print("✅ 散点图 + 回归线已保存为 regression_line.png")
    plt.close()


def main() -> None:
    """主函数：拟合简单回归并输出结果"""
    print("=" * 60)
    print("示例1: 从散点图到简单回归线")
    print("=" * 60)

    # 1. 生成数据
    df = generate_house_price_data(n_samples=100)
    print(f"\n📊 数据概览 (前5行):")
    print(df.head())

    # 2. 拟合回归模型
    X = df[['area_sqm']]  # sklearn 需要 2D 数组
    y = df['price_wan']

    model = LinearRegression()
    model.fit(X, y)

    # 3. 输出系数
    print(f"\n📈 回归系数:")
    print(f"  截距(β₀): {model.intercept_:.2f} 万元")
    print(f"  斜率(β₁): {model.coef_[0]:.2f} 万元/平米")
    print(f"\n解释: 面积每增加 1 平米,房价平均上涨 {model.coef_[0]:.2f} 万元")

    # 4. 输出拟合优度
    r_squared = model.score(X, y)
    print(f"\n📊 拟合优度:")
    print(f"  R² = {r_squared:.3f}")
    print(f"  解释: 模型解释了 {r_squared * 100:.1f}% 的房价变异")

    # 5. 计算残差
    df['predicted'] = model.predict(X)
    df['residual'] = df['price_wan'] - df['predicted']

    print(f"\n🔍 残差表 (前5行):")
    print(df[['area_sqm', 'price_wan', 'predicted', 'residual']].head())

    # 6. 画图
    plot_scatter_with_regression(df, model)

    # 7. 演示预测
    print(f"\n🔮 预测示例:")
    areas_to_predict = [60, 80, 100, 120]
    for area in areas_to_predict:
        pred_price = model.predict([[area]])[0]
        print(f"  {area:3d} 平米 -> 预测房价: {pred_price:.2f} 万元")

    print("\n" + "=" * 60)
    print("✅ 示例1完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
