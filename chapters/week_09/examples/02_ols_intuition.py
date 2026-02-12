"""
示例：最小二乘法的几何直觉——手动计算 OLS 系数

本例演示最小二乘法(OLS)的数学原理，包括：
1. 手动计算损失函数（残差平方和）
2. 用矩阵公式计算 OLS 系数：β = (X'X)^(-1)X'y
3. 对比 sklearn 结果验证一致性

运行方式：python3 chapters/week_09/examples/02_ols_intuition.py
预期输出：
- 损失函数的值（残差平方和）
- 手动计算和 sklearn 计算的系数对比
- 可视化：残差的平方（显示大误差被放大）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(42)


def generate_simple_data() -> pd.DataFrame:
    """生成简单的线性关系数据"""
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = 2 * x + 3 + np.array([0.5, -0.3, 0.8, -0.6, 0.4])  # 添加小噪音
    return pd.DataFrame({'x': x, 'y': y})


def compute_loss(y_true: np.ndarray, y_pred: np.ndarray,
                 loss_type: str = 'mse') -> float:
    """
    计算损失函数

    参数:
        y_true: 真实值
        y_pred: 预测值
        loss_type: 'mse'(均方误差) 或 'mae'(平均绝对误差)

    返回:
        损失值
    """
    residuals = y_true - y_pred
    if loss_type == 'mse':
        return np.mean(residuals ** 2)
    elif loss_type == 'mae':
        return np.mean(np.abs(residuals))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def manual_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    手动计算 OLS 系数（矩阵公式）

    公式: β = (X'X)^(-1)X'y

    参数:
        X: 自变量矩阵（含截距项）
        y: 因变量向量

    返回:
        系数向量 β
    """
    # β = (X'X)^(-1)X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.inv(XtX) @ Xty
    return beta


def visualize_residuals_squared(df: pd.DataFrame, model: LinearRegression) -> None:
    """
    可视化残差的平方（展示平方损失的放大效应）

    参数:
        df: 包含 x, y, predicted, residual 的数据
        model: 拟合的模型
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：残差（绝对值）
    axes[0].bar(df.index, np.abs(df['residual']), color='steelblue', alpha=0.7)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('观测索引', fontsize=12)
    axes[0].set_ylabel('|残差|', fontsize=12)
    axes[0].set_title('残差绝对值 (MAE)', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # 右图：残差平方（平方放大效应）
    axes[1].bar(df.index, df['residual'] ** 2, color='coral', alpha=0.7)
    axes[1].set_xlabel('观测索引', fontsize=12)
    axes[1].set_ylabel('残差²', fontsize=12)
    axes[1].set_title('残差平方 (MSE) - 大误差被放大', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ols_loss_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ 损失函数对比图已保存为 ols_loss_comparison.png")
    plt.close()


def main() -> None:
    """主函数：演示 OLS 几何直觉"""
    print("=" * 60)
    print("示例2: 最小二乘法的几何直觉")
    print("=" * 60)

    # 1. 生成数据
    df = generate_simple_data()
    print(f"\n📊 原始数据:")
    print(df)

    # 2. 用 sklearn 拟合
    X_sklearn = df[['x']]
    y = df['y']
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_sklearn, y)

    print(f"\n📈 sklearn 结果:")
    print(f"  截距: {model_sklearn.intercept_:.4f}")
    print(f"  斜率: {model_sklearn.coef_[0]:.4f}")
    print(f"  MSE (损失): {compute_loss(y, model_sklearn.predict(X_sklearn), 'mse'):.4f}")
    print(f"  MAE (对比): {compute_loss(y, model_sklearn.predict(X_sklearn), 'mae'):.4f}")

    # 3. 手动计算 OLS 系数
    X_with_intercept = np.column_stack([np.ones(len(df)), df['x']])
    beta_manual = manual_ols(X_with_intercept, y)

    print(f"\n🧮 手动计算 (矩阵公式 β = (X'X)^(-1)X'y):")
    print(f"  X'X 矩阵:")
    print(X_with_intercept.T @ X_with_intercept)
    print(f"\n  (X'X)^(-1):")
    print(np.linalg.inv(X_with_intercept.T @ X_with_intercept))
    print(f"\n  系数 β:")
    print(f"    截距: {beta_manual[0]:.4f}")
    print(f"    斜率: {beta_manual[1]:.4f}")

    # 4. 验证一致性
    print(f"\n✅ 验证:")
    intercept_match = np.isclose(beta_manual[0], model_sklearn.intercept_)
    slope_match = np.isclose(beta_manual[1], model_sklearn.coef_[0])
    print(f"  截距一致: {intercept_match}")
    print(f"  斜率一致: {slope_match}")
    print(f"  结论: 手动计算与 sklearn 结果{'一致 ✓' if intercept_match and slope_match else '不一致 ✗'}")

    # 5. 可视化残差
    df['predicted'] = model_sklearn.predict(X_sklearn)
    df['residual'] = df['y'] - df['predicted']
    visualize_residuals_squared(df, model_sklearn)

    # 6. 均值也是"最小二乘"
    print(f"\n🔗 回顾: 均值也是最小二乘估计")
    print(f"  y 的均值: {y.mean():.4f}")
    print(f"  最小化 Σ(yi - μ)² 的 μ: {y.mean():.4f}")
    print(f"  → 回归只是这个思想扩展到'带自变量'的场景")

    print("\n" + "=" * 60)
    print("✅ 示例2完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
