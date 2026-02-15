"""
Week 09 共享 Fixtures

提供测试用的共享数据和工具函数。

测试覆盖：
- 简单线性回归（OLS）
- 回归系数解释
- R² 和拟合优度
- 回归假设（LINE：线性、独立性、正态性、等方差）
- 模型诊断（残差分析、QQ图、Cook's 距离、VIF）
- 多元回归与多重共线性
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sys

# 添加 starter_code 到 Python 路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 简单线性回归测试 Fixtures
# =============================================================================

@pytest.fixture
def simple_linear_data():
    """
    标准简单线性回归数据
    y = 10 + 0.5 * x + noise
    """
    np.random.seed(42)
    x = np.random.normal(loc=50, scale=15, size=100)
    y = 10 + 0.5 * x + np.random.normal(loc=0, scale=5, size=100)
    return {'x': x, 'y': y}


@pytest.fixture
def simple_linear_perfect():
    """
    完美线性关系数据（无噪声）
    y = 5 + 2 * x
    """
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y = 5 + 2 * x
    return {'x': x, 'y': y}


@pytest.fixture
def simple_linear_negative_slope():
    """
    负斜率数据
    y = 100 - 0.8 * x + noise
    """
    np.random.seed(42)
    x = np.random.normal(loc=50, scale=15, size=100)
    y = 100 - 0.8 * x + np.random.normal(loc=0, scale=5, size=100)
    return {'x': x, 'y': y}


@pytest.fixture
def simple_linear_no_relationship():
    """
    无关系数据（斜率 = 0）
    y = 50 + noise
    """
    np.random.seed(42)
    x = np.random.normal(loc=50, scale=15, size=100)
    y = 50 + np.random.normal(loc=0, scale=10, size=100)
    return {'x': x, 'y': y}


@pytest.fixture
def simple_linear_high_r_squared():
    """
    高 R² 数据（强线性关系，低噪声）
    """
    np.random.seed(42)
    x = np.random.normal(loc=50, scale=15, size=200)
    y = 10 + 0.8 * x + np.random.normal(loc=0, scale=2, size=200)  # 低噪声
    return {'x': x, 'y': y}


@pytest.fixture
def simple_linear_low_r_squared():
    """
    低 R² 数据（弱线性关系，高噪声）
    """
    np.random.seed(42)
    x = np.random.normal(loc=50, scale=15, size=100)
    y = 10 + 0.3 * x + np.random.normal(loc=0, scale=20, size=100)  # 高噪声
    return {'x': x, 'y': y}


# =============================================================================
# 回归假设测试 Fixtures
# =============================================================================

@pytest.fixture
def assumption_linear_met():
    """
    满足线性假设的数据
    """
    np.random.seed(42)
    x = np.random.uniform(0, 100, 100)
    y = 5 + 0.7 * x + np.random.normal(0, 5, 100)
    return {'x': x, 'y': y}


@pytest.fixture
def assumption_linear_violated_quadratic():
    """
    违反线性假设的数据（二次关系）
    """
    np.random.seed(42)
    x = np.random.uniform(-10, 10, 100)
    y = 50 + 0.5 * x**2 + np.random.normal(0, 5, 100)  # 二次关系
    return {'x': x, 'y': y}


@pytest.fixture
def assumption_normality_met():
    """
    满足正态性假设的数据（残差近似正态）
    """
    np.random.seed(42)
    x = np.random.normal(50, 15, 100)
    y = 10 + 0.5 * x + np.random.normal(0, 5, 100)
    return {'x': x, 'y': y}


@pytest.fixture
def assumption_normality_violated_skewed():
    """
    违反正态性假设的数据（残差偏态）
    """
    np.random.seed(42)
    x = np.random.normal(50, 15, 100)
    # 使用指数分布噪声，导致残差偏态
    noise = np.random.exponential(scale=5, size=100) - 5
    y = 10 + 0.5 * x + noise
    return {'x': x, 'y': y}


@pytest.fixture
def assumption_homoscedasticity_met():
    """
    满足等方差假设的数据（同方差）
    """
    np.random.seed(42)
    x = np.random.uniform(0, 100, 100)
    y = 10 + 0.5 * x + np.random.normal(0, 5, 100)  # 恒定方差
    return {'x': x, 'y': y}


@pytest.fixture
def assumption_homoscedasticity_violated():
    """
    违反等方差假设的数据（异方差）
    方差随 x 增大而增大
    """
    np.random.seed(42)
    x = np.random.uniform(10, 100, 100)
    # 方差随 x 线性增长
    noise = np.random.normal(0, 1, 100) * (x / 10)
    y = 10 + 0.5 * x + noise
    return {'x': x, 'y': y}


@pytest.fixture
def assumption_independence_violated_autocorrelated():
    """
    违反独立性假设的数据（自相关残差）
    模拟时间序列数据
    """
    np.random.seed(42)
    n = 100
    x = np.arange(n)
    # 自相关噪声
    noise = np.zeros(n)
    noise[0] = np.random.normal(0, 1)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i-1] + np.random.normal(0, 1)
    y = 10 + 0.5 * x + noise
    return {'x': x, 'y': y}


# =============================================================================
# 模型诊断测试 Fixtures
# =============================================================================

@pytest.fixture
def diagnostics_with_outlier():
    """
    包含离群点的数据
    """
    np.random.seed(42)
    x = np.random.normal(50, 15, 95)
    y = 10 + 0.5 * x + np.random.normal(0, 5, 95)
    # 添加离群点
    x_with_outlier = np.concatenate([x, [150]])
    y_with_outlier = np.concatenate([y, [20]])  # 异常的 y 值
    return {'x': x_with_outlier, 'y': y_with_outlier}


@pytest.fixture
def diagnostics_with_high_leverage():
    """
    包含高杠杆点的数据
    （x 值远离数据中心，但 y 符合趋势）
    """
    np.random.seed(42)
    x = np.random.normal(50, 10, 95)
    y = 10 + 0.5 * x + np.random.normal(0, 5, 95)
    # 添加高杠杆点（x 远离中心，但 y 符合回归线）
    x_with_leverage = np.concatenate([x, [120]])
    y_with_leverage = np.concatenate([y, [10 + 0.5 * 120]])
    return {'x': x_with_leverage, 'y': y_with_leverage}


@pytest.fixture
def diagnostics_with_influential_point():
    """
    包含高影响点的数据
    （高杠杆 + 离群 y 值）
    """
    np.random.seed(42)
    x = np.random.normal(50, 10, 95)
    y = 10 + 0.5 * x + np.random.normal(0, 5, 95)
    # 添加高影响点
    x_with_influence = np.concatenate([x, [120]])
    y_with_influence = np.concatenate([y, [200]])  # 不符合趋势的 y 值
    return {'x': x_with_influence, 'y': y_with_influence}


@pytest.fixture
def diagnostics_multiple_outliers():
    """
    包含多个离群点的数据
    """
    np.random.seed(42)
    x = np.random.normal(50, 15, 90)
    y = 10 + 0.5 * x + np.random.normal(0, 5, 90)
    # 添加多个离群点
    x_outliers = np.array([130, 140, -20, 150])
    y_outliers = np.array([10, 15, 100, 5])  # 异常的 y 值
    x_with_outliers = np.concatenate([x, x_outliers])
    y_with_outliers = np.concatenate([y, y_outliers])
    return {'x': x_with_outliers, 'y': y_with_outliers}


# =============================================================================
# 多元回归测试 Fixtures
# =============================================================================

@pytest.fixture
def multiple_regression_data():
    """
    标准多元回归数据
    y = 10 + 0.5*x1 + 0.3*x2 - 0.2*x3 + noise
    """
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(50, 15, n)
    x2 = np.random.normal(30, 10, n)
    x3 = np.random.normal(20, 5, n)
    y = 10 + 0.5 * x1 + 0.3 * x2 - 0.2 * x3 + np.random.normal(0, 5, n)
    X = np.column_stack([x1, x2, x3])
    return {'X': X, 'y': y, 'var_names': ['x1', 'x2', 'x3']}


@pytest.fixture
def multiple_regression_perfect():
    """
    完美多元回归数据（无噪声）
    y = 5 + 2*x1 + 3*x2
    """
    x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    x2 = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=float)
    y = 5 + 2 * x1 + 3 * x2
    X = np.column_stack([x1, x2])
    return {'X': X, 'y': y, 'var_names': ['x1', 'x2']}


# =============================================================================
# 多重共线性测试 Fixtures
# =============================================================================

@pytest.fixture
def multicollinearity_none():
    """
    无多重共线性数据（自变量不相关）
    """
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = np.random.normal(0, 1, n)
    # 确保不相关
    X = np.column_stack([x1, x2, x3])
    y = 10 + 2 * x1 + 3 * x2 + 1.5 * x3 + np.random.normal(0, 1, n)
    return {'X': X, 'y': y, 'var_names': ['x1', 'x2', 'x3']}


@pytest.fixture
def multicollinearity_moderate():
    """
    中度多重共线性数据（自变量中度相关）
    """
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = 0.5 * x1 + np.random.normal(0, 0.5, n)  # 与 x1 相关
    x3 = np.random.normal(0, 1, n)
    X = np.column_stack([x1, x2, x3])
    y = 10 + 2 * x1 + 3 * x2 + 1.5 * x3 + np.random.normal(0, 1, n)
    return {'X': X, 'y': y, 'var_names': ['x1', 'x2', 'x3']}


@pytest.fixture
def multicollinearity_severe():
    """
    严重多重共线性数据（自变量高度相关）
    """
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = 0.95 * x1 + np.random.normal(0, 0.1, n)  # 与 x1 高度相关
    x3 = np.random.normal(0, 1, n)
    X = np.column_stack([x1, x2, x3])
    y = 10 + 2 * x1 + 3 * x2 + 1.5 * x3 + np.random.normal(0, 1, n)
    return {'X': X, 'y': y, 'var_names': ['x1', 'x2', 'x3']}


@pytest.fixture
def multicollinearity_perfect():
    """
    完全共线性数据（一个变量是其他变量的线性组合）
    这会导致设计矩阵奇异，OLS 无法估计
    """
    np.random.seed(42)
    n = 200
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    x3 = 2 * x1 + 3 * x2  # 完全共线
    X = np.column_stack([x1, x2, x3])
    y = 10 + 2 * x1 + 3 * x2 + 1.5 * x3 + np.random.normal(0, 1, n)
    return {'X': X, 'y': y, 'var_names': ['x1', 'x2', 'x3']}


# =============================================================================
# 边界情况 Fixtures
# =============================================================================

@pytest.fixture
def empty_data():
    """空数据"""
    return {'x': np.array([]), 'y': np.array([])}


@pytest.fixture
def single_point_data():
    """单点数据（无法拟合回归）"""
    return {'x': np.array([5.0]), 'y': np.array([10.0])}


@pytest.fixture
def two_points_data():
    """两点数据（可以拟合完美直线）"""
    return {'x': np.array([1.0, 5.0]), 'y': np.array([3.0, 11.0])}


@pytest.fixture
def constant_y_data():
    """y 为常量的数据（零方差）"""
    np.random.seed(42)
    x = np.random.normal(50, 15, 50)
    y = np.full(50, 10.0)
    return {'x': x, 'y': y}


@pytest.fixture
def constant_x_data():
    """x 为常量的数据（无法估计斜率）"""
    np.random.seed(42)
    x = np.full(50, 50.0)
    y = np.random.normal(35, 5, 50)
    return {'x': x, 'y': y}


@pytest.fixture
def nan_data():
    """包含 NaN 的数据"""
    np.random.seed(42)
    x = np.random.normal(50, 15, 50)
    y = 10 + 0.5 * x + np.random.normal(0, 5, 50)
    x[10] = np.nan
    y[20] = np.nan
    return {'x': x, 'y': y}


@pytest.fixture
def infinite_data():
    """包含 Inf 的数据"""
    np.random.seed(42)
    x = np.random.normal(50, 15, 50)
    y = 10 + 0.5 * x + np.random.normal(0, 5, 50)
    x[10] = np.inf
    y[20] = -np.inf
    return {'x': x, 'y': y}


@pytest.fixture
def mismatched_dimensions():
    """维度不匹配的数据"""
    np.random.seed(42)
    x = np.random.normal(50, 15, 100)
    y = np.random.normal(35, 5, 50)  # 不同的样本量
    return {'x': x, 'y': y}


# =============================================================================
# StatLab 测试 Fixtures
# =============================================================================

@pytest.fixture
def statlab_ad_sales_data():
    """
    StatLab 项目用例：广告投入 vs 销售额数据
    """
    np.random.seed(42)
    n = 500
    # 多个广告渠道
    ad_tv = np.random.uniform(10, 100, n)
    ad_online = np.random.uniform(5, 80, n)
    ad_social = np.random.uniform(1, 50, n)

    # 销售额受广告影响，加上噪声
    sales = 50 + 0.3 * ad_tv + 0.4 * ad_online + 0.2 * ad_social + np.random.normal(0, 10, n)
    sales = np.maximum(sales, 0)  # 确保非负

    return {
        'ad_tv': ad_tv,
        'ad_online': ad_online,
        'ad_social': ad_social,
        'sales': sales
    }


@pytest.fixture
def statlab_price_demand_data():
    """
    StatLab 项目用例：价格 vs 需求数据（负相关）
    """
    np.random.seed(42)
    prices = np.random.uniform(10, 100, 200)
    # 需求随价格下降
    demand = 1000 - 5 * prices + np.random.normal(0, 50, 200)
    demand = np.maximum(demand, 0)
    return {'price': prices, 'demand': demand}


@pytest.fixture
def statlab_experience_salary_data():
    """
    StatLab 项目用例：工作年限 vs 薪资数据
    """
    np.random.seed(42)
    n = 300
    experience = np.random.uniform(0, 30, n)
    # 薪资随经验增长，但有 diminishing returns
    salary = 40000 + 3000 * experience - 50 * experience**2 + np.random.normal(0, 5000, n)
    salary = np.maximum(salary, 20000)
    return {'experience': experience, 'salary': salary}


# =============================================================================
# 验证工具 Fixtures
# =============================================================================

@pytest.fixture
def known_regression_coefficients():
    """
    已知回归系数的数据
    用于验证回归计算的正确性
    """
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    # y = 5 + 2*x，无噪声
    y = np.array([7, 9, 11, 13, 15], dtype=float)
    return {'x': x, 'y': y, 'true_intercept': 5.0, 'true_slope': 2.0}


@pytest.fixture
def tolerance():
    """
    数值比较的容差
    """
    return {
        'rtol': 1e-5,  # 相对容差
        'atol': 1e-8,  # 绝对容差
        'r_squared_tol': 0.01,  # R² 容差（1%）
        'coef_tol': 0.1,  # 系数容差
    }
