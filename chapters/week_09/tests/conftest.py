"""
Week 09 共享 Fixtures

提供测试用的共享数据和工具函数，用于回归分析与模型诊断相关测试。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# 添加 starter_code 到导入路径（当存在时）
starter_code_path = Path(__file__).parent.parent / "starter_code"
if starter_code_path.exists():
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 房价数据 Fixtures - 贯穿案例
# =============================================================================

@pytest.fixture
def housing_data():
    """
    创建模拟房价数据（本章贯穿案例）

    包含：
    - area_sqm: 面积（平米）
    - price_wan: 售价（万元）
    - age_years: 房龄（年）
    - n_rooms: 房间数
    """
    np.random.seed(42)
    n = 200

    # 生成相关预测变量
    area = np.random.normal(loc=100, scale=20, size=n)
    age = np.random.normal(loc=10, scale=5, size=n)
    n_rooms = (area / 20 + np.random.normal(loc=0, scale=0.5, size=n)).astype(int)
    n_rooms = np.clip(n_rooms, 1, 10)

    # 生成目标变量（真实关系 + 噪音）
    true_coef_area = 1.2
    true_coef_age = -0.4
    true_coef_rooms = 3.0
    true_intercept = 25.0

    price = (true_intercept +
             true_coef_area * area +
             true_coef_age * age +
             true_coef_rooms * n_rooms +
             np.random.normal(loc=0, scale=10, size=n))

    df = pd.DataFrame({
        'area_sqm': area,
        'age_years': age,
        'n_rooms': n_rooms,
        'price_wan': price
    })

    return df


@pytest.fixture
def housing_data_simple():
    """简单回归用的房价数据（只有面积）"""
    np.random.seed(42)
    n = 100
    area = np.random.normal(loc=100, scale=20, size=n)
    price = 25 + 1.18 * area + np.random.normal(loc=0, scale=15, size=n)

    df = pd.DataFrame({
        'area_sqm': area,
        'price_wan': price
    })

    return df


# =============================================================================
# 多重共线性数据
# =============================================================================

@pytest.fixture
def data_with_multicollinearity():
    """
    创建具有多重共线性的数据

    特征：
    - x1, x2, x3 高度相关
    - VIF 会很高
    """
    np.random.seed(42)
    n = 100

    # 创建基础变量
    x_base = np.random.normal(loc=0, scale=1, size=n)

    # 创建相关变量（高度共线）
    x1 = x_base + np.random.normal(loc=0, scale=0.1, size=n)
    x2 = 0.8 * x_base + np.random.normal(loc=0, scale=0.2, size=n)
    x3 = 1.2 * x_base + np.random.normal(loc=0, scale=0.15, size=n)

    # 生成目标变量
    y = 5 + 2 * x_base + np.random.normal(loc=0, scale=1, size=n)

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })

    return df


@pytest.fixture
def data_no_multicollinearity():
    """
    创建无多重共线性的数据

    特征：
    - 各预测变量独立
    - VIF 应该接近 1
    """
    np.random.seed(42)
    n = 100

    x1 = np.random.normal(loc=0, scale=1, size=n)
    x2 = np.random.normal(loc=0, scale=1, size=n)
    x3 = np.random.normal(loc=0, scale=1, size=n)

    y = 5 + 2 * x1 + 1.5 * x2 - x3 + np.random.normal(loc=0, scale=1, size=n)

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'y': y
    })

    return df


# =============================================================================
# 残差诊断数据
# =============================================================================

@pytest.fixture
def data_violating_linearity():
    """
    创建违反线性假设的数据

    特征：
    - 真实关系是非线性（二次）
    - 残差图会呈现 U 型
    """
    np.random.seed(42)
    n = 100
    x = np.random.uniform(-10, 10, size=n)
    # 二次关系
    y = 5 + 0.5 * x + 0.3 * x**2 + np.random.normal(loc=0, scale=2, size=n)

    df = pd.DataFrame({'x': x, 'y': y})
    return df


@pytest.fixture
def data_violating_homoscedasticity():
    """
    创建违反等方差假设的数据（异方差）

    特征：
    - 残差方差随拟合值增大而增大
    - 残差图呈现"喇叭形"
    """
    np.random.seed(42)
    n = 100
    x = np.random.uniform(0, 10, size=n)
    # 方差随 x 增大而增大
    noise_std = 1 + 0.5 * x
    y = 5 + 2 * x + np.random.normal(loc=0, scale=noise_std, size=n)

    df = pd.DataFrame({'x': x, 'y': y})
    return df


@pytest.fixture
def data_meeting_assumptions():
    """
    创建满足所有 LINE 假设的数据

    特征：
    - 线性关系
    - 残差正态分布
    - 等方差
    - 独立观测
    """
    np.random.seed(42)
    n = 100
    x = np.random.normal(loc=50, scale=10, size=n)
    y = 10 + 1.5 * x + np.random.normal(loc=0, scale=5, size=n)

    df = pd.DataFrame({'x': x, 'y': y})
    return df


# =============================================================================
# 异常点数据
# =============================================================================

@pytest.fixture
def data_with_outlier():
    """包含离群点的数据"""
    np.random.seed(42)
    n = 50
    x = np.random.normal(loc=0, scale=1, size=n)
    y = 5 + 2 * x + np.random.normal(loc=0, scale=1, size=n)

    # 添加一个离群点
    x = np.append(x, 0)
    y = np.append(y, 15)  # 远离回归线

    df = pd.DataFrame({'x': x, 'y': y})
    return df


@pytest.fixture
def data_with_high_leverage():
    """包含高杠杆点的数据"""
    np.random.seed(42)
    n = 50
    x = np.random.normal(loc=0, scale=1, size=n)
    y = 5 + 2 * x + np.random.normal(loc=0, scale=1, size=n)

    # 添加一个高杠杆点（x 值远离其他点）
    x = np.append(x, 10)
    y = np.append(y, 5 + 2 * 10 + 1)  # 接近回归线

    df = pd.DataFrame({'x': x, 'y': y})
    return df


@pytest.fixture
def data_with_influential_point():
    """包含强影响点的数据（高杠杆 + 离群）"""
    np.random.seed(42)
    n = 50
    x = np.random.normal(loc=0, scale=1, size=n)
    y = 5 + 2 * x + np.random.normal(loc=0, scale=1, size=n)

    # 添加一个强影响点（高杠杆 + 大残差）
    x = np.append(x, 10)
    y = np.append(y, 0)  # 远离预期值（5 + 2*10 = 25）

    df = pd.DataFrame({'x': x, 'y': y})
    return df


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def empty_data():
    """空数据"""
    return pd.DataFrame({'x': [], 'y': []})


@pytest.fixture
def single_observation():
    """单观测数据"""
    return pd.DataFrame({'x': [1.0], 'y': [2.0]})


@pytest.fixture
def constant_x():
    """x 为常数（无法估计斜率）"""
    n = 50
    return pd.DataFrame({
        'x': [5.0] * n,
        'y': np.random.normal(loc=10, scale=2, size=n)
    })


@pytest.fixture
def perfect_collinearity():
    """完全共线性（x2 = 2 * x1）"""
    np.random.seed(42)
    n = 50
    x1 = np.random.normal(loc=0, scale=1, size=n)
    x2 = 2 * x1  # 完全共线
    y = 5 + 2 * x1 + np.random.normal(loc=0, scale=1, size=n)

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    return df


# =============================================================================
# AI 报告示例
# =============================================================================

@pytest.fixture
def good_regression_report():
    """一份合格的回归分析报告"""
    return """
## 回归分析报告

### 模型拟合
- 方程: 房价 = 20.5 + 1.25×面积 - 0.38×房龄
- R² = 0.82, 调整 R² = 0.80
- F(2, 97) = 45.6, p < 0.001

### 系数解释(95% CI)
| 变量 | 系数 | 标准误 | 95% CI | p 值 |
|------|------|--------|---------|------|
| 截距 | 20.50 | 3.20 | [14.12, 26.88] | <0.001 |
| 面积 | 1.25 | 0.18 | [0.89, 1.61] | <0.001 |
| 房龄 | -0.38 | 0.12 | [-0.62, -0.14] | 0.002 |

解释: 在其他变量不变的情况下,面积每增加 1 平米,房价平均上涨 1.25 万元(95% CI: [0.89, 1.61])。

### 残差诊断
- 线性假设: 残差 vs 拟合值图显示残差随机散布,无线性模式 ✅
- 正态性: QQ 图显示残差近似沿对角线分布,Shapiro-Wilk p = 0.08 ✅
- 等方差: 残差散布在所有拟合值上大致均匀 ✅
- 独立性: Durbin-Watson = 1.95, 接近理想值 2 ✅

### 多重共线性检查
- 面积 VIF = 1.2, 房龄 VIF = 1.1
- 无严重共线性问题 ✅

### 异常点分析
- Cook's D > 1 的观测: 2 个(索引 #45, #128)
- 删除后面积系数从 1.25 变为 1.18(变化 < 6%)
- 结论: 模型对异常点稳健 ✅

### 局限性与因果警告
⚠️ 本分析仅描述房价与面积、房龄的关联关系,不能直接推断因果。

可能的混杂变量:
- 地段(市中心 vs 郊区)
- 装修程度
- 学区质量
"""


@pytest.fixture
def bad_regression_report_no_diagnostics():
    """缺少残差诊断的糟糕报告"""
    return """
## 回归分析报告

### 模型拟合
- 方程: 房价 = 20.5 + 1.25×面积 - 0.38×房龄
- R² = 0.82
- 所有 p 值 < 0.05

### 结论
面积和房龄显著影响房价,模型拟合良好,可用于预测。
"""


@pytest.fixture
def bad_regression_report_causal_claim():
    """错误地将系数解释为因果的报告"""
    return """
## 回归分析报告

### 系数解释
- 面积系数 = 1.25 (p < 0.001)

结论: 面积每增加 1 平米,会导致房价上涨 1.25 万元。
建议: 如果要提升房价,应该增加房屋面积。
"""


@pytest.fixture
def bad_regression_report_no_vif():
    """缺少多重共线性检查的报告"""
    return """
## 回归分析报告

### 模型拟合
我们使用面积、房间数、卧室数、客厅数作为预测变量。

- 面积系数 = 0.85 (p = 0.02)
- 卧室数系数 = -2.30 (p = 0.15)
- 客厅数系数 = -1.80 (p = 0.22)
- R² = 0.78

结论: 客厅数和卧室数对房价有负面影响。
"""


# =============================================================================
# 临时输出目录
# =============================================================================

@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """创建临时输出目录"""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir
