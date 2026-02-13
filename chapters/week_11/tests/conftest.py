"""
Week 11 共享 Fixtures

提供测试用的共享数据和工具函数，用于树模型与集成学习相关测试。
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
# 房价预测数据 Fixtures - 贯穿案例
# =============================================================================

@pytest.fixture
def house_price_data():
    """
    创建模拟房价预测数据（本章贯穿案例）

    包含：
    - area_sqm: 面积（平方米）
    - bedrooms: 卧室数
    - bathrooms: 浴室数
    - age_years: 房龄（年）
    - distance_km: 距离市中心距离（公里）
    - price: 房价（万元）
    """
    np.random.seed(42)
    n = 500

    # 生成特征
    area = np.random.uniform(50, 200, n)
    bedrooms = np.random.randint(1, 5, n)
    bathrooms = np.random.randint(1, 3, n)
    age = np.random.randint(0, 30, n)
    distance = np.random.uniform(1, 30, n)

    # 生成目标变量（非线性关系 + 交互作用）
    # 房价 = 面积的平方根关系 + 房龄折旧 + 距离衰减 + 交互项 + 噪声
    price = (
        3.0 * np.sqrt(area) +           # 面积的边际递减
        10 * bedrooms +                   # 卧室数
        5 * bathrooms +                  # 浴室数
        -0.5 * age +                      # 房龄折旧
        -0.3 * distance +                 # 距离衰减
        0.02 * area * bedrooms / 10 +     # 面积和卧室的交互
        np.random.normal(0, 5, n)          # 噪声
    )

    # 确保房价为正
    price = np.maximum(price, 20)

    df = pd.DataFrame({
        'area_sqm': area,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age,
        'distance_km': distance,
        'price': price
    })

    return df


@pytest.fixture
def house_price_data_with_categories():
    """包含类别特征的房价数据"""
    np.random.seed(42)
    n = 500

    # 数值特征
    area = np.random.uniform(50, 200, n)
    bedrooms = np.random.randint(1, 5, n)
    age = np.random.randint(0, 30, n)

    # 类别特征
    city = np.random.choice(['北京', '上海', '深圳', '广州'], n, p=[0.4, 0.3, 0.2, 0.1])
    property_type = np.random.choice(['公寓', '别墅', '联排'], n, p=[0.7, 0.15, 0.15])

    # 生成目标变量
    base_price = 3.0 * np.sqrt(area) + 10 * bedrooms - 0.5 * age

    # 城市溢价
    city_premium = {'北京': 50, '上海': 45, '深圳': 48, '广州': 40}
    price = base_price + np.array([city_premium[c] for c in city])

    # 房型溢价
    type_premium = {'公寓': 0, '别墅': 30, '联排': 15}
    price += np.array([type_premium[t] for t in property_type])

    price += np.random.normal(0, 5, n)
    price = np.maximum(price, 30)

    df = pd.DataFrame({
        'area_sqm': area,
        'bedrooms': bedrooms,
        'age_years': age,
        'city': city,
        'property_type': property_type,
        'price': price
    })

    return df


@pytest.fixture
def house_price_data_correlated():
    """包含高度相关特征的数据（用于测试特征重要性陷阱）"""
    np.random.seed(42)
    n = 500

    # 生成一个基础变量
    base_size = np.random.uniform(50, 200, n)

    # 创建高度相关的特征
    area = base_size + np.random.normal(0, 5, n)
    living_area = base_size * 0.9 + np.random.normal(0, 3, n)  # 与面积高度相关
    rooms = base_size / 30 + np.random.randint(1, 2, n)  # 与面积相关

    # 房价主要由基础大小决定
    price = 3.0 * np.sqrt(base_size) + np.random.normal(0, 5, n)
    price = np.maximum(price, 20)

    df = pd.DataFrame({
        'area_sqm': area,
        'living_area': living_area,
        'rooms': rooms,
        'price': price
    })

    return df


# =============================================================================
# 分类数据 Fixtures
# =============================================================================

@pytest.fixture
def churn_classification_data():
    """客户流失分类数据（用于测试分类树和随机森林）"""
    np.random.seed(42)
    n = 500

    # 特征
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.random.uniform(20, 120, n)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, n)

    # 目标变量（与特征相关）
    logit = -3 + 0.05 * monthly_charges - 0.08 * tenure
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'tenure_months': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'churn': churn
    })

    return df


# =============================================================================
# 边界测试数据
# =============================================================================

@pytest.fixture
def single_feature_data():
    """只有一个特征的数据"""
    np.random.seed(42)
    n = 200

    x = np.random.uniform(0, 100, n)
    y = 2 * x + 10 + np.random.normal(0, 10, n)

    df = pd.DataFrame({
        'feature': x,
        'target': y
    })

    return df


@pytest.fixture
def very_small_dataset():
    """极小数据集（20 个样本）"""
    np.random.seed(42)
    n = 20

    df = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n),
        'feature_2': np.random.normal(0, 1, n),
        'target': np.random.normal(0, 1, n)
    })

    return df


@pytest.fixture
def constant_target_data():
    """目标变量几乎不变的数据（方差极小）"""
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n),
        'feature_2': np.random.normal(0, 1, n),
        'target': np.full(n, 100.0) + np.random.normal(0, 0.01, n)  # 几乎常数
    })

    return df


@pytest.fixture
def high_cardinality_categorical_data():
    """高基数类别特征数据（用于测试特征重要性陷阱）"""
    np.random.seed(42)
    n = 500

    # 数值特征
    feature_1 = np.random.normal(0, 1, n)

    # 高基数类别（如用户 ID）
    user_ids = [f"user_{i}" for i in range(n)]
    user_id = np.random.choice(user_ids, n)

    # 目标变量只与 feature_1 相关
    target = 2 * feature_1 + np.random.normal(0, 0.5, n)

    df = pd.DataFrame({
        'feature_1': feature_1,
        'user_id': user_id,
        'target': target
    })

    return df


# =============================================================================
# 过拟合测试数据
# =============================================================================

@pytest.fixture
def overfitting_scenario_data():
    """容易过拟合的数据（噪声大、样本少）"""
    np.random.seed(42)
    n_train = 50
    n_test = 200

    # 训练集（小样本 + 高噪声）
    X_train = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_train),
        'feature_2': np.random.normal(0, 1, n_train),
        'feature_3': np.random.normal(0, 1, n_train),
    })
    y_train = (
        X_train['feature_1'] +
        0.5 * X_train['feature_2'] +
        np.random.normal(0, 2, n_train)  # 高噪声
    )

    # 测试集（真实模式相同，但新数据）
    X_test = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, n_test),
        'feature_2': np.random.normal(0, 1, n_test),
        'feature_3': np.random.normal(0, 1, n_test),
    })
    y_test = (
        X_test['feature_1'] +
        0.5 * X_test['feature_2'] +
        np.random.normal(0, 0.5, n_test)  # 低噪声（真实关系）
    )

    return X_train, y_train, X_test, y_test


# =============================================================================
# 特征重要性测试数据
# =============================================================================

@pytest.fixture
def feature_importance_known_truth():
    """特征重要性真值已知的数据"""
    np.random.seed(42)
    n = 500

    # 特征重要性排序：feature_1 > feature_2 > feature_3 > feature_4（噪声）
    X = pd.DataFrame({
        'feature_1': np.random.uniform(0, 10, n),      # 最重要
        'feature_2': np.random.uniform(0, 10, n),      # 次重要
        'feature_3': np.random.uniform(0, 10, n),      # 弱重要
        'feature_4': np.random.uniform(0, 10, n),      # 噪声
    })

    # 目标变量：feature_1 权重最大，feature_4 不相关
    y = (
        5.0 * X['feature_1'] +
        2.0 * X['feature_2'] +
        0.5 * X['feature_3'] +
        0.0 * X['feature_4'] +  # 不相关
        np.random.normal(0, 1, n)
    )

    df = X.copy()
    df['target'] = y

    return df


# =============================================================================
# AI 树模型代码示例
# =============================================================================

@pytest.fixture
def good_tree_code_example():
    """示例：合格的树模型代码"""
    return """
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 决策树（带正则化）
dt = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=10,
    random_state=42
)
dt.fit(X_train, y_train)

# 随机森林（网格搜索调优）
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [1, 5, 10]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2'
)
rf_grid.fit(X_train, y_train)

# 评估
best_rf = rf_grid.best_estimator_
y_pred = best_rf.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"最佳参数: {rf_grid.best_params_}")
print(f"测试集 R²: {r2:.3f}")
"""


@pytest.fixture
def bad_tree_code_overfitting():
    """示例：存在过拟合风险的树模型代码"""
    return """
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 决策树（无限制，容易过拟合）
dt = DecisionTreeRegressor(random_state=42)  # 没有限制 max_depth
dt.fit(X_train, y_train)

# 评估
train_r2 = dt.score(X_train, y_train)
test_r2 = dt.score(X_test, y_test)

print(f"训练集 R²: {train_r2:.3f}")
print(f"测试集 R²: {test_r2:.3f}")
# 问题：训练集 R² 可能接近 1.0，但测试集很低
"""


@pytest.fixture
def bad_tree_code_no_tuning():
    """示例：缺少超参数调优的代码"""
    return """
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 随机森林（使用默认参数，未调优）
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# 直接评估，未与基线对比
test_r2 = rf.score(X_test, y_test)
print(f"测试集 R²: {test_r2:.3f}")
# 问题：未调优，未与基线对比，未检查特征重要性
"""


@pytest.fixture
def bad_tree_code_feature_importance_misinterpretation():
    """示例：错误解释特征重要性的代码"""
    return """
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# 获取特征重要性
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(importance)

# 错误解释示例：
# "面积重要性最高，所以扩大面积一定能让房价上涨"
# 问题：特征重要性 ≠ 因果关系
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
