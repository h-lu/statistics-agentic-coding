"""
Week 11 共享 Fixtures

提供测试用的共享数据和工具函数。

测试覆盖：
- 决策树分类
- 随机森林分类
- 特征重要性
- 基线对比（傻瓜基线、逻辑回归基线、单特征树基线）
- 过拟合检测
- StatLab 集成
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sys
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 添加 starter_code 到 Python 路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 决策树测试 Fixtures
# =============================================================================

@pytest.fixture
def simple_tree_classification_data():
    """
    标准二分类数据（适合决策树）
    使用非线性可分数据，展示决策树相对于线性模型的优势
    """
    np.random.seed(42)
    # 创建具有非线性边界的分类数据
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_redundant=0,
        n_informative=3,
        random_state=42,
        n_clusters_per_class=2,
        class_sep=1.0
    )
    return {'X': X, 'y': y}


@pytest.fixture
def xor_like_data():
    """
    XOR 类型的数据（线性模型难以处理，决策树可以处理）
    展示决策树相对于线性模型的优势
    """
    np.random.seed(42)
    # 创建 XOR 模式的数据
    X = np.random.randn(200, 2)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return {'X': X, 'y': y}


@pytest.fixture
def tree_overfitting_data():
    """
    容易导致决策树过拟合的数据
    用于测试过拟合检测和剪枝
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_redundant=5,
        n_informative=5,
        random_state=42,
        n_clusters_per_class=1,
        flip_y=0.1  # 添加噪声
    )
    return {'X': X, 'y': y}


@pytest.fixture
def single_feature_data():
    """
    单特征数据（用于测试决策树桩）
    """
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = (X[:, 0] > 0).astype(int)
    return {'X': X, 'y': y}


@pytest.fixture
def two_feature_tree_data():
    """
    两特征数据（便于可视化决策边界）
    """
    np.random.seed(42)
    X_class_0 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 50)
    X_class_1 = np.random.multivariate_normal([2, 2], [[1, -0.5], [-0.5, 1]], 50)
    X = np.vstack([X_class_0, X_class_1])
    y = np.array([0] * 50 + [1] * 50)
    return {'X': X, 'y': y}


# =============================================================================
# 随机森林测试 Fixtures
# =============================================================================

@pytest.fixture
def random_forest_data():
    """
    适合随机森林的中等规模数据
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_redundant=3,
        n_informative=5,
        random_state=42,
        n_clusters_per_class=2,
        class_sep=1.2
    )
    return {'X': X, 'y': y}


@pytest.fixture
def imbalanced_tree_data():
    """
    类别不平衡的树模型数据
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_redundant=1,
        n_informative=3,
        random_state=42,
        weights=[0.7, 0.3],
        class_sep=0.8
    )
    return {'X': X, 'y': y}


# =============================================================================
# 基线对比测试 Fixtures
# =============================================================================

@pytest.fixture
def baseline_comparison_data():
    """
    用于基线对比的数据
    包含训练集和测试集
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=6,
        n_redundant=2,
        n_informative=3,
        random_state=42,
        class_sep=0.9
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


@pytest.fixture
def easy_classification_data():
    """
    容易分类的数据（所有模型都应该表现好）
    """
    np.random.seed(42)
    X_class_0 = np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    X_class_1 = np.random.normal(loc=[3, 3], scale=0.5, size=(50, 2))
    X = np.vstack([X_class_0, X_class_1])
    y = np.array([0] * 50 + [1] * 50)
    return {'X': X, 'y': y}


@pytest.fixture
def hard_classification_data():
    """
    难以分类的数据（重叠严重）
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=400,
        n_features=5,
        n_redundant=2,
        n_informative=2,
        random_state=42,
        flip_y=0.2,  # 20% 标签噪声
        class_sep=0.5
    )
    return {'X': X, 'y': y}


# =============================================================================
# 特征重要性测试 Fixtures
# =============================================================================

@pytest.fixture
def feature_importance_data():
    """
    具有明显特征重要性差异的数据
    前 3 个特征重要，后 3 个特征不重要
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=6,
        n_redundant=0,
        n_informative=3,
        random_state=42,
        n_clusters_per_class=1,
        class_sep=1.5
    )
    # 创建特征名称
    feature_names = [f'feature_{i}' for i in range(6)]
    return {'X': X, 'y': y, 'feature_names': feature_names}


# =============================================================================
# 过拟合检测测试 Fixtures
# =============================================================================

@pytest.fixture
def overfitting_scenario_data():
    """
    容易过拟合的场景数据
    小样本 + 高维度
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=50,  # 小样本
        n_features=20,  # 高维度
        n_redundant=10,
        n_informative=5,
        random_state=42,
        n_clusters_per_class=1,
        flip_y=0.05
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


# =============================================================================
# 边界情况 Fixtures
# =============================================================================

@pytest.fixture
def minimal_tree_data():
    """
    最小决策树数据（每类 2 个样本）
    """
    X = np.array([[0, 0], [1, 1], [5, 5], [6, 6]], dtype=float)
    y = np.array([0, 0, 1, 1])
    return {'X': X, 'y': y}


@pytest.fixture
def constant_feature_tree_data():
    """
    包含常量特征的树模型数据
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100,
        n_features=3,
        n_redundant=0,
        random_state=42
    )
    X[:, 0] = 5.0  # 第一列是常量
    return {'X': X, 'y': y}


@pytest.fixture
def categorical_like_tree_data():
    """
    类别型特征的数据（少量唯一值）
    """
    np.random.seed(42)
    n_samples = 200
    # 创建少量唯一值的特征（类似类别型）
    feature_1 = np.random.choice([0, 1, 2], size=n_samples)
    feature_2 = np.random.choice([0.0, 1.0, 2.0, 3.0], size=n_samples)
    feature_3 = np.random.randn(n_samples)

    X = np.column_stack([feature_1, feature_2, feature_3])
    # 基于特征创建目标变量
    y = ((feature_1 == 1) & (feature_2 > 1)).astype(int)

    return {'X': X, 'y': y}


# =============================================================================
# StatLab 测试 Fixtures
# =============================================================================

@pytest.fixture
def statlab_customer_churn_data():
    """
    StatLab 项目用例：客户流失预测数据
    包含混合类型的特征
    """
    np.random.seed(42)
    n = 500

    # 数值型特征
    purchase_count = np.random.poisson(10, n)
    avg_spend = np.random.gamma(20, 10, n)
    days_since_last_purchase = np.random.exponential(30, n)
    registration_days = np.random.uniform(30, 365, n)

    # 流失概率（基于特征的简化模型）
    logit = -3 + 0.05 * purchase_count - 0.01 * avg_spend + 0.03 * days_since_last_purchase - 0.002 * registration_days
    prob = 1 / (1 + np.exp(-logit))
    is_churned = (np.random.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'purchase_count': purchase_count,
        'avg_spend': avg_spend,
        'days_since_last_purchase': days_since_last_purchase,
        'registration_days': registration_days,
        'is_churned': is_churned
    })

    return df


@pytest.fixture
def statlab_feature_lists():
    """
    StatLab 项目的特征列表
    """
    return {
        'numeric_features': ['purchase_count', 'avg_spend', 'days_since_last_purchase', 'registration_days'],
        'categorical_features': [],
        'target': 'is_churned'
    }


# =============================================================================
# 验证工具 Fixtures
# =============================================================================

@pytest.fixture
def tolerance():
    """
    数值比较的容差
    """
    return {
        'rtol': 1e-5,
        'atol': 1e-8,
        'metric_tol': 0.02,  # 评估指标容差（2%）
        'prob_tol': 0.05,    # 概率容差（5%）
        'importance_tol': 0.01,  # 特征重要性容差
    }


@pytest.fixture
def tree_model_params():
    """
    标准决策树参数
    """
    return {
        'max_depth': 5,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'random_state': 42
    }


@pytest.fixture
def rf_model_params():
    """
    标准随机森林参数
    """
    return {
        'n_estimators': 50,  # 测试时用较少的树以加快速度
        'max_depth': 5,
        'max_features': 'sqrt',
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'random_state': 42,
        'n_jobs': -1
    }


# =============================================================================
# 预期结果 Fixtures
# =============================================================================

@pytest.fixture
def expected_baseline_performance():
    """
    预期的基线模型性能范围
    """
    return {
        'dummy_auc_range': (0.45, 0.55),  # 傻瓜基线 AUC 应接近 0.5
        'min_improvement': 0.05,  # 相对于基线的最小提升量
        'reasonable_auc': 0.6,  # 合理的最低 AUC
    }
