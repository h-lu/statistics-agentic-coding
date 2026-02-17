"""
Week 10 共享 Fixtures

提供测试用的共享数据和工具函数。

测试覆盖：
- 逻辑回归分类
- 混淆矩阵与评估指标（准确率、精确率、召回率、F1）
- ROC 曲线与 AUC
- Pipeline 与数据泄漏防护
- 类别不平衡处理
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import sys
from sklearn.datasets import make_classification

# 添加 starter_code 到 Python 路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 逻辑回归测试 Fixtures
# =============================================================================

@pytest.fixture
def simple_binary_classification_data():
    """
    标准二分类数据（线性可分）
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1,
        class_sep=1.5
    )
    return {'X': X, 'y': y}


@pytest.fixture
def binary_classification_perfect():
    """
    完美线性可分数据（无重叠）
    """
    np.random.seed(42)
    # 创建两个完全分离的类别
    X_class_0 = np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    X_class_1 = np.random.normal(loc=[3, 3], scale=0.5, size=(50, 2))
    X = np.vstack([X_class_0, X_class_1])
    y = np.array([0] * 50 + [1] * 50)
    return {'X': X, 'y': y}


@pytest.fixture
def binary_classification_overlapping():
    """
    有重叠的类别（更真实）
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=300,
        n_features=3,
        n_redundant=1,
        n_informative=2,
        random_state=42,
        flip_y=0.1,  # 10% 标签噪声
        class_sep=0.8
    )
    return {'X': X, 'y': y}


@pytest.fixture
def imbalanced_classification_data():
    """
    类别不平衡数据（20% 正类）
    模拟流失预测场景
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=500,
        n_features=5,
        n_redundant=1,
        n_informative=3,
        random_state=42,
        weights=[0.8, 0.2],  # 80% 负类，20% 正类
        class_sep=0.7
    )
    return {'X': X, 'y': y}


@pytest.fixture
def highly_imbalanced_data():
    """
    极端类别不平衡数据（5% 正类）
    模拟欺诈检测场景
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        weights=[0.95, 0.05],  # 95% 负类，5% 正类
        class_sep=0.8
    )
    return {'X': X, 'y': y}


# =============================================================================
# 评估指标测试 Fixtures
# =============================================================================

@pytest.fixture
def known_predictions():
    """
    已知预测结果，用于验证评估指标计算
    """
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 0, 1, 1])
    # 混淆矩阵:
    # 位置 0,1,3,4: y_true=0, y_pred=0 → TN=4
    # 位置 2: y_true=0, y_pred=1 → FP=1
    # 位置 7: y_true=1, y_pred=0 → FN=1
    # 位置 5,6,8,9: y_true=1, y_pred=1 → TP=4
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'expected': {
            'TP': 4,
            'TN': 4,
            'FP': 1,
            'FN': 1,
            'accuracy': 0.8,  # (TP+TN)/total = 8/10
            'precision': 0.8,  # TP/(TP+FP) = 4/5
            'recall': 0.8,     # TP/(TP+FN) = 4/5
            'f1': 0.8,
            'specificity': 0.8  # TN/(TN+FP) = 4/5
        }
    }


@pytest.fixture
def perfect_predictions():
    """
    完美预测
    """
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'expected': {
            'TP': 3,
            'TN': 3,
            'FP': 0,
            'FN': 0,
            'accuracy': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0
        }
    }


@pytest.fixture
def worst_predictions():
    """
    最差预测（全错）
    """
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'expected': {
            'TP': 0,
            'TN': 0,
            'FP': 3,
            'FN': 3,
            'accuracy': 0.0,
            'precision': 0.0,  # TP=0
            'recall': 0.0,     # TP=0
            'f1': 0.0
        }
    }


@pytest.fixture
def probabilities_for_roc():
    """
    用于 ROC/AUC 测试的概率数据
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    # 完美排序：所有负类概率 < 所有正类概率
    return {
        'y_true': y_true,
        'y_prob': y_prob,
        'expected_auc': 1.0  # 完美分类
    }


# =============================================================================
# Pipeline 与数据泄漏测试 Fixtures
# =============================================================================

@pytest.fixture
def data_with_missing_values():
    """
    包含缺失值的数据
    用于测试 Pipeline 防泄漏
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200,
        n_features=3,
        n_redundant=0,
        n_informative=2,
        random_state=42
    )
    X = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3'])

    # 添加缺失值
    mask = np.random.random(X.shape) < 0.1
    X[mask] = np.nan

    return {'X': X, 'y': y}


@pytest.fixture
def mixed_type_classification_data():
    """
    混合类型特征数据（数值型 + 分类型）
    用于测试 ColumnTransformer
    """
    np.random.seed(42)

    # 数值型特征
    n_samples = 200
    numeric = np.random.randn(n_samples, 2)

    # 分类型特征
    categorical = np.random.choice(['A', 'B', 'C'], size=(n_samples, 2))

    # 目标变量
    X = np.hstack([numeric, categorical])
    y = (numeric[:, 0] + numeric[:, 1] > 0).astype(int)

    df = pd.DataFrame(X, columns=['num_1', 'num_2', 'cat_1', 'cat_2'])
    df['num_1'] = df['num_1'].astype(float)
    df['num_2'] = df['num_2'].astype(float)

    return {'X': df, 'y': y}


# =============================================================================
# 边界情况 Fixtures
# =============================================================================

@pytest.fixture
def empty_classification_data():
    """
    空分类数据
    """
    return {'X': np.array([]).reshape(0, 2), 'y': np.array([])}


@pytest.fixture
def single_class_data():
    """
    单类别数据（无法训练分类器）
    """
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = np.zeros(100)  # 全部是 0 类
    return {'X': X, 'y': y}


@pytest.fixture
def minimal_binary_data():
    """
    最小二分类数据（每类 2 个样本）
    """
    X = np.array([[0, 0], [1, 1], [5, 5], [6, 6]], dtype=float)
    y = np.array([0, 0, 1, 1])
    return {'X': X, 'y': y}


@pytest.fixture
def nan_classification_data():
    """
    包含 NaN 的分类数据
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100,
        n_features=3,
        n_redundant=0,
        random_state=42
    )
    X[10, 0] = np.nan
    X[20, 1] = np.nan
    return {'X': X, 'y': y}


@pytest.fixture
def constant_feature_data():
    """
    包含常量特征的数据
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        random_state=42
    )
    X[:, 0] = 5.0  # 第一列是常量
    return {'X': X, 'y': y}


# =============================================================================
# StatLab 测试 Fixtures
# =============================================================================

@pytest.fixture
def statlab_customer_churn_data():
    """
    StatLab 项目用例：客户流失预测数据
    """
    np.random.seed(42)
    n = 500

    # 特征
    purchase_count = np.random.poisson(10, n)
    avg_spend = np.random.gamma(20, 10, n)
    days_since_last_purchase = np.random.exponential(30, n)
    registration_days = np.random.uniform(30, 365, n)

    # 流失概率（基于特征的简化模型）
    logit = -3 + 0.05 * purchase_count - 0.01 * avg_spend + 0.03 * days_since_last_purchase - 0.002 * registration_days
    prob = 1 / (1 + np.exp(-logit))
    is_churned = (np.random.random(n) < prob).astype(int)

    return pd.DataFrame({
        'purchase_count': purchase_count,
        'avg_spend': avg_spend,
        'days_since_last_purchase': days_since_last_purchase,
        'registration_days': registration_days,
        'is_churned': is_churned
    })


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
        'metric_tol': 0.01,  # 评估指标容差（1%）
        'prob_tol': 0.05,    # 概率容差（5%）
    }
