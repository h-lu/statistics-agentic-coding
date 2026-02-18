"""
Week 12 共享 Fixtures

提供测试用的共享数据和工具函数。

测试覆盖：
- 特征重要性计算
- SHAP 值计算（局部可解释性）
- 偏见检测（数据偏见 vs 算法偏见）
- 公平性指标（统计均等、机会均等、校准）
- StatLab 集成（可解释性报告生成）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sys
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 添加 starter_code 到 Python 路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 特征重要性测试 Fixtures
# =============================================================================

@pytest.fixture
def feature_importance_data():
    """
    标准特征重要性测试数据
    包含 6 个特征，其中前 3 个重要，后 3 个不重要
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
    feature_names = [f'feature_{i}' for i in range(6)]
    return {'X': X, 'y': y, 'feature_names': feature_names}


@pytest.fixture
def correlated_features_data():
    """
    包含相关特征的数据（用于测试特征重要性的"分票"现象）
    feature_0 和 feature_1 高度相关
    """
    np.random.seed(42)
    n_samples = 500

    # 创建基础特征
    feature_0 = np.random.randn(n_samples)
    # 创建高度相关的特征
    feature_1 = feature_0 + 0.1 * np.random.randn(n_samples)
    # 创建独立特征
    feature_2 = np.random.randn(n_samples)
    feature_3 = np.random.randn(n_samples)

    # 创建目标变量（主要依赖 feature_0）
    logit = 0.5 * feature_0 - 0.3 * feature_2 + 0.2 * feature_3
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.random(n_samples) < prob).astype(int)

    X = np.column_stack([feature_0, feature_1, feature_2, feature_3])
    feature_names = ['feature_0', 'feature_1', 'feature_2', 'feature_3']

    return {'X': X, 'y': y, 'feature_names': feature_names}


@pytest.fixture
def single_important_feature_data():
    """
    单一重要特征的数据
    用于测试特征重要性是否正确识别
    """
    np.random.seed(42)
    n_samples = 500

    # 只有第一个特征重要
    feature_0 = np.random.randn(n_samples)
    # 噪声特征
    feature_1 = np.random.randn(n_samples) * 0.01
    feature_2 = np.random.randn(n_samples) * 0.01

    logit = feature_0
    prob = 1 / (1 + np.exp(-logit))
    y = (np.random.random(n_samples) < prob).astype(int)

    X = np.column_stack([feature_0, feature_1, feature_2])
    feature_names = ['important_feature', 'noise_1', 'noise_2']

    return {'X': X, 'y': y, 'feature_names': feature_names}


# =============================================================================
# SHAP 值测试 Fixtures
# =============================================================================

@pytest.fixture
def shap_test_data():
    """
    SHAP 值测试数据
    包含训练好的模型和测试数据
    """
    np.random.seed(42)
    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_redundant=1,
        n_informative=3,
        random_state=42,
        n_clusters_per_class=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 训练随机森林
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=4,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    feature_names = [f'feature_{i}' for i in range(5)]

    return {
        'model': rf,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names
    }


@pytest.fixture
def shap_single_sample_data():
    """
    单样本 SHAP 解释数据
    用于测试局部可解释性
    """
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    rf = RandomForestClassifier(
        n_estimators=30,
        max_depth=3,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # 选择一个测试样本
    sample_idx = 0

    return {
        'model': rf,
        'X_test': X_test,
        'sample_idx': sample_idx,
        'feature_names': ['feature_a', 'feature_b', 'feature_c']
    }


# =============================================================================
# 偏见检测测试 Fixtures
# =============================================================================

@pytest.fixture
def bias_detection_data():
    """
    偏见检测数据
    包含敏感属性（性别、地区）和预测结果
    """
    np.random.seed(42)
    n = 500

    # 创建特征
    feature_1 = np.random.randn(n)
    feature_2 = np.random.randn(n)
    feature_3 = np.random.randn(n)

    # 创建敏感属性（性别：0=男性，1=女性）
    gender = np.random.randint(0, 2, n)

    # 创建目标变量（与性别相关，模拟偏见）
    # 男性（gender=0）流失率 20%，女性（gender=1）流失率 30%
    logit = -2 + 0.5 * feature_1 - 0.3 * feature_2 + 0.8 * gender
    prob = 1 / (1 + np.exp(-logit))
    is_churned = (np.random.random(n) < prob).astype(int)

    X = np.column_stack([feature_1, feature_2, feature_3, gender])
    feature_names = ['feature_1', 'feature_2', 'feature_3', 'gender']

    return {
        'X': X,
        'y': is_churned,
        'sensitive_attr': gender,
        'feature_names': feature_names
    }


@pytest.fixture
def biased_predictions_data():
    """
    包含偏见预测的数据
    用于测试偏见检测函数
    """
    np.random.seed(42)
    n = 200

    y_true = np.random.randint(0, 2, n)
    # 创建敏感属性
    sensitive_attr = np.array([0] * 100 + [1] * 100)

    # 创建有偏见的预测
    # 对 group 0：预测较保守
    y_pred = np.zeros(n, dtype=int)
    y_pred[:100] = (y_true[:100] & (np.random.random(100) > 0.3)).astype(int)  # 30% 假阴性
    y_pred[100:] = (y_true[100:] | (np.random.random(100) > 0.7)).astype(int)   # 30% 假阳性

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive_attr': sensitive_attr
    }


@pytest.fixture
def unbiased_predictions_data():
    """
    无偏见预测的数据
    用于验证公平性指标
    """
    np.random.seed(42)
    n = 200

    y_true = np.random.randint(0, 2, n)
    sensitive_attr = np.array([0] * 100 + [1] * 100)

    # 创建无偏见的预测（两组表现相同）
    base_noise = np.random.random(n)
    y_pred = ((y_true + base_noise) > 0.8).astype(int)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive_attr': sensitive_attr
    }


@pytest.fixture
def multiple_sensitive_groups_data():
    """
    多个敏感群体的数据
    用于测试多群体公平性评估
    """
    np.random.seed(42)
    n = 300

    y_true = np.random.randint(0, 2, n)
    # 三个群体：A、B、C
    sensitive_attr = np.array([0] * 100 + [1] * 100 + [2] * 100)

    # 不同群体有不同的预测表现
    y_pred = np.zeros(n, dtype=int)
    # Group A: 高假阳性
    y_pred[:100] = (np.random.random(100) > 0.5).astype(int)
    # Group B: 平衡
    y_pred[100:200] = ((y_true[100:200] + np.random.random(100) * 0.3) > 0.7).astype(int)
    # Group C: 高假阴性
    y_pred[200:] = (y_true[200:] & (np.random.random(100) > 0.4)).astype(int)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive_attr': sensitive_attr,
        'group_names': ['Group_A', 'Group_B', 'Group_C']
    }


# =============================================================================
# 公平性指标测试 Fixtures
# =============================================================================

@pytest.fixture
def demographic_parity_data():
    """
    统计均等测试数据
    两组的预测正率应该相同
    """
    np.random.seed(42)
    n = 200

    # 两组真实正率不同，但模型应该给它们相同的预测正率
    y_true = np.random.randint(0, 2, n)
    sensitive_attr = np.array([0] * 100 + [1] * 100)

    # 模型给两组相似的预测正率
    y_pred = np.random.randint(0, 2, n)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive_attr': sensitive_attr
    }


@pytest.fixture
def equalized_odds_data():
    """
    机会均等测试数据
    需要 TPR 和 FPR 在各群体间相等
    """
    np.random.seed(42)
    n = 400

    y_true = np.random.randint(0, 2, n)
    sensitive_attr = np.array([0] * 200 + [1] * 200)

    # 模拟相对公平的预测
    y_pred = np.zeros(n, dtype=int)
    mask_0 = sensitive_attr == 0
    mask_1 = sensitive_attr == 1

    # 两组有相似的 TPR 和 FPR
    y_pred[mask_0] = (y_true[mask_0] & (np.random.random(200) > 0.2)).astype(int)
    y_pred[mask_1] = (y_true[mask_1] & (np.random.random(200) > 0.2)).astype(int)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive_attr': sensitive_attr
    }


@pytest.fixture
def calibration_test_data():
    """
    校准测试数据
    预测概率应与真实概率匹配
    """
    np.random.seed(42)
    n = 300

    # 生成校准良好的预测概率
    y_true = np.random.randint(0, 2, n)
    y_prob = np.random.uniform(0.3, 0.7, n)
    # 调整使概率更接近真实值
    y_prob = y_true * 0.7 + (1 - y_true) * 0.3 + np.random.randn(n) * 0.1
    y_prob = np.clip(y_prob, 0, 1)

    sensitive_attr = np.array([0] * 150 + [1] * 150)

    return {
        'y_true': y_true,
        'y_prob': y_prob,
        'sensitive_attr': sensitive_attr
    }


# =============================================================================
# StatLab 可解释性报告测试 Fixtures
# =============================================================================

@pytest.fixture
def statlab_interpretability_data():
    """
    StatLab 项目可解释性模块测试数据
    客户流失预测场景
    """
    np.random.seed(42)
    n = 500

    # 数值型特征
    purchase_count = np.random.poisson(10, n)
    avg_spend = np.random.gamma(20, 10, n)
    days_since_last_purchase = np.random.exponential(30, n)
    membership_days = np.random.uniform(30, 365, n)

    # 敏感属性
    gender = np.random.randint(0, 2, n)  # 0=男性, 1=女性
    age_group = np.random.choice([0, 1, 2], n)  # 0=<25, 1=25-35, 2=35+

    # 流失概率（基于特征的简化模型，包含性别偏见）
    logit = (-3 +
             0.05 * purchase_count -
             0.01 * avg_spend +
             0.03 * days_since_last_purchase -
             0.002 * membership_days +
             0.3 * gender)  # 性别偏见
    prob = 1 / (1 + np.exp(-logit))
    is_churned = (np.random.random(n) < prob).astype(int)

    # 创建 DataFrame
    feature_cols = ['purchase_count', 'avg_spend', 'days_since_last_purchase',
                    'membership_days', 'gender', 'age_group']
    X = pd.DataFrame({
        'purchase_count': purchase_count,
        'avg_spend': avg_spend,
        'days_since_last_purchase': days_since_last_purchase,
        'membership_days': membership_days,
        'gender': gender,
        'age_group': age_group
    })

    y = is_churned

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 训练随机森林
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 生成预测
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    return {
        'model': rf,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'feature_names': feature_cols,
        'sensitive_attrs': {
            'gender': X_test['gender'],
            'age_group': pd.cut(X_test['age_group'], bins=[-1, 0, 1, 2],
                                labels=['<25', '25-35', '35+'])
        }
    }


# =============================================================================
# 边界情况 Fixtures
# =============================================================================

@pytest.fixture
def minimal_fairness_data():
    """
    最小公平性测试数据
    每组只有少量样本
    """
    n = 20
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0] * 2)
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 1] * 2)
    sensitive_attr = np.array([0] * 10 + [1] * 10)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive_attr': sensitive_attr
    }


@pytest.fixture
def single_group_data():
    """
    单一群体数据
    用于测试只有一个群体时的边界情况
    """
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    sensitive_attr = np.zeros(n)  # 所有样本属于同一群体

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive_attr': sensitive_attr
    }


@pytest.fixture
def empty_group_data():
    """
    包含空群体的数据
    某些群体在测试集中没有样本
    """
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    # 只有 0 和 1 组，没有 2 组
    sensitive_attr = np.random.randint(0, 2, n)

    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive_attr': sensitive_attr
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
        'metric_tol': 0.05,     # 评估指标容差（5%）
        'prob_tol': 0.05,       # 概率容差（5%）
        'importance_tol': 0.02, # 特征重要性容差
        'fairness_tol': 0.1,    # 公平性指标容差（10%）
        'shap_tol': 0.01,       # SHAP 值容差
    }


@pytest.fixture
def expected_fairness_thresholds():
    """
    预期的公平性阈值
    """
    return {
        'demographic_parity_diff_threshold': 0.1,  # 统计均等差异阈值
        'equalized_odds_diff_threshold': 0.15,     # 机会均等差异阈值
        'calibration_diff_threshold': 0.1,         # 校准差异阈值
        'min_group_size': 10,                       # 最小群体大小
    }
