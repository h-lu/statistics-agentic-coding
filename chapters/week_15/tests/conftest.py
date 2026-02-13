"""
Pytest fixtures for Week 15 tests.

Provides test data and fixtures for testing:
- PCA dimensionality reduction
- K-means clustering
- Streaming statistics
- A/B testing engineering
"""

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Fixtures for PCA Tests
# =============================================================================

@pytest.fixture
def simple_2d_data():
    """
    Simple 2D data with clear direction for PCA testing.

    Returns:
        DataFrame with 2 correlated features
    """
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = 0.5 * x1 + 0.1 * np.random.randn(n)

    df = pd.DataFrame({'feature1': x1, 'feature2': x2})
    return df


@pytest.fixture
def high_dim_data():
    """
    High-dimensional data (100 samples x 50 features) for PCA testing.

    Simulates user behavior data with correlated features.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 50

    # Create data with correlated structure
    # First 10 features are highly correlated
    base = np.random.randn(n_samples)
    X = np.zeros((n_samples, n_features))

    for i in range(10):
        X[:, i] = base + 0.2 * np.random.randn(n_samples)

    # Remaining features are independent noise
    for i in range(10, n_features):
        X[:, i] = np.random.randn(n_samples)

    feature_names = [f'feature_{i}' for i in range(n_features)]
    return pd.DataFrame(X, columns=feature_names)


@pytest.fixture
def data_for_variance_threshold():
    """
    Data with known variance structure for testing variance threshold selection.

    First 5 components explain ~85% of variance.
    """
    np.random.seed(42)
    n = 200

    # Create data with decaying variance
    X = np.zeros((n, 20))
    for i in range(20):
        variance = np.exp(-0.3 * i)  # Decaying variance
        X[:, i] = np.sqrt(variance) * np.random.randn(n)

    return pd.DataFrame(X)


# =============================================================================
# Fixtures for Clustering Tests
# =============================================================================

@pytest.fixture
def well_separated_clusters():
    """
    Data with 3 well-separated clusters for K-means testing.

    Returns:
        DataFrame with 3 clearly separated clusters
    """
    np.random.seed(42)
    n_per_cluster = 50

    # Cluster 1: centered at (0, 0)
    cluster1 = np.random.randn(n_per_cluster, 2) * 0.5

    # Cluster 2: centered at (5, 5)
    cluster2 = np.random.randn(n_per_cluster, 2) * 0.5 + np.array([5, 5])

    # Cluster 3: centered at (-5, 5)
    cluster3 = np.random.randn(n_per_cluster, 2) * 0.5 + np.array([-5, 5])

    X = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0]*n_per_cluster + [1]*n_per_cluster + [2]*n_per_cluster)

    df = pd.DataFrame(X, columns=['x', 'y'])
    df['true_label'] = true_labels

    return df


@pytest.fixture
def overlapping_clusters():
    """
    Data with overlapping clusters for testing edge cases.

    Returns harder clustering problem.
    """
    np.random.seed(42)
    n_per_cluster = 50

    # Closer cluster centers
    cluster1 = np.random.randn(n_per_cluster, 2) * 1.0 + np.array([0, 0])
    cluster2 = np.random.randn(n_per_cluster, 2) * 1.0 + np.array([2, 2])
    cluster3 = np.random.randn(n_per_cluster, 2) * 1.0 + np.array([-2, 2])

    X = np.vstack([cluster1, cluster2, cluster3])

    return pd.DataFrame(X, columns=['x', 'y'])


@pytest.fixture
def single_cluster_data():
    """
    Data with only one cluster for testing K=1 edge case.
    """
    np.random.seed(42)
    X = np.random.randn(100, 2) * 0.5
    return pd.DataFrame(X, columns=['x', 'y'])


# =============================================================================
# Fixtures for Streaming Statistics Tests
# =============================================================================

@pytest.fixture
def streaming_data():
    """
    Sequential data for testing streaming statistics.

    Returns:
        Array of 1000 random values
    """
    np.random.seed(42)
    return np.random.randn(1000)


@pytest.fixture
def streaming_data_with_drift():
    """
    Data with mean drift for testing streaming statistics adaptation.

    Mean shifts from 0 to 5 at index 500.
    """
    np.random.seed(42)
    data = np.concatenate([
        np.random.randn(500),
        np.random.randn(500) + 5
    ])
    return data


# =============================================================================
# Fixtures for A/B Testing Tests
# =============================================================================

@pytest.fixture
def ab_test_data_significant():
    """
    A/B test data where B is significantly better than A.

    A: mean=100, std=20
    B: mean=108, std=20 (effect size=8)
    """
    np.random.seed(42)
    n_per_group = 200

    group_A = np.random.normal(100, 20, n_per_group)
    group_B = np.random.normal(108, 20, n_per_group)

    return pd.DataFrame({
        'group': ['A'] * n_per_group + ['B'] * n_per_group,
        'value': np.concatenate([group_A, group_B])
    })


@pytest.fixture
def ab_test_data_no_effect():
    """
    A/B test data where there's no significant difference.

    Both groups: mean=100, std=20
    """
    np.random.seed(42)
    n_per_group = 200

    group_A = np.random.normal(100, 20, n_per_group)
    group_B = np.random.normal(100, 20, n_per_group)

    return pd.DataFrame({
        'group': ['A'] * n_per_group + ['B'] * n_per_group,
        'value': np.concatenate([group_A, group_B])
    })


@pytest.fixture
def ab_test_data_with_srm():
    """
    A/B test data with Sample Ratio Mismatch.

    Expected 50:50, but actual is 60:40.
    """
    np.random.seed(42)
    n_A = 300
    n_B = 200  # Imbalanced

    group_A = np.random.normal(100, 20, n_A)
    group_B = np.random.normal(100, 20, n_B)

    return pd.DataFrame({
        'group': ['A'] * n_A + ['B'] * n_B,
        'value': np.concatenate([group_A, group_B])
    })


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def empty_data():
    """Empty DataFrame for testing error handling."""
    return pd.DataFrame({'x': [], 'y': []})


@pytest.fixture
def single_observation():
    """Single observation for testing minimum sample edge case."""
    return pd.DataFrame({'x': [1.0], 'y': [2.0]})


@pytest.fixture
def constant_data():
    """Constant data (zero variance) for testing edge cases."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': [5.0] * 100,
        'feature2': [5.0] * 100
    })


@pytest.fixture
def high_dim_curse_data():
    """
    Data demonstrating curse of dimensionality: p >> n.

    50 samples with 100 features (p > n problem).
    """
    np.random.seed(42)
    n_samples = 50
    n_features = 100

    X = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i}' for i in range(n_features)]

    return pd.DataFrame(X, columns=feature_names)


# =============================================================================
# AI Review Fixtures
# =============================================================================

@pytest.fixture
def good_pca_report():
    """A good PCA analysis report for testing AI review."""
    return """
    PCA 分析结果：
    - 原始特征数：500
    - 降维后成分数：47
    - 压缩率：10.6x
    - 保留方差：85.2%

    主成分解释：
    - 第1主成分解释方差32.1%，主要反映"活跃度"特征
    - 第2主成分解释方差18.5%，主要反映"购买倾向"特征

    方差解释比例检查：
    - 前47个主成分累积解释85.2%的方差
    - 满足85%阈值要求
    """


@pytest.fixture
def bad_pca_report_no_variance_check():
    """A bad PCA report missing variance threshold check."""
    return """
    PCA 分析结果：
    - 原始特征数：500
    - 降维后成分数：10
    - 压缩率：50x

    主成分解释：
    - 第1主成分主要反映活跃度
    - 第2主成分主要反映购买倾向
    """


@pytest.fixture
def good_clustering_report():
    """A good clustering analysis report for testing AI review."""
    return """
    K-means 聚类结果：
    - 最优簇数：5（肘部法则 + 轮廓系数）
    - 轮廓系数：0.382
    - 各簇样本数：[250, 180, 320, 80, 170]

    簇解释：
    - 簇0："低活跃用户"——各项指标较低
    - 簇3："超级活跃用户"——高点击、高停留、高消费

    业务解释：
    - 聚类结果符合业务预期，每组都有清晰的画像
    - 轮廓系数表明聚类质量中等，可接受
    """


@pytest.fixture
def bad_clustering_report_no_k_selection():
    """A bad clustering report missing K value selection rationale."""
    return """
    K-means 聚类结果：
    - 簇数：5
    - 各簇样本数：[250, 180, 320, 80, 170]

    簇解释：
    - 簇0：低活跃用户
    - 簇1：中等活跃用户
    """


@pytest.fixture
def good_ab_test_report():
    """A good A/B test report for testing AI review."""
    return """
    A/B 测试结果（优惠券实验）：
    - A组均值：100.5元（n=500）
    - B组均值：108.3元（n=500）
    - 效应量：7.8元
    - 95% CI：[3.2, 12.4]
    - p值：0.0008

    决策建议：
    - p < 0.05 且效应量 > 5元 → 建议上线B版本

    假设检验检查：
    - SRM检测：p=0.42，样本比例正常
    - 样本量满足功效分析要求
    - 置信区间不包含0，结论可靠
    """


@pytest.fixture
def bad_ab_test_report_no_srm_check():
    """A bad A/B test report missing SRM check."""
    return """
    A/B 测试结果：
    - A组均值：100.5元
    - B组均值：108.3元
    - p值：0.0008

    决策：
    - p < 0.05，建议上线B版本
    """


@pytest.fixture
def bad_ab_test_report_causal_claim():
    """A bad A/B test report with causal language."""
    return """
    A/B 测试结果：
    - B组比A组高7.8元
    - p值：0.0008

    结论：
    - 优惠券会导致用户消费增加7.8元
    - B版本使用户提升了消费
    """
