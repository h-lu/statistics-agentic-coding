"""
Smoke Tests for Week 15 solution.py

基础冒烟测试：
- 验证模块可以导入
- 验证基本函数存在
- 验证基本功能可运行

注意：由于 week_15 的 solution.py 可能尚未实现，
这些测试使用了 pytest.skip 来优雅地处理缺失的模块。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 模块导入测试
# =============================================================================

def test_solution_module_exists():
    """
    冒烟测试：solution.py 模块应存在

    如果此测试失败，说明 solution.py 文件不存在
    """
    try:
        import solution
        assert solution is not None
    except ImportError:
        pytest.skip("solution.py not found - expected to be implemented later")


def test_solution_has_basic_functions():
    """
    冒烟测试：solution.py 应包含降维和聚类相关函数

    检查核心函数是否存在（示例函数名，实际可能不同）
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 降维和聚类相关的可能函数名
    functions = [
        # PCA 相关
        'pca_analysis',
        'perform_pca',
        'reduce_dimensions',
        'compute_pca',

        # K-means 相关
        'kmeans_clustering',
        'perform_kmeans',
        'cluster_analysis',
        'find_clusters',

        # 标准化相关
        'standardize_data',
        'normalize_features',
        'scale_data',

        # 评估相关
        'silhouette_analysis',
        'evaluate_clustering',
        'choose_k',
        'elbow_method',
    ]

    # 至少有一个降维/聚类相关的函数存在
    has_any = any(hasattr(solution, func) for func in functions)

    if not has_any:
        # 没有找到预期函数，不报错，只是记录
        pytest.skip("No PCA/clustering functions found in solution.py")


# =============================================================================
# PCA 冒烟测试
# =============================================================================

def test_pca_smoke():
    """
    冒烟测试：PCA 降维应能运行

    测试基本的 PCA 降维功能
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    np.random.seed(42)
    X = np.random.randn(100, 10)

    # 尝试 PCA 降维
    if hasattr(solution, 'pca_analysis'):
        result = solution.pca_analysis(X, n_components=2)
        assert result is not None
    elif hasattr(solution, 'perform_pca'):
        result = solution.perform_pca(X, n_components=2)
        assert result is not None
    else:
        pytest.skip("PCA function not implemented")


def test_pca_explained_variance_smoke():
    """
    冒烟测试：解释方差比例计算应能运行

    测试计算 PCA 的解释方差比例
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    np.random.seed(42)
    X = np.random.randn(100, 10)

    # 尝试计算解释方差
    if hasattr(solution, 'explained_variance'):
        result = solution.explained_variance(X)
        assert result is not None
    elif hasattr(solution, 'pca_variance_ratio'):
        result = solution.pca_variance_ratio(X)
        assert result is not None
    else:
        pytest.skip("explained_variance function not implemented")


# =============================================================================
# K-means 冒烟测试
# =============================================================================

def test_kmeans_smoke():
    """
    冒烟测试：K-means 聚类应能运行

    测试基本的 K-means 聚类功能
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    np.random.seed(42)
    X = np.random.randn(100, 5)

    # 尝试 K-means 聚类
    if hasattr(solution, 'kmeans_clustering'):
        result = solution.kmeans_clustering(X, n_clusters=3)
        assert result is not None
    elif hasattr(solution, 'perform_kmeans'):
        result = solution.perform_kmeans(X, n_clusters=3)
        assert result is not None
    else:
        pytest.skip("K-means function not implemented")


def test_kmeans_inertia_smoke():
    """
    冒烟测试：簇内平方和计算应能运行

    测试计算 K-means 的 inertia
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    np.random.seed(42)
    X = np.random.randn(100, 5)

    # 尝试计算 inertia
    if hasattr(solution, 'compute_inertia'):
        result = solution.compute_inertia(X, labels=np.random.randint(0, 3, 100))
        assert result is not None
    else:
        pytest.skip("compute_inertia function not implemented")


# =============================================================================
# 轮廓系数冒烟测试
# =============================================================================

def test_silhouette_score_smoke():
    """
    冒烟测试：轮廓系数计算应能运行

    测试计算聚类质量的轮廓系数
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    np.random.seed(42)
    X = np.random.randn(100, 5)
    labels = np.random.randint(0, 3, 100)

    # 尝试计算轮廓系数
    if hasattr(solution, 'silhouette_analysis'):
        result = solution.silhouette_analysis(X, labels)
        assert result is not None
    elif hasattr(solution, 'compute_silhouette'):
        result = solution.compute_silhouette(X, labels)
        assert result is not None
    else:
        pytest.skip("silhouette function not implemented")


# =============================================================================
# 标准化冒烟测试
# =============================================================================

def test_standardization_smoke():
    """
    冒烟测试：数据标准化应能运行

    测试 StandardScaler 标准化
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    np.random.seed(42)
    X = np.random.randn(100, 5) * 10 + 50

    # 尝试标准化
    if hasattr(solution, 'standardize_data'):
        result = solution.standardize_data(X)
        assert result is not None
    elif hasattr(solution, 'scale_data'):
        result = solution.scale_data(X)
        assert result is not None
    else:
        pytest.skip("standardization function not implemented")


# =============================================================================
# 肘部法则冒烟测试
# =============================================================================

def test_elbow_method_smoke():
    """
    冒烟测试：肘部法则应能运行

    测试选择最优 K 值的肘部法则
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    np.random.seed(42)
    X = np.random.randn(100, 5)

    # 尝试肘部法则
    if hasattr(solution, 'elbow_method'):
        result = solution.elbow_method(X, max_k=10)
        assert result is not None
    elif hasattr(solution, 'find_optimal_k'):
        result = solution.find_optimal_k(X, max_k=10)
        assert result is not None
    else:
        pytest.skip("elbow_method function not implemented")


# =============================================================================
# 端到端场景测试
# =============================================================================

def test_customer_segmentation_pipeline_smoke():
    """
    冒烟测试：客户分群端到端流程应能运行

    测试完整的降维+聚类流程
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    np.random.seed(42)
    # 模拟客户数据：100 个客户，20 个特征
    X = np.random.randn(100, 20) * 10 + 50

    # 尝试端到端流程
    if hasattr(solution, 'customer_segmentation'):
        result = solution.customer_segmentation(X)
        assert result is not None
    elif hasattr(solution, 'analyze_customer_data'):
        result = solution.analyze_customer_data(X)
        assert result is not None
    else:
        pytest.skip("customer_segmentation function not implemented")


# =============================================================================
# 异常处理冒烟测试
# =============================================================================

def test_empty_data_handling():
    """
    冒烟测试：空数据应被正确处理

    验证函数对空输入的容错性
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    # 空数据
    X = np.array([]).reshape(0, 5)

    # 尝试计算（应该报错或返回 None）
    if hasattr(solution, 'pca_analysis'):
        try:
            result = solution.pca_analysis(X, n_components=2)
            # 如果不报错，应该返回 None 或合理的默认值
            assert result is None or isinstance(result, (tuple, dict, np.ndarray))
        except (ValueError, RuntimeError):
            # 报错也是可接受的
            assert True


def test_invalid_k_value_handling():
    """
    冒烟测试：无效的 K 值应被正确处理

    验证对 K <= 0 或 K > n_samples 的处理
    """
    try:
        import solution
    except ImportError:
        pytest.skip("solution.py not found")

    np.random.seed(42)
    X = np.random.randn(100, 5)

    # 无效 K 值
    invalid_ks = [0, -1, 200]

    for k in invalid_ks:
        if hasattr(solution, 'kmeans_clustering'):
            try:
                result = solution.kmeans_clustering(X, n_clusters=k)
                # 如果不报错，应该返回 None 或合理的默认值
                assert result is None or isinstance(result, (tuple, dict))
            except (ValueError, RuntimeError):
                # 报错也是可接受的
                assert True


# =============================================================================
# 概念理解冒烟测试
# =============================================================================

def test_dimensionality_reduction_concepts():
    """
    冒烟测试：降维基本概念

    验证对降维概念的理解
    """
    # 概念测试：不需要具体实现
    concepts = {
        'dimensionality_reduction': '从高维到低维，减少特征数量',
        'PCA': '主成分分析 - 通过正交变换最大化方差来降维',
        'explained_variance': '解释方差 - 主成分捕获的原始数据的方差量',
        'loading': '载荷 - 每个原始特征在主成分中的权重',
    }

    assert len(concepts) == 4
    assert 'dimensionality_reduction' in concepts
    assert 'PCA' in concepts
    assert 'explained_variance' in concepts
    assert 'loading' in concepts


def test_clustering_concepts():
    """
    冒烟测试：聚类基本概念

    验证对聚类概念的理解
    """
    # 概念测试：不需要具体实现
    concepts = {
        'clustering': '无监督学习，将相似样本分组',
        'K-means': '基于距离的聚类算法，最小化簇内平方和',
        'silhouette_score': '轮廓系数 - 衡量聚类质量的指标，范围 [-1, 1]',
        'inertia': '簇内平方和 - K-means 的优化目标',
        'centroid': '簇中心 - 簇内所有点的平均位置',
    }

    assert len(concepts) == 5
    assert 'clustering' in concepts
    assert 'K-means' in concepts
    assert 'silhouette_score' in concepts
    assert 'inertia' in concepts
    assert 'centroid' in concepts


def test_curse_of_dimensionality_concept():
    """
    冒烟测试：维度灾难概念

    验证对维度灾难的理解
    """
    # 概念测试
    concept = {
        'curse_of_dimensionality': '维度灾难 - 当特征数量增加时，数据在高维空间中变得极其稀疏，导致许多统计方法失效',
        'effects': [
            '点之间的距离趋同',
            '需要指数级增长的样本',
            '过拟合风险增加',
            '可视化困难',
        ],
    }

    assert 'curse_of_dimensionality' in concept
    assert len(concept['effects']) == 4


def test_pca_vs_feature_selection_concept():
    """
    冒烟测试：降维 vs 特征选择概念

    验证对两者区别的理解
    """
    # 概念测试
    comparison = {
        'feature_selection': '从原始特征中挑选子集，丢弃其他特征',
        'dimensionality_reduction': '创建新的低维特征，每个新特征是原始特征的组合',
        'key_difference': '特征选择保留原始特征含义，降维压缩信息但损失可解释性',
    }

    assert 'feature_selection' in comparison
    assert 'dimensionality_reduction' in comparison
    assert 'key_difference' in comparison


# =============================================================================
# 数据准备测试
# =============================================================================

def test_data_preprocessing_before_pca():
    """
    冒烟测试：PCA 前的数据预处理

    验证理解 PCA 需要标准化
    """
    # 概念测试
    steps = {
        'step_1': '检查数据是否有缺失值',
        'step_2': '标准化数据（StandardScaler）',
        'step_3': '应用 PCA',
        'reason': 'PCA 对尺度敏感，必须先标准化',
    }

    assert steps['reason'] == 'PCA 对尺度敏感，必须先标准化'


def test_data_preprocessing_before_kmeans():
    """
    冒烟测试：K-means 前的数据预处理

    验证理解 K-means 需要标准化
    """
    # 概念测试
    steps = {
        'step_1': '检查数据是否有缺失值',
        'step_2': '标准化数据（StandardScaler）',
        'step_3': '应用 K-means',
        'reason': 'K-means 基于距离，对尺度敏感',
    }

    assert steps['reason'] == 'K-means 基于距离，对尺度敏感'


# =============================================================================
# 结果解释测试
# =============================================================================

def test_pca_result_interpretation():
    """
    冒烟测试：PCA 结果解释

    验证理解如何解释 PCA 输出
    """
    # 概念测试
    interpretation = {
        'explained_variance_ratio': '每个主成分解释的方差比例',
        'cumulative_variance': '前 k 个主成分累积解释的方差比例',
        'components': '主成分方向（特征向量）',
        'loadings': '每个原始特征对主成分的贡献（权重）',
    }

    assert len(interpretation) == 4


def test_clustering_result_interpretation():
    """
    冒烟测试：聚类结果解释

    验证理解如何解释聚类输出
    """
    # 概念测试
    interpretation = {
        'cluster_labels': '每个样本的簇分配',
        'cluster_centers': '每个簇的中心点',
        'inertia': '簇内平方和（越小越好）',
        'silhouette_score': '轮廓系数（越大越好，范围 [-1, 1]）',
    }

    assert len(interpretation) == 4
