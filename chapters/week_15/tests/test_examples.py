"""
Week 15 示例代码测试

运行方式：
    pytest chapters/week_15/tests/ -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys


# 添加 examples 目录到路径
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir))


def test_pca_demo():
    """测试 PCA 降维示例"""
    # 导入
    from importlib import import_module
    pca_demo = import_module("01_pca_demo")

    # 生成数据
    X = pca_demo.generate_user_behavior_matrix(n_samples=100, n_features=50, random_seed=42)

    # 测试数据形状
    assert X.shape == (100, 50), f"Expected (100, 50), got {X.shape}"

    # 测试 PCA 降维
    X_transformed, pca, scaler, n_components = pca_demo.pca_dim_reduction(
        X, variance_threshold=0.85, plot_dir=None
    )

    # 验证降维后的形状
    assert n_components < X.shape[1], "n_components should be less than original features"
    assert X_transformed.shape[1] == n_components, f"Expected {n_components} components"

    # 验证方差保留
    assert sum(pca.explained_variance_ratio_) >= 0.85, "Should retain at least 85% variance"

    print(f"✅ test_pca_demo passed: {X.shape} -> {X_transformed.shape}")


def test_clustering_demo():
    """测试 K-means 聚类示例"""
    from importlib import import_module
    clustering_demo = import_module("02_clustering_demo")

    # 生成数据
    X, y_true = clustering_demo.generate_synthetic_clusters(
        n_samples=500, n_centers=5, n_features=10, random_seed=42
    )

    # 测试数据形状
    assert X.shape == (500, 10), f"Expected (500, 10), got {X.shape}"

    # 测试最优 K 值选择
    k_results = clustering_demo.find_optimal_k(X, k_range=range(2, 8), plot_dir=None)

    # 验证返回结果
    assert 'k_silhouette' in k_results, "Should return k_silhouette"
    assert 'k_elbow' in k_results, "Should return k_elbow"
    assert 2 <= k_results['k_silhouette'] <= 7, "k_silhouette should be in range"

    # 测试聚类
    k_final = k_results['k_silhouette']
    labels, kmeans, silhouette_avg = clustering_demo.perform_kmeans_clustering(
        X, n_clusters=k_final, plot_dir=None
    )

    # 验证聚类结果
    assert len(labels) == len(X), "Labels should match data length"
    assert -1 <= silhouette_avg <= 1, "Silhouette score should be between -1 and 1"
    assert len(np.unique(labels)) == k_final, f"Should have {k_final} clusters"

    print(f"✅ test_clustering_demo passed: K={k_final}, silhouette={silhouette_avg:.3f}")


def test_streaming_demo():
    """测试流式统计算法"""
    from importlib import import_module
    streaming_demo = import_module("03_streaming_demo")

    # 生成测试数据
    np.random.seed(42)
    data = np.random.randn(1000)

    # 测试 OnlineMean
    online_mean = streaming_demo.OnlineMean()
    for x in data:
        online_mean.update(x)

    batch_mean = np.mean(data)
    mean_error = abs(online_mean.mean() - batch_mean)

    assert mean_error < 1e-10, f"OnlineMean error too large: {mean_error}"

    # 测试 OnlineVariance
    online_var = streaming_demo.OnlineVariance()
    for x in data:
        online_var.update(x)

    batch_var = np.var(data, ddof=0)
    var_error = abs(online_var.variance() - batch_var)

    assert var_error < 1e-10, f"OnlineVariance error too large: {var_error}"

    # 测试 OnlineQuantile（近似算法，允许较大误差）
    online_quantile = streaming_demo.OnlineQuantile(num_bins=100)
    for x in data:
        online_quantile.update(x)

    batch_median = np.median(data)
    median_error = abs(online_quantile.quantile(0.5) - batch_median)

    # 在线分位数是近似算法，允许 5% 的误差
    assert median_error < abs(batch_median) * 0.05 + 0.1, \
        f"OnlineQuantile error too large: {median_error}"

    print(f"✅ test_streaming_demo passed: mean_err={mean_error:.2e}, "
          f"var_err={var_error:.2e}, median_err={median_error:.4f}")


def test_ab_test_demo():
    """测试 A/B 测试平台"""
    from importlib import import_module
    ab_test_demo = import_module("04_ab_test_demo")

    # 创建配置
    config = ab_test_demo.ExperimentConfig(
        name="测试实验",
        treatment_groups=["A", "B"],
        metric="avg_revenue",
        sample_ratio={"A": 0.5, "B": 0.5},
        min_sample_size=100,
        significance_level=0.05,
        min_effect_size=5.0
    )

    # 创建平台
    platform = ab_test_demo.ABTestPlatform(config)

    # 添加数据（B 组有真实效应）
    np.random.seed(42)
    for _ in range(200):
        platform.add_observation("A", np.random.normal(100, 20))
        platform.add_observation("B", np.random.normal(108, 20))

    # 分析
    result = platform.analyze()

    # 验证结果
    assert result.sample_sizes['A'] == 200, "A 组应有 200 个样本"
    assert result.sample_sizes['B'] == 200, "B 组应有 200 个样本"
    assert 'p_value' in result.__dict__, "应包含 p_value"
    assert 'decision' in result.__dict__, "应包含 decision"

    # 验证 SRM 检测
    srm_detected = platform.check_sample_ratio_mismatch()
    # numpy.bool_ is not exactly bool but behaves like it
    assert srm_detected in [True, False], "SRM 检测应返回布尔值"

    print(f"✅ test_ab_test_demo passed: decision={result.decision}, "
          f"p={result.p_value:.4f}")


def test_statlab_computational():
    """测试 StatLab 计算专题版本"""
    from importlib import import_module
    statlab = import_module("05_statlab_computational")

    # 生成数据
    df = statlab.generate_user_behavior_data(n_samples=500, random_seed=42)

    # 验证数据
    assert len(df) == 500, "应有 500 个样本"
    assert '消费金额' in df.columns, "应包含消费金额列"

    # 测试 PCA 降维
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    X = df[feature_cols]

    X_transformed, pca, n_components, scaler = statlab.pca_dim_reduction(
        X, variance_threshold=0.85
    )

    assert n_components > 0, "应至少有 1 个主成分"
    assert n_components < len(feature_cols), "主成分数应小于原始特征数"

    # 测试聚类
    cluster_labels, k_optimal, kmeans = statlab.kmeans_clustering(
        X_transformed, k_range=range(2, 6)
    )

    assert len(cluster_labels) == len(X), "聚类标签数应与样本数匹配"
    assert 2 <= k_optimal <= 5, f"最优 K 值应在 [2, 5] 范围内，实际为 {k_optimal}"

    # 测试流式统计
    cluster_stats = {i: statlab.StreamingClusterStats() for i in range(k_optimal)}

    # 模拟增量更新
    for i in range(50):
        user_features = X.iloc[i % len(X)].values
        user_transformed = scaler.transform([user_features])[0]
        user_pca = pca.transform([user_transformed])[0]
        user_cluster = kmeans.predict([user_pca])[0]
        user_spending = df['消费金额'].iloc[i % len(df)]

        cluster_stats[user_cluster].update(user_spending)

    # 验证流式统计
    for cluster_id, stats_obj in cluster_stats.items():
        stats_dict = stats_obj.get_stats()
        assert 'n' in stats_dict, "应包含样本数"
        assert 'mean' in stats_dict, "应包含均值"
        assert 'std' in stats_dict, "应包含标准差"

    print(f"✅ test_statlab_computational passed: K={k_optimal}, "
          f"n_components={n_components}")


def test_solution_code():
    """测试 starter_code/solution.py"""
    from importlib import import_module

    # 导入 solution
    solution_path = Path(__file__).parent.parent / "starter_code" / "solution.py"
    if not solution_path.exists():
        pytest.skip("solution.py not found")

    # 这里不直接运行 main()，而是测试各个类
    spec = __import__('importlib.util').util.spec_from_file_location(
        "solution", solution_path
    )
    solution = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(solution)

    # 测试 OnlineMean
    online_mean = solution.OnlineMean()
    online_mean.update(5.0)
    online_mean.update(10.0)

    assert online_mean.mean() == 7.5, "OnlineMean 应计算正确的均值"

    # 测试 OnlineVariance
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    online_var = solution.OnlineVariance()

    for x in data:
        online_var.update(x)

    expected_var = np.var(data, ddof=0)
    assert abs(online_var.variance() - expected_var) < 1e-10, \
        f"OnlineVariance 应计算正确的方差，期望 {expected_var}，得到 {online_var.variance()}"

    print("✅ test_solution_code passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
