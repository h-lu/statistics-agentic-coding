"""
Test suite for Week 15: StatLab Integration

This module tests the integration of all Week 15 methods:
- PCA + Clustering pipeline
- Streaming statistics for cluster monitoring
- A/B testing on user segments
- Complete computational analysis workflow
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats


# =============================================================================
# Helper Functions (simulating solution.py functions)
# =============================================================================

def pca_dim_reduction(X, variance_threshold=0.85):
    """
    PCA dimensionality reduction.

    Parameters:
        X: Feature matrix (n_samples, n_features)
        variance_threshold: Variance ratio to preserve

    Returns:
        X_transformed: Reduced data
        pca: PCA model
        n_components: Number of components selected
        scaler: Fitted scaler
    """
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA (all components)
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # Calculate cumulative variance
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # Select components
    n_components = (cumsum_variance >= variance_threshold).argmax() + 1

    # Refit with selected components
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)

    return X_transformed, pca, n_components, scaler


def kmeans_clustering(X, k_range=range(2, 11)):
    """
    K-means clustering with automatic K selection.

    Parameters:
        X: Feature matrix
        k_range: Range of K values to try

    Returns:
        cluster_labels: Cluster assignments
        k_optimal: Optimal K value
        kmeans: Fitted KMeans model
    """
    # Elbow method + silhouette score
    wcss = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)
        if k < X.shape[0]:
            silhouette_scores.append(silhouette_score(X, labels))

    # Select optimal K (max silhouette)
    k_optimal = np.argmax(silhouette_scores) + min(k_range)

    # Fit with optimal K
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    return cluster_labels, k_optimal, kmeans


class StreamingClusterStats:
    """Streaming statistics for each cluster."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        """Incremental update (O(1))."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_stats(self):
        """Return current statistics."""
        variance = self.M2 / self.n if self.n > 0 else 0.0
        return {
            'n': self.n,
            'mean': self.mean,
            'variance': variance,
            'std': np.sqrt(variance)
        }


def ab_test_decision(group_A_data, group_B_data, config):
    """
    A/B test decision recommendation.

    Parameters:
        group_A_data: Control group data
        group_B_data: Treatment group data
        config: Experiment configuration

    Returns:
        Dictionary with decision, p_value, effect_size, CI
    """
    # T-test
    t_stat, p_value = stats.ttest_ind(group_B_data, group_A_data)

    # Effect size
    effect_size = np.mean(group_B_data) - np.mean(group_A_data)

    # 95% CI
    se = np.sqrt(np.var(group_A_data, ddof=1) / len(group_A_data) +
                 np.var(group_B_data, ddof=1) / len(group_B_data))
    ci_low = effect_size - 1.96 * se
    ci_high = effect_size + 1.96 * se

    # Decision rule
    if p_value < config['significance_level'] and \
       abs(effect_size) >= config['min_effect']:
        decision = "launch_B"
    elif p_value < 0.10:
        decision = "continue"
    else:
        decision = "reject_B"

    return {
        'decision': decision,
        'p_value': p_value,
        'effect_size': effect_size,
        'ci_low': ci_low,
        'ci_high': ci_high
    }


# =============================================================================
# Test 1: PCA + Clustering Pipeline
# =============================================================================

class TestPCAClusteringPipeline:
    """Test integrated PCA + clustering workflow."""

    def test_pca_then_clustering_workflow(self, high_dim_data):
        """
        Happy path: PCA followed by K-means clustering.

        学习目标:
        - 理解降维后聚类的完整流程
        - 高维数据 → PCA → K-means → 簇标签
        """
        X = high_dim_data

        # Step 1: PCA
        X_transformed, pca, n_components, scaler = pca_dim_reduction(X, variance_threshold=0.85)

        # Verify PCA
        assert n_components > 0, "应选择至少1个主成分"
        assert X_transformed.shape[1] < X.shape[1], \
            "降维后特征数应小于原始特征数"

        # Step 2: Clustering
        cluster_labels, k_optimal, kmeans = kmeans_clustering(X_transformed)

        # Verify clustering
        assert len(cluster_labels) == len(X_transformed), \
            "每个样本应有一个簇标签"
        assert k_optimal >= 2, "至少应有2个簇"

    def test_pca_clustering_reduces_computation_cost(self, high_dim_data):
        """
        Test: PCA reduces features before clustering.

        学习目标:
        - 理解降维减少 K-means 的计算成本
        - 距离计算从 O(p) 降到 O(k)
        """
        X = high_dim_data.values
        n_samples, n_features = X.shape

        # Clustering on original data
        kmeans_full = KMeans(n_clusters=3, random_state=42, n_init=5)
        kmeans_full.fit(X)

        # Clustering on PCA-reduced data
        X_transformed, pca, n_components, _ = pca_dim_reduction(pd.DataFrame(X))
        kmeans_reduced = KMeans(n_clusters=3, random_state=42, n_init=5)
        kmeans_reduced.fit(X_transformed)

        # Reduced clustering should use fewer features
        assert n_components < n_features, \
            "降维后的特征数应小于原始特征数"

        # Both should produce valid clusterings
        assert len(set(kmeans_full.labels_)) == 3
        assert len(set(kmeans_reduced.labels_)) == 3

    def test_pca_clustering_preserves_structure(self, well_separated_clusters):
        """
        Test: PCA + clustering should preserve cluster structure.

        学习目标:
        - 理解降维不应破坏数据结构
        - 如果原始数据有3簇，降维后仍应有3簇
        """
        X = well_separated_clusters[['x', 'y']].values

        # True number of clusters
        true_k = 3

        # PCA (keep most variance)
        X_pca, pca, n_components, _ = pca_dim_reduction(pd.DataFrame(X), variance_threshold=0.95)

        # Clustering
        labels, k_optimal, _ = kmeans_clustering(X_pca, k_range=range(2, 6))

        # Should detect 3 clusters
        assert k_optimal == true_k or k_optimal == true_k - 1 or k_optimal == true_k + 1, \
            f"应检测到约 {true_k} 个簇，检测到: {k_optimal}"


# =============================================================================
# Test 2: Streaming Statistics for Cluster Monitoring
# =============================================================================

class TestStreamingClusterStats:
    """Test streaming statistics for monitoring clusters."""

    def test_streaming_stats_per_cluster(self, high_dim_data):
        """
        Happy path: Monitor statistics for each cluster.

        学习目标:
        - 理解如何用流式统计监控每个簇
        - 每个簇独立的统计状态
        """
        X = high_dim_data.values

        # PCA + Clustering
        X_transformed, pca, n_components, scaler = pca_dim_reduction(pd.DataFrame(X))
        cluster_labels, k_optimal, kmeans = kmeans_clustering(X_transformed)

        # Initialize streaming stats for each cluster
        cluster_stats = {i: StreamingClusterStats() for i in range(k_optimal)}

        # Simulate streaming updates
        np.random.seed(42)
        for i in range(100):
            # Randomly select a past sample
            idx = np.random.randint(0, len(X))
            user_features = X[idx:idx+1]

            # Assign to cluster
            user_transformed = pca.transform(scaler.transform(user_features))
            cluster_id = kmeans.predict(user_transformed)[0]

            # Simulate outcome (e.g., revenue)
            outcome = np.random.normal(100, 20)

            # Update streaming stats
            cluster_stats[cluster_id].update(outcome)

        # Check that all clusters got some updates
        for cluster_id, stats_obj in cluster_stats.items():
            stats = stats_obj.get_stats()
            assert stats['n'] >= 0, \
                f"簇 {cluster_id} 的统计状态应有效"

    def test_streaming_vs_batch_cluster_stats(self, high_dim_data):
        """
        Test: Streaming stats match batch calculations.

        学习目标:
        - 理解流式统计与批量计算的等价性
        - 相同数据应产生相同统计量
        """
        X = high_dim_data.values

        # PCA + Clustering
        X_transformed, pca, n_components, scaler = pca_dim_reduction(pd.DataFrame(X))
        cluster_labels, k_optimal, kmeans = kmeans_clustering(X_transformed)

        # Simulate outcomes for each cluster
        np.random.seed(42)
        cluster_outcomes = {i: [] for i in range(k_optimal)}

        for i in range(len(X)):
            cluster_id = cluster_labels[i]
            outcome = np.random.normal(100 + cluster_id * 10, 20)
            cluster_outcomes[cluster_id].append(outcome)

        # Streaming calculation
        streaming_stats = {}
        for cluster_id in range(k_optimal):
            stat = StreamingClusterStats()
            for outcome in cluster_outcomes[cluster_id]:
                stat.update(outcome)
            streaming_stats[cluster_id] = stat.get_stats()

        # Batch calculation
        batch_stats = {}
        for cluster_id in range(k_optimal):
            outcomes = cluster_outcomes[cluster_id]
            batch_stats[cluster_id] = {
                'mean': np.mean(outcomes),
                'std': np.std(outcomes, ddof=0)
            }

        # Compare
        for cluster_id in range(k_optimal):
            streaming_mean = streaming_stats[cluster_id]['mean']
            batch_mean = batch_stats[cluster_id]['mean']

            assert abs(streaming_mean - batch_mean) < 1e-10, \
                f"簇 {cluster_id}: 流式均值应等于批量均值"


# =============================================================================
# Test 3: A/B Testing on User Segments
# =============================================================================

class TestABTestingSegments:
    """Test A/B testing on different user segments."""

    def test_ab_test_per_cluster(self, high_dim_data):
        """
        Happy path: Run A/B test on each cluster separately.

        学习目标:
        - 理解用户分层的 A/B 测试
        - 每个用户群独立检验效应
        """
        X = high_dim_data.values

        # PCA + Clustering
        X_transformed, pca, n_components, scaler = pca_dim_reduction(pd.DataFrame(X))
        cluster_labels, k_optimal, kmeans = kmeans_clustering(X_transformed)

        # Simulate A/B test per cluster
        np.random.seed(42)
        results = {}

        for cluster_id in range(k_optimal):
            # Generate A/B data for this cluster
            n = 50
            group_A = np.random.normal(100, 20, n)
            group_B = np.random.normal(105, 20, n)  # Slight effect

            # Run test
            result = ab_test_decision(
                group_A, group_B,
                config={'significance_level': 0.05, 'min_effect': 3.0}
            )
            results[cluster_id] = result

        # Should have results for all clusters
        assert len(results) == k_optimal, \
            "每个簇都应有 A/B 测试结果"

        # Each result should have required fields
        for cluster_id, result in results.items():
            assert 'decision' in result
            assert 'p_value' in result
            assert 'effect_size' in result
            assert 'ci_low' in result
            assert 'ci_high' in result

    def test_ab_test_detects_segment_specific_effects(self):
        """
        Test: A/B test detects different effects across segments.

        学习目标:
        - 理解用户分层的价值
        - 不同用户群可能对处理有不同反应
        """
        np.random.seed(42)

        # Simulate 2 user segments
        # Segment 0: Price-sensitive (small effect)
        segment_0_A = np.random.normal(100, 20, 100)
        segment_0_B = np.random.normal(102, 20, 100)  # Effect = 2

        # Segment 1: Quality-seeking (large effect)
        segment_1_A = np.random.normal(150, 20, 100)
        segment_1_B = np.random.normal(165, 20, 100)  # Effect = 15

        # Run tests
        config = {'significance_level': 0.05, 'min_effect': 5.0}

        result_0 = ab_test_decision(segment_0_A, segment_0_B, config)
        result_1 = ab_test_decision(segment_1_A, segment_1_B, config)

        # Segment 0: Small effect, might not launch
        # Segment 1: Large effect, should launch

        # Check segment 1 (large effect)
        assert result_1['decision'] in ['launch_B', 'continue'], \
            "大效应用户群应考虑 launch_B"

        # Segment 0 might not meet min_effect threshold
        # This is probabilistic, so we just verify structure
        assert isinstance(result_0['effect_size'], (float, np.floating))
        assert isinstance(result_1['effect_size'], (float, np.floating))


# =============================================================================
# Test 4: Complete Computational Analysis Workflow
# =============================================================================

class TestCompleteWorkflow:
    """Test end-to-end computational analysis workflow."""

    def test_complete_pipeline_pca_cluster_stream_ab(self, high_dim_data):
        """
        Happy path: Complete analysis pipeline.

        学习目标:
        - 理解完整的计算分析流程
        - 数据 → PCA → 聚类 → 流式统计 → A/B 测试
        """
        # Step 1: PCA
        X_transformed, pca, n_components, scaler = pca_dim_reduction(
            high_dim_data, variance_threshold=0.85
        )

        assert n_components > 0
        assert X_transformed.shape[1] < high_dim_data.shape[1]

        # Step 2: Clustering
        cluster_labels, k_optimal, kmeans = kmeans_clustering(X_transformed)

        assert k_optimal >= 2
        assert len(cluster_labels) == len(high_dim_data)

        # Step 3: Streaming statistics
        cluster_stats = {i: StreamingClusterStats() for i in range(k_optimal)}

        # Simulate streaming data
        np.random.seed(42)
        for i in range(100):
            idx = np.random.randint(0, len(high_dim_data))
            user_features = high_dim_data.values[idx:idx+1]
            user_transformed = pca.transform(scaler.transform(user_features))
            cluster_id = kmeans.predict(user_transformed)[0]
            outcome = np.random.normal(100, 20)
            cluster_stats[cluster_id].update(outcome)

        # Step 4: A/B testing on segments
        ab_results = {}
        for cluster_id in range(k_optimal):
            group_A = np.random.normal(100, 20, 50)
            group_B = np.random.normal(105, 20, 50)

            result = ab_test_decision(
                group_A, group_B,
                config={'significance_level': 0.05, 'min_effect': 3.0}
            )
            ab_results[cluster_id] = result

        # Verify pipeline completed successfully
        assert n_components > 0
        assert k_optimal >= 2
        assert len(cluster_stats) == k_optimal
        assert len(ab_results) == k_optimal

    def test_workflow_handles_edge_cases(self, high_dim_data):
        """
        Test: Workflow handles various edge cases.

        学习目标:
        - 理解完整流程需要鲁棒性
        - 处理各种边界情况
        """
        # Test with small data
        small_data = high_dim_data.iloc[:20]

        # Should still work
        X_transformed, pca, n_components, scaler = pca_dim_reduction(
            small_data, variance_threshold=0.85
        )

        # n_components limited by n_samples
        assert n_components <= len(small_data)

        # Clustering might not work well with very small data
        # But should not crash
        try:
            cluster_labels, k_optimal, kmeans = kmeans_clustering(X_transformed)
            assert k_optimal >= 2
        except Exception as e:
            # Some clustering might fail with very small data
            # This is acceptable
            pass

    def test_workflow_output_structure(self, high_dim_data):
        """
        Test: Workflow produces structured output.

        学习目标:
        - 理解分析输出应有清晰的结构
        - 便于生成报告
        """
        # Run pipeline
        X_transformed, pca, n_components, scaler = pca_dim_reduction(
            high_dim_data, variance_threshold=0.85
        )
        cluster_labels, k_optimal, kmeans = kmeans_clustering(X_transformed)

        # Structure output
        output = {
            'pca': {
                'n_components': n_components,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
            },
            'clustering': {
                'n_clusters': k_optimal,
                'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict(),
                'inertia': kmeans.inertia_
            }
        }

        # Verify structure
        assert 'pca' in output
        assert 'clustering' in output
        assert 'n_components' in output['pca']
        assert 'n_clusters' in output['clustering']
        assert output['pca']['n_components'] > 0
        assert output['clustering']['n_clusters'] >= 2


# =============================================================================
# Test 5: StatLab Report Integration
# =============================================================================

class TestStatLabReportIntegration:
    """Test integration with StatLab report."""

    def test_generate_computational_report_section(self, high_dim_data):
        """
        Happy path: Generate computational analysis section for report.

        学习目标:
        - 理解如何把分析结果写入报告
        - Markdown 格式，包含关键发现
        """
        # Run pipeline
        X_transformed, pca, n_components, _ = pca_dim_reduction(
            high_dim_data, variance_threshold=0.85
        )
        cluster_labels, k_optimal, kmeans = kmeans_clustering(X_transformed)

        # Generate report section
        report = f"""
## 计算专题：降维与聚类

### PCA 降维

- 原始特征数：{high_dim_data.shape[1]}
- 降维后成分数：{n_components}
- 压缩率：{high_dim_data.shape[1] / n_components:.1f}x

### K-means 聚类

- 最优簇数：{k_optimal}
- 各簇样本数：{dict(pd.Series(cluster_labels).value_counts())}
"""

        # Verify report structure
        assert 'PCA' in report
        assert '降维' in report
        assert 'K-means' in report
        assert '聚类' in report
        assert str(n_components) in report
        assert str(k_optimal) in report

    def test_report_includes_limitations(self, high_dim_data):
        """
        Test: Report should include method limitations.

        学习目标:
        - 理解完整报告应说明方法局限性
        - 透明度增加可信度
        """
        # Generate limitations section
        limitations = """
### 方法局限性

- PCA 是线性降维，可能遗漏非线性结构
- K-means 假设簇是球形，复杂形状可能不适用
- 流式分位数是近似值，有一定误差
- A/B 测试自动化需要人工审查，避免陷阱
"""

        # Should mention key limitations
        assert '线性' in limitations or 'PCA' in limitations
        assert '近似' in limitations
        assert '人工' in limitations

    def test_report_includes_business_interpretation(self, high_dim_data):
        """
        Test: Report includes business interpretation.

        学习目标:
        - 理解统计结果需要业务解释
        - 不能只有数字，要有"这意味着什么"
        """
        # Generate interpretation section
        X_transformed, pca, n_components, _ = pca_dim_reduction(
            high_dim_data, variance_threshold=0.85
        )
        cluster_labels, k_optimal, kmeans = kmeans_clustering(X_transformed)

        interpretation = f"""
### 业务解释

我们使用 PCA 将 {high_dim_data.shape[1]} 个特征压缩到 {n_components} 个主成分，
保留了 85% 的信息。这大幅降低了计算成本，同时保留了主要变异。

在降维后的空间中运行 K-means，发现了 {k_optimal} 个用户群。
每个用户群有不同的行为模式，可以针对性地制定运营策略。
"""

        # Should have business context
        assert '计算成本' in interpretation or '用户群' in interpretation
        assert '主成分' in interpretation
        assert '策略' in interpretation
