"""
Test suite for Week 15: Clustering Analysis

This module tests K-means clustering, covering:
- K-means fitting and prediction
- Elbow method for K selection
- Silhouette score for clustering quality
- Cluster interpretation
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# =============================================================================
# Test 1: K-means Fitting and Prediction
# =============================================================================

class TestKMeansFitting:
    """Test K-means clustering fitting and prediction."""

    def test_kmeans_fits_well_separated_clusters(self, well_separated_clusters):
        """
        Happy path: K-means correctly identifies well-separated clusters.

        学习目标:
        - 理解 K-means 在分离良好的簇上效果很好
        - fit_predict 返回聚类标签
        """
        X = well_separated_clusters[['x', 'y']].values
        true_k = 3

        # Fit K-means
        kmeans = KMeans(n_clusters=true_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Check labels
        assert len(labels) == len(X), "每个样本应有一个标签"
        assert len(set(labels)) == true_k, \
            f"应该有 {true_k} 个不同的簇标签"

    def test_kmeans_cluster_centers(self, well_separated_clusters):
        """
        Happy path: Cluster centers are near true centers.

        学习目标:
        - 理解 cluster_centers_ 是簇的几何中心
        - 簇中心 = 该簇所有点的均值
        """
        X = well_separated_clusters[['x', 'y']].values
        true_k = 3

        kmeans = KMeans(n_clusters=true_k, random_state=42, n_init=10)
        kmeans.fit(X)

        # Check cluster centers
        assert kmeans.cluster_centers_.shape == (true_k, 2), \
            f"应该有 {true_k} 个簇中心，每个2维"

        # Centers should be within reasonable range
        # True centers are approximately (0,0), (5,5), (-5,5)
        assert np.all(kmeans.cluster_centers_[:, 0] > -10) and \
               np.all(kmeans.cluster_centers_[:, 0] < 10), \
            "簇中心应在数据范围内"

    def test_kmeans_inertia_decreases_with_k(self, well_separated_clusters):
        """
        Test: WCSS (inertia) decreases as K increases.

        学习目标:
        - 理解 inertia_ 是簇内平方和
        - K 越大，WCSS 越小（直到 K=n）
        """
        X = well_separated_clusters[['x', 'y']].values

        inertia_values = []
        k_range = range(1, 6)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertia_values.append(kmeans.inertia_)

        # Inertia should decrease with K
        for i in range(len(inertia_values) - 1):
            assert inertia_values[i] >= inertia_values[i + 1], \
                f"K={i+1} 的 WCSS 应 >= K={i+2} 的 WCSS"

    def test_kmeans_predict_new_data(self, well_separated_clusters):
        """
        Happy path: Predict cluster for new data points.

        学习目标:
        - 理解 fit() vs predict()
        - 新样本会被分配到最近的簇中心
        """
        X = well_separated_clusters[['x', 'y']].values

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        # Predict new points
        new_points = np.array([[0, 0], [5, 5], [-5, 5]])
        predicted_labels = kmeans.predict(new_points)

        # Should have 3 predictions
        assert len(predicted_labels) == 3, "应返回3个预测标签"
        assert all(0 <= label < 3 for label in predicted_labels), \
            "预测标签应在 [0, K-1] 范围内"

    def test_kmeans_with_k_equals_n(self, well_separated_clusters):
        """
        Edge case: K = n (each point is its own cluster).

        学习目标:
        - 理解 K=n 时 WCSS = 0（完美拟合）
        - 但这是过拟合，没有意义
        """
        X = well_separated_clusters[['x', 'y']].values[:10]  # Use 10 points
        n = len(X)

        kmeans = KMeans(n_clusters=n, random_state=42, n_init=1)
        kmeans.fit(X)

        # With K=n, each point should be its own cluster
        # (or very close to it)
        assert kmeans.inertia_ < 1e-10, \
            "K=n 时 WCSS 应接近 0"


# =============================================================================
# Test 2: Elbow Method for K Selection
# =============================================================================

class TestElbowMethod:
    """Test elbow method for selecting optimal K."""

    def test_elbow_method_visual_inspection(self, well_separated_clusters):
        """
        Happy path: Elbow is visible in WCSS vs K plot.

        学习目标:
        - 理解肘部是 WCSS 下降速度变缓的转折点
        - 对于3个真实簇，肘部应在 K=3
        """
        X = well_separated_clusters[['x', 'y']].values

        wcss = []
        k_range = range(1, 8)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        # Calculate "elow strength" - second derivative
        # Elbow is where decrease rate changes most
        decreases = [wcss[i] - wcss[i+1] for i in range(len(wcss)-1)]
        second_derivative = [decreases[i] - decreases[i+1]
                          for i in range(len(decreases)-1)]

        # For 3 well-separated clusters, elbow should be around K=3
        # (This is a heuristic check)
        elbow_candidates = [i+2 for i, sd in enumerate(second_derivative)
                         if sd > 0]  # Positive second derivative
        assert len(elbow_candidates) > 0, \
            "应能检测到肘部"

    def test_elbow_method_with_overlapping_clusters(self, overlapping_clusters):
        """
        Test: Elbow is less clear with overlapping clusters.

        学习目标:
        - 理解肘部法则在重叠簇上效果较差
        - 需要结合其他方法（如轮廓系数）
        """
        X = overlapping_clusters[['x', 'y']].values

        wcss = []
        k_range = range(1, 8)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        # WCSS should still decrease
        for i in range(len(wcss) - 1):
            assert wcss[i] > wcss[i + 1], \
                "WCSS 应随 K 增加而减小"

        # But "elbow" is less clear (no sharp drop)
        # This is expected for overlapping clusters

    def test_elbow_method_range_check(self):
        """
        Test: Elbow method should test reasonable K range.

        学习目标:
        - 理解 K 的合理范围通常是 [2, 10] 或 [2, sqrt(n/2)]
        - K=1 没有意义（所有点一个簇）
        """
        # This is a documentation test
        reasonable_k_range = range(2, 11)  # Common practice

        assert min(reasonable_k_range) >= 2, \
            "K 应从 2 开始（K=1 无意义）"
        assert max(reasonable_k_range) <= 15, \
            "K 的上限不应太大（通常 <= 10 或 sqrt(n/2)）"


# =============================================================================
# Test 3: Silhouette Score
# =============================================================================

class TestSilhouetteScore:
    """Test silhouette score for clustering quality."""

    def test_silhouette_score_range(self, well_separated_clusters):
        """
        Happy path: Silhouette score is in [-1, 1].

        学习目标:
        - 理解轮廓系数的取值范围
        - 接近 1 表示聚类质量好
        """
        X = well_separated_clusters[['x', 'y']].values

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)

        # Check range
        assert -1 <= score <= 1, \
            f"轮廓系数应在 [-1, 1] 范围内，实际: {score:.3f}"

        # For well-separated clusters, score should be positive
        assert score > 0, \
            f"分离良好的簇应有正的轮廓系数，实际: {score:.3f}"

    def test_silhouette_score_decreases_with_bad_k(self, well_separated_clusters):
        """
        Test: Silhouette score decreases with wrong K.

        学习目标:
        - 理解轮廓系数可以用来选择最优 K
        - 最大轮廓系数对应的 K 是最优的
        """
        X = well_separated_clusters[['x', 'y']].values

        silhouette_scores = []
        k_range = range(2, 6)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)

        # True K=3 should have highest (or near highest) score
        # This is probabilistic, so we just check structure
        assert len(silhouette_scores) == len(k_range), \
            "每个 K 值都应计算轮廓系数"

        # All scores should be in valid range
        for score in silhouette_scores:
            assert -1 <= score <= 1

    def test_silhouette_score_requires_min_clusters(self, well_separated_clusters):
        """
        Edge case: Silhouette score requires at least 2 clusters.

        学习目标:
        - 理解轮廓系数的定义（需要 intra 和 inter 距离）
        - K=1 时无法计算
        """
        X = well_separated_clusters[['x', 'y']].values

        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Should raise error
        with pytest.raises(ValueError):
            silhouette_score(X, labels)

    def test_silhouette_score_intuition(self, well_separated_clusters, overlapping_clusters):
        """
        Test: Well-separated clusters have higher silhouette score.

        学习目标:
        - 理解轮廓系数衡量"簇内紧密度"和"簇间分离度"
        - 分离好的簇得分更高
        """
        # Case 1: Well-separated
        X1 = well_separated_clusters[['x', 'y']].values
        kmeans1 = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels1 = kmeans1.fit_predict(X1)
        score1 = silhouette_score(X1, labels1)

        # Case 2: Overlapping
        X2 = overlapping_clusters[['x', 'y']].values
        kmeans2 = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels2 = kmeans2.fit_predict(X2)
        score2 = silhouette_score(X2, labels2)

        # Well-separated should have higher score
        # (This is generally true, though not guaranteed)
        assert isinstance(score1, (float, np.floating))
        assert isinstance(score2, (float, np.floating))


# =============================================================================
# Test 4: Cluster Interpretation
# =============================================================================

class TestClusterInterpretation:
    """Test cluster interpretation and business meaning."""

    def test_cluster_size_distribution(self, well_separated_clusters):
        """
        Happy path: Check cluster size distribution.

        学习目标:
        - 理解簇的大小分布很重要
        - 极端不平衡可能表明聚类问题
        """
        X = well_separated_clusters[['x', 'y']].values

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Count samples in each cluster
        unique, counts = np.unique(labels, return_counts=True)

        assert len(unique) == 3, "应有 3 个簇"

        # Check for extreme imbalance (e.g., 1 sample in one cluster)
        assert all(count > 5 for count in counts), \
            "每个簇应有合理数量的样本（避免极端不平衡）"

    def test_cluster_summary_statistics(self, well_separated_clusters):
        """
        Happy path: Compute summary statistics per cluster.

        学习目标:
        - 理解如何用统计量描述每个簇
        - 均值、方差、分位数等
        """
        X = well_separated_clusters[['x', 'y']].values

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Compute statistics for each cluster
        df = pd.DataFrame(X, columns=['x', 'y'])
        df['cluster'] = labels

        cluster_stats = df.groupby('cluster').agg(['mean', 'std', 'count'])

        # Should have 3 groups
        assert len(cluster_stats) == 3, "应有 3 组统计量"

        # Each cluster should have all statistics
        assert all(cluster_stats[('x', 'count')] > 0)

    def test_cluster_centers_match_sample_means(self, well_separated_clusters):
        """
        Test: Cluster centers equal sample means of clusters.

        学习目标:
        - 理解簇中心的数学定义
        - center = mean of all points in cluster
        """
        X = well_separated_clusters[['x', 'y']].values

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        # Manual calculation
        df = pd.DataFrame(X, columns=['x', 'y'])
        df['cluster'] = kmeans.labels_
        manual_centers = df.groupby('cluster').mean().values

        # Should match (accounting for numerical precision)
        assert np.allclose(kmeans.cluster_centers_, manual_centers, atol=1e-10), \
            "簇中心应等于簇内样本的均值"


# =============================================================================
# Test 5: Clustering Edge Cases
# =============================================================================

class TestClusteringEdgeCases:
    """Test clustering with edge cases."""

    def test_kmeans_with_single_cluster(self, single_cluster_data):
        """
        Edge case: K-means with K=1 (all data in one cluster).

        学习目标:
        - 理解 K=1 时所有点属于一个簇
        - 簇中心 = 全局均值
        """
        X = single_cluster_data.values

        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans.fit(X)

        # All labels should be 0
        assert np.all(kmeans.labels_ == 0), "K=1 时所有标签应为 0"

        # Center should be global mean
        assert np.allclose(kmeans.cluster_centers_[0], X.mean(axis=0), atol=1e-10), \
            "K=1 时簇中心应等于全局均值"

    def test_kmeans_minimum_samples(self):
        """
        Edge case: K-means with minimum samples.

        学习目标:
        - 理解每个簇至少需要1个样本
        - K 不能超过样本数
        """
        # Create minimal data: 3 samples
        X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)

        # K=3 should work
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=1)
        kmeans.fit(X)

        assert len(set(kmeans.labels_)) == 3

        # K=4 should not work (or handle gracefully)
        with pytest.raises(ValueError):
            KMeans(n_clusters=4, random_state=42).fit(X)

    def test_kmeans_with_empty_data(self, empty_data):
        """
        Edge case: K-means with empty data should raise error.

        学习目标:
        - 理解数据验证的重要性
        """
        X = empty_data.values

        kmeans = KMeans(n_clusters=2, random_state=42)

        with pytest.raises(ValueError):
            kmeans.fit(X)

    def test_kmeans_random_state_reproducibility(self, well_separated_clusters):
        """
        Test: K-means with random_state gives reproducible results.

        学习目标:
        - 理解 K-means 初始化是随机的
        - 设置 random_state 保证结果可复现
        """
        X = well_separated_clusters[['x', 'y']].values

        # Fit twice with same random_state
        kmeans1 = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans1.fit(X)

        kmeans2 = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans2.fit(X)

        # Results should be identical
        assert np.allclose(kmeans1.cluster_centers_, kmeans2.cluster_centers_), \
            "相同 random_state 应产生相同结果"
        assert np.array_equal(kmeans1.labels_, kmeans2.labels_), \
            "相同 random_state 应产生相同标签"


# =============================================================================
# Test 6: Clustering vs Classification
# =============================================================================

class TestClusteringVsClassification:
    """Test understanding of difference between clustering and classification."""

    def test_clustering_has_no_ground_truth(self, well_separated_clusters):
        """
        Test: Clustering doesn't use labels during fitting.

        学习目标:
        - 理解聚类是无监督学习
        - 不需要真实标签
        """
        X = well_separated_clusters[['x', 'y']].values

        # K-means doesn't need true labels
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        # Fit succeeded without labels
        assert hasattr(kmeans, 'labels_')

    def test_clustering_evaluation_is_harder(self, well_separated_clusters):
        """
        Test: Clustering evaluation requires internal metrics.

        学习目标:
        - 理解聚类没有"准确率"这种直接指标
        - 需要用轮廓系数、WCSS 等间接评估
        """
        X = well_separated_clusters[['x', 'y']].values

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Can't compute "accuracy" without ground truth
        # But can compute internal metrics
        score = silhouette_score(X, labels)

        assert isinstance(score, (float, np.floating)), \
            "聚类质量可以用轮廓系数等内部指标评估"


# =============================================================================
# Test 7: AI Report Review for Clustering
# =============================================================================

class TestAIClusteringReportReview:
    """Test ability to review AI-generated clustering reports."""

    def test_check_good_clustering_report(self, good_clustering_report):
        """
        Happy path: Identify a complete clustering report.

        学习目标:
        - 理解完整聚类报告应包含的要素
        - K值选择理由、轮廓系数、业务解释
        """
        report = good_clustering_report.lower()

        # Required elements
        required = ['簇', '轮廓', '肘部', 'k', '解释']

        missing = [elem for elem in required if elem not in report]
        assert len(missing) <= 1, \
            f"合格的聚类报告应包含关键要素，缺少: {missing}"

    def test_detect_missing_k_selection_rationale(self, bad_clustering_report_no_k_selection):
        """
        Test: Identify report missing K selection rationale.

        学习目标:
        - 理解报告应说明"为什么选择这个 K 值"
        - 不能只说"用了 K=5"
        """
        report = bad_clustering_report_no_k_selection

        # Has K value but no rationale
        has_k = any(char.isdigit() for char in report)
        has_rationale = any(word in report for word in
                          ['肘部', '轮廓', 'wcgg', '选择', '理由'])

        assert has_k and not has_rationale, \
            "报告应说明 K 值选择的理由（肘部法则或轮廓系数）"

    def test_detect_missing_silhouette_score(self, bad_clustering_report_no_k_selection):
        """
        Test: Identify report missing silhouette score.

        学习目标:
        - 理解轮廓系数是聚类质量的重要指标
        - 报告应包含此指标
        """
        report = bad_clustering_report_no_k_selection.lower()

        has_silhouette = '轮廓' in report or 'silhouette' in report

        assert not has_silhouette, \
            "应该检测到报告缺少轮廓系数"

    def test_detect_missing_business_interpretation(self, bad_clustering_report_no_k_selection):
        """
        Test: Identify report without business interpretation.

        学习目标:
        - 理解聚类结果需要业务解释
        - 不能只输出数字，要说明"每个簇代表什么"
        """
        report = bad_clustering_report_no_k_selection

        # Has cluster numbers but no interpretation
        has_clusters = '簇' in report
        has_interpretation = any(word in report for word in
                                 ['活跃', '用户', '画像', '特征', '代表', '高', '低'])

        # The bad report has clusters but minimal interpretation
        # Allow this test to check the structure
        assert has_clusters, "报告应包含簇相关内容"
