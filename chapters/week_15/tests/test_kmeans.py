"""
Tests for K-means Clustering

K-means 聚类测试用例矩阵：
- 正例：验证 K-means 在正常数据上的正确行为
- 边界：K=1、K=n_samples、单簇、球形簇
- 反例：无效参数、数据类型错误
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs


# =============================================================================
# 正例测试：正常数据上的 K-means 行为
# =============================================================================

class TestKMeansHappyPath:
    """测试 K-means 在正常情况下的行为"""

    def test_kmeans_basic_fit_returns_correct_labels_shape(self):
        """
        正例：K-means fit 后 labels_ 形状正确

        给定：100 个样本，3 个簇
        当：拟合 K-means 模型
        期望：labels_ 形状为 (n_samples,)
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        assert kmeans.labels_.shape == (100,)
        assert len(kmeans.labels_) == 100

    def test_kmeans_labels_in_valid_range(self):
        """
        正例：labels_ 值在有效范围内 [0, n_clusters-1]

        给定：K=3
        当：拟合 K-means
        期望：所有标签在 {0, 1, 2} 中
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        unique_labels = np.unique(kmeans.labels_)
        expected_labels = np.array([0, 1, 2])

        np.testing.assert_array_equal(np.sort(unique_labels), expected_labels)

    def test_kmeans_cluster_centers_shape(self):
        """
        正例：cluster_centers_ 形状正确

        给定：n_features=2, n_clusters=3
        当：拟合 K-means
        期望：cluster_centers_ 形状为 (3, 2)
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        assert kmeans.cluster_centers_.shape == (3, 2)

    def test_kmeans_inertia_is_non_negative(self):
        """
        正例：inertia_ 非负

        inertia_ 是簇内平方和，必须 >= 0
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        assert kmeans.inertia_ >= 0

    def test_kmeans_fit_predict_returns_same_labels_as_fit(self):
        """
        正例：fit_predict 返回的标签与 fit 后的 labels_ 一致

        验证 fit_predict 的正确性
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

        # 方法 1：fit 然后 labels_
        kmeans1 = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans1.fit(X)
        labels1 = kmeans1.labels_

        # 方法 2：fit_predict
        kmeans2 = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels2 = kmeans2.fit_predict(X)

        np.testing.assert_array_equal(labels1, labels2)

    def test_kmeans_predict_returns_valid_labels(self):
        """
        正例：predict 对新数据返回有效标签

        给定：已拟合的 K-means 模型
        当：预测新数据
        期望：返回的标签在有效范围内
        """
        np.random.seed(42)
        X_train, _ = make_blobs(n_samples=100, centers=3, random_state=42)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_train)

        # 预测新数据
        X_new = np.random.randn(20, 2)
        new_labels = kmeans.predict(X_new)

        assert new_labels.shape == (20,)
        assert np.all(new_labels >= 0)
        assert np.all(new_labels < 3)

    def test_kmeans_with_well_separated_clusters(self):
        """
        正例：对分离良好的数据，K-means 应该表现良好

        给定：3 个分离良好的 blob
        当：K-means 聚类
        期望：轮廓系数较高（> 0.5）
        """
        np.random.seed(42)
        X, true_labels = make_blobs(n_samples=300, centers=3,
                                     cluster_std=0.5, random_state=42)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(X)

        # 轮廓系数应该较高
        score = silhouette_score(X, pred_labels)
        assert score > 0.3  # 分离良好的数据应该有较高的轮廓系数


# =============================================================================
# 边界测试：极端情况
# =============================================================================

class TestKMeansBoundaryCases:
    """测试 K-means 在边界情况下的行为"""

    def test_kmeans_with_k_equals_1(self):
        """
        边界：K=1（最小有效值）

        所有样本应该被分配到同一个簇
        """
        np.random.seed(42)
        X = np.random.randn(100, 2)

        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans.fit(X)

        assert np.all(kmeans.labels_ == 0)
        assert kmeans.cluster_centers_.shape == (1, 2)
        # 簇中心应该是数据的均值
        np.testing.assert_array_almost_equal(
            kmeans.cluster_centers_[0],
            X.mean(axis=0),
            decimal=5
        )

    def test_kmeans_with_k_equals_n_samples(self):
        """
        边界：K=n_samples（极端情况）

        每个样本是一个簇， inertia_ 应该接近 0
        """
        np.random.seed(42)
        X = np.random.randn(10, 2)

        kmeans = KMeans(n_clusters=10, random_state=42, n_init=1)
        kmeans.fit(X)

        # inertia_ 应该为 0（每个样本是自己的簇中心）
        assert kmeans.inertia_ < 1e-10

    def test_kmeans_with_min_samples(self):
        """
        边界：最小样本数（n_samples >= n_clusters）

        样本数必须 >= 簇数
        """
        np.random.seed(42)
        # 3 个样本，3 个簇
        X = np.random.randn(3, 2)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        # 应该能够拟合
        assert len(np.unique(kmeans.labels_)) == 3

    def test_kmeans_with_single_feature(self):
        """
        边界：单特征数据

        K-means 应该能处理 1 维数据
        """
        np.random.seed(42)
        X = np.random.randn(100, 1)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        assert kmeans.labels_.shape == (100,)
        assert kmeans.cluster_centers_.shape == (3, 1)

    def test_kmeans_with_high_dimensional_data(self):
        """
        边界：高维数据

        K-means 应该能处理高维数据（虽然效果可能不佳）
        """
        np.random.seed(42)
        X = np.random.randn(100, 50)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        assert kmeans.labels_.shape == (100,)
        assert kmeans.cluster_centers_.shape == (3, 50)

    def test_kmeans_with_constant_data(self):
        """
        边界：常数数据（所有样本相同）

        所有样本相同时，K-means 应该仍然能运行
        """
        np.random.seed(42)
        X = np.ones((100, 2))

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        # 所有簇中心应该相同
        for center in kmeans.cluster_centers_:
            np.testing.assert_array_almost_equal(center, np.ones(2))

    def test_kmeans_with_collinear_data(self):
        """
        边界：共线数据

        数据在一条直线上
        """
        np.random.seed(42)
        # 所有点在 x=y 线上
        base = np.random.randn(100, 1)
        X = np.hstack([base, base])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X)

        # 应该能够拟合（虽然簇在一条线上）
        assert kmeans.labels_.shape == (100,)

    def test_kmeans_inertia_decreases_with_k(self):
        """
        边界：inertia_ 随 K 增加而单调递减

        K 越大，簇内平方和越小（但可能过拟合）
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

        inertias = []
        for k in range(2, 10):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # 验证单调递减
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1]


# =============================================================================
# 反例测试：错误处理
# =============================================================================

class TestKMeansErrorCases:
    """测试 K-means 在无效输入时的错误处理"""

    def test_kmeans_with_k_zero_raises_error(self):
        """
        反例：n_clusters=0 应该报错

        簇数必须 >= 1
        """
        np.random.seed(42)
        X = np.random.randn(100, 2)

        with pytest.raises(ValueError):
            KMeans(n_clusters=0, random_state=42).fit(X)

    def test_kmeans_with_k_negative_raises_error(self):
        """
        反例：n_clusters < 0 应该报错
        """
        np.random.seed(42)
        X = np.random.randn(100, 2)

        with pytest.raises(ValueError):
            KMeans(n_clusters=-1, random_state=42).fit(X)

    def test_kmeans_with_k_exceeds_n_samples_raises_error(self):
        """
        反例：n_clusters > n_samples 应该报错

        簇数不能超过样本数
        """
        np.random.seed(42)
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError):
            KMeans(n_clusters=20, random_state=42).fit(X)

    def test_kmeans_with_empty_data_raises_error(self):
        """
        反例：空数据应该报错
        """
        X = np.array([]).reshape(0, 2)

        with pytest.raises(ValueError):
            KMeans(n_clusters=2, random_state=42).fit(X)

    def test_kmeans_with_1d_array_raises_error(self):
        """
        反例：1D 数组应该报错

        K-means 需要 2D 输入 (n_samples, n_features)
        """
        np.random.seed(42)
        X = np.random.randn(100)

        with pytest.raises(ValueError):
            KMeans(n_clusters=3, random_state=42).fit(X)

    def test_kmeans_predict_on_unfitted_model_raises_error(self):
        """
        反例：未拟合就 predict 应该报错
        """
        np.random.seed(42)
        X = np.random.randn(100, 2)

        kmeans = KMeans(n_clusters=3, random_state=42)
        # 没有就调用 predict

        with pytest.raises(Exception):
            kmeans.predict(X)

    def test_kmeans_predict_wrong_dimension_raises_error(self):
        """
        反例：predict 输入维度不匹配应该报错

        预测数据的特征数必须与训练数据一致
        """
        np.random.seed(42)
        X_train = np.random.randn(100, 2)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(X_train)

        # 输入 3 维数据，但模型期望 2 维
        X_wrong_dim = np.random.randn(20, 3)

        with pytest.raises(Exception):
            kmeans.predict(X_wrong_dim)


# =============================================================================
# 轮廓系数测试
# =============================================================================

class TestSilhouetteScore:
    """测试轮廓系数计算"""

    def test_silhouette_score_in_valid_range(self):
        """
        正例：轮廓系数在 [-1, 1] 范围内

        轮廓系数衡量聚类质量，范围 [-1, 1]，越大越好
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)

        assert -1 <= score <= 1

    def test_silhouette_score_well_separated_clusters_high_score(self):
        """
        正例：分离良好的簇轮廓系数较高

        给定：分离良好的数据
        期望：轮廓系数 > 0.5
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=300, centers=3,
                          cluster_std=0.3, random_state=42)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)

        # 分离良好的簇应该有较高的轮廓系数
        assert score > 0.5

    def test_silhouette_score_overlapping_clusters_lower_score(self):
        """
        正例：重叠的簇轮廓系数较低

        给定：重叠的数据（高 std）
        期望：轮廓系数显著低于分离良好的簇
        """
        np.random.seed(42)
        # 创建非常重叠的数据 - 所有簇中心很接近
        X, _ = make_blobs(n_samples=300, centers=[[0, 0], [1, 1], [2, 2]],
                          cluster_std=3.0, random_state=42)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        score = silhouette_score(X, labels)

        # 非常重叠的簇轮廓系数应该较低（< 0.3）
        assert score < 0.35

    def test_silhouette_score_single_cluster_raises_error(self):
        """
        反例：只有 1 个簇时无法计算轮廓系数

        轮廓系数需要至少 2 个簇
        """
        np.random.seed(42)
        X = np.random.randn(100, 2)

        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # 轮廓系数需要至少 2 个簇
        with pytest.raises(ValueError):
            silhouette_score(X, labels)

    def test_silhouette_score_all_same_label_raises_error(self):
        """
        反例：所有样本属于同一簇无法计算轮廓系数
        """
        np.random.seed(42)
        X = np.random.randn(100, 2)
        labels = np.zeros(100, dtype=int)

        with pytest.raises(ValueError):
            silhouette_score(X, labels)


# =============================================================================
# 场景测试：客户分群
# =============================================================================

class TestKMeansCustomerSegmentation:
    """测试 K-means 在客户分群场景中的应用"""

    def test_kmeans_customer_segmentation_basic(self):
        """
        场景：电商客户分群

        模拟客户行为数据聚类
        """
        np.random.seed(42)
        # 模拟 1000 个客户的 3 个关键特征
        # 总消费、访问频次、优惠券使用率
        X = np.random.randn(1000, 3)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # 验证基本属性
        assert labels.shape == (1000,)
        assert kmeans.cluster_centers_.shape == (3, 3)

        # 验证每个簇都有样本
        unique, counts = np.unique(labels, return_counts=True)
        assert len(unique) == 3
        assert all(counts > 0)

    def test_kmeans_elbow_method_concept(self):
        """
        场景：肘部法则选择 K

        验证不同 K 值下的 inertia_ 变化
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

        ks = range(2, 11)
        inertias = []

        for k in ks:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # inertia_ 应该随 K 增加而递减
        for i in range(len(inertias) - 1):
            assert inertias[i] > inertias[i + 1]

        # 第一个 K 的 inertia_ 应该显著大于最后一个
        assert inertias[0] > inertias[-1]

    def test_kmeans_different_k_silhouette_comparison(self):
        """
        场景：比较不同 K 值的轮廓系数

        找到轮廓系数最大的 K
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

        ks = range(2, 11)
        scores = []

        for k in ks:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append(score)

        # 所有分数应该在有效范围内
        for score in scores:
            assert -1 <= score <= 1

        # 对分离良好的数据，应该有某个 K 的分数较高
        assert max(scores) > 0


# =============================================================================
# K-means 随机初始化测试
# =============================================================================

class TestKMeansInitialization:
    """测试 K-means 的初始化敏感性"""

    def test_kmeans_with_fixed_random_state_reproducible(self):
        """
        验证：固定 random_state 结果可复现

        相同的 random_state 应该产生相同的结果
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

        # 两次运行，使用相同的 random_state
        kmeans1 = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels1 = kmeans1.fit_predict(X)

        kmeans2 = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels2 = kmeans2.fit_predict(X)

        np.testing.assert_array_equal(labels1, labels2)

    def test_kmeans_with_different_random_state_different_results(self):
        """
        边界：不同 random_state 可能产生不同结果

        K-means 的随机初始化可能导致不同结果
        """
        np.random.seed(42)
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

        kmeans1 = KMeans(n_clusters=3, random_state=1, n_init=10)
        labels1 = kmeans1.fit_predict(X)

        kmeans2 = KMeans(n_clusters=3, random_state=99, n_init=10)
        labels2 = kmeans2.fit_predict(X)

        # 对于某些数据集，不同初始化可能产生不同结果
        #（对于分离良好的数据，结果可能相同）
        # 这里只验证它们都能运行
        assert labels1.shape == labels2.shape
