"""
Tests for StandardScaler and Data Preprocessing

StandardScaler 标准化测试用例矩阵：
- 正例：验证 StandardScaler 在正常数据上的正确行为
- 边界：常数列、零方差、单样本、单特征
- 反例：无效参数、数据类型错误
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# =============================================================================
# 正例测试：StandardScaler 基本行为
# =============================================================================

class TestStandardScalerHappyPath:
    """测试 StandardScaler 在正常情况下的行为"""

    def test_scaler_fit_transform_returns_zero_mean_unit_variance(self):
        """
        正例：fit_transform 后均值为 0，标准差为 1

        给定：任意数据
        当：StandardScaler fit_transform
        期望：每列均值约 0，标准差约 1
        """
        np.random.seed(42)
        X = np.random.randn(100, 5) * 10 + 50

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 每个特征的均值应该接近 0
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros(5), decimal=10)
        # 每个特征的标准差应该接近 1
        np.testing.assert_array_almost_equal(X_scaled.std(axis=0), np.ones(5), decimal=10)

    def test_scaler_transform_uses_fitted_mean_and_scale(self):
        """
        正例：transform 使用拟合时的均值和尺度

        验证 transform 使用的是训练集的统计量
        """
        np.random.seed(42)
        X_train = np.random.randn(100, 3) * 10 + 50
        X_test = np.random.randn(20, 3) * 5 + 30

        scaler = StandardScaler()
        scaler.fit(X_train)

        # 检查 scaler 存储的均值和尺度
        np.testing.assert_array_almost_equal(
            scaler.mean_,
            X_train.mean(axis=0),
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            scaler.scale_,
            X_train.std(axis=0, ddof=0),
            decimal=10
        )

        # 测试数据使用训练集的统计量
        X_test_scaled = scaler.transform(X_test)
        expected = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0, ddof=0)
        np.testing.assert_array_almost_equal(X_test_scaled, expected, decimal=10)

    def test_scaler_fit_transform_equivalent_to_fit_then_transform(self):
        """
        正例：fit_transform 等价于 fit 后 transform

        验证两个操作产生相同结果
        """
        np.random.seed(42)
        X = np.random.randn(100, 5) * 10 + 50

        # 方法 1：fit_transform
        scaler1 = StandardScaler()
        result1 = scaler1.fit_transform(X)

        # 方法 2：fit 然后 transform
        scaler2 = StandardScaler()
        scaler2.fit(X)
        result2 = scaler2.transform(X)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_scaler_inverse_transform_recovers_original(self):
        """
        正例：inverse_transform 恢复原始数据

        给定：标准化后的数据
        当：inverse_transform
        期望：恢复原始数据
        """
        np.random.seed(42)
        X = np.random.randn(100, 5) * 10 + 50

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X, X_recovered, decimal=10)

    def test_scaler_with_different_scales(self):
        """
        正例：不同尺度的特征被正确标准化

        给定：特征尺度差异很大的数据
        当：StandardScaler
        期望：所有特征具有相同的尺度
        """
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 0] *= 100    # 第一列尺度 100
        X[:, 1] *= 1      # 第二列尺度 1
        X[:, 2] *= 0.01   # 第三列尺度 0.01

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 所有特征的标准差应该接近 1
        np.testing.assert_array_almost_equal(X_scaled.std(axis=0), np.ones(3), decimal=10)

    def test_scaler_partial_fit_behavior(self):
        """
        正例：partial_fit 可以增量更新统计量

        验证 partial_fit 的正确性
        """
        np.random.seed(42)
        X1 = np.random.randn(50, 3) * 10 + 50
        X2 = np.random.randn(50, 3) * 5 + 30

        scaler = StandardScaler()
        scaler.partial_fit(X1)
        scaler.partial_fit(X2)

        # 合并数据的均值和标准差
        X_combined = np.vstack([X1, X2])
        expected_mean = X_combined.mean(axis=0)
        expected_std = X_combined.std(axis=0, ddof=0)

        np.testing.assert_array_almost_equal(scaler.mean_, expected_mean, decimal=10)
        np.testing.assert_array_almost_equal(scaler.scale_, expected_std, decimal=10)


# =============================================================================
# 边界测试：极端情况
# =============================================================================

class TestStandardScalerBoundaryCases:
    """测试 StandardScaler 在边界情况下的行为"""

    def test_scaler_with_single_feature(self):
        """
        边界：单特征数据

        StandardScaler 应该能处理单特征数据
        """
        np.random.seed(42)
        X = np.random.randn(100, 1) * 10 + 50

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        assert X_scaled.shape == (100, 1)
        assert abs(X_scaled.mean()) < 1e-10
        assert abs(X_scaled.std() - 1.0) < 1e-10

    def test_scaler_with_single_sample(self):
        """
        边界：单样本数据

        单样本时标准差为 0，需要特殊处理
        """
        np.random.seed(42)
        X = np.array([[1.0, 2.0, 3.0]])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 单样本的均值是样本本身，标准差为 1（sklearn 的默认行为）
        np.testing.assert_array_almost_equal(X_scaled, np.zeros((1, 3)), decimal=10)

    def test_scaler_with_constant_feature(self):
        """
        边界：包含常数特征（零方差）

        常数特征的方差为 0，标准化后应该为 0
        """
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 0] = 5.0  # 常数列

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 常数列标准化后应该为 0（避免除以 0）
        np.testing.assert_array_almost_equal(X_scaled[:, 0], np.zeros(100), decimal=10)

    def test_scaler_with_all_constant_data(self):
        """
        边界：所有特征都是常数

        整个数据集方差为 0
        """
        X = np.ones((100, 3)) * 5

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 所有值应该变为 0
        np.testing.assert_array_almost_equal(X_scaled, np.zeros((100, 3)), decimal=10)

    def test_scaler_with_zero_mean_data(self):
        """
        边界：零均值数据

        给定：已经零均值的数据
        期望：标准化后仍然是零均值
        """
        np.random.seed(42)
        X = np.random.randn(100, 3)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros(3), decimal=10)

    def test_scaler_with_negative_values(self):
        """
        边界：包含负值

        StandardScaler 应该正确处理负值
        """
        np.random.seed(42)
        X = np.random.randn(100, 3) * 10 - 5

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 标准化后均值应为 0，标准差为 1
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros(3), decimal=10)
        np.testing.assert_array_almost_equal(X_scaled.std(axis=0), np.ones(3), decimal=10)

    def test_scaler_with_large_values(self):
        """
        边界：极大数值

        StandardScaler 应该能处理大数值
        """
        np.random.seed(42)
        X = np.random.randn(100, 3) * 1e6 + 1e9

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 即使原始值很大，标准化后也应该是 0 均值 1 标准差
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros(3), decimal=10)
        np.testing.assert_array_almost_equal(X_scaled.std(axis=0), np.ones(3), decimal=10)

    def test_scaler_with_very_small_values(self):
        """
        边界：极小数值

        StandardScaler 应该能处理接近 0 的数值
        """
        np.random.seed(42)
        X = np.random.randn(100, 3) * 1e-10

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 即使原始值很小，标准化后也应该是 0 均值 1 标准差
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros(3), decimal=10)
        np.testing.assert_array_almost_equal(X_scaled.std(axis=0), np.ones(3), decimal=10)


# =============================================================================
# 反例测试：错误处理
# =============================================================================

class TestStandardScalerErrorCases:
    """测试 StandardScaler 在无效输入时的错误处理"""

    def test_scaler_with_empty_data_raises_error(self):
        """
        反例：空数据应该报错
        """
        X = np.array([]).reshape(0, 3)

        scaler = StandardScaler()
        with pytest.raises(ValueError):
            scaler.fit(X)

    def test_scaler_with_1d_array_raises_error(self):
        """
        反例：1D 数组应该报错

        StandardScaler 需要 2D 输入 (n_samples, n_features)
        """
        np.random.seed(42)
        X = np.random.randn(100)

        scaler = StandardScaler()
        # sklearn 的 StandardScaler 要求 2D 输入
        with pytest.raises(ValueError):
            scaler.fit_transform(X)

    def test_scaler_transform_on_unfitted_model_raises_error(self):
        """
        反例：未拟合就 transform 应该报错
        """
        np.random.seed(42)
        X = np.random.randn(100, 3)

        scaler = StandardScaler()
        # 没有就调用 transform

        with pytest.raises(Exception):
            scaler.transform(X)

    def test_scaler_transform_wrong_dimension_raises_error(self):
        """
        反例：transform 输入特征数不匹配

        预测数据的特征数必须与训练数据一致
        """
        np.random.seed(42)
        X_train = np.random.randn(100, 3)

        scaler = StandardScaler()
        scaler.fit(X_train)

        # 输入 5 维数据，但模型期望 3 维
        X_wrong_dim = np.random.randn(20, 5)

        with pytest.raises(ValueError):
            scaler.transform(X_wrong_dim)


# =============================================================================
# MinMaxScaler 测试
# =============================================================================

class TestMinMaxScaler:
    """测试 MinMaxScaler（另一种常用缩放方法）"""

    def test_minmax_scaler_scales_to_given_range(self):
        """
        正例：MinMaxScaler 缩放到 [0, 1]

        给定：任意数据
        当：MinMaxScaler fit_transform
        期望：所有值在 [0, 1] 范围内
        """
        np.random.seed(42)
        X = np.random.randn(100, 5) * 10 + 50

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # 所有值应该在 [0, 1] 范围内
        assert X_scaled.min() >= 0
        assert X_scaled.max() <= 1

    def test_minmax_scaler_custom_range(self):
        """
        正例：MinMaxScaler 自定义范围

        缩放到任意范围 [feature_range]
        """
        np.random.seed(42)
        X = np.random.randn(100, 3) * 10 + 50

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)

        # 所有值应该在 [-1, 1] 范围内
        # 使用 near 进行比较，因为浮点精度可能导致边界略超出范围
        assert X_scaled.min() >= -1.01
        assert X_scaled.max() <= 1.01

    def test_minmax_scaler_inverse_transform(self):
        """
        正例：MinMaxScaler inverse_transform 恢复原始数据
        """
        np.random.seed(42)
        X = np.random.randn(100, 3) * 10 + 50

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_recovered = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X, X_recovered, decimal=10)


# =============================================================================
# 场景测试：PCA 前的标准化
# =============================================================================

class TestScalerForPCA:
    """测试 PCA 前的标准化（Week 15 场景）"""

    def test_scaler_before_pca_changes_results(self):
        """
        场景：标准化对 PCA 结果的影响

        验证：未标准化和标准化的数据，PCA 结果不同
        """
        np.random.seed(42)
        X = np.random.randn(100, 3)
        # 让特征尺度差异很大
        X[:, 0] *= 100
        X[:, 1] *= 1
        X[:, 2] *= 0.01

        from sklearn.decomposition import PCA

        # 未标准化
        pca_unscaled = PCA(n_components=2)
        result_unscaled = pca_unscaled.fit_transform(X)

        # 标准化后
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca_scaled = PCA(n_components=2)
        result_scaled = pca_scaled.fit_transform(X_scaled)

        # 结果应该不同
        assert not np.allclose(result_unscaled, result_scaled)

    def test_scaler_pca_pipeline(self):
        """
        场景：Scaler + PCA 的标准流程

        验证整个流程能正常运行
        """
        np.random.seed(42)
        X = np.random.randn(100, 10) * 100 + 50

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_scaled)

        assert X_scaled.shape == (100, 10)
        assert X_pca.shape == (100, 5)

        # 验证标准化后的数据
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros(10), decimal=10)
        np.testing.assert_array_almost_equal(X_scaled.std(axis=0), np.ones(10), decimal=10)


# =============================================================================
# K-means 前的标准化测试
# =============================================================================

class TestScalerForKMeans:
    """测试 K-means 前的标准化（Week 15 场景）"""

    def test_scaler_before_kmeans_changes_clusters(self):
        """
        场景：标准化对 K-means 结果的影响

        K-means 基于距离，尺度敏感，标准化会改变聚类结果
        """
        np.random.seed(42)
        X = np.random.randn(100, 3)
        # 让特征尺度差异很大
        X[:, 0] *= 100
        X[:, 1] *= 1
        X[:, 2] *= 0.01

        from sklearn.cluster import KMeans

        # 未标准化
        kmeans_unscaled = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_unscaled = kmeans_unscaled.fit_predict(X)

        # 标准化后
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans_scaled = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_scaled = kmeans_scaled.fit_predict(X_scaled)

        # 结果应该不同（虽然不保证所有样本都不同）
        similarity = (labels_unscaled == labels_scaled).mean()
        # 至少有一些样本的簇分配不同
        assert similarity < 1.0

    def test_scaler_kmeans_pipeline(self):
        """
        场景：Scaler + K-means 的标准流程

        验证整个流程能正常运行
        """
        np.random.seed(42)
        X = np.random.randn(100, 5) * 100 + 50

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # K-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        assert labels.shape == (100,)
        assert len(np.unique(labels)) == 3

        # 验证标准化后的数据
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros(5), decimal=10)
