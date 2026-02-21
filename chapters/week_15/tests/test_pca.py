"""
Tests for PCA (Principal Component Analysis)

PCA 降维测试用例矩阵：
- 正例：验证 PCA 在正常数据上的正确行为
- 边界：空输入、单样本、单特征、零方差、n_components 极端值
- 反例：无效参数、数据类型错误
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 正例测试：正常数据上的 PCA 行为
# =============================================================================

class TestPCAHappyPath:
    """测试 PCA 在正常情况下的行为"""

    def test_pca_basic_fit_returns_correct_components_shape(self):
        """
        正例：PCA fit 后 components_ 形状正确

        给定：50 个特征的 100 个样本
        当：拟合 PCA 模型
        期望：components_ 形状为 (n_components, n_features)
        """
        np.random.seed(42)
        X = np.random.randn(100, 50)

        pca = PCA(n_components=5)
        pca.fit(X)

        assert pca.components_.shape == (5, 50)

    def test_pca_explained_variance_ratio_sum_less_than_one(self):
        """
        正例：explained_variance_ratio_ 之和 <= 1

        给定：随机数据
        当：拟合 PCA
        期望：解释方差比例之和在合理范围内（接近 1）
        """
        np.random.seed(42)
        X = np.random.randn(100, 10)

        pca = PCA()
        pca.fit(X)

        # 解释方差比例之和应该接近 1（可能有小浮点误差）
        assert pca.explained_variance_ratio_.sum() <= 1.0
        assert pca.explained_variance_ratio_.sum() >= 0.99

    def test_pca_transform_returns_correct_shape(self):
        """
        正例：transform 返回正确形状

        给定：100 个样本，50 个特征
        当：降到 5 维
        期望：输出形状 (100, 5)
        """
        np.random.seed(42)
        X = np.random.randn(100, 50)

        pca = PCA(n_components=5)
        X_transformed = pca.fit_transform(X)

        assert X_transformed.shape == (100, 5)

    def test_pca_fit_transform_equivalent_to_fit_then_transform(self):
        """
        正例：fit_transform 等价于 fit 后 transform

        验证两个操作产生相同结果
        """
        np.random.seed(42)
        X = np.random.randn(100, 20)

        # 方法 1：fit_transform
        pca1 = PCA(n_components=5)
        result1 = pca1.fit_transform(X)

        # 方法 2：fit 然后 transform
        pca2 = PCA(n_components=5)
        pca2.fit(X)
        result2 = pca2.transform(X)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_pca_variance_maximization(self):
        """
        正例：第一主成分解释最多方差

        给定：相关数据
        当：执行 PCA
        期望：第一主成分的解释方差比例最大
        """
        np.random.seed(42)
        # 构造有明显主方向的数据
        X = np.random.randn(100, 5)
        X[:, 1:] = X[:, :1] * 0.5 + np.random.randn(100, 4) * 0.1

        pca = PCA()
        pca.fit(X)

        # 第一主成分应该解释最多方差
        assert pca.explained_variance_ratio_[0] == max(pca.explained_variance_ratio_)

    def test_pca_cumulative_variance_increases(self):
        """
        正例：累积解释方差单调递增

        验证累积方差比例应该随主成分数量单调递增
        """
        np.random.seed(42)
        X = np.random.randn(100, 10)

        pca = PCA()
        pca.fit(X)
        cumulative = np.cumsum(pca.explained_variance_ratio_)

        # 验证单调递增
        assert all(cumulative[i] <= cumulative[i + 1] + 1e-10
                   for i in range(len(cumulative) - 1))

    def test_pca_reconstruction_with_full_components(self):
        """
        正例：使用全部主成分可以完美重构数据

        给定：标准化后的数据
        当：保留所有主成分
        期望：重构误差接近零
        """
        np.random.seed(42)
        X = np.random.randn(100, 10)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 保留所有成分
        pca = PCA(n_components=10)
        X_transformed = pca.fit_transform(X_scaled)
        X_reconstructed = pca.inverse_transform(X_transformed)

        # 重构应该很接近原始数据
        np.testing.assert_array_almost_equal(X_scaled, X_reconstructed, decimal=10)


# =============================================================================
# 边界测试：极端情况
# =============================================================================

class TestPCABoundaryCases:
    """测试 PCA 在边界情况下的行为"""

    def test_pca_with_n_components_1(self):
        """
        边界：n_components=1（最小有效值）

        降维到 1 维应该正常工作
        """
        np.random.seed(42)
        X = np.random.randn(100, 10)

        pca = PCA(n_components=1)
        X_transformed = pca.fit_transform(X)

        assert X_transformed.shape == (100, 1)
        assert pca.components_.shape == (1, 10)

    def test_pca_with_n_components_equals_n_features(self):
        """
        边界：n_components=n_features（保留全部）

        应该能保留所有成分
        """
        np.random.seed(42)
        X = np.random.randn(50, 10)

        pca = PCA(n_components=10)
        X_transformed = pca.fit_transform(X)

        assert X_transformed.shape == (50, 10)
        # 保留全部成分时，累积解释方差应该为 1
        assert pca.explained_variance_ratio_.sum() >= 0.99

    def test_pca_with_min_samples(self):
        """
        边界：最小样本数（n_samples >= n_components）

        样本数必须 >= 特征数（使用全部成分时）
        """
        np.random.seed(42)
        # 5 个样本，5 个特征
        X = np.random.randn(5, 5)

        pca = PCA(n_components=5)
        pca.fit(X)

        # 应该能够拟合（虽然结果不稳定）
        assert pca.components_.shape == (5, 5)

    def test_pca_with_constant_feature(self):
        """
        边界：包含零方差特征

        常数特征在 PCA 中应该被处理
        """
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X[:, 0] = 1.0  # 常数列

        pca = PCA()
        pca.fit(X)

        # 零方差特征对应的主成分解释方差应该为 0
        assert any(v == 0 for v in pca.explained_variance_)

    def test_pca_with_single_feature(self):
        """
        边界：单特征数据

        给定：只有 1 个特征的数据
        期望：PCA 应该能处理（结果就是数据本身）
        """
        np.random.seed(42)
        X = np.random.randn(100, 1)

        pca = PCA(n_components=1)
        X_transformed = pca.fit_transform(X)

        assert X_transformed.shape == (100, 1)
        # 单特征的解释方差应该是 1
        assert pca.explained_variance_ratio_[0] == 1.0

    def test_pca_with_highly_correlated_features(self):
        """
        边界：高度相关的特征

        高度相关的特征应该能被 PCA 压缩到少量主成分
        """
        np.random.seed(42)
        # 创建高度相关的数据：所有特征都是第一个特征的线性组合
        base = np.random.randn(100, 1)
        X = np.hstack([base + np.random.randn(100, 1) * 0.01 for _ in range(10)])

        pca = PCA()
        pca.fit(X)

        # 前几个主成分应该解释大部分方差
        cumulative_3 = np.cumsum(pca.explained_variance_ratio_)[2]
        assert cumulative_3 > 0.9  # 前 3 个主成分应该解释 > 90% 的方差

    def test_pca_with_scaled_vs_unscaled_data(self):
        """
        边界：标准化 vs 未标准化数据

        PCA 对尺度敏感，未标准化数据会被大方差特征主导
        """
        np.random.seed(42)
        X = np.random.randn(100, 3)
        # 故意让特征的尺度差异很大
        X[:, 0] *= 100
        X[:, 1] *= 1
        X[:, 2] *= 0.01

        # 未标准化
        pca_unscaled = PCA(n_components=2)
        pca_unscaled.fit(X)

        # 标准化后
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca_scaled = PCA(n_components=2)
        pca_scaled.fit(X_scaled)

        # 两种情况的 components_ 应该不同
        # （因为我们期望标准化会改变主成分的方向）
        assert not np.allclose(pca_unscaled.components_, pca_scaled.components_)


# =============================================================================
# 反例测试：错误处理
# =============================================================================

class TestPCAErrorCases:
    """测试 PCA 在无效输入时的错误处理"""

    def test_pca_with_negative_n_components_raises_error(self):
        """
        反例：n_components < 0 应该报错

        主成分数量不能为负数
        """
        np.random.seed(42)
        X = np.random.randn(100, 10)

        with pytest.raises(ValueError):
            PCA(n_components=-1).fit(X)

    def test_pca_with_n_components_exceeding_min_samples_raises_error(self):
        """
        反例：n_components > min(n_samples, n_features) 应该报错

        主成分数量不能超过样本数或特征数
        """
        np.random.seed(42)
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError):
            # n_components 不能超过 min(n_samples, n_features)
            PCA(n_components=20).fit(X)

    def test_pca_with_n_components_negative_raises_error(self):
        """
        反例：n_components < 0 应该报错
        """
        np.random.seed(42)
        X = np.random.randn(100, 10)

        with pytest.raises(ValueError):
            PCA(n_components=-1).fit(X)

    def test_pca_with_empty_data_raises_error(self):
        """
        反例：空数据应该报错
        """
        X = np.array([]).reshape(0, 5)

        with pytest.raises(ValueError):
            PCA().fit(X)

    def test_pca_with_1d_array_raises_error(self):
        """
        反例：1D 数组应该报错

        PCA 需要 2D 输入 (n_samples, n_features)
        """
        np.random.seed(42)
        X = np.random.randn(100)

        with pytest.raises(Exception):  # sklearn 可能抛出 ValueError 或其他异常
            PCA().fit(X)

    def test_pca_transform_on_unfitted_model_raises_error(self):
        """
        反例：未拟合就 transform 应该报错
        """
        np.random.seed(42)
        X = np.random.randn(100, 10)

        pca = PCA(n_components=5)
        # 没有就调用 transform

        with pytest.raises(Exception):
            pca.transform(X)

    def test_pca_inverse_transform_wrong_dimension_raises_error(self):
        """
        反例：inverse_transform 输入维度不匹配应该报错

        输入维度必须与 n_components 一致
        """
        np.random.seed(42)
        X = np.random.randn(100, 10)

        pca = PCA(n_components=5)
        pca.fit(X)

        # 输入 6 维，但 PCA 只有 5 个成分
        X_wrong_dim = np.random.randn(100, 6)

        with pytest.raises(ValueError):
            pca.inverse_transform(X_wrong_dim)


# =============================================================================
# PCA 专项测试：数学性质验证
# =============================================================================

class TestPCAMathematicalProperties:
    """测试 PCA 的数学性质"""

    def test_pca_components_are_orthogonal(self):
        """
        数学性质：主成分之间正交

        不同主成分的点积应该为 0
        """
        np.random.seed(42)
        X = np.random.randn(100, 10)

        pca = PCA(n_components=5)
        pca.fit(X)

        # 检查前两个主成分的正交性
        pc1 = pca.components_[0]
        pc2 = pca.components_[1]
        dot_product = np.dot(pc1, pc2)

        assert abs(dot_product) < 1e-10  # 接近 0

    def test_pca_components_are_unit_vectors(self):
        """
        数学性质：主成分是单位向量

        每个主成分的 L2 范数应该为 1
        """
        np.random.seed(42)
        X = np.random.randn(100, 10)

        pca = PCA(n_components=5)
        pca.fit(X)

        for i in range(5):
            component_norm = np.linalg.norm(pca.components_[i])
            assert abs(component_norm - 1.0) < 1e-10

    def test_pca_mean_centering(self):
        """
        数学性质：PCA 会对数据中心化

        PCA 默认会减去均值
        """
        np.random.seed(42)
        X = np.random.randn(100, 10) + 5  # 均值约为 5

        pca = PCA()
        pca.fit(X)

        # 拟合后的均值应该接近原始数据的均值
        np.testing.assert_array_almost_equal(pca.mean_, X.mean(axis=0), decimal=10)

    def test_pca_explained_variance_ordered(self):
        """
        数学性质：解释方差按降序排列

        第一主成分解释最多方差，第二主成分次之...
        """
        np.random.seed(42)
        X = np.random.randn(100, 10)

        pca = PCA()
        pca.fit(X)

        # 检查方差是否按降序排列
        for i in range(len(pca.explained_variance_) - 1):
            assert pca.explained_variance_[i] >= pca.explained_variance_[i + 1]


# =============================================================================
# 场景测试：电商客户数据降维
# =============================================================================

class TestPCACustomerSegmentationScenario:
    """测试 PCA 在客户分群场景中的应用"""

    def test_pca_customer_data_dimensionality_reduction(self):
        """
        场景：电商客户数据从 50 维降到 2 维

        模拟客户行为特征降维场景
        """
        np.random.seed(42)
        # 模拟 1000 个客户，50 个行为特征
        X = np.random.randn(1000, 50)

        # 降到 2 维用于可视化
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        assert X_2d.shape == (1000, 2)
        # 检查 2 个主成分的解释方差比例
        assert len(pca.explained_variance_ratio_) == 2
        # 第一主成分应该解释更多
        assert pca.explained_variance_ratio_[0] >= pca.explained_variance_ratio_[1]

    def test_pca_customer_data_80_percent_variance(self):
        """
        场景：保留 80% 方差的主成分数量

        找到能解释 80% 方差的最小主成分数
        """
        np.random.seed(42)
        X = np.random.randn(500, 50)

        pca = PCA()
        pca.fit(X)

        cumulative = np.cumsum(pca.explained_variance_ratio_)
        # 找到累积方差 >= 0.8 的最小索引
        n_components_80 = np.argmax(cumulative >= 0.8) + 1

        # 验证
        assert cumulative[n_components_80 - 1] >= 0.8
        if n_components_80 > 1:
            assert cumulative[n_components_80 - 2] < 0.8


# =============================================================================
# PCA 与 StandardScaler 组合测试
# =============================================================================

class TestPCAWithScaler:
    """测试 PCA 与 StandardScaler 的组合使用"""

    def test_pca_after_standardization_zero_mean_unit_variance(self):
        """
        验证：标准化后的数据均值为 0，标准差为 1
        """
        np.random.seed(42)
        X = np.random.randn(100, 5) * 10 + 50  # 不同的均值和尺度

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 每个特征的均值应该接近 0
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), np.zeros(5), decimal=10)
        # 每个特征的标准差应该接近 1
        np.testing.assert_array_almost_equal(X_scaled.std(axis=0), np.ones(5), decimal=10)

    def test_pca_pipeline_scaler_then_pca(self):
        """
        场景：Scaler + PCA 的标准流程

        先标准化，再做 PCA
        """
        np.random.seed(42)
        X = np.random.randn(100, 10) * 100 + 50

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA
        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_scaled)

        assert X_pca.shape == (100, 5)
        # 变换后的数据在第一主成分方向上应该中心化
        assert abs(X_pca[:, 0].mean()) < 1e-10
