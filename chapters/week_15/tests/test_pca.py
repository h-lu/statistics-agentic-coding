"""
Test suite for Week 15: PCA Dimensionality Reduction

This module tests PCA dimensionality reduction, covering:
- PCA fitting and transformation
- Explained variance ratio
- Cumulative variance threshold
- Feature loadings and interpretation
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Test 1: PCA Fitting and Transformation
# =============================================================================

class TestPCAFitting:
    """Test PCA fitting and basic transformation."""

    def test_pca_fits_on_simple_2d_data(self, simple_2d_data):
        """
        Happy path: Fit PCA on 2D data.

        学习目标:
        - 理解 PCA 基本拟合流程
        - n_components 参数控制降维后的维度
        """
        X = simple_2d_data.values

        # Fit PCA with 1 component
        pca = PCA(n_components=1)
        X_transformed = pca.fit_transform(X)

        # Check transformation
        assert X_transformed.shape == (len(X), 1), \
            "降维后应该有1列"

    def test_pca_requires_standardization(self, high_dim_data):
        """
        Test: PCA should be performed on standardized data.

        学习目标:
        - 理解 PCA 对尺度敏感
        - 不同量纲的特征需要标准化
        """
        X = high_dim_data

        # Without standardization
        pca_raw = PCA(n_components=5)
        pca_raw.fit(X)

        # With standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca_scaled = PCA(n_components=5)
        pca_scaled.fit(X_scaled)

        # Standardized PCA will have different explained variance
        # (because features now have equal variance)
        assert hasattr(pca_scaled, 'explained_variance_ratio_'), \
            "拟合后的PCA应该有 explained_variance_ratio_ 属性"

    def test_pca_max_components_limited_by_min(self, high_dim_data):
        """
        Edge case: n_components cannot exceed min(n_samples, n_features).

        学习目标:
        - 理解 PCA 的成分数限制
        - 成分数最多为 min(样本数, 特征数)
        """
        X = high_dim_data.values
        n_samples, n_features = X.shape

        max_components = min(n_samples, n_features)

        # Try to fit with too many components - should raise error
        pca = PCA(n_components=1000)

        # sklearn will raise ValueError for n_components > min(n_samples, n_features)
        with pytest.raises(ValueError):
            pca.fit(X)

    def test_pca_reconstruction_error(self, simple_2d_data):
        """
        Test: PCA reconstruction has some error (information loss).

        学习目标:
        - 理解降维会损失信息
        - 保留的主成分越少，重构误差越大
        """
        X = simple_2d_data.values

        # Fit PCA with 1 component (50% compression)
        pca = PCA(n_components=1)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        # Reconstruction should not be perfect
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)
        assert reconstruction_error > 0, \
            "降维后重构应该有误差"


# =============================================================================
# Test 2: Explained Variance Ratio
# =============================================================================

class TestExplainedVariance:
    """Test explained variance ratio and cumulative variance."""

    def test_explained_variance_ratio_decreases(self, high_dim_data):
        """
        Happy path: Explained variance ratio decreases with component index.

        学习目标:
        - 理解主成分按方差解释比例排序
        - PC1 解释最多方差，PC2 次之
        """
        X = StandardScaler().fit_transform(high_dim_data)

        pca = PCA()
        pca.fit(X)

        # Check that variance decreases
        evr = pca.explained_variance_ratio_
        for i in range(len(evr) - 1):
            assert evr[i] >= evr[i + 1], \
                f"PC{i+1} 的方差解释比例应 >= PC{i+2}"

    def test_explained_variance_sum_equals_one(self, high_dim_data):
        """
        Test: Sum of explained variance ratios should equal 1.

        学习目标:
        - 理解方差解释比例的归一化
        - 所有主成分的方差解释比例之和 = 100%
        """
        X = StandardScaler().fit_transform(high_dim_data)

        pca = PCA()
        pca.fit(X)

        total_variance = np.sum(pca.explained_variance_ratio_)

        assert abs(total_variance - 1.0) < 1e-10, \
            f"所有主成分的方差解释比例之和应等于1，实际: {total_variance:.10f}"

    def test_cumulative_variance_is_monotonic(self, high_dim_data):
        """
        Test: Cumulative variance should be monotonically increasing.

        学习目标:
        - 理解累积方差解释比例的概念
        - 每增加一个主成分，累积方差只增不减
        """
        X = StandardScaler().fit_transform(high_dim_data)

        pca = PCA()
        pca.fit(X)

        # Calculate cumulative variance
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

        # Check monotonic increase
        for i in range(len(cumsum_variance) - 1):
            assert cumsum_variance[i] <= cumsum_variance[i + 1], \
                f"累积方差应该单调递增"

    def test_select_components_by_variance_threshold(self, data_for_variance_threshold):
        """
        Happy path: Select components to reach 85% variance threshold.

        学习目标:
        - 理解如何根据方差阈值选择主成分数量
        - 目标：用最少的主成分达到目标方差保留比例
        """
        X = StandardScaler().fit_transform(data_for_variance_threshold)

        # Fit full PCA
        pca_full = PCA()
        pca_full.fit(X)

        # Calculate cumulative variance
        cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

        # Find components needed for 85% variance
        threshold = 0.85
        n_components = (cumsum_variance >= threshold).argmax() + 1

        # Verify
        assert cumsum_variance[n_components - 1] >= threshold, \
            f"选定的 {n_components} 个主成分应保留 >= 85% 的方差"

        # Check that one fewer component wouldn't meet threshold
        if n_components > 1:
            assert cumsum_variance[n_components - 2] < threshold, \
                f"{n_components-1} 个主成分应不足以达到 85% 阈值"


# =============================================================================
# Test 3: Feature Loadings and Interpretation
# =============================================================================

class TestFeatureLoadings:
    """Test feature loadings (components_) for interpretation."""

    def test_feature_loadings_shape(self, high_dim_data):
        """
        Test: Feature loadings matrix has correct shape.

        学习目标:
        - 理解 feature loadings (components_) 的结构
        - shape = (n_components, n_features)
        """
        X = StandardScaler().fit_transform(high_dim_data)

        n_components = 5
        pca = PCA(n_components=n_components)
        pca.fit(X)

        # Feature loadings shape
        loadings = pca.components_
        assert loadings.shape == (n_components, X.shape[1]), \
            f"特征载荷矩阵形状应为 ({n_components}, n_features)"

    def test_feature_loadings_unit_length(self, high_dim_data):
        """
        Test: Each principal component (loading vector) has unit length.

        学习目标:
        - 理解主成分是单位向量
        - ||loading_vector|| = 1
        """
        X = StandardScaler().fit_transform(high_dim_data)

        pca = PCA(n_components=5)
        pca.fit(X)

        # Check each component has unit length
        for i, component in enumerate(pca.components_):
            length = np.linalg.norm(component)
            assert abs(length - 1.0) < 1e-10, \
                f"PC{i+1} 的载荷向量长度应为1，实际: {length:.10f}"

    def test_feature_loadings_orthogonal(self, high_dim_data):
        """
        Test: Principal components are orthogonal (uncorrelated).

        学习目标:
        - 理解主成分之间相互正交
        - loading vectors 的点积 = 0
        """
        X = StandardScaler().fit_transform(high_dim_data)

        pca = PCA(n_components=5)
        pca.fit(X)

        # Check orthogonality between components
        for i in range(len(pca.components_)):
            for j in range(i + 1, len(pca.components_)):
                dot_product = np.dot(pca.components_[i], pca.components_[j])
                assert abs(dot_product) < 1e-10, \
                    f"PC{i+1} 和 PC{j+1} 应该正交（点积接近0）"

    def test_feature_loadings_interpretation(self, simple_2d_data):
        """
        Happy path: Interpret feature loadings for business meaning.

        学习目标:
        - 理解如何用特征载荷解释主成分
        - 载荷值大的特征对该主成分贡献大
        """
        X = StandardScaler().fit_transform(simple_2d_data)

        pca = PCA(n_components=1)
        pca.fit(X)

        # Feature loadings
        loadings = pca.components_[0]

        # Both features should have positive loadings (they're correlated)
        assert loadings[0] * loadings[1] > 0, \
            "相关特征的主成分载荷应该有相同符号"

        # Absolute values matter for interpretation
        abs_loadings = np.abs(loadings)
        assert abs_loadings[0] > 0 or abs_loadings[1] > 0, \
            "至少有一个特征对主成分有显著贡献"


# =============================================================================
# Test 4: PCA Edge Cases
# =============================================================================

class TestPCAEdgeCases:
    """Test PCA with edge cases and boundary conditions."""

    def test_pca_with_constant_data(self, constant_data):
        """
        Edge case: PCA with constant (zero variance) data.

        学习目标:
        - 理解方差为0的特征无法用于 PCA
        - 这些特征会被忽略或产生警告
        """
        X = constant_data.values

        # sklearn PCA will handle constant features
        pca = PCA(n_components=1)
        pca.fit(X)

        # Should still fit, but explained variance might be 0/NaN
        assert hasattr(pca, 'explained_variance_ratio_')

    def test_pca_with_single_feature(self, simple_2d_data):
        """
        Edge case: PCA with only 1 feature (no reduction possible).

        学习目标:
        - 理解单特征数据无法降维
        - 至少需要2个特征
        """
        X = simple_2d_data[['feature1']].values

        pca = PCA(n_components=1)
        pca.fit(X)

        # With 1 feature, PCA just returns (scaled) data
        assert pca.n_components_ == 1
        assert pca.explained_variance_ratio_[0] == 1.0, \
            "单特征的 PCA 应解释100%的方差"

    def test_pca_with_curse_of_dimensionality(self, high_dim_curse_data):
        """
        Edge case: PCA when p >> n (curse of dimensionality).

        学习目标:
        - 理解当特征数 > 样本数时的 PCA 限制
        - 最多只能得到 n 个主成分（而不是 p 个）
        """
        X = high_dim_curse_data.values
        n_samples = X.shape[0]

        # Fit PCA
        pca = PCA()
        pca.fit(X)

        # Number of components limited by n_samples
        assert pca.n_components_ == n_samples, \
            "当 p > n 时，主成分数最多为 n"

    def test_pca_minimum_sample_size(self):
        """
        Edge case: PCA with minimum sample size.

        学习目标:
        - 理解 PCA 至少需要 2 个样本
        """
        # Create minimal data
        X = np.array([[1, 2], [3, 4]], dtype=float)

        pca = PCA(n_components=1)
        pca.fit(X)

        # Should fit (2 samples, 2 features)
        assert pca.n_components_ == 1

    def test_pca_with_empty_data(self, empty_data):
        """
        Edge case: PCA with empty data should raise error.

        学习目标:
        - 理解数据验证的重要性
        """
        X = empty_data.values

        pca = PCA(n_components=1)

        # Should raise error
        with pytest.raises(ValueError):
            pca.fit(X)


# =============================================================================
# Test 5: PCA vs Feature Selection
# =============================================================================

class TestPCAvsFeatureSelection:
    """Test understanding of difference between PCA and feature selection."""

    def test_pca_creates_new_features(self, simple_2d_data):
        """
        Test: PCA creates NEW features (linear combinations).

        学习目标:
        - 理解 PCA 是"特征提取"，不是"特征选择"
        - 主成分是原始特征的线性组合
        """
        X = simple_2d_data.values

        pca = PCA(n_components=1)
        X_transformed = pca.fit_transform(X)

        # Transformed data is NOT a subset of original features
        # It's a linear combination
        assert not np.allclose(X_transformed, X[:, [0]]) and \
               not np.allclose(X_transformed, X[:, [1]]), \
            "PCA 应创建新特征（线性组合），不是选择原始特征的子集"

    def test_pca_loadings_determine_combination(self, simple_2d_data):
        """
        Test: Feature loadings define the linear combination.

        学习目标:
        - 理解 PC = loading1 * feature1 + loading2 * feature2 + ...
        """
        X = StandardScaler().fit_transform(simple_2d_data)

        pca = PCA(n_components=1)
        pca.fit(X)

        # Manual calculation: PC1 = loading1 * f1 + loading2 * f2
        loadings = pca.components_[0]
        X_manual = X @ loadings.reshape(-1, 1)

        X_pca = pca.transform(X)

        # Should match
        assert np.allclose(X_pca, X_manual, atol=1e-10), \
            "主成分应等于特征载荷的线性组合"


# =============================================================================
# Test 6: AI Report Review for PCA
# =============================================================================

class TestAIPCAReportReview:
    """Test ability to review AI-generated PCA reports."""

    def test_check_good_pca_report(self, good_pca_report):
        """
        Happy path: Identify a complete PCA report.

        学习目标:
        - 理解完整 PCA 报告应包含的要素
        - 方差解释比例、成分数、阈值检查
        """
        report = good_pca_report.lower()

        # Required elements
        required = ['方差', '成分', '主成分', '阈值', '保留']

        missing = [elem for elem in required if elem not in report]
        assert len(missing) <= 1, \
            f"合格的 PCA 报告应包含关键要素，缺少: {missing}"

    def test_detect_missing_variance_threshold_check(self, bad_pca_report_no_variance_check):
        """
        Test: Identify report missing variance threshold check.

        学习目标:
        - 理解方差阈值检查的重要性
        - 报告应说明选择了多少主成分及原因
        """
        report = bad_pca_report_no_variance_check.lower()

        # Missing key elements
        has_variance_threshold = any(word in report for word in
                                   ['85%', '阈值', '累积', '方差解释'])

        assert not has_variance_threshold, \
            "应该检测到报告缺少方差阈值检查"

    def test_detect_unclear_component_selection(self, bad_pca_report_no_variance_check):
        """
        Test: Identify report without clear component selection rationale.

        学习目标:
        - 理解报告应说明"为什么选择这么多主成分"
        - 不能只说"选了10个"
        """
        report = bad_pca_report_no_variance_check

        # Has component count but no rationale
        has_count = any(char.isdigit() for char in report)
        has_rationale = any(word in report for word in
                          ['方差', '阈值', '保留', '%', '累积'])

        assert has_count and not has_rationale, \
            "报告应说明主成分选择的理由（基于方差阈值）"
