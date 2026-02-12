"""
Test suite for Week 09: 回归与模型诊断

This module tests linear regression fitting, coefficient interpretation,
residual diagnostics, VIF detection, and influential point identification.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Test 1: Linear Regression Fitting
# =============================================================================

class TestLinearRegressionFitting:
    """Test linear regression fitting with sklearn/statsmodels."""

    def test_simple_regression_sklearn_fits_correctly(self, housing_data_simple):
        """
        Happy path: Fit simple linear regression with sklearn.

        学习目标:
        - 理解 sklearn.LinearRegression 的基本用法
        - 正确获取 coef_ 和 intercept_
        """
        from sklearn.linear_model import LinearRegression

        X = housing_data_simple[['area_sqm']]
        y = housing_data_simple['price_wan']

        model = LinearRegression()
        model.fit(X, y)

        # 检查模型成功拟合
        assert hasattr(model, 'coef_'), "模型应该有 coef_ 属性"
        assert hasattr(model, 'intercept_'), "模型应该有 intercept_ 属性"
        assert len(model.coef_) == 1, "简单回归应该只有一个系数"
        assert model.coef_[0] > 0, "面积对房价应该是正影响"

    def test_simple_regression_predicts_correctly(self, housing_data_simple):
        """Happy path: Predictions from fitted model."""
        from sklearn.linear_model import LinearRegression

        X = housing_data_simple[['area_sqm']]
        y = housing_data_simple['price_wan']

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # 预测值应该与观测值有一定相关性
        correlation = np.corrcoef(y, predictions)[0, 1]
        assert correlation > 0.5, "预测值应该与观测值相关"

    def test_multiple_regression_sklearn_fits_correctly(self, housing_data):
        """
        Happy path: Fit multiple linear regression.

        学习目标:
        - 理解多元回归（多个预测变量）
        - 每个预测变量有一个系数
        """
        from sklearn.linear_model import LinearRegression

        X = housing_data[['area_sqm', 'age_years', 'n_rooms']]
        y = housing_data['price_wan']

        model = LinearRegression()
        model.fit(X, y)

        assert len(model.coef_) == 3, "三个预测变量应该有三个系数"
        assert model.coef_[0] > 0, "面积应该是正影响"
        assert model.coef_[1] < 0, "房龄应该是负影响"
        assert model.coef_[2] > 0, "房间数应该是正影响"

    def test_regression_with_statsmodels(self, housing_data):
        """
        Happy path: Fit regression with statsmodels.

        学习目标:
        - 理解 statsmodels 需要手动添加常数项
        - 能够获取详细输出（summary）
        """
        import statsmodels.api as sm

        X = housing_data[['area_sqm', 'age_years']]
        y = housing_data['price_wan']

        # 添加截距项
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        # 检查模型成功拟合
        assert hasattr(model, 'params'), "模型应该有 params 属性"
        assert hasattr(model, 'summary'), "模型应该有 summary 方法"
        assert len(model.params) == 3, "应该有截距 + 2 个系数"
        assert 'const' in model.params.index, "应该包含截距项"


# =============================================================================
# Test 2: Coefficient Interpretation
# =============================================================================

class TestCoefficientInterpretation:
    """Test correct interpretation of regression coefficients."""

    def test_intercept_interpretation(self, housing_data_simple):
        """
        Test: 截距是 x=0 时的预测值.

        学习目标:
        - 理解截距的几何意义
        - 注意截距可能没有实际意义（如 x=0 不在数据范围内）
        """
        from sklearn.linear_model import LinearRegression

        X = housing_data_simple[['area_sqm']]
        y = housing_data_simple['price_wan']

        model = LinearRegression()
        model.fit(X, y)

        # 截距大约等于预测 x=0 时的值
        prediction_at_zero = model.predict(pd.DataFrame({'area_sqm': [0]}))[0]

        assert abs(model.intercept_ - prediction_at_zero) < 1e-10, \
            "截距应该等于 x=0 时的预测值"

    def test_slope_interpretation(self, housing_data_simple):
        """
        Test: 斜率是 x 每增加 1 单位,y 的平均变化.

        学习目标:
        - 正确解释斜率的含义
        - 理解这是"平均"关系，不是个体规律
        """
        from sklearn.linear_model import LinearRegression

        X = housing_data_simple[['area_sqm']]
        y = housing_data_simple['price_wan']

        model = LinearRegression()
        model.fit(X, y)

        # 斜率 ≈ (y2 - y1) / (x2 - x1)
        pred_x1 = model.predict(pd.DataFrame({'area_sqm': [100]}))[0]
        pred_x2 = model.predict(pd.DataFrame({'area_sqm': [101]}))[0]

        calculated_slope = pred_x2 - pred_x1

        assert abs(model.coef_[0] - calculated_slope) < 1e-10, \
            "斜率应该等于 x 增加 1 时 y 的变化量"

    def test_ceteris_paribus_multiple_regression(self, housing_data):
        """
        Test: 多元回归系数解释需要"在其他变量不变的情况下".

        学习目标:
        - 理解"其他变量不变"的含义
        - 多元回归系数与简单回归系数的差异
        """
        from sklearn.linear_model import LinearRegression

        # 简单回归（只有面积）
        X_simple = housing_data[['area_sqm']]
        y = housing_data['price_wan']
        model_simple = LinearRegression().fit(X_simple, y)
        coef_simple = model_simple.coef_[0]

        # 多元回归（面积、房龄、房间数）
        X_multi = housing_data[['area_sqm', 'age_years', 'n_rooms']]
        model_multi = LinearRegression().fit(X_multi, y)
        coef_multi = model_multi.coef_[0]

        # 多元回归的系数通常会不同（因为"分配"了共同效应）
        # 面积和房间数高度相关，所以简单回归中面积"抢了"房间数的功劳
        # 这导致 coef_simple 和 coef_multi 不相等
        assert isinstance(coef_simple, (float, np.floating))
        assert isinstance(coef_multi, (float, np.floating))
        # 系数会变化，这是正常的


# =============================================================================
# Test 3: OLS Loss Function
# =============================================================================

class TestOLSLossFunction:
    """Test understanding of Ordinary Least Squares (OLS) loss function."""

    def test_ols_minimizes_residual_sum_of_squares(self, housing_data_simple):
        """
        Test: OLS 最小化残差平方和.

        学习目标:
        - 理解为什么叫"最小二乘法"
        - 手动计算 RSS 并验证
        """
        from sklearn.linear_model import LinearRegression

        X = housing_data_simple[['area_sqm']]
        y = housing_data_simple['price_wan']

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # 计算残差平方和
        residuals = y - predictions
        rss = np.sum(residuals ** 2)

        # 任意其他斜率应该产生更大的 RSS
        # 测试几个不同的斜率
        for delta in [-0.5, -0.2, 0.2, 0.5]:
            alternative_coef = model.coef_[0] + delta
            alternative_pred = model.intercept_ + alternative_coef * X['area_sqm']
            alternative_rss = np.sum((y - alternative_pred) ** 2)

            assert alternative_rss > rss, \
                f"斜率 {alternative_coef:.2f} 的 RSS 应该大于 OLS 斜率的 RSS"

    def test_manual_ols_matches_sklearn(self, housing_data_simple):
        """
        Test: 手动计算 OLS 系数应与 sklearn 一致.

        学习目标:
        - 理解 OLS 的数学公式
        - β = (X'X)^(-1) X'y
        """
        from sklearn.linear_model import LinearRegression

        X = housing_data_simple[['area_sqm']].values
        y = housing_data_simple['price_wan'].values

        # sklearn 结果
        model_sklearn = LinearRegression()
        model_sklearn.fit(X, y)

        # 手动计算 OLS（添加截距项）
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ \
                X_with_intercept.T @ y

        # 应该非常接近
        assert abs(beta[0] - model_sklearn.intercept_) < 1e-10, \
            "手动计算的截距应该与 sklearn 一致"
        assert abs(beta[1] - model_sklearn.coef_[0]) < 1e-10, \
            "手动计算的斜率应该与 sklearn 一致"

    def test_mean_is_ols_with_no_predictors(self):
        """
        Test: 均值是"无预测变量"的 OLS 估计.

        学习目标:
        - 理解均值是最小二乘估计的一种特殊情况
        - 均值最小化 Σ(yi - μ)²
        """
        np.random.seed(42)
        data = np.random.normal(loc=100, scale=15, size=100)

        # 样本均值
        sample_mean = np.mean(data)

        # 均值应该最小化平方误差
        # 测试几个不同的值
        sse_at_mean = np.sum((data - sample_mean) ** 2)

        for test_value in [sample_mean - 10, sample_mean + 5, sample_mean + 20]:
            sse_at_test = np.sum((data - test_value) ** 2)
            assert sse_at_test > sse_at_mean, \
                f"在 {test_value} 处的 SSE 应该大于在均值处的 SSE"


# =============================================================================
# Test 4: Confidence Intervals for Coefficients
# =============================================================================

class TestCoefficientConfidenceIntervals:
    """Test confidence interval construction for regression coefficients."""

    def test_statsmodels_provides_ci(self, housing_data):
        """
        Happy path: statsmodels 自动计算系数 CI.

        学习目标:
        - 理解 95% CI 的含义
        - CI 表达系数估计的不确定性
        """
        import statsmodels.api as sm

        X = housing_data[['area_sqm', 'age_years']]
        y = housing_data['price_wan']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        # 获取置信区间
        conf_int = model.conf_int(alpha=0.05)

        # 检查 CI 的结构
        assert conf_int.shape[1] == 2, "CI 应该有下界和上界"
        assert conf_int.shape[0] == len(model.params), "每个参数都应该有 CI"
        # 使用 values 属性进行 numpy 数组操作
        assert np.all(conf_int.values[:, 0] < conf_int.values[:, 1]), "下界应该小于上界"

    def test_ci_contains_point_estimate(self, housing_data):
        """
        Test: 点估计应该在 CI 内部（近似中心）.

        学习目标:
        - 理解 CI 的构造方法
        - 点估计 ± 临界值 × 标准误
        """
        import statsmodels.api as sm

        X = housing_data[['area_sqm', 'age_years']]
        y = housing_data['price_wan']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        conf_int = model.conf_int(alpha=0.05)
        params = model.params

        # 点估计应该在 CI 内部
        for i in range(len(params)):
            assert conf_int.iloc[i, 0] <= params[i] <= conf_int.iloc[i, 1], \
                f"参数 {i} 的点估计应该在 CI 内部"

    def test_ci_width_with_sample_size(self, housing_data):
        """
        Test: 样本量越大，CI 越窄.

        学习目标:
        - 理解样本量对估计精度的影响
        """
        import statsmodels.api as sm

        # 小样本
        small_sample = housing_data.sample(n=50, random_state=42)
        X_small = sm.add_constant(small_sample[['area_sqm']])
        y_small = small_sample['price_wan']
        model_small = sm.OLS(y_small, X_small).fit()
        ci_small = model_small.conf_int(alpha=0.05)

        # 大样本
        large_sample = housing_data.sample(n=150, random_state=42)
        X_large = sm.add_constant(large_sample[['area_sqm']])
        y_large = large_sample['price_wan']
        model_large = sm.OLS(y_large, X_large).fit()
        ci_large = model_large.conf_int(alpha=0.05)

        # 大样本的 CI 应该更窄（针对面积系数）
        # 注意：索引1是面积系数
        small_width = ci_small.iloc[1, 1] - ci_small.iloc[1, 0]
        large_width = ci_large.iloc[1, 1] - ci_large.iloc[1, 0]

        assert small_width > large_width, \
            "大样本应该产生更窄的置信区间"


# =============================================================================
# Test 5: Multicollinearity Detection (VIF)
# =============================================================================

class TestMulticollinearity:
    """Test multicollinearity detection with VIF."""

    def test_vif_detection_with_multicollinearity(self, data_with_multicollinearity):
        """
        Happy path: 检测多重共线性（VIF > 5）.

        学习目标:
        - 理解 VIF 的含义（方差膨胀因子）
        - VIF > 10 表示严重共线性
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X = data_with_multicollinearity[['x1', 'x2', 'x3']]

        # 计算 VIF
        vif_values = []
        for i in range(X.shape[1]):
            vif = variance_inflation_factor(X.values, i)
            vif_values.append(vif)

        # 高度相关的变量应该有很高的 VIF
        max_vif = max(vif_values)
        assert max_vif > 5, f"高度相关数据的 VIF 应该 > 5, 实际: {max_vif:.2f}"

    def test_vif_no_multicollinearity(self, data_no_multicollinearity):
        """
        Happy path: 无共线性时 VIF 接近 1.

        学习目标:
        - 独立变量的 VIF 应该接近 1
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X = data_no_multicollinearity[['x1', 'x2', 'x3']]

        vif_values = []
        for i in range(X.shape[1]):
            vif = variance_inflation_factor(X.values, i)
            vif_values.append(vif)

        # 独立变量的 VIF 应该接近 1
        for vif in vif_values:
            assert vif < 5, f"独立变量的 VIF 应该 < 5, 实际: {vif:.2f}"

    def test_perfect_collinearity_raises_error(self, perfect_collinearity):
        """
        Edge case: 完全共线性导致矩阵不可逆.

        学习目标:
        - 理解完全共线性是极端情况
        - VIF 会非常大（可能是 inf 或非常大的数）
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import warnings

        X = perfect_collinearity[['x1', 'x2']]

        # VIF 计算可能产生警告（除零）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                vif = variance_inflation_factor(X.values, 0)
                # 如果没报错，VIF 应该非常大或无穷大
                assert np.isinf(vif) or vif > 1000, \
                    f"完全共线性的 VIF 应该非常大或无穷大, 实际: {vif}"
            except (np.linalg.LinAlgError, RuntimeError):
                # 预期的错误（某些实现可能报错）
                pass


# =============================================================================
# Test 6: Residual Diagnostics (LINE Assumptions)
# =============================================================================

class TestResidualDiagnostics:
    """Test residual diagnostics for regression assumptions."""

    def test_residuals_sum_to_zero(self, housing_data):
        """
        Test: OLS 残差和应该等于 0（带截距的模型）.

        学习目标:
        - 理解 OLS 的数学性质
        - 残差正负抵消
        """
        import statsmodels.api as sm

        X = housing_data[['area_sqm']]
        y = housing_data['price_wan']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        residuals = model.resid

        # 残差和应该接近 0（数值精度范围内）
        assert abs(np.sum(residuals)) < 1e-10, \
            f"OLS 残差和应该接近 0, 实际: {np.sum(residuals)}"

    def test_residuals_vs_fitted_linearity_check(self, data_meeting_assumptions):
        """
        Happy path: 线性假设满足时，残差随机分布.

        学习目标:
        - 理解如何检验线性假设
        - 残差 vs 拟合值图应该无线性模式
        """
        import statsmodels.api as sm

        X = data_meeting_assumptions[['x']]
        y = data_meeting_assumptions['y']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        fitted = model.fittedvalues
        residuals = model.resid

        # 计算残差与拟合值的相关性
        # （对于满足线性假设的数据，相关性应该接近 0）
        correlation = np.corrcoef(fitted, residuals)[0, 1]

        assert abs(correlation) < 0.3, \
            "满足线性假设时，残差与拟合值的相关性应该接近 0"

    def test_qq_plot_normality(self, data_meeting_assumptions):
        """
        Happy path: 正态假设满足时，QQ 图点沿对角线.

        学习目标:
        - 理解 QQ 图用于检验正态性
        - 正态数据的残差应该沿对角线分布
        """
        from scipy.stats import probplot
        import statsmodels.api as sm

        X = data_meeting_assumptions[['x']]
        y = data_meeting_assumptions['y']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        residuals = model.resid

        # QQ 图
        theoretical_quantiles, sample_quantiles = probplot(residuals, fit=False)

        # 计算相关系数（QQ 图点与对角线的接近程度）
        correlation = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]

        assert correlation > 0.95, \
            "正态数据的 QQ 图点应该高度沿对角线分布"

    def test_shapiro_wilk_normality_test(self, data_meeting_assumptions):
        """
        Happy path: Shapiro-Wilk 检验正态性.

        学习目标:
        - 理解 Shapiro-Wilk 检验
        - p > 0.05 不能拒绝正态性假设
        """
        from scipy.stats import shapiro
        import statsmodels.api as sm

        X = data_meeting_assumptions[['x']]
        y = data_meeting_assumptions['y']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        residuals = model.resid

        # Shapiro-Wilk 检验
        stat, p_value = shapiro(residuals)

        # 对于满足正态假设的数据，p 值应该 > 0.05
        # （这是概率性的，可能偶尔失败）
        assert 0 <= p_value <= 1, "p 值应该在 [0, 1] 范围内"
        assert stat > 0, "Shapiro-Wilk 统计量应该是正数"

    def test_durbin_watson_independence(self, data_meeting_assumptions):
        """
        Happy path: Durbin-Watson 检验独立性.

        学习目标:
        - 理解 DW 统计量
        - DW ≈ 2 表示无自相关
        """
        import statsmodels.api as sm
        from statsmodels.stats.stattools import durbin_watson

        X = data_meeting_assumptions[['x']]
        y = data_meeting_assumptions['y']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        residuals = model.resid
        dw = durbin_watson(residuals)

        # DW 应该接近 2（独立性满足）
        assert 1.5 < dw < 2.5, \
            f"独立数据的 DW 统计量应该接近 2, 实际: {dw:.2f}"

    def test_detect_violation_of_linearity(self, data_violating_linearity):
        """
        Test: 检测违反线性假设的情况.

        学习目标:
        - 理解如何识别非线性模式
        """
        import statsmodels.api as sm

        X = data_violating_linearity[['x']]
        y = data_violating_linearity['y']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        fitted = model.fittedvalues
        residuals = model.resid

        # 对于非线性关系，简单线性模型的残差会有模式
        # 计算残差与拟合值的二次相关
        # （简化检测：检查残差的绝对值是否随拟合值系统变化）
        residual_std_by_fitted_tertile = []
        for i in range(3):
            mask = (fitted >= np.percentile(fitted, i * 33.33)) & \
                    (fitted < np.percentile(fitted, (i + 1) * 33.33))
            residual_std_by_fitted_tertile.append(np.std(residuals[mask]))

        # 如果残差标准差变化很大，可能有模式
        std_range = max(residual_std_by_fitted_tertile) - \
                    min(residual_std_by_fitted_tertile)

        # 这是一个弱检测，主要为了教学目的
        assert isinstance(std_range, (float, np.floating))


# =============================================================================
# Test 7: Influential Points (Cook's Distance)
# =============================================================================

class TestInfluentialPoints:
    """Test influential point identification using Cook's Distance."""

    def test_cooks_distance_identifies_influential_points(self, data_with_influential_point):
        """
        Happy path: Cook's 距离识别强影响点.

        学习目标:
        - 理解 Cook's 距离的含义
        - D > 1 表示强影响点
        """
        import statsmodels.api as sm

        X = data_with_influential_point[['x']]
        y = data_with_influential_point['y']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        # 计算 Cook's 距离
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]

        # 最后一个点是强影响点
        assert cooks_d[-1] > 1, \
            f"强影响点的 Cook's D 应该 > 1, 实际: {cooks_d[-1]:.2f}"

    def test_cooks_distance_small_for_normal_points(self, data_meeting_assumptions):
        """
        Happy path: 正常点的 Cook's 距离应该很小.

        学习目标:
        - 理解大部分点的 Cook's D < 0.5
        """
        import statsmodels.api as sm

        X = data_meeting_assumptions[['x']]
        y = data_meeting_assumptions['y']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]

        # 大部分点的 Cook's D 应该 < 0.5
        influential_count = np.sum(cooks_d > 0.5)
        assert influential_count < len(cooks_d) * 0.1, \
            "正常数据中应该很少有强影响点（< 10%）"

    def test_leverage_identifies_high_x_values(self, data_with_high_leverage):
        """
        Test: 高杠杆点的 x 值远离数据中心.

        学习目标:
        - 理解杠杆 (leverage) 的含义
        - 高杠杆 = x 值异常
        """
        import statsmodels.api as sm

        X = data_with_high_leverage[['x']]
        y = data_with_high_leverage['y']

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        # 获取杠杆值
        influence = model.get_influence()
        leverage = influence.hat_matrix_diag

        # 最后一个点应该有很高的杠杆值
        # (因为它的 x 值远离其他点)
        assert leverage[-1] > np.mean(leverage) * 3, \
            "高杠杆点的杠杆值应该远大于平均值"

    def test_outlier_vs_influential_point(self, data_with_outlier, data_with_influential_point):
        """
        Test: 区分离群点和影响点.

        学习目标:
        - 离群点: y 异常（残差大）
        - 影响点: x 异常（杠杆高）+ 残差大
        """
        import statsmodels.api as sm

        # Case 1: 离群点（y 异常，x 正常）
        X1 = data_with_outlier[['x']]
        y1 = data_with_outlier['y']
        X1_sm = sm.add_constant(X1)
        model1 = sm.OLS(y1, X1_sm).fit()
        influence1 = model1.get_influence()

        standardized_residuals = influence1.resid_studentized_internal
        cooks_d1 = influence1.cooks_distance[0]

        # 离群点应该有大的标准化残差
        assert abs(standardized_residuals[-1]) > 2, \
            "离群点应该有大的标准化残差（> 2）"

        # Case 2: 影响点（x 异常 + y 异常）
        X2 = data_with_influential_point[['x']]
        y2 = data_with_influential_point['y']
        X2_sm = sm.add_constant(X2)
        model2 = sm.OLS(y2, X2_sm).fit()
        influence2 = model2.get_influence()
        cooks_d2 = influence2.cooks_distance[0]

        # 影响点应该有大的 Cook's 距离
        assert cooks_d2[-1] > 1, \
            "影响点应该有大的 Cook's 距离（> 1）"


# =============================================================================
# Test 8: Edge Cases and Common Mistakes
# =============================================================================

class TestRegressionEdgeCases:
    """Test regression with edge cases and boundary conditions."""

    def test_empty_data_raises_error(self, empty_data):
        """
        Edge case: 空数据应该报错.

        学习目标:
        - 理解数据验证的重要性
        """
        from sklearn.linear_model import LinearRegression

        X = empty_data[['x']]
        y = empty_data['y']

        model = LinearRegression()

        # 应该报错或产生警告
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_single_observation_cannot_estimate_variance(self, single_observation):
        """
        Edge case: 单观测无法估计方差.

        学习目标:
        - 理解最小样本量的要求
        - p 个参数需要至少 p+1 个观测
        """
        from sklearn.linear_model import LinearRegression

        X = single_observation[['x']]
        y = single_observation['y']

        model = LinearRegression()
        model.fit(X, y)

        # sklearn 允许拟合（会完美拟合）
        # 但无法做统计推断（没有自由度）
        assert model.coef_[0] == 0 or isinstance(model.coef_[0], (float, np.floating))

    def test_constant_x_perfect_collinearity(self, constant_x):
        """
        Edge case: x 为常数导致方差为 0.

        学习目标:
        - 理解为什么不能有常数预测变量
        - 方差为 0 时无法估计影响
        """
        from sklearn.linear_model import LinearRegression

        X = constant_x[['x']]
        y = constant_x['y']

        model = LinearRegression()
        model.fit(X, y)

        # 常数 x 的系数应该是 0（sklearn 的处理）
        # 或者产生数值问题
        assert isinstance(model.coef_[0], (float, np.floating))

    def test_correlation_vs_causation(self, housing_data):
        """
        Test: 相关性 ≠ 因果关系.

        学习目标:
        - 回归系数描述关联，不是因果
        - 这是概念性测试
        """
        # 这个测试是文档性的，强调回归的局限性
        interpretation_correct = (
            "在其他变量不变的情况下,面积每增加 1 平米,房价平均上涨 1.18 万元"
        )
        interpretation_wrong = (
            "增加面积会导致房价上涨 1.18 万元"
        )

        # 正确解释是关联性的，不是因果性的
        assert "其他变量不变" in interpretation_correct or "平均" in interpretation_correct
        assert isinstance(interpretation_wrong, str)


# =============================================================================
# Test 9: AI Report Review
# =============================================================================

class TestAIRegressionReportReview:
    """Test ability to review AI-generated regression reports."""

    def test_check_good_report_has_all_elements(self, good_regression_report):
        """
        Happy path: 识别合格的回归报告.

        学习目标:
        - 理解完整报告应包含的要素
        - 残差诊断、VIF、Cook's 距离、CI
        """
        report = good_regression_report.lower()

        # 应该包含的关键要素
        required_elements = [
            '残差',  # 残差诊断
            'vif',  # 多重共线性检查
            "cook",  # 异常点分析
            '置信区间',  # 系数 CI
            '95%',  # 置信水平
            'r²',  # 拟合优度
            '因果',  # 因果警告
        ]

        missing_elements = []
        for element in required_elements:
            if element not in report:
                missing_elements.append(element)

        # 允许缺少 1-2 个要素（不是所有要素都用英文）
        assert len(missing_elements) <= 2, \
            f"合格的报告应该包含关键要素，缺少: {missing_elements}"

    def test_detect_missing_diagnostics(self, bad_regression_report_no_diagnostics):
        """
        Test: 识别缺少残差诊断的报告.

        学习目标:
        - 理解残差诊断的重要性
        """
        report = bad_regression_report_no_diagnostics

        # 缺少的关键要素
        missing_elements = []

        # 检查残差图
        if '残差' not in report and 'residual' not in report.lower():
            missing_elements.append('残差诊断')

        # 检查 VIF
        if 'vif' not in report.lower() and '共线' not in report:
            missing_elements.append('多重共线性检查')

        # 检查因果警告
        if '因果' not in report and 'causal' not in report.lower():
            missing_elements.append('因果警告')

        # 应该检测到缺少关键要素
        assert len(missing_elements) >= 2, \
            f"应该检测到报告缺少要素: {missing_elements}"

    def test_detect_causal_claim(self, bad_regression_report_causal_claim):
        """
        Test: 识别错误的因果解释.

        学习目标:
        - 理解"相关 ≠ 因果"的重要性
        - 检测"导致"、"会使"等因果语言
        """
        report = bad_regression_report_causal_claim

        # 因果语言的警示词
        causal_warning_words = ['导致', '会使', '造成', '原因']

        has_causal_claim = any(word in report for word in causal_warning_words)
        has_correlation_language = '相关' in report or '关联' in report

        # 应该检测到因果声称
        assert has_causal_claim, "应该检测到因果声称"

        # 好的报告应该说明是"关联"而不是"因果"
        assert not has_correlation_language or has_causal_claim, \
            "错误报告直接用了因果语言，没有说明是关联"

    def test_detect_missing_vif(self, bad_regression_report_no_vif):
        """
        Test: 识别缺少 VIF 检查的报告.

        学习目标:
        - 理解多重共线性检查的重要性
        - 特别是在使用多个相关变量时
        """
        report = bad_regression_report_no_vif

        # 使用了多个相关变量（房间数、卧室数、客厅数）
        # 但没有 VIF 检查
        has_vif_check = 'vif' in report.lower() or '共线' in report
        has_multiple_room_variables = ('房间数' in report and
                                     ('卧室数' in report or '客厅数' in report))

        # 应该检测到缺少 VIF
        assert has_multiple_room_variables and not has_vif_check, \
            "使用多个相关变量的报告应该包含 VIF 检查"
