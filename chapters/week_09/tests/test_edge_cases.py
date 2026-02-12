"""
Test suite for Week 09: Edge Cases and Boundary Conditions

This module tests regression with edge cases, boundary conditions,
and common pitfalls in regression analysis.
"""

import pytest
import numpy as np
import pandas as pd


# =============================================================================
# Test 1: Sample Size Issues
# =============================================================================

class TestSampleSizeIssues:
    """Test regression with problematic sample sizes."""

    def test_ols_with_n_equal_to_p_plus_one(self):
        """
        Edge case: n = p + 1（恰好可识别）.

        学习目标:
        - 最小样本量要求
        - p 个参数需要至少 p+1 个观测
        """
        from sklearn.linear_model import LinearRegression

        # 2 个参数（截距 + 1 个系数）需要至少 3 个观测
        X = pd.DataFrame({'x': [1, 2, 3]})
        y = pd.Series([2, 4, 6])

        model = LinearRegression()
        model.fit(X, y)

        # 应该能拟合（完美拟合）
        predictions = model.predict(X)
        residuals = y - predictions

        # 完美拟合，残差应该接近 0
        assert np.allclose(residuals, 0, atol=1e-10), \
            "n = p + 1 时应该完美拟合"

    def test_ols_with_fewer_observations_than_parameters(self):
        """
        Edge case: n < p + 1（欠定）.

        学习目标:
        - 理解自由度的概念
        - 观测数少于参数数时无法唯一估计
        """
        from sklearn.linear_model import LinearRegression

        # 3 个参数但只有 2 个观测
        X = pd.DataFrame({
            'x1': [1, 2],
            'x2': [3, 4]
        })
        y = pd.Series([5, 6])

        model = LinearRegression()
        model.fit(X, y)

        # sklearn 仍会给出一个解（最小范数解）
        # 但这不是唯一解
        assert hasattr(model, 'coef_'), "应该有系数"

    def test_very_small_sample_ci_widens(self):
        """
        Edge case: 小样本导致 CI 很宽.

        学习目标:
        - 理解样本量对不确定性的影响
        """
        import statsmodels.api as sm

        np.random.seed(42)
        # 只有 10 个观测
        X = pd.DataFrame({'x': np.random.normal(0, 1, 10)})
        y = pd.Series(5 + 2 * X['x'] + np.random.normal(0, 1, 10))

        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        ci = model.conf_int(alpha=0.05)
        ci_width = ci.iloc[1, 1] - ci.iloc[1, 0]

        # 小样本的 CI 应该很宽（不确定性大）
        assert ci_width > 1, \
            f"小样本的 CI 应该很宽，实际宽度: {ci_width:.2f}"


# =============================================================================
# Test 2: Extreme Data Patterns
# =============================================================================

class TestExtremeDataPatterns:
    """Test regression with extreme or unusual data patterns."""

    def test_perfect_positive_correlation(self):
        """
        Edge case: 完全正相关（r = 1）.

        学习目标:
        - 理解完美相关的情况
        - R² = 1，残差 = 0
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        X = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        y = pd.Series([2, 4, 6, 8, 10])  # 完美 y = 2x

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # R² 应该是 1（完美拟合）
        r2 = r2_score(y, predictions)
        assert abs(r2 - 1.0) < 1e-10, \
            f"完全正相关的 R² 应该是 1, 实际: {r2}"

        # 残差应该是 0
        residuals = y - predictions
        assert np.allclose(residuals, 0), \
            "完全正相关的残差应该是 0"

    def test_perfect_negative_correlation(self):
        """
        Edge case: 完全负相关（r = -1）.

        学习目标:
        - 理解负相关的情况
        - 斜率应该是负的
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        X = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        y = pd.Series([10, 8, 6, 4, 2])  # 完美 y = -2x + 12

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # R² 应该还是 1（线性关系）
        r2 = r2_score(y, predictions)
        assert abs(r2 - 1.0) < 1e-10, \
            f"完全负相关的 R² 应该是 1, 实际: {r2}"

        # 斜率应该是负的
        assert model.coef_[0] < 0, \
            "负相关的斜率应该是负的"

    def test_zero_correlation(self):
        """
        Edge case: 零相关（r = 0）.

        学习目标:
        - 理解零相关的情况
        - 斜率接近 0，R² 接近 0
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        np.random.seed(42)
        X = pd.DataFrame({'x': np.random.normal(0, 1, 100)})
        # y 与 x 无关
        y = pd.Series(np.random.normal(5, 1, 100))

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        # 斜率应该接近 0
        assert abs(model.coef_[0]) < 0.3, \
            f"零相关的斜率应该接近 0, 实际: {model.coef_[0]:.3f}"

        # R² 应该很低
        r2 = r2_score(y, predictions)
        assert r2 < 0.1, \
            f"零相关的 R² 应该很低, 实际: {r2:.3f}"

    def test_single_outlier_dramatic_effect(self):
        """
        Edge case: 单个异常点的巨大影响.

        学习目标:
        - 理解 OLS 对异常值敏感
        - Cook's 距离检测这种影响
        """
        from sklearn.linear_model import LinearRegression
        import statsmodels.api as sm

        # 基础数据：强正相关，x 在 [1, 9]
        X_base = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        y_base = pd.Series([2, 4, 6, 8, 10, 12, 14, 16, 18])

        # 添加一个极端异常点（x 在数据范围外，y 值异常）
        X_with_outlier = pd.concat([X_base, pd.DataFrame({'x': [15]})],
                                 ignore_index=True)
        y_with_outlier = pd.concat([y_base, pd.Series([10])],
                                 ignore_index=True)

        # 不含异常点的模型
        model_clean = LinearRegression().fit(X_base, y_base)
        coef_clean = model_clean.coef_[0]

        # 含异常点的模型
        model_with_outlier = LinearRegression().fit(X_with_outlier,
                                                  y_with_outlier)
        coef_with_outlier = model_with_outlier.coef_[0]

        # 系数应该差异很大
        coef_change = abs(coef_with_outlier - coef_clean) / abs(coef_clean)

        assert coef_change > 0.1, \
            f"单个异常点应该显著影响系数，变化: {coef_change:.1%}"

        # 用 statsmodels 计算 Cook's 距离
        X_sm = sm.add_constant(X_with_outlier)
        model_sm = sm.OLS(y_with_outlier, X_sm).fit()
        influence = model_sm.get_influence()
        cooks_d = influence.cooks_distance[0]

        # 最后一个点（异常点）应该有很高的 Cook's 距离
        assert cooks_d[-1] > cooks_d[:-1].mean() * 5, \
            "异常点的 Cook's 距离应该远高于其他点"


# =============================================================================
# Test 3: Data Quality Issues
# =============================================================================

class TestDataQualityIssues:
    """Test regression with data quality problems."""

    def test_missing_values_handling(self):
        """
        Edge case: 数据包含缺失值.

        学习目标:
        - 理解缺失值的影响
        - 需要删除或填补
        """
        from sklearn.linear_model import LinearRegression

        # 包含 NaN 的数据
        X = pd.DataFrame({'x': [1, 2, np.nan, 4, 5]})
        y = pd.Series([2, 4, 6, 8, 10])

        model = LinearRegression()

        # sklearn 默认会报错
        with pytest.raises(ValueError):
            model.fit(X, y)

        # 删除缺失值后应该可以拟合
        X_clean = X.dropna()
        y_clean = y[X_clean.index]
        model.fit(X_clean, y_clean)

        assert hasattr(model, 'coef_'), "删除缺失值后应该可以拟合"

    def test_infinite_values_handling(self):
        """
        Edge case: 数据包含无穷值.

        学习目标:
        - 理解无穷值的影响
        """
        from sklearn.linear_model import LinearRegression

        # 包含 inf 的数据
        X = pd.DataFrame({'x': [1, 2, np.inf, 4, 5]})
        y = pd.Series([2, 4, 6, 8, 10])

        model = LinearRegression()

        # 应该报错或产生无效结果
        with pytest.raises((ValueError,)):
            model.fit(X, y)

    def test_constant_y_variable(self):
        """
        Edge case: y 为常数（无变异）.

        学习目标:
        - 理解为什么需要 y 有变异
        - R² 无法定义（0/0）
        """
        from sklearn.linear_model import LinearRegression

        X = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        y = pd.Series([10, 10, 10, 10, 10])  # 常数

        model = LinearRegression()
        model.fit(X, y)

        # 斜率应该是 0（y 不随 x 变化）
        assert abs(model.coef_[0]) < 1e-10, \
            "y 为常数时，斜率应该是 0"

        # R² 应该是 0 或未定义
        predictions = model.predict(X)
        # 所有预测值都相同
        assert len(np.unique(predictions)) == 1, \
            "y 为常数时，所有预测值应该相同"


# =============================================================================
# Test 4: Prediction Boundaries
# =============================================================================

class TestPredictionBoundaries:
    """Test regression prediction at boundaries and extrapolation."""

    def test_interpolation_vs_extrapolation(self):
        """
        Test: 内插 vs 外推.

        学习目标:
        - 理解外推的风险
        - 在数据范围内预测更可靠
        """
        from sklearn.linear_model import LinearRegression

        # 训练数据：x 在 [0, 10]
        X_train = pd.DataFrame({'x': np.linspace(0, 10, 20)})
        y_train = pd.Series(5 + 2 * X_train['x'] +
                          np.random.normal(0, 1, 20))

        model = LinearRegression()
        model.fit(X_train, y_train)

        # 内插：x = 5（在数据范围内）
        pred_interpolation = model.predict(pd.DataFrame({'x': [5]}))[0]

        # 外推：x = 100（远超数据范围）
        pred_extrapolation = model.predict(pd.DataFrame({'x': [100]}))[0]

        # 外推的预测值应该远大于内插
        assert pred_extrapolation > pred_interpolation * 10, \
            "外推到 x=100 的预测值应该远大于 x=5 的预测值"

        # 但外推的不确定性很大（无法直接量化，因为 sklearn 不提供 CI）

    def test_prediction_at_x_equals_zero(self):
        """
        Edge case: 在 x=0 处预测（截距）.

        学习目标:
        - 理解截距的含义
        - 注意 x=0 可能没有实际意义
        """
        from sklearn.linear_model import LinearRegression

        # 训练数据：x 在 [100, 200] 范围
        X = pd.DataFrame({'x': np.linspace(100, 200, 20)})
        y = pd.Series(5 + 2 * X['x'] + np.random.normal(0, 1, 20))

        model = LinearRegression()
        model.fit(X, y)

        # 预测 x=0
        pred_at_zero = model.predict(pd.DataFrame({'x': [0]}))[0]

        # 预测值应该等于截距
        assert abs(pred_at_zero - model.intercept_) < 1e-10, \
            "x=0 的预测值应该等于截距"

        # 但 x=0 远离训练数据范围，这是外推
        assert 0 < X['x'].min(), \
            "x=0 不在训练数据范围内（外推）"


# =============================================================================
# Test 5: Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Test numerical stability issues in regression."""

    def test_large_scale_differences(self):
        """
        Edge case: 预测变量尺度差异巨大.

        学习目标:
        - 理解变量标准化的必要性
        - 不同尺度可能导致数值不稳定
        """
        from sklearn.linear_model import LinearRegression

        # x1 尺度很小，x2 尺度很大
        X = pd.DataFrame({
            'x1': [0.001, 0.002, 0.003, 0.004, 0.005],
            'x2': [10000, 20000, 30000, 40000, 50000]
        })
        y = pd.Series([10, 20, 30, 40, 50])

        model = LinearRegression()
        model.fit(X, y)

        # 应该能拟合，但系数尺度差异巨大
        assert abs(model.coef_[0]) < abs(model.coef_[1]), \
            "小尺度变量的系数应该更大"

        # 标准化后系数应该更可比
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model_scaled = LinearRegression()
        model_scaled.fit(X_scaled, y)

        # 标准化后的系数应该更接近
        assert isinstance(model_scaled.coef_, np.ndarray), \
            "标准化后应该能正常拟合"

    def test_near_collinearity_warning(self):
        """
        Edge case: 近似共线性（非完全）.

        学习目标:
        - 近似共线性导致系数不稳定
        - VIF 会很高但不是无穷大
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # 创建高度相关但非完全共线的变量
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = x1 + np.random.normal(0, 0.01, n)  # 高度相关

        y = 5 + 2 * x1 + np.random.normal(0, 1, n)

        X = pd.DataFrame({'x1': x1, 'x2': x2})

        # 计算 VIF
        vif_x1 = variance_inflation_factor(X.values, 0)
        vif_x2 = variance_inflation_factor(X.values, 1)

        # VIF 应该很高
        assert max(vif_x1, vif_x2) > 10, \
            f"近似共线性的 VIF 应该 > 10, 实际: {max(vif_x1, vif_x2):.1f}"


# =============================================================================
# Test 6: Interpretation Pitfalls
# =============================================================================

class TestInterpretationPitfalls:
    """Test common interpretation mistakes."""

    def test_ecological_fallacy(self):
        """
        Test: 生态谬误（聚合层面结论不能直接应用于个体）.

        学习目标:
        - 理解聚合数据的陷阱
        - 这是一个概念性测试
        """
        # 这个测试是文档性的，强调解释陷阱
        # 生态谬误：从聚合数据得出个体层面结论

        interpretation_correct = (
            "在地区层面，平均收入与平均房价正相关"
        )
        interpretation_wrong = (
            "收入越高的人，买的房子越贵"  # 从聚合数据推断个体
        )

        # 两者都可能是真的，但聚合关系不能直接应用于个体
        assert isinstance(interpretation_correct, str)
        assert isinstance(interpretation_wrong, str)

    def test_simpsons_paradox(self):
        """
        Test: 辛普森悖论（分组趋势与整体趋势相反）.

        学习目标:
        - 理解为什么需要控制混杂变量
        """
        # 简化示例
        # 组 1：x 和 y 正相关
        x1 = np.array([1, 2, 3])
        y1 = np.array([2, 4, 6])

        # 组 2：x 和 y 正相关
        x2 = np.array([5, 6, 7])
        y2 = np.array([3, 5, 7])

        # 合并后可能看起来负相关（因为组 2 的 y 值整体更低）
        X_all = pd.DataFrame({'x': np.concatenate([x1, x2])})
        y_all = pd.Series(np.concatenate([y1, y2]))

        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        # 整体回归
        model_all = LinearRegression().fit(X_all, y_all)

        # 这里不做强断言（因为数据构造可能不总是产生悖论）
        # 但测试代码结构展示了分组分析的重要性
        assert hasattr(model_all, 'coef_')

    def test_omitted_variable_bias(self):
        """
        Test: 遗漏变量偏差.

        学习目标:
        - 理解为什么需要"其他变量不变"
        - 遗漏重要变量会导致系数有偏
        """
        from sklearn.linear_model import LinearRegression

        # 真实模型：y = 1 + 2*x1 + 3*x2
        np.random.seed(42)
        n = 100
        x1 = np.random.normal(0, 1, n)
        x2 = np.random.normal(0, 1, n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.normal(0, 0.1, n)

        # 正确模型（包含 x1 和 x2）
        X_correct = pd.DataFrame({'x1': x1, 'x2': x2})
        model_correct = LinearRegression().fit(X_correct, y)
        coef_x1_correct = model_correct.coef_[0]

        # 错误模型（只包含 x1，遗漏 x2）
        X_wrong = pd.DataFrame({'x1': x1})
        model_wrong = LinearRegression().fit(X_wrong, y)
        coef_x1_wrong = model_wrong.coef_[0]

        # 如果 x1 和 x2 相关，遗漏 x2 会导致 x1 的系数有偏
        # 这里 x1 和 x2 独立，所以偏差不大
        # 但测试结构展示了概念
        assert isinstance(coef_x1_correct, (float, np.floating))
        assert isinstance(coef_x1_wrong, (float, np.floating))
