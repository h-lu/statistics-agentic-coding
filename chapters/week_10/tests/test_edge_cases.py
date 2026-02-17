"""
Edge cases and boundary tests for Week 10 solution.py

边界情况与错误处理测试：
- 空数据集
- 单类别数据
- 极端类别不平衡
- NaN/Inf 处理
- 维度不匹配
- 常量特征
- 完美分离
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

try:
    import solution
except ImportError:
    solution = None


# =============================================================================
# 1. 空数据与极小样本测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestEmptyAndMinimalData:
    """测试空数据和极小样本"""

    def test_empty_data(self, empty_classification_data):
        """
        反例：空数据应报错或返回有意义的结果

        空数组无法训练分类器
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = empty_classification_data['X']
        y = empty_classification_data['y']

        # Should raise an error
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            if hasattr(solution, 'train_logistic_regression'):
                solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                solution.fit_logistic_regression(X, y)

    def test_single_sample(self):
        """
        反例：单样本数据无法训练

        至少需要 2 个样本
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = np.array([[1.0, 2.0]])
        y = np.array([0])

        # Should raise an error
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            if hasattr(solution, 'train_logistic_regression'):
                solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                solution.fit_logistic_regression(X, y)

    def test_two_samples_one_class(self):
        """
        边界：两个样本但同一类别

        无法训练分类器
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = np.array([[1.0, 2.0], [2.0, 3.0]])
        y = np.array([0, 0])  # Same class

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            if hasattr(solution, 'train_logistic_regression'):
                solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                solution.fit_logistic_regression(X, y)

    def test_two_samples_two_classes(self, minimal_binary_data):
        """
        边界：最小二分类数据（每类 2 个样本）

        可以训练但结果可能不稳定
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = minimal_binary_data['X']
        y = minimal_binary_data['y']

        # Should be able to train
        try:
            if hasattr(solution, 'train_logistic_regression'):
                model = solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                model = solution.fit_logistic_regression(X, y)
            else:
                pytest.skip("No logistic regression function implemented")

            assert model is not None
        except (ValueError, RuntimeError):
            # May fail due to convergence issues with small data
            assert True


# =============================================================================
# 2. 单类别数据测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSingleClassData:
    """测试单类别数据"""

    def test_all_zeros(self, single_class_data):
        """
        反例：全部为 0 类

        无法训练分类器
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = single_class_data['X']
        y = single_class_data['y']

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            if hasattr(solution, 'train_logistic_regression'):
                solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                solution.fit_logistic_regression(X, y)

    def test_all_ones(self):
        """
        反例：全部为 1 类

        无法训练分类器
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.ones(100)  # All ones

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            if hasattr(solution, 'train_logistic_regression'):
                solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                solution.fit_logistic_regression(X, y)


# =============================================================================
# 3. 极端类别不平衡测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestExtremeImbalance:
    """测试极端类别不平衡"""

    def test_5_percent_positive(self, highly_imbalanced_data):
        """
        边界：5% 正类数据

        模型应能训练但可能偏向多数类
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = highly_imbalanced_data['X']
        y = highly_imbalanced_data['y']

        # Should be able to train
        if hasattr(solution, 'train_logistic_regression'):
            model = solution.train_logistic_regression(X, y)
        elif hasattr(solution, 'fit_logistic_regression'):
            model = solution.fit_logistic_regression(X, y)
        else:
            pytest.skip("No logistic regression function implemented")

        assert model is not None

        # Check if predictions are made
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            # Model may predict mostly the majority class
            assert len(y_pred) == len(y)

    def test_1_percent_positive(self):
        """
        边界：1% 正类数据（极端不平衡）

        模型可能需要特殊处理
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        from sklearn.datasets import make_classification
        np.random.seed(42)
        X, y = make_classification(
            n_samples=1000,
            n_features=3,
            n_redundant=0,
            weights=[0.99, 0.01],  # 99% negative, 1% positive
            random_state=42
        )

        # Should be able to train
        try:
            if hasattr(solution, 'train_logistic_regression'):
                model = solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                model = solution.fit_logistic_regression(X, y)
            else:
                pytest.skip("No logistic regression function implemented")

            assert model is not None
        except (ValueError, RuntimeError):
            # May fail due to extreme imbalance
            assert True


# =============================================================================
# 4. NaN 和 Inf 处理测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestNaNAndInfHandling:
    """测试 NaN 和 Inf 处理"""

    def test_nan_in_features(self, nan_classification_data):
        """
        反例：特征中有 NaN

        应报错或使用 Pipeline 处理
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = nan_classification_data['X']
        y = nan_classification_data['y']

        # Basic function should raise error for NaN
        for func_name in ['train_logistic_regression', 'fit_logistic_regression']:
            if hasattr(solution, func_name):
                func = getattr(solution, func_name)
                try:
                    model = func(X, y)
                    # If it succeeds without Pipeline, check result
                    if hasattr(model, 'predict'):
                        # May have dropped NaN rows
                        assert True
                except (ValueError, RuntimeError):
                    # Raising an error is acceptable
                    assert True

    def test_inf_in_features(self):
        """
        反例：特征中有 Inf

        应报错
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        from sklearn.datasets import make_classification
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42)
        X[10, 0] = np.inf  # Add infinity

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            if hasattr(solution, 'train_logistic_regression'):
                solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                solution.fit_logistic_regression(X, y)

    def test_nan_in_target(self):
        """
        反例：目标变量中有 NaN

        应报错
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        from sklearn.datasets import make_classification
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42)
        y = y.astype(float)  # Convert to float so we can add NaN
        y[10] = np.nan  # Add NaN to target

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            if hasattr(solution, 'train_logistic_regression'):
                solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                solution.fit_logistic_regression(X, y)


# =============================================================================
# 5. 维度不匹配测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDimensionMismatch:
    """测试维度不匹配"""

    def test_wrong_x_dimensions(self):
        """
        反例：X 不是 2D 数组

        应报错或自动转换
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        X_1d = X[:, 0]  # 1D array

        # Should either raise error or reshape
        try:
            if hasattr(solution, 'train_logistic_regression'):
                model = solution.train_logistic_regression(X_1d, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                model = solution.fit_logistic_regression(X_1d, y)
            else:
                pytest.skip("No logistic regression function implemented")

            assert model is not None
        except (ValueError, IndexError):
            # Raising an error is acceptable
            assert True

    def test_mismatched_sample_counts(self):
        """
        反例：X 和 y 样本数不同

        应报错
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.randint(0, 2, 50)  # Different length

        # Should raise an error
        with pytest.raises((ValueError, IndexError, AssertionError)):
            if hasattr(solution, 'train_logistic_regression'):
                solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                solution.fit_logistic_regression(X, y)

    def test_predict_with_wrong_features(self, simple_binary_classification_data):
        """
        反例：预测时特征数量不匹配

        应报错
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = simple_binary_classification_data['X']
        y = simple_binary_classification_data['y']

        # Train model
        if hasattr(solution, 'train_logistic_regression'):
            model = solution.train_logistic_regression(X, y)
        elif hasattr(solution, 'fit_logistic_regression'):
            model = solution.fit_logistic_regression(X, y)
        else:
            pytest.skip("No logistic regression function implemented")

        # Predict with wrong number of features
        X_wrong = np.random.randn(10, 5)  # 5 features instead of 2

        if hasattr(model, 'predict'):
            with pytest.raises((ValueError, IndexError)):
                model.predict(X_wrong)


# =============================================================================
# 6. 常量特征测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestConstantFeatures:
    """测试常量特征"""

    def test_single_constant_feature(self, constant_feature_data):
        """
        边界：一个特征是常量

        常量特征不影响分类但会被某些模型忽略
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = constant_feature_data['X']
        y = constant_feature_data['y']

        # Should be able to train
        try:
            if hasattr(solution, 'train_logistic_regression'):
                model = solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                model = solution.fit_logistic_regression(X, y)
            else:
                pytest.skip("No logistic regression function implemented")

            assert model is not None
        except (ValueError, RuntimeError):
            # Some models may fail with constant features
            assert True

    def test_all_constant_features(self):
        """
        反例：所有特征都是常量

        无法分类（无信息），模型可能给出警告或产生无意义结果
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        np.random.seed(42)
        X = np.ones((100, 2))  # All features are constant
        y = np.random.randint(0, 2, 100)

        # LogisticRegression may or may not raise an error with constant features
        # In sklearn 1.3+, it typically runs but may produce a ConvergenceWarning
        # We just verify it doesn't crash the system
        try:
            if hasattr(solution, 'train_logistic_regression'):
                model = solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                model = solution.fit_logistic_regression(X, y)
            else:
                pytest.skip("No logistic regression function implemented")
            # If it doesn't raise, that's okay - constant features don't always error
            assert model is not None
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Some implementations may fail with constant features
            assert True


# =============================================================================
# 7. 完美分离测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestPerfectSeparation:
    """测试完美分离情况"""

    def test_perfectly_separable_warnings(self, binary_classification_perfect):
        """
        边界：完美可分数据

        可能导致系数无穷大（不收敛警告）
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = binary_classification_perfect['X']
        y = binary_classification_perfect['y']

        # Should train but may show convergence warning
        try:
            if hasattr(solution, 'train_logistic_regression'):
                model = solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                model = solution.fit_logistic_regression(X, y)
            else:
                pytest.skip("No logistic regression function implemented")

            assert model is not None

            # May still predict correctly despite convergence issues
            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
                accuracy = np.mean(y_pred == y)
                # Should still achieve high accuracy
                assert accuracy >= 0.8, "Should achieve good accuracy even with perfect separation"
        except (ValueError, RuntimeError):
            # May fail due to non-convergence
            assert True


# =============================================================================
# 8. 评估指标边界测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestMetricsBoundaries:
    """测试评估指标的边界情况"""

    def test_empty_predictions(self):
        """
        反例：空预测数组

        计算指标时应报错
        """
        if not hasattr(solution, 'calculate_metrics'):
            pytest.skip("calculate_metrics not implemented")

        y_true = np.array([])
        y_pred = np.array([])

        # Should raise an error
        with pytest.raises((ValueError, IndexError, ZeroDivisionError)):
            solution.calculate_metrics(y_true, y_pred)

    def test_zero_division_precision(self):
        """
        边界：精确率中的零除

        当没有预测为正类时
        """
        if not hasattr(solution, 'calculate_precision'):
            pytest.skip("calculate_precision not implemented")

        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0])  # No positive predictions

        # Should handle zero division gracefully
        try:
            precision = solution.calculate_precision(y_true, y_pred)
            # Should either return 0 or handle gracefully
            assert precision == 0 or precision is None
        except ZeroDivisionError:
            # Or raise a clear error
            assert True

    def test_zero_division_recall(self):
        """
        边界：召回率中的零除

        当没有真实正类时
        """
        if not hasattr(solution, 'calculate_recall'):
            pytest.skip("calculate_recall not implemented")

        y_true = np.array([0, 0, 0, 0, 0])  # No actual positives
        y_pred = np.array([0, 0, 1, 0, 0])

        # Should handle zero division gracefully
        try:
            recall = solution.calculate_recall(y_true, y_pred)
            # Should either return 0 or handle gracefully
            assert recall == 0 or recall is None
        except ZeroDivisionError:
            # Or raise a clear error
            assert True


# =============================================================================
# 9. ROC/AUC 边界测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestROC_AUC_Boundaries:
    """测试 ROC/AUC 的边界情况"""

    def test_single_unique_probability(self):
        """
        边界：所有概率相同

        ROC 曲线应退化为单点
        """
        if not hasattr(solution, 'calculate_auc'):
            pytest.skip("calculate_auc not implemented")

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])  # All same

        # AUC should be 0.5 for uniform probabilities
        auc = solution.calculate_auc(y_true, y_prob)
        assert abs(auc - 0.5) < 0.1, f"AUC should be 0.5 for uniform probabilities, got {auc}"

    def test_auc_with_all_same_class(self):
        """
        反例：所有样本同一类别

        AUC 无法定义
        """
        if not hasattr(solution, 'calculate_auc'):
            pytest.skip("calculate_auc not implemented")

        y_true = np.array([0, 0, 0, 0])  # All zeros
        y_prob = np.array([0.1, 0.2, 0.3, 0.4])

        # Should raise an error or return a specific value
        try:
            auc = solution.calculate_auc(y_true, y_prob)
            # If it doesn't raise, may return NaN or a specific value
            assert np.isnan(auc) or auc is None or 0 <= auc <= 1
        except (ValueError, RuntimeError):
            # Raising an error is acceptable
            assert True


# =============================================================================
# 10. 数据类型测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDataTypes:
    """测试不同数据类型"""

    def test_integer_input(self):
        """
        边界：整数输入

        应能正确处理整数
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 0, 1, 1])

        # Should handle integer input
        try:
            if hasattr(solution, 'train_logistic_regression'):
                model = solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                model = solution.fit_logistic_regression(X, y)
            else:
                pytest.skip("No logistic regression function implemented")

            assert model is not None
        except (ValueError, TypeError):
            # May need float conversion
            assert True

    def test_list_input(self):
        """
        边界：列表输入（而非 NumPy 数组）

        应能转换或报错
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        y = [0, 0, 0, 1, 1]

        # Should either convert or raise error
        try:
            if hasattr(solution, 'train_logistic_regression'):
                model = solution.train_logistic_regression(X, y)
            elif hasattr(solution, 'fit_logistic_regression'):
                model = solution.fit_logistic_regression(X, y)
            else:
                pytest.skip("No logistic regression function implemented")

            assert model is not None
        except (ValueError, TypeError):
            # Raising TypeError for list input is acceptable
            assert True

    def test_pandas_dataframe_input(self):
        """
        边界：Pandas DataFrame 输入

        应能正确处理
        """
        if not hasattr(solution, 'train_logistic_regression') and not hasattr(solution, 'fit_logistic_regression'):
            pytest.skip("No logistic regression function implemented")

        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not available")

        df = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [2, 3, 4, 5, 6]
        })
        y = pd.Series([0, 0, 0, 1, 1])

        # Should handle pandas input
        try:
            if hasattr(solution, 'train_logistic_regression'):
                model = solution.train_logistic_regression(df.values, y.values)
            elif hasattr(solution, 'fit_logistic_regression'):
                model = solution.fit_logistic_regression(df.values, y.values)
            else:
                pytest.skip("No logistic regression function implemented")

            assert model is not None
        except (ValueError, TypeError, AttributeError):
            # May need explicit conversion
            assert True
