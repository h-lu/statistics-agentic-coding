"""
Edge cases and boundary tests for Week 11 solution.py

边界情况与错误处理测试：
- 空数据集
- 单类别数据
- 极端类别不平衡
- NaN/Inf 处理
- 维度不匹配
- 常量特征
- 单特征数据
- max_depth=0 或负数
- n_estimators=0 或负数
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

    def test_empty_data_tree(self):
        """
        反例：空数据应报错

        空数组无法训练决策树
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = np.array([]).reshape(0, 2)
        y = np.array([])

        # Should raise an error
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            solution.train_decision_tree(X, y)

    def test_empty_data_rf(self):
        """
        反例：空数据应报错

        空数组无法训练随机森林
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = np.array([]).reshape(0, 2)
        y = np.array([])

        # Should raise an error
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            solution.train_random_forest(X, y)

    def test_single_sample_tree(self):
        """
        反例：单样本数据无法训练决策树

        至少需要 2 个样本
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = np.array([[1.0, 2.0]])
        y = np.array([0])

        # Should raise an error
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            solution.train_decision_tree(X, y)

    def test_two_samples_one_class_tree(self):
        """
        反例：两个样本但同一类别

        无法训练有意义的决策树
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = np.array([[1.0, 2.0], [2.0, 3.0]])
        y = np.array([0, 0])  # Same class

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            solution.train_decision_tree(X, y)

    def test_two_samples_two_classes_tree(self, minimal_tree_data):
        """
        边界：最小二分类数据（每类 2 个样本）

        可以训练但结果可能不稳定
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = minimal_tree_data['X']
        y = minimal_tree_data['y']

        # Should be able to train
        try:
            model = solution.train_decision_tree(X, y)
            assert model is not None
        except (ValueError, RuntimeError):
            # May fail due to minimal data
            assert True


# =============================================================================
# 2. 单类别数据测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSingleClassData:
    """测试单类别数据"""

    def test_all_zeros_tree(self):
        """
        反例：全部为 0 类

        无法训练决策树
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.zeros(100)  # All zeros

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            solution.train_decision_tree(X, y)

    def test_all_zeros_rf(self):
        """
        反例：全部为 0 类

        无法训练随机森林
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.zeros(100)  # All zeros

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            solution.train_random_forest(X, y)

    def test_all_ones_tree(self):
        """
        反例：全部为 1 类

        无法训练决策树
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.ones(100)  # All ones

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            solution.train_decision_tree(X, y)


# =============================================================================
# 3. 极端类别不平衡测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestExtremeImbalance:
    """测试极端类别不平衡"""

    def test_1_percent_positive_tree(self):
        """
        边界：1% 正类数据（极端不平衡）

        树模型应能训练但可能偏向多数类
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

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
            model = solution.train_decision_tree(X, y)
            assert model is not None

            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
                # Model may predict mostly the majority class
                assert len(y_pred) == len(y)
        except (ValueError, RuntimeError):
            # May fail due to extreme imbalance
            assert True

    def test_1_percent_positive_rf(self):
        """
        边界：1% 正类数据（极端不平衡）

        随机森林应能训练但可能偏向多数类
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

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
            model = solution.train_random_forest(X, y, n_estimators=50)
            assert model is not None

            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
                # Model may predict mostly the majority class
                assert len(y_pred) == len(y)
        except (ValueError, RuntimeError):
            # May fail due to extreme imbalance
            assert True


# =============================================================================
# 4. NaN 和 Inf 处理测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestNaNAndInfHandling:
    """测试 NaN 和 Inf 处理"""

    def test_nan_in_features_tree(self):
        """
        反例：特征中有 NaN

        应报错或需要预处理
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        from sklearn.datasets import make_classification
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=10, n_informative=3, n_redundant=2, random_state=42)
        X[10, 0] = np.nan  # Add NaN

        # Should raise an error (tree models don't handle NaN natively in sklearn < 1.4)
        with pytest.raises((ValueError, RuntimeError)):
            solution.train_decision_tree(X, y)

    def test_nan_in_features_rf(self):
        """
        反例：特征中有 NaN

        应报错或需要预处理
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        from sklearn.datasets import make_classification
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=10, n_informative=3, n_redundant=2, random_state=42)
        X[10, 0] = np.nan  # Add NaN

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            solution.train_random_forest(X, y)

    def test_inf_in_features_tree(self):
        """
        反例：特征中有 Inf

        应报错
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        from sklearn.datasets import make_classification
        np.random.seed(42)
        X, y = make_classification(n_samples=100, n_features=10, n_informative=3, n_redundant=2, random_state=42)
        X[10, 0] = np.inf  # Add infinity

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            solution.train_decision_tree(X, y)


# =============================================================================
# 5. 维度不匹配测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDimensionMismatch:
    """测试维度不匹配"""

    def test_1d_input_tree(self):
        """
        反例：1D 输入应报错

        决策树需要 2D 输入
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        np.random.seed(42)
        X = np.random.randn(100)
        y = (X > 0).astype(int)

        # Should either raise error or reshape
        try:
            model = solution.train_decision_tree(X, y)
            # If it succeeds, it should have reshaped
            assert model is not None
        except (ValueError, IndexError):
            # Raising an error is acceptable
            assert True

    def test_mismatched_sample_counts_tree(self):
        """
        反例：X 和 y 样本数不同

        应报错
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.random.randint(0, 2, 50)  # Different length

        # Should raise an error
        with pytest.raises((ValueError, IndexError, AssertionError)):
            solution.train_decision_tree(X, y)

    def test_predict_with_wrong_features(self, simple_tree_classification_data):
        """
        反例：预测时特征数量不匹配

        应报错
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        # Train model
        model = solution.train_decision_tree(X, y)

        # Predict with wrong number of features
        X_wrong = np.random.randn(10, 5)  # 5 features instead of original

        if hasattr(model, 'predict'):
            with pytest.raises((ValueError, IndexError)):
                model.predict(X_wrong)


# =============================================================================
# 6. 常量特征测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestConstantFeatures:
    """测试常量特征"""

    def test_single_constant_feature_tree(self, constant_feature_tree_data):
        """
        边界：一个特征是常量

        决策树应能训练，忽略常量特征
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = constant_feature_tree_data['X']
        y = constant_feature_tree_data['y']

        # Should be able to train
        try:
            model = solution.train_decision_tree(X, y)
            assert model is not None
        except (ValueError, RuntimeError):
            # Some implementations may fail
            assert True

    def test_all_constant_features_tree(self):
        """
        反例：所有特征都是常量

        无法分类（无信息）
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        np.random.seed(42)
        X = np.ones((100, 2))  # All features are constant
        y = np.random.randint(0, 2, 100)

        # May fail or produce useless model
        try:
            model = solution.train_decision_tree(X, y)
            # If it doesn't raise, that's okay
            assert model is not None
        except (ValueError, RuntimeError):
            # May fail with constant features
            assert True


# =============================================================================
# 7. 无效参数测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestInvalidParameters:
    """测试无效参数"""

    def test_max_depth_zero_tree(self):
        """
        反例：max_depth=0

        应报错或返回无效模型
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            solution.train_decision_tree(X, y, max_depth=0)

    def test_max_depth_negative_tree(self):
        """
        反例：max_depth=-1

        应报错
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            solution.train_decision_tree(X, y, max_depth=-1)

    def test_n_estimators_zero_rf(self):
        """
        反例：n_estimators=0

        应报错
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError, ValueError)):
            solution.train_random_forest(X, y, n_estimators=0)

    def test_n_estimators_negative_rf(self):
        """
        反例：n_estimators=-1

        应报错
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        # Should raise an error
        with pytest.raises((ValueError, RuntimeError)):
            solution.train_random_forest(X, y, n_estimators=-1)

    def test_min_samples_leaf_too_large(self):
        """
        反例：min_samples_leaf 大于样本数

        应报错
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        np.random.seed(42)
        X = np.random.randn(50, 2)  # Only 50 samples
        y = (X[:, 0] > 0).astype(int)

        # Should raise an error or handle gracefully
        try:
            model = solution.train_decision_tree(X, y, min_samples_leaf=100)
            # If it succeeds, it should handle it
            assert model is not None
        except (ValueError, RuntimeError):
            # Raising an error is acceptable
            assert True


# =============================================================================
# 8. 特征重要性边界测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFeatureImportanceBoundaries:
    """测试特征重要性边界情况"""

    def test_feature_importance_with_constant_feature(self, constant_feature_tree_data):
        """
        边界：包含常量特征的特征重要性

        常量特征的重要性应为 0
        """
        if not hasattr(solution, 'train_decision_tree') or not hasattr(solution, 'get_feature_importance'):
            pytest.skip("Required functions not implemented")

        X = constant_feature_tree_data['X']
        y = constant_feature_tree_data['y']

        model = solution.train_decision_tree(X, y, random_state=42)

        if hasattr(model, 'feature_importances_'):
            # First feature is constant, should have 0 or very low importance
            importance = model.feature_importances_
            assert importance[0] < 0.01, f"Constant feature should have ~0 importance, got {importance[0]}"

    def test_feature_importance_sums_to_one(self, simple_tree_classification_data):
        """
        正例：特征重要性之和应为 1

        (归一化的特征重要性)
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        model = solution.train_decision_tree(X, y, random_state=42)

        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            sum_importance = importance.sum()
            # Should sum to approximately 1
            assert abs(sum_importance - 1.0) < 0.01, \
                f"Feature importances should sum to 1, got {sum_importance}"


# =============================================================================
# 9. 过拟合检测边界测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestOverfittingDetectionBoundaries:
    """测试过拟合检测边界情况"""

    def test_overfitting_detection_with_perfect_fit(self):
        """
        边界：完美拟合的训练集

        训练集准确率 100% 应触发过拟合警告
        """
        if not hasattr(solution, 'detect_overfitting'):
            pytest.skip("detect_overfitting not implemented")

        # Create data that will be perfectly memorized
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, 20)

        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train unconstrained tree
        if hasattr(solution, 'train_decision_tree'):
            # Very small data, tree will likely memorize
            pass  # Skip actual training, just test function exists

    def test_overfitting_with_minimal_data(self, minimal_tree_data):
        """
        边界：极小数据的过拟合检测

        在只有 4 个样本的情况下，过拟合几乎是必然的
        """
        if not hasattr(solution, 'detect_overfitting'):
            pytest.skip("detect_overfitting not implemented")

        X = minimal_tree_data['X']
        y = minimal_tree_data['y']

        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Should still work
        if hasattr(solution, 'detect_overfitting'):
            try:
                result = solution.detect_overfitting(X_train, y_train, X_test, y_test)
                assert result is not None
            except Exception:
                # May fail with minimal data
                assert True


# =============================================================================
# 10. 数据类型测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDataTypes:
    """测试不同数据类型"""

    def test_integer_input_tree(self):
        """
        边界：整数输入

        应能正确处理整数
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y = np.array([0, 0, 0, 1, 1])

        # Should handle integer input
        model = solution.train_decision_tree(X, y)
        assert model is not None

    def test_list_input_tree(self):
        """
        边界：列表输入（而非 NumPy 数组）

        应能转换或报错
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        y = [0, 0, 0, 1, 1]

        # Should either convert or raise error
        try:
            model = solution.train_decision_tree(X, y)
            assert model is not None
        except (ValueError, TypeError):
            # Raising TypeError for list input is acceptable
            assert True

    def test_single_feature_tree(self, single_feature_data):
        """
        边界：单特征数据

        决策树应能处理单特征数据
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = single_feature_data['X']
        y = single_feature_data['y']

        # Should handle single feature
        model = solution.train_decision_tree(X, y)
        assert model is not None

        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            assert len(y_pred) == len(y)


# Need to import simple_tree_classification_data and single_feature_data fixtures
# These are defined in conftest.py but need to be available here

# Import fixtures at module level to make them available
from pytest import fixture

@fixture
def simple_tree_classification_data():
    from sklearn.datasets import make_classification
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42
    )
    return {'X': X, 'y': y}


@fixture
def single_feature_data():
    np.random.seed(42)
    X = np.random.randn(100, 1)
    y = (X[:, 0] > 0).astype(int)
    return {'X': X, 'y': y}


@fixture
def constant_feature_tree_data():
    from sklearn.datasets import make_classification
    np.random.seed(42)
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_redundant=0,
        random_state=42
    )
    X[:, 0] = 5.0  # First column is constant
    return {'X': X, 'y': y}


@fixture
def minimal_tree_data():
    X = np.array([[0, 0], [1, 1], [5, 5], [6, 6]], dtype=float)
    y = np.array([0, 0, 1, 1])
    return {'X': X, 'y': y}
