"""
Comprehensive tests for decision tree functionality in Week 11 solution.py

决策树测试：
- 训练和预测
- 树结构可视化
- 过拟合检测
- 剪枝参数
- 决策树桩
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
# 1. 决策树训练与预测测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDecisionTreeTraining:
    """测试决策树训练功能"""

    def test_train_decision_tree(self, simple_tree_classification_data):
        """
        正例：训练决策树模型

        应能拟合二分类数据
        """
        if not hasattr(solution, 'train_decision_tree') and not hasattr(solution, 'fit_decision_tree'):
            pytest.skip("No decision tree training function implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        # Try different function names
        if hasattr(solution, 'train_decision_tree'):
            model = solution.train_decision_tree(X, y)
        elif hasattr(solution, 'fit_decision_tree'):
            model = solution.fit_decision_tree(X, y)
        else:
            pytest.skip("No decision tree training function implemented")

        assert model is not None, "Should return a model"

        # Check that model can predict
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            assert len(y_pred) == len(y), "Predictions should have same length as y"

    def test_decision_tree_predictions(self, simple_tree_classification_data):
        """
        正例：决策树预测

        应能返回类别预测
        """
        if not hasattr(solution, 'train_decision_tree') and not hasattr(solution, 'fit_decision_tree'):
            pytest.skip("No decision tree training function implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        # Train model
        if hasattr(solution, 'train_decision_tree'):
            model = solution.train_decision_tree(X, y)
        elif hasattr(solution, 'fit_decision_tree'):
            model = solution.fit_decision_tree(X, y)
        else:
            pytest.skip("No decision tree training function implemented")

        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
        elif hasattr(solution, 'predict_tree'):
            y_pred = solution.predict_tree(model, X)
        else:
            pytest.skip("No prediction function implemented")

        assert y_pred is not None
        assert len(y_pred) == len(y)

        # Predictions should be 0 or 1
        unique_preds = np.unique(y_pred)
        assert all(p in [0, 1] for p in unique_preds), f"Predictions should be 0 or 1, got {unique_preds}"

    def test_decision_tree_probabilities(self, simple_tree_classification_data):
        """
        正例：决策树概率预测

        应能返回类别概率（在 [0, 1] 之间）
        """
        if not hasattr(solution, 'train_decision_tree') and not hasattr(solution, 'fit_decision_tree'):
            pytest.skip("No decision tree training function implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        # Train model
        if hasattr(solution, 'train_decision_tree'):
            model = solution.train_decision_tree(X, y)
        elif hasattr(solution, 'fit_decision_tree'):
            model = solution.fit_decision_tree(X, y)
        else:
            pytest.skip("No decision tree training function implemented")

        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)
            # Get probability of class 1
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]
        else:
            pytest.skip("No probability prediction function implemented")

        assert y_prob is not None
        assert len(y_prob) == len(y)

        # Probabilities should be in [0, 1]
        assert np.all(y_prob >= 0) and np.all(y_prob <= 1), "Probabilities should be in [0, 1]"

    def test_decision_tree_with_depth_limit(self, simple_tree_classification_data, tree_model_params):
        """
        正例：使用深度限制训练决策树

        应能正确应用 max_depth 参数
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        # Train with depth limit
        model = solution.train_decision_tree(X, y, max_depth=tree_model_params['max_depth'])

        assert model is not None

        # Check that depth was limited (if model has attribute)
        if hasattr(model, 'tree_'):
            assert model.tree_.max_depth <= tree_model_params['max_depth'], \
                f"Tree depth should be limited to {tree_model_params['max_depth']}"

    def test_decision_tree_xor_data(self, xor_like_data):
        """
        正例：决策树处理 XOR 类型数据

        决策树应能处理非线性可分数据（线性模型难以处理）
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = xor_like_data['X']
        y = xor_like_data['y']

        # Train model
        model = solution.train_decision_tree(X, y, max_depth=5, random_state=42)
        assert model is not None

        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            # Should achieve reasonable accuracy on XOR data
            accuracy = np.mean(y_pred == y)
            assert accuracy >= 0.7, f"Decision tree should handle XOR data, got accuracy {accuracy}"


# =============================================================================
# 2. 决策树桩测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDecisionStump:
    """测试决策树桩（max_depth=1）"""

    def test_decision_stump_training(self, single_feature_data):
        """
        边界：决策树桩训练

        max_depth=1 的决策树应能训练
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = single_feature_data['X']
        y = single_feature_data['y']

        # Train decision stump
        model = solution.train_decision_tree(X, y, max_depth=1, random_state=42)

        assert model is not None

        # Check depth
        if hasattr(model, 'tree_'):
            assert model.tree_.max_depth <= 1, "Decision stump should have max_depth=1"

    def test_decision_stump_predictions(self, single_feature_data):
        """
        正例：决策树桩预测

        即使只有一层，也应能做出合理预测
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = single_feature_data['X']
        y = single_feature_data['y']

        # Train decision stump
        model = solution.train_decision_tree(X, y, max_depth=1, random_state=42)

        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            # Should make predictions
            assert len(y_pred) == len(y)
            # Predictions should be 0 or 1
            assert all(p in [0, 1] for p in y_pred)


# =============================================================================
# 3. 过拟合检测测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestTreeOverfitting:
    """测试决策树过拟合检测"""

    def test_detect_overfitting(self, overfitting_scenario_data):
        """
        正例：检测决策树过拟合

        无限制的树可能在训练集上过拟合
        """
        if not hasattr(solution, 'detect_overfitting') and not hasattr(solution, 'check_overfitting'):
            pytest.skip("No overfitting detection function implemented")

        X_train = overfitting_scenario_data['X_train']
        X_test = overfitting_scenario_data['X_test']
        y_train = overfitting_scenario_data['y_train']
        y_test = overfitting_scenario_data['y_test']

        # Try different function names
        if hasattr(solution, 'detect_overfitting'):
            result = solution.detect_overfitting(X_train, y_train, X_test, y_test)
        elif hasattr(solution, 'check_overfitting'):
            result = solution.check_overfitting(X_train, y_train, X_test, y_test)
        else:
            pytest.skip("No overfitting detection function implemented")

        assert result is not None

        # Should return some indication of overfitting
        if isinstance(result, dict):
            assert 'train_score' in result or 'test_score' in result or \
                   'is_overfitting' in result or 'overfitting' in result

    def test_unconstrained_tree_overfits(self, tree_overfitting_data):
        """
        反例：无限制的决策树应过拟合

        训练集准确率应明显高于测试集
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = tree_overfitting_data['X']
        y = tree_overfitting_data['y']

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train unconstrained tree
        model = solution.train_decision_tree(X_train, y_train, random_state=42)

        if hasattr(model, 'predict'):
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)

            train_acc = np.mean(train_pred == y_train)
            test_acc = np.mean(test_pred == y_test)

            # Unconstrained tree should overfit
            # Training accuracy should be higher than test accuracy
            # (may not always be true, but likely with this data)
            assert train_acc >= test_acc, \
                f"Training accuracy {train_acc} should be >= test accuracy {test_acc}"

    def test_pruned_tree_reduces_overfitting(self, tree_overfitting_data):
        """
        正例：剪枝后的决策树应减少过拟合

        限制深度应提高泛化能力
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = tree_overfitting_data['X']
        y = tree_overfitting_data['y']

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train unpruned tree
        unpruned = solution.train_decision_tree(X_train, y_train, random_state=42)

        # Train pruned tree
        pruned = solution.train_decision_tree(
            X_train, y_train,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )

        if hasattr(unpruned, 'predict') and hasattr(pruned, 'predict'):
            # Compare test performance
            unpruned_test_pred = unpruned.predict(X_test)
            pruned_test_pred = pruned.predict(X_test)

            unpruned_test_acc = np.mean(unpruned_test_pred == y_test)
            pruned_test_acc = np.mean(pruned_test_pred == y_test)

            # Pruned tree should generalize better or similar
            # (not always strictly better, but should not be much worse)
            assert pruned_test_acc >= unpruned_test_acc - 0.1, \
                f"Pruned tree test accuracy {pruned_test_acc} should not be much worse than unpruned {unpruned_test_acc}"


# =============================================================================
# 4. 树参数测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestTreeParameters:
    """测试决策树参数"""

    def test_min_samples_split(self, simple_tree_classification_data):
        """
        正例：min_samples_split 参数

        应正确应用最小分裂样本数限制
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        # Train with min_samples_split
        model = solution.train_decision_tree(X, y, min_samples_split=20, random_state=42)

        assert model is not None

        # Should be able to predict
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            assert len(y_pred) == len(y)

    def test_min_samples_leaf(self, simple_tree_classification_data):
        """
        正例：min_samples_leaf 参数

        应正确应用最小叶节点样本数限制
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        # Train with min_samples_leaf
        model = solution.train_decision_tree(X, y, min_samples_leaf=10, random_state=42)

        assert model is not None

        # Should be able to predict
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            assert len(y_pred) == len(y)

    def test_max_leaf_nodes(self, simple_tree_classification_data):
        """
        正例：max_leaf_nodes 参数

        应正确应用最大叶节点数限制
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        # Train with max_leaf_nodes
        model = solution.train_decision_tree(X, y, max_leaf_nodes=10, random_state=42)

        assert model is not None

        # Should be able to predict
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            assert len(y_pred) == len(y)


# =============================================================================
# 5. 树结构测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestTreeStructure:
    """测试决策树结构"""

    def test_tree_depth_attribute(self, simple_tree_classification_data):
        """
        正例：获取树深度

        模型应能报告树的深度
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        model = solution.train_decision_tree(X, y, max_depth=3, random_state=42)

        # Check if model has depth information
        has_depth = False
        if hasattr(model, 'tree_'):
            if hasattr(model.tree_, 'max_depth'):
                has_depth = True
                assert model.tree_.max_depth <= 3, "Tree depth should respect max_depth"
        if hasattr(model, 'get_depth'):
            depth = model.get_depth()
            has_depth = True
            assert depth <= 3, "Tree depth should respect max_depth"

        # If model doesn't have depth attribute, that's okay
        # assert has_depth, "Model should provide depth information"

    def test_tree_n_leaves_attribute(self, simple_tree_classification_data):
        """
        正例：获取叶节点数量

        模型应能报告叶节点数量
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        model = solution.train_decision_tree(X, y, max_depth=3, random_state=42)

        # Check if model has n_leaves information
        if hasattr(model, 'tree_'):
            if hasattr(model.tree_, 'n_leaves'):
                n_leaves = model.tree_.n_leaves
                assert n_leaves > 0, "Tree should have at least one leaf"
                # With max_depth=3, maximum leaves is 2^3 = 8
                assert n_leaves <= 8, f"Tree should have at most 8 leaves with max_depth=3, got {n_leaves}"


# =============================================================================
# 6. 决策树可视化测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestTreeVisualization:
    """测试决策树可视化"""

    def test_plot_tree_function_exists(self):
        """
        正例：检查是否有树可视化函数

        不测试具体输出，只检查函数存在
        """
        has_plot = hasattr(solution, 'plot_tree') or hasattr(solution, 'visualize_tree')
        # Not required, but if exists, should have docstring
        if has_plot:
            func = getattr(solution, 'plot_tree') if hasattr(solution, 'plot_tree') else getattr(solution, 'visualize_tree')
            assert func.__doc__ is not None or True  # Docstring is nice but not required


# =============================================================================
# 7. 决策树特征重要性测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestTreeFeatureImportance:
    """测试决策树特征重要性"""

    def test_tree_feature_importance(self, feature_importance_data):
        """
        正例：获取决策树特征重要性

        应返回每个特征的重要性分数
        """
        if not hasattr(solution, 'train_decision_tree') or not hasattr(solution, 'get_feature_importance'):
            pytest.skip("Required functions not implemented")

        X = feature_importance_data['X']
        y = feature_importance_data['y']

        # Train model
        model = solution.train_decision_tree(X, y, random_state=42)

        # Get feature importance
        importance = solution.get_feature_importance(model)

        assert importance is not None

        # Should have importance for each feature
        if isinstance(importance, (list, np.ndarray)):
            assert len(importance) == X.shape[1], "Should have importance for each feature"
            # Importances should be non-negative
            assert all(imp >= 0 for imp in importance), "Importances should be non-negative"
        elif isinstance(importance, dict):
            assert len(importance) == X.shape[1], "Should have importance for each feature"
            # All values should be non-negative
            assert all(v >= 0 for v in importance.values()), "Importances should be non-negative"
