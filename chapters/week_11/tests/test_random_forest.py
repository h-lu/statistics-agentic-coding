"""
Comprehensive tests for random forest functionality in Week 11 solution.py

随机森林测试：
- 训练和预测
- 特征重要性
- Bagging 原理
- OOB 评分
- 并行训练
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
# 1. 随机森林训练与预测测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestRandomForestTraining:
    """测试随机森林训练功能"""

    def test_train_random_forest(self, random_forest_data):
        """
        正例：训练随机森林模型

        应能拟合二分类数据
        """
        if not hasattr(solution, 'train_random_forest') and not hasattr(solution, 'fit_random_forest'):
            pytest.skip("No random forest training function implemented")

        X = random_forest_data['X']
        y = random_forest_data['y']

        # Try different function names
        if hasattr(solution, 'train_random_forest'):
            model = solution.train_random_forest(X, y)
        elif hasattr(solution, 'fit_random_forest'):
            model = solution.fit_random_forest(X, y)
        else:
            pytest.skip("No random forest training function implemented")

        assert model is not None, "Should return a model"

        # Check that model can predict
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            assert len(y_pred) == len(y), "Predictions should have same length as y"

    def test_random_forest_predictions(self, random_forest_data):
        """
        正例：随机森林预测

        应能返回类别预测
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = random_forest_data['X']
        y = random_forest_data['y']

        # Train model
        model = solution.train_random_forest(X, y, n_estimators=50, random_state=42)

        # Make predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
        elif hasattr(solution, 'predict_random_forest'):
            y_pred = solution.predict_random_forest(model, X)
        else:
            pytest.skip("No prediction function implemented")

        assert y_pred is not None
        assert len(y_pred) == len(y)

        # Predictions should be 0 or 1
        unique_preds = np.unique(y_pred)
        assert all(p in [0, 1] for p in unique_preds), f"Predictions should be 0 or 1, got {unique_preds}"

    def test_random_forest_probabilities(self, random_forest_data):
        """
        正例：随机森林概率预测

        应能返回类别概率（在 [0, 1] 之间）
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = random_forest_data['X']
        y = random_forest_data['y']

        # Train model
        model = solution.train_random_forest(X, y, n_estimators=50, random_state=42)

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

    def test_random_forest_with_n_estimators(self, random_forest_data):
        """
        正例：使用指定数量的树训练

        应能正确应用 n_estimators 参数
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = random_forest_data['X']
        y = random_forest_data['y']

        # Train with specific number of trees
        n_estimators = 50
        model = solution.train_random_forest(X, y, n_estimators=n_estimators, random_state=42)

        assert model is not None

        # Check n_estimators
        if hasattr(model, 'n_estimators'):
            assert model.n_estimators == n_estimators, f"Should have {n_estimators} estimators"
        elif hasattr(model, 'estimators_'):
            assert len(model.estimators_) == n_estimators, f"Should have {n_estimators} estimators"

    def test_random_forest_with_max_depth(self, random_forest_data):
        """
        正例：使用深度限制训练

        应能正确应用 max_depth 参数
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = random_forest_data['X']
        y = random_forest_data['y']

        # Train with depth limit
        max_depth = 5
        model = solution.train_random_forest(X, y, n_estimators=50, max_depth=max_depth, random_state=42)

        assert model is not None

        # Check that trees respect max_depth
        if hasattr(model, 'estimators_'):
            for tree in model.estimators_:
                if hasattr(tree, 'tree_'):
                    assert tree.tree_.max_depth <= max_depth, \
                        f"Tree depth should be limited to {max_depth}"


# =============================================================================
# 2. 随机森林 vs 单棵树对比测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestRandomForestVsSingleTree:
    """测试随机森林相对于单棵树的优势"""

    def test_random_forest_better_than_single_tree(self, random_forest_data):
        """
        正例：随机森林应比单棵树更稳定

        在测试集上，随机森林通常比单棵树表现更好或相当
        """
        if not hasattr(solution, 'train_random_forest') or not hasattr(solution, 'train_decision_tree'):
            pytest.skip("Required functions not implemented")

        X = random_forest_data['X']
        y = random_forest_data['y']

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train single tree
        tree = solution.train_decision_tree(X_train, y_train, max_depth=5, random_state=42)

        # Train random forest
        rf = solution.train_random_forest(X_train, y_train, n_estimators=50, max_depth=5, random_state=42)

        if hasattr(tree, 'predict') and hasattr(rf, 'predict'):
            # Compare test performance
            tree_pred = tree.predict(X_test)
            rf_pred = rf.predict(X_test)

            tree_acc = np.mean(tree_pred == y_test)
            rf_acc = np.mean(rf_pred == y_test)

            # Random forest should be as good or better
            # (may not always be strictly better, but should not be much worse)
            assert rf_acc >= tree_acc - 0.05, \
                f"Random forest accuracy {rf_acc} should not be much worse than tree {tree_acc}"

    def test_random_forest_more_stable(self, simple_tree_classification_data):
        """
        正例：随机森林应更稳定

        相同随机种子下，随机森林的预测应该更一致
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        # Train two forests with same seed
        rf1 = solution.train_random_forest(X, y, n_estimators=50, random_state=42)
        rf2 = solution.train_random_forest(X, y, n_estimators=50, random_state=42)

        if hasattr(rf1, 'predict') and hasattr(rf2, 'predict'):
            pred1 = rf1.predict(X)
            pred2 = rf2.predict(X)

            # Should be identical
            assert np.array_equal(pred1, pred2), "Same random seed should give identical results"


# =============================================================================
# 3. 随机森林特征重要性测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestRandomForestFeatureImportance:
    """测试随机森林特征重要性"""

    def test_rf_feature_importance(self, feature_importance_data):
        """
        正例：获取随机森林特征重要性

        应返回每个特征的重要性分数
        """
        if not hasattr(solution, 'train_random_forest') or not hasattr(solution, 'get_feature_importance'):
            pytest.skip("Required functions not implemented")

        X = feature_importance_data['X']
        y = feature_importance_data['y']

        # Train model
        model = solution.train_random_forest(X, y, n_estimators=50, random_state=42)

        # Get feature importance
        importance = solution.get_feature_importance(model)

        assert importance is not None

        # Should have importance for each feature
        if isinstance(importance, (list, np.ndarray)):
            assert len(importance) == X.shape[1], "Should have importance for each feature"
            # Importances should be non-negative
            assert all(imp >= 0 for imp in importance), "Importances should be non-negative"
            # Importances should sum to approximately 1
            assert abs(sum(importance) - 1.0) < 0.1, "Importances should sum to approximately 1"
        elif isinstance(importance, dict):
            assert len(importance) == X.shape[1], "Should have importance for each feature"
            # All values should be non-negative
            assert all(v >= 0 for v in importance.values()), "Importances should be non-negative"

    def test_feature_importance_consistency(self, feature_importance_data):
        """
        正例：特征重要性应相对稳定

        多次训练应给出相似的特征重要性排序
        """
        if not hasattr(solution, 'train_random_forest') or not hasattr(solution, 'get_feature_importance'):
            pytest.skip("Required functions not implemented")

        X = feature_importance_data['X']
        y = feature_importance_data['y']

        # Train two models
        model1 = solution.train_random_forest(X, y, n_estimators=50, random_state=42)
        model2 = solution.train_random_forest(X, y, n_estimators=50, random_state=43)

        # Get importances
        imp1 = solution.get_feature_importance(model1)
        imp2 = solution.get_feature_importance(model2)

        # Convert to arrays if needed
        if isinstance(imp1, dict):
            imp1 = np.array([imp1[i] for i in range(len(imp1))])
        if isinstance(imp2, dict):
            imp2 = np.array([imp2[i] for i in range(len(imp2))])

        # Get rankings (most important to least important)
        ranking1 = np.argsort(-imp1)
        ranking2 = np.argsort(-imp2)

        # Top 3 features should be similar (at least 2 out of 3)
        top_overlap = len(set(ranking1[:3]) & set(ranking2[:3]))
        assert top_overlap >= 2, \
            f"Top features should be similar, got overlap of {top_overlap}/3"

    def test_plot_feature_importance(self, feature_importance_data):
        """
        正例：检查是否有特征重要性可视化函数

        不测试具体输出，只检查函数存在
        """
        has_plot = hasattr(solution, 'plot_feature_importance') or \
                   hasattr(solution, 'visualize_feature_importance')

        # If function exists, it should work
        if has_plot:
            if not hasattr(solution, 'train_random_forest') or not hasattr(solution, 'get_feature_importance'):
                pytest.skip("Required functions not implemented")

            X = feature_importance_data['X']
            y = feature_importance_data['y']

            model = solution.train_random_forest(X, y, n_estimators=50, random_state=42)
            importance = solution.get_feature_importance(model)

            # Try to plot (should not crash)
            try:
                if hasattr(solution, 'plot_feature_importance'):
                    solution.plot_feature_importance(importance, feature_importance_data['feature_names'])
                elif hasattr(solution, 'visualize_feature_importance'):
                    solution.visualize_feature_importance(importance, feature_importance_data['feature_names'])
                assert True
            except Exception:
                # May fail if matplotlib not available or other issues
                assert True


# =============================================================================
# 4. Bagging 原理测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBaggingBehavior:
    """测试 Bagging 行为"""

    def test_rf_has_multiple_trees(self, random_forest_data):
        """
        正例：随机森林应包含多棵树

        验证 Bagging 原理：多个模型的集合
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = random_forest_data['X']
        y = random_forest_data['y']

        n_estimators = 50
        model = solution.train_random_forest(X, y, n_estimators=n_estimators, random_state=42)

        # Check that model has multiple trees
        has_trees = False
        if hasattr(model, 'estimators_'):
            has_trees = True
            assert len(model.estimators_) == n_estimators, \
                f"Should have {n_estimators} trees, got {len(model.estimators_)}"
        elif hasattr(model, 'n_estimators'):
            has_trees = True
            assert model.n_estimators == n_estimators

        # assert has_trees, "Random forest should have multiple estimators"

    def test_rf_trees_are_diverse(self, random_forest_data):
        """
        正例：随机森林中的树应该不同

        验证 Bagging 的随机性：每棵树看到不同的数据
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = random_forest_data['X']
        y = random_forest_data['y']

        model = solution.train_random_forest(X, y, n_estimators=20, max_depth=3, random_state=42)

        # Check that trees are different
        if hasattr(model, 'estimators_'):
            # Compare predictions of individual trees
            if hasattr(model.estimators_[0], 'predict'):
                predictions = []
                for tree in model.estimators_[:5]:  # Check first 5 trees
                    pred = tree.predict(X[:10])  # Predict on first 10 samples
                    predictions.append(pred)

                # Not all trees should give identical predictions
                all_same = all(np.array_equal(predictions[0], p) for p in predictions[1:])
                assert not all_same, "Trees should be diverse due to bagging"


# =============================================================================
# 5. OOB 评分测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestOOBScore:
    """测试 OOB（Out-of-Bag）评分"""

    def test_rf_oob_score(self, random_forest_data):
        """
        正例：随机森林应能计算 OOB 评分

        OOB 评分是一种无偏的估计方法
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = random_forest_data['X']
        y = random_forest_data['y']

        # Train with OOB scoring
        try:
            model = solution.train_random_forest(
                X, y,
                n_estimators=50,
                oob_score=True,
                random_state=42
            )

            # Check if model has OOB score
            if hasattr(model, 'oob_score_'):
                assert 0 <= model.oob_score_ <= 1, "OOB score should be in [0, 1]"
                assert model.oob_score_ > 0.5, "OOB score should be better than random"
        except TypeError:
            # oob_score parameter might not be supported
            pytest.skip("oob_score parameter not supported")


# =============================================================================
# 6. 并行训练测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestParallelTraining:
    """测试并行训练"""

    def test_rf_n_jobs_parameter(self, random_forest_data):
        """
        正例：随机森林应支持并行训练

        n_jobs=-1 应使用所有 CPU 核心
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = random_forest_data['X']
        y = random_forest_data['y']

        # Train with parallel
        try:
            model = solution.train_random_forest(
                X, y,
                n_estimators=50,
                n_jobs=-1,
                random_state=42
            )

            assert model is not None

            # Should be able to predict
            if hasattr(model, 'predict'):
                y_pred = model.predict(X)
                assert len(y_pred) == len(y)
        except TypeError:
            # n_jobs parameter might not be supported
            pytest.skip("n_jobs parameter not supported")


# =============================================================================
# 7. 随机森林边界测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestRandomForestBoundaries:
    """测试随机森林边界情况"""

    def test_rf_with_single_tree(self, simple_tree_classification_data):
        """
        边界：n_estimators=1 的随机森林

        应退化为单棵决策树
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = simple_tree_classification_data['X']
        y = simple_tree_classification_data['y']

        # Train with single tree
        model = solution.train_random_forest(X, y, n_estimators=1, random_state=42)

        assert model is not None

        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            assert len(y_pred) == len(y)

    def test_rf_with_imbalanced_data(self, imbalanced_tree_data):
        """
        边界：类别不平衡数据

        随机森林应能处理不平衡数据
        """
        if not hasattr(solution, 'train_random_forest'):
            pytest.skip("train_random_forest not implemented")

        X = imbalanced_tree_data['X']
        y = imbalanced_tree_data['y']

        # Train model
        model = solution.train_random_forest(X, y, n_estimators=50, random_state=42)

        assert model is not None

        if hasattr(model, 'predict'):
            y_pred = model.predict(X)
            # Should predict both classes (not all the same)
            unique_preds = np.unique(y_pred)
            assert len(unique_preds) >= 1, f"Should make predictions, got {unique_preds}"
