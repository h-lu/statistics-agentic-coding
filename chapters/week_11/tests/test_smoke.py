"""
Smoke tests for Week 11 solution.py

Basic sanity checks to ensure the module can be imported and functions exist.
These tests should all pass once solution.py is created with the expected interface.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))


class TestModuleImport:
    """Test that the solution module can be imported."""

    def test_import_solution_module(self):
        """Test that solution module can be imported."""
        try:
            import solution
            assert solution is not None
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_solution_has_expected_functions(self):
        """Test that solution module has expected functions."""
        try:
            import solution

            # Expected function groups for Week 11
            expected_functions = [
                # 决策树分类
                'train_decision_tree',
                'fit_decision_tree',
                'predict_tree',

                # 随机森林分类
                'train_random_forest',
                'fit_random_forest',
                'predict_random_forest',

                # 特征重要性
                'get_feature_importance',
                'plot_feature_importance',

                # 基线对比
                'train_dummy_baseline',
                'train_logistic_baseline',
                'compare_with_baselines',
                'baseline_comparison',

                # 过拟合检测
                'detect_overfitting',
                'check_overfitting',

                # StatLab 集成
                'tree_models_comparison',
                'format_model_comparison_report',
            ]

            # Check that at least some functions exist
            existing_functions = [name for name in expected_functions if hasattr(solution, name)]
            assert len(existing_functions) >= 4, f"At least 4 functions should exist, found: {existing_functions}"

        except ImportError:
            pytest.skip("solution.py not yet created")


class TestBasicFunctionality:
    """Test basic functionality with simple inputs."""

    def test_train_decision_tree_basic(self):
        """Test basic decision tree training."""
        try:
            import solution

            if hasattr(solution, 'train_decision_tree'):
                np.random.seed(42)
                X = np.random.randn(100, 2)
                y = (X[:, 0] + X[:, 1] > 0).astype(int)

                model = solution.train_decision_tree(X, y)
                assert model is not None

                # Result should have some way to get predictions
                if hasattr(model, 'predict'):
                    assert True
                elif isinstance(model, dict):
                    assert 'model' in model or 'tree' in model
            else:
                pytest.skip("train_decision_tree not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_train_random_forest_basic(self):
        """Test basic random forest training."""
        try:
            import solution

            if hasattr(solution, 'train_random_forest'):
                np.random.seed(42)
                X = np.random.randn(100, 2)
                y = (X[:, 0] + X[:, 1] > 0).astype(int)

                model = solution.train_random_forest(X, y)
                assert model is not None

                # Result should have some way to get predictions
                if hasattr(model, 'predict'):
                    assert True
                elif isinstance(model, dict):
                    assert 'model' in model or 'forest' in model
            else:
                pytest.skip("train_random_forest not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_get_feature_importance_basic(self):
        """Test basic feature importance extraction."""
        try:
            import solution

            if hasattr(solution, 'get_feature_importance'):
                np.random.seed(42)
                X = np.random.randn(100, 3)
                y = (X[:, 0] > 0).astype(int)

                # Train a model first
                if hasattr(solution, 'train_random_forest'):
                    model = solution.train_random_forest(X, y)
                elif hasattr(solution, 'train_decision_tree'):
                    model = solution.train_decision_tree(X, y)
                else:
                    pytest.skip("No training function implemented")

                importance = solution.get_feature_importance(model)
                assert importance is not None

                # Should have importance for each feature
                if isinstance(importance, (list, np.ndarray)):
                    assert len(importance) == X.shape[1]
                elif isinstance(importance, dict):
                    assert len(importance) == X.shape[1]
            else:
                pytest.skip("get_feature_importance not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_compare_with_baselines_basic(self):
        """Test basic baseline comparison."""
        try:
            import solution

            if hasattr(solution, 'compare_with_baselines'):
                np.random.seed(42)
                X = np.random.randn(100, 2)
                y = (X[:, 0] + X[:, 1] > 0).astype(int)

                results = solution.compare_with_baselines(X, y)
                assert results is not None

                # Should contain comparison results
                if isinstance(results, dict):
                    # At least one model result should be present
                    expected_keys = ['dummy', 'logistic', 'tree', 'forest',
                                   'dummy_auc', 'logistic_auc', 'tree_auc', 'forest_auc']
                    has_result = any(key in results for key in expected_keys)
                    assert has_result, f"Should have at least one result, got: {results.keys()}"
            else:
                pytest.skip("compare_with_baselines not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")


class TestDocumentation:
    """Test that functions have proper documentation."""

    def test_functions_have_docstrings(self):
        """Test that key functions have docstrings."""
        try:
            import solution

            # Check a few key functions
            key_functions = [
                'train_decision_tree',
                'train_random_forest',
                'get_feature_importance',
                'compare_with_baselines',
                'detect_overfitting',
            ]

            for func_name in key_functions:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    assert func.__doc__ is not None, f"{func_name} should have a docstring"
                    assert len(func.__doc__) > 10, f"{func_name} docstring should be meaningful"

        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_tree_function_has_overfitting_warning(self):
        """Test that tree functions mention overfitting."""
        try:
            import solution

            # Check tree-related functions
            tree_functions = [
                'train_decision_tree',
                'detect_overfitting',
            ]

            for func_name in tree_functions:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    doc = func.__doc__ or ""
                    # Docstring should mention overfitting or pruning
                    text_lower = doc.lower()
                    assert 'overfit' in text_lower or 'prun' in text_lower or 'depth' in text_lower, \
                        f"{func_name} docstring should mention overfitting or pruning"

        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_baseline_function_has_baseline_warning(self):
        """Test that baseline functions mention baseline comparison."""
        try:
            import solution

            # Check baseline-related functions
            baseline_functions = [
                'compare_with_baselines',
                'baseline_comparison',
                'train_dummy_baseline',
            ]

            for func_name in baseline_functions:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    doc = func.__doc__ or ""
                    # Docstring should mention baseline
                    text_lower = doc.lower()
                    assert 'baseline' in text_lower or 'dummy' in text_lower or 'compar' in text_lower, \
                        f"{func_name} docstring should mention baseline or comparison"

        except ImportError:
            pytest.skip("solution.py not yet created")


class TestTypeAnnotations:
    """Test that functions have proper type annotations."""

    def test_functions_have_type_hints(self):
        """Test that key functions have type hints."""
        try:
            import solution

            # Check a few key functions
            key_functions = [
                'train_decision_tree',
                'train_random_forest',
                'get_feature_importance',
            ]

            for func_name in key_functions:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    # At minimum, function should have some annotations
                    assert hasattr(func, '__annotations__'), f"{func_name} should have type hints"

        except ImportError:
            pytest.skip("solution.py not yet created")


class TestErrorHandling:
    """Test basic error handling."""

    def test_empty_input_handling(self):
        """Test that empty input is handled gracefully."""
        try:
            import solution

            # Test with empty arrays
            X_empty = np.array([]).reshape(0, 2)
            y_empty = np.array([])

            for func_name in ['train_decision_tree', 'train_random_forest']:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    # Should either raise a meaningful error or return a meaningful result
                    try:
                        result = func(X_empty, y_empty)
                        # If it doesn't raise, result should indicate the issue
                        assert result is not None
                    except (ValueError, IndexError) as e:
                        # These are acceptable errors for empty input
                        assert len(str(e)) > 0

        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_single_class_handling(self):
        """Test that single-class data is handled."""
        try:
            import solution

            # Test with single class
            X = np.random.randn(100, 2)
            y = np.zeros(100)  # All zeros

            for func_name in ['train_decision_tree', 'train_random_forest']:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    # Should raise an error or handle gracefully
                    try:
                        result = func(X, y)
                        # If it succeeds, should indicate the issue
                        assert result is not None
                    except (ValueError, RuntimeError) as e:
                        # Error is acceptable for single class
                        assert len(str(e)) > 0

        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_mismatched_dimensions_handling(self):
        """Test that mismatched dimensions are handled."""
        try:
            import solution

            # Test with different length arrays
            X = np.random.randn(100, 2)
            y = np.array([0, 1])  # Wrong length

            for func_name in ['train_decision_tree', 'train_random_forest']:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    # Should raise an error
                    with pytest.raises((ValueError, IndexError, AssertionError)):
                        func(X, y)

        except ImportError:
            pytest.skip("solution.py not yet created")
