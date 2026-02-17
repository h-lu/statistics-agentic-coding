"""
Smoke tests for Week 10 solution.py

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

            # Expected function groups for Week 10
            expected_functions = [
                # 逻辑回归分类
                'train_logistic_regression',
                'fit_logistic_regression',
                'predict_logistic',

                # 混淆矩阵与评估指标
                'calculate_confusion_matrix',
                'calculate_accuracy',
                'calculate_precision',
                'calculate_recall',
                'calculate_f1',
                'calculate_metrics',

                # ROC 与 AUC
                'calculate_roc_curve',
                'calculate_auc',
                'plot_roc_curve',

                # Pipeline 与数据泄漏防护
                'create_classification_pipeline',
                'train_with_pipeline',
                'cross_val_with_pipeline',

                # StatLab 集成
                'classification_with_pipeline',
                'format_classification_report',
            ]

            # Check that at least some functions exist
            # (not all need to exist for smoke test)
            existing_functions = [name for name in expected_functions if hasattr(solution, name)]
            assert len(existing_functions) >= 4, f"At least 4 functions should exist, found: {existing_functions}"

        except ImportError:
            pytest.skip("solution.py not yet created")


class TestBasicFunctionality:
    """Test basic functionality with simple inputs."""

    def test_train_logistic_regression_basic(self):
        """Test basic logistic regression training."""
        try:
            import solution

            if hasattr(solution, 'train_logistic_regression'):
                np.random.seed(42)
                X = np.random.randn(100, 2)
                y = (X[:, 0] + X[:, 1] > 0).astype(int)

                model = solution.train_logistic_regression(X, y)
                assert model is not None

                # Result should have some way to get predictions
                if hasattr(model, 'predict'):
                    assert True
                elif isinstance(model, dict):
                    assert 'model' in model or 'coefficients' in model
            else:
                pytest.skip("train_logistic_regression not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_fit_logistic_regression_basic(self):
        """Test basic logistic regression fitting."""
        try:
            import solution

            if hasattr(solution, 'fit_logistic_regression'):
                np.random.seed(42)
                X = np.random.randn(100, 2)
                y = (X[:, 0] + X[:, 1] > 0).astype(int)

                model = solution.fit_logistic_regression(X, y)
                assert model is not None
            else:
                pytest.skip("fit_logistic_regression not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_calculate_confusion_matrix_basic(self):
        """Test basic confusion matrix calculation."""
        try:
            import solution

            if hasattr(solution, 'calculate_confusion_matrix'):
                y_true = np.array([0, 0, 1, 1])
                y_pred = np.array([0, 1, 1, 1])

                cm = solution.calculate_confusion_matrix(y_true, y_pred)
                assert cm is not None

                # Should be 2x2 matrix
                if isinstance(cm, np.ndarray):
                    assert cm.shape == (2, 2)
                elif isinstance(cm, dict):
                    # May return dict with TP, TN, FP, FN
                    assert any(key in cm for key in ['TP', 'TN', 'FP', 'fn', 'tp', 'tn', 'fp'])
            else:
                pytest.skip("calculate_confusion_matrix not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_calculate_metrics_basic(self):
        """Test basic metrics calculation."""
        try:
            import solution

            if hasattr(solution, 'calculate_metrics'):
                y_true = np.array([0, 0, 1, 1])
                y_pred = np.array([0, 1, 1, 1])

                metrics = solution.calculate_metrics(y_true, y_pred)
                assert metrics is not None

                # Should contain common metrics
                if isinstance(metrics, dict):
                    # At least one metric should be present
                    expected_keys = ['accuracy', 'precision', 'recall', 'f1', 'f1_score']
                    has_metric = any(key in metrics for key in expected_keys)
                    assert has_metric, f"Should have at least one metric, got: {metrics.keys()}"
            else:
                pytest.skip("calculate_metrics not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_calculate_auc_basic(self):
        """Test basic AUC calculation."""
        try:
            import solution

            if hasattr(solution, 'calculate_auc'):
                y_true = np.array([0, 0, 1, 1])
                y_prob = np.array([0.1, 0.4, 0.6, 0.9])

                auc = solution.calculate_auc(y_true, y_prob)
                assert auc is not None
                assert 0 <= auc <= 1, f"AUC should be in [0, 1], got {auc}"
            else:
                pytest.skip("calculate_auc not implemented")
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
                'train_logistic_regression',
                'fit_logistic_regression',
                'calculate_confusion_matrix',
                'calculate_metrics',
                'calculate_auc',
            ]

            for func_name in key_functions:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    assert func.__doc__ is not None, f"{func_name} should have a docstring"
                    assert len(func.__doc__) > 10, f"{func_name} docstring should be meaningful"

        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_pipeline_function_has_data_leakage_warning(self):
        """Test that pipeline functions mention data leakage."""
        try:
            import solution

            # Check pipeline-related functions
            pipeline_functions = [
                'create_classification_pipeline',
                'train_with_pipeline',
                'classification_with_pipeline',
            ]

            for func_name in pipeline_functions:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    doc = func.__doc__ or ""
                    # Docstring should mention data leakage or pipeline
                    text_lower = doc.lower()
                    assert 'leakage' in text_lower or 'pipeline' in text_lower, \
                        f"{func_name} docstring should mention data leakage or pipeline"

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
                'train_logistic_regression',
                'calculate_confusion_matrix',
                'calculate_metrics',
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

            for func_name in ['train_logistic_regression', 'fit_logistic_regression']:
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

            for func_name in ['train_logistic_regression', 'fit_logistic_regression']:
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

            for func_name in ['train_logistic_regression', 'fit_logistic_regression']:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    # Should raise an error
                    with pytest.raises((ValueError, IndexError, AssertionError)):
                        func(X, y)

        except ImportError:
            pytest.skip("solution.py not yet created")
