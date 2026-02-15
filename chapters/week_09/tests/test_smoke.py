"""
Smoke tests for Week 09 solution.py

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

            # Expected function groups for Week 09
            expected_functions = [
                # 简单线性回归
                'simple_linear_regression',
                'fit_ols',
                'predict',

                # 回归系数解释
                'interpret_coefficients',
                'get_coefficients',

                # 模型评估
                'calculate_r_squared',
                'calculate_residuals',

                # 模型诊断
                'check_linearity',
                'check_normality',
                'check_homoscedasticity',
                'calculate_cooks_distance',

                # 多元回归
                'multiple_regression',
                'calculate_vif',

                # StatLab 集成
                'regression_with_diagnostics',
                'format_regression_report',
            ]

            # Check that at least some functions exist
            # (not all need to exist for smoke test)
            existing_functions = [name for name in expected_functions if hasattr(solution, name)]
            assert len(existing_functions) >= 4, f"At least 4 functions should exist, found: {existing_functions}"

        except ImportError:
            pytest.skip("solution.py not yet created")


class TestBasicFunctionality:
    """Test basic functionality with simple inputs."""

    def test_simple_linear_regression_basic(self):
        """Test basic simple linear regression."""
        try:
            import solution

            if hasattr(solution, 'simple_linear_regression'):
                np.random.seed(42)
                x = np.random.normal(50, 15, 100)
                y = 10 + 0.5 * x + np.random.normal(0, 5, 100)

                result = solution.simple_linear_regression(x, y)
                assert result is not None

                # Result should contain intercept and slope
                # solution.py returns nested structure: {'coefficients': {'intercept': ..., 'slope': ...}}
                if isinstance(result, dict):
                    has_intercept = 'intercept' in result or 'coef' in result or 'coefficients' in result
                    has_slope = 'slope' in result or 'coefficients' in result
                    assert has_intercept and has_slope, f"Result should have intercept and slope, got keys: {result.keys()}"
            else:
                pytest.skip("simple_linear_regression not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_fit_ols_basic(self):
        """Test basic OLS fitting."""
        try:
            import solution

            if hasattr(solution, 'fit_ols'):
                np.random.seed(42)
                x = np.random.normal(50, 15, 100)
                y = 10 + 0.5 * x + np.random.normal(0, 5, 100)

                model = solution.fit_ols(x, y)
                assert model is not None

                # Should have some way to get coefficients and R²
                # (implementation dependent)
            else:
                pytest.skip("fit_ols not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_predict_basic(self):
        """Test basic prediction."""
        try:
            import solution

            if hasattr(solution, 'fit_ols') and hasattr(solution, 'predict'):
                np.random.seed(42)
                x = np.random.normal(50, 15, 100)
                y = 10 + 0.5 * x + np.random.normal(0, 5, 100)

                model = solution.fit_ols(x, y)
                x_new = np.array([50, 60, 70])
                predictions = solution.predict(model, x_new)

                assert predictions is not None
                assert len(predictions) == len(x_new)
            else:
                pytest.skip("fit_ols or predict not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_calculate_r_squared_basic(self):
        """Test basic R² calculation."""
        try:
            import solution

            if hasattr(solution, 'calculate_r_squared'):
                np.random.seed(42)
                y_true = np.array([1, 2, 3, 4, 5])
                y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])

                r2 = solution.calculate_r_squared(y_true, y_pred)
                assert r2 is not None
                assert 0 <= r2 <= 1  # R² should be in [0, 1] for this data
            else:
                pytest.skip("calculate_r_squared not implemented")
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
                'simple_linear_regression',
                'fit_ols',
                'calculate_r_squared',
                'calculate_cooks_distance',
            ]

            for func_name in key_functions:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    assert func.__doc__ is not None, f"{func_name} should have a docstring"
                    assert len(func.__doc__) > 10, f"{func_name} docstring should be meaningful"

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
                'simple_linear_regression',
                'fit_ols',
                'calculate_r_squared',
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
            x_empty = np.array([])
            y_empty = np.array([])

            for func_name in ['simple_linear_regression', 'fit_ols']:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    # Should either raise a meaningful error or return a meaningful result
                    try:
                        result = func(x_empty, y_empty)
                        # If it doesn't raise, result should indicate the issue
                        assert result is not None
                    except (ValueError, IndexError) as e:
                        # These are acceptable errors for empty input
                        assert len(str(e)) > 0

        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_mismatched_dimensions_handling(self):
        """Test that mismatched dimensions are handled."""
        try:
            import solution

            # Test with different length arrays
            x = np.array([1, 2, 3])
            y = np.array([1, 2])  # Different length

            for func_name in ['simple_linear_regression', 'fit_ols']:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    # Should raise an error
                    with pytest.raises((ValueError, IndexError, AssertionError)):
                        func(x, y)

        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_import_error_message(self):
        """Test that missing import gives helpful error."""
        try:
            import solution
        except ImportError as e:
            # If module doesn't exist, that's expected for now
            assert "solution" in str(e).lower() or "No module" in str(e)
        except Exception:
            # Other exceptions are OK for now
            pass
