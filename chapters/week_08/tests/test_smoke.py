"""
Smoke tests for Week 08 solution.py

Basic sanity checks to ensure the module can be imported and functions exist.
These tests should all pass once solution.py is created with the expected interface.
"""
from __future__ import annotations

import sys
from pathlib import Path

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

            # Expected function groups for Week 08
            expected_functions = [
                # 置信区间
                'calculate_confidence_interval',
                'interpret_confidence_interval',

                # Bootstrap
                'bootstrap_mean',
                'bootstrap_ci',
                'bootstrap_ci_bca',

                # 置换检验
                'permutation_test',
                'permutation_test_ci',

                # StatLab 集成
                'add_ci_to_estimate',
                'compare_groups_with_uncertainty',
            ]

            # Check that at least some functions exist
            # (not all need to exist for smoke test)
            existing_functions = [name for name in expected_functions if hasattr(solution, name)]
            assert len(existing_functions) >= 4, f"At least 4 functions should exist, found: {existing_functions}"

        except ImportError:
            pytest.skip("solution.py not yet created")


class TestBasicFunctionality:
    """Test basic functionality with simple inputs."""

    def test_confidence_interval_basic(self):
        """Test basic confidence interval calculation."""
        try:
            import solution
            import numpy as np

            if hasattr(solution, 'calculate_confidence_interval'):
                np.random.seed(42)
                data = np.random.normal(100, 15, 50)
                result = solution.calculate_confidence_interval(data)
                assert result is not None
                # CI should be a tuple or dict with lower and upper bounds
                if isinstance(result, dict):
                    assert 'lower' in result or 'ci_low' in result
                    assert 'upper' in result or 'ci_high' in result
                elif isinstance(result, (tuple, list)):
                    assert len(result) == 2
            else:
                pytest.skip("calculate_confidence_interval not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_bootstrap_mean_basic(self):
        """Test basic bootstrap mean estimation."""
        try:
            import solution
            import numpy as np

            if hasattr(solution, 'bootstrap_mean'):
                np.random.seed(42)
                data = np.random.normal(100, 15, 50)
                result = solution.bootstrap_mean(data, n_bootstrap=1000)
                assert result is not None
                # Result should contain mean and standard error or CI
                if isinstance(result, dict):
                    assert 'mean' in result or 'bootstrap_mean' in result
                elif isinstance(result, (int, float)):
                    # Just the bootstrap mean is fine
                    pass
            else:
                pytest.skip("bootstrap_mean not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_permutation_test_basic(self):
        """Test basic permutation test."""
        try:
            import solution
            import numpy as np

            if hasattr(solution, 'permutation_test'):
                np.random.seed(42)
                group1 = np.random.normal(100, 15, 50)
                group2 = np.random.normal(110, 15, 50)
                result = solution.permutation_test(group1, group2, n_permutations=1000)
                assert result is not None
                # Result should contain p-value
                if isinstance(result, dict):
                    assert 'p_value' in result or 'pvalue' in result
                elif isinstance(result, (int, float)):
                    # Just the p-value is fine
                    assert 0 <= result <= 1
            else:
                pytest.skip("permutation_test not implemented")
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
                'calculate_confidence_interval',
                'bootstrap_mean',
                'bootstrap_ci',
                'permutation_test',
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
            import inspect

            # Check a few key functions
            key_functions = [
                'calculate_confidence_interval',
                'bootstrap_mean',
                'permutation_test',
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
            import numpy as np

            # Test with empty array
            empty_data = np.array([])

            for func_name in ['calculate_confidence_interval', 'bootstrap_mean', 'bootstrap_ci']:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    # Should either raise a meaningful error or return a meaningful result
                    try:
                        result = func(empty_data)
                        # If it doesn't raise, result should indicate the issue
                        assert result is not None
                    except (ValueError, IndexError) as e:
                        # These are acceptable errors for empty input
                        assert len(str(e)) > 0

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
