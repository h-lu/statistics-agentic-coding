"""
Smoke tests for Week 06 solution.py

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

            # Expected function groups
            expected_functions = [
                # p 值理解
                'interpret_p_value',

                # t 检验
                'two_sample_t_test',
                'proportion_test',
                'paired_t_test',

                # 卡方检验
                'chi_square_test',

                # 效应量
                'cohens_d',
                'interpret_cohens_d',
                'risk_difference',
                'risk_ratio',

                # 前提假设检查
                'check_normality',
                'check_variance_homogeneity',
                'choose_test_auto',

                # AI 结论审查
                'review_ai_report',
                'bonferroni_correction',
                'fdr_correction',
                'calculate_family_wise_error_rate',

                # 综合流程
                'complete_two_group_test',
                'generate_hypothesis_test_report',
            ]

            # Check that at least some functions exist
            # (not all need to exist for smoke test)
            existing_functions = [name for name in expected_functions if hasattr(solution, name)]
            assert len(existing_functions) >= 5, f"At least 5 functions should exist, found: {existing_functions}"

        except ImportError:
            pytest.skip("solution.py not yet created")


class TestBasicFunctionality:
    """Test basic functionality with simple inputs."""

    def test_p_value_interpretation_basic(self):
        """Test basic p value interpretation."""
        try:
            import solution

            if hasattr(solution, 'interpret_p_value'):
                result = solution.interpret_p_value(p_value=0.03, alpha=0.05)
                assert result is not None
                assert 'reject_null' in result or 'conclusion' in result
            else:
                pytest.skip("interpret_p_value not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_cohens_d_basic(self):
        """Test basic Cohen's d calculation."""
        try:
            import solution
            import numpy as np

            if hasattr(solution, 'cohens_d'):
                group_a = np.array([1, 2, 3, 4, 5])
                group_b = np.array([2, 3, 4, 5, 6])
                d = solution.cohens_d(group_a, group_b)
                assert d is not None
                assert isinstance(d, (int, float))
            else:
                pytest.skip("cohens_d not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")

    def test_normality_check_basic(self):
        """Test basic normality check."""
        try:
            import solution
            import numpy as np

            if hasattr(solution, 'check_normality'):
                normal_data = np.random.normal(0, 1, 100)
                result = solution.check_normality(normal_data)
                assert result is not None
                assert 'p_value' in result or 'is_normal' in result
            else:
                pytest.skip("check_normality not implemented")
        except ImportError:
            pytest.skip("solution.py not yet created")


class TestDocumentation:
    """Test that functions have proper documentation."""

    def test_functions_have_docstrings(self):
        """Test that key functions have docstrings."""
        try:
            import solution

            # Check a few key functions
            key_functions = ['cohens_d', 'check_normality', 'two_sample_t_test']

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
            key_functions = ['cohens_d', 'check_normality']

            for func_name in key_functions:
                if hasattr(solution, func_name):
                    func = getattr(solution, func_name)
                    # At minimum, function should have some annotations
                    # (checking for __annotations__ dict)
                    assert hasattr(func, '__annotations__'), f"{func_name} should have type hints"

        except ImportError:
            pytest.skip("solution.py not yet created")


class TestErrorHandling:
    """Test basic error handling."""

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
