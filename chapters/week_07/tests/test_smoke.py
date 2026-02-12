"""
Week 07 烟雾测试（Smoke Test）

快速验证核心功能是否正常工作。
"""
from __future__ import annotations

import numpy as np
import pytest

# 导入需要测试的函数
try:
    from solution import (
        calculate_f_statistic,
        calculate_eta_squared,
        calculate_fwer,
        bonferroni_correction,
        interpret_tukey_hsd,
        calculate_cramers_v,
        chi_square_test,
        review_anova_report,
        anova_test,
    )
except ImportError:
    pytest.skip("solution.py not implemented yet", allow_module_level=True)


class TestSmokeBasicFunctionality:
    """测试基本功能是否可以运行"""

    @pytest.fixture
    def sample_groups(self):
        """创建样本数据"""
        np.random.seed(42)
        return [
            np.random.normal(100, 15, 50),
            np.random.normal(105, 15, 50),
            np.random.normal(98, 15, 50),
        ]

    def test_smoke_calculate_f_statistic(self, sample_groups):
        """烟雾测试：F 统计量计算"""
        result = calculate_f_statistic(sample_groups)
        assert 'f_statistic' in result
        assert 'p_value' in result

    def test_smoke_calculate_eta_squared(self, sample_groups):
        """烟雾测试：η² 效应量"""
        result = calculate_eta_squared(sample_groups)
        assert 'eta_squared' in result
        assert 0 <= result['eta_squared'] <= 1

    def test_smoke_calculate_fwer(self):
        """烟雾测试：FWER 计算"""
        result = calculate_fwer(alpha=0.05, n_tests=10)
        assert isinstance(result, (int, float))
        assert 0 <= result <= 1

    def test_smoke_bonferroni_correction(self):
        """烟雾测试：Bonferroni 校正"""
        result = bonferroni_correction(alpha=0.05, n_tests=10)
        assert 'corrected_alpha' in result
        assert result['corrected_alpha'] == 0.005

    def test_smoke_calculate_cramers_v(self):
        """烟雾测试：Cramér's V 计算"""
        result = calculate_cramers_v(chi2=10, n=100, min_dim=3)
        assert 'cramers_v' in result
        assert 0 <= result['cramers_v'] <= 1

    def test_smoke_chi_square_test(self):
        """烟雾测试：卡方检验"""
        import pandas as pd
        contingency_table = pd.DataFrame([
            [25, 25, 25, 25],
            [35, 15, 30, 20],
        ])
        result = chi_square_test(contingency_table)
        assert 'chi2' in result
        assert 'p_value' in result
        assert 'cramers_v' in result

    def test_smoke_review_anova_report(self):
        """烟雾测试：ANOVA 报告审查"""
        report = "ANOVA 结果：F=8.52, p=0.002"
        result = review_anova_report(report)
        assert 'has_issues' in result
        assert 'issues' in result

    def test_smoke_anova_test(self, sample_groups):
        """烟雾测试：完整 ANOVA 流程"""
        result = anova_test(sample_groups, check_assumptions=False)
        assert 'f_statistic' in result
        assert 'p_value' in result
        assert 'eta_squared' in result


class TestSmokeEndToEnd:
    """端到端工作流测试"""

    def test_complete_anova_workflow(self):
        """测试完整的 ANOVA 工作流"""
        # 1. 生成数据
        np.random.seed(42)
        groups = [
            np.random.normal(100, 15, 50),
            np.random.normal(108, 15, 50),
            np.random.normal(95, 15, 50),
        ]

        # 2. 计算 F 统计量
        f_result = calculate_f_statistic(groups)
        assert 'f_statistic' in f_result
        assert 'p_value' in f_result

        # 3. 计算效应量
        eta_result = calculate_eta_squared(groups)
        assert 'eta_squared' in eta_result
        assert 0 <= eta_result['eta_squared'] <= 1

        # 4. 完整 ANOVA
        anova_result = anova_test(groups, check_assumptions=False)
        assert 'decision' in anova_result

        # 5. 计算多重比较风险
        fwer = calculate_fwer(alpha=0.05, n_tests=10)
        assert fwer > 0.05

        # 6. Bonferroni 校正
        bonf_result = bonferroni_correction(alpha=0.05, n_tests=10)
        assert bonf_result['corrected_alpha'] == 0.005

        # 7. 流程成功
        assert True

    def test_complete_chisquare_workflow(self):
        """测试完整的卡方检验工作流"""
        # 1. 创建列联表
        import pandas as pd
        contingency_table = pd.DataFrame([
            [45, 30, 18, 7],
            [38, 32, 22, 8],
            [52, 28, 15, 5],
        ])

        # 2. 卡方检验
        result = chi_square_test(contingency_table)
        assert 'chi2' in result
        assert 'p_value' in result
        assert 'cramers_v' in result

        # 3. 流程成功
        assert True

    def test_complete_review_workflow(self):
        """测试完整的报告审查工作流"""
        # 有问题的报告
        bad_report = """
        ANOVA 结果：F=8.52, p=0.002。

        结论：
        1. 上海和深圳显著高于其他城市。
        2. 城市影响用户消费。
        """

        # 审查报告
        result = review_anova_report(bad_report)
        assert 'has_issues' in result
        assert 'issues' in result
        assert result['has_issues'] is True

        # 流程成功
        assert True
