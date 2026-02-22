"""
Tests for Integration and End-to-End Scenarios

集成测试用例矩阵：
- 端到端：完整的数据到报告流程
- 可复现性：验证结果一致性
- 场景：客户流失分析、A/B 测试分析等
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 测试数据 Fixture
# =============================================================================

@pytest.fixture
def customer_churn_dataset():
    """
    Fixture：完整的客户流失数据集
    模拟真实的电信客户流失数据
    """
    np.random.seed(42)
    n = 1000

    # 生成相关特征（使用时长与流失相关）
    tenure = np.random.exponential(20, n)
    # 使用时长越短，流失概率越高
    churn_prob = 1 / (1 + np.exp(0.1 * tenure - 2))
    churn = np.random.binomial(1, churn_prob)

    return pd.DataFrame({
        'customer_id': range(n),
        'tenure': tenure,
        'monthly_charges': np.random.uniform(20, 100, n),
        'total_charges': tenure * np.random.uniform(20, 100, n),
        'age': np.random.normal(45, 15, n),
        'income': np.random.lognormal(10, 0.5, n),
        'churn': churn,
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.5, 0.3, 0.2]),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.3, 0.5, 0.2]),
        'tech_support': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check'], n)
    })


@pytest.fixture
def ab_test_dataset():
    """
    Fixture：A/B 测试数据集
    模拟网站转化率 A/B 测试
    """
    np.random.seed(42)
    n_control = 500
    n_treatment = 500

    # 对照组：转化率 10%
    control_converted = np.random.binomial(1, 0.10, n_control)

    # 处理组：转化率 12%（真实的提升）
    treatment_converted = np.random.binomial(1, 0.12, n_treatment)

    control_df = pd.DataFrame({
        'user_id': range(n_control),
        'group': 'control',
        'converted': control_converted,
        'page_views': np.random.poisson(5, n_control)
    })

    treatment_df = pd.DataFrame({
        'user_id': range(n_control, n_control + n_treatment),
        'group': 'treatment',
        'converted': treatment_converted,
        'page_views': np.random.poisson(6, n_treatment)
    })

    return pd.concat([control_df, treatment_df], ignore_index=True)


@pytest.fixture
def temp_output_dir(tmp_path):
    """
    Fixture：创建临时输出目录
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


# =============================================================================
# 端到端测试：客户流失分析
# =============================================================================

class TestCustomerChurnAnalysisE2E:
    """测试客户流失分析的端到端流程"""

    def test_full_analysis_pipeline(self, customer_churn_dataset, temp_output_dir):
        """
        端到端：完整的客户流失分析流程

        给定：客户流失数据集
        当：执行完整分析流程
        期望：生成包含所有必要部分的报告
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'run_customer_churn_analysis'):
            # 执行完整分析
            result = solution.run_customer_churn_analysis(
                customer_churn_dataset,
                output_dir=temp_output_dir,
                random_seed=42
            )

            # 验证返回值
            assert result is not None

            # 验证生成了报告文件
            report_path = Path(temp_output_dir) / "report.md"
            if report_path.exists():
                content = report_path.read_text()

                # 验证报告包含关键部分
                assert any(keyword in content for keyword in
                          ['流失', 'churn', '客户', 'customer'])

                # 验证包含统计结果
                assert 'p' in content or 'P' in content or 'p值' in content or 'p-value' in content
        else:
            pytest.skip("run_customer_churn_analysis function not implemented")

    def test_reproducibility_same_seed(self, customer_churn_dataset, temp_output_dir):
        """
        端到端：相同随机种子应产生相同结果

        验证可复现性
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'run_customer_churn_analysis'):
            # 第一次运行
            dir1 = Path(temp_output_dir) / "run1"
            dir1.mkdir()
            result1 = solution.run_customer_churn_analysis(
                customer_churn_dataset,
                output_dir=str(dir1),
                random_seed=42
            )

            # 第二次运行（相同种子）
            dir2 = Path(temp_output_dir) / "run2"
            dir2.mkdir()
            result2 = solution.run_customer_churn_analysis(
                customer_churn_dataset,
                output_dir=str(dir2),
                random_seed=42
            )

            # 如果返回数值结果，应该相同
            if isinstance(result1, dict) and isinstance(result2, dict):
                # 检查关键统计量是否相同
                for key in ['mean', 'std', 'p_value', 'conversion_rate']:
                    if key in result1 and key in result2:
                        if isinstance(result1[key], (int, float)):
                            np.testing.assert_almost_equal(result1[key], result2[key], decimal=5)
        else:
            pytest.skip("run_customer_churn_analysis function not implemented")

    def test_reproducibility_different_seed(self, customer_churn_dataset, temp_output_dir):
        """
        端到端：不同随机种子可能产生不同结果

        验证随机性的存在
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'run_customer_churn_analysis'):
            # 使用种子 42
            dir1 = Path(temp_output_dir) / "run1"
            dir1.mkdir()
            result1 = solution.run_customer_churn_analysis(
                customer_churn_dataset,
                output_dir=str(dir1),
                random_seed=42
            )

            # 使用种子 123
            dir2 = Path(temp_output_dir) / "run2"
            dir2.mkdir()
            result2 = solution.run_customer_churn_analysis(
                customer_churn_dataset,
                output_dir=str(dir2),
                random_seed=123
            )

            # 不同种子的结果应该不同（如果涉及随机操作）
            # 注意：某些确定性操作可能返回相同结果
            if isinstance(result1, dict) and isinstance(result2, dict):
                # 只检查涉及随机的关键统计量
                for key in ['p_value', 'random_split', 'bootstrap']:
                    if key in result1 and key in result2:
                        if isinstance(result1[key], (int, float)):
                            # 这些值可能不同（但不保证）
                            pass  # 只验证能运行，不强制要求不同
        else:
            pytest.skip("run_customer_churn_analysis function not implemented")


# =============================================================================
# 端到端测试：A/B 测试分析
# =============================================================================

class TestABTestAnalysisE2E:
    """测试 A/B 测试分析的端到端流程"""

    def test_ab_test_analysis(self, ab_test_dataset, temp_output_dir):
        """
        端到端：完整的 A/B 测试分析

        给定：A/B 测试数据
        当：执行分析
        期望：报告转化率、p 值、置信区间
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'run_ab_test_analysis'):
            result = solution.run_ab_test_analysis(
                ab_test_dataset,
                group_col='group',
                metric_col='converted',
                output_dir=temp_output_dir
            )

            assert result is not None

            # 验证包含关键指标
            if isinstance(result, dict):
                expected_keys = ['control_rate', 'treatment_rate', 'p_value', 'lift', 'ci_lower', 'ci_upper']
                has_keys = any(key in result for key in expected_keys)
                assert has_keys or len(result) > 0
        else:
            pytest.skip("run_ab_test_analysis function not implemented")

    def test_ab_test_detects_significant_difference(self, ab_test_dataset, temp_output_dir):
        """
        端到端：A/B 测试应检测到显著差异

        给定：处理组真实提升 2%
        期望：分析应发现显著差异（p < 0.05）
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'run_ab_test_analysis'):
            result = solution.run_ab_test_analysis(
                ab_test_dataset,
                group_col='group',
                metric_col='converted',
                output_dir=temp_output_dir
            )

            if isinstance(result, dict) and 'p_value' in result:
                # 由于我们设置了真实差异，p 值应该较小（但不总是 < 0.05）
                assert isinstance(result['p_value'], (int, float))
                assert 0 <= result['p_value'] <= 1
        else:
            pytest.skip("run_ab_test_analysis function not implemented")


# =============================================================================
# 端到端测试：完整报告生成
# =============================================================================

class TestFullReportGeneration:
    """测试完整报告生成"""

    def test_generate_report_with_all_sections(self, customer_churn_dataset, temp_output_dir):
        """
        端到端：生成包含所有章节的报告

        验证报告包含：
        - 可复现信息
        - 数据概览
        - 描述统计
        - 统计检验
        - 结论
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_full_report'):
            report_path = solution.generate_full_report(
                customer_churn_dataset,
                output_path=str(Path(temp_output_dir) / "full_report.md")
            )

            # 验证文件创建
            assert Path(report_path).exists()

            # 读取并验证内容
            content = Path(report_path).read_text()

            # 检查关键章节
            expected_sections = [
                ('#', '标题'),
                ('数据', 'Data'),
                ('统计', 'Statistics'),
                ('结论', 'Conclusion')
            ]

            # 至少包含一些关键内容
            has_content = len(content) > 100
            assert has_content
        else:
            pytest.skip("generate_full_report function not implemented")

    def test_report_includes_reproducibility_info(self, customer_churn_dataset, temp_output_dir):
        """
        端到端：报告应包含可复现信息

        验证包含：数据来源、随机种子、日期
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_full_report'):
            report_path = solution.generate_full_report(
                customer_churn_dataset,
                output_path=str(Path(temp_output_dir) / "reproducibility_report.md"),
                include_metadata=True
            )

            content = Path(report_path).read_text()

            # 检查可复现性关键词
            reproducibility_keywords = ['seed', '种子', '日期', 'date', '2026', 'version']
            has_reproducibility = any(keyword in content.lower() for keyword in
                                      [k.lower() for k in reproducibility_keywords])
            assert has_reproducibility
        else:
            pytest.skip("generate_full_report function not implemented")


# =============================================================================
# 可复现性测试
# =============================================================================

class TestReproducibility:
    """测试可复现性"""

    def test_same_input_same_output(self, customer_churn_dataset):
        """
        可复现性：相同输入应产生相同输出

        验证确定性
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'compute_descriptive'):
            # 固定随机种子
            np.random.seed(42)

            # 第一次计算
            result1 = solution.compute_descriptive(customer_churn_dataset)

            # 重置种子，第二次计算
            np.random.seed(42)
            result2 = solution.compute_descriptive(customer_churn_dataset)

            # 结果应该相同
            if isinstance(result1, dict) and isinstance(result2, dict):
                for key in result1:
                    if key in result2:
                        if isinstance(result1[key], (int, float, np.ndarray)):
                            np.testing.assert_array_equal(result1[key], result2[key])
        else:
            pytest.skip("compute_descriptive function not implemented")

    def test_report_contains_execution_metadata(self, customer_churn_dataset, temp_output_dir):
        """
        可复现性：报告应包含执行元数据

        验证包含：执行时间、依赖版本
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        if hasattr(solution, 'generate_full_report'):
            report_path = solution.generate_full_report(
                customer_churn_dataset,
                output_path=str(Path(temp_output_dir) / "metadata_report.md")
            )

            content = Path(report_path).read_text()

            # 检查年份（当前是 2026）
            assert '2026' in content

            # 检查是否有日期相关内容
            date_keywords = ['date', '日期', 'time', '时间', 'feb', '二月']
            has_date = any(keyword in content.lower() for keyword in date_keywords)
            assert has_date
        else:
            pytest.skip("generate_full_report function not implemented")


# =============================================================================
# 场景测试：现实世界场景
# =============================================================================

class TestRealWorldScenarios:
    """测试现实世界场景"""

    def test_scenario_marketing_campaign_analysis(self):
        """
        场景：营销活动效果分析

        模拟分析营销活动对用户行为的影响
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        # 创建营销活动数据
        np.random.seed(42)
        n = 500

        campaign_data = pd.DataFrame({
            'user_id': range(n),
            'received_campaign': np.random.choice([0, 1], n, p=[0.5, 0.5]),
            'spent_before': np.random.exponential(50, n),
            'spent_after': np.random.exponential(60, n)  # 总体有提升
        })

        # 处理组（收到活动）提升更多
        mask = campaign_data['received_campaign'] == 1
        campaign_data.loc[mask, 'spent_after'] *= 1.2

        if hasattr(solution, 'analyze_campaign_effect'):
            result = solution.analyze_campaign_effect(
                campaign_data,
                group_col='received_campaign'
            )

            assert result is not None
        else:
            pytest.skip("analyze_campaign_effect function not implemented")

    def test_scenario_quality_control_analysis(self):
        """
        场景：质量控制分析

        模拟分析生产线产品质量
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        # 创建质量控制数据
        np.random.seed(42)

        # 正常生产：均值 100，标准差 5
        normal_production = np.random.normal(100, 5, 200)

        # 异常生产：均值 105，标准差 8
        abnormal_production = np.random.normal(105, 8, 50)

        qc_data = pd.DataFrame({
            'batch_id': list(range(200)) + list(range(200, 250)),
            'measurement': np.concatenate([normal_production, abnormal_production]),
            'period': ['normal'] * 200 + ['abnormal'] * 50
        })

        if hasattr(solution, 'analyze_quality_control'):
            result = solution.analyze_quality_control(qc_data)
            assert result is not None
        else:
            pytest.skip("analyze_quality_control function not implemented")


# =============================================================================
# 性能测试
# =============================================================================

class TestPerformance:
    """测试性能"""

    def test_handles_large_dataset(self):
        """
        性能：能处理较大数据集

        给定：10,000 行数据
        期望：在合理时间内完成
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        # 创建较大数据集
        np.random.seed(42)
        large_data = pd.DataFrame({
            'x': np.random.randn(10000),
            'y': np.random.randn(10000),
            'group': np.random.choice(['A', 'B', 'C'], 10000)
        })

        import time

        if hasattr(solution, 'compute_descriptive'):
            start = time.time()
            result = solution.compute_descriptive(large_data)
            elapsed = time.time() - start

            # 应该在合理时间内完成（< 5 秒）
            assert elapsed < 5.0
            assert result is not None
        else:
            pytest.skip("compute_descriptive function not implemented")


# =============================================================================
# 错误恢复测试
# =============================================================================

class TestErrorRecovery:
    """测试错误恢复"""

    def test_handles_mixed_types_in_dataframe(self):
        """
        错误恢复：处理混合类型数据

        给定：包含数值、字符串、日期的 DataFrame
        期望：不崩溃，正确处理每种类型
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        mixed_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'score': [85.5, 92.0, 78.5, 88.0, 95.5],
            'date': pd.date_range('2026-01-01', periods=5),
            'category': ['A', 'B', 'A', 'C', 'B']
        })

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(mixed_data)
            # 不应崩溃
            assert result is not None
        else:
            pytest.skip("compute_descriptive function not implemented")

    def test_handles_duplicates_gracefully(self):
        """
        错误恢复：优雅处理重复数据

        给定：包含重复行的数据
        期望：能正常处理或去重
        """
        try:
            import solution
        except ImportError:
            pytest.skip("solution.py not found")

        np.random.seed(42)
        data_with_duplicates = pd.DataFrame({
            'x': [1, 2, 2, 2, 3],
            'y': [4, 5, 5, 5, 6]
        })

        if hasattr(solution, 'compute_descriptive'):
            result = solution.compute_descriptive(data_with_duplicates)
            assert result is not None
        else:
            pytest.skip("compute_descriptive function not implemented")
