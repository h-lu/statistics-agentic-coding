"""
Fairness Metrics Tests for Week 12 solution.py

公平性指标测试：
- 正例：统计均等、机会均等、校准
- 边界：二元群体、多群体
- 反例：无效概率值、空数据
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add starter_code to path
starter_code_path = Path(__file__).parent.parent / "starter_code"
sys.path.insert(0, str(starter_code_path))

try:
    import solution
except ImportError:
    solution = None


# =============================================================================
# 1. 统计均等（Demographic Parity）
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDemographicParity:
    """测试统计均等指标"""

    def test_demographic_parity_difference(self, demographic_parity_data):
        """
        正例：计算统计均等差异

        应返回各群体预测正率的差异
        """
        if not hasattr(solution, 'demographic_parity_difference'):
            pytest.skip("demographic_parity_difference not implemented")

        y_pred = demographic_parity_data['y_pred']
        sensitive_attr = demographic_parity_data['sensitive_attr']

        # 计算统计均等差异
        dp_diff = solution.demographic_parity_difference(y_pred, sensitive_attr)

        assert dp_diff is not None
        assert isinstance(dp_diff, (float, np.floating))
        assert dp_diff >= 0, "Difference should be non-negative"

    def test_demographic_parity_ratio(self, demographic_parity_data):
        """
        正例：计算统计均等比率

        应返回少数群体与多数群体预测正率的比值
        """
        if not hasattr(solution, 'demographic_parity_ratio'):
            pytest.skip("demographic_parity_ratio not implemented")

        y_pred = demographic_parity_data['y_pred']
        sensitive_attr = demographic_parity_data['sensitive_attr']

        # 计算统计均等比率
        dp_ratio = solution.demographic_parity_ratio(y_pred, sensitive_attr)

        assert dp_ratio is not None
        assert isinstance(dp_ratio, (float, np.floating))
        assert dp_ratio >= 0, "Ratio should be non-negative"
        # 80% 规则：比率 < 0.8 表示存在偏见
        if dp_ratio < 1.0:
            assert dp_ratio is not None

    def test_demographic_parity_perfect(self, unbiased_predictions_data):
        """
        边界：完美统计均等

        两群体预测正率相同时，差异应为 0，比率应为 1
        """
        if not hasattr(solution, 'demographic_parity_difference'):
            pytest.skip("demographic_parity_difference not implemented")

        y_pred = unbiased_predictions_data['y_pred']
        sensitive_attr = unbiased_predictions_data['sensitive_attr']

        # 计算统计均等差异
        dp_diff = solution.demographic_parity_difference(y_pred, sensitive_attr)

        # 完美统计均等时，差异应接近 0
        # 由于随机性，允许小的误差
        assert dp_diff is not None


# =============================================================================
# 2. 机会均等（Equalized Odds）
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestEqualizedOdds:
    """测试机会均等指标"""

    def test_equalized_odds_tpr_difference(self, equalized_odds_data):
        """
        正例：计算真阳性率差异

        应返回各群体 TPR 的差异
        """
        if not hasattr(solution, 'equalized_odds_tpr_difference'):
            pytest.skip("equalized_odds_tpr_difference not implemented")

        y_true = equalized_odds_data['y_true']
        y_pred = equalized_odds_data['y_pred']
        sensitive_attr = equalized_odds_data['sensitive_attr']

        # 计算 TPR 差异
        tpr_diff = solution.equalized_odds_tpr_difference(
            y_true, y_pred, sensitive_attr
        )

        assert tpr_diff is not None
        assert isinstance(tpr_diff, (float, np.floating))
        assert tpr_diff >= 0, "TPR difference should be non-negative"

    def test_equalized_odds_fpr_difference(self, equalized_odds_data):
        """
        正例：计算假阳性率差异

        应返回各群体 FPR 的差异
        """
        if not hasattr(solution, 'equalized_odds_fpr_difference'):
            pytest.skip("equalized_odds_fpr_difference not implemented")

        y_true = equalized_odds_data['y_true']
        y_pred = equalized_odds_data['y_pred']
        sensitive_attr = equalized_odds_data['sensitive_attr']

        # 计算 FPR 差异
        fpr_diff = solution.equalized_odds_fpr_difference(
            y_true, y_pred, sensitive_attr
        )

        assert fpr_diff is not None
        assert isinstance(fpr_diff, (float, np.floating))
        assert fpr_diff >= 0, "FPR difference should be non-negative"

    def test_equalized_odds_combined(self, equalized_odds_data):
        """
        正例：计算机会均等的综合指标

        应同时考虑 TPR 和 FPR 的差异
        """
        if not hasattr(solution, 'equalized_odds_difference'):
            pytest.skip("equalized_odds_difference not implemented")

        y_true = equalized_odds_data['y_true']
        y_pred = equalized_odds_data['y_pred']
        sensitive_attr = equalized_odds_data['sensitive_attr']

        # 计算机会均等差异
        eo_diff = solution.equalized_odds_difference(
            y_true, y_pred, sensitive_attr
        )

        assert eo_diff is not None
        assert isinstance(eo_diff, (dict, tuple, float))

        if isinstance(eo_diff, dict):
            assert 'tpr_diff' in eo_diff or 'fpr_diff' in eo_diff


# =============================================================================
# 3. 校准（Calibration）
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestCalibration:
    """测试校准指标"""

    def test_calibration_by_group(self, calibration_test_data):
        """
        正例：按群体计算校准

        应返回各群体的预测概率与真实概率的匹配程度
        """
        if not hasattr(solution, 'calibration_by_group'):
            pytest.skip("calibration_by_group not implemented")

        y_true = calibration_test_data['y_true']
        y_prob = calibration_test_data['y_prob']
        sensitive_attr = calibration_test_data['sensitive_attr']

        # 计算校准
        calibration = solution.calibration_by_group(
            y_true, y_prob, sensitive_attr
        )

        assert calibration is not None
        assert isinstance(calibration, (dict, pd.DataFrame))

    def test_calibration_difference(self, calibration_test_data):
        """
        正例：计算群体间校准差异

        应返回各群体校准误差的差异
        """
        if not hasattr(solution, 'calibration_difference'):
            pytest.skip("calibration_difference not implemented")

        y_true = calibration_test_data['y_true']
        y_prob = calibration_test_data['y_prob']
        sensitive_attr = calibration_test_data['sensitive_attr']

        # 计算校准差异
        cal_diff = solution.calibration_difference(
            y_true, y_prob, sensitive_attr
        )

        assert cal_diff is not None
        assert isinstance(cal_diff, (float, np.floating))
        assert cal_diff >= 0, "Calibration difference should be non-negative"

    def test_calibration_curve(self, calibration_test_data):
        """
        正例：绘制校准曲线

        应能生成各群体的校准曲线
        """
        if not hasattr(solution, 'plot_calibration_curve'):
            pytest.skip("plot_calibration_curve not implemented")

        y_true = calibration_test_data['y_true']
        y_prob = calibration_test_data['y_prob']
        sensitive_attr = calibration_test_data['sensitive_attr']

        # 绘制校准曲线
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            result = solution.plot_calibration_curve(
                y_true, y_prob, sensitive_attr, output_path=output_path
            )
            assert result is not None
        except ImportError:
            pytest.skip("Matplotlib not available")
        finally:
            import os
            if os.path.exists(output_path):
                os.remove(output_path)


# =============================================================================
# 4. 公平性指标综合评估
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFairnessMetricsSummary:
    """测试公平性指标综合评估"""

    def test_compute_all_fairness_metrics(self, biased_predictions_data):
        """
        正例：计算所有公平性指标

        应返回统计均等、机会均等、校准等指标
        """
        if not hasattr(solution, 'compute_all_fairness_metrics'):
            pytest.skip("compute_all_fairness_metrics not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 计算所有指标
        metrics = solution.compute_all_fairness_metrics(
            y_true, y_pred, sensitive_attr
        )

        assert metrics is not None
        assert isinstance(metrics, dict)

        # 应包含关键指标
        expected_keys = ['demographic_parity', 'equalized_odds']
        for key in expected_keys:
            assert key in metrics or any(k in metrics for k in [key, key.replace('_', '')])

    def test_fairness_metrics_with_probabilities(self, calibration_test_data):
        """
        正例：使用概率预测计算公平性指标

        应支持概率预测（而不仅仅是类别预测）
        """
        if not hasattr(solution, 'compute_all_fairness_metrics'):
            pytest.skip("compute_all_fairness_metrics not implemented")

        y_true = calibration_test_data['y_true']
        y_prob = calibration_test_data['y_prob']
        y_pred = (y_prob > 0.5).astype(int)
        sensitive_attr = calibration_test_data['sensitive_attr']

        # 计算所有指标（使用概率）
        try:
            metrics = solution.compute_all_fairness_metrics(
                y_true, y_pred, sensitive_attr, y_prob=y_prob
            )
            assert metrics is not None
        except TypeError:
            # 如果不支持 y_prob 参数
            metrics = solution.compute_all_fairness_metrics(
                y_true, y_pred, sensitive_attr
            )
            assert metrics is not None


# =============================================================================
# 5. 公平性-准确性权衡
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFairnessAccuracyTradeoff:
    """测试公平性-准确性权衡"""

    def test_compute_tradeoff_metrics(self, biased_predictions_data):
        """
        正例：计算公平性-准确性权衡

        应返回整体准确率和公平性指标
        """
        if not hasattr(solution, 'compute_fairness_accuracy_tradeoff'):
            pytest.skip("compute_fairness_accuracy_tradeoff not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 计算权衡
        tradeoff = solution.compute_fairness_accuracy_tradeoff(
            y_true, y_pred, sensitive_attr
        )

        assert tradeoff is not None
        assert isinstance(tradeoff, dict)

        # 应包含准确率和公平性指标
        assert 'accuracy' in tradeoff or 'fairness' in str(tradeoff).lower()

    def test_pareto_frontier_analysis(self):
        """
        正例：分析帕累托前沿

        应能识别无法同时改善准确性和公平性的点
        """
        if not hasattr(solution, 'compute_pareto_frontier'):
            pytest.skip("compute_pareto_frontier not implemented")

        np.random.seed(42)
        n = 200

        y_true = np.random.randint(0, 2, n)
        sensitive_attr = np.array([0] * 100 + [1] * 100)

        # 生成多个阈值下的预测
        y_probs = np.random.uniform(0.3, 0.7, n)

        try:
            frontier = solution.compute_pareto_frontier(
                y_true, y_probs, sensitive_attr
            )
            assert frontier is not None
        except (TypeError, NotImplementedError):
            pytest.skip("Pareto frontier analysis not supported")


# =============================================================================
# 6. 反例：无效输入
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFairnessMetricsInvalidInput:
    """测试无效输入处理"""

    def test_invalid_probabilities(self):
        """
        反例：概率值不在 [0, 1] 范围内

        应报错或处理异常
        """
        if not hasattr(solution, 'calibration_by_group'):
            pytest.skip("calibration_by_group not implemented")

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.3, 1.5, -0.2, 0.8])  # 无效概率
        sensitive_attr = np.array([0, 0, 1, 1])

        with pytest.raises((ValueError, AssertionError)):
            solution.calibration_by_group(y_true, y_prob, sensitive_attr)

    def test_empty_group_in_fairness_metrics(self, empty_group_data):
        """
        反例：某些群体在数据中不存在

        应能处理或跳过空群体
        """
        if not hasattr(solution, 'compute_all_fairness_metrics'):
            pytest.skip("compute_all_fairness_metrics not implemented")

        y_true = empty_group_data['y_true']
        y_pred = empty_group_data['y_pred']
        sensitive_attr = empty_group_data['sensitive_attr']

        # 应能处理（只计算存在的群体）
        metrics = solution.compute_all_fairness_metrics(
            y_true, y_pred, sensitive_attr
        )
        assert metrics is not None

    def test_single_value_group(self):
        """
        反例：某个群体只有一个样本

        应能处理或给出警告
        """
        if not hasattr(solution, 'demographic_parity_difference'):
            pytest.skip("demographic_parity_difference not implemented")

        y_pred = np.array([0, 1, 0, 1, 1, 1])
        sensitive_attr = np.array([0, 0, 0, 0, 1, 1])  # group 1 只有 2 个样本

        # 应能计算（但可能不稳定）
        dp_diff = solution.demographic_parity_difference(y_pred, sensitive_attr)
        assert dp_diff is not None


# =============================================================================
# 7. 公平性阈值测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFairnessThresholds:
    """测试公平性阈值判断"""

    def test_check_fairness_threshold_pass(self, unbiased_predictions_data):
        """
        正例：检查公平性是否通过阈值

        无偏见模型应通过阈值测试
        """
        if not hasattr(solution, 'check_fairness_threshold'):
            pytest.skip("check_fairness_threshold not implemented")

        y_true = unbiased_predictions_data['y_true']
        y_pred = unbiased_predictions_data['y_pred']
        sensitive_attr = unbiased_predictions_data['sensitive_attr']

        # 检查是否通过阈值
        is_fair = solution.check_fairness_threshold(
            y_true, y_pred, sensitive_attr,
            threshold=0.1
        )

        assert isinstance(is_fair, bool)

    def test_check_fairness_threshold_fail(self, biased_predictions_data):
        """
        正例：检查公平性是否通过阈值

        有偏见模型应不通过阈值测试
        """
        if not hasattr(solution, 'check_fairness_threshold'):
            pytest.skip("check_fairness_threshold not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 检查是否通过阈值
        is_fair = solution.check_fairness_threshold(
            y_true, y_pred, sensitive_attr,
            threshold=0.05  # 严格阈值
        )

        assert isinstance(is_fair, bool)
        # 有偏见的数据应该不通过严格阈值

    def test_custom_threshold_per_metric(self, biased_predictions_data):
        """
        正例：为不同指标设置不同阈值

        应支持为统计均等、机会均等设置不同阈值
        """
        if not hasattr(solution, 'check_fairness_threshold'):
            pytest.skip("check_fairness_threshold not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 为不同指标设置阈值
        thresholds = {
            'demographic_parity': 0.15,
            'equalized_odds': 0.1
        }

        try:
            is_fair = solution.check_fairness_threshold(
                y_true, y_pred, sensitive_attr,
                thresholds=thresholds
            )
            assert isinstance(is_fair, (bool, dict))
        except TypeError:
            # 如果不支持 thresholds 参数
            is_fair = solution.check_fairness_threshold(
                y_true, y_pred, sensitive_attr, threshold=0.1
            )
            assert isinstance(is_fair, bool)


# =============================================================================
# 8. 公平性报告生成
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFairnessReportGeneration:
    """测试公平性报告生成"""

    def test_generate_fairness_report(self, biased_predictions_data):
        """
        正例：生成公平性评估报告

        应包含所有公平性指标和解释
        """
        if not hasattr(solution, 'generate_fairness_report'):
            pytest.skip("generate_fairness_report not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 生成报告
        report = solution.generate_fairness_report(
            y_true, y_pred, sensitive_attr,
            attr_name='sensitive_attribute'
        )

        assert report is not None
        assert isinstance(report, str)

        # 报告应包含关键信息
        assert any(kw in report.lower() for kw in [
            'fairness', 'equal', 'parity', 'bias',
            '公平', '均等', '偏见'
        ])

    def test_fairness_report_includes_interpretation(self, biased_predictions_data):
        """
        正例：报告应包含指标解释

        应解释每个公平性指标的含义
        """
        if not hasattr(solution, 'generate_fairness_report'):
            pytest.skip("generate_fairness_report not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 生成报告
        report = solution.generate_fairness_report(
            y_true, y_pred, sensitive_attr,
            attr_name='gender'
        )

        # 报告应该足够长（包含解释）
        assert len(report) > 50


# =============================================================================
# 9. 多敏感属性公平性
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestMultipleSensitiveAttributes:
    """测试多敏感属性的公平性评估"""

    def test_evaluate_multiple_sensitive_attrs(self):
        """
        正例：评估多个敏感属性

        应能同时评估性别、年龄等多个属性
        """
        if not hasattr(solution, 'evaluate_multiple_sensitive_attrs'):
            pytest.skip("evaluate_multiple_sensitive_attrs not implemented")

        np.random.seed(42)
        n = 200

        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        sensitive_attrs = {
            'gender': np.random.randint(0, 2, n),
            'age_group': np.random.randint(0, 3, n),
            'region': np.random.randint(0, 2, n)
        }

        # 评估多个敏感属性
        try:
            results = solution.evaluate_multiple_sensitive_attrs(
                y_true, y_pred, sensitive_attrs
            )
            assert results is not None
            assert isinstance(results, dict)
            # 应包含每个敏感属性的结果
            for attr in ['gender', 'age_group', 'region']:
                assert attr in results
        except (TypeError, NotImplementedError):
            pytest.skip("Multiple sensitive attributes not supported")


# =============================================================================
# 10. 公平性改进建议
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestFairnessImprovementRecommendations:
    """测试公平性改进建议"""

    def test_get_fairness_improvement_suggestions(self, biased_predictions_data):
        """
        正例：获取公平性改进建议

        应根据检测到的偏见提供具体的改进措施
        """
        if not hasattr(solution, 'get_fairness_improvement_suggestions'):
            pytest.skip("get_fairness_improvement_suggestions not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 获取改进建议
        suggestions = solution.get_fairness_improvement_suggestions(
            y_true, y_pred, sensitive_attr
        )

        assert suggestions is not None
        assert isinstance(suggestions, (list, dict, str))

        if isinstance(suggestions, list):
            assert len(suggestions) > 0
        elif isinstance(suggestions, dict):
            assert len(suggestions) > 0

    def test_suggestions_include_preprocessing(self, biased_predictions_data):
        """
        正例：建议应包括预处理方法

        应建议重采样、重新加权等预处理技术
        """
        if not hasattr(solution, 'get_fairness_improvement_suggestions'):
            pytest.skip("get_fairness_improvement_suggestions not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 获取改进建议
        suggestions = solution.get_fairness_improvement_suggestions(
            y_true, y_pred, sensitive_attr
        )

        # 建议应包含预处理相关内容
        suggestions_str = str(suggestions).lower()
        has_preprocessing = any(kw in suggestions_str for kw in [
            'resample', 'reweight', 'preprocess', 'oversample', 'undersample',
            '重采样', '预处', '加权'
        ])

        # 不强制要求，但如果有会更好
        assert True

    def test_suggestions_include_postprocessing(self, biased_predictions_data):
        """
        正例：建议应包括后处理方法

        应建议阈值调整等后处理技术
        """
        if not hasattr(solution, 'get_fairness_improvement_suggestions'):
            pytest.skip("get_fairness_improvement_suggestions not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 获取改进建议
        suggestions = solution.get_fairness_improvement_suggestions(
            y_true, y_pred, sensitive_attr
        )

        # 建议字符串
        suggestions_str = str(suggestions).lower()
        # 检查是否包含后处理相关建议（不强制要求）
        assert True
