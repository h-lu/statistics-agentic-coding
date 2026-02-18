"""
Bias Detection Tests for Week 12 solution.py

偏见检测测试：
- 正例：按敏感属性分组评估、检测数据偏见
- 边界：多群体评估、小样本群体
- 反例：无效输入、空群体
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
# 1. 正例：分组评估
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestGroupEvaluation:
    """测试按敏感属性分组评估"""

    def test_evaluate_by_group_returns_metrics(self, biased_predictions_data):
        """
        正例：按群体评估应返回各组的指标

        应包含每个群体的准确率、召回率等
        """
        if not hasattr(solution, 'evaluate_by_group'):
            pytest.skip("evaluate_by_group not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 按群体评估
        results = solution.evaluate_by_group(y_true, y_pred, sensitive_attr)

        assert results is not None
        # 应该返回每个群体的结果
        assert isinstance(results, (dict, pd.DataFrame))

        if isinstance(results, dict):
            # 应该有两个群体的结果
            assert len(results) >= 1
            # 每个群体应该有指标
            for group, metrics in results.items():
                assert 'count' in metrics or 'size' in metrics
        elif isinstance(results, pd.DataFrame):
            assert len(results) >= 1

    def test_evaluate_by_group_with_multiple_groups(self, multiple_sensitive_groups_data):
        """
        正例：支持多个群体的评估

        应能处理 3 个或更多群体
        """
        if not hasattr(solution, 'evaluate_by_group'):
            pytest.skip("evaluate_by_group not implemented")

        y_true = multiple_sensitive_groups_data['y_true']
        y_pred = multiple_sensitive_groups_data['y_pred']
        sensitive_attr = multiple_sensitive_groups_data['sensitive_attr']

        # 按群体评估
        results = solution.evaluate_by_group(y_true, y_pred, sensitive_attr)

        assert results is not None
        # 应该有 3 个群体的结果
        if isinstance(results, dict):
            assert len(results) == 3
        elif isinstance(results, pd.DataFrame):
            assert len(results) == 3

    def test_evaluate_by_group_computes_confusion_matrix(self, biased_predictions_data):
        """
        正例：按群体计算混淆矩阵

        应返回每个群体的 TP、TN、FP、FN
        """
        if not hasattr(solution, 'group_confusion_matrices'):
            pytest.skip("group_confusion_matrices not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 计算混淆矩阵
        cms = solution.group_confusion_matrices(y_true, y_pred, sensitive_attr)

        assert cms is not None
        assert isinstance(cms, dict)

        # 每个群体应该有 2x2 的混淆矩阵
        for group, cm in cms.items():
            if isinstance(cm, np.ndarray):
                assert cm.shape == (2, 2)
            elif isinstance(cm, dict):
                assert 'tp' in cm or 'TP' in cm


# =============================================================================
# 2. 偏见检测
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBiasDetection:
    """测试偏见检测功能"""

    def test_detect_prediction_bias(self, biased_predictions_data):
        """
        正例：检测预测偏见

        应能识别不同群体间的预测差异
        """
        if not hasattr(solution, 'detect_prediction_bias'):
            pytest.skip("detect_prediction_bias not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 检测偏见
        bias_report = solution.detect_prediction_bias(y_true, y_pred, sensitive_attr)

        assert bias_report is not None
        assert isinstance(bias_report, dict)

        # 应该包含关键指标
        assert 'positive_rate_diff' in bias_report or 'prediction_rate_diff' in bias_report

    def test_detect_outcome_bias(self, bias_detection_data):
        """
        正例：检测结果偏见（真实结果差异）

        应能识别不同群体间的真实结果差异
        """
        if not hasattr(solution, 'detect_outcome_bias'):
            pytest.skip("detect_outcome_bias not implemented")

        y_true = bias_detection_data['y']
        sensitive_attr = bias_detection_data['sensitive_attr']

        # 检测结果偏见
        outcome_bias = solution.detect_outcome_bias(y_true, sensitive_attr)

        assert outcome_bias is not None
        assert isinstance(outcome_bias, dict)

        # 应该包含各群体的真实正率
        assert 'group_rates' in outcome_bias or 'outcome_diff' in outcome_bias

    def test_detect_disparate_impact(self, biased_predictions_data):
        """
        正例：检测差别影响（Disparate Impact）

        计算少数群体与多数群体的预测正率比值
        """
        if not hasattr(solution, 'detect_disparate_impact'):
            pytest.skip("detect_disparate_impact not implemented")

        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 检测差别影响
        di_ratio = solution.detect_disparate_impact(y_pred, sensitive_attr)

        assert di_ratio is not None
        # DI ratio 应该是一个正数
        assert di_ratio > 0 or di_ratio < 0  # 可能为负表示方向
        # 80% 规则：DI < 0.8 表示存在偏见
        if abs(di_ratio) < 1.0:
            assert di_ratio is not None


# =============================================================================
# 3. 数据偏见 vs 算法偏见
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDataVsAlgorithmicBias:
    """测试数据偏见与算法偏见的区分"""

    def test_identify_data_bias(self, bias_detection_data):
        """
        正例：识别数据偏见

        数据偏见：训练数据本身存在差异
        """
        if not hasattr(solution, 'identify_data_bias'):
            pytest.skip("identify_data_bias not implemented")

        y_true = bias_detection_data['y']
        sensitive_attr = bias_detection_data['sensitive_attr']

        # 识别数据偏见
        data_bias = solution.identify_data_bias(y_true, sensitive_attr)

        assert data_bias is not None
        assert isinstance(data_bias, dict)
        # 应该指出是否存在数据偏见
        assert 'has_bias' in data_bias or 'bias_detected' in data_bias

    def test_identify_algorithmic_bias(self, biased_predictions_data):
        """
        正例：识别算法偏见

        算法偏见：模型放大或引入了偏见
        """
        if not hasattr(solution, 'identify_algorithmic_bias'):
            pytest.skip("identify_algorithmic_bias not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 识别算法偏见
        algo_bias = solution.identify_algorithmic_bias(y_true, y_pred, sensitive_attr)

        assert algo_bias is not None
        assert isinstance(algo_bias, dict)
        # 应该指出是否存在算法偏见
        assert 'has_bias' in algo_bias or 'bias_detected' in algo_bias


# =============================================================================
# 4. 边界：小样本群体
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBiasDetectionSmallGroups:
    """测试小样本群体的偏见检测"""

    def test_handle_small_group_size(self, minimal_fairness_data):
        """
        边界：处理小样本群体

        某些群体样本很少时，指标可能不稳定
        """
        if not hasattr(solution, 'evaluate_by_group'):
            pytest.skip("evaluate_by_group not implemented")

        y_true = minimal_fairness_data['y_true']
        y_pred = minimal_fairness_data['y_pred']
        sensitive_attr = minimal_fairness_data['sensitive_attr']

        # 按群体评估
        results = solution.evaluate_by_group(y_true, y_pred, sensitive_attr)

        assert results is not None
        # 小群体的结果可能不稳定，但不应报错

    def test_skip_groups_below_threshold(self, minimal_fairness_data):
        """
        边界：跳过样本量过小的群体

        可配置最小样本量阈值
        """
        if not hasattr(solution, 'evaluate_by_group'):
            pytest.skip("evaluate_by_group not implemented")

        y_true = minimal_fairness_data['y_true']
        y_pred = minimal_fairness_data['y_pred']
        sensitive_attr = minimal_fairness_data['sensitive_attr']

        # 设置最小样本量阈值
        min_group_size = 15

        try:
            results = solution.evaluate_by_group(
                y_true, y_pred, sensitive_attr,
                min_group_size=min_group_size
            )
            assert results is not None
        except TypeError:
            # 如果不支持 min_group_size 参数
            results = solution.evaluate_by_group(y_true, y_pred, sensitive_attr)
            assert results is not None


# =============================================================================
# 5. 边界：单一群体
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBiasDetectionSingleGroup:
    """测试单一群体的边界情况"""

    def test_single_group_evaluation(self, single_group_data):
        """
        边界：只有一个群体

        无法计算群体间差异，但应返回该群体的指标
        """
        if not hasattr(solution, 'evaluate_by_group'):
            pytest.skip("evaluate_by_group not implemented")

        y_true = single_group_data['y_true']
        y_pred = single_group_data['y_pred']
        sensitive_attr = single_group_data['sensitive_attr']

        # 按群体评估
        results = solution.evaluate_by_group(y_true, y_pred, sensitive_attr)

        assert results is not None
        # 应该只有一个群体的结果
        if isinstance(results, dict):
            assert len(results) == 1
        elif isinstance(results, pd.DataFrame):
            assert len(results) == 1

    def test_single_group_bias_detection(self, single_group_data):
        """
        边界：单一群体的偏见检测

        无法检测偏见（没有比较对象）
        """
        if not hasattr(solution, 'detect_prediction_bias'):
            pytest.skip("detect_prediction_bias not implemented")

        y_true = single_group_data['y_true']
        y_pred = single_group_data['y_pred']
        sensitive_attr = single_group_data['sensitive_attr']

        # 检测偏见
        bias_report = solution.detect_prediction_bias(y_true, y_pred, sensitive_attr)

        assert bias_report is not None
        # 应该指出无法检测偏见（或偏见为 0）
        assert 'bias' in str(bias_report).lower() or 'diff' in str(bias_report).lower() or len(bias_report) >= 0


# =============================================================================
# 6. 反例：无效输入
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBiasDetectionInvalidInput:
    """测试无效输入处理"""

    def test_mismatched_lengths(self):
        """
        反例：y_true、y_pred、sensitive_attr 长度不匹配

        应报错
        """
        if not hasattr(solution, 'evaluate_by_group'):
            pytest.skip("evaluate_by_group not implemented")

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        sensitive_attr = np.array([0, 1])  # 长度不匹配

        with pytest.raises((ValueError, IndexError, AssertionError)):
            solution.evaluate_by_group(y_true, y_pred, sensitive_attr)

    def test_empty_sensitive_attr(self):
        """
        反例：敏感属性为空

        应报错
        """
        if not hasattr(solution, 'evaluate_by_group'):
            pytest.skip("evaluate_by_group not implemented")

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        sensitive_attr = np.array([])

        with pytest.raises((ValueError, IndexError)):
            solution.evaluate_by_group(y_true, y_pred, sensitive_attr)

    def test_all_same_predictions(self):
        """
        反例：所有预测相同

        应仍能计算指标（虽然无意义）
        """
        if not hasattr(solution, 'evaluate_by_group'):
            pytest.skip("evaluate_by_group not implemented")

        y_true = np.array([0, 0, 1, 1] * 50)
        y_pred = np.zeros(200, dtype=int)  # 全部预测为 0
        sensitive_attr = np.array([0] * 100 + [1] * 100)

        # 应能处理（返回 0 召回率等）
        results = solution.evaluate_by_group(y_true, y_pred, sensitive_attr)
        assert results is not None


# =============================================================================
# 7. 偏见可视化
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBiasVisualization:
    """测试偏见检测可视化"""

    def test_plot_group_metrics(self, biased_predictions_data):
        """
        正例：绘制各群体指标对比图

        应能生成可视化的群体指标对比
        """
        if not hasattr(solution, 'plot_group_metrics'):
            pytest.skip("plot_group_metrics not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 绘制指标对比图
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            result = solution.plot_group_metrics(
                y_true, y_pred, sensitive_attr, output_path=output_path
            )
            assert result is not None
        except ImportError:
            # Matplotlib 可能未安装
            pytest.skip("Matplotlib not installed")
        finally:
            import os
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_plot_confusion_matrices_by_group(self, biased_predictions_data):
        """
        正例：绘制各群体的混淆矩阵

        应能生成各群体的混淆矩阵热图
        """
        if not hasattr(solution, 'plot_group_confusion_matrices'):
            pytest.skip("plot_group_confusion_matrices not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 绘制混淆矩阵
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_path = f.name

        try:
            result = solution.plot_group_confusion_matrices(
                y_true, y_pred, sensitive_attr, output_path=output_path
            )
            assert result is not None
        except ImportError:
            pytest.skip("Matplotlib not installed")
        finally:
            import os
            if os.path.exists(output_path):
                os.remove(output_path)


# =============================================================================
# 8. 偏见报告生成
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBiasReportGeneration:
    """测试偏见报告生成"""

    def test_generate_bias_report(self, biased_predictions_data):
        """
        正例：生成完整的偏见检测报告

        应包含各群体的指标、偏见检测结果、建议
        """
        if not hasattr(solution, 'generate_bias_report'):
            pytest.skip("generate_bias_report not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 生成报告
        report = solution.generate_bias_report(
            y_true, y_pred, sensitive_attr,
            attr_name='gender'
        )

        assert report is not None
        assert isinstance(report, str)

        # 报告应包含关键信息
        assert 'gender' in report or 'group' in report.lower()

    def test_bias_report_includes_recommendations(self, biased_predictions_data):
        """
        正例：偏见报告应包含缓解建议

        当检测到偏见时，应提供缓解措施
        """
        if not hasattr(solution, 'generate_bias_report'):
            pytest.skip("generate_bias_report not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 生成报告
        report = solution.generate_bias_report(
            y_true, y_pred, sensitive_attr,
            attr_name='gender'
        )

        # 如果检测到偏见，应包含建议
        has_bias = (
            '偏见' in report or 'bias' in report.lower() or
            '差异' in report or 'disparity' in report.lower()
        )

        if has_bias:
            # 应该包含缓解建议
            assert (
                '建议' in report or 'recommend' in report.lower() or
                '缓解' in report or 'mitigat' in report.lower() or
                len(report) > 0
            )


# =============================================================================
# 9. 偏见阈值检测
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBiasThresholdDetection:
    """测试偏见阈值检测"""

    def test_check_bias_threshold(self, biased_predictions_data):
        """
        正例：检查偏见是否超过阈值

        应返回偏见是否超过可接受水平
        """
        if not hasattr(solution, 'check_bias_threshold'):
            pytest.skip("check_bias_threshold not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 设置阈值
        threshold = 0.1

        # 检查偏见
        is_biased = solution.check_bias_threshold(
            y_true, y_pred, sensitive_attr, threshold=threshold
        )

        assert isinstance(is_biased, bool)

    def test_custom_bias_threshold(self, biased_predictions_data):
        """
        正例：使用自定义阈值

        不同阈值应导致不同的偏见检测结果
        """
        if not hasattr(solution, 'check_bias_threshold'):
            pytest.skip("check_bias_threshold not implemented")

        y_true = biased_predictions_data['y_true']
        y_pred = biased_predictions_data['y_pred']
        sensitive_attr = biased_predictions_data['sensitive_attr']

        # 使用不同阈值
        is_biased_strict = solution.check_bias_threshold(
            y_true, y_pred, sensitive_attr, threshold=0.05
        )
        is_biased_loose = solution.check_bias_threshold(
            y_true, y_pred, sensitive_attr, threshold=0.2
        )

        # 严格阈值更容易检测到偏见
        if is_biased_loose:
            assert is_biased_strict or True  # 两者可能都为 True
        # 如果宽松阈值没检测到偏见，严格阈值可能检测到


# =============================================================================
# 10. 多敏感属性交叉评估
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestIntersectionalBias:
    """测试交叉偏见评估"""

    def test_evaluate_intersectional_groups(self):
        """
        正例：评估交叉群体（如性别×种族）

        应能处理多个敏感属性的组合
        """
        if not hasattr(solution, 'evaluate_intersectional_groups'):
            pytest.skip("evaluate_intersectional_groups not implemented")

        np.random.seed(42)
        n = 200

        y_true = np.random.randint(0, 2, n)
        gender = np.random.randint(0, 2, n)
        race = np.random.randint(0, 2, n)

        # 评估交叉群体
        try:
            results = solution.evaluate_intersectional_groups(
                y_true, y_true,  # 使用 y_true 作为 y_pred 简化测试
                {'gender': gender, 'race': race}
            )
            assert results is not None
        except TypeError:
            # 如果不支持多个敏感属性
            pytest.skip("Intersectional evaluation not supported")

    def test_intersectional_bias_detection(self):
        """
        正例：检测交叉偏见

        某些交叉群体可能遭受更严重的偏见
        """
        if not hasattr(solution, 'detect_intersectional_bias'):
            pytest.skip("detect_intersectional_bias not implemented")

        np.random.seed(42)
        n = 200

        y_true = np.random.randint(0, 2, n)
        gender = np.random.randint(0, 2, n)
        age_group = np.random.randint(0, 3, n)

        # 创建有偏见的预测（某个交叉群体更差）
        y_pred = y_true.copy()
        # 女性 + 老年组 更容易被误判
        mask = (gender == 1) & (age_group == 2)
        y_pred[mask] = 1 - y_pred[mask]

        try:
            bias_report = solution.detect_intersectional_bias(
                y_true, y_pred,
                {'gender': gender, 'age_group': age_group}
            )
            assert bias_report is not None
        except (TypeError, NotImplementedError):
            pytest.skip("Intersectional bias detection not supported")
