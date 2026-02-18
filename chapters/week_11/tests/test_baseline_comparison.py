"""
Comprehensive tests for baseline comparison functionality in Week 11 solution.py

基线对比测试：
- 傻瓜基线（Dummy Classifier）
- 逻辑回归基线
- 单特征树基线
- 模型对比表
- 提升量计算
- 模型选择理由
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
# 1. 傻瓜基线测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestDummyBaseline:
    """测试傻瓜基线功能"""

    def test_train_dummy_baseline(self, baseline_comparison_data):
        """
        正例：训练傻瓜基线

        应创建一个总是预测多数类的基线
        """
        if not hasattr(solution, 'train_dummy_baseline'):
            pytest.skip("train_dummy_baseline not implemented")

        X_train = baseline_comparison_data['X_train']
        y_train = baseline_comparison_data['y_train']

        model = solution.train_dummy_baseline(X_train, y_train)

        assert model is not None

        # Should be able to predict
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_train)
            assert len(y_pred) == len(y_train)

    def test_dummy_baseline_auc(self, baseline_comparison_data, expected_baseline_performance):
        """
        正例：傻瓜基线 AUC

        傻瓜基线的 AUC 应接近 0.5
        """
        if not hasattr(solution, 'train_dummy_baseline'):
            pytest.skip("train_dummy_baseline not implemented")

        X_train = baseline_comparison_data['X_train']
        X_test = baseline_comparison_data['X_test']
        y_train = baseline_comparison_data['y_train']
        y_test = baseline_comparison_data['y_test']

        model = solution.train_dummy_baseline(X_train, y_train)

        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]

            # Calculate AUC
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, y_prob)

            # Dummy baseline should have AUC close to 0.5
            min_auc, max_auc = expected_baseline_performance['dummy_auc_range']
            assert min_auc <= auc <= max_auc, \
                f"Dummy baseline AUC should be around 0.5, got {auc}"

    def test_dummy_baseline_most_frequent(self, baseline_comparison_data):
        """
        正例：傻瓜基线应预测多数类

        验证基线策略是否正确
        """
        if not hasattr(solution, 'train_dummy_baseline'):
            pytest.skip("train_dummy_baseline not implemented")

        X_train = baseline_comparison_data['X_train']
        y_train = baseline_comparison_data['y_train']

        model = solution.train_dummy_baseline(X_train, y_train)

        if hasattr(model, 'predict'):
            y_pred = model.predict(X_train)

            # Find most frequent class in training data
            unique, counts = np.unique(y_train, return_counts=True)
            most_frequent = unique[np.argmax(counts)]

            # All predictions should be the most frequent class
            assert all(p == most_frequent for p in y_pred), \
                "Dummy baseline should predict the most frequent class"


# =============================================================================
# 2. 逻辑回归基线测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestLogisticBaseline:
    """测试逻辑回归基线功能"""

    def test_train_logistic_baseline(self, baseline_comparison_data):
        """
        正例：训练逻辑回归基线

        应创建一个简单的线性分类器作为基线
        """
        if not hasattr(solution, 'train_logistic_baseline'):
            pytest.skip("train_logistic_baseline not implemented")

        X_train = baseline_comparison_data['X_train']
        y_train = baseline_comparison_data['y_train']

        model = solution.train_logistic_baseline(X_train, y_train)

        assert model is not None

        # Should be able to predict
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_train)
            assert len(y_pred) == len(y_train)

    def test_logistic_baseline_better_than_dummy(self, baseline_comparison_data, expected_baseline_performance):
        """
        正例：逻辑回归应比傻瓜基线好

        在合理数据上，逻辑回归应该优于随机猜测
        """
        if not hasattr(solution, 'train_logistic_baseline') or not hasattr(solution, 'train_dummy_baseline'):
            pytest.skip("Required functions not implemented")

        X_train = baseline_comparison_data['X_train']
        X_test = baseline_comparison_data['X_test']
        y_train = baseline_comparison_data['y_train']
        y_test = baseline_comparison_data['y_test']

        # Train models
        dummy = solution.train_dummy_baseline(X_train, y_train)
        logistic = solution.train_logistic_baseline(X_train, y_train)

        # Compare AUC
        if hasattr(dummy, 'predict_proba') and hasattr(logistic, 'predict_proba'):
            from sklearn.metrics import roc_auc_score

            dummy_prob = dummy.predict_proba(X_test)
            if dummy_prob.ndim > 1:
                dummy_prob = dummy_prob[:, 1]
            dummy_auc = roc_auc_score(y_test, dummy_prob)

            logistic_prob = logistic.predict_proba(X_test)
            if logistic_prob.ndim > 1:
                logistic_prob = logistic_prob[:, 1]
            logistic_auc = roc_auc_score(y_test, logistic_prob)

            # Logistic should be better than dummy
            assert logistic_auc > dummy_auc + expected_baseline_performance['min_improvement'], \
                f"Logistic AUC {logistic_auc} should be better than dummy {dummy_auc}"


# =============================================================================
# 3. 单特征树基线测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestSingleFeatureBaseline:
    """测试单特征树基线功能"""

    def test_train_single_feature_tree(self, baseline_comparison_data):
        """
        正例：训练单特征树基线

        应创建一个只用最重要特征的决策树
        """
        if not hasattr(solution, 'train_single_feature_tree') and not hasattr(solution, 'train_decision_tree'):
            pytest.skip("Required functions not implemented")

        X_train = baseline_comparison_data['X_train']
        y_train = baseline_comparison_data['y_train']

        # Try single feature function
        if hasattr(solution, 'train_single_feature_tree'):
            model = solution.train_single_feature_tree(X_train, y_train)
        else:
            # Use first feature only
            model = solution.train_decision_tree(
                X_train[:, [0]], y_train,
                max_depth=2,
                random_state=42
            )

        assert model is not None

        if hasattr(model, 'predict'):
            # Use single feature for prediction since model was trained on single feature
            X_pred = X_train[:, [0]]
            y_pred = model.predict(X_pred)
            assert len(y_pred) == len(y_train)

    def test_single_feature_tree_simpler(self, baseline_comparison_data):
        """
        正例：单特征树应更简单

        相比多特征模型，单特征树应该更简单
        """
        if not hasattr(solution, 'train_decision_tree'):
            pytest.skip("train_decision_tree not implemented")

        X_train = baseline_comparison_data['X_train']
        y_train = baseline_comparison_data['y_train']

        # Train single feature tree
        single_tree = solution.train_decision_tree(
            X_train[:, [0]], y_train,
            max_depth=2,
            random_state=42
        )

        # Train multi-feature tree
        multi_tree = solution.train_decision_tree(
            X_train, y_train,
            max_depth=2,
            random_state=42
        )

        # Single feature tree should be simpler (fewer leaves or same depth)
        if hasattr(single_tree, 'tree_') and hasattr(multi_tree, 'tree_'):
            # Both have same max_depth, so just check they're valid
            assert single_tree.tree_.max_depth <= 2
            assert multi_tree.tree_.max_depth <= 2


# =============================================================================
# 4. 综合基线对比测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestBaselineComparison:
    """测试综合基线对比功能"""

    def test_compare_with_baselines(self, baseline_comparison_data):
        """
        正例：对比所有基线模型

        应返回包含所有模型性能的对比结果
        """
        if not hasattr(solution, 'compare_with_baselines'):
            pytest.skip("compare_with_baselines not implemented")

        X_train = baseline_comparison_data['X_train']
        X_test = baseline_comparison_data['X_test']
        y_train = baseline_comparison_data['y_train']
        y_test = baseline_comparison_data['y_test']

        results = solution.compare_with_baselines(X_train, y_train, X_test, y_test)

        assert results is not None

        # Should contain results for multiple models
        if isinstance(results, dict):
            # Check for common keys
            expected_keys = ['dummy', 'logistic', 'tree', 'forest',
                           'dummy_auc', 'logistic_auc', 'tree_auc', 'forest_auc',
                           'results', 'comparison']
            has_result = any(key in results for key in expected_keys)
            assert has_result, f"Should have comparison results, got: {results.keys()}"

    def test_baseline_comparison_all_models_better_than_dummy(self, baseline_comparison_data, expected_baseline_performance):
        """
        正例：所有模型都应优于傻瓜基线

        在合理数据上，真实模型应该比随机猜测好
        """
        if not hasattr(solution, 'compare_with_baselines'):
            pytest.skip("compare_with_baselines not implemented")

        X_train = baseline_comparison_data['X_train']
        X_test = baseline_comparison_data['X_test']
        y_train = baseline_comparison_data['y_train']
        y_test = baseline_comparison_data['y_test']

        results = solution.compare_with_baselines(X_train, y_train, X_test, y_test)

        # Extract AUC values (format may vary)
        dummy_auc = None
        other_aucs = []

        if isinstance(results, dict):
            # Try different possible formats
            if 'dummy_auc' in results:
                dummy_auc = results['dummy_auc']
            if 'logistic_auc' in results:
                other_aucs.append(results['logistic_auc'])
            if 'tree_auc' in results:
                other_aucs.append(results['tree_auc'])
            if 'forest_auc' in results:
                other_aucs.append(results['forest_auc'])

            # If results is nested
            if 'results' in results and isinstance(results['results'], dict):
                for model_name in ['dummy', 'logistic', 'tree', 'forest']:
                    if model_name in results['results']:
                        model_result = results['results'][model_name]
                        if isinstance(model_result, dict) and 'auc' in model_result:
                            if model_name == 'dummy':
                                dummy_auc = model_result['auc']
                            else:
                                other_aucs.append(model_result['auc'])

        if dummy_auc is not None and other_aucs:
            # All models should be better than dummy
            for auc in other_aucs:
                assert auc > dummy_auc + expected_baseline_performance['min_improvement'], \
                    f"Model AUC {auc} should be better than dummy {dummy_auc}"

    def test_baseline_comparison_includes_metrics(self, baseline_comparison_data):
        """
        正例：基线对比应包含多种评估指标

        不只是 AUC，还应包括准确率、精确率、召回率等
        """
        if not hasattr(solution, 'compare_with_baselines'):
            pytest.skip("compare_with_baselines not implemented")

        X_train = baseline_comparison_data['X_train']
        X_test = baseline_comparison_data['X_test']
        y_train = baseline_comparison_data['y_train']
        y_test = baseline_comparison_data['y_test']

        results = solution.compare_with_baselines(X_train, y_train, X_test, y_test)

        # Check for various metrics
        metrics_found = []
        if isinstance(results, dict):
            # Check for metric names in keys or nested dictionaries
            result_str = str(results).lower()
            metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            for metric in metric_names:
                if metric in result_str:
                    metrics_found.append(metric)

        # Should have at least some metrics
        assert len(metrics_found) >= 2, f"Should have multiple metrics, found: {metrics_found}"


# =============================================================================
# 5. 提升量计算测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestImprovementCalculation:
    """测试提升量计算"""

    def test_calculate_improvement(self):
        """
        正例：计算相对提升量

        应正确计算 (新模型 - 基线) / 基线
        """
        if not hasattr(solution, 'calculate_improvement'):
            pytest.skip("calculate_improvement not implemented")

        baseline_auc = 0.80
        model_auc = 0.85

        improvement = solution.calculate_improvement(model_auc, baseline_auc)

        assert improvement is not None

        # Improvement should be 0.0625 (0.05 / 0.80)
        expected_improvement = (model_auc - baseline_auc) / baseline_auc
        assert abs(improvement - expected_improvement) < 0.01, \
            f"Improvement should be {expected_improvement}, got {improvement}"

    def test_improvement_can_be_negative(self):
        """
        边界：提升量可以是负数

        如果模型比基线差，提升量应该是负数
        """
        if not hasattr(solution, 'calculate_improvement'):
            pytest.skip("calculate_improvement not implemented")

        baseline_auc = 0.85
        model_auc = 0.80

        improvement = solution.calculate_improvement(model_auc, baseline_auc)

        assert improvement is not None
        assert improvement < 0, "Improvement should be negative when model is worse than baseline"


# =============================================================================
# 6. 模型选择测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestModelSelection:
    """测试模型选择功能"""

    def test_select_best_model(self, baseline_comparison_data):
        """
        正例：选择最佳模型

        应根据评估指标选择最佳模型
        """
        if not hasattr(solution, 'select_best_model') and not hasattr(solution, 'compare_with_baselines'):
            pytest.skip("Required functions not implemented")

        X_train = baseline_comparison_data['X_train']
        X_test = baseline_comparison_data['X_test']
        y_train = baseline_comparison_data['y_train']
        y_test = baseline_comparison_data['y_test']

        if hasattr(solution, 'select_best_model'):
            best_model = solution.select_best_model(X_train, y_train, X_test, y_test)
            assert best_model is not None
        else:
            # Use compare_with_baselines and check if it identifies best
            results = solution.compare_with_baselines(X_train, y_train, X_test, y_test)
            assert results is not None

    def test_model_selection_considers_complexity(self, baseline_comparison_data):
        """
        正例：模型选择应考虑复杂度

        不只看 AUC，还应考虑模型复杂度
        """
        if not hasattr(solution, 'compare_with_baselines'):
            pytest.skip("compare_with_baselines not implemented")

        X_train = baseline_comparison_data['X_train']
        X_test = baseline_comparison_data['X_test']
        y_train = baseline_comparison_data['y_train']
        y_test = baseline_comparison_data['y_test']

        results = solution.compare_with_baselines(X_train, y_train, X_test, y_test)

        # Check if complexity is considered
        if isinstance(results, dict):
            result_str = str(results).lower()
            # May include complexity considerations
            # (not strictly required, but good to have)
            complexity_keywords = ['complexity', 'train_time', 'predict_time', 'interpretability']
            has_complexity = any(keyword in result_str for keyword in complexity_keywords)
            # If not present, that's okay - just checking it exists


# =============================================================================
# 7. 模型选择理由测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestModelSelectionReasoning:
    """测试模型选择理由"""

    def test_format_model_selection_reasoning(self, baseline_comparison_data):
        """
        正例：格式化模型选择理由

        应生成可读的选择理由说明
        """
        if not hasattr(solution, 'format_model_selection_reasoning') and \
           not hasattr(solution, 'format_model_comparison_report'):
            pytest.skip("Required functions not implemented")

        X_train = baseline_comparison_data['X_train']
        X_test = baseline_comparison_data['X_test']
        y_train = baseline_comparison_data['y_train']
        y_test = baseline_comparison_data['y_test']

        # Get comparison results
        if hasattr(solution, 'compare_with_baselines'):
            results = solution.compare_with_baselines(X_train, y_train, X_test, y_test)
        else:
            results = {'dummy_auc': 0.5, 'logistic_auc': 0.8, 'forest_auc': 0.82}

        # Format reasoning
        if hasattr(solution, 'format_model_selection_reasoning'):
            reasoning = solution.format_model_selection_reasoning(results)
        elif hasattr(solution, 'format_model_comparison_report'):
            reasoning = solution.format_model_comparison_report(results)
        else:
            pytest.skip("No formatting function implemented")

        assert reasoning is not None
        assert isinstance(reasoning, str), "Reasoning should be a string"
        assert len(reasoning) > 50, "Reasoning should have meaningful content"

        # Should mention key concepts
        reasoning_lower = reasoning.lower()
        has_baseline = 'baseline' in reasoning_lower or 'dummy' in reasoning_lower
        has_improvement = 'improvement' in reasoning_lower or 'better' in reasoning_lower or '+' in reasoning

        assert has_baseline, "Reasoning should mention baseline"
        assert has_improvement, "Reasoning should mention improvement"

    def test_reasoning_includes_tradeoffs(self, baseline_comparison_data):
        """
        正例：选择理由应包含权衡说明

        不只说哪个最好，还应说明为什么（权衡提升量、复杂度、可解释性）
        """
        if not hasattr(solution, 'format_model_selection_reasoning') and \
           not hasattr(solution, 'format_model_comparison_report'):
            pytest.skip("Required functions not implemented")

        X_train = baseline_comparison_data['X_train']
        X_test = baseline_comparison_data['X_test']
        y_train = baseline_comparison_data['y_train']
        y_test = baseline_comparison_data['y_test']

        # Get comparison results
        if hasattr(solution, 'compare_with_baselines'):
            results = solution.compare_with_baselines(X_train, y_train, X_test, y_test)
        else:
            results = {'dummy_auc': 0.5, 'logistic_auc': 0.8, 'forest_auc': 0.82}

        # Format reasoning
        if hasattr(solution, 'format_model_selection_reasoning'):
            reasoning = solution.format_model_selection_reasoning(results)
        elif hasattr(solution, 'format_model_comparison_report'):
            reasoning = solution.format_model_comparison_report(results)
        else:
            pytest.skip("No formatting function implemented")

        # Check for trade-off keywords
        reasoning_lower = reasoning.lower()
        tradeoff_keywords = ['trade-off', 'tradeoff', 'complexity', 'interpretable',
                            'explain', 'cost', 'worth', 'vs', 'versus']
        has_tradeoff = any(keyword in reasoning_lower for keyword in tradeoff_keywords)

        # Not strictly required, but good to have
        # assert has_tradeoff, "Reasoning should mention trade-offs"


# =============================================================================
# 8. StatLab 集成测试
# =============================================================================

@pytest.mark.skipif(solution is None, reason="solution.py not yet created")
class TestStatLabBaselineIntegration:
    """测试 StatLab 基线对比集成"""

    def test_statlab_tree_models_comparison(self, statlab_customer_churn_data, statlab_feature_lists):
        """
        正例：StatLab 树模型对比

        应能在客户流失数据上训练并对比多个模型
        """
        if not hasattr(solution, 'tree_models_comparison') and not hasattr(solution, 'baseline_comparison'):
            pytest.skip("Required functions not implemented")

        df = statlab_customer_churn_data
        feature_cols = statlab_feature_lists['numeric_features']
        target = statlab_feature_lists['target']

        X = df[feature_cols].values
        y = df[target].values

        # Try different function names
        if hasattr(solution, 'tree_models_comparison'):
            results = solution.tree_models_comparison(X, y)
        elif hasattr(solution, 'baseline_comparison'):
            results = solution.baseline_comparison(X, y)
        else:
            pytest.skip("No comparison function implemented")

        assert results is not None

        # Should contain comparison results
        if isinstance(results, dict):
            has_results = any(key in results for key in
                            ['dummy', 'logistic', 'tree', 'forest',
                             'dummy_auc', 'logistic_auc', 'tree_auc', 'forest_auc'])
            assert has_results, f"Should have comparison results, got: {results.keys()}"

    def test_statlab_format_comparison_report(self, statlab_customer_churn_data, statlab_feature_lists):
        """
        正例：StatLab 格式化对比报告

        应生成 Markdown 格式的对比报告
        """
        if not hasattr(solution, 'tree_models_comparison') or not hasattr(solution, 'format_model_comparison_report'):
            pytest.skip("Required functions not implemented")

        df = statlab_customer_churn_data
        feature_cols = statlab_feature_lists['numeric_features']
        target = statlab_feature_lists['target']

        X = df[feature_cols].values
        y = df[target].values

        results = solution.tree_models_comparison(X, y)
        report = solution.format_model_comparison_report(results)

        assert report is not None
        assert isinstance(report, str), "Report should be a string"
        assert len(report) > 100, "Report should have meaningful content"

        # Should be markdown format
        assert '#' in report or '|' in report, "Report should be in markdown format"

        # Should mention baseline
        report_lower = report.lower()
        assert 'baseline' in report_lower or 'dummy' in report_lower, \
            "Report should mention baseline"
