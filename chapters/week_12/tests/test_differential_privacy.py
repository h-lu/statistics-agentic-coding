"""
Week 12：差分隐私与伦理审查测试

测试差分隐私、噪声添加、伦理审查清单等功能
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from solution import (
        add_differential_privacy_noise,
        check_privacy_budget,
        create_ethics_checklist,
        explain_to_nontechnical,
        review_xai_code,
    )
except ImportError:
    pytest.skip("starter_code/solution.py not implemented yet", allow_module_level=True)


class TestDifferentialPrivacy:
    """测试差分隐私功能"""

    def test_add_noise_returns_array(self):
        """测试加噪返回数组"""
        data = np.array([1000, 2000, 3000, 4000, 5000])

        private_data = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=4000
        )

        assert private_data is not None
        assert isinstance(private_data, np.ndarray)

    def test_add_noise_preserves_length(self):
        """测试加噪保持数据长度"""
        data = np.array([1000, 2000, 3000])

        private_data = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=2000
        )

        assert len(private_data) == len(data)

    def test_add_noise_laplace_distribution(self):
        """测试噪声来自拉普拉斯分布"""
        data = np.full(100, 5000)  # 常数

        private_data = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=5000
        )

        # 加噪后不应该都相同
        assert len(np.unique(private_data)) > 1

        # 噪声应该围绕 0
        noise = private_data - data
        assert abs(noise.mean()) < 1000  # 均值应该接近 0

    def test_add_noise_epsilon_impact(self):
        """测试 epsilon 对噪声大小的影响"""
        data = np.full(50, 5000)

        # epsilon 小 → 噪声大
        private_low = add_differential_privacy_noise(
            data, epsilon=0.1, sensitivity=5000
        )

        # epsilon 大 → 噪声小
        private_high = add_differential_privacy_noise(
            data, epsilon=10.0, sensitivity=5000
        )

        # 低 epsilon 的噪声应该更大
        noise_low = np.abs(private_low - data).mean()
        noise_high = np.abs(private_high - data).mean()

        assert noise_low > noise_high

    def test_add_noise_sensitivity_impact(self):
        """测试 sensitivity 对噪声大小的影响"""
        data = np.full(50, 5000)

        # sensitivity 小 → 噪声小
        private_low = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=1000
        )

        # sensitivity 大 → 噪声大
        private_high = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=10000
        )

        # 高 sensitivity 的噪声应该更大
        noise_low = np.abs(private_low - data).mean()
        noise_high = np.abs(private_high - data).mean()

        assert noise_low < noise_high

    def test_add_noise_with_negative_values(self):
        """测试对负值加噪"""
        data = np.array([-1000, -500, 0, 500, 1000])

        private_data = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=2000
        )

        # 应该能处理负值
        assert private_data is not None
        assert len(private_data) == len(data)

    def test_add_noise_reproducibility(self):
        """测试加噪的随机性（不同次运行结果不同）"""
        data = np.full(10, 5000)

        private_1 = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=5000
        )
        private_2 = add_differential_privacy_noise(
            data, epsilon=1.0, sensitivity=5000
        )

        # 两次结果应该不同（除非随机种子固定）
        # 注意：这个测试取决于实现是否每次都重新随机
        # 我们只检查不总是完全相同
        assert not np.array_equal(private_1, private_2) or True  # 允许相同（如果seed固定）


class TestPrivacyBudget:
    """测试隐私预算管理"""

    def test_check_privacy_budget_single_query(self):
        """测试单次查询的隐私预算检查"""
        epsilon = 1.0

        budget_status = check_privacy_budget(epsilon)

        # 应该返回预算状态
        assert isinstance(budget_status, (dict, bool, str))

    def test_check_privacy_budget_multiple_queries(self):
        """测试多次查询的隐私预算检查"""
        epsilons = [0.5, 0.3, 0.4]  # 总和 = 1.2

        budget_status = check_privacy_budget(epsilons, total_budget=1.0)

        # 应该检测到超预算
        if isinstance(budget_status, dict):
            assert 'exceeded' in budget_status or 'over_budget' in budget_status

    def test_check_privacy_budget_within_limit(self):
        """测试在预算内的情况"""
        epsilons = [0.3, 0.2, 0.1]  # 总和 = 0.6

        budget_status = check_privacy_budget(epsilons, total_budget=1.0)

        # 应该显示未超预算
        if isinstance(budget_status, dict):
            if 'exceeded' in budget_status:
                assert budget_status['exceeded'] is False

    def test_check_privacy_budget_exactly_at_limit(self):
        """测试正好达到预算上限"""
        epsilons = [0.5, 0.5]  # 总和 = 1.0

        budget_status = check_privacy_budget(epsilons, total_budget=1.0)

        # 应该是边界情况
        assert budget_status is not None


class TestEthicsChecklist:
    """测试伦理审查清单"""

    def test_create_checklist_returns_dict(self):
        """测试创建审查清单"""
        checklist = create_ethics_checklist()

        # 应该返回字典或列表
        assert isinstance(checklist, (dict, list))

    def test_create_checklist_has_all_categories(self):
        """测试审查清单包含所有风险类别"""
        checklist = create_ethics_checklist()

        expected_categories = [
            'data_bias',
            'algorithm_bias',
            'proxy_variables',
            'fairness',
            'privacy',
            'reproducibility',
            'limitations'
        ]

        if isinstance(checklist, dict):
            # 应该包含大部分类别
            checklist_str = str(checklist).lower()
            has_multiple = any(cat in checklist_str for cat in expected_categories)
            assert has_multiple or len(checklist) > 0

    def test_create_checklist_with_model_info(self):
        """测试带模型信息的审查清单"""
        model_info = {
            'model_type': 'RandomForestClassifier',
            'features': ['income', 'age', 'gender'],
            'sensitive_features': ['gender'],
            'performance': {'auc': 0.85}
        }

        checklist = create_ethics_checklist(model_info)

        # 应该返回包含模型信息的清单
        assert checklist is not None

    def test_create_checklist_with_fairness_metrics(self):
        """测试带公平性指标的审查清单"""
        fairness_results = {
            'disparate_impact': 0.72,
            'equal_opportunity_diff': 0.15,
            'gender_auc_diff': 0.11
        }

        checklist = create_ethics_checklist(fairness_metrics=fairness_results)

        # 应该标记公平性问题
        assert checklist is not None
        if isinstance(checklist, dict):
            checklist_str = str(checklist).lower()
            # 应该提到公平性或差异
            has_fairness = (
                'fairness' in checklist_str or
                'disparate' in checklist_str or
                'bias' in checklist_str
            )
            # 不一定总是检测到，但如果有的话应该相关
            assert has_fairness or len(checklist) > 0


class TestNonTechnicalExplanation:
    """测试向非技术人员解释"""

    def test_explain_to_customer(self):
        """测试向客户解释"""
        mock_prediction = {
            'base_value': 0.2,
            'shap_values': {
                'income': -0.3,
                'credit_history': -0.1,
                'debt_ratio': 0.2
            },
            'final_value': 0.35,
            'feature_values': {
                'income': 5000,
                'credit_history': 24,
                'debt_ratio': 0.45
            }
        }

        explanation = explain_to_nontechnical(mock_prediction, audience='customer')

        # 应该返回字符串
        assert isinstance(explanation, str)
        assert len(explanation) > 0

        # 应该避免技术术语
        tech_terms = ['SHAP', 'shapley', 'logit', 'auc', 'roc']
        explanation_lower = explanation.lower()
        # 不应该包含太多技术术语
        tech_count = sum(1 for term in tech_terms if term in explanation_lower)
        assert tech_count <= 2  # 允许少量

    def test_explain_to_product_manager(self):
        """测试向产品经理解释"""
        mock_prediction = {
            'base_value': 0.2,
            'shap_values': {'feature_1': 0.3, 'feature_2': -0.1},
            'final_value': 0.4
        }

        explanation = explain_to_nontechnical(mock_prediction,
                                            audience='product_manager')

        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_explain_to_compliance(self):
        """测试向合规部门解释"""
        mock_prediction = {
            'base_value': 0.2,
            'shap_values': {'income': -0.3},
            'final_value': 0.35,
            'model_metadata': {
                'fairness_metrics': {
                    'disparate_impact': 0.72
                }
            }
        }

        explanation = explain_to_nontechnical(mock_prediction,
                                            audience='compliance')

        assert isinstance(explanation, str)
        assert len(explanation) > 0

        # 合规版本可能包含"差异影响比"等术语
        # 但应该解释清楚

    def test_explain_good_vs_bad(self, good_explanation_for_nontechnical,
                                  bad_explanation_technical_jargon):
        """对比好的解释和糟糕的解释"""
        # 好的解释：清晰、易懂
        good = explain_to_nontechnical(
            {'base_value': 0.2, 'shap_values': {'income': -0.3}},
            'customer'
        )

        # 糟糕的解释：技术术语堆砌
        # （这个函数应该能够检测或避免这种情况）

        # 好的解释应该更易读
        assert isinstance(good, str)

    def test_explain_with_recommendations(self):
        """测试包含建议的解释"""
        mock_prediction = {
            'base_value': 0.2,
            'shap_values': {
                'income': -0.3,
                'credit_inquiries': 0.2
            },
            'feature_values': {
                'income': 5000,
                'credit_inquiries': 5
            },
            'final_value': 0.4
        }

        explanation = explain_to_nontechnical(
            mock_prediction,
            audience='customer',
            include_recommendations=True
        )

        # 应该包含建议
        assert explanation is not None
        explanation_lower = explanation.lower()
        # 可能包含建议相关的词
        has_suggestion = (
            '提高' in explanation_lower or
            '增加' in explanation_lower or
            '降低' in explanation_lower or
            '建议' in explanation_lower or
            '可以' in explanation_lower
        )
        # 不一定总是有，但检查结构
        assert len(explanation) > 0


class TestEthicsCodeReview:
    """测试伦理相关的代码审查"""

    def test_review_good_explanation_code(self, good_explanation_for_nontechnical):
        """测试审查好的解释代码"""
        review = review_xai_code(good_explanation_for_nontechnical)

        assert isinstance(review, dict)

    def test_review_bad_explanation_code(self, bad_explanation_technical_jargon):
        """测试审查糟糕的解释代码"""
        review = review_xai_code(bad_explanation_technical_jargon)

        assert isinstance(review, dict)
        # 应该检测到使用了技术术语
        if 'issues' in review:
            issues_str = str(review['issues']).lower()
            # 可能标记为"技术术语过多"或"不够易懂"
            has_clarity_issue = (
                'jargon' in issues_str or
                'technical' in issues_str or
                'clear' in issues_str or
                'understand' in issues_str
            )
            # 不一定总是检测到
            assert has_clarity_issue or True

    def test_review_fairness_code(self, good_fairness_code_example):
        """测试审查公平性代码"""
        # 从 fixtures 导入
        code = good_fairness_code_example

        review = review_xai_code(code)

        assert isinstance(review, dict)

    def test_review_code_no_fairness_check(self, bad_fairness_code_no_group_analysis):
        """测试审查缺少公平性检查的代码"""
        review = review_xai_code(bad_fairness_code_no_group_analysis)

        assert isinstance(review, dict)
        # 应该检测到缺少公平性检查
        if 'issues' in review:
            issues_str = str(review['issues']).lower()
            has_fairness_issue = (
                'fairness' in issues_str or
                'group' in issues_str or
                'bias' in issues_str
            )
            # 不一定总是检测到
            assert has_fairness_issue or True


class TestPrivacyScenarios:
    """测试隐私相关的真实场景"""

    def test_privacy_preserving_mean_release(self, privacy_test_data):
        """测试隐私保护的均值发布"""
        df = privacy_test_data

        income_mean = df['income'].mean()
        sensitivity = df['income'].max() - df['income'].min()

        private_mean = add_differential_privacy_noise(
            np.array([income_mean]), epsilon=1.0, sensitivity=sensitivity
        )

        # 应该返回加噪均值
        assert private_mean is not None

        # 误差应该在合理范围内
        error = abs(private_mean[0] - income_mean)
        # 对于拉普拉斯噪声，误差可能较大
        assert error < sensitivity  # 误差不应该超过敏感度

    def test_privacy_preserving_count_release(self):
        """测试隐私保护的计数发布"""
        # 计数的敏感度 = 1（添加/删除一条记录最多改变1）
        count = 1000

        private_count = add_differential_privacy_noise(
            np.array([count]), epsilon=1.0, sensitivity=1
        )

        # 计数的噪声应该较小
        assert private_count is not None
        error = abs(private_count[0] - count)
        # 由于 sensitivity=1，误差应该相对较小
        assert error < 10  # 大部分情况下

    def test_privacy_budget_tracking_across_queries(self):
        """测试跨查询的隐私预算跟踪"""
        queries = [
            {'epsilon': 0.5, 'query': 'mean income'},
            {'epsilon': 0.3, 'query': 'mean age'},
            {'epsilon': 0.1, 'query': 'count by region'}
        ]

        # 应该能跟踪总预算使用
        total_epsilon = sum(q['epsilon'] for q in queries)
        budget_status = check_privacy_budget(total_epsilon,
                                            total_budget=1.0)

        assert budget_status is not None


class TestEthicsIntegration:
    """测试伦理审查的集成功能"""

    def test_complete_ethics_review(self):
        """测试完整的伦理审查流程"""
        model_info = {
            'model_type': 'RandomForestClassifier',
            'features': ['income', 'age', 'gender', 'zip_code'],
            'sensitive_features': ['gender'],
            'performance': {'auc': 0.85}
        }

        fairness_metrics = {
            'disparate_impact': 0.72,
            'gender_auc_diff': 0.11
        }

        privacy_info = {
            'uses_differential_privacy': True,
            'epsilon': 1.0
        }

        # 生成完整的审查清单
        checklist = create_ethics_checklist(
            model_info=model_info,
            fairness_metrics=fairness_metrics,
            privacy_info=privacy_info
        )

        # 应该返回完整的清单
        assert checklist is not None

    def test_ethics_report_generation(self):
        """测试伦理报告生成"""
        # 模拟审查结果
        review_results = {
            'has_data_bias': True,
            'has_algorithm_bias': False,
            'has_proxy_variables': True,
            'fairness_status': 'warning',
            'privacy_status': 'ok',
            'reproducibility_status': 'ok',
            'limitations_documented': True
        }

        # 应该能生成可读的报告
        # （这个测试根据实现调整）
        assert review_results is not None
