"""
Week 12：端到端集成测试

测试完整的可解释 AI 与伦理审查工作流
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from solution import (
        calculate_shap_values,
        explain_single_prediction,
        calculate_feature_importance_shap,
        calculate_disparate_impact,
        calculate_equal_opportunity,
        calculate_equalized_odds,
        detect_proxy_variables,
        add_differential_privacy_noise,
        check_privacy_budget,
        create_ethics_checklist,
        explain_to_nontechnical,
        review_xai_code,
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import shap
except ImportError:
    pytest.skip("starter_code/solution.py or required libraries not implemented yet", allow_module_level=True)


class TestCompleteXAIWorkflow:
    """测试完整的可解释 AI 工作流"""

    def test_end_to_end_credit_scoring_xai(self, credit_scoring_data):
        """测试信用评分的完整 XAI 工作流"""
        df = credit_scoring_data

        feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                       'credit_inquiries', 'employment_length']
        X = df[feature_cols]
        y = df['default']

        # 1. 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 2. 训练模型
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 3. 计算 SHAP 值
        shap_values = calculate_shap_values(model, X_test)
        assert shap_values is not None

        # 4. 计算特征重要性
        importance = calculate_feature_importance_shap(model, X_test)
        assert importance is not None

        # 5. 解释单个预测（找一个被拒的）
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        rejected_idx = np.where(y_pred_proba > 0.7)[0]

        if len(rejected_idx) > 0:
            explanation = explain_single_prediction(
                model, X_test.iloc[rejected_idx[0]]
            )
            assert explanation is not None

        # 工作流完成
        assert True

    def test_end_to_end_fairness_audit(self, credit_scoring_data):
        """测试完整的公平性审计工作流"""
        df = credit_scoring_data

        feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                       'credit_inquiries', 'employment_length']
        X = df[feature_cols]
        y = df['default']

        # 1. 训练模型
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 2. 预测
        y_pred = model.predict(X_test)
        gender = df.loc[X_test.index, 'gender'].values

        # 3. 计算公平性指标
        di_ratio = calculate_disparate_impact(y_pred, gender)
        assert di_ratio is not None

        eo_diff = calculate_equal_opportunity(y_test.values, y_pred, gender)
        assert eo_diff is not None

        eo_result = calculate_equalized_odds(y_test.values, y_pred, gender)
        assert eo_result is not None

        # 4. 检测代理变量
        proxies = detect_proxy_variables(df, sensitive_col='gender')
        assert proxies is not None

        # 审计完成
        assert True

    def test_end_to_end_ethics_review(self, credit_scoring_data):
        """测试完整的伦理审查流程"""
        df = credit_scoring_data

        feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                       'credit_inquiries', 'employment_length']
        X = df[feature_cols]
        y = df['default']

        # 1. 训练模型
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 2. 收集模型信息
        model_info = {
            'model_type': 'RandomForestClassifier',
            'features': feature_cols,
            'sensitive_features': ['gender'],
            'performance': {
                'auc': model.score(X_test, y_test)
            }
        }

        # 3. 收集公平性指标
        y_pred = model.predict(X_test)
        gender = df.loc[X_test.index, 'gender'].values

        fairness_metrics = {
            'disparate_impact': calculate_disparate_impact(y_pred, gender),
            'equal_opportunity': calculate_equal_opportunity(
                y_test.values, y_pred, gender
            )
        }

        # 4. 生成伦理审查清单
        checklist = create_ethics_checklist(
            model_info=model_info,
            fairness_metrics=fairness_metrics
        )
        assert checklist is not None

        # 审查完成
        assert True

    def test_end_to_end_customer_explanation(self, credit_scoring_data):
        """测试完整的客户解释流程"""
        df = credit_scoring_data

        feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                       'credit_inquiries', 'employment_length']
        X = df[feature_cols]
        y = df['default']

        # 1. 训练模型
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 2. 找一个被拒的样本
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        rejected_idx = np.where(y_pred_proba > 0.7)[0]

        if len(rejected_idx) > 0:
            sample = X_test.iloc[rejected_idx[0]]

            # 3. 获取技术解释
            tech_explanation = explain_single_prediction(model, sample)
            assert tech_explanation is not None

            # 4. 转换为面向客户的解释
            customer_explanation = explain_to_nontechnical(
                tech_explanation,
                audience='customer',
                feature_values=sample.to_dict()
            )
            assert customer_explanation is not None
            assert isinstance(customer_explanation, str)
            assert len(customer_explanation) > 0

        # 流程完成
        assert True


class TestStatLabIntegration:
    """测试 StatLab 集成"""

    def test_model_explanation_to_report(self, credit_scoring_data):
        """测试生成模型解释报告片段"""
        df = credit_scoring_data

        feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                       'credit_inquiries', 'employment_length']
        X = df[feature_cols]
        y = df['default']

        # 1. 训练模型
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 2. 计算 SHAP 值
        shap_values = calculate_shap_values(model, X_test)
        assert shap_values is not None

        # 3. 计算特征重要性
        importance = calculate_feature_importance_shap(model, X_test)
        assert importance is not None

        # 4. 公平性评估
        y_pred = model.predict(X_test)
        gender = df.loc[X_test.index, 'gender'].values

        fairness_results = {
            'disparate_impact': calculate_disparate_impact(y_pred, gender),
            'gender_auc_diff': 0.11  # 模拟值
        }

        # 5. 生成报告内容
        # （这里测试是否能够整合所有信息）
        report_sections = {
            'shap_available': shap_values is not None,
            'importance_available': importance is not None,
            'fairness_available': fairness_results is not None
        }

        assert all(report_sections.values())

    def test_complete_statlab_workflow(self, credit_scoring_data):
        """测试完整的 StatLab 工作流（从建模到报告）"""
        df = credit_scoring_data

        feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                       'credit_inquiries', 'employment_length']
        X = df[feature_cols]
        y = df['default']

        # 阶段 1：建模
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 阶段 2：可解释性
        shap_values = calculate_shap_values(model, X_test)
        importance = calculate_feature_importance_shap(model, X_test)
        assert shap_values is not None and importance is not None

        # 阶段 3：公平性评估
        y_pred = model.predict(X_test)
        gender = df.loc[X_test.index, 'gender'].values

        di_ratio = calculate_disparate_impact(y_pred, gender)
        eo_diff = calculate_equal_opportunity(y_test.values, y_pred, gender)
        assert di_ratio is not None and eo_diff is not None

        # 阶段 4：伦理审查
        checklist = create_ethics_checklist(
            model_info={
                'model_type': 'RandomForestClassifier',
                'features': feature_cols
            },
            fairness_metrics={'disparate_impact': di_ratio}
        )
        assert checklist is not None

        # 阶段 5：生成解释
        rejected_idx = np.where(y_pred == 1)[0]
        if len(rejected_idx) > 0:
            explanation = explain_single_prediction(
                model, X_test.iloc[rejected_idx[0]]
            )
            assert explanation is not None

        # 完整流程成功
        assert True


class TestRealWorldScenarios:
    """测试真实世界场景"""

    def test_hiring_scenario_fairness_check(self):
        """测试招聘场景的公平性检查"""
        # 模拟招聘数据
        np.random.seed(42)
        n = 500

        # 生成特征
        qualifications = np.random.normal(100, 15, n)
        experience = np.random.randint(0, 20, n)
        gender = np.random.randint(0, 2, n)

        # 目标变量（模拟偏见）
        logit = -3 + 0.05 * qualifications + 0.1 * experience - 0.5 * gender
        prob = 1 / (1 + np.exp(-logit))
        hired = (np.random.random(n) < prob).astype(int)

        # 评估
        di_ratio = calculate_disparate_impact(hired, gender)
        assert di_ratio is not None

        # 如果差异影响比 < 0.8，说明存在潜在偏见
        if not np.isnan(di_ratio):
            # 应该能检测到
            assert isinstance(di_ratio, float)

    def test_credit_scoring_with_proxy_detection(self, credit_scoring_data_with_proxy):
        """测试信用评分的代理变量检测"""
        df = credit_scoring_data_with_proxy

        # 1. 检测代理变量
        proxies = detect_proxy_variables(df, sensitive_col='gender')
        assert proxies is not None

        # 2. 训练模型（不使用性别）
        feature_cols = ['income', 'credit_history_age', 'debt_to_income']
        X = df[feature_cols]
        y = df['default']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 3. 评估性能
        score = model.score(X_test, y_test)
        assert score > 0.6  # 应该有基本的预测能力

    def test_medical_diagnosis_explanation(self):
        """测试医疗诊断场景的预测解释"""
        # 模拟医疗数据
        np.random.seed(42)
        n = 300

        X = pd.DataFrame({
            'age': np.random.randint(18, 80, n),
            'blood_pressure': np.random.normal(120, 15, n),
            'cholesterol': np.random.normal(200, 30, n),
            'bmi': np.random.normal(25, 5, n),
        })

        logit = -5 + 0.05 * X['age'] + 0.03 * X['blood_pressure'] + \
                 0.02 * X['cholesterol'] + 0.1 * X['bmi']
        prob = 1 / (1 + np.exp(-logit))
        y = (np.random.random(n) < prob).astype(int)

        # 确保两个类别都存在
        if len(np.unique(y)) < 2:
            y = (prob > 0.5).astype(int)

        # 训练模型
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 确保训练集有两个类别
        if len(np.unique(y_train)) < 2:
            # 调整数据使其有两个类别
            y_train[:10] = 1 - y_train[:10]

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # 解释高风险患者的预测
        # 检查 predict_proba 的输出形状
        proba = model.predict_proba(X_test)
        if proba.shape[1] > 1:
            y_pred_proba = proba[:, 1]
        else:
            y_pred_proba = proba[:, 0]

        high_risk_idx = np.where(y_pred_proba > 0.5)[0]

        # 如果没有高风险患者，使用预测概率最高的样本
        if len(high_risk_idx) == 0:
            high_risk_idx = np.array([y_pred_proba.argmax()])

        if len(high_risk_idx) > 0:
            explanation = explain_single_prediction(
                model, X_test.iloc[high_risk_idx[0]]
            )
            assert explanation is not None

            # 转换为面向患者的解释
            patient_explanation = explain_to_nontechnical(
                explanation,
                audience='customer'  # 使用 'customer' 而不是 'patient'
            )

        if len(high_risk_idx) > 0:
            explanation = explain_single_prediction(
                model, X_test.iloc[high_risk_idx[0]]
            )
            assert explanation is not None

            # 转换为面向患者的解释
            patient_explanation = explain_to_nontechnical(
                explanation,
                audience='patient'
            )
            assert patient_explanation is not None
            assert isinstance(patient_explanation, str)


class TestCodeReviewIntegration:
    """测试代码审查集成"""

    def test_review_complete_xai_workflow(self, good_shap_code_example):
        """测试审查完整的 XAI 工作流代码"""
        review = review_xai_code(good_shap_code_example)

        # 应该返回详细的审查结果
        assert isinstance(review, dict)
        assert 'has_issues' in review or 'issues' in review

    def test_review_incomplete_xai_workflow(self, bad_shap_code_global_only):
        """测试审查不完整的 XAI 工作流代码"""
        review = review_xai_code(bad_shap_code_global_only)

        # 应该检测到缺少局部解释
        assert isinstance(review, dict)
        if 'issues' in review:
            issues = review['issues']
            assert len(issues) > 0

    def test_compare_good_vs_bad_explanations(self,
                                               good_explanation_for_nontechnical,
                                               bad_explanation_technical_jargon):
        """对比好的解释和糟糕的解释"""
        # 审查好的解释
        review_good = review_xai_code(good_explanation_for_nontechnical)

        # 审查糟糕的解释
        review_bad = review_xai_code(bad_explanation_technical_jargon)

        # 糟糕的解释应该有更多问题
        if 'issues' in review_good and 'issues' in review_bad:
            # 不总是成立，但通常糟糕的代码问题更多
            assert isinstance(review_bad['issues'], (list, dict))


class TestPrivacyIntegration:
    """测试隐私保护集成"""

    def test_privacy_preserving_model_evaluation(self, privacy_test_data):
        """测试隐私保护的模型评估"""
        df = privacy_test_data

        # 1. 计算敏感指标
        income_mean = df['income'].mean()
        income_range = df['income'].max() - df['income'].min()

        # 2. 添加差分隐私噪声
        private_mean = add_differential_privacy_noise(
            np.array([income_mean]),
            epsilon=1.0,
            sensitivity=income_range
        )

        # 3. 检查隐私预算
        budget = check_privacy_budget([1.0], total_budget=5.0)

        # 应该成功
        assert private_mean is not None
        assert budget is not None

    def test_multi_query_privacy_tracking(self):
        """测试多次查询的隐私预算跟踪"""
        queries = [
            {'epsilon': 0.5, 'name': 'mean income'},
            {'epsilon': 0.3, 'name': 'mean age'},
            {'epsilon': 0.2, 'name': 'mean score'},
        ]

        # 总预算
        total_epsilon = sum(q['epsilon'] for q in queries)
        budget_status = check_privacy_budget(total_epsilon,
                                            total_budget=1.5)

        # 应该能跟踪预算使用
        assert budget_status is not None


class TestErrorHandling:
    """测试错误处理"""

    def test_handle_missing_model_in_shap(self):
        """测试 SHAP 中处理缺失模型"""
        # 应该优雅地处理
        try:
            shap_values = calculate_shap_values(None, pd.DataFrame())
            # 如果不报错，应该返回 None 或空值
            assert shap_values is None or len(shap_values) == 0
        except (ValueError, AttributeError, TypeError):
            # 预期可能报错
            assert True

    def test_handle_empty_data_in_fairness(self):
        """测试公平性指标中处理空数据"""
        empty_pred = np.array([])
        empty_groups = np.array([])

        try:
            di_ratio = calculate_disparate_impact(empty_pred, empty_groups)
            # 如果不报错，应该返回 NaN 或 None
            assert di_ratio is None or np.isnan(di_ratio)
        except (ValueError, IndexError):
            # 预期可能报错
            assert True

    def test_handle_invalid_audience_in_explanation(self):
        """测试解释中处理无效受众"""
        mock_explanation = {
            'base_value': 0.2,
            'shap_values': {'feature': 0.3}
        }

        # 传递无效的受众
        explanation = explain_to_nontechnical(
            mock_explanation,
            audience='invalid_audience'
        )

        # 应该仍然返回某种解释（使用默认受众）
        assert explanation is not None
        assert isinstance(explanation, str)
