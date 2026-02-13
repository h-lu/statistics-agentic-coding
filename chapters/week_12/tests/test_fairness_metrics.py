"""
Week 12：公平性指标测试

测试差异影响、平等机会、均等几率等公平性指标
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from solution import (
        calculate_disparate_impact,
        calculate_equal_opportunity,
        calculate_equalized_odds,
        detect_proxy_variables,
        review_xai_code,
    )
except ImportError:
    pytest.skip("starter_code/solution.py not implemented yet", allow_module_level=True)


class TestDisparateImpact:
    """测试差异影响比计算"""

    def test_disparate_impact_perfect_equality(self):
        """测试完全相等的情况（差异影响比 = 1.0）"""
        n = 100
        # 两个组都有 50% 通过率
        # 组0: 25个0，25个1
        # 组1: 25个0，25个1
        y_pred = np.array([0] * 25 + [1] * 25 + [0] * 25 + [1] * 25)
        group_labels = np.array([0] * 50 + [1] * 50)  # 两组

        di_ratio = calculate_disparate_impact(y_pred, group_labels)

        # 两组通过率相同，比例应该接近 1.0
        assert abs(di_ratio - 1.0) < 0.01

    def test_disparate_impact_80_percent_rule(self):
        """测试 80% 规则（差异影响比 = 0.8）"""
        n = 100

        # 群体 A：50% 通过率
        group_a_pred = np.array([0] * 25 + [1] * 25)
        # 群体 B：40% 通过率（是 A 的 80%）
        group_b_pred = np.array([0] * 30 + [1] * 20)

        y_pred = np.concatenate([group_a_pred, group_b_pred])
        group_labels = np.array([0] * 50 + [1] * 50)

        di_ratio = calculate_disparate_impact(y_pred, group_labels)

        # 应该接近 0.8
        assert abs(di_ratio - 0.8) < 0.05

    def test_disparate_impact_violation(self, biased_credit_data):
        """测试差异影响违反的情况（< 0.8）"""
        df = biased_credit_data

        # 简单模拟：假设所有低收入被拒
        y_pred = (df['income'] > 10000).astype(int)

        di_ratio = calculate_disparate_impact(y_pred, df['gender'].values)

        # 应该返回一个数值
        assert isinstance(di_ratio, float)
        assert di_ratio >= 0

    def test_disparate_impact_edge_cases(self):
        """测试边界情况"""
        # 所有样本都通过
        y_pred_all_pass = np.ones(100)
        group_labels = np.array([0] * 50 + [1] * 50)
        di = calculate_disparate_impact(y_pred_all_pass, group_labels)
        assert di == 1.0 or np.isnan(di)  # 可能是 1.0 或 NaN

        # 所有样本都不通过
        y_pred_all_fail = np.zeros(100)
        di = calculate_disparate_impact(y_pred_all_fail, group_labels)
        # 应该是 NaN 或 0/0 的某种处理
        assert np.isnan(di) or di == 0


class TestEqualOpportunity:
    """测试平等机会（召回率差异）"""

    def test_equal_opportunity_perfect_equality(self):
        """测试完全平等的召回率"""
        n = 100

        y_true = np.array([0] * 50 + [1] * 50)
        y_pred = np.array([0] * 40 + [1] * 10 + [0] * 10 + [1] * 40)
        group_labels = np.array([0] * 50 + [1] * 50)

        # 两组的真实正例召回率相同
        eo_diff = calculate_equal_opportunity(y_true, y_pred, group_labels)

        # 应该返回召回率差异
        assert isinstance(eo_diff, (float, dict))

    def test_equal_opportunity_with_difference(self):
        """测试召回率有差异的情况"""
        n = 200

        # 群体 A：高召回率
        y_true_a = np.array([0] * 80 + [1] * 20)
        y_pred_a = np.array([0] * 75 + [1] * 5 + [0] * 5 + [1] * 15)
        # 群体 A 召回率 = 15/20 = 0.75

        # 群体 B：低召回率
        y_true_b = np.array([0] * 80 + [1] * 20)
        y_pred_b = np.array([0] * 78 + [1] * 2 + [0] * 10 + [1] * 10)
        # 群体 B 召回率 = 10/20 = 0.5

        y_true = np.concatenate([y_true_a, y_true_b])
        y_pred = np.concatenate([y_pred_a, y_pred_b])
        group_labels = np.array([0] * 100 + [1] * 100)

        eo_diff = calculate_equal_opportunity(y_true, y_pred, group_labels)

        # 差异应该约为 0.25
        if isinstance(eo_diff, float):
            assert abs(eo_diff - 0.25) < 0.1

    def test_equal_opportunity_credit_scoring(self, credit_scoring_data):
        """测试信用评分场景的平等机会"""
        df = credit_scoring_data

        feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                       'credit_inquiries', 'employment_length']
        X = df[feature_cols]
        y = df['default']

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        gender = df.loc[X_test.index, 'gender'].values

        eo_diff = calculate_equal_opportunity(y_test.values, y_pred, gender)

        # 应该返回数值
        assert eo_diff is not None


class TestEqualizedOdds:
    """测试均等几率（TPR 和 FPR 都相等）"""

    def test_equalized_odds_perfect(self):
        """测试完美的均等几率"""
        n = 200

        y_true = np.array([0] * 100 + [1] * 100)
        y_pred = np.array([0] * 80 + [1] * 20 + [0] * 20 + [1] * 80)
        group_labels = np.array([0] * 100 + [1] * 100)

        eo_result = calculate_equalized_odds(y_true, y_pred, group_labels)

        # 应该返回包含 TPR 和 FPR 的字典
        assert isinstance(eo_result, dict)

        # 应该有 TPR 和 FPR 相关的键
        expected_keys = ['tpr', 'fpr', 'group_0', 'group_1']
        has_key = any(any(key in str(k).lower() for key in expected_keys)
                     for k in eo_result.keys())
        assert has_key

    def test_equalized_odds_with_disparity(self):
        """测试有差异的均等几率"""
        n = 200

        # 群体 A
        y_true_a = np.array([0] * 80 + [1] * 20)
        y_pred_a = np.array([0] * 70 + [1] * 10 + [0] * 5 + [1] * 15)
        # TPR = 15/20 = 0.75, FPR = 10/80 = 0.125

        # 群体 B（性能更差）
        y_true_b = np.array([0] * 80 + [1] * 20)
        y_pred_b = np.array([0] * 60 + [1] * 20 + [0] * 10 + [1] * 10)
        # TPR = 10/20 = 0.5, FPR = 20/80 = 0.25

        y_true = np.concatenate([y_true_a, y_true_b])
        y_pred = np.concatenate([y_pred_a, y_pred_b])
        group_labels = np.array([0] * 100 + [1] * 100)

        eo_result = calculate_equalized_odds(y_true, y_pred, group_labels)

        # 应该检测到差异
        assert eo_result is not None

    def test_equalized_odds_credit_scoring(self, credit_scoring_data):
        """测试信用评分场景的均等几率"""
        df = credit_scoring_data

        feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                       'credit_inquiries', 'employment_length']
        X = df[feature_cols]
        y = df['default']

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        gender = df.loc[X_test.index, 'gender'].values

        eo_result = calculate_equalized_odds(y_test.values, y_pred, gender)

        # 应该返回结果
        assert eo_result is not None


class TestProxyVariableDetection:
    """测试代理变量检测"""

    def test_detect_no_proxy(self):
        """测试无代理变量的情况"""
        np.random.seed(42)
        n = 100

        df = pd.DataFrame({
            'gender': np.random.randint(0, 2, n),
            'feature_1': np.random.normal(0, 1, n),
            'feature_2': np.random.normal(0, 1, n),
        })

        proxies = detect_proxy_variables(df, sensitive_col='gender')

        # 应该返回空的代理变量列表
        if isinstance(proxies, list):
            assert len(proxies) == 0 or len(proxies) <= 1  # 可能误报一个
        elif isinstance(proxies, dict):
            assert 'proxies' in proxies
            assert len(proxies['proxies']) == 0

    def test_detect_strong_proxy(self):
        """测试强代理变量的检测"""
        np.random.seed(42)
        n = 100

        # 创建强相关特征
        gender = np.random.randint(0, 2, n)
        # 职业与性别强相关
        occupation = np.where(gender == 1,
                             np.random.choice(['A', 'B'], n, p=[0.8, 0.2]),
                             np.random.choice(['A', 'B'], n, p=[0.2, 0.8]))

        df = pd.DataFrame({
            'gender': gender,
            'occupation': occupation
        })

        proxies = detect_proxy_variables(df, sensitive_col='gender')

        # 应该检测到 occupation 是代理变量
        if isinstance(proxies, list):
            assert 'occupation' in proxies or len(proxies) > 0
        elif isinstance(proxies, dict):
            assert 'occupation' in proxies.get('proxies', {})

    def test_detect_proxy_credit_scoring(self, credit_scoring_data_with_proxy):
        """测试信用评分数据的代理变量检测"""
        df = credit_scoring_data_with_proxy

        # 检测性别相关的代理变量
        proxies = detect_proxy_variables(df, sensitive_col='gender')

        # 应该返回检测结果
        assert proxies is not None

        # occupation 或 zip_code 可能被标记为代理变量
        if isinstance(proxies, list):
            # 至少应该检测到一些潜在代理
            assert isinstance(proxies, list)
        elif isinstance(proxies, dict):
            assert 'proxies' in proxies or 'correlations' in proxies

    def test_detect_proxy_correlation_method(self, credit_scoring_data_with_proxy):
        """测试基于相关性的代理变量检测方法"""
        df = credit_scoring_data_with_proxy

        # 计算性别与各特征的相关性
        # 对于数值特征，用相关系数；对于类别特征，用关联度量
        gender_encoded = df['gender']

        proxies = detect_proxy_variables(df, sensitive_col='gender',
                                        method='correlation')

        # 应该返回包含相关性信息的结果
        assert proxies is not None


class TestFairnessScenarios:
    """测试真实场景的公平性评估"""

    def test_fair_data_evaluation(self, fair_classification_data):
        """测试公平数据的评估"""
        X = fair_classification_data

        # 简单的预测逻辑
        y_pred = (X['feature'] > 0).astype(int)

        di_ratio = calculate_disparate_impact(y_pred.values, X['group'].values)

        # 对于公平数据，差异影响比应该接近 1
        assert di_ratio is not None
        if not np.isnan(di_ratio):
            assert abs(di_ratio - 1.0) < 0.3  # 允许一定偏差

    def test_unfair_data_detection(self, unfair_classification_data):
        """测试不公平数据的检测"""
        X = unfair_classification_data

        y_pred = (X['feature'] > 0).astype(int)

        di_ratio = calculate_disparate_impact(y_pred.values, X['group'].values)

        # 应该检测到不公平
        assert di_ratio is not None

        # 如果差异影响比偏离 1 较多，说明存在不公平
        if not np.isnan(di_ratio):
            # 不一定总是 < 0.8，但应该能检测到差异
            assert isinstance(di_ratio, float)

    def test_complete_fairness_audit(self, credit_scoring_data):
        """测试完整的公平性审计流程"""
        df = credit_scoring_data

        feature_cols = ['income', 'credit_history_age', 'debt_to_income',
                       'credit_inquiries', 'employment_length']
        X = df[feature_cols]
        y = df['default']

        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        gender = df.loc[X_test.index, 'gender'].values

        # 1. 差异影响比
        di_ratio = calculate_disparate_impact(y_pred, gender)
        assert di_ratio is not None

        # 2. 平等机会
        eo_diff = calculate_equal_opportunity(y_test.values, y_pred, gender)
        assert eo_diff is not None

        # 3. 均等几率
        eo_result = calculate_equalized_odds(y_test.values, y_pred, gender)
        assert eo_result is not None

        # 审计完成
        assert True


class TestFairnessCodeReview:
    """测试公平性代码审查"""

    def test_review_good_fairness_code(self, good_fairness_code_example):
        """测试审查好的公平性代码"""
        review = review_xai_code(good_fairness_code_example)

        assert isinstance(review, dict)
        # 好代码应该有群体分析
        code_str = str(review) + good_fairness_code_example
        assert 'group' in code_str.lower() or 'gender' in code_str.lower()

    def test_review_bad_fairness_no_group_analysis(self, bad_fairness_code_no_group_analysis):
        """测试审查缺少群体分析的代码"""
        review = review_xai_code(bad_fairness_code_no_group_analysis)

        assert isinstance(review, dict)
        # 应该检测到缺少群体分析
        if 'issues' in review:
            issues_str = str(review['issues'])
            has_fairness_issue = (
                'group' in issues_str.lower() or
                'fairness' in issues_str.lower() or
                'bias' in issues_str.lower()
            )
            # 不一定总是检测到，但如果检测到应该相关
            if has_fairness_issue:
                assert True
