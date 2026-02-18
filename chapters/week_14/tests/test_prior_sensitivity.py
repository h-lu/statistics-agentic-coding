"""
Test Suite: Prior Sensitivity Analysis（先验敏感性分析）

测试先验敏感性分析：
1. 比较不同先验对后验的影响
2. 判断结论是否对先验敏感
3. 数据量与先验敏感性的关系
4. 部门间先验分歧的处理

测试覆盖：
- 正确比较多个先验的后验
- 正确判断敏感性
- 理解数据量对敏感性的影响
- 理解强先验的作用
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

# 添加 starter_code 到路径
starter_code_path = Path(__file__).parent.parent / "starter_code"
if str(starter_code_path) not in sys.path:
    sys.path.insert(0, str(starter_code_path))


# =============================================================================
# 多先验比较测试
# =============================================================================

class TestMultiplePriorsComparison:
    """测试多个先验的比较"""

    def test_compare_three_priors(self, multiple_priors_data):
        """
        正例：比较三种先验的后验

        无信息、弱信息、信息性先验
        """
        n = multiple_priors_data['n']
        churned = multiple_priors_data['churned']
        priors = multiple_priors_data['priors']

        posteriors = {}
        for name, (alpha, beta) in priors.items():
            alpha_post = alpha + churned
            beta_post = beta + (n - churned)
            posterior_mean = alpha_post / (alpha_post + beta_post)
            posteriors[name] = {
                'mean': posterior_mean,
                'alpha': alpha_post,
                'beta': beta_post
            }

        # 验证：所有后验均值都合理
        for name, post in posteriors.items():
            assert 0 < post['mean'] < 1

        # 验证：不同先验产生不同后验
        means = [post['mean'] for post in posteriors.values()]
        assert len(set(means)) > 1  # 后验不完全相同

    def test_posterior_means_ordering(self, multiple_priors_data):
        """
        正例：后验均值的排序

        先验均值越低，后验均值通常越低（数据相同）
        """
        n = multiple_priors_data['n']
        churned = multiple_priors_data['churned']
        priors = multiple_priors_data['priors']

        # 计算先验均值
        prior_means = {}
        for name, (alpha, beta) in priors.items():
            prior_means[name] = alpha / (alpha + beta)

        # 计算后验均值
        posterior_means = {}
        for name, (alpha, beta) in priors.items():
            alpha_post = alpha + churned
            beta_post = beta + (n - churned)
            posterior_means[name] = alpha_post / (alpha_post + beta_post)

        # 验证：先验和后验的排序通常一致
        # （对于共轭先验，这个性质成立）
        sorted_priors = sorted(prior_means.items(), key=lambda x: x[1])
        sorted_posteriors = sorted(posterior_means.items(), key=lambda x: x[1])

        # 至少端点应该一致
        assert sorted_priors[0][0] == sorted_posteriors[0][0] or \
               sorted_priors[-1][0] == sorted_posteriors[-1][0]


# =============================================================================
# 敏感性判断测试
# =============================================================================

class TestSensitivityJudgment:
    """测试先验敏感性的判断"""

    def test_not_sensitive_with_large_data(self, prior_sensitivity_scenarios):
        """
        正例：大数据量时不敏感

        当数据量足够大时，不同先验的后验会收敛
        """
        scenario = prior_sensitivity_scenarios['large_data']
        n = scenario['n']
        churned = scenario['churned']
        priors = scenario['priors']

        # 计算后验均值
        posterior_means = []
        for alpha, beta in priors.values():
            alpha_post = alpha + churned
            beta_post = beta + (n - churned)
            posterior_mean = alpha_post / (alpha_post + beta_post)
            posterior_means.append(posterior_mean)

        # 计算范围
        mean_range = max(posterior_means) - min(posterior_means)

        # 验证：范围小于阈值
        assert mean_range < scenario['expected_mean_range']
        assert not scenario['is_sensitive']

    def test_sensitive_with_small_data(self, prior_sensitivity_scenarios):
        """
        正例：小数据量时敏感

        当数据量较小时，先验对后验影响大
        """
        scenario = prior_sensitivity_scenarios['small_data']
        n = scenario['n']
        churned = scenario['churned']
        priors = scenario['priors']

        # 计算后验均值
        posterior_means = []
        for alpha, beta in priors.values():
            alpha_post = alpha + churned
            beta_post = beta + (n - churned)
            posterior_mean = alpha_post / (alpha_post + beta_post)
            posterior_means.append(posterior_mean)

        # 计算范围
        mean_range = max(posterior_means) - min(posterior_means)

        # 验证：范围大于阈值
        assert mean_range > 0.02  # 敏感
        assert scenario['is_sensitive']

    def test_sensitivity_with_extreme_priors(self, prior_sensitivity_scenarios):
        """
        正例：极端先验导致高敏感性

        即使数据量中等，极端先验也会产生分歧
        """
        scenario = prior_sensitivity_scenarios['extreme_prior']
        n = scenario['n']
        churned = scenario['churned']
        priors = scenario['priors']

        # 计算后验均值
        posterior_means = {}
        for name, (alpha, beta) in priors.items():
            alpha_post = alpha + churned
            beta_post = beta + (n - churned)
            posterior_mean = alpha_post / (alpha_post + beta_post)
            posterior_means[name] = posterior_mean

        # 验证：极端先验的后验差异很大
        pessimistic = posterior_means['极端悲观']
        optimistic = posterior_means['极端乐观']
        medium = posterior_means['中等']

        # 验证：极端先验的后验仍受先验影响
        assert pessimistic < medium < optimistic
        assert optimistic - pessimistic > 0.10  # 差异 > 10%


# =============================================================================
# 数据量与敏感性关系测试
# =============================================================================

class TestSampleSizeVsSensitivity:
    """测试数据量与先验敏感性的关系"""

    @pytest.mark.parametrize("n,expected_sensitivity", [
        (10, "high"),      # 极小样本：高敏感
        (50, "high"),      # 小样本：高敏感
        (100, "medium"),   # 中等样本：中等敏感
        (1000, "low"),     # 大样本：低敏感
        (10000, "none"),   # 极大样本：几乎不敏感
    ])
    def test_sensitivity_decreases_with_sample_size(self, n, expected_sensitivity):
        """
        正例：敏感性随样本量增加而降低

        数据越多，先验的影响越小
        """
        # 固定流失率
        churn_rate = 0.18
        churned = int(n * churn_rate)

        # 两个极端先验
        priors = {
            '悲观': (1, 99),    # 均值 1%
            '乐观': (99, 1),    # 均值 99%
        }

        # 计算后验
        posteriors = []
        for alpha, beta in priors.values():
            alpha_post = alpha + churned
            beta_post = beta + (n - churned)
            posterior_mean = alpha_post / (alpha_post + beta_post)
            posteriors.append(posterior_mean)

        # 计算差异
        diff = abs(posteriors[0] - posteriors[1])

        # 验证：样本量越大，差异越小
        # 注意：对于极端先验（1,99）和（99,1），即使有数据，差异也会较大
        # 这里测试的是差异的相对变化趋势
        if expected_sensitivity == "high":
            assert diff > 0.40  # n=10,50: 极端先验仍主导
        elif expected_sensitivity == "medium":
            assert 0.20 < diff < 0.60  # n=100: 开始收敛但仍明显
        elif expected_sensitivity == "low":
            assert 0.05 < diff < 0.20  # n=1000: 数据开始发挥作用
        elif expected_sensitivity == "none":
            assert diff < 0.05  # n=10000: 数据主导

    def test_prior_strength_vs_data_strength(self):
        """
        正例：先验强度 vs 数据强度

        比较先验"伪观测数"和实际观测数
        """
        # 强先验：1000 个伪观测
        alpha_prior, beta_prior = 150, 850
        prior_strength = alpha_prior + beta_prior  # 1000

        # 数据：100 个观测
        n = 100
        churned = 18
        data_strength = n

        # 先验强度 > 数据强度，先验主导
        assert prior_strength > data_strength

        # 后验均值更接近先验均值
        prior_mean = alpha_prior / (alpha_prior + beta_prior)
        data_mean = churned / n

        alpha_post = alpha_prior + churned
        beta_post = beta_prior + (n - churned)
        posterior_mean = alpha_post / (alpha_post + beta_post)

        # 验证：后验更接近先验
        assert abs(posterior_mean - prior_mean) < abs(posterior_mean - data_mean)


# =============================================================================
# 部门间先验分歧测试
# =============================================================================

class TestDepartmentPriorDisagreement:
    """测试部门间先验分歧的处理"""

    def test_marketing_vs_product_priors(self, department_prior_disagreement):
        """
        正例：市场部 vs 产品部的先验分歧

        场景：两个部门对流失率有不同看法
        """
        marketing = department_prior_disagreement['marketing']
        product = department_prior_disagreement['product']
        current_data = department_prior_disagreement['current_data']

        # 市场部先验
        alpha_mkt, beta_mkt = marketing['prior']
        prior_mkt_mean = alpha_mkt / (alpha_mkt + beta_mkt)

        # 产品部先验
        alpha_prod, beta_prod = product['prior']
        prior_prod_mean = alpha_prod / (alpha_prod + beta_prod)

        # 验证：先验分歧
        assert abs(prior_mkt_mean - prior_prod_mean) > 0.05  # 至少差 5%

        # 计算后验
        n = current_data['n']
        churned = current_data['churned']

        alpha_mkt_post = alpha_mkt + churned
        beta_mkt_post = beta_mkt + (n - churned)
        posterior_mkt_mean = alpha_mkt_post / (alpha_mkt_post + beta_mkt_post)

        alpha_prod_post = alpha_prod + churned
        beta_prod_post = beta_prod + (n - churned)
        posterior_prod_mean = alpha_prod_post / (alpha_prod_post + beta_prod_post)

        # 验证：后验仍有分歧（但可能缩小）
        posterior_divergence = abs(posterior_mkt_mean - posterior_prod_mean)

        # 数据应该"拉近距离"，但分歧可能仍然存在
        assert posterior_divergence > 0

    def test_data_convergence_with_more_data(self):
        """
        正例：更多数据使后验收敛

        即使先验分歧很大，足够的数据也能让后验收敛
        """
        # 两个分歧的先验
        prior1 = (10, 90)   # 均值 10%
        prior2 = (30, 70)   # 均值 30%

        # 随着数据量增加，后验差异应该减小
        divergences = []
        data_sizes = [100, 500, 1000, 5000]

        for n in data_sizes:
            churned = int(n * 0.20)  # 真实流失率 20%

            # 后验1
            alpha1_post = prior1[0] + churned
            beta1_post = prior1[1] + (n - churned)
            post1_mean = alpha1_post / (alpha1_post + beta1_post)

            # 后验2
            alpha2_post = prior2[0] + churned
            beta2_post = prior2[1] + (n - churned)
            post2_mean = alpha2_post / (alpha2_post + beta2_post)

            divergences.append(abs(post1_mean - post2_mean))

        # 验证：数据量增加，分歧减小
        assert divergences[0] > divergences[1] > divergences[2] > divergences[3]

        # 验证：大数据后分歧很小
        assert divergences[-1] < 0.01


# =============================================================================
# 先验敏感性报告测试
# =============================================================================

class TestSensitivityReport:
    """测试先验敏感性分析的报告"""

    def test_sensitivity_summary_statistics(self, multiple_priors_data):
        """
        正例：敏感性分析的统计摘要

        计算后验均值、标准差、区间等
        """
        n = multiple_priors_data['n']
        churned = multiple_priors_data['churned']
        priors = multiple_priors_data['priors']

        summary = {}
        for name, (alpha, beta) in priors.items():
            alpha_post = alpha + churned
            beta_post = beta + (n - churned)

            # 均值
            mean = alpha_post / (alpha_post + beta_post)

            # 方差
            var = (alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1))

            # 95% 可信区间
            ci_low, ci_high = stats.beta.interval(0.95, alpha_post, beta_post)

            summary[name] = {
                'mean': mean,
                'std': np.sqrt(var),
                'ci': (ci_low, ci_high)
            }

        # 验证：摘要完整
        assert len(summary) == len(priors)

        for name, stat_dict in summary.items():
            assert 'mean' in stat_dict
            assert 'std' in stat_dict
            assert 'ci' in stat_dict
            assert 0 < stat_dict['mean'] < 1
            assert stat_dict['std'] > 0

    def test_sensitivity_conclusion(self, prior_sensitivity_scenarios):
        """
        正例：敏感性分析的结论

        根据后验差异判断是否敏感
        """
        large_data = prior_sensitivity_scenarios['large_data']
        small_data = prior_sensitivity_scenarios['small_data']

        # 大数据：结论应该是不敏感
        assert not large_data['is_sensitive']

        # 小数据：结论应该是敏感
        assert small_data['is_sensitive']


# =============================================================================
# 先验敏感性可视化数据测试
# =============================================================================

class TestSensitivityVisualization:
    """测试先验敏感性分析的可视化数据"""

    def test_plot_data_for_prior_comparison(self, multiple_priors_data):
        """
        正例：先验比较的绘图数据

        生成用于绘制先验/后验分布的数据
        """
        n = multiple_priors_data['n']
        churned = multiple_priors_data['churned']
        priors = multiple_priors_data['priors']

        # x 轴：概率值
        x = np.linspace(0, 1, 100)

        plot_data = {}
        for name, (alpha, beta) in priors.items():
            # 先验 PDF
            prior_pdf = stats.beta.pdf(x, alpha, beta)

            # 后验 PDF
            alpha_post = alpha + churned
            beta_post = beta + (n - churned)
            posterior_pdf = stats.beta.pdf(x, alpha_post, beta_post)

            plot_data[name] = {
                'x': x,
                'prior_pdf': prior_pdf,
                'posterior_pdf': posterior_pdf,
            }

        # 验证：数据可以用于绘图
        for name, data in plot_data.items():
            assert len(data['x']) == 100
            assert len(data['prior_pdf']) == 100
            assert len(data['posterior_pdf']) == 100
            assert np.all(data['prior_pdf'] >= 0)
            assert np.all(data['posterior_pdf'] >= 0)


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_identical_priors(self):
        """
        边界：相同的先验

        相同先验应该产生相同后验
        """
        n = 100
        churned = 18
        prior = (15, 85)

        # 两个"不同"但相同的先验
        priors = {
            '先验A': prior,
            '先验B': prior,
        }

        posteriors = []
        for alpha, beta in priors.values():
            alpha_post = alpha + churned
            beta_post = beta + (n - churned)
            posterior_mean = alpha_post / (alpha_post + beta_post)
            posteriors.append(posterior_mean)

        # 验证：后验完全相同
        assert abs(posteriors[0] - posteriors[1]) < 1e-10

    def test_single_prior(self):
        """
        边界：单个先验

        只有一个先验时，无法比较
        """
        n = 100
        churned = 18
        priors = {
            '唯一先验': (15, 85),
        }

        # 计算后验
        posteriors = []
        for alpha, beta in priors.values():
            alpha_post = alpha + churned
            beta_post = beta + (n - churned)
            posterior_mean = alpha_post / (alpha_post + beta_post)
            posteriors.append(posterior_mean)

        # 验证：只有一个后验
        assert len(posteriors) == 1

        # 敏感性分析：单个先验时，敏感性无法定义
        # 但可以报告"无比较"
        assert True

    def test_zero_data(self):
        """
        边界：零数据

        没有数据时，后验 = 先验
        """
        n = 0
        churned = 0
        priors = {
            '先验A': (15, 85),
            '先验B': (5, 20),
        }

        prior_means = []
        posterior_means = []

        for alpha, beta in priors.values():
            # 先验均值
            prior_means.append(alpha / (alpha + beta))

            # 后验 = 先验（没有数据）
            posterior_means.append(alpha / (alpha + beta))

        # 验证：后验 = 先验
        for i in range(len(prior_means)):
            assert abs(prior_means[i] - posterior_means[i]) < 1e-10


# =============================================================================
# 实际应用场景测试
# =============================================================================

class TestRealWorldScenarios:
    """测试实际应用场景"""

    def test_ab_test_prior_sensitivity(self):
        """
        正例：A/B 测试中的先验敏感性

        场景：比较两个版本的转化率
        """
        # 版本 A：1000 次展示，200 次转化
        n_a, converted_a = 1000, 200

        # 版本 B：1000 次展示，220 次转化
        n_b, converted_b = 1000, 220

        # 不同先验
        priors = [
            (1, 1),      # 无信息
            (10, 40),    # 弱信息，均值 20%
            (50, 200),   # 信息性，均值 20%
        ]

        results = []
        for alpha, beta in priors:
            # A 的后验
            alpha_a_post = alpha + converted_a
            beta_a_post = beta + (n_a - converted_a)
            mean_a = alpha_a_post / (alpha_a_post + beta_a_post)

            # B 的后验
            alpha_b_post = alpha + converted_b
            beta_b_post = beta + (n_b - converted_b)
            mean_b = alpha_b_post / (alpha_b_post + beta_b_post)

            results.append({
                'prior': (alpha, beta),
                'mean_a': mean_a,
                'mean_b': mean_b,
                'diff': mean_b - mean_a,
            })

        # 验证：所有先验下 B 都优于 A
        for result in results:
            assert result['mean_b'] > result['mean_a']

        # 验证：差异大小对先验不敏感（数据量大）
        diffs = [r['diff'] for r in results]
        assert max(diffs) - min(diffs) < 0.01

    def test_churn_prediction_prior_sensitivity(self, churn_bayes_data):
        """
        正例：流失率预测中的先验敏感性

        场景：不同部门对流失率有不同预期
        """
        n = churn_bayes_data['n']
        churned = churn_bayes_data['churned']

        # 不同部门的先验
        department_priors = {
            '市场部（乐观）': (10, 90),   # 均值 10%
            '数据部门（中性）': (15, 85),  # 均值 15%
            '产品部（悲观）': (25, 75),   # 均值 25%
        }

        posteriors = {}
        for dept, (alpha, beta) in department_priors.items():
            alpha_post = alpha + churned
            beta_post = beta + (n - churned)
            posterior_mean = alpha_post / (alpha_post + beta_post)
            posteriors[dept] = posterior_mean

        # 验证：所有后验都在数据均值附近
        data_mean = churned / n
        for mean in posteriors.values():
            assert abs(mean - data_mean) < 0.03  # 差异 < 3%

        # 结论：数据量大（n=1000），先验影响小
        assert max(posteriors.values()) - min(posteriors.values()) < 0.02
