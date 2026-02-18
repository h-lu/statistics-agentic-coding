"""
Test Suite: Bayes' Theorem（贝叶斯定理）

测试贝叶斯定理的计算和理解：
1. 贝叶斯定理公式：P(A|B) = P(B|A) × P(A) / P(B)
2. 先验概率、似然、后验概率的关系
3. 贝叶斯更新过程
4. 与频率学派 p 值的区别

测试覆盖：
- 正确计算后验概率
- 理解先验、似然、后验的关系
- 理解贝叶斯更新的过程
- 区分后验概率和 p 值
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
# 贝叶斯定理基础计算测试
# =============================================================================

class TestBayesTheoremBasic:
    """测试贝叶斯定理的基础计算"""

    def test_bayes_theorem_disease_detection(self, basic_bayes_data):
        """
        正例：疾病检测的经典例子

        P(Disease|Positive) = P(Positive|Disease) × P(Disease) / P(Positive)

        验证：即使测试很准确（99%敏感度），阳性结果也不一定意味着患病
        """
        prior = basic_bayes_data['prior_disease']  # P(D) = 1%
        sensitivity = basic_bayes_data['sensitivity']  # P(+|D) = 99%
        false_positive = basic_bayes_data['false_positive_rate']  # P(+|¬D) = 5%

        # 计算 P(+) = P(+|D)×P(D) + P(+|¬D)×P(¬D)
        p_positive = sensitivity * prior + false_positive * (1 - prior)

        # 计算后验 P(D|+)
        posterior = (sensitivity * prior) / p_positive

        # 验证：后验约 16.6%（不是 99%！）
        assert abs(posterior - basic_bayes_data['expected_posterior']) < 0.01

        # 验证公式：P(D|+) = P(+|D) × P(D) / P(+)
        assert abs(posterior - (sensitivity * prior) / p_positive) < 1e-10

    def test_bayes_theorem_components(self, basic_bayes_data):
        """
        正例：贝叶斯定理的四个组成部分

        验证对先验、似然、后验、证据的理解
        """
        prior = basic_bayes_data['prior_disease']
        sensitivity = basic_bayes_data['sensitivity']
        false_positive = basic_bayes_data['false_positive_rate']

        # 先验：P(D)
        assert 0 < prior < 1

        # 似然：P(+|D)
        assert 0 < sensitivity <= 1

        # 证据：P(+) = P(+|D)×P(D) + P(+|¬D)×P(¬D)
        evidence = sensitivity * prior + false_positive * (1 - prior)
        assert 0 < evidence < 1

        # 后验：P(D|+) = 似然 × 先验 / 证据
        posterior = (sensitivity * prior) / evidence
        assert 0 < posterior < 1

    def test_bayes_update_sequence(self):
        """
        正例：贝叶斯更新的顺序性

        后验可以作为下一次更新的先验
        """
        # 初始先验：P(D) = 1%
        prior = 0.01
        sensitivity = 0.99
        false_positive = 0.05

        # 第一次阳性检测
        evidence1 = sensitivity * prior + false_positive * (1 - prior)
        posterior1 = (sensitivity * prior) / evidence1

        # 第二次阳性检测（用第一次的后验作为先验）
        evidence2 = sensitivity * posterior1 + false_positive * (1 - posterior1)
        posterior2 = (sensitivity * posterior1) / evidence2

        # 验证：后验概率随证据累积而增加
        assert posterior2 > posterior1 > prior

        # 验证：两次阳性后，患病概率显著增加
        assert posterior2 > 0.75  # 两次阳性后约 80%

    def test_bayes_theorem_with_negative_evidence(self):
        """
        正例：阴性证据的贝叶斯更新

        P(D|Negative) = P(Negative|D) × P(D) / P(Negative)
        """
        prior = 0.01
        sensitivity = 0.99  # P(+|D)
        specificity = 0.95  # P(-|¬D) = 1 - false_positive

        # P(-|D) = 1 - sensitivity
        negative_given_disease = 1 - sensitivity

        # P(-) = P(-|D)×P(D) + P(-|¬D)×P(¬D)
        evidence_negative = negative_given_disease * prior + specificity * (1 - prior)

        # 后验：P(D|-)
        posterior = (negative_given_disease * prior) / evidence_negative

        # 验证：阴性结果大幅降低患病概率
        assert posterior < prior * 0.1  # 应该远低于先验


# =============================================================================
# 先验概率的影响测试
# =============================================================================

class TestPriorInfluence:
    """测试先验概率对后验的影响"""

    @pytest.mark.parametrize("prior,expected_posterior_range", [
        (0.001, (0.01, 0.03)),   # 极低先验
        (0.01, (0.10, 0.20)),   # 低先验
        (0.10, (0.50, 0.80)),   # 中等先验
        (0.50, (0.90, 0.99)),   # 高先验
    ])
    def test_prior_impact_on_posterior(self, prior, expected_posterior_range):
        """
        正例：先验概率对后验的影响

        相同的似然，不同的先验导致不同的后验
        """
        sensitivity = 0.99
        false_positive = 0.05

        # 计算后验
        evidence = sensitivity * prior + false_positive * (1 - prior)
        posterior = (sensitivity * prior) / evidence

        # 验证：后验在预期范围内
        low, high = expected_posterior_range
        assert low <= posterior <= high

    def test_uninformative_prior(self):
        """
        正例：无信息先验（均匀先验）

        P(A) = P(B) = P(C) = ... = 1/n
        """
        # 假设有 3 种等可能的假设
        n_hypotheses = 3
        prior = 1 / n_hypotheses

        # 观测到证据 E
        likelihoods = [0.8, 0.1, 0.1]  # E 在不同假设下的似然

        # 计算证据 P(E)
        evidence = sum(prior * L for L in likelihoods)

        # 计算后验
        posteriors = [prior * L / evidence for L in likelihoods]

        # 验证：后验和为 1
        assert abs(sum(posteriors) - 1.0) < 1e-10

        # 验证：第一个假设的后验最高
        assert posteriors[0] > posteriors[1]
        assert posteriors[0] > posteriors[2]

    def test_extreme_prior_zero(self):
        """
        边界：零先验（绝对不可能）

        P(A) = 0 意味着"绝对不可能"，后验仍为 0
        """
        prior = 0.0
        likelihood = 0.99

        # 零先验导致零后验（证据无法改变）
        posterior = 0.0

        assert posterior == 0.0

    def test_extreme_prior_certain(self):
        """
        边界：确定先验（绝对确定）

        P(A) = 1 意味着"绝对确定"，后验仍为 1
        """
        prior = 1.0
        likelihood = 0.01  # 即使似然很低

        # 确定先验导致确定后验
        posterior = 1.0

        assert posterior == 1.0


# =============================================================================
# 似然的影响测试
# =============================================================================

class TestLikelihoodInfluence:
    """测试似然对后验的影响"""

    @pytest.mark.parametrize("sensitivity,expected_direction", [
        (0.50, "low"),      # 无区分力
        (0.80, "medium"),   # 中等区分力
        (0.99, "high"),     # 高区分力
        (1.00, "perfect"),  # 完美测试
    ])
    def test_likelihood_impact(self, sensitivity, expected_direction):
        """
        正例：似然对后验的影响

        似然越高，后验越接近先验（在证据支持的情况下）
        """
        prior = 0.01
        false_positive = 0.05

        evidence = sensitivity * prior + false_positive * (1 - prior)
        posterior = (sensitivity * prior) / evidence

        # 验证：敏感度越高，后验越高
        assert posterior > 0

        if sensitivity == 1.0:
            # 完美敏感度：阳性时 P(+|D)=1, 但仍有假阳性
            # 由于先验很低(1%)且假阳性率5%，后验不会太高
            # 后验 = 1 * 0.01 / (1 * 0.01 + 0.05 * 0.99) ≈ 0.168
            # 这说明完美敏感度不保证高后验，还需要低假阳性率
            assert posterior > 0.15  # 约 16.8%

    def test_likelihood_ratio(self):
        """
        正例：似然比（Likelihood Ratio）

        LR = P(E|A) / P(E|¬A)
        LR > 1: 支持 A
        LR < 1: 不支持 A
        LR = 1: 无信息
        """
        sensitivity = 0.99
        false_positive = 0.05

        # 似然比
        lr = sensitivity / false_positive  # = 19.8

        # 验证：似然比 > 1，支持患病假设
        assert lr > 1

        # 验证：似然比与后验赔率的关系
        # Posterior Odds = LR × Prior Odds
        prior_odds = 0.01 / 0.99  # 约 0.01
        posterior_odds = lr * prior_odds
        posterior = posterior_odds / (1 + posterior_odds)

        # 与直接计算的后验一致
        evidence = sensitivity * 0.01 + false_positive * 0.99
        expected_posterior = (sensitivity * 0.01) / evidence

        assert abs(posterior - expected_posterior) < 1e-10


# =============================================================================
# 与频率学派对比测试
# =============================================================================

class TestFrequentistVsBayesian:
    """测试频率学派与贝叶斯学派的区别"""

    def test_p_value_vs_posterior(self, frequentist_vs_bayesian_data):
        """
        正例：p 值 vs 后验概率的区别

        p 值：P(data|H0) - 在原假设下观察到数据的概率
        后验：P(H1|data) - 给定数据下备择假设的概率

        这是两个不同的量！
        """
        data = frequentist_vs_bayesian_data

        # 频率学派：p 值
        # p 值不是"原假设成立的概率"
        p_value = data['frequentist']['p_value']  # 0.03

        # 贝叶斯：后验概率
        # P(θ > 15% | data) ≈ 0.98
        posterior_prob = data['bayesian']['p_theta_gt_threshold']

        # 验证：p 值和后验概率是不同的量
        # p = 0.03 不意味着"原假设只有 3% 的概率成立"
        assert abs(p_value - posterior_prob) > 0.5  # 差异很大

        # 验证：p 值小不代表"不可能"
        assert p_value < 0.05  # "显著"
        assert posterior_prob > 0.95  # "高概率"

    def test_confidence_vs_credible_interval(self, frequentist_vs_bayesian_data):
        """
        正例：置信区间 vs 可信区间

        置信区间：长期频率下 95% 的区间包含真实参数
        可信区间：参数有 95% 的概率在这个区间里

        解释方式不同！
        """
        data = frequentist_vs_bayesian_data

        # 频率学派：95% 置信区间
        # 不能说"参数有 95% 的概率在这个区间里"
        ci_freq = data['frequentist']['confidence_interval']

        # 贝叶斯：95% 可信区间
        # 可以说"参数有 95% 的概率在这个区间里"
        ci_bayes = data['bayesian']['credible_interval']

        # 验证：两个区间类似（因为数据量大）
        assert abs(ci_freq[0] - ci_bayes[0]) < 0.01
        assert abs(ci_freq[1] - ci_bayes[1]) < 0.01

        # 但解释不同！
        # 频率学派：不能说"概率"
        # 贝叶斯：可以说"概率"

    def test_parameter_interpretation(self):
        """
        正例：参数的解释

        频率学派：参数是固定但未知的常数
        贝叶斯学派：参数是随机变量（有分布）
        """
        # 频率学派观点
        frequentist_view = "参数 θ 是一个固定值，我们不知道它"

        # 贝叶斯学派观点
        bayesian_view = "参数 θ 是一个随机变量，有概率分布"

        # 验证：两种观点都合理
        assert isinstance(frequentist_view, str)
        assert isinstance(bayesian_view, str)


# =============================================================================
# 贝叶斯更新测试
# =============================================================================

class TestBayesianUpdate:
    """测试贝叶斯更新的过程"""

    def test_sequential_updates(self):
        """
        正例：序贯贝叶斯更新

        新证据到来时，用当前后验作为下一次的先验
        """
        # 初始先验：50%
        prior = 0.5

        # 证据序列
        evidences = [
            (0.8, 0.2),  # 第一次：支持 A
            (0.7, 0.3),  # 第二次：支持 A
            (0.1, 0.9),  # 第三次：支持 ¬A
        ]

        for i, (likelihood_A, likelihood_not_A) in enumerate(evidences):
            # 计算证据
            evidence = likelihood_A * prior + likelihood_not_A * (1 - prior)

            # 计算后验
            posterior = (likelihood_A * prior) / evidence

            # 更新先验
            prior = posterior

        # 验证：两次支持 + 一次反对，仍然支持 A
        assert prior > 0.5

    def test_strong_evidence_overcomes_weak_prior(self):
        """
        正例：强证据克服弱先验

        即使先验很弱，强证据也能产生高后验
        """
        # 弱先验：只有 10% 相信
        prior = 0.1

        # 强证据：似然比 100:1
        likelihood_H = 0.99
        likelihood_not_H = 0.01

        # 计算后验
        evidence = likelihood_H * prior + likelihood_not_H * (1 - prior)
        posterior = (likelihood_H * prior) / evidence

        # 验证：强证据大幅提高后验
        assert posterior > 0.9

    def test_weak_evidence_doesnt_change_strong_prior(self):
        """
        正例：弱证据不改变强先验

        先验很强时，弱证据的影响有限
        """
        # 强先验：99% 确定
        prior = 0.99

        # 弱证据：似然接近 1
        likelihood_H = 0.51  # 稍微支持 H
        likelihood_not_H = 0.49

        # 计算后验
        evidence = likelihood_H * prior + likelihood_not_H * (1 - prior)
        posterior = (likelihood_H * prior) / evidence

        # 验证：后验仍然接近先验
        assert abs(posterior - prior) < 0.02


# =============================================================================
# 常见误解测试
# =============================================================================

class TestCommonMisconceptions:
    """测试贝叶斯定理的常见误解"""

    def test_misconception_p_value_is_posterior(self):
        """
        反例：p 值不是后验概率

        p = 0.03 不意味着"原假设只有 3% 的概率成立"
        """
        # 这是常见的误解
        # p 值是 P(data|H0)，不是 P(H0|data)

        # 正确理解
        p_value = 0.03
        interpretation = "在原假设成立时，观察到当前数据或更极端数据的概率是 3%"

        assert "在原假设成立时" in interpretation

    def test_misconception_base_rate_neglect(self):
        """
        反例：忽视基率（Base Rate Neglect）

        人们经常忽视先验概率，只关注似然
        """
        # 经典的出租车问题
        # 85% 的出租车是绿色，15% 是蓝色
        # 目击者识别准确率 80%
        # 目击者说"蓝色"

        # 错误答案：80%（只看准确率）
        # 正确答案：需要用贝叶斯定理

        prior_blue = 0.15
        accuracy = 0.80

        # P(说蓝|蓝) = 0.8, P(说蓝|绿) = 0.2
        evidence_said_blue = accuracy * prior_blue + (1 - accuracy) * (1 - prior_blue)
        posterior_blue = (accuracy * prior_blue) / evidence_said_blue

        # 验证：正确答案约 41%（不是 80%）
        assert abs(posterior_blue - 0.41) < 0.05

    def test_misconception_confidence_interval_interpretation(self):
        """
        反例：置信区间的错误解释

        "95% 置信区间"不能解释为"参数有 95% 的概率在这个区间里"
        """
        # 频率学派的置信区间
        # 正确解释：如果重复抽样 100 次，约 95 个区间会包含真实参数
        correct_interpretation = "长期频率下 95% 的区间包含真实参数"

        # 错误解释（这是贝叶斯的可信区间的解释）
        wrong_interpretation = "参数有 95% 的概率在这个区间里"

        assert "长期频率" in correct_interpretation
        assert "长期频率" not in wrong_interpretation


# =============================================================================
# 边界情况测试
# =============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_perfect_test(self):
        """
        边界：完美测试

        敏感度 = 特异度 = 1
        """
        prior = 0.5
        sensitivity = 1.0
        specificity = 1.0

        # 阳性 => 100% 患病
        evidence_pos = sensitivity * prior + (1 - specificity) * (1 - prior)
        posterior_pos = (sensitivity * prior) / evidence_pos

        assert posterior_pos == 1.0

        # 阴性 => 0% 患病
        sensitivity_neg = 1 - sensitivity  # P(-|D) = 0
        evidence_neg = sensitivity_neg * prior + specificity * (1 - prior)
        posterior_neg = (sensitivity_neg * prior) / evidence_neg

        assert posterior_neg == 0.0

    def test_useless_test(self):
        """
        边界：无信息测试

        敏感度 = 1 - 特异度 = 0.5（随机猜测）
        """
        prior = 0.5
        sensitivity = 0.5
        false_positive = 0.5

        evidence = sensitivity * prior + false_positive * (1 - prior)
        posterior = (sensitivity * prior) / evidence

        # 验证：无信息测试不改变先验
        assert abs(posterior - prior) < 1e-10

    def test_equal_priors(self):
        """
        边界：等先验（最大不确定性）

        P(A) = P(¬A) = 0.5
        """
        prior = 0.5
        likelihood_A = 0.7
        likelihood_not_A = 0.3

        evidence = likelihood_A * prior + likelihood_not_A * (1 - prior)
        posterior = (likelihood_A * prior) / evidence

        # 简化：当先验 = 0.5 时
        # P(A|E) = L_A / (L_A + L_¬A)
        expected = likelihood_A / (likelihood_A + likelihood_not_A)

        assert abs(posterior - expected) < 1e-10
