"""
条件概率与贝叶斯定理测试

测试条件概率计算和贝叶斯定理应用的正确性。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats


# =============================================================================
# 条件概率计算测试
# =============================================================================

class TestConditionalProbability:
    """测试条件概率计算"""

    def test_conditional_probability_basic(self):
        """测试基本的条件概率计算 P(A|B) = P(A∩B) / P(B)"""
        # 构造简单的联合概率表
        #        健康  患病
        # 阳性   99    99    P(阳性)=198/10000=0.0198
        # 阴性   9801  1     P(阴性)=9802/10000=0.9802
        #        9900  100

        p_sick = 100 / 10000
        p_positive_given_sick = 99 / 100
        p_positive_given_healthy = 99 / 9900

        p_positive = p_positive_given_sick * p_sick + p_positive_given_healthy * (1 - p_sick)
        p_sick_given_positive = (p_positive_given_sick * p_sick) / p_positive

        # 在这个经典例子中，P(患病|阳性) 应该约等于 0.5
        assert 0.45 <= p_sick_given_positive <= 0.55

    def test_conditional_probability_formula(self):
        """测试条件概率公式的正确性"""
        # 简单例子：掷骰子
        # P(偶数|大于3) = P(偶数且大于3) / P(大于3)
        # 大于3的数：{4,5,6}，其中偶数：{4,6}
        # P(偶数|大于3) = 2/3

        p_even_and_gt3 = 2 / 6  # {4, 6}
        p_gt3 = 3 / 6  # {4, 5, 6}

        p_even_given_gt3 = p_even_and_gt3 / p_gt3

        assert abs(p_even_given_gt3 - 2/3) < 0.001

    def test_conditional_probability_direction_matters(self):
        """测试 P(A|B) ≠ P(B|A) 的反直觉"""
        # 构造一个例子
        # P(医生|女性) ≠ P(女性|医生)

        n_doctors = 100
        n_females = 10000
        n_female_doctors = 30

        p_doctor_given_female = n_female_doctors / n_females
        p_female_given_doctor = n_female_doctors / n_doctors

        # 这两个概率应该非常不同
        # p_doctor_given_female ≈ 0.003, p_female_given_doctor = 0.3
        # 差值约 0.297，显著大于 0
        assert abs(p_doctor_given_female - p_female_given_doctor) > 0.25

    def test_impossible_condition(self):
        """测试不可能条件下的概率"""
        # P(A|不可能事件) 未定义

        with pytest.raises(ZeroDivisionError):
            _ = 10 / 0  # P(A) / P(不可能)


# =============================================================================
# 贝叶斯定理测试
# =============================================================================

class TestBayesTheorem:
    """测试贝叶斯定理应用"""

    @pytest.mark.parametrize("prevalence,sensitivity,specificity,expected_range", [
        (0.01, 0.99, 0.99, (0.45, 0.55)),  # 经典例子
        (0.001, 0.99, 0.99, (0.05, 0.15)),  # 极罕见病
        (0.1, 0.99, 0.99, (0.90, 0.99)),  # 较高发病率
        (0.5, 0.90, 0.90, (0.85, 0.95)),  # 均衡情况
    ])
    def test_bayes_disease_detection(self, prevalence, sensitivity, specificity, expected_range):
        """参数化测试：医疗检测问题的贝叶斯计算"""
        # P(患病|阳性) = P(阳性|患病) × P(患病) / P(阳性)

        p_positive_given_sick = sensitivity
        p_sick = prevalence
        p_positive_given_healthy = 1 - specificity

        p_positive = (p_positive_given_sick * p_sick +
                     p_positive_given_healthy * (1 - p_sick))

        p_sick_given_positive = (p_positive_given_sick * p_sick) / p_positive

        low, high = expected_range
        assert low <= p_sick_given_positive <= high

    def test_bayes_components(self):
        """测试贝叶斯定理的各组成部分"""
        # 后验 = (似然 × 先验) / 证据

        # 似然 P(阳性|患病) = 0.99
        likelihood = 0.99

        # 先验 P(患病) = 0.01
        prior = 0.01

        # 证据 P(阳性) = P(阳性|患病)×P(患病) + P(阳性|健康)×P(健康)
        p_positive_given_healthy = 0.01
        evidence = likelihood * prior + p_positive_given_healthy * (1 - prior)

        # 后验 P(患病|阳性)
        posterior = (likelihood * prior) / evidence

        # 验证关系
        assert 0 < posterior < 1
        assert posterior > prior  # 阳性证据提高了患病概率

    def test_bayes_extreme_cases(self, disease_test_data):
        """测试边界情况"""
        # 情况 1：完全准确的检测（敏感性=特异性=1）
        p_sick_given_positive = (
            disease_test_data['sensitivity'] * disease_test_data['prevalence']
        ) / (
            disease_test_data['sensitivity'] * disease_test_data['prevalence'] +
            (1 - disease_test_data['specificity']) * (1 - disease_test_data['prevalence'])
        )

        # 如果检测完美，P(患病|阳性) 应该等于 P(阳性|患病) = 1
        # 但实际上因为特异性=1，P(阳性|健康)=0，所以简化

        # 情况 2：先验概率为 0.5（无信息）
        prevalence_neutral = 0.5
        # ...更多边界测试


# =============================================================================
# 模拟验证测试
# =============================================================================

class TestSimulationVerification:
    """测试用模拟验证贝叶斯计算"""

    def test_simulate_disease_detection(self, disease_test_data):
        """测试：模拟结果应该接近理论值"""
        population = disease_test_data['population']
        prevalence = disease_test_data['prevalence']
        sensitivity = disease_test_data['sensitivity']
        specificity = disease_test_data['specificity']

        np.random.seed(42)

        # 生成真实患病状态
        true_sick = np.random.random(population) < prevalence

        # 生成检测结果
        test_positive = np.where(
            true_sick,
            np.random.random(population) < sensitivity,
            np.random.random(population) < (1 - specificity)
        )

        # 计算条件概率
        n_positive = test_positive.sum()
        n_sick_and_positive = (true_sick & test_positive).sum()

        p_sick_given_positive_sim = n_sick_and_positive / n_positive

        # 理论值
        p_positive_given_sick = sensitivity
        p_positive_given_healthy = 1 - specificity
        p_sick = prevalence
        p_positive = p_positive_given_sick * p_sick + p_positive_given_healthy * (1 - p_sick)
        p_sick_given_positive_theory = (p_positive_given_sick * p_sick) / p_positive

        # 模拟值应该接近理论值（在大样本下）
        assert abs(p_sick_given_positive_sim - p_sick_given_positive_theory) < 0.05

    def test_simulation_stability(self):
        """测试：多次模拟结果应该稳定"""
        np.random.seed(42)

        results = []
        for seed in range(10):
            np.random.seed(seed)
            population = 10000
            prevalence = 0.01
            sensitivity = 0.99

            true_sick = np.random.random(population) < prevalence
            test_positive = np.where(
                true_sick,
                np.random.random(population) < sensitivity,
                np.random.random(population) < 0.01
            )

            n_positive = test_positive.sum()
            n_sick_and_positive = (true_sick & test_positive).sum()
            p = n_sick_and_positive / n_positive
            results.append(p)

        # 标准差应该较小（结果稳定）
        # 由于样本随机性，允许稍大一些的标准差
        assert np.std(results) < 0.06


# =============================================================================
# 反例测试
# =============================================================================

class TestConditionalProbabilityMistakes:
    """测试条件概率常见错误"""

    def test_confusing_direction(self):
        """测试：混淆 P(A|B) 和 P(B|A)"""
        # P(医生|女性) ≈ 0.003
        # P(女性|医生) ≈ 0.3

        # 如果有人混淆了方向，会得到错误的结论
        p_doctor_given_female = 30 / 10000
        p_female_given_doctor = 30 / 100

        # 混淆的结果
        confused_result = p_doctor_given_female  # 但以为是 P(女性|医生)
        correct_result = p_female_given_doctor

        assert confused_result != correct_result

    def test_ignoring_base_rate(self):
        """测试：忽略基础率（base rate neglect）"""
        # 如果 P(B|A) 很高，不代表 P(A|B) 也很高
        # 必须考虑先验 P(A)

        # 几乎不可能的事件 A，P(A) ≈ 0
        p_A = 0.0001
        p_B_given_A = 0.99
        p_B_given_not_A = 0.01

        # 贝叶斯计算
        p_B = p_B_given_A * p_A + p_B_given_not_A * (1 - p_A)
        p_A_given_B = (p_B_given_A * p_A) / p_B

        # 即使 P(B|A) = 99%，P(A|B) 也只有约 1%
        assert p_A_given_B < 0.02
