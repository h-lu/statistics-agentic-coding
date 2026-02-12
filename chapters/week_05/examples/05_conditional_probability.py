#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立示例 1：条件概率与贝叶斯定理

本示例独立演示条件概率计算和贝叶斯定理应用，
与主示例文件分离，便于单独学习和引用。

运行方式：python3 chapters/week_05/examples/05_conditional_probability.py
"""
from __future__ import annotations

import numpy as np


def calculate_conditional_probability(
    n_A_and_B: int,
    n_B: int
) -> float:
    """
    计算条件概率 P(A|B)

    参数：
        n_A_and_B: 同时满足 A 和 B 的数量
        n_B: 满足 B 的数量

    返回：
        float: 条件概率 P(A|B)
    """
    if n_B == 0:
        raise ValueError("n_B 不能为 0，因为不能以不可能事件为条件")
    return n_A_and_B / n_B


def bayes_theorem(
    p_B_given_A: float,  # 似然 P(B|A)
    p_A: float,          # 先验 P(A)
    p_B_given_not_A: float,  # P(B|¬A)
) -> dict[str, float]:
    """
    应用贝叶斯定理计算后验概率

    参数：
        p_B_given_A: P(B|A)，似然
        p_A: P(A)，先验概率
        p_B_given_not_A: P(B|¬A)，假阳性率

    返回：
        dict: 包含后验概率和中间步骤
    """
    p_not_A = 1 - p_A
    p_B = p_B_given_A * p_A + p_B_given_not_A * p_not_A

    if p_B == 0:
        raise ValueError("p_B 不能为 0")

    p_A_given_B = (p_B_given_A * p_A) / p_B

    return {
        'prior': p_A,
        'likelihood': p_B_given_A,
        'evidence': p_B,
        'posterior': p_A_given_B
    }


def disease_detection_example():
    """医疗检测的经典贝叶斯问题"""
    print("\n" + "=" * 70)
    print("条件概率与贝叶斯定理：医疗检测问题")
    print("=" * 70)

    # 参数设置
    prevalence = 0.01      # 发病率 P(患病)
    sensitivity = 0.99      # 敏感性 P(阳性|患病)
    specificity = 0.99      # 特异性 P(阴性|健康)

    print(f"\n场景设置：")
    print(f"  - 发病率（先验）：{prevalence:.1%}")
    print(f"  - 检测敏感性 P(阳性|患病)：{sensitivity:.1%}")
    print(f"  - 检测特异性 P(阴性|健康)：{specificity:.1%}")

    # 应用贝叶斯定理
    result = bayes_theorem(
        p_B_given_A=sensitivity,
        p_A=prevalence,
        p_B_given_not_A=1 - specificity  # P(阳性|健康)
    )

    print(f"\n计算结果：")
    print(f"  - 先验 P(患病)：{result['prior']:.1%}")
    print(f"  - 似然 P(阳性|患病)：{result['likelihood']:.1%}")
    print(f"  - 证据 P(阳性)：{result['evidence']:.4f}")
    print(f"  - 后验 P(患病|阳性)：{result['posterior']:.1%}")

    print(f"\n关键结论：")
    print(f"  即使检测准确率 99%，检测阳性后真正患病的概率只有")
    print(f"  {result['posterior']:.1%}，远低于直觉的 99%！")

    print(f"\n小北的困惑：")
    print(f"  '这意味着检测阳性的人，一半以上其实是健康的？'")
    print(f"  '那这个检测还有什么用？'")

    print(f"\n老潘的解释：")
    print(f"  '检测的用途是筛检，不是确诊。'")
    print(f"  '阳性后需要进一步检查确认。'")
    print(f"  '如果发病率更低（如 0.1%），P(患病|阳性) 会更低。'")

    return result


def simulate_detection(
    population: int = 100000,
    prevalence: float = 0.01,
    sensitivity: float = 0.99,
    specificity: float = 0.99,
    seed: int = 42
) -> None:
    """用模拟验证贝叶斯计算"""
    np.random.seed(seed)

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

    p_sick_given_positive = n_sick_and_positive / n_positive

    print(f"\n模拟验证（{population:,} 人）：")
    print(f"  - 真实患病：{true_sick.sum():,} 人")
    print(f"  - 检测阳性：{n_positive:,} 人")
    print(f"  - 真阳性（患病且阳性）：{n_sick_and_positive:,} 人")
    print(f"  - 假阳性（健康但阳性）：{n_positive - n_sick_and_positive:,} 人")
    print(f"  - P(患病|阳性) = {p_sick_given_positive:.1%}")


def main() -> None:
    """主函数"""
    print("\n" + "=" * 70)
    print("独立示例 1：条件概率与贝叶斯定理")
    print("=" * 70)

    # 理论计算
    disease_detection_example()

    # 模拟验证
    simulate_detection()

    # 不同发病率下的后验概率
    print("\n" + "=" * 70)
    print("扩展：不同发病率下的 P(患病|阳性)")
    print("=" * 70)

    prevalences = [0.001, 0.01, 0.05, 0.1, 0.5]
    print(f"\n假设检测敏感性=特异性=99%：")
    print(f"{'发病率':<10} {'P(患病|阳性)':<15} {'解读'}")
    print("-" * 70)

    for prev in prevalences:
        result = bayes_theorem(
            p_B_given_A=0.99,
            p_A=prev,
            p_B_given_not_A=0.01
        )
        posterior = result['posterior']

        if prev == 0.001:
            interp = "极罕见病，阳性几乎全是假阳性"
        elif prev == 0.01:
            interp = "罕见病，阳性仅 50% 真阳性"
        elif prev == 0.05:
            interp = "较低发病率，阳性约 84% 真阳性"
        elif prev == 0.1:
            interp = "中等发病率，阳性约 92% 真阳性"
        else:
            interp = "高发病率（50%），检测基本可靠"

        print(f"{prev:<10.1%} {posterior:<15.1%} {interp}")

    print("\n" + "=" * 70)
    print("关键洞察")
    print("=" * 70)
    print("贝叶斯定理的核心是'更新信念'：")
    print("  - 先验（发病率）越低，后验（阳性后患病概率）越低")
    print("  - 即使检测很准确，罕见病的阳性结果也不可靠")
    print("  - 这就是为什么罕见病筛查需要多次确认")
    print("=" * 70)


if __name__ == "__main__":
    main()
