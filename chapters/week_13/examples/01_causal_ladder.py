"""
示例：因果推断三层级——从关联到干预到反事实

本例演示 Judea Pearl 的因果推断三层级：
1. 关联（Association）：看到 X 如何变化，P(y|x)
2. 干预（Intervention）：如果做 X 会怎样，P(y|do(x))
3. 反事实（Counterfactual）：如果当时没做 X 会怎样，P(y_x|x',y')

运行方式：python3 chapters/week_13/examples/01_causal_ladder.py
预期输出：stdout 输出三层级的区别和示例
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================================
# 第1层：关联（Association）——模型能做什么
# ============================================================================

def demonstrate_association(df: pd.DataFrame) -> dict:
    """
    第1层：关联

    问题是：看到 X 和 Y 相关
    方法：相关系数、回归系数、监督学习
    陷阱：相关 != 因果

    示例：使用优惠券的客户流失率更低（P(churn|coupon) < P(churn|no coupon)）
    但这不意味着优惠券降低了流失率（可能是高价值客户才收到优惠券）
    """
    print("=" * 60)
    print("第1层：关联（Association）")
    print("=" * 60)

    # 计算条件概率
    churn_given_coupon = df[df['coupon'] == 1]['churn'].mean()
    churn_given_no_coupon = df[df['coupon'] == 0]['churn'].mean()

    print(f"\n观察数据：")
    print(f"  收到优惠券的客户流失率: {churn_given_coupon:.3f}")
    print(f"  未收到优惠券的客户流失率: {churn_given_no_coupon:.3f}")
    print(f"  差异: {churn_given_no_coupon - churn_given_coupon:.3f}")

    print("\n这是 P(churn|coupon) - 关联层面的观察")
    print("陷阱：相关 != 因果")
    print("可能的解释：")
    print("  1. 优惠券降低了流失率（因果）")
    print("  2. 高价值客户收到优惠券，且高价值客户本身流失率低（混杂）")

    return {
        'P(churn|coupon)': churn_given_coupon,
        'P(churn|no coupon)': churn_given_no_coupon,
        'association': churn_given_no_coupon - churn_given_coupon
    }


# ============================================================================
# 第2层：干预（Intervention）——因果推断要回答的问题
# ============================================================================

def demonstrate_intervention(n_samples: int = 10000, random_state: int = 42) -> dict:
    """
    第2层：干预

    问题是：如果做 X 会怎样？
    方法：随机对照试验（RCT）、因果推断方法（DID、IV、PSM）
    关键：需要 P(y|do(x)) —— "如果主动做 X，Y 会怎样"

    示例：随机发放优惠券，比较实验组和对照组的流失率
    在 RCT 中，关联 = 因果（因为随机化切断了混杂路径）
    """
    print("\n" + "=" * 60)
    print("第2层：干预（Intervention）")
    print("=" * 60)

    rng = np.random.default_rng(random_state)

    # 模拟 RCT：随机分配优惠券
    coupon = rng.binomial(1, 0.5, n_samples)

    # 真实因果效应：优惠券降低流失率 5 个百分点
    # 潜在结果框架
    churn_potential_0 = rng.binomial(1, 0.30, n_samples)  # 不发优惠券时的流失率
    churn_potential_1 = rng.binomial(1, 0.25, n_samples)  # 发优惠券时的流失率

    # 实际观察到的结果（取决于实际接受的 treatment）
    churn = np.where(coupon == 1, churn_potential_1, churn_potential_0)

    df_rct = pd.DataFrame({
        'coupon': coupon,
        'churn': churn
    })

    # 计算因果效应
    churn_treated = df_rct[df_rct['coupon'] == 1]['churn'].mean()
    churn_control = df_rct[df_rct['coupon'] == 0]['churn'].mean()
    causal_effect = churn_control - churn_treated

    print(f"\n随机对照试验（RCT）：")
    print(f"  实验组（发优惠券）流失率: {churn_treated:.3f}")
    print(f"  对照组（不发优惠券）流失率: {churn_control:.3f}")
    print(f"  因果效应（ATE）: {causal_effect:.3f}")

    print("\n这是 P(churn|do(coupon)) - 干预层面的因果效应")
    print("为什么 RCT 有效？")
    print("  随机化确保实验组和对照组在所有变量上平衡")
    print("  因此，两组的差异只能是 treatment（优惠券）导致的")
    print("  关联 = 因果（在 RCT 中）")

    return {
        'P(churn|do(coupon)=1)': churn_treated,
        'P(churn|do(coupon)=0)': churn_control,
        'causal_effect': causal_effect
    }


# ============================================================================
# 第3层：反事实（Counterfactual）——因果推断的最高层级
# ============================================================================

def demonstrate_counterfactual() -> dict:
    """
    第3层：反事实

    问题是：如果当时没做 X 会怎样？
    方法：结构因果模型（SCM）
    关键：需要 P(y_x|x',y') —— "给定实际观察到的 X 和 Y，如果 X 取不同值，Y 会是什么"

    示例：对某个具体客户，计算"如果当时没发优惠券，他的流失概率会是多少"
    这需要估计个体的潜在结果，比第2层更难

    注意：本例只是概念演示，实际反事实推断需要更复杂的模型
    """
    print("\n" + "=" * 60)
    print("第3层：反事实（Counterfactual）")
    print("=" * 60)

    print("\n问题：如果当时没做 X 会怎样？")
    print("示例：客户 A 收到了优惠券且没有流失。")
    print("反事实问题：如果当时没发优惠券，客户 A 会流失吗？")

    print("\n这是 P(y_x|x',y') - 反事实层面的推理")
    print("需要：结构因果模型（SCM）")
    print("难度：需要估计个体层面的潜在结果，且无法完全验证")

    print("\n反事实推理的应用：")
    print("  - 医学：如果当时没给这个病人用药，他会怎样？")
    print("  - 政策：如果当时没实施这个政策，经济会怎样？")
    print("  - 营销：如果当时没发这个优惠券，客户会购买吗？")

    return {
        'note': '反事实推断需要结构因果模型（SCM），本例只是概念演示'
    }


# ============================================================================
# 坏例子：混淆关联和因果
# ============================================================================

def bad_example_confounding(n_samples: int = 10000, random_state: int = 42) -> dict:
    """
    坏例子：把关联当成因果

    场景：高价值客户（unobserved）更容易收到优惠券，且本身流失率低
    结果：观察数据中，优惠券和流失率负相关，但这不是因果效应
    """
    print("\n" + "=" * 60)
    print("坏例子：把关联当成因果")
    print("=" * 60)

    rng = np.random.default_rng(random_state)

    # 未观测的混杂变量：高价值客户
    high_value = rng.binomial(1, 0.3, n_samples)  # 30% 是高价值客户

    # 高价值客户更容易收到优惠券
    coupon_prob = np.where(high_value == 1, 0.8, 0.2)
    coupon = rng.binomial(1, coupon_prob)

    # 流失率由高价值决定（优惠券没有因果效应）
    # 高价值客户流失率 10%，低价值客户流失率 30%
    churn_prob = np.where(high_value == 1, 0.10, 0.30)
    churn = rng.binomial(1, churn_prob)

    df_biased = pd.DataFrame({
        'coupon': coupon,
        'churn': churn
        # high_value 是未观测的
    })

    # 计算观察到的关联
    churn_given_coupon = df_biased[df_biased['coupon'] == 1]['churn'].mean()
    churn_given_no_coupon = df_biased[df_biased['coupon'] == 0]['churn'].mean()

    print(f"\n观察数据（存在未观测混杂）：")
    print(f"  收到优惠券的客户流失率: {churn_given_coupon:.3f}")
    print(f"  未收到优惠券的客户流失率: {churn_given_no_coupon:.3f}")
    print(f"  观察到的差异: {churn_given_no_coupon - churn_given_coupon:.3f}")

    print("\n错误结论：优惠券降低了流失率")
    print("真相：优惠券没有因果效应，只是高价值客户更容易收到优惠券")

    print("\n这个例子说明：")
    print("  - 观察数据中的关联可能是混杂导致的")
    print("  - 需要因果识别策略（如 RCT）才能估计因果效应")

    return {
        'observed_effect': churn_given_no_coupon - churn_given_coupon,
        'true_effect': 0.0,  # 优惠券没有因果效应
        'bias': churn_given_no_coupon - churn_given_coupon
    }


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("因果推断三层级演示")
    print("=" * 60)

    # 第1层：关联
    # 生成有混杂的数据
    rng = np.random.default_rng(42)
    n = 10000

    high_value = rng.binomial(1, 0.3, n)
    coupon_prob = np.where(high_value == 1, 0.8, 0.2)
    coupon = rng.binomial(1, coupon_prob)
    churn_prob = np.where(high_value == 1, 0.10, 0.30)
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame({
        'coupon': coupon,
        'churn': churn
    })

    association_results = demonstrate_association(df)

    # 第2层：干预
    intervention_results = demonstrate_intervention()

    # 第3层：反事实
    counterfactual_results = demonstrate_counterfactual()

    # 坏例子
    bad_results = bad_example_confounding()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    print("\n因果推断的三层级：")
    print("  第1层（关联）：P(y|x) - 看到相关")
    print("  第2层（干预）：P(y|do(x)) - 如果做 X 会怎样")
    print("  第3层（反事实）：P(y_x|x',y') - 如果当时没做 X 会怎样")

    print("\n你的模型能做什么？")
    print("  - 监督学习（逻辑回归、随机森林）：第1层 - 预测和关联")
    print("  - 因果推断（RCT、DID、IV）：第2层 - 干预效应")
    print("  - 结构因果模型（SCM）：第3层 - 反事实推理")

    print("\n关键教训：")
    print("  - 不要把回归系数直接当成因果效应")
    print("  - 相关 != 因果")
    print("  - 需要因果识别策略（RCT 或观察研究方法）")


if __name__ == "__main__":
    main()
