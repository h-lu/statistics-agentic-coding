"""
示例：后门准则（Backdoor Criterion）——该调整什么，不该调整什么

本例演示如何使用 DoWhy 库自动识别后门路径和调整集。

后门准则的三条规则：
1. 调整集中不包含处理变量的后代（避免调整中介变量）
2. 调整集阻断所有后门路径（消除混杂）
3. 调整集不打开新的虚假路径（避免调整对撞变量）

运行方式：python3 chapters/week_13/examples/03_backdoor_criterion.py
预期输出：
- DoWhy 自动识别的调整集
- 后门路径分析
- 如果没有 DoWhy，用纯 Python 演示
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def try_dowhy_demo():
    """
    尝试使用 DoWhy 演示后门准则

    如果 DoWhy 未安装，返回 False
    """
    try:
        from dowhy import CausalModel
        return True
    except ImportError:
        return False


def demonstrate_with_dowhy():
    """
    使用 DoWhy 演示后门准则的自动识别
    """
    print("=" * 70)
    print("DoWhy: 因果推断自动化工具")
    print("=" * 70)

    from dowhy import CausalModel

    # 生成模拟数据
    np.random.seed(42)
    n = 1000

    # 混杂变量
    activity = np.random.normal(50, 15, n)
    history_spend = np.random.normal(100, 30, n)

    # 处理变量（受混杂影响）
    coupon_prob = 0.2 + 0.006 * activity + 0.002 * history_spend
    coupon = np.random.binomial(1, np.clip(coupon_prob, 0, 1))

    # 结果变量（受混杂和处理影响）
    spending = 50 + 1.5 * activity + 0.3 * history_spend + 30 * coupon + np.random.normal(0, 15, n)

    df = pd.DataFrame({
        '用户活跃度': activity,
        '历史消费': history_spend,
        '优惠券使用': coupon,
        '消费金额': spending
    })

    print("\n📊 数据概览:")
    print(df.head())

    # 定义因果图（DOT 格式）
    causal_graph = """digraph {
        用户活跃度 -> 优惠券使用;
        用户活跃度 -> 消费金额;
        历史消费 -> 优惠券使用;
        历史消费 -> 消费金额;
        优惠券使用 -> 消费金额;
    }"""

    print("\n📈 因果图（DAG）:")
    print(causal_graph)

    # 创建因果模型
    print("\n🔧 创建因果模型...")
    model = CausalModel(
        data=df,
        treatment="优惠券使用",
        outcome="消费金额",
        graph=causal_graph.replace('\n', ' ')
    )

    # 识别因果效应（自动应用后门准则）
    print("\n🔍 识别因果效应...")
    identified_estimand = model.identify_effect()

    print("\n✅ 识别结果:")
    print(identified_estimand)

    # 提取后门调整集
    print("\n📋 DoWhy 自动识别的调整集:")
    if hasattr(identified_estimand, 'backdoor_variables'):
        backdoor_vars = identified_estimand.backdoor_variables
        if backdoor_vars:
            print(f"  调整变量: {backdoor_vars}")
        else:
            print("  无需调整（随机对照试验）")
    else:
        # 从 estimand 表达式中提取
        estimand_str = str(identified_estimand)
        if 'backdoor' in estimand_str.lower():
            print("  需要调整后门变量（见上方表达式）")

    print("\n💡 解读:")
    print("  - DoWhy 根据因果图自动应用后门准则")
    print("  - 它识别出需要调整'用户活跃度'和'历史消费'")
    print("  - 这些变量同时影响处理（优惠券）和结果（消费）")

    # 估计因果效应
    print("\n📊 估计因果效应...")

    # 方法 1: 基于回归的估计
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )

    print(f"\n✅ 因果效应估计:")
    print(f"  方法: 线性回归（带后门调整）")
    print(f"  估计值: {estimate.value:.2f} 元")

    # 真实值是 30 元（我们生成的数据）
    print(f"  真实值: 30.00 元（数据生成时设定的）")
    print(f"  误差: {abs(estimate.value - 30):.2f} 元")

    # 鲁棒性检查（敏感性分析）
    print("\n🛡️  鲁棒性检查（随机原因模型）...")
    refutation = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter"
    )

    print(f"\n  新效应（安慰剂处理）: {refutation.new_effect:.2f} 元")
    print(f"  原效应: {refutation.estimated_effect:.2f} 元")
    print(f"  解读: {'✅ 通过' if abs(refutation.new_effect) < abs(refutation.estimated_effect) else '❌ 未通过'}")

    return True


def demonstrate_manual_backdoor():
    """
    手动演示后门准则（不依赖 DoWhy）
    """
    print("\n" + "=" * 70)
    print("后门准则：手动演示")
    print("=" * 70)

    print("\n📚 后门准则的定义:")
    print("-" * 70)
    print("给定因果图 G 和处理变量 X、结果变量 Y，")
    print("调整集 Z 满足后门准则，如果：")
    print("")
    print("  1. Z 中没有 X 的后代（不调整中介变量）")
    print("  2. Z 阻断了所有 X → Y 的后门路径")
    print("  3. Z 不打开任何新的虚假路径（不调整对撞变量）")

    # 优惠券案例
    print("\n" + "=" * 70)
    print("💼 优惠券案例：后门路径分析")
    print("=" * 70)

    print("\n因果图:")
    print("""
  用户活跃度 → 优惠券使用
      ↓           ↓
      消费金额 ←────┘
  历史消费 → ↑
    """)

    print("\n路径分析:")
    print("-" * 70)

    paths = {
        "因果路径": {
            "path": "优惠券 → 消费金额",
            "type": "因果路径",
            "action": "保留（这是我们想估计的）",
            "correct": True
        },
        "后门路径 1": {
            "path": "优惠券 ← 活跃度 → 消费金额",
            "type": "后门路径（虚假关联）",
            "action": "阻断：调整'活跃度'",
            "correct": True
        },
        "后门路径 2": {
            "path": "优惠券 ← 历史消费 → 消费金额",
            "type": "后门路径（虚假关联）",
            "action": "阻断：调整'历史消费'",
            "correct": True
        }
    }

    for name, info in paths.items():
        print(f"\n{name}:")
        print(f"  路径: {info['path']}")
        print(f"  类型: {info['type']}")
        print(f"  行动: {info['action']}")

    print("\n✅ 正确的调整集:")
    print("  - 调整: 用户活跃度、历史消费（阻断后门路径）")
    print("  - 不调整: 使用频率（中介变量，会切断因果路径）")

    # 演示调整效果
    print("\n" + "=" * 70)
    print("📊 调整效果对比")
    print("=" * 70)

    # 生成数据
    np.random.seed(42)
    n = 1000

    activity = np.random.normal(50, 15, n)
    history_spend = np.random.normal(100, 30, n)
    coupon_prob = 0.2 + 0.006 * activity + 0.002 * history_spend
    coupon = np.random.binomial(1, np.clip(coupon_prob, 0, 1))
    spending = 50 + 1.5 * activity + 0.3 * history_spend + 30 * coupon + np.random.normal(0, 15, n)

    df = pd.DataFrame({
        '用户活跃度': activity,
        '历史消费': history_spend,
        '优惠券使用': coupon,
        '消费金额': spending
    })

    # 未调整的估计
    untreated_mean = df[df['优惠券使用'] == 0]['消费金额'].mean()
    treated_mean = df[df['优惠券使用'] == 1]['消费金额'].mean()
    naive_effect = treated_mean - untreated_mean

    print(f"\n未调整的估计（有偏）:")
    print(f"  用券用户平均消费: {treated_mean:.2f} 元")
    print(f"  未用券用户平均消费: {untreated_mean:.2f} 元")
    print(f"  差异（关联）: {naive_effect:.2f} 元")
    print(f"  ⚠️ 被混杂夸大了: {naive_effect - 30:.2f} 元")

    # 调整后的估计（回归）
    from sklearn.linear_model import LinearRegression

    X = df[['优惠券使用', '用户活跃度', '历史消费']]
    y = df['消费金额']
    model = LinearRegression()
    model.fit(X, y)

    adjusted_effect = model.coef_[0]

    print(f"\n调整后的估计（正确）:")
    print(f"  回归系数: {adjusted_effect:.2f} 元")
    print(f"  真实效应: 30.00 元")
    print(f"  误差: {abs(adjusted_effect - 30):.2f} 元")
    print(f"  ✅ 调整后接近真实值！")


def bad_example_wrong_adjustment():
    """
    反例：错误的调整策略
    """
    print("\n" + "=" * 70)
    print("❌ 反例：常见的调整错误")
    print("=" * 70)

    errors = {
        "错误 1：盲目调整一切": {
            "description": "把所有变量都放进回归",
            "problem": "可能调整了中介变量或对撞变量",
            "consequence": "低估因果效应或制造虚假关联",
            "example": "调整'使用频率'（中介），会低估优惠券效果"
        },
        "错误 2：不调整混杂": {
            "description": "只放处理变量，不调整混杂",
            "problem": "后门路径未阻断",
            "consequence": "关联被夸大（小北的错误）",
            "example": "直接比较用券和未用券用户的消费"
        },
        "错误 3：调整对撞变量": {
            "description": "调整了结果变量的后代",
            "problem": "打开虚假路径（选择偏差）",
            "consequence": "制造不存在的关联",
            "example": "在'录取'模型中调整'面试分数'（对撞）"
        }
    }

    for name, info in errors.items():
        print(f"\n{name}")
        print("-" * 70)
        print(f"描述: {info['description']}")
        print(f"问题: {info['problem']}")
        print(f"后果: {info['consequence']}")
        print(f"示例: {info['example']}")


def main():
    """主函数"""
    print("=" * 70)
    print("后门准则（Backdoor Criterion）演示")
    print("=" * 70)

    # 尝试使用 DoWhy
    dowhy_available = try_dowhy_demo()

    if dowhy_available:
        print("\n✅ DoWhy 已安装，使用自动化工具演示")
        demonstrate_with_dowhy()
    else:
        print("\n⚠️  DoWhy 未安装，使用手动演示")
        print("   安装方法: pip install dowhy")
        print("   文档: https://www.pywhy.org/dowhy/\n")

    # 手动演示（无论如何都运行）
    demonstrate_manual_backdoor()

    # 反例
    bad_example_wrong_adjustment()

    print("\n" + "=" * 70)
    print("💡 关键要点")
    print("=" * 70)
    print("""
1. 后门准则的三条规则:
   - 不调整 X 的后代（避免切路径）
   - 阻断所有后门路径（消除混杂）
   - 不打开新路径（避免对撞）

2. 常见错误:
   - 盲目调整一切（可能调整中介变量）
   - 不调整混杂（后门路径未阻断）
   - 调整对撞变量（打开虚假路径）

3. 实践建议:
   - 先画因果图（明确假设）
   - 用 DoWhy 自动识别调整集
   - 手动验证（看哪些是混杂/中介/对撞）

下一步: 因果效应估计（回归 + 匹配）
    """)


if __name__ == "__main__":
    main()
