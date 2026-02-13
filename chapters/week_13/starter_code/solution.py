"""
Week 13 作业参考解答

本文件提供 Week 13 作业的基础部分参考实现。
学生应在完成作业后参考此文件，而不是直接复制。

作业要求详见 chapters/week_13/ASSIGNMENT.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy import stats

# 设置输出目录
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_13"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_coupon_data(n=1000, seed=42):
    """
    练习 1：生成优惠券模拟数据

    真实因果效应：30 元

    结构:
      活跃度 → 优惠券 → 消费
           ↘          ↗
           历史消费
    """
    np.random.seed(seed)

    # 混杂变量
    activity = np.random.normal(50, 15, n)
    history_spend = np.random.normal(100, 30, n)

    # 处理变量（受混杂影响）
    coupon_prob = 0.2 + 0.006 * activity + 0.002 * history_spend
    coupon = np.random.binomial(1, np.clip(coupon_prob, 0, 1))

    # 结果变量（受混杂和处理影响）
    # 真实因果效应 = 30 元
    spending = (
        50 +                      # 基础消费
        1.5 * activity +         # 活跃度影响
        0.3 * history_spend +    # 历史消费影响
        30 * coupon +             # 优惠券因果效应
        np.random.normal(0, 15, n)  # 噪声
    )

    df = pd.DataFrame({
        '用户活跃度': activity,
        '历史消费': history_spend,
        '优惠券使用': coupon,
        '消费金额': spending
    })

    return df


def exercise_1_association_vs_causation():
    """
    练习 1：关联 vs 因果

    要求：
    1. 计算用券和未用券用户的平均消费差异（关联）
    2. 用带调整集的回归估计因果效应
    3. 比较两者差异，解释混杂偏差
    """
    print("=" * 70)
    print("练习 1：关联 vs 因果")
    print("=" * 70)

    # 生成数据
    df = generate_coupon_data(n=1000, seed=42)

    # 1. 未调整的关联（小北的错误）
    treated_mean = df[df['优惠券使用'] == 1]['消费金额'].mean()
    control_mean = df[df['优惠券使用'] == 0]['消费金额'].mean()
    naive_diff = treated_mean - control_mean

    print(f"\n未调整的估计（关联）:")
    print(f"  用券用户平均消费: {treated_mean:.2f} 元")
    print(f"  未用券用户平均消费: {control_mean:.2f} 元")
    print(f"  差异: {naive_diff:.2f} 元")

    # 2. 调整后的因果效应
    X = df[['优惠券使用', '用户活跃度', '历史消费']]
    y = df['消费金额']

    model = LinearRegression()
    model.fit(X, y)

    causal_effect = model.coef_[0]

    print(f"\n调整后的估计（因果）:")
    print(f"  回归系数: {causal_effect:.2f} 元")

    # 3. 解释差异
    bias = naive_diff - causal_effect

    print(f"\n混杂偏差:")
    print(f"  被夸大: {bias:.2f} 元")
    print(f"  真实效应: 30.00 元")
    print(f"  关联估计误差: {abs(naive_diff - 30):.2f} 元")
    print(f"  因果估计误差: {abs(causal_effect - 30):.2f} 元")

    print(f"\n结论:")
    print(f"  活跃用户既更可能用券，也消费更高")
    print(f"  不调整活跃度，会把活跃度的效应归功于优惠券")

    return {
        'naive_effect': naive_diff,
        'causal_effect': causal_effect,
        'bias': bias
    }


def exercise_2_draw_cag():
    """
    练习 2：画因果图

    要求：
    1. 用 NetworkX 画优惠券案例的因果图
    2. 标注处理变量、结果变量、混杂变量
    3. 保存为 causal_dag.png
    """
    print("\n" + "=" * 70)
    print("练习 2：画因果图")
    print("=" * 70)

    try:
        import networkx as nx

        # 创建 DAG
        G = nx.DiGraph()
        G.add_edges_from([
            ("用户活跃度", "优惠券使用"),
            ("用户活跃度", "消费金额"),
            ("历史消费", "优惠券使用"),
            ("历史消费", "消费金额"),
            ("优惠券使用", "消费金额")
        ])

        # 布局
        pos = {
            "用户活跃度": (0, 2),
            "历史消费": (0, 0),
            "优惠券使用": (1, 1),
            "消费金额": (2, 1)
        }

        # 画图
        plt.figure(figsize=(10, 6))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=3000, alpha=0.9)
        nx.draw_networkx_edges(G, pos, edge_color='gray',
                              arrowsize=20, width=2, alpha=0.7)
        nx.draw_networkx_labels(G, pos, font_size=12,
                                font_family='sans-serif')
        plt.title("优惠券案例的因果图（DAG）", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'causal_dag.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✅ 因果图已保存: causal_dag.png")

        # 解释结构
        print(f"\n图解:")
        print(f"  - 处理变量（X）: 优惠券使用")
        print(f"  - 结果变量（Y）: 消费金额")
        print(f"  - 混杂变量（Z）: 用户活跃度、历史消费")

        return True

    except ImportError:
        print(f"\n⚠️  NetworkX 未安装")
        print(f"  安装方法: pip install networkx")
        return False


def exercise_3_backdoor_criterion():
    """
    练习 3：后门准则

    要求：
    1. 识别优惠券案例的后门路径
    2. 说明为什么需要调整用户活跃度和历史消费
    3. 解释为什么不调整使用频率（如果有的话）
    """
    print("\n" + "=" * 70)
    print("练习 3：后门准则")
    print("=" * 70)

    print(f"\n后门路径分析:")
    print(f"  因果路径: 优惠券 → 消费金额（保留）")
    print(f"  后门路径 1: 优惠券 ← 活跃度 → 消费金额（需阻断）")
    print(f"  后门路径 2: 优惠券 ← 历史消费 → 消费金额（需阻断）")

    print(f"\n调整策略:")
    print(f"  ✅ 必须调整: 用户活跃度、历史消费")
    print(f"     原因: 阻断后门路径，消除混杂")
    print(f"  ❌ 不能调整: 使用频率（中介变量）")
    print(f"     原因: 会切断因果路径，低估真实效应")

    print(f"\n后门准则三条规则:")
    print(f"  1. 调整集中不包含处理变量的后代")
    print(f"  2. 调整集阻断所有后门路径")
    print(f"  3. 调整集不打开新的虚假路径")

    return {
        'backdoor_paths': [
            '优惠券 ← 活跃度 → 消费金额',
            '优惠券 ← 历史消费 → 消费金额'
        ],
        'adjustment_set': ['用户活跃度', '历史消费']
    }


def exercise_4_psm():
    """
    练习 4：倾向评分匹配

    要求：
    1. 用 Logistic 回归估计倾向评分
    2. 用 1:1 最近邻匹配
    3. 计算 ATT 和 95% CI
    """
    print("\n" + "=" * 70)
    print("练习 4：倾向评分匹配")
    print("=" * 70)

    # 生成数据
    df = generate_coupon_data(n=1000, seed=42)

    # 1. 估计倾向评分
    confounders = ['用户活跃度', '历史消费']
    treatment = '优惠券使用'

    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(df[confounders], df[treatment])

    df['propensity_score'] = ps_model.predict_proba(df[confounders])[:, 1]

    print(f"\n倾向评分估计:")
    print(f"  平均倾向评分: {df['propensity_score'].mean():.3f}")

    # 2. 匹配
    treated = df[df[treatment] == 1].copy()
    control = df[df[treatment] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    matched_control = control.iloc[indices.flatten()].copy()

    # 3. 计算 ATT
    att = (treated['消费金额'].values - matched_control['消费金额'].values).mean()

    # Bootstrap CI
    n_boot = 500
    att_samples = []

    for i in range(n_boot):
        treated_boot = treated.sample(n=len(treated), replace=True)
        control_boot = control.sample(n=len(control), replace=True)

        nn_boot = NearestNeighbors(n_neighbors=1)
        nn_boot.fit(control_boot[['propensity_score']])
        _, indices_boot = nn_boot.kneighbors(treated_boot[['propensity_score']])
        matched_boot = control_boot.iloc[indices_boot.flatten()]

        att_boot = (treated_boot['消费金额'].values - matched_boot['消费金额'].values).mean()
        att_samples.append(att_boot)

    att_ci_low = np.percentile(att_samples, 2.5)
    att_ci_high = np.percentile(att_samples, 97.5)

    print(f"\n倾向评分匹配结果:")
    print(f"  ATT: {att:.2f} 元")
    print(f"  95% CI (Bootstrap): [{att_ci_low:.2f}, {att_ci_high:.2f}]")
    print(f"  真实效应: 30.00 元")
    print(f"  误差: {abs(att - 30):.2f} 元")

    return {
        'att': att,
        'ci_low': att_ci_low,
        'ci_high': att_ci_high
    }


def exercise_5_interpret_results():
    """
    练习 5：结果解释

    要求：
    1. 解释关联 vs 因果的差异
    2. 说明因果结论的边界
    3. 列出分析的局限性
    """
    print("\n" + "=" * 70)
    print("练习 5：结果解释")
    print("=" * 70)

    print(f"\n关联 vs 因果:")
    print(f"  关联问题: 用券用户和未用券用户的消费差异是多少？")
    print(f"  因果问题: 如果给用户发券，他的消费会提高多少？")
    print(f"  关联估计: ~50 元（被混杂夸大）")
    print(f"  因果估计: ~30 元（真实效应）")

    print(f"\n因果结论边界:")
    print(f"  ✅ 能回答的:")
    print(f"     - 优惠券的平均因果效应约为 30 元")
    print(f"     - 在调整了混杂变量后，效应统计显著")
    print(f"  ❌ 不能回答的:")
    print(f"     - 个体因果效应（反事实）")
    print(f"     - 长期效应（数据范围外）")
    print(f"     - 效应异质性（不同人群的效应）")

    print(f"\n局限性:")
    print(f"  - 存在未观察混杂的可能（如用户收入）")
    print(f"  - 回归假设线性关系")
    print(f"  - 匹配会丢弃无法匹配的样本")
    print(f"  - 真实数据可能有更复杂的因果结构")


def main():
    """主函数：运行所有练习"""
    print("=" * 70)
    print("Week 13 作业参考解答")
    print("=" * 70)
    print("\n注意：本文件仅供参考，请在完成作业后查看")
    print("直接复制无法达到学习效果\n")

    # 运行练习 1
    result_1 = exercise_1_association_vs_causation()

    # 运行练习 2
    result_2 = exercise_2_draw_cag()

    # 运行练习 3
    result_3 = exercise_3_backdoor_criterion()

    # 运行练习 4
    result_4 = exercise_4_psm()

    # 运行练习 5
    exercise_5_interpret_results()

    print("\n" + "=" * 70)
    print("✅ 所有练习完成")
    print("=" * 70)
    print("""
进一步学习建议:
1. 尝试不同的数据生成参数（如更强的混杂）
2. 对比回归和倾向评分匹配的结果
3. 思考真实数据中的因果问题
4. 学习 DoWhy 库的自动化工具
5. 阅读《因果推断》教材（如 Hernán & Robins）

下周预告: 贝叶斯统计
- 从"频率学派"到"贝叶斯学派"
- 先验分布、似然函数、后验分布
- 贝叶斯因果推断
    """)


if __name__ == "__main__":
    main()
