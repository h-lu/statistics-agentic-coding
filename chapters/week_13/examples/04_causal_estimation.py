"""
示例：因果效应估计——从识别到数值

本例演示两种常用的因果效应估计方法：
1. 带后门调整集的回归（简单快速）
2. 倾向评分匹配（Propensity Score Matching，灵活可检查）

运行方式：python3 chapters/week_13/examples/04_causal_estimation.py
预期输出：
- stdout 输出两种方法的估计结果
- 保存倾向评分匹配可视化图（psm_comparison.png）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_coupon_data(n=1000, seed=42):
    """
    生成优惠券模拟数据

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


def method_1_regression_adjustment(df):
    """
    方法 1：带后门调整集的回归

    最简单的方法——用回归控制混杂变量
    """
    print("\n" + "=" * 70)
    print("📊 方法 1：带后门调整集的回归")
    print("=" * 70)

    print("\n原理:")
    print("  在回归方程中包含混杂变量，")
    print("  处理变量的系数就是调整后的因果效应")

    print("\n回归方程:")
    print("  消费金额 = β0 + β1×优惠券 + β2×活跃度 + β3×历史消费 + ε")

    # 拟合模型
    X = df[['优惠券使用', '用户活跃度', '历史消费']]
    y = df['消费金额']

    model = LinearRegression()
    model.fit(X, y)

    # 提取结果
    coef_coupon = model.coef_[0]
    coef_activity = model.coef_[1]
    coef_history = model.coef_[2]
    intercept = model.intercept_

    # 计算标准误差（简化版）
    from scipy import stats
    n = len(df)
    k = 3  # 自变量数量
    y_pred = model.predict(X)
    residuals = y - y_pred
    mse = np.sum(residuals**2) / (n - k - 1)

    # 系数的协方差矩阵（简化）
    X_with_intercept = np.column_stack([np.ones(n), X.values])
    cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    se_coupon = np.sqrt(cov_matrix[1, 1])

    # t 检验
    t_stat = coef_coupon / se_coupon
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 1))

    # 95% 置信区间
    ci_low = coef_coupon - 1.96 * se_coupon
    ci_high = coef_coupon + 1.96 * se_coupon

    print("\n结果:")
    print("-" * 70)
    print(f"截距: {intercept:.2f}")
    print(f"优惠券系数: {coef_coupon:.2f} 元 (SE: {se_coupon:.2f})")
    print(f"活跃度系数: {coef_activity:.2f}")
    print(f"历史消费系数: {coef_history:.2f}")
    print(f"\n因果效应估计:")
    print(f"  优惠券 → 消费金额: {coef_coupon:.2f} 元")
    print(f"  95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"  t 值: {t_stat:.2f}")
    print(f"  p 值: {p_value:.4f}")

    # 对比真实值
    true_effect = 30
    print(f"\n对比真实值:")
    print(f"  真实效应: {true_effect:.2f} 元")
    print(f"  估计误差: {abs(coef_coupon - true_effect):.2f} 元")

    return {
        'method': '回归（带调整集）',
        'effect': coef_coupon,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': p_value
    }


def method_2_psm(df):
    """
    方法 2：倾向评分匹配（Propensity Score Matching）

    核心思想：如果两个用户倾向评分相近（特征相似），但一个用券、一个不用，
    那他们的差异就是因果效应。
    """
    print("\n" + "=" * 70)
    print("🎯 方法 2：倾向评分匹配（PSM）")
    print("=" * 70)

    print("\n原理:")
    print("  1. 估计倾向评分：P(用券 | 活跃度, 历史消费)")
    print("  2. 为每个用券用户找未用券的'相似用户'（1:1 匹配）")
    print("  3. 计算匹配后的消费差异（ATT）")

    # ========== 第 1 步：估计倾向评分 ==========
    print("\n第 1 步：估计倾向评分...")
    print("-" * 70)

    confounders = ['用户活跃度', '历史消费']
    treatment = '优惠券使用'

    ps_model = LogisticRegression(random_state=42)
    ps_model.fit(df[confounders], df[treatment])

    df['propensity_score'] = ps_model.predict_proba(df[confounders])[:, 1]

    print(f"倾向评分模型:")
    print(f"  特征: {confounders}")
    print(f"  算法: Logistic Regression")
    print(f"  平均倾向评分: {df['propensity_score'].mean():.3f}")

    # ========== 第 2 步：匹配 ==========
    print("\n第 2 步：匹配（1:1 最近邻）...")
    print("-" * 70)

    treated = df[df[treatment] == 1].copy()
    control = df[df[treatment] == 0].copy()

    print(f"处理组（用券）: {len(treated)} 人")
    print(f"对照组（未用券）: {len(control)} 人")

    # 1:1 最近邻匹配
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])

    distances, indices = nn.kneighbors(treated[['propensity_score']])
    matched_control = control.iloc[indices.flatten()].copy()

    # 计算匹配质量
    mean_distance = distances.mean()

    print(f"\n匹配质量:")
    print(f"  平均倾向评分距离: {mean_distance:.4f}")
    print(f"  匹配成功: {len(matched_control)} 对")

    # ========== 第 3 步：计算 ATT ==========
    print("\n第 3 步：计算 ATT（处理组平均处理效应）...")
    print("-" * 70)

    treated_outcome = treated['消费金额'].values
    control_outcome = matched_control['消费金额'].values

    att = (treated_outcome - control_outcome).mean()

    # Bootstrap 置信区间
    print("\nBootstrap 95% CI (500 次重采样)...")

    n_boot = 500
    att_samples = []

    for i in range(n_boot):
        # 重采样
        treated_boot = treated.sample(n=len(treated), replace=True)
        control_boot = control.sample(n=len(control), replace=True)

        # 重新匹配
        nn_boot = NearestNeighbors(n_neighbors=1)
        nn_boot.fit(control_boot[['propensity_score']])
        _, indices_boot = nn_boot.kneighbors(treated_boot[['propensity_score']])
        matched_boot = control_boot.iloc[indices_boot.flatten()]

        # 计算 ATT
        att_boot = (treated_boot['消费金额'].values - matched_boot['消费金额'].values).mean()
        att_samples.append(att_boot)

    att_ci_low = np.percentile(att_samples, 2.5)
    att_ci_high = np.percentile(att_samples, 97.5)

    print(f"\n因果效应估计:")
    print(f"  ATT（处理组平均处理效应）: {att:.2f} 元")
    print(f"  95% CI (Bootstrap): [{att_ci_low:.2f}, {att_ci_high:.2f}]")

    # 对比真实值
    true_effect = 30
    print(f"\n对比真实值:")
    print(f"  真实效应: {true_effect:.2f} 元")
    print(f"  估计误差: {abs(att - true_effect):.2f} 元")

    # ========== 第 4 步：可视化 ==========
    print("\n第 4 步：保存匹配前后可视化...")

    plot_psm_comparison(
        treated['propensity_score'].values,
        control['propensity_score'].values,
        matched_control['propensity_score'].values,
        output_path='psm_comparison.png'
    )

    return {
        'method': '倾向评分匹配（PSM）',
        'effect': att,
        'ci_low': att_ci_low,
        'ci_high': att_ci_high,
        'p_value': None  # PSM 不直接给出 p 值
    }


def plot_psm_comparison(treated_ps, control_ps, matched_ps, output_path='psm_comparison.png'):
    """
    画倾向评分匹配前后的对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 匹配前
    axes[0].hist(treated_ps, alpha=0.5, label='用券（处理组）', bins=20, color='blue')
    axes[0].hist(control_ps, alpha=0.5, label='未用券（对照组）', bins=20, color='red')
    axes[0].set_xlabel('倾向评分', fontsize=12)
    axes[0].set_ylabel('人数', fontsize=12)
    axes[0].set_title('匹配前：倾向评分分布差异大', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # 匹配后
    axes[1].hist(treated_ps, alpha=0.5, label='用券（处理组）', bins=20, color='blue')
    axes[1].hist(matched_ps, alpha=0.5, label='匹配的未用券（对照组）', bins=20, color='green')
    axes[1].set_xlabel('倾向评分', fontsize=12)
    axes[1].set_ylabel('人数', fontsize=12)
    axes[1].set_title('匹配后：倾向评分分布接近', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ 图已保存: {output_path}")


def compare_methods(result_reg, result_psm):
    """
    对比两种方法的结果
    """
    print("\n" + "=" * 70)
    print("📋 两种方法的对比")
    print("=" * 70)

    print(f"\n{'方法':<20} {'估计值':<12} {'95% CI':<20} {'结论'}")
    print("-" * 70)

    print(f"{result_reg['method']:<20} "
          f"{result_reg['effect']:>8.2f} 元  "
          f"[{result_reg['ci_low']:.2f}, {result_reg['ci_high']:.2f}]  "
          f"{'显著' if result_reg['p_value'] < 0.05 else '不显著'}")

    print(f"{result_psm['method']:<20} "
          f"{result_psm['effect']:8.2f} 元  "
          f"[{result_psm['ci_low']:.2f}, {result_psm['ci_high']:.2f}]  "
          f"Bootstrap")

    # 一致性检查
    effect_diff = abs(result_reg['effect'] - result_psm['effect'])
    ci_overlap = not (result_reg['ci_high'] < result_psm['ci_low'] or
                     result_psm['ci_high'] < result_reg['ci_low'])

    print("\n一致性评估:")
    print(f"  估计值差异: {effect_diff:.2f} 元")
    print(f"  置信区间重叠: {'✅ 是' if ci_overlap else '❌ 否'}")

    if effect_diff < 5 and ci_overlap:
        print(f"\n✅ 两种方法结果接近，结论稳健！")
    else:
        print(f"\n⚠️  两种方法差异较大，需要检查假设")

    # 实践建议
    print("\n实践建议:")
    print("-" * 70)
    print("1. 先用回归（快速得到基线）")
    print("2. 再用匹配（检查稳健性）")
    print("3. 如果两者接近，结论可靠")
    print("4. 如果差异大，检查假设（模型形式、匹配质量）")


def bad_example_naive_comparison(df):
    """
    反例：简单均值比较（小北的错误）
    """
    print("\n" + "=" * 70)
    print("❌ 反例：小北的错误——简单均值比较")
    print("=" * 70)

    print("\n小北的做法:")
    print("  直接比较用券和未用券用户的平均消费")

    treated_mean = df[df['优惠券使用'] == 1]['消费金额'].mean()
    control_mean = df[df['优惠券使用'] == 0]['消费金额'].mean()
    naive_effect = treated_mean - control_mean

    print(f"\n结果:")
    print(f"  用券用户平均消费: {treated_mean:.2f} 元")
    print(f"  未用券用户平均消费: {control_mean:.2f} 元")
    print(f"  差异: {naive_effect:.2f} 元")

    print(f"\n⚠️ 问题:")
    print(f"  真实因果效应: 30.00 元")
    print(f"  小北的估计: {naive_effect:.2f} 元")
    print(f"  混杂偏差: {naive_effect - 30:.2f} 元")

    print(f"\n原因:")
    print(f"  活跃用户既更可能用券，也消费更高")
    print(f"  不调整活跃度，会把活跃度的效应归功于优惠券")


def main():
    """主函数"""
    print("=" * 70)
    print("因果效应估计：回归 vs 倾向评分匹配")
    print("=" * 70)

    # 创建输出目录
    output_dir = Path("report")
    output_dir.mkdir(exist_ok=True)

    # 生成数据
    print("\n📊 生成模拟数据...")
    df = generate_coupon_data(n=1000, seed=42)

    print(f"数据规模: {len(df)} 用户")
    print(f"用券比例: {df['优惠券使用'].mean():.1%}")
    print(f"平均消费: {df['消费金额'].mean():.2f} 元")

    # 反例：小北的错误
    bad_example_naive_comparison(df)

    # 方法 1：回归
    result_reg = method_1_regression_adjustment(df)

    # 方法 2：倾向评分匹配
    result_psm = method_2_psm(df)

    # 对比
    compare_methods(result_reg, result_psm)

    print("\n" + "=" * 70)
    print("💡 关键要点")
    print("=" * 70)
    print("""
1. 回归（带调整集）:
   - 简单、快速、可用标准误差
   - 假设线性、容易模型错设
   - 适用于混杂变量少、关系简单的场景

2. 倾向评分匹配:
   - 不假设线性、可视化强、直觉清晰
   - 丢弃无法匹配的样本、效率低
   - 适用于非线性关系、需要可比性检查的场景

3. 实践建议:
   - 先用回归（快速得到基线）
   - 再用匹配（检查稳健性）
   - 如果两者接近，结论可靠

4. 小北的错误:
   - 直接比较均值（未调整混杂）
   - 结果被夸大（50 vs 30 元）
   - 正确做法：先画因果图，用后门准则选择调整集
    """)


if __name__ == "__main__":
    main()
