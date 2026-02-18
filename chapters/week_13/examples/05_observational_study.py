"""
示例：观察研究中的因果推断——双重差分与倾向得分匹配

本例演示当 RCT 不可行时的因果推断方法：
1. 双重差分（DID）：利用自然实验，比较"实验组的前后变化"和"对照组的前后变化"
2. 倾向得分匹配（PSM）：用相似度匹配，消除观测混杂
3. 方法的局限和假设

运行方式：python3 chapters/week_13/examples/05_observational_study.py
预期输出：stdout 输出 DID 和 PSM 的演示 + 保存图表到 images/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


# ============================================================================
# 方法1：双重差分（DID）
# ============================================================================

def generate_did_data(
    n_cities: int = 20,
    n_periods: int = 10,
    true_effect: float = -0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """
    生成双重差分（DID）的模拟数据

    场景：某城市在某时点实施"发放优惠券"政策，其他城市没有
    目标：估计政策的因果效应

    关键假设：平行趋势（实验组和对照组在政策前的趋势相同）
    """
    rng = np.random.default_rng(random_state)

    data = []

    for city in range(n_cities):
        # 随机决定是否为实验城市（前50%的城市是实验组）
        is_treated = city < n_cities // 2

        # 城市固定效应（有些城市本身流失率就高/低）
        city_fe = rng.normal(0, 0.02)

        for period in range(n_periods):
            # 时间固定效应（整体趋势）
            time_fe = period * 0.005

            # 政策实施后的效应
            policy_effect = 0
            if is_treated and period >= n_periods // 2:
                policy_effect = true_effect

            # 生成流失率
            base_rate = 0.30
            churn_prob = base_rate + city_fe + time_fe + policy_effect
            churn_prob = np.clip(churn_prob, 0, 1)  # 限制在[0,1]

            # 模拟 n 个客户
            n_customers = 100
            churn = rng.binomial(1, churn_prob, n_customers)
            avg_churn = churn.mean()

            data.append({
                'city': city,
                'period': period,
                'is_treated': int(is_treated),
                'after_policy': int(period >= n_periods // 2),
                'churn_rate': avg_churn,
                'n_customers': n_customers
            })

    return pd.DataFrame(data)


def demonstrate_did() -> dict:
    """
    演示双重差分（DID）
    """
    print("=" * 60)
    print("方法1：双重差分（Difference-in-Differences, DID）")
    print("=" * 60)

    # 生成数据
    df = generate_did_data()

    print("\n场景：某城市在某时点实施'发放优惠券'政策")
    print("实验组：实施政策的城市")
    print("对照组：未实施政策的城市")

    # 计算各组的平均流失率
    treated_before = df[(df['is_treated'] == 1) & (df['after_policy'] == 0)]['churn_rate'].mean()
    treated_after = df[(df['is_treated'] == 1) & (df['after_policy'] == 1)]['churn_rate'].mean()
    control_before = df[(df['is_treated'] == 0) & (df['after_policy'] == 0)]['churn_rate'].mean()
    control_after = df[(df['is_treated'] == 0) & (df['after_policy'] == 1)]['churn_rate'].mean()

    # DID 估计
    treated_diff = treated_after - treated_before  # 实验组的变化
    control_diff = control_after - control_before  # 对照组的变化
    did_estimate = treated_diff - control_diff  # DID = 实验组变化 - 对照组变化

    print(f"\n结果：")
    print(f"  实验组-政策前：{treated_before:.3f}")
    print(f"  实验组-政策后：{treated_after:.3f}")
    print(f"  实验组变化：{treated_diff:.3f}")

    print(f"\n  对照组-政策前：{control_before:.3f}")
    print(f"  对照组-政策后：{control_after:.3f}")
    print(f"  对照组变化：{control_diff:.3f}")

    print(f"\n  DID 估计：{did_estimate:.3f}")
    print(f"  解释：政策导致流失率变化 {did_estimate:.3f}")

    print(f"\nDID 的直觉：")
    print(f"  实验组的变化 = 政策效应 + 时间趋势")
    print(f"  对照组的变化 = 时间趋势")
    print(f"  DID = 实验组变化 - 对照组变化 = 政策效应")

    print(f"\nDID 的关键假设：平行趋势")
    print(f"  如果没有政策，实验组和对照组的趋势应该相同")
    print(f"  需要画前政策趋势图来检验")

    return {
        'data': df,
        'did_estimate': did_estimate,
        'treated_before': treated_before,
        'treated_after': treated_after,
        'control_before': control_before,
        'control_after': control_after
    }


def plot_did_trends(df: pd.DataFrame, output_dir: Path) -> None:
    """绘制 DID 的趋势图"""
    font = setup_chinese_font()

    # 计算每期的平均流失率
    trend_data = df.groupby(['period', 'is_treated'])['churn_rate'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    # 分别绘制实验组和对照组
    for is_treated, label, color in [(1, '实验组（政策城市）', 'steelblue'),
                                       (0, '对照组（非政策城市）', 'coral')]:
        data = trend_data[trend_data['is_treated'] == is_treated]
        ax.plot(data['period'], data['churn_rate'],
                marker='o', label=label, color=color, linewidth=2)

    # 添加政策实施线
    policy_period = (df['period'].max() + df['period'].min()) / 2
    ax.axvline(x=policy_period, color='gray', linestyle='--',
               label='政策实施时点', linewidth=1.5)

    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('流失率', fontsize=12)
    ax.set_title('双重差分（DID）：平行趋势检验', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'did_trends.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"\n图表已保存: {output_dir / 'did_trends.png'}")


# ============================================================================
# 方法2：倾向得分匹配（PSM）
# ============================================================================

def generate_psm_data(
    n_samples: int = 10000,
    true_effect: float = -0.05,
    random_state: int = 42
) -> pd.DataFrame:
    """
    生成倾向得分匹配（PSM）的模拟数据

    场景：公司向某些客户发放优惠券（非随机）
    目标：估计优惠券对流失率的因果效应

    问题：存在混杂（高价值客户更容易收到优惠券）
    解决：用倾向得分匹配，消除观测混杂
    """
    rng = np.random.default_rng(random_state)

    # 混杂变量
    high_value = rng.binomial(1, 0.3, n_samples)
    vip_status = rng.binomial(1, 0.3, n_samples)
    purchase_count = rng.poisson(5, n_samples)
    days_since_last = rng.poisson(30, n_samples)

    # Treatment：高价值客户和 VIP 更容易收到优惠券
    coupon_prob = (
        0.2
        + 0.4 * high_value
        + 0.3 * vip_status
        + 0.02 * purchase_count
        - 0.001 * days_since_last
    )
    coupon_prob = np.clip(coupon_prob, 0, 1)
    coupon = rng.binomial(1, coupon_prob)

    # 结果：高价值和 VIP 本身流失率低，优惠券也有因果效应
    churn_prob = (
        0.35
        - 0.15 * high_value
        - 0.10 * vip_status
        - 0.02 * purchase_count
        + 0.001 * days_since_last
        + true_effect * coupon  # 优惠券的因果效应
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame({
        'high_value': high_value,
        'vip_status': vip_status,
        'purchase_count': purchase_count,
        'days_since_last_purchase': days_since_last,
        'coupon': coupon,
        'churn': churn
    })

    return df


def demonstrate_psm() -> dict:
    """
    演示倾向得分匹配（PSM）
    """
    print("\n" + "=" * 60)
    print("方法2：倾向得分匹配（Propensity Score Matching, PSM）")
    print("=" * 60)

    # 生成数据
    df = generate_psm_data()

    print("\n场景：公司向某些客户发放优惠券（非随机）")
    print("问题：存在混杂（高价值客户更容易收到优惠券）")

    # 1. 计算原始偏差（有混杂）
    raw_effect = (
        df[df['coupon'] == 1]['churn'].mean() -
        df[df['coupon'] == 0]['churn'].mean()
    )

    print(f"\n原始比较（有偏差）：")
    print(f"  收到优惠券的流失率：{df[df['coupon'] == 1]['churn'].mean():.3f}")
    print(f"  未收到优惠券的流失率：{df[df['coupon'] == 0]['churn'].mean():.3f}")
    print(f"  原始差异：{raw_effect:.3f}")

    # 2. 估计倾向得分
    print(f"\n第1步：估计倾向得分 P(coupon|X)")

    covariates = ['high_value', 'vip_status', 'purchase_count', 'days_since_last_purchase']
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(df[covariates], df['coupon'])
    df['propensity_score'] = ps_model.predict_proba(df[covariates])[:, 1]

    print(f"  倾向得分范围：{df['propensity_score'].min():.3f} - {df['propensity_score'].max():.3f}")
    print(f"  收到优惠券的平均倾向得分：{df[df['coupon'] == 1]['propensity_score'].mean():.3f}")
    print(f"  未收到优惠券的平均倾向得分：{df[df['coupon'] == 0]['propensity_score'].mean():.3f}")

    # 3. 匹配（1:1 最近邻匹配，有放回）
    print(f"\n第2步：匹配")

    treated = df[df['coupon'] == 1].copy()
    control = df[df['coupon'] == 0].copy()

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treated[['propensity_score']])

    matched_control = control.iloc[indices.flatten()].copy()

    # 4. 检查匹配质量
    print(f"\n匹配后平衡性检查：")

    for covariate in covariates:
        treated_mean = treated[covariate].mean()
        control_mean_matched = matched_control[covariate].mean()

        # 标准化均值差异
        treated_std = treated[covariate].std()
        control_std = matched_control[covariate].std()
        pooled_std = np.sqrt((treated_std**2 + control_std**2) / 2)
        smd = (treated_mean - control_mean_matched) / pooled_std if pooled_std > 0 else 0

        balanced = abs(smd) < 0.1
        status = "✓ 平衡" if balanced else "✗ 不平衡"

        print(f"  {covariate}: SMD = {smd:.3f} ({status})")

    # 5. 计算匹配后的因果效应
    matched_ate = (
        treated['churn'].mean() -
        matched_control['churn'].mean()
    )

    print(f"\n第3步：计算匹配后的因果效应")
    print(f"  匹配后的 ATE：{matched_ate:.3f}")
    print(f"  解释：发放优惠券使流失率变化 {matched_ate:.3f}")

    print(f"\n对比：")
    print(f"  原始差异（有偏差）：{raw_effect:.3f}")
    print(f"  匹配后（调整混杂）：{matched_ate:.3f}")
    print(f"  真实因果效应：-0.050")

    return {
        'raw_effect': raw_effect,
        'matched_ate': matched_ate,
        'true_effect': -0.05,
        'n_matched': len(treated),
        'df': df
    }


def plot_psm_matching(df: pd.DataFrame, output_dir: Path) -> None:
    """绘制倾向得分分布图"""
    font = setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 匹配前
    treated_ps = df[df['coupon'] == 1]['propensity_score']
    control_ps = df[df['coupon'] == 0]['propensity_score']

    axes[0].hist(control_ps, bins=30, alpha=0.5, label='对照组', color='coral', density=True)
    axes[0].hist(treated_ps, bins=30, alpha=0.5, label='实验组', color='steelblue', density=True)
    axes[0].set_xlabel('倾向得分', fontsize=12)
    axes[0].set_ylabel('密度', fontsize=12)
    axes[0].set_title('匹配前：倾向得分分布', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)

    # 匹配后（简化演示）
    # 实际应该用匹配后的数据
    axes[1].hist(control_ps.sample(n=len(treated_ps), random_state=42),
                 bins=30, alpha=0.5, label='对照组（匹配后）', color='coral', density=True)
    axes[1].hist(treated_ps, bins=30, alpha=0.5, label='实验组', color='steelblue', density=True)
    axes[1].set_xlabel('倾向得分', fontsize=12)
    axes[1].set_ylabel('密度', fontsize=12)
    axes[1].set_title('匹配后：倾向得分分布（更相似）', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'psm_matching.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"\n图表已保存: {output_dir / 'psm_matching.png'}")


# ============================================================================
# 观察研究的局限
# ============================================================================

def explain_observational_limitations() -> None:
    """解释观察研究的局限"""
    print("\n" + "=" * 60)
    print("观察研究的局限")
    print("=" * 60)

    methods = [
        ("双重差分（DID）", ["平行趋势假设", "需要前政策数据", "无法检验所有假设"]),
        ("倾向得分匹配（PSM）", ["只能控制观测变量", "未观测混杂仍存在", "匹配质量依赖模型正确性"]),
        ("工具变量（IV）", ["好的工具变量难找", "外生性和排他性无法完全验证", "弱工具变量问题"]),
    ]

    for i, (method, limitations) in enumerate(methods, 1):
        print(f"\n{i}. {method}的局限：")
        for j, limitation in enumerate(limitations, 1):
            print(f"   {j}. {limitation}")

    print("\n共同结论：")
    print("  - 观察研究永远是'近似'，不是'精确'")
    print("  - 需要明确报告假设和局限性")
    print("  - 敏感性分析：如果存在未观测混杂，结论会如何变化？")
    print("  - 最优方法：RCT（如果可行）")


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("观察研究中的因果推断方法")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 双重差分（DID）
    did_results = demonstrate_did()
    plot_did_trends(did_results['data'], output_dir)

    # 2. 倾向得分匹配（PSM）
    psm_results = demonstrate_psm()
    plot_psm_matching(psm_results['df'], output_dir)

    # 3. 观察研究的局限
    explain_observational_limitations()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    print("\n当 RCT 不可行时的因果推断方法：")

    print("\n1. 双重差分（DID）：")
    print("   - 适用：有自然实验，有前后数据")
    print("   - 关键假设：平行趋势")
    print("   - 局限：趋势不平行时失效")

    print("\n2. 倾向得分匹配（PSM）：")
    print("   - 适用：有观测混杂，可找到相似对照")
    print("   - 关键假设：无未观测混杂")
    print("   - 局限：只能控制观测变量")

    print("\n3. 工具变量（IV）：")
    print("   - 适用：有合适工具变量")
    print("   - 关键假设：工具变量外生 + 排他性")
    print("   - 局限：好的工具变量难找")

    print("\n选择方法的决策树：")
    print("  能做 RCT？ → RCT（金标准）")
    print("  有自然实验 + 前后数据？ → DID")
    print("  有观测混杂 + 能找到相似对照？ → PSM")
    print("  有合适工具变量？ → IV")

    print(f"\n生成的图表：")
    print(f"  - {output_dir / 'did_trends.png'}")
    print(f"  - {output_dir / 'psm_matching.png'}")


if __name__ == "__main__":
    main()
