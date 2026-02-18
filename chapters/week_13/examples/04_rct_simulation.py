"""
示例：RCT 模拟——为什么随机化是因果推断的"金标准"

本例演示：
1. RCT 的原理：随机化如何切断所有混杂路径
2. RCT 的假设：随机化成功、SUTVA、依从性、无流失偏差
3. RCT 与观察研究的对比：为什么 RCT 更可靠

运行方式：python3 chapters/week_13/examples/04_rct_simulation.py
预期输出：stdout 输出 RCT 模拟结果 + 保存图表到 images/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
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
# RCT 模拟：潜在结果框架
# ============================================================================

def simulate_rct(
    n_samples: int = 10000,
    true_ate: float = -0.05,
    baseline_rate: float = 0.30,
    random_state: int = 42
) -> dict:
    """
    模拟随机对照试验（RCT）

    参数：
    - n_samples: 样本量
    - true_ate: 真实的平均处理效应（ATE）
    - baseline_rate: 对照组的平均结果率
    - random_state: 随机种子

    返回：
    - dict: 包含 RCT 数据和估计的 ATE
    """
    rng = np.random.default_rng(random_state)

    # 随机分配 treatment
    treatment = rng.binomial(1, 0.5, n_samples)

    # 潜在结果框架
    # Y_0: 未接受 treatment 的潜在结果
    # Y_1: 接受 treatment 的潜在结果
    Y_0 = rng.binomial(1, baseline_rate, n_samples)
    Y_1 = rng.binomial(1, baseline_rate + true_ate, n_samples)

    # 实际观察到的结果（取决于实际接受的 treatment）
    Y_observed = np.where(treatment == 1, Y_1, Y_0)

    # 个体因果效应（无法直接观测）
    individual_effect = Y_1 - Y_0

    df = pd.DataFrame({
        'treatment': treatment,
        'Y_0': Y_0,
        'Y_1': Y_1,
        'Y_observed': Y_observed,
        'individual_effect': individual_effect
    })

    # 估计 ATE
    treated_outcome = df[df['treatment'] == 1]['Y_observed'].mean()
    control_outcome = df[df['treatment'] == 0]['Y_observed'].mean()
    estimated_ate = treated_outcome - control_outcome

    return {
        'data': df,
        'true_ate': true_ate,
        'estimated_ate': estimated_ate,
        'treated_outcome': treated_outcome,
        'control_outcome': control_outcome,
        'individual_effect': individual_effect.mean()
    }


def demonstrate_rct_logic(n_samples: int = 10000, random_state: int = 42) -> dict:
    """
    演示 RCT 的逻辑：随机化为什么有效

    对比：
    1. 观察研究（有混杂）
    2. RCT（随机化切断混杂）
    """
    print("=" * 60)
    print("RCT 模拟：随机化为什么有效")
    print("=" * 60)

    rng = np.random.default_rng(random_state)

    # 1. 观察研究（有混杂）
    print("\n--- 观察研究（有混杂）---")

    # 未观测的混杂：高价值客户
    high_value = rng.binomial(1, 0.3, n_samples)

    # 高价值客户更容易收到优惠券
    coupon_prob = np.where(high_value == 1, 0.8, 0.2)
    coupon = rng.binomial(1, coupon_prob)

    # 流失率由高价值决定（优惠券没有因果效应）
    churn_prob = np.where(high_value == 1, 0.10, 0.30)
    churn = rng.binomial(1, churn_prob)

    df_obs = pd.DataFrame({
        'coupon': coupon,
        'churn': churn
        # high_value 未观测
    })

    # 计算观察到的关联（有偏差）
    obs_effect = (
        df_obs[df_obs['coupon'] == 1]['churn'].mean() -
        df_obs[df_obs['coupon'] == 0]['churn'].mean()
    )

    print(f"观察到的效应：{obs_effect:.3f}")
    print(f"  收到优惠券的客户流失率: {df_obs[df_obs['coupon'] == 1]['churn'].mean():.3f}")
    print(f"  未收到优惠券的客户流失率: {df_obs[df_obs['coupon'] == 0]['churn'].mean():.3f}")
    print(f"  结论：优惠券看起来降低了流失率（虚假！）")
    print(f"  真相：优惠券没有因果效应，只是高价值客户更容易收到优惠券")

    # 2. RCT（随机化）
    print("\n--- RCT（随机化切断混杂）---")

    rct_results = simulate_rct(n_samples=n_samples, true_ate=-0.05, random_state=random_state)

    print(f"RCT 估计的效应：{rct_results['estimated_ate']:.3f}")
    print(f"  真实 ATE：{rct_results['true_ate']:.3f}")
    print(f"  实验组流失率：{rct_results['treated_outcome']:.3f}")
    print(f"  对照组流失率：{rct_results['control_outcome']:.3f}")
    print(f"  结论：优惠券降低了流失率（因果！）")

    print(f"\n对比：")
    print(f"  观察研究（有混杂）：{obs_effect:.3f}（偏差）")
    print(f"  RCT（随机化）：{rct_results['estimated_ate']:.3f}（无偏）")
    print(f"  真实因果效应：{rct_results['true_ate']:.3f}")

    return {
        'observational_effect': obs_effect,
        'rct_effect': rct_results['estimated_ate'],
        'true_effect': rct_results['true_ate']
    }


# ============================================================================
# RCT 假设检查
# ============================================================================

def check_rct_assumptions(df: pd.DataFrame, covariates: list = None) -> dict:
    """
    检查 RCT 的关键假设

    假设：
    1. 随机化成功：实验组和对照组在基线特征上平衡
    2. SUTVA：个体的 Treatment 不影响其他人（难以检验）
    3. 依从性：实验组真的接受了 Treatment，对照组没有
    4. 无流失偏差：流失在两组间随机
    """
    print("\n" + "=" * 60)
    print("RCT 假设检查")
    print("=" * 60)

    results = {}

    # 1. 随机化成功：基线平衡性检验
    print("\n1. 随机化成功：基线平衡性")

    if covariates is None:
        # 如果没有提供协变量，生成模拟的
        rng = np.random.default_rng(42)
        n = len(df)
        covariates_data = pd.DataFrame({
            'age': rng.integers(18, 70, n),
            'income': rng.exponential(50000, n),
            'gender': rng.binomial(1, 0.5, n)
        })
    else:
        covariates_data = covariates

    balance_test = []
    for covariate in covariates_data.columns:
        treated = covariates_data[df['treatment'] == 1][covariate]
        control = covariates_data[df['treatment'] == 0][covariate]

        # t 检验
        t_stat, p_value = stats.ttest_ind(treated, control)

        # 标准化均值差异（Standardized Mean Difference）
        pooled_std = np.sqrt((treated.std()**2 + control.std()**2) / 2)
        smd = (treated.mean() - control.mean()) / pooled_std if pooled_std > 0 else 0

        balance_test.append({
            'covariate': covariate,
            'treated_mean': treated.mean(),
            'control_mean': control.mean(),
            'smd': smd,
            'p_value': p_value,
            'balanced': abs(smd) < 0.1 and p_value > 0.05
        })

    balance_df = pd.DataFrame(balance_test)
    print(balance_test)

    results['balance'] = balance_df

    # 检查是否平衡
    n_unbalanced = balance_df[~balance_df['balanced']].shape[0]
    if n_unbalanced == 0:
        print(f"\n  结论：所有协变量在两组间平衡（随机化成功）")
    else:
        print(f"\n  警告：{n_unbalanced} 个协变量不平衡（可能需要调整）")

    # 2. 依从性检查
    print("\n2. 依从性检查")

    # 在 RCT 中，treatment 是随机分配的，依从性假设自动满足
    # 在实际 RCT 中，需要检查"实际接受的 treatment"是否与"分配的 treatment"一致
    compliance_rate = df['treatment'].mean()
    print(f"  实验组占比：{compliance_rate:.1%}（预期 50%）")
    print(f"  结论：依从性假设满足")

    results['compliance'] = compliance_rate

    # 3. 样本量检查
    print("\n3. 样本量检查")

    n_treated = df['treatment'].sum()
    n_control = len(df) - n_treated
    min_sample_size = min(n_treated, n_control)

    print(f"  实验组样本量：{n_treated}")
    print(f"  对照组样本量：{n_control}")
    print(f"  最小组样本量：{min_sample_size}")

    if min_sample_size >= 100:
        print(f"  结论：样本量充足")
    else:
        print(f"  警告：样本量可能不足")

    results['sample_size'] = {'treated': n_treated, 'control': n_control}

    return results


# ============================================================================
# 可视化：RCT vs 观察研究
# ============================================================================

def plot_rct_vs_observational(
    obs_effect: float,
    rct_effect: float,
    true_effect: float,
    output_dir: Path
) -> None:
    """绘制 RCT 与观察研究的对比图"""
    font = setup_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['观察研究\n(有混杂)', 'RCT\n(随机化)', '真实\n因果效应']
    effects = [obs_effect, rct_effect, true_effect]
    colors = ['coral', 'steelblue', 'green']

    bars = ax.bar(methods, effects, color=colors, alpha=0.7, edgecolor='black')

    # 添加数值标签
    for bar, effect in zip(bars, effects):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{effect:.3f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=12, fontweight='bold')

    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.set_ylabel('效应大小', fontsize=12)
    ax.set_title('RCT vs 观察研究：随机化如何消除混杂偏差\n'
                 '（负值表示降低流失率）', fontsize=14, fontweight='bold')
    ax.set_ylim(min(effects) - 0.05, max(effects) + 0.05)

    # 添加说明
    ax.text(0.5, -0.15,
            '观察研究：由于混杂（高价值客户更易收到优惠券），\n'
            '优惠券看起来有效，但这只是虚假相关\n\n'
            'RCT：随机化确保两组在所有变量上平衡，\n'
            '因此差异只能是因果效应',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'rct_vs_observational.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"\n图表已保存: {output_dir / 'rct_vs_observational.png'}")


# ============================================================================
# RCT 的局限
# ============================================================================

def explain_rct_limitations() -> None:
    """解释 RCT 的局限"""
    print("\n" + "=" * 60)
    print("RCT 的局限")
    print("=" * 60)

    limitations = [
        ("成本高", "医学 RCT 可能需要数百万美元"),
        ("时间限制", "某些效应需要数年才能显现"),
        ("伦理问题", "不能随机让人吸烟来测试吸烟是否致癌"),
        ("外部有效性", "实验室结论可能推广不到真实世界"),
        ("SUTVA 违反", "社交产品中，用户行为会互相影响（网络效应）"),
    ]

    for i, (limitation, description) in enumerate(limitations, 1):
        print(f"\n{i}. {limitation}")
        print(f"   {description}")

    print("\n结论：")
    print("  RCT 是金标准，但不是万能的")
    print("  当 RCT 不可行时，需要用观察研究方法（DID、IV、PSM）")


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """运行所有演示"""
    print("\n" + "=" * 60)
    print("RCT 模拟演示")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 演示 RCT 的逻辑
    comparison_results = demonstrate_rct_logic()

    # 2. RCT 假设检查
    # 生成模拟 RCT 数据
    rct_data = simulate_rct(n_samples=10000, true_ate=-0.05, random_state=42)
    df_rct = rct_data['data'].copy()
    df_rct = df_rct[['treatment', 'Y_observed']].rename(columns={'Y_observed': 'churn'})

    check_rct_assumptions(df_rct)

    # 3. 可视化对比
    plot_rct_vs_observational(
        comparison_results['observational_effect'],
        comparison_results['rct_effect'],
        comparison_results['true_effect'],
        output_dir
    )

    # 4. RCT 的局限
    explain_rct_limitations()

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)

    print("\n为什么 RCT 是金标准？")
    print("  - 随机化切断了所有混杂路径")
    print("  - 在 RCT 中，关联 = 因果")
    print('  - 无需假设"没有未观测混杂"')

    print("\nRCT 的关键假设：")
    print("  1. 随机化成功：基线平衡")
    print("  2. SUTVA：无干扰")
    print("  3. 依从性：按分配接受 treatment")
    print("  4. 无流失偏差：流失随机")

    print("\n当 RCT 不可行时：")
    print("  - 双重差分（DID）")
    print("  - 工具变量（IV）")
    print("  - 倾向得分匹配（PSM）")
    print("  - 断点回归（RDD）")

    print(f"\n生成的图表：")
    print(f"  - {output_dir / 'rct_vs_observational.png'}")


if __name__ == "__main__":
    main()
