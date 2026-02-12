"""
示例：残差诊断——检验回归的 LINE 假设

本例演示如何通过残差诊断图检验回归模型的四大假设：
- L: Linearity (线性)
- I: Independence (独立性)
- N: Normality (正态性)
- E: Equal variance (等方差)

运行方式：python3 chapters/week_09/examples/05_residual_diagnostics.py
预期输出：
- 2x2 残差诊断图（保存为 residual_diagnostics.png）
- 假设检验统计量（Durbin-Watson, Shapiro-Wilk, Breusch-Pagan）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro, probplot
from statsmodels.stats.diagnostic import het_breuschpagan

np.random.seed(42)


def generate_housing_data(n_samples: int = 100) -> pd.DataFrame:
    """生成房价数据（满足线性假设）"""
    area = np.random.uniform(40, 150, n_samples)
    age = np.random.uniform(0, 30, n_samples)

    # 线性关系 + 正态噪音
    price = 20 + 1.0 * area - 0.4 * age + np.random.normal(0, 12, n_samples)

    return pd.DataFrame({
        'area_sqm': area,
        'age_years': age,
        'price_wan': price
    })


def generate_heteroscedastic_data(n_samples: int = 100) -> pd.DataFrame:
    """生成异方差数据（违反等方差假设，用于对比）"""
    area = np.random.uniform(40, 150, n_samples)

    # 方差随面积增大而增大
    noise_std = 5 + 0.15 * area
    price = 20 + 1.0 * area + np.random.normal(0, 1, n_samples) * noise_std

    return pd.DataFrame({
        'area_sqm': area,
        'price_wan': price
    })


def plot_residual_diagnostics(model: sm.regression.linear_model.RegressionResults,
                             df: pd.DataFrame,
                             output_name: str = 'residual_diagnostics.png') -> None:
    """
    画 2x2 残差诊断图

    参数:
        model: 拟合的回归模型
        df: 原始数据
        output_name: 输出文件名
    """
    residuals = model.resid
    fitted = model.fittedvalues
    influence = model.get_influence()
    standardized_residuals = influence.resid_studentized_internal

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 残差 vs 拟合值 (检验线性和等方差)
    axes[0, 0].scatter(fitted, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('拟合值', fontsize=12)
    axes[0, 0].set_ylabel('残差', fontsize=12)
    axes[0, 0].set_title('残差 vs 拟合值 (检验线性与等方差)', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 添加 LOWESS 平滑线（帮助识别模式）
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(residuals, fitted, frac=0.6)
    axes[0, 0].plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, alpha=0.7, label='LOWESS')
    axes[0, 0].legend(loc='upper right')

    # 2. QQ 图 (检验正态性)
    probplot(residuals, plot=axes[0, 1])
    axes[0, 1].set_title('QQ 图 (检验正态性)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 尺度-位置图 (检验等方差)
    axes[1, 0].scatter(fitted, np.sqrt(np.abs(standardized_residuals)),
                      alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[1, 0].set_xlabel('拟合值', fontsize=12)
    axes[1, 0].set_ylabel('√|标准化残差|', fontsize=12)
    axes[1, 0].set_title('尺度-位置图 (检验同方差)', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 添加水平参考线
    axes[1, 0].axhline(y=np.sqrt(np.abs(standardized_residuals)).mean(),
                       color='red', linestyle='--', linewidth=2)

    # 4. 标准化残差 vs 观测索引 (检验独立性)
    axes[1, 1].scatter(range(len(residuals)), standardized_residuals,
                      alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].axhline(y=2, color='orange', linestyle=':', linewidth=1)
    axes[1, 1].axhline(y=-2, color='orange', linestyle=':', linewidth=1)
    axes[1, 1].set_xlabel('观测索引', fontsize=12)
    axes[1, 1].set_ylabel('标准化残差', fontsize=12)
    axes[1, 1].set_title('标准化残差序列 (检验独立性)', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"✅ 残差诊断图已保存为 {output_name}")
    plt.close()


def run_assumption_tests(model: sm.regression.linear_model.RegressionResults) -> dict:
    """
    运行假设检验

    参数:
        model: 拟合的回归模型

    返回:
        包含各项检验结果的字典
    """
    results = {}
    residuals = model.resid

    # 1. Durbin-Watson 检验 (独立性)
    from statsmodels.stats.stattools import durbin_watson
    dw = durbin_watson(residuals)
    results['durbin_watson'] = dw
    results['independent'] = 1.5 < dw < 2.5  # 经验判断

    # 2. Shapiro-Wilk 检验 (正态性)
    stat, p_value = shapiro(residuals)
    results['shapiro_stat'] = stat
    results['shapiro_p'] = p_value
    results['normal'] = p_value > 0.05

    # 3. Breusch-Pagan 检验 (异方差)
    # H0: 误差方差齐性 (等方差)
    # H1: 误差方差不齐 (异方差)
    X = model.model.exog
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
    results['bp_stat'] = bp_stat
    results['bp_p'] = bp_p
    results['homoscedastic'] = bp_p > 0.05

    return results


def print_diagnostics_report(results: dict) -> None:
    """打印诊断报告"""
    print(f"\n{'=' * 60}")
    print("回归假设检验报告")
    print('=' * 60)

    # 1. 线性假设
    print(f"\n1. 线性假设 (Linearity)")
    print(f"   方法: 残差 vs 拟合值图")
    print(f"   判断: 点应随机散布在 y=0 线上下,无线性模式")
    print(f"   → 见残差诊断图左上")

    # 2. 独立性假设
    print(f"\n2. 独立性假设 (Independence)")
    print(f"   Durbin-Watson 统计量: {results['durbin_watson']:.3f}")
    print(f"   判断: DW ≈ 2 表示无自相关")
    if 1.5 < results['durbin_watson'] < 2.5:
        print(f"   结果: ✓ 独立性假设满足")
    else:
        print(f"   结果: ✗ 可能存在自相关")

    # 3. 正态性假设
    print(f"\n3. 正态性假设 (Normality)")
    print(f"   Shapiro-Wilk 检验:")
    print(f"     统计量: {results['shapiro_stat']:.4f}")
    print(f"     p 值: {results['shapiro_p']:.4f}")
    if results['normal']:
        print(f"   结果: ✓ 不能拒绝正态性假设 (p > 0.05)")
    else:
        print(f"   结果: ✗ 拒绝正态性假设 (p ≤ 0.05)")
    print(f"   → 见残差诊断图右上 (QQ 图)")

    # 4. 等方差假设
    print(f"\n4. 等方差假设 (Equal Variance)")
    print(f"   Breusch-Pagan 检验:")
    print(f"     统计量: {results['bp_stat']:.4f}")
    print(f"     p 值: {results['bp_p']:.4f}")
    if results['homoscedastic']:
        print(f"   结果: ✓ 不能拒绝等方差假设 (p > 0.05)")
    else:
        print(f"   结果: ✗ 拒绝等方差假设,存在异方差 (p ≤ 0.05)")
    print(f"   → 见残差诊断图左下和右下")


def main() -> None:
    """主函数：演示残差诊断"""
    print("=" * 60)
    print("示例5: 残差诊断——检验 LINE 假设")
    print("=" * 60)

    # ========================================
    # 场景1: 满足所有假设的好数据
    # ========================================
    print(f"\n{'='*60}")
    print("场景1: 满足 LINE 假设的数据")
    print('='*60)

    df1 = generate_housing_data(n_samples=100)
    X1 = sm.add_constant(df1[['area_sqm', 'age_years']])
    model1 = sm.OLS(df1['price_wan'], X1).fit()

    print(f"\n模型拟合结果:")
    print(f"  R² = {model1.rsquared:.3f}")
    print(f"  面积系数 = {model1.params['area_sqm']:.3f}")
    print(f"  房龄系数 = {model1.params['age_years']:.3f}")

    plot_residual_diagnostics(model1, df1, 'residual_diagnostics_good.png')

    results1 = run_assumption_tests(model1)
    print_diagnostics_report(results1)

    # ========================================
    # 场景2: 异方差数据（违反等方差）
    # ========================================
    print(f"\n\n{'='*60}")
    print("场景2: 违反等方差假设的数据（异方差）")
    print('='*60)

    df2 = generate_heteroscedastic_data(n_samples=100)
    X2 = sm.add_constant(df2[['area_sqm']])
    model2 = sm.OLS(df2['price_wan'], X2).fit()

    print(f"\n模型拟合结果:")
    print(f"  R² = {model2.rsquared:.3f}")
    print(f"  面积系数 = {model2.params['area_sqm']:.3f}")

    plot_residual_diagnostics(model2, df2, 'residual_diagnostics_bad.png')

    results2 = run_assumption_tests(model2)
    print_diagnostics_report(results2)

    # ========================================
    # 处理策略
    # ========================================
    print(f"\n\n{'='*60}")
    print("假设违反时的处理策略")
    print('='*60)
    print("""
    1. 线性假设违反:
       - 加入多项式项 (如 x²)
       - 变换变量 (如 log, sqrt)

    2. 正态性假设违反:
       - 变换因变量 (如 log(y))
       - 使用 Bootstrap 置信区间 (不依赖正态假设)
       - 使用稳健回归

    3. 等方差假设违反 (异方差):
       - 使用稳健标准误 (HC0, HC1, HC2, HC3)
       - 变换因变量 (如 log(y))
       - 加权最小二乘 (WLS)

    4. 独立性假设违反:
       - 检查数据采集方式
       - 使用聚类稳健标准误
       - 考虑时间序列模型
    """)

    # 演示稳健标准误
    print(f"\n{'='*60}")
    print("演示: 使用稳健标准误修正异方差")
    print('='*60)

    model_robust = sm.OLS(df2['price_wan'], X2).fit(cov_type='HC3')

    print(f"\n对比普通标准误 vs HC3 稳健标准误:")
    comparison = pd.DataFrame({
        '系数': model2.params,
        '普通标准误': model2.bse,
        'HC3 标准误': model_robust.bse,
    })
    print(comparison.round(4))
    print(f"\n结论: HC3 标准误更大,更诚实地表达了不确定性")

    print("\n" + "=" * 60)
    print("✅ 示例5完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
