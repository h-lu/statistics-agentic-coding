"""
示例：异常点与影响点分析——Cook's 距离

本例演示如何识别和处理回归分析中的异常点：
1. 三种异常点：离群点、高杠杆点、强影响点
2. Cook's 距离的计算与可视化
3. 杠杆图 (Leverage vs 标准化残差)
4. 删除前后模型对比

运行方式：python3 chapters/week_09/examples/06_cooks_distance.py
预期输出：
- Cook's 距离图（保存为 cooks_distance.png）
- 杠杆图（保存为 leverage_plot.png）
- 删除前后模型系数对比
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_leverage_resid2

np.random.seed(42)


def generate_data_with_outliers(n_samples: int = 50) -> pd.DataFrame:
    """
    生成包含异常点的房价数据

    参数:
        n_samples: 正常样本数量

    返回:
        包含异常点的 DataFrame
    """
    # 正常数据
    area = np.random.uniform(50, 120, n_samples)
    price = 20 + 1.0 * area + np.random.normal(0, 10, n_samples)

    df = pd.DataFrame({
        'area_sqm': area,
        'price_wan': price
    })

    # 添加 3 个异常点
    # 异常点1: 高杠杆点 (面积异常大)
    df.loc[len(df)] = [180, 20 + 1.0 * 180 + np.random.normal(0, 10)]

    # 异常点2: 离群点 (房价异常高，但面积正常)
    df.loc[len(df)] = [80, 20 + 1.0 * 80 + 80]  # 残差约80万

    # 异常点3: 强影响点 (面积和房价都异常，且拽动回归线)
    df.loc[len(df)] = [180, 20 + 1.0 * 180 + 120]  # 高杠杆 + 大残差

    return df.reset_index(drop=True)


def plot_cooks_distance(model: sm.regression.linear_model.RegressionResults,
                       df: pd.DataFrame,
                       output_name: str = 'cooks_distance.png') -> None:
    """
    画 Cook's 距离图

    参数:
        model: 拟合的回归模型
        df: 原始数据
        output_name: 输出文件名
    """
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]

    plt.figure(figsize=(12, 5))

    # 左图：Cook's 距离条形图
    plt.subplot(1, 2, 1)
    plt.bar(df.index, cooks_d, color='steelblue', alpha=0.7, edgecolor='k')
    plt.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, label='中等影响 (D=0.5)')
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='强影响 (D=1.0)')
    plt.xlabel('观测索引', fontsize=12)
    plt.ylabel("Cook's 距离", fontsize=12)
    plt.title("Cook's 距离 - 识别强影响点", fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    # 右图：Cook's 距离散点图（更清晰）
    plt.subplot(1, 2, 2)
    colors = ['red' if d >= 1 else 'orange' if d >= 0.5 else 'steelblue'
              for d in cooks_d]
    plt.scatter(df.index, cooks_d, c=colors, s=80, alpha=0.7, edgecolors='k')
    plt.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('观测索引', fontsize=12)
    plt.ylabel("Cook's 距离", fontsize=12)
    plt.title("Cook's 距离 - 散点图", fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # 标注强影响点
    high_influence = np.where(cooks_d >= 1)[0]
    for idx in high_influence:
        plt.annotate(f'#{idx}', (idx, cooks_d[idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"✅ Cook's 距离图已保存为 {output_name}")
    plt.close()


def plot_leverage_resid2(model: sm.regression.linear_model.RegressionResults,
                        output_name: str = 'leverage_plot.png') -> None:
    """
    画杠杆图 (Leverage vs 标准化残差²)

    参数:
        model: 拟合的回归模型
        output_name: 输出文件名
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_leverage_resid2(model, ax=ax)
    ax.set_title('杠杆图 (Leverage vs 标准化残差²)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"✅ 杠杆图已保存为 {output_name}")
    plt.close()


def identify_outliers(model: sm.regression.linear_model.RegressionResults,
                    df: pd.DataFrame) -> dict:
    """
    识别三类异常点

    参数:
        model: 拟合的回归模型
        df: 原始数据

    返回:
        包含三类异常点索引的字典
    """
    influence = model.get_influence()

    # 1. Cook's 距离 (强影响点)
    cooks_d = influence.cooks_distance[0]
    influential_points = np.where(cooks_d >= 1)[0]

    # 2. 标准化残差 (离群点)
    standardized_resid = influence.resid_studentized_internal
    outliers = np.where(np.abs(standardized_resid) > 2)[0]

    # 3. 杠杆值 (高杠杆点)
    leverage = influence.hat_matrix_diag
    # 杠杆阈值: 2 * (k+1) / n, k为自变量数
    n = len(df)
    k = len(model.params) - 1  # 不包括截距
    leverage_threshold = 2 * (k + 1) / n
    high_leverage = np.where(leverage > leverage_threshold)[0]

    return {
        'cook': influential_points,
        'outlier': outliers,
        'leverage': high_leverage,
        'cooks_d': cooks_d,
        'standardized_resid': standardized_resid,
        'leverage_values': leverage
    }


def print_outlier_summary(outlier_info: dict, df: pd.DataFrame) -> None:
    """打印异常点摘要"""
    print(f"\n{'=' * 70}")
    print("异常点识别结果")
    print('=' * 70)

    print(f"\n1. 强影响点 (Cook's D ≥ 1):")
    if len(outlier_info['cook']) > 0:
        for idx in outlier_info['cook']:
            print(f"   观测 #{idx}: Cook's D = {outlier_info['cooks_d'][idx]:.3f}")
            print(f"      面积 = {df.loc[idx, 'area_sqm']:.1f}, 房价 = {df.loc[idx, 'price_wan']:.1f}")
    else:
        print(f"   无")

    print(f"\n2. 离群点 (|标准化残差| > 2):")
    if len(outlier_info['outlier']) > 0:
        for idx in outlier_info['outlier'][:10]:  # 最多显示10个
            print(f"   观测 #{idx}: 标准化残差 = {outlier_info['standardized_resid'][idx]:.2f}")
    else:
        print(f"   无")

    print(f"\n3. 高杠杆点 (Leverage > 阈值):")
    if len(outlier_info['leverage']) > 0:
        for idx in outlier_info['leverage']:
            print(f"   观测 #{idx}: Leverage = {outlier_info['leverage_values'][idx]:.3f}")
    else:
        print(f"   无")


def compare_models(original_model: sm.regression.linear_model.RegressionResults,
                 cleaned_model: sm.regression.linear_model.RegressionResults,
                 removed_indices: list) -> None:
    """
    对比删除异常点前后的模型

    参数:
        original_model: 原始模型
        cleaned_model: 删除异常点后的模型
        removed_indices: 被删除的观测索引
    """
    print(f"\n{'=' * 70}")
    print(f"对比: 删除异常点前后的模型")
    print(f"删除了 {len(removed_indices)} 个观测: {removed_indices}")
    print('=' * 70)

    comparison = pd.DataFrame({
        '原始模型': original_model.params,
        '删除后模型': cleaned_model.params.reindex(original_model.params.index),
        '变化%': ((cleaned_model.params.reindex(original_model.params.index) -
                 original_model.params) / original_model.params * 100).round(1)
    })
    print(comparison)

    # 判断模型是否稳健
    max_change = comparison['变化%'].abs().max()
    print(f"\n最大系数变化: {max_change:.1f}%")
    if max_change < 10:
        print(f"结论: ✓ 模型对异常点稳健")
    else:
        print(f"结论: ✗ 模型对异常点敏感,结论可能被少数点'绑架'")


def main() -> None:
    """主函数：演示异常点分析"""
    print("=" * 70)
    print("示例6: 异常点与影响点分析——Cook's 距离")
    print("=" * 70)

    # 1. 生成包含异常点的数据
    df = generate_data_with_outliers(n_samples=50)
    print(f"\n📊 数据概览:")
    print(f"  样本量: {len(df)}")
    print(f"  面积范围: [{df['area_sqm'].min():.1f}, {df['area_sqm'].max():.1f}] 平米")
    print(f"  房价范围: [{df['price_wan'].min():.1f}, {df['price_wan'].max():.1f}] 万元")

    print(f"\n最后3行 (可能包含异常点):")
    print(df.tail(3))

    # 2. 拟合原始模型
    X = sm.add_constant(df[['area_sqm']])
    y = df['price_wan']
    model_original = sm.OLS(y, X).fit()

    print(f"\n📈 原始模型:")
    print(f"  截距: {model_original.params['const']:.2f}")
    print(f"  斜率: {model_original.params['area_sqm']:.3f}")
    print(f"  R²: {model_original.rsquared:.3f}")

    # 3. 识别异常点
    outlier_info = identify_outliers(model_original, df)
    print_outlier_summary(outlier_info, df)

    # 4. 画 Cook's 距离图
    plot_cooks_distance(model_original, df)

    # 5. 画杠杆图
    plot_leverage_resid2(model_original)

    # 6. 删除强影响点，重新拟合
    strong_influence = outlier_info['cook']
    if len(strong_influence) > 0:
        df_cleaned = df.drop(strong_influence)
        X_cleaned = sm.add_constant(df_cleaned[['area_sqm']])
        y_cleaned = df_cleaned['price_wan']
        model_cleaned = sm.OLS(y_cleaned, X_cleaned).fit()

        # 7. 对比模型
        compare_models(model_original, model_cleaned, list(strong_influence))
    else:
        print(f"\n✓ 无强影响点需要删除")

    # ========================================
    # 处理策略总结
    # ========================================
    print(f"\n{'=' * 70}")
    print("异常点处理策略")
    print('=' * 70)
    print("""
    1. 核实数据:
       - 检查是否为录入错误 (如 50 写成 500)
       - 修正后重新拟合

    2. 保留但标注:
       - 在报告中说明异常点的性质
       - 提供业务解释 (如市中心豪华小户型)

    3. 敏感性分析:
       - 对比删除前后的模型
       - 如果系数变化不大 → 模型稳健
       - 如果系数变化剧烈 → 需要谨慎解释结论

    4. 稳健方法:
       - 使用稳健回归 (如 RLM、M-estimator)
       - 降低异常点的权重

    5. 领域判断:
       - 结合业务知识决定是否删除
       - 不要只依赖统计规则
    """)

    print("\n" + "=" * 70)
    print("✅ 示例6完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
