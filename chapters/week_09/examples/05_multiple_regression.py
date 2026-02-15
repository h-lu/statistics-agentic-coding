"""
示例：多元回归与多重共线性——从"简单回归"到"多元回归"。

本例演示多元回归的核心概念：
- 多元回归方程：y = a + b₁x₁ + b₂x₂ + ... + bₖxₖ
- 控制其他变量后：多元回归系数的含义
- 多重共线性：自变量之间高度相关的问题
- VIF（方差膨胀因子）：检测多重共线性
- 调整 R²：惩罚模型复杂度的 R²

运行方式：python3 chapters/week_09/examples/05_multiple_regression.py
预期输出：
  - stdout 输出多元回归结果和 VIF 分析
  - 展示多重共线性的影响
  - 保存图表到 images/05_multiple_regression.png

核心概念：
  - 多元回归：多个自变量同时预测因变量
  - "控制其他变量后"：每个系数表示其他变量不变时的影响
  - 多重共线性：自变量高度相关导致系数不稳定
  - VIF > 10：严重多重共线性
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path


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


def generate_ecommerce_data(n: int = 100, random_state: int = 42) -> pd.DataFrame:
    """
    生成模拟电商数据

    变量：
    - ad_spend: 广告投入
    - price: 价格
    - promotion: 是否促销（0/1）
    - competitor_price: 竞争对手价格（与 price 高度相关）
    - sales: 销售额
    """
    np.random.seed(random_state)

    ad_spend = np.random.normal(loc=50, scale=15, size=n)
    price = np.random.normal(loc=100, scale=20, size=n)
    promotion = np.random.binomial(n=1, p=0.3, size=n)

    # 竞争对手价格与价格高度相关（制造多重共线性）
    competitor_price = price * 1.2 + np.random.normal(loc=0, scale=5, size=n)

    # 销售额方程
    sales = 20 + 0.5 * ad_spend - 0.3 * price + 10 * promotion
    sales += np.random.normal(loc=0, scale=5, size=n)

    return pd.DataFrame({
        'ad_spend': ad_spend,
        'price': price,
        'promotion': promotion,
        'competitor_price': competitor_price,
        'sales': sales
    })


def simple_vs_multiple_regression(df: pd.DataFrame) -> None:
    """对比简单回归和多元回归"""
    print("=" * 70)
    print("简单回归 vs 多元回归：系数的变化")
    print("=" * 70)

    X_simple = df['ad_spend']
    y = df['sales']

    # 简单回归
    X_simple_const = sm.add_constant(X_simple)
    model_simple = sm.OLS(y, X_simple_const).fit()

    print("\n简单回归：sales ~ ad_spend")
    print(f"  sales = {model_simple.params['const']:.2f} + {model_simple.params['ad_spend']:.4f} × ad_spend")
    print(f"  斜率 = {model_simple.params['ad_spend']:.4f} (p = {model_simple.pvalues['ad_spend']:.4f})")

    # 多元回归（控制价格和促销）
    X_multi = df[['ad_spend', 'price', 'promotion']]
    X_multi_const = sm.add_constant(X_multi)
    model_multi = sm.OLS(y, X_multi_const).fit()

    print("\n多元回归：sales ~ ad_spend + price + promotion")
    print(f"  sales = {model_multi.params['const']:.2f}")
    print(f"         + {model_multi.params['ad_spend']:.4f} × ad_spend")
    print(f"         + {model_multi.params['price']:.4f} × price")
    print(f"         + {model_multi.params['promotion']:.4f} × promotion")

    print(f"\n广告投入的系数变化：")
    print(f"  简单回归: {model_simple.params['ad_spend']:.4f}")
    print(f"  多元回归: {model_multi.params['ad_spend']:.4f}")
    print(f"  差异: {model_simple.params['ad_spend'] - model_multi.params['ad_spend']:.4f}")

    print("\n解读：")
    print("  - 简单回归中，广告投入的系数'吸收'了价格的影响")
    print("    （因为广告投入和价格相关）")
    print("  - 多元回归中，价格的影响被单独分出来")
    print("  - 多元回归系数 = '控制其他变量后'的影响")
    print("    （保持价格和促销不变，广告投入每增加 1 万，销售增加多少）")


def bad_example_multicollinearity(df: pd.DataFrame) -> None:
    """❌ 坏例子：忽略多重共线性"""
    print("\n" + "=" * 70)
    print("❌ 坏例子：忽略多重共线性")
    print("=" * 70)

    # 拟合包含高度相关变量的模型
    X = df[['price', 'competitor_price']]
    y = df['sales']

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    print("\n模型：sales ~ price + competitor_price")
    print("\n回归结果：")
    print(f"  R² = {model.rsquared:.4f}（很高！）")

    print("\n系数：")
    for var in ['price', 'competitor_price']:
        coef = model.params[var]
        p_val = model.pvalues[var]
        print(f"  {var}: {coef:.4f} (p = {p_val:.4f})")

    print("\n问题：")
    print("  - R² 很高（> 0.7）")
    print("  - 但单个系数都不显著（p > 0.05）")
    print("  - 系数符号可能反直觉（比如价格系数为正）")
    print("  → 这是多重共线性的典型症状！")

    print("\n为什么？")
    print("  - price 和 competitor_price 高度相关（r > 0.9）")
    print("  - 模型无法区分两者的独立影响")
    print("  - 系数不稳定（小样本变化导致大变化）")


def good_example_check_vif(df: pd.DataFrame) -> None:
    """✅ 好例子：检查 VIF 并处理多重共线性"""
    print("\n" + "=" * 70)
    print("✅ 好例子：检查 VIF 并处理多重共线性")
    print("=" * 70)

    # 计算相关矩阵
    X = df[['ad_spend', 'price', 'promotion', 'competitor_price']]
    corr_matrix = X.corr()

    print("\n1. 相关系数矩阵：")
    print(corr_matrix.round(3))

    print("\n2. 检查高度相关的变量对：")
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

    if high_corr:
        print("   发现高度相关的变量对：")
        for var1, var2, corr_val in high_corr:
            print(f"   - {var1} vs {var2}: r = {corr_val:.3f}")
    else:
        print("   未发现高度相关的变量对（|r| > 0.7）")

    # 计算 VIF
    print("\n3. VIF（方差膨胀因子）：")

    def calculate_vif(X_df: pd.DataFrame) -> pd.DataFrame:
        """计算 VIF"""
        X_const = sm.add_constant(X_df)
        vif_data = []
        for i in range(1, X_const.shape[1]):  # 跳过截距
            vif = variance_inflation_factor(X_const.values, i)
            vif_data.append({
                'variable': X_df.columns[i-1],
                'VIF': vif,
                'sqrt_VIF': np.sqrt(vif),
                'diagnosis': '⚠️ 严重' if vif > 10 else '⚠️ 中等' if vif > 5 else '✅'
            })
        return pd.DataFrame(vif_data)

    # 所有变量
    vif_all = calculate_vif(X)
    print(vif_all.to_string(index=False))

    # 处理：删除 competitor_price
    print("\n4. 处理多重共线性：删除 competitor_price")
    X_reduced = df[['ad_spend', 'price', 'promotion']]
    vif_reduced = calculate_vif(X_reduced)
    print(vif_reduced.to_string(index=False))

    # 拟合处理后的模型
    y = df['sales']
    X_reduced_const = sm.add_constant(X_reduced)
    model_reduced = sm.OLS(y, X_reduced_const).fit()

    print("\n5. 处理后的模型：sales ~ ad_spend + price + promotion")
    print(f"   R² = {model_reduced.rsquared:.4f}")
    print(f"   调整 R² = {model_reduced.rsquared_adj:.4f}")

    print("\n系数：")
    for var in ['ad_spend', 'price', 'promotion']:
        coef = model_reduced.params[var]
        p_val = model_reduced.pvalues[var]
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        print(f"  {var}: {coef:.4f} (p = {p_val:.4f}) {sig}")

    print("\n结论：")
    print("  - 删除高度相关的变量后，VIF 降到安全范围")
    print("  - 系数变得显著且稳定")
    print("  - 调整 R² 可能略有下降，但模型更可靠")


def explain_adjusted_r2() -> None:
    """解释调整 R²"""
    print("\n" + "=" * 70)
    print("调整 R²：惩罚模型复杂度")
    print("=" * 70)

    print("\nR² 的问题：")
    print("  - R² 会随特征增加而增加")
    print("  - 即使加入无意义的噪声特征，R² 也会变大")
    print("  → R² 不能用于比较不同复杂度的模型")

    print("\n调整 R² 的公式：")
    print("  调整 R² = 1 - (1 - R²) × (n - 1) / (n - k - 1)")
    print("  其中：n = 样本量，k = 自变量数")

    print("\n调整 R² 的特点：")
    print("  - 惩罚模型复杂度（k 越大，惩罚越重）")
    print("  - 可能随特征增加而下降（如果新特征没有足够贡献）")
    print("  - 可用于比较不同复杂度的模型")

    print("\n经验法则：")
    print("  - 优先看调整 R²，而不是 R²")
    print("  - 如果 R² 上升但调整 R² 下降，新特征可能不有用")


def plot_multiple_regression(df: pd.DataFrame) -> None:
    """绘制多元回归相关图表"""
    setup_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 相关系数热图
    ax = axes[0, 0]
    X = df[['ad_spend', 'price', 'promotion', 'competitor_price']]
    corr = X.corr()

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1,
                cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_title('相关系数矩阵：识别多重共线性', fontsize=13)

    # 2. 系数对比（简单回归 vs 多元回归）
    ax = axes[0, 1]

    y = df['sales']

    # 简单回归系数
    simple_coefs = []
    for var in ['ad_spend', 'price']:
        X_var = sm.add_constant(df[var])
        model = sm.OLS(y, X_var).fit()
        simple_coefs.append(model.params[var])

    # 多元回归系数
    X_multi = sm.add_constant(df[['ad_spend', 'price', 'promotion']])
    model_multi = sm.OLS(y, X_multi).fit()
    multi_coefs = [model_multi.params['ad_spend'], model_multi.params['price']]

    variables = ['广告投入', '价格']
    x_pos = np.arange(len(variables))
    width = 0.35

    ax.bar(x_pos - width/2, simple_coefs, width, label='简单回归',
           color='#2E86AB', alpha=0.7)
    ax.bar(x_pos + width/2, multi_coefs, width, label='多元回归（控制其他变量）',
           color='#A23B72', alpha=0.7)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(variables)
    ax.set_ylabel('系数值', fontsize=12)
    ax.set_title('系数对比：简单回归 vs 多元回归', fontsize=13)
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. VIF 柱状图
    ax = axes[1, 0]

    X = df[['ad_spend', 'price', 'promotion', 'competitor_price']]
    vif_values = []
    for i in range(X.shape[1]):
        X_temp = sm.add_constant(X)
        vif = variance_inflation_factor(X_temp.values, i+1)
        vif_values.append(vif)

    colors = ['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_values]

    ax.barh(X.columns, vif_values, color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='阈值 = 10')

    for i, (var, vif) in enumerate(zip(X.columns, vif_values)):
        ax.text(vif + 0.5, i, f'{vif:.1f}',
                va='center', fontsize=10)

    ax.set_xlabel('VIF', fontsize=12)
    ax.set_title('VIF（方差膨胀因子）：检测多重共线性\n'
                 '红色 > 10: 严重，橙色 > 5: 中等，绿色 ≤ 5: 安全', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')

    # 4. R² vs 调整 R²（逐步添加变量）
    ax = axes[1, 1]

    variable_sets = [
        ['ad_spend'],
        ['ad_spend', 'price'],
        ['ad_spend', 'price', 'promotion'],
        ['ad_spend', 'price', 'promotion', 'competitor_price']
    ]

    r2_values = []
    adj_r2_values = []
    labels = ['广告', '广告+价格', '广告+价格+促销', '+竞争价格']

    for vars in variable_sets:
        X_temp = sm.add_constant(df[vars])
        model = sm.OLS(y, X_temp).fit()
        r2_values.append(model.rsquared)
        adj_r2_values.append(model.rsquared_adj)

    x_pos = np.arange(len(labels))

    ax.plot(x_pos, r2_values, marker='o', linewidth=2.5,
            label='R²', color='#2E86AB')
    ax.plot(x_pos, adj_r2_values, marker='s', linewidth=2.5,
            label='调整 R²', color='#A23B72')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('值', fontsize=12)
    ax.set_title('R² vs 调整 R²：逐步添加变量\n'
                 '注意：添加竞争价格后，R² 上升但调整 R² 下降', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 标注调整 R² 下降的点
    if len(adj_r2_values) > 1:
        for i in range(1, len(adj_r2_values)):
            if adj_r2_values[i] < adj_r2_values[i-1]:
                ax.scatter(i, adj_r2_values[i], s=200, color='red',
                          marker='x', zorder=5, linewidth=3)

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '05_multiple_regression.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("\n图表已保存到: images/05_multiple_regression.png")


def main() -> None:
    """主函数"""
    print("多元回归与多重共线性\n")

    # 生成数据
    df = generate_ecommerce_data(n=100, random_state=42)

    # 简单 vs 多元回归
    simple_vs_multiple_regression(df)

    # 坏例子 vs 好例子
    bad_example_multicollinearity(df)
    good_example_check_vif(df)

    # 调整 R²
    explain_adjusted_r2()

    # 绘图
    plot_multiple_regression(df)

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n关键要点：")
    print("  1. 多元回归系数 = '控制其他变量后'的影响")
    print("  2. 多重共线性：自变量高度相关导致系数不稳定")
    print("  3. VIF > 10：严重多重共线性，需要处理")
    print("  4. 解决方法：删除高度相关的变量、PCA、正则化")
    print("  5. 调整 R²：惩罚复杂度，可比较不同模型")
    print("\n模型选择：")
    print("  - 预测优先：用交叉验证（Week 10）")
    print("  - 解释优先：选调整 R² 高、系数显著的简单模型")
    print("  - 简洁原则：在拟合相近时，选更简单的模型")
    print("\n在报告中：")
    print("  ❌ 只报告 R² 和系数")
    print("  ✅ 报告调整 R²、VIF、系数显著性检验")
    print()


if __name__ == "__main__":
    main()
