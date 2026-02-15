"""
示例：解读回归系数与 R²——从"拟合模型"到"解释模型"。

本例演示如何正确解读回归分析的结果：
- 截距（intercept）：x = 0 时 y 的预测值
- 斜率（slope）：x 每增加 1 单位，y 的变化量
- R²：模型解释的方差比例
- 系数的显著性检验：t 检验和 p 值

运行方式：python3 chapters/week_09/examples/02_interpret_coefficients.py
预期输出：
  - stdout 输出回归系数的详细解读
  - 展示 statsmodels 的完整输出
  - 保存图表到 images/02_coefficient_interpretation.png

核心概念：
  - 回归系数的解读：斜率表示"影响大小"，截距表示"基准值"
  - R² 的含义：模型解释了 y 的多少变异
  - t 检验：检验"系数是否显著不为 0"
  - 置信区间：系数的不确定性范围
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
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
    生成模拟电商数据：广告投入、价格、促销、销售额

    参数:
        n: 样本量
        random_state: 随机种子

    返回:
        DataFrame 包含 ad_spend, price, promotion, sales
    """
    np.random.seed(random_state)

    ad_spend = np.random.normal(loc=50, scale=15, size=n)
    price = np.random.normal(loc=100, scale=20, size=n)
    promotion = np.random.binomial(n=1, p=0.3, size=n)  # 30% 有促销

    # 销售额 = 基准 + 0.5*广告 - 0.3*价格 + 10*促销 + 噪声
    sales = 20 + 0.5 * ad_spend - 0.3 * price + 10 * promotion
    sales += np.random.normal(loc=0, scale=5, size=n)

    return pd.DataFrame({
        'ad_spend': ad_spend,
        'price': price,
        'promotion': promotion,
        'sales': sales
    })


def interpret_simple_regression(df: pd.DataFrame) -> None:
    """简单线性回归：广告投入 -> 销售额"""
    print("=" * 70)
    print("简单线性回归：广告投入对销售额的影响")
    print("=" * 70)

    X = df['ad_spend']
    y = df['sales']

    # 添加截距项
    X_with_const = sm.add_constant(X)

    # 拟合 OLS 模型
    model = sm.OLS(y, X_with_const).fit()

    print("\n回归方程:")
    intercept = model.params['const']
    slope = model.params['ad_spend']
    print(f"  sales = {intercept:.2f} + {slope:.4f} × ad_spend")

    print("\n系数解读:")
    print(f"  截距 ({intercept:.2f}):")
    print(f"    → 广告投入为 0 时，销售额的预测值")
    print(f"    → 注意：这可能没有实际意义（广告投入为 0 不在数据范围内）")

    print(f"\n  斜率 ({slope:.4f}):")
    print(f"    → 广告投入每增加 1 万元，销售额平均增加 {slope:.2f} 万元")
    print(f"    → 斜率 > 0：正相关（投入越多，销售越高）")

    # 显著性检验
    t_stat = model.tvalues['ad_spend']
    p_value = model.pvalues['ad_spend']

    print(f"\n  t 检验:")
    print(f"    → H0: 斜率 = 0（广告投入对销售额没有影响）")
    print(f"    → H1: 斜率 ≠ 0（广告投入对销售额有影响）")
    print(f"    → t = {t_stat:.4f}, p = {p_value:.4f}")

    if p_value < 0.001:
        print(f"    → p < 0.001：拒绝 H0，广告投入对销售额有显著影响 ***")
    elif p_value < 0.01:
        print(f"    → p < 0.01：拒绝 H0，广告投入对销售额有显著影响 **")
    elif p_value < 0.05:
        print(f"    → p < 0.05：拒绝 H0，广告投入对销售额有显著影响 *")
    else:
        print(f"    → p >= 0.05：不能拒绝 H0，影响不显著")

    # 置信区间
    ci_low, ci_high = model.conf_int().loc['ad_spend']
    print(f"\n  95% 置信区间:")
    print(f"    → [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"    → 解读：我们有 95% 的信心，真实斜率在 {ci_low:.4f} 到 {ci_high:.4f} 之间")

    # R²
    print(f"\n  R² = {model.rsquared:.4f}:")
    print(f"    → 模型解释了销售额 {model.rsquared*100:.1f}% 的变异")
    print(f"    → 剩余 {100-model.rsquared*100:.1f}% 的变异由其他因素或随机性解释")

    print("\n" + "-" * 70)
    print("完整回归结果:")
    print("-" * 70)
    print(model.summary().tables[1])


def bad_example_interpretation() -> None:
    """❌ 坏例子：错误解读"""
    print("\n" + "=" * 70)
    print("❌ 坏例子：错误解读回归系数")
    print("=" * 70)

    print("\n假设回归结果: sales = 20 + 0.5 × ad_spend")
    print("\n错误解读 1：'广告投入每增加 1 万元，销售额增加 0.5 万元，'")
    print("              '所以广告投入决定销售额。'")
    print("  → 问题：相关不等于因果")

    print("\n错误解读 2：'R² = 0.56，模型很好，假设肯定满足。'")
    print("  → 问题：高 R² 不等于假设满足")

    print("\n错误解读 3：'p < 0.001，所以广告投入的影响很大。'")
    print("  → 问题：显著性不等于效应量大小（需要看系数和置信区间）")

    print("\n错误解读 4：'截距 = 20，说明不做广告也有 20 万销售额。'")
    print("  → 问题：截距可能外推到数据范围之外")


def good_example_interpretation() -> None:
    """✅ 好例子：正确解读"""
    print("\n" + "=" * 70)
    print("✅ 好例子：正确解读回归系数")
    print("=" * 70)

    print("\n假设回归结果: sales = 20 + 0.5 × ad_spend (p < 0.001, 95% CI [0.4, 0.6])")
    print("\n正确解读：")
    print("\n1. 系数解读：")
    print("   '广告投入每增加 1 万元，销售额平均增加 0.5 万元")
    print("    （95% CI: [0.4, 0.6], p < 0.001）。'")
    print("   → 说清楚：'平均增加'（不是精确的）")
    print("   → 给出不确定性：置信区间")
    print("   → 给出显著性：p 值")

    print("\n2. R² 解读：")
    print("   '模型解释了销售额 56% 的变异（R² = 0.56）。'")
    print("   → 不说'模型很好'（需要看假设检验）")

    print("\n3. 因果谨慎：")
    print("   '我们观测到广告投入和销售额正相关，'")
    print("   '但不能断定因果关系（可能有其他混淆因素）。'")

    print("\n4. 模型局限：")
    print("   '模型的假设需要验证（线性、独立性、正态性、等方差）。'")


def interpret_multiple_regression(df: pd.DataFrame) -> None:
    """多元回归：多个自变量"""
    print("\n" + "=" * 70)
    print("多元回归：控制其他变量后的影响")
    print("=" * 70)

    X = df[['ad_spend', 'price', 'promotion']]
    y = df['sales']

    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    print("\n回归方程:")
    print(f"  sales = {model.params['const']:.2f}")
    print(f"         + {model.params['ad_spend']:.4f} × ad_spend")
    print(f"         + {model.params['price']:.4f} × price")
    print(f"         + {model.params['promotion']:.4f} × promotion")

    print("\n系数解读（控制其他变量后）：")
    for var in ['ad_spend', 'price', 'promotion']:
        coef = model.params[var]
        p_val = model.pvalues[var]
        ci_low, ci_high = model.conf_int().loc[var]

        print(f"\n  {var}:")
        print(f"    → 系数 = {coef:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}], p = {p_val:.4f})")

        if var == 'ad_spend':
            print(f"    → 控制价格和促销后，广告投入每增加 1 万元，")
            print(f"       销售额平均增加 {coef:.2f} 万元")
        elif var == 'price':
            print(f"    → 控制广告和促销后，价格每增加 1 元，")
            print(f"       销售额平均变化 {coef:.2f} 万元")
            print(f"    → 负系数：价格越高，销售越低")
        elif var == 'promotion':
            print(f"    → 控制广告和价格后，有促销比无促销，")
            print(f"       销售额平均增加 {coef:.2f} 万元")

        # 显著性标记
        if p_val < 0.001:
            print(f"    → p < 0.001: 影响显著 ***")
        elif p_val < 0.01:
            print(f"    → p < 0.01: 影响显著 **")
        elif p_val < 0.05:
            print(f"    → p < 0.05: 影响显著 *")
        else:
            print(f"    → p >= 0.05: 影响不显著")

    print(f"\n  R² = {model.rsquared:.4f}（调整 R² = {model.rsquared_adj:.4f}）")
    print(f"    → 多元回归模型解释了销售额 {model.rsquared*100:.1f}% 的变异")


def plot_coefficient_interpretation(df: pd.DataFrame) -> None:
    """绘制系数解读图"""
    setup_chinese_font()

    # 简单回归
    X_simple = df['ad_spend']
    y = df['sales']
    X_simple_const = sm.add_constant(X_simple)
    model_simple = sm.OLS(y, X_simple_const).fit()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 散点图 + 回归线
    ax = axes[0, 0]
    ax.scatter(df['ad_spend'], df['sales'], alpha=0.6, s=50, color='#2E86AB')

    # 绘制回归线
    x_range = np.linspace(df['ad_spend'].min(), df['ad_spend'].max(), 100)
    y_pred = model_simple.params['const'] + model_simple.params['ad_spend'] * x_range
    ax.plot(x_range, y_pred, color='#A23B72', linewidth=2.5,
            label=f"y = {model_simple.params['const']:.2f} + {model_simple.params['ad_spend']:.2f}x")

    # 标注斜率
    ax.annotate('', xy=(70, 50), xytext=(50, 40),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(60, 42, f'斜率 = {model_simple.params["ad_spend"]:.2f}\n(每增加 1 万，增加 {model_simple.params["ad_spend"]:.2f} 万)',
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    ax.set_xlabel('广告投入（万元）', fontsize=12)
    ax.set_ylabel('销售额（万元）', fontsize=12)
    ax.set_title('回归系数解读：斜率的含义', fontsize=13)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)

    # 2. R² 可视化
    ax = axes[0, 1]
    y_pred_full = model_simple.predict(X_simple_const)
    residuals = y - y_pred_full

    ax.scatter(y_pred_full, residuals, alpha=0.6, s=50, color='#2E86AB')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('预测值', fontsize=12)
    ax.set_ylabel('残差', fontsize=12)
    ax.set_title(f'残差图（R² = {model_simple.rsquared:.3f}）\n'
                 f'模型解释了 {model_simple.rsquared*100:.1f}% 的变异', fontsize=13)
    ax.grid(True, alpha=0.3)

    # 3. 系数置信区间
    ax = axes[1, 0]
    params = model_simple.params
    conf_int = model_simple.conf_int()

    variables = ['截距', '斜率']
    x_pos = np.arange(len(variables))

    ax.errorbar(x_pos, params.values,
                yerr=[params.values - conf_int[0], conf_int[1] - params.values],
                fmt='o', capsize=10, capthick=2, linewidth=2.5,
                markersize=10, color='#2E86AB', ecolor='#A23B72')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(variables, fontsize=11)
    ax.set_ylabel('系数值', fontsize=12)
    ax.set_title('回归系数及其 95% 置信区间', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for i, (var, val) in enumerate(zip(variables, params.values)):
        ci_low, ci_high = conf_int.iloc[i]
        ax.text(i, val + 1, f'{val:.2f}\n[{ci_low:.2f}, {ci_high:.2f}]',
                ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # 4. 多元回归系数对比
    ax = axes[1, 1]

    # 多元回归
    X_multi = df[['ad_spend', 'price', 'promotion']]
    X_multi_const = sm.add_constant(X_multi)
    model_multi = sm.OLS(y, X_multi_const).fit()

    variables = ['广告投入', '价格', '促销']
    var_keys = ['ad_spend', 'price', 'promotion']
    x_pos = np.arange(len(variables))
    params_multi = model_multi.params[var_keys]
    conf_int_multi = model_multi.conf_int().loc[var_keys]
    p_values_multi = model_multi.pvalues[var_keys]

    colors = ['green' if p < 0.05 else 'gray' for p in p_values_multi]

    ax.barh(x_pos, params_multi.values, color=colors, alpha=0.7, edgecolor='black')

    # 添加误差条
    ax.errorbar(params_multi.values, x_pos,
                xerr=[params_multi.values - conf_int_multi[0], conf_int_multi[1] - params_multi.values],
                fmt='none', ecolor='black', capsize=5, linewidth=2)

    ax.set_yticks(x_pos)
    ax.set_yticklabels(variables, fontsize=11)
    ax.set_xlabel('系数值', fontsize=12)
    ax.set_title('多元回归：控制其他变量后的影响\n'
                 '(绿色 = 显著, 灰色 = 不显著)', fontsize=13)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    # 标注显著性
    for i, var_key in enumerate(var_keys):
        p_val = model_multi.pvalues[var_key]
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.text(params_multi.values[i], i, f'  {sig}',
                va='center', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '02_coefficient_interpretation.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("\n图表已保存到: images/02_coefficient_interpretation.png")


def summarize_r_squared() -> None:
    """总结 R² 的解读"""
    print("\n" + "=" * 70)
    print("R² 的正确解读")
    print("=" * 70)

    print("\nR² 是什么？")
    print("  → R² = 1 - (SS_res / SS_tot)")
    print("  → SS_res：残差平方和（模型无法解释的变异）")
    print("  → SS_tot：总平方和（y 的总变异）")
    print("  → R²：模型解释的变异比例")

    print("\nR² 的经验法则：")
    print("  - R² < 0.3：模型解释力很弱，可能需要更多特征或换模型")
    print("  - 0.3 ≤ R² < 0.7：模型有一定解释力，但还有改进空间")
    print("  - R² ≥ 0.7：模型解释力较强（但也要检查假设）")

    print("\nR² 的陷阱：")
    print("  ❌ R² 高不等于假设满足（需要残差诊断）")
    print("  ❌ R² 会随特征增加而增加（需要看调整 R²）")
    print("  ❌ R² 无法判断因果关系（高相关不等于因果）")

    print("\n记住：R² 只是参考，不是金标准！")


def main() -> None:
    """主函数"""
    print("解读回归系数与 R²\n")

    # 生成数据
    df = generate_ecommerce_data(n=100, random_state=42)

    # 简单回归
    interpret_simple_regression(df)

    # 坏例子 vs 好例子
    bad_example_interpretation()
    good_example_interpretation()

    # 多元回归
    interpret_multiple_regression(df)

    # R² 总结
    summarize_r_squared()

    # 绘图
    plot_coefficient_interpretation(df)

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n关键要点：")
    print("  1. 斜率：x 每增加 1 单位，y 的平均变化量")
    print("  2. 截距：x = 0 时 y 的预测值（注意外推问题）")
    print("  3. t 检验：检验'系数是否显著不为 0'")
    print("  4. 95% CI：系数的不确定性范围")
    print("  5. R²：模型解释的变异比例（不是模型好坏的唯一标准）")
    print("\n在报告中：")
    print("  ❌ '系数是 0.5，p < 0.05'")
    print("  ✅ '广告投入每增加 1 万元，销售额平均增加 0.5 万元")
    print("      （95% CI: [0.40, 0.60], t = 8.5, p < 0.001）'")
    print()


if __name__ == "__main__":
    main()
