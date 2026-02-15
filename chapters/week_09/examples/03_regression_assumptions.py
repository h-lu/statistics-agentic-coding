"""
示例：回归假设检查——LINE 假设。

本例演示如何检查回归模型的四大假设（LINE）：
- L（Linear）：线性——y 和 x 之间是线性关系
- I（Independence）：独立性——残差之间相互独立
- N（Normal）：正态性——残差服从正态分布
- E（Equal variance）：等方差——残差的方差恒定

运行方式：python3 chapters/week_09/examples/03_regression_assumptions.py
预期输出：
  - stdout 输出假设检验的结果
  - 展示满足假设 vs 违反假设的对比
  - 保存图表到 images/03_residuals_vs_fitted.png 和 images/03_qq_plot.png

核心概念：
  - 残差图（Residuals vs Fitted）：检查线性和等方差
  - QQ 图（Q-Q Plot）：检查正态性
  - Durbin-Watson 统计量：检查独立性
  - 假设违反的后果和解决方法
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


def generate_good_model_data(n: int = 100, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """生成满足假设的数据"""
    np.random.seed(random_state)
    x = np.random.normal(loc=50, scale=15, size=n)
    y = 10 + 0.5 * x + np.random.normal(loc=0, scale=5, size=n)
    return x, y


def generate_bad_model_data(n: int = 100, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """生成违反假设的数据（非线性关系）"""
    np.random.seed(random_state)
    x = np.random.uniform(low=0, high=100, size=n)
    # 二次关系：y = 10 + 0.1*x + 0.005*x² + 噪声
    y = 10 + 0.1 * x + 0.005 * x ** 2 + np.random.normal(loc=0, scale=5, size=n)
    return x, y


def generate_heteroscedastic_data(n: int = 100, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """生成违反等方差假设的数据（异方差）"""
    np.random.seed(random_state)
    x = np.random.uniform(low=10, high=100, size=n)
    # 方差随 x 增加：noise ~ N(0, 0.1*x)
    noise = np.random.normal(loc=0, scale=0.1 * x, size=n)
    y = 10 + 0.5 * x + noise
    return x, y


def fit_and_check_assumptions(x: np.ndarray, y: np.ndarray, title: str) -> dict:
    """
    拟合模型并检查假设

    返回:
        dict 包含模型和诊断结果
    """
    X_with_const = sm.add_constant(x)
    model = sm.OLS(y, X_with_const).fit()

    residuals = model.resid
    fitted = model.fittedvalues

    # 1. 线性假设：通过残差图判断
    # （这里不做正式检验，依赖图形）

    # 2. 独立性假设：Durbin-Watson 检验
    # DW ≈ 2 表示无自相关，DW < 1 或 > 3 表示有问题
    dw = sm.stats.stattools.durbin_watson(residuals)

    # 3. 正态性假设：Shapiro-Wilk 检验
    shapiro_stat, shapiro_p = stats.shapiro(residuals)

    # 4. 等方差假设：Breusch-Pagan 检验
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)

    return {
        'model': model,
        'residuals': residuals,
        'fitted': fitted,
        'dw': dw,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'bp_stat': bp_stat,
        'bp_p': bp_p,
        'title': title
    }


def print_assumption_check(results: dict) -> None:
    """打印假设检查结果"""
    print("\n" + "=" * 70)
    print(f"假设检查：{results['title']}")
    print("=" * 70)

    print(f"\n模型摘要：")
    print(f"  R² = {results['model'].rsquared:.4f}")
    print(f"  F 检验: F = {results['model'].fvalue:.2f}, p = {results['model'].f_pvalue:.4f}")

    print("\n" + "-" * 70)
    print("LINE 假设检查")
    print("-" * 70)

    # L: 线性
    print("\nL（Linear）线性假设：")
    print("  检查方法：残差 vs 拟合值图")
    print("  判断标准：残差应在 0 上下随机分布，无明显模式")
    print("  → 需要查看图形判断")

    # I: 独立性
    print("\nI（Independence）独立性假设：")
    print(f"  Durbin-Watson 统计量 = {results['dw']:.4f}")
    if 1.5 <= results['dw'] <= 2.5:
        print("  → DW ≈ 2，独立性假设满足 ✅")
    else:
        print("  → DW 偏离 2，可能存在自相关 ⚠️")

    # N: 正态性
    print("\nN（Normal）正态性假设：")
    print(f"  Shapiro-Wilk 检验: W = {results['shapiro_stat']:.4f}, p = {results['shapiro_p']:.4f}")
    if results['shapiro_p'] > 0.05:
        print("  → p > 0.05，不能拒绝正态性假设 ✅")
    else:
        print("  → p < 0.05，拒绝正态性假设 ⚠️")

    # E: 等方差
    print("\nE（Equal variance）等方差假设：")
    print(f"  Breusch-Pagan 检验: BP = {results['bp_stat']:.4f}, p = {results['bp_p']:.4f}")
    if results['bp_p'] > 0.05:
        print("  → p > 0.05，不能拒绝等方差假设 ✅")
    else:
        print("  → p < 0.05，拒绝等方差假设（存在异方差）⚠️")


def plot_residuals_vs_fitted(results_list: list[dict]) -> None:
    """绘制残差 vs 拟合值图"""
    setup_chinese_font()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, results in zip(axes, results_list):
        residuals = results['residuals']
        fitted = results['fitted']

        # 散点
        ax.scatter(fitted, residuals, alpha=0.6, s=50, color='#2E86AB')

        # 添加 LOWESS 平滑线
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals, fitted, frac=0.66)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color='#A23B72',
                linewidth=2.5, label='LOWESS')

        # 0 线
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)

        ax.set_xlabel('拟合值', fontsize=12)
        ax.set_ylabel('残差', fontsize=12)
        ax.set_title(results['title'], fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 判断文本
        if '好模型' in results['title']:
            judgment = '✅ 假设满足：残差随机分布'
        elif '非线性' in results['title']:
            judgment = '⚠️ 违反线性：残差有U型模式'
        else:  # 异方差
            judgment = '⚠️ 违反等方差：残差宽度变化'

        ax.text(0.5, 0.95, judgment, transform=ax.transAxes,
                fontsize=11, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    plt.suptitle('残差 vs 拟合值图：检查线性和等方差假设', fontsize=14, y=1.02)
    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '03_residuals_vs_fitted.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("\n图表已保存到: images/03_residuals_vs_fitted.png")


def plot_qq_plots(results_list: list[dict]) -> None:
    """绘制 QQ 图"""
    setup_chinese_font()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, results in zip(axes, results_list):
        residuals = results['residuals']

        # QQ 图
        stats.probplot(residuals, dist="norm", plot=ax)

        ax.set_title(results['title'], fontsize=13)
        ax.grid(True, alpha=0.3)

        # 判断文本
        shapiro_p = results['shapiro_p']
        if shapiro_p > 0.05:
            judgment = f'✅ 正态性: p = {shapiro_p:.3f} > 0.05'
        else:
            judgment = f'⚠️ 非正态: p = {shapiro_p:.3f} < 0.05'

        ax.text(0.5, 0.05, judgment, transform=ax.transAxes,
                fontsize=10, ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.suptitle('QQ 图：检查正态性假设', fontsize=14, y=1.02)
    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '03_qq_plot.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("图表已保存到: images/03_qq_plot.png")


def bad_example_no_assumption_check() -> None:
    """❌ 坏例子：不做假设检查"""
    print("\n" + "=" * 70)
    print("❌ 坏例子：不做假设检查")
    print("=" * 70)

    print("\n报告：")
    print("  '广告投入对销售额有显著影响（β = 0.5, p < 0.001, R² = 0.75）。'")
    print("  '因此，我们建议增加广告投入以提高销售额。'")

    print("\n问题：")
    print("  - 没有检查残差图")
    print("  - 没有验证正态性")
    print("  - 没有检查等方差")
    print("  - 如果假设违反，p 值和 CI 都不可信")

    print("\n老潘的点评：")
    print("  '这不是分析，是自欺欺人。'")


def good_example_with_assumption_check() -> None:
    """✅ 好例子：做假设检查"""
    print("\n" + "=" * 70)
    print("✅ 好例子：做假设检查")
    print("=" * 70)

    print("\n报告：")
    print("  '广告投入对销售额有显著影响（β = 0.5, p < 0.001, R² = 0.75）。'")
    print("")
    print("  模型诊断：")
    print("  - 残差 vs 拟合值图显示残差随机分布，线性假设满足 ✅")
    print("  - QQ 图显示残差近似正态，Shapiro-Wilk p = 0.23 > 0.05 ✅")
    print("  - Breusch-Pagan 检验 p = 0.45 > 0.05，等方差假设满足 ✅")
    print("  - Durbin-Watson = 1.98，独立性假设满足 ✅")
    print("")
    print("  结论：模型假设满足，回归结果可信。'")

    print("\n老潘的点评：")
    print("  '这才是分析。'")


def explain_consequences() -> None:
    """解释假设违反的后果"""
    print("\n" + "=" * 70)
    print("假设违反的后果与解决方法")
    print("=" * 70)

    print("\n违反线性假设：")
    print("  后果：系数有偏，预测不准")
    print("  解决：加入二次项、对 x 转换、非线性模型")

    print("\n违反独立性假设：")
    print("  后果：p 值不可信，CI 不准确")
    print("  解决：时间序列模型、混合效应模型、聚类稳健标准误")

    print("\n违反正态性假设：")
    print("  后果：小样本时 p 值、CI 不准确")
    print("  解决：对 y 转换（对数、Box-Cox）、Bootstrap 稳健标准误")
    print("  注：大样本时回归对正态性偏离较稳健")

    print("\n违反等方差假设（异方差）：")
    print("  后果：p 值、CI 不准确")
    print("  解决：对 y 转换、稳健标准误（HC3）、加权最小二乘法（WLS）")


def main() -> None:
    """主函数"""
    print("回归假设检查：LINE 假设\n")

    # 生成不同类型的数据
    x_good, y_good = generate_good_model_data(n=100, random_state=42)
    x_bad, y_bad = generate_bad_model_data(n=100, random_state=43)
    x_hetero, y_hetero = generate_heteroscedastic_data(n=100, random_state=44)

    # 拟合模型并检查假设
    results_good = fit_and_check_assumptions(x_good, y_good, "好模型：满足假设")
    results_bad = fit_and_check_assumptions(x_bad, y_bad, "坏模型：违反线性（U型残差）")
    results_hetero = fit_and_check_assumptions(x_hetero, y_hetero, "坏模型：违反等方差（漏斗型）")

    results_list = [results_good, results_bad, results_hetero]

    # 打印结果
    for results in results_list:
        print_assumption_check(results)

    # 坏例子 vs 好例子
    bad_example_no_assumption_check()
    good_example_with_assumption_check()

    # 解释后果
    explain_consequences()

    # 绘图
    plot_residuals_vs_fitted(results_list)
    plot_qq_plots(results_list)

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n关键要点：")
    print("  1. 回归有四大假设：LINE（线性、独立性、正态性、等方差）")
    print("  2. 残差图：检查线性和等方差")
    print("  3. QQ 图：检查正态性")
    print("  4. Durbin-Watson：检查独立性")
    print("  5. 假设违反时，p 值和 CI 不可信")
    print("\n老潘的经验法则：")
    print("  '先画图，再做检验。图比检验更直观。'")
    print("  '小偏差可以容忍，严重违反要处理。'")
    print("\n在报告中：")
    print("  ❌ 只报告 R² 和系数")
    print("  ✅ 报告系数 + 假设检验结果 + 诊断图表")
    print()


if __name__ == "__main__":
    main()
