"""
示例：模型诊断——Cook's 距离与 Breusch-Pagan 检验。

本例演示回归模型的诊断工具：
- Cook's 距离：识别高影响点（对回归系数影响大的观测）
- Breusch-Pagan 检验：检验等方差假设（异方差）
- 标准化残差：识别异常值
- 杠杆值（Leverage）：识别高杠杆点

运行方式：python3 chapters/week_09/examples/04_model_diagnostics.py
预期输出：
  - stdout 输出诊断结果
  - 展示如何识别和处理高影响点
  - 保存图表到 images/04_cooks_distance.png

核心概念：
  - Cook's 距离：衡量每个观测对回归系数的影响
  - 高影响点：Cook's 距离 > 4/n 的点需要关注
  - 异常值 vs 高影响点：残差大 vs 杠杆高
  - 异方差检验：Breusch-Pagan 检验
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
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


def generate_data_with_outlier(n: int = 50, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """生成包含异常值的数据"""
    np.random.seed(random_state)
    x = np.random.normal(loc=50, scale=10, size=n)
    y = 10 + 0.5 * x + np.random.normal(loc=0, scale=3, size=n)

    # 添加一个高影响点（极端 x 值）
    x = np.append(x, 120)
    y = np.append(y, 30)  # 这个点在回归线下面很远

    return x, y


def generate_data_with_influential_point(n: int = 50, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """生成包含高影响点的数据"""
    np.random.seed(random_state)
    x = np.random.normal(loc=50, scale=10, size=n)
    y = 10 + 0.5 * x + np.random.normal(loc=0, scale=3, size=n)

    # 添加一个高影响点（极端 x，且 y 值"拉动"回归线）
    x = np.append(x, 120)
    y = np.append(y, 100)  # 这个点会"拉动"回归线

    return x, y


def calculate_cooks_distance(model: sm.regression.linear_model.RegressionResultsWrapper) -> np.ndarray:
    """计算 Cook's 距离"""
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    return cooks_d


def diagnose_model(model: sm.regression.linear_model.RegressionResultsWrapper,
                   x: np.ndarray, y: np.ndarray) -> dict:
    """
    完整的模型诊断

    返回:
        dict 包含诊断结果
    """
    # 1. Cook's 距离
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(y)
    high_influence = np.where(cooks_d > threshold)[0]

    # 2. 标准化残差（绝对值 > 3 可能是异常值）
    residuals = model.resid
    standardized_residuals = influence.resid_studentized_internal
    outliers = np.where(np.abs(standardized_residuals) > 3)[0]

    # 3. 杠杆值（hat matrix 对角元素）
    leverage = influence.hat_matrix_diag
    # 杠杆值 > 2*(k+1)/n 需要关注（k 是自变量数）
    k = len(model.params) - 1  # 减去截距
    high_leverage = np.where(leverage > 2 * (k + 1) / len(y))[0]

    # 4. Breusch-Pagan 检验（异方差）
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)

    return {
        'cooks_d': cooks_d,
        'cooks_threshold': threshold,
        'high_influence': high_influence,
        'standardized_residuals': standardized_residuals,
        'outliers': outliers,
        'leverage': leverage,
        'high_leverage': high_leverage,
        'bp_stat': bp_stat,
        'bp_p': bp_p,
        'fitted': model.fittedvalues,
        'residuals': residuals
    }


def print_diagnostics(diag: dict, title: str) -> None:
    """打印诊断结果"""
    print("\n" + "=" * 70)
    print(f"模型诊断：{title}")
    print("=" * 70)

    # Cook's 距离
    print(f"\n1. Cook's 距离（高影响点）：")
    print(f"   阈值: {diag['cooks_threshold']:.4f} (4/n)")
    print(f"   高影响点数量: {len(diag['high_influence'])}")

    if len(diag['high_influence']) > 0:
        print(f"   高影响点索引: {diag['high_influence'].tolist()}")
        max_idx = np.argmax(diag['cooks_d'])
        print(f"   最大 Cook's 距离: {diag['cooks_d'][max_idx]:.4f} (索引 {max_idx})")
    else:
        print("   ✅ 未发现高影响点")

    # 标准化残差
    print(f"\n2. 标准化残差（异常值）：")
    print(f"   异常值数量 (|residual| > 3): {len(diag['outliers'])}")

    if len(diag['outliers']) > 0:
        print(f"   异常值索引: {diag['outliers'].tolist()}")
        print(f"   最大标准化残差: {diag['standardized_residuals'][diag['outliers'][0]]:.2f}")
    else:
        print("   ✅ 未发现异常值")

    # 杠杆值
    print(f"\n3. 杠杆值（高杠杆点）：")
    print(f"   高杠杆点数量: {len(diag['high_leverage'])}")

    if len(diag['high_leverage']) > 0:
        print(f"   高杠杆点索引: {diag['high_leverage'].tolist()}")
    else:
        print("   ✅ 未发现高杠杆点")

    # Breusch-Pagan 检验
    print(f"\n4. Breusch-Pagan 检验（异方差）：")
    print(f"   BP 统计量: {diag['bp_stat']:.4f}")
    print(f"   p 值: {diag['bp_p']:.4f}")

    if diag['bp_p'] > 0.05:
        print("   ✅ p > 0.05，不能拒绝等方差假设")
    else:
        print("   ⚠️ p < 0.05，拒绝等方差假设（存在异方差）")


def bad_example_delete_points() -> None:
    """❌ 坏例子：直接删除高影响点"""
    print("\n" + "=" * 70)
    print("❌ 坏例子：看到高影响点就删除")
    print("=" * 70)

    print("\n场景：发现 Cook's 距离大于阈值的点")
    print("\n错误做法：")
    print("  1. 直接删除这些点")
    print("  2. 重新拟合模型")
    print("  3. R² 从 0.75 提升到 0.85")
    print("  4. 报告：'模型很好，R² = 0.85'")

    print("\n问题：")
    print("  - 数据造假：为了提高 R² 而删点")
    print("  - 可能删除了真实但有价值的极端观测")
    print("  - 没有记录删除原因，结果不可复现")

    print("\n老潘的点评：")
    print("  '如果你为了提高 R² 而删点，这是学术不端。'")


def good_example_handle_points() -> None:
    """✅ 好例子：正确处理高影响点"""
    print("\n" + "=" * 70)
    print("✅ 好例子：正确处理高影响点")
    print("=" * 70)

    print("\n场景：发现 Cook's 距离大于阈值的点")
    print("\n正确做法：")
    print("  1. 检查是否为录入错误")
    print("     → 如果是，修正或删除并记录")
    print("  2. 检查是否为真实极端值")
    print("     → 如果是，保留并报告")
    print("  3. 比较删除前后的模型")
    print("     → 如果结论改变，报告敏感性分析")
    print("  4. 考虑稳健回归（RLM）")
    print("     → 对异常值不敏感的方法")

    print("\n报告示例：")
    print("  '模型拟合结果为 β = 0.5 (p < 0.001)。'")
    print("  '诊断发现 1 个高影响点（索引 50，Cook's D = 0.15）。'")
    print("  '删除该点后，β = 0.6 (p < 0.001)，结论方向一致。'")
    print("  '保留该点的主结果：β = 0.5 [95% CI: 0.4, 0.6]。'")

    print("\n老潘的点评：")
    print("  '诚实报告比高 R² 更重要。'")


def plot_cooks_distance(x: np.ndarray, y: np.ndarray,
                         diag_normal: dict, diag_outlier: dict) -> None:
    """绘制 Cook's 距离图"""
    setup_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 正常数据：散点图 + 回归线
    ax = axes[0, 0]
    X_with_const = sm.add_constant(x)
    model_normal = sm.OLS(y, X_with_const).fit()

    ax.scatter(x, y, alpha=0.6, s=50, color='#2E86AB', label='观测值')
    x_range = np.linspace(x.min(), x.max(), 100)
    y_pred = model_normal.params[0] + model_normal.params[1] * x_range
    ax.plot(x_range, y_pred, color='#A23B72', linewidth=2.5, label='回归线')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('正常数据：散点图与回归线', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. 正常数据：Cook's 距离
    ax = axes[0, 1]
    cooks_d = diag_normal['cooks_d']
    threshold = diag_normal['cooks_threshold']

    ax.stem(np.arange(len(cooks_d)), cooks_d, linefmt='-', markerfmt='.',
             basefmt=' ')
    ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                label=f'阈值 = {threshold:.3f}')
    ax.set_xlabel('观测索引', fontsize=12)
    ax.set_ylabel("Cook's 距离", fontsize=12)
    ax.set_title("正常数据：Cook's 距离", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 带异常值数据：散点图 + 回归线
    ax = axes[1, 0]
    x_out, y_out = generate_data_with_outlier(n=50, random_state=42)

    # 正常点
    ax.scatter(x_out[:-1], y_out[:-1], alpha=0.6, s=50,
                color='#2E86AB', label='正常点')
    # 异常值
    ax.scatter(x_out[-1], y_out[-1], s=200, color='red',
                marker='*', label='高影响点', zorder=5)

    X_out_const = sm.add_constant(x_out)
    model_out = sm.OLS(y_out, X_out_const).fit()

    x_range_out = np.linspace(x_out.min(), x_out.max(), 100)
    y_pred_out = model_out.params[0] + model_out.params[1] * x_range_out

    # 不带异常值的回归线（虚线）
    X_normal_const = sm.add_constant(x_out[:-1])
    model_normal_only = sm.OLS(y_out[:-1], X_normal_const).fit()
    y_pred_normal = (model_normal_only.params[0] +
                     model_normal_only.params[1] * x_range_out)

    ax.plot(x_range_out, y_pred_out, color='#A23B72',
            linewidth=2.5, label='含异常值的回归线')
    ax.plot(x_range_out, y_pred_normal, color='green',
            linewidth=2.5, linestyle='--', label='不含异常值的回归线')

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('带异常值数据：回归线被"拉动"', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. 带异常值数据：Cook's 距离
    ax = axes[1, 1]
    cooks_d_out = diag_outlier['cooks_d']
    threshold_out = diag_outlier['cooks_threshold']

    colors = ['#2E86AB'] * len(cooks_d_out)
    colors[-1] = 'red'

    ax.stem(np.arange(len(cooks_d_out)), cooks_d_out, linefmt='-',
            markerfmt='.', basefmt=' ')

    # 标记高影响点
    high_inf = diag_outlier['high_influence']
    if len(high_inf) > 0:
        ax.scatter(high_inf, cooks_d_out[high_inf], s=200, color='red',
                   marker='*', zorder=5, label='高影响点')

    ax.axhline(y=threshold_out, color='red', linestyle='--', linewidth=2,
                label=f'阈值 = {threshold_out:.3f}')
    ax.set_xlabel('观测索引', fontsize=12)
    ax.set_ylabel("Cook's 距离", fontsize=12)
    ax.set_title("带异常值数据：Cook's 距离识别高影响点", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '04_cooks_distance.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print("\n图表已保存到: images/04_cooks_distance.png")


def explain_outlier_vs_influence() -> None:
    """解释异常值 vs 高影响点"""
    print("\n" + "=" * 70)
    print("异常值 vs 高影响点")
    print("=" * 70)

    print("\n异常值（Outlier）：")
    print("  - 定义：y 值远离其他点（残差大）")
    print("  - 识别：标准化残差 |residual| > 3")
    print("  - 影响：可能增加残差方差，但不一定影响回归系数")

    print("\n高影响点（Influential Point）：")
    print("  - 定义：删除后会显著改变回归系数的点")
    print("  - 识别：Cook's 距离 > 4/n")
    print("  - 来源：高杠杆（x 极端）或大残差（y 极端）")

    print("\n杠杆值（Leverage）：")
    print("  - 定义：x 值远离 x 的均值（极端 x）")
    print("  - 识别：hat matrix 对角元素 > 2*(k+1)/n")
    print('  - 影响：高杠杆点对回归线有"拉动"作用')

    print("\n关系：")
    print("  - 高杠杆 + 大残差 = 高影响点（最危险）")
    print("  - 高杠杆 + 小残差 = 高杠杆但影响小")
    print("  - 低杠杆 + 大残差 = 异常值但影响小")


def main() -> None:
    """主函数"""
    print("模型诊断：Cook's 距离与 Breusch-Pagan 检验\n")

    # 生成数据
    x_normal, y_normal = generate_data_with_outlier(n=50, random_state=42)
    # 拟合模型
    X_normal = sm.add_constant(x_normal)
    model_normal = sm.OLS(y_normal, X_normal).fit()

    x_outlier, y_outlier = generate_data_with_influential_point(n=50, random_state=42)
    X_outlier = sm.add_constant(x_outlier)
    model_outlier = sm.OLS(y_outlier, X_outlier).fit()

    # 诊断
    diag_normal = diagnose_model(model_normal, x_normal, y_normal)
    diag_outlier = diagnose_model(model_outlier, x_outlier, y_outlier)

    # 打印结果
    print_diagnostics(diag_normal, "正常数据")
    print_diagnostics(diag_outlier, "带高影响点数据")

    # 坏例子 vs 好例子
    bad_example_delete_points()
    good_example_handle_points()

    # 解释概念
    explain_outlier_vs_influence()

    # 绘图
    # 重新生成正常数据用于绘图
    x_plot, y_plot = generate_data_with_outlier(n=50, random_state=42)
    X_plot = sm.add_constant(x_plot)
    model_plot = sm.OLS(y_plot, X_plot).fit()

    diag_normal_plot = diagnose_model(model_plot, x_plot, y_plot)

    x_out_plot, y_out_plot = generate_data_with_influential_point(n=50, random_state=42)
    X_out_plot = sm.add_constant(x_out_plot)
    model_out_plot = sm.OLS(y_out_plot, X_out_plot).fit()

    diag_outlier_plot = diagnose_model(model_out_plot, x_out_plot, y_out_plot)

    plot_cooks_distance(x_plot, y_plot, diag_normal_plot, diag_outlier_plot)

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n关键要点：")
    print("  1. Cook's 距离：衡量每个观测对回归系数的影响")
    print("  2. 阈值：4/n，超过的点需要关注")
    print("  3. 异常值 ≠ 高影响点：前者 y 极端，后者对系数影响大")
    print('  4. 杠杆值：x 极端的点，有"拉动"回归线的潜力')
    print("  5. Breusch-Pagan 检验：检验异方差")
    print("\n处理高影响点的决策流程：")
    print("  1. 检查是否录入错误 → 修正或删除")
    print("  2. 检查是否真实极端值 → 保留并报告")
    print("  3. 比较删除前后模型 → 报告敏感性分析")
    print("  4. 考虑稳健回归 → 对异常值不敏感")
    print("\n老潘的经验法则：")
    print("  '不要为了高 R² 删点。'")
    print("  '诚实报告比好看的结果更重要。'")
    print()


if __name__ == "__main__":
    main()
