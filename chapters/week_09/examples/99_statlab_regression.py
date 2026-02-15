"""
示例：StatLab 超级线——带诊断的回归报告生成。

本例演示如何在 StatLab 报告中添加回归分析模块，包括：
- 模型拟合与系数解释
- 假设检验（LINE 假设）
- 模型诊断（残差图、QQ 图、Cook's 距离）
- 多重共线性检查（VIF）
- 自动生成 Markdown 报告

这是 StatLab 超级线在 Week 09 的进展，在上周（区间估计与重采样）的基础上
添加回归分析功能。

运行方式：python3 chapters/week_09/examples/99_statlab_regression.py
预期输出：
  - stdout 输出回归分析结果
  - 生成图表到 images/
  - 生成报告到 output/regression_report.md

核心概念：
  - 回归分析的完整流程：拟合 → 检验假设 → 诊断 → 报告
  - 可复现报告：脚本自动生成，可审计
  - 模型诊断是底线：没有诊断的回归不是分析
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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
from typing import Optional, Union


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


def regression_with_diagnostics(
    y: np.ndarray,
    X: Union[np.ndarray, pd.DataFrame],
    var_names: Optional[list[str]] = None,
    confidence: float = 0.95
) -> dict:
    """
    拟合回归模型并输出诊断报告

    参数:
        y: 因变量（array-like）
        X: 自变量矩阵（array-like, 每列是一个自变量）
        var_names: 自变量名称列表（可选）
        confidence: 置信水平（默认 0.95）

    返回:
        dict: 包含模型、诊断结果、图表数据的字典
    """
    # 转换为 DataFrame
    if isinstance(X, np.ndarray):
        if var_names is None:
            var_names = [f"x{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=var_names)
    else:
        X_df = X
        var_names = X_df.columns.tolist()

    # 添加截距
    X_with_const = sm.add_constant(X_df)

    # 拟合 OLS 模型
    model = sm.OLS(y, X_with_const).fit()

    # 1. 提取模型结果
    results = {
        "model": model,
        "summary": model.summary(),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "n_obs": len(y),
        "n_vars": len(var_names),
        "var_names": var_names
    }

    # 2. 置信区间
    ci = model.conf_int(alpha=1-confidence)
    ci.columns = ['lower', 'upper']
    results["confidence_intervals"] = ci

    # 3. 残差分析
    residuals = model.resid
    fitted = model.fittedvalues

    # 3.1 正态性检验（Shapiro-Wilk）
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    results["normality_test"] = {
        "statistic": float(shapiro_stat),
        "p_value": float(shapiro_p),
        "is_normal": shapiro_p > 0.05
    }

    # 3.2 同方差检验（Breusch-Pagan）
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, model.model.exog)
    results["homoscedasticity_test"] = {
        "statistic": float(bp_stat),
        "p_value": float(bp_p),
        "is_homoscedastic": bp_p > 0.05
    }

    # 4. 高影响点（Cook's 距离）
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    threshold = 4 / len(cooks_d)
    high_influence = np.where(cooks_d > threshold)[0]

    results["high_influence_points"] = {
        "cook_distances": cooks_d,
        "threshold": threshold,
        "high_influence_indices": high_influence.tolist(),
        "n_high_influence": len(high_influence)
    }

    # 5. 多重共线性（VIF）——如果是多元回归
    if X_df.shape[1] > 1:
        vif_data = []
        for i in range(X_df.shape[1]):
            vif = variance_inflation_factor(X_with_const.values, i+1)
            vif_data.append({
                "variable": var_names[i],
                "vif": float(vif),
                "is_high_collinear": vif > 10
            })
        results["multicollinearity"] = vif_data
    else:
        results["multicollinearity"] = []

    # 6. 图表数据
    results["plots"] = {
        "residuals_vs_fitted": {"x": fitted.values, "y": residuals.values},
        "qq_plot": {"residuals": residuals.values},
        "cook_distance": {"index": np.arange(len(cooks_d)),
                         "cook_d": cooks_d,
                         "threshold": threshold},
    }

    return results


def plot_diagnostics(diag_results: dict, figsize: tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    画诊断图表

    参数:
        diag_results: regression_with_diagnostics 的返回结果
        figsize: 图形大小

    返回:
        matplotlib Figure 对象
    """
    setup_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. 残差 vs 拟合值
    plot_data = diag_results["plots"]["residuals_vs_fitted"]
    axes[0, 0].scatter(plot_data["x"], plot_data["y"], alpha=0.6, s=50,
                       color='#2E86AB', edgecolors='black', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)

    # 添加 LOWESS 线
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(plot_data["y"], plot_data["x"], frac=0.66)
    axes[0, 0].plot(smoothed[:, 0], smoothed[:, 1], color='#A23B72',
                    linewidth=2.5, label='LOWESS')

    axes[0, 0].set_xlabel('拟合值', fontsize=12)
    axes[0, 0].set_ylabel('残差', fontsize=12)
    axes[0, 0].set_title('残差 vs 拟合值', fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. QQ 图
    residuals = diag_results["plots"]["qq_plot"]["residuals"]
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('正态 Q-Q 图', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Cook's 距离
    cook_data = diag_results["plots"]["cook_distance"]
    axes[1, 0].stem(cook_data["index"], cook_data["cook_d"],
                    linefmt='-', markerfmt='.', basefmt=' ')
    axes[1, 0].axhline(y=cook_data["threshold"], color='red',
                       linestyle='--', linewidth=2, label=f'阈值 = {cook_data["threshold"]:.3f}')

    # 标记高影响点
    high_inf = diag_results["high_influence_points"]["high_influence_indices"]
    if high_inf:
        axes[1, 0].scatter(high_inf, [cook_data["cook_d"][i] for i in high_inf],
                          s=150, color='red', marker='*', zorder=5,
                          label='高影响点')

    axes[1, 0].set_xlabel('观测索引', fontsize=12)
    axes[1, 0].set_ylabel("Cook's 距离", fontsize=12)
    axes[1, 0].set_title("Cook's 距离：识别高影响点", fontsize=13, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 4. 残差直方图
    axes[1, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7,
                    color='#2E86AB')

    # 添加正态分布曲线
    mu, std = residuals.mean(), residuals.std(ddof=1)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1, 1].plot(x, stats.norm.pdf(x, mu, std) * len(residuals) *
                    (residuals.max() - residuals.min()) / 20,
                    color='#A23B72', linewidth=2.5, label='正态分布')

    axes[1, 1].set_xlabel('残差', fontsize=12)
    axes[1, 1].set_ylabel('频数', fontsize=12)
    axes[1, 1].set_title('残差分布直方图', fontsize=13, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def format_regression_report(diag_results: dict, confidence: float = 0.95) -> str:
    """
    格式化回归结果为 Markdown 报告

    参数:
        diag_results: regression_with_diagnostics 的返回结果
        confidence: 置信水平

    返回:
        Markdown 格式的报告字符串
    """
    model = diag_results["model"]
    ci_pct = int(confidence * 100)

    md = ["## 回归分析\n\n"]

    # 1. 模型摘要
    md.append("### 模型拟合摘要\n\n")
    md.append("| 指标 | 值 |\n")
    md.append("|------|-----|\n")
    md.append(f"| 样本量 | {diag_results['n_obs']} |\n")
    md.append(f"| 自变量数 | {diag_results['n_vars']} |\n")
    md.append(f"| R² | {diag_results['r_squared']:.4f} |\n")
    md.append(f"| 调整 R² | {diag_results['adj_r_squared']:.4f} |\n")
    md.append(f"| F 统计量 | {diag_results['f_statistic']:.4f} |\n")
    md.append(f"| F 检验 p 值 | {diag_results['f_pvalue']:.4f} |\n\n")

    # 2. 回归系数
    md.append("### 回归系数\n\n")
    md.append(f"| 变量 | 系数 | 标准误 | t 值 | p 值 | {ci_pct}% CI |\n")
    md.append("|------|------|--------|------|------|-------|\n")

    ci = diag_results["confidence_intervals"]
    for idx in model.params.index:
        row = model.params[idx]
        se = model.bse[idx]
        t = model.tvalues[idx]
        p = model.pvalues[idx]
        lower = ci.loc[idx, 'lower']
        upper = ci.loc[idx, 'upper']

        # 显著性标记
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        md.append(f"| {idx} | {row:.4f} | {se:.4f} | {t:.4f} | {p:.4f} {sig} | "
                  f"[{lower:.4f}, {upper:.4f}] |\n")
    md.append("\n")

    # 3. 假设检验
    md.append("### 假设检验\n\n")

    # 正态性
    norm = diag_results["normality_test"]
    md.append(f"**正态性检验（Shapiro-Wilk）**：\n")
    md.append(f"- 统计量：{norm['statistic']:.4f}\n")
    md.append(f"- p 值：{norm['p_value']:.4f}\n")
    if norm['is_normal']:
        md.append(f"- 结论：p > 0.05，不能拒绝正态性假设 ✅\n\n")
    else:
        md.append(f"- 结论：p < 0.05，拒绝正态性假设 ⚠️\n\n")

    # 同方差
    homo = diag_results["homoscedasticity_test"]
    md.append(f"**同方差检验（Breusch-Pagan）**：\n")
    md.append(f"- 统计量：{homo['statistic']:.4f}\n")
    md.append(f"- p 值：{homo['p_value']:.4f}\n")
    if homo['is_homoscedastic']:
        md.append(f"- 结论：p > 0.05，不能拒绝同方差假设 ✅\n\n")
    else:
        md.append(f"- 结论：p < 0.05，拒绝同方差假设（存在异方差）⚠️\n\n")

    # 4. 高影响点
    md.append("### 高影响点（Cook's 距离）\n\n")
    inf = diag_results["high_influence_points"]
    md.append(f"- Cook's 距离阈值：{inf['threshold']:.4f}\n")
    md.append(f"- 高影响点数量：{inf['n_high_influence']}\n")
    if inf['n_high_influence'] > 0:
        md.append(f"- 高影响点索引：{inf['high_influence_indices']}\n")
        md.append(f"- ⚠️ 建议：检查这些数据点是否为录入错误或极端值\n\n")
    else:
        md.append(f"- ✅ 未发现高影响点\n\n")

    # 5. 多重共线性（如果有）
    if diag_results.get("multicollinearity"):
        md.append("### 多重共线性（VIF）\n\n")
        md.append("| 变量 | VIF | 诊断 |\n")
        md.append("|------|-----|------|\n")
        for vif in diag_results["multicollinearity"]:
            diagnosis = "⚠️ VIF > 10" if vif["is_high_collinear"] else "✅"
            md.append(f"| {vif['variable']} | {vif['vif']:.2f} | {diagnosis} |\n")
        md.append("\n")

    # 6. 结论
    md.append("### 结论\n\n")

    # 判断整体模型质量
    issues = []
    if not norm['is_normal']:
        issues.append("正态性假设违反")
    if not homo['is_homoscedastic']:
        issues.append("等方差假设违反")
    if inf['n_high_influence'] > 0:
        issues.append(f"存在 {inf['n_high_influence']} 个高影响点")
    if any(v['is_high_collinear'] for v in diag_results.get("multicollinearity", [])):
        issues.append("存在多重共线性")

    if issues:
        md.append("**⚠️ 注意**：模型诊断发现以下问题：\n")
        for issue in issues:
            md.append(f"- {issue}\n")
        md.append("\n建议在使用模型结论前谨慎处理这些问题。\n\n")
    else:
        md.append("**✅ 模型诊断通过**：所有假设检验均满足，模型结果可信。\n\n")

    return "".join(md)


def main() -> None:
    """主函数：演示 StatLab 回归报告生成"""
    print("=" * 70)
    print("StatLab 超级线：带诊断的回归报告生成")
    print("=" * 70)

    # 加载数据（使用 seaborn 的 penguins 数据集）
    print("\n加载数据...")
    penguins = sns.load_dataset("penguins")
    penguins = penguins.dropna()

    # 准备数据
    print("准备数据...")

    # 简单回归：喙长度预测喙深度
    print("\n" + "=" * 70)
    print("示例 1：简单回归——喙长度预测喙深度")
    print("=" * 70)

    X_simple = penguins[["bill_length_mm"]].values
    y = penguins["bill_depth_mm"].values

    diag_simple = regression_with_diagnostics(
        y, X_simple,
        var_names=["喙长度"],
        confidence=0.95
    )

    # 生成报告
    report_simple = format_regression_report(diag_simple)
    print("\n简单回归报告：")
    print(report_simple)

    # 画诊断图
    fig_simple = plot_diagnostics(diag_simple)

    # 保存图表
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    fig_simple.savefig(output_dir / '99_statlab_simple_diagnostics.png',
                       dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
    plt.close(fig_simple)
    print(f"\n诊断图已保存到: images/99_statlab_simple_diagnostics.png")

    # 多元回归：喙长度 + 翼展预测喙深度
    print("\n" + "=" * 70)
    print("示例 2：多元回归——喙长度 + 翼展预测喙深度")
    print("=" * 70)

    X_multiple = penguins[["bill_length_mm", "flipper_length_mm"]].values
    diag_multiple = regression_with_diagnostics(
        y, X_multiple,
        var_names=["喙长度", "翼展"],
        confidence=0.95
    )

    # 生成报告
    report_multiple = format_regression_report(diag_multiple)
    print("\n多元回归报告：")
    print(report_multiple)

    # 画诊断图
    fig_multiple = plot_diagnostics(diag_multiple)

    # 保存图表
    fig_multiple.savefig(output_dir / '99_statlab_multiple_diagnostics.png',
                        dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
    plt.close(fig_multiple)
    print(f"\n诊断图已保存到: images/99_statlab_multiple_diagnostics.png")

    # 保存报告到文件
    output_path = Path(__file__).parent.parent / 'output' / 'regression_report.md'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_multiple)

    print(f"\n报告已保存到: output/regression_report.md")

    # 与上周的对比
    print("\n" + "=" * 70)
    print("StatLab 进度：本周 vs 上周")
    print("=" * 70)

    print("\n上周（Week 08）：")
    print("  - 点估计 + 95% CI")
    print("  - Bootstrap 方法")
    print("  - 置换检验")
    print("  → 量化了'不确定性'")

    print("\n本周（Week 09）：")
    print("  - 回归分析（系数、R²、F 检验）")
    print("  - 假设检验（LINE 假设）")
    print("  - 模型诊断（残差图、QQ 图、Cook's 距离）")
    print("  - 多重共线性检查（VIF）")
    print("  → 量化了'关系' + 验证了'模型可信性'")

    print("\n老潘的点评：")
    print("  '上周你学会了说：均值差异是 2.1 [95% CI: 1.5, 2.7]。'")
    print("  '本周你学会了说：广告投入每增加 1 万，销售增加 0.5'")
    print("  '                [95% CI: 0.4, 0.6], p < 0.001，'")
    print("  '                且模型假设满足，诊断通过。'")
    print("  '这才是完整的分析。'")

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
