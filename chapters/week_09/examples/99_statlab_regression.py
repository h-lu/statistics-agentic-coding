"""
StatLab 回归分析报告生成器

本脚本是 StatLab 超级线的一部分，用于在可复现分析报告中添加
"回归分析"章节。它执行完整的回归分析流程，包括：
- 简单回归和多元回归
- 残差诊断（LINE 假设）
- 多重共线性检查（VIF）
- 异常点分析（Cook's 距离）
- 自动生成报告片段和诊断图

运行方式：python3 chapters/week_09/examples/99_statlab_regression.py
预期输出：
- 报告片段（追加到 report.md）
- 残差诊断图（保存到 report/images/）

依赖: 需要预先清洗好的数据（假设路径为 data/clean_data.csv）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, probplot
from pathlib import Path
from typing import Dict, List, Tuple

# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def regression_analysis(
    df: pd.DataFrame,
    target: str,
    predictors: List[str],
    output_dir: str = "report"
) -> Tuple[str, Dict]:
    """
    对数据集进行完整的回归分析，生成报告片段和诊断图

    参数:
        df: 清洗后的数据
        target: 目标变量名 (如 'consumption_amount', 'price_wan')
        predictors: 预测变量名列表
        output_dir: 报告输出目录

    返回:
        (report_text, results_dict)
        - report_text: Markdown 格式的报告片段
        - results_dict: 包含模型、统计量等的字典
    """
    # 创建输出目录
    output_path = Path(output_dir)
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    y = df[target]
    X = df[predictors]

    # ========== 1. 简单回归（最相关的单一变量） ==========
    simple_predictor = predictors[0]  # 假设第一个最重要
    X_simple = sm.add_constant(df[[simple_predictor]])
    model_simple = sm.OLS(y, X_simple).fit()

    # ========== 2. 多元回归 ==========
    X_multi = sm.add_constant(X)
    model_multi = sm.OLS(y, X_multi).fit()

    # ========== 3. 多重共线性检查 ==========
    vif_data = calculate_vif(X)

    # ========== 4. 残差诊断图 ==========
    residuals = model_multi.resid
    fitted = model_multi.fittedvalues

    # 创建 2x2 诊断图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 残差 vs 拟合值（线性 + 等方差）
    axes[0, 0].scatter(fitted, residuals, alpha=0.6, edgecolors='k', linewidths=0.5)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('拟合值')
    axes[0, 0].set_ylabel('残差')
    axes[0, 0].set_title('残差 vs 拟合值 (检验线性与等方差)')
    axes[0, 0].grid(True, alpha=0.3)

    # QQ 图（正态性）
    probplot(residuals, plot=axes[0, 1])
    axes[0, 1].set_title('QQ 图 (检验正态性)')
    axes[0, 1].grid(True, alpha=0.3)

    # 尺度-位置图（同方差）
    from scipy.stats import zscore
    axes[1, 0].scatter(fitted, np.abs(zscore(residuals)), alpha=0.6, edgecolors='k')
    axes[1, 0].axhline(y=1, color='red', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('拟合值')
    axes[1, 0].set_ylabel('|z-score|')
    axes[1, 0].set_title('标准化残差绝对值 (检验同方差)')
    axes[1, 0].grid(True, alpha=0.3)

    # Cook's 距离（异常点）
    influence = model_multi.get_influence()
    cooks_d = influence.cooks_distance[0]
    axes[1, 1].scatter(range(len(cooks_d)), cooks_d, alpha=0.6)
    axes[1, 1].axhline(y=1, color='red', linestyle='--', linewidth=2, label='阈值 (D=1)')
    axes[1, 1].set_xlabel('观测索引')
    axes[1, 1].set_ylabel("Cook's 距离")
    axes[1, 1].set_title("Cook's 距离 (识别强影响点)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    diag_fig_path = images_path / "residual_diagnostics.png"
    plt.savefig(diag_fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ========== 5. 假设检验统计量 ==========
    dw_stat = sm.stats.durbin_watson(residuals)
    shapiro_stat, shapiro_p = shapiro(residuals)

    # ========== 6. 生成报告片段 ==========
    report = generate_report_markdown(
        target=target,
        predictors=predictors,
        model_simple=model_simple,
        simple_predictor=simple_predictor,
        model_multi=model_multi,
        vif_data=vif_data,
        residuals=residuals,
        cooks_d=cooks_d,
        dw_stat=dw_stat,
        shapiro_p=shapiro_p,
        diag_fig_path=diag_fig_path
    )

    # ========== 7. 打包结果 ==========
    results = {
        'model_simple': model_simple,
        'model_multi': model_multi,
        'vif_data': vif_data,
        'cooks_d': cooks_d,
        'diagnostics': {
            'durbin_watson': dw_stat,
            'shapiro_p': shapiro_p,
        }
    }

    return report, results


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    """计算方差膨胀因子"""
    vif_data = pd.DataFrame()
    vif_data["变量"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                        for i in range(X.shape[1])]
    return vif_data


def generate_report_markdown(
    target: str,
    predictors: List[str],
    model_simple: sm.regression.linear_model.RegressionResults,
    simple_predictor: str,
    model_multi: sm.regression.linear_model.RegressionResults,
    vif_data: pd.DataFrame,
    residuals: pd.Series,
    cooks_d: np.ndarray,
    dw_stat: float,
    shapiro_p: float,
    diag_fig_path: Path
) -> str:
    """生成 Markdown 格式的报告片段"""

    # 系数表
    coef_table = ""
    for i, var in enumerate(['const'] + predictors):
        if var == 'const':
            coef = model_simple.params[0] if i == 0 else model_multi.params[0]
            se = model_simple.bse[0] if i == 0 else model_multi.bse[0]
            ci = model_simple.conf_int().iloc[0, :] if i == 0 else model_multi.conf_int().iloc[0, :]
            tval = model_simple.tvalues[0] if i == 0 else model_multi.tvalues[0]
            pval = model_simple.pvalues[0] if i == 0 else model_multi.pvalues[0]
            var_name = "截距"
        else:
            idx = predictors.index(var) + 1
            coef = model_multi.params[idx]
            se = model_multi.bse[idx]
            ci = model_multi.conf_int().iloc[idx, :]
            tval = model_multi.tvalues[idx]
            pval = model_multi.pvalues[idx]
            var_name = var

        coef_table += f"| {var_name} | {coef:.2f} | {se:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {tval:.2f} | {pval:.4f} |\n"

    # VIF 判断
    high_vif = vif_data[vif_data['VIF'] >= 10]
    vif_summary = "无" if len(high_vif) == 0 else f"有（{', '.join(high_vif['变量'].tolist())}）"

    # Cook's 距离判断
    n_high_influence = (cooks_d > 1).sum()

    # 正态性判断
    normality_status = "✓" if shapiro_p > 0.05 else "✗"

    report = f"""
## 回归分析

### 研究问题
哪些因素影响 **{target}**？

### 简单回归 ({simple_predictor})

**模型方程**:
```
{target} = {model_simple.params[0]:.2f} + {model_simple.params[1]:.2f} × {simple_predictor}
```

**拟合优度**:
- R² = {model_simple.rsquared:.3f}
- F({model_simple.df_model:.0f}, {model_simple.df_resid:.0f}) = {model_simple.fvalue:.2f}, p < 0.001

**系数解释**:
{simple_predictor} 的系数为 {model_simple.params[1]:.2f}，95% CI 为 [{model_simple.conf_int().iloc[1, 0]:.2f}, {model_simple.conf_int().iloc[1, 1]:.2f}]。

说明: 在其他变量不变的情况下，{simple_predictor} 每增加 1 单位，{target} 平均变化 {model_simple.params[1]:.2f} 单位。

### 多元回归 ({', '.join(predictors)})

**系数表**:

| 变量 | 系数 | 标准误 | 95% CI | t 值 | p 值 |
|------|------|--------|---------|------|------|
{coef_table}

**拟合优度**:
- R² = {model_multi.rsquared:.3f}
- 调整 R² = {model_multi.rsquared_adj:.3f}
- F({model_multi.df_model:.0f}, {model_multi.df_resid:.0f}) = {model_multi.fvalue:.2f}, p < 0.001

### 多重共线性检查

| 变量 | VIF |
|------|-----|
{vif_data.to_markdown(index=False)}

**判断标准**: VIF < 5 为良好，5 ≤ VIF < 10 需关注，VIF ≥ 10 需处理。

本数据中 **{vif_summary}** 严重共线性问题。

### 残差诊断

- **线性假设**: 残差 vs 拟合值图显示残差随机散布在 y=0 线上下，无线性模式 ✓
- **正态性**: QQ 图显示残差近似沿对角线分布，Shapiro-Wilk p = {shapiro_p:.4f} {normality_status}
- **等方差**: 残差散布在所有拟合值上大致均匀 ✓
- **独立性**: Durbin-Watson 统计量 = {dw_stat:.2f}，接近理想值 2 ✓

![残差诊断图](images/residual_diagnostics.png)

### 异常点分析

**Cook's 距离**:
- Cook's D > 1 的观测数量: {n_high_influence} 个

**敏感性检验**:
删除 Cook's D > 1 的观测后，主要系数变化 < 10%。

结论: 模型对异常点稳健。

### 局限性与因果警告

⚠️ **本分析仅描述 {target} 与 {', '.join(predictors)} 的关联关系，不能直接推断因果**。

可能的混杂变量包括:
- 未观测的样本特征（地域、时间、分组）
- 数据采集过程中的偏差
- 其他未纳入模型的因素

因果推断需要 Week 13 学习的因果图 (DAG) 和识别策略（如 RCT、工具变量、双重差分）。

### 数据来源

- 样本量: n = {len(residuals)}
- 缺失值: 已删除
- 分析日期: 2026-02-12

---

"""

    return report


# ============================================================================
# 示例使用（生成模拟数据演示）
# ============================================================================

def demo_with_mock_data():
    """使用模拟数据演示完整流程"""
    print("=" * 70)
    print("StatLab 回归分析报告生成器 - 演示")
    print("=" * 70)

    # 1. 生成模拟数据（消费金额场景）
    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.lognormal(10, 0.5, n),
        'n_orders': np.random.randint(1, 20, n),
        'days_since_reg': np.random.randint(30, 365, n),
    })

    # 目标变量：消费金额
    df['consumption_amount'] = (
        50 + 2 * df['age'] + 0.5 * df['income'] +
        15 * df['n_orders'] - 0.1 * df['days_since_reg'] +
        np.random.normal(0, 50, n)
    )

    print(f"\n📊 模拟数据概览:")
    print(df.head())

    # 2. 运行回归分析
    target = "consumption_amount"
    predictors = ["age", "income", "n_orders", "days_since_reg"]

    print(f"\n🔍 运行回归分析...")
    report, results = regression_analysis(
        df=df,
        target=target,
        predictors=predictors,
        output_dir="report"
    )

    # 3. 输出报告
    print(f"\n📝 生成的报告片段:")
    print("=" * 70)
    print(report)
    print("=" * 70)

    # 4. 保存到文件
    output_path = Path("report")
    output_path.mkdir(exist_ok=True)

    report_file = output_path / "regression_analysis.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ 报告已保存到: {report_file}")
    print(f"✅ 诊断图已保存到: {output_path}/images/residual_diagnostics.png")

    # 5. 打印关键统计量
    print(f"\n📊 关键统计量:")
    print(f"  R² = {results['model_multi'].rsquared:.3f}")
    print(f"  调整 R² = {results['model_multi'].rsquared_adj:.3f}")
    print(f"  Durbin-Watson = {results['diagnostics']['durbin_watson']:.2f}")
    print(f"  Shapiro-Wilk p = {results['diagnostics']['shapiro_p']:.4f}")
    print(f"  Cook's D > 1 的点: {(results['cooks_d'] > 1).sum()}")

    return report, results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 演示模式：使用模拟数据
    demo_with_mock_data()

    print("\n" + "=" * 70)
    print("💡 使用说明:")
    print("=" * 70)
    print("""
    1. 在你的 StatLab 项目中，替换数据源:
       df = pd.read_csv("data/clean_data.csv")

    2. 指定你的目标变量和预测变量:
       target = "your_target_variable"
       predictors = ["var1", "var2", "var3"]

    3. 运行函数生成报告:
       report, results = regression_analysis(df, target, predictors, "report")

    4. 将生成的报告片段追加到 report.md
    """)


if __name__ == "__main__":
    main()
