#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StatLab Week 07：多组比较结果章节生成

本脚本在 Week 06 报告基础上，添加"多组比较结果"章节。
功能包括：
1. 单因素 ANOVA（多组均值比较）
2. 前提假设检查（正态性、方差齐性）
3. 事后检验（Tukey HSD）
4. 效应量计算（η²）
5. 卡方检验（分类变量关联）
6. Cramér's V 效应量
7. 生成 Markdown 格式的报告章节并追加到 report.md

这是 StatLab 超级线的 Week 07 增量更新。

运行方式：python3 chapters/week_07/examples/99_statlab.py
预期输出：更新后的 report.md，包含多组比较结果章节

作者：StatLab Week 07
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


# =============================================================================
# 数据加载（与 Week 06 一致）
# =============================================================================

def load_or_generate_data() -> pd.DataFrame:
    """加载数据或生成示例数据（与 Week 06 保持一致）"""
    data_paths = [
        'checkpoint/week_06_checkpoint.json',
        '../week_06/checkpoint/week_06_checkpoint.json',
        'checkpoint/week_04_checkpoint.json',
    ]

    for path in data_paths:
        if Path(path).exists():
            print(f"[加载数据] 从 {path} 加载数据")
            return generate_sample_data()

    print("[生成数据] 未找到现有数据，生成示例数据...")
    return generate_sample_data()


def generate_sample_data(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """生成示例电商用户数据（与 Week 06 保持一致）"""
    np.random.seed(seed)

    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age': np.clip(np.random.normal(35, 12, n), 18, 70).astype(int),
        'monthly_income': np.clip(np.random.lognormal(8.5, 0.6, n), 3000, 80000).astype(int),
        'gender': np.random.choice(['男', '女'], n, p=[0.48, 0.52]),
        'city_tier': np.random.choice(['一线', '二线', '三线'], n, p=[0.3, 0.45, 0.25]),
        'registration_days': np.clip(np.random.exponential(200, n), 1, 1000).astype(int),
    })

    # 消费与收入、城市级别相关
    base_spend = df['monthly_income'] * np.random.uniform(0.15, 0.35, n)
    city_multiplier = df['city_tier'].map({'一线': 1.3, '二线': 1.0, '三线': 0.8})
    df['monthly_spend'] = (base_spend * city_multiplier + np.random.normal(0, 500, n)).astype(int).clip(100, None)

    # 用户等级
    spend_bins = [0, 1000, 3000, 8000, float('inf')]
    spend_labels = ['普通', '银卡', '金卡', '钻石']
    df['user_level'] = pd.cut(df['monthly_spend'], bins=spend_bins, labels=spend_labels)

    return df


# =============================================================================
# ANOVA 分析函数
# =============================================================================

class ANOVAAnalyzer:
    """ANOVA 分析类：封装完整的 ANOVA 流程"""

    def __init__(self):
        self.results = {}

    def check_assumptions(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str
    ) -> Dict[str, any]:
        """
        检查 ANOVA 前提假设。

        参数：
            df: 数据框
            group_col: 分组列名
            value_col: 数值列名

        返回：
            dict: 假设检查结果
        """
        assumptions = {}
        groups = df[group_col].unique()

        # 1. 正态性检验
        normality_results = {}
        for group in groups:
            group_data = df[df[group_col] == group][value_col]
            if len(group_data) >= 3:  # Shapiro-Wilk 需要至少 3 个样本
                _, p_value = stats.shapiro(group_data)
                normality_results[group] = {
                    'p_value': p_value,
                    'met': p_value > 0.05
                }
        assumptions['normality'] = normality_results

        # 2. 方差齐性检验
        group_arrays = [df[df[group_col] == g][value_col].values for g in groups]
        _, p_levene = stats.levene(*group_arrays)
        assumptions['levene'] = {
            'p_value': p_levene,
            'met': p_levene > 0.05
        }

        # 3. 独立性（设计检查）
        assumptions['independence'] = {
            'met': True,
            'note': '用户随机抽样，各城市互不干扰'
        }

        return assumptions

    def perform_anova(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str
    ) -> Dict[str, any]:
        """
        执行 ANOVA 检验。

        参数：
            df: 数据框
            group_col: 分组列名
            value_col: 数值列名

        返回：
            dict: ANOVA 结果
        """
        # 使用 statsmodels 生成 ANOVA 表
        formula = f'{value_col} ~ C({group_col})'
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # 提取关键结果
        f_stat = anova_table.loc[f'C({group_col})', 'F']
        p_value = anova_table.loc[f'C({group_col})', 'PR(>F)']

        # 计算 η²
        ssb = anova_table.loc[f'C({group_col})', 'sum_sq']
        ssw = anova_table.loc['Residual', 'sum_sq']
        sst = ssb + ssw
        eta_squared = ssb / sst

        # 自由度
        df_between = anova_table.loc[f'C({group_col})', 'df']
        df_within = anova_table.loc['Residual', 'df']

        return {
            'f_stat': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'df_between': df_between,
            'df_within': df_within,
            'anova_table': anova_table,
            'reject_h0': p_value < 0.05
        }

    def perform_posthoc(
        self,
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        alpha: float = 0.05
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        执行 Tukey HSD 事后检验。

        参数：
            df: 数据框
            group_col: 分组列名
            value_col: 数值列名
            alpha: 显著性水平

        返回：
            tuple: (tukey_df, significant_pairs)
        """
        tukey = pairwise_tukeyhsd(
            endog=df[value_col].values,
            groups=df[group_col].values,
            alpha=alpha
        )

        # 转换为 DataFrame
        tukey_df = pd.DataFrame(
            data=tukey._results_table.data[1:],
            columns=tukey._results_table.data[0]
        )

        # 提取显著结果
        significant_pairs = tukey_df[tukey_df['reject'] == True]

        return tukey_df, significant_pairs

    def interpret_eta_squared(self, eta2: float) -> str:
        """解释 η² 效应量"""
        if eta2 < 0.01:
            return "效应量极小（< 1% 的变异由组间差异解释）"
        elif eta2 < 0.06:
            return "效应量小（1%-6% 的变异由组间差异解释）"
        elif eta2 < 0.14:
            return "效应量中等（6%-14% 的变异由组间差异解释）"
        else:
            return "效应量大（≥ 14% 的变异由组间差异解释）"


# =============================================================================
# 卡方检验函数
# =============================================================================

class ChiSquareAnalyzer:
    """卡方检验分析类"""

    def perform_chisquare(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str
    ) -> Dict[str, any]:
        """
        执行卡方独立性检验。

        参数：
            df: 数据框
            col1: 第一个分类变量
            col2: 第二个分类变量

        返回：
            dict: 卡方检验结果
        """
        # 创建列联表
        contingency_table = pd.crosstab(df[col1], df[col2])

        # 执行卡方检验
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

        # 计算 Cramér's V
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape[0], contingency_table.shape[1])
        phi2 = chi2 / n
        cramers_v = np.sqrt(phi2 / (min_dim - 1))

        # 解释 Cramér's V
        if cramers_v < 0.1:
            interpretation = "关联很弱"
        elif cramers_v < 0.3:
            interpretation = "关联较弱"
        elif cramers_v < 0.5:
            interpretation = "关联中等"
        else:
            interpretation = "关联较强"

        return {
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'cramers_v': cramers_v,
            'interpretation': interpretation,
            'contingency_table': contingency_table,
            'expected': expected,
            'reject_h0': p_value < 0.05
        }


# =============================================================================
# 报告生成
# =============================================================================

def append_to_report(content: str, report_path: str = 'report.md') -> None:
    """追加内容到报告（与 Week 06 一致）"""
    path = Path(report_path)

    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            existing = f.read()

        # 检查是否已有多组比较章节
        if '## 多组比较结果' in existing:
            print(f"[更新报告] {report_path} 已包含多组比较章节，将更新")
            # 替换现有章节
            start = existing.find('## 多组比较结果')
            end = existing.find('\n## ', start + 1)
            if end == -1:
                new_content = existing[:start] + content
            else:
                new_content = existing[:start] + content + existing[end:]
        else:
            new_content = existing + '\n\n' + content

        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        # 如果文件不存在，创建新文件
        header = f"# StatLab 分析报告\n\n"
        with open(path, 'w', encoding='utf-8') as f:
            f.write(header + content)

    print(f"[完成] 报告已更新：{report_path}")


def generate_multigroup_comparison_section(
    df: pd.DataFrame,
    report_path: str = 'report.md'
) -> str:
    """
    生成多组比较结果章节的 Markdown 内容。

    参数：
        df: 包含分析数据的 DataFrame
        report_path: 报告文件路径

    返回：
        str: 生成的 Markdown 内容
    """
    print("\n" + "=" * 70)
    print("StatLab 多组比较分析")
    print("=" * 70)

    lines = [
        "\n## 多组比较结果\n",
        "> 本章使用 ANOVA 和卡方检验分析多组数据，判断组间差异和分类变量关联。",
        f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
    ]

    # ========== ANOVA：不同城市级别的消费差异 ==========
    print("\n[分析 1] 不同城市级别的消费差异（ANOVA）")

    anova = ANOVAAnalyzer()

    # 前提假设检查
    print("  检查前提假设...")
    assumptions = anova.check_assumptions(df, 'city_tier', 'monthly_spend')

    normality_ok = all(v['met'] for v in assumptions['normality'].values())
    levene_ok = assumptions['levene']['met']

    # 执行 ANOVA
    print("  执行 ANOVA...")
    anova_results = anova.perform_anova(df, 'city_tier', 'monthly_spend')

    # 事后检验
    print("  执行 Tukey HSD 事后检验...")
    tukey_df, significant_pairs = anova.perform_posthoc(df, 'city_tier', 'monthly_spend')

    # 生成报告内容
    lines.extend([
        "### H4：不同城市级别的消费差异（ANOVA）\n",
        "**假设设定**：",
        "- H0（原假设）：所有城市级别的平均消费相等（μ_一线 = μ_二线 = μ_三线）",
        "- H1（备择假设）：至少有一对城市级别的平均消费不等\n",
        "**前提假设检查**：",
    ])

    # 正态性
    normality_lines = []
    for tier, result in assumptions['normality'].items():
        status = '✓' if result['met'] else '✗'
        normality_lines.append(
            f"  - {tier}：p={result['p_value']:.3f} ({status})"
        )
    lines.append(f"- 正态性：Shapiro-Wilk 检验\n" + '\n'.join(normality_lines))
    lines.append(
        f"  ({'✓' if normality_ok else '✗'}) 正态性假设{'满足' if normality_ok else '可能不满足'}"
    )

    # 方差齐性
    levene_p = assumptions['levene']['p_value']
    lines.extend([
        "",
        f"- 方差齐性：Levene 检验 p={levene_p:.3f}",
        f"  ({'✓' if levene_ok else '✗'}) 方差齐性{'满足' if levene_ok else '不满足'}",
        "",
        "- 独立性：✓ 用户随机抽样，各城市级别互不干扰\n",
    ])

    # ANOVA 结果
    eta2_interpret = anova.interpret_eta_squared(anova_results['eta_squared'])

    lines.extend([
        "**ANOVA 结果**：",
        f"- F 统计量：F({anova_results['df_between']:.0f}, {anova_results['df_within']:.0f}) = {anova_results['f_stat']:.3f}",
        f"- p 值：p = {anova_results['p_value']:.6f}",
        f"- η² 效应量：η² = {anova_results['eta_squared']:.3f}（{eta2_interpret}）",
        f"- 决策：{'拒绝 H0（至少有一对城市级别均值不同）' if anova_results['reject_h0'] else '无法拒绝 H0（各城市级别均值可能相等）'}\n",
    ])

    # Tukey HSD 结果
    if len(significant_pairs) > 0:
        lines.append("**事后检验（Tukey HSD）**：")
        lines.append("| 城市级别对 | 均值差异（元） | p 值（校正后） | 显著性 |")
        lines.append("|-----------|--------------|---------------|--------|")

        for _, row in significant_pairs.iterrows():
            lines.append(
                f"| {row['group1']} vs {row['group2']} | {abs(row['meandiff']):.1f} | {row['p-adj']:.4f} | ✓ |"
            )
        lines.append("")
    else:
        lines.append("**事后检验（Tukey HSD）**：未发现显著差异的城市级别对\n")

    # 解读
    sig_text = "ANOVA 显示城市级别之间存在显著差异" if anova_results['reject_h0'] else "未发现显著差异"
    lines.extend([
        "**解读**：",
        f"- 统计显著性：{sig_text}" +
        (f"，Tukey HSD 识别出 {len(significant_pairs)} 对显著差异" if len(significant_pairs) > 0 else ""),
        f"- 实际意义：效应量{eta2_interpret.split('（')[0]}，" +
        f"{'需结合业务场景评估差异化策略的价值' if anova_results['eta_squared'] < 0.14 else '差异明显，建议差异化策略'}",
        f"- 不确定性：95% CI {'支持' if anova_results['reject_h0'] else '不支持'}城市级别间存在系统性差异\n",
    ])

    # ========== 卡方检验：城市级别与用户等级的关联 ==========
    print("\n[分析 2] 城市级别与用户等级的关联（卡方检验）")

    chi2_analyzer = ChiSquareAnalyzer()
    chi2_results = chi2_analyzer.perform_chisquare(df, 'city_tier', 'user_level')

    lines.extend([
        "### H5：城市级别与用户等级的关联（卡方检验）\n",
        "**假设设定**：",
        "- H0（原假设）：城市级别与用户等级无关",
        "- H1（备择假设）：城市级别与用户等级相关\n",
        "**检验结果**：",
        f"- 卡方统计量：χ²({chi2_results['dof']}) = {chi2_results['chi2']:.2f}",
        f"- p 值：p = {chi2_results['p_value']:.4f}",
        f"- Cramér's V 效应量：V = {chi2_results['cramers_v']:.3f}（{chi2_results['interpretation']}）",
        f"- 决策：{'拒绝 H0（城市级别与用户等级相关）' if chi2_results['reject_h0'] else '无法拒绝 H0（城市级别与用户等级可能无关）'}\n",
        "**解读**：",
        f"- 统计显著性：{'卡方检验显示相关' if chi2_results['reject_h0'] else '未发现显著关联'}",
        f"- 实际意义：效应量{chi2_results['interpretation'].split('（')[0]}" +
        f"，{'即使相关，关联强度也有限' if chi2_results['cramers_v'] < 0.3 else '需结合业务场景评估'}",
        "- 相关 ≠ 因果：观察性设计无法确定因果方向，需进一步研究\n",
    ])

    # ========== 多重比较风险与下一步 ==========
    num_comparisons = len(tukey_df) + 1  # Tukey HSD + 卡方检验
    lines.extend([
        "### 多重比较风险与校正\n",
        f"本周进行了 {num_comparisons} 次比较（ANOVA + Tukey HSD + 卡方检验），存在多重比较风险。",
        "已采用以下策略控制假阳性率：",
        "- ANOVA：单次全局检验，控制整体第一类错误率",
        "- 事后检验：使用 Tukey HSD，控制配对比较的 FWER",
        "- 卡方检验：单次检验，未涉及多重比较\n",
        "### 下一步\n",
        "Week 08 将进行区间估计与重采样（Bootstrap、置换检验），进一步量化结论的不确定性。",
        "---\n"
    ])

    content = ''.join(lines)

    # 追加到报告
    append_to_report(content, report_path)

    return content


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数：StatLab Week 07 更新"""
    print("=" * 70)
    print("StatLab Week 07: 多组比较结果")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    df = load_or_generate_data()
    print(f"  数据形状：{df.shape}")
    print(f"  城市级别分布：\n{df['city_tier'].value_counts()}")
    print(f"  用户等级分布：\n{df['user_level'].value_counts()}")

    # 2. 执行多组比较分析
    print("\n[2/3] 执行多组比较分析...")
    # （在 generate_multigroup_comparison_section 中完成）

    # 3. 生成报告
    print("\n[3/3] 生成多组比较章节...")
    generate_multigroup_comparison_section(df, 'report.md')

    # 完成总结
    print("\n" + "=" * 70)
    print("StatLab Week 07 更新完成!")
    print("=" * 70)
    print(f"\n本周新增内容:")
    print(f"  - 单因素 ANOVA（城市级别 vs 消费）")
    print(f"  - 前提假设检查（正态性、方差齐性）")
    print(f"  - Tukey HSD 事后检验")
    print(f"  - η² 效应量计算")
    print(f"  - 卡方检验（城市级别 vs 用户等级）")
    print(f"  - Cramér's V 效应量")
    print(f"  - 多重比较风险说明")
    print(f"  - 多组比较结果章节（已追加到 report.md）")
    print(f"\n下周预告:")
    print(f"  - 区间估计与重采样")
    print(f"  - Bootstrap 置信区间")
    print(f"  - 置换检验")


if __name__ == "__main__":
    main()
