"""
示例：StatLab 超级线 - Week 04 更新脚本。

本周在 Week 01-03 的基础上，加入相关分析、分组比较和假设清单。
这是从"只看描述"到"提出可检验假设"的重要升级。

运行方式：python3 chapters/week_04/examples/99_statlab.py
预期输出：
- examples/output/report.md：包含数据卡 + 描述统计 + 可视化 + 清洗日志 + **相关性分析 + 分组比较 + 假设清单**
- examples/output/ 目录下生成必要的图表

设计思路：
1. 保留 Week 01-03 的所有内容（数据卡、描述统计、可视化、清洗日志）
2. 新增：相关性分析
3. 新增：分组比较（按物种）
4. 新增：可检验假设清单
5. 所有内容整合为一份可审计的报告

本周改进：
- 上周：数据卡 + 描述统计 + 可视化 + 清洗日志
- 本周：+ 相关性分析 + 分组比较 + 假设清单
- 目标：让报告变成"可检验的故事"，为 Week 06-08 做铺垫
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Literal
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =============================================================================
# 配置
# =============================================================================

OUTPUT_DIR = Path(__file__).parent / "output"
REPORT_PATH = OUTPUT_DIR / "report.md"


# =============================================================================
# 假设清单类（来自示例05）
# =============================================================================

class Priority(str, Enum):
    """优先级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Hypothesis:
    """假设数据类"""
    observation: str
    explanation: str
    test_method: str
    priority: Priority

    def to_dict(self) -> dict:
        return {
            "observation": self.observation,
            "explanation": self.explanation,
            "test_method": self.test_method,
            "priority": self.priority.value
        }


class HypothesisList:
    """假设清单：记录可检验的假设"""

    def __init__(self):
        self.hypotheses: list[Hypothesis] = []

    def add(self, observation: str, explanation: str, test_method: str,
            priority: Priority = Priority.MEDIUM) -> None:
        self.hypotheses.append(Hypothesis(
            observation=observation,
            explanation=explanation,
            test_method=test_method,
            priority=priority
        ))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([h.to_dict() for h in self.hypotheses])

    def to_markdown_section(self) -> str:
        if not self.hypotheses:
            return "## 6. 可检验假设清单\n\n本数据集暂无待检验假设。\n"

        df = self.to_dataframe()
        md = ["## 6. 可检验假设清单\n\n"]
        md.append("以下假设基于 EDA 观察，将在后续章节用统计方法验证。\n\n")

        for priority in ["high", "medium", "low"]:
            priority_hyps = df[df["priority"] == priority]
            if priority_hyps.empty:
                continue

            priority_label = {"high": "高", "medium": "中", "low": "低"}[priority]
            md.append(f"### {priority_label}优先级\n\n")

            for _, row in priority_hyps.iterrows():
                md.append(f"- **观察**：{row['observation']}\n")
                md.append(f"  - **解释**：{row['explanation']}\n")
                md.append(f"  - **检验方法**：{row['test_method']}\n\n")

        return "".join(md)


# =============================================================================
# 清洗日志类（来自 Week 03）
# =============================================================================

class CleaningLog:
    """清洗日志：记录每一个数据决策"""

    def __init__(self):
        self.logs = []

    def add(self, variable: str, issue: str, action: str,
            reason: str, n_affected: int = None) -> None:
        self.logs.append({
            "variable": variable,
            "issue": issue,
            "action": action,
            "reason": reason,
            "n_affected": n_affected
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.logs)

    def to_markdown_section(self) -> str:
        df = self.to_dataframe()
        if df.empty:
            return "## 3. 数据清洗\n\n本数据集质量良好，未发现需要处理的问题。\n"

        md = ["## 3. 数据清洗与决策记录\n\n"]
        md.append("以下记录了本报告对数据的所有处理决策及其理由。\n\n")

        md.append("### 3.1 缺失值处理\n\n")
        missing_logs = df[df["issue"] == "missing_values"]
        if not missing_logs.empty:
            for _, row in missing_logs.iterrows():
                md.append(f"- **{row['variable']}**：{row['action']}\n")
                md.append(f"  - {row['reason']}")
                if row["n_affected"]:
                    md.append(f"（影响 {row['n_affected']} 行）\n")
                else:
                    md.append("\n")
        else:
            md.append("本数据集无显著缺失值问题。\n")

        md.append("\n### 3.2 异常值处理\n\n")
        outlier_logs = df[df["issue"] == "potential_outliers"]
        if not outlier_logs.empty:
            for _, row in outlier_logs.iterrows():
                md.append(f"- **{row['variable']}**：{row['action']}\n")
                md.append(f"  - {row['reason']}")
                if row["n_affected"]:
                    md.append(f"（影响 {row['n_affected']} 个观测值）\n")
                else:
                    md.append("\n")
        else:
            md.append("本数据集未发现需要处理的异常值。\n")

        return "".join(md)


# =============================================================================
# 数据加载
# =============================================================================

def load_data() -> pd.DataFrame:
    """加载 Palmer Penguins 数据集"""
    return sns.load_dataset("penguins")


# =============================================================================
# Week 01-02 内容（数据卡、描述统计、可视化）
# =============================================================================

def generate_data_card(df: pd.DataFrame) -> str:
    """生成数据卡（Week 01 内容）"""
    lines = [
        "# StatLab 分析报告",
        "",
        f"**生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. 数据卡",
        "",
        "### 1.1 数据来源",
        "- 数据集：Palmer Penguins",
        "- 来源：seaborn 内置数据集",
        "- 描述：Palmer Archipelago（南极）的 3 种企鹅数据",
        "",
        "### 1.2 字段字典",
        "",
        "| 字段名 | 类型 | 说明 | 缺失率 |",
        "|--------|------|------|--------|",
    ]

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_rate = (df[col].isna().mean() * 100).round(1)
        desc = {
            "species": "企鹅物种 (Adelie, Chinstrap, Gentoo)",
            "island": "岛屿 (Biscoe, Dream, Torgersen)",
            "bill_length_mm": "嘴峰长度 (mm)",
            "bill_depth_mm": "嘴峰深度 (mm)",
            "flipper_length_mm": "鳍肢长度 (mm)",
            "body_mass_g": "体重 (g)",
            "sex": "性别 (Male, Female)",
        }.get(col, "未知字段")
        lines.append(f"| {col} | {dtype} | {desc} | {missing_rate}% |")

    lines.extend([
        "",
        "### 1.3 数据规模",
        f"- 总行数：{len(df)}",
        f"- 总列数：{df.shape[1]}",
        "",
        "### 1.4 缺失值概览",
        f"- 有缺失值的列数：{df.isna().any().sum()}",
        f"- 完整行数（无缺失）：{df.dropna().shape[0]}",
        "",
    ])

    return "\n".join(lines)


def generate_summary_stats(df: pd.DataFrame) -> str:
    """生成描述统计表（Week 02 内容）"""
    lines = [
        "## 2. 描述统计",
        "",
        "### 2.1 数值型字段统计",
        "",
        "#### 2.1.1 整体统计",
        "",
        "| 字段 | 计数 | 均值 | 中位数 | 标准差 | 最小值 | Q25 | Q75 | 最大值 |",
        "|------|------|------|--------|--------|--------|-----|-----|--------|",
    ]

    numeric_cols = ["bill_length_mm", "bill_depth_mm",
                    "flipper_length_mm", "body_mass_g"]
    for col in numeric_cols:
        data = df[col].dropna()
        lines.append(
            f"| {col} | {len(data):.0f} | {data.mean():.1f} | "
            f"{data.median():.1f} | {data.std():.1f} | {data.min():.1f} | "
            f"{data.quantile(0.25):.1f} | {data.quantile(0.75):.1f} | {data.max():.1f} |"
        )

    return "\n".join(lines)


def generate_plots(df: pd.DataFrame, output_dir: Path) -> list[str]:
    """生成可视化图表（Week 02 内容）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    species_colors = {"Adelie": "steelblue", "Chinstrap": "orange", "Gentoo": "green"}

    # 1. 按物种分组的体重箱线图
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="species", y="body_mass_g", hue="species",
               palette=species_colors, legend=False)
    plt.xlabel("Species")
    plt.ylabel("Body Mass (g)")
    plt.title("Body Mass by Species (Boxplot)")
    filename = "boxplot_by_species.png"
    plt.savefig(output_dir / filename, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(filename)

    print(f"图表已保存到 {output_dir}/")
    return generated_files


def generate_visualization_section(generated_files: list[str]) -> str:
    """生成可视化章节（Week 02 内容）"""
    lines = [
        "## 4. 可视化",
        "",
        "### 4.1 按物种分组的箱线图",
        "",
        "![按物种分组的箱线图](boxplot_by_species.png)",
        "",
        "**观察**：",
        "- Gentoo 企鹅的体重分布整体高于其他两种",
        "- Adelie 和 Chinstrap 的体重分布有部分重叠",
        "- Gentoo 的 IQR 较小，说明体重分布更集中",
        "",
    ]

    return "\n".join(lines)


# =============================================================================
# Week 03 内容：清洗日志生成
# =============================================================================

def analyze_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """分析数据并生成清洗日志"""
    log = CleaningLog()
    df_cleaned = df.copy()

    # 缺失值分析
    print("分析缺失值...")
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_rate = df[col].isna().mean()
            log.add(
                variable=col,
                issue="missing_values",
                action="documented_only",
                reason=f"缺失率 {missing_rate*100:.1f}%，在可接受范围内。未进行填充，分析时将忽略缺失值。",
                n_affected=missing_count
            )

    # 异常值检测
    print("检测异常值...")
    numeric_cols = ["bill_length_mm", "bill_depth_mm",
                    "flipper_length_mm", "body_mass_g"]

    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue

        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr

        outliers = (data < lower) | (data > upper)
        n_outliers = outliers.sum()

        if n_outliers > 0:
            log.add(
                variable=col,
                issue="potential_outliers",
                action="keep_as_is",
                reason=f"检测到 {n_outliers} 个候选异常值（IQR 规则）。经检查，这些值在合理范围内，保留不做处理。",
                n_affected=n_outliers
            )

    return df_cleaned, log


# =============================================================================
# Week 04 新增：相关性分析、分组比较、假设清单
# =============================================================================

def generate_correlation_section(df: pd.DataFrame) -> str:
    """生成相关性分析的 Markdown 片段（本周新增）"""
    numeric_cols = ["bill_length_mm", "bill_depth_mm",
                    "flipper_length_mm", "body_mass_g"]
    corr_matrix = df[numeric_cols].corr(method="pearson")

    md = ["## 5. 相关性分析\n\n"]
    md.append("以下展示数值型变量两两之间的 Pearson 相关系数。\n\n")

    # 找出最强的相关
    corr_unstacked = corr_matrix.abs().unstack()
    corr_unstacked = corr_unstacked[corr_unstacked < 1]
    if not corr_unstacked.empty:
        max_corr = corr_unstacked.idxmax()
        max_corr_value = corr_unstacked.max()
        md.append(f"**最强相关**：{max_corr[0]} 与 {max_corr[1]} ")
        md.append(f"（r = {corr_matrix.loc[max_corr[0], max_corr[1]]:.2f}）\n\n")

    # 展示相关矩阵（简化版）
    md.append("### 相关系数矩阵\n\n")
    for i, col1 in enumerate(numeric_cols):
        row_vals = []
        for col2 in numeric_cols:
            if col1 == col2:
                row_vals.append("—")
            else:
                r = corr_matrix.loc[col1, col2]
                row_vals.append(f"{r:.2f}")
        md.append(f"| {col1} | {' | '.join(row_vals)} |\n")

    md.append("\n**解释**：\n")
    md.append("- 相关系数范围 -1 到 1，接近 1 表示强正相关\n")
    md.append("- body_mass_g 与 flipper_length_mm 呈强正相关\n")
    md.append("- bill_depth_mm 与其他变量呈弱负相关\n")

    return "".join(md)


def generate_group_comparison_section(df: pd.DataFrame) -> str:
    """生成分组比较的 Markdown 片段（本周新增）"""
    md = ["## 6. 分组比较\n\n"]
    md.append("以下按物种（species）分组，比较数值型变量的分布。\n\n")

    numeric_cols = ["bill_length_mm", "bill_depth_mm",
                    "flipper_length_mm", "body_mass_g"]

    for col in numeric_cols:
        md.append(f"### {col} 按物种分组\n\n")
        md.append("| 物种 | 计数 | 均值 | 中位数 | 标准差 |\n")
        md.append("|------|------|------|--------|--------|\n")

        for species in df["species"].unique():
            data = df[df["species"] == species][col].dropna()
            md.append(
                f"| {species} | {len(data):.0f} | {data.mean():.1f} | "
                f"{data.median():.1f} | {data.std():.1f} |\n"
            )
        md.append("\n")

    return "".join(md)


def generate_hypothesis_list(df: pd.DataFrame) -> HypothesisList:
    """基于数据生成假设清单（本周新增）"""
    hypotheses = HypothesisList()

    # 假设 1：体重差异
    adelie_mean = df[df["species"] == "Adelie"]["body_mass_g"].mean()
    gentoo_mean = df[df["species"] == "Gentoo"]["body_mass_g"].mean()
    diff = gentoo_mean - adelie_mean

    hypotheses.add(
        observation=f"Gentoo 企鹅的平均体重（{gentoo_mean:.0f}g）显著高于 Adelie（{adelie_mean:.0f}g），相差 {diff:.0f}g",
        explanation="Gentoo 体型更大，体重可能是物种特征",
        test_method="方差分析 ANOVA（Week 07）",
        priority=Priority.HIGH
    )

    # 假设 2：翅长与体重相关
    corr_fb = df["flipper_length_mm"].corr(df["body_mass_g"])
    hypotheses.add(
        observation=f"flipper_length_mm 与 body_mass_g 呈强正相关（Pearson r = {corr_fb:.2f}）",
        explanation="体型更大的企鹅翅长更长，两者呈正相关",
        test_method="相关系数的置信区间（Week 08）",
        priority=Priority.MEDIUM
    )

    # 假设 3：嘴峰深度差异
    chinstrap_depth = df[df["species"] == "Chinstrap"]["bill_depth_mm"].mean()
    adelie_depth = df[df["species"] == "Adelie"]["bill_depth_mm"].mean()

    hypotheses.add(
        observation=f"Adelie 的嘴峰深度（{adelie_depth:.1f}mm）比 Chinstrap（{chinstrap_depth:.1f}mm）更深",
        explanation="不同物种的嘴峰形状可能适应不同的食物来源",
        test_method="双样本 t 检验（Week 06）",
        priority=Priority.MEDIUM
    )

    return hypotheses


# =============================================================================
# 主要发现生成
# =============================================================================

def generate_key_findings(df: pd.DataFrame, log: CleaningLog, hypotheses: HypothesisList) -> str:
    """生成主要发现"""
    adelie_mean = df[df["species"] == "Adelie"]["body_mass_g"].mean()
    gentoo_mean = df[df["species"] == "Gentoo"]["body_mass_g"].mean()
    diff = gentoo_mean - adelie_mean

    lines = [
        "## 8. 主要发现",
        "",
        "### 8.1 集中趋势",
        f"- 三种企鹅的平均体重差异明显：Adelie ({adelie_mean:.0f}g) < "
        f"Chinstrap ({df[df['species']=='Chinstrap']['body_mass_g'].mean():.0f}g) < "
        f"Gentoo ({gentoo_mean:.0f}g)",
        f"- Gentoo 比 Adelie 平均重 {diff:.0f}g",
        "",
        "### 8.2 数据质量评估",
        f"- 数据集整体质量良好，清洗日志中记录了 {len(log.to_dataframe())} 条数据决策",
        "- 所有异常值经检查后判定为合理值，未进行删除",
        "",
        "### 8.3 相关性与分组差异",
        "- 体重与翅长呈强正相关，符合生物学预期",
        "- 不同物种在所有测量维度上均有显著差异",
        "",
        "### 8.4 下一步建议",
        f"- 验证 {len(hypotheses.to_dataframe())} 个可检验假设（Week 06-08）",
        "- 重点关注物种间体重差异的统计显著性",
        "",
    ]

    return "\n".join(lines)


# =============================================================================
# 分析日志生成
# =============================================================================

def generate_analysis_log(hypotheses: HypothesisList) -> str:
    """生成分析日志（记录每周的改进）"""
    lines = [
        "---",
        "",
        "## 9. 分析日志",
        "",
        "### Week 01",
        "- 创建数据卡",
        "",
        "### Week 02",
        "- 新增：描述统计表",
        "- 新增：2 张可视化图表（直方图、箱线图）",
        "",
        "### Week 03",
        "- 新增：数据清洗与决策记录",
        "- 缺失值：经诊断，缺失率低，保留原样",
        "- 异常值：经 IQR 规则检测，判定为合理值",
        "",
        "### Week 04",
        "- **新增**：相关性分析（Pearson 相关系数矩阵）",
        "- **新增**：分组比较（按物种分组统计）",
        "- **新增**：可检验假设清单（" + str(len(hypotheses.to_dataframe())) + " 个假设）",
        "- **改进**：从'看数据'到'提出可检验问题'",
        "- **改进**：为 Week 06-08 的统计推断做准备",
        "",
    ]

    return "\n".join(lines)


# =============================================================================
# 报告生成入口
# =============================================================================

def generate_report(df: pd.DataFrame, output_dir: Path) -> None:
    """生成完整的 StatLab 报告"""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("StatLab 报告生成器 - Week 04")
    print("="*60)

    # 1. 数据清洗与分析
    df_cleaned, log = analyze_and_clean(df)

    # 2. 生成可视化
    print("\n生成可视化图表...")
    generated_files = generate_plots(df, output_dir)

    # 3. 生成假设清单（本周新增）
    print("\n生成假设清单...")
    hypotheses = generate_hypothesis_list(df)

    # 4. 组装报告
    print("\n组装报告...")
    report_sections = [
        generate_data_card(df),
        generate_summary_stats(df),
        log.to_markdown_section(),
        generate_visualization_section(generated_files),
        generate_correlation_section(df),  # 本周新增
        generate_group_comparison_section(df),  # 本周新增
        hypotheses.to_markdown_section(),  # 本周新增
        generate_key_findings(df, log, hypotheses),
        generate_analysis_log(hypotheses),
    ]

    report_content = "\n".join(report_sections)

    # 写入报告
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n报告已生成：{REPORT_PATH}")
    print(f"  - 数据卡：1 个")
    print(f"  - 描述统计表：1 个")
    print(f"  - 清洗日志：{len(log.to_dataframe())} 条记录")
    print(f"  - 可视化图表：{len(generated_files)} 张")
    print(f"  - 相关性分析：1 个")
    print(f"  - 分组比较：4 个变量")
    print(f"  - 假设清单：{len(hypotheses.to_dataframe())} 个")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数"""
    df = load_data()
    generate_report(df, OUTPUT_DIR)

    print("\n" + "="*60)
    print("Week 04 更新完成！")
    print("="*60)
    print("从 Week 03 的'数据卡 + 描述统计 + 可视化 + 清洗日志'")
    print("升级为'... + **相关性分析 + 分组比较 + 假设清单**'")
    print()
    print("关键改进：")
    print("  - 从'只看描述'到'提出可检验假设'")
    print("  - 相关分析识别变量关系")
    print("  - 分组比较发现组间差异")
    print("  - 假设清单为 Week 06-08 做铺垫")


if __name__ == "__main__":
    main()
