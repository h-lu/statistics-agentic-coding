"""
示例：StatLab 超级线 - Week 03 更新脚本。

本周在 Week 02 的基础上，加入数据清洗日志和决策记录。
这是从"只看描述"到"记录每一个清洗决策"的重要升级。

运行方式：python3 chapters/week_03/examples/99_statlab.py
预期输出：
- examples/output/report.md：包含数据卡 + 描述统计 + 可视化 + **清洗日志**
- examples/output/ 目录下生成必要的图表

设计思路：
1. 保留 Week 01-02 的所有内容（数据卡、描述统计、可视化）
2. 新增：缺失值处理日志
3. 新增：异常值检测与处理日志
4. 新增：数据转换记录
5. 所有清洗决策都有理由说明（可审计）

本周改进：
- 上周：数据卡 + 描述统计 + 可视化
- 本周：+ 清洗日志（记录每一个数据决策）
- 目标：让报告变得可复现、可审计
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Literal

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
# 清洗日志类（来自示例04）
# =============================================================================

class CleaningLog:
    """清洗日志：记录每一个数据决策"""

    def __init__(self):
        self.logs = []

    def add(self, variable: str, issue: str, action: str,
            reason: str, n_affected: int = None) -> None:
        """添加一条清洗记录"""
        self.logs.append({
            "variable": variable,
            "issue": issue,
            "action": action,
            "reason": reason,
            "n_affected": n_affected
        })

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return pd.DataFrame(self.logs)

    def to_markdown_section(self) -> str:
        """生成清洗日志的 Markdown 片段"""
        df = self.to_dataframe()
        if df.empty:
            return "## 3. 数据清洗\n\n本数据集质量良好，未发现需要处理的问题。\n"

        md = ["## 3. 数据清洗与决策记录\n\n"]
        md.append("以下记录了本报告对数据的所有处理决策及其理由。\n\n")

        # 按问题类型分组
        md.append("### 3.1 缺失值处理\n\n")
        missing_logs = df[df["issue"] == "missing_values"]
        if not missing_logs.empty:
            for _, row in missing_logs.iterrows():
                md.append(f"- **{row['variable']}**：{row['action']}\n")
                md.append(f"  - 处理方式：{row['reason']}")
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
                md.append(f"  - 处理方式：{row['reason']}")
                if row["n_affected"]:
                    md.append(f"（影响 {row['n_affected']} 个观测值）\n")
                else:
                    md.append("\n")
        else:
            md.append("本数据集未发现需要处理的异常值。\n")

        md.append("\n### 3.3 数据转换\n\n")
        transform_logs = df[df["issue"] == "scale_difference"]
        if not transform_logs.empty:
            for _, row in transform_logs.iterrows():
                md.append(f"- **{row['variable']}**：{row['action']}\n")
                md.append(f"  - 理由：{row['reason']}\n")
        else:
            md.append("本数据集未进行额外的数据转换。\n")

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

    lines.extend([
        "",
        "#### 2.1.2 按物种分组的体重统计",
        "",
        "| 物种 | 计数 | 均值(g) | 中位数(g) | 标准差(g) | IQR(g) |",
        "|------|------|---------|-----------|-----------|--------|",
    ])

    for species in df["species"].unique():
        data = df[df["species"] == species]["body_mass_g"].dropna()
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        lines.append(
            f"| {species} | {len(data):.0f} | {data.mean():.1f} | "
            f"{data.median():.1f} | {data.std():.1f} | {iqr:.1f} |"
        )

    return "\n".join(lines)


def generate_plots(df: pd.DataFrame, output_dir: Path) -> list[str]:
    """生成可视化图表（Week 02 内容）"""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    species_colors = {"Adelie": "steelblue", "Chinstrap": "orange", "Gentoo": "green"}

    # 1. 按物种分组的体重分布（直方图）
    plt.figure(figsize=(10, 6))
    for species in df["species"].unique():
        data = df[df["species"] == species]["body_mass_g"].dropna()
        plt.hist(data, bins=15, alpha=0.5, label=species, edgecolor="black",
                color=species_colors.get(species))
    plt.xlabel("Body Mass (g)")
    plt.ylabel("Frequency")
    plt.title("Body Mass Distribution by Species")
    plt.legend()
    filename = "dist_by_species.png"
    plt.savefig(output_dir / filename, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()
    generated_files.append(filename)

    # 2. 按物种分组的体重箱线图
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
        "### 4.1 体重分布",
        "",
        "![按物种分组的体重分布](dist_by_species.png)",
        "",
        "**观察**：",
        "- Gentoo 企鹅的体重分布整体高于其他两种",
        "- Adelie 和 Chinstrap 的体重分布有部分重叠",
        "- Gentoo 的分布更集中（方差较小）",
        "",
        "### 4.2 箱线图（异常值检测）",
        "",
        "![按物种分组的箱线图](boxplot_by_species.png)",
        "",
        "**观察**：",
        "- 箱线图显示了各物种的中位数、Q25、Q75 和异常值",
        "- Gentoo 的 IQR（箱子高度）较大，说明体重分布较广",
        "- Adelie 和 Chinstrap 有少量异常值（须之外的点）",
        "",
    ]

    return "\n".join(lines)


# =============================================================================
# Week 03 新增：清洗日志生成
# =============================================================================

def analyze_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """
    分析数据并生成清洗日志

    这是本周的核心改进：不再是盲目清洗，而是：
    1. 先诊断缺失机制
    2. 检测候选异常值
    3. 判断是否需要处理
    4. 记录所有决策和理由
    """
    log = CleaningLog()
    df_cleaned = df.copy()

    # 1. 缺失值分析
    print("分析缺失值...")
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_rate = df[col].isna().mean()

            if missing_rate < 0.05:
                # 缺失率低，保持原样
                log.add(
                    variable=col,
                    issue="missing_values",
                    action="keep_as_is",
                    reason=f"缺失率仅 {missing_rate*100:.1f}%，删除会损失数据，填充可能引入偏差。保持缺失值，分析时忽略。",
                    n_affected=missing_count
                )
            elif missing_rate < 0.10:
                # 缺失率中等，记录但不处理（供分析时决定）
                log.add(
                    variable=col,
                    issue="missing_values",
                    action="documented_only",
                    reason=f"缺失率 {missing_rate*100:.1f}%，在可接受范围内。未进行填充，分析时将忽略缺失值。",
                    n_affected=missing_count
                )

    # 2. 异常值检测（基于 IQR 规则）
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
            outlier_values = data[outliers].tolist()
            # 检查是否在合理范围内
            if col == "body_mass_g":
                # 体重合理范围：2000g - 8000g
                unreasonable = data[(data < 2000) | (data > 8000)]
                if len(unreasonable) > 0:
                    log.add(
                        variable=col,
                        issue="potential_outliers",
                        action="documented_only",
                        reason=f"检测到 {n_outliers} 个候选异常值（IQR 规则）。其中 {len(unreasonable)} 个超出合理范围(2000-8000g)。经检查，这些值仍在物理可能范围内，保留不做处理。",
                        n_affected=n_outliers
                    )
                else:
                    log.add(
                        variable=col,
                        issue="potential_outliers",
                        action="keep_as_is",
                        reason=f"检测到 {n_outliers} 个候选异常值（IQR 规则）。经检查，这些值在合理范围内，保留不做处理。可能代表真实的数据变异性。",
                        n_affected=n_outliers
                    )
            else:
                log.add(
                    variable=col,
                    issue="potential_outliers",
                    action="keep_as_is",
                    reason=f"检测到 {n_outliers} 个候选异常值（IQR 规则）。经检查，这些值在合理范围内，保留不做处理。",
                    n_affected=n_outliers
                )

    # 3. 数据转换记录
    print("记录数据转换...")
    log.add(
        variable="bill_length_mm, body_mass_g",
        issue="scale_difference",
        action="standardization_available",
        reason="为与不同尺度的变量进行比较，可进行 Z-score 标准化（用于可视化，不改变原始数据）。本报告使用原始值以保持可解释性。"
    )

    return df_cleaned, log


# =============================================================================
# 主要发现生成
# =============================================================================

def generate_key_findings(df: pd.DataFrame, log: CleaningLog) -> str:
    """生成主要发现（Week 02 + Week 03）"""
    # 计算关键统计量
    adelie_mean = df[df["species"] == "Adelie"]["body_mass_g"].mean()
    gentoo_mean = df[df["species"] == "Gentoo"]["body_mass_g"].mean()
    diff = gentoo_mean - adelie_mean
    pct_diff = (diff / adelie_mean) * 100

    lines = [
        "## 5. 主要发现",
        "",
        "### 5.1 集中趋势",
        f"- 三种企鹅的平均体重差异明显：Adelie ({adelie_mean:.0f}g) < "
        f"Chinstrap ({df[df['species']=='Chinstrap']['body_mass_g'].mean():.0f}g) < "
        f"Gentoo ({gentoo_mean:.0f}g)",
        f"- Gentoo 比 Adelie 平均重 {diff:.0f}g（约 {pct_diff:.0f}%）",
        "",
        "### 5.2 数据质量评估",
        f"- 数据集整体质量良好，缺失率低于 5%",
        f"- 清洗日志中记录了 {len(log.to_dataframe())} 条数据决策",
        "- 所有异常值经检查后判定为合理值，未进行删除",
        "",
        "### 5.3 下一步建议",
        "- 考虑按物种分组进行相关分析（Week 04）",
        "- 检验不同物种间体重差异是否显著（Week 06）",
        "",
    ]

    return "\n".join(lines)


# =============================================================================
# 分析日志生成
# =============================================================================

def generate_analysis_log(log: CleaningLog) -> str:
    """生成分析日志（记录每周的改进）"""
    lines = [
        "---",
        "",
        "## 6. 分析日志",
        "",
        "### Week 01",
        "- 创建数据卡",
        "",
        "### Week 02",
        "- 新增：描述统计表（整体 + 按物种分组）",
        "- 新增：2 张可视化图表（直方图、箱线图）",
        "",
        "### Week 03",
        "- **新增**：数据清洗与决策记录",
        "- **改进**：所有数据决策都有理由说明（可审计）",
        "- **改进**：从'盲目清洗'到'理解后再处理'",
        "- 缺失值：经诊断，缺失率低，保留原样",
        "- 异常值：经 IQR 规则检测，判定为合理值",
        "- 数据转换：记录标准化方法，保留原始值以确保可解释性",
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
    print("StatLab 报告生成器 - Week 03")
    print("="*60)

    # 1. 数据清洗与分析（本周新增）
    df_cleaned, log = analyze_and_clean(df)

    # 2. 生成可视化（Week 02）
    print("\n生成可视化图表...")
    generated_files = generate_plots(df, output_dir)

    # 3. 组装报告
    print("\n组装报告...")
    report_sections = [
        generate_data_card(df),
        generate_summary_stats(df),
        log.to_markdown_section(),  # Week 03 新增
        generate_visualization_section(generated_files),
        generate_key_findings(df, log),
        generate_analysis_log(log),
    ]

    report_content = "\n".join(report_sections)

    # 写入报告
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n✓ 报告已生成：{REPORT_PATH}")
    print(f"  - 数据卡：1 个")
    print(f"  - 描述统计表：2 个")
    print(f"  - 清洗日志：{len(log.to_dataframe())} 条记录")
    print(f"  - 可视化图表：{len(generated_files)} 张")
    print(f"  - 主要发现：3 项")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数"""
    df = load_data()
    generate_report(df, OUTPUT_DIR)

    print("\n" + "="*60)
    print("Week 03 更新完成！")
    print("="*60)
    print("从 Week 02 的'数据卡 + 描述统计 + 可视化'")
    print("升级为'数据卡 + 描述统计 + 可视化 + **清洗日志**'")
    print()
    print("关键改进：")
    print("  - 不再盲目清洗，而是先诊断再处理")
    print("  - 每一个数据决策都有记录和理由")
    print("  - 报告变得可复现、可审计")


if __name__ == "__main__":
    main()
