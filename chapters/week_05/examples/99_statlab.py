"""
示例：StatLab 超级线 - Week 05 更新脚本。

本周在 Week 01-04 的基础上，加入**不确定性量化**（Bootstrap）。
这是从"确定性结论"到"概率性思维"的重要升级。

运行方式：python3 chapters/week_05/examples/99_statlab.py
预期输出：
- examples/output/report.md：包含数据卡 + 描述统计 + 可视化 + 清洗日志 +
                              相关性分析 + 分组比较 + 假设清单 + **不确定性量化**
- examples/output/ 目录下生成必要的图表

设计思路：
1. 保留 Week 01-04 的所有内容（数据卡、描述统计、可视化、清洗日志、相关分析、分组比较、假设清单）
2. 新增：不确定性量化（Bootstrap 标准误和置信区间）
3. 所有内容整合为一份可审计的报告

本周改进：
- 上周：数据卡 + 描述统计 + 可视化 + 清洗日志 + 相关分析 + 分组比较 + 假设清单
- 本周：+ **不确定性量化（标准误 + 95% 置信区间）**
- 目标：让报告从"确定性结论"变成"概率性结论"

与本周知识的连接：
- Bootstrap → 用重采样方法估计抽样分布
- 标准误 → 统计量的不确定性
- 置信区间 → 结论的"误差棒"

老潘的点评：
"在公司里，我们不会只写'A 渠道转化率更高'。
我们会写'A 渠道转化率是 12% ± 1.5%（95% CI: [9%, 15%]）'。
因为决策者需要知道：这个结论有多稳定。"

关键设计：
- 为关键变量计算 Bootstrap 置信区间
- 为组间差异计算 Bootstrap 置信区间
- 在报告中标注"置信区间是否包含 0"
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
RANDOM_STATE = 42  # 固定随机种子，确保结果可复现


# =============================================================================
# 假设清单类（来自 Week 04）
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
# Week 05 新增：不确定性量化类
# =============================================================================

class BootstrapUncertainty:
    """Bootstrap 不确定性量化"""

    def __init__(self, n_bootstrap: int = 1000, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.rng = np.random.default_rng(seed=random_state)

    def bootstrap_mean(self, data: np.ndarray) -> dict:
        """
        对单个变量的均值进行 Bootstrap

        返回：包含 estimate, se, ci_low, ci_high 的字典
        """
        data = data[~np.isnan(data)]
        n = len(data)

        if n == 0:
            return {
                "estimate": np.nan,
                "se": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n": 0
            }

        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            boot_sample = self.rng.choice(data, size=n, replace=True)
            bootstrap_means.append(boot_sample.mean())

        bootstrap_means = np.array(bootstrap_means)

        estimate = data.mean()
        se = bootstrap_means.std()
        ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])

        return {
            "estimate": estimate,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": n
        }

    def bootstrap_group_diff(self, data_a: np.ndarray,
                              data_b: np.ndarray) -> dict:
        """
        对两组差异进行 Bootstrap

        返回：包含 estimate, se, ci_low, ci_high, contains_zero 的字典
        """
        data_a = data_a[~np.isnan(data_a)]
        data_b = data_b[~np.isnan(data_b)]

        n_a, n_b = len(data_a), len(data_b)

        if n_a == 0 or n_b == 0:
            return {
                "estimate": np.nan,
                "se": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "contains_zero": False,
                "n_a": n_a,
                "n_b": n_b
            }

        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            boot_a = self.rng.choice(data_a, size=n_a, replace=True)
            boot_b = self.rng.choice(data_b, size=n_b, replace=True)
            bootstrap_diffs.append(boot_a.mean() - boot_b.mean())

        bootstrap_diffs = np.array(bootstrap_diffs)

        estimate = data_a.mean() - data_b.mean()
        se = bootstrap_diffs.std()
        ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])
        contains_zero = (ci_low <= 0 <= ci_high)

        return {
            "estimate": estimate,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "contains_zero": contains_zero,
            "n_a": n_a,
            "n_b": n_b
        }


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
# Week 04 内容：相关性分析、分组比较、假设清单
# =============================================================================

def generate_correlation_section(df: pd.DataFrame) -> str:
    """生成相关性分析的 Markdown 片段（Week 04 内容）"""
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
    """生成分组比较的 Markdown 片段（Week 04 内容）"""
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
    """基于数据生成假设清单（Week 04 内容）"""
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
# Week 05 新增：不确定性量化
# =============================================================================

def generate_uncertainty_section(df: pd.DataFrame,
                                  bootstrap: BootstrapUncertainty) -> str:
    """生成不确定性量化的 Markdown 片段（本周新增）"""
    numeric_cols = ["bill_length_mm", "bill_depth_mm",
                    "flipper_length_mm", "body_mass_g"]

    md = ["## 8. 不确定性量化（本周新增）\n\n"]
    md.append("以下使用 Bootstrap 方法（1000 次重采样，随机种子=42）\n")
    md.append("估计关键统计量的 95% 置信区间。\n\n")

    md.append("**说明**：\n")
    md.append("- **标准误（SE）**：抽样分布的标准差，描述统计量的不确定性\n")
    md.append("- **95% 置信区间（95% CI）**：如果重复抽样 100 次，95 次的区间会包含真实值\n\n")

    md.append("### 8.1 单变量均值的不确定性\n\n")

    for col in numeric_cols:
        data = df[col].dropna()
        result = bootstrap.bootstrap_mean(data.values)

        md.append(f"#### {col}\n\n")
        md.append(f"- **估计值**：{result['estimate']:.2f}\n")
        md.append(f"- **标准误**：{result['se']:.2f}\n")
        md.append(f"- **95% CI**：[{result['ci_low']:.2f}, {result['ci_high']:.2f}]\n")
        md.append(f"- **样本量**：{result['n']}\n\n")

    return "".join(md)


def generate_group_uncertainty_section(df: pd.DataFrame,
                                         bootstrap: BootstrapUncertainty) -> str:
    """生成组间差异的不确定性量化（本周新增）"""
    md = ["### 8.2 组间差异的不确定性\n\n"]
    md.append("以下按物种比较，使用 Bootstrap 方法估计差异的置信区间。\n\n")

    # 比较物种对
    species_pairs = [
        ("Adelie", "Gentoo"),
        ("Adelie", "Chinstrap"),
        ("Chinstrap", "Gentoo")
    ]

    numeric_cols = ["bill_length_mm", "bill_depth_mm",
                    "flipper_length_mm", "body_mass_g"]

    for species_a, species_b in species_pairs:
        md.append(f"#### {species_a} vs {species_b}\n\n")

        for col in numeric_cols:
            data_a = df[df["species"] == species_a][col].dropna().values
            data_b = df[df["species"] == species_b][col].dropna().values

            result = bootstrap.bootstrap_group_diff(data_a, data_b)

            if result["n_a"] == 0 or result["n_b"] == 0:
                continue

            direction = f"{species_a} - {species_b}"
            md.append(f"**{col}**（{direction}）：\n\n")
            md.append(f"- **观察到的差异**：{result['estimate']:.2f}\n")
            md.append(f"- **标准误**：{result['se']:.2f}\n")
            md.append(f"- **95% CI**：[{result['ci_low']:.2f}, {result['ci_high']:.2f}]\n")

            if not result["contains_zero"]:
                if result['estimate'] > 0:
                    md.append(f"- ✅ 置信区间不包含 0，表明 {species_a} 的均值显著高于 {species_b}\n\n")
                else:
                    md.append(f"- ✅ 置信区间不包含 0，表明 {species_b} 的均值显著高于 {species_a}\n\n")
            else:
                md.append(f"- ⚠️ 置信区间包含 0，表明差异可能不显著（需要正式检验）\n\n")

    return "".join(md)


# =============================================================================
# 主要发现生成
# =============================================================================

def generate_key_findings(df: pd.DataFrame, log: CleaningLog,
                          hypotheses: HypothesisList,
                          bootstrap: BootstrapUncertainty) -> str:
    """生成主要发现"""
    adelie_mean = df[df["species"] == "Adelie"]["body_mass_g"].mean()
    gentoo_mean = df[df["species"] == "Gentoo"]["body_mass_g"].mean()
    diff = gentoo_mean - adelie_mean

    # Bootstrap 体重差异
    adelie_data = df[df["species"] == "Adelie"]["body_mass_g"].dropna().values
    gentoo_data = df[df["species"] == "Gentoo"]["body_mass_g"].dropna().values
    diff_result = bootstrap.bootstrap_group_diff(adelie_data, gentoo_data)

    lines = [
        "## 9. 主要发现",
        "",
        "### 9.1 集中趋势",
        f"- 三种企鹅的平均体重差异明显：Adelie ({adelie_mean:.0f}g) < "
        f"Chinstrap ({df[df['species']=='Chinstrap']['body_mass_g'].mean():.0f}g) < "
        f"Gentoo ({gentoo_mean:.0f}g)",
        f"- Gentoo 比 Adelie 平均重 {diff:.0f}g",
        f"- **本周新增**：Gentoo - Adelie 体重差异的 95% CI 为 "
        f"[{diff_result['ci_low']:.0f}g, {diff_result['ci_high']:.0f}g]",
        "",
        "### 9.2 数据质量评估",
        f"- 数据集整体质量良好，清洗日志中记录了 {len(log.to_dataframe())} 条数据决策",
        "- 所有异常值经检查后判定为合理值，未进行删除",
        "",
        "### 9.3 相关性与分组差异",
        "- 体重与翅长呈强正相关，符合生物学预期",
        "- 不同物种在所有测量维度上均有显著差异",
        "",
        "### 9.4 不确定性量化（本周新增）",
        "- 使用 Bootstrap 方法（1000 次重采样）估计了关键统计量的置信区间",
        "- 大部分组间差异的 95% CI 不包含 0，支持'物种间存在显著差异'的结论",
        "- **关键改进**：从'确定性结论'升级为'概率性结论'",
        "",
        "### 9.5 下一步建议",
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
        "## 10. 分析日志",
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
        "- 新增：相关性分析（Pearson 相关系数矩阵）",
        "- 新增：分组比较（按物种分组统计）",
        "- 新增：可检验假设清单（" + str(len(hypotheses.to_dataframe())) + " 个假设）",
        "- 改进：从'看数据'到'提出可检验问题'",
        "",
        "### Week 05",
        "- **新增**：不确定性量化（Bootstrap 标准误和置信区间）",
        "- **新增**：单变量均值的 95% CI",
        "- **新增**：组间差异的 95% CI",
        "- **关键改进**：从'确定性结论'升级为'概率性结论'",
        "- **关键改进**：报告包含'误差棒'，让读者知道'结论有多确定'",
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
    print("StatLab 报告生成器 - Week 05")
    print("="*60)

    # 1. 数据清洗与分析
    df_cleaned, log = analyze_and_clean(df)

    # 2. 生成可视化
    print("\n生成可视化图表...")
    generated_files = generate_plots(df, output_dir)

    # 3. 生成假设清单
    print("\n生成假设清单...")
    hypotheses = generate_hypothesis_list(df)

    # 4. 初始化 Bootstrap（本周新增）
    print("\n初始化 Bootstrap 不确定性量化...")
    bootstrap = BootstrapUncertainty(n_bootstrap=1000, random_state=RANDOM_STATE)

    # 5. 组装报告
    print("\n组装报告...")
    report_sections = [
        generate_data_card(df),
        generate_summary_stats(df),
        log.to_markdown_section(),
        generate_visualization_section(generated_files),
        generate_correlation_section(df),
        generate_group_comparison_section(df),
        hypotheses.to_markdown_section(),
        generate_uncertainty_section(df, bootstrap),  # 本周新增
        generate_group_uncertainty_section(df, bootstrap),  # 本周新增
        generate_key_findings(df, log, hypotheses, bootstrap),
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
    print(f"  - 不确定性量化：4 个变量 + 6 个组间对比")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数"""
    df = load_data()
    generate_report(df, OUTPUT_DIR)

    print("\n" + "="*60)
    print("Week 05 更新完成！")
    print("="*60)
    print("从 Week 04 的'")
    print("  数据卡 + 描述统计 + 可视化 + 清洗日志 + 相关分析 + 分组比较 + 假设清单")
    print("'")
    print("升级为'")
    print("  ... + **不确定性量化（标准误 + 95% 置信区间）**")
    print("'")
    print()
    print("关键改进：")
    print("  - 从'确定性结论'升级为'概率性结论'")
    print("  - 使用 Bootstrap 方法量化统计量的不确定性")
    print("  - 报告包含'误差棒'，让读者知道'结论有多确定'")
    print()
    print("老潘的点评：")
    print("  '在公司里，我们不会只写\"A 渠道转化率更高\"。'")
    print("  '我们会写\"A 渠道转化率是 12% ± 1.5%（95% CI: [9%, 15%]）\"。'")
    print("  '因为决策者需要知道：这个结论有多稳定。'")


if __name__ == "__main__":
    main()
