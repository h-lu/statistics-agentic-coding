"""
示例：增强版数据卡 —— 自动推断统计类型。

本示例演示如何自动推断列的"统计类型"（而非 pandas dtype），
帮助你判断每列应该用哪种统计量和图表。

运行方式：python3 chapters/week_01/examples/05_enhanced_data_card.py
"""
from __future__ import annotations

import pandas as pd
import seaborn as sns
from typing import Dict, Any, Literal


def infer_statistical_type(series: pd.Series) -> Dict[str, Any]:
    """
    推断一列数据的统计类型。

    统计类型和 pandas dtype 不同：
    - pandas dtype 是存储类型（int64, float64, object...）
    - 统计类型是语义类型（连续数值、离散数值、名义分类、有序分类）

    Parameters
    ----------
    series : pd.Series
        要推断的列

    Returns
    -------
    dict
        包含 stat_type（统计类型）和 reason（推断依据）
    """
    n_unique = series.nunique()
    n_total = len(series)
    dtype = str(series.dtype)

    # 1. 数值型判断
    if pd.api.types.is_numeric_dtype(series):
        # 离散数值：唯一值少，或明显是整数计数
        if n_unique <= 10 or (pd.api.types.is_integer_dtype(series) and n_unique / n_total < 0.05):
            return {
                "stat_type": "discrete_numeric",
                "reason": f"数值型，但只有 {n_unique} 个唯一值（可能代表类别或计数）"
            }
        # 连续数值：唯一值多，有小数
        else:
            return {
                "stat_type": "continuous_numeric",
                "reason": f"连续数值型，{n_unique} 个唯一值"
            }

    # 2. 分类型判断
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        # 尝试检测是否有自然顺序（如 Low/Medium/High）
        sample_values = series.dropna().head(20).tolist()
        ordinal_hints = ["low", "medium", "high", "小", "中", "大", "差", "良", "优"]

        if any(hint in str(v).lower() for v in sample_values for hint in ordinal_hints):
            return {
                "stat_type": "ordinal_categorical",
                "reason": "可能是有序分类（检测到顺序性关键词）"
            }
        else:
            return {
                "stat_type": "nominal_categorical",
                "reason": f"名义分类，{n_unique} 个类别"
            }

    # 3. 时间型（简化处理）
    if pd.api.types.is_datetime64_any_dtype(series):
        return {
            "stat_type": "datetime",
            "reason": "时间/日期类型"
        }

    # 4. 布尔型
    if pd.api.types.is_bool_dtype(series):
        return {
            "stat_type": "binary",
            "reason": "二值变量（True/False）"
        }

    # 5. 兜底
    return {
        "stat_type": "unknown",
        "reason": f"无法确定类型，pandas dtype: {dtype}"
    }


def generate_enhanced_data_card(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """
    生成增强版数据卡，包含自动推断的统计类型。

    Parameters
    ----------
    df : pd.DataFrame
        数据集
    metadata : dict
        数据的元信息

    Returns
    -------
    str
        Markdown 格式的增强版数据卡
    """
    lines = []
    lines.append("# 增强版数据卡（Enhanced Data Card）\n")

    # 1. 数据来源
    lines.append("## 数据来源\n")
    for key, value in metadata.items():
        lines.append(f"- **{key}**：{value}")
    lines.append("\n")

    # 2. 增强字段字典
    lines.append("## 字段字典（含统计类型）\n")
    lines.append("| 字段名 | pandas 类型 | 统计类型 | 推断依据 | 缺失率 |")
    lines.append("|--------|------------|---------|---------|--------|")

    for col in df.columns:
        dtype = str(df[col].dtype)
        type_info = infer_statistical_type(df[col])
        missing_rate = round(df[col].isna().sum() / len(df) * 100, 1)
        lines.append(f"| {col} | {dtype} | {type_info['stat_type']} | {type_info['reason']} | {missing_rate}% |")
    lines.append("\n")

    # 3. 统计类型分布
    lines.append("## 统计类型分布\n")
    type_counts: Dict[str, int] = {}
    for col in df.columns:
        type_info = infer_statistical_type(df[col])
        t = type_info["stat_type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    for t, count in sorted(type_counts.items()):
        lines.append(f"- **{t}**：{count} 列")
    lines.append("\n")

    # 4. 分析建议
    lines.append("## 分析建议\n")
    lines.append("根据统计类型，建议以下分析方法：\n")

    if "continuous_numeric" in type_counts:
        lines.append("**连续数值型列**：")
        lines.append("- 描述统计：均值、中位数、标准差、分位数")
        lines.append("- 可视化：直方图、箱线图、密度图")
        lines.append("- 相关性：Pearson 相关系数\n")

    if "discrete_numeric" in type_counts:
        lines.append("**离散数值型列**：")
        lines.append("- 描述统计：频数表、众数")
        lines.append("- 可视化：柱状图")
        lines.append("- 注意：可能需要当作分类型处理\n")

    if "nominal_categorical" in type_counts:
        lines.append("**名义分类型列**：")
        lines.append("- 描述统计：频数表、比例")
        lines.append("- 可视化：柱状图、饼图")
        lines.append("- 检验：卡方检验（与另一分类列）\n")

    if "ordinal_categorical" in type_counts:
        lines.append("**有序分类型列**：")
        lines.append("- 描述统计：频数表、中位数（可解释）")
        lines.append("- 可视化：有序柱状图")
        lines.append("- 相关性：Spearman 相关系数\n")

    # 5. 规模概览
    lines.append("## 规模概览\n")
    lines.append(f"- **行数**：{len(df)}")
    lines.append(f"- **列数**：{len(df.columns)}")
    lines.append("\n")

    # 6. 缺失概览
    lines.append("## 缺失概览\n")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        for col, count in missing.items():
            rate = round(count / len(df) * 100, 1)
            lines.append(f"- **{col}**：{count} ({rate}%)")
    else:
        lines.append("- 无缺失值")
    lines.append("\n")

    return "\n".join(lines)


def main() -> None:
    # 加载数据
    penguins = sns.load_dataset("penguins")

    # 准备元数据
    metadata = {
        "数据集名称": "Palmer Penguins",
        "来源": "seaborn 内置数据集",
        "原始来源": "Palmer Station, Antarctica LTER",
        "描述": "南极 Palmer Station 的三种企鹅（Adelie, Chinstrap, Gentoo）的形态测量数据",
        "收集时间": "2007-2009 年",
        "单位说明": "长度单位为毫米（mm），重量单位为克（g）"
    }

    # 生成增强版数据卡
    enhanced_card = generate_enhanced_data_card(penguins, metadata)

    # 打印到 stdout
    print(enhanced_card)

    # 写入文件
    output_path = "enhanced_data_card.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(enhanced_card)

    print(f"\n增强版数据卡已保存到：{output_path}")

    # 额外演示：检测"伪装成数值的分类变量"
    print("\n" + "=" * 50)
    print("【额外演示】检测伪装成数值的分类变量")
    print("=" * 50)

    # 创建一个包含 zipcode 的测试数据
    test_df = pd.DataFrame({
        "zipcode": [10001, 90210, 60601, 77001, 33101],  # 看起来像数字，实际是分类
        "age": [25, 30, 35, 40, 45],
        "satisfaction": ["low", "medium", "high", "medium", "low"]
    })

    for col in test_df.columns:
        type_info = infer_statistical_type(test_df[col])
        print(f"\n{col}:")
        print(f"  pandas dtype: {test_df[col].dtype}")
        print(f"  统计类型: {type_info['stat_type']}")
        print(f"  推断依据: {type_info['reason']}")


if __name__ == "__main__":
    main()
