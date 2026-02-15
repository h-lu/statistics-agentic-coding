"""
示例：生成数据卡（Data Card）——数据的"身份证"。

运行方式：python3 chapters/week_01/examples/04_data_card.py
预期输出：在 examples/ 目录下生成 data_card.md 文件
"""
from __future__ import annotations

import pandas as pd
import seaborn as sns
from typing import Dict, Any


def generate_data_card(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """
    生成数据卡（Markdown 格式）

    Parameters
    ----------
    df : pd.DataFrame
        数据集
    metadata : dict
        数据的元信息，包含 source、description、collection_date 等

    Returns
    -------
    str
        Markdown 格式的数据卡
    """
    lines = []
    lines.append("# 数据卡（Data Card）\n")

    # 1. 数据来源
    lines.append("## 数据来源\n")
    for key, value in metadata.items():
        lines.append(f"- **{key}**：{value}")
    lines.append("\n")

    # 2. 字段字典
    lines.append("## 字段字典\n")
    lines.append("| 字段名 | 数据类型 | 描述 | 缺失率 |")
    lines.append("|--------|---------|------|--------|")

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_rate = round(df[col].isna().sum() / len(df) * 100, 1)
        lines.append(f"| {col} | {dtype} | （待补充） | {missing_rate}% |")
    lines.append("\n")

    # 3. 规模概览
    lines.append("## 规模概览\n")
    lines.append(f"- **行数**：{len(df)}")
    lines.append(f"- **列数**：{len(df.columns)}")
    lines.append("\n")

    # 4. 缺失概览
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

    # 5. 数据类型分布
    lines.append("## 数据类型分布\n")
    type_counts = df.dtypes.value_counts()
    for dtype, count in type_counts.items():
        lines.append(f"- **{dtype}**：{count} 列")
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

    # 生成数据卡
    data_card = generate_data_card(penguins, metadata)

    # 打印到 stdout
    print(data_card)

    # 写入文件
    output_path = "data_card.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(data_card)

    print(f"\n数据卡已保存到：{output_path}")


if __name__ == "__main__":
    main()
