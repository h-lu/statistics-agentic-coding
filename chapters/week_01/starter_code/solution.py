"""
Week 01 作业参考实现

本文件提供了 Week 01 作业的参考答案。
当你在作业中遇到困难时，可以查看此文件来理解正确的实现方式。

建议：先自己尝试完成作业，实在卡住时再参考答案。
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


def identify_data_type(df: pd.DataFrame, col_name: str) -> str:
    """
    判断某列的数据类型（统计学角度，不是 pandas dtype）

    Parameters
    ----------
    df : pd.DataFrame
        数据集
    col_name : str
        列名

    Returns
    -------
    str
        数据类型描述：'数值型（连续）'、'数值型（离散）'、'分类型（名义）'、'分类型（有序）'
    """
    if col_name not in df.columns:
        return f"错误：列 '{col_name}' 不存在"

    col = df[col_name]
    dtype = col.dtype

    # 数值型判断
    if pd.api.types.is_numeric_dtype(dtype):
        # 判断是连续还是离散
        unique_count = col.nunique()
        if unique_count > 10:  # 经验法则：唯一值超过10个可能是连续型
            return "数值型（连续）"
        else:
            return "数值型（离散）"

    # 分类型判断
    else:
        # 判断是有序还是名义
        if dtype.name == "category" and col.cat.ordered:
            return "分类型（有序）"
        else:
            return "分类型（名义）"


def load_and_inspect_data(source: str = "seaborn") -> pd.DataFrame:
    """
    加载数据并进行基础检查

    Parameters
    ----------
    source : str
        数据来源：'seaborn' 使用内置数据集，其他值视为文件路径

    Returns
    -------
    pd.DataFrame
        加载的数据集
    """
    if source == "seaborn":
        df = sns.load_dataset("penguins")
        print(f"成功加载 seaborn 内置数据集：Palmer Penguins")
    else:
        try:
            df = pd.read_csv(source)
            print(f"成功加载文件：{source}")
        except FileNotFoundError:
            print(f"错误：文件 '{source}' 未找到")
            raise
        except UnicodeDecodeError:
            print(f"错误：文件编码问题，尝试指定 encoding='gbk' 或 'gb18030'")
            raise

    # 打印基础信息
    print(f"\n数据规模：{df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"\n数据类型：")
    print(df.dtypes)
    print(f"\n缺失值统计：")
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            rate = round(count / len(df) * 100, 1)
            print(f"  - {col}: {count} ({rate}%)")
    else:
        print("  无缺失值")

    return df


def main() -> None:
    """主函数：演示完整的数据卡生成流程"""
    print("=" * 70)
    print("Week 01 参考实现：数据卡生成器")
    print("=" * 70)
    print()

    # 1. 加载数据
    df = load_and_inspect_data("seaborn")
    print()

    # 2. 演示数据类型判断
    print("数据类型判断示例：")
    for col in ["species", "bill_length_mm", "island", "body_mass_g"]:
        data_type = identify_data_type(df, col)
        print(f"  - {col}: {data_type}")
    print()

    # 3. 生成数据卡
    metadata = {
        "数据集名称": "Palmer Penguins",
        "来源": "seaborn 内置数据集",
        "原始来源": "Palmer Station, Antarctica LTER",
        "描述": "南极 Palmer Station 的三种企鹅（Adelie, Chinstrap, Gentoo）的形态测量数据",
        "收集时间": "2007-2009 年",
        "单位说明": "长度单位为毫米（mm），重量单位为克（g）"
    }

    data_card = generate_data_card(df, metadata)

    # 4. 保存数据卡
    output_file = "starter_code_data_card.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(data_card)

    print(f"数据卡已生成：{output_file}")
    print()
    print("=" * 70)
    print("数据卡内容预览：")
    print("=" * 70)
    print(data_card)


if __name__ == "__main__":
    main()
