"""
示例：数据卡生成器 —— 给数据办一张"身份证"。

运行方式：python3 chapters/week_01/examples/04_data_card.py
预期输出：在当前目录生成 data_card.md 文件。
"""
from __future__ import annotations

import pandas as pd
from datetime import datetime
from typing import Any


def create_data_card(
    df: pd.DataFrame,
    title: str,
    data_source: str,
    description: str,
    field_meanings: dict[str, str] | None = None,
    time_range: str | None = None,
    analysis_type: str = "描述（Description）",
    limitations: str | None = None,
) -> str:
    """
    生成数据卡的 Markdown 文本

    Args:
        df: pandas DataFrame
        title: 数据集标题
        data_source: 数据来源描述
        description: 数据描述
        field_meanings: 字段含义字典 {列名: 业务含义}
        time_range: 时间范围
        analysis_type: 分析类型（描述/推断/预测）
        limitations: 使用限制说明

    Returns:
        Markdown 格式的数据卡文本
    """
    card_lines = [
        f"# {title}",
        "",
        "## 数据来源",
        f"- **来源**：{data_source}",
        f"- **生成时间**：{datetime.now().strftime('%Y-%m-%d')}",
        "",
        "## 数据描述",
        description,
        "",
        "## 统计三问",
        f"本周的分析目标属于：**{analysis_type}**",
        "",
        "**三类目标的区别：**",
        "- **描述（Description）**：说明数据本身的特点，结论只适用于这批数据",
        "- **推断（Inference）**：从样本推断总体，结论带有不确定性",
        "- **预测（Prediction）**：对未来或未见样本做出判断，需要建模",
        "",
        "## 样本规模",
        f"- **行数**：{df.shape[0]:,}",
        f"- **列数**：{df.shape[1]}",
        "",
    ]

    # 时间范围
    if time_range:
        card_lines.extend([
            "## 时间范围",
            time_range,
            "",
        ])

    # 字段字典
    card_lines.extend([
        "## 字段字典",
        "",
        "| 字段名 | 数据类型 | 业务含义 | 缺失率 | 示例值 |",
        "|--------|----------|----------|--------|--------|",
    ])

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_rate = df[col].isna().mean() * 100
        meaning = field_meanings.get(col, "待补充") if field_meanings else "待补充"

        # 获取示例值（取第一个非空值）
        sample_value = "N/A"
        non_null_vals = df[col].dropna()
        if len(non_null_vals) > 0:
            val = non_null_vals.iloc[0]
            if isinstance(val, str):
                sample_value = val[:20] + "..." if len(val) > 20 else val
            else:
                sample_value = str(val)

        card_lines.append(
            f"| {col} | {dtype} | {meaning} | {missing_rate:.1f}% | {sample_value} |"
        )

    card_lines.append("")

    # 缺失概览
    card_lines.extend([
        "## 缺失概览",
        "",
    ])

    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

    if len(missing_summary) > 0:
        for col, count in missing_summary.items():
            rate = count / len(df) * 100
            card_lines.append(f"- **{col}**：{count} 个缺失 ({rate:.1f}%)")
    else:
        card_lines.append("- ✅ 无缺失值")

    card_lines.append("")

    # 基本统计（数值列）
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        card_lines.extend([
            "## 数值列基本统计",
            "",
        ])

        stats = df[numeric_cols].describe().round(2)
        card_lines.append("| 列名 | 均值 | 标准差 | 最小值 | 25% | 50% | 75% | 最大值 |")
        card_lines.append("|------|------|--------|--------|-----|-----|-----|--------|")

        for col in numeric_cols:
            row = stats[col]
            card_lines.append(
                f"| {col} | {row['mean']:.2f} | {row['std']:.2f} | "
                f"{row['min']:.2f} | {row['25%']:.2f} | {row['50%']:.2f} | "
                f"{row['75%']:.2f} | {row['max']:.2f} |"
            )

        card_lines.append("")

    # 使用限制
    if limitations:
        card_lines.extend([
            "## 使用限制与注意事项",
            "",
            limitations,
            "",
        ])
    else:
        card_lines.extend([
            "## 使用限制与注意事项",
            "",
            "待补充：本数据集能回答什么问题？不能回答什么问题？",
            "",
        ])

    # 元数据
    card_lines.extend([
        "---",
        "",
        "*本数据卡由脚本自动生成，请根据实际情况补充完整信息。*",
    ])

    return "\n".join(card_lines)


def main() -> None:
    """主函数 —— 演示数据卡生成"""
    # 创建示例数据
    data = {
        "user_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "register_date": ["2025-01-15", "2025-02-20", "2025-03-10", None, "2025-04-05", "2025-05-12", "2025-06-01", "2025-06-15"],
        "age": [25, 30, None, 28, 35, 22, 29, 31],
        "gender": [1, 2, 1, 2, 1, None, 2, 1],  # 1=男, 2=女
        "city": ["北京", "上海", "深圳", "北京", "上海", "深圳", None, "北京"],
        "total_spend": [1200.50, 3500.00, 800.00, 2100.00, 5600.00, 450.00, 1800.00, 3200.00],
        "purchase_count": [3, 12, 1, 5, 20, 1, 4, 8],
    }

    df = pd.DataFrame(data)

    # 字段含义
    field_meanings = {
        "user_id": "用户唯一标识",
        "register_date": "注册日期",
        "age": "用户年龄（岁）",
        "gender": "性别（1=男，2=女，0=未知）",
        "city": "所在城市",
        "total_spend": "年度消费金额（元）",
        "purchase_count": "年度购买次数",
    }

    # 生成数据卡
    card = create_data_card(
        df=df,
        title="电商用户行为分析数据卡",
        data_source="公司内部用户行为数据库，SQL 导出",
        description="2025 年上半年（1-6 月）活跃用户的注册、浏览、购买行为数据。包含用户基本信息和消费行为指标。",
        field_meanings=field_meanings,
        time_range="2025-01-01 至 2025-06-30",
        analysis_type="描述（Description）",
        limitations=(
            "⚠️ **重要限制：**\n"
            "1. 本数据仅包含活跃用户（至少有过一次购买），不包含注册后未购买的用户；因此不适用于'新用户转化率'等分析。\n"
            "2. user_id 已脱敏，无法关联到其他数据源。\n"
            "3. 部分字段存在缺失值（register_date、age、gender、city），分析时需注意。"
        )
    )

    # 保存到文件
    output_path = "data_card.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(card)

    print(f"✅ 数据卡已生成到 {output_path}")
    print("\n预览：")
    print("-" * 70)
    print(card[:500] + "...")
    print("-" * 70)


if __name__ == "__main__":
    main()
