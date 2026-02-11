"""
示例：StatLab 超级线 —— 初始化可复现分析报告项目。

这是 Week 01 的 StatLab 入口脚本，用于生成第一版 report.md。
后续每周会在本周基础上增量修改。

运行方式：python3 chapters/week_01/examples/99_statlab.py
预期输出：在当前目录生成 report.md（StatLab 的权威输出文件）

注意：这是 Week 01 的初始版本，只包含数据卡部分。
后续会逐周添加：描述统计、可视化、推断、建模等内容。
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Any


def generate_statlab_data_card(
    df: pd.DataFrame,
    metadata: dict[str, Any],
) -> str:
    """
    为 StatLab 生成初始数据卡

    这是整个分析报告的"地基"——它告诉读者：
    - 数据从哪来
    - 每列是什么意思
    - 数据规模和缺失情况
    - 能回答什么问题，不能回答什么

    Args:
        df: pandas DataFrame（你的分析数据）
        metadata: 元数据字典，包含以下键：
            - title: 报告标题
            - source: 数据来源
            - collection_date: 数据采集时间
            - contact: 数据联系人（可选）
            - description: 数据描述
            - analysis_type: 分析类型（描述/推断/预测）
            - time_range: 时间范围
            - field_meanings: 字段含义字典 {列名: 业务含义}
            - limitations: 使用限制说明

    Returns:
        Markdown 格式的数据卡文本
    """
    # 从 metadata 中获取字段含义
    field_meanings = metadata.get('field_meanings', {})

    # 构建报告
    report_lines = [
        f"# {metadata['title']}",
        "",
        "---",
        "",
        "## 数据来源",
        "",
        f"- **来源**：{metadata['source']}",
        f"- **采集时间**：{metadata['collection_date']}",
    ]

    if 'contact' in metadata:
        report_lines.append(f"- **数据联系人**：{metadata['contact']}")

    report_lines.extend([
        "",
        "## 数据描述",
        "",
        metadata['description'],
        "",
        "---",
        "",
        "## 统计三问",
        "",
        f"**本周的分析目标属于：{metadata['analysis_type']}**",
        "",
        "统计三问是明确分析目标的第一步：",
        "",
        "| 类型 | 英文 | 说明 | 适用方法 |",
        "|------|------|------|----------|",
        "| **描述** | Description | 说明数据本身的特点，不外推 | 均值、中位数、直方图 |",
        "| **推断** | Inference | 从样本推断总体，带不确定性 | 置信区间、假设检验 |",
        "| **预测** | Prediction | 对未来或未见样本做判断 | 回归模型、分类模型 |",
        "",
        "当前报告的目标类型决定了后续分析方法的选择。",
        "",
        "---",
        "",
        "## 样本规模",
        "",
        f"- **行数**：{df.shape[0]:,}",
        f"- **列数**：{df.shape[1]}",
        "",
    ])

    # 时间范围
    if 'time_range' in metadata:
        report_lines.extend([
            "## 时间范围",
            "",
            metadata['time_range'],
            "",
            "---",
            "",
        ])

    # 字段字典（核心部分）
    report_lines.extend([
        "## 字段字典",
        "",
        "这是数据的「名词解释」——每个列的业务含义、数据类型、缺失情况。",
        "",
        "| 字段名 | 数据类型 | 业务含义 | 缺失率 |",
        "|--------|----------|----------|--------|",
    ])

    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_rate = df[col].isna().mean() * 100
        meaning = field_meanings.get(col, "⚠️ 待补充")

        # 缺失率标记
        missing_mark = ""
        if missing_rate > 20:
            missing_mark = " ⚠️ 高缺失"
        elif missing_rate > 0:
            missing_mark = f" ({missing_rate:.1f}%)"

        report_lines.append(
            f"| {col} | {dtype} | {meaning} | {missing_rate:.1f}%{missing_mark} |"
        )

    report_lines.append("")

    # 缺失概览
    report_lines.extend([
        "## 缺失概览",
        "",
    ])

    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

    if len(missing_summary) > 0:
        report_lines.append("| 字段名 | 缺失数量 | 缺失率 |")
        report_lines.append("|--------|----------|--------|")
        for col, count in missing_summary.items():
            rate = count / len(df) * 100
            report_lines.append(f"| {col} | {count} | {rate:.1f}% |")
    else:
        report_lines.append("✅ 无缺失值")

    report_lines.append("")

    # 使用限制（非常重要！）
    report_lines.extend([
        "---",
        "",
        "## 使用限制与注意事项",
        "",
    ])

    limitations = metadata.get('limitations')
    if limitations:
        report_lines.append(limitations)
    else:
        report_lines.extend([
            "⚠️ **请根据实际数据补充以下信息：**",
            "",
            "1. 本数据集能回答什么问题？",
            "2. 不能回答什么问题？（边界在哪里）",
            "3. 数据采集过程中可能存在的偏差？",
            "4. 缺失值的可能机制？（MCAR/MAR/MNAR）",
        ])

    report_lines.extend([
        "",
        "---",
        "",
        "## StatLab 进展记录",
        "",
        "本报告会逐周迭代更新。记录每周的重要改动：",
        "",
        "| 周次 | 改动内容 | 日期 |",
        "|------|----------|------|",
        f"| Week 01 | 初始化数据卡，建立数据地基 | {pd.Timestamp.now().strftime('%Y-%m-%d')} |",
        "",
        "---",
        "",
        "*本报告由 StatLab 脚本自动生成，是可复现分析的核心交付物。*",
    ])

    return "\n".join(report_lines)


def main() -> None:
    """
    StatLab Week 01 主函数

    使用说明：
    1. 准备你的数据集 CSV 文件
    2. 修改下面的 metadata 字典，填入你的数据信息
    3. 运行脚本，生成 report.md
    """
    print("=" * 70)
    print("StatLab Week 01：初始化数据卡")
    print("=" * 70)

    # ============================================================
    # 用户配置区：根据你的实际数据修改以下内容
    # ============================================================

    # 1. 数据路径（修改为你的数据文件路径）
    DATA_PATH = "data/users.csv"  # 相对路径或绝对路径

    # 2. 元数据配置
    metadata = {
        'title': '电商用户行为分析报告',
        'source': '公司内部用户行为数据库，通过 SQL 导出',
        'collection_date': '2026-01-15',
        'contact': 'data-team@example.com',  # 可选
        'description': (
            '本报告分析 2025 年全年活跃用户的注册、浏览、购买行为数据。'
            '重点关注用户消费模式与城市、性别等特征的关系。'
        ),
        'analysis_type': '描述（Description）',  # 或推断/预测
        'time_range': '2025-01-01 至 2025-12-31',
        'field_meanings': {
            # 根据你的实际列名填写
            'user_id': '用户唯一标识（脱敏）',
            'register_date': '注册日期',
            'age': '用户年龄（岁）',
            'gender': '性别（1=男，2=女，0=未知）',
            'city': '所在城市',
            'total_spend': '年度消费金额（元）',
            'purchase_count': '年度购买次数',
            # ... 其他字段
        },
        'limitations': (
            '⚠️ **重要限制：**\n\n'
            '1. **样本偏差**：本数据仅包含活跃用户（至少有一次购买），'
            '不包含注册后未购买的用户。因此不能用于分析"新用户转化率"等问题。\n\n'
            '2. **时间限制**：数据仅覆盖 2025 年全年，不能用于推断 2024 年或 2026 年的趋势。\n\n'
            '3. **变量限制**：缺少用户收入、教育水平等重要特征，无法进行更深入的用户画像分析。\n\n'
            '4. **因果推断**：本报告仅进行描述性分析，任何因果结论需要进一步的设计（如 A/B 测试）支持。'
        )
    }

    # ============================================================
    # 生成报告（无需修改以下代码）
    # ============================================================

    # 检查数据文件是否存在
    data_file = Path(DATA_PATH)
    if not data_file.exists():
        print(f"\n❌ 错误：数据文件不存在：{DATA_PATH}")
        print("\n提示：")
        print("1. 请确保数据文件路径正确")
        print("2. 或者，先创建一个示例数据文件用于测试")

        # 创建示例数据用于演示
        print("\n创建示例数据文件用于演示...")
        create_sample_data("data/users.csv")
        print(f"✅ 示例数据已创建：data/users.csv")

    # 读取数据
    print(f"\n正在读取数据：{DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ 数据读取成功：{df.shape[0]} 行 × {df.shape[1]} 列")

    # 生成报告
    print("\n正在生成 report.md...")
    report = generate_statlab_data_card(df, metadata)

    # 保存报告
    output_path = "report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ 报告已生成：{output_path}")
    print("\n" + "=" * 70)
    print("StatLab Week 01 完成！")
    print("=" * 70)
    print("\n下一步：")
    print("1. 打开 report.md，检查数据卡内容")
    print("2. 根据实际情况补充字段含义和使用限制")
    print("3. 用 git 提交：git add report.md && git commit -m 'draft: initial data card'")
    print("4. 下周我们会在这份报告上添加描述统计和可视化")


def create_sample_data(path: str) -> None:
    """创建示例数据文件（用于演示）"""
    # 确保目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # 创建示例数据
    import numpy as np
    np.random.seed(42)

    n = 1000
    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'register_date': pd.date_range('2025-01-01', periods=n, freq='1h'),
        'age': np.random.randint(18, 65, n),
        'gender': np.random.choice([1, 2, 0], n, p=[0.45, 0.5, 0.05]),
        'city': np.random.choice(['北京', '上海', '深圳', '杭州', '成都'], n),
        'total_spend': np.random.exponential(2000, n),
        'purchase_count': np.random.poisson(5, n),
    })

    # 添加一些缺失值
    df.loc[np.random.choice(n, 50, replace=False), 'age'] = np.nan
    df.loc[np.random.choice(n, 30, replace=False), 'city'] = np.nan

    df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
