"""
示例：StatLab 超级线入口脚本 —— 生成可复现的分析报告。

运行方式：python3 chapters/week_01/examples/99_statlab.py
预期输出：在 examples/output/ 目录下生成 report.md
"""
from __future__ import annotations

import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, Any


def generate_data_card(df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
    """生成数据卡（Markdown 格式）"""
    lines = []
    lines.append("## 数据来源\n")
    for key, value in metadata.items():
        lines.append(f"- **{key}**：{value}")
    lines.append("\n")

    # 字段字典
    lines.append("## 字段字典\n")
    lines.append("| 字段名 | 数据类型 | 描述 | 缺失率 |")
    lines.append("|--------|---------|------|--------|")
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_rate = round(df[col].isna().sum() / len(df) * 100, 1)
        lines.append(f"| {col} | {dtype} | （待补充） | {missing_rate}% |")
    lines.append("\n")

    # 规模概览
    lines.append("## 规模概览\n")
    lines.append(f"- **行数**：{len(df)}")
    lines.append(f"- **列数**：{len(df.columns)}")
    lines.append("\n")

    # 缺失概览
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


def generate_report(data_df: pd.DataFrame, output_path: str = "output/report.md") -> Path:
    """
    生成 StatLab 报告

    Parameters
    ----------
    data_df : pd.DataFrame
        数据集
    output_path : str
        输出文件路径

    Returns
    -------
    Path
        输出文件的绝对路径
    """
    # 准备元数据
    metadata = {
        "数据集名称": "Palmer Penguins",
        "来源": "seaborn 内置数据集",
        "原始来源": "Palmer Station, Antarctica LTER",
        "描述": "南极 Palmer Station 的三种企鹅（Adelie, Chinstrap, Gentoo）的形态测量数据",
        "收集时间": "2007-2009 年",
        "单位说明": "长度单位为毫米（mm），重量单位为克（g）"
    }

    # 1. 生成数据卡
    data_card = generate_data_card(data_df, metadata)

    # 2. 组装报告
    report = f"""# StatLab 分析报告

> 本报告由 StatLab 流水线自动生成
> 生成时间：{pd.Timestamp.now().strftime('%Y-%m-%d')}

{data_card}

---

## 下一步

- [ ] 补充字典字典的业务含义
- [ ] 补充描述统计（Week 02）
- [ ] 生成可视化图表（Week 02）

---

## StatLab 进度记录

### Week 01
- ✅ 选择数据集：Palmer Penguins
- ✅ 生成数据卡（来源、字典、规模、缺失概览）
- ✅ 建立可复现报告流水线
"""

    # 3. 写入文件
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"报告已生成：{output_file.absolute()}")
    return output_file


def main() -> None:
    # 加载数据
    penguins = sns.load_dataset("penguins")

    # 生成报告
    generate_report(penguins, "output/report.md")


if __name__ == "__main__":
    main()
