#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StatLab 示例：报告摘要生成模块

本模块用于在 StatLab 报告中追加完整的"不确定性量化"章节摘要。
功能包括：
1. 汇总 Bootstrap 分析结果到表格
2. 生成规范的 Markdown 章节格式
3. 为 report.md 提供可直接粘贴的内容

运行方式：python3 chapters/week_05/examples/05_statlab_summary.py
预期输出：打印可粘贴到 report.md 的内容
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# 报告模板生成
# =============================================================================

def generate_uncertainty_summary_template(
    report_path: str = 'report.md'
) -> str:
    """
    生成不确定性量化章节的模板

    参数：
        report_path: 报告文件路径（用于追加内容）

    返回：
        str: Markdown 格式的章节内容
    """

    template = """

## 不确定性量化

> 本章说明关键统计量的波动范围，为后续假设检验提供基础。
> 生成时间：2026-02-12

### 核心统计量的稳定性

| 统计量 | 点估计 | 95% CI (Bootstrap) | 标准误 | 解读 |
|--------|--------|---------------------|---------|------|
| 钻石用户平均消费 | [TODO] 元 | [[TODO_LOW], [TODO_HIGH]] | [TODO] 元 | [填写解读] |
| 普通用户平均消费 | [TODO] 元 | [[TODO_LOW], [TODO_HIGH]] | [TODO] 元 | [填写解读] |
| 均值差异 | [TODO] 元 | [[TODO_LOW], [TODO_HIGH]] | [TODO] 元 | **[是否包含0]** |
| 收入-消费相关系数 | r = [TODO] | [[TODO_LOW], [TODO_HIGH]] | [TODO] | [填写解读] |

### 关键发现

1. **均值差异稳定性**：Bootstrap 95% CI [包含/不包含] 0，说明 [结论]
2. **相关性稳健性**：Pearson 和 Spearman 相关系数 [一致/不一致]，[解读]
3. **样本量充足性**：当前样本量下，标准误 [TODO]，[评估]

### 敏感性分析

- **剔除极端值前后**：均值差异变化 < [TODO]%
- **改变 Bootstrap 次数（1000 vs 10000）**：CI 边界变化 < [TODO]%
- **使用中位数代替均值**：钻石用户中位数仍是普通用户的 [TODO] 倍

### 数据局限

- **Bootstrap 假设**：样本代表性良好，否则 CI 无效
- **未控制混杂**：[列出 Week 04 识别的混杂变量]
- **横截面数据**：无法确定因果方向
- **样本量偏差**：[如钻石用户样本量较小]

### 下一步

Week 06 将进行正式统计检验：
- 假设清单 H1-H3 的 t 检验/ANOVA
- 效应量计算（Cohen's d, η²）
- 前提假设检查（正态性、方差齐性）

---
"""
    return template


def append_to_report(
    content: str,
    report_path: str = 'report.md'
) -> None:
    """
    将内容追加到报告文件

    参数：
        content: 要追加的内容
        report_path: 报告文件路径
    """
    try:
        with open(report_path, 'a', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ 内容已追加到 {report_path}")
    except FileNotFoundError:
        print(f"✗ 文件 {report_path} 不存在，请先创建报告")
        print("\n请手动复制以下内容到 report.md:\n")
        print(content)


def generate_example_summary() -> str:
    """
    生成一个完整的示例不确定性章节（使用模拟数据结果）
    """
    return """

## 不确定性量化

> 本章说明关键统计量的波动范围，为后续假设检验提供基础。
> 生成时间：2026-02-12

### 核心统计量的稳定性

基于 Bootstrap 重采样（10000 次）的结果汇总：

| 统计量 | 点估计 | 95% CI (Bootstrap) | 标准误 | 解读 |
|--------|--------|---------------------|---------|------|
| 钻石用户平均消费 | 4,200 元 | [3,800, 4,650] | 210 元 | 区间较窄，估计稳定 |
| 普通用户平均消费 | 850 元 | [780, 920] | 85 元 | 样本量较大，波动较小 |
| 均值差异 | 3,350 元 | [2,980, 3,720] | 190 元 | **不包含 0，预期显著** |
| 收入-消费相关系数 | 0.520 | [0.380, 0.640] | 0.067 | 中度正相关，稳定 |

### 关键发现

1. **均值差异稳定**：Bootstrap 95% CI 不包含 0，说明钻石用户与普通用户的消费差异在统计上显著（Week 06 将进行正式 t 检验）
2. **相关性稳健**：Pearson 和 Spearman 相关系数一致（r ≈ 0.5），异常值影响有限
3. **样本量充足**：当前样本量下，标准误已控制在可接受范围（< 10% 点估计）

### 敏感性分析

为检验结论的稳健性，进行了以下敏感性测试：

- **剔除前 5% 极端值**：均值差异变化 < 8%，结论稳健
- **改变 Bootstrap 次数（1000 vs 10000）**：CI 边界变化 < 2%，收敛稳定
- **使用中位数代替均值**：钻石用户中位数仍是普通用户的 4.2 倍
- **Bootstrap 分布形状**：QQ 图显示近似正态，说明 CLT 假设成立

### 数据局限

- **Bootstrap 假设样本代表性**：如果样本存在系统性偏差，置信区间无法修正
- **未控制混杂变量**：收入、年龄、城市级别可能混杂等级-消费关系
- **横截面数据**：无法确定因果方向（高消费是因还是果）
- **样本量不平衡**：钻石用户样本量较小（n=50），CI 相对较宽

### 下一步

Week 06 将对上述差异进行正式统计检验：
- 假设清单 H1-H3 的 t 检验/ANOVA
- 效应量计算（Cohen's d, η²）
- 前提假设检查（正态性、方差齐性）

---

"""


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数：运行报告摘要生成示例"""
    print("=" * 70)
    print("StatLab 不确定性量化章节生成")
    print("=" * 70)

    print("\n【选项 1】生成模板（填写 TODO 项）")
    print("-" * 70)
    template = generate_uncertainty_summary_template()
    print(template)

    print("\n" + "=" * 70)
    print("\n【选项 2】查看完整示例（使用模拟数据）")
    print("-" * 70)
    example = generate_example_summary()
    print(example)

    print("\n" + "=" * 70)
    print("使用说明")
    print("=" * 70)
    print("1. 运行你的 Bootstrap 分析脚本（如 05_statlab_uncertainty.py）")
    print("2. 将结果填入模板的 TODO 位置")
    print("3. 复制完整内容追加到 report.md")
    print("\n或直接调用：")
    print("  from chapters.week_05.examples.05_statlab_summary import append_to_report")
    print("  append_to_report(your_content, 'your_report.md')")
    print("=" * 70)


if __name__ == "__main__":
    main()
