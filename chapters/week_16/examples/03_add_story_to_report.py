"""
示例：用"问题-证据-不确定性-边界"结构重写报告

本例演示如何在报告开头添加"数据故事概要"章节，
将散落的技术分析串联成完整的叙事。

运行方式：python3 chapters/week_16/examples/03_add_story_to_report.py
预期输出：在 report/report.md 开头插入"数据故事概要"章节
"""
from __future__ import annotations

import re
from pathlib import Path


def generate_story_summary() -> str:
    """
    生成数据故事概要（Markdown 格式）

    返回：
        str: 数据故事概要的 Markdown 内容
    """
    story_summary = """

## 数据故事概要

### 研究问题

优惠券能提升消费金额吗？

公司计划全面上线优惠券策略，需要评估其对用户消费金额的实际效果。核心问题是：**投入优惠券成本后，消费金额的提升是否值得？**

### 证据

**关键发现**：

- 优惠券组平均消费：108 元（n=1000）
- 对照组平均消费：100 元（n=1000）
- 效应量：8 元 [95% CI: 3.2, 12.8]
- Cohen's d：0.48（中等效应）
- p 值：0.014（统计显著）

**解释**：优惠券使平均消费提升 8 元（相对提升 8%），95% 置信区间不包含 0，说明在统计上显著。效应量 Cohen's d = 0.48，属于中等效应，具有实际意义。

### 不确定性

- 95% 信心认为优惠券有效
- 置信区间宽度：9.6 元（不确定性中等）
- 结论基于 2025 年 1-3 月数据（时间范围有限）
- 样本量：1000 人/组，功效 80%（可检测 d = 0.35）

**注意**：p 值不是"结论成立的概率"，而是"在 H0 为真时观测到当前或更极端数据的概率"。真正的结论强度来自效应量和置信区间。

### 结论边界

**适用场景**：
- 结论适用于 2025 年 1-3 月的一线城市用户
- 用户随机分配（无选择偏差）
- 消费行为相对稳定的时期（非购物季）

**失效条件**：
- 购物季（如 618、双 11）用户行为变化，结论可能失效
- 如果优惠券不是随机分配（有选择偏差），结论不可靠
- 推广到二线城市或不同用户群需谨慎

**未来工作**：
- 收集二线城市数据，测试结论普适性
- 分析不同用户群（高价值 vs 低价值）的效应差异
- 在购物季重复实验，验证结论稳定性

---

"""
    return story_summary


def add_story_to_report(report_path: str = "report/report.md",
                       output_path: str = "report/report_with_story.md"):
    """
    在报告开头添加"数据故事概要"

    参数：
        report_path: 原始报告路径
        output_path: 输出报告路径

    返回：
        bool: 是否成功添加
    """
    report_file = Path(report_path)

    # 检查文件是否存在
    if not report_file.exists():
        print(f"⚠️  报告文件不存在: {report_path}")
        return False

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report = f.read()
    except Exception as e:
        print(f"⚠️  读取报告失败: {e}")
        return False

    # 检查是否已经有"数据故事概要"
    if "## 数据故事概要" in report:
        print("⚠️  报告已经包含'数据故事概要'，跳过")
        return False

    # 查找插入位置（在第一个二级标题之前）
    first_heading_match = re.search(r"\n## ", report)
    if not first_heading_match:
        print("⚠️  未找到二级标题，无法插入")
        return False

    insert_pos = first_heading_match.start()

    # 生成数据故事概要
    story_summary = generate_story_summary()

    # 插入数据故事概要
    updated_report = report[:insert_pos] + story_summary + report[insert_pos:]

    # 保存更新后的报告
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(updated_report)

    print(f"✅ 数据故事概要已添加到: {output_path}")
    return True


def print_story_structure():
    """打印数据故事的结构说明"""
    print("\n" + "=" * 50)
    print("数据故事的四要素结构：")
    print("=" * 50)
    print("""
1. 研究问题（Question）
   - 不是"我做了什么分析"
   - 而是"我想回答什么问题"

2. 证据（Evidence）
   - 不是"所有图表和 p 值"
   - 而是"支持结论的关键证据"
   - 先说效应量，再说 p 值，最后说置信区间

3. 不确定性（Uncertainty）
   - 不是"结论成立"
   - 而是"结论有多可靠"
   - 包含置信区间、p 值解释、样本量

4. 结论边界（Limitations）
   - 不是"结论完美"
   - 而是"结论在哪里不适用"
   - 包含适用场景、失效条件、未来工作
    """)


def main():
    """执行完整的流程"""
    print("\n" + "=" * 50)
    print("数据故事概要生成工具")
    print("=" * 50 + "\n")

    # 1. 打印结构说明
    print_story_structure()

    # 2. 添加数据故事到报告
    print("\n" + "=" * 50)
    print("添加数据故事概要到报告...")
    print("=" * 50)

    success = add_story_to_report(
        report_path="report/report.md",
        output_path="report/report_with_story.md"
    )

    if success:
        print("\n" + "=" * 50)
        print("✅ 数据故事概要生成完成")
        print("=" * 50)
        print("\n建议：")
        print("  1. 检查 report/report_with_story.md")
        print("  2. 根据实际分析结果修改数据故事内容")
        print("  3. 将 report_with_story.md 重命名为 report.md")
    else:
        print("\n" + "=" * 50)
        print("⚠️  处理未完成，请检查报告文件")
        print("=" * 50)


if __name__ == "__main__":
    main()
