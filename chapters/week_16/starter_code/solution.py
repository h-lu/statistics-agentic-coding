"""
Week 16 作业参考实现

本文件是 Week 16 作业的参考答案，供学生在遇到困难时查看。
只实现基础作业要求，不覆盖进阶/挑战部分。

运行方式：python3 chapters/week_16/starter_code/solution.py
预期输出：生成完整的期末项目包（数据故事、可复现章节、展示材料、HTML）
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path


# ==================== 第 1 题：数据故事概要 ====================

def add_data_story_summary(report_path: str = "report/report.md",
                          output_path: str = "report/report_solution.md") -> str:
    """
    在报告开头添加"数据故事概要"章节

    要求：
        1. 包含四个要素：研究问题、证据、不确定性、结论边界
        2. 每个要素简洁明了（1-3 段话）
        3. 使用"问题-证据-不确定性-边界"的结构

    返回：
        str: 更新后的报告路径
    """
    print("\n" + "=" * 50)
    print("第 1 题：添加数据故事概要")
    print("=" * 50)

    # 数据故事概要模板
    story_summary = """

## 数据故事概要

### 研究问题

优惠券能提升消费金额吗？

### 证据

- 优惠券组平均消费：108 元（n=1000）
- 对照组平均消费：100 元（n=1000）
- 效应量：8 元 [95% CI: 3.2, 12.8]
- Cohen's d：0.48（中等效应）
- p 值：0.014（统计显著）

### 不确定性

- 95% 信心认为优惠券有效
- 置信区间宽度：9.6 元（不确定性中等）

### 结论边界

**适用场景**：
- 结论适用于 2025 年 1-3 月的一线城市用户

**失效条件**：
- 购物季（如 618、双 11）用户行为变化，结论可能失效

**未来工作**：
- 收集二线城市数据，测试结论普适性

---

"""

    # 读取原始报告
    report_file = Path(report_path)
    if not report_file.exists():
        print(f"⚠️  报告文件不存在，创建示例报告: {report_path}")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# StatLab 分析报告\n\n## 研究问题\n\n优惠券能提升消费金额吗？\n")

    with open(report_file, "r", encoding="utf-8") as f:
        report = f.read()

    # 检查是否已有数据故事概要
    if "## 数据故事概要" in report:
        print("✓ 报告已包含数据故事概要")
        return report_path

    # 在第一个二级标题前插入
    first_heading_match = re.search(r"\n## ", report)
    if first_heading_match:
        insert_pos = first_heading_match.start()
        updated_report = report[:insert_pos] + story_summary + report[insert_pos:]
    else:
        updated_report = story_summary + report

    # 保存更新后的报告
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(updated_report)

    print(f"✅ 数据故事概要已添加: {output_path}")
    return output_path


# ==================== 第 2 题：可复现分析章节 ====================

def add_reproducibility_section(report_path: str = "report/report_solution.md",
                              output_path: str = "report/report_with_reproducibility.md") -> str:
    """
    在报告末尾添加"可复现分析"章节

    要求：
        1. 包含依赖版本（requirements.txt 内容）
        2. 包含固定随机性说明
        3. 包含数据版本记录
        4. 包含运行命令

    返回：
        str: 更新后的报告路径
    """
    print("\n" + "=" * 50)
    print("第 2 题：添加可复现分析章节")
    print("=" * 50)

    reproducibility_section = """

## 可复现分析

### 依赖安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt**：
```text
numpy==1.26.4
pandas==2.2.1
scipy==1.13.0
scikit-learn==1.5.0
```

### 固定随机性

所有分析使用固定随机种子 `RANDOM_SEED = 42`。

### 数据版本

- 数据来源：`data/user_behavior_20250315.csv`
- 数据版本：2025-03-15

### 运行分析

```bash
git clone https://github.com/yourusername/statlab-project.git
cd statlab-project
python scripts/run_full_analysis.py
```

---
"""

    # 读取报告
    with open(report_path, "r", encoding="utf-8") as f:
        report = f.read()

    # 检查是否已有可复现分析章节
    if "## 可复现分析" in report:
        print("✓ 报告已包含可复现分析章节")
        return report_path

    # 在报告末尾添加
    updated_report = report + reproducibility_section

    # 保存更新后的报告
    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(updated_report)

    print(f"✅ 可复现分析章节已添加: {output_path}")
    return output_path


# ==================== 第 3 题：生成展示材料 ====================

def generate_presentation_slides(report_path: str = "report/report_with_reproducibility.md",
                                 output_path: str = "presentation/slides_solution.md") -> str:
    """
    从报告生成展示材料（10-15 分钟版本）

    要求：
        1. 四段式结构：开场、证据、不确定性与边界、行动建议
        2. 每页 1-3 分钟
        3. 技术细节放在附录

    返回：
        str: 展示材料路径
    """
    print("\n" + "=" * 50)
    print("第 3 题：生成展示材料")
    print("=" * 50)

    # 读取报告
    with open(report_path, "r", encoding="utf-8") as f:
        report = f.read()

    # 提取关键信息
    question_match = re.search(r"## 研究问题\n\n(.*?)\n##", report, re.DOTALL)
    question = question_match.group(1).strip() if question_match else "优惠券能提升消费金额吗？"

    # 生成展示幻灯片
    slides = f"""# 期末展示：优惠券效果分析

**日期**：{datetime.now().strftime("%Y-%m-%d")}

---

## 第 1 页：研究问题（1 分钟）

**优惠券能提升消费金额吗？**

- 背景：公司计划全面上线优惠券
- 数据：2025 年 1-3 月一线城市用户
- 方法：A/B 测试 + 效应量估计

---

## 第 2 页：关键发现（5 分钟）

**优惠券组平均消费 108 元，对照组 100 元**

- 效应量 8 元，相对提升 8%
- 95% 置信区间：[3.2, 12.8]
- Cohen's d = 0.48（中等效应）

---

## 第 3 页：不确定性与边界（3 分钟）

**结论有多可靠？**
- 95% 信心认为优惠券有效（p = 0.014）

**结论在哪里不适用？**
- 数据仅来自 1-3 月，可能不适用于购物季
- 样本仅来自一线城市

---

## 第 4 页：行动建议（2 分钟）

**所以呢？**

**短期**：在一线城市全面上线优惠券
**长期**：收集二线城市数据

---

## 谢谢！

Q & A
"""

    # 保存展示材料
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(slides)

    print(f"✅ 展示材料已生成: {output_file}")
    return output_path


# ==================== 第 4 题：导出 HTML ====================

def export_html_report(markdown_path: str = "report/report_with_reproducibility.md",
                      output_path: str = "report/report_solution.html") -> str:
    """
    从 Markdown 导出 HTML 展示版

    要求：
        1. 使用 pandoc 或 markdown 库
        2. 包含基本样式
        3. 代码高亮

    返回：
        str: HTML 文件路径
    """
    print("\n" + "=" * 50)
    print("第 4 题：导出 HTML 展示版")
    print("=" * 50)

    # 优先使用 pandoc
    try:
        cmd = ["pandoc", markdown_path, "-o", output_path, "--standalone"]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ HTML 已导出（使用 pandoc）: {output_path}")
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  pandoc 不可用，使用 markdown 库")

    # 备选：使用 markdown 库
    try:
        import markdown

        # 读取 Markdown
        with open(markdown_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 转换为 HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

        # 生成完整的 HTML
        full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>StatLab 终稿报告</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/github-markdown-css@5.6.1/github-markdown.min.css">
    <style>
        body {{ max-width: 900px; margin: 2rem auto; padding: 0 1rem; }}
        img {{ max-width: 100%; }}
    </style>
</head>
<body>
    <article class="markdown-body">
        {html_content}
    </article>
</body>
</html>
"""

        # 保存 HTML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_html)

        print(f"✅ HTML 已导出（使用 markdown 库）: {output_path}")
        return output_path

    except ImportError:
        print("❌ 未安装 pandoc 或 markdown 库")
        print("建议安装: pip install markdown")
        return None


# ==================== 主函数 ====================

def main():
    """执行完整的作业流程"""
    print("\n" + "=" * 60)
    print("Week 16 作业参考实现")
    print("=" * 60)
    print("\n本脚本将生成：")
    print("  1. 数据故事概要")
    print("  2. 可复现分析章节")
    print("  3. 展示材料（slides.md）")
    print("  4. HTML 展示版")

    # 第 1 题：数据故事概要
    report_path = add_data_story_summary(
        report_path="report/report.md",
        output_path="report/report_solution.md"
    )

    # 第 2 题：可复现分析章节
    report_path = add_reproducibility_section(
        report_path=report_path,
        output_path="report/report_with_reproducibility.md"
    )

    # 第 3 题：展示材料
    slides_path = generate_presentation_slides(
        report_path=report_path,
        output_path="presentation/slides_solution.md"
    )

    # 第 4 题：HTML 导出
    html_path = export_html_report(
        markdown_path=report_path,
        output_path="report/report_solution.html"
    )

    # 总结
    print("\n" + "=" * 60)
    print("✅ Week 16 作业完成")
    print("=" * 60)
    print("\n生成的文件：")
    print(f"  1. {report_path} - 包含数据故事和可复现章节的报告")
    print(f"  2. {slides_path} - 10-15 分钟展示材料")
    if html_path:
        print(f"  3. {html_path} - HTML 展示版")
    print("\n建议：")
    print("  1. 检查生成的文件是否符合要求")
    print("  2. 根据实际分析结果修改内容")
    print("  3. 在浏览器中打开 HTML 文件查看效果")


if __name__ == "__main__":
    main()
