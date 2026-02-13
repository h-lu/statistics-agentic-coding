"""
示例：期末展示模板

本例演示如何从 report.md 自动生成展示材料（Markdown 幻灯片）。
结构来自报告，而不是从头编。

运行方式：python3 chapters/week_16/examples/02_presentation_template.py
预期输出：生成 presentation/slides.md，可用于转换为 PPT/PDF
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def extract_story_from_report(report_path: str = "report/report.md"):
    """
    从报告中提取数据故事的核心要素

    返回：
        dict: 包含 question, evidence, uncertainty, limitations, action
    """
    # 检查文件是否存在
    report_file = Path(report_path)
    if not report_file.exists():
        print(f"⚠️  报告文件不存在: {report_path}")
        print("使用默认示例数据...")
        return get_default_story()

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            report = f.read()
    except Exception as e:
        print(f"⚠️  读取报告失败: {e}")
        print("使用默认示例数据...")
        return get_default_story()

    # 提取研究问题（从"研究问题"章节）
    question_match = re.search(r"## 研究问题\n\n(.*?)\n##", report, re.DOTALL)
    question = question_match.group(1).strip() if question_match else "优惠券能提升消费金额吗？"

    # 提取关键证据（从"推断检验"章节找 p 值和效应量）
    evidence_match = re.search(
        r"效应量.*?([\d.]+).*?95% CI: \[([\d.]+), ([\d.]+)\]",
        report
    )
    if evidence_match:
        effect_size = evidence_match.group(1)
        ci_low = evidence_match.group(2)
        ci_high = evidence_match.group(3)
        evidence = f"效应量 {effect_size} 元，95% CI: [{ci_low}, {ci_high}]"
    else:
        evidence = "优惠券组平均消费 108 元，对照组 100 元"

    # 提取不确定性（从"不确定性量化"章节）
    uncertainty_match = re.search(r"## 不确定性量化\n\n(.*?)\n##", report, re.DOTALL)
    uncertainty = uncertainty_match.group(1).strip()[:200] if uncertainty_match else "95% 信心认为优惠券有效"

    # 提取结论边界（从"结论边界"章节）
    limitations_match = re.search(r"## 结论边界\n\n(.*?)\n##", report, re.DOTALL)
    limitations = limitations_match.group(1).strip()[:300] if limitations_match else "结论仅适用于 1-3 月一线城市数据"

    # 提取行动建议（从"结论"章节）
    action_match = re.search(r"## 行动建议\n\n(.*?)\n##", report, re.DOTALL)
    action = action_match.group(1).strip()[:200] if action_match else "建议在一线城市先试点，收集更多数据后再推广"

    return {
        "question": question,
        "evidence": evidence,
        "uncertainty": uncertainty,
        "limitations": limitations,
        "action": action
    }


def get_default_story():
    """返回默认的示例数据故事（当报告不存在时）"""
    return {
        "question": "优惠券能提升消费金额吗？",
        "evidence": "效应量 8 元，95% CI: [3.2, 12.8]",
        "uncertainty": "95% 信心认为优惠券有效",
        "limitations": "结论仅适用于 1-3 月一线城市数据",
        "action": "建议在一线城市先试点，收集更多数据后再推广"
    }


def generate_presentation_slides(story: dict) -> str:
    """
    生成展示幻灯片（Markdown 格式）

    参数：
        story: 数据故事字典（从 extract_story_from_report 返回）

    返回：
        str: Markdown 格式的幻灯片内容
    """
    slides = f"""# 期末展示：优惠券效果分析

**日期**：{datetime.now().strftime("%Y-%m-%d")}
**演讲者**：小北

---

## 第 1 页：研究问题（1 分钟）

**优惠券能提升消费金额吗？**

- 背景：公司计划全面上线优惠券，需要评估效果
- 数据：2025 年 1-3 月一线城市用户消费记录
- 方法：A/B 测试 + 效应量估计

**关键问题**：投入优惠券成本后，消费金额的提升是否值得？

---

## 第 2 页：关键发现（5 分钟）

**优惠券组平均消费 108 元，对照组 100 元**

{story['evidence']}

**可视化**：（放优惠券 vs 对照组的箱线图）

**解释**：
- 效应量 8 元，相对提升 8%
- 95% 置信区间：[3.2, 12.8]，不包含 0 → 统计显著
- Cohen's d = 0.48，中等效应

**结论**：优惠券有统计显著且实际意义的提升效果

---

## 第 3 页：不确定性与边界（3 分钟）

**结论有多可靠？**

- 95% 信心认为优惠券有效（p = 0.014）
- 置信区间宽度 9.6 元，不确定性中等

**结论在哪里不适用？**

{story['limitations']}

- 数据仅来自 1-3 月，可能不适用于购物季
- 样本仅来自一线城市，推广到其他城市需谨慎
- 假设：用户随机分配（无选择偏差）

---

## 第 4 页：行动建议（2 分钟）

**所以呢？**

**短期**：
- 在一线城市全面上线优惠券（证据支持有效）
- 收集 2-3 月数据，验证结论稳定性

**长期**：
- 收集二线城市数据，测试结论普适性
- 分析不同用户群（高价值 vs 低价值）的效应差异

**风险提示**：
- 购物季（如 618、双 11）需重新评估（用户行为变化）

---

## 附录（备用 Q&A）

**技术细节**：
- 检验方法：Welch's t 检验（方差不齐）
- 前提检查：Shapiro-Wilk 检验（p > 0.05，正态假设成立）
- 样本量：1000 人/组，功效 80%（可检测 d = 0.35）

**完整报告**：
- 见 `report/report.md`（包含所有技术细节、代码、图表）

**可复现性**：
- 随机种子：42
- 代码仓库：github.com/yourusername/statlab-project

---

## 谢谢！

Q & A
"""
    return slides


def save_presentation(slides: str, output_path: str = "presentation/slides.md"):
    """保存幻灯片到文件"""
    # 创建输出目录
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 写入文件
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(slides)

    return output_file


def print_conversion_tips():
    """打印转换工具建议"""
    print("\n" + "=" * 50)
    print("转换为 PPT/PDF 的建议工具：")
    print("=" * 50)
    print("\n1. Marp（推荐）:")
    print("   - 安装: npm install -g @marp-team/marp-cli")
    print("   - 转换: marp slides.md -o slides.pdf")
    print("   - 转换: marp slides.md -o slides.pptx")
    print("   - 实时预览: marp -s slides.md")
    print("\n2. Pandoc:")
    print("   - 安装: https://pandoc.org/installing.html")
    print("   - 转换: pandoc slides.md -o slides.pptx")
    print("   - 转换: pandoc slides.md -o slides.pdf --pdf-engine=xelatex")
    print("\n3. Reveal.js:")
    print("   - 在线编辑: https://slides.com/")
    print("   - 本地运行: 配置 HTML 输出\n")


def main():
    """执行完整的展示材料生成流程"""
    print("\n" + "=" * 50)
    print("期末展示材料生成")
    print("=" * 50 + "\n")

    # 1. 提取数据故事
    print("步骤 1: 从报告中提取数据故事...")
    story = extract_story_from_report()
    print(f"  ✓ 研究问题: {story['question']}")
    print(f"  ✓ 关键证据: {story['evidence']}")
    print()

    # 2. 生成幻灯片
    print("步骤 2: 生成展示幻灯片...")
    slides = generate_presentation_slides(story)
    print(f"  ✓ 幻灯片页数: {slides.count('---')}")
    print()

    # 3. 保存文件
    print("步骤 3: 保存展示材料...")
    output_file = save_presentation(slides)
    print(f"  ✓ 已保存: {output_file}")
    print()

    # 4. 打印转换建议
    print_conversion_tips()

    print("=" * 50)
    print("✅ 展示材料生成完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
