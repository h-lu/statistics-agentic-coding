"""
示例：展示材料生成器

本例演示如何从分析报告自动生成展示材料。
遵循"问题-方法-发现-边界-反思"的叙事结构。

核心原则：
1. 展示不是报告的复制粘贴，而是故事的重新讲述
2. 每张图只传递一个核心信息
3. 为非技术受众简化术语，但保留不确定性

运行方式：python3 chapters/week_16/examples/16_presentation_generator.py

预期输出：
- 生成演讲脚本（Markdown 格式）
- 生成幻灯片大纲
- 为每张图写好"听众应该看到什么"的说明
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


# ===== 展示叙事结构模板 =====
PRESENTATION_TEMPLATE = {
    "structure": [
        {
            "section": "问题",
            "duration_minutes": 1,
            "purpose": "让听众关心这个问题",
            "slides": [
                {
                    "title": "背景与问题",
                    "content": ["业务背景", "研究问题", "为什么重要"],
                    "visual": None,
                    "speaker_notes": "从听众熟悉的业务场景开始，不是从技术细节开始"
                },
                {
                    "title": "分析目标",
                    "content": ["要回答的问题", "预期的商业价值"],
                    "visual": None,
                    "speaker_notes": "用一句话说清楚分析的目标"
                }
            ]
        },
        {
            "section": "方法",
            "duration_minutes": 2,
            "purpose": "建立信任（这是个靠谱的分析）",
            "slides": [
                {
                    "title": "数据与方法",
                    "content": ["数据来源", "样本量", "分析方法概述"],
                    "visual": "data_overview.png",
                    "speaker_notes": "简述数据和方法，不要深入技术细节。听众需要知道'数据可靠'，不需要知道公式"
                }
            ]
        },
        {
            "section": "发现",
            "duration_minutes": 4,
            "purpose": "展示核心发现（用图表）",
            "slides": [
                {
                    "title": "关键发现 1：使用时长与流失",
                    "content": ["流失客户的使用时长显著更短", "统计显著性", "置信区间"],
                    "visual": "tenure_by_churn.png",
                    "speaker_notes": "这张图要传递的信息：使用时长是流失的强预测因子。听众应该看到：两组的分布有明显差异，误差棒表示不确定性"
                },
                {
                    "title": "关键发现 2：模型预测能力",
                    "content": ["模型 AUC", "关键预测因子", "商业含义"],
                    "visual": "model_performance.png",
                    "speaker_notes": "这张图要传递的信息：模型能有效识别高风险客户。听众应该看到：AUC 是什么意思，我们能捕获多少流失客户"
                }
            ]
        },
        {
            "section": "边界",
            "duration_minutes": 1,
            "purpose": "诚实地说明限制（建立信任）",
            "slides": [
                {
                    "title": "分析的局限性",
                    "content": ["数据代表性", "因果 vs 相关", "模型不确定性"],
                    "visual": None,
                    "speaker_notes": "主动说明限制比被发现问题更专业。区分'我们发现相关'和'我们能证明因果'"
                }
            ]
        },
        {
            "section": "反思",
            "duration_minutes": 1,
            "purpose": "回到原始问题，给出行动建议",
            "slides": [
                {
                    "title": "结论与建议",
                    "content": ["可操作建议", "下一步计划"],
                    "visual": None,
                    "speaker_notes": "把分析收束回原始问题：我们能不能预测流失？能，那接下来该做什么？"
                }
            ]
        }
    ]
}


# ===== 展示脚本生成器 =====
class PresentationGenerator:
    """展示材料生成器"""

    def __init__(self, report_results: Dict[str, Any]):
        """
        初始化生成器

        参数:
            report_results: 分析流水线的结果字典
        """
        self.results = report_results
        self.slides = []
        self.speaker_notes = []

    def generate_slides_outline(self) -> List[Dict]:
        """
        生成幻灯片大纲

        基于分析结果，自动填充模板内容
        """
        outline = []

        # 1. 标题页
        outline.append({
            "title": "客户流失预测分析",
            "subtitle": f"基于 {self.results['data']['n_samples']} 个客户的数据分析",
            "date": datetime.now().strftime("%Y年%m月%d日"),
            "section": "封面"
        })

        # 2. 问题
        outline.append({
            "title": "问题：为什么客户流失？",
            "points": [
                f"当前流失率：{self.results['data']['churn_rate']:.1%}",
                "识别高风险客户，提前干预",
                "降低流失，提升客户价值"
            ],
            "visual": None,
            "speaker_notes": "各位好，今天分享的客户流失分析。当前流失率约20%，我们的目标是识别高风险客户，提前采取留存措施。",
            "section": "问题"
        })

        # 3. 方法
        outline.append({
            "title": "方法：数据与分析",
            "points": [
                f"数据：{self.results['data']['n_samples']} 个客户，{self.results['data']['n_features']} 个特征",
                "方法：统计检验 + 逻辑回归",
                "评估：AUC、置信区间",
                f"随机种子：{self.results['reproducibility']['random_seed']}（可复现）"
            ],
            "visual": "data_overview.png",
            "speaker_notes": "我们分析了1000个客户的三个关键指标。使用统计检验发现差异，用逻辑回归预测流失。所有分析固定随机种子，可复现。",
            "section": "方法"
        })

        # 4. 发现 - 统计检验
        tenure_test = self.results['tests']['tenure']
        outline.append({
            "title": "发现 1：使用时长与流失显著相关",
            "points": [
                f"检验方法：{tenure_test['test']}",
                f"p 值：{tenure_test['p_value']:.4f}",
                "结论：流失客户的使用时长明显更短"
            ],
            "visual": "tenure_distribution.png",
            "speaker_notes": "这张图显示，流失客户（红色）的使用时长明显短于留存客户（绿色）。Mann-Whitney U 检验的 p 值小于 0.001，说明这个差异不是偶然。",
            "section": "发现"
        })

        # 5. 发现 - 模型
        outline.append({
            "title": "发现 2：模型能有效预测流失",
            "points": [
                f"模型 AUC：{self.results['model']['auc']:.2f}",
                "准确率：{:.1%}".format(self.results['model']['accuracy']),
                "最强预测因子：使用时长、客服联系次数"
            ],
            "visual": "roc_curve.png",
            "speaker_notes": "逻辑回归模型的 AUC 为 0.78，意味着如果我们用模型识别前 20% 的高风险客户，能捕获 60% 的实际流失者。SHAP 值显示使用时长和客服联系是最重要的预测因子。",
            "section": "发现"
        })

        # 6. 边界
        outline.append({
            "title": "边界：分析的限制",
            "points": [
                "数据来自单一平台，外推性有限",
                "统计检验显示相关，不能证明因果",
                "模型有一定不确定性（置信区间）",
                "未考虑季节性因素"
            ],
            "visual": None,
            "speaker_notes": "需要说明三个限制：第一，数据代表性；第二，我们只能说明相关，不能证明因果——要回答因果问题需要随机对照实验；第三，模型预测有不确定性。",
            "section": "边界"
        })

        # 7. 反思/建议
        outline.append({
            "title": "建议：接下来做什么？",
            "points": [
                "用模型识别前 20% 高风险客户（覆盖 60% 流失）",
                "针对低活跃客户设计留存方案",
                "开展 A/B 测试验证干预效果",
                "定期更新模型，监测变化"
            ],
            "visual": None,
            "speaker_notes": "回到最初的问题：我们能预测流失吗？能。建议是：用模型识别高风险客户，主动联系。下一步是做 A/B 测试，把相关发现变成因果结论。",
            "section": "反思"
        })

        # 8. 结束页
        outline.append({
            "title": "谢谢",
            "points": [
                "分析可复现：固定随机种子",
                "问题？"
            ],
            "visual": None,
            "speaker_notes": "报告已开源，任何人都能复现。谢谢大家，欢迎提问。",
            "section": "结束"
        })

        self.slides = outline
        return outline

    def generate_speaker_script(self, slide_outline: List[Dict]) -> str:
        """
        生成完整的演讲脚本

        格式：Markdown，包含幻灯片内容 + 演讲者备注
        """
        lines = []

        lines.append("# 客户流失分析展示脚本\n")
        lines.append(f"> **生成时间**：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        lines.append("---\n\n")

        for i, slide in enumerate(slide_outline, 1):
            section = slide.get('section', '')
            lines.append(f"## 幻灯片 {i}：{slide['title']}\n\n")
            if section:
                lines.append(f"*章节：{section}*\n\n")

            # 幻灯片内容
            if 'subtitle' in slide:
                lines.append(f"**副标题**：{slide['subtitle']}\n\n")
            if 'date' in slide:
                lines.append(f"**日期**：{slide['date']}\n\n")

            if 'points' in slide:
                lines.append("**内容要点**：\n\n")
                for point in slide['points']:
                    lines.append(f"- {point}\n")
                lines.append("\n")

            if slide.get('visual'):
                lines.append(f"**视觉元素**：`{slide['visual']}`\n\n")

            # 演讲者备注
            if slide.get('speaker_notes'):
                lines.append("**🗣️ 演讲者备注**：\n\n")
                lines.append(f"{slide['speaker_notes']}\n\n")
                lines.append("---\n\n")

        script = "".join(lines)
        self.speaker_notes = script
        return script

    def export_to_revealjs(self, slide_outline: List[Dict],
                           output_path: str = 'output/presentation.html') -> str:
        """
        导出为 Reveal.js HTML 幻灯片

        Reveal.js 是一个 HTML 幻灯片框架，支持：
        - 键盘导航
        - 嵌入图表
        - 演讲者备注
        - 响应式设计

        参数:
            slide_outline: 幻灯片大纲
            output_path: 输出文件路径

        返回:
            HTML 字符串
        """
        html_parts = []

        html_parts.append("""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>客户流失分析展示</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4.5.0/dist/reveal.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4.5.0/dist/theme/moon.css">
    <style>
        .reveal { font-family: "Microsoft YaHei", sans-serif; }
        .reveal h1, .reveal h2, .reveal h3 { color: #3498db; }
        .reveal ul { text-align: left; }
        .reveal .speaker-notes { color: #888; font-size: 0.7em; font-style: italic; }
    </style>
</head>
<body>
    <div class="reveal">
        <div class="slides">
""")

        for slide in slide_outline:
            html_parts.append('            <section>\n')
            html_parts.append(f'                <h2>{slide["title"]}</h2>\n')

            if 'subtitle' in slide:
                html_parts.append(f'                <h3>{slide["subtitle"]}</h3>\n')

            if 'points' in slide:
                html_parts.append('                <ul>\n')
                for point in slide['points']:
                    html_parts.append(f'                    <li>{point}</li>\n')
                html_parts.append('                </ul>\n')

            if slide.get('visual'):
                html_parts.append(f'                <p><img src="{slide["visual"]}" style="max-height: 400px;"></p>\n')

            if slide.get('speaker_notes'):
                html_parts.append(f'                <p class="speaker-notes">🗣️ {slide["speaker_notes"]}</p>\n')

            html_parts.append('            </section>\n')

        html_parts.append("""        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/reveal.js@4.5.0/dist/reveal.js"></script>
    <script>
        Reveal.initialize({
            hash: true,
            transition: 'slide',
            controls: true,
            progress: true,
            center: true
        });
    </script>
</body>
</html>
""")

        html = "".join(html_parts)

        # 写入文件
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"幻灯片已导出: {output_path}")

        return html

    def export_to_marp(self, slide_outline: List[Dict],
                       output_path: str = 'output/presentation.md') -> str:
        """
        导出为 Marp Markdown 幻灯片

        Marp 是一个 Markdown 幻灯片生态系统，
        可以在 VS Code 中预览和导出 PDF/HTML

        参数:
            slide_outline: 幻灯片大纲
            output_path: 输出文件路径

        返回:
            Markdown 字符串
        """
        lines = []

        lines.append("---\n")
        lines.append("marp: true\n")
        lines.append("theme: gaia\n")
        lines.append("paginate: true\n")
        lines.append("---\n\n")

        for slide in slide_outline:
            lines.append("---\n\n")
            lines.append(f"# {slide['title']}\n\n")

            if 'subtitle' in slide:
                lines.append(f"## {slide['subtitle']}\n\n")

            if 'points' in slide:
                for point in slide['points']:
                    lines.append(f"- {point}\n")
                lines.append("\n")

            if slide.get('visual'):
                lines.append(f"![{slide['visual']}]({slide['visual']})\n\n")

            if slide.get('speaker_notes'):
                lines.append(f"<!-- \n")
                lines.append(f"演讲者备注: {slide['speaker_notes']}\n")
                lines.append(f"-->\n\n")

        markdown = "".join(lines)

        # 写入文件
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)

        print(f"Marp 幻灯片已导出: {output_path}")

        return markdown


# ===== 演示展示生成 =====
def demo_presentation_generation():
    """演示展示材料生成流程"""
    print("=" * 60)
    print("展示材料生成器")
    print("=" * 60)

    # 模拟分析结果
    mock_results = {
        'data': {
            'n_samples': 1000,
            'n_features': 3,
            'churn_rate': 0.2
        },
        'tests': {
            'tenure': {
                'test': 'Mann-Whitney U',
                'p_value': 0.0001,
                'significant': True
            },
            'spend': {
                'test': 'Mann-Whitney U',
                'p_value': 0.0032,
                'significant': True
            }
        },
        'model': {
            'auc': 0.78,
            'accuracy': 0.81,
            'top_features': ['tenure', 'support_calls']
        },
        'reproducibility': {
            'random_seed': 42
        }
    }

    # 创建生成器
    generator = PresentationGenerator(mock_results)

    # 生成幻灯片大纲
    print("\n生成幻灯片大纲...")
    outline = generator.generate_slides_outline()
    print(f"生成了 {len(outline)} 张幻灯片")

    # 打印大纲
    print("\n幻灯片大纲：")
    print("-" * 40)
    for i, slide in enumerate(outline, 1):
        section = f"[{slide.get('section', '')}] " if slide.get('section') else ""
        print(f"{i}. {section}{slide['title']}")

    # 生成演讲脚本
    print("\n生成演讲脚本...")
    script = generator.generate_speaker_script(outline)

    # 保存脚本
    script_path = Path('output/speaker_script.md')
    script_path.parent.mkdir(exist_ok=True)
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)
    print(f"演讲脚本已保存: {script_path}")

    # 导出 Reveal.js
    print("\n导出 Reveal.js 幻灯片...")
    generator.export_to_revealjs(outline, 'output/presentation_reveal.html')

    # 导出 Marp
    print("\n导出 Marp 幻灯片...")
    generator.export_to_marp(outline, 'output/presentation_marp.md')

    return outline


# ===== 展示原则检查清单 =====
PRESENTATION_CHECKLIST = {
    "structure": "是否遵循'问题-方法-发现-边界-反思'结构？",
    "one_idea_per_slide": "每张幻灯片是否只传递一个核心信息？",
    "visual_first": "是否用图表而非文字讲故事？",
    "simplify_jargon": "是否为非技术受众简化了术语？",
    "honest_uncertainty": "是否诚实地表达了不确定性？",
    "time_management": "总时长是否控制在 10 分钟内？",
    "actionable_takeaway": "听众离开时能否记住一个行动建议？"
}


def check_presentation_quality() -> None:
    """打印展示质量检查清单"""
    print("\n" + "=" * 60)
    print("展示质量检查清单")
    print("=" * 60)

    for i, (key, question) in enumerate(PRESENTATION_CHECKLIST.items(), 1):
        print(f"{i}. {question}")


# ===== 主函数 =====
def main() -> None:
    """运行展示材料生成演示"""
    demo_presentation_generation()
    check_presentation_quality()

    print("\n" + "=" * 60)
    print("使用建议")
    print("=" * 60)
    print("""
小北问："我能不能把所有 16 周的内容都放进去？"

老潘说："那不是展示，是数据倾倒。记住三点：

1. **从问题开头**：不是'我用了什么方法'，而是'为什么要做这个分析'

2. **一张图一个信息**：每张幻灯片回答一个问题，不要堆砌

3. **诚实表达不确定性**：不说'证明'了，说'支持'了结论

展示的目的不是展示你知道多少，而是让听众理解并信任你的结论。"

展示工具选择：
- **快速迭代**：Marp（Markdown 直接转幻灯片）
- **专业展示**：Reveal.js（HTML 幻灯片，可交互）
- **传统**：PowerPoint（手工制作，但最可控）

安装 Marp：
  npm install -g @marp-team/marp-cli

使用：
  marp presentation_marp.md --pdf
  marp presentation_marp.md --html
    """)


if __name__ == "__main__":
    main()
