"""
示例：最终交付脚本

本例是 StatLab 超级线的最终交付入口，整合所有模块：
- 运行完整分析流水线
- 生成 Markdown 报告
- 导出 HTML 版本
- 运行审计清单
- 生成展示材料

这是 16 周学习的收束：从零散的分析到可复现、可审计、可对外展示的交付物。

运行方式：python3 chapters/week_16/examples/16_final_delivery.py

预期输出：
- 完整的分析报告（report.md + report.html）
- 审计清单（audit_checklist.md）
- 展示材料（presentation.html + speaker_script.md）
- 所有图表和中间文件
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import json

# 添加当前目录到路径，以便导入其他示例模块
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


# ===== 交付物清单 =====
DELIVERABLES = {
    "report": {
        "name": "分析报告",
        "files": ["report.md", "report.html"],
        "required": True
    },
    "figures": {
        "name": "图表",
        "files": ["figures/*.png"],
        "required": True
    },
    "audit": {
        "name": "审计清单",
        "files": ["audit_checklist.md"],
        "required": True
    },
    "presentation": {
        "name": "展示材料",
        "files": ["presentation_reveal.html", "speaker_script.md"],
        "required": False
    },
    "code": {
        "name": "分析代码",
        "files": ["scripts/*.py"],
        "required": True
    },
    "dependencies": {
        "name": "依赖说明",
        "files": ["requirements.txt"],
        "required": True
    }
}


# ===== 最终交付流水线 =====
class FinalDeliveryPipeline:
    """最终交付流水线"""

    def __init__(self, output_dir: str = 'output',
                  random_state: int = 42):
        """
        初始化流水线

        参数:
            output_dir: 输出目录
            random_state: 随机种子
        """
        self.output_dir = Path(output_dir)
        self.random_state = random_state
        self.analysis_results = None
        self.delivery_status = {
            "pipeline": False,
            "report": False,
            "html": False,
            "audit": False,
            "presentation": False
        }

    def print_header(self, title: str):
        """打印章节标题"""
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

    def step_1_run_analysis_pipeline(self) -> Dict[str, Any]:
        """
        步骤 1：运行分析流水线

        导入并执行 16_report_pipeline.py 的主逻辑
        """
        self.print_header("步骤 1：运行分析流水线")

        try:
            # 导入流水线模块
            from week_16.examples import report_pipeline

            # 运行流水线
            results = report_pipeline.run_analysis_pipeline(
                output_dir=str(self.output_dir / 'intermediate'),
                random_state=self.random_state
            )

            self.analysis_results = results
            self.delivery_status["pipeline"] = True

            print("\n✓ 分析流水线执行完成")

            return results

        except ImportError as e:
            print(f"\n✗ 无法导入流水线模块: {e}")
            print("使用模拟数据继续...")
            return self._get_mock_results()

        except Exception as e:
            print(f"\n✗ 流水线执行失败: {e}")
            return self._get_mock_results()

    def _get_mock_results(self) -> Dict[str, Any]:
        """获取模拟结果（用于演示）"""
        return {
            'data': {
                'n_samples': 1000,
                'n_features': 3,
                'churn_rate': 0.2
            },
            'tests': {
                'tenure': {
                    'test': 'Mann-Whitney U',
                    'statistic': 95234.5,
                    'p_value': 0.0001,
                    'significant': True
                },
                'spend': {
                    'test': 'Mann-Whitney U',
                    'statistic': 112456.0,
                    'p_value': 0.0032,
                    'significant': True
                }
            },
            'model': {
                'coefficients': {
                    'tenure': -0.0523,
                    'monthly_spend': 0.0021,
                    'support_calls': 0.2345
                },
                'auc': 0.78,
                'accuracy': 0.81
            },
            'reproducibility': {
                'execution_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'random_seed': self.random_state,
                'python_version': '3.11.0',
                'dependencies': {
                    'numpy': '1.24.0',
                    'pandas': '2.0.0',
                    'scikit-learn': '1.3.0'
                }
            }
        }

    def step_2_generate_report(self) -> bool:
        """
        步骤 2：生成 Markdown 报告
        """
        self.print_header("步骤 2：生成 Markdown 报告")

        if self.analysis_results is None:
            print("✗ 没有分析结果，无法生成报告")
            return False

        try:
            from week_16.examples import markdown_generator

            report_path = self.output_dir / 'report.md'
            markdown = markdown_generator.generate_markdown_with_fstring(
                self.analysis_results,
                output_path=str(report_path)
            )

            self.delivery_status["report"] = True
            print(f"\n✓ Markdown 报告已生成: {report_path}")
            return True

        except Exception as e:
            print(f"\n✗ 报告生成失败: {e}")
            return False

    def step_3_export_html(self) -> bool:
        """
        步骤 3：导出 HTML 版本
        """
        self.print_header("步骤 3：导出 HTML 版本")

        report_md = self.output_dir / 'report.md'
        if not report_md.exists():
            print(f"✗ Markdown 报告不存在: {report_md}")
            return False

        try:
            from week_16.examples import html_export

            html_path = self.output_dir / 'report.html'
            success = html_export.convert_with_python_markdown(
                str(report_md),
                str(html_path)
            )

            if success:
                self.delivery_status["html"] = True
                print(f"\n✓ HTML 报告已导出: {html_path}")
                return True
            else:
                print("\n✗ HTML 导出失败")
                return False

        except Exception as e:
            print(f"\n✗ HTML 导出失败: {e}")
            return False

    def step_4_run_audit(self) -> bool:
        """
        步骤 4：运行审计清单
        """
        self.print_header("步骤 4：运行审计清单")

        report_md = self.output_dir / 'report.md'
        if not report_md.exists():
            print(f"✗ Markdown 报告不存在: {report_md}")
            return False

        try:
            from week_16.examples import audit_checklist

            auditor = audit_checklist.ReportAuditor(
                str(report_md),
                code_dir=str(current_dir)
            )

            audit_results = auditor.run_all_checks()

            audit_path = self.output_dir / 'audit_checklist.md'
            auditor.generate_markdown_report(
                audit_results,
                output_path=str(audit_path)
            )

            self.delivery_status["audit"] = True
            print(f"\n✓ 审计清单已生成: {audit_path}")

            # 检查关键项是否通过
            critical_failed = sum(
                1 for cat in audit_results["categories"].values()
                for check in cat["checks"]
                if not check["passed"] and check["severity"] == "critical"
            )

            if critical_failed > 0:
                print(f"\n⚠️  警告：{critical_failed} 个关键检查项未通过")
            else:
                print("\n✓ 所有关键检查项通过")

            return True

        except Exception as e:
            print(f"\n✗ 审计失败: {e}")
            return False

    def step_5_generate_presentation(self) -> bool:
        """
        步骤 5：生成展示材料
        """
        self.print_header("步骤 5：生成展示材料")

        if self.analysis_results is None:
            print("✗ 没有分析结果，无法生成展示材料")
            return False

        try:
            from week_16.examples import presentation_generator

            generator = presentation_generator.PresentationGenerator(
                self.analysis_results
            )

            # 生成大纲
            outline = generator.generate_slides_outline()

            # 生成演讲脚本
            script_path = self.output_dir / 'speaker_script.md'
            script = generator.generate_speaker_script(outline)
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script)

            # 导出 Reveal.js
            presentation_path = self.output_dir / 'presentation_reveal.html'
            generator.export_to_revealjs(outline, str(presentation_path))

            self.delivery_status["presentation"] = True
            print(f"\n✓ 展示材料已生成")
            print(f"  - 演讲脚本: {script_path}")
            print(f"  - HTML 幻灯片: {presentation_path}")

            return True

        except Exception as e:
            print(f"\n✗ 展示材料生成失败: {e}")
            return False

    def step_6_generate_delivery_manifest(self) -> Path:
        """
        步骤 6：生成交付清单
        """
        self.print_header("步骤 6：生成交付清单")

        manifest = {
            "delivery_date": datetime.now().isoformat(),
            "random_seed": self.random_state,
            "status": self.delivery_status,
            "deliverables": {}
        }

        # 检查每个交付物的存在性
        for key, spec in DELIVERABLES.items():
            files_exist = []
            for file_pattern in spec["files"]:
                if '*' in file_pattern:
                    # 通配符模式
                    matching = list(self.output_dir.glob(file_pattern))
                    files_exist.extend([str(f.relative_to(self.output_dir))
                                       for f in matching])
                else:
                    file_path = self.output_dir / file_pattern
                    if file_path.exists():
                        files_exist.append(file_pattern)

            manifest["deliverables"][key] = {
                "name": spec["name"],
                "required": spec["required"],
                "files": files_exist,
                "complete": len(files_exist) > 0
            }

        # 保存清单
        manifest_path = self.output_dir / 'delivery_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"\n✓ 交付清单已生成: {manifest_path}")

        # 打印摘要
        print("\n交付物摘要：")
        print("-" * 40)

        all_complete = True
        for key, data in manifest["deliverables"].items():
            status = "✓" if data["complete"] else "✗"
            required = "（必需）" if data["required"] else "（可选）"
            print(f"{status} {data['name']}{required}: {len(data['files'])} 个文件")

            if data["required"] and not data["complete"]:
                all_complete = False

        if all_complete:
            print("\n✓ 所有必需的交付物已就绪")
        else:
            print("\n⚠️  部分必需交付物缺失")

        return manifest_path

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        运行完整交付流水线

        返回:
            包含所有状态和路径的字典
        """
        self.print_header("StatLab 最终交付流水线")

        print("\n老潘说：'交付不是堆文件，而是给对方一个")
        print("可以理解、可以验证、可以行动的完整故事。'\n")

        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)

        # 运行所有步骤
        self.step_1_run_analysis_pipeline()
        self.step_2_generate_report()
        self.step_3_export_html()
        self.step_4_run_audit()
        self.step_5_generate_presentation()
        manifest_path = self.step_6_generate_delivery_manifest()

        # 最终总结
        self.print_summary(manifest_path)

        return {
            "output_dir": str(self.output_dir),
            "status": self.delivery_status,
            "manifest": str(manifest_path)
        }

    def print_summary(self, manifest_path: Path):
        """打印最终总结"""
        self.print_header("交付完成")

        print(f"\n所有文件已保存到: {self.output_dir}")
        print(f"\n交付清单: {manifest_path}")

        print("\n交付物文件列表：")
        print("-" * 40)

        for file_path in sorted(self.output_dir.rglob('*')):
            if file_path.is_file():
                size = file_path.stat().st_size
                size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
                print(f"  {file_path.relative_to(self.output_dir)} ({size_str})")

        print("\n" + "=" * 60)
        print("下一步建议")
        print("=" * 60)
        print("""
1. 查看 report.md，确认分析结果正确
2. 查看 audit_checklist.md，修复关键问题
3. 在浏览器中打开 report.html 和 presentation_reveal.html
4. 准备展示：练习演讲脚本
5. 将整个 output/ 目录打包交付

老潘说：'好的交付物是这样的：给一个完全陌生的同事，
他能在 30 分钟内理解你的分析、重现你的结果、
并对你的结论有信心。'
        """)


# ===== AI 使用日志模板 =====
AI_USAGE_LOG_TEMPLATE = """# AI 使用日志

> **记录目的**：在 AI 辅助下完成分析的同时，保留人类审查的证据。
> 这是 "human-in-the-loop" 的核心：AI 加速，你负责。

## 采纳的建议

记录你在 AI 辅助下接受并应用的建议。

### 示例

- **原文**："模型证明了客服联系次数导致流失"
  - **AI 建议**：改为"模型显示客服联系次数与流失相关"
  - **采纳理由**：避免因果声明，改为相关。统计检验只能确认相关，不能证明因果。
  - **修改位置**：report.md 第 45 行

## 拒绝的建议

记录你考虑后拒绝的 AI 建议，并说明理由。

### 示例

- **AI 建议**：删除置信区间误差棒，"让图表更简洁"
  - **拒绝理由**：不确定性量化是统计推断的核心原则。删除误差棒会误导读者认为估计是精确的。
  - **决策**：保留误差棒，但调整图例说明更清晰

## 自己的修改

记录你独立完成的改进。

### 示例

- 补充了模型假设检查的章节（残差正态性、异方差性检验）
- 添加了数据代表性的限制说明（样本来自单一平台）
- 明确了商业建议的可操作性（具体的前 20% 客户识别）

## 总结

- **AI 辅助工具**：[列出使用的 AI 工具]
- **AI 主要帮助**：代码编写、文字润色、格式调整
- **人类关键贡献**：
  - 研究问题设计
  - 统计方法选择
  - 假设验证与模型诊断
  - 商业解读与建议
  - 最终审查与决策

---

*本日志证明：分析是在人类主导下完成的，AI 是辅助工具。*
"""


def generate_ai_usage_log(output_path: str = 'output/ai_usage_log.md'):
    """生成 AI 使用日志模板"""
    log_path = Path(output_path)
    log_path.parent.mkdir(exist_ok=True)

    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(AI_USAGE_LOG_TEMPLATE)

    print(f"AI 使用日志模板已创建: {log_path}")
    print("\n阿码问：'为什么要写这个？'")
    print("\n老潘说：'因为在 AI 时代，证明'这是我的工作'")
    print("比'我完成了工作'更重要。日志证明你思考过、'")
    print("质疑过、改进过——这比一份完美的 AI 生成报告'")
    print("更有价值。'")


# ===== 主函数 =====
def main() -> None:
    """运行最终交付演示"""
    pipeline = FinalDeliveryPipeline(output_dir='output/final_delivery')
    results = pipeline.run_full_pipeline()

    # 生成 AI 使用日志模板
    print("\n" + "=" * 60)
    print("AI 使用记录")
    print("=" * 60)
    generate_ai_usage_log('output/final_delivery/ai_usage_log.md')


if __name__ == "__main__":
    main()
