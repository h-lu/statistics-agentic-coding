"""
示例：分析报告审计清单脚本

本例演示如何用脚本自动检查报告的可复现性和完整性。
这是交付前的最后一道防线，确保报告经得起审查。

审计维度：
1. 数据与可复现性
2. 统计假设与方法
3. 诚实性与透明度
4. 叙事与结构

运行方式：python3 chapters/week_16/examples/16_audit_checklist.py

预期输出：
- 打印审计结果（通过/不通过的项目）
- 生成审计清单 Markdown 文件
- 给出改进建议
"""
from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


# ===== 审计检查项定义 =====
AUDIT_CHECKS = {
    "data_reproducibility": {
        "name": "数据与可复现性",
        "checks": [
            {
                "id": "data_source",
                "name": "数据来源明确",
                "description": "报告写清楚数据从哪来（URL、采集时间、数据集名称）",
                "check_func": "check_data_source",
                "severity": "critical"
            },
            {
                "id": "random_seed",
                "name": "随机种子固定",
                "description": "所有随机操作都固定了种子",
                "check_func": "check_random_seed",
                "severity": "critical"
            },
            {
                "id": "dependency_version",
                "name": "依赖版本记录",
                "description": "列出了所有库的版本号",
                "check_func": "check_dependency_version",
                "severity": "high"
            },
            {
                "id": "code_runnable",
                "name": "代码可运行",
                "description": "附带可运行的脚本或 README 说明",
                "check_func": "check_code_runnable",
                "severity": "high"
            }
        ]
    },
    "statistical_assumptions": {
        "name": "统计假设与方法",
        "checks": [
            {
                "id": "assumption_checked",
                "name": "检验前提验证",
                "description": "在使用 t 检验/ANOVA/回归前检查了假设（正态性、方差齐性、线性等）",
                "check_func": "check_assumption",
                "severity": "high"
            },
            {
                "id": "confidence_interval",
                "name": "不确定性量化",
                "description": "报告了置信区间或标准误，不只是点估计",
                "check_func": "check_confidence_interval",
                "severity": "high"
            },
            {
                "id": "multiple_comparison",
                "name": "多重比较校正",
                "description": "一次性检验多个指标时说明是否做了校正",
                "check_func": "check_multiple_comparison",
                "severity": "medium"
            },
            {
                "id": "model_diagnostics",
                "name": "模型诊断",
                "description": "回归模型包含残差诊断、异常点影响分析",
                "check_func": "check_model_diagnostics",
                "severity": "high"
            }
        ]
    },
    "honesty_transparency": {
        "name": "诚实性与透明度",
        "checks": [
            {
                "id": "chart_honesty",
                "name": "图表诚实性",
                "description": "Y 轴未截断（柱状图从 0 开始），标注了样本量",
                "check_func": "check_chart_honesty",
                "severity": "high"
            },
            {
                "id": "missing_disclosed",
                "name": "缺失处理说明",
                "description": "写清楚了缺失值机制和处理策略",
                "check_func": "check_missing_disclosed",
                "severity": "high"
            },
            {
                "id": "causation_boundary",
                "name": "因果声明边界",
                "description": "区分'相关'与'因果'，不说'证明'了因果关系",
                "check_func": "check_causation_boundary",
                "severity": "critical"
            },
            {
                "id": "limitations_stated",
                "name": "模型限制说明",
                "description": "明确模型适用范围和失效场景",
                "check_func": "check_limitations",
                "severity": "medium"
            }
        ]
    },
    "narrative_structure": {
        "name": "叙事与结构",
        "checks": [
            {
                "id": "question_clear",
                "name": "研究问题清晰",
                "description": "报告开头明确要回答的问题",
                "check_func": "check_question_clear",
                "severity": "high"
            },
            {
                "id": "method_traceable",
                "name": "方法可追溯",
                "description": "每个结论对应的分析方法写清楚",
                "check_func": "check_method_traceable",
                "severity": "medium"
            },
            {
                "id": "results_discussion_separated",
                "name": "结果与讨论分离",
                "description": "结果是'发现了什么'，讨论是'意味着什么'",
                "check_func": "check_results_discussion_separated",
                "severity": "low"
            },
            {
                "id": "conclusion_no_exaggeration",
                "name": "结论不夸大",
                "description": "不说'证明了'，说'支持了/暗示了'",
                "check_func": "check_conclusion_no_exaggeration",
                "severity": "high"
            }
        ]
    }
}


# ===== 检查函数实现 =====

class ReportAuditor:
    """报告审计器"""

    def __init__(self, report_path: str, code_dir: str = None):
        """
        初始化审计器

        参数:
            report_path: 报告文件路径（.md 或 .html）
            code_dir: 代码目录路径（可选）
        """
        self.report_path = Path(report_path)
        self.code_dir = Path(code_dir) if code_dir else None
        self.report_content = ""
        self.audit_results = {}

        # 读取报告内容
        if self.report_path.exists():
            with open(self.report_path, 'r', encoding='utf-8') as f:
                self.report_content = f.read()
        else:
            raise FileNotFoundError(f"报告文件不存在: {report_path}")

    def check_data_source(self) -> Tuple[bool, str]:
        """检查数据来源是否明确"""
        keywords = ['数据来源', '数据集', 'dataset', 'source', 'Kaggle', 'UCI']
        found = any(kw in self.report_content for kw in keywords)
        return found, "找到数据来源描述" if found else "未找到数据来源描述"

    def check_random_seed(self) -> Tuple[bool, str]:
        """检查是否固定了随机种子"""
        keywords = ['random_seed', '随机种子', 'seed', 'np.random.seed']
        found = any(kw in self.report_content for kw in keywords)
        return found, "找到随机种子设置" if found else "未找到随机种子设置"

    def check_dependency_version(self) -> Tuple[bool, str]:
        """检查是否记录了依赖版本"""
        keywords = ['依赖版本', 'requirements', 'version', '版本']
        found = any(kw in self.report_content for kw in keywords)
        return found, "找到依赖版本记录" if found else "未找到依赖版本记录"

    def check_code_runnable(self) -> Tuple[bool, str]:
        """检查是否有可运行的代码"""
        if self.code_dir is None:
            return False, "未指定代码目录，无法检查"

        # 检查是否有 Python 脚本或 README
        py_files = list(self.code_dir.glob("*.py"))
        readme = self.code_dir / "README.md"

        has_script = len(py_files) > 0
        has_readme = readme.exists()

        if has_script or has_readme:
            return True, f"找到 {len(py_files)} 个 Python 脚本" if has_script else "找到 README"
        return False, "未找到可运行的代码说明"

    def check_assumption(self) -> Tuple[bool, str]:
        """检查是否验证了统计假设"""
        keywords = ['假设', '正态性', '方差齐性', '残差', 'assumption',
                    'Shapiro', 'Levene', '残差图']
        found = any(kw in self.report_content for kw in keywords)
        return found, "找到假设检验相关内容" if found else "未找到假设检验"

    def check_confidence_interval(self) -> Tuple[bool, str]:
        """检查是否报告了置信区间"""
        keywords = ['置信区间', '95% CI', 'confidence interval', 'CI:', '标准误']
        found = any(kw in self.report_content for kw in keywords)
        return found, "找到置信区间报告" if found else "未找到置信区间"

    def check_multiple_comparison(self) -> Tuple[bool, str]:
        """检查是否处理了多重比较"""
        keywords = ['多重比较', 'Bonferroni', 'FDR', '校正', 'correction']
        found = any(kw in self.report_content for kw in keywords)
        # 如果没有多个检验，这个检查通过
        test_count = self.report_content.count('p 值') + self.report_content.count('p-value')
        if test_count <= 1:
            return True, "只有一个检验，无需多重比较校正"
        return found, "找到多重比较处理" if found else "有多个检验但未说明是否校正"

    def check_model_diagnostics(self) -> Tuple[bool, str]:
        """检查是否有模型诊断"""
        keywords = ['残差', '诊断', 'diagnostic', 'residual', 'QQ', '残差图']
        found = any(kw in self.report_content for kw in keywords)
        return found, "找到模型诊断内容" if found else "未找到模型诊断"

    def check_chart_honesty(self) -> Tuple[bool, str]:
        """检查图表诚实性（简化版：只检查是否提到了样本量）"""
        keywords = ['样本量', 'n=', 'N=', 'sample size']
        found = any(kw in self.report_content for kw in keywords)
        return found, "找到样本量标注" if found else "未找到样本量标注"

    def check_missing_disclosed(self) -> Tuple[bool, str]:
        """检查是否说明了缺失值处理"""
        keywords = ['缺失', 'missing', '插补', '删除', 'impute']
        found = any(kw in self.report_content for kw in keywords)
        return found, "找到缺失值处理说明" if found else "未找到缺失值说明"

    def check_causation_boundary(self) -> Tuple[bool, str]:
        """检查是否区分了相关和因果"""
        # 检查是否避免了"证明"因果的说法
        bad_patterns = [
            r'证明.*导致', r'证实.*因果', r'proves.*caus',
            r'必然导致', r'肯定.*因果'
        ]
        has_bad_claim = any(re.search(p, self.report_content) for p in bad_patterns)

        # 检查是否有恰当的边界声明
        good_patterns = [
            r'相关.*因果', r'不.*因果', r'不能.*因果',
            r'correlation.*causation', r'不等于'
        ]
        has_boundary = any(re.search(p, self.report_content) for p in good_patterns)

        if has_bad_claim:
            return False, "报告中有绝对的因果声明，建议修改"
        if has_boundary:
            return True, "找到了相关/因果边界声明"
        return False, "未明确区分相关和因果"

    def check_limitations(self) -> Tuple[bool, str]:
        """检查是否说明了研究局限"""
        keywords = ['局限', '限制', 'limitation', '边界', '假设条件']
        found = any(kw in self.report_content for kw in keywords)
        return found, "找到研究局限说明" if found else "未找到局限说明"

    def check_question_clear(self) -> Tuple[bool, str]:
        """检查研究问题是否清晰"""
        # 检查是否有问题相关的章节标题或内容
        has_question_section = re.search(r'##.*问题|##.*目标', self.report_content)
        return (has_question_section is not None,
                "找到问题/目标章节" if has_question_section else "未找到明确的问题陈述")

    def check_method_traceable(self) -> Tuple[bool, str]:
        """检查方法是否可追溯"""
        keywords = ['方法', 'method', 't 检验', 'ANOVA', '回归', '逻辑回归']
        found = any(kw in self.report_content for kw in keywords)
        return found, "找到方法描述" if found else "未找到方法描述"

    def check_results_discussion_separated(self) -> Tuple[bool, str]:
        """检查是否分离了结果和讨论"""
        has_results = re.search(r'##.*结果', self.report_content)
        has_discussion = re.search(r'##.*讨论|##.*结论', self.report_content)
        separated = has_results and has_discussion
        return separated, "结果和讨论分离" if separated else "结果和讨论未明确分离"

    def check_conclusion_no_exaggeration(self) -> Tuple[bool, str]:
        """检查结论是否夸大"""
        # 检查是否避免使用"证明"等绝对词汇
        bad_words = ['证明了', '证实了', '确凿']
        has_bad = any(bw in self.report_content for bw in bad_words)
        return (not has_bad,
                "结论用词谨慎" if not has_bad else "结论中有夸大用词（'证明'等）")

    def run_all_checks(self) -> Dict:
        """
        运行所有审计检查

        返回:
            审计结果字典
        """
        print("\n开始审计...")
        print("=" * 60)

        results = {
            "timestamp": datetime.now().isoformat(),
            "report_path": str(self.report_path),
            "categories": {}
        }

        # 获取所有检查方法
        check_methods = {
            name: getattr(self, name)
            for name in dir(self)
            if name.startswith('check_') and callable(getattr(self, name))
        }

        # 遍历每个类别
        for cat_id, category in AUDIT_CHECKS.items():
            cat_results = {
                "name": category["name"],
                "checks": []
            }

            print(f"\n类别: {category['name']}")
            print("-" * 40)

            for check in category["checks"]:
                check_id = check["id"]
                check_func_name = check["check_func"]

                if check_func_name in check_methods:
                    passed, message = check_methods[check_func_name]()
                else:
                    passed, message = False, f"检查函数未实现: {check_func_name}"

                check_result = {
                    "id": check_id,
                    "name": check["name"],
                    "description": check["description"],
                    "severity": check["severity"],
                    "passed": passed,
                    "message": message
                }

                cat_results["checks"].append(check_result)

                # 打印结果
                status = "✓" if passed else "✗"
                print(f"  {status} {check['name']}: {message}")

            results["categories"][cat_id] = cat_results

        return results

    def generate_markdown_report(self, audit_results: Dict,
                                  output_path: str = 'output/audit_checklist.md') -> str:
        """
        生成 Markdown 格式的审计清单

        参数:
            audit_results: 审计结果字典
            output_path: 输出文件路径

        返回:
            Markdown 字符串
        """
        lines = []

        lines.append("# 分析报告审计清单\n")
        lines.append(f"> **审计时间**：{audit_results['timestamp']}\n")
        lines.append(f"> **报告路径**：{audit_results['report_path']}\n")
        lines.append("---\n")

        # 统计
        total_checks = 0
        passed_checks = 0
        critical_failed = 0

        for cat_data in audit_results["categories"].values():
            for check in cat_data["checks"]:
                total_checks += 1
                if check["passed"]:
                    passed_checks += 1
                elif check["severity"] == "critical":
                    critical_failed += 1

        lines.append("## 审计摘要\n\n")
        lines.append(f"- **总检查项**：{total_checks}\n")
        lines.append(f"- **通过项**：{passed_checks}\n")
        lines.append(f"- **不通过项**：{total_checks - passed_checks}\n")
        lines.append(f"- **关键失败**：{critical_failed}\n")

        if critical_failed > 0:
            lines.append(f"\n⚠️ **警告**：有 {critical_failed} 个关键检查项未通过，建议修复后再交付。\n")
        elif passed_checks == total_checks:
            lines.append("\n✓ **所有检查项通过**，报告可以交付。\n")
        else:
            lines.append("\n**注意**：有非关键检查项未通过，建议审查后决定是否交付。\n")

        lines.append("\n---\n")

        # 详细清单
        for cat_id, cat_data in audit_results["categories"].items():
            lines.append(f"## {cat_data['name']}\n\n")

            for check in cat_data["checks"]:
                status_icon = "✅" if check["passed"] else "❌"
                severity_badge = f"`{check['severity'].upper()}`" if not check["passed"] else ""

                lines.append(f"### {status_icon} {check['name']} {severity_badge}\n\n")
                lines.append(f"{check['description']}\n\n")
                lines.append(f"**审计结果**：{check['message']}\n\n")

                if not check["passed"] and check["severity"] in ["high", "critical"]:
                    lines.append(f"💡 **建议**：优先处理此项。\n\n")

            lines.append("\n")

        # 添加改进建议
        lines.append("---\n")
        lines.append("## 改进建议\n\n")

        has_suggestions = False
        for cat_data in audit_results["categories"].values():
            for check in cat_data["checks"]:
                if not check["passed"] and check["severity"] in ["high", "critical"]:
                    has_suggestions = True
                    lines.append(f"- **{check['name']}**：{check['description']}\n")

        if not has_suggestions:
            lines.append("所有关键检查项已通过，无需改进建议。\n")

        markdown = "".join(lines)

        # 写入文件
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)

        print(f"\n审计清单已保存到: {output_path}")

        return markdown


# ===== 演示审计流程 =====
def demo_audit():
    """演示审计流程"""
    print("=" * 60)
    print("分析报告审计清单")
    print("=" * 60)

    # 首先创建一个示例报告
    sample_report = """# 客户流失分析报告

> **报告生成时间**：2026-02-21
> **随机种子**：42

## 可复现信息

- **数据来源**：Kaggle 电商客户数据集（2025年采集）
- **样本数量**：1000 个客户
- **依赖版本**：
  - numpy: 1.24.0
  - pandas: 2.0.0
  - scikit-learn: 1.3.0

## 研究问题

本分析旨在回答：哪些客户特征与流失行为相关？

## 描述统计

| 指标 | 均值 | 标准差 |
|------|------|--------|
| 使用时长 | 24.5 | 15.3 |
| 月消费 | 85.2 | 45.6 |

## 统计检验

我们使用 Mann-Whitney U 检验（因为数据不完全满足正态假设）：

- **使用时长差异**：p < 0.001（显著）
- **消费金额差异**：p = 0.003 [95% CI: 2.1, 4.8]（显著）

## 建模结果

逻辑回归模型的 AUC 为 0.78，95% 置信区间 [0.72, 0.84]。

**残差诊断**：残差大致正态分布，无明显异方差性。

## 结论

分析**支持**以下结论：
1. 使用时长与流失相关（但不能确定因果关系）
2. 消费行为是预测流失的重要因子

### 研究局限

1. 数据为横截面数据，无法确定因果方向
2. 样本来自单一平台，外推性有限
3. 未考虑季节性因素

---

*本报告由可复现分析流水线自动生成*
"""

    # 写入示例报告
    report_path = Path('output/sample_report_for_audit.md')
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(sample_report)
    print(f"示例报告已创建: {report_path}")

    # 创建审计器并运行检查
    auditor = ReportAuditor(str(report_path))
    results = auditor.run_all_checks()

    # 生成审计清单
    auditor.generate_markdown_report(results, 'output/audit_checklist.md')

    # 打印摘要
    print("\n" + "=" * 60)
    print("审计摘要")
    print("=" * 60)

    total = sum(len(c["checks"]) for c in results["categories"].values())
    passed = sum(
        sum(1 for check in cat["checks"] if check["passed"])
        for cat in results["categories"].values()
    )

    print(f"\n总检查项: {total}")
    print(f"通过: {passed}")
    print(f"不通过: {total - passed}")
    print(f"通过率: {passed/total*100:.1f}%")

    if passed == total:
        print("\n✓ 所有检查通过，报告可以交付！")
    else:
        print(f"\n⚠ 有 {total - passed} 个检查项未通过，请查看审计清单详情。")


# ===== 主函数 =====
def main() -> None:
    """运行审计演示"""
    demo_audit()

    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    print("""
在实际项目中使用审计器：

1. 对你的 report.md 运行审计：
   ```python
   auditor = ReportAuditor('path/to/report.md', code_dir='path/to/scripts')
   results = auditor.run_all_checks()
   auditor.generate_markdown_report(results, 'audit_checklist.md')
   ```

2. 同行评审时交换审计清单，逐项检查

3. 修复关键问题后重新审计

4. 将审计清单随报告一起交付

阿码问：'能不能用 AI 来做审计？'

老潘说：'AI 可以帮你检查格式和关键词，但它不知道
你的数据来源是否真实、假设是否合理。审计清单是
工具，判断还得靠人。'
    """)


if __name__ == "__main__":
    main()
