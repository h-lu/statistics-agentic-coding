"""
示例：假设清单生成 - 将 EDA 观察转化为可检验假设。

本例演示：
1. 创建一个假设清单类
2. 基于前几节的观察添加假设
3. 按优先级组织假设
4. 生成 Markdown 格式的假设清单

运行方式：python3 chapters/week_04/examples/05_hypothesis_list.py
预期输出：
- stdout 输出假设清单表格
- examples/output/hypothesis_list.md：Markdown 格式的假设清单
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import pandas as pd


class Priority(str, Enum):
    """优先级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Hypothesis:
    """假设数据类"""
    observation: str  # 观察到什么
    explanation: str  # 解释为什么
    test_method: str  # 如何检验
    priority: Priority  # 优先级

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "observation": self.observation,
            "explanation": self.explanation,
            "test_method": self.test_method,
            "priority": self.priority.value
        }


class HypothesisList:
    """假设清单：记录可检验的假设"""

    def __init__(self):
        self.hypotheses: list[Hypothesis] = []

    def add(self, observation: str, explanation: str, test_method: str,
            priority: Priority = Priority.MEDIUM) -> None:
        """
        添加一个假设

        参数：
            observation: 观察到什么（基于 EDA）
            explanation: 解释为什么（业务逻辑）
            test_method: 如何验证（统计方法）
            priority: 优先级（high/medium/low）
        """
        hyp = Hypothesis(
            observation=observation,
            explanation=explanation,
            test_method=test_method,
            priority=priority
        )
        self.hypotheses.append(hyp)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        data = [h.to_dict() for h in self.hypotheses]
        return pd.DataFrame(data)

    def to_markdown(self) -> str:
        """生成 Markdown 格式的假设清单"""
        if not self.hypotheses:
            return "## 可检验假设清单\n\n本数据集暂无待检验假设。\n"

        df = self.to_dataframe()

        md = []
        md.append("## 可检验假设清单\n\n")
        md.append("以下假设基于 EDA 观察，将在后续章节用统计方法验证。\n\n")

        # 按优先级分组
        for priority in ["high", "medium", "low"]:
            priority_hyps = df[df["priority"] == priority]
            if priority_hyps.empty:
                continue

            priority_label = {
                "high": "高优先级",
                "medium": "中优先级",
                "low": "低优先级"
            }[priority]

            md.append(f"### {priority_label}\n\n")

            for idx, row in priority_hyps.iterrows():
                md.append(f"#### 假设 {idx + 1}\n\n")
                md.append(f"**观察**：{row['observation']}\n\n")
                md.append(f"**解释**：{row['explanation']}\n\n")
                md.append(f"**检验方法**：{row['test_method']}\n\n")

        return "".join(md)

    def save(self, filepath: Path) -> None:
        """保存到文件"""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.to_markdown(), encoding="utf-8")


def create_sample_hypotheses() -> HypothesisList:
    """创建示例假设清单"""
    hypotheses = HypothesisList()

    # 示例 1：渠道差异（高优先级）
    hypotheses.add(
        observation="搜索渠道的平均购买金额（$120）显著高于社交渠道（$80）",
        explanation="搜索渠道的用户有明确的购买意图，而社交渠道用户更多是浏览",
        test_method="双样本 t 检验（Week 06）",
        priority=Priority.HIGH
    )

    # 示例 2：年龄与购买（中优先级）
    hypotheses.add(
        observation="年龄与购买金额呈正相关（Pearson r = 0.65）",
        explanation="年龄大的用户购买力更强，可能因为收入更高",
        test_method="回归分析（Week 09）",
        priority=Priority.MEDIUM
    )

    # 示例 3：周末效应（高优先级）
    hypotheses.add(
        observation="周末的平均购买金额比工作日高 20%",
        explanation="周末用户有更多时间浏览和购买",
        test_method="按星期分组的方差分析（Week 07）",
        priority=Priority.HIGH
    )

    # 示例 4：停留时长（低优先级）
    hypotheses.add(
        observation="停留时长与购买金额的相关性较弱（Pearson r = 0.3）",
        explanation="停留时长可能包含浏览行为（如比较价格），不直接转化为购买",
        test_method="相关系数的置信区间（Week 08）",
        priority=Priority.LOW
    )

    # 示例 5：季节性（中优先级）
    hypotheses.add(
        observation="夏季月份（6-8月）销售额比冬季高 15%",
        explanation="夏季是消费旺季，可能因为假期多、用户活动频繁",
        test_method="时间序列分解与季节性检验（Week 04 后续深入学习）",
        priority=Priority.MEDIUM
    )

    return hypotheses


def print_hypothesis_summary(hypotheses: HypothesisList) -> None:
    """打印假设清单摘要"""
    df = hypotheses.to_dataframe()

    print("=" * 70)
    print("可检验假设清单")
    print("=" * 70)
    print()

    # 统计
    n_by_priority = df["priority"].value_counts().to_dict()
    print(f"总计：{len(df)} 个假设")
    print(f"  - 高优先级：{n_by_priority.get('high', 0)} 个")
    print(f"  - 中优先级：{n_by_priority.get('medium', 0)} 个")
    print(f"  - 低优先级：{n_by_priority.get('low', 0)} 个")
    print()

    # 表格形式
    print("假设列表（表格形式）")
    print("-" * 70)
    print(df.to_string(index=False))
    print()


def main() -> None:
    """主函数"""
    # 设置输出路径
    output_dir = Path(__file__).parent / "output"
    output_path = output_dir / "hypothesis_list.md"

    # 创建示例假设清单
    hypotheses = create_sample_hypotheses()

    # 打印摘要
    print_hypothesis_summary(hypotheses)

    # 保存为 Markdown
    hypotheses.save(output_path)
    print(f"假设清单已保存到 {output_path}")

    # 打印 Markdown 内容
    print("\n" + "=" * 70)
    print("Markdown 预览")
    print("=" * 70)
    print(hypotheses.to_markdown())

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("假设生成要点：")
    print("  1. 基于观察：每个假设都来自 EDA 的实际发现")
    print("  2. 可检验：能用数据验证对错")
    print("  3. 具体：明确'谁比谁高'、'趋势是什么方向'")
    print("  4. 优先级：按业务影响和数据可检验性排序")
    print("\n老潘的经验：")
    print("  '图表不是结论。假设才是结论。'")
    print("  EDA 是'提出问题'，统计检验是'回答问题'")
    print("\nWeek 06-08 我们将学习：")
    print("  - 假设检验（t 检验、ANOVA）")
    print("  - 置信区间")
    print("  - 效应量")
    print("  来验证这些假设")


if __name__ == "__main__":
    main()
