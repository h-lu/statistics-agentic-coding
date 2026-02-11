"""
示例：统计三问分类器（描述/推断/预测）。

运行方式：python3 chapters/week_01/examples/01_three_questions.py
预期输出：stdout 输出几个典型分析问题的分类结果。
"""
from __future__ import annotations

from enum import Enum
from dataclasses import dataclass


class QuestionType(Enum):
    """统计分析的三类目标"""
    DESCRIPTION = "description"  # 描述：说明数据本身的特点
    INFERENCE = "inference"    # 推断：从样本推断总体
    PREDICTION = "prediction"  # 预测：对未来或未见样本做判断


@dataclass
class QuestionExample:
    """分析问题示例"""
    question: str
    q_type: QuestionType
    reason: str  # 为什么属于这一类


# 常见问题示例库
EXAMPLES = [
    QuestionExample(
        question="这批用户的平均消费金额是多少？",
        q_type=QuestionType.DESCRIPTION,
        reason="只说明'这批数据'的特征，不外推到其他用户或时间。"
    ),
    QuestionExample(
        question="根据这 1000 个样本，全国用户的平均消费金额落在什么范围？",
        q_type=QuestionType.INFERENCE,
        reason="从样本推断总体，结论带有不确定性（需要置信区间）。"
    ),
    QuestionExample(
        question="这个新用户下周会不会购买？",
        q_type=QuestionType.PREDICTION,
        reason="对未见样本做判断，需要建模和泛化能力检验。"
    ),
    QuestionExample(
        question="A/B 测试中，哪个版本更好？",
        q_type=QuestionType.INFERENCE,
        reason="从实验样本推断到未来用户，是推断问题。"
    ),
    QuestionExample(
        question="下个月的销售额会是多少？",
        q_type=QuestionType.PREDICTION,
        reason="对未来的赌注，需要时序模型或预测模型。"
    ),
]


def classify_question(question: str) -> QuestionType | None:
    """
    简单的关键词分类器（演示用，实际需要更复杂的 NLP）

    Args:
        question: 分析问题文本

    Returns:
        QuestionType 或 None（如果无法分类）
    """
    question_lower = question.lower()

    # 描述性关键词
    desc_keywords = ["平均", "中位数", "分布", "占比", "有多少", "前几", "排名"]
    # 推断性关键词
    infer_keywords = ["推断", "总体", "样本", "范围", "差异是否", "显著", "a/b"]
    # 预测性关键词
    pred_keywords = ["预测", "会", "将", "未来", "下周", "下个月", "新用户", "会不会"]

    desc_score = sum(1 for kw in desc_keywords if kw in question_lower)
    infer_score = sum(1 for kw in infer_keywords if kw in question_lower)
    pred_score = sum(1 for kw in pred_keywords if kw in question_lower)

    if pred_score > 0:
        return QuestionType.PREDICTION
    elif infer_score > 0:
        return QuestionType.INFERENCE
    elif desc_score > 0:
        return QuestionType.DESCRIPTION
    return None


def print_examples() -> None:
    """打印示例分类结果"""
    print("=" * 60)
    print("统计三问分类示例")
    print("=" * 60)

    for ex in EXAMPLES:
        print(f"\n问题：{ex.question}")
        print(f"类型：{ex.q_type.value}")
        print(f"理由：{ex.reason}")
        print("-" * 60)


def interactive_demo() -> None:
    """交互式演示"""
    print("\n" + "=" * 60)
    print("输入你的分析问题（输入 q 退出）：")
    print("=" * 60)

    while True:
        question = input("\n> ").strip()
        if question.lower() in ["q", "exit", "退出"]:
            print("退出分类器。")
            break

        if not question:
            continue

        q_type = classify_question(question)
        if q_type:
            type_name = {
                QuestionType.DESCRIPTION: "描述（Description）",
                QuestionType.INFERENCE: "推断（Inference）",
                QuestionType.PREDICTION: "预测（Prediction）",
            }[q_type]
            print(f"→ 分类结果：{type_name}")

            # 给出建议
            if q_type == QuestionType.DESCRIPTION:
                print("  建议方法：均值、中位数、标准差、直方图等")
            elif q_type == QuestionType.INFERENCE:
                print("  建议方法：置信区间、假设检验、效应量等")
            else:  # PREDICTION
                print("  建议方法：回归模型、分类模型、交叉验证等")
        else:
            print("→ 无法分类：请确保问题包含明确的意图关键词")


def main() -> None:
    """主函数"""
    # 先打印示例
    print_examples()

    # 进入交互模式
    try:
        interactive_demo()
    except (KeyboardInterrupt, EOFError):
        print("\n\n退出分类器。")


if __name__ == "__main__":
    main()
