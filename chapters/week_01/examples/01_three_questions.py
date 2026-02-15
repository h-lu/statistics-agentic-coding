"""
示例：用 Palmer Penguins 数据集演示三类统计问题（描述/推断/预测）的区别。

运行方式：python3 chapters/week_01/examples/01_three_questions.py
预期输出：stdout 输出描述性统计结果（三种企鹅的平均嘴峰长度）
"""
from __future__ import annotations

import seaborn as sns


def main() -> None:
    # 加载数据
    penguins = sns.load_dataset("penguins")

    print("=" * 60)
    print("统计三问：描述 vs 推断 vs 预测")
    print("=" * 60)
    print()

    # 1. 描述（Description）：数据长什么样？
    print("1. 描述性统计（Description）")
    print("   问题：这三种企鹅的嘴峰长度平均是多少？")
    print()
    print("   三种企鹅的平均嘴峰长度：")
    species_means = penguins.groupby("species")["bill_length_mm"].mean()
    for species, mean_length in species_means.items():
        print(f"   - {species}: {mean_length:.2f} mm")
    print()
    print("   这是一个'描述'——我们只是在报告手头样本的事实，")
    print("   不涉及对总体的推断，也不涉及预测新样本。")
    print()

    # 2. 推断（Inference）：差异是真差异吗？
    print("2. 统计推断（Inference）")
    print("   问题：Adelie 和 Chinstrap 的嘴峰长度差异是'真差异'，")
    print("         还是抽样造成的偶然？")
    print()
    print("   这类问题需要回答：我们看到的差异，是否能推广到整个")
    print("   南极企鹅种群？需要引入'不确定性'的量化。")
    print("   （Week 06-08 会学习假设检验来回答这类问题）")
    print()

    # 3. 预测（Prediction）：给定特征，猜物种
    print("3. 预测（Prediction）")
    print("   问题：给定一只新企鹅的嘴峰长度、嘴峰深度、鳍肢长度，")
    print("         能不能猜出它的物种？")
    print()
    print("   这类问题的目标是'猜得准'，不一定要理解变量之间的关系。")
    print("   （Week 09-10 会学习分类模型来回答这类问题）")
    print()

    print("=" * 60)
    print("关键区分：")
    print("- 描述：报告样本事实（无需推断）")
    print("- 推断：从样本推断总体（需要量化不确定性）")
    print("- 预测：给定 X，猜 Y（目标是准确性）")
    print("=" * 60)


if __name__ == "__main__":
    main()
