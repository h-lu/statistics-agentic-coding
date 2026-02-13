#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诚实可视化示例：误导图 vs 诚实图

本示例展示常见的可视化误导陷阱，
以及如何修正它们。

运行方式：python 04_honest_visualization.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_02"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_style():
    """设置绘图风格"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False


def demonstrate_truncated_y_axis():
    """演示截断 Y 轴的误导性"""
    print("=" * 60)
    print("误导 #1：截断 Y 轴 —— 让小差异看起来很大")
    print("=" * 60)

    # 数据：两组差异很小（2% vs 3%）
    groups = ['A组', 'B组']
    values = [2.0, 3.0]  # 单位：%

    print(f"\n真实数据：{dict(zip(groups, values))}")
    print(f"差异：{values[1] / values[0] - 1:.2%}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：误导（Y 轴从 1.5 开始）
    bars1 = axes[0].bar(groups, values, color=['coral', 'steelblue'])
    axes[0].set_ylim(1.5, 3.5)
    axes[0].set_ylabel('转化率 (%)')
    axes[0].set_title('❌ 误导：Y 轴从 1.5 开始', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)

    # 添加数据标签
    for i, (group, val) in enumerate(zip(groups, values)):
        axes[0].text(i, val + 0.1, f'{val}%', ha='center', fontsize=12)

    # 右图：诚实（Y 轴从 0 开始）
    bars2 = axes[1].bar(groups, values, color=['coral', 'steelblue'])
    axes[1].set_ylim(0, 3.5)
    axes[1].set_ylabel('转化率 (%)')
    axes[1].set_title('✅ 诚实：Y 轴从 0 开始', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)

    for i, (group, val) in enumerate(zip(groups, values)):
        axes[1].text(i, val + 0.1, f'{val}%', ha='center', fontsize=12)

    plt.tight_layout()
    print(f"\n输出：{OUTPUT_DIR / 'truncated_y_axis_comparison.png'}")
    plt.savefig(OUTPUT_DIR / 'truncated_y_axis_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n老潘的点评：")
    print("'在公司里，截断 Y 轴的图表会被打回来的。'")
    print("'你想突出差异，但同时掩盖了真实范围。'")

    return {
        'misleading_ylim': (1.5, 3.5),
        'honest_ylim': (0, 3.5),
        'data': dict(zip(groups, values))
    }


def demonstrate_area_deception():
    """演示面积误导"""
    print("\n" + "=" * 60)
    print("误导 #2：面积误导 —— 让人误以为总量不同")
    print("=" * 60)

    # 单变量时间序列数据
    years = [2018, 2019, 2020, 2021, 2022]
    sales = [100, 110, 120, 135, 150]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：误导（3D 效果/面积填充）
    axes[0].fill_between(years, 0, sales, color='steelblue', alpha=0.5)
    axes[0].plot(years, sales, 'o-', color='darkblue', markersize=8)
    axes[0].set_ylabel('销售额')
    axes[0].set_title('❌ 误导：面积填充强调视觉', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # 右图：诚实（线条）
    axes[1].plot(years, sales, 'o-', color='steelblue', markersize=8)
    axes[1].set_ylabel('销售额')
    axes[1].set_title('✅ 诚实：线条展示趋势', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # 标注最终值
    axes[0].text(years[-1], sales[-1], f' {sales[-1]}', fontsize=10)
    axes[1].text(years[-1], sales[-1], f' {sales[-1]}', fontsize=10)

    plt.tight_layout()
    print("\n说明：")
    print("• 左图：面积填充让最后一年'看起来占比很大")
    print("• 右图：线条展示的是'增长趋势'，视觉上更平衡")

    print(f"\n输出：{OUTPUT_DIR / 'area_deception_comparison.png'}")
    plt.savefig(OUTPUT_DIR / 'area_deception_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def demonstrate_dual_axis_trick():
    """演示双 Y 轴陷阱"""
    print("\n" + "=" * 60)
    print("误导 #3：双 Y 轴 —— 相关性错觉")
    print("=" * 60)

    years = [2018, 2019, 2020, 2021, 2022]
    revenue = [100, 105, 108, 112, 118]  # 左轴
    growth_rate = [5, 8, 12, 15, 18]  # 右轴

    fig, ax = plt.subplots(figsize=(10, 5))

    color1 = 'steelblue'
    color2 = 'coral'

    # 左轴：收入
    line1 = ax.plot(years, revenue, 'o-', color=color1, markersize=8, label='收入（万元）')
    ax.set_ylabel('收入（万元）', color=color1, fontsize=12)
    ax.tick_params(axis='y', colors=color1)

    # 右轴：增长率
    ax2 = ax.twinx()
    line2 = ax2.plot(years, revenue, 's--', color=color2, markersize=8, label='增长率（%）')
    ax2.set_ylabel('增长率（%）', color=color2, fontsize=12)
    ax2.tick_params(axis='y', colors=color2)

    ax.set_title('同一数据，两个 Y 轴的"相关"错觉', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    plt.tight_layout()
    print("\n说明：")
    print("• 两条线走的是同一数据！")
    print("• 但双 Y 轴给人'两个指标一起变化'的错觉")
    print("• 如果想看相关，应该画散点图，不是双 Y 轴折线")

    print(f"\n输出：{OUTPUT_DIR / 'dual_axis_trick.png'}")
    plt.savefig(OUTPUT_DIR / 'dual_axis_trick.png', dpi=150, bbox_inches='tight')
    plt.close()


def demonstrate_honest_alternatives():
    """演示诚实的替代方案"""
    print("\n" + "=" * 60)
    print("✅ 诚实可视化的原则")
    print("=" * 60)

    # 生成示例数据
    np.random.seed(42)
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 25, 21, 27, 24]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：带误差条的柱状图
    bars = axes[0].bar(categories, values, yerr=2, capsize=5,
                          color='steelblue', alpha=0.7)
    axes[0].set_ylabel('数值')
    axes[0].set_title('带误差条：展示不确定性', fontsize=12)
    axes[0].set_ylim(0, 35)
    axes[0].grid(axis='y', alpha=0.3)

    # 右图：完整标签
    bars2 = axes[1].bar(categories, values, color='steelblue', alpha=0.7)
    axes[1].set_ylabel('数值')
    axes[1].set_title('完整标签：直接标注数值', fontsize=12)
    axes[1].set_ylim(0, 35)
    axes[1].grid(axis='y', alpha=0.3)

    for i, (cat, val) in enumerate(zip(categories, values)):
        axes[1].text(i, val + 0.5, f'{val}', ha='center', fontsize=11)

    plt.tight_layout()
    print(f"\n输出：{OUTPUT_DIR / 'honest_alternatives.png'}")
    plt.savefig(OUTPUT_DIR / 'honest_alternatives.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n诚实可视化清单：")
    print("1. Y 轴从 0 开始（除非有充分理由）")
    print("2. 比例时保持比例（1:1）")
    print("3. 用误差条展示不确定性")
    print("4. 标注实际数值，不只是看图")
    print("5. 避免双 Y 轴（除非必要）")
    print("6. 用面积填充时要谨慎")


def create_honesty_checklist():
    """创建可视化检查清单"""
    print("\n" + "=" * 60)
    print("可视化诚实性自检清单")
    print("=" * 60)

    checklist = """
在发布图表前，问自己这些问题：

□ Y 轴是否从 0 开始？
  → 如果否，我是否有充分理由截断？
  → 我是否在标题/注释中说明了截断？

□ 比例是否真实反映了数据关系？
  → 如果是 1:1，我是否标注了？
  → 如果不是，是否用散点图替代？

□ 是否有误导性的视觉效果？
  → 面积填充是否夸大了差异？
  → 3D 效果是否必要？

□ 标签是否完整清晰？
  → 坐标轴有单位吗？
  → 图例能解释所有元素吗？

□ 数据来源是否说明？
  → 是否标注了样本量？
  → 是否标注了时间范围？
  → 是否说明了数据来源？

□ 如果 AI 生成这张图，我审查过吗？
  → 图表类型适合数据类型吗？
  → 有没有误导性截断？
  → 结论是否与图表一致？
    """

    print(checklist)

    # 写入文件
    checklist_path = OUTPUT_DIR / 'honesty_checklist.md'
    with open(checklist_path, 'w', encoding='utf-8') as f:
        f.write("# 可视化诚实性自检清单\n\n")
        f.write(checklist)

    print(f"\n清单已保存到 {checklist_path}")


if __name__ == "__main__":
    setup_style()

    # 运行所有演示
    demonstrate_truncated_y_axis()
    demonstrate_area_deception()
    demonstrate_dual_axis_trick()
    demonstrate_honest_alternatives()
    create_honesty_checklist()

    print("\n" + "=" * 60)
    print("核心要点总结")
    print("=" * 60)
    print("1. 截断 Y 轴是最高发的误导性做法")
    print("2. 面积填充会夸大视觉差异")
    print("3. 双 Y 轴容易造成相关错觉")
    print("4. 诚实原则：Y 轴从 0、标注数值、说明来源")
    print("5. 老潘：'图表是武器，你要为它负责'")
