"""
示例：抛硬币模拟——理解随机性与长期频率。

本例演示如何用模拟建立概率直觉：
1. 抛10次硬币：单次实验的随机波动大
2. 抛1000次硬币：大数定律，频率趋近概率
3. 重复1000次"每次抛100次"：正面次数的抽样分布

运行方式：python3 chapters/week_05/examples/01_simulation_intro.py
预期输出：stdout 输出模拟结果 + 图表保存到 output/simulation_coin_flip.png

核心概念：
- 随机性：单次实验结果不可预测
- 长期频率：重复次数足够多时，频率趋近真实概率
- 抽样分布：统计量（正面次数）的分布

关键发现：
- 小北：单次抛10次可能得到7正3反，这不算"异常"，只是随机波动
- 阿码：重复1000次实验后，大部分实验的正面次数集中在40-60之间
- 老潘：钟形曲线的宽度描述了"随机性有多大"，这是标准误的前身
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
    chinese_fonts = ['SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS',
                     'PingFang SC', 'Microsoft YaHei']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


def main() -> None:
    """主函数：运行模拟实验"""
    setup_chinese_font()

    # 使用现代 NumPy 随机数生成器
    rng = np.random.default_rng(seed=42)

    print("=" * 60)
    print("抛硬币模拟实验")
    print("=" * 60)

    # 实验 1：抛 10 次硬币
    print("\n【实验 1】抛 10 次硬币")
    flips_10 = rng.choice(["H", "T"], size=10)
    heads_10 = (flips_10 == "H").sum()
    print(f"结果：{flips_10}")
    print(f"正面 {heads_10} 次，反面 {10 - heads_10} 次")
    print(f"正面频率：{heads_10 / 10:.1%}")

    # 实验 2：抛 1000 次硬币
    print("\n【实验 2】抛 1000 次硬币")
    flips_1000 = rng.choice(["H", "T"], size=1000)
    heads_1000 = (flips_1000 == "H").sum()
    print(f"正面 {heads_1000} 次，反面 {1000 - heads_1000} 次")
    print(f"正面频率：{heads_1000 / 1000:.1%}")

    # 实验 3：重复 1000 次"每次抛 100 次"
    print("\n【实验 3】重复 1000 次'每次抛 100 次'")
    n_simulations = 1000
    n_flips_per_sim = 100
    heads_counts = []

    for i in range(n_simulations):
        flips = rng.choice([0, 1], size=n_flips_per_sim)
        heads_counts.append(flips.sum())

    heads_counts = np.array(heads_counts)

    print(f"1000次实验的正面次数统计：")
    print(f"  均值：{heads_counts.mean():.1f}")
    print(f"  标准差：{heads_counts.std():.1f}")
    print(f"  最小值：{heads_counts.min()}")
    print(f"  最大值：{heads_counts.max()}")
    print(f"  落在 [40, 60] 范围的比例：{((heads_counts >= 40) & (heads_counts <= 60)).mean():.1%}")

    # 可视化
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：单次实验的结果分布
    axes[0].bar(["正面", "反面"], [heads_10, 10 - heads_10],
                color=['steelblue', 'orange'])
    axes[0].set_ylabel("次数")
    axes[0].set_title("抛 10 次的结果（单次实验）")
    axes[0].set_ylim(0, 10)

    # 右图：1000 次实验的正面次数分布
    axes[1].hist(heads_counts, bins=np.arange(35, 66, 1),
                 edgecolor="black", alpha=0.7, color='steelblue')
    axes[1].axvline(50, color="red", linestyle="--", linewidth=2,
                    label="期望值=50")
    axes[1].axvline(heads_counts.mean(), color="blue", linestyle="-",
                    linewidth=2, label=f"样本均值={heads_counts.mean():.1f}")
    axes[1].set_xlabel("正面次数")
    axes[1].set_ylabel("实验次数")
    axes[1].set_title("重复 1000 次'每次抛 100 次'的正面次数分布")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "simulation_coin_flip.png", dpi=150,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    print(f"\n图表已保存到 {output_dir}/simulation_coin_flip.png")

    # 核心结论
    print("\n" + "=" * 60)
    print("核心结论")
    print("=" * 60)
    print("1. 单次实验（抛10次）结果不稳定：可能得到7正3反")
    print("2. 大数定律：抛1000次后，频率趋近真实概率（50%）")
    print("3. 抽样分布：重复1000次实验，正面次数形成'钟形曲线'")
    print("4. 标准差（约5次）描述了'随机性有多大'")
    print("\n这就是'从确定性到概率性思维'的第一步！")


if __name__ == "__main__":
    main()
