"""
示例：p 值的可视化——理解观察值在抽样分布中的位置。

本例演示如何通过模拟"无差异"场景下的抽样分布，来可视化 p 值的含义。
p 值是"在原假设成立时，观察到当前数据或更极端数据的概率"。

运行方式：python3 chapters/week_06/examples/01_p_value_intuition.py
预期输出：
  - stdout 输出观察到的差异和 p 值
  - 图表保存到 output/p_value_visualization.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
    import matplotlib.font_manager as fm
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
    """运行 p 值可视化演示"""
    setup_chinese_font()

    # 固定随机种子确保可复现
    np.random.seed(42)

    # 设定：真实无差异（原假设）
    true_diff = 0
    sample_size = 1000
    n_simulations = 10000

    print("=== p 值可视化演示 ===\n")
    print(f"模拟设定：")
    print(f"  - 原假设：真实差异 = {true_diff}")
    print(f"  - 样本量：{sample_size}")
    print(f"  - 模拟次数：{n_simulations}")
    print(f"  - 真实转化率：10%（两组相同）\n")

    # 模拟"无差异"场景下的抽样分布
    print("正在模拟抽样分布...")
    differences = []
    for _ in range(n_simulations):
        sample_a = np.random.binomial(n=1, p=0.10, size=sample_size)
        sample_b = np.random.binomial(n=1, p=0.10, size=sample_size)
        differences.append(sample_a.mean() - sample_b.mean())

    differences = np.array(differences)

    # 你观察到的差异
    observed_diff = 0.03

    # 计算 p 值（双尾）
    p_value_two_tailed = (np.abs(differences) >= np.abs(observed_diff)).mean()
    print(f"\n=== 结果 ===")
    print(f"观察到的差异：{observed_diff:.2%}")
    print(f"双尾 p 值：{p_value_two_tailed:.4f}")
    print(f"\n解释：如果真实没有差异，观察到 {observed_diff:.1%} 或更大差异的概率约为 {p_value_two_tailed:.1%}")

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(differences, bins=50, edgecolor="black", alpha=0.7, density=True, color='steelblue')
    ax.axvline(0, color="black", linestyle="-", linewidth=2, label="原假设：差异=0")
    ax.axvline(observed_diff, color="red", linestyle="--", linewidth=2, label=f"观察到的差异={observed_diff:.2%}")
    ax.axvline(-observed_diff, color="red", linestyle="--", linewidth=2)

    # 标注"极端区域"（p 值区域）
    xtreme_right = differences[differences >= observed_diff]
    xtreme_left = differences[differences <= -observed_diff]
    ax.fill_between(xtreme_right, 0, 20, alpha=0.3, color="red", label="p 值区域")
    ax.fill_between(xtreme_left, 0, 20, alpha=0.3, color="red")

    ax.set_xlabel("A 渠道转化率 - B 渠道转化率", fontsize=12)
    ax.set_ylabel("密度", fontsize=12)
    ax.set_title("p 值的可视化：观察值在抽样分布中的位置", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, min(20, ax.get_ylim()[1]))

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'output'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'p_value_visualization.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"\n图表已保存到 {output_dir}/p_value_visualization.png")


if __name__ == "__main__":
    main()
