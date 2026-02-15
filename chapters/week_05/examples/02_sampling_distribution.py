"""
示例：抽样分布模拟——理解"如果真实没有差异，会看到多大的虚假差异"。

本例演示从同一总体重复抽样的过程：
1. 假设 A 渠道和 B 渠道的真实转化率都是 10%（无差异）
2. 模拟 1000 次实验，每次都抽取两个样本并计算差异
3. 观察"虚假差异"的分布

运行方式：python3 chapters/week_05/examples/02_sampling_distribution.py
预期输出：stdout 输出模拟结果 + 图表保存到 output/sampling_distribution_null.png

核心概念：
- 抽样分布：统计量（差异）的分布
- 标准误：抽样分布的标准差
- 虚假差异：即使真实无差异，抽样也可能产生"看起来显著"的差异

关键发现：
- 阿码：即使真实转化率都是 10%，偶尔也会看到 12% vs 9% 的 3% 差异
- 小北：但如果这个 3% 处于分布的边缘（如前 5%），就更可能是真实差异
- 老潘：这就是 p 值的直觉——你的观察值在抽样分布中的"排名"

场景：
阿码上周算出"A 渠道的转化率是 12%，B 渠道是 9%"，差异是 3%。
老潘问："如果真实转化率都是 10%，你会不会偶尔也看到 12% vs 9%？"
这个模拟就是回答这个问题。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def setup_chinese_font() -> str:
    """配置中文字体"""
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
    """主函数：模拟无差异场景下的抽样分布"""
    setup_chinese_font()

    # 使用现代 NumPy 随机数生成器
    rng = np.random.default_rng(seed=42)

    print("=" * 60)
    print("抽样分布模拟：'如果真实没有差异'")
    print("=" * 60)

    # 设定：真实转化率都是 10%（无差异）
    true_rate = 0.10
    sample_size = 1000  # 每个渠道 1000 个用户

    print(f"\n设定：")
    print(f"  - A 渠道真实转化率：{true_rate:.0%}")
    print(f"  - B 渠道真实转化率：{true_rate:.0%}")
    print(f"  - 样本量：每个渠道 {sample_size} 个用户")
    print(f"  - 真实差异：{true_rate - true_rate:.0%}")

    # 模拟 1000 次实验
    n_simulations = 1000
    differences = []

    print(f"\n开始模拟 {n_simulations} 次实验...")

    for i in range(n_simulations):
        # 从同一个总体抽样（A 和 B 的真实转化率相同）
        sample_a = rng.binomial(n=1, p=true_rate, size=sample_size)
        sample_b = rng.binomial(n=1, p=true_rate, size=sample_size)

        # 计算转化率
        rate_a = sample_a.mean()
        rate_b = sample_b.mean()

        # 记录差异
        differences.append(rate_a - rate_b)

    differences = np.array(differences)

    # 计算统计量
    std_error = differences.std()
    prob_ge_3pc = (differences >= 0.03).mean()

    print(f"\n模拟结果：")
    print(f"  - 差异的标准差（标准误）：{std_error:.2%}")
    print(f"  - 差异 ≥ 3% 的比例：{prob_ge_3pc:.2%}")
    print(f"  - 差异落在 ±2% 的比例：{((differences >= -0.02) & (differences <= 0.02)).mean():.1%}")

    # 可视化
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：单次模拟的样本分布（A 渠道）
    # 用最后一次模拟的结果作为示例
    axes[0].bar(["未转化", "转化"], [(1 - sample_a).sum(), sample_a.sum()],
                color=['orange', 'steelblue'])
    axes[0].set_ylabel("用户数")
    axes[0].set_title(f"单次模拟：A 渠道的 {sample_size} 个用户\n（真实转化率 10%）")
    axes[0].set_ylim(0, sample_size)

    # 右图：1000 次模拟的差异分布
    axes[1].hist(differences, bins=np.arange(-0.04, 0.04, 0.005),
                 edgecolor="black", alpha=0.7, color='steelblue')
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2,
                    label="真实差异=0")
    axes[1].axvline(0.03, color="blue", linestyle="--", linewidth=2,
                    label="你观察到的差异=3%")
    axes[1].axvline(-0.03, color="blue", linestyle="--", linewidth=2)
    axes[1].set_xlabel("A 渠道转化率 - B 渠道转化率")
    axes[1].set_ylabel("模拟次数")
    axes[1].set_title("重复 1000 次的'虚假差异'分布\n（真实无差异）")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "sampling_distribution_null.png", dpi=150,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    print(f"\n图表已保存到 {output_dir}/sampling_distribution_null.png")

    # 核心结论
    print("\n" + "=" * 60)
    print("核心结论")
    print("=" * 60)
    print("1. 抽样分布：即使真实无差异，重复抽样会产生不同的'虚假差异'")
    print("2. 标准误：抽样分布的标准差，描述'统计量的不确定性'")
    print(f"3. 你观察到的 3% 差异：在本次模拟中出现的概率是 {prob_ge_3pc:.1%}")
    print("4. p 值的直觉：把你的观察值放在抽样分布中，看它排在什么位置")
    print("\n阿码：'所以如果 3% 处于分布的边缘，就说明不太可能是运气？'")
    print("老潘：'对。这就是 Week 06 假设检验的核心思想。'")


if __name__ == "__main__":
    main()
