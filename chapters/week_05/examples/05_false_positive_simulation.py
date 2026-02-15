"""
示例：假阳性模拟——理解"多次检验的风险"。

本例演示假阳性（第一类错误）的问题：
1. 假设真实没有差异，但你检验了 5 个假设
2. 每次检验都有 5% 的概率误判为"有差异"（α = 0.05）
3. 模拟 1000 次"实验"，看看"至少出现 1 个假阳性"的概率

运行方式：python3 chapters/week_05/examples/05_false_positive_simulation.py
预期输出：stdout 输出模拟结果 + 图表保存到 output/false_positive_distribution.png

核心概念：
- 假阳性（第一类错误）：真实没有差异，但检验结论说"有差异"
- 显著性水平 α：即使真实无差异，仍有 α 的概率误判为"有差异"
- 多重比较问题：检验的假设越多，假阳性的概率越高
- 至少 1 个假阳性的概率 = 1 - (1 - α)^k

关键发现：
- 小北："如果我的置信区间不包含 0，是不是就证明差异真实存在？"
- 老潘："你一开始想检验多少个假设？"
- 阿码："5% 的假阳性概率不大吧？"
- 老潘："如果你检验 5 个假设，假阳性的概率会变成 23%！"

场景：
小北用 Bootstrap 算出了 95% 置信区间 [0.5%, 5.5%]，不包含 0。
他激动地说："所以 A 渠道真的比 B 渠道好！"
老潘问："你上周的假设清单里有 5 个假设。如果你同时检验 5 个，"
       "即使真实都没有差异，你也有很大概率至少看到 1 个'假阳性'。"

计算：
- 只检验 1 个：假阳性概率 = 5%
- 检验 5 个：假阳性概率 = 1 - (1 - 0.05)^5 ≈ 23%
- 检验 10 个：假阳性概率 = 1 - (1 - 0.05)^10 ≈ 40%

这就是为什么需要多重比较校正（Week 07）！
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


def simulate_false_positive_rate(n_hypotheses: int,
                                   alpha: float = 0.05,
                                   n_simulations: int = 1000,
                                   random_state: int = 42) -> float:
    """
    模拟假阳性率

    参数:
        n_hypotheses: 检验的假设数量
        alpha: 显著性水平（每次检验的假阳性概率）
        n_simulations: 模拟次数
        random_state: 随机种子

    返回:
        至少 1 个假阳性的概率
    """
    rng = np.random.default_rng(seed=random_state)

    false_positive_counts = []

    for _ in range(n_simulations):
        # 每次实验检验 n_hypotheses 个假设（真实都无差异）
        # p 值服从均匀分布 U(0, 1)
        p_values = rng.uniform(0, 1, size=n_hypotheses)

        # 统计有多少个"假阳性"（p < alpha）
        false_positives = (p_values < alpha).sum()
        false_positive_counts.append(false_positives)

    false_positive_counts = np.array(false_positive_counts)

    # 计算概率
    prob_at_least_one = (false_positive_counts >= 1).mean()

    return prob_at_least_one, false_positive_counts


def main() -> None:
    """主函数：假阳性模拟"""
    setup_chinese_font()

    print("=" * 60)
    print("假阳性模拟：'多次检验的风险'")
    print("=" * 60)

    # 设定
    alpha = 0.05  # 显著性水平
    n_simulations = 1000  # 模拟次数

    print(f"\n设定：")
    print(f"  - 显著性水平 α = {alpha}")
    print(f"  - 每次检验的假阳性概率：{alpha:.0%}")
    print(f"  - 模拟次数：{n_simulations}")

    # 模拟不同假设数量下的假阳性率
    print(f"\n" + "=" * 60)
    print("不同假设数量下的假阳性率")
    print("=" * 60)

    hypothesis_counts = [1, 2, 5, 10, 20]
    results = []

    for n_hyp in hypothesis_counts:
        prob, counts = simulate_false_positive_rate(
            n_hyp, alpha, n_simulations, random_state=42
        )
        theoretical = 1 - (1 - alpha) ** n_hyp
        results.append((n_hyp, prob, theoretical))
        print(f"\n检验 {n_hyp} 个假设：")
        print(f"  - 模拟结果：至少 1 个假阳性的概率 = {prob:.2%}")
        print(f"  - 理论值：1 - (1 - α)^{n_hyp} = {theoretical:.2%}")

    # 详细模拟：5 个假设的情况
    print(f"\n" + "=" * 60)
    print("详细模拟：检验 5 个假设")
    print("=" * 60)

    n_hypotheses = 5
    prob_at_least_one, false_positive_counts = simulate_false_positive_rate(
        n_hypotheses, alpha, n_simulations, random_state=42
    )

    print(f"\n模拟结果：")
    print(f"  - 至少 1 个假阳性的概率：{prob_at_least_one:.2%}")
    print(f"  - 假阳性数量分布：")

    for n_fp in range(n_hypotheses + 1):
        count = (false_positive_counts == n_fp).sum()
        pct = count / n_simulations
        print(f"    {n_fp} 个假阳性：{count} 次（{pct:.1%}）")

    # 可视化
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 图 1：假阳性率 vs 假设数量
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：假阳性率随假设数量的变化
    n_hyps_range = np.arange(1, 21)
    theoretical_probs = 1 - (1 - alpha) ** n_hyps_range

    axes[0].plot(n_hyps_range, theoretical_probs, 'o-', color='steelblue',
                 linewidth=2, markersize=6)
    axes[0].axhline(alpha, color='red', linestyle='--',
                    linewidth=2, label=f'单次检验 α={alpha}')
    axes[0].set_xlabel("检验的假设数量")
    axes[0].set_ylabel("至少 1 个假阳性的概率")
    axes[0].set_title("假阳性率随假设数量的变化")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图：5 个假设的假阳性数量分布
    counts, bins, patches = axes[1].hist(
        false_positive_counts, bins=np.arange(-0.5, 5.5, 1),
        edgecolor="black", alpha=0.7, color='steelblue'
    )
    axes[1].set_xlabel("假阳性数量")
    axes[1].set_ylabel("模拟次数")
    axes[1].set_title(f"重复 {n_simulations} 次的假阳性分布\n（检验了 {n_hypotheses} 个假设）")

    # 标注"至少 1 个假阳性"的区域
    for i in range(1, len(patches)):
        patches[i].set_facecolor("coral")
    axes[1].axvline(0.5, color="green", linestyle="--", linewidth=2,
                    label="无假阳性")
    axes[1].axvline(1.5, color="red", linestyle="--", linewidth=2,
                    label="≥1 个假阳性")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "false_positive_distribution.png", dpi=150,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    print(f"\n图表已保存到 {output_dir}/false_positive_distribution.png")

    # 核心结论
    print("\n" + "=" * 60)
    print("核心结论")
    print("=" * 60)
    print("1. 假阳性（第一类错误）：")
    print("   真实没有差异，但检验结论说'有差异'")
    print("2. 多重比较问题：")
    print("   检验的假设越多，假阳性的概率越高")
    print(f"3. 公式：至少 1 个假阳性的概率 = 1 - (1 - α)^k")
    print(f"   - k=1：{alpha:.0%}")
    print(f"   - k=5：{1 - (1 - alpha)**5:.0%}")
    print(f"   - k=10：{1 - (1 - alpha)**10:.0%}")
    print("\n阿码：'所以如果我检验 5 个假设，")
    print("       即使真实都没有差异，也有 23% 的概率看到'显著'结果？'")
    print("老潘：'对。所以你在写报告时，必须说明：你检验了多少个假设。'")
    print("\n小北：'那我怎么办？难道不能检验多个假设吗？'")
    print("老潘：'可以，但需要做多重比较校正（Week 07）。")
    print("        Bonferroni 校正：把 α 从 0.05 调整为 0.05/k'")
    print(f"        检验 5 个时：α' = 0.05/5 = 0.01'")
    print("        只有 p < 0.01 时，才认为'显著''")


if __name__ == "__main__":
    main()
