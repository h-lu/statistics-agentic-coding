"""
示例：Bootstrap 方法入门——用一个样本估计抽样分布。

本例演示如何用 Bootstrap 方法量化不确定性：
1. 从真实数据（A 渠道 12% vs B 渠道 9%，差异 3%）出发
2. 用重采样方法估计差异的抽样分布
3. 计算标准误和 95% 置信区间

运行方式：python3 chapters/week_05/examples/04_bootstrap_intro.py
预期输出：stdout 输出 Bootstrap 结果 + 图表保存到 output/bootstrap_distribution.png

核心概念：
- Bootstrap：从样本中"有放回地"重抽样，模拟"从总体重复抽样"的过程
- 标准误：Bootstrap 分布的标准差
- 置信区间：Bootstrap 分布的 2.5% 和 97.5% 分位数
- 非参数方法：不假设总体分布是什么形状

关键发现：
- 阿码："我只有一个样本，怎么算标准误？"
- 小北："Bootstrap 是不是'自己抽自己'？这靠谱吗？"
- 老潘："Bootstrap 估计的是'抽样不确定性'，不是'所有不确定性'"

场景：
阿码上周算出"A 渠道的转化率是 12%，B 渠道是 9%"，差异是 3%。
老潘问："这个 3% 的标准误是多少？"
阿码愣住了："我只有一个样本啊！"
Bootstrap 的答案：用你唯一的样本"假装"重复抽样 1000 次。

Bootstrap 的有效性依赖于一个假设：
- 你的样本是总体的代表
- 如果样本有偏差，Bootstrap 无法修复这个偏差

应用场景（2026 年）：
- 深度学习模型的"预测区间"
- 大模型输出的"不确定性"量化
- A/B 测试的"转化率差异置信区间"
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


def bootstrap_difference(data_a: np.ndarray, data_b: np.ndarray,
                          n_bootstrap: int = 1000,
                          random_state: int = 42) -> np.ndarray:
    """
    Bootstrap 重采样：计算两组差异的分布

    参数:
        data_a: A 组数据（如 A 渠道的转化数据）
        data_b: B 组数据（如 B 渠道的转化数据）
        n_bootstrap: Bootstrap 次数
        random_state: 随机种子

    返回:
        Bootstrap 差异分布（array）
    """
    rng = np.random.default_rng(seed=random_state)

    n_a = len(data_a)
    n_b = len(data_b)
    bootstrap_diffs = []

    for _ in range(n_bootstrap):
        # 有放回地重抽样
        boot_sample_a = rng.choice(data_a, size=n_a, replace=True)
        boot_sample_b = rng.choice(data_b, size=n_b, replace=True)

        # 计算差异
        boot_diff = boot_sample_a.mean() - boot_sample_b.mean()
        bootstrap_diffs.append(boot_diff)

    return np.array(bootstrap_diffs)


def main() -> None:
    """主函数：Bootstrap 不确定性量化"""
    setup_chinese_font()

    print("=" * 60)
    print("Bootstrap 方法：量化'3% 差异'的不确定性")
    print("=" * 60)

    # 假设这是你收集的真实数据
    # A 渠道：120 个转化（12%）
    # B 渠道：90 个转化（9%）
    conversions_a = np.array([1] * 120 + [0] * 880)
    conversions_b = np.array([1] * 90 + [0] * 910)

    # 观察到的差异
    observed_diff = conversions_a.mean() - conversions_b.mean()
    rate_a = conversions_a.mean()
    rate_b = conversions_b.mean()

    print(f"\n观察到的数据：")
    print(f"  - A 渠道：{conversions_a.sum()}/{len(conversions_a)} 转化（{rate_a:.1%}）")
    print(f"  - B 渠道：{conversions_b.sum()}/{len(conversions_b)} 转化（{rate_b:.1%}）")
    print(f"  - 差异：{observed_diff:.2%}")

    # Bootstrap：重复 1000 次
    n_bootstrap = 1000
    print(f"\n开始 Bootstrap（{n_bootstrap} 次重采样）...")

    bootstrap_diffs = bootstrap_difference(
        conversions_a, conversions_b,
        n_bootstrap=n_bootstrap,
        random_state=42
    )

    # 计算统计量
    standard_error = bootstrap_diffs.std()
    ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])

    print(f"\nBootstrap 结果：")
    print(f"  - 标准误：{standard_error:.2%}")
    print(f"  - 95% 置信区间：[{ci_low:.2%}, {ci_high:.2%}]")

    # 判断是否包含 0
    if ci_low > 0:
        print(f"  - 结论：置信区间不包含 0，表明 A 渠道可能真的更好")
    elif ci_high < 0:
        print(f"  - 结论：置信区间不包含 0，表明 B 渠道可能真的更好")
    else:
        print(f"  - 结论：置信区间包含 0，差异可能不显著")

    # 可视化
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))

    # 直方图
    plt.hist(bootstrap_diffs, bins=30, edgecolor="black",
             alpha=0.7, color='steelblue')

    # 标记线
    plt.axvline(observed_diff, color="red", linestyle="--", linewidth=2,
                label=f"观察到的差异={observed_diff:.2%}")
    plt.axvline(0, color="black", linestyle="-", linewidth=1,
                label="无差异线")
    plt.axvline(ci_low, color="blue", linestyle=":", linewidth=2,
                label=f"95% CI 下限={ci_low:.2%}")
    plt.axvline(ci_high, color="blue", linestyle=":", linewidth=2,
                label=f"95% CI 上限={ci_high:.2%}")

    # 填充置信区间
    plt.axvspan(ci_low, ci_high, alpha=0.2, color='blue')

    plt.xlabel("A 渠道转化率 - B 渠道转化率")
    plt.ylabel("Bootstrap 次数")
    plt.title("Bootstrap 抽样分布：差异的分布")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_distribution.png", dpi=150,
                bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    print(f"\n图表已保存到 {output_dir}/bootstrap_distribution.png")

    # 核心结论
    print("\n" + "=" * 60)
    print("核心结论")
    print("=" * 60)
    print("1. Bootstrap 的核心思想：")
    print("   从样本中'有放回地'重抽样，模拟'从总体重复抽样'")
    print("2. 标准误：Bootstrap 分布的标准差，描述'统计量的不确定性'")
    print("3. 置信区间：Bootstrap 分布的 2.5% 和 97.5% 分位数")
    print("4. 非参数方法：不假设总体分布是什么形状")
    print("\n小北：'所以 Bootstrap 不用假设数据服从正态分布？'")
    print("老潘：'对。但前提是——你的样本是总体的代表。'")
    print("\n阿码：'如果样本有偏差怎么办？'")
    print("老潘：'Bootstrap 无法修复偏差。它只能告诉你：")
    print("        如果你的样本有偏差，结论的不确定性有多大。'")
    print("\n应用场景（2026）：")
    print("  - 深度学习模型的'预测区间'")
    print("  - 大模型输出的'不确定性'量化")
    print("  - A/B 测试的'转化率差异置信区间'")


if __name__ == "__main__":
    main()
