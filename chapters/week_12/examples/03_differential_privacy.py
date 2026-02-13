"""
示例：差分隐私——通过噪声保护隐私

本例演示：
1. 什么是差分隐私（Differential Privacy）
2. 拉普拉斯机制的实现
3. 如何选择隐私预算 ε
4. 差分隐私与数据实用性的权衡

运行方式：python3 chapters/week_12/examples/03_differential_privacy.py
预期输出：
- 真实统计量 vs 噪声统计量对比
- 不同 ε 值下的隐私-效用权衡图
- 理解 ε 的含义

依赖安装：
pip install pandas numpy matplotlib
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_12"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子保证可复现
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def laplace_mechanism(true_value: float, sensitivity: float, epsilon: float) -> float:
    """
    拉普拉斯机制：添加噪声保护隐私

    参数：
        true_value: 真实的统计量值（如均值、计数等）
        sensitivity: 全局敏感度（删除一条记录对统计量的最大影响）
        epsilon: 隐私预算（越小，隐私保护越强）

    返回：
        添加噪声后的值
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return true_value + noise


def count_sensitivity(data_range: int) -> float:
    """
    计数查询的全局敏感度

    对于计数查询，删除一条记录最多使计数变化 1
    因此敏感度 = 1
    """
    return 1.0


def mean_sensitivity(data_min: float, data_max: float, n: int) -> float:
    """
    计算均值查询的全局敏感度

    对于均值查询，敏感度 = (max - min) / n
    其中 max 和 min 是数据的已知范围
    """
    return (data_max - data_min) / n


def sum_sensitivity(data_range: float) -> float:
    """
    计算求和查询的全局敏感度

    对于求和查询，敏感度 = max - min
    其中 max 和 min 是单条记录的最大最小值
    """
    return data_range


def demonstrate_differential_privacy():
    """演示差分隐私的基本原理"""
    print("=" * 70)
    print("差分隐私示例：通过噪声保护隐私")
    print("=" * 70)

    # ========== 1. 什么是差分隐私？ ==========
    print("\n" + "-" * 70)
    print("1. 什么是差分隐私？")
    print("-" * 70)
    print("""
核心思想：查询结果不应该因为"有没有某条记录"而有显著差异。

攻击场景：
  - 攻击者知道"张三"不在数据集 D 中
  - 攻击者查询"收入均值"，得到 50000
  - 攻击者将张三加入数据集，再次查询，得到 50100
  - 攻击者推断：张三的收入约为 60000

差分隐私的解决方案：
  - 在查询结果中添加"精心控制的噪声"
  - 噪声大小由"全局敏感度"和"隐私预算 ε"决定
  - 攻击者无法确定"某个特定个体是否在数据集中"

两个参数：
  1. ε（epsilon）：隐私损失上限
     - ε 越小，噪声越大，隐私保护越强，但数据实用性越低
     - ε = 1.0 通常被认为是"可接受的平衡"
     - ε < 0.1 是"强隐私"，但噪声很大

  2. 敏感度（Sensitivity）：删除一条记录对查询结果的最大影响
     - 计数：敏感度 = 1
     - 均值：敏感度 = (max - min) / n
     - 求和：敏感度 = max - min
    """)

    # ========== 2. 生成示例数据 ==========
    print("\n" + "-" * 70)
    print("2. 生成示例数据")
    print("-" * 70)

    n = 1000
    income = np.random.lognormal(10, 0.5, n)

    print(f"\n模拟 {n} 个人的收入数据：")
    print(f"  收入均值: {income.mean():.2f}")
    print(f"  收入中位数: {np.median(income):.2f}")
    print(f"  收入范围: [{income.min():.2f}, {income.max():.2f}]")

    # ========== 3. 差分隐私：均值查询 ==========
    print("\n" + "-" * 70)
    print("3. 差分隐私：均值查询")
    print("-" * 70)

    true_mean = income.mean()

    # 计算敏感度
    data_min, data_max = 0, 200000  # 假设已知收入范围
    sensitivity = mean_sensitivity(data_min, data_max, n)

    print(f"\n真实均值: {true_mean:.2f}")
    print(f"敏感度: {sensitivity:.2f}")
    print(f"  （删除一条记录最多使均值变化 {sensitivity:.2f}）")

    # 尝试不同的 ε 值
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print(f"\n不同 ε 值下的噪声均值：")
    print(f"{'ε':>8s} {'噪声均值':>15s} {'误差':>15s} {'隐私级别':>15s}")
    print("-" * 60)

    for eps in epsilons:
        private_mean = laplace_mechanism(true_mean, sensitivity, eps)
        error = abs(private_mean - true_mean)

        if eps < 0.5:
            level = "强隐私"
        elif eps < 2:
            level = "中等"
        else:
            level = "弱隐私"

        print(f"{eps:8.1f} {private_mean:15.2f} {error:15.2f} {level:>15s}")

    print(f"\n关键观察：")
    print(f"  - ε = 0.1：噪声很大（误差可能 > 10000），但隐私保护很强")
    print(f"  - ε = 1.0：噪声适中（误差 ~ 1000-3000），常用选择")
    print(f"  - ε = 10：噪声很小（误差 < 1000），但隐私保护弱")

    # ========== 4. 多次查询的隐私损失累积 ==========
    print("\n" + "-" * 70)
    print("4. 隐私预算的累积性")
    print("-" * 70)
    print("""
重要：差分隐私具有"累积性"。

如果你进行多次查询，总隐私损失是各次查询的 ε 之和。

例如：
  - 查询 1：ε = 0.5
  - 查询 2：ε = 0.5
  - 查询 3：ε = 0.5
  - 总隐私损失：ε_total = 0.5 + 0.5 + 0.5 = 1.5

这意味着：你不能无限制地查询数据！
每次查询都在"消耗"隐私预算，预算用完后，数据就不可用了。

解决方案：
  1. 预先规划所有查询，分配隐私预算
  2. 使用组合定理（如 advanced composition）
  3. 对数据进行快照，每次快照有独立的隐私预算
    """)

    # ========== 5. 隐私-效用权衡可视化 ==========
    print("\n" + "-" * 70)
    print("5. 隐私-效用权衡可视化")
    print("-" * 70)

    # 对每个 ε 值进行多次实验
    n_trials = 100
    epsilons_plot = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    results = []

    for eps in epsilons_plot:
        errors = []
        for _ in range(n_trials):
            private_mean = laplace_mechanism(true_mean, sensitivity, eps)
            error = abs(private_mean - true_mean)
            errors.append(error)
        results.append({
            'epsilon': eps,
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
        })

    results_df = pd.DataFrame(results)

    print("\n不同 ε 值下的误差统计（100 次实验）：")
    print(results_df.round(2))

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：误差 vs ε
    ax1 = axes[0]
    ax1.plot(results_df['epsilon'], results_df['mean_error'], 'o-',
             linewidth=2, markersize=8, label='平均误差')
    ax1.fill_between(
        results_df['epsilon'],
        results_df['mean_error'] - results_df['std_error'],
        results_df['mean_error'] + results_df['std_error'],
        alpha=0.3, label='±1 标准差'
    )
    ax1.set_xlabel('隐私预算 ε', fontsize=12)
    ax1.set_ylabel('误差（元）', fontsize=12)
    ax1.set_title('隐私-效用权衡', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 右图：误差分布
    ax2 = axes[1]
    eps_to_show = [0.5, 1.0, 5.0]
    colors = ['red', 'orange', 'green']
    labels = ['ε=0.5', 'ε=1.0', 'ε=5.0']

    for i, eps in enumerate(eps_to_show):
        errors = []
        for _ in range(n_trials):
            private_mean = laplace_mechanism(true_mean, sensitivity, eps)
            errors.append(abs(private_mean - true_mean))

        ax2.hist(errors, bins=20, alpha=0.5, label=labels[i],
                 color=colors[i], density=True)

    ax2.set_xlabel('误差（元）', fontsize=12)
    ax2.set_ylabel('密度', fontsize=12)
    ax2.set_title('不同 ε 值下的误差分布', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'differential_privacy_utility.png', dpi=150, bbox_inches='tight')
    print("\n✅ 已保存: differential_privacy_utility.png")
    plt.close()

    # ========== 6. 攻击场景演示 ==========
    print("\n" + "-" * 70)
    print("6. 攻击场景：差分隐私如何防御成员推断攻击")
    print("-" * 70)

    # 创建两个数据集：D 和 D'（D' 多了一条记录）
    person_income = 80000

    # 数据集 D（不包含该记录）
    mean_D = income[:n-1].mean()

    # 数据集 D'（包含该记录）
    income_with = np.append(income[:n-1], person_income)
    mean_D_prime = income_with.mean()

    print(f"\n成员推断攻击：")
    print(f"  数据集 D（不包含张三）的均值: {mean_D:.2f}")
    print(f"  数据集 D'（包含张三）的均值: {mean_D_prime:.2f}")
    print(f"  差异: {abs(mean_D_prime - mean_D):.2f}")

    # 如果没有差分隐私，攻击者可以推断张三的收入
    print(f"\n❌ 无差分隐私：")
    print(f"  攻击者观察差异 ≈ {abs(mean_D_prime - mean_D):.2f}")
    print(f"  可以推断：张三的收入约为 {person_income} 元")

    # 使用差分隐私
    eps = 1.0
    private_mean_D = laplace_mechanism(mean_D, sensitivity, eps)
    private_mean_D_prime = laplace_mechanism(mean_D_prime, sensitivity, eps)

    print(f"\n✅ 有差分隐私（ε={eps}）：")
    print(f"  数据集 D 的噪声均值: {private_mean_D:.2f}")
    print(f"  数据集 D' 的噪声均值: {private_mean_D_prime:.2f}")
    print(f"  差异: {abs(private_mean_D_prime - private_mean_D):.2f}")

    if abs(private_mean_D_prime - private_mean_D) > sensitivity * 2:
        print(f"  ⚠️  噪声不够大，攻击者可能仍有推断能力")
    else:
        print(f"  ✓ 噪声足够大，攻击者无法确定张三是否在数据集中")

    # ========== 7. 总结 ==========
    print("\n" + "=" * 70)
    print("总结：差分隐私的核心要点")
    print("=" * 70)
    print(f"""
1. 差分隐私的定义：
   - 对任意两个相邻数据集 D 和 D'（只有一条记录不同）
   - 对任意查询结果 S
   - P(M(D) ∈ S) ≤ e^ε · P(M'D') ∈ S)

   简化理解：攻击者无法通过查询结果确定"某条记录是否在数据集中"

2. 拉普拉斯机制：
   - 噪声 ~ Laplace(0, sensitivity/ε)
   - 敏感度越大，噪声越大
   - ε 越小，噪声越大

3. ε 的选择：
   - ε = 0.1：强隐私，噪声很大
   - ε = 1.0：常用选择，平衡隐私和实用性
   - ε = 10：弱隐私，噪声很小

4. 隐私预算的累积：
   - 多次查询的 ε 总和不能超过预算
   - 需要预先规划所有查询

5. 实际应用：
   - Google、Apple、Microsoft 都在使用差分隐私
   - 用途：用户统计、数据分析、机器学习
   - 合规：满足 GDPR、CCPA 等隐私法规

工业界工具：
   - Google's DP Library（开源）
   - OpenDP（Harvard）
   - SmartNoise SDK（OpenMined）
    """)


if __name__ == "__main__":
    demonstrate_differential_privacy()
