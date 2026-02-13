"""
示例：StatLab 超级线——不确定性量化报告生成器。

本例是 Week 08 StatLab 的入口脚本，用于在 report.md 中添加
"不确定性量化"章节，包含：
- 置信区间（t 公式 + Bootstrap）
- 置换检验
- 效应量的 CI
- 贝叶斯框架简介

运行方式：python3 chapters/week_08/examples/99_statlab.py
预期输出：
  - 在当前目录生成或更新 report.md
  - 追加"不确定性量化"章节

注意：
  - 本示例使用模拟数据演示
  - 实际使用时，需要传入真实数据集
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Optional


# ==================== 核心分析函数 ====================

def bootstrap_ci_diff(group1: np.ndarray,
                      group2: np.ndarray,
                      n_bootstrap: int = 10000,
                      conf_level: float = 0.95,
                      seed: int = 42) -> tuple[np.ndarray, float, float]:
    """Bootstrap 均值差 CI（复用 examples/02_bootstrap_ci.py）"""
    np.random.seed(seed)
    n1, n2 = len(group1), len(group2)

    bootstrap_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_sample1 = np.random.choice(group1, size=n1, replace=True)
        boot_sample2 = np.random.choice(group2, size=n2, replace=True)
        bootstrap_diffs[i] = np.mean(boot_sample1) - np.mean(boot_sample2)

    alpha = 1 - conf_level
    ci_low = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_high = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return bootstrap_diffs, ci_low, ci_high


def permutation_test(group1: np.ndarray,
                     group2: np.ndarray,
                     n_permutations: int = 10000,
                     seed: int = 42) -> tuple[float, np.ndarray, float]:
    """置换检验（复用 examples/05_permutation_test.py）"""
    np.random.seed(seed)
    n1, n2 = len(group1), len(group2)
    combined = np.concatenate([group1, group2])

    observed_stat = np.mean(group1) - np.mean(group2)
    perm_stats = np.empty(n_permutations)

    for i in range(n_permutations):
        permuted = np.random.permutation(combined)
        perm_group1 = permuted[:n1]
        perm_group2 = permuted[n1:n1 + n2]
        perm_stats[i] = np.mean(perm_group1) - np.mean(perm_group2)

    if observed_stat >= 0:
        p_value = (np.mean(perm_stats >= observed_stat) +
                   np.mean(perm_stats <= -observed_stat))
    else:
        p_value = (np.mean(perm_stats <= observed_stat) +
                   np.mean(perm_stats >= -observed_stat))

    return observed_stat, perm_stats, p_value


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """计算 Cohen's d 效应量"""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) +
                          (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def t_ci_diff(group1: np.ndarray,
              group2: np.ndarray,
              conf_level: float = 0.95) -> tuple[float, float, float]:
    """t 公式计算均值差 CI（Week 06 的方法）"""
    n1, n2 = len(group1), len(group2)
    mean_diff = np.mean(group1) - np.mean(group2)

    # 合并标准误
    se_diff = np.sqrt(np.var(group1, ddof=1) / n1 + np.var(group2, ddof=1) / n2)

    # t 临界值
    df = n1 + n2 - 2
    t_critical = stats.t.ppf((1 + conf_level) / 2, df)

    margin_of_error = t_critical * se_diff
    ci_low = mean_diff - margin_of_error
    ci_high = mean_diff + margin_of_error

    return mean_diff, ci_low, ci_high


# ==================== 报告生成函数 ====================

def generate_uncertainty_section(group1: np.ndarray,
                                 group2: np.ndarray,
                                 group1_name: str = "新用户",
                                 group2_name: str = "老用户",
                                 unit: str = "元",
                                 seed: int = 42) -> str:
    """
    生成不确定性量化报告章节。

    参数：
        group1, group2: 两组数据
        group1_name, group2_name: 组名
        unit: 单位
        seed: 随机种子

    返回：
        report_section: Markdown 格式的报告章节
    """
    np.random.seed(seed)

    # 计算各种统计量
    mean_diff, t_ci_low, t_ci_high = t_ci_diff(group1, group2)
    bootstrap_diffs, boot_ci_low, boot_ci_high = bootstrap_ci_diff(group1, group2)
    observed_diff, perm_stats, p_value = permutation_test(group1, group2)
    d = cohens_d(group1, group2)

    # 置换 CI（从置换分布计算）
    perm_ci_low = np.percentile(perm_stats, 2.5)
    perm_ci_high = np.percentile(perm_stats, 97.5)

    # 效应量解释
    if abs(d) < 0.2:
        d_interpretation = "极小（negligible）"
    elif abs(d) < 0.5:
        d_interpretation = "小（small）"
    elif abs(d) < 0.8:
        d_interpretation = "中等（medium）"
    else:
        d_interpretation = "大（large）"

    # 显著性判断
    significance = "拒绝 H0（差异显著）" if p_value < 0.05 else "无法拒绝 H0（差异不显著）"
    consistency = "一致" if (t_ci_low > 0) == (boot_ci_low > 0) else "不一致"

    # 生成报告
    report = f"""

## 不确定性量化

> 本章使用置信区间、Bootstrap 和置换检验量化结论的不确定性。
> 生成时间：2026-02-12

### H2：{group1_name} vs {group2_name} 的差异（区间估计）

**频率学派置信区间（t 公式）**：
- 均值差：{mean_diff:.2f} {unit}
- 95% CI：[{t_ci_low:.2f}, {t_ci_high:.2f}]
- 解释：如果我们重复抽样 100 次，约 95 个区间会覆盖真值

**Bootstrap 置信区间（重采样）**：
- 均值差：{mean_diff:.2f} {unit}
- 95% Bootstrap CI：[{boot_ci_low:.2f}, {boot_ci_high:.2f}]
- Bootstrap 次数：10000 次重采样
- 解释：Bootstrap CI 不依赖正态假设，是稳健的区间估计

**置换检验（分布无关）**：
- 观测均值差：{observed_diff:.2f} {unit}
- 置换 p 值：{p_value:.6f}
- 置换次数：10000 次随机打乱
- 置换 95% CI：[{perm_ci_low:.2f}, {perm_ci_high:.2f}]
- 结论：{significance}

**效应量**：
- Cohen's d：{d:.3f}（{d_interpretation}）
- 解释：两组差异相当于 {d:.3f} 个标准差

**结果一致性**：
- t 公式 CI、Bootstrap CI、置换检验结果 {consistency}
- 稳健性结论：{"结论稳健，可信" if consistency == "一致" else "结论不一致，需进一步检查假设"}

### 方法选择：为什么用 Bootstrap 和置换检验？

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **t 公式 CI** | 有理论支持，计算快 | 假设正态分布 | 样本量大、正态性满足 |
| **Bootstrap CI** | 不依赖分布假设，通用 | 计算成本高（需重采样） | 任何场景，尤其是分布未知 |
| **置换检验** | 分布无关，稳健 | 计算成本高 | 数据不满足正态假设 |

本周选择 Bootstrap 和置换检验，原因：
1. 数据分布可能偏离正态（需检查正态性）
2. Bootstrap 和置换检验是稳健的替代方案
3. 重采样方法更直观，可解释性更强

### 频率学派 vs 贝叶斯学派

本周使用频率学派框架（CI 是方法的覆盖率）。

**频率学派**：
- 参数是固定但未知的值
- 95% CI：如果我们重复抽样，约 95% 的区间会覆盖真值
- 不能说"参数有 95% 的概率在区间内"

**贝叶斯学派**（Week 14 将深入）：
- 参数是随机变量，有概率分布
- 95% 可信区间：参数有 95% 的概率落在区间内
- 解释更直观，但需要选择先验

### 下一步

Week 09 将学习回归分析与模型诊断，进一步量化预测的不确定性（预测区间、置信带）。

---

"""

    return report


def generate_uncertainty_visualizations(group1: np.ndarray,
                                        group2: np.ndarray,
                                        output_prefix: str = "statlab",
                                        seed: int = 42) -> dict:
    """
    生成不确定性量化的可视化图片。

    参数：
        group1, group2: 两组数据
        output_prefix: 输出文件名前缀
        seed: 随机种子

    返回：
        filenames: 生成的图片文件名字典
    """
    np.random.seed(seed)
    filenames = {}

    # 1. Bootstrap 分布图
    bootstrap_diffs, ci_low, ci_high = bootstrap_ci_diff(group1, group2)
    observed_diff = np.mean(group1) - np.mean(group2)

    plt.figure(figsize=(10, 6))
    plt.hist(bootstrap_diffs, bins=50, density=True, alpha=0.7, color='steelblue')
    plt.axvline(ci_low, color='red', linestyle='--', linewidth=2, label=f'2.5% ({ci_low:.2f})')
    plt.axvline(ci_high, color='red', linestyle='--', linewidth=2, label=f'97.5% ({ci_high:.2f})')
    plt.axvline(0, color='black', linestyle='-', linewidth=2, label='零线（无差异）')
    plt.axvline(observed_diff, color='green', linestyle='-', linewidth=2,
                label=f'原始均值差 ({observed_diff:.2f})')
    plt.xlabel('均值差')
    plt.ylabel('密度')
    plt.title('Bootstrap 均值差分布（10000 次重采样）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    bootstrap_file = f'{output_prefix}_bootstrap_dist.png'
    plt.savefig(bootstrap_file, dpi=150)
    plt.close()
    filenames['bootstrap'] = bootstrap_file
    print(f"✓ 生成 Bootstrap 分布图：{bootstrap_file}")

    # 2. 置换分布图
    _, perm_stats, p_value = permutation_test(group1, group2)

    plt.figure(figsize=(10, 6))
    plt.hist(perm_stats, bins=50, density=True, alpha=0.7, color='steelblue',
             label='置换分布（H0 为真时）')
    plt.axvline(observed_diff, color='red', linestyle='-', linewidth=3,
                label=f'观测值 ({observed_diff:.2f})')
    plt.axvline(-observed_diff, color='red', linestyle='--', linewidth=2,
                label=f'负观测值 ({-observed_diff:.2f})')
    perm_95_low = np.percentile(perm_stats, 2.5)
    perm_95_high = np.percentile(perm_stats, 97.5)
    plt.axvline(perm_95_low, color='orange', linestyle=':', linewidth=2,
                label=f'2.5% ({perm_95_low:.2f})')
    plt.axvline(perm_95_high, color='orange', linestyle=':', linewidth=2,
                label=f'97.5% ({perm_95_high:.2f})')
    plt.axvspan(perm_stats.min(), perm_95_low, alpha=0.3, color='red',
                label='极端区域（α=0.05）')
    plt.axvspan(perm_95_high, perm_stats.max(), alpha=0.3, color='red')
    plt.xlabel('均值差')
    plt.ylabel('密度')
    plt.title(f'置换检验：均值差的零分布（{len(perm_stats)} 次置换）\np = {p_value:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    permutation_file = f'{output_prefix}_permutation_dist.png'
    plt.savefig(permutation_file, dpi=150)
    plt.close()
    filenames['permutation'] = permutation_file
    print(f"✓ 生成置换分布图：{permutation_file}")

    # 3. CI 对比图
    mean_diff, t_ci_low, t_ci_high = t_ci_diff(group1, group2)

    fig, ax = plt.subplots(figsize=(10, 4))
    methods = ['t 公式\nCI', 'Bootstrap\nCI', '置换检验\nCI']
    ci_lows = [t_ci_low, ci_low, perm_95_low]
    ci_highs = [t_ci_high, ci_high, perm_95_high]

    colors = ['blue', 'green', 'orange']
    for i, (method, low, high, color) in enumerate(zip(methods, ci_lows, ci_highs, colors)):
        ax.plot([low, high], [i, i], color=color, linewidth=3, label=method.replace('\n', ' '))
        ax.plot(mean_diff, i, 'ro', markersize=8)

    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='零线')
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_xlabel('均值差')
    ax.set_title('三种方法的 95% 置信区间对比')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    comparison_file = f'{output_prefix}_ci_comparison.png'
    plt.savefig(comparison_file, dpi=150)
    plt.close()
    filenames['comparison'] = comparison_file
    print(f"✓ 生成 CI 对比图：{comparison_file}")

    return filenames


def main() -> None:
    """主函数：生成 StatLab 不确定性量化报告。"""
    print("=" * 60)
    print("StatLab: 不确定性量化报告生成器")
    print("=" * 60)

    # 模拟数据（实际使用时从文件读取）
    print("\n[1/3] 加载数据...")
    np.random.seed(42)
    new_users = np.random.normal(loc=315, scale=50, size=100)
    old_users = np.random.normal(loc=300, scale=50, size=100)
    print("  使用模拟数据演示（实际使用时请传入真实数据）")

    # 生成报告章节
    print("\n[2/3] 生成不确定性量化章节...")
    uncertainty_section = generate_uncertainty_section(
        new_users, old_users,
        group1_name="新用户",
        group2_name="老用户",
        unit="元"
    )

    # 追加到报告
    report_path = "report.md"
    with open(report_path, 'a', encoding='utf-8') as f:
        f.write(uncertainty_section)
    print(f"  ✓ 不确定性量化章节已追加到 {report_path}")

    # 生成可视化
    print("\n[3/3] 生成可视化图片...")
    filenames = generate_uncertainty_visualizations(
        new_users, old_users,
        output_prefix="statlab_week08"
    )

    # 完成
    print("\n" + "=" * 60)
    print("StatLab 不确定性量化报告生成完成！")
    print("=" * 60)
    print(f"\n报告文件：{report_path}")
    print("可视化图片：")
    for name, path in filenames.items():
        print(f"  - {path}")

    print("\n下一步：")
    print("  1. 检查 report.md 中的不确定性量化章节")
    print("  2. 查看生成的可视化图片")
    print("  3. 根据实际数据调整分析和解释")


if __name__ == "__main__":
    main()
