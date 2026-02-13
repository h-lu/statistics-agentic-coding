#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
常见错误示例：Week 05 概率与统计

本模块展示新手在学习概率与统计时常犯的错误，
每个错误配有"正确做法"对比。

运行方式：python3 chapters/week_05/examples/05_common_mistakes.py
预期输出：错误 vs 正确的对比说明
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_05"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 错误 1：混淆条件概率的方向 P(A|B) vs P(B|A)
# =============================================================================

def mistake_1_conditional_direction():
    """
    错误：混淆 P(患病|阳性) 和 P(阳性|患病)

    场景：医疗检测假阳性问题
    """
    print("\n" + "=" * 70)
    print("错误 1：混淆条件概率的方向")
    print("=" * 70)

    # 正确的贝叶斯计算
    prevalence = 0.01        # P(患病) = 1%
    sensitivity = 0.99       # P(阳性|患病) = 99%
    specificity = 0.99       # P(阴性|健康) = 99%

    # ❌ 错误直觉：检测准确率 99% = 阳性后患病概率 99%
    wrong_answer = 0.99

    # ✅ 正确计算：贝叶斯定理
    p_positive_given_sick = sensitivity
    p_positive_given_healthy = 1 - specificity
    p_sick = prevalence
    p_healthy = 1 - prevalence

    p_positive = (p_positive_given_sick * p_sick +
                  p_positive_given_healthy * p_healthy)

    # P(患病|阳性) = P(阳性|患病) × P(患病) / P(阳性)
    correct_answer = (p_positive_given_sick * p_sick) / p_positive

    print("\n场景：某疾病发病率 1%，检测准确率 99%（敏感性=特异性=99%）")
    print(f"\n❌ 错误理解：")
    print(f"   '检测准确率 99%，所以检测阳性 = 99% 患病'")
    print(f"   P(患病|阳性) ≈ {wrong_answer:.1%}  <- 这是错的！")

    print(f"\n✅ 正确理解：")
    print(f"   P(阳性|患病) = {sensitivity:.1%}  <- 这是检测准确率")
    print(f"   P(患病|阳性) = {correct_answer:.1%}  <- 这才是阳性后患病的概率！")
    print(f"   差异：{abs(wrong_answer - correct_answer):.1%}")

    print("\n小北的困惑：")
    print("  '但是医生说检测准确率 99% 啊？'")
    print("\n老潘的解释：")
    print("  '准确率指的是 P(阳性|患病) 和 P(阴性|健康)。'")
    print("  '但你要问的是 P(患病|阳性)，这是反过来的条件概率。'")
    print("  '在发病率很低时，假阳性会淹没真阳性，导致 P(患病|阳性) 远低于 99%。'")


# =============================================================================
# 错误 2：滥用正态分布假设
# =============================================================================

def mistake_2_normal_assumption_abuse():
    """
    错误：对所有数据都假设正态分布

    场景：右偏的收入数据
    """
    print("\n" + "=" * 70)
    print("错误 2：滥用正态分布假设")
    print("=" * 70)

    # 生成右偏数据（收入分布）
    np.random.seed(42)
    skewed_data = np.random.lognormal(8.5, 0.5, 1000)

    mean = np.mean(skewed_data)
    std = np.std(skewed_data, ddof=1)

    print("\n场景：分析用户收入数据（典型的右偏分布）")
    print(f"  均值 = {mean:.0f} 元")
    print(f"  标准差 = {std:.0f} 元")

    print("\n❌ 错误做法：用正态分布 3σ 原则判断异常值")
    print(f"  '均值 ± 3×标准差 = [{mean - 3*std:.0f}, {mean + 3*std:.0f}]'")
    print(f"  '超过 {mean + 3*std:.0f} 的都是异常值'")
    print("  问题：右偏数据的正态假设不成立，会误判大量正常数据为异常")

    print("\n✅ 正确做法：使用分位数方法（IQR）")
    q1 = np.percentile(skewed_data, 25)
    q3 = np.percentile(skewed_data, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr

    print(f"  Q1 = {q1:.0f} 元")
    print(f"  Q3 = {q3:.0f} 元")
    print(f"  IQR = {iqr:.0f} 元")
    print(f"  上界 = Q3 + 1.5×IQR = {upper_bound:.0f} 元")
    print("  优势：不依赖分布假设，对偏态数据更稳健")

    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：数据分布
    axes[0].hist(skewed_data, bins=50, density=True, alpha=0.7, color='steelblue')
    axes[0].axvline(mean, color='r', linestyle='--', label=f'均值={mean:.0f}')
    axes[0].axvline(mean + 3*std, color='orange', linestyle='--',
                   label=f'均值+3σ={mean + 3*std:.0f}')
    axes[0].set_xlabel('收入（元）')
    axes[0].set_ylabel('密度')
    axes[0].set_title('错误：用正态 3σ 判断异常值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图：IQR 方法
    axes[1].hist(skewed_data, bins=50, density=True, alpha=0.7, color='green')
    axes[1].axvline(q1, color='orange', linestyle='--', label=f'Q1={q1:.0f}')
    axes[1].axvline(q3, color='orange', linestyle='--', label=f'Q3={q3:.0f}')
    axes[1].axvline(upper_bound, color='r', linestyle='--',
                   label=f'上界={upper_bound:.0f}')
    axes[1].set_xlabel('收入（元）')
    axes[1].set_ylabel('密度')
    axes[1].set_title('正确：用 IQR 判断异常值')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'mistake_2_normal_assumption.png', dpi=150)
    print(f"\n[图表已保存] /tmp/mistake_2_normal_assumption.png")
    plt.close()


# =============================================================================
# 错误 3：Bootstrap 时忘记有放回抽样
# =============================================================================

def mistake_3_bootstrap_without_replacement():
    """
    错误：Bootstrap 时忘记有放回抽样

    场景：估计样本均值的抽样分布
    """
    print("\n" + "=" * 70)
    print("错误 3：Bootstrap 时忘记有放回抽样")
    print("=" * 70)

    np.random.seed(42)
    original_sample = np.random.normal(100, 15, 50)
    n_bootstrap = 10000

    # ❌ 错误做法：无放回抽样
    # 注意：当 n_bootstrap > sample_size 时会报错！
    # 无放回抽样只是在"排列"原始数据，不能产生新的变异
    boot_without_replace = []
    for _ in range(min(n_bootstrap, 1000)):  # 限制次数避免重复
        resample = np.random.choice(original_sample, size=len(original_sample), replace=False)
        boot_without_replace.append(np.mean(resample))

    # ✅ 正确做法：有放回抽样
    boot_with_replace = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(original_sample, size=len(original_sample), replace=True)
        boot_with_replace.append(np.mean(resample))

    print("\n场景：用 Bootstrap 估计样本均值的抽样分布")
    print(f"  原始样本量：{len(original_sample)}")
    print(f"  Bootstrap 重复次数：{n_bootstrap}")

    print("\n❌ 错误做法：无放回抽样 (replace=False)")
    print(f"  Bootstrap 分布的标准差：{np.std(boot_without_replace):.4f}")
    print("  问题：")
    print("    1. 无放回抽样只是在'排列'原始数据，不产生新变异")
    print("    2. 当重复次数 > 样本量时，会穷尽所有排列")
    print("    3. 低估了真实的抽样变异性！")

    print("\n✅ 正确做法：有放回抽样 (replace=True)")
    print(f"  Bootstrap 分布的标准差：{np.std(boot_with_replace):.4f}")
    print("  优势：")
    print("    1. 每次重采样都是独立的，产生真实的变异")
    print("    2. 可以进行任意次数的 Bootstrap")
    print("    3. 正确估计了抽样分布的标准误")

    print("\n阿码的疑问：")
    print("  '为什么要放回抽样？这不符合直觉。'")
    print("\n老潘的解释：")
    print("  'Bootstrap 的思想是：把你的样本当作'虚拟总体'。'")
    print("  '每次从这个虚拟总体中抽取一个同样大小的样本，就要放回。'")
    print("  '这样才能模拟'从总体中重复抽样'的过程。'")


# =============================================================================
# 错误 4：混淆标准差和标准误
# =============================================================================

def mistake_4_sd_vs_se():
    """
    错误：混淆标准差（SD）和标准误（SE）

    场景：报告统计结果时
    """
    print("\n" + "=" * 70)
    print("错误 4：混淆标准差和标准误")
    print("=" * 70)

    # 模拟：总体 + 样本
    np.random.seed(42)
    population = np.random.normal(100, 15, 100000)
    sample = np.random.choice(population, 100, replace=False)

    # 计算
    sample_mean = np.mean(sample)
    sample_sd = np.std(sample, ddof=1)  # 标准差
    sample_se = sample_sd / np.sqrt(len(sample))  # 标准误

    print("\n场景：报告用户平均消费及其不确定性")
    print(f"  样本量：n = {len(sample)}")
    print(f"  样本均值：{sample_mean:.1f} 元")

    print("\n❌ 错误报告：使用标准差作为均值的不确定性")
    print(f"  '用户平均消费：{sample_mean:.1f} ± {sample_sd:.1f} 元'")
    print("  问题：")
    print("    - 标准差描述的是'数据本身的分散程度'")
    print(f"    - 大约 68% 的数据落在均值 ± {sample_sd:.1f} 范围内")
    print("    - 但这不是'均值估计的不确定性'！")

    print("\n✅ 正确报告：使用标准误")
    print(f"  '用户平均消费：{sample_mean:.1f} ± {sample_se:.1f} 元 (SE)'")
    print("  说明：")
    print("    - 标准误描述的是'均值估计的精确度'")
    print(f"    - 如果重复抽样，约 68% 的样本均值会落在 {sample_mean:.1f} ± {sample_se:.1f}")
    print("    - SE = SD / √n，样本量越大，SE 越小")

    print("\n公式对比：")
    print("  标准差（SD）：σ = √[Σ(xi - x̄)² / (n-1)]")
    print("              → 描述数据有多'散'")
    print()
    print("  标准误（SE）：SE = σ / √n")
    print("              → 描述均值估计有多'稳'")

    print("\n小北的疑问：")
    print("  '那报告里到底应该写哪一个？'")
    print("\n老潘的回答：")
    print("  '都要写，但用在不同的地方。'")
    print("  '描述数据特征时，报告 SD（说明数据分布）。'")
    print("  '报告统计推断结果时，报告 SE（说明估计精度）。'")
    print("  '或者干脆报告 95% 置信区间：x̄ ± 1.96×SE'")


# =============================================================================
# 错误 5：小样本时滥用 CLT
# =============================================================================

def mistake_5_clt_small_sample():
    """
    错误：小样本时滥用中心极限定理

    场景：对严重偏态数据的小样本均值做正态假设
    """
    print("\n" + "=" * 70)
    print("错误 5：小样本时滥用中心极限定理")
    print("=" * 70)

    np.random.seed(42)

    # 严重偏态的总体
    population = np.random.exponential(scale=10, size=100000)

    # 小样本
    sample_sizes = [5, 10, 30]
    n_simulations = 5000

    print("\n场景：从严重偏态的指数分布总体抽样")
    print(f"  总体分布：指数分布（严重右偏）")
    print(f"  模拟次数：{n_simulations}")

    for n in sample_sizes:
        # 生成样本均值分布
        sample_means = [
            np.mean(np.random.choice(population, n, replace=False))
            for _ in range(n_simulations)
        ]

        # 正态性检验（Shapiro-Wilk）
        _, p_value = stats.shapiro(sample_means)

        print(f"\n样本量 n = {n}:")
        print(f"  正态性检验 p 值：{p_value:.4f}")

        if p_value > 0.05:
            print(f"  ✓ 样本均值分布接近正态（p > 0.05）")
            print(f"    → 可以使用 CLT 的正态近似")
        else:
            print(f"  ✗ 样本均值分布偏离正态（p < 0.05）")
            print(f"    → 不应使用 CLT 的正态近似！")
            print(f"    → 考虑使用 Bootstrap 或增加样本量")

    print("\n阿码的追问：")
    print("  '那到底样本量多大才算'够大'？'")
    print("\n老潘的解释：")
    print("  '取决于总体分布的偏度。'")
    print("  '如果总体接近正态，n=30 可能就够了。'")
    print("  '如果总体严重偏态，可能需要 n=100 或更多。'")
    print("  '最保险的做法：做正态性检验，或者直接用 Bootstrap。'")


# =============================================================================
# 主函数
# =============================================================================

def main() -> None:
    """主函数：运行所有常见错误示例"""
    print("=" * 70)
    print("Week 05 常见错误示例")
    print("=" * 70)
    print("\n本模块展示 5 个新手常犯的错误：")
    print("1. 混淆条件概率的方向 P(A|B) vs P(B|A)")
    print("2. 滥用正态分布假设")
    print("3. Bootstrap 时忘记有放回抽样")
    print("4. 混淆标准差和标准误")
    print("5. 小样本时滥用 CLT")

    mistake_1_conditional_direction()
    mistake_2_normal_assumption_abuse()
    mistake_3_bootstrap_without_replacement()
    mistake_4_sd_vs_se()
    mistake_5_clt_small_sample()

    print("\n" + "=" * 70)
    print("总结：避免这些错误的关键")
    print("=" * 70)
    print("1. 条件概率：明确'在什么条件下'，不要想当然地反向")
    print("2. 分布假设：先看数据形状（直方图/QQ图），再选择分布")
    print("3. Bootstrap：必须 replace=True，否则只是排列组合")
    print("4. SD vs SE：SD 描述数据，SE 描述统计量")
    print("5. CLT：样本量够不够大，要做正态性检验")
    print("=" * 70)


if __name__ == "__main__":
    main()
