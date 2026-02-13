"""
示例：PyMC 贝叶斯 A/B 测试——MCMC 采样

运行方式：python3 chapters/week_14/examples/05_pymc_ab_test.py
预期输出：后验摘要、B 比 A 好的概率、trace plot 和 posterior plot

依赖：pip install pymc arviz
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pymc as pm
import arviz as az

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_14"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pymc_ab_test(conversions_A: int, exposures_A: int,
                conversions_B: int, exposures_B: int,
                draws: int = 5000, tune: int = 2000,
                chains: int = 4, target_accept: float = 0.9,
                random_seed: int = 42) -> az.InferenceData:
    """
    使用 PyMC 实现贝叶斯 A/B 测试

    参数:
        conversions_A: A 版本转化数
        exposures_A: A 版本曝光数
        conversions_B: B 版本转化数
        exposures_B: B 版本曝光数
        draws: 采样数量
        tune: 调整/预热步数
        chains: 马尔可夫链数量
        target_accept: 目标接受率（0-1）
        random_seed: 随机种子

    返回:
        ArviZ InferenceData 对象
    """
    # 定义贝叶斯模型
    with pm.Model() as ab_model:
        # 先验：转化率的均匀分布（无信息先验）
        theta_A = pm.Uniform("theta_A", lower=0, upper=1)
        theta_B = pm.Uniform("theta_B", lower=0, upper=1)

        # 似然：Binomial 分布
        obs_A = pm.Binomial("obs_A", n=exposures_A, p=theta_A,
                           observed=conversions_A)
        obs_B = pm.Binomial("obs_B", n=exposures_B, p=theta_B,
                           observed=conversions_B)

        # 感兴趣的量：B 比 A 好多少
        delta = pm.Deterministic("delta", theta_B - theta_A)
        rel_uplift = pm.Deterministic("rel_uplift",
                                    (theta_B - theta_A) / theta_A)

        # MCMC 采样
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True
        )

    return idata


def check_convergence(idata: az.InferenceData,
                    var_names: list[str] = None) -> dict:
    """
    检查 MCMC 收敛性

    参数:
        idata: ArviZ InferenceData 对象
        var_names: 要检查的变量名列表

    返回:
        包含 R-hat 和 ESS 的字典
    """
    if var_names is None:
        var_names = ["theta_A", "theta_B", "delta", "rel_uplift"]

    print("=" * 60)
    print("MCMC 收敛诊断")
    print("=" * 60)

    # 计算 R-hat
    rhat = az.rhat(idata, var_names=var_names)
    print("\n【R-hat】（应该 < 1.01）")
    print(rhat)

    # 计算 ESS（有效样本量）
    ess = az.ess(idata, var_names=var_names)
    print("\n【ESS】（有效样本量，应该 > 400）")
    print(ess)

    # 检查是否满足收敛标准
    rhat_max = max(rhat[var].values for var in var_names)
    ess_min = min(ess[var].values for var in var_names)

    converged = rhat_max < 1.01 and ess_min > 400

    print(f"\n【收敛判断】")
    print(f"  R-hat 最大值: {rhat_max:.4f} {'✅' if rhat_max < 1.01 else '❌'}")
    print(f"  ESS 最小值: {ess_min:.0f} {'✅' if ess_min > 400 else '❌'}")
    print(f"  整体判断: {'收敛 ✅' if converged else '未收敛 ❌'}")

    return {'rhat': rhat, 'ess': ess, 'converged': converged}


def compute_prob_B_better(idata: az.InferenceData) -> dict:
    """
    计算 B 比 A 好的概率

    参数:
        idata: ArviZ InferenceData 对象

    返回:
        包含概率和统计量的字典
    """
    # 提取 delta 样本
    delta_samples = idata.posterior["delta"].values.flatten()
    rel_uplift_samples = idata.posterior["rel_uplift"].values.flatten()

    # B 比 A 好的概率
    prob_B_better = (delta_samples > 0).mean()

    # 相对提升的中位数和区间
    median_uplift = np.median(rel_uplift_samples)
    ci_low, ci_high = np.percentile(rel_uplift_samples, [2.5, 97.5])

    return {
        'prob_B_better': prob_B_better,
        'median_uplift': median_uplift,
        'ci_low': ci_low,
        'ci_high': ci_high
    }


# 坏例子：不检查收敛就下结论
def bad_ignore_convergence(idata: az.InferenceData) -> str:
    """
    反例：不检查收敛就使用后验结果

    问题：
    1. R-hat > 1.05 说明链未混合，结果不可信
    2. ESS 太小说明采样效率低，方差估计不准
    3. 未收敛的 MCMC 可能给出完全错误的结论
    """
    delta_samples = idata.posterior["delta"].values.flatten()
    prob_B_better = (delta_samples > 0).mean()

    return f"B 比 A 好的概率: {prob_B_better:.1%}"


def main() -> None:
    # 示例数据
    conversions_A, exposures_A = 52, 1000
    conversions_B, exposures_B = 58, 1000

    print("=" * 60)
    print("PyMC 贝叶斯 A/B 测试")
    print("=" * 60)
    print(f"\n数据：A={conversions_A}/{exposures_A}, B={conversions_B}/{exposures_B}")
    print()

    # 运行 MCMC
    print("运行 MCMC 采样...")
    idata = pymc_ab_test(conversions_A, exposures_A,
                        conversions_B, exposures_B,
                        draws=5000, tune=2000, chains=4)

    # 后验摘要
    print("\n" + "=" * 60)
    print("后验摘要")
    print("=" * 60)
    summary = az.summary(idata, var_names=["theta_A", "theta_B", "delta", "rel_uplift"])
    print(summary)

    # 检查收敛
    convergence = check_convergence(idata)

    # 计算概率
    if convergence['converged']:
        print("\n" + "=" * 60)
        print("决策相关量")
        print("=" * 60)

        prob_stats = compute_prob_B_better(idata)
        print(f"\nB 比 A 好的概率: {prob_stats['prob_B_better']:.1%}")
        print(f"相对提升中位数: {prob_stats['median_uplift']:.2%}")
        print(f"95% 可信区间: [{prob_stats['ci_low']:.2%}, {prob_stats['ci_high']:.2%}]")

        # 决策建议
        print("\n" + "=" * 60)
        print("决策建议")
        print("=" * 60)
        if prob_stats['prob_B_better'] >= 0.90:
            print(f"✅ 推荐 B：有 {prob_stats['prob_B_better']:.1%} 的把握")
        elif prob_stats['prob_B_better'] >= 0.70:
            print(f"⚠️  倾向 B：有 {prob_stats['prob_B_better']:.1%} 的把握，建议继续收集数据")
        else:
            print(f"❌ 不推荐 B：只有 {prob_stats['prob_B_better']:.1%} 的把握")

    # 绘制 trace plot 和 posterior plot
    print("\n" + "=" * 60)
    print("生成诊断图...")
    print("=" * 60)

    # Trace plot
    az.plot_trace(idata, var_names=["theta_A", "theta_B"])
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'trace_plot.png', dpi=150)
    print("✅ Trace plot 已保存: trace_plot.png")

    # Posterior plot
    az.plot_posterior(idata, var_names=["delta", "rel_uplift"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'posterior_plot.png', dpi=150)
    print("✅ Posterior plot 已保存: posterior_plot.png")

    # 对比：坏例子
    print("\n" + "=" * 60)
    print("【反例】不检查收敛就下结论")
    print("=" * 60)
    bad_result = bad_ignore_convergence(idata)
    print(f"❌ 错误做法：{bad_result}")
    print("   问题：如果 MCMC 未收敛，这个概率可能完全错误！")
    print("   正确做法：必须先检查 R-hat < 1.01 和 ESS > 400")


if __name__ == "__main__":
    main()
