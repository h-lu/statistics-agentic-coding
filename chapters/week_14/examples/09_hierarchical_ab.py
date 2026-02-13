"""
示例：层次模型 A/B 测试——多组信息共享

运行方式：python3 chapters/week_14/examples/09_hierarchical_ab.py
预期输出：层次模型的后验分布、shrinkage 效应、与非层次模型对比

依赖：pip install pymc arviz numpy
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pymc as pm
import arviz as az

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_14"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_hierarchical_data(
    countries: list[str],
    n_per_country: list[int],
    base_rate: float = 0.05,
    rate_std: float = 0.01,
    random_seed: int = 42
) -> dict:
    """
    生成层次结构的 A/B 测试数据

    参数:
        countries: 国家列表
        n_per_country: 每个国家的样本量列表
        base_rate: 基础转化率
        rate_std: 国家间转化率标准差
        random_seed: 随机种子

    返回:
        包含转化数、曝光数等数据的字典
    """
    np.random.seed(random_seed)

    n_countries = len(countries)

    # 每个国家的真实转化率（从全局分布中采样）
    true_rates = np.random.normal(base_rate, rate_std, n_countries)
    true_rates = np.clip(true_rates, 0.001, 0.999)  # 限制在合理范围

    # 生成观测数据
    conversions = []
    exposures = []

    for i, n in enumerate(n_per_country):
        conv = np.random.binomial(n, true_rates[i])
        conversions.append(conv)
        exposures.append(n)

    return {
        'countries': countries,
        'conversions': np.array(conversions),
        'exposures': np.array(exposures),
        'true_rates': true_rates,
        'n_countries': n_countries
    }


def hierarchical_ab_test(
    data: dict,
    draws: int = 5000,
    tune: int = 2000,
    chains: int = 4,
    random_seed: int = 42
) -> az.InferenceData:
    """
    层次模型 A/B 测试

    让不同国家的数据"互相借力"

    参数:
        data: generate_hierarchical_data 的返回结果
        draws: 采样数量
        tune: 调整步数
        chains: 马尔可夫链数量
        random_seed: 随机种子

    返回:
        ArviZ InferenceData 对象
    """
    countries = data['countries']
    conversions = data['conversions']
    exposures = data['exposures']
    n_countries = data['n_countries']

    # 定义层次模型
    with pm.Model() as hierarchical_model:
        # 超先验：全局参数
        # 全局平均转化率
        mu = pm.Beta("mu", alpha=1, beta=1)

        # 国家间差异（转换为 Beta 参数）
        # 使用一个更简单的参数化
        sigma = pm.HalfNormal("sigma", sigma=0.05)

        # 每个国家的先验（来自共同分布）
        # 使用 reparameterization
        # theta_i ~ Normal(mu_transformed, sigma)
        # 然后通过 logistic 变换到 [0, 1]

        # 更简单的参数化：直接用 Beta 分布
        # 从 mu 和 sigma 推导 Beta 的 alpha, beta
        # 近似：alpha = mu * kappa, beta = (1-mu) * kappa
        # 其中 kappa 与 sigma 相关

        # 使用更简单的方法：每个国家的 theta 来自同一个 Beta 分布
        # 但这里 Beta 分布的参数本身是随机的

        # 最简层次模型
        # theta_i ~ Beta(mu * scale, (1-mu) * scale)
        # 其中 scale 控制方差

        scale = pm.HalfNormal("scale", sigma=10)

        # 每个国家的转化率
        theta = pm.Beta("theta",
                       alpha=mu * scale + 1,
                       beta=(1 - mu) * scale + 1,
                       shape=n_countries)

        # 似然
        obs = pm.Binomial("obs", n=exposures, p=theta, observed=conversions)

        # 采样
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9,
            random_seed=random_seed,
            return_inferencedata=True
        )

    return idata


def non_hierarchical_ab_test(
    data: dict,
    draws: int = 5000,
    tune: int = 2000,
    chains: int = 4,
    random_seed: int = 42
) -> az.InferenceData:
    """
    非层次模型（每个国家独立分析）

    用于对比层次模型的效果

    参数:
        data: generate_hierarchical_data 的返回结果
        draws: 采样数量
        tune: 调整步数
        chains: 马尔可夫链数量
        random_seed: 随机种子

    返回:
        ArviZ InferenceData 对象
    """
    countries = data['countries']
    conversions = data['conversions']
    exposures = data['exposures']
    n_countries = data['n_countries']

    with pm.Model() as non_hierarchical_model:
        # 每个国家的转化率（独立先验）
        theta = pm.Beta("theta", alpha=1, beta=1, shape=n_countries)

        # 似然
        obs = pm.Binomial("obs", n=exposures, p=theta, observed=conversions)

        # 采样
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9,
            random_seed=random_seed,
            return_inferencedata=True
        )

    return idata


# 坏例子：合并所有数据（忽略组间差异）
def bad_pooled_analysis(data: dict) -> dict:
    """
    反例：合并所有国家的数据

    问题：
    1. 忽略了国家间的差异
    2. 可能误导决策（小国家的信号被淹没）
    3. 无法回答"每个国家的情况如何"
    """
    total_conversions = data['conversions'].sum()
    total_exposures = data['exposures'].sum()

    pooled_rate = total_conversions / total_exposures

    return {
        'pooled_rate': pooled_rate,
        'total_conversions': total_conversions,
        'total_exposures': total_exposures
    }


def analyze_shrinkage(idata_hier: az.InferenceData,
                    data: dict) -> dict:
    """
    分析 shrinkage 效应

    对比观测值和后验均值，看小样本国家的估计是否向全局均值收缩

    参数:
        idata_hier: 层次模型的 InferenceData
        data: 原始数据

    返回:
        包含 shrinkage 分析的字典
    """
    countries = data['countries']
    conversions = data['conversions']
    exposures = data['exposures']

    # 观测转化率
    observed_rates = conversions / exposures

    # 后验均值
    posterior_means = idata_hier.posterior["theta"].mean(dim=["draw", "chain"]).values

    # 全局均值
    global_mean = idata_hier.posterior["mu"].mean().values

    shrinkage = {
        'country': countries,
        'observed_rate': observed_rates,
        'posterior_mean': posterior_means,
        'global_mean': global_mean,
        'sample_size': exposures,
        'shrinkage_amount': observed_rates - posterior_means
    }

    return shrinkage


def print_comparison(
    data: dict,
    idata_hier: az.InferenceData,
    idata_non_hier: az.InferenceData
):
    """
    打印层次模型与非层次模型的对比
    """
    countries = data['countries']

    print("\n" + "=" * 60)
    print("层次模型 vs 非层次模型 vs 合并分析")
    print("=" * 60)

    print(f"\n{'国家':<10} {'样本量':>8} {'观测值':>10} {'层次模型':>12} {'独立模型':>12}")
    print("-" * 60)

    for i, country in enumerate(countries):
        obs_rate = data['conversions'][i] / data['exposures'][i]
        hier_mean = idata_hier.posterior["theta"].mean()[i].values
        non_hier_mean = idata_non_hier.posterior["theta"].mean()[i].values

        print(f"{country:<10} {data['exposures'][i]:>8} {obs_rate:>10.4f} "
              f"{hier_mean:>12.4f} {non_hier_mean:>12.4f}")


def main() -> None:
    print("=" * 60)
    print("层次模型 A/B 测试")
    print("=" * 60)

    # 生成模拟数据
    countries = ["美国", "英国", "德国", "法国"]
    n_per_country = [10000, 10000, 200, 200]  # 大样本 vs 小样本

    data = generate_hierarchical_data(
        countries=countries,
        n_per_country=n_per_country,
        base_rate=0.05,
        rate_std=0.01,
        random_seed=42
    )

    print(f"\n生成的数据:")
    print(f"{'国家':<10} {'样本量':>8} {'转化数':>8} {'观测转化率':>12} {'真实转化率':>12}")
    print("-" * 60)
    for i, country in enumerate(countries):
        obs_rate = data['conversions'][i] / data['exposures'][i]
        print(f"{country:<10} {data['exposures'][i]:>8} {data['conversions'][i]:>8} "
              f"{obs_rate:>12.4f} {data['true_rates'][i]:>12.4f}")

    # 运行层次模型
    print("\n" + "=" * 60)
    print("运行层次模型...")
    print("=" * 60)

    idata_hier = hierarchical_ab_test(data, draws=5000, tune=2000)

    summary_hier = az.summary(idata_hier, var_names=["theta", "mu", "sigma"])
    print("\n层次模型后验摘要:")
    print(summary_hier)

    # 运行非层次模型（对比）
    print("\n" + "=" * 60)
    print("运行非层次模型（每个国家独立）...")
    print("=" * 60)

    idata_non_hier = non_hierarchical_ab_test(data, draws=5000, tune=2000)

    summary_non_hier = az.summary(idata_non_hier, var_names=["theta"])
    print("\n非层次模型后验摘要:")
    print(summary_non_hier)

    # 对比
    print_comparison(data, idata_hier, idata_non_hier)

    # 分析 shrinkage
    print("\n" + "=" * 60)
    print("Shrinkage 效应分析")
    print("=" * 60)

    shrinkage = analyze_shrinkage(idata_hier, data)

    print(f"\n全局均值: {shrinkage['global_mean']:.4f}")
    print(f"\n{'国家':<10} {'样本量':>8} {'观测值':>10} {'后验均值':>12} {'收缩量':>10}")
    print("-" * 60)

    for i in range(len(shrinkage['country'])):
        print(f"{shrinkage['country'][i]:<10} {shrinkage['sample_size'][i]:>8} "
              f"{shrinkage['observed_rate'][i]:>10.4f} {shrinkage['posterior_mean'][i]:>12.4f} "
              f"{shrinkage['shrinkage_amount'][i]:>10.4f}")

    print("\n解读:")
    print("  - 大样本国家（美、英）：收缩量小（数据主导）")
    print("  - 小样本国家（德、法）：收缩量大（向全局均值学习）")

    # 坏例子：合并分析
    print("\n" + "=" * 60)
    print("【反例】合并所有数据")
    print("=" * 60)

    pooled = bad_pooled_analysis(data)
    print(f"\n合并转化率: {pooled['pooled_rate']:.4f}")
    print("问题：")
    print("  1. 无法区分不同国家的表现")
    print("  2. 小国家的特殊性被淹没")
    print("  3. 可能误导决策（某些国家适合 B，某些不适合）")

    # 绘制对比图
    print("\n" + "=" * 60)
    print("生成可视化...")
    print("=" * 60)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图 1：后验分布对比
    ax1 = axes[0]
    x = np.linspace(0.02, 0.08, 200)

    colors = ['blue', 'green', 'red', 'orange']

    for i, country in enumerate(countries):
        # 层次模型的后验
        theta_hier = idata_hier.posterior["theta"].values[:, :, i].flatten()
        from scipy import stats
        kde_hier = stats.gaussian_kde(theta_hier)
        ax1.plot(x, kde_hier(x), color=colors[i], linewidth=2,
                label=f'{country}（层次）')

        # 非层次模型的后验
        theta_non_hier = idata_non_hier.posterior["theta"].values[:, :, i].flatten()
        kde_non_hier = stats.gaussian_kde(theta_non_hier)
        ax1.plot(x, kde_non_hier(x), color=colors[i], linestyle='--',
                alpha=0.5, label=f'{country}（独立）')

    ax1.axvline(shrinkage['global_mean'], color='black', linestyle=':',
               linewidth=2, label='全局均值')
    ax1.set_xlabel('转化率')
    ax1.set_ylabel('概率密度')
    ax1.set_title('层次 vs 独立模型后验分布')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 子图 2：Shrinkage 效应
    ax2 = axes[1]
    ax2.scatter(shrinkage['sample_size'], shrinkage['shrinkage_amount'],
               s=100, alpha=0.6, color='red')

    # 标注国家
    for i, country in enumerate(countries):
        ax2.annotate(country, (shrinkage['sample_size'][i],
                             shrinkage['shrinkage_amount'][i]),
                   fontsize=9)

    ax2.set_xlabel('样本量')
    ax2.set_ylabel('收缩量（观测值 - 后验均值）')
    ax2.set_title('Shrinkage 效应')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'hierarchical_comparison.png', dpi=150)
    print("✅ 对比图已保存: hierarchical_comparison.png")


if __name__ == "__main__":
    main()
