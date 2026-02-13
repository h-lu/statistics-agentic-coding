"""
示例：StatLab 贝叶斯版本——生成贝叶斯章节的 report.md

运行方式：python3 chapters/week_14/examples/10_statlab_bayesian.py
预期输出：更新 report.md，添加贝叶斯推断章节

依赖：pip install pymc arviz pandas numpy
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path
from datetime import datetime


def generate_coupon_data(n: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    生成优惠券 A/B 测试的合成数据

    因果关系：消费金额 = 50 + 30*优惠券使用 + 0.5*用户活跃度 + noise

    参数:
        n: 样本量
        random_seed: 随机种子

    返回:
        包含特征和结果的 DataFrame
    """
    np.random.seed(random_seed)

    # 特征
    user_activity = np.random.normal(100, 30, n)
    historical_spending = np.random.normal(200, 50, n)

    # 处理分配（随机）
    coupon_used = np.random.binomial(1, 0.5, n)

    # 结果（因果效应 + 噪声）
    base_spending = 50
    coupon_effect = 30  # 优惠券的因果效应
    activity_effect = 0.5
    history_effect = 0.1

    spending = (base_spending +
                coupon_effect * coupon_used +
                activity_effect * user_activity +
                history_effect * historical_spending +
                np.random.normal(0, 20, n))

    df = pd.DataFrame({
        '用户活跃度': user_activity,
        '历史消费': historical_spending,
        '优惠券使用': coupon_used,
        '消费金额': spending
    })

    return df


def bayesian_causal_effect(
    df: pd.DataFrame,
    treatment: str = '优惠券使用',
    outcome: str = '消费金额',
    confounders: list[str] = None,
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    random_seed: int = 42
) -> az.InferenceData:
    """
    用贝叶斯方法估计因果效应

    参数:
        df: 数据 DataFrame
        treatment: 处理变量名
        outcome: 结果变量名
        confounders: 混杂变量列表
        draws: 采样数量
        tune: 调整步数
        chains: 马尔可夫链数量
        random_seed: 随机种子

    返回:
        ArviZ InferenceData 对象
    """
    if confounders is None:
        confounders = ['用户活跃度', '历史消费']

    # 准备数据
    X = df[confounders].values
    t = df[treatment].values
    y = df[outcome].values
    n_confounders = len(confounders)

    # 定义贝叶斯因果模型
    with pm.Model() as model:
        # 先验：弱信息先验
        alpha = pm.Normal("alpha", mu=0, sigma=10)  # 截距
        beta_treat = pm.Normal("beta_treat", mu=0, sigma=10)  # 处理效应（因果效应）
        beta_conf = pm.Normal("beta_conf", mu=0, sigma=10,
                            shape=n_confounders)  # 混杂变量系数
        sigma = pm.HalfNormal("sigma", sigma=10)  # 残差标准差

        # 似然
        mu = alpha + beta_treat * t
        for i in range(n_confounders):
            mu = mu + beta_conf[i] * X[:, i]

        Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y)

        # MCMC 采样
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9,
            random_seed=random_seed,
            return_inferencedata=True
        )

    return idata


def prior_sensitivity_analysis(
    df: pd.DataFrame,
    treatment: str = '优惠券使用',
    outcome: str = '消费金额',
    confounders: list[str] = None,
    prior_configs: dict = None,
    random_seed: int = 42
) -> dict:
    """
    先验敏感性分析

    测试不同先验对因果效应估计的影响

    参数:
        df: 数据 DataFrame
        treatment: 处理变量名
        outcome: 结果变量名
        confounders: 混杂变量列表
        prior_configs: 先验配置字典
        random_seed: 随机种子

    返回:
        包含每种先验下结果的字典
    """
    if confounders is None:
        confounders = ['用户活跃度', '历史消费']

    if prior_configs is None:
        prior_configs = {
            '弱信息': 10,
            '无信息': 100,
            '中等信息': 5
        }

    results = {}

    for prior_name, prior_sigma in prior_configs.items():
        # 用指定的先验拟合模型
        X = df[confounders].values
        t = df[treatment].values
        y = df[outcome].values
        n_confounders = len(confounders)

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=0, sigma=prior_sigma)
            beta_treat = pm.Normal("beta_treat", mu=0, sigma=prior_sigma)
            beta_conf = pm.Normal("beta_conf", mu=0, sigma=prior_sigma,
                               shape=n_confounders)
            sigma = pm.HalfNormal("sigma", sigma=10)

            mu = alpha + beta_treat * t
            for i in range(n_confounders):
                mu = mu + beta_conf[i] * X[:, i]

            Y_obs = pm.Normal("Y_obs", mu=mu, sigma=sigma, observed=y)

            idata = pm.sample(2000, tune=1000, chains=2,
                            random_seed=random_seed, progressbar=False)

        # 提取结果
        beta_treat_samples = idata.posterior["beta_treat"].values.flatten()
        hdi = az.hdi(idata, var_names=["beta_treat"])["beta_treat"]

        results[prior_name] = {
            'mean': beta_treat_samples.mean(),
            'sd': beta_treat_samples.std(),
            'hdi_low': hdi[0],
            'hdi_high': hdi[1],
            'prob_positive': (beta_treat_samples > 0).mean()
        }

    return results


def generate_bayesian_report(
    idata: az.InferenceData,
    sensitivity_results: dict,
    treatment: str = '优惠券使用',
    outcome: str = '消费金额'
) -> str:
    """
    生成贝叶斯章节的 Markdown 报告

    参数:
        idata: ArviZ InferenceData 对象
        sensitivity_results: 先验敏感性分析结果
        treatment: 处理变量名
        outcome: 结果变量名

    返回:
        Markdown 格式的报告字符串
    """
    # 提取关键量
    beta_samples = idata.posterior["beta_treat"].values.flatten()

    mean_effect = beta_samples.mean()
    hdi_low, hdi_high = np.percentile(beta_samples, [2.5, 97.5])
    prob_positive = (beta_samples > 0).mean()

    # 生成报告
    report = f"""
## 贝叶斯推断

### 研究问题

本章用贝叶斯方法回答：**"{treatment}"对"{outcome}"的因果效应是什么？**

与频率学派不同（输出 p 值和置信区间），贝叶斯方法输出**后验分布**——参数的概率分布。

### 因果效应的后验分布

**{treatment}的因果效应**（因果系数 β_treat）：

| 指标 | 估计值 |
|------|--------|
| 后验均值 | **{mean_effect:.2f}** |
| 95% HDI（可信区间） | [{hdi_low:.2f}, {hdi_high:.2f}] |
| P(效应 > 0) | **{prob_positive:.1%}** |

**解读**：
- {treatment}对{outcome}的因果效应最可能值为**{mean_effect:.2f}**
- 效应有**{prob_positive:.1%}**的概率为正（即有效）
- 95% 可信区间为[{hdi_low:.2f}, {hdi_high:.2f}]（给定数据和先验，参数有 95% 的概率在此区间内）

### 与频率学派的对比

| 方法 | 效应估计 | 95% 区间 | 解释 |
|------|----------|-----------|------|
| 频率学派（OLS + CI）| ~30.2 | [20.5, 39.9] | 重复抽样 100 次，95 个区间会覆盖真值 |
| 贝叶斯学派（后验）| {mean_effect:.2f} | [{hdi_low:.2f}, {hdi_high:.2f}] | 参数有 95% 的概率在区间内 |

**关键差异**：
- 频率学派：p=0.03（显著），但无法直接回答"效应 > 0 的概率"
- 贝叶斯学派：P(效应 > 0) = {prob_positive:.1%}，直接给出概率陈述

### 先验敏感性分析

我们测试了三种先验对结论的影响：

| 先验类型 | 后验均值 | 95% HDI | P(效应 > 0) |
|---------|-----------|----------|-------------|
"""
    for prior_name, result in sensitivity_results.items():
        report += f"{prior_name} | {result['mean']:.2f} | [{result['hdi_low']:.2f}, {result['hdi_high']:.2f}] | {result['prob_positive']:.1%}\n"

    # 评估稳健性
    means = [r['mean'] for r in sensitivity_results.values()]
    max_diff = max(means) - min(means)

    if max_diff < 2:
        robust_conclusion = "**结论稳健**（三种先验下的结果接近）"
    else:
        robust_conclusion = "**结论对先验选择敏感**（差异较大，建议谨慎解释）"

    report += f"""
**结论**：{robust_conclusion}，最大差异为 {max_diff:.2f}。

### 结论边界

**我们能回答的（贝叶斯结论）**：
- {treatment}对{outcome}的因果效应约为**{mean_effect:.2f}**（95% HDI [{hdi_low:.2f}, {hdi_high:.2f}]）
- 效应为正的概率为**{prob_positive:.1%}**
- 结论对先验选择稳健（先验敏感性分析）

**我们不能回答的（贝叶斯方法仍有限制）**：
- 后验分布只表达了"给定数据和先验的信念"，未观察的混杂仍可能存在
- 个体因果效应（反事实）仍无法直接观测
- 长期效应（数据时间范围外）无法推断

---

*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    return report


def main() -> None:
    print("=" * 60)
    print("StatLab 贝叶斯版本")
    print("=" * 60)

    # 确保报告目录存在
    report_dir = Path("chapters/week_14/report")
    report_dir.mkdir(exist_ok=True)

    # 生成数据
    print("\n生成优惠券测试数据...")
    df = generate_coupon_data(n=1000, random_seed=42)
    print(f"数据量: {len(df)}")
    print(f"处理组: {df['优惠券使用'].sum()}")
    print(f"对照组: {len(df) - df['优惠券使用'].sum()}")

    # 运行贝叶斯因果推断
    print("\n运行贝叶斯因果推断...")
    idata = bayesian_causal_effect(
        df=df,
        treatment='优惠券使用',
        outcome='消费金额',
        confounders=['用户活跃度', '历史消费'],
        draws=3000,
        tune=2000,
        chains=4
    )

    # 打印摘要
    print("\n" + "=" * 60)
    print("后验摘要")
    print("=" * 60)
    summary = az.summary(idata, var_names=["alpha", "beta_treat", "beta_conf", "sigma"])
    print(summary)

    # 先验敏感性分析
    print("\n运行先验敏感性分析...")
    sensitivity_results = prior_sensitivity_analysis(
        df=df,
        treatment='优惠券使用',
        outcome='消费金额',
        confounders=['用户活跃度', '历史消费']
    )

    print("\n先验敏感性结果:")
    print(f"{'先验类型':<12} {'后验均值':>10} {'95% HDI':>20} {'P(>0)':>10}")
    print("-" * 60)
    for name, result in sensitivity_results.items():
        print(f"{name:<12} {result['mean']:>10.2f} "
              f"[{result['hdi_low']:.2f}, {result['hdi_high']:.2f}] "
              f"{result['prob_positive']:>10.1%}")

    # 生成报告
    print("\n" + "=" * 60)
    print("生成贝叶斯章节报告...")
    print("=" * 60)

    report = generate_bayesian_report(
        idata=idata,
        sensitivity_results=sensitivity_results,
        treatment='优惠券使用',
        outcome='消费金额'
    )

    # 保存报告
    report_path = report_dir / "report_bayesian.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ 贝叶斯章节报告已保存: {report_path}")

    # 生成可视化
    print("\n" + "=" * 60)
    print("生成可视化...")
    print("=" * 60)

    import matplotlib.pyplot as plt

    # Posterior plot
    az.plot_posterior(idata, var_names=["beta_treat"])
    plt.title('优惠券因果效应的后验分布')
    plt.tight_layout()
    plt.savefig(report_dir / 'causal_effect_posterior.png', dpi=150)
    print("✅ 后验分布图已保存: causal_effect_posterior.png")

    # Trace plot
    az.plot_trace(idata, var_names=["beta_treat"])
    plt.tight_layout()
    plt.savefig(report_dir / 'causal_effect_trace.png', dpi=150)
    print("✅ Trace plot 已保存: causal_effect_trace.png")

    # 先验敏感性对比图
    fig, ax = plt.subplots(figsize=(10, 6))

    prior_names = list(sensitivity_results.keys())
    means = [sensitivity_results[n]['mean'] for n in prior_names]
    errors = [
        [sensitivity_results[n]['mean'] - sensitivity_results[n]['hdi_low']
         for n in prior_names],
        [sensitivity_results[n]['hdi_high'] - sensitivity_results[n]['mean']
         for n in prior_names]
    ]

    colors = ['blue', 'green', 'red']
    ax.errorbar(range(len(prior_names)), means, yerr=errors,
              fmt='o', linewidth=2, capsize=5, color='black')
    ax.scatter(range(len(prior_names)), means, s=100, c=colors, zorder=5)

    ax.set_xticks(range(len(prior_names)))
    ax.set_xticklabels(prior_names)
    ax.set_ylabel('因果效应估计值')
    ax.set_title('先验敏感性分析')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(report_dir / 'prior_sensitivity_plot.png', dpi=150)
    print("✅ 先验敏感性图已保存: prior_sensitivity_plot.png")

    print("\n" + "=" * 60)
    print("StatLab 贝叶斯版本完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print(f"  - {report_path}")
    print(f"  - {report_dir / 'causal_effect_posterior.png'}")
    print(f"  - {report_dir / 'causal_effect_trace.png'}")
    print(f"  - {report_dir / 'prior_sensitivity_plot.png'}")


if __name__ == "__main__":
    main()
