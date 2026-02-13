"""
示例：PyMC 贝叶斯线性回归——从 OLS 到后验分布

运行方式：python3 chapters/week_14/examples/07_pymc_regression.py
预期输出：回归系数的后验分布、95% HDI、trace plot

依赖：pip install pymc arviz numpy pandas
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_14"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(n: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """
    生成合成的线性回归数据

    y = 50 + 1.5 * x1 + 0.3 * x2 + noise

    参数:
        n: 样本量
        random_seed: 随机种子

    返回:
        包含 x1, x2, y 的 DataFrame
    """
    np.random.seed(random_seed)

    x1 = np.random.normal(50, 15, n)
    x2 = np.random.normal(100, 30, n)

    # 真实系数
    true_intercept = 50
    true_beta1 = 1.5
    true_beta2 = 0.3

    # 生成目标变量
    y = (true_intercept + true_beta1 * x1 + true_beta2 * x2 +
         np.random.normal(0, 10, n))

    df = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'y': y
    })

    return df, {'intercept': true_intercept, 'beta1': true_beta1, 'beta2': true_beta2}


def bayesian_linear_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    prior_sigma: float = 10,
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    random_seed: int = 42
) -> az.InferenceData:
    """
    贝叶斯线性回归（PyMC 实现）

    参数:
        df: 数据 DataFrame
        feature_cols: 特征列名列表
        target_col: 目标列名
        prior_sigma: 先验标准差（弱信息先验）
        draws: 采样数量
        tune: 调整步数
        chains: 马尔可夫链数量
        random_seed: 随机种子

    返回:
        ArviZ InferenceData 对象
    """
    # 准备数据
    X = df[feature_cols].values
    y = df[target_col].values
    n_features = len(feature_cols)

    # 定义贝叶斯模型
    with pm.Model() as model:
        # 先验：弱信息正态分布
        intercept = pm.Normal("intercept", mu=0, sigma=prior_sigma)
        betas = pm.Normal("betas", mu=0, sigma=prior_sigma,
                        shape=n_features)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # 似然
        mu = intercept
        for i in range(n_features):
            mu = mu + betas[i] * X[:, i]

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


# 坏例子：只用点估计，忽略不确定性
def bad_point_estimate_only(idata: az.InferenceData, var_name: str = "betas") -> dict:
    """
    反例：只用后验均值做预测

    问题：
    1. 丢失了预测的不确定性信息
    2. 无法评估预测风险
    3. 违背贝叶斯方法的核心优势
    """
    posterior_mean = idata.posterior[var_name].mean().values

    return {'point_estimate': posterior_mean}


def compute_coefficient_probability(idata: az.InferenceData,
                                  feature_idx: int = 0,
                                  threshold: float = 0) -> dict:
    """
    计算系数相关的概率

    参数:
        idata: ArviZ InferenceData 对象
        feature_idx: 特征索引（0=第一个特征）
        threshold: 阈值（默认 0）

    返回:
        包含概率的字典
    """
    beta_samples = idata.posterior["betas"].values[:, :, feature_idx].flatten()

    return {
        'prob_positive': (beta_samples > threshold).mean(),
        'prob_negative': (beta_samples < threshold).mean(),
        'mean': beta_samples.mean(),
        'median': np.median(beta_samples),
        'ci_95': np.percentile(beta_samples, [2.5, 97.5])
    }


def compare_frequentist_vs_bayesian(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str
) -> None:
    """
    对比频率学派（OLS）和贝叶斯学派的结果

    参数:
        df: 数据 DataFrame
        feature_cols: 特征列名列表
        target_col: 目标列名
    """
    from sklearn.linear_model import LinearRegression

    # 频率学派：OLS
    X = df[feature_cols].values
    y = df[target_col].values

    ols = LinearRegression()
    ols.fit(X, y)

    print("\n" + "=" * 60)
    print("频率学派 vs 贝叶斯学派对比")
    print("=" * 60)

    print("\n【频率学派（OLS）】")
    print(f"  截距: {ols.intercept_:.2f}")
    for i, col in enumerate(feature_cols):
        print(f"  {col} 系数: {ols.coef_[i]:.2f}")

    print("\n【贝叶斯学派】")
    # 这里的 idata 需要从外部传入，简化处理
    print("  （见后续输出：后验均值、95% HDI、P(系数>0)）")


def main() -> None:
    print("=" * 60)
    print("PyMC 贝叶斯线性回归")
    print("=" * 60)

    # 生成合成数据
    print("\n生成合成数据...")
    df, true_params = generate_synthetic_data(n=500, random_seed=42)

    print(f"\n真实参数:")
    print(f"  截距: {true_params['intercept']}")
    print(f"  x1 系数: {true_params['beta1']}")
    print(f"  x2 系数: {true_params['beta2']}")

    print(f"\n数据预览:")
    print(df.head())

    # 运行贝叶斯回归
    print("\n" + "=" * 60)
    print("运行贝叶斯回归...")
    print("=" * 60)

    idata = bayesian_linear_regression(
        df=df,
        feature_cols=['x1', 'x2'],
        target_col='y',
        prior_sigma=10,
        draws=3000,
        tune=2000,
        chains=4
    )

    # 后验摘要
    print("\n" + "=" * 60)
    print("后验摘要")
    print("=" * 60)

    summary = az.summary(idata, var_names=["intercept", "betas", "sigma"])
    print(summary)

    # 对比真实值
    print("\n" + "=" * 60)
    print("后验均值 vs 真实值")
    print("=" * 60)

    posterior_intercept = idata.posterior["intercept"].mean().values
    posterior_betas = idata.posterior["betas"].mean().values

    print(f"\n{'参数':<15} {'真实值':>10} {'后验均值':>10} {'差异':>10}")
    print("-" * 60)
    print(f"{'截距':<15} {true_params['intercept']:>10.2f} {posterior_intercept:>10.2f} {posterior_intercept - true_params['intercept']:>10.2f}")
    print(f"{'x1 系数':<15} {true_params['beta1']:>10.2f} {posterior_betas[0]:>10.2f} {posterior_betas[0] - true_params['beta1']:>10.2f}")
    print(f"{'x2 系数':<15} {true_params['beta2']:>10.2f} {posterior_betas[1]:>10.2f} {posterior_betas[1] - true_params['beta2']:>10.2f}")

    # 计算系数相关的概率
    print("\n" + "=" * 60)
    print("系数概率分析")
    print("=" * 60)

    for i, col in enumerate(['x1', 'x2']):
        prob_stats = compute_coefficient_probability(idata, feature_idx=i)
        print(f"\n{col}:")
        print(f"  P(系数 > 0): {prob_stats['prob_positive']:.1%}")
        print(f"  后验均值: {prob_stats['mean']:.2f}")
        print(f"  中位数: {prob_stats['median']:.2f}")
        print(f"  95% HDI: [{prob_stats['ci_95'][0]:.2f}, {prob_stats['ci_95'][1]:.2f}]")

    # 绘制诊断图
    print("\n" + "=" * 60)
    print("生成诊断图...")
    print("=" * 60)

    import matplotlib.pyplot as plt

    # Trace plot
    az.plot_trace(idata, var_names=["intercept", "betas", "sigma"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'regression_trace_plot.png', dpi=150)
    print("✅ Trace plot 已保存: regression_trace_plot.png")

    # Posterior plot
    az.plot_posterior(idata, var_names=["betas"])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'regression_posterior_plot.png', dpi=150)
    print("✅ Posterior plot 已保存: regression_posterior_plot.png")

    # 对比：坏例子
    print("\n" + "=" * 60)
    print("【反例】只用点估计")
    print("=" * 60)
    bad_result = bad_point_estimate_only(idata)
    print(f"❌ 错误做法：只用点估计 {bad_result['point_estimate']}")
    print("   问题：丢失了贝叶斯方法的核心优势——不确定性量化！")
    print("   正确做法：使用完整的后验分布，给出概率陈述")


if __name__ == "__main__":
    main()
