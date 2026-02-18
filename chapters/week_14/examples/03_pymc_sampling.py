"""
示例：PyMC MCMC 采样——后验分布计算

本例演示如何使用 PyMC 进行 MCMC 采样，估计后验分布。
即使先验和似然不是共轭的，MCMC 也能计算后验。

运行方式：python3 chapters/week_14/examples/03_pymc_sampling.py

预期输出：
- MCMC 采样结果统计
- 收敛性诊断（R-hat, ESS）
- 生成迹图和后验分布图到 images/

依赖：
    pip install pymc arviz numpy scipy matplotlib
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# PyMC 和 ArviZ 可能不可用，提供降级方案
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("⚠️  PyMC/ArviZ 未安装，将使用 SciPy 模拟后验分布")
    print("   安装命令: pip install pymc arviz")


def setup_chinese_font() -> str:
    """配置中文字体"""
    import matplotlib.font_manager as fm
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


class PyMCSampler:
    """
    PyMC MCMC 采样器封装

    提供简单的接口来定义模型、运行采样和获取结果
    """

    def __init__(self, random_seed: int = 42):
        """
        初始化采样器

        参数:
            random_seed: 随机种子，确保可复现性
        """
        self.random_seed = random_seed
        self.trace = None
        self.model = None

    def fit_beta_binomial(self, n: int, successes: int,
                         prior_alpha: float = 15, prior_beta: float = 85,
                         draws: int = 2000, tune: int = 1000,
                         chains: int = 4) -> az.InferenceData | None:
        """
        拟合 Beta-Binomial 模型

        参数:
            n: 总样本数
            successes: 成功次数（如流失客户数）
            prior_alpha: 先验 Beta 分布的 alpha 参数
            prior_beta: 先验 Beta 分布的 beta 参数
            draws: 每条链的采样次数
            tune: 调谐步数（丢弃，不计入结果）
            chains: 并行链的数量

        返回:
            ArviZ InferenceData 对象（包含所有采样结果）
        """
        if not PYMC_AVAILABLE:
            return None

        with pm.Model() as model:
            # 先验：Beta 分布
            theta = pm.Beta('theta', alpha=prior_alpha, beta=prior_beta)

            # 似然：Binomial 分布
            likelihood = pm.Binomial('likelihood', n=n, p=theta,
                                    observed=successes)

            # MCMC 采样（使用 NUTS 采样器）
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=self.random_seed,
                return_inferencedata=True
            )
            self.model = model

        return self.trace

    def check_convergence(self) -> dict:
        """
        检查 MCMC 收敛性

        返回:
            包含 R-hat 和 ESS 的字典
        """
        if self.trace is None:
            raise ValueError("请先调用 fit_beta_binomial() 进行采样")

        # R-hat：应该接近 1（< 1.05 表示收敛）
        rhat = az.rhat(self.trace)

        # ESS（有效样本量）：应该足够大（> 400）
        ess = az.ess(self.trace)

        results = {}
        for var in rhat.data_vars:
            results[var] = {
                'rhat': float(rhat[var].values),
                'ess': float(ess[var].values)
            }

        return results

    def summary(self) -> dict:
        """
        获取后验统计摘要

        返回:
            包含均值、标准差、分位数的字典
        """
        if self.trace is None:
            raise ValueError("请先调用 fit_beta_binomial() 进行采样")

        summary = az.summary(self.trace, hdi_prob=0.95)
        results = {}
        for var in summary.index:
            results[var] = {
                'mean': float(summary.loc[var, 'mean']),
                'sd': float(summary.loc[var, 'sd']),
                'hdi_3%': float(summary.loc[var, 'hdi_3%']),
                'hdi_97%': float(summary.loc[var, 'hdi_97%']),
            }
        return results

    def plot_trace(self, output_path: Path) -> None:
        """
        绘制迹图（trace plot），检查链的混合情况

        参数:
            output_path: 图片保存路径
        """
        if self.trace is None:
            raise ValueError("请先调用 fit_beta_binomial() 进行采样")

        setup_chinese_font()
        az.plot_trace(self.trace, var_names=['theta'])
        plt.suptitle('MCMC 迹图：检查链的混合情况', y=1.02, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

    def plot_posterior(self, output_path: Path) -> None:
        """
        绘制后验分布图

        参数:
            output_path: 图片保存路径
        """
        if self.trace is None:
            raise ValueError("请先调用 fit_beta_binomial() 进行采样")

        setup_chinese_font()
        az.plot_posterior(self.trace, var_names=['theta'])
        plt.suptitle('', y=1.02)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()


def simulate_with_scipy(n: int, successes: int,
                       prior_alpha: float, prior_beta: float) -> dict:
    """
    使用 SciPy 模拟后验分布（PyMC 不可用时的降级方案）

    对于 Beta-Binomial 共轭先验，后验有解析解
    """
    from scipy import stats

    alpha_post = prior_alpha + successes
    beta_post = prior_beta + (n - successes)

    # 采样
    samples = stats.beta.rvs(alpha_post, beta_post, size=4000, random_state=42)

    return {
        'samples': samples,
        'mean': float(alpha_post / (alpha_post + beta_post)),
        'hdi_3%': float(stats.beta.ppf(0.03, alpha_post, beta_post)),
        'hdi_97%': float(stats.beta.ppf(0.97, alpha_post, beta_post)),
        'alpha_post': alpha_post,
        'beta_post': beta_post
    }


def plot_scipy_posterior(samples: np.ndarray, mean: float,
                        output_path: Path) -> None:
    """使用 SciPy 采样结果绘制后验分布"""
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 左图：后验分布
    ax = axes[0]
    ax.hist(samples, bins=50, density=True, alpha=0.7, color='#3498db', edgecolor='black')
    ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'均值 = {mean:.3f}')
    ax.set_xlabel('流失率 θ', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('后验分布（SciPy 解析解）', fontsize=13, fontweight='bold')
    ax.legend()

    # 右图：迹图模拟
    ax = axes[1]
    ax.plot(samples, alpha=0.6, linewidth=0.5)
    ax.axhline(mean, color='red', linestyle='--', linewidth=2, label=f'均值 = {mean:.3f}')
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('θ 值', fontsize=12)
    ax.set_title('采样轨迹', fontsize=13, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()


def main() -> None:
    """主函数：运行所有示例"""
    print("\n" + "=" * 60)
    print("PyMC MCMC 采样示例")
    print("=" * 60)

    # 数据：1000 个客户中 180 个流失
    n = 1000
    churned = 180
    prior_alpha = 15
    prior_beta = 85

    print(f"\n数据：{churned}/{n} = {churned/n:.1%}")
    print(f"先验：Beta({prior_alpha}, {prior_beta}) → 均值 {prior_alpha/(prior_alpha+prior_beta):.1%}")

    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)

    if PYMC_AVAILABLE:
        # 使用 PyMC
        print("\n### 使用 PyMC 进行 MCMC 采样")
        print("-" * 40)

        sampler = PyMCSampler(random_seed=42)

        # 拟合模型
        trace = sampler.fit_beta_binomial(
            n=n,
            successes=churned,
            prior_alpha=prior_alpha,
            prior_beta=prior_beta,
            draws=2000,
            tune=1000,
            chains=4
        )

        # 检查收敛性
        print("\n### 收敛性诊断")
        print("-" * 40)
        conv = sampler.check_convergence()
        for var, metrics in conv.items():
            print(f"\n参数：{var}")
            print(f"  R-hat: {metrics['rhat']:.4f}", end='')
            if metrics['rhat'] < 1.05:
                print(" ✅ (收敛良好，< 1.05)")
            else:
                print(" ⚠️  (可能未收敛，建议增加采样量)")
            print(f"  ESS: {metrics['ess']:.0f}", end='')
            if metrics['ess'] > 400:
                print(" ✅ (有效样本量充足)")
            else:
                print(" ⚠️  (有效样本量不足)")

        # 后验统计
        print("\n### 后验分布统计")
        print("-" * 40)
        stats_dict = sampler.summary()
        for var, s in stats_dict.items():
            print(f"\n参数：{var}")
            print(f"  后验均值: {s['mean']:.4f}")
            print(f"  后验标准差: {s['sd']:.4f}")
            print(f"  94% HDI: [{s['hdi_3%']:.4f}, {s['hdi_97%']:.4f}]")

        # 绘图
        print("\n### 生成图表")
        print("-" * 40)
        sampler.plot_trace(output_dir / '03_mcmc_trace.png')
        print(f"✅ 迹图已保存: images/03_mcmc_trace.png")

        sampler.plot_posterior(output_dir / '03_mcmc_posterior.png')
        print(f"✅ 后验分布图已保存: images/03_mcmc_posterior.png")

    else:
        # 使用 SciPy 降级方案
        print("\n### 使用 SciPy 解析解（降级方案）")
        print("-" * 40)

        result = simulate_with_scipy(n, churned, prior_alpha, prior_beta)

        print(f"\n后验：Beta({result['alpha_post']}, {result['beta_post']})")
        print(f"  后验均值: {result['mean']:.4f}")
        print(f"  94% HDI: [{result['hdi_3%']:.4f}, {result['hdi_97%']:.4f}]")

        plot_scipy_posterior(result['samples'], result['mean'],
                            output_dir / '03_scipy_posterior.png')
        print(f"\n✅ 图片已保存: images/03_scipy_posterior.png")

    print("\n### 关键概念回顾")
    print("-" * 40)
    print("1. MCMC（Markov Chain Monte Carlo）：通过采样近似后验分布")
    print("2. R-hat：衡量链间收敛程度，应 < 1.05")
    print("3. ESS（有效样本量）：衡量独立信息量，应 > 400")
    print("4. HDI（Highest Density Interval）：包含 94% 后验概率的最窄区间")


if __name__ == "__main__":
    main()
