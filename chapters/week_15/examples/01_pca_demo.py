"""
示例：PCA 降维——从 5000 维到 47 维

运行方式：python3 chapters/week_15/examples/01_pca_demo.py
预期输出：降维结果、方差解释比例、主成分分析图表
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path


def generate_user_behavior_matrix(n_samples: int = 1000, n_features: int = 500,
                                  random_seed: int = 42) -> pd.DataFrame:
    """
    生成高维用户行为矩阵

    模拟 500 个行为特征（点击、停留、购买、搜索等）

    参数:
        n_samples: 样本数
        n_features: 特征数
        random_seed: 随机种子

    返回:
        特征矩阵 DataFrame
    """
    np.random.seed(random_seed)

    # 生成相关特征（模拟用户行为的内在结构）
    # 潜在因子：活跃度、购买倾向、浏览深度
    n_latent = 3

    activity = np.random.normal(0, 1, n_samples)  # 活跃度
    purchase_intent = np.random.normal(0, 1, n_samples)  # 购买倾向
    browse_depth = np.random.normal(0, 1, n_samples)  # 浏览深度

    # 特征是潜在因子的线性组合 + 噪声
    features = {}
    feature_idx = 0

    # 每个潜在因子生成一组相关特征
    for latent_factor in [activity, purchase_intent, browse_depth]:
        n_features_per_factor = n_features // n_latent
        for i in range(n_features_per_factor):
            loading = np.random.uniform(0.3, 0.8)  # 因子载荷
            noise = np.random.normal(0, 0.5, n_samples)
            features[f'feature_{feature_idx}'] = latent_factor * loading + noise
            feature_idx += 1

    # 补充剩余特征（纯噪声）
    while feature_idx < n_features:
        features[f'feature_{feature_idx}'] = np.random.normal(0, 1, n_samples)
        feature_idx += 1

    df = pd.DataFrame(features)
    return df


def pca_dim_reduction(X: pd.DataFrame, variance_threshold: float = 0.85,
                      plot_dir: Path = None) -> tuple:
    """
    PCA 降维

    参数:
        X: 特征矩阵
        variance_threshold: 保留的方差比例阈值
        plot_dir: 图片保存目录

    返回:
        X_transformed: 降维后的数据
        pca: PCA 模型
        scaler: 标准化器
        n_components: 选择的成分数
    """
    # 1. 标准化（PCA 前必须做）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 拟合 PCA（保留所有成分，看方差解释）
    pca_full = PCA()
    pca_full.fit(X_scaled)

    # 3. 计算累积方差解释比例
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)

    # 4. 选择成分数
    n_components = (cumsum_variance >= variance_threshold).argmax() + 1

    # 5. 用选定数量重新拟合
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_scaled)

    # 6. 可视化（如果提供了保存目录）
    if plot_dir is not None:
        _plot_pca_results(pca_full, cumsum_variance, n_components,
                         X_transformed, plot_dir, variance_threshold)

    return X_transformed, pca, scaler, n_components


def _plot_pca_results(pca_full, cumsum_variance, n_components,
                      X_transformed, plot_dir, variance_threshold):
    """生成 PCA 相关图表"""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(15, 4))

    # 左图：方差解释比例（前 50 个成分）
    plt.subplot(1, 3, 1)
    plt.bar(range(1, 51), pca_full.explained_variance_ratio_[:50], alpha=0.7)
    plt.xlabel('主成分编号')
    plt.ylabel('方差解释比例')
    plt.title('各主成分的方差解释比例')
    plt.grid(True, alpha=0.3)

    # 中图：累积方差解释比例
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance,
             linewidth=2)
    plt.axhline(y=variance_threshold, color='red', linestyle='--',
                label=f'{variance_threshold:.0%} 阈值')
    plt.axvline(x=n_components, color='red', linestyle='--',
                label=f'{n_components} 个主成分')
    plt.xlabel('主成分数量')
    plt.ylabel('累积方差解释比例')
    plt.title('累积方差解释比例')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：前 2 个主成分的散点图
    plt.subplot(1, 3, 3)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5, s=20)
    plt.xlabel(f'PC1 (方差解释 {pca_full.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 (方差解释 {pca_full.explained_variance_ratio_[1]:.1%})')
    plt.title('前 2 个主成分的样本分布')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / 'pca_variance_explained.png', dpi=150)
    print(f"✅ 图表已保存: {plot_dir / 'pca_variance_explained.png'}")


def main() -> None:
    print("=" * 60)
    print("PCA 降维示例：从 5000 维到 47 维")
    print("=" * 60)

    # 生成高维用户行为数据
    print("\n生成高维用户行为数据...")
    X = generate_user_behavior_matrix(n_samples=5000, n_features=500, random_seed=42)
    print(f"原始数据形状: {X.shape}")

    # PCA 降维
    print("\n运行 PCA 降维...")
    report_dir = Path("chapters/week_15/report")
    X_transformed, pca, scaler, n_components = pca_dim_reduction(
        X, variance_threshold=0.85, plot_dir=report_dir
    )

    print(f"降维后数据形状: {X_transformed.shape}")
    print(f"压缩率: {X.shape[1] / n_components:.1f}x")
    print(f"保留方差: {sum(pca.explained_variance_ratio_):.1%}")

    # 打印主成分信息
    print("\n" + "=" * 60)
    print("前 10 个主成分的方差解释比例")
    print("=" * 60)
    for i in range(min(10, n_components)):
        cumsum = sum(pca.explained_variance_ratio_[:i+1])
        print(f"PC{i+1:2d}: {pca.explained_variance_ratio_[i]:.4f} "
              f"(累积: {cumsum:.4f})")

    # 坏例子：不标准化直接做 PCA
    print("\n" + "=" * 60)
    print("坏例子：不标准化直接做 PCA")
    print("=" * 60)

    X_no_scale = X.values
    pca_bad = PCA(n_components=n_components)
    X_bad = pca_bad.fit_transform(X_no_scale)

    print(f"不标准化的保留方差: {sum(pca_bad.explained_variance_ratio_):.1%}")
    print("问题：PCA 对特征尺度敏感，标准化是必需步骤！")

    print("\n" + "=" * 60)
    print("PCA 降维完成")
    print("=" * 60)
    print("\n关键结论:")
    print("1. PCA 降维前必须标准化")
    print("2. 选择成分数基于累积方差解释比例（如 85%）")
    print("3. 压缩率越高，信息损失越多，需要权衡")
    print("4. 前 2-3 个主成分通常可用于可视化")


if __name__ == "__main__":
    main()
