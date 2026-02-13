"""
Week 15 作业参考实现

本章作业：
1. 实现 PCA 降维并解释主成分
2. 实现 K-means 聚类并选择最优 K 值
3. 实现流式统计算法（在线均值、在线方差）
4. 设计一个简单的 A/B 测试自动化流程

运行方式：
    python3 chapters/week_15/starter_code/solution.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import matplotlib.pyplot as plt


# ============================================================================
# 练习 1: PCA 降维
# ============================================================================

def exercise_1_pca():
    """
    练习 1：PCA 降维

    任务：
    1. 标准化数据（PCA 前必须做）
    2. 拟合 PCA 模型
    3. 选择保留 85% 方差的主成分数量
    4. 解释前 2 个主成分的业务含义
    """
    print("=" * 60)
    print("练习 1：PCA 降维")
    print("=" * 60)

    # 生成示例数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    # 数据有 3 个潜在因子
    latent1 = np.random.randn(n_samples)  # 因子 1
    latent2 = np.random.randn(n_samples)  # 因子 2
    latent3 = np.random.randn(n_samples)  # 因子 3

    X = []
    for i in range(n_features):
        # 每个特征是 3 个因子的线性组合 + 噪声
        loading1 = np.random.uniform(0.5, 1.0)
        loading2 = np.random.uniform(0.0, 0.5)
        loading3 = np.random.uniform(0.0, 0.3)
        feature = (latent1 * loading1 + latent2 * loading2 +
                  latent3 * loading3 + np.random.randn(n_samples) * 0.5)
        X.append(feature)

    X = np.array(X).T  # (1000, 50)

    print(f"\n数据形状: {X.shape}")

    # --- 学生代码开始 ---
    # 1. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. 拟合 PCA（保留所有成分）
    pca = PCA()
    pca.fit(X_scaled)

    # 3. 计算累积方差解释比例
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)

    # 4. 选择保留 85% 方差的主成分数量
    n_components = (cumsum_variance >= 0.85).argmax() + 1

    print(f"\n保留 85% 方差需要 {n_components} 个主成分")
    print(f"压缩率: {X.shape[1] / n_components:.1f}x")

    # 5. 降维
    pca_reduced = PCA(n_components=n_components)
    X_transformed = pca_reduced.fit_transform(X_scaled)

    # 6. 解释主成分
    print(f"\n前 3 个主成分的方差解释比例:")
    for i in range(3):
        print(f"  PC{i+1}: {pca_reduced.explained_variance_ratio_[i]:.2%}")

    # --- 学生代码结束 ---

    print("\n✅ 练习 1 完成")


# ============================================================================
# 练习 2: K-means 聚类
# ============================================================================

def exercise_2_kmeans():
    """
    练习 2：K-means 聚类

    任务：
    1. 使用肘部法则选择最优 K 值
    2. 使用轮廓系数验证聚类质量
    3. 解释每个簇的特征
    """
    print("\n" + "=" * 60)
    print("练习 2：K-means 聚类")
    print("=" * 60)

    # 使用上一节 PCA 降维后的数据
    np.random.seed(42)
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=1000, centers=5, n_features=10,
                       cluster_std=1.5, random_state=42)

    print(f"\n数据形状: {X.shape}")

    # --- 学生代码开始 ---
    # 1. 肘部法则：尝试不同的 K 值
    K_range = range(2, 11)
    wcss = []
    silhouette_scores = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    # 2. 选择最优 K 值（轮廓系数最大）
    k_optimal = np.argmax(silhouette_scores) + min(K_range)

    print(f"\n最优 K 值: {k_optimal}")
    print(f"最大轮廓系数: {max(silhouette_scores):.3f}")

    # 3. 用最优 K 值拟合
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    print(f"\n各簇样本数: {np.bincount(labels)}")

    # --- 学生代码结束 ---

    print("\n✅ 练习 2 完成")


# ============================================================================
# 练习 3: 流式统计
# ============================================================================

class OnlineMean:
    """在线均值计算器"""

    def __init__(self):
        self.n = 0
        self.sum = 0.0

    def update(self, x: float) -> None:
        """更新状态"""
        self.n += 1
        self.sum += x

    def mean(self) -> float:
        """返回当前均值"""
        return self.sum / self.n if self.n > 0 else 0.0


class OnlineVariance:
    """在线方差计算器（Welford 算法）"""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        """更新状态"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self) -> float:
        """返回当前方差"""
        return self.M2 / self.n if self.n > 0 else 0.0


def exercise_3_streaming():
    """
    练习 3：流式统计

    任务：
    1. 实现 OnlineMean 类
    2. 实现 OnlineVariance 类（Welford 算法）
    3. 对比流式统计和批量统计的结果
    """
    print("\n" + "=" * 60)
    print("练习 3：流式统计")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    data = np.random.randn(1000)

    # --- 学生代码开始 ---
    # 1. 流式计算
    online_mean = OnlineMean()
    online_var = OnlineVariance()

    for x in data:
        online_mean.update(x)
        online_var.update(x)

    # 2. 批量计算（验证）
    batch_mean = np.mean(data)
    batch_var = np.var(data, ddof=0)

    print(f"\n均值对比:")
    print(f"  流式: {online_mean.mean():.6f}")
    print(f"  批量: {batch_mean:.6f}")
    print(f"  误差: {abs(online_mean.mean() - batch_mean):.8f}")

    print(f"\n方差对比:")
    print(f"  流式: {online_var.variance():.6f}")
    print(f"  批量: {batch_var:.6f}")
    print(f"  误差: {abs(online_var.variance() - batch_var):.8f}")

    # --- 学生代码结束 ---

    print("\n✅ 练习 3 完成")


# ============================================================================
# 练习 4: A/B 测试工程化
# ============================================================================


class ExperimentConfig:
    """实验配置"""

    def __init__(self, treatment_groups, metric, sample_ratio, min_sample_size,
                 significance_level=0.05, min_effect_size=0.01):
        self.treatment_groups = treatment_groups
        self.metric = metric
        self.sample_ratio = sample_ratio
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        self.min_effect_size = min_effect_size


class ExperimentResult:
    """实验结果"""

    def __init__(self, sample_sizes, metrics, p_value, effect_size,
                 ci_low, ci_high, decision):
        self.sample_sizes = sample_sizes
        self.metrics = metrics
        self.p_value = p_value
        self.effect_size = effect_size
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.decision = decision


class SimpleABTest:
    """简化的 A/B 测试平台"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data = {group: [] for group in config.treatment_groups}

    def add_observation(self, group: str, value: float) -> None:
        """添加观测值"""
        if group in self.data:
            self.data[group].append(value)

    def analyze(self) -> ExperimentResult:
        """运行 A/B 测试分析"""
        # 计算统计量
        metrics = {}
        for group in self.config.treatment_groups:
            metrics[group] = np.mean(self.data[group])

        # t 检验
        group_A, group_B = self.config.treatment_groups
        data_A = np.array(self.data[group_A])
        data_B = np.array(self.data[group_B])

        t_stat, p_value = stats.ttest_ind(data_B, data_A)

        # 效应量
        effect_size = metrics[group_B] - metrics[group_A]

        # 置信区间
        se = np.sqrt(data_A.var(ddof=1)/len(data_A) +
                    data_B.var(ddof=1)/len(data_B))
        ci_low = effect_size - 1.96 * se
        ci_high = effect_size + 1.96 * se

        # 决策
        if p_value < self.config.significance_level:
            decision = f"launch_{group_B}"
        else:
            decision = "continue"

        return ExperimentResult(
            sample_sizes={g: len(self.data[g]) for g in self.config.treatment_groups},
            metrics=metrics,
            p_value=p_value,
            effect_size=effect_size,
            ci_low=ci_low,
            ci_high=ci_high,
            decision=decision
        )


def exercise_4_ab_test():
    """
    练习 4：A/B 测试工程化

    任务：
    1. 实现 SimpleABTest 类
    2. 运行 A/B 测试并输出决策
    3. 解释结果
    """
    print("\n" + "=" * 60)
    print("练习 4：A/B 测试工程化")
    print("=" * 60)

    # 创建配置
    config = ExperimentConfig(
        treatment_groups=["A", "B"],
        metric="avg_revenue",
        sample_ratio={"A": 0.5, "B": 0.5},
        min_sample_size=1000,
        significance_level=0.05,
        min_effect_size=5.0
    )

    # --- 学生代码开始 ---
    # 创建 A/B 测试平台
    platform = SimpleABTest(config)

    # 模拟数据（B 组有真实效应）
    np.random.seed(42)
    for _ in range(1000):
        value_A = np.random.normal(100, 20)
        value_B = np.random.normal(108, 20)  # 效应 +8
        platform.add_observation("A", value_A)
        platform.add_observation("B", value_B)

    # 分析
    result = platform.analyze()

    print(f"\nA/B 测试结果:")
    print(f"  A 组均值: {result.metrics['A']:.2f}")
    print(f"  B 组均值: {result.metrics['B']:.2f}")
    print(f"  效应量: {result.effect_size:.2f}")
    print(f"  p 值: {result.p_value:.4f}")
    print(f"  95% CI: [{result.ci_low:.2f}, {result.ci_high:.2f}]")
    print(f"  决策: {result.decision}")

    # --- 学生代码结束 ---

    print("\n✅ 练习 4 完成")


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有练习"""
    print("\n" + "=" * 60)
    print("Week 15 作业参考实现")
    print("=" * 60)

    exercise_1_pca()
    exercise_2_kmeans()
    exercise_3_streaming()
    exercise_4_ab_test()

    print("\n" + "=" * 60)
    print("所有练习完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
