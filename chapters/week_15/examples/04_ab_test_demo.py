"""
示例：A/B 测试工程化——从手动分析到自动化决策平台

运行方式：python3 chapters/week_15/examples/04_ab_test_demo.py
预期输出：A/B 测试自动化流程、SRM 检测、决策建议
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List
import datetime
from pathlib import Path


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    treatment_groups: List[str]  # 如 ["A", "B"]
    metric: str  # 如 "conversion_rate", "avg_revenue"
    sample_ratio: Dict[str, float]  # 如 {"A": 0.5, "B": 0.5}
    min_sample_size: int  # 最小样本量
    significance_level: float = 0.05  # 显著性水平
    min_effect_size: float = 0.01  # 最小可检测效应


@dataclass
class ExperimentResult:
    """实验结果"""
    timestamp: datetime.datetime
    sample_sizes: Dict[str, int]
    metrics: Dict[str, float]
    p_value: float
    ci_low: float
    ci_high: float
    decision: str  # "launch_B", "continue", "reject_B"
    srm_detected: bool = False  # 样本比例异常


class ABTestPlatform:
    """A/B 测试自动化平台"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.data = {group: [] for group in config.treatment_groups}

    def add_observation(self, group: str, value: float) -> None:
        """
        添加一个观测值（流式更新）

        参数:
            group: 实验组名称
            value: 观测值
        """
        if group in self.data:
            self.data[group].append(value)

    def check_sample_ratio_mismatch(self) -> bool:
        """
        检查样本比例是否异常（SRM 检测）

        使用卡方检验检测样本比例是否符合预期

        返回:
            True 表示检测到 SRM（样本比例异常）
        """
        total_observed = sum(len(self.data[g]) for g in self.config.treatment_groups)

        # 如果样本量太小，不检测
        if total_observed < self.config.min_sample_size:
            return False

        observed_sizes = {g: len(self.data[g]) for g in self.config.treatment_groups}
        expected_sizes = {g: total_observed * self.config.sample_ratio[g]
                         for g in self.config.treatment_groups}

        # 卡方检验
        observed = np.array([observed_sizes[g] for g in self.config.treatment_groups])
        expected = np.array([expected_sizes[g] for g in self.config.treatment_groups])

        # 避免除零
        expected = np.maximum(expected, 1)

        chi2, p_value = stats.chisquare(observed, expected)

        return p_value < 0.05  # p < 0.05 表示样本比例异常

    def analyze(self) -> ExperimentResult:
        """
        运行完整的 A/B 测试分析

        返回:
            ExperimentResult 对象
        """
        # 1. 计算统计量
        metrics = {}
        for group in self.config.treatment_groups:
            data = np.array(self.data[group])
            if len(data) > 0:
                if self.config.metric == "conversion_rate":
                    metrics[group] = data.mean()
                elif self.config.metric == "avg_revenue":
                    metrics[group] = data.mean()
                else:
                    metrics[group] = data.mean()

        # 2. 假设检验（假设是 A/B 两组）
        if len(self.config.treatment_groups) >= 2:
            group_A = self.config.treatment_groups[0]
            group_B = self.config.treatment_groups[1]
            data_A = np.array(self.data[group_A])
            data_B = np.array(self.data[group_B])

            # 双样本 t 检验
            t_stat, p_value = stats.ttest_ind(data_B, data_A)

            # 3. 计算置信区间（效应量的 CI）
            effect_size = metrics[group_B] - metrics[group_A]
            se = np.sqrt(data_A.var(ddof=1)/len(data_A) +
                        data_B.var(ddof=1)/len(data_B))
            ci_low = effect_size - 1.96 * se
            ci_high = effect_size + 1.96 * se

            # 4. 决策规则
            min_samples = min(len(data_A), len(data_B))

            if min_samples < self.config.min_sample_size:
                decision = "continue"  # 样本量不足，继续收集
            elif p_value < self.config.significance_level and abs(effect_size) >= self.config.min_effect_size:
                decision = f"launch_{group_B}"  # 上线 B
            elif p_value < 0.10:  # 边界显著
                decision = "continue"  # 继续收集数据
            else:
                decision = f"reject_{group_B}"  # 放弃 B
        else:
            p_value = 1.0
            ci_low = 0.0
            ci_high = 0.0
            decision = "continue"

        # 5. 检查 SRM
        srm_detected = self.check_sample_ratio_mismatch()

        return ExperimentResult(
            timestamp=datetime.datetime.now(),
            sample_sizes={g: len(self.data[g]) for g in self.config.treatment_groups},
            metrics=metrics,
            p_value=p_value,
            ci_low=ci_low,
            ci_high=ci_high,
            decision=decision,
            srm_detected=srm_detected
        )


def generate_ab_test_data(n_per_group: int = 1000, true_effect: float = 0.0,
                          random_seed: int = 42) -> pd.DataFrame:
    """
    生成 A/B 测试数据

    参数:
        n_per_group: 每组样本量
        true_effect: 真实效应量（B 组相对于 A 组的提升）
        random_seed: 随机种子

    返回:
        包含组别和结果的 DataFrame
    """
    np.random.seed(random_seed)

    # A 组：对照组
    data_A = np.random.normal(100, 20, n_per_group)

    # B 组：处理组（加上真实效应）
    data_B = np.random.normal(100 + true_effect, 20, n_per_group)

    df = pd.DataFrame({
        'group': ['A'] * n_per_group + ['B'] * n_per_group,
        'value': np.concatenate([data_A, data_B])
    })

    return df


def bad_early_stopping_example() -> None:
    """
    反例：早期停止（Early Stopping）导致的假阳性

    问题：在样本量不足时提前停止，导致假阳性率增加
    """
    print("\n" + "=" * 60)
    print("反例：早期停止（Early Stopping）")
    print("=" * 60)

    config = ExperimentConfig(
        name="早期停止测试",
        treatment_groups=["A", "B"],
        metric="avg_revenue",
        sample_ratio={"A": 0.5, "B": 0.5},
        min_sample_size=1000,  # 预定样本量
        significance_level=0.05,
        min_effect_size=5.0
    )

    platform = ABTestPlatform(config)

    # 模拟流式数据到来（真实效应为 0）
    np.random.seed(42)
    early_significant_count = 0
    total_simulations = 100

    for sim in range(total_simulations):
        platform.data = {"A": [], "B": []}

        # 生成数据（真实效应为 0）
        for i in range(1000):
            value_A = np.random.normal(100, 20)
            value_B = np.random.normal(100, 20)  # 无真实效应
            platform.add_observation("A", value_A)
            platform.add_observation("B", value_B)

            # 每 100 个样本检查一次
            if (i + 1) % 100 == 0 and (i + 1) < 1000:
                result = platform.analyze()
                if result.decision == "launch_B":
                    early_significant_count += 1
                    break

    false_positive_rate = early_significant_count / total_simulations

    print(f"\n模拟 100 次，真实效应为 0 时的假阳性率: {false_positive_rate:.1%}")
    print(f"理论假阳性率（α=0.05）: 5.0%")
    print(f"结论: 早期停止会显著增加假阳性率！")


def bad_srm_example() -> None:
    """
    反例：样本比例异常（SRM）

    问题：随机化代码有 bug，导致样本比例不符合预期
    """
    print("\n" + "=" * 60)
    print("反例：样本比例异常（SRM）")
    print("=" * 60)

    # 模拟 SRM：A 组样本量是 B 组的 2 倍
    data_A = np.random.normal(100, 20, 2000)
    data_B = np.random.normal(108, 20, 1000)  # B 组均值更高

    # 直接 t 检验（不考虑 SRM）
    t_stat, p_value = stats.ttest_ind(data_B, data_A)

    print(f"\nA 组样本量: {len(data_A)}")
    print(f"B 组样本量: {len(data_B)}")
    print(f"样本比例: {len(data_A) / (len(data_A) + len(data_B)):.1%} vs {len(data_B) / (len(data_A) + len(data_B)):.1%}")
    print(f"\nt 检验结果: p={p_value:.4f}")

    if p_value < 0.05:
        print("结论: 显著（但这不可信，因为样本比例异常！）")

    # 卡方检验检测 SRM
    observed = np.array([len(data_A), len(data_B)])
    expected = np.array([1500, 1500])  # 预期 50:50
    chi2, p_srm = stats.chisquare(observed, expected)

    print(f"\nSRM 检测: χ²={chi2:.2f}, p={p_srm:.4f}")
    print(f"结论: {'检测到 SRM！' if p_srm < 0.05 else '未检测到 SRM'}")


def main() -> None:
    print("=" * 60)
    print("A/B 测试工程化示例")
    print("=" * 60)

    # 创建实验配置
    config = ExperimentConfig(
        name="优惠券效果测试",
        treatment_groups=["A", "B"],
        metric="avg_revenue",
        sample_ratio={"A": 0.5, "B": 0.5},
        min_sample_size=1000,
        significance_level=0.05,
        min_effect_size=5.0
    )

    platform = ABTestPlatform(config)

    # 模拟流式数据到来（B 组有真实效应：+8）
    np.random.seed(42)
    print("\n模拟流式数据到来...")
    print(f"真实效应: B 组比 A 组高 8 元")

    for i in range(1000):
        # A 组：均值为 100，标准差 20
        value_A = np.random.normal(100, 20)
        platform.add_observation("A", value_A)

        # B 组：均值为 108（效应量 8），标准差 20
        value_B = np.random.normal(108, 20)
        platform.add_observation("B", value_B)

        # 每 200 个样本分析一次
        if (i + 1) % 200 == 0:
            result = platform.analyze()
            print(f"\n样本量 {i+1}:")
            print(f"  A 组均值: {result.metrics['A']:.2f}")
            print(f"  B 组均值: {result.metrics['B']:.2f}")
            print(f"  效应量: {result.metrics['B'] - result.metrics['A']:.2f}")
            print(f"  p 值: {result.p_value:.4f}")
            print(f"  95% CI: [{result.ci_low:.2f}, {result.ci_high:.2f}]")
            print(f"  决策: {result.decision}")

            if result.srm_detected:
                print(f"  ⚠️  警告：样本比例异常（SRM 检测）")

    # 反例：早期停止
    bad_early_stopping_example()

    # 反例：SRM
    bad_srm_example()

    # 工程化建议
    print("\n" + "=" * 60)
    print("A/B 测试工程化建议")
    print("=" * 60)

    print("\n1. 强制最小样本量")
    print("   - 避免早期停止导致的假阳性")
    print("   - 样本量应基于功效分析（Power Analysis）")

    print("\n2. SRM 检测")
    print("   - 每次分析前检查样本比例")
    print("   - 如果检测到 SRM，检查随机化代码")

    print("\n3. 分组审查")
    print("   - 检查不同分组（国家、设备）的结论")
    print("   - 避免辛普森悖论")

    print("\n4. Human-in-the-loop")
    print("   - 系统提供建议，最终决策由人负责")
    print("   - 自动化 ≠ 无监督")

    print("\n" + "=" * 60)
    print("A/B 测试工程化完成")
    print("=" * 60)
    print("\n关键结论:")
    print("1. A/B 测试工程化是系统，不只是算法")
    print("2. SRM 检测、早期停止、辛普森悖论是常见陷阱")
    print("3. Human-in-the-loop 是必要的")
    print("4. 自动化平台应提供建议，而非直接决策")


if __name__ == "__main__":
    main()
