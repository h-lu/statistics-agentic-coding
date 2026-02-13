"""
示例：流式统计算法——在线均值、在线方差、在线分位数

运行方式：python3 chapters/week_15/examples/03_streaming_demo.py
预期输出：流式统计与批量统计的对比、更新复杂度分析
"""
from __future__ import annotations

import numpy as np
import time
from typing import Optional
import matplotlib.pyplot as plt
from pathlib import Path


class OnlineMean:
    """增量计算均值"""

    def __init__(self):
        self.n = 0
        self.sum = 0.0

    def update(self, x: float) -> float:
        """
        更新状态（O(1)）

        参数:
            x: 新观测值

        返回:
            更新后的均值
        """
        self.n += 1
        self.sum += x
        return self.mean()

    def mean(self) -> float:
        """返回当前均值"""
        return self.sum / self.n if self.n > 0 else 0.0


class OnlineVariance:
    """增量计算均值和方差（Welford's Online Algorithm）"""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0  # 平方和

    def update(self, x: float) -> None:
        """
        更新状态（O(1)）

        参数:
            x: 新观测值
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def mean(self) -> float:
        """返回当前均值"""
        return self.mean if self.n > 0 else 0.0

    def variance(self) -> float:
        """返回当前方差（总体方差）"""
        return self.M2 / self.n if self.n > 0 else 0.0

    def std(self) -> float:
        """返回当前标准差"""
        return np.sqrt(self.variance())


class OnlineQuantile:
    """增量估计分位数（近似算法：分箱法）"""

    def __init__(self, num_bins: int = 100):
        self.num_bins = num_bins
        self.bins = np.zeros(num_bins)  # 每个箱的计数
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.total_count = 0

    def update(self, x: float) -> None:
        """
        更新状态（O(1)）

        参数:
            x: 新观测值
        """
        self.total_count += 1

        # 更新最小值和最大值
        self.min_val = min(self.min_val, x)
        self.max_val = max(self.max_val, x)

        # 确定箱的编号
        if self.max_val > self.min_val:
            bin_idx = int((x - self.min_val) / (self.max_val - self.min_val) * self.num_bins)
            bin_idx = max(0, min(bin_idx, self.num_bins - 1))
            self.bins[bin_idx] += 1

    def quantile(self, q: float) -> float:
        """
        返回分位数（近似）

        参数:
            q: 分位数（0-1之间）

        返回:
            估计的分位数值
        """
        if self.total_count == 0:
            return 0.0

        target_count = q * self.total_count
        cumulative = 0

        for i, count in enumerate(self.bins):
            cumulative += count
            if cumulative >= target_count:
                # 线性插值
                ratio = i / self.num_bins
                return self.min_val + ratio * (self.max_val - self.min_val)

        return self.max_val


def compare_online_vs_batch(data: np.ndarray, plot_dir: Path = None) -> dict:
    """
    对比流式统计和批量统计的结果

    参数:
        data: 数据数组
        plot_dir: 图片保存目录

    返回:
        对比结果字典
    """
    # 流式计算
    online_mean = OnlineMean()
    online_var = OnlineVariance()
    online_quantile = OnlineQuantile(num_bins=100)

    # 记录中间结果（用于绘图）
    mean_history = []
    var_history = []
    median_history = []

    for i, x in enumerate(data):
        online_mean.update(x)
        online_var.update(x)
        online_quantile.update(x)

        if i % 100 == 0:
            mean_history.append(online_mean.mean())
            var_history.append(online_var.variance())
            median_history.append(online_quantile.quantile(0.5))

    # 批量计算（验证）
    batch_mean = np.mean(data)
    batch_var = np.var(data, ddof=0)
    batch_median = np.median(data)

    # 可视化
    if plot_dir is not None:
        _plot_streaming_comparison(mean_history, var_history, median_history,
                                  batch_mean, batch_var, batch_median, plot_dir)

    return {
        'online_mean': online_mean.mean(),
        'batch_mean': batch_mean,
        'mean_error': abs(online_mean.mean() - batch_mean),
        'online_var': online_var.variance(),
        'batch_var': batch_var,
        'var_error': abs(online_var.variance() - batch_var),
        'online_median': online_quantile.quantile(0.5),
        'batch_median': batch_median,
        'median_error': abs(online_quantile.quantile(0.5) - batch_median)
    }


def _plot_streaming_comparison(mean_history, var_history, median_history,
                               batch_mean, batch_var, batch_median, plot_dir):
    """绘制流式统计对比图"""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    x = np.arange(len(mean_history)) * 100

    # 均值
    axes[0].plot(x, mean_history, 'b-o', linewidth=2, markersize=6, label='在线均值')
    axes[0].axhline(y=batch_mean, color='red', linestyle='--',
                    label=f'批量均值={batch_mean:.4f}', linewidth=2)
    axes[0].set_xlabel('样本数')
    axes[0].set_ylabel('均值')
    axes[0].set_title('在线均值 vs 批量均值')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 方差
    axes[1].plot(x, var_history, 'g-o', linewidth=2, markersize=6, label='在线方差')
    axes[1].axhline(y=batch_var, color='red', linestyle='--',
                    label=f'批量方差={batch_var:.4f}', linewidth=2)
    axes[1].set_xlabel('样本数')
    axes[1].set_ylabel('方差')
    axes[1].set_title('在线方差 vs 批量方差')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 中位数
    axes[2].plot(x, median_history, 'purple', linestyle='-', linewidth=2,
                 marker='o', markersize=6, label='在线中位数')
    axes[2].axhline(y=batch_median, color='red', linestyle='--',
                    label=f'批量中位数={batch_median:.4f}', linewidth=2)
    axes[2].set_xlabel('样本数')
    axes[2].set_ylabel('中位数')
    axes[2].set_title('在线中位数 vs 批量中位数')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / 'streaming_comparison.png', dpi=150)
    print(f"✅ 图表已保存: {plot_dir / 'streaming_comparison.png'}")


def benchmark_complexity(n_samples_list: list = None) -> dict:
    """
    对比流式统计和批量统计的计算复杂度

    参数:
        n_samples_list: 测试的样本量列表

    返回:
        性能对比结果
    """
    if n_samples_list is None:
        n_samples_list = [1000, 5000, 10000, 50000, 100000]

    results = {
        'n_samples': [],
        'batch_time': [],
        'online_time': []
    }

    for n in n_samples_list:
        # 生成数据
        data = np.random.randn(n)

        # 批量计算计时
        start = time.time()
        for _ in range(100):
            mean = np.mean(data)
            var = np.var(data, ddof=0)
        batch_time = (time.time() - start) / 100

        # 流式计算计时
        start = time.time()
        for _ in range(100):
            online_mean = OnlineMean()
            online_var = OnlineVariance()
            for x in data:
                online_mean.update(x)
                online_var.update(x)
        online_time = (time.time() - start) / 100

        results['n_samples'].append(n)
        results['batch_time'].append(batch_time)
        results['online_time'].append(online_time)

        print(f"样本量 {n:6d}: 批量={batch_time*1000:.3f}ms, "
              f"流式={online_time*1000:.3f}ms, "
              f"加速比={batch_time/online_time:.1f}x")

    return results


def main() -> None:
    print("=" * 60)
    print("流式统计算法示例")
    print("=" * 60)

    # 生成测试数据
    np.random.seed(42)
    data = np.random.randn(10000)
    print(f"\n测试数据量: {len(data)}")

    # 对比流式和批量统计
    print("\n" + "=" * 60)
    print("流式统计 vs 批量统计")
    print("=" * 60)

    report_dir = Path("chapters/week_15/report")
    results = compare_online_vs_batch(data, plot_dir=report_dir)

    print(f"\n均值对比:")
    print(f"  流式: {results['online_mean']:.6f}")
    print(f"  批量: {results['batch_mean']:.6f}")
    print(f"  误差: {results['mean_error']:.8f}")

    print(f"\n方差对比:")
    print(f"  流式: {results['online_var']:.6f}")
    print(f"  批量: {results['batch_var']:.6f}")
    print(f"  误差: {results['var_error']:.8f}")

    print(f"\n中位数对比:")
    print(f"  流式: {results['online_median']:.6f}")
    print(f"  批量: {results['batch_median']:.6f}")
    print(f"  误差: {results['median_error']:.6f} (近似算法)")

    # 复杂度对比
    print("\n" + "=" * 60)
    print("计算复杂度对比（每次更新的平均时间）")
    print("=" * 60)

    benchmark_results = benchmark_complexity()

    # 坏例子：每次都重新计算
    print("\n" + "=" * 60)
    print("坏例子：每次新数据都重新计算")
    print("=" * 60)

    print("\n模拟场景：每秒有 100 个新数据到来，持续 100 秒")
    print("总数据量: 10000")

    # 坏方法：每次重新计算
    batch_data = []
    start = time.time()
    for i in range(100):
        new_data = np.random.randn(100).tolist()
        batch_data.extend(new_data)
        mean = np.mean(batch_data)
        var = np.var(batch_data, ddof=0)
    batch_total_time = time.time() - start

    # 好方法：流式更新
    online_mean = OnlineMean()
    online_var = OnlineVariance()
    start = time.time()
    for i in range(100):
        new_data = np.random.randn(100)
        for x in new_data:
            online_mean.update(x)
            online_var.update(x)
    online_total_time = time.time() - start

    print(f"\n批量方法（每次重算）: {batch_total_time*1000:.2f}ms")
    print(f"流式方法（增量更新）: {online_total_time*1000:.2f}ms")
    print(f"加速比: {batch_total_time/online_total_time:.1f}x")

    print("\n问题：随着数据积累，批量方法会越来越慢")
    print("解法：流式统计每次更新都是 O(1)，速度恒定")

    # 限制和陷阱
    print("\n" + "=" * 60)
    print("流式统计的限制和陷阱")
    print("=" * 60)

    print("\n1. 无法回溯")
    print("   - 批量：可以随时重新计算")
    print("   - 流式：状态错误后无法恢复")

    print("\n2. 近似误差")
    print(f"   - 在线分位数是近似值（误差: {results['median_error']:.6f}）")
    print("   - 增加箱数量可以提高精度，但牺牲内存")

    print("\n3. 状态维护")
    print("   - 需要持久化状态（防止系统崩溃）")
    print("   - 分布式环境需要合并状态（如 map-reduce）")

    print("\n" + "=" * 60)
    print("流式统计算法完成")
    print("=" * 60)
    print("\n关键结论:")
    print("1. 流式统计通过维护状态变量实现 O(1) 更新")
    print("2. Welford 算法是增量计算方差的经典方法")
    print("3. 在线分位数是近似算法，精度取决于箱数量")
    print("4. 流式统计是实时系统的必需品，但有使用限制")


if __name__ == "__main__":
    main()
