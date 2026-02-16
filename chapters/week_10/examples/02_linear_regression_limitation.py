"""
示例：为什么线性回归不适合分类——预测值越界与误差假设不匹配。

本例演示用线性回归做分类的三个问题：
1. 预测值越界：回归输出可能超出 [0,1] 范围
2. 误差假设不匹配：分类问题的误差是二元的，不服从正态分布
3. 准确率幻觉：在类别不平衡时，高准确率可能是误导性的

运行方式：python3 chapters/week_10/examples/02_linear_regression_limitation.py
预期输出：
  - stdout 输出回归和逻辑回归的预测对比
  - 展示预测值越界问题
  - 保存图表到 images/02_regression_limitation.png

核心概念：
  - 线性回归输出范围：(-∞, +∞)
  - 分类概率范围：[0, 1]
  - Sigmoid 函数：将线性输出压缩到 (0, 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def setup_chinese_font() -> str:
    """配置中文字体，返回使用的字体名称"""
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


def generate_imbalanced_data(n: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    生成类别不平衡的客户流失数据。
    高价值客户仅占 5%，模拟真实业务场景。
    
    参数:
        n: 样本量
        random_state: 随机种子
        
    返回:
        DataFrame 包含客户特征
    """
    np.random.seed(random_state)
    
    # 生成特征
    data = pd.DataFrame({
        '注册月数': np.random.randint(1, 48, n),
        '月均浏览次数': np.random.poisson(30, n),
        '月均消费次数': np.random.poisson(3, n),
        '最近登录距今天数': np.random.randint(1, 60, n)
    })
    
    # 生成高价值标签（少数类，约 5%）
    # 高价值客户的特征：注册久、浏览多、消费频繁、最近登录
    score = (
        0.1 * data['注册月数'] +
        0.05 * data['月均浏览次数'] +
        0.5 * data['月均消费次数'] -
        0.02 * data['最近登录距今天数'] +
        np.random.normal(0, 1, n)
    )
    data['是否高价值'] = (score > np.percentile(score, 95)).astype(int)
    
    return data


def demonstrate_boundary_violation(df: pd.DataFrame) -> None:
    """
    演示问题 1：线性回归预测值越界。
    """
    print("=" * 70)
    print("问题 1：线性回归预测值越界")
    print("=" * 70)
    
    X = df[['注册月数', '月均浏览次数', '月均消费次数', '最近登录距今天数']]
    y = df['是否高价值']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 用线性回归做分类
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    
    y_pred_reg = reg_model.predict(X_test)
    
    print("\n线性回归预测结果示例：")
    print(f"{'实际':<10} {'预测值':<12} {'是否越界':<10}")
    print("-" * 40)
    
    out_of_bounds = []
    for i in range(20):
        actual = y_test.iloc[i]
        pred = y_pred_reg[i]
        status = ""
        if pred > 1:
            status = "⚠️ >1"
            out_of_bounds.append(('>1', pred))
        elif pred < 0:
            status = "⚠️ <0"
            out_of_bounds.append(('<0', pred))
        print(f"{actual:<10} {pred:<12.4f} {status:<10}")
    
    # 统计越界情况
    n_above_1 = (y_pred_reg > 1).sum()
    n_below_0 = (y_pred_reg < 0).sum()
    
    print(f"\n越界统计：")
    print(f"  预测值 > 1 的样本数: {n_above_1} ({n_above_1/len(y_pred_reg)*100:.1f}%)")
    print(f"  预测值 < 0 的样本数: {n_below_0} ({n_below_0/len(y_pred_reg)*100:.1f}%)")
    
    print(f"\n问题分析：")
    print(f"  ❌ 回归输出范围是 (-∞, +∞)，但分类概率必须在 [0, 1] 之间")
    print(f"  ❌ 预测值 1.3 或 -0.2 无法解释为'概率'")
    print(f"  ❌ 信息损失：1.3 和 2.5 都被截断为 1，但'高价值程度'不同")


def demonstrate_accuracy_trap(df: pd.DataFrame) -> None:
    """
    演示问题 2：准确率幻觉（类别不平衡时的误导性准确率）。
    """
    print("\n" + "=" * 70)
    print("问题 2：准确率幻觉（类别不平衡）")
    print("=" * 70)
    
    X = df[['注册月数', '月均浏览次数', '月均消费次数', '最近登录距今天数']]
    y = df['是否高价值']
    
    # 类别分布
    class_dist = y.value_counts().sort_index()
    imbalance_ratio = class_dist.max() / class_dist.min()
    
    print(f"\n类别分布：")
    print(f"  低价值客户 (0): {class_dist[0]} ({class_dist[0]/len(y)*100:.1f}%)")
    print(f"  高价值客户 (1): {class_dist[1]} ({class_dist[1]/len(y)*100:.1f}%)")
    print(f"  不平衡比例: 1:{imbalance_ratio:.0f}")
    
    # "愚蠢"的基准模型：全部预测为多数类（低价值）
    baseline_pred = np.zeros(len(y))
    baseline_accuracy = (baseline_pred == y).mean()
    
    print(f"\n⚠️  '愚蠢'基准模型（全部预测为低价值）：")
    print(f"    准确率 = {baseline_accuracy:.2%}")
    print(f"    但：把所有高价值客户都漏掉了！")
    
    # 线性回归模型的表现
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    y_pred_reg = (reg_model.predict(X_test) > 0.5).astype(int)
    
    reg_accuracy = (y_pred_reg == y_test).mean()
    
    print(f"\n线性回归模型（阈值 0.5）：")
    print(f"  准确率 = {reg_accuracy:.2%}")
    
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_reg)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n混淆矩阵分析：")
    print(f"  真正例 (TP): {tp} - 正确识别的高价值客户")
    print(f"  假正例 (FP): {fp} - 错误识别为高的低价值客户")
    print(f"  真反例 (TN): {tn} - 正确识别的低价值客户")
    print(f"  假反例 (FN): {fn} - 漏掉的高价值客户")
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\n  查全率 (Recall): {recall:.2%} - 只找到了 {recall:.0%} 的高价值客户！")


def demonstrate_error_assumption(df: pd.DataFrame) -> None:
    """
    演示问题 3：误差假设不匹配。
    """
    print("\n" + "=" * 70)
    print("问题 3：误差假设不匹配")
    print("=" * 70)
    
    print("\n线性回归的误差假设（LINE）：")
    print("  - L: Linearity（线性关系）")
    print("  - I: Independence（误差独立）")
    print("  - N: Normality（误差正态分布） ← 这里出问题")
    print("  - E: Equal Variance（等方差）")
    
    print("\n分类问题的误差特征：")
    print("  - 误差是二元的：对 (0) 或错 (1)")
    print("  - 服从伯努利分布，不是正态分布")
    print("  - 误差的方差随预测概率变化：p(1-p)")
    
    print("\n后果：")
    print("  ❌ 系数的 p 值不可靠")
    print("  ❌ 置信区间不准确")
    print("  ❌ 残差图无法解释")
    print("  ❌ 统计推断失效")


def compare_with_logistic(df: pd.DataFrame) -> None:
    """
    对比线性回归和逻辑回归的表现。
    """
    print("\n" + "=" * 70)
    print("解决方案：逻辑回归")
    print("=" * 70)
    
    X = df[['注册月数', '月均浏览次数', '月均消费次数', '最近登录距今天数']]
    y = df['是否高价值']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 逻辑回归
    logit_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    logit_model.fit(X_train_scaled, y_train)
    
    y_prob = logit_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = logit_model.predict(X_test_scaled)
    
    print("\n逻辑回归预测结果示例：")
    print(f"{'实际':<10} {'预测概率':<12} {'预测类别':<12} {'状态':<10}")
    print("-" * 55)
    for i in range(10):
        status = "✅" if y_pred[i] == y_test.iloc[i] else "❌"
        print(f"{y_test.iloc[i]:<10} {y_prob[i]:<12.4f} {y_pred[i]:<12} {status:<10}")
    
    print(f"\n逻辑回归的优势：")
    print(f"  ✅ 预测概率始终在 (0, 1) 之间")
    print(f"  ✅ 使用对数似然损失，适合分类问题")
    print(f"  ✅ 输出可解释为'属于某类的概率'")
    
    # 概率范围检查
    print(f"\n概率范围检查：")
    print(f"  最小概率: {y_prob.min():.6f} (接近 0)")
    print(f"  最大概率: {y_prob.max():.6f} (接近 1)")


def plot_limitations(df: pd.DataFrame) -> None:
    """绘制回归局限性的可视化"""
    setup_chinese_font()
    
    X = df[['注册月数', '月均浏览次数', '月均消费次数', '最近登录距今天数']]
    y = df['是否高价值']
    
    # 只用一个特征便于可视化
    X_simple = df[['月均消费次数']]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_simple, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 线性回归
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    
    # 逻辑回归
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logit_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    logit_model.fit(X_train_scaled, y_train)
    
    # 创建平滑的 x 值用于绘图
    x_range = np.linspace(X_simple.min().values[0], X_simple.max().values[0], 100).reshape(-1, 1)
    x_range_scaled = scaler.transform(x_range)
    
    # 预测
    y_pred_reg = reg_model.predict(x_range)
    y_pred_logit = logit_model.predict_proba(x_range_scaled)[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：线性回归的问题
    ax1 = axes[0]
    ax1.scatter(X_test, y_test, alpha=0.5, s=50, color='#2E86AB', 
                label='实际数据', edgecolors='black', linewidth=0.5)
    ax1.plot(x_range, y_pred_reg, 'r-', linewidth=2.5, label='线性回归预测')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax1.fill_between(x_range.flatten(), 1, y_pred_reg.clip(max=2), 
                      where=(y_pred_reg > 1), alpha=0.3, color='red', label='越界区域 (>1)')
    ax1.fill_between(x_range.flatten(), y_pred_reg.clip(min=-1), 0, 
                      where=(y_pred_reg < 0), alpha=0.3, color='orange', label='越界区域 (<0)')
    ax1.set_xlabel('月均消费次数', fontsize=12)
    ax1.set_ylabel('预测值', fontsize=12)
    ax1.set_title('线性回归的问题\n预测值超出 [0,1] 范围', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.set_ylim(-0.5, 1.5)
    ax1.grid(True, alpha=0.3)
    
    # 右图：逻辑回归的解决方案
    ax2 = axes[1]
    ax2.scatter(X_test, y_test, alpha=0.5, s=50, color='#2E86AB', 
                label='实际数据', edgecolors='black', linewidth=0.5)
    ax2.plot(x_range, y_pred_logit, 'g-', linewidth=2.5, label='逻辑回归 (sigmoid)')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='阈值 = 0.5')
    ax2.set_xlabel('月均消费次数', fontsize=12)
    ax2.set_ylabel('预测概率', fontsize=12)
    ax2.set_title('逻辑回归的解决方案\n预测概率始终在 (0,1) 之间', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_ylim(-0.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '02_regression_limitation.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n图表已保存到: images/02_regression_limitation.png")


def main() -> None:
    """主函数"""
    print("线性回归的局限性：为什么不能用回归做分类\n")
    
    # 生成不平衡数据
    df = generate_imbalanced_data(n=1000, random_state=42)
    print(f"数据集：{len(df)} 个客户样本")
    print(f"高价值客户占比: {(df['是否高价值'] == 1).mean()*100:.1f}%\n")
    
    # 演示三个问题
    demonstrate_boundary_violation(df)
    demonstrate_accuracy_trap(df)
    demonstrate_error_assumption(df)
    compare_with_logistic(df)
    
    # 绘图
    plot_limitations(df)
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n线性回归不适合分类的三个原因：")
    print("  1. 预测值越界：回归输出 (-∞,+∞)，但概率必须在 [0,1]")
    print("  2. 准确率幻觉：在类别不平衡时，高准确率可能是误导性的")
    print("  3. 误差假设不匹配：分类误差是二元的，不服从正态分布")
    print("\n解决方案：逻辑回归")
    print("  - 使用 sigmoid 函数将线性输出压缩到 (0,1)")
    print("  - 使用对数似然损失，适合分类问题")
    print("  - 输出可解释为概率")
    print()


if __name__ == "__main__":
    main()
