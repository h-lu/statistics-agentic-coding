"""
示例：从散点图到 Sigmoid 函数——为什么不能用线性回归做分类

本例演示：
1. 为什么线性回归不适合分类问题（预测值超出 [0,1]）
2. Sigmoid 函数如何把线性预测映射到概率空间
3. 逻辑回归的基本概念

运行方式：python3 chapters/week_10/examples/01_logistic_regression_basics.py
预期输出：
- 散点图展示分类问题（保存为 classification_scatter.png）
- 线性回归预测图（展示问题：预测值超出 [0,1]）
- Sigmoid 函数图（保存为 sigmoid_function.png）
- 控制台输出对比结果
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "week_10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 设置随机种子保证可复现
np.random.seed(42)


def generate_binary_classification_data(n_samples: int = 200) -> pd.DataFrame:
    """
    生成模拟的二分类数据（客户流失场景）

    参数:
        n_samples: 样本数量

    返回:
        包含特征和二分类目标的 DataFrame
    """
    # 生成特征：合同期（月）
    tenure_months = np.random.uniform(1, 72, n_samples)

    # 真实概率：合同期越短，流失概率越高
    # 使用真实的 Sigmoid 关系
    true_prob = 1 / (1 + np.exp(0.15 * (tenure_months - 24)))

    # 生成二分类标签
    churn = np.random.binomial(1, true_prob)

    return pd.DataFrame({
        'tenure_months': tenure_months,
        'churn': churn
    })


def plot_classification_scatter(df: pd.DataFrame) -> None:
    """画分类数据的散点图"""
    plt.figure(figsize=(10, 6))

    # 分别画出两类样本
    churn_no = df[df['churn'] == 0]
    churn_yes = df[df['churn'] == 1]

    plt.scatter(churn_no['tenure_months'], churn_no['churn'],
                alpha=0.5, label='不流失 (Churn=0)', s=80)
    plt.scatter(churn_yes['tenure_months'], churn_yes['churn'],
                alpha=0.5, label='流失 (Churn=1)', s=80, marker='x')

    plt.xlabel('合同期 (月)', fontsize=12)
    plt.ylabel('是否流失', fontsize=12)
    plt.yticks([0, 1])
    plt.title('客户流失数据：二分类问题', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'classification_scatter.png', dpi=150, bbox_inches='tight')
    print("✅ 散点图已保存为 classification_scatter.png")
    plt.close()


def plot_linear_regression_problem(df: pd.DataFrame) -> None:
    """展示线性回归在分类问题上的缺陷"""
    # 拟合线性回归
    X = df[['tenure_months']]
    y = df['churn']

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred_linear = lin_reg.predict(X)

    # 画图
    plt.figure(figsize=(10, 6))

    # 散点图
    churn_no = df[df['churn'] == 0]
    churn_yes = df[df['churn'] == 1]
    plt.scatter(churn_no['tenure_months'], churn_no['churn'],
                alpha=0.5, label='不流失', s=80)
    plt.scatter(churn_yes['tenure_months'], churn_yes['churn'],
                alpha=0.5, label='流失', s=80, marker='x')

    # 线性回归线
    plt.plot(df['tenure_months'], y_pred_linear,
             color='red', linewidth=2, label='线性回归预测')

    # 标注问题区域
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axhspan(1, 1.5, alpha=0.2, color='red', label='超出概率范围')
    plt.axhspan(-0.5, 0, alpha=0.2, color='red')

    plt.xlabel('合同期 (月)', fontsize=12)
    plt.ylabel('是否流失', fontsize=12)
    plt.yticks([0, 1])
    plt.title('线性回归的致命缺陷：预测值超出 [0, 1]', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'linear_regression_problem.png', dpi=150, bbox_inches='tight')
    print("✅ 线性回归问题图已保存为 linear_regression_problem.png")
    plt.close()


def plot_sigmoid_function() -> None:
    """画 Sigmoid 函数图"""
    z = np.linspace(-6, 6, 200)
    sigmoid = 1 / (1 + np.exp(-z))

    plt.figure(figsize=(10, 6))
    plt.plot(z, sigmoid, linewidth=3, color='steelblue', label='Sigmoid(z)')
    plt.axhline(y=0.5, color='red', linestyle='--',
                linewidth=2, label='决策阈值 0.5')
    plt.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5)

    # 标注关键区域
    plt.fill_between(z[z < 0], 0, sigmoid[z < 0],
                     alpha=0.2, color='blue', label='P(y=1) < 0.5')
    plt.fill_between(z[z > 0], 0.5, sigmoid[z > 0],
                     alpha=0.2, color='red', label='P(y=1) > 0.5')

    plt.xlabel('线性得分 z = β₀ + β₁x', fontsize=12)
    plt.ylabel('概率 P(y=1|x)', fontsize=12)
    plt.title('Sigmoid 函数：把任意实数映射到 [0, 1]', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sigmoid_function.png', dpi=150, bbox_inches='tight')
    print("✅ Sigmoid 函数图已保存为 sigmoid_function.png")
    plt.close()


def compare_linear_vs_logistic(df: pd.DataFrame) -> None:
    """对比线性回归和逻辑回归的预测结果"""
    X = df[['tenure_months']]
    y = df['churn']

    # 线性回归
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    y_pred_linear = lin_reg.predict(X)

    # 逻辑回归
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X, y)
    y_proba_logistic = log_reg.predict_proba(X)[:, 1]

    # 打印对比
    print("\n" + "=" * 60)
    print("线性回归 vs 逻辑回归：预测值对比")
    print("=" * 60)

    sample_indices = [0, 50, 100, 150, 199]
    print(f"\n{'样本':<6} {'合同期':<10} {'真实标签':<10} {'线性回归预测':<15} {'逻辑回归预测':<15}")
    print("-" * 60)

    for idx in sample_indices:
        tenure = df.loc[idx, 'tenure_months']
        true_label = df.loc[idx, 'churn']
        linear_pred = y_pred_linear[idx]
        logistic_pred = y_proba_logistic[idx]

        label_str = "流失" if true_label == 1 else "不流失"
        print(f"{idx:<6} {tenure:<10.1f} {label_str:<10} {linear_pred:<15.3f} {logistic_pred:<15.3f}")

    # 问题总结
    print("\n" + "=" * 60)
    print("线性回归的问题：")
    print("=" * 60)
    print(f"  预测值最小值: {y_pred_linear.min():.3f}")
    print(f"  预测值最大值: {y_pred_linear.max():.3f}")
    print(f"  ❌ 问题：概率可以为负数或超过1！")

    print("\n" + "=" * 60)
    print("逻辑回归的优势：")
    print("=" * 60)
    print(f"  预测概率最小值: {y_proba_logistic.min():.3f}")
    print(f"  预测概率最大值: {y_proba_logistic.max():.3f}")
    print(f"  ✅ 所有预测值都在 [0, 1] 范围内！")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("示例1: 为什么不能用线性回归做分类？")
    print("=" * 60)

    # 1. 生成数据
    df = generate_binary_classification_data(n_samples=200)
    print(f"\n📊 数据概览:")
    print(df.head(10))
    print(f"\n流失率: {df['churn'].mean():.2%}")

    # 2. 画散点图
    plot_classification_scatter(df)

    # 3. 展示线性回归的问题
    plot_linear_regression_problem(df)

    # 4. 画 Sigmoid 函数
    plot_sigmoid_function()

    # 5. 对比两种方法
    compare_linear_vs_logistic(df)

    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("""
线性回归的三个致命缺陷：
1. 预测值无界：可以是任意实数，而概率必须在 [0, 1]
2. 误差项假设不成立：二分类残差显然不是正态分布
3. 同方差假设违反：在 x=0.5 处方差最大，在 0 或 1 处方差最小

逻辑回归的解决方案：
- 用 Sigmoid 函数把线性预测压缩到 [0, 1]
- 最小化对数损失（log loss），而非残差平方和
- 输出有意义的概率估计
    """)

    print("\n" + "=" * 60)
    print("✅ 示例1完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
