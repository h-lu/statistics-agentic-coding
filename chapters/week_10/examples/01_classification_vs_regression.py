"""
示例：分类 vs 回归对比——预测类别 vs 预测连续值

运行方式：python3 chapters/week_10/examples/01_classification_vs_regression.py
预期输出：对比回归和分类的预测结果，展示为什么线性回归不适合分类问题
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# 配置中文字体
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


def generate_binary_data(n_samples: int = 100, random_state: int = 42) -> tuple:
    """生成二元分类数据：基于年龄预测是否购买"""
    np.random.seed(random_state)
    age = np.random.uniform(18, 70, n_samples)
    # 真实的概率随年龄增加
    true_prob = 1 / (1 + np.exp(-(age - 40) / 10))
    purchase = (np.random.random(n_samples) < true_prob).astype(int)
    return age.reshape(-1, 1), purchase


def bad_linear_regression_for_classification() -> None:
    """
    错误示范：用线性回归做分类

    问题：
    1. 预测值可能超出 [0, 1] 范围
    2. 假设残差正态，但 0/1 数据的残差不可能正态
    3. 对异常值敏感
    """
    print("=" * 60)
    print("错误示范：用线性回归做分类")
    print("=" * 60)

    X, y = generate_binary_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 线性回归
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # 预测
    y_pred = lin_reg.predict(X_test)

    print("\n线性回归预测值（前 10 个）：")
    for i in range(10):
        print(f"  年龄 {X_test[i][0]:.1f} -> 预测值 {y_pred[i]:.3f} (实际: {y_test[i]})")

    # 统计超出范围的预测
    out_of_range = (y_pred < 0) | (y_pred > 1)
    print(f"\n超出 [0, 1] 范围的预测: {out_of_range.sum()} / {len(y_pred)}")
    print(f"  最小预测值: {y_pred.min():.3f}")
    print(f"  最大预测值: {y_pred.max():.3f}")

    # 绘图
    font = setup_chinese_font()
    fig, ax = plt.subplots(figsize=(10, 6))

    # 画数据点
    ax.scatter(X_test, y_test, alpha=0.5, label='实际数据')

    # 画回归线
    X_plot = np.linspace(18, 70, 100).reshape(-1, 1)
    y_plot = lin_reg.predict(X_plot)
    ax.plot(X_plot, y_plot, 'r-', linewidth=2, label='线性回归预测')

    # 标注问题区域
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax.text(25, -0.15, '预测 < 0（无意义）', fontsize=10, color='red')
    ax.text(50, 1.05, '预测 > 1（无意义）', fontsize=10, color='red')

    ax.set_xlabel('年龄')
    ax.set_ylabel('是否购买 (0/1)')
    ax.set_title('线性回归做分类的问题：预测值超出 [0, 1] 范围')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.2)

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'classification_vs_regression.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n图片已保存到: {output_dir / 'classification_vs_regression.png'}")


def good_logistic_regression() -> None:
    """
    正确做法：用逻辑回归做分类

    优势：
    1. 预测概率值在 [0, 1] 范围内
    2. Sigmoid 函数将线性预测映射为概率
    3. 输出可以解释为概率
    """
    print("\n" + "=" * 60)
    print("正确做法：用逻辑回归做分类")
    print("=" * 60)

    X, y = generate_binary_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 逻辑回归
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train, y_train)

    # 预测概率
    y_prob = log_reg.predict_proba(X_test)[:, 1]

    print("\n逻辑回归预测概率（前 10 个）：")
    for i in range(10):
        print(f"  年龄 {X_test[i][0]:.1f} -> 预测概率 {y_prob[i]:.3f} (实际: {y_test[i]})")

    # 统计范围
    print(f"\n概率范围: [{y_prob.min():.3f}, {y_prob.max():.3f}]")
    print(f"所有概率值都在 [0, 1] 范围内: {(y_prob >= 0).all() and (y_prob <= 1).all()}")

    # 打印系数
    print(f"\n逻辑回归系数: 截距={log_reg.intercept_[0]:.3f}, 斜率={log_reg.coef_[0][0]:.3f}")
    print("解读：年龄每增加 1 岁，购买的对数几率增加约 {:.3f}".format(log_reg.coef_[0][0]))


def compare_predictions() -> None:
    """对比回归和分类的输出差异"""
    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)

    print("""
    回归 vs 分类对比：

    | 维度       | 回归 (Regression)    | 分类 (Classification)  |
    |-----------|---------------------|----------------------|
    | 预测目标   | 连续值               | 类别标签              |
    | 输出示例   | 销售额 35.6 万       | 会流失 / 不会流失     |
    | 评估指标   | MSE, RMSE, R²        | 准确率, 精确率, 召回率 |
    | 常用算法   | 线性回归             | 逻辑回归              |
    | 输出范围   | (-∞, +∞)            | [0, 1] (概率)         |

    关键差异：
    - 回归：预测"多少"（数量、金额、温度等）
    - 分类：预测"是或否"（类别、标签、决策等）

    什么时候用分类？
    - 目标变量是离散的类别（流失/不流失、购买/不购买）
    - 需要预测概率，而不仅仅是数值
    - 需要评估"抓到多少"（召回率）和"误判多少"（精确率）
    """)


def main() -> None:
    """主函数"""
    from pathlib import Path
    bad_linear_regression_for_classification()
    good_logistic_regression()
    compare_predictions()


if __name__ == "__main__":
    main()
