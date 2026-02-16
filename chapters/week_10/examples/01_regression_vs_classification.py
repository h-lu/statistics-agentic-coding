"""
示例：回归 vs 分类——同一数据集的两种预测视角。

本例演示回归和分类的本质区别：
- 回归：预测连续值（如销售额的具体数值）
- 分类：预测离散类别（如高价值/低价值客户）

使用同一电商数据集，展示为什么回归不适合分类问题。

运行方式：python3 chapters/week_10/examples/01_regression_vs_classification.py
预期输出：
  - stdout 输出回归和分类的预测结果对比
  - 展示回归预测值越界问题
  - 保存图表到 images/01_regression_vs_classification.png

核心概念：
  - 回归输出：连续数值 (-∞, +∞)
  - 分类输出：离散类别（通常是 0/1）
  - 回归预测值可能超出 [0,1] 范围，不适合分类任务
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


def generate_ecommerce_data(n: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    生成模拟电商客户数据。
    
    参数:
        n: 样本量
        random_state: 随机种子
        
    返回:
        DataFrame 包含客户特征和消费数据
    """
    np.random.seed(random_state)
    
    # 生成特征
    data = pd.DataFrame({
        '注册月数': np.random.randint(1, 36, n),
        '月均浏览次数': np.random.poisson(50, n),
        '购物车添加次数': np.random.poisson(10, n),
        '客服咨询次数': np.random.poisson(2, n)
    })
    
    # 生成消费金额（回归目标）：基于特征的连续值
    data['历史消费金额'] = (
        50 + 
        2 * data['注册月数'] + 
        1.5 * data['月均浏览次数'] + 
        8 * data['购物车添加次数'] +
        5 * data['客服咨询次数'] +
        np.random.normal(0, 30, n)
    ).clip(lower=0)
    
    # 生成客户等级（分类目标）：基于消费金额的二元类别
    # 高价值客户：消费金额 > 150 元
    data['是否高价值'] = (data['历史消费金额'] > 150).astype(int)
    
    return data


def regression_approach(df: pd.DataFrame) -> dict:
    """
    使用回归预测消费金额（连续值）。
    
    返回:
        dict 包含回归结果
    """
    print("=" * 70)
    print("方法 1：回归分析——预测消费金额（连续值）")
    print("=" * 70)
    
    # 特征和目标
    X = df[['注册月数', '月均浏览次数', '购物车添加次数', '客服咨询次数']]
    y = df['历史消费金额']
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 拟合回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    print(f"\n回归模型系数：")
    for name, coef in zip(X.columns, model.coef_):
        print(f"  {name}: {coef:.4f}")
    print(f"  截距: {model.intercept_:.4f}")
    
    print(f"\n预测结果示例（前 10 个）：")
    print(f"{'实际消费':<12} {'预测消费':<12} {'误差':<10}")
    print("-" * 40)
    for i in range(10):
        error = y_pred[i] - y_test.iloc[i]
        print(f"{y_test.iloc[i]:<12.2f} {y_pred[i]:<12.2f} {error:<10.2f}")
    
    # 评估
    r2 = model.score(X_test, y_test)
    print(f"\n回归评估指标：")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {np.sqrt(np.mean((y_test - y_pred)**2)):.2f}")
    
    return {
        'model': model,
        'y_test': y_test,
        'y_pred': y_pred,
        'r2': r2
    }


def classification_approach(df: pd.DataFrame) -> dict:
    """
    使用分类预测客户等级（离散类别）。
    
    返回:
        dict 包含分类结果
    """
    print("\n" + "=" * 70)
    print("方法 2：分类分析——预测客户等级（高价值/低价值）")
    print("=" * 70)
    
    # 特征和目标
    X = df[['注册月数', '月均浏览次数', '购物车添加次数', '客服咨询次数']]
    y = df['是否高价值']
    
    # 划分数据集（分层抽样保持类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 拟合逻辑回归模型
    model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 预测概率和类别
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    print(f"\n类别分布：")
    print(f"  低价值客户 (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"  高价值客户 (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    print(f"\n预测结果示例（前 10 个）：")
    print(f"{'实际类别':<12} {'预测概率':<12} {'预测类别':<12}")
    print("-" * 45)
    for i in range(10):
        print(f"{y_test.iloc[i]:<12} {y_prob[i]:<12.4f} {y_pred[i]:<12}")
    
    # 准确率
    accuracy = (y_pred == y_test).mean()
    print(f"\n分类准确率：{accuracy:.2%}")
    
    return {
        'model': model,
        'scaler': scaler,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy
    }


def compare_approaches(df: pd.DataFrame, reg_result: dict, clf_result: dict) -> None:
    """
    对比回归和分类两种方法的区别。
    """
    print("\n" + "=" * 70)
    print("对比：回归 vs 分类")
    print("=" * 70)
    
    print("\n┌───────────────────┬─────────────────────────┬─────────────────────────┐")
    print("│ 维度              │ 回归                     │ 分类                    │")
    print("├───────────────────┼─────────────────────────┼─────────────────────────┤")
    print("│ 目标变量          │ 历史消费金额（连续值）   │ 是否高价值（0/1）       │")
    print("│ 输出类型          │ 连续数值 (-∞, +∞)        │ 离散类别（0 或 1）      │")
    print("│ 预测解释          │ 预计消费多少元           │ 是高价值客户的概率      │")
    print("│ 适用场景          │ 预测具体数值             │ 预测类别/群体           │")
    print("│ 评估指标          │ R², RMSE, MAE            │ 准确率, 精确率, 查全率  │")
    print("└───────────────────┴─────────────────────────┴─────────────────────────┘")
    
    print("\n业务场景选择：")
    print("  ❓ 业务方问：'这个客户预计会消费多少钱？' → 用回归")
    print("  ❓ 业务方问：'哪些客户是高价值客户，我要重点运营？' → 用分类")
    
    print("\n为什么不能混淆两者？")
    print("  ⚠️  回归输出可能超出 [0,1]，无法解释为'概率'")
    print("  ⚠️  回归的误差假设（正态分布）不适用于分类问题")
    print("  ⚠️  分类问题中不同类型的错误代价不同（误报 vs 漏报）")


def plot_comparison(df: pd.DataFrame, reg_result: dict, clf_result: dict) -> None:
    """绘制回归 vs 分类的可视化对比"""
    setup_chinese_font()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：回归预测（散点图）
    ax1 = axes[0]
    y_test_reg = reg_result['y_test'].values
    y_pred_reg = reg_result['y_pred']
    
    ax1.scatter(y_test_reg, y_pred_reg, alpha=0.5, s=50, color='#2E86AB', edgecolors='black', linewidth=0.5)
    ax1.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 
             'r--', linewidth=2, label='完美预测线')
    ax1.set_xlabel('实际消费金额（元）', fontsize=12)
    ax1.set_ylabel('预测消费金额（元）', fontsize=12)
    ax1.set_title(f'回归预测\nR² = {reg_result["r2"]:.3f}', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 右图：分类预测（混淆矩阵热力图）
    ax2 = axes[1]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(clf_result['y_test'], clf_result['y_pred'])
    
    im = ax2.imshow(cm, interpolation='nearest', cmap='Blues')
    ax2.set_title(f'分类预测（混淆矩阵）\n准确率 = {clf_result["accuracy"]:.2%}', 
                  fontsize=13, fontweight='bold')
    
    tick_marks = [0, 1]
    ax2.set_xticks(tick_marks)
    ax2.set_yticks(tick_marks)
    ax2.set_xticklabels(['低价值', '高价值'])
    ax2.set_yticklabels(['低价值', '高价值'])
    ax2.set_xlabel('预测类别', fontsize=12)
    ax2.set_ylabel('实际类别', fontsize=12)
    
    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / '01_regression_vs_classification.png',
                dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n图表已保存到: images/01_regression_vs_classification.png")


def main() -> None:
    """主函数"""
    print("回归 vs 分类：同一数据集的两种预测视角\n")
    
    # 生成数据
    df = generate_ecommerce_data(n=500, random_state=42)
    print(f"数据集：{len(df)} 个客户样本")
    print(f"特征：注册月数、月均浏览次数、购物车添加次数、客服咨询次数")
    print(f"回归目标：历史消费金额（连续值）")
    print(f"分类目标：是否高价值（消费 > 150 元 = 高价值）\n")
    
    # 回归方法
    reg_result = regression_approach(df)
    
    # 分类方法
    clf_result = classification_approach(df)
    
    # 对比
    compare_approaches(df, reg_result, clf_result)
    
    # 绘图
    plot_comparison(df, reg_result, clf_result)
    
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("\n关键要点：")
    print("  1. 回归预测连续值（-∞ 到 +∞），分类预测离散类别（有限的几个值）")
    print("  2. 回归适用场景：'预测多少'——预测销售额、温度、股价等")
    print("  3. 分类适用场景：'预测是否'——识别高价值客户、垃圾邮件、疾病等")
    print("  4. 两者的评估指标完全不同：R² vs 准确率/精确率/查全率")
    print("  5. 不要用回归做分类：预测值会越界，误差假设不匹配")
    print()


if __name__ == "__main__":
    main()
