"""
示例：模型偏见检测——按敏感属性分组评估

本例演示：
1. 如何按敏感属性（性别、地区等）分组评估模型
2. 通过混淆矩阵分解发现不公平现象
3. 数据偏见 vs 算法偏见

运行方式：python3 chapters/week_12/examples/12_03_bias_detection.py
预期输出：stdout 输出分组评估结果 + 保存图表到 output/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


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


def generate_biased_data(n_samples: int = 2000, random_state: int = 42) -> tuple:
    """
    生成包含偏见的数据

    数据偏见示例：
    - 某个地区（region_B）的客户历史上获得的服务较少
    - 但这与真实风险无关，只是历史上的分配不公
    """
    rng = np.random.default_rng(random_state)

    # 生成特征
    n = n_samples

    # 地区：A（60%）、B（20%）、C（20%）
    region = rng.choice(['A', 'B', 'C'], n, p=[0.6, 0.2, 0.2])

    # 性别：0（女，50%）、1（男，50%）
    gender = rng.binomial(1, 0.5, n)

    # 其他特征
    days = rng.poisson(30, n)
    count = rng.poisson(5, n)
    spend = rng.exponential(100, n)
    vip = rng.binomial(1, 0.3, n)

    # 生成目标
    # 真实风险只与这些特征相关，与地区和性别无关
    logit = -2 + 0.05 * days - 0.2 * count - 0.01 * spend - 1.0 * vip
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    # 但是！我们在训练数据中加入"数据偏见"
    # 历史上，region B 的客户流失被标记得更频繁（因为服务差，不是风险高）
    # 这是一个"历史偏见"被编码进标签的例子
    region_b_mask = region == 'B'
    churn[region_b_mask] = rng.binomial(1, prob[region_b_mask] * 1.3)  # 提高区域 B 的标签流失率

    X = pd.DataFrame({
        'days_since_last_purchase': days,
        'purchase_count': count,
        'avg_spend': spend,
        'vip_status': vip,
        'region': region,
        'gender': gender
    })
    y = pd.Series(churn, name='churn')

    return X, y


def train_model_with_bias(X: pd.DataFrame, y: pd.Series) -> tuple:
    """训练模型（会学到数据中的偏见）"""
    # One-hot 编码分类型特征
    X_encoded = pd.get_dummies(X, columns=['region'], drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    return rf, X_train, X_test, y_train, y_test


def fairness_evaluation(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    sensitive_attr: pd.Series
) -> pd.DataFrame:
    """
    按敏感属性分组评估模型

    参数:
    - y_true: 真实标签
    - y_pred: 预测标签
    - y_prob: 预测概率
    - sensitive_attr: 敏感属性（如性别、地区）

    返回:
    - 分组评估结果 DataFrame
    """
    df = pd.DataFrame({
        'y_true': y_true.values,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'sensitive': sensitive_attr.values
    })

    results = []

    for group in sorted(df['sensitive'].unique()):
        group_df = df[df['sensitive'] == group]

        if len(group_df) < 10:
            continue

        cm = confusion_matrix(group_df['y_true'], group_df['y_pred'])
        tn, fp, fn, tp = cm.ravel()

        results.append({
            'group': group,
            'count': len(group_df),
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'positive_rate': group_df['y_pred'].mean(),
            'avg_predicted_prob': group_df['y_prob'].mean(),
            'true_churn_rate': group_df['y_true'].mean(),
            'accuracy': accuracy_score(group_df['y_true'], group_df['y_pred'])
        })

    return pd.DataFrame(results)


def plot_group_confusion_matrices(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive_attr: pd.Series,
    output_dir: Path
) -> None:
    """绘制分组的混淆矩阵"""
    font = setup_chinese_font()

    groups = sorted(sensitive_attr.unique())
    n_groups = len(groups)

    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4))
    if n_groups == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        mask = sensitive_attr == group
        cm = confusion_matrix(y_true[mask], y_pred[mask])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['非流失', '流失'],
                   yticklabels=['非流失', '流失'])
        ax.set_title(f'{group} 组混淆矩阵')
        ax.set_xlabel('预测')
        ax.set_ylabel('真实')

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_dir / 'group_confusion_matrices.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"分组混淆矩阵图已保存到: {output_dir / 'group_confusion_matrices.png'}")


def plot_fairness_comparison(
    fairness_results: pd.DataFrame,
    sensitive_attr_name: str,
    output_dir: Path
) -> None:
    """绘制公平性对比图"""
    font = setup_chinese_font()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [
        ('true_positive_rate', '真阳性率（召回率）'),
        ('false_positive_rate', '假阳性率'),
        ('positive_rate', '预测正率'),
        ('accuracy', '准确率')
    ]

    colors = plt.cm.Set2(range(len(fairness_results)))

    for ax, (metric, title) in zip(axes.flat, metrics):
        bars = ax.bar(fairness_results['group'].astype(str),
                     fairness_results[metric],
                     color=colors, alpha=0.7)

        ax.set_ylabel(title)
        ax.set_title(f'{title} 按 {sensitive_attr_name} 分组')
        ax.set_ylim(0, 1)

        # 添加数值标签
        for bar, val in zip(bars, fairness_results[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        # 添加平均线
        avg = fairness_results[metric].mean()
        ax.axhline(y=avg, color='red', linestyle='--', linewidth=1,
                  label=f'平均值: {avg:.3f}')
        ax.legend()

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_dir / 'fairness_comparison.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"公平性对比图已保存到: {output_dir / 'fairness_comparison.png'}")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("模型偏见检测演示")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成包含偏见的数据
    print("\n1. 生成包含'历史偏见'的数据...")
    print("   说明: region B 的客户在历史上被标记为流失的频率更高")
    print("         但这不是真实风险差异，只是历史上的服务不公")
    X, y = generate_biased_data()

    # 2. 训练模型（会学到偏见）
    print("\n2. 训练模型...")
    rf, X_train, X_test, y_train, y_test = train_model_with_bias(X, y)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print(f"   整体准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   整体召回率: {recall_score(y_test, y_pred):.4f}")

    # 3. 按地区分组评估
    print("\n3. 按地区分组评估...")
    # 从 X_test 中恢复地区信息（One-Hot 之前）
    # 我们需要从原始 X 中获取地区信息
    X_test_original = X.loc[X_test.index]
    region_test = X_test_original['region']

    region_results = fairness_evaluation(y_test, y_pred, y_prob, region_test)

    print("\n" + "=" * 60)
    print("按地区分组的评估结果")
    print("=" * 60)
    print(region_results.to_string(index=False))

    # 4. 分析偏见
    print("\n" + "=" * 60)
    print("偏见分析")
    print("=" * 60)

    tpr_diff = region_results['true_positive_rate'].max() - region_results['true_positive_rate'].min()
    fpr_diff = region_results['false_positive_rate'].max() - region_results['false_positive_rate'].min()

    print(f"\n真阳性率差异: {tpr_diff:.3f}")
    print(f"假阳性率差异: {fpr_diff:.3f}")

    if tpr_diff > 0.1 or fpr_diff > 0.1:
        print("\n警告: 检测到显著的分组差异！")
        print("这可能是数据偏见（历史不公）被模型学到并放大的结果。")
    else:
        print("\n分组差异在可接受范围内。")

    # 5. 按性别分组评估
    print("\n4. 按性别分组评估...")
    gender_test = X_test_original['gender']
    gender_results = fairness_evaluation(y_test, y_pred, y_prob, gender_test)

    print("\n" + "=" * 60)
    print("按性别分组的评估结果")
    print("=" * 60)
    print(gender_results.to_string(index=False))

    # 6. 绘制图表
    print("\n5. 生成图表...")
    plot_group_confusion_matrices(y_test, y_pred, region_test, output_dir)
    plot_fairness_comparison(region_results, '地区', output_dir)

    # 7. 解释数据偏见 vs 算法偏见
    print("\n" + "=" * 60)
    print("数据偏见 vs 算法偏见")
    print("=" * 60)
    print("""
1. 数据偏见（Data Bias）
   - 训练数据本身反映了历史偏见
   - 示例：历史上 region B 的客户被标记为流失的频率更高
         但这不是真实风险差异，只是服务不公

2. 算法偏见（Algorithmic Bias）
   - 算法可能放大数据中的偏见
   - 示例：随机森林可能"过分关注"地区特征，导致对 region B
         的客户系统性高估流失风险

3. 如何判断？
   - 按敏感属性分组评估
   - 比较各群体的真阳性率、假阳性率
   - 如果差异显著，需要调查数据来源或考虑后处理校准

4. 校准检查
   - 比较各群体的"真实流失率"和"预测流失率"
   - 如果真实率相同但预测率不同 -> 模型有偏见
   - 如果真实率不同且预测率也不同 -> 模型可能校准良好
    """)

    # 检查校准
    print("\n校准检查:")
    for _, row in region_results.iterrows():
        calibration_diff = row['avg_predicted_prob'] - row['true_churn_rate']
        print(f"  {row['group']} 组: 预测流失率={row['avg_predicted_prob']:.3f}, "
              f"真实流失率={row['true_churn_rate']:.3f}, "
              f"差异={calibration_diff:.3f}")

    print("\n生成的文件:")
    print(f"  1. {output_dir / 'group_confusion_matrices.png'}")
    print(f"  2. {output_dir / 'fairness_comparison.png'}")


if __name__ == "__main__":
    main()
