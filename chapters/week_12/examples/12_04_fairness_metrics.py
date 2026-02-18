"""
示例：公平性指标计算——三种公平性定义及其权衡

本例演示：
1. 统计均等（Demographic Parity）：各群体的预测正率相同
2. 机会均等（Equalized Odds）：各群体的真阳性率、假阳性率相同
3. 校准（Calibration）：各群体的预测概率与真实概率匹配
4. 公平性-准确性权衡

运行方式：python3 chapters/week_12/examples/12_04_fairness_metrics.py
预期输出：stdout 输出公平性指标对比 + 保存图表到 output/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


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


def generate_fairness_scenario(n_samples: int = 2000, random_state: int = 42) -> tuple:
    """
    生成用于公平性评估的场景数据

    场景设置：
    - 群体 A：60%，真实流失率 20%
    - 群体 B：40%，真实流失率 25%
    - 这种真实差异可能导致"公平的模型"产生"不公平的结果"
    """
    rng = np.random.default_rng(random_state)

    n = n_samples

    # 敏感属性：群体（0=A, 1=B）
    group = rng.binomial(1, 0.4, n)

    # 其他特征
    days = rng.poisson(30, n)
    count = rng.poisson(5, n)
    spend = rng.exponential(100, n)
    vip = rng.binomial(1, 0.3, n)

    # 生成目标：群体 B 的真实风险略高
    base_logit = -2 + 0.05 * days - 0.2 * count - 0.01 * spend - 1.0 * vip

    # 群体 B 的流失风险略高（+0.3 log-odds）
    group_effect = 0.3 * group

    logit = base_logit + group_effect
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)

    X = pd.DataFrame({
        'days_since_last_purchase': days,
        'purchase_count': count,
        'avg_spend': spend,
        'vip_status': vip,
        'group': group  # 0=A, 1=B
    })
    y = pd.Series(churn, name='churn')

    return X, y


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group: np.ndarray
) -> dict:
    """
    计算三种公平性指标
    """
    from sklearn.metrics import confusion_matrix

    results = {'group_0': {}, 'group_1': {}, 'differences': {}}

    for g in [0, 1]:
        mask = group == g

        if mask.sum() < 10:
            continue

        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        y_prob_g = y_prob[mask]

        cm = confusion_matrix(y_true_g, y_pred_g)
        tn, fp, fn, tp = cm.ravel()

        results[f'group_{g}'] = {
            'count': mask.sum(),
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'positive_rate': y_pred_g.mean(),
            'true_churn_rate': y_true_g.mean(),
            'avg_predicted_prob': y_prob_g.mean()
        }

    # 计算差异
    g0 = results['group_0']
    g1 = results['group_1']

    results['differences'] = {
        'tpr_diff': abs(g1['true_positive_rate'] - g0['true_positive_rate']),
        'fpr_diff': abs(g1['false_positive_rate'] - g0['false_positive_rate']),
        'positive_rate_diff': abs(g1['positive_rate'] - g0['positive_rate']),
        'calibration_diff_0': abs(g0['avg_predicted_prob'] - g0['true_churn_rate']),
        'calibration_diff_1': abs(g1['avg_predicted_prob'] - g1['true_churn_rate'])
    }

    return results


def apply_demographic_parity_threshold(
    y_prob: np.ndarray,
    group: np.ndarray,
    target_rate: float
) -> np.ndarray:
    """
    应用统计均等约束：调整阈值使各群体预测正率相同

    这是一个简化的后处理方法
    """
    y_pred = np.zeros_like(y_prob, dtype=int)

    for g in [0, 1]:
        mask = group == g
        prob_g = y_prob[mask]

        # 找到能使预测正率等于 target_rate 的阈值
        thresholds = np.sort(prob_g)
        for t in thresholds:
            if (prob_g >= t).mean() <= target_rate:
                threshold = t
                break
        else:
            threshold = 1.0

        y_pred[mask] = (prob_g >= threshold).astype(int)

    return y_pred


def plot_fairness_tradeoff(
    original_metrics: dict,
    adjusted_metrics: dict,
    original_accuracy: float,
    adjusted_accuracy: float,
    output_dir: Path
) -> None:
    """绘制公平性-准确性权衡图"""
    font = setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. 公平性指标对比
    metrics_names = ['真阳性率差异', '假阳性率差异', '预测正率差异']
    original_values = [
        original_metrics['differences']['tpr_diff'],
        original_metrics['differences']['fpr_diff'],
        original_metrics['differences']['positive_rate_diff']
    ]
    adjusted_values = [
        adjusted_metrics['differences']['tpr_diff'],
        adjusted_metrics['differences']['fpr_diff'],
        adjusted_metrics['differences']['positive_rate_diff']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    axes[0].bar(x - width/2, original_values, width, label='原始模型',
               color='coral', alpha=0.7)
    axes[0].bar(x + width/2, adjusted_values, width, label='公平性调整后',
               color='steelblue', alpha=0.7)

    axes[0].set_ylabel('差异值')
    axes[0].set_title('公平性指标对比（越小越公平）')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics_names)
    axes[0].legend()
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, (orig, adj) in enumerate(zip(original_values, adjusted_values)):
        axes[0].text(i - width/2, orig + 0.01, f'{orig:.3f}',
                    ha='center', va='bottom', fontsize=9)
        axes[0].text(i + width/2, adj + 0.01, f'{adj:.3f}',
                    ha='center', va='bottom', fontsize=9)

    # 2. 准确率 vs 公平性
    scenarios = ['原始模型', '公平性调整后']
    accuracies = [original_accuracy, adjusted_accuracy]
    fairness_scores = [
        np.mean([original_metrics['differences']['tpr_diff'],
                original_metrics['differences']['fpr_diff']]),
        np.mean([adjusted_metrics['differences']['tpr_diff'],
                adjusted_metrics['differences']['fpr_diff']])
    ]

    colors_scatter = ['coral', 'steelblue']
    for i, (acc, fair) in enumerate(zip(accuracies, fairness_scores)):
        axes[1].scatter(fair, acc, s=200, c=colors_scatter[i], alpha=0.7,
                       label=scenarios[i], edgecolors='black', linewidth=1.5)
        axes[1].annotate(scenarios[i], (fair, acc),
                        xytext=(10, 0), textcoords='offset points',
                        fontsize=10, va='center')

    axes[1].set_xlabel('公平性得分（平均差异，越小越公平）')
    axes[1].set_ylabel('准确率')
    axes[1].set_title('公平性-准确性权衡')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_dir / 'fairness_tradeoff.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"公平性-准确性权衡图已保存到: {output_dir / 'fairness_tradeoff.png'}")


def plot_calibration_comparison(
    original_metrics: dict,
    adjusted_metrics: dict,
    output_dir: Path
) -> None:
    """绘制校准对比图"""
    font = setup_chinese_font()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    groups = ['群体 A', '群体 B']

    # 1. 原始模型的校准
    x = np.arange(len(groups))
    width = 0.35

    original_true = [
        original_metrics['group_0']['true_churn_rate'],
        original_metrics['group_1']['true_churn_rate']
    ]
    original_pred = [
        original_metrics['group_0']['avg_predicted_prob'],
        original_metrics['group_1']['avg_predicted_prob']
    ]

    axes[0].bar(x - width/2, original_true, width, label='真实流失率',
               color='lightcoral', alpha=0.7)
    axes[0].bar(x + width/2, original_pred, width, label='平均预测概率',
               color='steelblue', alpha=0.7)

    axes[0].set_ylabel('概率')
    axes[0].set_title('原始模型：校准检查')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups)
    axes[0].legend()
    axes[0].set_ylim(0, 0.5)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 2. 调整后的校准
    adjusted_true = [
        adjusted_metrics['group_0']['true_churn_rate'],
        adjusted_metrics['group_1']['true_churn_rate']
    ]
    adjusted_pred = [
        adjusted_metrics['group_0']['avg_predicted_prob'],
        adjusted_metrics['group_1']['avg_predicted_prob']
    ]

    axes[1].bar(x - width/2, adjusted_true, width, label='真实流失率',
               color='lightcoral', alpha=0.7)
    axes[1].bar(x + width/2, adjusted_pred, width, label='平均预测概率',
               color='steelblue', alpha=0.7)

    axes[1].set_ylabel('概率')
    axes[1].set_title('调整后：校准检查')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups)
    axes[1].legend()
    axes[1].set_ylim(0, 0.5)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图片
    plt.savefig(output_dir / 'calibration_comparison.png',
                dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"校准对比图已保存到: {output_dir / 'calibration_comparison.png'}")


def main() -> None:
    """主函数"""
    print("=" * 60)
    print("公平性指标计算与权衡演示")
    print("=" * 60)

    # 创建输出目录
    output_dir = Path(__file__).parent.parent.parent.parent / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 生成场景数据
    print("\n1. 生成公平性评估场景数据...")
    print("   场景：群体 A（60%）和群体 B（40%）的真实流失率不同")
    X, y = generate_fairness_scenario()

    print(f"   数据规模: {X.shape[0]} 行")
    print(f"   群体 A 占比: {(X['group'] == 0).mean():.1%}")
    print(f"   群体 B 占比: {(X['group'] == 1).mean():.1%}")

    # 2. 训练模型
    print("\n2. 训练模型...")
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop('group', axis=1), y, test_size=0.3, random_state=42, stratify=y
    )
    group_test = X.loc[X_test.index, 'group'].values

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    original_accuracy = accuracy_score(y_test, y_pred)

    # 3. 计算原始模型的公平性指标
    print("\n3. 计算原始模型的公平性指标...")
    original_metrics = compute_fairness_metrics(
        y_test.values, y_pred, y_prob, group_test
    )

    print("\n" + "=" * 60)
    print("原始模型的公平性指标")
    print("=" * 60)

    print("\n群体 A:")
    for k, v in original_metrics['group_0'].items():
        if k != 'count':
            print(f"  {k}: {v:.4f}")

    print("\n群体 B:")
    for k, v in original_metrics['group_1'].items():
        if k != 'count':
            print(f"  {k}: {v:.4f}")

    print("\n群体差异:")
    for k, v in original_metrics['differences'].items():
        print(f"  {k}: {v:.4f}")

    # 4. 应用统计均等约束
    print("\n4. 应用统计均等约束...")
    target_rate = 0.2  # 目标预测正率
    y_pred_fair = apply_demographic_parity_threshold(y_prob, group_test, target_rate)

    adjusted_accuracy = accuracy_score(y_test, y_pred_fair)

    # 计算调整后的公平性指标
    adjusted_metrics = compute_fairness_metrics(
        y_test.values, y_pred_fair, y_prob, group_test
    )

    print("\n调整后的公平性指标:")
    print("\n群体差异:")
    for k, v in adjusted_metrics['differences'].items():
        print(f"  {k}: {v:.4f}")

    # 5. 公平性-准确性权衡
    print("\n5. 公平性-准确性权衡...")
    print(f"\n原始模型准确率: {original_accuracy:.4f}")
    print(f"调整后准确率: {adjusted_accuracy:.4f}")
    print(f"准确率下降: {original_accuracy - adjusted_accuracy:.4f} ({(original_accuracy - adjusted_accuracy)/original_accuracy*100:.1f}%)")

    # 6. 绘制图表
    print("\n6. 生成图表...")
    plot_fairness_tradeoff(
        original_metrics, adjusted_metrics,
        original_accuracy, adjusted_accuracy, output_dir
    )
    plot_calibration_comparison(original_metrics, adjusted_metrics, output_dir)

    # 7. 总结三种公平性定义
    print("\n" + "=" * 60)
    print("三种公平性定义总结")
    print("=" * 60)
    print("""
1. 统计均等（Demographic Parity）
   定义: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
   含义: 各群体的预测正率相同
   问题: 忽略真实风险差异
   适用场景: 营销定向（避免资源分配不公）

2. 机会均等（Equalized Odds）
   定义: TPR(A=0) = TPR(A=1) 且 FPR(A=0) = FPR(A=1)
   含义: 各群体的真阳性率、假阳性率相同
   问题: 可能无法同时满足
   适用场景: 信贷审批、医疗筛查

3. 校准（Calibration）
   定义: P(Y=1|Ŷ=p, A=0) = P(Y=1|Ŷ=p, A=1)
   含义: 各群体的预测概率与真实概率匹配
   问题: 可能掩盖分配不公
   适用场景: 风险评估

重要结论:
- 没有三种定义都满足的"完美公平"模型
- 公平性工程不是"消除偏见"（不可能），而是"管理偏见"（权衡）
- 需要根据业务场景选择合适的公平性定义
    """)

    print("\n生成的文件:")
    print(f"  1. {output_dir / 'fairness_tradeoff.png'}")
    print(f"  2. {output_dir / 'calibration_comparison.png'}")


if __name__ == "__main__":
    main()
