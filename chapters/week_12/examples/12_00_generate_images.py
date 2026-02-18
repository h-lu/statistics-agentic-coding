#!/usr/bin/env python3
"""
Week 12 图片生成脚本
生成 SHAP 和公平性评估的示意图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

import os
os.makedirs('chapters/week_12/images', exist_ok=True)

# ============================================
# 图 1: SHAP 汇总图示意
# ============================================
print("生成 SHAP 汇总图...")

np.random.seed(42)
n_samples = 200
n_features = 6
feature_names = ['days_since_purchase', 'purchase_count', 'avg_spend',
                 'vip_status', 'support_calls', 'tenure_months']

# 生成模拟 SHAP 值
shap_data = []
for i, feat in enumerate(feature_names):
    # 每个特征的 SHAP 值分布不同
    base_value = np.random.uniform(-0.3, 0.3)
    spread = np.random.uniform(0.1, 0.4)
    feat_shap = np.random.normal(base_value, spread, n_samples)
    feat_values = np.random.uniform(0, 1, n_samples)  # 特征值（归一化）

    for j in range(n_samples):
        shap_data.append({
            'feature': feat,
            'shap_value': feat_shap[j],
            'feature_value': feat_values[j]
        })

df_shap = pd.DataFrame(shap_data)

# 绘制 SHAP 汇总图
fig, ax = plt.subplots(figsize=(10, 6))

colors = plt.cm.coolwarm(df_shap['feature_value'])

for i, feat in enumerate(feature_names):
    feat_data = df_shap[df_shap['feature'] == feat]
    y_positions = np.full(len(feat_data), i) + np.random.uniform(-0.2, 0.2, len(feat_data))

    scatter = ax.scatter(
        feat_data['shap_value'],
        y_positions,
        c=feat_data['feature_value'],
        cmap='coolwarm',
        alpha=0.6,
        s=30,
        vmin=0, vmax=1
    )

ax.set_yticks(range(len(feature_names)))
ax.set_yticklabels(feature_names)
ax.set_xlabel('SHAP Value (impact on churn probability)')
ax.set_title('SHAP Summary Plot: Feature Importance Distribution')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Feature Value\n(Low → High)')

plt.tight_layout()
plt.savefig('chapters/week_12/images/shap_summary_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: shap_summary_plot.png")

# ============================================
# 图 2: SHAP 瀑布图示意（单样本解释）
# ============================================
print("生成 SHAP 瀑布图...")

fig, ax = plt.subplots(figsize=(10, 6))

# 模拟单个样本的 SHAP 值
sample_shap = {
    'days_since_purchase': 0.28,
    'purchase_count': -0.12,
    'avg_spend': 0.08,
    'vip_status': -0.15,
    'support_calls': 0.05,
    'tenure_months': -0.03
}

features = list(sample_shap.keys())
values = list(sample_shap.values())
baseline = 0.0  # log-odds 基线
colors_bar = ['#e74c3c' if v > 0 else '#3498db' for v in values]

# 计算累积位置
positions = np.cumsum([baseline] + values[:-1])

y_pos = np.arange(len(features))
bars = ax.barh(y_pos, values, left=positions, color=colors_bar, alpha=0.8, height=0.6)

ax.set_yticks(y_pos)
ax.set_yticklabels(features)
ax.set_xlabel('Contribution to Prediction (log-odds)')
ax.set_title('SHAP Waterfall Plot: Single Prediction Explanation')
ax.axvline(x=baseline, color='gray', linestyle='--', alpha=0.5, label='Baseline')

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, values)):
    width = bar.get_width()
    label_x = bar.get_x() + width + 0.02
    ax.text(label_x, bar.get_y() + bar.get_height()/2,
            f'{val:+.2f}', va='center', fontsize=10)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', alpha=0.8, label='Increases churn risk'),
    Patch(facecolor='#3498db', alpha=0.8, label='Decreases churn risk')
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig('chapters/week_12/images/shap_waterfall_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: shap_waterfall_plot.png")

# ============================================
# 图 3: 分组混淆矩阵对比
# ============================================
print("生成分组混淆矩阵对比图...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 男性组
cm_male = np.array([[85, 15], [20, 80]])
im1 = axes[0].imshow(cm_male, cmap='Blues')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Predicted: No Churn', 'Predicted: Churn'])
axes[0].set_yticklabels(['Actual: No Churn', 'Actual: Churn'])
axes[0].set_title('Confusion Matrix: Male Group')

# 添加数值标签
for i in range(2):
    for j in range(2):
        text = axes[0].text(j, i, cm_male[i, j],
                          ha="center", va="center", color="white" if cm_male[i, j] > 50 else "black",
                          fontsize=14)

# 女性组
cm_female = np.array([[80, 20], [15, 85]])
im2 = axes[1].imshow(cm_female, cmap='Oranges')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Predicted: No Churn', 'Predicted: Churn'])
axes[1].set_yticklabels(['Actual: No Churn', 'Actual: Churn'])
axes[1].set_title('Confusion Matrix: Female Group')

for i in range(2):
    for j in range(2):
        text = axes[1].text(j, i, cm_female[i, j],
                          ha="center", va="center", color="white" if cm_female[i, j] > 50 else "black",
                          fontsize=14)

plt.suptitle('Fairness Check: Confusion Matrices by Gender Group', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('chapters/week_12/images/group_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: group_confusion_matrices.png")

# ============================================
# 图 4: 公平性指标对比
# ============================================
print("生成公平性指标对比图...")

fig, ax = plt.subplots(figsize=(10, 6))

groups = ['Male', 'Female']
metrics = {
    'True Positive Rate': [0.80, 0.85],
    'False Positive Rate': [0.15, 0.20],
    'Accuracy': [0.825, 0.825],
    'Predicted Positive Rate': [0.475, 0.525]
}

x = np.arange(len(groups))
width = 0.2
multiplier = 0

colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

for i, (metric, values) in enumerate(metrics.items()):
    offset = width * multiplier
    rects = ax.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8)
    ax.bar_label(rects, padding=3, fontsize=8, fmt='%.2f')
    multiplier += 1

ax.set_ylabel('Rate')
ax.set_title('Fairness Metrics Comparison by Gender')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(groups)
ax.legend(loc='upper right')
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('chapters/week_12/images/fairness_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  保存: fairness_metrics_comparison.png")

print("\n所有图片生成完成！")
print("生成的图片：")
for f in os.listdir('chapters/week_12/images'):
    print(f"  - {f}")
