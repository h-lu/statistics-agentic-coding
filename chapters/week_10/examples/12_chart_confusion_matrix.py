"""
生成混淆矩阵热力图，展示分类模型评估的基础工具。

运行方式：python3 chapters/week_10/examples/12_chart_confusion_matrix.py
预期输出：生成 images/02_confusion_matrix.png
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path


def setup_chinese_font() -> str:
    """配置中文字体"""
    # 重新扫描字体缓存以确保找到系统字体
    fm.fontManager.__init__()
    
    chinese_fonts = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'SimHei', 
                     'Arial Unicode MS', 'PingFang SC', 'Microsoft YaHei',
                     'WenQuanYi Micro Hei']
    available = [f.name for f in fm.fontManager.ttflist]
    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


def main() -> None:
    font = setup_chinese_font()
    print(f"使用字体: {font}")

    # 混淆矩阵数据 (示例值)
    # 行：实际类别，列：预测类别
    # [TN, FP]
    # [FN, TP]
    cm = np.array([[850, 50],   # 实际负类：850 TN, 50 FP
                   [80, 420]])  # 实际正类：80 FN, 420 TP

    # 类别标签
    labels = ['负类 (Negative)', '正类 (Positive)']

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 7))

    # 绘制热力图
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('样本数', rotation=-90, va="bottom", fontsize=11)

    # 设置坐标轴
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['预测负类', '预测正类'],
           yticklabels=['实际负类', '实际正类'],
           title='混淆矩阵：分类结果可视化',
           ylabel='实际类别',
           xlabel='预测类别')

    # 旋转 x 轴标签
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center", fontsize=11)

    # 在每个单元格中添加数值和标签
    text_colors = ['white', 'black', 'black', 'white']
    annotations = [
        ('TN\n真负例', 850),
        ('FP\n假正例', 50),
        ('FN\n假负例', 80),
        ('TP\n真正例', 420)
    ]

    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            label, value = annotations[idx]
            text = ax.text(j, i, f'{label}\n\n{value}',
                          ha="center", va="center", 
                          color=text_colors[idx], fontsize=13, fontweight='bold')

    # 添加关键指标计算
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / cm.sum()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # 在图下方添加指标说明
    metrics_text = (f'准确率 (Accuracy): {accuracy:.3f}  |  '
                   f'精确率 (Precision): {precision:.3f}  |  '
                   f'查全率 (Recall): {recall:.3f}  |  '
                   f'特异度 (Specificity): {specificity:.3f}')

    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / '02_confusion_matrix.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"图片已保存: {output_path}")
    print(f"混淆矩阵指标: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


if __name__ == '__main__':
    main()
