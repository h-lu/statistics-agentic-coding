"""
生成精确率-查全率曲线，展示分类器在精确率和查全率之间的权衡。

运行方式：python3 chapters/week_10/examples/14_chart_pr_curve.py
预期输出：生成 images/04_precision_recall_curve.png
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


def generate_pr_curve(n_points=100):
    """
    生成示例精确率-查全率曲线数据。
    模拟一个 AP ≈ 0.82 的分类器。
    """
    np.random.seed(42)
    
    # 生成查全率从 0 到 1
    recall = np.linspace(0, 1, n_points)
    
    # 生成精确率：随着查全率增加，精确率通常下降
    # 使用递减函数模拟这种权衡关系
    precision = 0.95 - 0.3 * recall + 0.15 * np.sin(recall * np.pi)
    precision = precision + np.random.normal(0, 0.02, n_points)
    precision = np.clip(precision, 0, 1)
    
    # 确保单调递减（PR 曲线的标准做法）
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    
    # 计算平均精确率 (AP)
    ap = np.mean(precision)
    
    return precision, recall, ap


def main() -> None:
    font = setup_chinese_font()
    print(f"使用字体: {font}")

    # 生成 PR 曲线数据
    precision, recall, ap = generate_pr_curve()

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制基线（正样本比例）
    baseline = 0.35  # 假设正样本占 35%
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=2,
              label=f'随机基线 (AP = {baseline:.2f})', alpha=0.6)

    # 绘制 PR 曲线
    ax.plot(recall, precision, 'g-', linewidth=3,
           label=f'分类器 PR 曲线 (AP = {ap:.3f})')

    # 填充曲线下方面积
    ax.fill_between(recall, 0, precision, alpha=0.2, color='green')

    # 标注权衡区域
    ax.annotate('高精确率\n低查全率',
               xy=(0.1, 0.92), xytext=(0.05, 0.75),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

    ax.annotate('高查全率\n低精确率',
               xy=(0.9, 0.55), xytext=(0.75, 0.35),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))

    # 标注一个关键操作点
    key_idx = 50
    ax.plot(recall[key_idx], precision[key_idx], 'ro', markersize=10)
    ax.annotate(f'操作点\nR={recall[key_idx]:.2f}\nP={precision[key_idx]:.2f}',
               xy=(recall[key_idx], precision[key_idx]),
               xytext=(recall[key_idx] - 0.25, precision[key_idx] - 0.12),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

    # 添加说明框
    ax.text(0.45, 0.25, 
           'PR 曲线解读：\n'
           '• 右上角为理想区域\n'
           '• 曲线越高，模型越好\n'
           '• 不平衡数据更适用\n'
           '• 需在精确率与查全率间权衡',
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # 设置坐标轴
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('查全率 (Recall)', fontsize=12)
    ax.set_ylabel('精确率 (Precision)', fontsize=12)
    ax.set_title('精确率-查全率曲线：权衡分析', fontsize=14, fontweight='bold')

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower left', fontsize=11)

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / '04_precision_recall_curve.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"图片已保存: {output_path}")
    print(f"平均精确率 (AP): {ap:.3f}")


if __name__ == '__main__':
    main()
