"""
生成 ROC 曲线图，展示分类器在不同阈值下的表现。

运行方式：python3 chapters/week_10/examples/13_chart_roc.py
预期输出：生成 images/03_roc_curve.png
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


def generate_roc_curve(n_points=100):
    """
    生成示例 ROC 曲线数据。
    模拟一个 AUC ≈ 0.85 的分类器。
    """
    np.random.seed(42)
    
    # 生成假阳性率 (FPR) 从 0 到 1
    fpr = np.linspace(0, 1, n_points)
    
    # 生成真阳性率 (TPR)：使用一个略优于随机猜测的曲线
    # 使用凸曲线模拟较好的分类器
    tpr = np.power(fpr, 0.3) * 0.95 + np.random.normal(0, 0.02, n_points)
    tpr = np.clip(tpr, 0, 1)
    tpr[0] = 0  # 确保起点在原点
    tpr[-1] = 1  # 确保终点在 (1,1)
    
    # 计算 AUC（梯形法则）
    auc = np.trapezoid(tpr, fpr)
    
    return fpr, tpr, auc


def main() -> None:
    font = setup_chinese_font()
    print(f"使用字体: {font}")

    # 生成 ROC 数据
    fpr, tpr, auc = generate_roc_curve()

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制对角线（随机猜测基线）
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, 
            label='随机猜测 (AUC = 0.5)', alpha=0.6)

    # 绘制 ROC 曲线
    ax.plot(fpr, tpr, 'b-', linewidth=3, 
            label=f'分类器 ROC 曲线 (AUC = {auc:.3f})')

    # 填充曲线下方面积
    ax.fill_between(fpr, 0, tpr, alpha=0.2, color='blue')

    # 标注几个关键阈值点
    key_points = [20, 50, 80]
    for idx in key_points:
        ax.plot(fpr[idx], tpr[idx], 'ro', markersize=8)
        ax.annotate(f'阈值 {idx}',
                   xy=(fpr[idx], tpr[idx]),
                   xytext=(fpr[idx] + 0.1, tpr[idx] - 0.08),
                   fontsize=9,
                   arrowprops=dict(arrowstyle='->', color='red', lw=1))

    # 添加 AUC 说明框
    ax.text(0.6, 0.2, f'AUC = {auc:.3f}\n\nAUC 解释：\n• 0.5 = 随机猜测\n• 0.7-0.8 = 可接受\n• 0.8-0.9 = 优秀\n• >0.9 = 卓越',
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 设置坐标轴
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=12)
    ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=12)
    ax.set_title('ROC 曲线：评估分类器在不同阈值下的性能', fontsize=14, fontweight='bold')

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='lower right', fontsize=11)

    # 添加象限说明
    ax.text(0.25, 0.75, '理想区域\n高 TPR, 低 FPR', ha='center', va='center',
           fontsize=10, style='italic', color='green',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / '03_roc_curve.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"图片已保存: {output_path}")
    print(f"AUC 值: {auc:.3f}")


if __name__ == '__main__':
    main()
