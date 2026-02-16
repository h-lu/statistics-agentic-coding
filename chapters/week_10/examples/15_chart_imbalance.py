"""
生成类别不平衡可视化图，对比平衡与不平衡数据集。

运行方式：python3 chapters/week_10/examples/15_chart_imbalance.py
预期输出：生成 images/05_class_imbalance.png
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

    # 数据集
    categories = ['负类 (多数)', '正类 (少数)']
    
    # 平衡数据集
    balanced = [500, 500]
    
    # 不平衡数据集 (1:10 比例)
    imbalanced = [909, 91]  # 总计 1000，正类约 9.1%
    
    # 严重不平衡数据集 (1:100 比例)
    severe_imbalanced = [990, 10]  # 正类仅 1%

    # 创建图形，使用子图布局
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    datasets = [
        (balanced, '平衡数据集', '1:1', 'steelblue'),
        (imbalanced, '轻度不平衡', '10:1', 'orange'),
        (severe_imbalanced, '严重不平衡', '100:1', 'crimson')
    ]

    for idx, (data, title, ratio, color_base) in enumerate(datasets):
        ax = axes[idx]
        
        colors = [color_base, 'lightcoral']
        bars = ax.bar(categories, data, color=colors, edgecolor='black', linewidth=1.2)
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, data):
            height = bar.get_height()
            percentage = value / sum(data) * 100
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value}\n({percentage:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 设置标题和标签
        ax.set_title(f'{title}\n比例 {ratio}', fontsize=13, fontweight='bold')
        ax.set_ylabel('样本数', fontsize=11)
        ax.set_ylim(0, max(data) * 1.2)
        
        # 添加总数
        total = sum(data)
        ax.text(0.5, 0.95, f'总计: {total} 个样本', 
               transform=ax.transAxes, ha='center', va='top',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 总标题
    fig.suptitle('类别不平衡问题：从平衡到严重不平衡', 
                fontsize=15, fontweight='bold', y=1.02)

    # 添加说明文字
    explanation = (
        '类别不平衡会导致模型偏向多数类，常见处理方法：\n'
        '• 重采样：过采样少数类 / 欠采样多数类  |  '
        '• 类别权重：给少数类更高权重  |  '
        '• 阈值调整：降低正类判定阈值  |  '
        '• 评估指标：使用 F1、AUC-PR 而非准确率'
    )
    fig.text(0.5, -0.02, explanation, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / '05_class_imbalance.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"图片已保存: {output_path}")
    print("数据集对比:")
    print(f"  平衡: {balanced[0]}:{balanced[1]} = 1:1")
    print(f"  轻度不平衡: {imbalanced[0]}:{imbalanced[1]} ≈ 10:1")
    print(f"  严重不平衡: {severe_imbalanced[0]}:{severe_imbalanced[1]} = 100:1")


if __name__ == '__main__':
    main()
