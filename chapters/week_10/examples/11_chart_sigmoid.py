"""
生成 Sigmoid 函数图，展示逻辑回归的核心激活函数。

运行方式：python3 chapters/week_10/examples/11_chart_sigmoid.py
预期输出：生成 images/01_sigmoid_function.png
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


def sigmoid(z):
    """Sigmoid 函数"""
    return 1 / (1 + np.exp(-z))


def main() -> None:
    font = setup_chinese_font()
    print(f"使用字体: {font}")

    # 生成 z 值范围
    z = np.linspace(-10, 10, 1000)
    p = sigmoid(z)

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制 sigmoid 曲线
    ax.plot(z, p, 'b-', linewidth=3, label=r'$\sigma(z) = \frac{1}{1+e^{-z}}$')

    # 添加水平参考线
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # 标注关键点
    # z=0, p=0.5
    ax.plot(0, 0.5, 'ro', markersize=10)
    ax.annotate('z=0, p=0.5\n(决策边界)', 
                xy=(0, 0.5), xytext=(3, 0.35),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    # z→∞, p→1
    ax.annotate('z→+∞, p→1\n(正类概率趋近 1)', 
                xy=(8, sigmoid(8)), xytext=(5, 0.92),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

    # z→-∞, p→0
    ax.annotate('z→-∞, p→0\n(负类概率趋近 1)', 
                xy=(-8, sigmoid(-8)), xytext=(-5, 0.08),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))

    # 设置标签和标题
    ax.set_xlabel('z (线性组合值)', fontsize=12)
    ax.set_ylabel('p (属于正类的概率)', fontsize=12)
    ax.set_title('Sigmoid 函数：将实数映射到 (0,1) 概率空间', fontsize=14, fontweight='bold')

    # 设置坐标轴范围
    ax.set_xlim(-10, 10)
    ax.set_ylim(-0.05, 1.05)

    # 添加网格
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right', fontsize=11)

    # 保存图片
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / '01_sigmoid_function.png'

    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"图片已保存: {output_path}")


if __name__ == '__main__':
    main()
