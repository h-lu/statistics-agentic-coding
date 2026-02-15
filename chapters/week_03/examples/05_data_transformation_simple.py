"""
生成数据转换对比图（标准化 vs 归一化）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def setup_output_dir() -> Path:
    """设置输出目录"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def main() -> None:
    """生成数据转换对比图"""
    output_dir = setup_output_dir()

    # 加载企鹅数据
    penguins = sns.load_dataset("penguins")
    numeric_cols = ["bill_length_mm", "body_mass_g"]
    data = penguins[numeric_cols].dropna()

    # 手动实现标准化
    data_standardized = data.copy()
    for col in numeric_cols:
        mean = data[col].mean()
        std = data[col].std()
        data_standardized[col] = (data[col] - mean) / std

    # 手动实现归一化
    data_normalized = data.copy()
    for col in numeric_cols:
        min_val = data[col].min()
        max_val = data[col].max()
        data_normalized[col] = (data[col] - min_val) / (max_val - min_val)

    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 左图：原始数据（不同尺度）
    data["bill_length_mm"].plot(kind="hist", ax=axes[0], alpha=0.5, label="bill_length_mm")
    data["body_mass_g"].plot(kind="hist", ax=axes[0], alpha=0.5, label="body_mass_g")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Original: Different Scales")
    axes[0].legend()

    # 中图：标准化后（相同尺度）
    data_standardized["bill_length_mm"].plot(kind="hist", ax=axes[1], alpha=0.5, label="bill_length_mm")
    data_standardized["body_mass_g"].plot(kind="hist", ax=axes[1], alpha=0.5, label="body_mass_g")
    axes[1].set_xlabel("Z-score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Standardized: Same Scale")
    axes[1].legend()

    # 右图：归一化后（都在 [0,1]）
    data_normalized["bill_length_mm"].plot(kind="hist", ax=axes[2], alpha=0.5, label="bill_length_mm")
    data_normalized["body_mass_g"].plot(kind="hist", ax=axes[2], alpha=0.5, label="body_mass_g")
    axes[2].set_xlabel("Normalized Value")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Normalized: All in [0,1]")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "data_transformation_comparison.png", dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"图表已保存：{output_dir / 'data_transformation_comparison.png'}")


if __name__ == "__main__":
    main()
