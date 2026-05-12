"""生成 DID 平行趋势示意图。

本脚本演示：
- 输入：脚本内构造的政策前/后时间序列
- 输出：`chapters/week_13/images/did_parallel_trends.png`
- 核心概念：DID 的平行趋势假设；政策前趋势若不平行，因果解释会站不稳
- 常见错误：只看政策后差异，不检查政策前趋势
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


def setup_chinese_font() -> str:
    """设置 Linux 服务器上的中文字体，避免 PNG 中文变方框。"""
    font_files = [
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/google-droid-fonts/DroidSansFallback.ttf",
    ]
    for font_file in font_files:
        if Path(font_file).is_file():
            fm.fontManager.addfont(font_file)
            font_name = fm.FontProperties(fname=font_file).get_name()
            plt.rcParams["font.sans-serif"] = [font_name, "Droid Sans Fallback", "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return font_name
    raise RuntimeError("未找到可用中文字体；请安装 Noto Sans CJK 或 Droid Sans Fallback")


def main() -> None:
    setup_chinese_font()

    t = np.array([-4, -3, -2, -1, 0, 1, 2, 3])
    control = np.array([30, 29, 28, 27, 26, 25, 24, 23])
    treated_parallel = np.array([34, 33, 32, 31, 24, 22, 21, 20])
    treated_bad = np.array([40, 37, 34, 31, 24, 22, 21, 20])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)

    panels = [
        (axes[0], treated_parallel, "较可信：政策前趋势大致平行", "可以继续做 DID（仍需检查其他假设）"),
        (axes[1], treated_bad, "风险高：政策前趋势已经不同", "政策后差异不能直接解释为因果效应"),
    ]
    for ax, treated, title, note in panels:
        ax.plot(t, control, marker="o", linewidth=2.2, label="对照组：非试点城市", color="#4E79A7")
        ax.plot(t, treated, marker="o", linewidth=2.2, label="处理组：试点城市", color="#E15759")
        ax.axvline(0, color="#333333", linestyle="--", linewidth=1.5)
        ax.text(0.08, 39.0, "政策开始", rotation=90, va="top", fontsize=10, color="#333333")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("相对政策时间")
        ax.grid(True, alpha=0.25)
        ax.text(0.02, 0.04, note, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF7D6", edgecolor="#D4A017"))
    axes[0].set_ylabel("流失率（%）")
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("DID 平行趋势检查：先看政策前，再解释政策后", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output = Path(__file__).resolve().parents[1] / "images" / "did_parallel_trends.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
