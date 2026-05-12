"""生成聚类稳定性示意图。

本脚本演示：
- 输入：脚本内构造的三簇二维样本
- 输出：`chapters/week_15/images/04_cluster_stability.png`
- 核心概念：聚类不是只跑一次 K-means；需要看不同 random_state / bootstrap 子样本下是否稳定
- 常见错误：只报告一次聚类标签，把随机初始化造成的差异当成真实分群
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


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
    rng = np.random.default_rng(42)
    X, _ = make_blobs(
        n_samples=360,
        centers=[(-3, 0), (0, 2.5), (3, 0)],
        cluster_std=[0.9, 1.0, 0.9],
        random_state=7,
    )

    base = KMeans(n_clusters=3, n_init=20, random_state=0).fit(X)
    base_labels = base.labels_

    scores = []
    examples = []
    for seed in range(30):
        sample_idx = rng.choice(len(X), size=int(len(X) * 0.8), replace=True)
        model = KMeans(n_clusters=3, n_init=10, random_state=seed).fit(X[sample_idx])
        labels_full = model.predict(X)
        scores.append(adjusted_rand_score(base_labels, labels_full))
        if seed in (1, 9):
            examples.append((seed, labels_full))

    fig = plt.figure(figsize=(10, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    ax0.plot(range(1, len(scores) + 1), scores, marker="o", color="#4E79A7", linewidth=2)
    ax0.axhline(0.8, color="#E15759", linestyle="--", linewidth=1.8, label="经验警戒线：ARI = 0.8")
    ax0.set_ylim(0, 1.02)
    ax0.set_xlabel("重复运行 / bootstrap 次数")
    ax0.set_ylabel("与基准聚类的一致性（ARI）")
    ax0.set_title("聚类稳定性：同一数据多跑几次，结果是否一致？", fontsize=14, fontweight="bold")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="lower right")
    ax0.text(0.02, 0.08, "ARI 越接近 1，说明分群越稳定；\n若大幅波动，不宜直接给业务贴标签。",
             transform=ax0.transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F0F7FF", edgecolor="#4E79A7"))

    for ax, (seed, labels) in zip((ax1, ax2), examples):
        ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="Set2", s=24, alpha=0.8, edgecolor="none")
        ax.set_title(f"一次运行结果（random_state={seed}）", fontsize=12, fontweight="bold")
        ax.set_xlabel("主成分 1")
        ax.set_ylabel("主成分 2")
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    output = Path(__file__).resolve().parents[1] / "images" / "04_cluster_stability.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
