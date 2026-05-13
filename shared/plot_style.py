"""Shared plotting style for course images.

The course images are used as lecture assets, so they should be legible on slides
and consistent across chapters.  This module intentionally keeps the API tiny:
call :func:`install` once before running a plotting script.  It configures a
CJK-safe font stack and monkey-patches Matplotlib save helpers so legacy scripts
also get the refreshed typography, grid, and export defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

_INSTALLED = False

COURSE_FACE = "#F8FAFC"
AXIS_FACE = "#FFFFFF"
TEXT = "#0F172A"
GRID = "#E2E8F0"
SPINE = "#CBD5E1"

COURSE_PALETTE = [
    "#2563EB",  # blue
    "#F97316",  # orange
    "#10B981",  # emerald
    "#8B5CF6",  # violet
    "#EF4444",  # red
    "#06B6D4",  # cyan
    "#F59E0B",  # amber
    "#64748B",  # slate
]


def _font_stack() -> list[str]:
    import matplotlib.font_manager as fm

    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Noto Sans CJK JP",  # common Linux family name for the Noto CJK TTC
        "Source Han Sans SC",
        "Microsoft YaHei",
        "PingFang SC",
        "WenQuanYi Micro Hei",
        "DejaVu Sans",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    selected = next((font for font in preferred if font in available), "DejaVu Sans")
    return [selected, "DejaVu Sans"] if selected != "DejaVu Sans" else [selected]


def apply_rc() -> None:
    """Apply global Matplotlib/Seaborn rc params."""
    import matplotlib.pyplot as plt

    fonts = _font_stack()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": fonts,
        "axes.unicode_minus": False,
        "figure.facecolor": COURSE_FACE,
        "axes.facecolor": AXIS_FACE,
        "axes.edgecolor": SPINE,
        "axes.labelcolor": "#334155",
        "axes.titlecolor": TEXT,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "text.color": TEXT,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=COURSE_PALETTE),
        "legend.frameon": False,
        "savefig.facecolor": COURSE_FACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 160,
    })

    try:
        import seaborn as sns

        sns.set_theme(
            style="whitegrid",
            context="notebook",
            palette=COURSE_PALETTE,
            rc={
                "font.sans-serif": fonts,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
            },
        )
    except Exception:
        pass


def style_figure(fig: Any) -> None:
    """Polish an existing figure before export."""
    apply_rc()
    fonts = _font_stack()
    try:
        fig.set_facecolor(COURSE_FACE)
    except Exception:
        pass

    for ax in getattr(fig, "axes", []):
        try:
            ax.set_facecolor(AXIS_FACE)
            ax.grid(True, axis="y", alpha=0.75)
            ax.grid(False, axis="x")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(SPINE)
                ax.spines[side].set_linewidth(0.8)
            ax.title.set_fontsize(max(ax.title.get_fontsize(), 13))
            ax.title.set_fontweight("bold")
            ax.xaxis.label.set_color("#334155")
            ax.yaxis.label.set_color("#334155")
            ax.tick_params(colors="#475569")
        except Exception:
            continue

        for text in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]:
            try:
                text.set_fontfamily(fonts)
            except Exception:
                pass
        legend = ax.get_legend()
        if legend is not None:
            try:
                legend.set_frame_on(False)
                for text in legend.get_texts():
                    text.set_fontfamily(fonts)
            except Exception:
                pass

    for text in getattr(fig, "texts", []):
        try:
            text.set_fontfamily(fonts)
        except Exception:
            pass


def install() -> None:
    """Install course plotting style and export hooks."""
    global _INSTALLED
    if _INSTALLED:
        apply_rc()
        return

    apply_rc()

    import matplotlib.figure as mfigure
    import matplotlib.pyplot as plt

    original_plt_savefig = plt.savefig
    original_fig_savefig = mfigure.Figure.savefig

    def styled_plt_savefig(*args: Any, **kwargs: Any) -> Any:
        fig = plt.gcf()
        style_figure(fig)
        kwargs.setdefault("dpi", 160)
        kwargs.setdefault("facecolor", COURSE_FACE)
        kwargs.setdefault("bbox_inches", "tight")
        return original_plt_savefig(*args, **kwargs)

    def styled_fig_savefig(self: Any, fname: str | Path, *args: Any, **kwargs: Any) -> Any:
        style_figure(self)
        kwargs.setdefault("dpi", 160)
        kwargs.setdefault("facecolor", COURSE_FACE)
        kwargs.setdefault("bbox_inches", "tight")
        return original_fig_savefig(self, fname, *args, **kwargs)

    plt.savefig = styled_plt_savefig  # type: ignore[assignment]
    mfigure.Figure.savefig = styled_fig_savefig  # type: ignore[assignment]
    _INSTALLED = True
