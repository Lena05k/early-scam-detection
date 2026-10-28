"""Общий стиль рисунков статьи PIERE 2026 (ревизия v3).

Семантика цвета едина во всех рисунках:
GOLD   — всё про человеческие метки; SILVER — всё про машинные (LLM) метки;
NEUTRAL — общее; ACCENT — третий цвет, только если без него нельзя.
"""
import matplotlib

GOLD    = "#009E73"   # всё про человеческие метки
SILVER  = "#D55E00"   # всё про машинные (LLM) метки
NEUTRAL = "#7F7F7F"   # общее
ACCENT  = "#0072B2"   # третий цвет, только если без него нельзя
FONT = "Times New Roman"; FONTSIZE_LABEL = 8; FONTSIZE_ANNOT = 7
COL_WIDTH_IN = 3.38; FULL_WIDTH_IN = 7.01; DPI = 300  # колонки A4-шаблона PIERE


def apply():
    matplotlib.rcParams.update({
        "font.family": FONT,
        "font.size": FONTSIZE_LABEL,
        "axes.labelsize": FONTSIZE_LABEL,
        "axes.titlesize": FONTSIZE_LABEL,
        "xtick.labelsize": FONTSIZE_LABEL,
        "ytick.labelsize": FONTSIZE_LABEL,
        "legend.fontsize": FONTSIZE_ANNOT,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.dpi": DPI,
        "pdf.fonttype": 42,
        "axes.unicode_minus": False,
    })


def save(fig, stem):
    """Сохранить .pdf + .png 300 dpi; ширина задана при создании figure."""
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
