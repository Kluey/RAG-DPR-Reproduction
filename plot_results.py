import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

N_DEV = 1000
TOP_K_ACCURACY = {1: 37.3, 5: 68.2, 20: 84.4, 100: 93.3}
EXACT_MATCH = {"Closed-book BART": 2.9, "RAG-Token": 27.1, "RAG-Sequence": 29.1}
N_DOCS = 5

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0, labelsize=10)
    ax.grid(color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)


def ci95(pct: float, n: int = N_DEV) -> float:
    p = pct / 100
    return 196 * math.sqrt(p * (1 - p) / n)


def plot_retrieval(ax):
    ks, acc = list(TOP_K_ACCURACY), list(TOP_K_ACCURACY.values())
    ax.plot(ks, acc, color=SERIES, linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.scatter(ks, acc, s=64, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(ks, [f"top-{k}" for k in ks])
    ax.minorticks_off()
    ax.set_xlim(0.7, 160)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.grid(axis="x", visible=False)

    for k, label in ((N_DOCS, f"{TOP_K_ACCURACY[N_DOCS]}%  ← RAG reads top-{N_DOCS}"), (100, f"{TOP_K_ACCURACY[100]}%")):
        ax.annotate(label, (k, TOP_K_ACCURACY[k]), xytext=(8, -14), textcoords="offset points",
                    color=INK, fontsize=10)

    ax.set_title("DPR retrieval accuracy", loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.02, "Dev questions with an answer-containing passage in the top k",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5)


def plot_exact_match(ax):
    names, em = list(EXACT_MATCH), list(EXACT_MATCH.values())
    ys = range(len(names))
    ax.barh(ys, em, height=0.3, color=SERIES, zorder=3)
    ax.errorbar(em, ys, xerr=[ci95(v) for v in em], fmt="none", ecolor=INK_MUTED, elinewidth=1, capsize=3, zorder=4)

    for y, v in zip(ys, em):
        ax.text(v + ci95(v) + 1, y, f"{v}%", va="center", color=INK, fontsize=10)

    ax.set_yticks(list(ys), names)
    ax.set_xlim(0, 40)
    ax.set_xticks(range(0, 41, 10), [f"{v}%" for v in range(0, 41, 10)])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)

    ax.set_title("Answer exact match", loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.02, f"NQ dev subset (n = {N_DEV}); whiskers are 95% confidence intervals",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/results.png")
    args = parser.parse_args()

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.38})
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        style_axes(ax)
    plot_retrieval(left)
    plot_exact_match(right)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
