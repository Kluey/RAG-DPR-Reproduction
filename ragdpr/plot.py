import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from ragdpr.config import N_DOCS, RESULTS_FILE
from ragdpr.results import load_results

EM_LABELS = {"closed_book": "Closed-book BART", "rag_token": "RAG-Token", "rag_seq": "RAG-Sequence"}

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


def ci95(pct: float, n: int) -> float:
    p = pct / 100
    return 196 * math.sqrt(p * (1 - p) / n)


def mark_missing(ax, title: str, command: str):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    ax.text(0.5, 0.5, f"Not run yet\n{command}", transform=ax.transAxes, ha="center", va="center",
            color=INK_SECONDARY, fontsize=10)


def plot_retrieval(ax, retrieval: dict):
    top_k = {int(k): v for k, v in retrieval["top_k"].items()}
    ks, acc = sorted(top_k), [top_k[k] for k in sorted(top_k)]
    ax.plot(ks, acc, color=SERIES, linewidth=2, solid_capstyle="round", solid_joinstyle="round", zorder=3)
    ax.scatter(ks, acc, s=64, color=SERIES, edgecolors=SURFACE, linewidths=2, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(ks, [f"top-{k}" for k in ks])
    ax.minorticks_off()
    ax.set_xlim(min(ks) * 0.7, max(ks) * 1.6)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax.grid(axis="x", visible=False)

    labels = {max(ks): f"{top_k[max(ks)]}%"}
    if N_DOCS in top_k:
        labels[N_DOCS] = f"{top_k[N_DOCS]}%  ← RAG reads top-{N_DOCS}"
    for k, label in labels.items():
        ax.annotate(label, (k, top_k[k]), xytext=(8, -14), textcoords="offset points", color=INK, fontsize=10)

    ax.set_title("DPR retrieval accuracy", loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.02, f"Dev questions (n = {retrieval['n']}) with an answer-containing passage in the top k",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5)


def plot_exact_match(ax, exact_match: dict):
    keys = [k for k in EM_LABELS if k in exact_match]
    em = [exact_match[k]["em"] for k in keys]
    ci = [ci95(exact_match[k]["em"], exact_match[k]["n"]) for k in keys]
    ys = range(len(keys))
    ax.barh(ys, em, height=0.3, color=SERIES, zorder=3)
    ax.errorbar(em, ys, xerr=ci, fmt="none", ecolor=INK_MUTED, elinewidth=1, capsize=3, zorder=4)

    for y, v, c in zip(ys, em, ci):
        ax.text(v + c + 1, y, f"{v}%", va="center", color=INK, fontsize=10)

    x_max = 10 * math.ceil((max(e + c for e, c in zip(em, ci)) + 6) / 10)
    ax.set_yticks(list(ys), [EM_LABELS[k] for k in keys])
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.set_xlim(0, x_max)
    ax.set_xticks(range(0, x_max + 1, 10), [f"{v}%" for v in range(0, x_max + 1, 10)])
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)

    ax.set_title("Answer exact match", loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    ns = sorted({exact_match[k]["n"] for k in keys})
    ax.text(0, 1.02, f"NQ dev subset (n = {', '.join(map(str, ns))}); whiskers are 95% confidence intervals",
            transform=ax.transAxes, color=INK_SECONDARY, fontsize=9.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_FILE)
    parser.add_argument("--out", default="figures/results.png")
    args = parser.parse_args()

    results = load_results(args.results)
    if not results:
        raise SystemExit(f"no results in {args.results}; run the evaluations first")

    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.38})
    fig.patch.set_facecolor(SURFACE)
    for ax in (left, right):
        style_axes(ax)
    if results.get("retrieval"):
        plot_retrieval(left, results["retrieval"])
    else:
        mark_missing(left, "DPR retrieval accuracy", "python -m ragdpr.dpr.index --eval")
    if results.get("exact_match"):
        plot_exact_match(right, results["exact_match"])
    else:
        mark_missing(right, "Answer exact match", "python -m ragdpr.rag.evaluate")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
