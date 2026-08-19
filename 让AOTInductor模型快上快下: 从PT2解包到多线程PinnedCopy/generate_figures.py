#!/usr/bin/env python3
"""Generate the explanatory figures used by this article."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent

NAVY = "#183153"
BLUE = "#2878B5"
CYAN = "#49B3D1"
GREEN = "#2A9D8F"
ORANGE = "#F4A261"
RED = "#E76F51"
PURPLE = "#7B61A8"
LIGHT = "#F3F6FA"
GRID = "#D9E2EC"
TEXT = "#203040"
PAPER = "#FFFDF7"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": TEXT,
        "axes.edgecolor": GRID,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
    }
)


def save(fig, name):
    fig.savefig(ROOT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, width, height, text, color, subtitle=None, text_color="white"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.8,
        edgecolor=NAVY,
        facecolor=color,
    )
    patch.set_sketch_params(scale=1.2, length=85, randomness=2.0)
    ax.add_patch(patch)
    cx, cy = xy[0] + width / 2, xy[1] + height / 2
    ax.text(cx, cy + (0.06 if subtitle else 0), text, ha="center", va="center",
            color=text_color, fontsize=12, weight="bold")
    if subtitle:
        ax.text(cx, cy - 0.11, subtitle, ha="center", va="center",
                color=text_color, fontsize=9)
    return patch


def arrow(ax, start, end, color=NAVY, text=None, rad=0):
    p = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=2,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    p.set_sketch_params(scale=1.2, length=85, randomness=2.0)
    ax.add_patch(p)
    if text:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.07,
                text, ha="center", color=color, fontsize=9)


def hot_path_comparison():
    labels = ["PT2 hot", "Direct SO hot"]
    values = [9.2645, 0.8227]
    colors = [ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.4f}s",
                ha="center", va="bottom", weight="bold", color=TEXT)
    ax.set_ylabel("Load latency (seconds)")
    ax.set_title("Hot load: skip repeated PT2 extraction")
    ax.set_ylim(0, 10.4)
    ax.grid(axis="y", color=GRID, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(1, 3.0, "11.26x faster", ha="center", color=GREEN,
            fontsize=12, weight="bold")
    save(fig, "hot_pt2_vs_direct_so.png")


def pt2_loader_flow():
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.text(0.2, 4.3, "Standard PT2 load (repeated for every switch)",
            fontsize=14, weight="bold", color=NAVY)
    xs = [0.3, 2.7, 5.2, 7.7, 9.5]
    widths = [1.5, 1.6, 1.6, 1.2, 1.2]
    texts = ["PT2", "new /tmp", "extract ~5GB", "dlopen", "Runner"]
    colors = [PURPLE, ORANGE, RED, BLUE, GREEN]
    for x, w, t, c in zip(xs, widths, texts, colors):
        box(ax, (x, 3.1), w, 0.75, t, c)
    for a, b in zip(range(4), range(1, 5)):
        arrow(ax, (xs[a] + widths[a], 3.47), (xs[b], 3.47))

    ax.text(0.2, 1.9, "Pre-extracted artifact (one-time release work)",
            fontsize=14, weight="bold", color=NAVY)
    box(ax, (0.3, 0.7), 2.0, 0.75, "PT2", PURPLE, "release time")
    box(ax, (4.3, 0.7), 2.1, 0.75, "immutable SO", BLUE, "validated once")
    box(ax, (8.5, 0.7), 1.7, 0.75, "Runner", GREEN, "load time")
    arrow(ax, (2.3, 1.07), (4.3, 1.07), text="extract once")
    arrow(ax, (6.4, 1.07), (8.5, 1.07), text="direct")
    save(fig, "pt2_loader_flow.png")


def runner_chain():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    steps = [
        (0.2, 1.35, 1.8, "SO + CUBIN", PURPLE, "stable artifact"),
        (2.5, 1.35, 1.6, "dlopen", BLUE, "map ELF"),
        (4.6, 1.35, 1.8, "dlsym", CYAN, "AOTI exports"),
        (6.9, 1.35, 1.8, "Container", ORANGE, "GPU blob"),
        (9.2, 1.35, 1.5, "Runner", GREEN, "ready"),
    ]
    for x, y, w, t, c, sub in steps:
        box(ax, (x, y), w, 0.9, t, c, sub)
    for i in range(len(steps) - 1):
        x, y, w, *_ = steps[i]
        nx = steps[i + 1][0]
        arrow(ax, (x + w, 1.8), (nx, 1.8))
    ax.text(0.2, 3.1, "Direct Runner still performs real initialization",
            fontsize=15, weight="bold", color=NAVY)
    save(fig, "runner_init_chain.png")


def pinned_double_buffer():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.text(0.2, 4.55, "Ping-pong pinned staging", fontsize=15,
            weight="bold", color=NAVY)
    box(ax, (0.3, 2.0), 2.1, 1.1, "SO mmap", PURPLE, "pageable pages")
    box(ax, (4.0, 3.0), 2.0, 0.85, "PING", ORANGE, "CPU filling")
    box(ax, (4.0, 1.2), 2.0, 0.85, "PONG", BLUE, "GPU reading")
    box(ax, (8.4, 2.0), 2.0, 1.1, "GPU blob", GREEN, "constants")
    arrow(ax, (2.4, 2.55), (4.0, 3.4), color=ORANGE, text="std::memcpy")
    arrow(ax, (6.0, 1.62), (8.4, 2.45), color=BLUE, text="H2D async")
    arrow(ax, (5.0, 2.15), (5.0, 2.9), color=NAVY, text="swap")
    ax.text(0.35, 0.35,
            "The pipeline is limited by the slower stage: CPU producer or GPU consumer.",
            fontsize=11, color=TEXT)
    save(fig, "pinned_double_buffer.png")


def bandwidth_budget():
    labels = ["Pageable\nH2D", "Pinned H2D\n(trace)", "Pinned H2D\n(microbench)",
              "1-ch memcpy\npayload ceiling**", "DDR4-3200\n1-ch bus*",
              "PCIe 4 x16\ntheory*"]
    values = [9.0, 22.0, 26.0, 12.8, 25.6, 31.5]
    colors = [RED, BLUE, GREEN, CYAN, ORANGE, PURPLE]
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    bars = ax.barh(labels, values, color=colors, height=0.62)
    for b, v in zip(bars, values):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:g} GB/s",
                va="center", weight="bold", color=TEXT)
    ax.set_xlim(0, 35)
    ax.set_xlabel("Bandwidth")
    ax.set_title("Measured transfer rates vs. theoretical ceilings")
    ax.grid(axis="x", color=GRID, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0, -1.02,
            "* Theory is not an application measurement.  ** Ideal bus/2 for one read + one write.",
            color=TEXT, fontsize=9)
    ax.invert_yaxis()
    save(fig, "bandwidth_budget.png")


def weight_coalescing():
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), gridspec_kw={"hspace": 0.55})
    fig.patch.set_facecolor(PAPER)
    specs = [
        (axes[0], "Per-weight submission", False),
        (axes[1], "Cross-weight coalescing", True),
    ]
    weights = [0.7, 1.0, 0.45, 1.25, 0.6, 0.8]
    colors = [BLUE, GREEN, ORANGE, PURPLE, CYAN, RED]
    for ax, title, coalesced in specs:
        ax.set_facecolor(PAPER)
        ax.set_xlim(0, 7.1)
        ax.set_ylim(0, 1.45)
        ax.axis("off")
        ax.set_title(title, loc="left", color=NAVY, fontsize=13, weight="bold")
        if not coalesced:
            x = 0.15
            for i, (w, c) in enumerate(zip(weights, colors), 1):
                patch = Rectangle((x, 0.45), w, 0.55, facecolor=c,
                                  edgecolor=NAVY, linewidth=1.2)
                patch.set_sketch_params(scale=1.0, length=75, randomness=1.8)
                ax.add_patch(patch)
                ax.text(x + w / 2, 0.72, f"W{i}", color="white", ha="center",
                        va="center", weight="bold", fontsize=9)
                ax.text(x + w / 2, 0.2, "H2D", ha="center", color=RED, fontsize=8)
                x += w + 0.35
        else:
            x = 0.15
            for i, (w, c) in enumerate(zip(weights, colors), 1):
                patch = Rectangle((x, 0.45), w, 0.55, facecolor=c,
                                  edgecolor=NAVY, linewidth=1.2)
                patch.set_sketch_params(scale=1.0, length=75, randomness=1.8)
                ax.add_patch(patch)
                ax.text(x + w / 2, 0.72, f"W{i}", color="white", ha="center",
                        va="center", weight="bold", fontsize=9)
                x += w
                if i in (2, 4):
                    gap = Rectangle((x, 0.45), 0.12, 0.55,
                                    facecolor="#CBD5E1", edgecolor=NAVY,
                                    linewidth=1.0)
                    gap.set_sketch_params(scale=1.0, length=75, randomness=1.8)
                    ax.add_patch(gap)
                    x += 0.12
            arrow(ax, (0.2, 0.2), (x - 0.05, 0.2), color=GREEN,
                  text="one contiguous H2D window")
    fig.suptitle("Use staging capacity across weight boundaries",
                 fontsize=15, weight="bold", color=NAVY, y=1.01)
    save(fig, "weight_coalescing.png")


def optimization_waterfall():
    labels = [
        "Pageable\nsync",
        "Pinned\n1 thread",
        "Pinned + coalescing\n1 thread",
        "Pinned\n4 threads",
        "Pinned + coalescing\n4 threads",
    ]
    runner = [0.8987, 0.9340, 0.8970, 0.6717, 0.6345]
    phase = np.array([668, 654, 626, 399, 356]) / 1000.0
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    bars = ax.bar(x, runner, color=[NAVY, RED, ORANGE, BLUE, GREEN], width=0.62,
                  label="Runner construction")
    ax.plot(x, phase, marker="o", markersize=8, linewidth=2.3, color=PURPLE,
            label="Constant load/copy phase")
    for b, v in zip(bars, runner):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.4f}s",
                ha="center", fontsize=9, weight="bold", color=TEXT)
    for xx, v in zip(x, phase):
        ax.text(xx, v - 0.065, f"{v*1000:.0f}ms", ha="center", fontsize=9,
                color=PURPLE, weight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Latency (seconds)")
    ax.set_title("Direct-SO loading: threads provide the main gain")
    ax.grid(axis="y", color=GRID, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "optimization_waterfall.png")


if __name__ == "__main__":
    hot_path_comparison()
    pt2_loader_flow()
    runner_chain()
    pinned_double_buffer()
    bandwidth_budget()
    weight_coalescing()
    optimization_waterfall()
