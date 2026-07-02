#!/usr/bin/env python3
"""把 results.json 画成对比图"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "own-no-destroy":        "#2a9d8f",   # 推荐:adapter 持有
    "own-recreate":          "#8a63d2",   # 改造前现状(HIP handle 复用)
    "own-recreate-distinct": "#d95f5f",   # 真实切换:handle 不复用
    "pool-cached":           "#4b78d8",   # 推荐:hybrid 方案
    "pool-fresh":            "#c18d19",   # 反例:每轮 fresh pool
}
LABELS = {
    "own-no-destroy":        "A. own-no-destroy (adapter-owned)",
    "own-recreate":          "B. own-recreate (HIP handle reused)",
    "own-recreate-distinct": "C. own-recreate-distinct (real switch)",
    "pool-cached":           "D. pool-cached (hybrid)",
    "pool-fresh":            "E. pool-fresh (round-robin spreads)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results.json")
    ap.add_argument("--out", default="memory_trend.png")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for mode, recs in data["records"].items():
        rounds = [r["round"] for r in recs]
        alloc = [r["allocated_MB"] for r in recs]
        reserved = [r["reserved_MB"] for r in recs]
        ax1.plot(rounds, reserved, "-o", color=COLORS[mode],
                 label=LABELS[mode], linewidth=2, markersize=5)
        ax2.plot(rounds, alloc, "-o", color=COLORS[mode],
                 label=LABELS[mode], linewidth=2, markersize=5)

    cfg = data["args"]
    title_cfg = (f"streams={cfg['num_streams']}  "
                 f"steps/round={cfg['steps_per_round']}  "
                 f"carry={cfg['carry_mb']:.0f}MB×{cfg['carry_keep']}  "
                 f"batch×dim={cfg['batch']}×{cfg['dim']}")

    for ax, ylabel, title in [
        (ax1, "reserved (MB)", "reserved_bytes (caching allocator)"),
        (ax2, "allocated (MB)", "allocated_bytes (active tensor)"),
    ]:
        ax.set_xlabel("snapshot switch round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"StreamManager switch memory comparison  |  {title_cfg}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
