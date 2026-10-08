# Data figures for the post. Numbers are the measurements quoted in the text.
# Usage: MPLCONFIGDIR=/tmp/mpl python plot_figures.py   (writes ../images/*.png)
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")
plt.xkcd(scale=1, length=100, randomness=2)
plt.rcParams.update({"font.size": 13, "path.effects": []})

# 1. One core, measured with membw.c (1 GiB buffers, core 8, node0 memory).
labels = ["read\nonly", "write\nregular", "write\nnon-temporal",
          "copy\nregular store", "copy\nnon-temporal", "glibc\nmemcpy"]
gbps = [16.9, 14.2, 21.2, 10.1, 16.4, 15.5]
colors = ["#7fb3d5", "#a9cce3", "#a9cce3", "#eb984e", "#52be80", "#52be80"]
fig, ax = plt.subplots(figsize=(10, 5.2))
bars = ax.bar(range(len(gbps)), gbps, color=colors, edgecolor="#333333")
for b, v in zip(bars, gbps):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v}", ha="center", fontsize=13)
ax.axhline(26.3, color="#c0392b", ls="--", lw=1.5, label="H2D via PCIe DMA: 26.3 GB/s")
ax.legend(loc="upper left", fontsize=11, frameon=False)
ax.set_xticks(range(len(gbps)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("GB/s, one core (copy counts bytes once)")
ax.set_ylim(0, 35)
ax.set_title("One core can't keep up with one PCIe x16 link", pad=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "single_core.png"), dpi=150)
plt.close(fig)

# 2. 5.39 GB load through the staging pipeline (replica of the PR's path,
#    source freshly mmap'd so first touches fault), vs copy threads.
threads = [1, 2, 4, 8]
ms = [687, 425, 222, 210]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(threads, ms, marker="o", color="#2e86c1", lw=2.5, ms=8)
for t, v in zip(threads, ms):
    ax.text(t, v + 22, f"{v} ms", ha="center", fontsize=12)
ax.axhline(205, color="#c0392b", ls="--", lw=1.5)
ax.text(1, 150, "DMA floor: 5.39 GB / 26.3 GB/s = 205 ms", ha="left", color="#c0392b", fontsize=11)
ax.annotate("close to the\nDMA floor", xy=(4, 222), xytext=(4.6, 420),
            arrowprops=dict(arrowstyle="->", color="#555555"), fontsize=12)
ax.set_xscale("log", base=2)
ax.set_xticks(threads)
ax.set_xticklabels([str(t) for t in threads])
ax.set_xlabel("H2H copy threads")
ax.set_ylabel("load time (ms)")
ax.set_ylim(0, 780)
ax.set_title("5.39 GB through the pinned staging pipeline", pad=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "threads.png"), dpi=150)
plt.close(fig)
print("ok")
