"""画 blog 里的数据图. 只读 ../data 下的汇总数据, 输出到 ../images.

    MPLCONFIGDIR=/tmp/mpl python plot_figures.py
"""
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore", message="findfont")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
IMG = os.path.join(HERE, "..", "images")

# 颜色只跟"策略"走, 全文一致
C_DEFAULT = "#eb6834"   # 默认内存策略 (自动均衡生效)
C_LOCAL = "#2a78d6"     # 显式 MPOL_LOCAL
C_NUMACTL = "#1baf7a"   # numactl --preferred
INK = "#333333"


def load(name):
    return json.load(open(os.path.join(DATA, name)))


def bars(ax, labels, values, colors, fmt="{:.0f}", hatches=None):
    xs = range(len(values))
    rects = ax.bar(xs, values, color=colors, edgecolor=INK, linewidth=1.2,
                   hatch=None if hatches is None else None)
    if hatches:
        for r, h in zip(rects, hatches):
            r.set_hatch(h)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    for x, v in zip(xs, values):
        ax.annotate(fmt.format(v), (x, v), ha="center", va="bottom", xytext=(0, 3),
                    textcoords="offset points", color=INK, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    top = max(values)
    if ax.get_yscale() == "log":
        ax.set_ylim(top=top * 4)
    else:
        ax.set_ylim(0, top * 1.22)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, name), dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def fig_numactl(svc):
    b = svc["numactl_batch"]
    arms = ["default (no numactl)", "numactl --preferred=0", "numactl --preferred=1 (run a)",
            "numactl --preferred=1 (run b)"]
    labels = ["default", "preferred\n=0", "preferred\n=1 (a)", "preferred\n=1 (b)"]
    colors = [C_DEFAULT, C_NUMACTL, C_NUMACTL, C_NUMACTL]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, node in zip(axes, (0, 1)):
        bars(ax, labels, [b[a][f"node{node}_per_so_ms_median"] for a in arms], colors)
        ax.set_title(f"node{node} loaders ({'4' if node == 0 else '2'} GPUs)")
    axes[0].set_ylabel("median load time per model file (ms)")
    fig.suptitle("numactl on EITHER node makes node0 fast", fontsize=14)
    save(fig, "fig1_numactl_clue.png")


def fig_mpol(svc):
    b = svc["mpol_batch"]
    arms = ["default policy", "MPOL_LOCAL (exec wrapper)", "MPOL_LOCAL (set at process start)"]
    labels = ["default", "LOCAL\n(wrapper)", "LOCAL\n(at start)"]
    colors = [C_DEFAULT, C_LOCAL, C_LOCAL]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, node in zip(axes[:2], (0, 1)):
        bars(ax, labels, [b[a][f"node{node}_per_so_ms_median"] for a in arms], colors)
        ax.set_title(f"node{node}\nms per model file (median)")
    bars(axes[2], labels, [b[a]["prepare_ms_median"] for a in arms], colors)
    axes[2].set_title("all loads\nprepare ms (median)")
    axes[1].set_ylim(axes[0].get_ylim())
    fig.suptitle("Same placement rule, just explicit: node0 back to normal", fontsize=14)
    save(fig, "fig2_mpol_local.png")


def fig_scan(svc):
    d = svc["mpol_batch_vmstat_delta"]
    arms = ["default policy", "MPOL_LOCAL (exec wrapper)", "MPOL_LOCAL (set at process start)"]
    labels = ["default", "LOCAL\n(wrapper)", "LOCAL\n(at start)"]
    colors = [C_DEFAULT, C_LOCAL, C_LOCAL]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    series = axes[0]
    for arm, color in (("default policy", C_DEFAULT), ("MPOL_LOCAL (set at process start)", C_LOCAL)):
        pts = svc["vmstat_series"][arm]
        series.plot([p["t_s"] / 60 for p in pts], [max(p["pte_updates_per_s"], 1) for p in pts],
                    color=color, linewidth=2)
    series.set_yscale("log")
    series.set_xlabel("minutes into replay")
    series.set_title("PTEs made inaccessible per second")
    series.annotate("default", xy=(0.02, 0.9), xycoords="axes fraction", color=C_DEFAULT, fontsize=12)
    series.annotate("MPOL_LOCAL", xy=(0.02, 0.08), xycoords="axes fraction", color=C_LOCAL, fontsize=12)
    axes[1].set_yscale("log")
    bars(axes[1], labels, [d[a]["numa_pte_updates"] / 1e6 for a in arms], colors, fmt="{:,.1f}M")
    axes[1].set_ylabel("millions (log scale)")
    axes[1].set_title("numa_pte_updates, whole replay")
    bars(axes[2], labels, [d[a]["numa_pages_migrated"] / 1e3 for a in arms], colors, fmt="{:,.1f}K")
    axes[2].set_title("numa_pages_migrated, whole replay")
    save(fig, "fig3_scan_volume.png")


def fig_mechanism(mech):
    L = mech["during_node0_loading"]["node0_loader_threads"]
    arms = ["default policy", "MPOL_LOCAL"]
    labels = ["default", "MPOL_LOCAL"]
    colors = [C_DEFAULT, C_LOCAL]
    panels = [
        ("(1) loader's own kernel time\n(cores)", [L["kernel_time_cores"][a] for a in arms], "{:.2f}"),
        ("(2) TLB shootdown IRQ/s\non node0 CPUs",
         [mech["during_node0_loading"]["tlb_shootdown_irq_per_s_on_node0_cpus"][a] for a in arms], "{:,.0f}"),
        ("(3) loader samples blocked\non memory-map lock",
         [100 * L["blocked_on_address_space_lock_sample_fraction"][a] for a in arms], "{:.1f}%"),
        ("(4) loader run-queue wait\n÷ running time", [L["wait_over_run"][a] for a in arms], "{:.2f}"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    for ax, (title, vals, fmt) in zip(axes, panels):
        bars(ax, labels, vals, colors, fmt=fmt)
        ax.set_title(title, fontsize=11, pad=14)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    fig.suptitle("Where do node0 loaders lose time? (while node0 is loading)", fontsize=14)
    save(fig, "fig5_mechanism.png")


TL_ARMS = [("default", C_DEFAULT), ("MPOL_LOCAL", C_LOCAL)]
TL_BEFORE, TL_AFTER = 10, 42          # 从滚动加载开始前 10 s 画到开始后 42 s


def _window(arm):
    rolling = [l["start_s"] for l in arm["loads"] if l["start_s"] > 600]
    return min(rolling)


def fig_timeline(tl):
    """上: 每秒扫描量 (整机); 下: 每次加载在各 GPU 上的起止. 两臂同一时间轴."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 7.8), sharex="col",
                             gridspec_kw={"height_ratios": [1, 1.4]})
    for col, (name, color) in enumerate(TL_ARMS):
        arm = tl["arms"][name]
        z = _window(arm)
        top, bot = axes[0][col], axes[1][col]
        pts = [(p["t_s"] - z, max(p["pte_updates_per_s"], 1)) for p in arm["scan"]
               if -TL_BEFORE <= p["t_s"] - z <= TL_AFTER]
        top.plot([x for x, _ in pts], [y for _, y in pts], color=color, linewidth=1.8)
        top.set_yscale("log")
        top.set_ylim(1e3, 3e7)
        top.set_title(f"{name}: numa_pte_updates per second (whole host)", fontsize=12)
        for l in arm["loads"]:
            x0, x1, x2 = l["start_s"] - z, l["load_end_s"] - z, l["end_s"] - z
            if x2 < -TL_BEFORE or x0 > TL_AFTER:
                continue
            y = 5 - l["gpu"] + (0.5 if l["node"] == 0 else 0)
            bot.barh(y, x1 - x0, left=x0, height=0.7, color=color, edgecolor=INK, linewidth=0.8)
            bot.barh(y, x2 - x1, left=x1, height=0.7, color="white", edgecolor=color, hatch="///", linewidth=0.8)
        bot.axhline(1.75, color="#999999", linestyle="--", linewidth=1)
        bot.set_yticks([5.5, 4.5, 3.5, 2.5, 1, 0])
        bot.set_yticklabels(["GPU0", "GPU1", "GPU2", "GPU3", "GPU4", "GPU5"])
        bot.annotate("node0", (TL_AFTER, 5.9), ha="right", fontsize=11, color="#666666")
        bot.annotate("node1", (TL_AFTER, 1.35), ha="right", fontsize=11, color="#666666")
        med0 = sorted(l["load_ms"] for l in arm["loads"] if l["node"] == 0 and l["start_s"] > 600)
        med1 = sorted(l["load_ms"] for l in arm["loads"] if l["node"] == 1 and l["start_s"] > 600)
        bot.set_title(f"each load (solid = load, hatched = warmup); median load: node0 {med0[len(med0)//2]:.0f} ms, "
                      f"node1 {med1[len(med1)//2]:.0f} ms", fontsize=11)
        bot.set_xlabel("seconds since the rolling loads start")
        bot.set_xlim(-TL_BEFORE, TL_AFTER)
        bot.set_ylim(-0.6, 6.2)
        for a in (top, bot):
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
    save(fig, "fig8_timeline.png")


def fig_thread_timeline(tl):
    """node0 绑定线程 / 未绑核线程: 每 ~2 s 的运行核数与排队核数 (堆叠), 灰底是 node0 在加载的时段."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 7.8), sharex="col")
    ymax = {"node0": 0, "unbound": 0}
    for name, _ in TL_ARMS:
        arm = tl["arms"][name]
        z = _window(arm)
        for g in ymax:
            for p in arm["threads"]:
                if -TL_BEFORE <= p["t_s"] - z <= TL_AFTER:
                    ymax[g] = max(ymax[g], p[f"{g}_run_cores"] + p[f"{g}_wait_cores"])
    titles = {"node0": "threads bound to node0 (loaders, copy workers)", "unbound": "unbound threads (rest of the service)"}
    for col, (name, color) in enumerate(TL_ARMS):
        arm = tl["arms"][name]
        z = _window(arm)
        pts = [p for p in arm["threads"] if -TL_BEFORE <= p["t_s"] - z <= TL_AFTER]
        xs = [p["t_s"] - z for p in pts]
        for row, g in enumerate(("node0", "unbound")):
            ax = axes[row][col]
            for l in arm["loads"]:
                if l["node"] == 0 and -TL_BEFORE <= l["start_s"] - z <= TL_AFTER:
                    ax.axvspan(l["start_s"] - z, l["load_end_s"] - z, color="#e6e6e6", zorder=0, linewidth=0)
            run = [p[f"{g}_run_cores"] for p in pts]
            wait = [p[f"{g}_wait_cores"] for p in pts]
            ax.fill_between(xs, 0, run, color=color, alpha=0.9, step="mid", linewidth=0)
            ax.fill_between(xs, run, [a + b for a, b in zip(run, wait)], facecolor="white", edgecolor=color,
                            hatch="///", step="mid", linewidth=0.8)
            ax.set_ylim(0, ymax[g] * 1.15)
            ax.set_title(f"{name}: {titles[g]}", fontsize=11)
            ax.set_ylabel("cores")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[1][col].set_xlabel("seconds since the rolling loads start")
        axes[1][col].set_xlim(-TL_BEFORE, TL_AFTER)
    axes[0][0].annotate("solid = running\nhatched = waiting for a CPU\ngrey = node0 loading", (0.02, 0.97),
                        xycoords="axes fraction", va="top", fontsize=10, color=INK)
    save(fig, "fig9_thread_timeline.png")


def fig_placement(pl):
    """未绑核线程跑在哪个 node、偏好哪个 node, 以及两个 node 的忙碌度, 随时间变化."""
    arms = [("default", C_DEFAULT), ("MPOL_LOCAL", C_LOCAL)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8.4), sharex=True)

    def marks(ax, arm, color, label_start=False):
        a = pl["arms"][arm]
        w0, w1 = a["rolling_load_window_s"]
        ax.axvspan(w0, w1, color=color, alpha=0.12, linewidth=0)
        ax.axvline(a["processing_start_s"], color=color, linestyle=":", linewidth=1.5)
        if label_start:
            ax.annotate("real-time processing\nstarts", (a["processing_start_s"], 0.05), xytext=(-8, 0),
                        textcoords="offset points", ha="right", fontsize=9, color="#555555")
            ax.annotate("rolling\nloads", ((w0 + w1) / 2, 0.05), ha="center", fontsize=9, color="#555555")

    ax = axes[0][0]
    for arm, color in arms:
        pts = [(x["t_mid"], x["run_share_node0"]) for x in pl["arms"][arm]["series"] if x["run_share_node0"] is not None]
        ax.plot([t for t, _ in pts], [100 * v for _, v in pts], color=color, linewidth=2.2, label=arm)
        marks(ax, arm, color, label_start=(arm == "default"))
    ax.axhline(50, color="#999999", linestyle="--", linewidth=1)
    ax.set_ylim(0, 105)
    ax.set_ylabel("%")
    ax.set_title("unbound threads: share of CPU time on node0", fontsize=12)
    ax.legend(loc="lower left", fontsize=10)

    ax = axes[0][1]
    for arm, color in arms:
        pts = [(x["t_mid"], x["run_share_pref0"]) for x in pl["arms"][arm]["series"] if x["run_share_pref0"] is not None]
        ax.plot([t for t, _ in pts], [100 * v for _, v in pts], color=color, linewidth=2.2, label=arm)
        marks(ax, arm, color)
    ax.set_ylim(-3, 105)
    ax.annotate("MPOL_LOCAL: no preferred node (-1)", (20, 6), fontsize=10, color=C_LOCAL)
    ax.set_title("unbound threads: share of CPU time from threads preferring node0", fontsize=12)

    for ax, (arm, color) in zip(axes[1], arms):
        pts = pl["arms"][arm]["series"]
        xs = [x["t_mid"] for x in pts if x["busy_node0"] is not None]
        ax.plot(xs, [100 * x["busy_node0"] for x in pts if x["busy_node0"] is not None], color=color, linewidth=2.2)
        ax.plot(xs, [100 * x["busy_node1"] for x in pts if x["busy_node0"] is not None], color=color, linewidth=2.2, linestyle="--")
        y0, y1 = 100 * pts[-2]["busy_node0"], 100 * pts[-2]["busy_node1"]
        dy = 9 if abs(y0 - y1) < 8 else 0          # 两条线挨得近时把标签上下错开
        ax.annotate("node0", (xs[-1], y0), xytext=(4, -dy if y0 <= y1 else dy), textcoords="offset points", fontsize=10, color=INK, va="center")
        ax.annotate("node1", (xs[-1], y1), xytext=(4, dy if y0 <= y1 else -dy), textcoords="offset points", fontsize=10, color=INK, va="center")
        marks(ax, arm, color)
        ax.set_ylim(0, 105)
        ax.set_ylabel("% busy")
        ax.set_title(f"{arm}: CPU busy per node (solid node0, dashed node1)", fontsize=12)
        ax.set_xlabel("seconds since the worker processes started")
    for row in axes:
        for ax in row:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    save(fig, "fig10_placement.png")


REPRO_ROWS = [
    ("cpu90-split", "90 CPU-bound background procs, memory on node0, start split 45/45"),
]


def fig_repro(repro):
    import statistics as st
    rows = [(k, t) for k, t in REPRO_ROWS if k in repro and {"default", "local"} <= set(repro[k])]
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.9 * len(rows)), squeeze=False)
    arms = [("default", C_DEFAULT, "default"), ("local", C_LOCAL, "MPOL_LOCAL")]
    for row, (key, title) in zip(axes, rows):
        runs = repro[key]
        n_bg = runs["default"][0]["bg_procs"]
        # (a) 每个 node 的 loader: 各轮各卡中位数再取中位数, 单位秒
        ax = row[0]
        w = 0.38
        for i, (arm, color, label) in enumerate(arms):
            vals = [st.median(L["load_ms_median"] for r in runs[arm] for L in r["loaders"] if L["node"] == n) / 1000
                    for n in (0, 1)]
            xs = [n + (i - 0.5) * w for n in (0, 1)]
            ax.bar(xs, vals, w, color=color, edgecolor=INK, label=label)
            for x, v in zip(xs, vals):
                ax.annotate(f"{v:.2f}", (x, v), ha="center", va="bottom", xytext=(0, 2),
                            textcoords="offset points", fontsize=10, color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["node0 loaders", "node1 loaders"])
        ax.set_ylabel("seconds per 2 GiB load")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_title(title, fontsize=11, loc="left")
        # (b) 后台进程平均有几个跑在 node0
        ax = row[1]
        vals = [st.mean(r["bg_procs_on_node0_mean"] for r in runs[a]) for a, _, _ in arms]
        bars(ax, [l for _, _, l in arms], vals, [c for _, c, _ in arms], fmt="{:.1f}")
        ax.axhline(n_bg / 2, color="#999999", linestyle=":", linewidth=1)
        ax.annotate("even split", (1.45, n_bg / 2), ha="right", va="bottom", fontsize=9, color="#666666")
        ax.axhline(64, color="#999999", linestyle="--", linewidth=1)
        ax.annotate("node0 has 64 CPUs", (1.45, 64), ha="right", va="bottom", fontsize=9, color="#666666")
        ax.set_ylim(0, max(max(vals), 64) * 1.25)
        ax.set_title(f"background procs on node0 (of {n_bg})", fontsize=11)
        # (c) 每个 node 的 CPU 忙碌度
        ax = row[2]
        for i, (arm, color, label) in enumerate(arms):
            b0 = st.mean(r["node_cpu_busy_fraction"]["0"] for r in runs[arm]) * 100
            b1 = st.mean(r["node_cpu_busy_fraction"]["1"] for r in runs[arm]) * 100
            ax.bar(i - w / 2, b0, w, color=color, edgecolor=INK)
            ax.bar(i + w / 2, b1, w, color="white", edgecolor=color, hatch="//", linewidth=1.5)
            ax.annotate(f"{b0:.0f}%", (i - w / 2, b0), ha="center", va="bottom", fontsize=10, color=INK)
            ax.annotate(f"{b1:.0f}%", (i + w / 2, b1), ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([l for _, _, l in arms])
        ax.set_ylim(0, 118)
        ax.set_title("CPU busy: solid = node0, hatched = node1", fontsize=11)
        for a in row:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
    save(fig, "fig7_repro.png")


if __name__ == "__main__":
    plt.xkcd(scale=0.8, length=120, randomness=2)
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    svc = load("service_replay_summary.json")
    mech = load("mechanism_summary.json")
    fig_numactl(svc)
    fig_mpol(svc)
    fig_scan(svc)
    fig_mechanism(mech)
    fig_placement(load("placement_timeline.json"))
    tl = load("timeline_mech.json")
    fig_timeline(tl)
    fig_thread_timeline(tl)
    if os.path.exists(os.path.join(DATA, "repro_results.json")):
        fig_repro(load("repro_results.json"))
