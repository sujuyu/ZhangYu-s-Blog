#!/usr/bin/env python3
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OLD_TRACE = Path("/home/zy429782/fx_experiments/debug_onelive/dynamicLib/fx_user_model_header_model/search/tracing_files/batch_size_1280_timeline_tracing.json")
NEW_TRACE = Path("/home/zy429782/fx_experiments/debug_onelive/biz_plugin/search/tracing_files/batch_size_1280_timeline_tracing.json")
EAGER_TRACE = Path("/home/zy429782/fx_experiments/debug_onelive/experiment/eager_broadcast_matmul_traces/20260509_140734/aten_matmul_implicit_broadcast_trace.json")


PALETTE = {
    "clone_expand/copy": "#d95f5f",
    "broadcast_bmm": "#4b78d8",
    "custom_matmul": "#2a9d8f",
    "attention": "#8a63d2",
    "dense/elementwise": "#c18d19",
    "other": "#8f969e",
}


def load_events(path):
    with path.open() as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def classify_kernel(name):
    if "bmm_clone_expand" in name or "_unsafe_view_bmm" in name or "_bmm_" in name or name.endswith("bmm"):
        return "broadcast_bmm"
    if "clone_expand" in name or "direct_copy" in name or "copy" in name and "large_batch" not in name:
        return "clone_expand/copy"
    if "large_batch_per_token_broadcast_weight_matmul" in name:
        return "custom_matmul"
    if "attention" in name:
        return "attention"
    if any(token in name for token in ["layer_norm", "addmm", "elu", "cat", "select", "squeeze", "_to_copy"]):
        return "dense/elementwise"
    return "other"


def summarize_trace(path):
    events = load_events(path)
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    sync = sorted([e for e in events if e.get("name") == "hipEventSynchronize"], key=lambda e: e["ts"])
    by_name = defaultdict(lambda: [0.0, 0])
    by_cat = defaultdict(lambda: [0.0, 0])
    for e in kernels:
        by_name[e["name"]][0] += float(e.get("dur", 0.0))
        by_name[e["name"]][1] += 1
        cat = classify_kernel(e["name"])
        by_cat[cat][0] += float(e.get("dur", 0.0))
        by_cat[cat][1] += 1
    gaps = [sync[i + 1]["ts"] - sync[i]["ts"] for i in range(len(sync) - 1)]
    return {
        "kernel_total_us": sum(e.get("dur", 0.0) for e in kernels),
        "kernel_count": len(kernels),
        "sync_count": len(sync),
        "sync_gap_avg_us": sum(gaps) / len(gaps),
        "sync_gap_first_us": gaps[0],
        "by_name": by_name,
        "by_cat": by_cat,
    }


def svg_text(x, y, text, size=14, weight="400", fill="#222", anchor="start"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
        f'{html.escape(str(text))}</text>'
    )


def draw_timeline(path, out_name, title, window_index=1):
    events = load_events(path)
    sync = sorted([e for e in events if e.get("name") == "hipEventSynchronize"], key=lambda e: e["ts"])
    kernels = sorted([e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"], key=lambda e: e["ts"])
    start = sync[window_index]["ts"]
    end = sync[window_index + 1]["ts"]
    span = end - start
    win = [e for e in kernels if start <= e["ts"] <= end + 80]
    rows = ["clone_expand/copy", "broadcast_bmm", "custom_matmul", "attention", "dense/elementwise", "other"]
    rows = [r for r in rows if any(classify_kernel(e["name"]) == r for e in win)]

    width = 1180
    left = 190
    top = 92
    row_h = 44
    timeline_w = 900
    height = top + row_h * len(rows) + 92
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(34, 34, title, 22, "700"),
        svg_text(34, 60, f"one request window: {span / 1000:.2f} ms, kernels: {len(win)}", 13, fill="#555"),
    ]
    tick_count = 6
    for i in range(tick_count + 1):
        x = left + timeline_w * i / tick_count
        ts = span * i / tick_count / 1000
        parts.append(f'<line x1="{x:.2f}" y1="{top - 18}" x2="{x:.2f}" y2="{top + row_h * len(rows) - 12}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(svg_text(x, top - 28, f"{ts:.1f}ms", 11, fill="#666", anchor="middle"))
    parts.append(f'<line x1="{left}" y1="{top - 18}" x2="{left + timeline_w}" y2="{top - 18}" stroke="#222" stroke-width="1"/>')
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        parts.append(svg_text(34, y + 23, row, 14, "700", PALETTE[row]))
        parts.append(f'<rect x="{left}" y="{y}" width="{timeline_w}" height="30" fill="#f8fafc" stroke="#e5e7eb" rx="3"/>')
        label_budget = 0
        for e in win:
            if classify_kernel(e["name"]) != row:
                continue
            x = left + (e["ts"] - start) / span * timeline_w
            w = max(1.3, e["dur"] / span * timeline_w)
            if x > left + timeline_w:
                continue
            w = min(w, left + timeline_w - x)
            parts.append(
                f'<rect x="{x:.2f}" y="{y + 5}" width="{w:.2f}" height="20" '
                f'fill="{PALETTE[row]}" opacity="0.88" rx="2">'
                f'<title>{html.escape(e["name"])} {e["dur"]:.2f} us</title></rect>'
            )
            if w > 72 and label_budget < 5:
                label = e["name"]
                if "large_batch_per_token_broadcast_weight_matmul" in label:
                    label = "custom matmul"
                elif "clone_expand" in label and "bmm" not in label:
                    label = "clone_expand"
                elif "bmm_clone_expand" in label:
                    label = "bmm"
                elif "attention" in label:
                    label = "attention"
                else:
                    label = "kernel"
                parts.append(svg_text(x + 5, y + 20, label, 10, "700", "#ffffff"))
                label_budget += 1
    legend_x = left
    legend_y = height - 32
    for row in rows:
        parts.append(f'<rect x="{legend_x}" y="{legend_y - 11}" width="12" height="12" fill="{PALETTE[row]}" rx="2"/>')
        parts.append(svg_text(legend_x + 18, legend_y, row, 12, fill="#555"))
        legend_x += 160
    parts.append("</svg>")
    (ROOT / out_name).write_text("\n".join(parts))


def draw_lowering():
    width, height = 1050, 330
    boxes = [
        (45, 105, 195, 88, "FX graph", "[B,T,1,K] @ [1,T,K,N]", "#e8f1ff"),
        (292, 105, 195, 88, "aten.matmul", "broadcast batch dim", "#fff3df"),
        (539, 105, 195, 88, "expand + clone", "[B,T,K,N] materialized", "#ffe8e8"),
        (786, 105, 195, 88, "bmm", "[B*T,1,K] x [B*T,K,N]", "#e9f7ef"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(42, 42, "broadcast matmul 在通用实现里的执行链路", 23, "700"),
        svg_text(42, 68, "weight 的 batch 维本来是 1，通用 matmul 先广播成 B，再走 bmm", 14, fill="#555"),
    ]
    for i, (x, y, w, h, title, sub, fill) in enumerate(boxes):
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="#333" stroke-width="1.2" rx="6"/>')
        parts.append(svg_text(x + w / 2, y + 36, title, 17, "700", anchor="middle"))
        parts.append(svg_text(x + w / 2, y + 62, sub, 12, fill="#333", anchor="middle"))
        if i + 1 < len(boxes):
            ax = x + w + 16
            ay = y + h / 2
            parts.append(f'<line x1="{ax}" y1="{ay}" x2="{ax + 34}" y2="{ay}" stroke="#333" stroke-width="1.6" marker-end="url(#arrow)"/>')
    parts.append(
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#333"/></marker></defs>'
    )
    parts.append(svg_text(42, 252, "问题点：expand 本身可以是 metadata view，但 broadcast 后的 weight stride-0 无法直接喂给 bmm 的常规 batch 矩阵布局，reshape/clone 触发实际 copy。", 14, fill="#333"))
    parts.append("</svg>")
    (ROOT / "image-3.svg").write_text("\n".join(parts))


def draw_tiling():
    width, height = 1120, 620
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(42, 42, "Triton tiling：batch block 复用 per-token weight", 24, "700"),
        svg_text(42, 70, "grid = (ceil(N / BLOCK_N), T, ceil(B / BLOCK_B))", 14, fill="#555"),
        '<defs>'
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#333"/></marker>'
        '<pattern id="diagBlue" patternUnits="userSpaceOnUse" width="8" height="8"><path d="M0,8 L8,0" stroke="#8fb1e8" stroke-width="1"/></pattern>'
        '<pattern id="diagGreen" patternUnits="userSpaceOnUse" width="8" height="8"><path d="M0,0 L8,8" stroke="#8bc9aa" stroke-width="1"/></pattern>'
        '</defs>',
    ]

    # Left: logical tensors.
    parts += [
        svg_text(70, 123, "logical tensors", 16, "700", "#333"),
        '<rect x="70" y="150" width="285" height="115" fill="#f8fafc" stroke="#cbd5e1" rx="6"/>',
        svg_text(92, 178, "x: [B, T, 1, K]", 15, "700", "#244a8f"),
        '<rect x="92" y="196" width="210" height="34" fill="#dbeafe" stroke="#3b6fc4" rx="3"/>',
        '<rect x="126" y="196" width="58" height="34" fill="url(#diagBlue)" stroke="#3b6fc4" rx="3"/>',
        svg_text(155, 218, "BLOCK_B x K", 12, "700", "#244a8f", "middle"),
        svg_text(92, 250, "selected token t, selected batch block", 12, fill="#64748b"),
        '<rect x="70" y="292" width="285" height="115" fill="#f8fafc" stroke="#cbd5e1" rx="6"/>',
        svg_text(92, 320, "weight: [1, T, K, N]", 15, "700", "#26734d"),
        '<rect x="92" y="338" width="210" height="34" fill="#dcfce7" stroke="#2f855a" rx="3"/>',
        '<rect x="180" y="338" width="66" height="34" fill="url(#diagGreen)" stroke="#2f855a" rx="3"/>',
        svg_text(213, 360, "K x BLOCK_N", 12, "700", "#26734d", "middle"),
        svg_text(92, 392, "batch dim is broadcast, no materialization", 12, fill="#64748b"),
    ]

    # Middle: one Triton program.
    parts += [
        '<rect x="430" y="150" width="270" height="257" fill="#ffffff" stroke="#334155" stroke-width="1.5" rx="8"/>',
        svg_text(565, 183, "one Triton program", 17, "700", "#111827", "middle"),
        svg_text(565, 209, "pid_n, pid_t, pid_b", 13, fill="#475569", anchor="middle"),
        '<rect x="468" y="236" width="96" height="72" fill="#dbeafe" stroke="#3b6fc4" rx="4"/>',
        svg_text(516, 267, "x", 18, "700", "#244a8f", "middle"),
        svg_text(516, 289, "evict_first", 12, fill="#244a8f", anchor="middle"),
        '<text x="584" y="279" font-family="Arial, sans-serif" font-size="26" font-weight="700" fill="#334155">×</text>',
        '<rect x="618" y="236" width="46" height="72" fill="#dcfce7" stroke="#2f855a" rx="4"/>',
        svg_text(641, 267, "w", 18, "700", "#26734d", "middle"),
        svg_text(641, 289, "evict_last", 12, fill="#26734d", anchor="middle"),
        '<line x1="516" y1="318" x2="516" y2="346" stroke="#334155" stroke-width="1.7" marker-end="url(#arrow)"/>',
        '<line x1="641" y1="318" x2="641" y2="346" stroke="#334155" stroke-width="1.7" marker-end="url(#arrow)"/>',
        '<rect x="494" y="352" width="140" height="34" fill="#fff7ed" stroke="#c77700" rx="4"/>',
        svg_text(564, 374, "tl.dot -> out tile", 13, "700", "#9a6500", "middle"),
    ]

    # Right: output and launch space.
    parts += [
        svg_text(785, 123, "program grid", 16, "700", "#333"),
        '<rect x="785" y="150" width="260" height="160" fill="#f8fafc" stroke="#cbd5e1" rx="6"/>',
    ]
    gx, gy, cell_w, cell_h = 820, 185, 44, 28
    for by in range(4):
        for nx in range(5):
            fill = "#e2e8f0"
            stroke = "#94a3b8"
            if by == 1 and nx == 2:
                fill = "#fff7ed"
                stroke = "#c77700"
            parts.append(f'<rect x="{gx + nx * cell_w}" y="{gy + by * cell_h}" width="{cell_w - 5}" height="{cell_h - 5}" fill="{fill}" stroke="{stroke}" rx="3"/>')
    parts += [
        svg_text(930, 172, "fixed token t", 12, fill="#64748b", anchor="middle"),
        svg_text(930, 320, "N blocks × B blocks", 12, fill="#64748b", anchor="middle"),
        '<rect x="785" y="338" width="260" height="69" fill="#fff7ed" stroke="#c77700" rx="6"/>',
        svg_text(915, 367, "out: [B, T, 1, N]", 15, "700", "#9a6500", "middle"),
        svg_text(915, 390, "store [BLOCK_B, BLOCK_N]", 12, fill="#9a6500", anchor="middle"),
    ]

    parts += [
        '<line x1="365" y1="214" x2="420" y2="260" stroke="#334155" stroke-width="1.7" marker-end="url(#arrow)"/>',
        '<line x1="365" y1="350" x2="420" y2="292" stroke="#334155" stroke-width="1.7" marker-end="url(#arrow)"/>',
        '<line x1="710" y1="370" x2="775" y2="370" stroke="#334155" stroke-width="1.7" marker-end="url(#arrow)"/>',
    ]

    y0 = 470
    notes = [
        "T 维作为 grid 维度：每个 token 使用独立 KxN weight。",
        "N 维切 BLOCK_N：输出列方向独立，控制单个 program 的寄存器和矩阵规模。",
        "B 维切 BLOCK_B：提高 block/warp 数，同时让同一份 weight tile 被一组 batch 复用。",
        "K 维不切不循环：一次 tl.dot 完成 reduce，直接走 tensorcore。",
    ]
    for i, note in enumerate(notes):
        parts.append(svg_text(62, y0 + i * 28, f"{i + 1}. {note}", 14, fill="#333"))
    parts.append("</svg>")
    (ROOT / "image-4.svg").write_text("\n".join(parts))


def draw_speedup(old, new):
    width, height = 880, 420
    left, bottom = 95, 330
    chart_w, chart_h = 690, 245
    old_ms = old["sync_gap_avg_us"] / 1000
    new_ms = new["sync_gap_avg_us"] / 1000
    values = [old_ms, new_ms]
    labels = ["原始", "优化后"]
    colors = ["#d95f5f", "#2a9d8f"]
    max_v = max(values) * 1.18
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(42, 42, "整体耗时对比", 23, "700"),
        svg_text(42, 68, f"hipEventSynchronize 起始间隔均值：{old_ms:.2f} ms -> {new_ms:.2f} ms，约 {old_ms / new_ms:.1f}x", 14, fill="#555"),
        f'<line x1="{left}" y1="{bottom}" x2="{left + chart_w}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{bottom - chart_h}" x2="{left}" y2="{bottom}" stroke="#333"/>',
    ]
    for i, (label, val, color) in enumerate(zip(labels, values, colors)):
        bw = 170
        x = left + 120 + i * 250
        h = val / max_v * chart_h
        y = bottom - h
        parts.append(f'<rect x="{x}" y="{y:.2f}" width="{bw}" height="{h:.2f}" fill="{color}" rx="4"/>')
        parts.append(svg_text(x + bw / 2, y - 10, f"{val:.2f} ms", 16, "700", "#222", "middle"))
        parts.append(svg_text(x + bw / 2, bottom + 30, label, 15, "700", "#333", "middle"))
    parts.append(svg_text(42, 382, f"kernel total / 10 runs: {old['kernel_total_us'] / 1000:.1f} ms -> {new['kernel_total_us'] / 1000:.1f} ms", 13, fill="#555"))
    parts.append("</svg>")
    (ROOT / "image-5.svg").write_text("\n".join(parts))


def write_stats(old, new):
    lines = []
    for title, stat in [("old", old), ("new", new)]:
        lines.append(f"[{title}]")
        lines.append(f"kernel_total_us={stat['kernel_total_us']:.3f}")
        lines.append(f"kernel_count={stat['kernel_count']}")
        lines.append(f"sync_gap_avg_us={stat['sync_gap_avg_us']:.3f}")
        lines.append(f"sync_gap_first_us={stat['sync_gap_first_us']:.3f}")
        lines.append("top kernels:")
        for name, (dur, cnt) in sorted(stat["by_name"].items(), key=lambda kv: kv[1][0], reverse=True)[:10]:
            lines.append(f"  {dur:10.3f} us  {cnt:4d}x  avg={dur / cnt:8.3f} us  {name}")
        lines.append("categories:")
        for cat, (dur, cnt) in sorted(stat["by_cat"].items(), key=lambda kv: kv[1][0], reverse=True):
            lines.append(f"  {dur:10.3f} us  {cnt:4d}x  {cat}")
        lines.append("")

    if EAGER_TRACE.exists():
        events = load_events(EAGER_TRACE)
        cats = Counter()
        for e in events:
            if e.get("cat") == "kernel":
                cats[e["name"]] += 1
        lines.append("[eager trace kernel names]")
        for name, cnt in cats.items():
            lines.append(f"  {cnt}x {name}")
    (ROOT / "stats.txt").write_text("\n".join(lines))


def main():
    old = summarize_trace(OLD_TRACE)
    new = summarize_trace(NEW_TRACE)
    draw_timeline(OLD_TRACE, "image-1.svg", "优化前：broadcast weight matmul 被拆成 copy + bmm", 1)
    draw_timeline(NEW_TRACE, "image-2.svg", "优化后：per-token broadcast matmul 直接在 kernel 内完成", 1)
    draw_lowering()
    draw_tiling()
    draw_speedup(old, new)
    write_stats(old, new)


if __name__ == "__main__":
    main()
