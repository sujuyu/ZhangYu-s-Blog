#!/usr/bin/env python3
"""Generate dependency-free, sanitized SVG figures for the article."""

from __future__ import annotations

from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent


def write_svg(name: str, body: str, width: int, height: int, background: str = "white") -> None:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="{background}"/>
<style>
  text {{ font-family: Arial, Helvetica, sans-serif; }}
  .title {{ font-size: 23px; font-weight: 700; }}
  .subtitle {{ font-size: 14px; fill: #555; }}
  .label {{ font-size: 13px; }}
  .small {{ font-size: 12px; }}
  .axis {{ stroke: #333; stroke-width: 1.2; }}
  .grid {{ stroke: #999; stroke-width: 0.7; opacity: 0.25; }}
</style>
{body}
</svg>
'''
    (OUT_DIR / name).write_text(svg)


def bar_chart(
    x0: float,
    y0: float,
    width: float,
    height: float,
    values: list[float],
    labels: list[str],
    colors: list[str],
    max_value: float,
    title: str,
    unit: str,
) -> str:
    parts = [f'<text x="{x0 + width / 2}" y="{y0 - 24}" text-anchor="middle" class="label" font-weight="700">{title}</text>']
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" class="axis"/>')
    parts.append(f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" y2="{y0 + height}" class="axis"/>')
    for i in range(5):
        value = max_value * i / 4
        y = y0 + height - height * i / 4
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + width}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.0f}</text>')
    slot = width / len(values)
    for i, (value, label, color) in enumerate(zip(values, labels, colors)):
        bar_w = slot * 0.58
        x = x0 + slot * i + (slot - bar_w) / 2
        h = height * value / max_value
        y = y0 + height - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}" stroke="#333"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" class="small">{value:.2f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y0 + height + 21:.1f}" text-anchor="middle" class="small">{label}</text>')
    parts.append(f'<text x="{x0 - 50}" y="{y0 + height / 2}" text-anchor="middle" class="small" transform="rotate(-90 {x0 - 50} {y0 + height / 2})">{unit}</text>')
    return "\n".join(parts)


def plot_rounding_error() -> None:
    labels = ["folded BF16", "eager no-fold"]
    colors = ["#9b2226", "#2a9d8f"]
    relative_l2 = [46.327, 0.2459]
    max_abs = [256.0, 1.20096]
    body = '''
<text x="700" y="42" text-anchor="middle" class="title">Same-input error comparison on one BN + BMM module</text>
<text x="700" y="67" text-anchor="middle" class="subtitle">first-layer output, B=128, E=171, K=7926, N=256; FP32 reference with TF32 disabled</text>
'''
    body += bar_chart(90, 130, 560, 300, relative_l2, labels, colors, 50, "Relative L2", "%")
    body += bar_chart(790, 130, 520, 300, max_abs, labels, colors, 280, "Max absolute error", "absolute error")
    write_svg("bn_fold_rounding_error.svg", body, 1400, 520)


def plot_timeline() -> None:
    scale = 455
    origin = 230
    no_fold = [
        (0.000, 1.412700, "BN affine + cast", "#e76f51"),
        (1.413628, 0.815071, "BMM0", "#457b9d"),
        (2.229691, 0.009824, "", "#2a9d8f"),
    ]
    folded = [(0.000, 0.637695, "BMM0 + bias + ReLU", "#457b9d")]
    parts = [
        '<text x="700" y="42" text-anchor="middle" class="title" fill="white">Actual profiler timeline</text>',
        '<text x="700" y="68" text-anchor="middle" class="subtitle" fill="#cccccc">B=128, E=171, K=7926, N=256</text>',
    ]
    for label, y, events in [("No fold", 145, no_fold), ("BN folded", 260, folded)]:
        parts.append(f'<text x="195" y="{y + 27}" text-anchor="end" class="label" fill="white">{label}</text>')
        for start, duration, name, color in events:
            x = origin + start * scale
            w = max(duration * scale, 4)
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="52" rx="4" fill="{color}" stroke="#f1faee"/>')
            if duration > 0.2:
                parts.append(f'<text x="{x + w / 2:.1f}" y="{y + 23}" text-anchor="middle" class="label" fill="white">{name}</text>')
                parts.append(f'<text x="{x + w / 2:.1f}" y="{y + 42}" text-anchor="middle" class="small" fill="white">{duration:.3f} ms</text>')
    for i in range(6):
        x = origin + i * 0.5 * scale
        parts.append(f'<line x1="{x:.1f}" y1="118" x2="{x:.1f}" y2="336" class="grid" stroke="white"/>')
        parts.append(f'<text x="{x:.1f}" y="360" text-anchor="middle" class="small" fill="white">{i * 0.5:.1f}</text>')
    parts.append('<text x="700" y="392" text-anchor="middle" class="label" fill="white">time from the beginning of this operator region (ms)</text>')
    parts.append('<text x="700" y="430" text-anchor="middle" class="small" fill="#cccccc">The no-fold path writes the expanded BN result to global memory before BMM0.</text>')
    write_svg("timeline_bn_bmm.svg", "\n".join(parts), 1400, 470, "#171717")


def plot_latency_mfu() -> None:
    labels = ["Folded BF16", "Custom no-BN", "Custom FP32-BN", "Folded FP32+TF32"]
    colors = ["#457b9d", "#2a9d8f", "#e9c46a", "#e76f51"]
    body = '''
<text x="700" y="42" text-anchor="middle" class="title">Isolated BN + BMM performance on NVIDIA A100</text>
<text x="700" y="67" text-anchor="middle" class="subtitle">E=171, K=7926, N=256; median CUDA Event latency</text>
'''
    body += bar_chart(100, 130, 520, 300, [0.644, 0.656, 0.813, 1.453], labels, colors, 1.6, "B=128 latency", "ms")
    body += bar_chart(800, 130, 500, 300, [44.20, 43.37, 35.02, 39.19], labels, colors, 60, "B=128 MFU", "% of dtype peak")
    body += '<text x="700" y="500" text-anchor="middle" class="small">BF16 MFU uses 312 TFLOP/s peak; TF32 MFU uses 156 TFLOP/s peak.</text>'
    write_svg("latency_mfu.svg", body, 1400, 540)


if __name__ == "__main__":
    plot_rounding_error()
    plot_timeline()
    plot_latency_mfu()
