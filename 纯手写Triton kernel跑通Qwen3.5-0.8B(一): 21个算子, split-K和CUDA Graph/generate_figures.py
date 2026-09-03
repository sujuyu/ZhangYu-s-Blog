#!/usr/bin/env python3
"""生成本文用的手绘卡通风格 SVG 配图, 不依赖任何第三方库.

手绘感靠三样东西堆出来:
  1. feTurbulence + feDisplacementMap 把直线抖成手画的歪线;
  2. 圆头线帽 + 略粗的描边;
  3. 楷体/Comic Sans 一类的字体栈.
"""

from __future__ import annotations

import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# 字体栈分两段: 前半段挑手写体, 后半段兜底中文.
# 浏览器是逐字符回退的, 所以英文数字会命中 Comic Sans / 楷体一类,
# 中文在没装楷体的机器上会落到 Noto / 苹方 / 雅黑, 至少不会变成豆腐块.
FONT = ('"Comic Sans MS", "Chalkboard SE", "Kaiti SC", "STKaiti", KaiTi, '
        '"Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", '
        '"Hiragino Sans GB", "Microsoft YaHei", sans-serif')

# 本机没有浏览器时用 ImageMagick 粗看布局: FIG_FONT=... python generate_figures.py
# ImageMagick 自带的 SVG 渲染器不做字体回退, 也不支持 filter, 只能用来检查有没有
# 文字溢出/重叠, 手绘效果得在浏览器里看.
FONT = os.environ.get("FIG_FONT", FONT)

# 配色: 低饱和, 像马克笔
INK = "#3d3a36"
GDN = "#7fb3d5"
ATTN = "#f0a868"
IDLE = "#e8e4de"
BUSY = "#82c09a"
WARN = "#e07a6b"
PAPER = "#fdfcf8"


def svg(name: str, body: str, w: int, h: int) -> None:
    out = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
<defs>
  <filter id="wob" x="-6%" y="-6%" width="112%" height="112%">
    <feTurbulence type="fractalNoise" baseFrequency="0.018" numOctaves="3" seed="7" result="n"/>
    <feDisplacementMap in="SourceGraphic" in2="n" scale="2.4" xChannelSelector="R" yChannelSelector="G"/>
  </filter>
  <filter id="wob2" x="-6%" y="-6%" width="112%" height="112%">
    <feTurbulence type="fractalNoise" baseFrequency="0.03" numOctaves="2" seed="19" result="n"/>
    <feDisplacementMap in="SourceGraphic" in2="n" scale="1.5" xChannelSelector="R" yChannelSelector="G"/>
  </filter>
  <marker id="arw" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 1 L 9 5 L 0 9 z" fill="{INK}"/>
  </marker>
</defs>
<rect width="100%" height="100%" fill="{PAPER}"/>
<style>
  text {{ font-family: {FONT}; fill: {INK}; }}
  .t  {{ font-size: 21px; font-weight: 700; }}
  .h  {{ font-size: 15px; font-weight: 700; }}
  .l  {{ font-size: 13px; }}
  .s  {{ font-size: 11.5px; fill: #6b6660; }}
  .box {{ stroke: {INK}; stroke-width: 2; stroke-linejoin: round; }}
  .ln {{ stroke: {INK}; stroke-width: 2; fill: none;
         stroke-linecap: round; stroke-linejoin: round; }}
  .dash {{ stroke: {INK}; stroke-width: 1.8; fill: none;
           stroke-dasharray: 6 5; stroke-linecap: round; }}
</style>
{body}
</svg>
'''
    (OUT_DIR / name).write_text(out, encoding="utf-8")
    print(f"  {name}")


def box(x, y, w, h, fill, rx=6, cls="box"):
    return f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>'


def txt(x, y, s, cls="l", anchor="middle", extra=""):
    return f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}" {extra}>{s}</text>'


# --------------------------------------------------------------------------
# 图 1: Qwen3.5-0.8B 的层结构
# --------------------------------------------------------------------------
def fig_structure() -> None:
    W, H = 900, 470
    b = [txt(W / 2, 40, "Qwen3.5-0.8B 的 24 层长这样", "t")]
    b.append(txt(W / 2, 64, "3 个 Gated DeltaNet + 1 个全注意力, 重复 6 次", "s"))

    # 24 层的条带
    x0, y0, bw, bh, gap = 46, 92, 30, 42, 5
    b.append('<g filter="url(#wob2)">')
    for i in range(24):
        x = x0 + i * (bw + gap)
        is_attn = i in (3, 7, 11, 15, 19, 23)
        b.append(box(x, y0, bw, bh, ATTN if is_attn else GDN))
        b.append(txt(x + bw / 2, y0 + 27, str(i), "s"))
    b.append("</g>")

    # 第一个重复单元的括号
    ux = x0 - 3
    uw = 4 * (bw + gap) - gap + 6
    b.append(f'<path class="ln" d="M {ux} {y0 - 12} L {ux} {y0 - 20} L {ux + uw} {y0 - 20} '
             f'L {ux + uw} {y0 - 12}" filter="url(#wob2)"/>')
    b.append(txt(ux + uw / 2, y0 - 27, "一个重复单元", "s"))

    # 两个说明卡片
    cy = 178
    b.append('<g filter="url(#wob)">')
    b.append(box(46, cy, 390, 250, "#eaf3fa"))
    b.append(box(468, cy, 390, 250, "#fdf1e4"))
    b.append("</g>")

    b.append(txt(64, cy + 30, "■ 18 层 Gated DeltaNet", "h", "start"))
    for i, s in enumerate([
        "线性注意力. 每层带一个 kernel=4 的",
        "depthwise causal Conv1D, 外加 delta rule 递推.",
        "",
        "状态是一个 [16,128,128] 的矩阵,",
        "跟上下文多长完全没关系.",
        "",
        "decode 时只要把这个矩阵原地更新一次,",
        "不需要回看任何历史 token.",
    ]):
        b.append(txt(64, cy + 58 + i * 22, s, "l", "start"))

    b.append(txt(486, cy + 30, "■ 6 层 GQA 全注意力", "h", "start"))
    for i, s in enumerate([
        "层号 3 / 7 / 11 / 15 / 19 / 23.",
        "8 个 Q head, 2 个 KV head, head_dim 256.",
        "",
        "RoPE 只作用在前 64 维 (partial rotary),",
        "后 192 维原样不动.",
        "",
        "KV cache 随序列线性增长,",
        "decode 时要把整段历史全读一遍. ← 麻烦在这",
    ]):
        cls = "l"
        b.append(txt(486, cy + 58 + i * 22, s, cls, "start"))

    b.append(txt(W / 2, 452,
                 "两种层的缓存机制完全不同, 一个引擎里要同时管好三类缓存", "s"))
    svg("qwen35_layers.svg", "\n".join(b), W, H)


# --------------------------------------------------------------------------
# 图 2: CTA 荒
# --------------------------------------------------------------------------
def fig_cta() -> None:
    W, H = 900, 606
    b = [txt(W / 2, 38, "108 个 SM, 只有 2 个 CTA 在干活", "t")]

    cols, rows, s, g = 12, 9, 22, 4
    grid_w = cols * (s + g) - g

    def draw_grid(ox, oy, busy):
        out = ['<g filter="url(#wob2)">']
        for i in range(108):
            r, c = divmod(i, cols)
            out.append(box(ox + c * (s + g), oy + r * (s + g), s, s,
                           BUSY if i < busy else IDLE, rx=4))
        out.append("</g>")
        return "\n".join(out)

    lx, rx = 60, 500
    gy = 90
    b.append(txt(lx + grid_w / 2, 78, "grid = (num_kv_heads,) = 2", "h"))
    b.append(draw_grid(lx, gy, 2))
    b.append(txt(rx + grid_w / 2, 78, "split-K: grid = (2, 128) = 256", "h"))
    b.append(draw_grid(rx, gy, 108))

    ey = gy + rows * (s + g) + 26
    b.append(txt(lx + grid_w / 2, ey, "106 个 SM 在看戏", "l"))
    b.append(txt(lx + grid_w / 2, ey + 20, "60 GB/s = 峰值的 4%", "s"))
    b.append(txt(rx + grid_w / 2, ey, "全都排上了队", "l"))
    b.append(txt(rx + grid_w / 2, ey + 20, "881 GB/s = 峰值的 57%", "s"))

    # 底部数据表
    ty = ey + 52
    b.append(f'<g filter="url(#wob)">{box(60, ty, 780, 132, "#ffffff")}</g>')
    heads = ["seq_len", "不分 split", "split-K", "加速"]
    xs = [150, 350, 550, 730]
    for x, hd in zip(xs, heads):
        b.append(txt(x, ty + 26, hd, "h"))
    # 注意不能写成一条纯水平的直线再套 filter: filter 区域默认按 objectBoundingBox
    # 算, 水平线的 bbox 高度为 0, 区域归零就整条线都不渲染了. 这里画成微微起伏的
    # 曲线, 既绕开这个坑, 看着也更像手画的.
    y = ty + 38
    b.append(f'<path class="ln" d="M 80 {y} Q 265 {y - 3} 450 {y} T 820 {y - 1}"/>')
    for i, row in enumerate([("1024", "38.7us", "14.4us", "2.7x"),
                             ("4096", "140.4us", "15.8us", "8.9x"),
                             ("8000", "265.9us", "18.6us", "14.3x")]):
        for x, cell in zip(xs, row):
            b.append(txt(x, ty + 66 + i * 24, cell, "l"))

    b.append(txt(W / 2, ty + 162,
                 "不分 split 时序列越长越慢, split-K 之后基本是平的", "s"))
    svg("cta_starvation.svg", "\n".join(b), W, H)


# --------------------------------------------------------------------------
# 图 3: CUDA Graph 把标量烧进 launch 配置
# --------------------------------------------------------------------------
def fig_graph_scalar() -> None:
    W, H = 900, 430
    b = [txt(W / 2, 38, "为什么 seq_len 必须搬到显存里去", "t")]

    for ox, title, tint in ((50, "python int 直接传", "#fbe9e6"),
                            (480, "显存里的 pos tensor", "#e9f4ec")):
        b.append(f'<g filter="url(#wob)">{box(ox, 62, 370, 300, tint)}</g>')
        b.append(txt(ox + 185, 90, title, "h"))

    # 左: 烧死
    lx = 50
    b.append(f'<g filter="url(#wob2)">{box(lx + 30, 112, 310, 44, "#ffffff")}</g>')
    b.append(txt(lx + 185, 140, 'kernel[grid](..., seq_len=20)', "l"))
    b.append(f'<path class="ln" d="M {lx + 185} 160 L {lx + 185} 186" marker-end="url(#arw)"/>')
    b.append(f'<g filter="url(#wob2)">{box(lx + 60, 190, 250, 40, "#ffffff")}</g>')
    b.append(txt(lx + 185, 216, "capture 时烧进 launch 配置", "l"))
    b.append(f'<path class="ln" d="M {lx + 185} 234 L {lx + 185} 260" marker-end="url(#arw)"/>')
    b.append(f'<g filter="url(#wob2)">{box(lx + 40, 264, 290, 76, "#f8d9d4")}</g>')
    b.append(txt(lx + 185, 290, "replay 第 1 次: seq_len = 20  ✓", "l"))
    b.append(txt(lx + 185, 312, "replay 第 2 次: seq_len = 20  ✗", "l"))
    b.append(txt(lx + 185, 332, "第 100 次也还是 20", "s"))

    # 右: 活的
    rx = 480
    b.append(f'<g filter="url(#wob2)">{box(rx + 30, 112, 310, 44, "#ffffff")}</g>')
    b.append(txt(rx + 185, 140, 'kernel[grid](..., pos_ptr=pos)', "l"))
    b.append(f'<path class="ln" d="M {rx + 185} 160 L {rx + 185} 186" marker-end="url(#arw)"/>')
    b.append(f'<g filter="url(#wob2)">{box(rx + 60, 190, 250, 40, "#ffffff")}</g>')
    b.append(txt(rx + 185, 216, "烧进去的只是一个地址", "l"))
    b.append(f'<path class="ln" d="M {rx + 185} 234 L {rx + 185} 260" marker-end="url(#arw)"/>')
    b.append(f'<g filter="url(#wob2)">{box(rx + 40, 264, 290, 76, "#d6ecdd")}</g>')
    b.append(txt(rx + 185, 290, "kernel 里 tl.load(pos_ptr)", "l"))
    b.append(txt(rx + 185, 312, "执行那一刻才去读", "l"))
    b.append(txt(rx + 185, 332, "图内 pos.add_(1) 自增, 每次都对", "s"))

    b.append(txt(W / 2, 396,
                 "同理: 切片下标也不能用 python int, k_cache[:, t, :] = k 要换成 index_copy_", "s"))
    svg("graph_scalar.svg", "\n".join(b), W, H)


if __name__ == "__main__":
    print("生成配图:")
    fig_structure()
    fig_cta()
    fig_graph_scalar()
