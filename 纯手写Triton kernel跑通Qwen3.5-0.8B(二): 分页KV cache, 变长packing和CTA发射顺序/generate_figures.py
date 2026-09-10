#!/usr/bin/env python3
"""生成本文用的手绘卡通风格 SVG 配图, 不依赖任何第三方库.

滤镜, 配色, 字体栈全部沿用第一篇的 generate_figures.py, 两篇看起来是一套的.

手绘感靠三样东西堆出来:
  1. feTurbulence + feDisplacementMap 把直线抖成手画的歪线;
  2. 圆头线帽 + 略粗的描边;
  3. 楷体/Comic Sans 一类的字体栈.
"""

from __future__ import annotations

import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

FONT = ('"Comic Sans MS", "Chalkboard SE", "Kaiti SC", "STKaiti", KaiTi, '
        '"Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", '
        '"Hiragino Sans GB", "Microsoft YaHei", sans-serif')
FONT = os.environ.get("FIG_FONT", FONT)

# 配色: 低饱和, 像马克笔
INK = "#3d3a36"
GDN = "#7fb3d5"      # 蓝
ATTN = "#f0a868"     # 橙
IDLE = "#e8e4de"     # 灰
BUSY = "#82c09a"     # 绿
WARN = "#e07a6b"     # 红
PURPLE = "#b39ddb"
PAPER = "#fdfcf8"

# 四条序列贯穿全文, 固定配色, 读者一眼能认出是同一批请求
SEQ = ["#7fb3d5", "#f0a868", "#82c09a", "#b39ddb"]


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
  <marker id="arws" viewBox="0 0 10 10" refX="8" refY="5"
          markerWidth="5" markerHeight="5" orient="auto-start-reverse">
    <path d="M 0 1 L 9 5 L 0 9 z" fill="#6b6660"/>
  </marker>
</defs>
<rect width="100%" height="100%" fill="{PAPER}"/>
<style>
  text {{ font-family: {FONT}; fill: {INK}; }}
  .t  {{ font-size: 21px; font-weight: 700; }}
  .h  {{ font-size: 15px; font-weight: 700; }}
  .l  {{ font-size: 13px; }}
  .s  {{ font-size: 11.5px; fill: #6b6660; }}
  .xs {{ font-size: 10px; fill: #6b6660; }}
  .box {{ stroke: {INK}; stroke-width: 2; stroke-linejoin: round; }}
  .thin {{ stroke: {INK}; stroke-width: 1.2; stroke-linejoin: round; }}
  .ln {{ stroke: {INK}; stroke-width: 2; fill: none;
         stroke-linecap: round; stroke-linejoin: round; }}
  .thinln {{ stroke: #6b6660; stroke-width: 1.4; fill: none;
             stroke-linecap: round; }}
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


def wavy(x1, y, x2, cls="ln", amp=3):
    """水平分隔线. 不能画成纯直线再套 filter -- filter 区域默认按 objectBoundingBox
    算, 水平线的 bbox 高度为 0, 区域归零就整条线都不渲染了. 画成微微起伏的曲线,
    既绕开这个坑, 看着也更像手画的. (第一篇里踩过, 这里直接封成函数. )
    """
    mid = (x1 + x2) / 2
    return (f'<path class="{cls}" d="M {x1} {y} Q {(x1 + mid) / 2} {y - amp} {mid} {y} '
            f'T {x2} {y - amp / 3}"/>')


# --------------------------------------------------------------------------
# 图 1: 整块预分配的浪费
# --------------------------------------------------------------------------
def fig_waste() -> None:
    W, H = 900, 452
    b = [txt(W / 2, 38, "为什么一块连续的 KV cache 撑不住多条请求", "t")]
    b.append(txt(W / 2, 62, "4 条请求, 长度 40 / 200 / 10 / 130, 引擎按 max_len = 256 开", "s"))

    lengths = [40, 200, 10, 130]
    MAXLEN, PAGE = 256, 16
    BAR = 280.0

    # ---- 左: 整块 ----
    ox = 46
    b.append(f'<g filter="url(#wob)">{box(ox, 84, 380, 268, "#fbe9e6")}</g>')
    b.append(txt(ox + 190, 110, "整块: 每条按 max_len 各开一份", "h"))
    for i, n in enumerate(lengths):
        y = 128 + i * 46
        b.append(txt(ox + 20, y + 20, f"#{i}", "s", "start"))
        b.append(f'<g filter="url(#wob2)">')
        b.append(box(ox + 46, y, BAR, 26, IDLE, rx=4))
        b.append(box(ox + 46, y, BAR * n / MAXLEN, 26, SEQ[i], rx=4))
        b.append("</g>")
        b.append(txt(ox + 46 + BAR + 8, y + 19, str(n), "xs", "start"))
    b.append(wavy(ox + 30, 320, ox + 350, "thinln"))
    b.append(txt(ox + 190, 340, "开了 1024 个槽, 只用了 380 个 -- 63% 是空的", "l"))

    # ---- 右: 分页 ----
    rx0 = 474
    b.append(f'<g filter="url(#wob)">{box(rx0, 84, 380, 268, "#e9f4ec")}</g>')
    b.append(txt(rx0 + 190, 110, f"分页: 按 {PAGE} 个 token 一页, 用多少要多少", "h"))
    pw, pg = 15, 2
    for i, n in enumerate(lengths):
        y = 128 + i * 46
        npage = (n + PAGE - 1) // PAGE
        b.append(txt(rx0 + 20, y + 20, f"#{i}", "s", "start"))
        b.append('<g filter="url(#wob2)">')
        for p in range(npage):
            # 最后一页通常没填满, 用浅一档表示内部碎片
            full = (p + 1) * PAGE <= n
            b.append(box(rx0 + 46 + p * (pw + pg), y, pw, 26,
                         SEQ[i] if full else IDLE, rx=3))
        b.append("</g>")
        b.append(txt(rx0 + 46 + npage * (pw + pg) + 6, y + 19,
                     f"{npage} 页", "xs", "start"))
    b.append(wavy(rx0 + 30, 320, rx0 + 350, "thinln"))
    b.append(txt(rx0 + 190, 340, "26 页 = 416 个槽, 浪费只剩每条尾页那几个", "l"))

    b.append(txt(W / 2, 386,
                 "浪费从 \"按最长的开\" 变成 \"每条尾页最多空 15 个\" -- 内部碎片, 有上界",
                 "l"))
    b.append(txt(W / 2, 412,
                 "而且序列还在一边 decode 一边长: 整块方案里紧挨着排就会撞上邻居, 得搬数据;",
                 "s"))
    b.append(txt(W / 2, 432,
                 "分页之后长满一页就再要一页, 落在哪都无所谓", "s"))
    svg("paged_waste.svg", "\n".join(b), W, H)


# --------------------------------------------------------------------------
# 图 2: 地址是怎么算出来的
# --------------------------------------------------------------------------
def fig_mapping() -> None:
    W, H = 900, 396
    b = [txt(W / 2, 38, "一个 token 的 K 到底存在哪", "t")]
    b.append(txt(W / 2, 62, "以第 0 条序列的第 37 个 token, KV head 1 为例 (页表就是下一张图那一行)", "s"))

    steps = [
        ("逻辑 token", "t = 37", "序列自己的第几个 token", GDN),
        ("逻辑页 / 槽位", "37 // 16 = 2\n37 % 16 = 5", "第 2 页, 页内第 5 格", ATTN),
        ("查页表", "block_table[0][2]\n= 3", "本条序列私有的一行", BUSY),
        ("物理行号", "(3*2 + 1)*16 + 5\n= 117", "页池里的绝对位置", PURPLE),
    ]
    bw, gap = 190, 42
    x0 = (W - (len(steps) * bw + (len(steps) - 1) * gap)) / 2
    for i, (title, expr, note, color) in enumerate(steps):
        x = x0 + i * (bw + gap)
        b.append(f'<g filter="url(#wob)">{box(x, 96, bw, 128, color + "44")}</g>')
        b.append(txt(x + bw / 2, 122, title, "h"))
        for j, line in enumerate(expr.split("\n")):
            b.append(txt(x + bw / 2, 152 + j * 22, line, "l"))
        b.append(txt(x + bw / 2, 208, note, "xs"))
        if i < len(steps) - 1:
            b.append(f'<path class="ln" d="M {x + bw + 6} 160 L {x + bw + gap - 8} 160" '
                     f'marker-end="url(#arw)"/>')

    b.append(f'<g filter="url(#wob)">{box(120, 254, 660, 60, "#ffffff")}</g>')
    b.append(txt(W / 2, 280, "row(t, h) = (block_table[b][t // PAGE] * H_kv + h) * PAGE + t % PAGE",
                 "h"))
    b.append(txt(W / 2, 302, "整条式子里没有一个 python int -- 全在显存上算, 所以能进 CUDA Graph",
                 "xs"))

    b.append(txt(W / 2, 348,
                 "查表这一下就是分页的全部代价. 换来的是逻辑上连续的序列", "l"))
    b.append(txt(W / 2, 372, "可以散落在物理上任意的页里", "l"))
    svg("paged_mapping.svg", "\n".join(b), W, H)


# --------------------------------------------------------------------------
# 图 3: 页池共享, 页表的行私有   <- 本文最核心的一张
# --------------------------------------------------------------------------
def fig_table() -> None:
    W, H = 900, 500
    b = [txt(W / 2, 38, "池子是共享的, 表的行是私有的", "t")]
    b.append(txt(W / 2, 62, "理解了这一句, 分页 KV cache 的其余部分都是推论", "s"))

    # ---- 上: 两条序列的页表. 左右并排, 不上下叠 --
    # 叠起来的话上面那行的连线要穿过下面那行的格子和标注, 字就被压住了.
    tables = [(110, "序列 0  (长 40, 3 页)", [5, 12, 3], SEQ[0]),
              (590, "序列 1  (长 30, 2 页)", [8, 1, None], SEQ[1])]
    cw, cell_y = 62, 126
    arrows = []
    for (gx, label, row, color) in tables:
        b.append(txt(gx, 106, label, "l", "start"))
        for c in range(3):
            x = gx + c * (cw + 8)
            v = row[c]
            b.append(f'<g filter="url(#wob2)">'
                     f'{box(x, cell_y, cw, 44, color + "55" if v is not None else IDLE, rx=5)}</g>')
            b.append(txt(x + cw / 2, cell_y + 28, str(v) if v is not None else "-", "l"))
            b.append(txt(x + cw / 2, cell_y + 62, f"逻辑页 {c}", "xs"))
            if v is not None:
                arrows.append((x + cw / 2, cell_y + 44, v, color))

    # 中间那条空走廊, 正好放页表本身的说明
    b.append(txt(W / 2, 122, "block_table", "h"))
    b.append(txt(W / 2, 144, "[max_batch, max_pages]", "xs"))
    b.append(txt(W / 2, 170, "一条序列一行, 互不相干", "xs"))

    # ---- 下: 共享的页池 ----
    pool_y = 300
    owner = {5: 0, 12: 0, 3: 0, 8: 1, 1: 1}
    ppw, ppg = 52, 8
    px0 = 60
    b.append('<g filter="url(#wob2)">')
    for p in range(14):
        x = px0 + p * (ppw + ppg)
        o = owner.get(p)
        b.append(box(x, pool_y, ppw, 46, SEQ[o] + "55" if o is not None else "#ffffff", rx=5))
    b.append("</g>")
    for p in range(14):
        x = px0 + p * (ppw + ppg)
        b.append(txt(x + ppw / 2, pool_y + 29, str(p), "l"))
    # 池子的标题放在下面: 放上面会被那一束连线穿过去
    b.append(txt(px0, pool_y + 72, "页池 k_cache / v_cache  [num_pages, H_kv, PAGE, D]  -- 只有一份",
                 "h", "start"))
    b.append(txt(px0, pool_y + 94, "空白的页在空闲链表里, reset 时全部归还", "s", "start"))

    # ---- 连线: 故意交叉 ----
    for (ax, ay, phys, color) in arrows:
        bx = px0 + phys * (ppw + ppg) + ppw / 2
        my = (ay + pool_y) / 2
        # stroke 得写进 style: 同名的表现属性打不过 .thinln 里的 stroke, 写成属性就全灰了
        b.append(f'<path class="thinln" d="M {ax} {ay + 26} C {ax} {my} {bx} {my} {bx} {pool_y - 4}" '
                 f'style="stroke:{color}" marker-end="url(#arws)"/>')

    b.append(txt(W / 2, 444,
                 "连线是乱的 -- 逻辑上连续的 token 落在物理上不相邻的页里, 这正是分页的意义",
                 "l"))
    b.append(txt(W / 2, 472,
                 "推论: kernel 里 k_cache 的基址完全不带 batch 偏移, 只有 block_table 按 pid_b 偏",
                 "s"))
    svg("paged_table.svg", "\n".join(b), W, H)


# --------------------------------------------------------------------------
# 图 4: 页内布局与 PAGE = 16 的取舍
# --------------------------------------------------------------------------
def fig_layout() -> None:
    W, H = 900, 430
    b = [txt(W / 2, 38, "一页里面长什么样, 以及为什么 PAGE = 16", "t")]

    # 左: 一页展开
    b.append(f'<g filter="url(#wob)">{box(46, 68, 400, 250, "#eaf3fa")}</g>')
    b.append(txt(246, 94, "一页 = [H_kv, PAGE, D] = [2, 16, 256]", "h"))
    for hh in range(2):
        y = 116 + hh * 92
        b.append(f'<g filter="url(#wob2)">{box(76, y, 340, 76, GDN + "55" if hh == 0 else ATTN + "55")}</g>')
        b.append(txt(246, y + 30, f"KV head {hh}", "l"))
        b.append(txt(246, y + 52, "16 x 256 x 2B = 8 KiB, 完全连续", "xs"))
    b.append(txt(246, 336, "固定 (page, head) 之后就是一整块连续内存,", "s"))
    b.append(txt(246, 354, "和整块方案里 k_cache[h, t0:t0+BLOCK_T, :] 的形态一样", "s"))

    # 右: BLOCK_T == PAGE
    b.append(f'<g filter="url(#wob)">{box(474, 68, 380, 250, "#e9f4ec")}</g>')
    b.append(txt(664, 94, "BLOCK_T 钉死等于 PAGE", "h"))
    code = [
        "for lp in range(page_start, page_end):",
        "    phys = tl.load(bt_ptr + lp)   # 标量!",
        "    k = tl.load(kc_ptr + phys * stride_p",
        "                + ... )",
    ]
    b.append(f'<g filter="url(#wob2)">{box(494, 112, 340, 96, "#ffffff")}</g>')
    for i, line in enumerate(code):
        b.append(txt(506, 132 + i * 20, line.replace("<", "&lt;"), "xs", "start"))
    for i, line in enumerate([
        "一个 tile 恰好覆盖一个页,",
        "页表只查一次, 查出来还是个标量.",
        "",
        "如果 BLOCK_T > PAGE, 一个 tile 跨多个页,",
        "就得 gather 一个页号向量再逐元素算地址 --",
        "复杂度和寄存器压力都上一个台阶.",
    ]):
        # 行距收到 16, 最后一行才不会压在面板下边框上
        b.append(txt(494, 226 + i * 16, line, "xs", "start"))

    b.append(txt(W / 2, 396,
                 "代价是 BLOCK_T 不再能被 autotune 选. 实测整块版本 seq=4095 时 16 vs 32 差距在 3% 以内, 认了",
                 "s"))
    svg("paged_layout.svg", "\n".join(b), W, H)


# --------------------------------------------------------------------------
# 图 5: CTA 发射顺序
# --------------------------------------------------------------------------
def fig_lpt() -> None:
    W, H = 900, 560
    b = [txt(W / 2, 38, "causal 让每个 CTA 的活不一样重", "t")]
    b.append(txt(W / 2, 62, "8 个 q 块, 4 个 SM. 块号越大要扫的 KV 越多, 工作量正比于 m+1", "s"))

    # 阶梯: 工作量示意
    sy = 84
    b.append(txt(60, sy + 16, "工作量", "s", "start"))
    for m in range(8):
        x = 132 + m * 44
        h = 8 + m * 7
        b.append(f'<g filter="url(#wob2)">{box(x, sy + 62 - h, 34, h, ATTN, rx=3)}</g>')
        b.append(txt(x + 17, sy + 78, f"q{m}", "xs"))
    b.append(txt(520, sy + 34, "q0 只要扫 1 个 KV 块, q7 要扫 8 个", "s", "start"))

    UNIT = 24.0
    tx0 = 132

    def gantt(oy, title, order, tint):
        out = [f'<g filter="url(#wob)">{box(46, oy - 26, 808, 152, tint)}</g>']
        out.append(txt(60, oy - 6, title, "h", "start"))
        free = [0.0] * 4
        placed = []
        for m in order:
            k = free.index(min(free))
            st = free[k]
            placed.append((k, st, m + 1, m))
            free[k] += m + 1
        for (k, st, dur, m) in placed:
            x = tx0 + st * UNIT
            y = oy + 14 + k * 26
            out.append(f'<g filter="url(#wob2)">{box(x, y, dur * UNIT - 3, 21, ATTN, rx=3)}</g>')
            out.append(txt(x + (dur * UNIT - 3) / 2, y + 15, f"q{m}", "xs"))
        for k in range(4):
            out.append(txt(tx0 - 12, oy + 29 + k * 26, f"SM{k}", "xs", "end"))
        span = max(free)
        # 完成线
        ex = tx0 + span * UNIT
        out.append(f'<path class="dash" d="M {ex} {oy + 8} L {ex} {oy + 120}"/>')
        out.append(txt(ex + 8, oy + 74, f"{int(span)}", "h", "start"))
        return "\n".join(out), span

    g1, s1 = gantt(212, "正序发射 q0 -> q7: 轻的先上, 最重的最后才开始",
                   list(range(8)), "#fbe9e6")
    b.append(g1)
    g2, s2 = gantt(384, "逆序发射 q7 -> q0: 重的先占住 SM, 轻的填空", list(range(7, -1, -1)), "#e9f4ec")
    b.append(g2)

    # 第二块面板下边到 510, 这句得再往下让一让, 否则压在甘特图上
    b.append(txt(W / 2, 540,
                 f"总工作量 36 不变, 4 个 SM 的理论下界就是 9 -- 逆序一行代码就够到了下界",
                 "l"))
    svg("lpt_dispatch.svg", "\n".join(b), W, H)


# --------------------------------------------------------------------------
# 图 6: packing 而不是 padding
# --------------------------------------------------------------------------
def fig_packing() -> None:
    W, H = 900, 480
    b = [txt(W / 2, 38, "变长 prefill: packing, 而不是 padding", "t")]
    b.append(txt(W / 2, 62, "4 条 prompt, 长度 3 / 6 / 1 / 4", "s"))

    lengths = [3, 6, 1, 4]
    CW = 26

    # 上: padding
    b.append(f'<g filter="url(#wob)">{box(46, 84, 380, 170, "#fbe9e6")}</g>')
    b.append(txt(236, 108, "padding 到 max = 6", "h"))
    for i, n in enumerate(lengths):
        y = 122 + i * 30
        for c in range(6):
            x = 96 + c * (CW + 3)
            b.append(f'<g filter="url(#wob2)">{box(x, y, CW, 22, SEQ[i] if c < n else IDLE, rx=3)}</g>')
    b.append(txt(236, 268, "24 个格子里 10 个是废的, 而且要造 mask 把它们挡掉", "s"))

    # 下: packing
    b.append(f'<g filter="url(#wob)">{box(474, 84, 380, 170, "#e9f4ec")}</g>')
    b.append(txt(664, 108, "拼成一条扁平序列", "h"))
    # packing 后一共 14 格, 用左边那个 CW=26 会顶出面板右边框, 这里收窄一档
    CW2, GAP2 = 22, 2
    x = 474 + (380 - (14 * (CW2 + GAP2) - GAP2)) / 2
    for i, n in enumerate(lengths):
        for c in range(n):
            b.append(f'<g filter="url(#wob2)">{box(x, 140, CW2, 22, SEQ[i], rx=3)}</g>')
            x += CW2 + GAP2
    b.append(txt(664, 186, "total_tokens = 14, 一个废格子都没有", "s"))
    b.append(txt(664, 214, "cu_seqlens = [0, 3, 9, 10, 14]", "l"))
    b.append(txt(664, 236, "边界就靠这一个数组说清楚", "xs"))

    # 底部: 块对角
    b.append(f'<g filter="url(#wob)">{box(46, 288, 808, 130, "#ffffff")}</g>')
    b.append(txt(70, 312, "为什么不需要造一个块对角的 mask 矩阵", "h", "start"))
    for i, line in enumerate([
        "因为压根不会算到. 每条序列的 attention 循环边界来自它自己的 cu_seqlens --",
        "q 在第 b 条里, 内层就只从 cu_seqlens[b] 扫到 cu_seqlens[b+1], 跨序列的位置",
        "从来没有进过循环. 块对角是这么 \"自动\" 出来的, 不是靠一个矩阵挡出来的.",
        "",
        "整个 24 层里只有三个算子需要知道边界: attention, conv4_prefill, GDN 递推.",
    ]):
        b.append(txt(70, 336 + i * 19, line, "l" if i < 4 else "s", "start"))

    b.append(txt(W / 2, 452,
                 "其余逐 token 的算子拿 [total_tokens, ...] 一行都不用改", "l"))
    svg("varlen_packing.svg", "\n".join(b), W, H)


# --------------------------------------------------------------------------
# 图 7: prefill 从 CPU-bound 变成 GPU-bound
# --------------------------------------------------------------------------
def fig_bound() -> None:
    W, H = 900, 442
    b = [txt(W / 2, 38, "prefill 的 bubble 是怎么一步步消掉的", "t")]
    b.append(txt(W / 2, 62, "B = 32, T = 128, packing 成一次前向. 深色是 GPU 真在算, 浅色是 GPU 等 CPU 发射", "s"))

    rows = [
        ("最初", 77.4, 131.9, 6937),
        ("slice_of 不再取标量", 70.3, 77.1, 5337),
        ("+ 变长 scatter kernel", 61.2, 22.0, 2271),
        ("+ 批量抽 conv state", 57.9, 4.3, 561),
    ]
    SCALE = 2.3
    x0 = 300
    for i, (label, gpu, bub, nk) in enumerate(rows):
        y = 100 + i * 62
        # 标签要停在条形图左边: 原来 anchor=end 停在 286, 而条子从 268 就开始画,
        # 后面几个字直接被压到条子底下去了
        b.append(txt(x0 - 14, y + 22, label, "l", "end"))
        b.append('<g filter="url(#wob2)">')
        b.append(box(x0, y, gpu * SCALE, 30, BUSY, rx=4))
        b.append(box(x0 + gpu * SCALE, y, bub * SCALE, 30, IDLE, rx=4))
        b.append("</g>")
        b.append(txt(x0 + (gpu + bub) * SCALE + 10, y + 21,
                     f"{gpu + bub:.1f}ms", "h", "start"))
        b.append(txt(x0 + 8, y + 46, f"GPU {gpu:.1f}", "xs", "start"))
        if bub * SCALE > 46:
            b.append(txt(x0 + gpu * SCALE + 8, y + 46, f"等 CPU {bub:.1f}", "xs", "start"))
        b.append(txt(x0 - 14, y + 40, f"{nk} 次发射", "xs", "end"))

    b.append(wavy(100, 358, 800, "thinln"))
    b.append(txt(W / 2, 386,
                 "三处改动全部在 CPU 侧, GPU 干的活基本没变 (77.4 -> 57.9 只是少了些冗余拷贝)",
                 "l"))
    b.append(txt(W / 2, 414,
                 "最后 4.3ms 的 bubble 说明 prefill 终于变成 GPU-bound -- 再压发射数已经没有意义了",
                 "l"))
    svg("prefill_bound.svg", "\n".join(b), W, H)


if __name__ == "__main__":
    print("生成配图:")
    fig_waste()
    fig_mapping()
    fig_table()
    fig_layout()
    fig_lpt()
    fig_packing()
    fig_bound()
