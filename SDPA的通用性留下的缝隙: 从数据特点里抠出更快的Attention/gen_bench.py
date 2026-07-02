"""
可分离 mask Attention 优化的性能对比 (cudaEvent 计时)

生产形态: batch=1700, num_head=8, q_len=15, kv_len=1458, d_model=64, fp16
  q     : (B, 8, 15, 64)
  k / v : (1, 8, 1458, 64)  逻辑上跨 batch 共享(线上是 expand 出来的)
  mask  : gamma(1,8,15,1) * pad(B,1,1,1458)   —— 可分离(秩1)加性 mask

三个变体:
  V0 baseline : 物化 mask (B,8,15,1458) + 物化 expand K/V (B,8,1458,64), 走 F.sdpa
  V1 自定义kernel(未折叠): 不物化 mask(kernel 内算 gamma*pad), 但仍物化 expand K/V
  V2 自定义kernel(折叠)  : 不物化 mask, 也不物化 expand(K/V 传 (1,...) stride=0)

每个变体分三段计时(cudaEvent, 两次 synchronize 取间隔):
  mask-prep     : 物化 (B,8,15,1458) mask 的耗时(只有 V0 有)
  kv-expand-prep: 物化 (B,8,1458,64) K/V 的耗时(V0/V1 有, V2 没有)
  attention     : sdpa / 自定义 kernel 本身
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

torch.manual_seed(0)
DEV = "cuda"
B, H, Q, KV, D = 1700, 8, 15, 1458, 64

# ----------------------------------------------------------------------------
# 可分离 mask 的 flash-attention kernel (与线上自定义算子同构, 这里内联便于复现)
# ----------------------------------------------------------------------------
_cfgs = [
    triton.Config({"BLOCK_Q_S": 16, "BLOCK_KV_S": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_Q_S": 32, "BLOCK_KV_S": 32}, num_warps=2, num_stages=2),
]


@triton.autotune(configs=_cfgs, key=["q_seq_len", "kv_seq_len", "d_model"])
@triton.jit
def _sep_mask_attn(
    q_ptr, k_ptr, v_ptr, gamma_ptr, pad_ptr, o_ptr,
    sm_scale: tl.constexpr,
    sqb, sqh: tl.constexpr, sqs: tl.constexpr, sqd: tl.constexpr,
    skb, skh: tl.constexpr, sks: tl.constexpr, skd: tl.constexpr,
    svb, svh: tl.constexpr, svs: tl.constexpr, svd: tl.constexpr,
    sgb, sgh: tl.constexpr, sgs: tl.constexpr,   # gamma: 只用 batch/head/q
    spb, sps: tl.constexpr,                      # pad:   只用 batch/kv
    sob, soh: tl.constexpr, sos: tl.constexpr, sod: tl.constexpr,
    q_seq_len: tl.constexpr, kv_seq_len: tl.constexpr,
    BLOCK_Q_S: tl.constexpr, BLOCK_KV_S: tl.constexpr, d_model: tl.constexpr,
):
    gemm_dtype = tl.float16
    bid = tl.program_id(2)
    hid = tl.program_id(1)
    qid = tl.program_id(0)

    qb = q_ptr + bid * sqb + hid * sqh
    kb = k_ptr + bid * skb + hid * skh
    vb = v_ptr + bid * svb + hid * svh
    ob = o_ptr + bid * sob + hid * soh

    q_bp = tl.make_block_ptr(qb, (q_seq_len, d_model), (sqs, sqd), (qid * BLOCK_Q_S, 0), (BLOCK_Q_S, d_model), (1, 0))
    k_bp = tl.make_block_ptr(kb, (d_model, kv_seq_len), (skd, sks), (0, 0), (d_model, BLOCK_KV_S), (0, 1))
    v_bp = tl.make_block_ptr(vb, (kv_seq_len, d_model), (svs, svd), (0, 0), (BLOCK_KV_S, d_model), (1, 0))

    offs_q = qid * BLOCK_Q_S + tl.arange(0, BLOCK_Q_S)
    offs_kv = tl.arange(0, BLOCK_KV_S)
    q_valid = offs_q < q_seq_len

    # gamma 向量: (BLOCK_Q_S,), 只随 head/query 变
    gamma_vec = tl.load(gamma_ptr + bid * sgb + hid * sgh + offs_q * sgs, mask=q_valid, other=0.0).to(tl.float32)

    m_i = tl.zeros([BLOCK_Q_S], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_Q_S], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q_S, d_model], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_bp, boundary_check=(0,))
    q = (q * qk_scale).to(gemm_dtype)

    for start_kv in range(0, kv_seq_len, BLOCK_KV_S):
        kv_valid = start_kv + offs_kv < kv_seq_len
        valid = q_valid[:, None] & kv_valid[None, :]
        k = tl.load(k_bp, boundary_check=(1,)).to(gemm_dtype)
        v = tl.load(v_bp, boundary_check=(0,)).to(gemm_dtype)

        # pad 向量: (BLOCK_KV_S,), 只随 batch/kv 变
        pad_vec = tl.load(pad_ptr + bid * spb + (start_kv + offs_kv) * sps, mask=kv_valid, other=0.0).to(tl.float32)

        qk = tl.dot(q, k)
        # 可分离 additive mask: bias[i,j] = gamma[i] * pad[j]
        bias = gamma_vec[:, None] * pad_vec[None, :]
        bias = tl.minimum(tl.maximum(bias, -1.0e4), 1.0e4)
        qk = qk + bias * 1.44269504
        qk = tl.where(valid, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - tl.where(m_new == -float("inf"), 0.0, m_new))
        p = tl.math.exp2(qk - m_new[:, None])
        p = tl.where(valid, p, 0.0)          # valid_mask 之外置零, 否则被 mask 的位置会漏进分母
        acc *= alpha[:, None]
        acc += tl.dot(p.to(gemm_dtype), v.to(gemm_dtype))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        k_bp = tl.advance(k_bp, (0, BLOCK_KV_S))
        v_bp = tl.advance(v_bp, (BLOCK_KV_S, 0))

    l_i = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i[:, None]
    o_bp = tl.make_block_ptr(ob, (q_seq_len, d_model), (sos, sod), (qid * BLOCK_Q_S, 0), (BLOCK_Q_S, d_model), (1, 0))
    tl.store(o_bp, acc.to(o_ptr.dtype.element_ty), boundary_check=(0,))


def sep_attn(q, k, v, gamma, pad):
    Bb, Hh, Qq, Dd = q.shape
    KVv = k.shape[2]
    o = torch.empty_like(q)

    def sz0(t):
        s = list(t.stride())
        for i in range(4):
            if t.size(i) == 1:
                s[i] = 0
        return s

    sk = sz0(k)
    sv = sz0(v)
    sg = sz0(gamma)
    sp = sz0(pad)

    def grid(meta):
        return (triton.cdiv(Qq, meta["BLOCK_Q_S"]), Hh, Bb)

    _sep_mask_attn[grid](
        q, k, v, gamma, pad, o, 1.0 / math.sqrt(Dd),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        sk[0], sk[1], sk[2], sk[3],
        sv[0], sv[1], sv[2], sv[3],
        sg[0], sg[1], sg[2],
        sp[0], sp[3],
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        Qq, KVv, d_model=Dd,
    )
    return o


# ----------------------------------------------------------------------------
# cudaEvent 计时: warmup 后, 两次 event + synchronize 取间隔
# ----------------------------------------------------------------------------
def timeit(fn, warmup=30, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/次


def main():
    dt = torch.float16
    q = torch.randn(B, H, Q, D, device=DEV, dtype=dt)
    k1 = torch.randn(1, H, KV, D, device=DEV, dtype=dt)      # 逻辑上跨 batch 共享
    v1 = torch.randn(1, H, KV, D, device=DEV, dtype=dt)
    gamma = torch.randn(1, H, Q, 1, device=DEV, dtype=dt)
    pad = torch.zeros(B, 1, 1, KV, device=DEV, dtype=dt)
    pad[:, :, :, -40:] = -1e4                                # 尾部 padding 屏蔽

    # ---- 三段可复用的算子 ----
    def build_mask():                       # 物化 (B,H,Q,KV) mask
        return (gamma * pad).contiguous()

    def expand_kv():                        # 物化 (B,H,KV,D) K/V
        return k1.expand(B, -1, -1, -1).contiguous(), v1.expand(B, -1, -1, -1).contiguous()

    k_mat, v_mat = expand_kv()
    mask_mat = build_mask()

    def attn_sdpa():                        # baseline: F.sdpa(物化 K/V, 物化 mask)
        return F.scaled_dot_product_attention(q, k_mat, v_mat, mask_mat)

    def attn_ours_nofold():                 # 自定义 kernel, 读物化 K/V
        return sep_attn(q, k_mat, v_mat, gamma, pad)

    def attn_ours_fold():                   # 自定义 kernel, 读 (1,...) 共享 K/V
        return sep_attn(q, k1, v1, gamma, pad)

    # ---- 分段计时 ----
    t_mask = timeit(build_mask)
    t_expand = timeit(lambda: expand_kv())
    t_sdpa = timeit(attn_sdpa)
    t_nofold = timeit(attn_ours_nofold)
    t_fold = timeit(attn_ours_fold)

    # 每个变体的 (mask-prep, kv-expand-prep, attention)
    variants = [
        ("baseline (F.sdpa)",       t_mask, t_expand, t_sdpa),
        ("ours no-fold",            0.0,    t_expand, t_nofold),
        ("ours +fold",              0.0,    0.0,      t_fold),
    ]
    print(f"{'variant':22s} {'mask-prep':>10s} {'kv-expand':>10s} {'attn':>10s} {'total':>10s}")
    for name, a, b_, c in variants:
        n = name.replace(chr(10), " ")
        print(f"{n:22s} {a:10.3f} {b_:10.3f} {c:10.3f} {a+b_+c:10.3f}")

    # 数值 sanity: 折叠版 vs baseline
    ref = attn_sdpa().float()
    got = attn_ours_fold().float()
    print(f"max_diff(fold vs sdpa) = {(ref-got).abs().max().item():.5f}")

    _plot_timeline(variants)
    _plot_breakdown(variants)


# ----------------------------------------------------------------------------
# 图1: perfetto 风格 timeline (深色, 横向色块, 宽度∝耗时)
# ----------------------------------------------------------------------------
def _plot_timeline(variants):
    BG = "#2d2d2d"
    C = {"mask": "#e06c75", "expand": "#e5c07b", "attn": "#61afef"}
    LBL = {"mask": "materialize mask", "expand": "materialize K/V expand", "attn": "attention kernel"}
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    row_h = 0.62
    ys = [2, 1, 0]
    xmax = max(a + b + c for _, a, b, c in variants)
    for y, (name, a, b_, c) in zip(ys, variants):
        x = 0.0
        for key, w in [("mask", a), ("expand", b_), ("attn", c)]:
            if w <= 0:
                continue
            ax.add_patch(Rectangle((x, y), w, row_h, facecolor=C[key], edgecolor=BG, linewidth=1.5))
            if w > xmax * 0.045:
                ax.text(x + w / 2, y + row_h / 2, f"{w:.2f}", ha="center", va="center",
                        fontsize=8.5, color="#1a1a1a", fontweight="bold")
            x += w
        ax.text(-0.02 * xmax, y + row_h / 2, name, ha="right", va="center", color="#dddddd", fontsize=9)
        ax.text(x + 0.01 * xmax, y + row_h / 2, f"total {a+b_+c:.2f} ms", ha="left", va="center",
                color="#98c379", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.26 * xmax, xmax * 1.18)
    ax.set_ylim(-0.35, 2 + row_h + 0.35)
    ax.set_yticks([])
    ax.set_xlabel("per-forward latency (ms, cudaEvent)", color="#cccccc")
    ax.tick_params(colors="#cccccc")
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, facecolor=C[k]) for k in ["mask", "expand", "attn"]]
    ax.legend(handles, [LBL[k] for k in ["mask", "expand", "attn"]], loc="upper right",
              facecolor="#3a3a3a", edgecolor="none", labelcolor="#dddddd", fontsize=8.5, ncol=3)
    ax.set_title("separable-mask attention: latency breakdown (B=1700, H=8, Q=15, KV=1458, D=64, fp16)",
                 color="#eeeeee", fontsize=10.5, pad=10)
    fig.tight_layout()
    fig.savefig("timeline_compare.png", facecolor=BG, bbox_inches="tight")
    print("saved timeline_compare.png")


# ----------------------------------------------------------------------------
# 图2: 堆叠柱状图 breakdown
# ----------------------------------------------------------------------------
def _plot_breakdown(variants):
    names = [n.replace(chr(10), "\n") for n, _, _, _ in variants]
    mask = np.array([a for _, a, _, _ in variants])
    exp = np.array([b for _, _, b, _ in variants])
    att = np.array([c for _, _, _, c in variants])
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=140)
    x = np.arange(len(names))
    ax.bar(x, mask, 0.55, label="materialize mask", color="#e06c75")
    ax.bar(x, exp, 0.55, bottom=mask, label="materialize K/V expand", color="#e5c07b")
    ax.bar(x, att, 0.55, bottom=mask + exp, label="attention kernel", color="#61afef")
    tot = mask + exp + att
    for i, t in enumerate(tot):
        ax.text(i, t + tot.max() * 0.01, f"{t:.2f} ms", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("per-forward latency (ms, cudaEvent)")
    ax.set_title("latency breakdown: mask / K/V expand / attention")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("latency_breakdown.png", bbox_inches="tight")
    print("saved latency_breakdown.png")


if __name__ == "__main__":
    main()
