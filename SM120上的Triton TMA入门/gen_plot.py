"""
生成 TMA benchmark 热力图
固定 batch=32, heads=4, d_model=64
横轴: q_seq_len, 纵轴: kv_seq_len
色值: TMA耗时 / block_ptr耗时 (>1 表示 block_ptr 快, <1 表示 TMA 快)
另外一张图: 三方绝对耗时对比 (选几个典型 kv 长度)
"""
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

triton.set_allocator(lambda s, a, st: torch.empty((s,), device="cuda", dtype=torch.int8))

@triton.jit
def _attn_block_ptr(
    q_ptr, k_ptr, v_ptr, o_ptr, sm_scale: tl.constexpr,
    stride_q_b, stride_q_h: tl.constexpr, stride_q_s: tl.constexpr, stride_q_d: tl.constexpr,
    stride_k_b, stride_k_h: tl.constexpr, stride_k_s: tl.constexpr, stride_k_d: tl.constexpr,
    stride_v_b, stride_v_h: tl.constexpr, stride_v_s: tl.constexpr, stride_v_d: tl.constexpr,
    stride_o_b, stride_o_h: tl.constexpr, stride_o_s: tl.constexpr, stride_o_d: tl.constexpr,
    q_seq_len: tl.constexpr, kv_seq_len: tl.constexpr,
    BLOCK_Q_S: tl.constexpr, BLOCK_KV_S: tl.constexpr, d_model: tl.constexpr,
):
    gemm_dtype = tl.float16
    bid = tl.program_id(2); hid = tl.program_id(1); qid = tl.program_id(0)
    qb = q_ptr + bid * stride_q_b + hid * stride_q_h
    kb = k_ptr + bid * stride_k_b + hid * stride_k_h
    vb = v_ptr + bid * stride_v_b + hid * stride_v_h
    ob = o_ptr + bid * stride_o_b + hid * stride_o_h
    q_bp = tl.make_block_ptr(qb, (q_seq_len, d_model), (stride_q_s, stride_q_d), (qid*BLOCK_Q_S, 0), (BLOCK_Q_S, d_model), (1,0))
    k_bp = tl.make_block_ptr(kb, (d_model, kv_seq_len), (stride_k_d, stride_k_s), (0,0), (d_model, BLOCK_KV_S), (0,1))
    v_bp = tl.make_block_ptr(vb, (kv_seq_len, d_model), (stride_v_s, stride_v_d), (0,0), (BLOCK_KV_S, d_model), (1,0))
    qk_scale = sm_scale * 1.44269504
    m_i = tl.zeros([BLOCK_Q_S], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_Q_S], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q_S, d_model], dtype=tl.float32)
    q = (tl.load(q_bp, boundary_check=(0,)) * qk_scale).to(gemm_dtype)
    for _ in range(0, kv_seq_len, BLOCK_KV_S):
        k = tl.load(k_bp, boundary_check=(1,)).to(gemm_dtype)
        v = tl.load(v_bp, boundary_check=(0,)).to(gemm_dtype)
        qk = tl.dot(q, k); m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_new); p = tl.math.exp2(qk - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(gemm_dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1); m_i = m_new
        k_bp = tl.advance(k_bp, (0, BLOCK_KV_S)); v_bp = tl.advance(v_bp, (BLOCK_KV_S, 0))
    acc = acc / tl.where(l_i == 0, 1.0, l_i)[:, None]
    o_bp = tl.make_block_ptr(ob, (q_seq_len, d_model), (stride_o_s, stride_o_d), (qid*BLOCK_Q_S, 0), (BLOCK_Q_S, d_model), (1,0))
    tl.store(o_bp, acc.to(o_ptr.dtype.element_ty), boundary_check=(0,))

@triton.jit
def _attn_tma(
    q_ptr, k_ptr, v_ptr, o_ptr, sm_scale: tl.constexpr,
    stride_q_b, stride_q_h: tl.constexpr, stride_q_s: tl.constexpr, stride_q_d: tl.constexpr,
    stride_k_b, stride_k_h: tl.constexpr, stride_k_s: tl.constexpr, stride_k_d: tl.constexpr,
    stride_v_b, stride_v_h: tl.constexpr, stride_v_s: tl.constexpr, stride_v_d: tl.constexpr,
    stride_o_b, stride_o_h: tl.constexpr, stride_o_s: tl.constexpr, stride_o_d: tl.constexpr,
    q_seq_len: tl.constexpr, kv_seq_len: tl.constexpr,
    BLOCK_Q_S: tl.constexpr, BLOCK_KV_S: tl.constexpr, d_model: tl.constexpr,
):
    gemm_dtype = tl.float16
    bid = tl.program_id(2); hid = tl.program_id(1); qid = tl.program_id(0)
    qb = q_ptr + bid * stride_q_b + hid * stride_q_h
    kb = k_ptr + bid * stride_k_b + hid * stride_k_h
    vb = v_ptr + bid * stride_v_b + hid * stride_v_h
    ob = o_ptr + bid * stride_o_b + hid * stride_o_h
    q_desc = tl.make_tensor_descriptor(qb, [q_seq_len, d_model], [stride_q_s, stride_q_d], [BLOCK_Q_S, d_model])
    k_desc = tl.make_tensor_descriptor(kb, [kv_seq_len, d_model], [stride_k_s, stride_k_d], [BLOCK_KV_S, d_model])
    v_desc = tl.make_tensor_descriptor(vb, [kv_seq_len, d_model], [stride_v_s, stride_v_d], [BLOCK_KV_S, d_model])
    o_desc = tl.make_tensor_descriptor(ob, [q_seq_len, d_model], [stride_o_s, stride_o_d], [BLOCK_Q_S, d_model])
    qk_scale = sm_scale * 1.44269504
    m_i = tl.zeros([BLOCK_Q_S], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_Q_S], dtype=tl.float32)
    acc = tl.zeros([BLOCK_Q_S, d_model], dtype=tl.float32)
    q = (q_desc.load([qid * BLOCK_Q_S, 0]) * qk_scale).to(gemm_dtype)
    for start_kv in range(0, kv_seq_len, BLOCK_KV_S):
        k = k_desc.load([start_kv, 0]).to(gemm_dtype)
        v = v_desc.load([start_kv, 0]).to(gemm_dtype)
        qk = tl.dot(q, k.trans(1, 0)); m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_new); p = tl.math.exp2(qk - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(gemm_dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1); m_i = m_new
    acc = acc / tl.where(l_i == 0, 1.0, l_i)[:, None]
    o_desc.store([qid * BLOCK_Q_S, 0], acc.to(o_ptr.dtype.element_ty))


def bench(kernel_fn, B, H, SQ, SKV, D):
    q = torch.randn(B, H, SQ, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, SKV, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, SKV, D, device="cuda", dtype=torch.float16)
    o = torch.empty_like(q)
    sm = 1.0 / math.sqrt(D)
    BLOCK_Q_S = min(32, SQ) if SQ >= 16 else 16
    BLOCK_KV_S = 64
    grid = (triton.cdiv(SQ, BLOCK_Q_S), H, B)
    def fn():
        kernel_fn[grid](q, k, v, o, sm,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            SQ, SKV, BLOCK_Q_S=BLOCK_Q_S, BLOCK_KV_S=BLOCK_KV_S, d_model=D)
    fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench(fn, warmup=50, rep=100)


def bench_sdpa(B, H, SQ, SKV, D):
    q = torch.randn(B, H, SQ, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, SKV, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, SKV, D, device="cuda", dtype=torch.float16)
    return triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v), warmup=50, rep=100)


if __name__ == "__main__":
    B, H, D = 32, 4, 64
    q_seqs = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    kv_seqs = [64, 128, 256, 512, 1024, 2048, 4096]

    print(f"Benchmarking: B={B}, H={H}, D={D}")

    # 收集数据
    data_blk = np.zeros((len(kv_seqs), len(q_seqs)))
    data_tma = np.zeros((len(kv_seqs), len(q_seqs)))
    data_sdpa = np.zeros((len(kv_seqs), len(q_seqs)))

    for j, sq in enumerate(q_seqs):
        for i, skv in enumerate(kv_seqs):
            if sq > skv:
                data_blk[i, j] = np.nan
                data_tma[i, j] = np.nan
                data_sdpa[i, j] = np.nan
                continue
            print(f"  SQ={sq}, SKV={skv}...", end=" ")
            t_blk = bench(_attn_block_ptr, B, H, sq, skv, D)
            t_tma = bench(_attn_tma, B, H, sq, skv, D)
            t_sdpa = bench_sdpa(B, H, sq, skv, D)
            data_blk[i, j] = t_blk
            data_tma[i, j] = t_tma
            data_sdpa[i, j] = t_sdpa
            print(f"blk={t_blk:.4f} tma={t_tma:.4f} sdpa={t_sdpa:.4f}")

    # 图1: TMA / block_ptr 耗时比 热力图
    ratio = data_tma / data_blk
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(ratio, cmap='RdYlGn_r', vmin=0.7, vmax=1.3, aspect='auto')
    ax.set_xticks(range(len(q_seqs)))
    ax.set_xticklabels(q_seqs)
    ax.set_yticks(range(len(kv_seqs)))
    ax.set_yticklabels(kv_seqs)
    ax.set_xlabel('Q sequence length')
    ax.set_ylabel('KV sequence length')
    ax.set_title(f'TMA / block_ptr latency ratio (B={B}, H={H}, D={D}, fp16)\n'
                 f'Green = TMA faster, Red = block_ptr faster')
    for i in range(len(kv_seqs)):
        for j in range(len(q_seqs)):
            if not np.isnan(ratio[i, j]):
                ax.text(j, i, f'{ratio[i, j]:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax, label='TMA time / block_ptr time')
    plt.tight_layout()
    plt.savefig('/home/zy429782/debug_data/ZhangYu-s-Blog/SM120上的Triton TMA入门/tma_ratio_heatmap.png', dpi=150)
    print("Saved tma_ratio_heatmap.png")

    # 图2: 三方绝对耗时对比 (固定 kv=1024 和 kv=4096)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, kv_idx, kv_val in [(axes[0], 4, 1024), (axes[1], 6, 4096)]:
        valid_mask = ~np.isnan(data_blk[kv_idx])
        qs = [q_seqs[j] for j in range(len(q_seqs)) if valid_mask[j]]
        blk_vals = [data_blk[kv_idx, j] for j in range(len(q_seqs)) if valid_mask[j]]
        tma_vals = [data_tma[kv_idx, j] for j in range(len(q_seqs)) if valid_mask[j]]
        sdpa_vals = [data_sdpa[kv_idx, j] for j in range(len(q_seqs)) if valid_mask[j]]
        x = range(len(qs))
        ax.plot(x, blk_vals, 'o-', label='block_ptr (cp.async)', color='#2196F3')
        ax.plot(x, tma_vals, 's-', label='TMA (make_tensor_descriptor)', color='#FF5722')
        ax.plot(x, sdpa_vals, '^-', label='PyTorch SDPA (cuDNN)', color='#4CAF50')
        ax.set_xticks(x)
        ax.set_xticklabels(qs, rotation=45)
        ax.set_xlabel('Q sequence length')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'KV_seq = {kv_val}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    plt.suptitle(f'Attention Latency: B={B}, H={H}, D={D}, fp16 (no mask)', fontsize=13)
    plt.tight_layout()
    plt.savefig('/home/zy429782/debug_data/ZhangYu-s-Blog/SM120上的Triton TMA入门/latency_comparison.png', dpi=150)
    print("Saved latency_comparison.png")
