import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl
import math


'''
GQA Attention实现
head维度上分为按照Q和KV的num_head切分两种方式 
前者并行度大 需要L2 cache降低对global memory的读取压力 
后者对于kv的读取量减少 同时启动的block数量也会降低
'''

# 测试场景模拟decode 
# 即Q[batch, q_num_head, 1, d_model]
# KV [batch, kv_num_head, seq_len, d_model]
# seq_len取1024 再长需要使用split-k之类的并行策略
# q head数量64, kv head数量4

autotune_configs = [
    triton.Config({"TILE_KV_S": 128}, num_warps=2, num_stages=2),
    triton.Config({"TILE_KV_S": 64}, num_warps=2, num_stages=2),
    triton.Config({"TILE_KV_S": 32}, num_warps=2, num_stages=2),

    triton.Config({"TILE_KV_S": 128}, num_warps=4, num_stages=2),
    triton.Config({"TILE_KV_S": 64}, num_warps=4, num_stages=2),
    triton.Config({"TILE_KV_S": 32}, num_warps=4, num_stages=2),
]
@triton.autotune(
    configs=autotune_configs,
    key=["d_model"],
)
@triton.jit
def _gqa_attention_split_kv_head_triton(
    q_ptr, k_ptr, v_ptr, o_ptr, 
    stride_q_b, stride_q_h, stride_q_s, stride_q_d, 
    stride_k_b, stride_k_h, stride_k_s, stride_k_d, 
    stride_v_b, stride_v_h, stride_v_s, stride_v_d, 
    stride_o_b, stride_o_h, stride_o_s, stride_o_d, 
    kv_seq_len, 
    sm_scale: tl.constexpr, 
    d_model: tl.constexpr, 
    group_size: tl.constexpr,
    TILE_KV_S: tl.constexpr
):
    # 按照kv的num head起block
    batch_id = tl.program_id(1)
    head_id = tl.program_id(0)

    q_base_ptr = q_ptr + batch_id * stride_q_b + head_id * group_size * stride_q_h
    o_base_ptr = o_ptr + batch_id * stride_o_b + head_id * group_size * stride_o_h
    # 对kv直接执行batch head维度消除
    k_base_ptr = k_ptr + batch_id * stride_k_b + head_id * stride_k_h 
    v_base_ptr = v_ptr + batch_id * stride_v_b + head_id * stride_v_h

    
    # seq_q=1, 直接squeeze掉seq维度, 把group_size当作pseudo seq_len
    # Q: [group_size, d_model], K: [d_model, TILE_KV_S], V: [TILE_KV_S, d_model]
    # 所有操作都是2D, tl.dot可以正常工作
    offset_d = tl.arange(0, d_model)
    offset_q_h = tl.arange(0, group_size)
    q = tl.load(
        q_base_ptr + offset_q_h[:, None] * stride_q_h + offset_d[None, :] * stride_q_d,
    ) # [group_size, d_model]

    k_block_ptr = tl.make_block_ptr(
        base = k_base_ptr,
        shape = (d_model, kv_seq_len),
        strides = (stride_k_d, stride_k_s),
        offsets = (0, 0),
        block_shape = (d_model, TILE_KV_S),
        order = (0, 1)
    )

    v_block_ptr = tl.make_block_ptr(
        base = v_base_ptr,
        shape = (kv_seq_len, d_model),
        strides = (stride_v_s, stride_v_d),
        offsets = (0, 0),
        block_shape = (TILE_KV_S, d_model),
        order = (1, 0)
    )

    m_i = tl.zeros([group_size], dtype = tl.float32) - float('inf')
    l_i = tl.zeros([group_size], dtype = tl.float32)
    acc = tl.zeros([group_size, d_model], dtype = tl.float32)

    qk_scale = sm_scale * math.log2(math.e)

    q = (q * qk_scale).to(tl.float16)

    for start_kv_s in tl.range(0, kv_seq_len, TILE_KV_S):
        k = tl.load(k_block_ptr, boundary_check = (1,)).to(tl.float16)
        v = tl.load(v_block_ptr, boundary_check = (0, )).to(tl.float16)
        # [group_size, d_model] @ [d_model, TILE_KV_S] -> [group_size, TILE_KV_S]
        qk = tl.dot(q, k)

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_i_new)

        p = tl.math.exp2(qk - m_i_new[:, None])

        acc *= alpha[:, None]
        # [group_size, TILE_KV_S] @ [TILE_KV_S, d_model] -> [group_size, d_model]
        acc += tl.dot(p.to(tl.float16), v)

        l_i = l_i * alpha + tl.sum(p, axis=1)

        m_i = m_i_new

        k_block_ptr = tl.advance(k_block_ptr, (0, TILE_KV_S))
        v_block_ptr = tl.advance(v_block_ptr, (TILE_KV_S, 0))

    acc = acc / l_i[:, None]

    tl.store(
        o_base_ptr + \
            tl.arange(0, group_size)[:, None] * stride_o_h + \
            offset_d[None, :] * stride_o_d,
        acc
    )

def call_gqa_attention_split_kv_head_triton(
    q, k, v     
):
    batch, q_num_head, seq_len_q, d_model = q.shape
    kv_seq_len = k.shape[2]
    kv_num_head = k.shape[1]

    def grid(META):
        return (kv_num_head, batch)

    o = torch.empty_like(q)

    torch.library.wrap_triton(_gqa_attention_split_kv_head_triton)[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        kv_seq_len, 1.0 / math.sqrt(d_model),
        d_model, q_num_head // kv_num_head
    )
    return o 




def extract_kernel_time_us(prof):
    total = 0.0
    for evt in prof.key_averages():
        if evt.device_time_total > 0:
            total += evt.device_time_total
    return total


def benchmark():
    q_num_head = 64
    kv_num_head = 4
    kv_seq_len = 1024
    d_model = 32
    warmup = 5
    repeat = 20

    print(f"{'batch':>6} | {'Triton(us)':>12} | {'SDPA(us)':>12} | {'Speedup':>8}")
    print("-" * 50)

    batch_size = 1
    while batch_size <= 2048:
        q = torch.randn(batch_size, q_num_head, 1, d_model, device="cuda", dtype=torch.float16)
        k = torch.randn(batch_size, kv_num_head, kv_seq_len, d_model, device="cuda", dtype=torch.float16)
        v = torch.randn(batch_size, kv_num_head, kv_seq_len, d_model, device="cuda", dtype=torch.float16)

        # warmup
        for _ in range(warmup):
            call_gqa_attention_split_kv_head_triton(q, k, v)
            F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
        ) as prof_triton:
            for _ in range(repeat):
                call_gqa_attention_split_kv_head_triton(q, k, v)
            torch.cuda.synchronize()
        triton_us = extract_kernel_time_us(prof_triton) / repeat

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
        ) as prof_sdpa:
            for _ in range(repeat):
                F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            torch.cuda.synchronize()
        sdpa_us = extract_kernel_time_us(prof_sdpa) / repeat

        speedup = sdpa_us / triton_us if triton_us > 0 else float('inf')
        print(f"{batch_size:>6} | {triton_us:>12.1f} | {sdpa_us:>12.1f} | {speedup:>7.2f}x")

        # 导出timeline供详细分析
        prof_triton.export_chrome_trace(f"trace_triton_bs{batch_size}.json")
        prof_sdpa.export_chrome_trace(f"trace_sdpa_bs{batch_size}.json")

        batch_size *= 2


if __name__ == "__main__":
    # 正确性验证
    for desc, gen_fn in [
        ("randn", lambda shape: torch.randn(shape, device="cuda", dtype=torch.float16)),
        ("randn % 3", lambda shape: torch.randn(shape, device="cuda", dtype=torch.float16) % 3),
    ]:
        q = gen_fn((4, 64, 1, 32))
        k = gen_fn((4, 4, 1024, 32))
        v = gen_fn((4, 4, 1024, 32))

        output = call_gqa_attention_split_kv_head_triton(q, k, v)
        output_sdpa = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        diff = torch.abs(output - output_sdpa)
        print(f"[{desc}] Max diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}")

    # 性能对比
    benchmark()


# ==================== Benchmark Results ====================
#
# 测试配置: q_num_head=64, kv_num_head=4, kv_seq_len=1024, d_model=32
#           Q: [batch, 64, 1, 32], K/V: [batch, 4, 1024, 32]
#
# --- NVIDIA PRO5000 (SM120), CUDA 13.1, PyTorch 2.9.1 ---
#  batch |   Triton(us) |     SDPA(us) |  Speedup
# --------------------------------------------------
#      1 |          7.8 |          6.1 |    0.79x
#      2 |          7.8 |          6.2 |    0.79x
#      4 |          7.8 |          6.3 |    0.80x
#      8 |          7.9 |          7.7 |    0.97x
#     16 |          8.1 |          9.8 |    1.21x
#     32 |         10.0 |         14.5 |    1.45x
#     64 |         11.3 |         28.3 |    2.50x
#    128 |         21.1 |         45.8 |    2.17x
#    256 |        111.3 |        118.6 |    1.07x
#    512 |        223.7 |        235.1 |    1.05x
#   1024 |        445.5 |        458.4 |    1.03x
#   2048 |        889.5 |        902.6 |    1.01x
#
# 结论: 小batch时SDPA的flash attention更快(kernel launch overhead占主导),
#       中等batch(16-128)时Triton有1.2-2.5x加速,
#       大batch时两者趋于一致(均为memory-bound, 带宽打满)
#
# --- AMD MI308X (ROCm), PyTorch 2.9.0 ---
#  batch |   Triton(us) |     SDPA(us) |  Speedup
# --------------------------------------------------
#      1 |         20.0 |         46.8 |    2.34x
#      2 |         20.0 |         69.8 |    3.49x
#      4 |         20.2 |        132.8 |    6.59x
#      8 |         20.2 |        226.6 |   11.20x
#     16 |         20.6 |        412.6 |   20.07x
#     32 |         29.3 |        813.5 |   27.75x
#     64 |         54.2 |       1986.7 |   36.63x
#    128 |         91.3 |       3190.4 |   34.95x
#    256 |        164.9 |       6341.7 |   38.46x
#    512 |        325.8 |      12660.8 |   38.86x
#   1024 |        651.8 |      25301.7 |   38.82x
#   2048 |       1288.2 |      50594.4 |   39.28x
#
# 结论: AMD上SDPA的CK flash attention在GQA场景下性能极差(~39x慢),
#       疑似与CK内部的seqlenq_ngroups_swapped优化有关:
#         q.reshape(b, kv_heads, groups, d).transpose(1, 2)
#       用户传入的Q本身是连续的, 但CK内部的transpose(1,2)将其变为非连续
#       (strides从连续的(2048,512,32,1)经transpose变为(2048,32,512,1)),
#       CK kernel直接用非连续strides访存而未做.contiguous()拷贝,
#       可能导致GPU coalescing变差, 但具体根因仍需进一步profiling验证