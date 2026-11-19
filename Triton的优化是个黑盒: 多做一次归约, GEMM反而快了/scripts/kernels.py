"""文中几个 GEMM kernel 的精简版, 只保留 fp16 路径和影响布局的部分, 用来离线编译对比 TTGIR / SASS.

x [B, K] FP32, BN 参数 mean / scale / beta [M, K] FP32, 权重 [M, K, O] fp16 (或 bf16 / FP32), 输出 [M, B, O] FP32.
grid = (输出列块, 行块, 分支); 每个 program 算一个分支的 BLOCK_B x BLOCK_OUT 输出块.
"""

import triton
import triton.language as tl


@triton.jit
def _normed(x_ptr, stride_x_b, mean_ptr, scale_ptr, beta_ptr, bn_offset, offset_b, offset_k, batch, k):
    x = tl.load(
        x_ptr + offset_b[:, None] * stride_x_b + offset_k[None, :],
        mask=(offset_b[:, None] < batch) & (offset_k[None, :] < k),
        other=0.0,
    )
    bn_mask = offset_k < k
    mean = tl.load(mean_ptr + bn_offset, mask=bn_mask, other=0.0)
    scale = tl.load(scale_ptr + bn_offset, mask=bn_mask, other=0.0)
    beta = tl.load(beta_ptr + bn_offset, mask=bn_mask, other=0.0)
    return (x - mean[None, :]) * scale[None, :] + beta[None, :]


@triton.jit
def _row_scale(peak):  # 行最大值缩到 (2^14, 2^15] 的 2 的幂, 2^-100 下限
    return tl.exp2(15.0 - tl.ceil(tl.log2(tl.maximum(peak, 7.888609052210118e-31))))


@triton.jit
def _gemm_loop(x_ptr, stride_x_b, mean_ptr, scale_ptr, beta_ptr, stride_bn_m, weight_ptr, stride_w_m, stride_w_k,
               pid_m, offset_b, offset_o, row_scale, batch, k, d_out,
               BLOCK_B: tl.constexpr, BLOCK_OUT: tl.constexpr, TILE_K: tl.constexpr, SCALED: tl.constexpr,
               PREC: tl.constexpr):
    acc = tl.zeros([BLOCK_B, BLOCK_OUT], dtype=tl.float32)
    for start_k in tl.range(0, k, TILE_K):
        offset_k = start_k + tl.arange(0, TILE_K)
        normed = _normed(x_ptr, stride_x_b, mean_ptr, scale_ptr, beta_ptr, pid_m * stride_bn_m + offset_k,
                         offset_b, offset_k, batch, k)
        weight = tl.load(weight_ptr + pid_m * stride_w_m + offset_k[:, None] * stride_w_k + offset_o[None, :],
                         mask=(offset_k[:, None] < k) & (offset_o[None, :] < d_out), other=0.0)
        if weight.dtype == tl.float32:
            acc += tl.dot(normed, weight, input_precision=PREC)
        elif SCALED:
            acc += tl.dot((normed * row_scale[:, None]).to(weight.dtype), weight)
        else:
            acc += tl.dot(normed.to(weight.dtype), weight)
    return acc


@triton.jit
def _epilogue(acc, bias_ptr, stride_bias_m, out_ptr, stride_o_m, stride_o_b, pid_m, offset_b, offset_o, batch, d_out):
    bias = tl.load(bias_ptr + pid_m * stride_bias_m + offset_o[None, :], mask=offset_o[None, :] < d_out, other=0.0)
    acc = tl.maximum(acc + bias, 0.0)
    tl.store(out_ptr + pid_m * stride_o_m + offset_b[:, None] * stride_o_b + offset_o[None, :], acc,
             mask=(offset_b[:, None] < batch) & (offset_o[None, :] < d_out))


@triton.jit
def gemm_kernel(x_ptr, stride_x_b, mean_ptr, scale_ptr, beta_ptr, stride_bn_m, weight_ptr, stride_w_m, stride_w_k,
                bias_ptr, stride_bias_m, rs_ptr, out_ptr, stride_o_m, stride_o_b, batch, k, d_out,
                BLOCK_B: tl.constexpr, BLOCK_OUT: tl.constexpr, TILE_K: tl.constexpr, MODE: tl.constexpr,
                PREC: tl.constexpr):
    """MODE 决定行系数从哪来:

    - "v1":     kernel 内先走一遍 K 求行峰值 (每个 K 块做一次行归约), 再走一遍做 GEMM
    - "v2":     从显存读现成的行系数 [M, B]
    - "v2p":    从显存读行峰值 [M, B], kernel 内算 exp2 / log2
    - "v3":     从显存读 TILE_K 道部分最大值 [M, B, TILE_K], kernel 内归约再算系数 (现行写法)
    - "plain":  不缩放 (bf16 权重直接转, FP32 权重走 tf32 dot)
    """
    pid_m, pid_b, pid_o = tl.program_id(2), tl.program_id(1), tl.program_id(0)
    offset_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offset_o = pid_o * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    if MODE == "v1":
        peak = tl.zeros([BLOCK_B], dtype=tl.float32)
        for start_k in tl.range(0, k, TILE_K):
            offset_k = start_k + tl.arange(0, TILE_K)
            normed = _normed(x_ptr, stride_x_b, mean_ptr, scale_ptr, beta_ptr, pid_m * stride_bn_m + offset_k,
                             offset_b, offset_k, batch, k)
            peak = tl.maximum(peak, tl.max(tl.abs(normed), axis=1))
        row_scale = _row_scale(peak)
    elif MODE == "v2":
        row_scale = tl.load(rs_ptr + pid_m * batch + offset_b, mask=offset_b < batch, other=1.0)
    elif MODE == "v2p":
        row_scale = _row_scale(tl.load(rs_ptr + pid_m * batch + offset_b, mask=offset_b < batch, other=0.0))
    elif MODE == "v3":
        lanes = tl.arange(0, TILE_K)
        peak = tl.load(rs_ptr + (pid_m * batch + offset_b[:, None]) * TILE_K + lanes[None, :],
                       mask=offset_b[:, None] < batch, other=0.0)
        row_scale = _row_scale(tl.max(peak, axis=1))
    else:
        row_scale = tl.full([BLOCK_B], 1.0, tl.float32)
    acc = _gemm_loop(x_ptr, stride_x_b, mean_ptr, scale_ptr, beta_ptr, stride_bn_m, weight_ptr, stride_w_m,
                     stride_w_k, pid_m, offset_b, offset_o, row_scale, batch, k, d_out,
                     BLOCK_B, BLOCK_OUT, TILE_K, MODE != "plain", PREC)
    if MODE != "plain":
        acc = acc / row_scale[:, None]
    _epilogue(acc, bias_ptr, stride_bias_m, out_ptr, stride_o_m, stride_o_b, pid_m, offset_b, offset_o, batch, d_out)
