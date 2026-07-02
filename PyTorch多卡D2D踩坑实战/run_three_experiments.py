#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3 个实验复现：
1) target guard + .to()（同一 target stream）
2) src 多 stream + .to()（仍是同一 target stream）
3) 手动 hipMemcpyAsync + 两段 event 同步（并行）

运行环境：ROCm PyTorch（torch.cuda 接口）
"""

import ctypes
import json
import time
import statistics
from typing import Dict, List

import torch
from torch.profiler import profile, ProfilerActivity

TARGET = 0
SRCS = [1, 2, 3]
SIZE_MB = 384
N = (SIZE_MB * 1024 * 1024) // 4


def sync_all() -> None:
    for d in [TARGET] + SRCS:
        with torch.cuda.device(d):
            torch.cuda.synchronize()


def percentile(values: List[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    k = (len(values) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    if lo == hi:
        return float(values[lo])
    ratio = k - lo
    return float(values[lo] * (1 - ratio) + values[hi] * ratio)


def make_trace(path: str, fn) -> None:
    sync_all()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        out = fn()
    sync_all()
    del out
    prof.export_chrome_trace(path)


def bench(fn, warmup: int = 8, iters: int = 30) -> Dict[str, float]:
    for _ in range(warmup):
        out = fn()
        sync_all()
        del out

    lat = []
    for _ in range(iters):
        sync_all()
        t0 = time.perf_counter()
        out = fn()
        sync_all()
        t1 = time.perf_counter()
        del out
        lat.append((t1 - t0) * 1000.0)

    return {
        "mean_ms": statistics.mean(lat),
        "p50_ms": percentile(lat, 0.50),
        "p95_ms": percentile(lat, 0.95),
        "std_ms": statistics.pstdev(lat),
    }


def main() -> None:
    assert torch.cuda.is_available(), "CUDA/HIP 不可用"
    assert torch.cuda.device_count() >= 4, "需要至少 4 张 GPU"

    src = {d: torch.randn(N, device=f"cuda:{d}", dtype=torch.float32) for d in SRCS}

    # Exp1: target guard + .to()，同一 target stream
    target_stream1 = torch.cuda.Stream(device=TARGET)

    def exp1():
        outs = []
        with torch.cuda.device(TARGET), torch.cuda.stream(target_stream1):
            for d in SRCS:
                outs.append(src[d].to(device=f"cuda:{TARGET}", non_blocking=True))
        return outs

    # Exp2: src 多 stream + .to()，但 target 仍是同一 stream
    src_streams2 = {d: torch.cuda.Stream(device=d) for d in SRCS}
    target_stream2 = torch.cuda.Stream(device=TARGET)

    def exp2():
        outs = []
        with torch.cuda.device(TARGET):
            torch.cuda.set_stream(target_stream2)
        for d in SRCS:
            with torch.cuda.device(d), torch.cuda.stream(src_streams2[d]):
                outs.append(src[d].to(device=f"cuda:{TARGET}", non_blocking=True))
        return outs

    # Exp3: 手动 hipMemcpyAsync + 两段 event
    lib = ctypes.CDLL("libamdhip64.so.6")
    hipMemcpyAsync = lib.hipMemcpyAsync
    hipMemcpyAsync.restype = ctypes.c_int
    hipMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    hipGetErrorString = lib.hipGetErrorString
    hipGetErrorString.restype = ctypes.c_char_p
    hipGetErrorString.argtypes = [ctypes.c_int]
    HIP_MEMCPY_D2D = 3

    src_streams3 = {d: torch.cuda.Stream(device=d) for d in SRCS}
    dst_streams3 = {d: torch.cuda.Stream(device=TARGET) for d in SRCS}
    consume_stream3 = torch.cuda.Stream(device=TARGET)
    dst_buf = {d: torch.empty_like(src[d], device=f"cuda:{TARGET}") for d in SRCS}
    dst_ready = {d: torch.cuda.Event(enable_timing=False) for d in SRCS}
    copy_done = {d: torch.cuda.Event(enable_timing=False) for d in SRCS}

    def exp3():
        outs = []
        for d in SRCS:
            # phase-A: 在目标卡声明 dst 可写
            with torch.cuda.device(TARGET), torch.cuda.stream(dst_streams3[d]):
                dst_ready[d].record(dst_streams3[d])

            # phase-B: 源卡等待 dst_ready 后，发起 D2D
            with torch.cuda.device(d), torch.cuda.stream(src_streams3[d]):
                src_streams3[d].wait_event(dst_ready[d])
                err = hipMemcpyAsync(
                    ctypes.c_void_p(dst_buf[d].data_ptr()),
                    ctypes.c_void_p(src[d].data_ptr()),
                    src[d].numel() * 4,
                    HIP_MEMCPY_D2D,
                    ctypes.c_void_p(src_streams3[d].cuda_stream),
                )
                if err != 0:
                    raise RuntimeError(hipGetErrorString(err).decode())
                copy_done[d].record(src_streams3[d])

            outs.append(dst_buf[d])

        # phase-C: target consume stream 等待所有 copy_done
        with torch.cuda.device(TARGET), torch.cuda.stream(consume_stream3):
            for d in SRCS:
                consume_stream3.wait_event(copy_done[d])
            _ = outs[0].narrow(0, 0, 1024) + outs[1].narrow(0, 0, 1024) + outs[2].narrow(0, 0, 1024)

        return outs

    stats = {
        "tensor_size_per_copy_mb": SIZE_MB,
        "copy_paths": len(SRCS),
        "exp1_shared_target_to": bench(exp1),
        "exp2_src_stream_to_shared_target": bench(exp2),
        "exp3_manual_memcpy_events": bench(exp3),
    }

    make_trace("exp1_shared_target_to.json", exp1)
    make_trace("exp2_src_stream_to_shared_target.json", exp2)
    make_trace("exp3_manual_memcpy_events.json", exp3)

    with open("latency_stats.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
