#!/usr/bin/env python3
"""为 blog 生成 Chrome trace, scp 回本地用 https://ui.perfetto.dev 打开.

    python dump_traces.py [输出目录]

刻意做了三件事让时间轴好看:
  1. 用 record_function 给区段起名, 否则 Perfetto 里是一堵 kernel 名的墙;
  2. 只跑 1~2 次迭代, 文件小、打开快、横轴不会被压扁;
  3. 每个文件只讲一件事, 不混在一起.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, "/wydata/src_home/zhangy2/qwen3.5-0.8b-inference-engine")

from engine.cache import allocate_caches
from engine.loader import load_text_weights
from engine.runner import GraphedDecoder, Qwen35Runner

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
OUT.mkdir(parents=True, exist_ok=True)
ACT = [ProfilerActivity.CPU, ProfilerActivity.CUDA]


def dump(name: str, fn, warmup: int = 3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=ACT) as p:
        fn()
        torch.cuda.synchronize()
    path = OUT / f"{name}.json"
    p.export_chrome_trace(str(path))
    print(f"  {name:<28} {path.stat().st_size / 1024:>7.0f} KB")


w = load_text_weights("/wydata/src_home/zhangy2/qwen3.5-0.8b-inference-engine/Qwen3.5-0.8B")
r = Qwen35Runner(w, compile=False)
meta = torch.load(
    "/wydata/src_home/zhangy2/qwen3.5-0.8b-inference-engine/oracle/meta.pt",
    weights_only=False,
)
base = meta["input_ids"].to(torch.int32).cuda()
mk = lambda n: base.repeat((n + base.numel() - 1) // base.numel())[:n].contiguous()

print("生成 trace:")

# ---- 1/2: decode 一步, eager vs CUDA Graph -------------------------------
# 看点: GPU 那条几乎一样长, 但 eager 的 CPU 行密密麻麻, graph 的是空的.
c = allocate_caches(w, 512, paged=True)
c.reset(); r.prefill(mk(23), c)
slot = torch.zeros(1, dtype=torch.int32, device="cuda")


def eager_step():
    with record_function("decode step (eager)"):
        r.decode_step(slot, c)
        c.pos.add_(1)


dump("decode_eager", eager_step)

dec = GraphedDecoder(r, c)
dec.capture()
c.reset(); r.prefill(mk(23), c)


def graph_step():
    with record_function("decode step (CUDA Graph)"):
        dec.graph.replay()


dump("decode_cudagraph", graph_step)
del c, dec
torch.cuda.empty_cache()

# ---- 3/4: prefill B=4, 逐条 vs 打包 --------------------------------------
# 看点: 逐条是四段几乎一样的块首尾相接, 打包只有一段.
B, T = 4, 128
prompts = [mk(T) for _ in range(B)]
c = allocate_caches(w, T + 64, paged=True, batch=B)


def one_by_one():
    with record_function(f"prefill x{B} (逐条)"):
        c.reset()
        for b in range(B):
            with record_function(f"  seq {b}"):
                r.prefill(prompts[b], c, seq=b)


def packed():
    with record_function(f"prefill x{B} (打包)"):
        c.reset()
        r.prefill_packed(prompts, c)


dump("prefill_one_by_one", one_by_one)
dump("prefill_packed", packed)
del c
torch.cuda.empty_cache()

# ---- 5: batch decode, B=1 vs B=32 ---------------------------------------
# 看点: kernel 数量完全一样, 每个变宽了一点, 但吞吐涨了 7.5 倍.
for B in (1, 32):
    c = allocate_caches(w, 512, paged=True, batch=B)
    c.reset()
    for b in range(B):
        r.prefill(mk(23), c, seq=b)
    dec = GraphedDecoder(r, c)
    dec.capture()
    c.reset()
    for b in range(B):
        r.prefill(mk(23), c, seq=b)

    def step(dec=dec, B=B):
        with record_function(f"decode step (B={B})"):
            dec.graph.replay()

    dump(f"decode_batch{B}", step)
    del c, dec
    torch.cuda.empty_cache()

print(f"\n全部写到 {OUT}")
print("本地打开:  scp 回去后拖进 https://ui.perfetto.dev  (纯客户端, 文件不上传)")
