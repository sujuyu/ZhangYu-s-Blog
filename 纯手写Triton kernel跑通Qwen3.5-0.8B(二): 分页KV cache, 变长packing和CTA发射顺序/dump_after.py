#!/usr/bin/env python3
"""只重跑 B=32 的 decode step, 用来和 BLOCK_B 改动之前的那张做对照.

配置必须和 dump_traces.py 第 5 段**逐字一致**(prompt 23 token, cache 512,
warmup 3, 一次 replay), 否则两张图的差异里会混进配置差异.
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

OUT = Path("traces"); OUT.mkdir(exist_ok=True)
ROOT = "/wydata/src_home/zhangy2/qwen3.5-0.8b-inference-engine"
w = load_text_weights(f"{ROOT}/Qwen3.5-0.8B")
r = Qwen35Runner(w, compile=False)
base = torch.load(f"{ROOT}/oracle/meta.pt", weights_only=False)["input_ids"].to(torch.int32).cuda()
mk = lambda n: base.repeat((n + base.numel() - 1) // base.numel())[:n].contiguous()

B = 32
c = allocate_caches(w, 512, paged=True, batch=B)
c.reset()
for b in range(B):
    r.prefill(mk(23), c, seq=b)
dec = GraphedDecoder(r, c); dec.capture()
c.reset()
for b in range(B):
    r.prefill(mk(23), c, seq=b)

def step():
    with record_function(f"decode step (B={B})"):
        dec.graph.replay()

for _ in range(3):
    step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    step(); torch.cuda.synchronize()
path = OUT / "decode_batch32_after.json"
p.export_chrome_trace(str(path))
print(f"  {path}  {path.stat().st_size/1024:.0f} KB")
