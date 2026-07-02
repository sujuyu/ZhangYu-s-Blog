#!/usr/bin/env python3
"""
复现:用 hipStreamCreate/Destroy 自管 stream + carry-over tensor 时,reserved 显存随轮次单调上涨。

五组对照:
  A. own-no-destroy        : 一次建 N 条,永不 destroy(对应 adapter-owned 方案)
  B. own-recreate          : 每轮 destroy 旧 + 建新 N 条 - HIP 会复用 handle,allocator 视角不变
  C. own-recreate-distinct : 每轮先建新再 destroy 旧 - handle 不会复用,模拟真实 snapshot 切换
                             中新旧 snapshot 短暂共存时 HIP 给新 stream 分新 handle 的场景
  D. pool-cached           : 一次从 PyTorch pool 拿 N 条缓存复用(对应 hybrid 方案)
  E. pool-fresh            : 每轮重新 getStreamFromPool 拿 N 条(round-robin 走过越来越多 stream)

每轮内 forward 产生的 carry 按 stream 分桶持有,确保每条旧 stream 的 segment 都钉着活 block。
这样旧 stream "死掉"后,carry 钉住 segment,新 stream 上的 alloc 无法复用旧段的 free hole,
reserved 单调累积 —— 这正是生产中 emptyCache 收不回的根因。
"""
import argparse
import ctypes
import gc
import json

import torch


HIP_LIB = "/opt/rocm/lib/libamdhip64.so"
_libhip = ctypes.CDLL(HIP_LIB)
_libhip.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_libhip.hipStreamDestroy.argtypes = [ctypes.c_void_p]
_libhip.hipDeviceSynchronize.argtypes = []


def hip_stream_create() -> int:
    s = ctypes.c_void_p()
    rc = _libhip.hipStreamCreate(ctypes.byref(s))
    assert rc == 0, f"hipStreamCreate failed rc={rc}"
    return s.value


def hip_stream_destroy(stream_ptr: int) -> None:
    rc = _libhip.hipStreamDestroy(ctypes.c_void_p(stream_ptr))
    assert rc == 0, f"hipStreamDestroy failed rc={rc}"


def hip_device_sync() -> None:
    _libhip.hipDeviceSynchronize()


def sample(tag: str, round_idx: int) -> dict:
    s = torch.cuda.memory_stats()
    return {
        "tag": tag,
        "round": round_idx,
        "allocated_MB": s["allocated_bytes.all.current"] / 1024 / 1024,
        "reserved_MB": s["reserved_bytes.all.current"] / 1024 / 1024,
        "inactive_split_MB": s["inactive_split_bytes.all.current"] / 1024 / 1024,
        "num_alloc_retries": s["num_alloc_retries"],
        "num_ooms": s["num_ooms"],
    }


def forward_once(stream, weight: torch.Tensor, batch: int, dim: int,
                 carry_elems: int) -> torch.Tensor:
    """模拟一次 forward,返回需要"常驻"的 tensor(carry-over,走 large pool)"""
    with torch.cuda.stream(stream):
        x = torch.randn(batch, dim, device="cuda", dtype=torch.float32)
        y = x @ weight                                          # 大中间 buffer
        z = torch.nn.functional.gelu(y)
        # carry-over:模拟 Biz/Model 内部常驻的中等 buffer
        # (如 fused workspace、prefill cache、KV cache 片段)
        # 大小必须 > caching allocator small/large 阈值 (1MB),才走 large pool 跟 transient 同段
        carry = torch.randn(carry_elems, device="cuda", dtype=torch.float32)
        torch.cuda.current_stream().synchronize()
        del x, y, z
        return carry


def run_round(streams: list, weight: torch.Tensor, batch: int, dim: int,
              steps_per_round: int, carry_pool: list, carry_keep: int,
              carry_elems: int) -> None:
    """一轮 forward + FIFO carry-over,保持总数恒定(alloc 稳定),只看 reserved 涨幅"""
    n = len(streams)
    for step in range(steps_per_round):
        c = forward_once(streams[step % n], weight, batch, dim, carry_elems)
        carry_pool.append(c)
        while len(carry_pool) > carry_keep:
            carry_pool.pop(0)


def build_streams_own(num: int):
    raw = [hip_stream_create() for _ in range(num)]
    wrapped = [torch.cuda.ExternalStream(p, device=0) for p in raw]
    return list(zip(raw, wrapped))


def destroy_streams_own(streams: list) -> None:
    hip_device_sync()
    for raw, _ in streams:
        hip_stream_destroy(raw)


def build_streams_pool(num: int):
    return [(None, torch.cuda.Stream(priority=0)) for _ in range(num)]


def run_experiment(mode: str, num_streams: int, num_rounds: int,
                   steps_per_round: int, batch: int, dim: int,
                   carry_keep: int, carry_elems: int) -> list:
    torch.cuda.empty_cache()
    gc.collect()
    weight = torch.randn(dim, dim, device="cuda", dtype=torch.float32)

    records = [sample(mode, 0)]
    carry_pool: list = []

    streams = None
    pool_streams_cached = None

    for r in range(1, num_rounds + 1):
        if mode == "own-no-destroy":
            if streams is None:
                streams = build_streams_own(num_streams)
        elif mode == "own-recreate":
            if streams is not None:
                wrappers = [w for _, w in streams]; del wrappers
                destroy_streams_own(streams)
            streams = build_streams_own(num_streams)
        elif mode == "own-recreate-distinct":
            # 先建新再 destroy 旧,HIP 必须给新 stream 分配新 handle
            new_streams = build_streams_own(num_streams)
            if streams is not None:
                wrappers = [w for _, w in streams]; del wrappers
                destroy_streams_own(streams)
            streams = new_streams
        elif mode == "pool-cached":
            if pool_streams_cached is None:
                pool_streams_cached = build_streams_pool(num_streams)
            streams = pool_streams_cached
        elif mode == "pool-fresh":
            streams = build_streams_pool(num_streams)
        else:
            raise ValueError(mode)

        wrapped_streams = [w for _, w in streams]
        run_round(wrapped_streams, weight, batch, dim, steps_per_round,
                  carry_pool, carry_keep, carry_elems)

        torch.cuda.empty_cache()
        records.append(sample(mode, r))

    # 收尾
    del carry_pool
    if mode in ("own-no-destroy", "own-recreate", "own-recreate-distinct") and streams is not None:
        wrappers = [w for _, w in streams]; del wrappers
        destroy_streams_own(streams)
    del weight
    torch.cuda.empty_cache()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-streams", type=int, default=4)
    ap.add_argument("--num-rounds", type=int, default=12)
    ap.add_argument("--steps-per-round", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=4096)
    ap.add_argument("--carry-keep", type=int, default=8,
                    help="跨轮持有的 carry-over tensor 数量上限")
    ap.add_argument("--carry-mb", type=float, default=8.0,
                    help="单个 carry-over tensor 大小 (MB),>1MB 走 large pool")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()
    carry_elems = int(args.carry_mb * 1024 * 1024 / 4)   # fp32

    assert torch.cuda.is_available()
    torch.cuda.init()
    torch.cuda.set_device(0)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"workload: batch={args.batch} dim={args.dim} "
          f"per-step transient ~ {args.batch * args.dim * 4 / 1024 / 1024:.1f}MB")
    print(f"streams/round={args.num_streams}, rounds={args.num_rounds}, "
          f"steps/round={args.steps_per_round}, "
          f"carry_keep={args.carry_keep}, carry={args.carry_mb:.1f}MB/each\n")

    all_records = {}
    for mode in ["own-no-destroy", "own-recreate", "own-recreate-distinct",
                 "pool-cached", "pool-fresh"]:
        print(f"=== {mode} ===")
        recs = run_experiment(
            mode=mode,
            num_streams=args.num_streams,
            num_rounds=args.num_rounds,
            steps_per_round=args.steps_per_round,
            batch=args.batch,
            dim=args.dim,
            carry_keep=args.carry_keep,
            carry_elems=carry_elems,
        )
        for r in recs:
            print(f"  round={r['round']:3d}  "
                  f"alloc={r['allocated_MB']:7.1f}MB  "
                  f"reserved={r['reserved_MB']:7.1f}MB  "
                  f"inact_split={r['inactive_split_MB']:6.1f}MB  "
                  f"retries={r['num_alloc_retries']}")
        all_records[mode] = recs
        print()

    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "records": all_records}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
