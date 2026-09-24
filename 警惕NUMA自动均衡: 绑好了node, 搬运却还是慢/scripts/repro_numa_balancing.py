"""NUMA 自动均衡 (kernel.numa_balancing=1) 拖慢绑核搬运的最小复现.

场景: 两个 NUMA node, 每个 node 各 2 张 GPU (负载完全对称). 每张卡一个 loader 线程,
CPU 亲和 + MPOL_BIND 都绑在自己卡所在的 node 上, 循环做
    文件(page cache) --read()--> pinned buffer --H2D--> GPU
另外起一批 "后台忙碌进程" 模拟真实服务里大量未绑核的计算线程: 它们不绑核,
工作内存按 first-touch 落在 node0 (模拟服务里大部分内存在 node0 的情况).

两臂只差一件事: 进程启动时是否 set_mempolicy(MPOL_LOCAL).
  default: 默认策略, 自动均衡会扫描后台进程, 并按 hint fault 统计挪动它们
  local  : 显式 MPOL_LOCAL (放置规则同 first-touch), 自动均衡跳过整个进程树

用法 (blog 里的最终配置; --gpus 按机器改成两个 node 各两张卡):
  python repro_numa_balancing.py --policy default --bg-procs 90 --bg-kind cpu --bg-start split \
      --gpus 2,3,4,5 --data-dir /tmp/numa_repro --out result_default.json
  python repro_numa_balancing.py --policy local   --bg-procs 90 --bg-kind cpu --bg-start split \
      --gpus 2,3,4,5 --data-dir /tmp/numa_repro --out result_local.json
"""

import argparse
import ctypes
import os
import sys

# ---- 必须在 import torch 之前设策略: 之后创建的线程/子进程都会继承 -------------------------
_ap = argparse.ArgumentParser()
_ap.add_argument("--policy", choices=["default", "local"], required=True)
_ap.add_argument("--gpus", default="2,3,4,5")
_ap.add_argument("--bg-procs", type=int, default=80)
_ap.add_argument("--bg-mem-mib", type=int, default=256)
_ap.add_argument("--bg-mem-node", type=int, default=0)
_ap.add_argument("--bg-kind", choices=["stream", "cpu"], default="stream",
                 help="stream: 持续流式读写整块工作内存 (吃内存带宽); cpu: 在一小块缓存内数组上算,"
                      " 只稀疏地碰一下大块内存 (主要吃 CPU, 更像真实服务里的计算线程)")
_ap.add_argument("--bg-start", choices=["memnode", "split"], default="memnode",
                 help="memnode: 分配完内存后从 mem-node 起跑; split: 一半先在另一个 node 上跑 1 秒再放开,"
                      " 让两臂的起点一样均衡, 差别只来自自动均衡")
_ap.add_argument("--file-gib", type=float, default=2.0)
_ap.add_argument("--chunk-mib", type=int, default=256)
_ap.add_argument("--duration", type=float, default=90.0)
_ap.add_argument("--data-dir", required=True)
_ap.add_argument("--out", required=True)
ARGS = _ap.parse_args()

MPOL_DEFAULT, MPOL_BIND, MPOL_LOCAL = 0, 2, 4
libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
libnuma.set_mempolicy.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong)
libnuma.get_mempolicy.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_ulong,
                                  ctypes.c_void_p, ctypes.c_ulong)
libnuma.numa_move_pages.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int)


def set_policy(mode, node=None):
    if node is None:
        rc = libnuma.set_mempolicy(mode, None, 0)
    else:
        mask = ctypes.c_ulong(1 << node)
        rc = libnuma.set_mempolicy(mode, ctypes.byref(mask), 64)
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def get_policy_mode():
    mode = ctypes.c_int(-1)
    libnuma.get_mempolicy(ctypes.byref(mode), None, 0, None, 0)
    return mode.value


if ARGS.policy == "local":
    set_policy(MPOL_LOCAL)
PROCESS_POLICY_MODE = get_policy_mode()

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import statistics  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402


def node_cpus(node):
    text = open(f"/sys/devices/system/node/node{node}/cpulist").read().strip()
    cpus = set()
    for part in text.split(","):
        a, _, b = part.partition("-")
        cpus |= set(range(int(a), int(b or a) + 1))
    return cpus


NODE_CPUS = {0: node_cpus(0), 1: node_cpus(1)}
ALL_CPUS = NODE_CPUS[0] | NODE_CPUS[1]


def gpu_node(index):
    import torch
    p = torch.cuda.get_device_properties(index)
    addr = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
    return int(open(f"/sys/bus/pci/devices/{addr}/numa_node").read())


def pages_node(address, length, samples=16):
    """用 move_pages(nodes=NULL) 查询几页实际落在哪个 node."""
    page = os.sysconf("SC_PAGE_SIZE")
    addrs = [(address + i * (length // samples)) // page * page for i in range(samples)]
    arr = (ctypes.c_void_p * samples)(*addrs)
    status = (ctypes.c_int * samples)()
    libnuma.numa_move_pages(0, samples, arr, None, status, 0)
    return sorted(set(status))


# ---- 后台忙碌进程 ------------------------------------------------------------------------------
def background_worker(stop_at, mem_mib, mem_node, kind, start_node, ready):
    # 分配期间临时绑在 mem_node 的 CPU 上, first-touch 让工作内存落在 mem_node; 之后放开亲和,
    # 让调度器自由安排. 进程策略从父进程继承 (default 或 MPOL_LOCAL).
    os.sched_setaffinity(0, NODE_CPUS[mem_node])
    buf = np.ones(mem_mib * 1024 * 1024 // 8, dtype=np.float64)
    if start_node is not None:                # 先在指定 node 上跑一会儿, 再放开亲和
        os.sched_setaffinity(0, NODE_CPUS[start_node])
        t = time.time()
        while time.time() - t < 1.0:
            buf[:4096] += 1.0
    os.sched_setaffinity(0, ALL_CPUS)
    ready.set()
    if kind == "stream":
        while time.time() < stop_at:
            buf *= 1.0000001            # 持续读写整块内存: 吃内存带宽
            buf[:: 4096 // 8] += 1.0
    else:
        hot = buf[: 256 * 1024 // 8]    # 256 KiB, 常驻 L2
        pages = buf[:: 4096 // 8]       # 每页一个元素
        i = 0
        while time.time() < stop_at:
            for _ in range(200):
                hot *= 1.0000001        # 纯计算, 几乎不出缓存
            pages[i % 64 :: 64] += 1.0  # 每轮只碰 1/64 的页: 给扫描器留下"内存在 node0"的统计, 带宽很小
            i += 1


# ---- loader 线程 -------------------------------------------------------------------------------
def loader(gpu, node, path, chunk, stop_at, result, barrier):
    import torch
    os.sched_setaffinity(0, NODE_CPUS[node])
    set_policy(MPOL_BIND, node)
    torch.cuda.set_device(gpu)
    stream = torch.cuda.Stream()
    size = os.path.getsize(path)
    bufs = [torch.empty(chunk, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    events = [torch.cuda.Event(), torch.cuda.Event()]
    dev = torch.empty(size, dtype=torch.uint8, device=f"cuda:{gpu}")
    result["pinned_pages_node"] = pages_node(bufs[0].data_ptr(), chunk)
    # 用本线程 (已绑 node) 重新读一遍文件, 让 page cache 落在本 node
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    while os.read(fd, 64 << 20):
        pass
    loads, h2d_gibps = [], []
    tid = threading.get_native_id()
    barrier.wait()
    sched0 = open(f"/proc/self/task/{tid}/schedstat").read().split()
    while time.time() < stop_at:
        t0 = time.perf_counter()
        os.lseek(fd, 0, os.SEEK_SET)
        offset, i = 0, 0
        while offset < size:
            b = bufs[i % 2]
            events[i % 2].synchronize()                      # 等这块 pinned 上一次 H2D 完成
            view = memoryview(b.numpy())
            n = os.readv(fd, [view[: min(chunk, size - offset)]])   # page cache -> pinned (H2H)
            with torch.cuda.stream(stream):
                dev[offset: offset + n].copy_(b[:n], non_blocking=True)  # pinned -> GPU (H2D)
                events[i % 2].record(stream)
            offset += n
            i += 1
        stream.synchronize()
        loads.append((time.perf_counter() - t0) * 1000)
    sched1 = open(f"/proc/self/task/{tid}/schedstat").read().split()
    os.close(fd)
    # 单独测一次纯 H2D (pinned -> GPU) 的带宽, 作为本机上限参照
    with torch.cuda.stream(stream):
        t0 = time.perf_counter()
        for _ in range(8):
            dev[:chunk].copy_(bufs[0], non_blocking=True)
        stream.synchronize()
        h2d_gibps = 8 * chunk / (time.perf_counter() - t0) / 2**30
    result.update(
        gpu=gpu, node=node, loads_ms=loads, file_gib=size / 2**30,
        run_ns=int(sched1[0]) - int(sched0[0]), wait_ns=int(sched1[1]) - int(sched0[1]),
        pure_h2d_gibps=h2d_gibps,
    )


# ---- 采样: vmstat NUMA 计数, 每个 node 的 CPU 忙碌度, 后台进程跑在哪 -------------------------
VM_KEYS = ("numa_pte_updates", "numa_huge_pte_updates", "numa_hint_faults", "numa_hint_faults_local",
           "numa_pages_migrated")


def vmstat():
    d = {}
    for line in open("/proc/vmstat"):
        k, v = line.split()
        if k in VM_KEYS:
            d[k] = int(v)
    return d


def cpu_busy():
    busy = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            vals = list(map(int, f[1:]))
            busy[int(f[0][3:])] = (sum(vals) - vals[3] - vals[4], sum(vals))
    return busy


def bg_placement(pids):
    on0 = on1 = 0
    pref = {}
    for pid in pids:
        try:
            cpu = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[36])
            for line in open(f"/proc/{pid}/sched"):
                if line.startswith("numa_preferred_nid"):
                    n = int(line.split(":")[1])
                    pref[n] = pref.get(n, 0) + 1
        except (OSError, ValueError, IndexError):
            continue
        if cpu in NODE_CPUS[0]:
            on0 += 1
        else:
            on1 += 1
    return on0, on1, pref


def main():
    import torch
    gpus = [int(x) for x in ARGS.gpus.split(",")]
    chunk = ARGS.chunk_mib << 20
    os.makedirs(ARGS.data_dir, exist_ok=True)
    paths = {}
    for g in gpus:
        p = os.path.join(ARGS.data_dir, f"blob_gpu{g}.bin")
        want = int(ARGS.file_gib * 2**30)
        if not os.path.exists(p) or os.path.getsize(p) != want:
            with open(p, "wb") as f:
                block = os.urandom(64 << 20)
                for _ in range(want // len(block)):
                    f.write(block)
        paths[g] = p

    # 先 fork 后台进程 (在任何 CUDA 初始化之前), 它们继承本进程的内存策略
    ctx = mp.get_context("fork")
    warm = 20.0                                             # 给扫描/放置留出建立统计的时间
    t_start = time.time()
    stop_at = t_start + warm + ARGS.duration
    readies = [ctx.Event() for _ in range(ARGS.bg_procs)]
    starts = [None if ARGS.bg_start == "memnode" else i % 2 for i in range(ARGS.bg_procs)]
    procs = [ctx.Process(target=background_worker,
                         args=(stop_at, ARGS.bg_mem_mib, ARGS.bg_mem_node, ARGS.bg_kind, sn, r))
             for r, sn in zip(readies, starts)]
    for p in procs:
        p.start()
    for r in readies:
        r.wait()
    pids = [p.pid for p in procs]

    nodes = {g: gpu_node(g) for g in gpus}
    results = {g: {} for g in gpus}
    barrier = threading.Barrier(len(gpus) + 1)
    threads = [threading.Thread(target=loader, args=(g, nodes[g], paths[g], chunk, stop_at, results[g], barrier))
               for g in gpus]
    for t in threads:
        t.start()
    # 预热期: 后台进程已在跑, loader 在准备 (读文件落 page cache). 预热结束后才开始计时.
    while time.time() < t_start + warm:
        time.sleep(0.5)
    barrier.wait()
    vm0, cpu0 = vmstat(), cpu_busy()
    placement = []
    while any(t.is_alive() for t in threads):
        placement.append(bg_placement(pids))
        time.sleep(2.0)
    vm1, cpu1 = vmstat(), cpu_busy()
    for p in procs:
        p.join()

    node_busy = {}
    for n, cpus in NODE_CPUS.items():
        b = sum(cpu1[c][0] - cpu0[c][0] for c in cpus)
        t = sum(cpu1[c][1] - cpu0[c][1] for c in cpus)
        node_busy[n] = b / t if t else 0.0
    on0 = statistics.mean(x[0] for x in placement) if placement else None
    on1 = statistics.mean(x[1] for x in placement) if placement else None
    pref_last = placement[-1][2] if placement else {}
    summary = {
        "policy_arg": ARGS.policy,
        "process_policy_mode": PROCESS_POLICY_MODE,
        "bg_procs": ARGS.bg_procs, "bg_kind": ARGS.bg_kind, "bg_start": ARGS.bg_start, "bg_mem_node": ARGS.bg_mem_node, "duration_s": ARGS.duration,
        "vmstat_delta": {k: vm1[k] - vm0[k] for k in VM_KEYS},
        "node_cpu_busy_fraction": node_busy,
        "bg_procs_on_node0_mean": on0, "bg_procs_on_node1_mean": on1,
        "bg_numa_preferred_nid_last": pref_last,
        "loaders": [],
    }
    for g in gpus:
        r = results[g]
        loads = r["loads_ms"]
        summary["loaders"].append({
            "gpu": g, "node": r["node"], "pinned_pages_node": r["pinned_pages_node"],
            "n_loads": len(loads), "load_ms_median": statistics.median(loads) if loads else None,
            "load_ms_p90": sorted(loads)[int(len(loads) * 0.9)] if loads else None,
            "effective_gibps": r["file_gib"] / (statistics.median(loads) / 1000) if loads else None,
            "wait_over_run": r["wait_ns"] / r["run_ns"] if r["run_ns"] else None,
            "pure_h2d_gibps": r["pure_h2d_gibps"],
            "loads_ms": loads,
        })
    json.dump(summary, open(ARGS.out, "w"), indent=1)
    brief = {k: v for k, v in summary.items() if k != "loaders"}
    print(json.dumps(brief, indent=1))
    for L in summary["loaders"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in L.items() if k != "loads_ms"})


if __name__ == "__main__":
    main()
