## scripts

- `repro_numa_balancing.py`: 正文"一个两 node 对称的最小复现"用的脚本
- `collect_repro.py`: 把多轮输出的 `result_*.json` 汇总成 `../data/repro_results.json` (会覆盖这个文件)
- `plot_figures.py`: 从 `../data` 画正文里的数据图, 输出到 `../images`

### 跑复现

需要两个 NUMA node 各有 GPU, 以及 `libnuma`.

```bash
# 默认策略 vs MPOL_LOCAL, 90 个 CPU 密集型后台进程, 起跑对半分
python repro_numa_balancing.py --policy default --bg-procs 90 --bg-kind cpu --bg-start split \
    --gpus 2,3,4,5 --data-dir /tmp/numa_repro --out result_default.json
python repro_numa_balancing.py --policy local   --bg-procs 90 --bg-kind cpu --bg-start split \
    --gpus 2,3,4,5 --data-dir /tmp/numa_repro --out result_local.json
```

- `--gpus` 要按你的机器改成两个 node 各两张卡
- 每轮先预热 20 秒, 再计时 90 秒 (`--duration`); 正文的结果是两组交替各跑两轮
- 脚本会在 `--data-dir` 下生成每卡 2 GiB 的测试文件, 跑完记得删掉

`../data/repro_results.json` 里除了正文用到的这组配置, 还留着几组探索时跑的配置 (80 / 100 个后台进程、全部从 node0 起跑、后台进程流式读写内存), 供参考.

### 画图

```bash
MPLCONFIGDIR=/tmp/mpl python plot_figures.py
```
