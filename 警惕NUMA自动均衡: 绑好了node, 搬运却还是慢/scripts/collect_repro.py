"""把 repro_numa_balancing.py 各轮输出的 result_*.json 汇总成 ../data/repro_results.json, 并打印一张表.

    python collect_repro.py <放 result_*.json 的目录>
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
runs = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "result_*.json"))):
    r = json.load(open(path))
    r["run"] = os.path.basename(path)[len("result_"):-len(".json")]
    key = f"{r.get('bg_kind', 'stream')}{r['bg_procs']}" + ("-split" if r.get("bg_start") == "split" else "")
    runs.setdefault(key, {}).setdefault(r["policy_arg"], []).append(r)

print(f"{'run':22s} {'policy':8s} {'pte_updates':>12s} {'busy n0/n1':>11s} {'bg on n0':>8s}  loader median ms (wait/run)")
for key, arms in runs.items():
    for arm, rs in arms.items():
        for r in rs:
            L = sorted(r["loaders"], key=lambda x: x["gpu"])
            b = r["node_cpu_busy_fraction"]
            print(f"{r['run']:22s} {arm:8s} {r['vmstat_delta']['numa_pte_updates']:>12,} "
                  f"{100 * b['0']:4.0f}%/{100 * b['1']:3.0f}% {r['bg_procs_on_node0_mean'] or 0:8.1f}  "
                  + "  ".join(f"n{x['node']}:{x['load_ms_median']:.0f}({x['wait_over_run']:.2f})" for x in L))
json.dump(runs, open(os.path.join(HERE, "..", "data", "repro_results.json"), "w"), indent=1)
