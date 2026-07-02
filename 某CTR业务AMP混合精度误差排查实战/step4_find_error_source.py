"""
Step 4: 逐节点单步误差分析
对每个 call_function 节点，用 fp32 的输入分别在 fp32 和 autocast fp16 下执行，
比较输出差异，定位每个节点自身引入了多少误差。
"""
import torch
import pickle
import os
from torch.fx.interpreter import Interpreter
from collections import defaultdict

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

model = torch.export.load("fx_user_model/exported_model.pt")
inputs_dict = pickle.load(open("fx_user_model/inputs_dict.pkl", "rb"))
graph_module = model.module()

placeholders = [n.name for n in graph_module.graph.nodes if n.op == "placeholder"]
args = [inputs_dict[name] for name in placeholders]

# 1. 收集 fp32 下每个节点的值
class CollectInterpreter(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.node_values = {}
    def run_node(self, n):
        result = super().run_node(n)
        if isinstance(result, torch.Tensor):
            self.node_values[n.name] = result.detach().clone()
        return result

print("Collecting fp32 values...")
interp32 = CollectInterpreter(graph_module)
with torch.inference_mode():
    result_fp32 = interp32.run(*args)
fp32_values = interp32.node_values

# 2. 对每个 call_function 节点，用 fp32 输入，在 autocast 下执行单个 op
print("Measuring per-node introduced error...")

results = []
for node in graph_module.graph.nodes:
    if node.op != "call_function":
        continue
    if node.name not in fp32_values:
        continue

    fp32_args = []
    has_tensor_input = False
    for arg in node.args:
        if isinstance(arg, torch.fx.Node) and arg.name in fp32_values:
            fp32_args.append(fp32_values[arg.name])
            has_tensor_input = True
        elif isinstance(arg, torch.fx.Node):
            fp32_args.append(interp32.env.get(arg, arg))
        else:
            fp32_args.append(arg)

    if not has_tensor_input:
        continue

    fp32_kwargs = {}
    for k, v in node.kwargs.items():
        if isinstance(v, torch.fx.Node) and v.name in fp32_values:
            fp32_kwargs[k] = fp32_values[v.name]
        elif isinstance(v, torch.fx.Node):
            fp32_kwargs[k] = interp32.env.get(v, v)
        else:
            fp32_kwargs[k] = v

    try:
        with torch.inference_mode():
            result_fp32_node = node.target(*fp32_args, **fp32_kwargs)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            with torch.inference_mode():
                result_fp16_node = node.target(*fp32_args, **fp32_kwargs)

        if isinstance(result_fp32_node, torch.Tensor) and isinstance(result_fp16_node, torch.Tensor):
            if result_fp32_node.shape == result_fp16_node.shape and result_fp32_node.numel() > 0:
                diff = (result_fp32_node.float() - result_fp16_node.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                output_scale = result_fp32_node.abs().mean().item()
                rel_err = mean_diff / (output_scale + 1e-10)
                out_dtype = str(result_fp16_node.dtype).replace('torch.', '')
                results.append((node.name, str(node.target).split('aten.')[-1],
                              out_dtype, max_diff, mean_diff, rel_err, output_scale))
    except Exception:
        pass

# 按 mean_diff 排序
results.sort(key=lambda x: x[4], reverse=True)

print(f"\n{'Node':<45} {'op':<25} {'dtype':>8} {'max_diff':>10} {'mean_diff':>10} {'rel_err':>10} {'out_scale':>10}")
print("=" * 120)
for name, op, dtype, max_diff, mean_diff, rel_err, out_scale in results[:40]:
    print(f"{name:<45} {op:<25} {dtype:>8} {max_diff:>10.6f} {mean_diff:>10.6f} {rel_err:>10.6f} {out_scale:>10.4f}")

# 统计各类 op 的总误差贡献
print("\n\n--- 按 op 类型汇总 ---")
op_stats = defaultdict(lambda: {'count': 0, 'total_mean_diff': 0, 'max_max_diff': 0})
for name, op, dtype, max_diff, mean_diff, rel_err, out_scale in results:
    op_stats[op]['count'] += 1
    op_stats[op]['total_mean_diff'] += mean_diff
    op_stats[op]['max_max_diff'] = max(op_stats[op]['max_max_diff'], max_diff)

print(f"{'op':<30} {'count':>6} {'total_mean_diff':>15} {'max_max_diff':>15}")
print("=" * 70)
for op, stats in sorted(op_stats.items(), key=lambda x: x[1]['total_mean_diff'], reverse=True):
    print(f"{op:<30} {stats['count']:>6} {stats['total_mean_diff']:>15.6f} {stats['max_max_diff']:>15.6f}")
