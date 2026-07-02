"""
Step 5: 逐节点误差传播分析
每次只让一个节点用 fp16 执行，其他全部用 fp32 正确值，
测量该节点的 fp16 误差独立传播到最终输出的贡献。
"""
import torch
import pickle
import os
from torch.fx.interpreter import Interpreter

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

# fp16 baseline
interp16 = CollectInterpreter(graph_module)
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    with torch.inference_mode():
        result_fp16 = interp16.run(*args)
total_fp16_diff = (result_fp32.float() - result_fp16.float()).abs().mean().item()
print(f"Total fp16 mean_diff: {total_fp16_diff:.6f}")

# 2. 建立节点拓扑序
all_nodes = list(graph_module.graph.nodes)
node_order = {n.name: i for i, n in enumerate(all_nodes)}

# 3. 找出所有产生误差的节点 (matmul + linear)
target_nodes = []
for node in graph_module.graph.nodes:
    if node.op == "call_function" and node.target in (
        torch.ops.aten.matmul.default,
        torch.ops.aten.linear.default,
    ):
        if node.name in fp32_values:
            target_nodes.append(node)

print(f"Testing {len(target_nodes)} nodes...")

class SingleNodeFP16Interpreter(Interpreter):
    """
    目标节点之前: 直接用 fp32 正确值
    目标节点: autocast fp16 执行
    目标节点之后: 正常 fp32 执行（让误差自然传播到输出）
    """
    def __init__(self, module, fp16_node_name, fp32_values, target_order):
        super().__init__(module)
        self.fp16_node_name = fp16_node_name
        self.fp32_values = fp32_values
        self.target_order = target_order
        self.passed_target = False

    def run_node(self, n):
        if n.name == self.fp16_node_name:
            self.passed_target = True
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                result = super().run_node(n)
            if isinstance(result, torch.Tensor) and result.dtype == torch.float16:
                result = result.float()
            self.env[n] = result
            return result
        elif not self.passed_target:
            if n.name in self.fp32_values:
                val = self.fp32_values[n.name]
                self.env[n] = val
                return val
            else:
                return super().run_node(n)
        else:
            return super().run_node(n)

results = []
for i, node in enumerate(target_nodes):
    order = node_order[node.name]
    interp = SingleNodeFP16Interpreter(graph_module, node.name, fp32_values, order)
    with torch.inference_mode():
        result = interp.run(*args)

    diff = (result.float() - result_fp32.float()).abs()
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()

    op_name = 'matmul' if 'matmul' in str(node.target) else 'linear'
    out_scale = fp32_values[node.name].abs().mean().item()
    results.append((node.name, op_name, mean_diff, max_diff, out_scale))

    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(target_nodes)}")

# 排序
results.sort(key=lambda x: x[2], reverse=True)

print(f"\n{'Node':<45} {'op':>8} {'mean_diff':>12} {'max_diff':>12} {'out_scale':>10} {'pct':>8}")
print("=" * 100)
cumulative = 0
for name, op, mean_diff, max_diff, out_scale in results[:50]:
    pct = mean_diff / total_fp16_diff * 100
    cumulative += pct
    marker = " <---" if pct > 3 else ""
    print(f"{name:<45} {op:>8} {mean_diff:>12.6f} {max_diff:>12.6f} {out_scale:>10.4f} {pct:>7.2f}%{marker}")

total_contribution = sum(r[2] for r in results)
print(f"\nTop 50 cumulative: {cumulative:.1f}%")
print(f"Sum of all individual contributions: {total_contribution:.6f}")
print(f"Total fp16 error: {total_fp16_diff:.6f}")
