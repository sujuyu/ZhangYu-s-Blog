"""
Step 3: 修复 BN eps —— 将方差过小的 BN 层的 eps 从 0.001 增大到合适值
对比修复前后的误差变化

注意：当前 fx_user_model/user_model/module.py 中的 eps 已经是修复后的值。
本脚本通过 update_arg 将 eps 还原为 0.001，复现修复前的误差，再对比修复后的效果。
"""
import torch
import pickle
import copy
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

model = torch.export.load("fx_user_model/exported_model.pt")
inputs_dict = pickle.load(open("fx_user_model/inputs_dict.pkl", "rb"))

# ========== 当前（已修复 eps）的误差 ==========
graph_module_fixed = model.module()

print("Running fp32 baseline (fixed eps)...")
with torch.inference_mode():
    result_fp32_fixed = graph_module_fixed(**inputs_dict)

print("Running AMP fp16 (fixed eps)...")
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    with torch.inference_mode():
        result_fp16_fixed = graph_module_fixed(**inputs_dict)

diff_fixed = (result_fp32_fixed.float() - result_fp16_fixed.float()).abs()

# ========== 还原 eps=0.001，复现修复前的误差 ==========
graph_module_orig = copy.deepcopy(model.module())

# 找到所有 batch_norm 节点，将 eps > 0.001 的还原为 0.001
restored_nodes = []
for node in graph_module_orig.graph.nodes:
    if node.target == torch.ops.aten.batch_norm.default:
        current_eps = node.args[7]
        if current_eps > 0.001:
            # eps 是 args[7]，用 update_arg 还原
            node.update_arg(7, 0.001)
            restored_nodes.append((node.name, current_eps, 0.001))

graph_module_orig.recompile()

print(f"\nRestored {len(restored_nodes)} BN nodes to eps=0.001:")
for name, old_eps, new_eps in restored_nodes:
    print(f"  {name}: {old_eps} -> {new_eps}")

print("\nRunning fp32 baseline (original eps=0.001)...")
with torch.inference_mode():
    result_fp32_orig = graph_module_orig(**inputs_dict)

print("Running AMP fp16 (original eps=0.001)...")
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    with torch.inference_mode():
        result_fp16_orig = graph_module_orig(**inputs_dict)

diff_orig = (result_fp32_orig.float() - result_fp16_orig.float()).abs()

# ========== 对比 ==========
output_mean = result_fp32_fixed.abs().mean().item()

print(f"\n{'='*70}")
print(f"FP32 output mean: {output_mean:.6f}")
print(f"")
print(f"{'Metric':<20} {'eps=0.001 (before)':>20} {'eps fixed (after)':>20} {'Improvement':>15}")
print(f"{'-'*75}")
print(f"{'max_diff':<20} {diff_orig.max().item():>20.6f} {diff_fixed.max().item():>20.6f} {diff_orig.max().item()/max(diff_fixed.max().item(),1e-10):>14.1f}x")
print(f"{'mean_diff':<20} {diff_orig.mean().item():>20.6f} {diff_fixed.mean().item():>20.6f} {diff_orig.mean().item()/max(diff_fixed.mean().item(),1e-10):>14.1f}x")
rel_orig = diff_orig.mean().item() / (output_mean + 1e-10)
rel_fixed = diff_fixed.mean().item() / (output_mean + 1e-10)
print(f"{'rel_err':<20} {rel_orig*100:>19.2f}% {rel_fixed*100:>19.2f}% {rel_orig/max(rel_fixed,1e-10):>14.1f}x")
print(f"{'='*70}")
