"""
Step 1: 基线对比 —— fp32 vs AMP fp16 输出差异
展示模型在 AMP 混合精度下的初始误差水平
"""
import torch
import pickle
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

model = torch.export.load("fx_user_model/exported_model.pt")
inputs_dict = pickle.load(open("fx_user_model/inputs_dict.pkl", "rb"))

graph_module = model.module()

# fp32 baseline
print("Running fp32 baseline...")
with torch.inference_mode():
    result_fp32 = graph_module(**inputs_dict)

# AMP fp16
print("Running AMP fp16...")
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    with torch.inference_mode():
        result_fp16 = graph_module(**inputs_dict)

diff = (result_fp32.float() - result_fp16.float()).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
output_mean = result_fp32.abs().mean().item()
rel_err = mean_diff / (output_mean + 1e-10)

print(f"\n{'='*60}")
print(f"FP32 output mean:  {output_mean:.6f}")
print(f"Max abs diff:      {max_diff:.6f}")
print(f"Mean abs diff:     {mean_diff:.6f}")
print(f"Relative error:    {rel_err*100:.2f}%")
print(f"{'='*60}")

print(f"\nPer-sample comparison (first 20):")
print(f"{'idx':>5} {'fp32':>12} {'fp16':>12} {'abs_diff':>12}")
for i in range(min(20, result_fp32.shape[0])):
    v32 = result_fp32[i].item()
    v16 = result_fp16[i].item()
    print(f"{i:>5} {v32:>12.6f} {v16:>12.6f} {abs(v32-v16):>12.6f}")
