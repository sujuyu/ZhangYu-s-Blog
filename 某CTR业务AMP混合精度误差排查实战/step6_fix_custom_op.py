"""
Step 6: 使用 custom_op 替换关键 linear 节点为 fp32 版本
验证最终修复效果：BN eps 修复 + linear fp32 替换的叠加效果
"""
import torch
import pickle
import copy
import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# 注册 fp32_linear custom op
@torch.library.custom_op("rtp_lib::fp32_linear", mutates_args=())
def fp32_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=False):
        if x.dtype != torch.float32:
            x = x.float()
        return torch.ops.aten.linear.default(x, weight, bias).clone()

@torch.library.register_fake("rtp_lib::fp32_linear")
def fp32_linear_fake(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None):
    return torch.empty_like(torch.ops.aten.linear.default(x, weight))

# 关键节点 (误差传播贡献 > 3%)
critical_linears = {
    'linear_105',  # 87.7%  main_net.net.2.output_fc
    'linear_111',  # 34.4%  bias_net.net.2.output_fc
    'linear_109',  # 9.7%   bias_net.net.1.output_fc
    'linear_103',  # 6.3%   main_net.net.1.output_fc
    'linear_107',  # 4.6%   bias_net.net.0.output_fc
    'linear_102',  # 3.8%
}

if __name__ == "__main__":
    model = torch.export.load("fx_user_model/exported_model.pt")
    inputs_dict = pickle.load(open("fx_user_model/inputs_dict.pkl", "rb"))
    graph_module = model.module()

    # fp32 baseline
    print("Running fp32 baseline...")
    with torch.inference_mode():
        result_fp32 = graph_module(**inputs_dict)

    # 原始 autocast fp16（BN eps 已修复，但 linear 还是 fp16）
    print("Running autocast fp16 (BN eps fixed only)...")
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        with torch.inference_mode():
            result_fp16_bn_only = graph_module(**inputs_dict)

    # 替换关键 linear 节点
    graph_module_fixed = copy.deepcopy(model.module())

    replaced = []
    for node in list(graph_module_fixed.graph.nodes):
        if node.target == torch.ops.aten.linear.default and node.name in critical_linears:
            with graph_module_fixed.graph.inserting_after(node):
                cast_linear_node = graph_module_fixed.graph.create_node(
                    "call_function",
                    torch.ops.rtp_lib.fp32_linear,
                    args=node.args,
                )
            node.replace_all_uses_with(cast_linear_node)
            graph_module_fixed.graph.erase_node(node)
            replaced.append(cast_linear_node.name)

    graph_module_fixed.graph.lint()
    graph_module_fixed.recompile()
    print(f"Replaced {len(replaced)} linear nodes: {replaced}")

    # 验证 fp32 等价性
    print("Verifying fp32 equivalence...")
    with torch.inference_mode():
        result_fp32_fixed = graph_module_fixed(**inputs_dict)
    fp32_diff = (result_fp32_fixed.float() - result_fp32.float()).abs()
    print(f"FP32 equivalence: max_diff={fp32_diff.max().item():.10f}")

    # 修复后 autocast fp16（BN eps 修复 + linear fp32）
    print("Running autocast fp16 (BN eps fixed + linear fp32)...")
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        with torch.inference_mode():
            result_fp16_full_fix = graph_module_fixed(**inputs_dict)

    # ========== 还原 eps=0.001，展示完全未修复的误差 ==========
    graph_module_raw = copy.deepcopy(model.module())
    for node in graph_module_raw.graph.nodes:
        if node.target == torch.ops.aten.batch_norm.default:
            current_eps = node.args[7]
            if current_eps > 0.001:
                node.update_arg(7, 0.001)
    graph_module_raw.recompile()

    with torch.inference_mode():
        result_fp32_raw = graph_module_raw(**inputs_dict)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        with torch.inference_mode():
            result_fp16_raw = graph_module_raw(**inputs_dict)

    diff_raw = (result_fp32_raw.float() - result_fp16_raw.float()).abs()
    diff_bn_only = (result_fp32.float() - result_fp16_bn_only.float()).abs()
    diff_full = (result_fp32_fixed.float() - result_fp16_full_fix.float()).abs()
    output_mean = result_fp32.abs().mean().item()

    print(f"\n{'='*80}")
    print(f"FP32 output mean: {output_mean:.6f}")
    print(f"")
    print(f"{'Stage':<30} {'max_diff':>12} {'mean_diff':>12} {'rel_err':>10}")
    print(f"{'-'*70}")
    print(f"{'Original (eps=0.001)':<30} {diff_raw.max().item():>12.6f} {diff_raw.mean().item():>12.6f} {diff_raw.mean().item()/(output_mean+1e-10)*100:>9.2f}%")
    print(f"{'After BN eps fix':<30} {diff_bn_only.max().item():>12.6f} {diff_bn_only.mean().item():>12.6f} {diff_bn_only.mean().item()/(output_mean+1e-10)*100:>9.2f}%")
    print(f"{'After BN + linear fp32':<30} {diff_full.max().item():>12.6f} {diff_full.mean().item():>12.6f} {diff_full.mean().item()/(output_mean+1e-10)*100:>9.2f}%")
    print(f"{'='*80}")
    print(f"\nTotal improvement: {diff_raw.mean().item()/max(diff_full.mean().item(),1e-10):.1f}x mean_diff reduction")

    # 逐样本对比
    print(f"\nPer-sample comparison (first 10):")
    print(f"{'fp32':>10} {'raw_fp16':>10} {'bn_fix':>10} {'full_fix':>10} {'raw_err':>10} {'final_err':>10}")
    for i in range(min(10, result_fp32.shape[0])):
        v32 = result_fp32[i].item()
        v_raw = result_fp16_raw[i].item()
        v_bn = result_fp16_bn_only[i].item()
        v_full = result_fp16_full_fix[i].item()
        print(f"{v32:>10.4f} {v_raw:>10.4f} {v_bn:>10.4f} {v_full:>10.4f} {abs(v32-v_raw):>10.6f} {abs(v32-v_full):>10.6f}")
