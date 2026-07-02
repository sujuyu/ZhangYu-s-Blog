"""
Step 2: 检查 batch_norm 的 running_var
找出哪些 BN 层的方差特别小，可能导致数值不稳定
"""
import torch
import pickle
import os
from torch.fx.interpreter import Interpreter

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

model = torch.export.load("fx_user_model/exported_model.pt")
inputs_dict = pickle.load(open("fx_user_model/inputs_dict.pkl", "rb"))

graph_module = model.module()

# 收集所有节点的值
def get_all_node_value(interpreter, graph_model, inputs_dict):
    import copy
    original_env = copy.deepcopy(interpreter.env)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        with torch.inference_mode():
            executed_nodes = dict()
            for ph in [_ for _ in graph_model.graph.nodes if _.op == "placeholder"]:
                interpreter.env[ph] = inputs_dict[ph.name]
                executed_nodes[ph] = inputs_dict[ph.name]

            def execute_node(node):
                if node in executed_nodes:
                    return
                for input_node in node.all_input_nodes:
                    execute_node(input_node)
                value = interpreter.run_node(node)
                executed_nodes[node] = value if isinstance(value, torch.Tensor) else None
                interpreter.env[node] = value

            for node in graph_model.graph.nodes:
                if node in executed_nodes:
                    continue
                execute_node(node)
            interpreter.env = original_env
            return executed_nodes

interpreter = Interpreter(graph_module)
all_node_value = get_all_node_value(interpreter, graph_module, inputs_dict)
all_node_value = {str(k): v for k, v in all_node_value.items()}

# 检查每个 batch_norm 的 running_var
print(f"{'BN Node':<30} {'eps':>10} {'var_min':>12} {'var_mean':>12} {'var_max':>12} {'var<eps count':>15}")
print("=" * 95)

for node in graph_module.graph.nodes:
    if node.target == torch.ops.aten.batch_norm.default:
        # batch_norm args: input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled
        var_node = node.args[4]
        eps = node.args[7]
        var_tensor = all_node_value.get(str(var_node))
        if var_tensor is not None:
            var_min = var_tensor.min().item()
            var_mean = var_tensor.mean().item()
            var_max = var_tensor.max().item()
            small_var_count = (var_tensor < eps).sum().item()
            total = var_tensor.numel()
            marker = " <--- DANGER!" if var_min < eps else ""
            print(f"{node.name:<30} {eps:>10.4f} {var_min:>12.6f} {var_mean:>12.6f} {var_max:>12.6f} {small_var_count:>7}/{total:<7}{marker}")
