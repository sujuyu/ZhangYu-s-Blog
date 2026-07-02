import torch
import time
import os
from transformers import AutoConfig, AutoModel

# --- configs ---
MODELS_TO_BENCHMARK = [
    {
        "model_id": "google/vit-base-patch16-224", 
        "model_type": "vision"
    },
    {
        "model_id": "facebook/deit-base-distilled-patch16-224",
        "model_type": "vision"
    },
    {
        "model_id": "microsoft/swin-base-patch4-window7-224",
        "model_type": "vision"
    },
    {
        "model_id": "bert-base-uncased",
        "model_type": "text"
    },
    {
        "model_id": "roberta-base",
        "model_type": "text"
    },
]

BATCH_SIZE = 4
# --- Benchmark Configs ---
WARMUP_RUNS = 20
MAIN_RUNS = 100

def prepare_inputs_for_export(model, config, model_type, device):
    """
    Prepares a dictionary of sample inputs for the given model.
    The model object is needed to check for 'token_type_ids'.
    """
    if model_type == "vision":
        num_channels = config.num_channels
        height = config.image_size
        width = config.image_size
        pixel_values = torch.randn(BATCH_SIZE, num_channels, height, width, device=device)
        return {"pixel_values": pixel_values}
    
    elif model_type == "text":
        vocab_size = config.vocab_size
        max_position_embeddings = config.max_position_embeddings
        seq_len = min(128, max_position_embeddings)

        input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, seq_len), device=device, dtype=torch.long)
        attention_mask = torch.ones(BATCH_SIZE, seq_len, device=device, dtype=torch.long)
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        if "token_type_ids" in model.forward.__code__.co_varnames:
            token_type_ids = torch.zeros(BATCH_SIZE, seq_len, device=device, dtype=torch.long)
            inputs["token_type_ids"] = token_type_ids
        return inputs
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def compile_and_package(model, inputs, model_id):
    """
    AOT compiles the model and packages it into a .pt2 file.
    """
    print(f"--- Compiling {model_id} with AOTInductor ---")
    package_path = os.path.join("model_pts", f"{model_id.replace('/', '_')}.pt2")
    
    # Skip compilation if the package already exists
    if os.path.exists(package_path):
        print(f"Package already exists at {package_path}. Skipping compilation.")
        return package_path

    dynamic_shapes = {}
    # for key, value in inputs.items():
    #     dynamic_shapes[key] = {0: torch.export.Dim("batch", min=1, max=16)}

    with torch.inference_mode():
        exported_model = torch.export.export(
            model,
            args=(),
            kwargs=inputs,
            dynamic_shapes=dynamic_shapes
        )

    inductor_configs = {}
    torch._inductor.aoti_compile_and_package(
        exported_model,
        package_path=package_path,
        inductor_configs=inductor_configs
    )
    print(f"Successfully compiled and saved to {package_path}")
    return package_path

def benchmark_latency(func, warmup_runs, main_runs):
    """
    Benchmarks the latency of a given function using CUDA events for precision.
    """
    # Warmup runs
    for _ in range(warmup_runs):
        func()
    torch.cuda.synchronize()

    # Main benchmark runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(main_runs):
        func()
    end_event.record()
    
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = total_time_ms / main_runs
    return avg_latency_ms

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is required for this benchmark script."
    assert not torch._C._GLIBCXX_USE_CXX11_ABI, "This script requires GLIBCXX_USE_CXX11_ABI=0"

    device = torch.device("cuda")
    os.makedirs("model_pts", exist_ok=True)

    for model_info in MODELS_TO_BENCHMARK:
        model_id = model_info["model_id"]
        model_type = model_info["model_type"]
        
        print(f"\n{'='*20} Benchmarking: {model_id} {'='*20}")

        # 1. Load original model and prepare inputs
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModel.from_config(config).eval().to(device)
        inputs_dict = prepare_inputs_for_export(model, config, model_type, device)
        # The AOT runner expects a list/tuple of tensors, not a dict
        inputs_list = list(inputs_dict.values())

        # 2. Compile the model
        package_path = compile_and_package(model, inputs_dict, model_id)

        # 3. Load the AOT-compiled model
        print("--- Loading AOT package ---")
        # run_single_threaded=True simplifies the C++ backend by not using the 
        # thread pool, which is fine and often slightly faster for single-stream benchmarks.
        aot_model = torch._inductor.aoti_load_package(package_path, run_single_threaded=True)
        
        # --- 4. Benchmark Eager Mode ---
        print("--- Benchmarking Eager Mode Latency ---")
        eager_func = lambda: aot_model(inputs_list)
        eager_latency = benchmark_latency(eager_func, WARMUP_RUNS, MAIN_RUNS)
        print(f"Eager mode average latency: {eager_latency:.4f} ms")

        # --- 5. Capture CUDA Graph ---
        print("--- Capturing CUDA Graph ---")
        graph = torch.cuda.CUDAGraph()
        
        # Do a single run to ensure any lazy-loading of kernels happens before capture
        aot_model(inputs_list)
        torch.cuda.synchronize()
        
        # Capture
        graph.capture_begin()
        aot_model(inputs_list)
        graph.capture_end()
        print("CUDA Graph captured successfully!")

        # --- 6. Benchmark CUDA Graph Mode ---
        print("--- Benchmarking CUDA Graph Mode Latency ---")
        graph_func = lambda: graph.replay()
        graph_latency = benchmark_latency(graph_func, WARMUP_RUNS, MAIN_RUNS)
        print(f"CUDA Graph mode average latency: {graph_latency:.4f} ms")
        
        # --- 7. Report Results ---
        print("\n--- Performance Summary ---")
        print(f"Model: {model_id}")
        print(f"  Eager Latency:       {eager_latency:.4f} ms")
        print(f"  CUDA Graph Latency:  {graph_latency:.4f} ms")
        if graph_latency > 0:
            speedup = eager_latency / graph_latency
            print(f"  Speedup Factor:      {speedup:.2f}x")
        print(f"{'='*60}")
