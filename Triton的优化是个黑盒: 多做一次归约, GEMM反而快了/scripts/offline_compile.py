"""不占 GPU, 把 kernels.py 里的几个版本对 sm80 (A100) 编译, 导出 TTGIR / SASS, 统计 GEMM 主循环的指令.

    python offline_compile.py <输出目录>                 # 编译全部版本, 打印循环体统计
    python offline_compile.py <输出目录> --find-pass v2  # 每个 pass 后 dump IR, 找出 BN 被挪进 dot 操作数布局的那个 pass

特化方式模仿 AOTI 的实际调用: 指针和整除 16 的整数参数带 tt.divisibility, 最内维步长是常量 1 (kernel 里直接写死).
实测环境 triton 3.7.1; 换版本后指令数可能不同. kernel 是精简版, 和线上算子的指令数可能差几条
(比如这里 v1 第二遍是 274 条, 线上算子是 272 条), 布局的结论相同.
"""

import collections
import os
import re
import subprocess
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kernels  # noqa: E402

ARGS = ["x_ptr", "stride_x_b", "mean_ptr", "scale_ptr", "beta_ptr", "stride_bn_m", "weight_ptr", "stride_w_m",
        "stride_w_k", "bias_ptr", "stride_bias_m", "rs_ptr", "out_ptr", "stride_o_m", "stride_o_b", "batch", "k", "d_out"]
DIV16 = {"stride_x_b", "stride_bn_m", "stride_w_m", "stride_w_k", "stride_bias_m", "stride_o_m", "stride_o_b", "k", "d_out"}
# (名字, MODE, 权重 dtype)
VARIANTS = [
    ("v1", "v1", "fp16"),
    ("v2", "v2", "fp16"),
    ("v2p", "v2p", "fp16"),
    ("v3", "v3", "fp16"),
    ("bf16", "plain", "bf16"),
    ("tf32", "plain", "fp32"),
]
NV_BIN = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin")


def compile_variant(mode, wdtype):
    sig = {n: (f"*{wdtype}" if n == "weight_ptr" else "*fp32") if n.endswith("_ptr") else "i32" for n in ARGS}
    for name in ("BLOCK_B", "BLOCK_OUT", "TILE_K", "MODE", "PREC"):
        sig[name] = "constexpr"
    attrs = {(i,): [["tt.divisibility", 16]] for i, n in enumerate(ARGS) if n.endswith("_ptr") or n in DIV16}
    consts = {"BLOCK_B": 64, "BLOCK_OUT": 256, "TILE_K": 32, "MODE": mode, "PREC": "tf32"}
    src = ASTSource(kernels.gemm_kernel, sig, constexprs=consts, attrs=attrs)
    return triton.compile(src, target=GPUTarget("cuda", 80, 32), options={"num_warps": 4, "num_stages": 3})


def sass_loops(sass):
    """返回每个循环 (回跳的 BRA) 的指令计数器."""
    labels, insts, pending = {}, [], None
    for line in sass.splitlines():
        label = re.match(r"\s*\.(L_x_\d+):", line)
        if label:
            pending = label.group(1)
            continue
        inst = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?);", line)
        if inst:
            addr = int(inst.group(1), 16)
            if pending:
                labels[pending], pending = addr, None
            insts.append((addr, inst.group(2).strip()))
    loops = []
    for addr, text in insts:
        target = re.search(r"BRA[^`]*`\(\.(L_x_\d+)\)", text)
        if target and labels.get(target.group(1), addr + 1) < addr:
            start = labels[target.group(1)]
            ops = [re.sub(r"^@!?U?P\w+\s+", "", t).split()[0].split(".")[0] for a, t in insts if start <= a <= addr]
            loops.append(collections.Counter(ops))
    return loops


def main():
    out_dir = sys.argv[1]
    if "--find-pass" in sys.argv:
        find_pass(out_dir, sys.argv[sys.argv.index("--find-pass") + 1])
        return
    keys = ["HMMA", "FFMA", "FADD", "FMUL", "F2FP", "LDS", "STS", "LDSM", "BAR", "LDGSTS", "LDG"]
    print(f"{'kernel':6s} {'regs':>4s} {'loop':>5s}  " + " ".join(f"{k:>6s}" for k in keys))
    for name, mode, wdtype in VARIANTS:
        ck = compile_variant(mode, wdtype)
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "kernel.ttgir"), "w") as f:
            f.write(ck.asm["ttgir"])
        cubin = os.path.join(d, "kernel.cubin")
        with open(cubin, "wb") as f:
            f.write(ck.asm["cubin"])
        sass = subprocess.run([os.path.join(NV_BIN, "nvdisasm"), "-c", cubin], capture_output=True, text=True).stdout
        with open(os.path.join(d, "kernel.sass"), "w") as f:
            f.write(sass)
        res = subprocess.run([os.path.join(NV_BIN, "cuobjdump"), "-res-usage", cubin], capture_output=True, text=True)
        regs = re.search(r"REG:(\d+)", res.stdout).group(1)
        gemm_loop = max(sass_loops(sass), key=lambda c: c["HMMA"])  # 有 HMMA 的那个循环
        print(f"{name:6s} {regs:>4s} {sum(gemm_loop.values()):>5d}  " + " ".join(f"{gemm_loop[k]:>6d}" for k in keys))


def find_pass(out_dir, name):
    """子进程里开 MLIR_ENABLE_DUMP 编译, 按 pass 切开 IR, 报告 FP32 的 dot 操作数张量第一次出现在哪个 pass 之后."""
    mode, wdtype = next((m, w) for n, m, w in VARIANTS if n == name)
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ, MLIR_ENABLE_DUMP="1", TRITON_CACHE_DIR=os.path.join(out_dir, f"cache_{name}"))
    code = (f"import sys; sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r}); "
            f"import offline_compile as o; o.compile_variant({mode!r}, {wdtype!r})")
    dump = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True).stderr
    parts = re.split(r"// -----// IR Dump Before (\S+)", dump)
    previous = None
    for i in range(1, len(parts), 2):
        count = len(re.findall(r"64x32xf32, #ttg\.dot_op", parts[i + 1]))
        if previous == 0 and count > 0:
            print(f"{name}: FP32 的 dot 操作数张量出现在 {parts[i - 2]} 之后 (下一个 pass 是 {parts[i]})")
            return
        previous = count
    print(f"{name}: 没有出现 FP32 的 dot 操作数张量, BN 留在 blocked 布局")


if __name__ == "__main__":
    main()
