# 从一条 Issue 到 AOTInductor 跨 SM 部署

2025 年 5 月，我在 PyTorch 提过一条 [Issue #153697](https://github.com/pytorch/pytorch/issues/153697)。

主要问题是我的开发机是NVIDIA A10, 线上运行的机器是NVIDIA H20. AOTInductor 会把模型里的 PyTorch 算子 codegen 成 Triton kernel，Triton 再把 kernel 编译成 cubin。cubin 已经是面向特定 NVIDIA SM 架构的机器码，因此在一张卡上生成的 AOTInductor package，换到另一代 GPU 上不一定还能加载。

这对于离线编译和线上部署是一个很现实的问题。训练或者临时实验通常可以在目标机器上重新编译，但模型发布系统更希望同一份产物能够部署到多种 GPU，而不是为 A100、L40、H100 和更新的 GPU 各维护一份 package。

我当时只是把问题和需求提了出来。最近重新翻 PyTorch main 的代码时，意外发现这项能力真的已经被实现了。🥹感恩 [Bin Bao](https://github.com/binbao)：2025 年 5 月 27 日，他在 [PR #154413](https://github.com/pytorch/pytorch/pull/154413) 中加入了最初的 multi-arch kernel binary 选项。后来社区继续完善这条链路，例如 Jason Ansel 在 [PR #185328](https://github.com/pytorch/pytorch/pull/185328) 中修正了在新 GPU 上面向较老部署架构生成 PTX 和 fatbin 的行为。

🥹感恩社区, 让兄弟们狠狠白嫖了. 我要狠狠歌颂🥹

## 原来的 AOTInductor package 为什么不能跨 SM

默认的 CUDA 编译链路可以简化成：

```text
Inductor 生成 Triton kernel
        ↓
Triton 生成 PTX
        ↓ ptxas
生成目标 SM 的 cubin
        ↓
打包进 PT2，由 C++ wrapper 加载
```

Triton 并不是没有 PTX。它的编译结果里同时保留了 `binary.asm["ptx"]` 和 `binary.asm["cubin"]`，只是过去 AOTInductor 最终保存和加载的是单架构 cubin。

PTX 是 NVIDIA 的虚拟指令集。相比 cubin，它可以由 CUDA driver 在较新的 GPU 上 JIT 成本机指令，但它也不是完全与架构无关：面向较高 compute capability 生成的 PTX 不能向下运行，旧 driver 也未必认识较新的 PTX ISA。

因此，更实用的部署产物是 fatbin：为已知 GPU 保存对应的 SASS，同时保留一份较低架构的 PTX，给未来更高架构的 GPU 兜底。

## 现在怎样生成多架构 package

当前 PyTorch main 可以这样配置。这里使用的是仍可能继续演进的内部配置，具体名称和行为应以所用 PyTorch 版本为准：

```python
import os

import torch


os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;9.0"

package = torch._inductor.aoti_compile_and_package(
    exported_program,
    package_path="model.pt2",
    inductor_configs={
        "aot_inductor.emit_multi_arch_kernel": True,
        "cuda.arch": "80",
    },
)
```

这里两个配置的作用不同：

- `cuda.arch="80"` 指定 Triton PTX 的最低目标是 `compute_80`。
- `TORCH_CUDA_ARCH_LIST` 指定希望提前生成哪些架构的 SASS。

上面的 fatbin 大致包含：

```text
sm_80 SASS
sm_86 SASS
sm_90 SASS
compute_80 PTX
```

运行在 SM80、SM86 或 SM90 时，CUDA driver 优先选择对应的 SASS，不需要现场编译。运行在没有匹配 SASS 的更新 GPU 上时，则可以选择 `compute_80` PTX，在加载阶段 JIT 成当前 GPU 的机器码。

`cuda.arch` 应当设置为计划支持的最低架构。例如 PTX target 是 `compute_90`，它不能反过来支持 SM80。当前实现也会忽略 `TORCH_CUDA_ARCH_LIST` 中低于 PTX target 的条目。

## 一个容易忽略的例子

假设编译机器是 SM120，设置了：

```python
inductor_configs={
    "aot_inductor.emit_multi_arch_kernel": True,
    "cuda.arch": "80",
}
```

但没有设置 `TORCH_CUDA_ARCH_LIST`。Inductor 会先在 SM100 上 autotune，选出配置，然后为了打包而把获胜配置重新编译到 SM80 target。最终 fatbin 里只有：

```text
sm_80 SASS
compute_80 PTX
```

它不会因为编译机器是 SM120，就自动把 SM100 SASS 放进最终 package。在 SM100 上加载这份产物时，CUDA driver 会从 `compute_80` PTX 做 JIT。这样通常可以运行，但会增加首次加载开销，也无法完整获得 Triton 面向 SM100 lowering 时可能使用的新架构优化。

如果既要支持 SM80，又希望 SM100 使用预编译的 SM100 SASS、避免运行时 JIT，就应该把 SM100 也加入 `TORCH_CUDA_ARCH_LIST`。具体架构名称需要以所用 CUDA Toolkit 支持的写法为准。不过这份 SASS 仍然来自低目标 PTX，并不等同于 Triton 直接面向 SM100 完成 lowering，后文会继续解释这个区别。

## 使用时还要注意什么

这项功能解决的是 NVIDIA GPU 之间的跨 SM 部署，不是 CUDA、ROCm 和 XPU 之间共享同一份二进制。跨厂商仍然需要分别编译对应 backend 的产物。

另外，生成额外架构的 SASS 时会使用 CUDA Toolkit 处理 Triton 生成的 PTX。系统 CUDA Toolkit 如果比 Triton 自带的 ptxas 太旧，可能无法识别对应 PTX 版本。因此在真正发布前，仍然应该在每一种目标 GPU 和 driver 组合上做加载、正确性和性能验证。

## Autotune 和跨架构打包不是一回事

这个新特性还有一个很重要的使用边界。假设编译机器是 SM120，但设置了 `cuda.arch="80"`：

```text
在 SM100 上编译、运行候选 kernel，并完成 autotune
                        ↓
             选出 SM100 上的最佳配置
                        ↓
      只把获胜配置重新编译到 SM80 target
                        ↓
          保存 sm_80 cubin 和 compute_80 PTX
```

因此，autotune 选出的 block size、num warps 和 num stages 等配置，反映的是它们在 SM100 上的表现；最终打包的 kernel 却已经按 SM80 做了 lowering，而且这个 SM80 版本没有重新 benchmark。它未必是 SM80 上的最佳配置，在其他 GPU 上也未必最优。

更容易误解的是，之后把这份 package 放回 SM120，CUDA driver 即使能从 `compute_80` PTX JIT 出 SM100 机器码，也不会重新执行 Triton 的高层 lowering。面向 SM80 生成的 PTX 里没有 TMA 等新架构路径，driver JIT 不会凭空把它们恢复出来。即便通过 `TORCH_CUDA_ARCH_LIST` 额外生成了 `sm_120` SASS，它仍然来自这份低目标 PTX，作用主要是避免运行时 JIT，而不是获得原生 SM100 Triton 编译的全部优化。

如果 SM100 上的原生性能很重要，最稳妥的方式仍然是为 SM100 单独编译和 autotune 一份产物。多架构 fatbin 更适合解决“同一份产物能够加载和运行”的兼容性问题，不能把它等同于“每一种架构都自动获得原生最优性能”。最终性能应该以各目标 GPU 上实际加载 package 后的 benchmark 为准。
