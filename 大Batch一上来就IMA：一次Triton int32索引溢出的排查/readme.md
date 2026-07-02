## 大 Batch 一上来就 IMA：一次 Triton int32 索引溢出的排查

最近在排查一个线上压测里的 GPU illegal memory access 问题.

先说明一下版本背景: 这次线上出问题的编译链路基于 PyTorch 2.9, 也就是
AOTInductor 生成的 Triton BMM template 里还没有把相关 index 统一提升到 int64.

我们线上实际运行的不是 Python eager 模型, 而是把 PyTorch 模型先导出成 FX Graph,
再通过 AOTInductor 编译出来的产物文件, 即 `model.pt2`. 可以简单理解为: FX Graph
是编译前能读懂的计算图, `model.pt2` 是线上真正跑的编译产物. 这个产物里面已经
不是原始 PyTorch op, 而是经过 lowering、fuse 之后 codegen 出来的 C++ wrapper
和各种 Triton kernel.

这个问题比较坑爹. 从编译前的 FX Graph 看, 没有明显的非法 index; 从输入看,
也没有奇怪的 NaN/Inf. 但同一份模型小 batch 跑得好好的, batch 一大线上就开始
illegal memory access.

最后发现问题藏在 Triton kernel 的地址计算里: `tl.program_id` 默认是 int32,
`tl.arange` 默认也是 int32. GPU 上的地址当然是 64 位的, 但如果 offset 在加到
pointer 之前已经用 int32 算溢出了, 后面再接到 64 位地址上也救不回来.

---

### 背景: FX Graph 编辑 + AOTInductor codegen

更具体一点, 我们线上这条链路大概是这样的:

1. 算法侧交付 PyTorch 模型.
2. 通过 `torch.export` 拿到 FX Graph.
3. 在 FX Graph 上做一些图编辑, 比如替换算子, 插入自定义 op, 做精度或性能相关的 pass.
4. 使用 AOTInductor 编译成 `model.pt2`.
5. 在线上 C++ 环境里通过 `AOTIModelPackageLoader` 加载执行.

`model.pt2` 本质是一个 zip 包. 解压之后里面能看到 AOTInductor 生成的 C++ wrapper,
也能在 wrapper 的注释里看到 Triton kernel 的源码. 这点很重要, 因为这次问题最后
就是靠解包之后反查 kernel 才定位到的.

压测时我们发现模型偶发 IMA. 一开始还没意识到它和 batch size 有关, 只知道某些
请求会把 GPU 打到不可恢复状态.

我第一反应当然是把出问题的输入保存下来. 于是很自然地在 catch 里对 GPU tensor
做保存, 结果落下来的文件只有 798B. 这个大小一看就不对, 更像是只写了个文件头,
真正的 tensor 数据根本没存进去.

后来想了一下也合理: IMA 发生之后, 当前进程里的 GPU context 基本已经污染了.
这个时候再指望 `torch.save(gpu_tensor)` 去读 GPU 上的数据, 本身就不靠谱.

所以最后换了一个更简单粗暴的办法: 在 forward 之前先把输入备份一份到 CPU.
如果 forward 过程中炸了, 就保存 CPU 上这份输入. CPU tensor 不依赖已经挂掉的
GPU context, 至少能把现场留下来.

伪代码大概是这样:

```cpp
std::vector<at::Tensor> cpuBackup;
for (const auto& t : realInputs) {
    cpuBackup.push_back(t.to(torch::kCPU));
}

try {
    outputs = aotiModel.run(realInputs);
} catch (...) {
    TensorUtils::saveToTempFile(cpuBackup, "crash_inputs_cpu");
    throw;
}
```

这个细节很朴素, 但对于排查 GPU IMA 很关键. 没有这份 CPU 输入, 后面基本就是盲猜.

---

### 第一反应: 是不是 FX Graph 里有非法索引?

拿到输入之后, 我们第一反应还是怀疑图里有某个索引越界. 这也很自然, 因为 IMA
听起来最像:

1. `index_select` 的 index 超范围.
2. `gather` 的 index 超范围.
3. 某个动态 shape 下 mask 没挡住非法地址.

于是先回到 FX Graph 上看, 重点检查所有 `index_select` 和 `gather`.

图里确实有一些索引操作, 但看起来没有特别离谱的地方. 为了排除它们, 我们干脆在
这些可能越界的位置前面插入 `clamp`, 把 index 强制限制在合法范围内.

结果: 还是 IMA.

这一步虽然没有解决问题, 但方向上很有帮助. 看起来不是模型写法里某个显式索引
直接越界, 至少不是那种一眼能在 FX Graph 里抓出来的问题. 真实的问题大概率藏得
更深, 得继续往 AOTInductor 生成的产物里看.

---

### 线下复现: 小 batch 居然不炸

有了 CPU 输入之后, 我们在线下把整条 AOTI forward 跑起来, IMA 可以稳定复现.

然后出现了第一个关键信号: 如果把 batch size 改小, 它就不炸了.

这件事一开始有点反直觉. 因为我们之前检查过:

1. 输入没有 NaN/Inf.
2. gather/index_select 的 index 都在合法范围内.
3. fp16/fp32, autocast 开不开, 都不是决定因素.
4. FX Graph 版本在 fp32 下可以正常跑, IMA 主要出现在 AOTInductor 产物里.

继续二分 batch size, 现象就很清楚了:

```text
s10=1000: 成功
s10=1372: 成功
s10=1373: IMA
s10=1374: IMA
```

到这里我们只是知道它和 batch size 强相关, 但还不知道到底是哪一个 op 在炸.
AOTInductor 产物里面有一大堆 codegen 出来的 Triton kernel, 不可能靠肉眼在
FX Graph 里直接猜中. 下一步就得让 coredump 告诉我们, IMA 当时到底死在哪个
kernel 上.

---

### 解包 pt2 + debug-agent: crash kernel 指向一个 BMM

接下来直接解包 `user_model.pt2`:

```bash
unzip -q user_model.pt2 \
  'user_model/data/aotinductor/model/*.cpp' \
  'user_model/data/aotinductor/model/*.json' \
  -d /tmp/aoti_model
```

在 AMD 上可以加 debug-agent, 让 coredump 信息更具体一点:

```bash
export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
export ROCM_DEBUG_AGENT_OPTIONS=-p
export HSA_ENABLE_DEBUG=1
export HIP_LAUNCH_BLOCKING=1
```

这次 debug-agent 指向的 kernel 是:

```text
triton_tem_fused__to_copy__unsafe_view_add_addmm_bmm_clone_mean_mul_pow_rsqrt_transpose_view_139
```

名字非常长, 但里面最关键的是 `bmm`. 它对应 target attention 里的 `Q @ K^T`:

```text
Q: [8*s10, 84, 32]
K: [8*s10, 32, 2328]
O: [8*s10, 84, 2328]
```

这里的 `8` 是 head 数量, `32` 是 head dim. 也就是说, AOTInductor 把
`[s10, 8, ...]` 的 batch/head 合并成了 BMM 的 batch 维.

知道 crash kernel 是这个 BMM 之后, 再回头看刚才 `s10=1372/1373` 的阈值,
味道就出来了:

```text
输出 shape = [8*s10, 84, 2328]
单个 batch/head 的矩阵大小 = 84 * 2328 = 195552 elements

s10=1372: 8*1372*84*2328 = 2,146,378,752 elements
s10=1373: 8*1373*84*2328 = 2,147,943,168 elements
```

注意 `int32_t` 最大值是:

```text
2,147,483,647
```

好家伙, `s10=1373` 刚刚越过 `int32_t` 的正数上界.

再看 wrapper 里 kernel 的源码注释, 问题已经呼之欲出:

```python
pid = tl.program_id(0)
...
rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
...
idx_q = tl.program_id(1)  # batch dimension for BMM
...
idx_m = rm[:, None]
idx_n = rn[None, :]

xindex = idx_n + 2328 * idx_m + 195552 * idx_q
tl.store(out_ptr0 + tl.broadcast_to(xindex, acc.shape), acc, mask)
```

`195552 = 84 * 2328`, 也就是每个 BMM batch 的输出元素数.

当 `idx_q` 逐渐增大时, `195552 * idx_q` 很快就接近 `int32_t` 上界.
在 `s10=1373` 时, `idx_q` 最大是 `8*s10-1 = 10983`, 最大 offset 约等于:

```text
195552 * 10983 + 2328 * 83 + 2327 = 2,147,943,167
```

这个值已经超过 `2,147,483,647`.

于是事情就很清楚了: 不是 base pointer 不是 64 位, 而是 `xindex` 这个 offset
表达式在中间计算时走了 int32, 溢出了.

这里还有个容易误导人的细节: 生成出来的 kernel 里能看到类似
`INDEX_DTYPE : tl.constexpr = tl.int64` 的常量, 但这不等于 `tl.program_id`
参与的每个中间表达式都已经是 int64. 还是要看真正的 `pid`, `idx_q`,
`xindex` 这些变量是怎么一路推导出来的.

---

### 为什么 mask 没救它?

Triton kernel 里通常都会有 mask:

```python
mask = (idx_m < M) & (idx_n < N)
tl.store(ptr + offset, value, mask)
```

但是 mask 只能挡住逻辑 shape 外的访问. 这次的问题是, 从逻辑上看:

```text
idx_m < 84
idx_n < 2328
idx_q < 8*s10
```

这些条件全部成立.

错的是 `offset = idx_n + 2328*idx_m + 195552*idx_q` 的整数类型. 逻辑 index
合法, offset 算炸, mask 自然也不知道.

这也是这个 bug 隐蔽的地方. 你在 FX Graph 里看不出任何异常, 在高层 shape
推导里也看不出任何越界, 甚至单看 Triton 源码里的 mask 也会觉得挺正常.

---

### 最小修复: 把参与地址计算的 pid cast 到 int64

我们把这个 BMM kernel 单独抽出来做最小复现, 然后改一行:

```python
pid = tl.program_id(0).to(tl.int64)
idx_q = tl.program_id(1).to(tl.int64)
```

再跑 `s10=1373`, IMA 消失.

这里还有一个小细节: `tl.arange` 本身通常是 int32, 但如果一端操作数已经是 int64,
表达式会被提升到 int64. 所以在这个 kernel 里, 不一定要把每个 `tl.arange`
都显式 `.to(tl.int64)`. 真正关键的是所有参与大 offset 的 `tl.program_id`
不能保持 int32.

不过我更倾向于把规则写得保守一点:

> 只要 `pid` 会参与全局内存地址计算, 尤其是 `pid * stride` 这种 stride 可能很大的表达式,
> 就显式 `.to(tl.int64)`.

例如:

```python
pid = tl.program_id(0).to(tl.int64)
pid_b = tl.program_id(1).to(tl.int64)

offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs = offs_n + stride_m * offs_m + stride_b * pid_b
```

如果只 cast `pid = tl.program_id(0)`, 但真正把 offset 撑爆的是
`idx_q = tl.program_id(1)`, 那还是会炸. 这点我们也验证过.

---

### GitHub 上不是第一次有人踩

后来去 GitHub 上搜了一下, 果然有人很早就提过类似问题.

Triton issue [#832](https://github.com/triton-lang/triton/issues/832) 的标题就非常直白:

```text
Index computations are done in int32 even for large tensors, and overflow causing IMA
```

另一个 issue [#1058](https://github.com/openai/triton/issues/1058) 里也有人提到,
把:

```python
pid = tl.program_id(axis=0)
```

改成:

```python
pid = tl.program_id(axis=0).to(tl.int64)
```

问题就消失了.

看到这里, 心情就比较复杂. 一方面说明我们没有抓到什么玄学问题, 另一方面也说明
这个坑确实足够隐蔽, 隐蔽到几年后依然可能在真实模型里踩到.

---

### 性能会不会变差?

直觉上, pointer 本身就是 64 位, 所以把 offset 提升到 int64 不应该带来灾难性的
性能损失. 但也不能说绝对零成本, 因为 64 位整数中间值可能增加寄存器使用量,
也可能影响某些 kernel 的 occupancy.

所以这个事情不用过度纠结. 64 位 offset 可能会多占一点寄存器, 极端情况下影响一点
occupancy, 但比起一点点性能损失, 线上推理肯定还是稳定性优先. IMA 一旦触发,
就不是慢一点的问题了.

更麻烦的是, 这不是某一个模型的偶然问题. 只要模型里出现超过 4GiB 级别的 tensor,
又刚好被 AOTInductor codegen 成 Triton kernel, 地址计算里还保留着 int32 offset,
就有机会踩到类似的坑. 换句话说, 大 tensor + Triton codegen + int32 offset,
这个组合本身就需要警惕.

---

### 最后总结一下

1. GPU pointer 是 64 位, 不代表 Triton 里的 offset 表达式自动是 64 位.
2. 大 tensor 下的 IMA 不一定是业务 index 越界, 也可能是 `pid * stride` 的 int32 溢出.

以后如果看到大 batch 才稳定触发的 IMA, 尤其阈值卡在 `2^31` elements 附近,
不要只盯着 `gather` 和 `index_select`. 顺手把 AOTInductor 产物解开, 搜一下
crash kernel 里的 `tl.program_id`, 看看有没有哪个 `pid * stride` 正悄悄越过
`int32_t` 的边界.

补充一句: 我后来简单看了一下 PyTorch 2.12 分支, 这条 AOTInductor BMM template
路径已经明显修过了. 2.12 里的 `triton_bmm.py.jinja` 会把 `pid`,
`idx_q` 以及相关 `tl.arange` cast 到 `INDEX_DTYPE`; 而 `INDEX_DTYPE` 会根据
输出 numel 和相关 buffer storage size 判断, 大 tensor 下会走 `tl.int64`.

所以这次 PyTorch 2.9 上踩到的问题, 在 PyTorch 2.12 的 AOTInductor BMM/template
路径上大概率已经修复. 但这不等于所有 Triton kernel 都自动安全, 手写 Triton、
自定义 template 或其它特殊 codegen 路径, 看到 4GiB 级别的大 tensor 时还是要
顺手检查一下 offset 类型.
