# Triton入门: 从零实现面向Decode场景的GQA Attention

上一篇文章我们用Triton实现了contiguous-batch的flash-attention，面向的是prefill阶段，Q和KV的序列长度都比较长的场景。这篇文章我们来聊一个不同的话题——GQA (Grouped Query Attention)，并且面向decode阶段，Q的序列长度为1的场景。

## 什么是GQA

标准的MHA (Multi-Head Attention)中，Q、K、V的head数量是一样的，例如都是64个head。而GQA的核心思想非常简单：**KV的head数量比Q少**，多个Q head共享同一组KV head。

| 类型 | Q head数 | KV head数 | 比例 | 代表模型 |
|------|---------|----------|------|---------|
| MHA | 64 | 64 | 1:1 | GPT-2, BERT |
| GQA | 64 | 4 | 16:1 | LLaMA-2/3, Qwen |
| MQA | 64 | 1 | 64:1 | PaLM, Falcon |

GQA的动机很直接：在推理阶段，KV Cache是显存的主要瓶颈，减少KV head数量可以线性降低KV Cache的大小，同时实验表明对模型质量的影响很小。

## Decode场景下的Attention长什么样

在推理的decode阶段，每次只生成一个token，因此Q的序列长度是1，而KV是之前所有token的缓存。维度上来说：

```
Q:  [batch, q_num_head, 1, d_model]        # seq_len_q = 1
K:  [batch, kv_num_head, kv_seq_len, d_model]   # kv_seq_len可能很长
V:  [batch, kv_num_head, kv_seq_len, d_model]
```

以LLaMA-2 70B为例，`q_num_head=64, kv_num_head=8, d_model=128`，group_size = 64/8 = 8，也就是每8个Q head共享一组KV。

这个场景有两个显著特征：
1. **Q的序列长度极短（=1）**，计算量小，是典型的memory-bound场景
2. **Q head数量远大于KV head数量**，存在数据复用的机会

## 在head维度上怎么起block

在MHA里面，一般在head维度上每个head起一个block，没什么好纠结的。但在GQA里面，Q和KV的head数量不同，就有了两种选择：

### 方案一：按Q的num_head起block
启动`q_num_head`个block，每个block负责一个Q head。每个block需要自己去读取对应的KV head的数据。

- 优点：并行度大（64个block vs 4个block）
- 缺点：同一组的多个Q head会重复读取相同的KV数据，增加global memory访问压力

### 方案二：按KV的num_head起block
启动`kv_num_head`个block，每个block负责一组KV head对应的所有Q head（即`group_size`个Q head）。

- 优点：KV数据只需要读取一次，组内复用
- 缺点：并行度低（只有4个block），单个block的计算量大

**那到底选哪个？**

在decode场景下（seq_q=1），单个Q head的计算量非常小，瓶颈在于KV的访存。如果按Q head起block，64个block都要去读同一份KV数据，对global memory的压力很大。

而按KV head起block，虽然并行度只有4，但每个block只需要读一次KV，然后在block内部让`group_size=16`个Q head共享这份KV数据。**对于decode这种memory-bound的场景，减少访存才是王道**。

所以接下来的写法里面选择了**按KV head起block**的方案。

## seq_q=1带来的维度问题

选定了方案之后，我们面临一个实际的工程问题：**Triton的`tl.dot`只支持2D矩阵乘法**。

一个block要处理的数据是：
```
Q:  [group_size, 1, d_model]      # group_size个Q head，每个seq_len=1
K:  [d_model, TILE_KV_S]          # 转置后的K
V:  [TILE_KV_S, d_model]
```

Q是3D的，`tl.dot`不认。如果强行调用`tl.dot(q[16,1,32], k[1,32,128])`，会报错：
```
'tt.dot' op expected the first dimension of the first operand 
to be equal to the first dimension of the result
```

解决方案很自然：既然seq_q=1，那就把这个维度squeeze掉就完事了

```
Q:  [group_size, d_model]         # 把group_size当作pseudo seq_len
K:  [d_model, TILE_KV_S]
V:  [TILE_KV_S, d_model]
```

这样所有矩阵乘法都是2D的：
- `Q @ K^T`: `[group_size, d_model] @ [d_model, TILE_KV_S]` → `[group_size, TILE_KV_S]`
- `P @ V`: `[group_size, TILE_KV_S] @ [TILE_KV_S, d_model]` → `[group_size, d_model]`

`group_size`充当了"伪序列长度"的角色。这个trick不仅解决了维度问题，还完美利用了TensorCore——`group_size`一般是8或16，刚好满足TensorCore对M维度的最小要求。

## 实现

### kernel签名
```Python
@triton.jit
def _gqa_attention_split_kv_head_triton(
    q_ptr, k_ptr, v_ptr, o_ptr, 
    stride_q_b, stride_q_h, stride_q_s, stride_q_d, 
    stride_k_b, stride_k_h, stride_k_s, stride_k_d, 
    stride_v_b, stride_v_h, stride_v_s, stride_v_d, 
    stride_o_b, stride_o_h, stride_o_s, stride_o_d, 
    kv_seq_len, 
    sm_scale: tl.constexpr, 
    d_model: tl.constexpr, 
    group_size: tl.constexpr,
    TILE_KV_S: tl.constexpr
):
```

Q K V O都是4D tensor `[batch, num_head, seq_len, d_model]`，因此需要传入四组stride。

`sm_scale`、`d_model`、`group_size`声明为`tl.constexpr`，编译期硬编码到kernel中。关于`tl.constexpr`的好处，可以参考[追寻PyTorch推理中Attention计算的最优解](https://zhuanlan.zhihu.com/p/2012166866475460344)中"更极致的静态优化"一节的讨论。

`TILE_KV_S`是在KV序列方向上的tiling大小，由于flash-attention算法需要在 kv 的序列维度上进行遍历，因此需要在这个方向上展开循环，TILE_KV_S即每次循环处理的数据个数

### grid规划

按`[kv_num_head, batch]`起block，每个block负责一个batch、一个KV head对应的全部Q head：

```Python
def grid(META):
    return (kv_num_head, batch)
```

非常简洁，没有sequence维度的tiling（因为Q的seq_len=1，不需要在Q序列方向划分block）。

### 指针偏移与降维

进入kernel后，首先获取block_id并计算基础指针：

```Python
batch_id = tl.program_id(1)
head_id = tl.program_id(0)

# Q和O需要偏移到当前KV head对应的第一个Q head
q_base_ptr = q_ptr + batch_id * stride_q_b + head_id * group_size * stride_q_h
o_base_ptr = o_ptr + batch_id * stride_o_b + head_id * group_size * stride_o_h

# KV直接偏移到当前KV head
k_base_ptr = k_ptr + batch_id * stride_k_b + head_id * stride_k_h 
v_base_ptr = v_ptr + batch_id * stride_v_b + head_id * stride_v_h
```

注意Q的偏移是`head_id * group_size * stride_q_h`——因为一个KV head对应`group_size`个Q head，所以需要跳过`head_id * group_size`个Q head。

### 加载Q并squeeze掉seq维度

这是整个kernel最关键的设计。我们不使用`tl.make_block_ptr`来加载Q，而是用手动的指针广播，直接构造一个`[group_size, d_model]`的2D数据块：

```Python
offset_d = tl.arange(0, d_model)
offset_q_h = tl.arange(0, group_size)
q = tl.load(
    q_base_ptr + offset_q_h[:, None] * stride_q_h + offset_d[None, :] * stride_q_d,
) # [group_size, d_model]
```

`offset_q_h[:, None] * stride_q_h`在head维度上遍历`group_size`个Q head，`offset_d[None, :] * stride_q_d`在d_model维度上遍历。seq维度的偏移是0（因为seq_len=1），直接省略了。

这样就把原本`[group_size, 1, d_model]`的3D数据，自然地加载为`[group_size, d_model]`的2D tensor。

### 构造KV的block指针

K需要转置（用于Q @ K^T），所以构造时交换shape和stride的维度：

```Python
k_block_ptr = tl.make_block_ptr(
    base = k_base_ptr,
    shape = (d_model, kv_seq_len),       # 转置后的shape
    strides = (stride_k_d, stride_k_s),  # 转置后的stride
    offsets = (0, 0),
    block_shape = (d_model, TILE_KV_S),
    order = (0, 1)
)

v_block_ptr = tl.make_block_ptr(
    base = v_base_ptr,
    shape = (kv_seq_len, d_model),
    strides = (stride_v_s, stride_v_d),
    offsets = (0, 0),
    block_shape = (TILE_KV_S, d_model),
    order = (1, 0)
)
```

与上一篇contiguous-batch的文章一样，K和V的`offsets`都从`(0, 0)`开始——这些是要被遍历的维度，而不是被tiling的维度。

### Online Softmax主循环

初始化统计变量后，进入KV序列方向的遍历循环：

```Python
m_i = tl.zeros([group_size], dtype=tl.float32) - float('inf')
l_i = tl.zeros([group_size], dtype=tl.float32)
acc = tl.zeros([group_size, d_model], dtype=tl.float32)

qk_scale = sm_scale * math.log2(math.e)
q = (q * qk_scale).to(tl.float16)

for start_kv_s in tl.range(0, kv_seq_len, TILE_KV_S):
    k = tl.load(k_block_ptr, boundary_check=(1,)).to(tl.float16)
    v = tl.load(v_block_ptr, boundary_check=(0,)).to(tl.float16)
    
    # [group_size, d_model] @ [d_model, TILE_KV_S] -> [group_size, TILE_KV_S]
    qk = tl.dot(q, k)

    m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])

    acc *= alpha[:, None]
    # [group_size, TILE_KV_S] @ [TILE_KV_S, d_model] -> [group_size, d_model]
    acc += tl.dot(p.to(tl.float16), v)

    l_i = l_i * alpha + tl.sum(p, axis=1)
    m_i = m_i_new

    k_block_ptr = tl.advance(k_block_ptr, (0, TILE_KV_S))
    v_block_ptr = tl.advance(v_block_ptr, (TILE_KV_S, 0))
```

这里和标准flash-attention的online-softmax完全一致，唯一的区别是统计变量`m_i`和`l_i`的shape是`[group_size]`而不是`[BLOCK_Q_S]`——因为`group_size`就是我们的"伪序列长度"。

注意这里没有causal mask——decode阶段Q的序列长度为1，每个Q token都可以attend到所有KV token，不需要casual。

### 写回结果

```Python
acc = acc / l_i[:, None]
tl.store(
    o_base_ptr + \
        tl.arange(0, group_size)[:, None] * stride_o_h + \
        offset_d[None, :] * stride_o_d,
    acc
)
```

写回时，同样沿着head维度展开`group_size`个Q head的结果。

## 封装接口

```Python
def call_gqa_attention_split_kv_head_triton(q, k, v):
    batch, q_num_head, seq_len_q, d_model = q.shape
    kv_seq_len = k.shape[2]
    kv_num_head = k.shape[1]

    def grid(META):
        return (kv_num_head, batch)

    o = torch.empty_like(q)
    torch.library.wrap_triton(_gqa_attention_split_kv_head_triton)[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        kv_seq_len, 1.0 / math.sqrt(d_model),
        d_model, q_num_head // kv_num_head
    )
    return o 
```

`torch.library.wrap_triton`是PyTorch 2.9引入的接口，可以将Triton kernel注册为PyTorch的自定义算子。

## Auto-Tuning

```Python
autotune_configs = [
    triton.Config({"TILE_KV_S": 128}, num_warps=2, num_stages=2),
    triton.Config({"TILE_KV_S": 64}, num_warps=2, num_stages=2),
    triton.Config({"TILE_KV_S": 32}, num_warps=2, num_stages=2),
    triton.Config({"TILE_KV_S": 128}, num_warps=4, num_stages=2),
    triton.Config({"TILE_KV_S": 64}, num_warps=4, num_stages=2),
    triton.Config({"TILE_KV_S": 32}, num_warps=4, num_stages=2),
]
```

只需要tune `TILE_KV_S`（KV序列方向的tile大小）和`num_warps`。`d_model`和`group_size`这些维度在一次推理中都是固定的，不需要进行tiling。

## 正确性验证

```
$ python GQA_Attention.py
[randn] Max diff: 0.000122, Mean diff: 0.000014
[randn % 3] Max diff: 0.000977, Mean diff: 0.000036
```

与PyTorch的`F.scaled_dot_product_attention(q, k, v, enable_gqa=True)`对比，randn数据下最大误差1.2e-4，平均误差1.4e-5；`randn % 3`（值域限制在`[0, 3)`）下最大误差约1e-3，平均误差3.6e-5。fp16下矩阵乘法的误差与随机种子和数据分布有关，每次运行结果会略有波动，但平均误差稳定在1e-5量级，完全在可接受范围内。

## 性能对比

测试配置：`q_num_head=64, kv_num_head=4, kv_seq_len=1024, d_model=32`

Q: `[batch, 64, 1, 32]`, K/V: `[batch, 4, 1024, 32]`

### NVIDIA PRO5000 (SM120)

| batch | Triton(us) | SDPA(us) | Speedup |
|------:|-----------:|---------:|--------:|
| 1 | 7.8 | 6.1 | 0.79x |
| 2 | 7.8 | 6.2 | 0.79x |
| 4 | 7.8 | 6.3 | 0.80x |
| 8 | 7.9 | 7.7 | 0.97x |
| 16 | 8.1 | 9.8 | 1.21x |
| 32 | 10.0 | 14.5 | 1.45x |
| 64 | 11.3 | 28.3 | 2.50x |
| 128 | 21.1 | 45.8 | 2.17x |
| 256 | 111.3 | 118.6 | 1.07x |
| 512 | 223.7 | 235.1 | 1.05x |
| 1024 | 445.5 | 458.4 | 1.03x |
| 2048 | 889.5 | 902.6 | 1.01x |

在NVIDIA上，小batch（1~8）时SDPA更快，中等batch（16~128）时Triton有1.2x~2.5x的加速优势，大batch（256+）时两者趋于一致。

### AMD MI308X (ROCm)

| batch | Triton(us) | SDPA(us) | Speedup |
|------:|-----------:|---------:|--------:|
| 1 | 20.0 | 46.8 | 2.34x |
| 2 | 20.0 | 69.8 | 3.49x |
| 4 | 20.2 | 132.8 | 6.59x |
| 8 | 20.2 | 226.6 | 11.20x |
| 16 | 20.6 | 412.6 | 20.07x |
| 32 | 29.3 | 813.5 | 27.75x |
| 64 | 54.2 | 1986.7 | 36.63x |
| 128 | 91.3 | 3190.4 | 34.95x |
| 256 | 164.9 | 6341.7 | 38.46x |
| 512 | 325.8 | 12660.8 | 38.86x |
| 1024 | 651.8 | 25301.7 | 38.82x |
| 2048 | 1288.2 | 50594.4 | 39.28x |

AMD上SDPA在GQA场景下的性能非常差，Triton有近**39倍**的加速。具体根因尚未深入分析。

### 测时方法

性能对比使用`torch.profiler`采集kernel级别的device耗时，排除了host端的overhead：

```Python
def extract_kernel_time_us(prof):
    total = 0.0
    for evt in prof.key_averages():
        if evt.device_time_total > 0:
            total += evt.device_time_total
    return total

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
) as prof:
    for _ in range(repeat):
        call_gqa_attention_split_kv_head_triton(q, k, v)
    torch.cuda.synchronize()
kernel_time_us = extract_kernel_time_us(prof) / repeat
```

同时导出了chrome trace文件，可以在`chrome://tracing`中打开进行详细的timeline分析。

## 总结

| | 优势 | 适用场景 |
|---|---|---|
| 按Q head起block | 并行度高 | prefill阶段，计算密集 |
| 按KV head起block | KV访存少，组内复用 | decode阶段，访存密集 |

我这里实现的是后者。核心trick是将seq_q=1 squeeze掉，用`group_size`作为矩阵乘法的M维度，这样既解决了`tl.dot`不支持3D的问题，又天然利用了TensorCore的算力。

完整代码放在了 [GitHub](https://github.com/sujuyu/GQA_attention_decode_triton)。
