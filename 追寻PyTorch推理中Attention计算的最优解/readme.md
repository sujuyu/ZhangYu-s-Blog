# 追寻PyTorch推理中Attention计算的最优解

这篇文章的标题取得比较大, 其实应该加一个限定条件——网络需要能够使用图来表示. 
也就是说网络能够被TorchDynamo trace成一张fx graph的情况, 一般来说这个条件都是满足的. 

## 专门的attention算子
如果手搓attention的实现, 在实际执行的时候会对应到aten层面的多个算子, matmul, add, softmax, div等, 对于attention这种性能热点算子, 显然是不划算的. 
于是PyTorch为了加速Attention的计算, 专门有一个SDPA算子

```Python
import torch.nn.functional as F
F.scaled_dot_product_attention(q, k, v, attn_mask = attn_mask)
```

如果对这点不熟悉的同学可以参考这篇[PyTorch里面怎么做维度设计才能让attention跑得更快](https://zhuanlan.zhihu.com/p/1951418724851619024)

scaled_dot_product_attention在内部会根据实际输入的情况, dispatch到不同的attention实现上, 例如flash-attention、efficient-attention或者兜底的math-attention等. 而在具体的某一种实现内部, 也会根据输入的shape情况等, 选择最优的一组grid-block并行config

而PyTorch自身也在不断迭代SDPA的实现, 例如在23年9月份的时候, 官方把flahs-attentionv2搬进去了

所以在相当长的一段时间内, 我一直认为使用PyTorch进行训练和推理的搜推广或者视觉模型里面, attention操作默认使用PyTorch官方的原生实现是肉眼所及的最优解. 除非是对一些维度特殊的Attention, 例如batch_size和seq_len相差悬殊的Target-Attention, 重写一遍特定tile的Attention会有收益之外, 正常我们重写Attention操作的收益不会很明显

## 在Inductor中里面会发生变化吗
PyTorch在2.0之后推出了torch.compile特性对网络进行整体的jit加速, torch.compile后端依赖于TorchInductor. 而RTP-Torch的部署链路依赖的AOTInductor本质上也是在TorchInductor之上包了一层, 提供了在C++中被方便调用的特性. 

对Inductor不熟悉的同学可以参考[TorchInductor设计](https://zhuanlan.zhihu.com/p/1960385575526852600?share_code=qiyE5ErT8up2&utm_psn=1960825411299697324), 想要了解AOTInductor的同学可以参考[基于计算图捕获和AOT预编译技术的Pytorch模型部署链路](https://zhuanlan.zhihu.com/p/707694134), [讨论一下AOTInductor在C++和Python下的并发设计](https://zhuanlan.zhihu.com/p/1933245343149032668)

一张被TorchDynamo捕获的fx graph经过TorchInductor的处理过程可以概括为4步. 
1. decomposition
    1. 将一些复杂的或者功能重复算子进行分解, 成为一组小op的组合
    2. 例如 resize_as操作会被分解为 torch.contiguous_format + aten.resize, inplace的 aten.addmm_ 操作会被分解为 aten.addmm + aten.copy_
    3. 分解的主要作用是将需要支持的算子总数控制在200个上下, 否则需要对aten.addmm_和aten.addmm都写一遍IR转换的代码, 尽管两者的功能几乎完全一致, 只有inplace和outplace的区别
2. lowering
    1. 对于常用算子, PyTorch都注册了对应的lowering函数, 表示输出的每一个元素是如何从输入中运算获取的, 例如对于clone操作, 他的lowering函数是
 ```python
        @register_lowering(aten.clone)
        def clone(x, *, memory_format=None):
            # TODO(jansel): memory format
            return Pointwise.create(device=x.get_device(),
                dtype=x.get_dtype(),
                inner_fn=x.make_loader(),
                ranges=list(x.get_size()),
            )
```
    2. 知道一个op的lowering函数之后, 就可以转换为Inductor的IR表示, 这个IR可以简单理解为"op结果的每一个元素是怎么获取的"
3. IR层面的op融合
    1. 知道了每一个op的IR表示之后, 对于存在边的op之间, 会检测两者是否可以fuse
    2. 例如存在边的aten.add和aten.div操作, 可以直接使用一个运算逻辑是ops.add + ops.div的融合IR进行表示. 当然实际情况要复杂一点, 对于这种简单的连续element-wise操作, 其实在生成IR的阶段就融合好了
    3. 一遍一遍循环, 直到整张图没办法被继续融合
4. 生成对应的Triton代码
    1. 对于每一个IR表示, 生成对应的Triton核函数

如果是torch.compile调用, 到这里就结束了. 如果是AOTInductor, 则还会多一个生成C++代码, 调用系统gcc/clang进行编译的过程. 

根据上面所述, 对于每一个上层的PyTorch操作, 都会被分解到一个受支持的op子集上, 然后转换为IR, 再然后生成Triton代码. 那么在inductor环境下, 最终不会有scaled_dot_peoduct_attention调用, 因为一切op都被转换为了Triton核函数. 

事实真的是这样吗？很明显当然不会
1. 一方面如果在decomposition阶段将SDPA算子分解为matmul+div+softmax+add操作, 那么会失去融合的性能优势
2. 另一方面SDPA运算复杂, 很难写出一个对应的lowering函数. 
因此TorchInductor设置了一个[fallback ops](https://github.com/pytorch/pytorch/blob/main/torchgen/aoti/fallback_ops.py)列表, 对于在其中的算子, 会直接调用PyTorch的实现, 不再生成Triton, SDPA也在其列. 

所以, 即便有inductor的存在, scaled_dot_product_attention还是会调用PyTorch自身的kernel实现, 不影响本文的讨论. 

## 我们能超越原生的实现吗
PyTorch的scaled_dot_product_attention实现, 本身是非常高效的, 而且对于不同的输入case内部还会使用不同的算法. 

因此很长一段时间内我觉得绝大多数情况下, 我们自己用Triton重写一遍SDPA算子意义不大, Inductor的作者或许也是同样的想法, 因此在Inductor中继续使用了PyTorch原生的实现. 

不过我最近有了一些特别的想法, 概括来说就是:
1. 在代码实现层面, 我们用triton写的attention算子, 大概率击败不了使用CUDA等更low-level语言实现的SDPA算子
2. 但是我们比SDPA算子的作者, 更加了解我们的网络模型, 也就是说我们掌握更多的"**先验知识**"

#### 更极致的运行时finetune
SDPA的作者可不知道我们的输入维度是多少, 所以只能尽量枚举输入的情况, 使用if-else来选择预先知道的最佳config, 伪代码可能长这样(Gemini写的):
```C++
// 伪代码
Config get_best_config(int M, int N, int K, GPU_Arch arch) {
    if (arch == "Ampere" && M <= 256 && N <= 256) {
        // 对小矩阵, 使用专门优化的配置, 可能用更小的block tile
        return {grid(calc_grid_A), block(256), smem(size_A)};
    } else if (K % 8 != 0) {
        // 如果K维度没有对齐, 不能用Tensor Core, 回退到普通的CUDA Core Kernel
        return {grid(calc_grid_B), block(128), smem(size_B)};
    } else {
        // 默认的、适用于大多数情况的配置
        return {grid(calc_grid_C), block(256), smem(size_C)};
    }
}
```
写下C++代码的时候并不能枚举出将来需要遇到的全部case, 但是对于一张输入维度已经固定的fx graph而言, 输入的维度却是已知的, 借助Triton提供的Auto-Tuning能力, 可以把最佳配置的选择推迟到运行时(如果是AOTInductor的话则是编译期)
```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q_S": 16, "BLOCK_KV_S": 16}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q_S": 16, "BLOCK_KV_S": 16}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_Q_S": 32, "BLOCK_KV_S": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_Q_S": 32, "BLOCK_KV_S": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_Q_S": 16, "BLOCK_KV_S": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_Q_S": 16, "BLOCK_KV_S": 64}, num_warps=4, num_stages=4),
    ],
    key=["q_seq_len", "kv_seq_len", "d_model"],
)
```
通过尽可能把更多编译期的参数放在key中, 同时枚举足够多的config, 我们总能比SDPA更接近这个维度下的最佳配置. 而代价仅仅是在第一次运行的时候, 多消耗一些编译时间而已. 

#### 更极致的静态优化
已一个Attention算子为例, 假设输入的QKV的维度都是[batch, num_head, sequence_length, d_model]

对于一个网络设计来说, 在写好模型定义的时候, 更准确说是被dynamo捕获成一张fx graph的时候, num_head, sequence_length, d_model就已经确定了, 一般而言只有batch维度是可变的(语言模型里面sequence维度也是可变的, 但是搜推模型里面一般是填充到max_sequence_length)

并且一般而言, QKV三个Tensor在head, sequence, d_model三个维度上的stride也是一个常量

但是写PyTorch SDPA算子的人可不知道你的网络里面这些常量维度是多少, stride是多少. 所以SDPA的实现里面这些都是外部传参. 体现在核函数上可能长这样:
```C++
__global__ void attention_forward_kernel_templated(
    // ---------- 数据指针 ----------
    const float* q_ptr,
    const float* k_ptr,
    const float* v_ptr,
    float* o_ptr,

    // ---------- 维度信息 (运行时可变) ----------
    const int batch_size,
    const int num_heads,
    const int seq_len_q,
    const int seq_len_kv,
    
    // ---------- Stride 信息 (运行时可变) ----------
    const int stride_q_b,
    const int stride_q_h,
    const int stride_q_s,
```
然后在函数的执行过程中, 去取每一个QKV的stride, 然后H2D传输到device, 函数内部再去取值. 

既然我们已知这些参数是一个编译期的常量, 在写Triton算子的时候, 直接定义为tl.constexpr即可, 例如一个flash-attention核函数的接口定义为: 
```Python
@triton.jit
def flash_attention_forward(
    q_ptr, k_ptr, v_ptr, o_ptr, attn_mask_ptr,
    batch, 
    num_head: tl.constexpr,
    seq_len_q: tl.constexpr,
    seq_len_kv: tl.constexpr,
    d_model: tl.constexpr,

    stride_q_b, stride_k_b, stride_v_b, stride_o_b, stride_attn_mask_b,

    stride_q_h: tl.constexpr,
    stride_k_h: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_attn_mask_h: tl.constexpr,

    stride_q_s: tl.constexpr,
    stride_k_s: tl.constexpr,
    stride_v_s: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_attn_mask_q_s: tl.constexpr,
    stride_attn_mask_kv_s: tl.constexpr,

    stride_q_d: tl.constexpr,
    stride_k_d: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_o_d: tl.constexpr
)
```
tl.constexpr类型的传参在编译期就被硬编码到了函数内部, 当然这种做法也不是没有副作用——同一个Triton函数, 但是tl.constexpr类型的参数不一样时, 函数会被分别编译, 导致Jit的时间变长. 


#### 简化输入检查
PyTorch的SDPA实现里面对于输入会有额外的检查, 例如需要检查dtype, device, shape等, 在一张已经捕获的fx graph中, 一般只有batch-size维度可变, dtype, device等都是固定的, 没必要在每一次执行的时候都去做一遍检查. 

## 实际效果
我选取了多个当前实际上线或者实验中的搜推业务, 摘取了其中的attention维度, 在fp16下进行实验. 

对于flash-attention, 我有两版实现: 
1. flash_attention_v2
    - 常规的flash-attention实现
    - d_model维度上不进行split, 每个Batch和Head维度上单独起block, Q的序列维度上进行划分Block
2. short_query_attention_v1 & short_query_attention_v2
    - 两阶段的flash-attention, 在KV的序列维度上进行切分, 启动第二个kernel进行规约, 主要面向KV序列很长但是Q序列长度较短的场景
    - v1和v2仅在KV序列的切分上存在区别
    - 具体思想参考[搜推场景下的Target-Attention加速策略](https://zhuanlan.zhihu.com/p/1939818412491641816)中关于Target-Attention的讨论

测试环境是NVIDIA A10, 循环100次取均值
测试过程使用AOTInductor尽量消除python造成的overhead, 但是依旧包含kernel launch的开销, 所以真实差距会比测试结果要更大

| q Dimensions      | k Dimensions      | v Dimensions      | attn_mask Dimensions  | original_sdpa (ms) | flash_attention_v2 (ms) | short_query_attention_v1 (ms) | short_query_attention_v2 (ms) | _scaled_dot_product_attention_math (ms) |
|-------------------|-------------------|-------------------|-----------------------|--------------------|--------------------------|--------------------------------|--------------------------------|-----------------------------------------|
| [1, 8, 1, 64]     | [1, 8, 890, 64]  | [1, 8, 890, 64]  | [1, 1, 1, 890]       | 0.10178           | 0.05020                 | 0.05977                        | 0.06131                        | 0.08167                                  |
| [1, 8, 890, 64]   | [1, 8, 890, 64]  | [1, 8, 890, 64]  | [1, 1, 1, 890]       | 0.12535           | 0.11147                 | 0.12102                        | 0.14464                        | 0.36629                                  |
| [1, 4, 20, 64]    | [1, 4, 2099, 64] | [1, 4, 2099, 64] | [1, 4, 1, 2099]      | 0.17835           | 0.06373                 | 0.06039                        | 0.05993                        | 0.13946                                  |
| [1, 4, 20, 64]    | [1, 4, 20, 64]   | [1, 4, 20, 64]   | [1, 4, 20, 20]       | 0.07551           | 0.05127                 | 0.06005                        | 0.06002                        | 0.07721                                  |
| [1, 1, 50, 256]   | [1, 1, 10000, 256] | [1, 1, 10000, 256] | [1, 1, 50, 10000] | 0.79547           | 0.63372                 | 0.11615                        | 0.09463                        | 0.20302                                  |
| [1, 8, 1, 128]    | [1, 8, 50, 128]  | [1, 8, 50, 128]  | [1, 1, 1, 50]        | 0.07616           | 0.05040                 | 0.06164                        | 0.07173                        | 0.07739                                  |
| [9, 8, 1, 128]    | [9, 8, 200, 128] | [9, 8, 200, 128] | [1, 1, 1, 200]       | 0.06940           | 0.05107                 | 0.06037                        | 0.07162                        | 0.11531                                  |

所以, 只要给的先验条件足够多, 并且Auto-Tuning的空间足够大, 我们总是能比PyTorch SDPA更快一点点.