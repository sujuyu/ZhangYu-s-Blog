## 某CTR业务AMP混合精度误差排查实战

该业务是首猜的一个重要打分模型，使用RTP-Torch进行部署，模型通过`torch.export`导出为FX Graph后，使用`torch.amp.autocast(fp16)`进行混合精度下的PyTorch Inductor编译。上线前的精度校验中发现，AMP fp16的输出与fp32相比存在不可接受的误差。

这个模型有个特点：输出经过sigmoid之后，分值均值只有1e-2量级（大部分样本的预测概率很低）。这意味着哪怕绝对误差只有1e-3，相对误差也会飙到~10%，对打分排序的影响是不可接受的。

整个排查过程分为两个阶段，像剥洋葱一样，修好一层露出下一层：
1. 第一阶段：误差 ~0.5，完全不可用 → 定位到 batch_norm 的 eps 问题
2. 第二阶段：误差 ~0.001，相对误差 ~10% → 定位到关键路径 linear 的 fp16 精度问题

---

### 第一阶段：batch_norm eps 导致的巨大误差

#### 发现问题

最初开启AMP fp16后，输出和fp32对比，误差直接到了0.5量级——模型输出完全不可用，sigmoid之后的概率值面目全非。

直觉上怀疑是某些op在fp16下数值不稳定。通过`torch.fx.interpreter.Interpreter`逐节点执行并收集中间值，我们检查了模型中所有6个batch_norm的`running_var`：

```python
# batch_norm args: input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled
for node in graph_module.graph.nodes:
    if node.target == torch.ops.aten.batch_norm.default:
        var_node = node.args[4]  # running_var
        eps = node.args[7]
        # 检查 var 的最小值是否小于 eps
```

结果一目了然——6个BN中，第3个和第6个的`running_var`非常小：

```
BN Node                  eps      var_min     var_mean      var_max   var<eps count
===================================================================================
batch_norm               0.001    0.281394    2.035522    59.780880       0/1024
batch_norm_1             0.001    0.109407    0.676400    20.920765       0/512
batch_norm_2             0.010    0.000478    0.003083     0.027805     240/256   <--- DANGER!
batch_norm_3             0.001    0.006409    0.489867    31.881142       0/512
batch_norm_4             0.001    0.007030    0.262783     3.447351       0/256
batch_norm_5             0.010    0.000221    0.020712     0.148691      64/128   <--- DANGER!
```

（注：表中的eps已经是修复后的值0.01，修复前都是0.001）

batch_norm_2的256个维度中有240个的方差小于eps，batch_norm_5的128个维度中有64个方差小于eps。方差最小值只有0.0002~0.0005，比eps=0.001还小。

#### 为什么 var 小会炸？

batch_norm的核心计算是：

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

当$\sigma^2$只有0.0005，$\epsilon$也只有0.001时，分母$\sqrt{0.0005 + 0.001} \approx 0.039$。

fp16的精度大约是1e-3量级，分子$x - \mu$本身就带着fp16的量化噪声，再除以一个0.039的分母，噪声直接被放大25倍。更要命的是，如果var比eps还小，那这些维度的归一化结果几乎完全由数值噪声决定了。

#### 修复

很直接——把这两个BN层的eps从0.001增大到0.01，让分母更大、数值更稳定。修复前后对比：

```
Metric              eps=0.001 (before)    eps=0.01 (after)     Improvement
---------------------------------------------------------------------------
max_diff                      0.017759            0.010487            1.7x
mean_diff                     0.004263            0.002582            1.7x
rel_err                          1.89%               1.14%            1.7x
```

误差下降了接近一倍。但注意，这里展示的是当前模型（已修复eps）还原eps=0.001后的对比。在最初排查时，由于还有其他因素叠加，原始误差远大于这里的数字，实际体感是从"完全不可用"变成了"基本能用但精度不够"。

---

### 第二阶段：关键路径 linear 的 fp16 精度问题

BN eps修复后，残余误差的mean_diff约0.0026，相对误差约1.14%。看起来不大，但对于输出均值只有0.01~0.02的样本来说，相对误差依然可以到10%以上，还是不够。

#### 首先排除 batch_norm

第二阶段一开始还是怀疑batch_norm。实测发现`aten.batch_norm.default`在autocast下虽然输入/输出是fp16，但内部计算走fp32（running_mean/var/weight/bias都是fp32参数），单独把batch_norm强制fp32执行，最终误差几乎不变。排除。

#### 逐节点单步误差分析：谁在产生误差？

第一个关键实验：对每个op，喂fp32输入，分别在fp32和autocast fp16下执行，比较输出差异。回答的问题是：**每个op自身引入了多少误差？**

```python
# fp32 执行
result_fp32_node = node.target(*fp32_args, **fp32_kwargs)

# autocast fp16 执行（同样用 fp32 输入，让 autocast 决定是否 cast）
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    result_fp16_node = node.target(*fp32_args, **fp32_kwargs)
```

结果非常清晰——只有matmul和linear产生误差，其他所有op（batch_norm、layer_norm、sigmoid、add、mul等）引入误差为0：

```
op                          count total_mean_diff    max_max_diff
======================================================================
matmul.default                 18        0.121061        0.272247
linear.default                116        0.010778        0.034382
layer_norm.default             82        0.000000        0.000000
batch_norm.default              6        0.000000        0.000000
sigmoid.default                 1        0.000000        0.000000
... (其他所有 op 均为 0)
```

attention matmul的单步误差遥遥领先，因为Q*K^T的中间值scale在50~140，fp16精度不足以表示这么大的数值。

到这里，直觉会说：那就把attention matmul改成fp32呗？

别急，按照既往的经验，attention的matmul出现误差的几率很小，由于QK^T是一种规约操作，本身这个方向上的值超过100也是很正常的事情。

#### 逐节点误差传播分析：谁的误差真正影响了输出？（关键实验）

单步误差大 ≠ 对最终输出影响大。第二个关键实验更精确：每次只让一个节点用fp16执行，其他全部用fp32正确值，测量该节点的fp16误差独立传播到最终输出的贡献。

```python
class SingleNodeFP16Interpreter(Interpreter):
    def run_node(self, n):
        if n.name == self.fp16_node_name:
            # 目标节点: autocast fp16 执行
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                result = super().run_node(n)
            return result.float()  # cast回fp32让误差自然传播
        elif not self.passed_target:
            # 目标节点之前: 用 fp32 正确值
            return self.fp32_values[n.name]
        else:
            # 目标节点之后: 正常 fp32 执行
            return super().run_node(n)
```

结果完全颠覆了单步误差的排名——误差高度集中在main_net和bias_net最后几层linear：

```
Node                          op    mean_diff     max_diff  out_scale      pct
================================================================================
linear_105                linear     0.002264     0.009185     1.1625   87.69% <---
linear_111                linear     0.000888     0.006545     2.5643   34.42% <---
linear_109                linear     0.000250     0.003121     1.3213    9.67% <---
linear_103                linear     0.000162     0.000879     1.8891    6.27% <---
linear_107                linear     0.000119     0.001128     0.7051    4.59% <---
linear_102                linear     0.000098     0.000650     0.0266    3.80% <---
...
matmul_2                  matmul     0.000047     0.000283    76.1582    1.80%
matmul                    matmul     0.000028     0.000390   141.5600    1.08%
```

仅`linear_105`（main_net最后一层output_fc）一个节点，就贡献了87.7%的最终输出误差。而单步误差最大的attention matmul？传播到输出后贡献只有1.8%和1.08%。

#### 为什么是这几个 linear？

看一下模型的尾部结构就明白了：

```
main_net.net.2.output_fc (linear_105)
    → batch_norm → leaky_relu
    → logit_net (linear_113)  ─┐
                                ├─ add → add bias → sigmoid → output
bias_net.net.2.output_fc (linear_111)
    → batch_norm → leaky_relu
    → logit_net (linear_115)  ─┘
```

这几个linear的输出直接（或经过很少的层）连接到sigmoid前的logit，误差没有被后续网络层"稀释"的机会。

而attention matmul虽然单步误差大，但它的输出还要经过十几层网络计算才到达输出，误差在传播过程中被大量稀释和抵消了。

一句话总结：**不是这几个linear自身误差大，而是它们处在模型的关键路径上，误差无处稀释。**

#### 修复

使用`torch.library.custom_op`注册一个fp32版本的linear op，在AMP上下文下仍然以fp32执行：

```python
@torch.library.custom_op("rtp_lib::fp32_linear", mutates_args=())
def fp32_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
    with torch.amp.autocast('cuda', enabled=False):
        if x.dtype != torch.float32:
            x = x.float()
        return torch.ops.aten.linear.default(x, weight, bias).clone()
```

在FX Graph中把上述6个关键linear节点替换为`fp32_linear`即可。

#### 线上部署：C++ 实现

上面的`torch.library.custom_op`是Python层面的方案，用于验证和离线实验。但线上推理环境是纯C++的（AOTInductor导出后），需要一个C++版本的fp32_linear算子。

好在实现非常简单——`at::linear`底层调用的是cuBLAS/hipBLAS，性能本身就很好，我们只需要在调用前把输入强转为fp32：

```cpp
at::Tensor fp32_linear(const at::Tensor& input, const at::Tensor& weight,
                       const c10::optional<at::Tensor>& bias) {
    auto input_fp32 = input.to(at::kFloat);
    return at::linear(input_fp32, weight, bias);
}
```

通过`ExcludeDispatchKeyGuard`排除AutocastCUDA的dispatch key，确保这个op在AMP上下文下不会被自动cast回fp16。

该功能已在RTP-Torch框架中提供支持，编译阶段会自动识别精度敏感的linear节点并替换。

---

### 最终效果

两阶段修复的叠加效果：

```
Stage                              max_diff    mean_diff    rel_err
----------------------------------------------------------------------
Original (eps=0.001, all fp16)     0.017759     0.004263      1.89%
After BN eps fix                   0.010487     0.002582      1.14%
After BN eps + linear fp32         0.001413     0.000273      0.12%
```

逐样本对比也很直观：

```
      fp32   raw_fp16     bn_fix   full_fix    raw_err  final_err
    0.0970     0.0729     0.0961     0.0968   0.024127   0.000236
    0.0838     0.0572     0.0826     0.0840   0.026586   0.000279
    0.1569     0.0744     0.1595     0.1574   0.082562   0.000470
    0.0454     0.0545     0.0465     0.0453   0.009115   0.000117
    0.0638     0.0529     0.0609     0.0638   0.010853   0.000067
    0.2299     0.2100     0.2358     0.2296   0.019915   0.000248
    0.3733     0.4510     0.3822     0.3733   0.077697   0.000066
```

总改善15.6倍。只替换了6个linear（占全部116个的5%），对推理性能几乎无影响。

---

### 总结与方法论

这次排查的核心方法论是**逐节点隔离实验**，分两步走：

1. **单步误差分析**：回答"哪个op产生了误差"——给每个op喂fp32输入，对比fp32和fp16输出
2. **误差传播分析**：回答"哪个op的误差对输出影响最大"——每次只让一个op用fp16，测量其误差传播到输出的贡献

这两个实验的结论可能完全不同，甚至相反。attention matmul单步误差最大，但对输出贡献最小；最后几层linear单步误差不大，但对输出贡献最大。如果只做了第一个实验就动手优化attention，方向就完全跑偏了。

关键insight：**在深度网络中，误差的影响不取决于它有多大，而取决于它离输出有多远。** 靠近输出的节点，即使自身误差很小，也会对最终结果产生显著影响，因为误差没有被后续计算稀释的机会。
