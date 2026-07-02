### 欢迎来到 ZhangYu 的 blog

这里记录一些 PyTorch / Triton / GPU kernel 性能优化的实战与踩坑, 大多来自搜推场景的真实问题. 欢迎一起交流.

### 文章列表 (按更新时间倒序)

- [SDPA的通用性留下的缝隙: 从数据特点里抠出更快的Attention](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/SDPA%E7%9A%84%E9%80%9A%E7%94%A8%E6%80%A7%E7%95%99%E4%B8%8B%E7%9A%84%E7%BC%9D%E9%9A%99%3A%20%E4%BB%8E%E6%95%B0%E6%8D%AE%E7%89%B9%E7%82%B9%E9%87%8C%E6%8A%A0%E5%87%BA%E6%9B%B4%E5%BF%AB%E7%9A%84Attention)
- [SM120上的Triton TMA入门](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/SM120%E4%B8%8A%E7%9A%84Triton%20TMA%E5%85%A5%E9%97%A8)
- [大Batch一上来就IMA：一次Triton int32索引溢出的排查](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/%E5%A4%A7Batch%E4%B8%80%E4%B8%8A%E6%9D%A5%E5%B0%B1IMA%EF%BC%9A%E4%B8%80%E6%AC%A1Triton%20int32%E7%B4%A2%E5%BC%95%E6%BA%A2%E5%87%BA%E7%9A%84%E6%8E%92%E6%9F%A5)
- [Triton入门: 从零实现面向Decode场景的GQA Attention](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/GQA_ATTENTION)
- [stream_manager_leak_repro](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/stream_manager_leak_repro)
- [per-token ffn推理优化与Triton Tiling技巧](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/per-token%20ffn%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96%E4%B8%8ETriton%20Tiling%E6%8A%80%E5%B7%A7)
- [PyTorch多卡D2D踩坑实战](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/PyTorch%E5%A4%9A%E5%8D%A1D2D%E8%B8%A9%E5%9D%91%E5%AE%9E%E6%88%98)
- [PyTorch AOTInductor设计解读: C++和Python下的并发设计](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/PyTorch%20AOTInductor%E8%AE%BE%E8%AE%A1%E8%A7%A3%E8%AF%BB%3A%20C%2B%2B%E5%92%8CPython%E4%B8%8B%E7%9A%84%E5%B9%B6%E5%8F%91%E8%AE%BE%E8%AE%A1)
- [RTP-Torch 2025新进展](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/RTP-Torch%202025%E6%96%B0%E8%BF%9B%E5%B1%95)
- [某直播推荐业务kernel优化实战](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/%E6%9F%90%E7%9B%B4%E6%92%AD%E6%8E%A8%E8%8D%90%E4%B8%9A%E5%8A%A1kernel%E4%BC%98%E5%8C%96%E5%AE%9E%E6%88%98)
- [搜推场景下的Target-Attention和可变长Flash-attention的加速策略](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/%E6%90%9C%E6%8E%A8%E5%9C%BA%E6%99%AF%E4%B8%8B%E7%9A%84Target-Attention%E5%92%8C%E5%8F%AF%E5%8F%98%E9%95%BFFlash-attention%E7%9A%84%E5%8A%A0%E9%80%9F%E7%AD%96%E7%95%A5)
- [PyTorch: 你比以前快了](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/PyTorch%3A%20%E4%BD%A0%E6%AF%94%E4%BB%A5%E5%89%8D%E5%BF%AB%E4%BA%86)
- [怎么样让我的PyTorch代码跑得更快](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/%E6%80%8E%E4%B9%88%E6%A0%B7%E8%AE%A9%E6%88%91%E7%9A%84PyTorch%E4%BB%A3%E7%A0%81%E8%B7%91%E5%BE%97%E6%9B%B4%E5%BF%AB)
- [PyTorch在线预测启航](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/PyTorch%E5%9C%A8%E7%BA%BF%E9%A2%84%E6%B5%8B%E5%90%AF%E8%88%AA)
- [基于图追踪和AOT(ahead of time)技术的Pytorch 2+模型部署链路初探(一)](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/torch_fx_aot_chapter)
- [追寻PyTorch推理中Attention计算的最优解](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/%E8%BF%BD%E5%AF%BBPyTorch%E6%8E%A8%E7%90%86%E4%B8%ADAttention%E8%AE%A1%E7%AE%97%E7%9A%84%E6%9C%80%E4%BC%98%E8%A7%A3)
- [TA&&var_flash_attn](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/TA%26%26var_flash_attn)
- [PyTorch坑点随记: 谨慎使用masked_fill](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/PyTorch%E5%9D%91%E7%82%B9%E9%9A%8F%E8%AE%B0%3A%20%E8%B0%A8%E6%85%8E%E4%BD%BF%E7%94%A8masked_fill)
- [某CTR业务AMP混合精度误差排查实战](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/%E6%9F%90CTR%E4%B8%9A%E5%8A%A1AMP%E6%B7%B7%E5%90%88%E7%B2%BE%E5%BA%A6%E8%AF%AF%E5%B7%AE%E6%8E%92%E6%9F%A5%E5%AE%9E%E6%88%98)
- [Speed Up PyTorch Model Inference on CPU with Intel OneDNN Integration](https://github.com/sujuyu/ZhangYu-s-Blog/tree/main/torch_onednn_quantization)
