# 本轮关键论断与支持位置

核对日期：2026-10-15。先阅读正文和参考原文，再按下表核对改动涉及的关键机制。HTTP 成功只表示可访问，不表示原文支持论断；下表不是全书所有参考文献的逐句审计。

| 正文位置 / 论断 | 实际支持位置 | 本书采用的范围 |
| --- | --- | --- |
| 阶段导读：OLMo 2 主训练与 mid-training 分开数据及学习率计划 | [2 OLMo 2 Furious §2.1](https://arxiv.org/html/2501.00656v1#S2.SS1)、[§2.3](https://arxiv.org/html/2501.00656v1#S2.SS3)、[§4](https://arxiv.org/html/2501.00656v1#S4) | 核对主阶段余弦、后阶段 Dolmino 与线性退火；不称 OLMo 2 全过程为 WSD，不复用其能力分数 |
| 阶段导读：SmolLM3 使用 WSD、改变阶段配比 | [官方文章 Architecture and training details](https://huggingface.co/blog/smollm3#architecture-and-training-details)、[Data mixture and training stages](https://huggingface.co/blog/smollm3#data-mixture-and-training-stages) | 原文说明先稳定后衰减，并随阶段调整代码 / 数学份额；未将该配置称为最优通则 |
| 阶段导读：SmolLM3 将长上下文适配独立安排，4K→32K→64K | [Long Context extension](https://huggingface.co/blog/smollm3#long-context-extension) 及其前面的 Mid-training 定义 | 同时核对 RoPE 参数调整；不把 128K 推理外推当作 128K 训练长度；不使用本站假设图复现其 recipe |
| 2.1 / 阶段导读：数据比较应控制模型与训练 recipe | [DataComp-LM Introduction](https://arxiv.org/html/2406.11794v1#S1) 的混淆因素讨论和固定 recipe 工作流 | 2406.11794 是数据 benchmark 论文，不把它当作 WSD 原始论文；本站配比对照只是未运行设计 |
| 3.5：参数、token、预算存在带条件的经验关系 | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，固定训练预算的实验设置与结果 | 不从约 20 token/参数推出数据上限、阶段长度或通用最优配比 |
| 3.2：RoPE、RMSNorm、SwiGLU、GQA 是不同部件 | [RoFormer](https://arxiv.org/abs/2104.09864)、[RMSNorm](https://arxiv.org/abs/1910.07467)、[GLU Variants](https://arxiv.org/abs/2002.05202)、[GQA](https://arxiv.org/abs/2305.13245) 的对应定义 / 公式 | 只解释位置、去均值差异、门控投影和共享 K/V；不声称 TinyPretrain 实现或复现其结果 |
| 4.1：标准 DDP 平均 rank 梯度，sum-loss 需适当缩放 | [PyTorch 2.11 DDP](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html) 中 summed loss 相对单节点梯度缩小的 Note、no_sync 接口 | 仅针对等参与数、标准均值归约、无自定义 hook / join 的假设；100/900 例子是独立推导，另做实际 CPU 数值对照 |
| 4.3：累积期间保持 scale，结束后一次 unscale，再裁剪；scaler 可跳更新 | [PyTorch 2.11 AMP Gradient clipping](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html#gradient-clipping)、[Gradient accumulation](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html#gradient-accumulation)、Typical Mixed Precision Training | 机制说明，不声称 CPU 教学代码实现 AMP；计数 / 调度需区分尝试与成功更新 |
| 4.3：分片梯度范数不是单 rank 的局部范数 | [PyTorch 2.11 FSDP.clip_grad_norm_](https://docs.pytorch.org/docs/2.11/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_) | 只给选择对应分布式机制的原则，未提供跨所有分片布局通用的一行调用 |
| 5.4：MFU 的模型 FLOPs 排除实现所需的重计算，与硬件执行量不同 | [PaLM §4.1 与 Appendix B](https://arxiv.org/html/2204.02311v5#A2) 的定义与含 / 不含 rematerialization 的对照 | 100/130/200 的数字是独立假设算例；硬件执行量可为分析估计，不伪称 profiler 测量 |
| 1.3 / 2.4：Hi!、OK 的 ID 与 mask | UTF-8 编码规范及项目实际 `text.encode('utf8')`、因果 mask 实现；具体 ID 可运行核对 | ID 是真实字节编码；概率、隔离 packing 矩阵是教学设计。默认代码没有文档隔离或 padding loss mask |
| 4.6 / 6.2–6.4：严格恢复、新运行与实测曲线 | 项目 `train.py`、`verify_resume.py`、`verify_ddp.py` 和 [cpu-reference.json](examples/results/cpu-reference.json) | 用实际程序与测试验证，不借论文声誉为本项目结果背书；单步 DDP 不等于完整多机验收 |

阶段图中 120B token、配比与长度，以及 1B 预算校准案例的耗时均为本站假设，未归为他人实验或本机实测。提示 / 抽取造成 50%→100% 的两题例子是人工构造评分例子，没有训练或模型成绩宣称。

原有来源归属保留于 [ATTRIBUTION.md](ATTRIBUTION.md)。技术定义和算例仍需结合实现版本理解，不能仅因概念相似把机器人实时频率或纵向指令层次套入训练系统。
