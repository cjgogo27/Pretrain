# 实测记录怎样对应到代码

这些 JSON 是 2026-10-15 本机实际运行的记录，不是期望结果。完整权重、逐行日志和约 16MB/份的原始 profiler trace 留在各自 `runs/` 目录，不提交到普通 Git；以下保留配置、身份和核对所需数据。训练代码严格身份采用规范化 `train.py` 哈希；其他脚本的 `script_sha256` 是执行文件原始字节哈希。

| 文件 | 实际覆盖 | 重跑命令（仓库根目录） |
| --- | --- | --- |
| cpu-reference.json | 默认 100 步、单进程恢复 12 项、原单步 DDP 对照、仅加载权重新运行 | `python examples/prepare_data.py --out runs/data`；`python examples/train.py`；`python examples/verify_resume.py`；`python examples/verify_ddp.py` |
| cleaning-reference.json | 十份人工标注文档、两阈值决定与误删/漏删；非网页质量抽样 | `python examples/cleaning_case.py --out runs/cleaning` |
| mixture-reference.json | 3 个配对种子 × 2 个概率，每次 25,600 token；两池固定验证索引 | `python examples/mixture_case.py --out runs/mixture` |
| parallel-reference.json | 同进程 FP64 FFN 切分代数与理想 GPipe 依赖/计数；非真实 TP/PP | `python examples/parallel_case.py --out runs/parallel` |
| profile-reference.json | 受控 4ms 等待、三次配对基准、两次独立 CPU profiler | `python examples/profile_case.py --out runs/profile` |
| profile-ranges.json | 从两份原始 trace 提取全部 160 个 case 范围事件；dur 单位微秒 | 同上；逐名称汇总 dur 可核对阶段总时间，完整 trace 身份也保留 |
| ddp-training-reference.json | 五步跨布局容差对照，真实循环八步同布局更新边界恢复与拒绝检查 | `python examples/verify_ddp_training.py --out runs/ddp-training` |

每次重跑选择新的输出目录。计时和 trace 哈希会变化；同平台同软件的数值测试仍应满足明确断言。默认训练本轮重跑的全部 loss/PPL/lr/范数点与上轮逐点相同，计时重新观测；不把不同运行的时间拼成一条速度结论。

配比实验的重复单位是训练种子，不是验证窗口或 token；64 窗口允许重叠，仅降低此固定分布上测量的抽样噪声，不产生 64 次独立训练证据。三种子只作教学对照，不声称统计功效或普遍最优配方。清洗夹具没有用于这些训练池，不能推导清洗的模型收益。

实验设计使用 experimental-design 技能的配对、控制变量、预先保存设计与避免伪重复流程，未增加其 DOE 软件依赖。工具方法归属：[Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*](https://arxiv.org/abs/2609.00065)（已核对当前 v2 元数据）。此来源只说明方法工具，不为本项目数值或模型能力背书。
