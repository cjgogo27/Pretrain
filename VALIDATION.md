# 实际验证记录：三组优先修订

日期：2026-10-15（Asia/Shanghai）。本轮只修订清洗/配比实跑、TP/PP与性能案例、多步DDP训练及恢复。保留六章29节87问答；部署、后训练和其他进阶专题未扩展。以下通过项均在本轮实际执行。

## 环境与执行身份

- Windows 11（10.0.26200）、Python 3.12.5、PyTorch 2.11.0+cpu，CUDA available=False。FP32训练、每进程1个CPU线程；TP代数对照使用FP64。
- Node.js 24.14.0、npm 11.9.0；锁定Astro 7.3.1 / Starlight 0.42.0，未新增依赖。
- 执行基线HEAD：c15e884458ed4890fd233f42ff6d30350b465c69，当时有本轮工作区修改。
- 规范化train.py SHA-256：596cbe6485f4e405b6abc399e0d2b793a812b36caa4096b6b65a902960487403。Git HEAD作来源，严格恢复仍核对训练文件身份。
- 证据索引见[结果说明](examples/results/README.md)。JSON保存配置、身份、索引、实测与边界；完整权重、日志和约16MB/份trace留在忽略的runs/中。已提交全部160个profiler范围事件与原trace哈希，支持独立汇总。

## 清洗与配比实跑

实际执行：

    python examples/cleaning_case.py --out runs/priority-cleaning
    python examples/mixture_case.py --out runs/priority-mixture-final

清洗使用十份原创、预先人工标注保留意图的文档：统一换行、明确导航行、替换字符/长度过滤、SHA-256精确去重、连续五字符集合Jaccard。不是MinHash。实际核对精确副本识别和代码四空格缩进保留。

| Jaccard阈值 | 保留 | 误删 | 漏删 |
| --- | --- | --- | --- |
| 0.70 | 4/10 | changed-claim、short-valid | paraphrase |
| 0.95 | 6/10 | short-valid | near-copy、paraphrase |

raw.jsonl与逐项report实际生成，发布摘要包含规范化文本、哈希、匹配对象、相似度和原因。夹具刻意包含边界样例，不估计网页错误率，未用于训练或测清洗收益。

配比执行前保存design，以seed=2026打乱六次运行顺序；训练种子42/43/44各配对比较领域概率0.2/0.5。默认437,888参数、B4/S64/A1/DP1、100步，每次25,600个有效目标位置，其余优化计划相同。领域为优化器主题，通用为其余九主题；训练池20/180文档，含EOS的原始位置数5,684/51,256。

| 种子 | A领域消费token | B领域消费token | 领域NLL：B−A | 通用NLL：B−A |
| --- | --- | --- | --- | --- |
| 42 | 4,928 | 12,032 | −0.07123165 | +0.00806963 |
| 43 | 4,992 | 12,672 | −0.04270026 | +0.02401473 |
| 44 | 4,352 | 12,800 | −0.08602060 | +0.01164870 |

平均配对差−0.06665084 / +0.01457769 nats/token。最终每池固定64个S64验证窗口、4,096目标位置，seed=2026；六个模型共用该池索引，完整索引和哈希已保存。训练中默认16窗口日志另列。已核对六次均完成、两池计数和为25,600、loss有限、身份正确。初跑与最终脚本重跑数值相同，本表绑定最终重跑。

仅支持共用模板分布上的预测取舍。三个种子不是有统计功效的最优配比研究，窗口不能当独立训练重复，重复消费比不等于覆盖率。模板/主题留出未运行。

## TP/PP代数与调度

    python examples/parallel_case.py --out runs/priority-parallel

同进程FP64对比X[6,4]、W1[4,8]、W2[8,4]的原FFN与两路分解，输出、dX、dW1、dW2最大绝对差8.881784197001252e-16，atol=rtol=1e-12通过。检查两阶段四微批次GPipe的前后向依赖：20设备格有4格空闲，总时间口径20%；相对理想计算时间开销25%，分母明确区分。

16MiB payload及每rank前后向发送合计32MiB是经典ring AllReduce假设下的字节算例，没有真实TP/PP通信与耗时测量。

## CPU性能诊断

    python examples/profile_case.py --out runs/priority-profile-final

width32/1层/B4/S32，固定初始化和窗口，每次预热3步、测20步。明确注入4ms sleep，再移除；各三次非profiler基准，随机运行顺序，另各跑一次CPU profiler。

- 三次步中位数再取中位数：等待6.441900ms，移除1.737150ms。
- profiler累计20步取数范围88.4312 / 0.6041ms；含等待条件前向28.2528、反向32.3288、裁剪更新32.3456ms。
- 六次基准与两次profiler最终参数字节哈希相同；实际导出两份有效JSON trace，提取160个case事件，按名称汇总dur与report差小于0.001ms。

范围时间含等待，不等于CPU浮点忙碌；嵌套范围/子算子不能重复计费，内存归属不是峰值显存。没有GPU MFU、默认loader瓶颈或生产加速结论，未在trace查看器点击验收。profiler提示事件按周期清理；本次无循环schedule，已核对每条件完整20步、80个阶段事件。

## 单进程与DDP恢复

    python examples/verify_resume.py
    python examples/verify_ddp.py
    python examples/verify_ddp_training.py --out runs/priority-ddp

1. 单进程12项重新通过：因果可见范围与形状；12步连续和6步保存恢复的模型/AdamW/RNG/采样器完全相同，768 token；数据、实际lr代码乘1.01、缺代码身份、world size、PyTorch版本、计划变更均拒绝；仅权重新运行重置状态。
2. 原100/900有效token单步CPU Gloo对照通过：梯度最大差1.4901161193847656e-08、参数1.0496507485413531e-07、固定8窗口验证loss差1.9073486345888568e-07。
3. 新五步相同全局样本对照：S8、每rank B2/A2、DP2，每更新64 token；首微批前后向no_sync，loss/2，最后同步后阈值0.2裁剪，五步均触发。最大差：梯度3.725290298461914e-08、裁剪前范数5.960464477539063e-08、参数2.51146389018686e-07，atol1e-6/rtol1e-4通过。
4. 新真实循环：launch_cpu建立两进程Gloo后直接调用train.run。width32/1层/S16/B2/A2/DP2、领域概率0.5、总计划8步，比较连续与第4更新边界正常保存退出再以新进程恢复。最终step8、1,024 token、领域480/通用544；模型、AdamW、两rank随机/采样状态、消费计数逐元素一致，最终lr、loss、范数、验证身份与生成文件一致。改概率或池文件均拒绝恢复。

默认torchrun的libuv限制未验证解除，启动器显式使用TCPStore(use_libuv=False)；有地址重试警告，最终完成。只验证正常更新边界保存/重启，不是中途写盘失败、rank崩溃、弹性恢复或长期稳定性。

## 默认回归与仅权重初始化

    python examples/prepare_data.py --out runs/priority-default-data
    python examples/train.py --data runs/priority-default-data --out runs/priority-default --steps 100
    python examples/train.py --data runs/priority-default-data --out runs/priority-weights --steps 20 --lr 0.0001 --warmup 5 --init-weights runs/priority-default/latest.pt

默认437,888参数、25,600 token；20/40/60/80/100步loss/PPL/lr/范数与上轮逐点完全相同，最终val_loss=3.44255793094635；计时重新观测，JSON身份更新。仅权重阶段从step/token=0开始，20步/5,120 token，val_loss=3.360415041446686。同语料，只说明初始化和状态重置，不是领域收益实验。原曲线数值未变，未声称本轮重新截图。

## 文档与产物检查

| 本轮实际执行 | 结果 |
| --- | --- |
| npm run check:docs | PASS：40源页面、29节、87问答 |
| npm run build | PASS：40 HTML、搜索索引、sitemap |
| npm run check:built | PASS：87个KaTeX实例无解析错误，内部目标/锚点缺失0，29组答案结构完整 |
| python -m compileall -q examples | PASS |
| git diff --check | PASS |
| 结果核对脚本 | 训练/执行脚本身份、六次配比计数/有限loss/固定64索引、PPL、逐点默认回归、160个范围dur汇总通过 |
| 接口与图示 | 22个列出的CLI字段对应配置，TP张量、PP依赖及气泡分母、两池计数已核对 |
| 来源核对 | 新TP/PP/no_sync/profiler论断已阅读主来源，支持范围写入SOURCE_AUDIT；不以HTTP可达替代论断核验 |
| UI库存 | 只有空MCP Apps，无可操作网页标签，未做桌面/移动截图、折叠点击、搜索与预算交互验收 |

构建仍有依赖层use astro:head-inject警告及已有404路由提示，没有构建失败。静态检查不替代浏览器显示验收。

## 失败尝试与边界

- 首轮验证捕获Windows原生诊断时UTF-8解码失败。已为Python子进程指定UTF-8，对原生无法解码字节采用替换，仍检查退出码和ASCII错误身份；随后12项及完整DDP/拒绝检查重新运行通过，首轮失败未计为通过。
- 性能脚本初版漏执行非profiler分支循环，空均值报错。修复后重新跑全部六次基准与两次profiler，表内只使用完成后的结果。
- 未执行CUDA/NCCL、多机、AMP、状态分片、真实TP/PP/CP、长上下文、写盘故障/SDC、长期稳定性、生产吞吐/GPU MFU、模板/主题留出或真实领域能力测评。

实验设计方法与工具归属已写入[结果说明](examples/results/README.md)。发布状态以本轮GitHub Actions记录为准。
