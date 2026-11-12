# 实际验证记录：《听懂大模型预训练》

## 2026-11-12：站内中文/英文切换

本轮基线 `95762dd68e47331305855538b6f216d8929a30a1`。保留中文正文、六章29节及87个角色问答，增加41个对应英文静态页面（包含404页）。页眉按钮在同一章节中切换语言，启用JavaScript时保留锚点；导航、正文、问答、搜索及交互组件的说明随页面语言切换。英文代码块与公式保留原文。

英文为机器翻译稿，并对书名、全部页面标题、导航、系统总图及部分关键术语做针对性校订，不代表逐句人工审定。英文页展示此范围说明，技术歧义以中文原文为准。翻译输入已与公开GitHub Pages的正文逐条匹配；仅一次性生成翻译词典，构建、CI和阅读时不调用翻译服务。源码变化产生新文本时构建拒绝缺失翻译，维护流程见README。

### 实际执行

环境：Windows、Node24.14.0、npm11.9.0、固定Astro7.3.1/Starlight0.42.0；浏览器为Edge157.0.4341.46，复用已有Playwright，使用全新上下文。`npm ci --no-audit --no-fund` 实际完成，替换了不能正常构建的旧依赖目录；新增显式依赖parse5/pagefind原已在锁文件中。

| 命令或检查 | 实际结果 |
| --- | --- |
| `npm run check:docs` | PASS：41个中文页面、29节、87个角色问答 |
| `npm test` | PASS：4项原有Node测试 |
| `npm run build` | PASS：41个中文页面、41个英文页面；7176处文本替换，无运行时翻译 |
| `npm run check:built` | PASS：82页、174处KaTeX渲染、缺失链接0；41组语言页面、24个代码块与87个公式块逐项相同；29组英文角色问答；可翻译可见文本残留中文0 |
| 缺失词条反例 | 临时移除一个未被人工覆盖的词条，`node scripts/bilingual.mjs` 实际退出1并提示需要翻译；随后恢复词典 |
| `node scripts/check-bilingual-browser.cjs` | PASS：8组检查，JavaScript页面错误0；报告见[validation/bilingual-browser.json](validation/bilingual-browser.json) |
| `git diff --check` | PASS |

浏览器检查覆盖1280×900和390×844：当前章节/锚点往返切换、英文折叠答案、五维布局和预算计算动态文字；桌面搜索返回英文页面；禁用JavaScript仍可阅读英文并使用静态语言链接。实际查看桌面及移动宽度截图，修复英文长串导致的移动横向溢出与标题拥挤。没有声称逐页截图审阅、真机测试或全面语言校对。截图和临时构建保留于忽略的`runs/bilingual-browser/`，脚本可复现。

本轮仅修改站点呈现和构建，不修改训练算法或重新执行历史训练/GPU实验；下面的训练结果仍按原日期、版本及环境理解。

## 2026-10-22：数据与Infra、五维并行、弹性迁移

基线HEAD369b5bba89e3a817dc3781310de766230f772dca；远端没有新增主干变化。核对现有清洗/配比、TP/PP代数/调度、CPU profile及DDP恢复后补内容，没有把已有章节当缺失。保留六章29节87问答；新增1页跨章地图，共41页。部署和后训练未扩展。

### 实际修改与证据层级

- 1.1/1.3、首页和导读将数据、学习、Infra列为共同主线；2.1/2.4/2.5补实际配比消费、供给/缓存/预取与已提交数据游标。
- 4.5新增DP/TP/PP/CP/EP、MoE路由、ETP/EDP与SP/CP区分；32逻辑设备的Attention PP2/DP4/CP2/TP2和专家PP2/EDP2/EP8/ETP1复用同一设备，PP组一致。全局原始目标131,072，不机械乘EP变256设备。
- 4.6区分集群准入、成员重建、训练状态迁移；5.4补互斥时间段下排队/占用/已保留更新，48 GPU小时仅是假设算例。
- 6.3提供独立CPU边界迁移程序；默认train.py与其格式2严格恢复没有修改或放宽，规范化训练哈希仍为596cbe6485f4e405b6abc399e0d2b793a812b36caa4096b6b65a902960487403。

### 运行环境

Windows11（10.0.26200）、Python3.12.5、PyTorch2.11.0+cpu；新成员迁移使用FP64、每进程1线程、Gloo/TCPStore(use_libuv=False)。无CUDA、NCCL或生产调度平台运行。站点仍用固定Astro7.3.1/Starlight0.42.0，无新增训练或生产依赖。浏览器复用现有Playwright和Edge155.0.4283.62，两个全新浏览器上下文，不使用个人浏览配置。

### 实际逻辑布局与路由核对

```bash
python examples/topology_case.py --out runs/topology-20261022-final
npm test
```

PASS。枚举32个逻辑rank，检查每维完整覆盖、无重复、PP组成员完全一致；EP3无法整除的布局拒绝。16原始token、top-k2为32次专家分配，均衡负载每专家4；集中到两个专家时各16，容量定义ceil(1.25×16×2/8)=5，超容量22次分配。没有把22写成不同token数量，也未运行All-to-All、32进程或MoE训练。Python实际报告与界面共用JSON；4个Node测试检查手算成员、PP一致、计数与无效归属拒绝，全部通过。来源位置见SOURCE_AUDIT.md。

### 实际进程故障与边界成员迁移

```bash
python examples/elastic_case.py --out runs/elastic-20261022
```

PASS，完整报告examples/results/elastic-reference.json。它复用TinyLM类但不调用train.run：width16/layers1/heads4/S8、固定随机ID窗口、无dropout、FP64完整复制模型与AdamW；lr0.001、decay0.1、clip1。每更新8全局窗口、64有效目标位置；B2，DP2时A2，DP1时A4。

1. 固定两进程执行6更新作基线。
2. 另一路两进程提交2步后，实际注入rank1退出码86；父进程只接受这个指定故障和标记，确认最新快照仍为第2步且文件SHA不变。
3. 以单进程重建到第4步，再以两进程重建到第6步，实际DP顺序2/2/1/1/2/2，A为2/2/4/4/2/2。
4. 两路最终均保留384位置，窗口ID账本0…47相同。完整模型最大绝对差9.490718672634083e-16，完整Adam状态1.0408340855860843e-17，通过atol1e-9/rtol1e-8。
5. 失败尝试已记录两rank首微批次共32目标位置，后续可能还有部分计算，所以32只是可确认的丢弃计算下界。对应ID16/18/17/19被重放；384不冒充总实际计算量。

有Windows地址连接重试警告，以及注入故障后父进程终止其他worker的诊断，最终完成。没有发生写盘中断；只验证失败没有提交新快照，不能当成写入事务故障验收。程序不保存rank RNG，因为无随机采样层且窗口全局固定；不覆盖生产dropout/shuffle/多worker数据加载器。它不是torchrun自动弹性准入、Kubernetes/Kueue/Slurm或分片/TP/PP/CP/EP状态迁移，不支持生产性能推论。

### 原有行为的实际回归

```bash
python examples/verify_resume.py
python examples/parallel_case.py --out runs/parallel-20261022
python -m compileall -q examples
```

- 原12项严格恢复检查重新PASS，包括因果/形状、12步连续与6步恢复状态、RNG/采样器、768 token、改数据/代码/缺代码身份/world size/PyTorch/计划拒绝、仅权重状态重建。
- 原TP同进程FP64代数重新PASS，最大差8.881784197001252e-16；PP理想GPipe依赖与4/20空闲格重新PASS；不是新TP/PP分布式运行。
- compileall通过。回归JSON为regression-20261022.json。原清洗、三种子配比、profile、默认100步和真实train.run多步DDP实验本轮未重跑；历史产物/数值保留，不修改日期冒充新结果。

### 文档与浏览器实际检查

| 执行项 | 结果 |
| --- | --- |
| npm run check:docs | PASS：41源页面、29节、87问答，正文中文约4.55万字 |
| npm run build | PASS：41 HTML、Pagefind、sitemap |
| npm run check:built | PASS：87个KaTeX实例，内部资源/目标/锚点缺失0，29组三答案 |
| npm test | PASS：4个逻辑布局/计数/归属测试；CI已加入 |
| git diff --check | PASS，包含新增文件暂存后复查 |
| scripts/check-infra-browser.cjs | PASS：1280×900与390×844，10组检查，页面JS异常0 |

浏览器使用实际构建预览端口4326；PLAYWRIGHT_MODULE指向现有工作区运行时，BROWSER_EXECUTABLE指向本机Edge，BROWSER_CHECK_OUT=runs/infra-browser-20261022。10组包括三主线图/宽度、32rank各层成员/坐标/键盘、进阶折叠和角色答案、主题与响应布局、弹性证据和成本示例显示。截图已实际打开检查；修正Starlight通用相邻元素margin造成的rank按钮/卡片错位，最终增加几何对齐断言。报告infra-browser-reference.json，截图留在runs中。

首次尝试沿用旧Chromium缓存路径，因文件已不存在而失败；随后改用现有Edge实际运行通过，没有把浏览器启动失败计成通过。构建仍有既有依赖级head-inject、i18n/404警告，无构建错误。静态HTML检查与浏览器检查各自标明范围，未测试所有页面/搜索、Firefox/Safari或真机触控。

### 仍未运行的范围

真实5D MoE训练、All-to-All性能、长上下文CP、CUDA/NCCL、多机、生产资源调度、分片优化器/DCP加载、随机层弹性迁移、写盘故障、长期稳定性与GPU MFU均未运行。逻辑32rank、真实本机CPU进程、官方项目机制和假设预算分别标记。严格恢复未被改造成可任意换布局的入口。

---

## 2026-10-15历史记录：三组优先修订


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

## 发布后实际检查

内容提交为1f0144d5100ab3e3e4669627fc028ccd45a3c080，[GitHub Actions构建与Pages发布](https://github.com/cjgogo27/Pretrain/actions/runs/38688161602)已完成，结论success。

发布后实际HTTP读取2.1配比、4.5 TP/PP、5.4性能、6.3多步DDP四个线上页面，均返回200，并分别检出本轮新数值或图示标记。此项只确认线上内容已更新，不等同于截图、布局和交互验收。本记录后续独立文档提交不改变该内容提交的实测源码身份。
