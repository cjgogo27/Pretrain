# 实际验证记录：教学与技术修订

日期：2026-10-15（Asia/Shanghai）。本轮先拉取并通读原版，再复核发现、修改、运行检查。以下仅记录实际执行的工作；实验设计、预算假设和生产经验不混作本项目实测。

## 环境与执行身份

- Windows，Python 3.12.5，PyTorch 2.11.0+cpu；训练使用 FP32、1 个 PyTorch CPU 线程、seed=42。
- Node.js 24.14.0，npm 11.9.0，Astro 7.3.1、Starlight 0.42.0；依赖继续由锁文件及 Rolldown 1.2.13 override 固定。
- 本轮执行时的基线 HEAD：`53a23ae09156e5f2ef0384c25e278a7a4440cdc6`，工作区已有修订。结果身份以规范化 train.py SHA-256 为准：`1aa57e7ad787e8446db35a25883864381e645800222a3fcba0eee7db2fe53c6b`。
- 配置、数据身份、窗口索引、原始指标、生成字符串和代码检查结果保存在 [examples/results/cpu-reference.json](examples/results/cpu-reference.json)。它不包含模型权重或生产性能承诺。

## 修改前实际复核

| 审阅发现 | 本轮复核结果 |
| --- | --- |
| 保存 code_revision / code_sha256，但恢复没有核对 | 代码阅读确认仍然存在 |
| 学习率乘以 1.01 后仍接受严格恢复 | 在临时副本保存 4 步计划的第 2 步后修改实际 lr 计算，恢复退出码 0，问题复现 |
| 原 CPU verify_resume.py 可通过 | 实际运行退出码 0，原 6 项检查通过 |
| 原记录未完成 DDP / CUDA 验证 | 记录阅读确认；不能据此推断当前数值路径已验证 |

## 默认 CPU 100 步实跑

本轮使用新的目录，实际执行：

```bash
python examples/prepare_data.py --out runs/revision-data
python examples/train.py --data runs/revision-data --out runs/revision-default --steps 100
```

默认 2 层、width=128、heads=4、seq-len=64，实际独立参数 **437,888**。B=4、A=1、DP=1，每更新 256 token；100 步累计 **25,600**。原创语料为 200 train / 40 val 文档，共用模板和主题。

| step | 累计 token | 最近更新 train loss | 固定窗口 val loss | PPL |
| --- | --- | --- | --- | --- |
| 20 | 5,120 | 4.748636 | 4.682485 | 108.0382 |
| 40 | 10,240 | 4.109130 | 4.074328 | 58.8109 |
| 60 | 15,360 | 3.588485 | 3.687630 | 39.9501 |
| 80 | 20,480 | 3.602930 | 3.512109 | 33.5189 |
| 100 | 25,600 | 3.687634 | 3.442558 | 31.2668 |

输出 `metrics.jsonl`、`latest.pt`、`sample.txt` 均产生。验证是固定全局 16 个窗口、1,024 个目标位置的估计，窗口可重叠。原始生成含乱码与控制字符；正文以 JSON 转义形式展示，没有人工改写成通顺输出。

使用已有 Matplotlib 从实际 JSON 绘制 [loss 图](public/figures/cpu-loss.svg)，并打开 PNG 预览检查了图的标签、图例和数值范围。此项是**独立曲线图片检查**，不是网站桌面 / 移动截图验收。单种子、模板语料与极小模型的 loss 改善，不支持通用能力结论。

## 严格恢复和新运行检查

```bash
python examples/verify_resume.py
```

本轮修改后实际通过 12 项检查：

1. 改未来 token，历史 logits 不变。
2. 输出形状正确。
3. 固定 12 步计划：连续运行与第 6 步中断恢复的模型 / AdamW 逐元素一致。
4. RNG、训练采样器与状态一致。
5. 全局 token 计数为 768。
6. 改训练数据后拒绝恢复。
7. 将实际 lr 计算乘以 1.01 后，以代码身份不符拒绝恢复。
8. 检查点缺失代码身份时拒绝恢复。
9. 改 world size 字段时拒绝恢复。
10. 改 PyTorch 版本字段时拒绝恢复。
11. 改训练 lr 配置时以计划不符拒绝恢复。
12. 数据变化后可以仅加载权重开始新运行，step/token 从零建立，AdamW 的内部 step 为新运行的更新数。

另外实际执行默认模型的权重初始化新阶段：

```bash
python examples/train.py --data runs/revision-data --out runs/revision-stage2 --steps 20 --lr 0.0001 --warmup 5 --init-weights runs/revision-default/latest.pt
```

启动记录为 `weights_only, source_step=100, start_step=0, start_tokens=0`；结束为 step=20、tokens=5,120，并保留来源检查点哈希。使用的是相同语料，只证明初始化 / 状态重置机制；领域适应收益与通用遗忘对照没有执行。

格式版本 2 的代码身份只规范 UTF-8 BOM 与换行，未忽略源码、注释或空白修改。Git HEAD 保留为来源信息，不强制整个提交相同。哈希不是安全签名，不保证跨硬件逐位复现；旧格式不能自动视作已通过新约束的严格恢复。

## 实际两进程 CPU DDP 数值对照

```bash
python examples/verify_ddp.py
```

实际在本机完成两个 Gloo 进程。固定初始化与全部样本，rank 0 / rank 1 分别处理 100 / 900 个有效 token；局部 summed loss 按 `2/1000` 缩放，与单进程全局 token 平均比较。随后执行一次相同 AdamW 更新；无 AMP、无裁剪，以隔离归约。

| 比较 | 实测最大绝对差 | 验收条件 |
| --- | --- | --- |
| 全参数梯度 | 1.4901161193847656e-08 | atol=1e-6，rtol=1e-4 |
| 一次更新后的参数 | 1.0496507485413531e-07 | atol=1e-6，rtol=1e-4 |
| 固定窗口验证 loss | 1.9073486345888568e-07 | 绝对差 ≤ 1e-6 |

全部逐参数检查通过。验证使用一个全局 8 窗口索引表，按 `indices[rank::2]` 分配，索引身份与单进程相同。它与默认 16 窗口报告是两项不同检查，未混成一组指标。

环境说明：默认 torchrun 的 libuv 限制仍存在；两次 FileStore 尝试返回 No such file or directory，未证实具体路径原因。最终通过显式 `TCPStore(use_libuv=False)` 避开该启动依赖；运行有地址重试警告，随后完成。没有把前面的失败记为通过。

该结果覆盖固定样本、单步 CPU DDP 数值路径，**不覆盖完整多步 DDP 训练 / 恢复、CUDA / NCCL、多机或吞吐扩展**。

## 文档、公式、默认值和产物检查

| 实际执行的检查 | 结果 |
| --- | --- |
| `npm run check:docs` | 通过：主体仍为 29 教学节、87 问答；40 份页面源文件（含新增阶段导读与 404） |
| `npm run build` | 通过：40 个 HTML、Pagefind 搜索索引和 sitemap |
| `npm run check:built` | 通过：本地页面、锚点、CSS/JS/图片资源无缺失；86 个 KaTeX 实例无 parse error；29 组各有 3 个折叠答案 |
| `python -m compileall -q examples` | 通过 |
| 附录中的 CLI 默认值对照 parse_args | 21 个已列字段一致 |
| 数学 / 单位核对 | 真实 Hi!/OK 字节 ID、100/900 加权均值 3.8、假设 token loss 1.386294、437,888 参数、token 与 PPL 记录均一致 |
| 参考 JSON 对当前训练源码 | code_sha256 一致；逐点 token 与 exp(val_loss) 一致 |
| 预算输入默认约束 | 默认值均满足 min/max/step；将与 min=0.001 不对齐的定步长改为 step=any，GPU 数仍要求整数 |
| 关键引用 | 已阅读并核对 [SOURCE_AUDIT.md](SOURCE_AUDIT.md) 所列支持位置；不以 HTTP 状态代替论断核验 |

构建仍有依赖级 `use astro:head-inject` 等警告，但没有构建失败或生成公式错误。静态检查能发现缺失链接、资源和结构，不能替代真实浏览器的布局与交互检查。

## 明确未执行的事项

- 浏览器工具库存仅有空的 MCP Apps 表面，没有可操作的普通浏览器标签。因此未执行网站桌面 / 移动视口截图、折叠答案点击、主题 / 搜索或预算输入交互验收；只做了生成 HTML、样式引用和默认输入约束检查。
- 没有运行领域配比 A/B、模板 / 主题留出对照、生产规模预算校准。正文全部按实验设计或假设标明，不填写结果。
- 没有运行低精度、状态分片、TP/PP/CP、长上下文扩展，也没有复现 OLMo 2 / SmolLM3 的模型或分数。
- GitHub Actions 发布后是否成功，以该次工作流和线上检查为准。
