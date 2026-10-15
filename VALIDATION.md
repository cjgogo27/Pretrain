# 实际验证记录

日期：2026-10-15（Asia/Shanghai）。本记录只描述实际执行的检查，不将教学实验解释为大模型能力结果。

## 环境

- Windows，Python 3.12.5，PyTorch 2.11.0+cpu。
- Node.js 24.14.0，npm 11.9.0。
- Astro 7.3.1，Starlight 0.42.0；站点完整依赖由 package-lock.json 固定。
- CPU 教学训练使用 1 个 PyTorch 线程、FP32、固定种子。

## 文档与静态站点

| 检查 | 结果 |
| --- | --- |
| `npm run check:docs` | 通过：29 节、87 角色问答、39 份页面源文件（含自定义 404） |
| `npm run build` | 通过：39 个 HTML、Pagefind 搜索索引、sitemap |
| 构建后内部链接与 CSS/JS/favicon 文件 | 全量检查通过，无缺失目标 |
| 数学公式 | 82 个 KaTeX 实例，无 katex-error 节点 |
| 教学节问答 | 每节开头 3 个问题，末尾 3 个折叠答案，均存在 |
| 技术参考链接 | 52 个去重 URL 均完成 HTTP 可达检查；1 个初次 TLS 失败，重试返回 200 |
| 本地代表页 HTTP | 系统架构页返回 200 |

源码检查统计约 2.8 万汉字，不含独立 JSON 中的问答答案。HTTP 可达只证明链接能访问，不代替逐条论断与原文之间的学术核验。

Astro/Vite 构建输出包含 `use astro:head-inject` 的依赖级警告，但构建完成；已核对生成 HTML 中的样式、脚本和公式资源引用。当前会话没有可用的浏览器自动化表面，未完成截图、移动视口和交互点击验收，因此不声称已做视觉逐页审阅。搜索与主题切换由 Starlight 提供，搜索索引成功生成；预算交互已构建，运行时交互仍需浏览器验证。

## 默认 CPU 教学训练

命令：

```bash
python examples/prepare_data.py --out runs/data
python examples/train.py --data runs/data --out runs/tiny --steps 100
```

原创语料：200 个训练文档、40 个验证文档；默认模型实际参数数 437,888。每次更新消费 256 个字节 token，100 步累计 25,600 个训练 token。

| 更新步 | 消费 token | 固定抽样窗口验证 loss | PPL |
| --- | --- | --- | --- |
| 20 | 5,120 | 4.682485 | 108.0382 |
| 40 | 10,240 | 4.074328 | 58.8109 |
| 60 | 15,360 | 3.687630 | 39.9501 |
| 80 | 20,480 | 3.512109 | 33.5189 |
| 100 | 25,600 | 3.442558 | 31.2668 |

成功生成 metrics.jsonl、latest.pt 与 sample.txt。验证集合很小且与训练共享教学主题；结果仅说明当前实现能够学习这类数据，不支持真实任务泛化结论。CPU 短跑吞吐不外推到 GPU 或大模型。

## 因果性与恢复验证

```bash
python examples/verify_resume.py
```

通过以下检查：

1. 改变未来 token 后，前半段 logits 保持不变，符合因果可见性。
2. 输出形状为批次 × 序列 × 257。
3. 固定 12 步总计划，连续运行与第 6 步中断恢复的最终模型、优化器逐元素一致。
4. RNG、训练采样器、步数、数据身份与 token 计数一致；全局累计 768 token。
5. 修改训练文件后，恢复以 fingerprint mismatch 明确拒绝。

上述一致性结论限于本次固定 CPU 环境，不保证跨硬件或版本一致。

## 未完成的运行验证

- 本机尝试 Windows 两进程 CPU DDP，启动器在进入训练脚本前因 PyTorch 构建缺少 libuv 支持而失败；关闭 USE_LIBUV 的尝试也未通过。DDP 代码路径尚未在此环境验证。
- 未运行 CUDA、NCCL、多机、ZeRO/FSDP、TP/PP/CP 或低精度训练。后四类生产机制为概念讲解，并非 TinyPretrain 的实现能力。
- 预算计算器的 1B / 20B / 8 卡 / 100 TFLOP/s 是假设算例，不是本项目的已执行训练。

## 发布

GitHub Actions 在每次 main 推送后独立运行文档检查与构建。线上可访问状态与 Actions 记录应以仓库实际结果为准。
