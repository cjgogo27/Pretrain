import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { unified } from '@astrojs/markdown-remark';

export default defineConfig({
  site: 'https://cjgogo27.github.io',
  base: '/Pretrain',
  trailingSlash: 'always',
  markdown: { processor: unified({ remarkPlugins: [remarkMath], rehypePlugins: [rehypeKatex] }) },
  integrations: [starlight({
    title: '《听懂大模型预训练》',
    description: '从原始文本到模型权重：大模型预训练的原理、系统与工程实践。',
    locales: { root: { label: '简体中文', lang: 'zh-CN' } },
    social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/cjgogo27/Pretrain' }],
    editLink: { baseUrl: 'https://github.com/cjgogo27/Pretrain/edit/main/src/content/docs/' },
    sidebar: [
      { label: '从这里开始', items: [{ autogenerate: { directory: '00-reading-guides' } }] },
      { label: '第 1 章 系统总览', items: [{ autogenerate: { directory: '01-system-overview' } }] },
      { label: '第 2 章 数据与分词', items: [{ autogenerate: { directory: '02-data-tokenization' } }] },
      { label: '第 3 章 模型与计算', items: [{ autogenerate: { directory: '03-model-compute' } }] },
      { label: '第 4 章 优化与分布式训练', items: [{ autogenerate: { directory: '04-training-distributed' } }] },
      { label: '第 5 章 评估与训练诊断', items: [{ autogenerate: { directory: '05-evaluation-diagnosis' } }] },
      { label: '第 6 章 软件工具与实战', items: [{ autogenerate: { directory: '06-tools-practice' } }] },
      { label: '附录', items: [{ autogenerate: { directory: '07-appendix' } }] },
    ],
    tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
    pagination: true,
    customCss: ['./src/styles/custom.css'],
  })],
});
