import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve('src/content/docs');
async function walk(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  return (await Promise.all(entries.map(entry => entry.isDirectory() ? walk(path.join(dir, entry.name)) : path.join(dir, entry.name)))).flat();
}
const files = (await walk(root)).filter(file => /\.mdx?$/.test(file));
const routes = new Set(files.map(file => '/Pretrain/' + path.relative(root, file).replaceAll('\\', '/').replace(/\.mdx?$/, '').replace(/^index$/, '') + '/'));
routes.add('/Pretrain/');
const qa = JSON.parse(await readFile('src/data/section-qa.json', 'utf8'));
const chapters = JSON.parse(await readFile('src/data/chapters.json', 'utf8'));
const errors = [];
let characters = 0;
for (const file of files) {
  const text = await readFile(file, 'utf8');
  const relative = path.relative(root, file).replaceAll('\\', '/');
  characters += (text.match(/[\u3400-\u9fff]/g) ?? []).length;
  if (!/^---\r?\ntitle:/m.test(text)) errors.push(`${relative}: missing title`);
  if (/TODO|TBD|待补充|PLACEHOLDER/.test(text)) errors.push(`${relative}: unfinished placeholder`);
  for (const match of text.matchAll(/(?:\]\(|(?:href|link)=['"])(\/Pretrain\/[^)'"\s]*)/g)) {
    const route = match[1].split('#')[0];
    if (!routes.has(route)) errors.push(`${relative}: unknown internal link ${route}`);
  }
  if (/^0[1-6]-/.test(relative)) {
    const chapter = chapters.find(item => relative === item.path + '.mdx');
    if (!chapter) { errors.push(`${relative}: not indexed`); continue; }
    const id = chapter.title.split(' ')[0];
    const items = qa[id];
    if (!items || items.length !== 3 || items.some(item => !item.q || !item.a || !item.role)) errors.push(`${id}: expected three complete role QAs`);
    for (const marker of ['## 现场问题', '## 参考资料', '一句话听懂', `id="${id}" mode="ask"`, `id="${id}" mode="answer"`]) {
      if (!text.includes(marker)) errors.push(`${relative}: missing ${marker}`);
    }
    if ((text.match(/\[\d+\] \[/g) ?? []).length < 2) errors.push(`${relative}: expected primary references`);
  }
}
if (chapters.length !== 29 || Object.keys(qa).length !== 29) errors.push('Expected 29 sections and 29 QA groups');
if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
console.log(JSON.stringify({ status: 'PASS', pages: files.length, sections: chapters.length, role_questions: Object.values(qa).flat().length, chinese_characters: characters }));
