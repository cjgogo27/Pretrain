// Validate generated local targets/anchors/assets and role-answer structure.
// This inspects HTML; it is not a browser, viewport or interaction test.
import { readdir, readFile, stat } from 'node:fs/promises';
import path from 'node:path';
const root = path.resolve('dist');
async function walk(dir) {
  return (await Promise.all((await readdir(dir, {withFileTypes:true})).map(e => e.isDirectory() ? walk(path.join(dir,e.name)) : path.join(dir,e.name)))).flat();
}
const files = (await walk(root)).filter(f=>f.endsWith('.html'));
const documents = new Map(await Promise.all(files.map(async f=>[f,await readFile(f,'utf8')])));
const ids = html=>new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]));
const errors=[]; let formulaCount=0;
for (const [file, html] of documents) {
  const label=path.relative(root,file);
  const classes=[...html.matchAll(/\bclass="([^"]+)"/g)].map(m=>m[1].split(/\s+/));
  formulaCount+=classes.filter(list=>list.includes('katex')).length;
  if(classes.some(list=>list.includes('katex-error'))) errors.push(`${label}: formula parse error`);
  for(const match of html.matchAll(/\b(?:href|src)="([^"]+)"/g)) {
    const link=match[1];
    if(!link.startsWith('/Pretrain/')&&!link.startsWith('#')) continue;
    const url=new URL(link,'https://local.invalid/Pretrain/');
    let target=link.startsWith('#')?file:path.join(root,decodeURIComponent(url.pathname.slice('/Pretrain/'.length)));
    try {
      if((await stat(target)).isDirectory()) target=path.join(target,'index.html');
      await stat(target);
      const fragment=decodeURIComponent(url.hash.slice(1));
      if(fragment && documents.has(target) && !ids(documents.get(target)).has(fragment)) errors.push(`${label}: missing anchor ${link}`);
    } catch { errors.push(`${label}: missing local target ${link}`); }
  }
}
const chapters=JSON.parse(await readFile('src/data/chapters.json','utf8'));
for(const chapter of chapters) {
  const html=documents.get(path.join(root,chapter.path,'index.html'))||'';
  const answer=html.split('section-qa-answer')[1]?.split('</aside>')[0]||'';
  if((answer.match(/<details\b/g)||[]).length!==3) errors.push(`${chapter.path}: expected three role answers`);
}
if(errors.length) { console.error(errors.join('\n'));process.exit(1); }
console.log(JSON.stringify({status:'PASS',html_pages:files.length,katex_formulas:formulaCount,missing_targets_or_anchors:0,role_answer_groups:chapters.length,browser_check:false}));
