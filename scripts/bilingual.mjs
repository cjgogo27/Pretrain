// Deterministic local English editions. No translation calls during build or reading.
// Chinese MDX stays authoritative; missing/changed text fails the release build.
import { parse, serialize } from 'parse5';
import { createHash } from 'node:crypto';
import { readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const hasChinese = value => /[\u3400-\u9fff]/u.test(value);
export const keyFor = value => createHash('sha256').update(value).digest('hex').slice(0,24);
const excluded = new Set(['script','style','code','pre','math','annotation','textarea']);
const attributes = new Set(['title','alt','aria-label','placeholder','label','data-label']);
const attr = (node,name) => node.attrs?.find(item=>item.name===name)?.value;
const set = (node,name,value) => {
  const item=node.attrs?.find(item=>item.name===name);
  if(item) item.value=value; else (node.attrs??=[]).push({name,value});
};
export function visitText(node, fn) {
  if(excluded.has(node.tagName) || attr(node,'translate')==='no' || /(?:^|\s)katex(?:\s|$)/.test(attr(node,'class')||'')) return;
  if(node.nodeName==='#text' && hasChinese(node.value)) fn(node.value.trim(),value=>{node.value=node.value.replace(node.value.trim(),value);});
  for(const item of node.attrs||[]) if(attributes.has(item.name)&&hasChinese(item.value)) fn(item.value,value=>{item.value=value;});
  if(node.tagName==='meta' && ['description','og:title','og:description','og:site_name','twitter:title','twitter:description'].includes(attr(node,'name')||attr(node,'property')) && hasChinese(attr(node,'content')||'')) {
    const text=attr(node,'content');fn(text,value=>set(node,'content',value));
  }
  if(node.tagName==='site-search') {
    const translations=JSON.parse(attr(node,'data-translations')||'{}');
    for(const key of Object.keys(translations)) if(hasChinese(translations[key])) fn(translations[key],value=>{translations[key]=value;});
    set(node,'data-translations',JSON.stringify(translations));
  }
  for(const child of node.childNodes||[]) visitText(child,fn);
}
export async function htmlFiles(root) {
  const result=[];
  for(const item of await readdir(root,{withFileTypes:true})) {
    if(['en','midtraining','pagefind','_astro'].includes(item.name)) continue;
    const file=path.join(root,item.name);
    if(item.isDirectory()) result.push(...await htmlFiles(file)); else if(file.endsWith('.html')) result.push(file);
  }
  return result.sort();
}
function walk(node,fn){fn(node);for(const child of node.childNodes||[])walk(child,fn);}
const pagePath = relative => relative==='index.html'?'':relative==='404.html'?relative:relative.replace(/index\.html$/,'').replace(/\.html$/, '/');

export async function run(mode='build') {
  const pkg=JSON.parse(await readFile('package.json','utf8'));
  const base=pkg.name.startsWith('pretrain-')?'/Pretrain/':'/postraining/';
  const root=path.resolve('dist');
  const files=await htmlFiles(root);
  const catalogPath='src/i18n/en.json';
  const overrides=JSON.parse(await readFile('src/i18n/overrides.json','utf8'));
  let catalog={schema:1,sourceLanguage:'zh-CN',targetLanguage:'en',entries:{}};
  try {catalog=JSON.parse(await readFile(catalogPath,'utf8'));} catch(error){if(mode!=='collect')throw error;}
  const missing=new Map();
  const documents=[];
  for(const file of files) {
    const doc=parse(await readFile(file,'utf8'));
    visitText(doc,(text)=>{if(!overrides[text]&&!catalog.entries[keyFor(text)]?.en)missing.set(keyFor(text),text);});
    documents.push([file,doc]);
  }
  if(mode==='collect') {
    await mkdir('runs',{recursive:true});
    await writeFile('runs/translation-needed.json',JSON.stringify(Object.fromEntries(missing),null,2)+'\n');
    console.log(JSON.stringify({pages:files.length,missing:missing.size,characters:[...missing.values()].join('').length}));return;
  }
  if(missing.size) throw new Error(`${missing.size} new/changed text segments need English translations. Run npm run i18n:collect after the Chinese build.`);
  const known=new Set(documents.map(([f])=>base+pagePath(path.relative(root,f).replaceAll('\\','/'))));
  let translated=0;
  for(const [file,doc] of documents) {
    const relative=path.relative(root,file).replaceAll('\\','/');
    const chinese=base+pagePath(relative),english=base+'en/'+pagePath(relative);
    // Add reciprocal language metadata to the original without changing its content.
    const head=doc.childNodes.find(n=>n.tagName==='html').childNodes.find(n=>n.tagName==='head');
    head.childNodes=head.childNodes.filter(n=>!(n.tagName==='link'&&attr(n,'rel')==='alternate'&&['en','zh-CN'].includes(attr(n,'hreflang'))));
    const fragment=source=>parse(source).childNodes.find(n=>n.tagName==='html').childNodes.find(n=>n.tagName==='head').childNodes;
    head.childNodes.push(...fragment(`<link rel="alternate" hreflang="zh-CN" href="https://cjgogo27.github.io${chinese}"><link rel="alternate" hreflang="en" href="https://cjgogo27.github.io${english}">`));
    walk(doc,node=>{if(node.tagName==='a'&&attr(node,'data-language-switch')!==undefined)set(node,'href',english);});
    await writeFile(file,serialize(doc));
    // Starlight protects its brand span from browser translation. This is our
    // own published English title, so replace that span explicitly.
    walk(doc,node=>{if(node.tagName==='span'&&/(?:^|\s)site-title(?:\s|$)/.test(attr(node.parentNode,'class')||'')) {
      set(node.parentNode,'data-mobile-title',base==='/Pretrain/'?'LLM Pretraining':'LLM Post-Training');
      for(const child of node.childNodes||[]) if(child.nodeName==='#text'&&hasChinese(child.value)) {
        const source=child.value.trim();const title=overrides[source]||catalog.entries[keyFor(source)]?.en||source;
        child.value=child.value.replace(source,title);set(node.parentNode,'aria-label',title);
      }
    }});
    visitText(doc,(text,replace)=>{const record=catalog.entries[keyFor(text)];if(!overrides[text]&&record.zh!==text)throw new Error('Translation source mismatch');replace(overrides[text]||record.en);translated++;});
    walk(doc,node=>{
      if(node.tagName==='html')set(node,'lang','en');
      if(node.tagName==='site-search') {
        const data=JSON.parse(attr(node,'data-translations')||'{}');
        for(const key of Object.keys(data)) {
          const record=catalog.entries[keyFor(String(data[key]))];
          if(record) data[key]=record.en;
        }
        set(node,'data-translations',JSON.stringify(data));
      }
      if(node.tagName==='a' && attr(node,'data-language-switch')!==undefined) {
        set(node,'href',chinese);set(node,'lang','zh-CN');set(node,'hreflang','zh-CN');set(node,'aria-label','切换为中文');
        const span=node.childNodes?.find(n=>n.tagName==='span');if(span)span.childNodes=[{nodeName:'#text',value:'中文',parentNode:span}];
      } else if(node.tagName==='a') {
        const href=attr(node,'href');
        if(href) {
          const url=new URL(href,'https://cjgogo27.github.io'+chinese);
          if(url.origin==='https://cjgogo27.github.io' && known.has(url.pathname))set(node,'href',base+'en/'+url.pathname.slice(base.length)+url.search+url.hash);
        }
      }
      if(node.tagName==='link' && attr(node,'rel')==='canonical')set(node,'href','https://cjgogo27.github.io'+english);
      if(node.tagName==='meta' && ['og:url'].includes(attr(node,'property')))set(node,'content','https://cjgogo27.github.io'+english);
      if(node.tagName==='meta' && attr(node,'property')==='og:locale')set(node,'content','en_US');
      if(node.tagName==='meta' && ['description','og:title','og:description','twitter:title','twitter:description'].includes(attr(node,'name')||attr(node,'property'))) {
        const text=attr(node,'content');const record=catalog.entries[keyFor(text||'')];if(record)set(node,'content',record.en);
      }
    });
    // A small, explicit editorial note; code and exact quoted examples retain their source form.
    walk(doc,node=>{if(node.tagName==='main') {
      const note=parse('<!doctype html><html><body><p class="translation-note" data-pagefind-ignore>Machine-translated English edition with targeted terminology review. Code and formulas are preserved. For technical ambiguity, consult the <a href="'+chinese+'">Chinese original</a>.</p></body></html>').childNodes.find(n=>n.tagName==='html').childNodes.find(n=>n.tagName==='body').childNodes[0];
      node.childNodes.unshift(note);
    }});
    const out=path.join(root,'en',relative);await mkdir(path.dirname(out),{recursive:true});await writeFile(out,serialize(doc));
  }
  // Search discovers English pages by html[lang]; rebuild both language indexes together.
  const pagefind=await import('pagefind');
  const {index,errors}=await pagefind.createIndex({});
  if(errors.length)throw new Error(errors.join('\n'));
  try {
    const added=await index.addDirectory({path:root});if(added.errors.length)throw new Error(added.errors.join('\n'));
    const written=await index.writeFiles({outputPath:path.join(root,'pagefind')});if(written.errors.length)throw new Error(written.errors.join('\n'));
  }finally{await pagefind.close();}
  const sitemap=path.join(root,'sitemap-0.xml');
  try {let xml=await readFile(sitemap,'utf8');xml=xml.replace(new RegExp('<url>\\s*<loc>https://cjgogo27\\.github\\.io'+base+'en/[^<]*</loc>[\\s\\S]*?</url>','g'),'');const urls=documents.filter(([f])=>!f.endsWith('404.html')).map(([f])=>'<url><loc>https://cjgogo27.github.io'+base+'en/'+pagePath(path.relative(root,f).replaceAll('\\','/'))+'</loc></url>').join('');xml=xml.replace('</urlset>',urls+'</urlset>');await writeFile(sitemap,xml);}catch(error){if(error.code!=='ENOENT')throw error;}
  console.log(JSON.stringify({status:'PASS',chinese_pages:files.length,english_pages:files.length,translated_segments:translated,external_runtime_translation:false}));
}
if(process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url))await run(process.argv[2]);
