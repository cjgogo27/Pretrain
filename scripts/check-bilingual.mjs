import { parse, serialize } from 'parse5';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {htmlFiles,visitText} from './bilingual.mjs';
const attr=(node,name)=>node.attrs?.find(a=>a.name===name)?.value;
function all(node,predicate,out=[]){if(predicate(node))out.push(node);for(const child of node.childNodes||[])all(child,predicate,out);return out;}
const files=await htmlFiles('dist');
let code=0,math=0,answers=0;
for(const file of files){
  const relative=path.relative('dist',file);
  const zh=parse(await readFile(file,'utf8'));
  const en=parse(await readFile(path.join('dist/en',relative),'utf8'));
  const html=all(en,n=>n.tagName==='html')[0];assert.equal(attr(html,'lang'),'en',file);
  const brand=all(en,n=>/(?:^|\s)site-title(?:\s|$)/.test(attr(n,'class')||''))[0];
  assert.doesNotMatch(serialize(brand),/[\u3400-\u9fff]/u,`${file}: site title should be English`);
  const switchEN=all(zh,n=>n.tagName==='a'&&attr(n,'data-language-switch')!==undefined)[0];
  const switchZH=all(en,n=>n.tagName==='a'&&attr(n,'data-language-switch')!==undefined)[0];
  assert.ok(switchEN&&switchZH,file);assert.ok(attr(switchEN,'href').includes('/en/'),file);assert.ok(!attr(switchZH,'href').includes('/en/'),file);
  const remaining=[];visitText(en,(text)=>remaining.push(text));assert.deepEqual(remaining,[],`${file}: untranslated visible text`);
  for(const tag of ['pre','math']){
    const before=all(zh,n=>n.tagName===tag).map(serialize);
    const after=all(en,n=>n.tagName===tag).map(serialize);
    assert.deepEqual(after,before,`${file}: ${tag} content changed`);
    if(tag==='pre')code+=before.length;else math+=before.length;
  }
  const qa=all(en,n=>/\bsection-qa-answer\b/.test(attr(n,'class')||''));
  for(const item of qa){assert.equal(all(item,n=>n.tagName==='details').length,3,file);answers++;}
  const original=all(en,n=>attr(n,'class')==='translation-note')[0];assert.ok(original,file);
}
console.log(JSON.stringify({status:'PASS',paired_pages:files.length,unchanged_code_blocks:code,unchanged_math_blocks:math,english_role_answer_groups:answers,untranslated_visible_segments:0}));
