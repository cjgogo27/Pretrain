import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {groupFor} from '../src/lib/topology.mjs';
const report=JSON.parse(await readFile(new URL('../examples/results/topology-reference.json',import.meta.url),'utf8'));
const l=report.layout;
test('Same rank has different layer-specific groups; EP is not an extra world factor',()=>{
  assert.deepEqual(groupFor(l.attention_groups,'DP',0),[0,4,8,12]);
  assert.deepEqual(groupFor(l.expert_groups,'EP',0),[0,1,2,3,4,5,6,7]);
  assert.deepEqual(groupFor(l.expert_groups,'EDP',0),[0,8]);
  assert.equal(l.world_size,32);assert.equal(l.wrong_five_factor_product,256);
});
test('Both layer mappings preserve identical PP groups',()=>{
  assert.deepEqual(l.attention_groups.PP,l.expert_groups.PP);
  assert.deepEqual(groupFor(l.attention_groups,'DP',31),[19,23,27,31]);
});
test('Batch counts original targets, routing counts assignments',()=>{
  assert.equal(l.tokens_per_update.effective_targets,131072);
  const r=report.routing;
  assert.equal(r.original_tokens,16);assert.equal(r.topk*r.original_tokens,32);
  assert.equal(r.capacity_per_expert,5);assert.equal(r.cases.concentrated.overflow_assignments,22);
});
test('Unknown or overlapping group ownership is rejected',()=>{
  assert.throws(()=>groupFor(l.attention_groups,'EP',0));
  assert.throws(()=>groupFor(l.attention_groups,'TP',32));
  assert.throws(()=>groupFor({X:[[0],[0]]},'X',0));
});
