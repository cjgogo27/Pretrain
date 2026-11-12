// Optional Playwright QA against the static build, including no-JS reading.
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const base=process.env.BROWSER_CHECK_BASE||'http://127.0.0.1:4331/Pretrain/';
const out=process.env.BROWSER_CHECK_OUT||'runs/bilingual-browser';
const pre=base.includes('/Pretrain/');
async function main(){
  await fs.mkdir(out,{recursive:true});
  const browser=await chromium.launch({headless:true,executablePath:process.env.BROWSER_EXECUTABLE});
  const checks=[],errors=[];
  try{
    for(const viewport of [{width:1280,height:900},{width:390,height:844}]){
      const context=await browser.newContext({viewport});const page=await context.newPage();page.on('pageerror',e=>errors.push(e.message));
      await page.goto(base+'00-reading-guides/00-outline/',{waitUntil:'networkidle'});
      const button=page.locator('[data-language-switch]');assert.ok(await button.isVisible());assert.equal((await button.innerText()).trim(),'English');
      await page.evaluate(()=>location.hash='系统架构');await button.click();await page.waitForURL('**/en/00-reading-guides/00-outline/**');
      assert.equal(await page.locator('html').getAttribute('lang'),'en');assert.equal(decodeURIComponent(new URL(page.url()).hash),'#系统架构');
      assert.equal((await page.locator('[data-language-switch]').innerText()).trim(),'中文');
      assert.match(await page.locator('h1').innerText(),/Chapters|Outline|Architecture/i);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);
      checks.push({viewport:viewport.width,name:'same_chapter_switch_and_anchor',status:'PASS'});
      await page.screenshot({path:path.join(out,`outline-en-${viewport.width}.png`)});
      await page.locator('[data-language-switch]').click();await page.waitForURL('**/00-reading-guides/00-outline/**');assert.equal(await page.locator('html').getAttribute('lang'),'zh-CN');
      const route=pre?'04-training-distributed/05-tensor-pipeline-context-parallel/':'04-online-rl-distillation/02-ppo-advantage-and-kl/';
      await page.goto(base+'en/'+route,{waitUntil:'networkidle'});
      const answer=page.locator('.section-qa-answer details').first();await answer.locator('summary').click();assert.notEqual(await answer.getAttribute('open'),null);
      assert.doesNotMatch(await answer.innerText(),/[\u3400-\u9fff]/u);
      if(pre){const widget=page.locator('[data-five-d]');await widget.locator('select').selectOption('attention:DP');assert.match(await widget.locator('output').innerText(),/\[0, 4, 8, 12\]/);assert.doesNotMatch(await widget.locator('output').innerText(),/[\u3400-\u9fff]/u);}
      else{const widget=page.locator('[data-ppo-explorer]');await widget.locator('[data-case="high-negative"]').click();assert.match(await widget.locator('output').innerText(),/objective=-1.50/);assert.doesNotMatch(await widget.locator('output').innerText(),/[\u3400-\u9fff]/u);}
      checks.push({viewport:viewport.width,name:'english_answers_and_live_widget',status:'PASS'});
      if(pre){await page.goto(base+'en/07-appendix/c-budget/',{waitUntil:'networkidle'});await page.locator('#budget-p').fill('16');assert.match(await page.locator('#budget-result').innerText(),/hours/);}
      else{
        await page.goto(base+'en/02-data-supervision/02-template-and-mask/',{waitUntil:'networkidle'});await page.locator('[data-token-explorer] [data-index="0"]').click();assert.match(await page.locator('[data-token-explorer] output').innerText(),/Position 0/);
        await page.goto(base+'en/03-supervised-preference/04-dpo-objective/',{waitUntil:'networkidle'});await page.locator('[data-preset="bothDown"]').click();assert.match(await page.locator('[data-preset-hint]').innerText(),/Both fall/);
        await page.goto(base+'en/04-online-rl-distillation/03-grpo-and-rlvr/',{waitUntil:'networkidle'});await page.locator('[data-rl-explorer] [data-step="3"]').click();assert.match(await page.locator('[data-rl-explorer] output').innerText(),/Mean=0.50/);
      }
      checks.push({viewport:viewport.width,name:'other_dynamic_messages_stay_english',status:'PASS'});
      if(viewport.width===1280){
        await page.locator('button[data-open-modal]').click();
        await page.locator('.pagefind-ui__search-input').fill('token');
        await page.locator('.pagefind-ui__result-link').first().waitFor();
        const hrefs=await page.locator('.pagefind-ui__result-link').evaluateAll(nodes=>nodes.map(n=>n.getAttribute('href')));
        assert.ok(hrefs.length>0);assert.ok(hrefs.every(href=>href.includes('/en/')));
        checks.push({name:'english_search_returns_english_pages',status:'PASS'});
      }
      await context.close();
    }
    const context=await browser.newContext({javaScriptEnabled:false});const page=await context.newPage();await page.goto(base+'en/00-reading-guides/00-outline/');assert.equal(await page.locator('html').getAttribute('lang'),'en');assert.ok(await page.locator('[data-language-switch]').isVisible());assert.match(await page.locator('h1').innerText(),/Chapters|Outline|Architecture/i);checks.push({name:'english_reading_without_javascript',status:'PASS'});await context.close();
    assert.deepEqual(errors,[]);
    const report={status:'PASS',base,browser:await browser.version(),checks,errors,count:checks.length};await fs.writeFile(path.join(out,'report.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
  }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exit(1);});
