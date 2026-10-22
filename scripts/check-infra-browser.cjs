// Optional local browser QA; prepare Playwright/Chromium and a built-site preview.
const fs=require('node:fs/promises');
const path=require('node:path');
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE||'playwright');
const root=path.resolve(__dirname,'..');
const folder=path.resolve(root,process.env.BROWSER_CHECK_OUT||'runs/infra-browser-checks');
const base=process.env.BROWSER_CHECK_BASE||'http://127.0.0.1:4326/Pretrain/';

async function main(){
  await fs.mkdir(folder,{recursive:true});
  const browser=await chromium.launch({headless:true,...(process.env.BROWSER_EXECUTABLE?{executablePath:process.env.BROWSER_EXECUTABLE}:{})});
  const checks=[],errors=[];
  try{
    for(const viewport of [{width:1280,height:900},{width:390,height:844}]){
      const context=await browser.newContext({viewport,reducedMotion:'reduce'});
      const page=await context.newPage();page.on('pageerror',error=>errors.push(error.message));
      const go=async(route)=>{await page.goto(base+route,{waitUntil:'networkidle'});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true,route);};
      await go('00-reading-guides/05-data-and-infrastructure/');
      assert.equal(await page.locator('.data-infra-map .tracks section').count(),3);
      if(viewport.width===1280){
        const cards=await page.locator('.data-infra-map .tracks section').all();
        const a=await cards[0].boundingBox(),b=await cards[1].boundingBox();
        assert.ok(Math.abs(a.y-b.y)<1,'Desktop data/infra cards should align');
      }
      assert.match(await page.locator('main').innerText(),/提交游标|数据游标/);
      checks.push({viewport:viewport.width,name:'data_learning_infra_map_and_page_width',status:'PASS'});
      await go('04-training-distributed/05-tensor-pipeline-context-parallel/');
      const widget=page.locator('[data-five-d]');
      assert.equal(await widget.locator('[data-rank]').count(),32);
      const first=await widget.locator('[data-rank="0"]').boundingBox();
      const second=await widget.locator('[data-rank="1"]').boundingBox();
      assert.ok(Math.abs(first.y-second.y)<1,'Rank buttons in a row should align');
      assert.match(await widget.locator('output').innerText(),/\[0, 1\]/);
      for(const [axis,expected,count] of [
        ['attention:CP','[0, 2]',2],['attention:DP','[0, 4, 8, 12]',4],
        ['expert:EP','[0, 1, 2, 3, 4, 5, 6, 7]',8],['expert:EDP','[0, 8]',2],['expert:PP','[0, 16]',2]
      ]){
        await widget.locator('select').selectOption(axis);
        assert.ok((await widget.locator('output').innerText()).includes(expected));
        assert.equal(await widget.locator('button.member').count(),count);
      }
      await widget.locator('[data-rank="31"]').click();
      await widget.locator('select').selectOption('attention:DP');
      assert.match(await widget.locator('output').innerText(),/\[19, 23, 27, 31\]/);
      await widget.locator('select').selectOption('expert:ETP');
      assert.match(await widget.locator('output').innerText(),/\[31\]/);
      // Native buttons must also work with keyboard activation.
      await widget.locator('[data-rank="0"]').focus();await page.keyboard.press('Enter');
      assert.equal(await widget.locator('[data-rank="0"]').getAttribute('aria-pressed'),'true');
      checks.push({viewport:viewport.width,name:'five_d_membership_coordinate_and_keyboard',status:'PASS'});
      await page.getByText('16个token选2个专家，为什么会有32次分配却没有32个新token？',{exact:true}).click();
      const detail=page.locator('details').filter({has:page.getByText('16个token选2个专家，为什么会有32次分配却没有32个新token？',{exact:true})});
      assert.notEqual(await detail.getAttribute('open'),null);assert.match(await detail.innerText(),/22次/);
      const answer=page.locator('.section-qa-answer details').first();await answer.locator('summary').click();assert.notEqual(await answer.getAttribute('open'),null);
      checks.push({viewport:viewport.width,name:'advanced_details_and_role_answer',status:'PASS'});
      await widget.locator('select').selectOption('expert:EP');
      await widget.evaluate(el=>window.scrollTo(0,el.getBoundingClientRect().top+scrollY-130));
      await page.screenshot({path:path.join(folder,`topology-${viewport.width}.png`)});
      let opened=false;
      if(await page.locator('starlight-theme-select select:visible').count()===0){await page.getByRole('button',{name:'菜单',exact:true}).click();opened=true;}
      await page.locator('starlight-theme-select select:visible').first().selectOption('dark');
      assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
      if(opened)await page.getByRole('button',{name:'菜单',exact:true}).click();
      await widget.evaluate(el=>window.scrollTo(0,el.getBoundingClientRect().top+scrollY-130));
      await page.screenshot({path:path.join(folder,`topology-dark-${viewport.width}.png`)});
      checks.push({viewport:viewport.width,name:'theme_and_responsive_topology',status:'PASS'});
      await go('04-training-distributed/06-checkpoint-and-recovery/');
      assert.match(await page.locator('main').innerText(),/131,072/);
      await go('06-tools-practice/03-single-to-multi-gpu/');
      assert.match(await page.locator('main').innerText(),/9\.490718672634083e-16/);
      await go('05-evaluation-diagnosis/04-throughput-and-profiling/');
      assert.match(await page.locator('main').innerText(),/48 GPU小时/);
      checks.push({viewport:viewport.width,name:'elastic_evidence_and_cost_examples_rendered',status:'PASS'});
      await context.close();
    }
    assert.deepEqual(errors,[]);
    const result={status:'PASS',browser:await browser.version(),viewports:[1280,390],checks,count:checks.length,page_errors:errors,
                  scope:'Local built site: data/infra map, topology groups, details, role answers, themes, changed pages and overflow. No GPU/browser performance or all-page/search audit.'};
    await fs.writeFile(path.join(folder,'report.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
  }finally{await browser.close();}
}
main().catch(error=>{console.error(error);process.exitCode=1;});
