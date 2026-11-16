import {icon} from './icons.js';
import {escape,markdown,renderMessage} from './markdown.js?v=20261013-teaching-panels';
import {teachingFeedback} from './teaching-workspace.js?v=20261013-teaching-panels';
import {renderLearningAnalytics} from './learning-visuals.js?v=20261009-guided-learning';
import {teachingReferenceCards,bibliographyText,updateReferenceLinks} from './reference-cards.js?v=20261011-reference-cleanup';

const outcomes={covered:'本题要点已覆盖',needs_review:'待复习',in_progress:'进行中',interrupted:'已中断',unassessed:'尚无评价'};
const modules={specialty:'MDRPI专项案例训练',foundation:'基础知识练习'};
const populations={adult:'成人',pediatric:'儿童',neonatal:'新生儿'};
const devices={niv_mask:'无创通气面罩',nasogastric_tube:'鼻胃管',endotracheal_tube:'气管插管',tracheostomy_tube:'气管切开套管',pulse_oximeter:'血氧探头',cervical_collar:'颈托',cast:'石膏/支具',nasal_cannula:'鼻导管',nasojejunal_tube:'鼻空肠管',nasal_prong_cpap:'鼻塞式CPAP',bite_block:'口咬器',general_mdrpi:'其他器械'};
const options=(values,selected)=>Object.entries(values).map(([value,title])=>`<option value="${escape(value)}" ${value===selected?'selected':''}>${escape(title)}</option>`).join('');
const field=(id,title,value,help='')=>`<label class="field"><span>${escape(title)}</span><input id="${id}" value="${escape(value||'')}" maxlength="80">${help?`<small>${escape(help)}</small>`:''}</label>`;
const button=(label,command,extra='',primary=false)=>`<button type="button" class="button ${primary?'primary':''}" data-command="${command}" ${extra}>${label}</button>`;
const when=value=>value?new Date(value).toLocaleString('zh-CN',{year:'numeric',month:'long',day:'numeric',hour:'2-digit',minute:'2-digit',timeZoneName:'short'}):'尚未安排';
const emptyCatalogFilters=()=>({population:'',device_type:'',difficulty:'',question_type:'',learning_topic:''});

export function createPanels(C) {
  let catalog=[],category='specialty',panelSid='',catalogRequest=0,facets=null,learningRequest=0,recommendationRequest=0,launchingRecommendation=false,openingCaseSetup=false;
  let catalogQuery='',catalogLoading=false,catalogFiltersOpen=!(globalThis.matchMedia?.('(max-width:760px)')?.matches);
  let filters=emptyCatalogFilters();

  function learnerAccessCode() {
    let value=localStorage.getItem('mdrpi.learner-key')||'';
    if(!/^[a-f0-9]{64}$/.test(value)){
      value=Array.from(crypto.getRandomValues(new Uint8Array(32)),part=>part.toString(16).padStart(2,'0')).join('');
      localStorage.setItem('mdrpi.learner-key',value);
    }
    return value;
  }

  async function copyLearningAccess() {
    const input=document.querySelector('#setting-access-code');
    if(!input)return;
    if(!/^[a-f0-9]{64}$/.test(input.value.trim())){C.toast('访问码应为64位十六进制字符。',true);return;}
    try {
      if(!navigator.clipboard?.writeText)throw new Error('clipboard_unavailable');
      await navigator.clipboard.writeText(input.value.trim());
      C.toast('访问码已复制，可在另一浏览器的教学设置中恢复同名档案。');
    }catch(error){C.toast('当前浏览器未允许复制，请点“显示访问码”后手动复制。',true);}
  }

  async function caseSetup(action) {
    if(!['choose_case','choose_topic'].includes(action))throw new Error('未识别的案例准备方式。');
    if(openingCaseSetup)return;
    openingCaseSetup=true;
    try{await C.startTurn('',action);}finally{openingCaseSetup=false;}
    const state=C.S.view?.state,generated=action==='choose_topic',expected=generated?'topic':'case';
    if(!state||state.stage!==expected||state.error){C.toast(state?.error||C.S.view?.connection_error||'请先结束当前练习，再开始新的案例。',true);return;}
    const setupSid=state.session_id,spec=state.case_spec||{},title=generated?'定制案例训练':'导入自有脱敏案例';
    C.openDialog(title,`<p class="panel-note">${generated?'生成内容标注为“AI生成·仅供练习”，通过统一六步流程训练。':'仅用于教学，请先删除姓名、身份证号、住院号、联系方式、详细地址及其他可识别患者的信息。自动检查可能遗漏，请自行确认脱敏。'}</p><form id="case-setup-form">
      <label class="field"><span>练习主题</span><input id="case-topic" maxlength="300" value="${escape(spec.topic||'MDRPI风险与预防')}" required></label>
      <div class="case-spec-grid"><label class="field"><span>人群</span><select id="case-population">${options(populations,spec.population||'adult')}</select></label><label class="field"><span>医疗器械</span><select id="case-device">${options(devices,spec.device_type||'niv_mask')}</select></label><label class="field"><span>案例难度</span><select id="case-difficulty">${options({'基础':'基础','进阶':'进阶','复杂':'复杂'},spec.difficulty||'基础')}</select></label></div>
      <label class="field"><span>练习重点</span><textarea id="case-focus" rows="2" maxlength="600" placeholder="例如：观察皮肤变化、区分风险与损伤、保护器械治疗功能">${escape(spec.focus||'六步临床推理')}</textarea></label>
      ${generated?'<p class="case-origin-label">AI生成·仅供练习</p>':'<label class="field"><span>脱敏后的案例描述</span><textarea id="case-description" rows="7" maxlength="12000" required placeholder="填写人群、器械及使用过程、局部皮肤或黏膜变化、现有处理；未知信息可明确写“未提供”。"></textarea><small>缺少临床信息会提示补充，系统不会将未知内容编成患者事实。</small></label><label class="privacy-confirmation"><input id="case-privacy-confirmed" type="checkbox" required><span>我已检查并去除患者识别信息，确认此案例仅供教学练习。</span></label>'}
      <div id="case-intake-report" class="panel-note" aria-live="polite"></div><div id="case-setup-error" class="inline-error" role="alert" hidden></div><div class="form-footer">${button('取消','close-dialog')}<button type="submit" class="button primary">${generated?'生成并开始练习':'导入并开始练习'}</button></div></form>`,true);
    document.querySelector('#case-setup-form').addEventListener('submit',async event=>{
      event.preventDefault();const submit=event.submitter||event.currentTarget.querySelector('[type="submit"]'),errorBox=document.querySelector('#case-setup-error');
      if(submit.disabled)return;submit.disabled=true;
      try {
        if(C.S.view?.state.session_id!==setupSid||C.S.mode!=='teaching')throw new Error('会话已切换，请重新打开案例表单。');
        const read=id=>document.getElementById(id)?.value.trim()||'';
        const requested={topic:read('case-topic'),population:read('case-population'),device_type:read('case-device'),difficulty:read('case-difficulty'),focus:read('case-focus')};
        if(!requested.topic)throw new Error('请填写练习主题。');
        let text=generated?`练习主题：${requested.topic}；人群：${populations[requested.population]}；器械：${devices[requested.device_type]}；难度：${requested.difficulty}；重点：${requested.focus}`:read('case-description');
        if(!text)throw new Error('请填写脱敏后的案例描述。');
        if(!generated&&!document.getElementById('case-privacy-confirmed').checked)throw new Error('请检查并确认案例已去除患者识别信息。');
        errorBox.hidden=true;
        if(!generated){
          const intake=await C.api(`/sessions/${setupSid}/case-intake`,{text,case_spec:requested});
          if(C.S.view?.state.session_id!==setupSid)throw new Error('会话已切换，请重新打开案例表单。');
          text=intake.sanitized_text;
          const report=document.querySelector('#case-intake-report');
          if(report)report.textContent=`${intake.redaction_count?'已隐藏 '+intake.redaction_count+' 处可能识别患者的信息。':'已完成导入前隐私检查。'}${intake.missing_fields?.length?'未提供：'+intake.missing_fields.join('、')+'；训练中将保留为未知。':''}`;
          if(intake.scope_conflicts?.length){
            const details=intake.scope_conflicts.map(row=>row.field==='population'?`人群应与案例中的“${populations[row.explicit_case_value]||row.explicit_case_value}”一致`:`器械应与案例中的“${devices[row.explicit_case_value]||row.explicit_case_value}”一致`).join('；');
            document.getElementById('case-description').value=text;
            throw new Error('所选范围与案例中的明确描述不一致：'+details+'。请确认并修改后再导入。');
          }
          if(intake.redaction_count)C.toast('已隐藏 '+intake.redaction_count+' 处可能识别患者的信息，再进入教学流程。');
        }
        await C.startTurn(text,'send',{case_spec:requested,privacy_confirmed:true});
        if(C.S.view?.state.session_id!==setupSid)return;
        const result=C.S.view.state;
        if(result.error||C.S.view.connection_error||!result.clinical_flow)throw new Error(result.error||C.S.view.connection_error||'案例尚未准备完成，请补充信息后再提交。');
        C.closeDialog();
      }catch(error){if(errorBox.isConnected){errorBox.textContent=error.message;errorBox.hidden=false;}}
      finally{submit.disabled=false;}
    });
  }

  function settings() {
    const state=C.S.view?.state,mode=C.S.mode,values=state||C.defaults(mode);
    const openedAccessCode=mode==='teaching'?learnerAccessCode():'';
    panelSid=state?.session_id||'';
    C.openDialog('偏好与设置',`
      <div class="settings-section">外观</div><div class="theme-options">${[['light','sun','浅色'],['dark','moon','深色'],['system','settings','跟随系统']].map(([value,shape,label])=>`<button class="theme-choice ${C.appearance()===value?'selected':''}" data-theme-value="${value}">${icon(shape)}${label}</button>`).join('')}</div>
      <form id="settings-form"><div class="settings-section">${mode==='teaching'?'教学设置':mode==='performance'?'测试设置':'回答设置'}</div>
      ${mode==='performance'?`<label class="field"><span>联网方式</span><select id="setting-web-mode">${options({live:'实时联网',off:'关闭联网',replay:'使用已保存证据'},values.web_mode||'live')}</select></label>${field('setting-replay','回放记录编号',values.replay_run_id,'选择使用已保存证据时填写原始运行编号。')}`:'<p class="panel-note">资料范围：网页＋知识库。</p>'}
      ${mode==='chat'?field('setting-timezone','时间显示时区',values.timezone||'Asia/Shanghai','默认北京时间。也可填写 UTC、Europe/London 等时区。')+field('setting-memory','偏好档案名称（可选）',values.memory_profile,'同名会话共用偏好。只保存你明确要求记住的内容。'):''}
      ${mode==='teaching'?`<label class="field"><span>回复语言</span><select id="setting-language">${options({auto:'跟随我的输入',zh:'中文',en:'English'},values.response_language||'auto')}</select></label>`+field('setting-learning','学习档案名称（可选）',values.learning_profile,'填写后可跨会话续学并安排复习。开始作答后，需完成本题才能切换。'):''}
      ${mode==='teaching'?`<label class="field"><span>学习档案访问码</span><input id="setting-access-code" type="password" value="${escape(openedAccessCode)}" maxlength="64" minlength="64" pattern="[a-f0-9]{64}" autocomplete="off" spellcheck="false" required><small>在另一浏览器输入相同访问码和档案名称，可恢复你的学习记录。请妥善保存；持有此码可访问相应档案。更换访问码会开启新会话，旧记录保留。</small></label><div class="action-row">${button('显示访问码','toggle-learning-access','aria-controls="setting-access-code" aria-pressed="false"')}${button('复制访问码','copy-learning-access')}</div>`:''}
      <div id="settings-error" class="inline-error" hidden></div><div class="form-footer">${button('取消','close-dialog')}<button type="submit" class="button primary">保存设置</button></div></form>`);
    document.querySelector('#settings-form').addEventListener('submit',async event=>{
      event.preventDefault();const submit=event.submitter||event.currentTarget.querySelector('[type="submit"]');if(submit.disabled)return;submit.disabled=true;
      try {
        const current=C.S.view?.state; if((current?.session_id||'')!==panelSid||C.S.mode!==mode)throw new Error('会话已切换，请重新打开设置。');
        const read=id=>document.getElementById(id)?.value;
        const body={policy:mode==='performance'?(values.policy||'none'):'web',web_mode:read('setting-web-mode')||values.web_mode||'live',replay_run_id:read('setting-replay')??values.replay_run_id??''};
        if(mode==='performance'&&body.web_mode==='replay'&&!body.replay_run_id.trim())throw new Error('使用已保存证据时，请填写原始运行编号。');
        if(mode==='chat')Object.assign(body,{timezone:read('setting-timezone'),memory_profile:read('setting-memory')});
        if(mode==='teaching')Object.assign(body,{response_language:read('setting-language'),learning_profile:read('setting-learning')});
        const accessCode=mode==='teaching'?read('setting-access-code').trim():'',changed=mode==='teaching'&&accessCode!==openedAccessCode;
        if(mode==='teaching'&&!/^[a-f0-9]{64}$/.test(accessCode))throw new Error('访问码应为64位小写十六进制字符。');
        if(mode==='teaching'&&localStorage.getItem('mdrpi.learner-key')!==openedAccessCode)throw new Error('访问身份已改变，请重新打开教学设置。');
        if(changed){
          if(current?.pending||C.S.view?.active||C.S.jobs?.size||current?.attempts?.length||current?.clinical_flow?.steps?.some(step=>step.answers?.length))throw new Error('当前练习已有作答或正在处理，请先新建空白教学会话，再更换档案访问码；旧记录会保留。');
          const previousView=C.S.view,previousDefaults=C.defaults(mode);
          localStorage.setItem('mdrpi.learner-key',accessCode);
          C.saveDefaults(mode,{...previousDefaults,...body});C.S.view=null;
          try{await C.ensureSession();await C.refresh();await C.loadHistory();}
          catch(error){localStorage.setItem('mdrpi.learner-key',openedAccessCode);C.saveDefaults(mode,previousDefaults);C.S.view=previousView;throw error;}
          C.closeDialog();C.render();C.toast('访问码已应用，已开启新会话；可以查看同名学习档案。');
        }else{
          if(panelSid){await C.api(`/sessions/${panelSid}/settings`,body,'PUT');await C.refresh();}
          C.saveDefaults(mode,{...C.defaults(mode),...body});C.closeDialog();C.render();C.toast('设置已保存');
        }
      }catch(error){const node=document.querySelector('#settings-error');node.textContent=error.message;node.hidden=false;}
      finally{submit.disabled=false;}
    });
  }

  async function exercises() {
    C.openDialog('选择一道练习','<p class="panel-note">正在读取题库…</p>',true);
    await loadCatalog();
  }
  async function loadCatalog(focus='') {
    const request=++catalogRequest,query=new URLSearchParams({module:category});
    for(const [key,value] of Object.entries(filters))if(value)query.set(key,value);
    catalogLoading=true;
    const list=document.querySelector('#catalog-results');
    if(list){list.setAttribute('aria-busy','true');list.querySelectorAll('button').forEach(node=>node.disabled=true);}
    document.querySelectorAll('[data-catalog-filter]').forEach(node=>node.disabled=true);
    const count=document.querySelector('#catalog-count');if(count)count.textContent='正在更新练习…';
    try {
      const [rows,data]=await Promise.all([C.api('/practice-cases?'+query),C.api('/practice-cases/taxonomy?'+query)]);
      if(request!==catalogRequest||!catalogActive())return;
      catalog=rows;facets=data;catalogLoading=false;renderCatalog();
      if(focus)document.querySelector('#catalog-'+focus)?.focus();
    } catch(error) {
      if(request!==catalogRequest||!catalogActive())return;
      catalogLoading=false;
      document.querySelector('#dialog-body').innerHTML=`<p class="inline-error" role="alert">题库未能加载：${escape(error.message)}</p><div class="action-row">${button('重新加载题库','catalog')}${button('清除搜索与筛选','catalog-reset')}</div>`;
    }
  }
  function catalogActive(){return document.querySelector('#dialog').open&&document.querySelector('#dialog-title').textContent==='选择一道练习';}
  function catalogSelect(key,title,rows=[]) {
    const disabled=key==='device_type'&&!rows.length;
    return `<label class="field"><span>${title}</span><select id="catalog-${key}" data-catalog-filter="${key}" ${disabled?'disabled':''}><option value="">${disabled?'当前人群暂无器械':'全部'}</option>${rows.map(row=>`<option value="${escape(row.id)}" ${filters[key]===row.id?'selected':''}>${escape(row.label)} · ${key==='learning_topic'&&Number(row.count)===0?'当前筛选暂无题':row.count}</option>`).join('')}</select></label>`;
  }
  function catalogMatches(item) {
    const normalize=value=>String(value||'').normalize('NFKC').toLocaleLowerCase().replace(/[\s·／/—-]+/g,'');
    const haystack=normalize([item.id,item.title,...(item.population_labels||[]),...(item.device_labels||[]),item.question_type_label,item.learning_topic_label,item.topic_label,item.difficulty].join(' '));
    return catalogQuery.trim().split(/\s+/).every(part=>haystack.includes(normalize(part)));
  }
  function renderCatalogResults() {
    const specialty=category==='specialty',rows=catalog.filter(catalogMatches);
    const topic=facets.topics?.find(row=>row.id===filters.learning_topic),searched=Boolean(catalogQuery.trim());
    document.querySelector('#catalog-count').textContent=catalogLoading?'正在更新练习…':`${topic?'当前分类：'+topic.label+' · ':''}符合条件：${rows.length} 道练习${searched?' · 已按关键词搜索':''}`;
    document.querySelector('#catalog-clear-search').hidden=!catalogQuery;
    const list=document.querySelector('#catalog-results');list.setAttribute('aria-busy',String(catalogLoading));
    list.innerHTML=rows.map(item=>{
      // Review status and history are developer records, not learner-facing labels.
      const source=item.kind==='standardized_case'?'标准化案例':['','textbook_key','project_adapted_source_reference'].includes(item.answer_origin||'')&&item.has_reference_answer?'教材配套练习':item.has_reference_answer?'项目练习':'项目自编开放练习';
      const scope=[item.population_labels?.join('、'),item.device_labels?.join('、')].filter(Boolean).join(' · ')||(item.kind==='knowledge_card'?'通用概念卡':'练习情境');
      const label=item.learning_topic_pending?'待归类':item.learning_topic_label||(specialty?item.topic_label:'待归类');
      return `<button type="button" class="exercise-card ${specialty?'':'foundation-card'}" data-case="${escape(item.id)}" ${catalogLoading?'disabled':''}>${!specialty?`<span class="exercise-topic">${escape(label)}</span>`:''}<div class="exercise-tags"><span>${escape(item.difficulty)}</span><span>${escape(item.question_type_label||'练习')}</span>${specialty?`<span>${escape(scope)}</span>`:''}</div><strong>${escape(item.title.split(' / ')[0])}</strong><small>${specialty?escape(label)+' · ':''}${source}</small>${item.source_difficulty!==item.difficulty?`<small>原始难度：${escape(item.source_difficulty)}</small>`:''}${!specialty&&item.learning_topic_pending?'<small>知识分类待核对，不限定分类时可查看本题。</small>':specialty&&item.classification_pending?'<small>人群和器械分类待核对</small>':''}</button>`;
    }).join('')||`<div class="catalog-empty"><p>${!specialty&&topic&&!searched?'“'+escape(topic.label)+'”在当前条件下暂无练习。':'没有找到符合条件的练习。'}</p><p class="panel-note">${specialty?'可换个关键词、放宽筛选，或切换到另一题库模块。':'可调整关键词、题型或难度，或选择其他知识分类。'}</p><div class="action-row">${!specialty&&filters.learning_topic?button('保留分类，清除其他筛选','catalog-clear-refinements'):''}${button('清除全部筛选','catalog-reset')}</div></div>`;
  }
  function availableCatalogTopics() {
    return (facets.topics||[]).filter(row=>row.id!=='related_nursing'&&Number(row.available_count??row.count)>0);
  }
  function foundationTopicSelect() {
    const topics=availableCatalogTopics();
    const sourceTopics=facets.classification_source?.educational_topics||[];
    const paperIds=new Set(sourceTopics.map(row=>typeof row==='string'?row:row.id).filter(Boolean));
    if(!paperIds.size)['skin_anatomy_physiology','wound_healing','wound_assessment','pressure_injury_classification','pressure_injury_prevention_management','wound_infection','support_surfaces','lower_extremity_ulcers','iad','wound_dressings'].forEach(id=>paperIds.add(id));
    const isPaper=row=>row.origin==='paper_table'||!row.origin&&paperIds.has(row.id);
    const paperTopics=topics.filter(isPaper),supplementalTopics=topics.filter(row=>!isPaper(row));
    const topicOption=row=>{
      const count=Number(row.count)||0;
      return `<option value="${escape(row.id)}" ${filters.learning_topic===row.id?'selected':''}>${escape(row.label)} · ${count?count+' 题':'当前筛选暂无题'}</option>`;
    };
    const groups=[{id:'paper',title:'皮肤与伤口护理',rows:paperTopics},{id:'supplement',title:'护理操作与专业能力',rows:supplementalTopics}].filter(group=>group.rows.length);
    const unavailable=!topics.length;
    return `<label class="field catalog-knowledge-filter"><span>知识分类</span><select id="catalog-learning_topic" data-catalog-filter="learning_topic" ${unavailable?'disabled aria-describedby="catalog-classification-help"':''}><option value="" ${filters.learning_topic?'':'selected'}>按知识分类</option>${groups.map(group=>`<optgroup label="${escape(group.title)}">${group.rows.map(topicOption).join('')}</optgroup>`).join('')}</select></label>`;
  }
  function renderCatalog() {
    const specialty=category==='specialty';
    const selected=Object.values(filters).filter(Boolean).length;
    const controls=specialty?`${catalogSelect('population','人群',facets.populations)}${catalogSelect('device_type','医疗器械',facets.devices)}${catalogSelect('difficulty','案例难度',facets.difficulties)}${catalogSelect('question_type','题型',facets.question_types)}${catalogSelect('learning_topic','案例训练主题',availableCatalogTopics())}`:`${catalogSelect('question_type','题型',facets.question_types)}${catalogSelect('difficulty','练习难度',facets.difficulties)}${foundationTopicSelect()}`;
    const classificationNote=!specialty&&!availableCatalogTopics().length?`<div class="catalog-filter-warning"><p class="panel-note" id="catalog-classification-help">分类信息暂未加载，请重新加载题库。</p>${button('重新加载分类','catalog')}</div>`:'';
    document.querySelector('#dialog-body').innerHTML=`<p class="panel-note catalog-intro">${specialty?'按人群、器械和案例主题选择专项练习。':'按题型、练习难度和知识分类选择基础知识练习。'}客观题在总结中查看参考；主观题参考答案在三条提示后解锁，答对后可复盘。</p><div class="catalog-tabs" role="group" aria-label="题库模块">${facets.modules.map(row=>`<button type="button" class="tab ${row.id===category?'selected':''}" id="catalog-module-${escape(row.id)}" data-category="${escape(row.id)}" aria-pressed="${row.id===category}">${escape(row.label)}<span> · ${row.count}</span></button>`).join('')}</div><label class="catalog-search">${icon('search')}<span class="sr-only">搜索当前模块的练习</span><input type="search" id="catalog-search" value="${escape(catalogQuery)}" maxlength="120" placeholder="${specialty?'搜索题目、人群、器械或案例主题':'搜索题目、知识分类或题型'}" autocomplete="off"><button type="button" id="catalog-clear-search" data-command="catalog-clear-search" aria-label="清除搜索" ${catalogQuery?'':'hidden'}>${icon('x')}</button></label><details class="catalog-refine" ${catalogFiltersOpen?'open':''}><summary>${specialty?'按人群、器械、难度、题型和案例主题筛选':'按题型、练习难度和知识分类筛选'}<span>${selected?'已选 '+selected+' 项':'展开筛选'}</span></summary><div class="catalog-filters ${specialty?'':'foundation-filters'}">${controls}</div>${classificationNote}<p class="panel-note">${specialty?'案例难度不改变六步推理框架。':'三项可组合筛选，关键词搜索限定在当前筛选结果中。'}</p></details><div class="catalog-result-heading"><p class="catalog-count" id="catalog-count" role="status" aria-live="polite" aria-atomic="true"></p>${button('重置','catalog-reset')}</div><div class="catalog-list" id="catalog-results"></div>`;
    renderCatalogResults();
    document.querySelector('.catalog-refine').addEventListener('toggle',event=>{catalogFiltersOpen=event.target.open;});
    document.querySelector('#catalog-search').addEventListener('input',event=>{catalogQuery=event.target.value;renderCatalogResults();});
    document.querySelectorAll('[data-catalog-filter]').forEach(field=>field.addEventListener('change',event=>{
      const key=event.target.dataset.catalogFilter;
      if(catalogLoading||category==='foundation'&&['population','device_type'].includes(key))return;
      filters[key]=event.target.value;
      if(key==='population')filters.device_type='';
      loadCatalog(key);
    }));
  }

  async function learning(profile=false) {
    const state=await C.ensureSession(),learningSid=state.session_id,request=++learningRequest;panelSid=learningSid;
    if(profile&&!state.learning_profile){C.openDialog('学习档案',`<p class="panel-note">在设置中填写一个档案名称，就能跨会话保存已完成练习，并按自己的回忆情况安排复习。</p>${button('填写档案名称','settings','',true)}`);return;}
    const title=profile?'学习档案':'本次学习进度';
    C.openDialog(title,'<p class="panel-note">正在读取真实学习记录…</p>',true);
    const active=()=>request===learningRequest&&C.S.view?.state.session_id===learningSid&&document.querySelector('#dialog').open&&document.querySelector('#dialog-title').textContent===title;
    let data;
    try{data=await C.api(`/sessions/${learningSid}/${profile?'learning-profile':'learning'}`);}
    catch(error){if(active())document.querySelector('#dialog-body').innerHTML=`<p class="inline-error" role="alert">${escape(error.message)}</p>`;return;}
    if(!active())return;
    const count=profile?data.completed_rounds:data.completed_exercises;
    const review=profile?data.due_count:data.review_count;
    const rows=data.exercises||[];
    const stats=`<div class="stat-grid"><div class="stat"><b>${count}</b><span>已完成轮次</span></div><div class="stat"><b>${data.unique_questions}</b><span>练习题目</span></div><div class="stat"><b>${review}</b><span>${profile?'到期复习':'待复习题目'}</span></div></div>`;
    const history=data.analytics?.history||[];
    const records=rows.map((row,index)=>{
      const metric=history.find(item=>item.exercise_id===row.exercise_id),ended=metric?.completion_status==='interrupted';
      return `<div class="record-row"><strong>${escape(row.title)}<span class="record-tag">${ended?'已中断':outcomes[row.outcome]||'已记录'}</span></strong><p>${escape(when(row.completed_at||row.updated_at||metric?.at))}${metric?.guided?' · 六步完成 '+metric.passed_steps+'/6':''}</p>${row.superseded_by?`<p>当轮：${outcomes[row.outcome]} · 最新：${outcomes[row.latest_outcome]||'已记录'}</p>`:''}${row.gaps?.length?`<p>待复习：${escape(row.gaps.join('；'))}</p>`:''}${profile&&row.complete?`<p>下次复习：${escape(when(row.next_due))} · ${row.self_rating?'已自评':'尚未自评'}</p>`:''}<div class="action-row">${profile?button('查看原会话','open-session',`data-sid="${escape(row.source_session_id)}"`):button('查看第 '+(index+1)+' 轮','exercise-record',`data-exercise="${escape(row.exercise_id)}"`)}${!profile&&(row.complete||ended)?button('重练本题','learning-action',`data-operation="retry_exercise" data-exercise="${escape(row.exercise_id)}"`):''}</div></div>`;
    }).join('');
    const footer=profile?`${review?button('开始到期复习','learning-action','data-operation="due"',true):''}<a class="button" href="/api/sessions/${learningSid}/learning-profile/export" download>${icon('download')}导出档案</a>${button('导出科研数据','research-export','data-scope="profile"')}${button('清空档案','clear-learning')}`:button('导出科研数据','research-export');
    document.querySelector('#dialog-body').innerHTML=`${profile?`<p class="panel-note">${escape(data.profile)} · 当前浏览器的独立档案。复习时间根据你的自评安排，作答评价与提示使用分别记录。</p>`:`<p class="panel-note">来自本会话保存的真实评价；中断练习保留已作答内容，不计完整六步完成。</p>`}${data.analytics?renderLearningAnalytics(data.analytics):stats}<div class="panel-footer">${button('查看推荐练习','recommendations','',true)}</div><h3 class="panel-heading">训练历史</h3>${records||'<p class="panel-note">还没有训练记录。先开始一道练习吧。</p>'}${footer?`<div class="panel-footer">${footer}</div>`:''}`;
  }

  function exerciseRecord(id) {
    const state=C.S.view.state;
    const row=state.practice_history?.find(record=>record.exercise_id===id)||(id===(state.exercise_id||state.session_id+':first')?state:null);
    if(!row){C.toast('未找到这轮练习记录',true);return;}
    C.S.panelMessages=row.messages.filter(item=>item.kind!=='welcome');
    C.openDialog('练习记录',`<div class="markdown">${markdown(row.case)}</div><div class="record-row">${C.S.panelMessages.map((message,index)=>renderMessage(message,index,{mode:'teaching',messages:C.S.panelMessages,teachingFeedback})).join('')}</div>`,true);
  }

  function stepRecord(index) {
    const flow=C.S.view?.state.clinical_flow,step=flow?.steps?.[Number(index)];
    if(!step||!Number.isInteger(Number(index))){C.toast('没有找到这一步的保存记录',true);return;}
    const titles=['识别线索','分析线索','确定优先问题','制定解决方案','采取行动','评价结果'];
    const status={pending:'尚未开始',active:'正在作答',awaiting_next:'已通过，等待下一步',passed:'已完成'}[step.status]||'已保存';
    const criterionLabels={met:'已覆盖',partial:'待补充',incorrect:'需要修正',uncertain:'待确认',missing:'待补充'};
    const answers=(step.answers||[]).map((answer,position)=>`<div class="record-row" id="step-answer-${Number(index)}-${position+1}" tabindex="-1"><strong>第 ${position+1} 次作答</strong><pre class="learner-quote">${escape(answer)}</pre></div>`).join('');
    const checks=(step.criterion_checks||[]).map(row=>{
      const evidence=(row.evidence||[]).filter(citation=>Number.isInteger(citation.attempt)&&citation.attempt>0&&citation.attempt<=step.answers.length&&typeof citation.quote==='string'&&citation.quote.trim()&&step.answers[citation.attempt-1].includes(citation.quote));
      const attempts=[...new Set(evidence.map(citation=>citation.attempt))],note=typeof row.explanation==='string'&&!evidence.some(citation=>row.explanation.includes(citation.quote))?row.explanation:'';
      return `<div class="record-row status-${escape(row.status)}"><strong>${escape(typeof row.criterion==='object'?row.criterion.zh||row.criterion.en:row.criterion||row.id)} <span class="criterion-status">${criterionLabels[row.status]||'待确认'}</span></strong>${attempts.length?`<p>${attempts.map(attempt=>`<a class="criterion-evidence-link" href="#step-answer-${Number(index)}-${attempt}">定位第 ${attempt} 次作答</a>`).join(' · ')}</p>`:''}${note?`<p>${escape(note)}</p>`:''}</div>`;
    }).join('');
    const hints=(step.hints||[]).map(hint=>`<div class="record-row"><strong>${hint.level?'第 '+escape(hint.level)+' 级提示':'已查看提示'}</strong><div class="markdown">${markdown(hint.text||'')}</div></div>`).join('');
    C.openDialog(`第 ${Number(index)+1} 步 · ${titles[Number(index)]}`,`<p class="panel-note">${status} · 此处只回看保存的作答和评价，不改变训练进度。</p><h3 class="panel-heading">学生作答</h3>${answers||'<p class="panel-note">该步骤暂无已保存作答。</p>'}<h3 class="panel-heading">必答点核对</h3>${checks||'<p class="panel-note">该步骤尚未保存评价。</p>'}${hints?'<h3 class="panel-heading">提示使用</h3>'+hints:''}`,true);
  }

  function showSummary() {
    const state=C.S.view?.state;
    if(!state)return;
    const messages=C.S.view.messages||[];let start=0;messages.forEach((row,index)=>{if(row.kind==='learning_section')start=index+1;});
    const summary=messages.slice(start).reverse().find(row=>row.role==='assistant'&&row.kind==='summary');
    const content=state.summary?(summary?.display_content??state.summary):'';
    C.openDialog('本轮练习总结',`<p class="panel-note">${state.clinical_flow?.status==='interrupted'?'本轮提前结束，记录为中断；以下讲解不计作你的作答。':'根据本轮保存的作答与评价整理。'}</p><div class="markdown">${content?markdown(content):'<p>本轮尚未生成总结，可以返回训练后继续。</p>'}</div>`,true);
  }

  async function recommendations() {
    const state=await C.ensureSession(),recommendSid=state.session_id,request=++recommendationRequest,title='推荐下一次练习';
    C.openDialog(title,'<p class="panel-note">正在根据真实训练记录选择案例…</p>',true);
    const active=()=>request===recommendationRequest&&C.S.view?.state.session_id===recommendSid&&document.querySelector('#dialog').open&&document.querySelector('#dialog-title').textContent===title;
    try {
      const data=await C.api(`/sessions/${recommendSid}/recommendations`);
      if(!active())return;
      const cards=(data.items||[]).map(item=>`<div class="record-row recommendation-card"><strong>${escape(item.title)}<span class="record-tag">${escape(item.difficulty)}</span></strong><p>${escape(item.reason)}</p><div class="action-row">${button('开始这道练习','recommended-case',`data-case-id="${escape(item.case_id)}" data-from-sid="${escape(recommendSid)}"`,true)}</div></div>`).join('');
      document.querySelector('#dialog-body').innerHTML=`<p class="panel-note">${escape(data.basis||'按已保存的作答、提示与覆盖情况推荐。')}每个推荐都有原因；开始后会新建训练会话，原记录保留。</p>${data.progression?`<p class="panel-note">建议难度：${escape(data.progression.target_difficulty)} · ${escape(data.progression.basis)}</p>`:''}${cards||`<p class="panel-note">${escape(data.no_match_reason||'暂时没有合适推荐，可以在题库中主动选择。')}</p>`}<div class="panel-footer">${button('查看完整题库','catalog')}</div>`;
    }catch(error){if(active())document.querySelector('#dialog-body').innerHTML=`<p class="inline-error" role="alert">${escape(error.message)}</p>`;}
  }

  async function researchExport(scope='session') {
    const state=C.S.view?.state;
    if(!state){C.toast('请先打开需要导出的会话。',true);return;}
    const exportSid=state.session_id;
    C.openDialog('导出科研数据',`<p class="panel-note">${scope==='profile'?'导出当前访问身份和档案中的练习记录。':'导出当前会话的已保存记录。'}默认仅包含结构化指标，以临时编号替代会话身份，省略档案名和正文。用途标签按本次选择记录。</p><form id="research-export-form">
      <label class="field"><span>数据用途</span><select id="research-purpose">${options({development:'开发测试',pilot:'试用／预实验',research:'正式研究'},'development')}</select></label>
      ${field('research-study','研究编号（正式研究必填）','','使用项目编码，不填写姓名。')}${field('research-protocol','研究方案版本（正式研究必填）','')}
      <label class="field"><span>文件格式</span><select id="research-format">${options(state.mode==='teaching'?{json:'JSON：完整记录与来源说明',csv:'CSV：逐步骤结构化指标'}:state.mode==='performance'?{json:'JSON：完整记录与来源说明',csv:'CSV：性能运行结构化指标'}:{json:'JSON：完整记录与来源说明'},'json')}</select></label>
      <label class="privacy-confirmation"><input type="checkbox" id="research-include-text"><span>包含病例、作答和反馈正文</span></label>
      <label class="privacy-confirmation" id="research-text-confirmation" hidden><input type="checkbox" id="research-confirm-text"><span>我已核查正文中的身份信息，并会在分享前再次检查。自动过滤可能遗漏。</span></label>
      <div id="research-export-error" class="inline-error" role="alert" hidden></div><div class="form-footer">${button('取消','close-dialog')}<button type="submit" class="button primary">导出文件</button></div></form>`,true);
    document.querySelector('#research-include-text').addEventListener('change',event=>{document.querySelector('#research-text-confirmation').hidden=!event.target.checked;});
    document.querySelector('#research-export-form').addEventListener('submit',async event=>{
      event.preventDefault();const submit=event.submitter,errorBox=document.querySelector('#research-export-error');
      if(submit.disabled)return;submit.disabled=true;
      try {
        if(C.S.view?.state.session_id!==exportSid)throw new Error('会话已切换，请重新打开导出窗口。');
        const read=id=>document.getElementById(id).value.trim();
        const body={scope,purpose:read('research-purpose'),study_id:read('research-study'),protocol_version:read('research-protocol'),format:read('research-format'),include_text:document.querySelector('#research-include-text').checked,text_review_confirmed:document.querySelector('#research-confirm-text').checked};
        if(body.include_text&&!body.text_review_confirmed)throw new Error('包含正文时请先确认已核查身份信息。');
        const data=await C.api(`/sessions/${exportSid}/research-export`,body);
        const blob=new Blob([body.format==='csv'?'\ufeff':'',data.content],{type:data.content_type});
        const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=data.filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
        C.closeDialog();C.toast((state.mode==='teaching'?`已导出 ${data.record_count} 条练习记录`:'会话科研资料已导出')+(data.free_text_included?'，包含经规则过滤的正文':'，不含正文'));
      }catch(error){errorBox.textContent=error.message;errorBox.hidden=false;}finally{submit.disabled=false;}
    });
  }

  async function recommendedCase(target) {
    if(target.disabled||launchingRecommendation)return;
    if(target.dataset.fromSid&&C.S.view?.state.session_id!==target.dataset.fromSid){C.toast('会话已切换，请重新读取推荐。',true);return;}
    const caseId=target.dataset.caseId;
    if(!caseId)return;
    target.disabled=true;launchingRecommendation=true;
    try {
      // Keep the previous persisted session intact; recommendations launch a
      // new round with no old answers entering its assessment context.
      C.S.mode='teaching';C.S.view=null;
      await C.ensureSession();
      C.closeDialog();
      await C.startTurn(caseId,'library_case');
    }catch(error){C.toast(error.message,true);}
    finally{target.disabled=false;launchingRecommendation=false;}
  }

  function clearLearning() {
    C.openDialog('清空当前学习档案',`<p class="panel-note">会清空“${escape(C.S.view.state.learning_profile)}”的完成记录和复习安排。原始会话、学生作答与评价仍可在历史中查看。</p><div class="form-footer">${button('保留档案','learning-profile')}${button('清空学习档案','confirm-clear-learning')}</div>`);
  }

  async function preferences() {
    const state=await C.ensureSession();panelSid=state.session_id;
    if(!state.memory_profile){C.openDialog('偏好记忆',`<p class="panel-note">在设置中选择偏好档案后，可以通过“记住：以后回答简短一些”保存偏好；临时任务不会自动写入。</p>${button('设置偏好档案','settings','',true)}`);return;}
    const data=await C.api(`/sessions/${panelSid}/preferences`);
    C.openDialog('偏好记忆',`<p class="panel-note">${escape(data.profile)} · 只保存你明确要求记住的内容。</p>${data.items.map(item=>`<div class="record-row"><strong>${escape(item.key)}</strong><p>${escape(item.value)}</p>${button('删除此项','delete-preference',`data-key="${escape(item.key)}"`)}</div>`).join('')||'<p class="panel-note">尚未保存偏好。可以在对话中说“记住……”来添加。</p>'}${data.items.length?`<div class="panel-footer">${button('清空此档案偏好','clear-preferences')}</div>`:''}`);
  }

  let evidenceRequest=0;
  let evidenceRows=[];
  function referenceRows(data) {
    const rows=data.references||[];
    if(C.S.view?.state.mode==='teaching')return teachingReferenceCards(data,C.S.view?.notices||[]);
    return (C.S.view?.notices||[]).map(note=>`<p class="panel-note">${escape(note)}</p>`).join('')+`<p class="panel-note">参考文献依据 ${escape(data.standard||'GB/T 7714-2025')} 整理。原文链接可跳转到来源页面，全文访问取决于来源权限。</p>`+rows.map(row=>{
      const link=row.link||{};
      return `<div class="record-row reference-card"><strong>${escape(row.title)}</strong><p class="reference-citation">${escape(row.reference)}</p>
        ${row.resource_type?`<p>资料类型：${escape(row.resource_type)}</p>`:''}
        ${row.standard_number?`<p>标准编号：${escape(row.standard_number)}</p>`:''}
        ${row.publication_date||row.effective_date?`<p>${row.publication_date?'发布日期：'+escape(row.publication_date):''}${row.effective_date?' · 实施日期：'+escape(row.effective_date):''}</p>`:''}
        ${row.source_name||row.year?`<p>${escape(row.source_name)}${row.year?' · '+escape(row.year):''}</p>`:''}
        ${row.locations?.length?`<p>原文位置：${escape(row.locations.join('；'))}</p>`:''}
        ${row.excerpt_count>1?`<p>同一文献的 ${row.excerpt_count} 个参考片段已合并</p>`:''}
        ${row.metadata_status==='incomplete'?`<p class="reference-quality">题录不完整${row.missing_metadata?.length?'：缺少'+escape(row.missing_metadata.join('、')):''}；待核对补充。</p>`:''}
        ${row.metadata_warnings?.length?`<p class="reference-quality">${escape(row.metadata_warnings.join('；'))}</p>`:''}
        <div class="action-row">${link.status==='verified'&&/^https:\/\//i.test(link.url)?`<a class="button" href="${escape(link.url)}" target="_blank" rel="noopener noreferrer">查看原文 ${icon('arrow')}</a>`:`<span class="panel-note">${escape(link.message||'暂无法提供原文链接')}</span>`}</div>
      </div>`;
    }).join('')+(rows.length?'':'<p class="panel-note">这轮没有保存可展示的外部参考资料。</p>');
  }
  async function evidence() {
    const state=C.S.view?.state;
    if(!state)return;
    const sid=state.session_id,request=++evidenceRequest;
    const teaching=state.mode==='teaching';
    evidenceRows=[];
    const active=()=>request===evidenceRequest&&C.S.view?.state.session_id===sid&&document.querySelector('#dialog').open&&document.querySelector('#dialog-title').textContent==='本轮参考资料';
    C.openDialog('本轮参考资料','<p class="panel-note">正在整理参考文献…</p>',true);
    try {
      const data=await C.api(`/sessions/${sid}/references`);
      if(!active())return;
      evidenceRows=data.references||[];
      document.querySelector('#dialog-body').innerHTML=referenceRows(data);
      if(!data.document_count)return;
      try {
        const verified=await C.api(`/sessions/${sid}/references?verify=true`);
        if(!active())return;
        if(teaching)updateReferenceLinks(verified);
        else document.querySelector('#dialog-body').innerHTML=referenceRows(verified);
      }catch(error){
        // A link check never interrupts the teaching bibliography or its copy
        // controls; the author can search using the already displayed citation.
        if(active()&&!teaching)document.querySelector('#dialog-body').insertAdjacentHTML('beforeend',`<p class="panel-note">原文链接核验暂未完成：${escape(error.message)}</p>`);
      }
    } catch(error) {
      if(!active())return;
      if(teaching)document.querySelector('#dialog-body').innerHTML=`<p class="panel-note">参考资料未能加载，请重试。</p>${button('重新加载参考资料','evidence')}`;
      else document.querySelector('#dialog-body').insertAdjacentHTML('beforeend',`<p class="panel-note">原文链接核验暂未完成：${escape(error.message)}</p>`);
    }
  }

  function rename(sid,title) {
    C.openDialog('重命名对话',`<form id="rename-form">${field('rename-title','对话名称',title)}<div id="rename-error" class="inline-error" hidden></div><div class="form-footer">${button('取消','close-dialog')}<button class="button primary" type="submit">保存名称</button></div></form>`);
    const input=document.querySelector('#rename-title');input.maxLength=120;input.focus();input.select();
    document.querySelector('#rename-form').addEventListener('submit',async event=>{event.preventDefault();const target=event.submitter;target.disabled=true;try{await C.api(`/sessions/${sid}/metadata`,{title:input.value},'PATCH');C.closeDialog();await C.loadHistory();if(C.S.view?.state.session_id===sid)await C.refresh();C.toast('对话名称已保存');}catch(error){const box=document.querySelector('#rename-error');box.textContent=error.message;box.hidden=false;}finally{target.disabled=false;}});
  }

  function help() {
    C.openDialog('用起来更顺手',`<p class="panel-note">侧栏切换普通问答和教学练习，点击历史标题继续原会话。刷新自动恢复当前对话；“对话选项”可以重命名、置顶、归档和导出。</p><div class="help-shortcuts"><div><span>发送消息</span><kbd>Enter</kbd></div><div><span>输入换行</span><kbd>Shift + Enter</kbd></div><div><span>搜索会话</span><kbd>Ctrl / ⌘ + K</kbd></div><div><span>新建对话</span><kbd>Ctrl / ⌘ + Shift + O</kbd></div><div><span>关闭面板</span><kbd>Esc</kbd></div></div><div class="settings-section">教学练习</div><p class="panel-note">选题后先尝试作答，再按需请求提示、重新核对或结束总结。填入学习档案名称后，可跨会话查看记录；完成题目后选择复习自评，再使用到期复习。</p><div class="settings-section">文件</div><p class="panel-note">点击输入框旁的加号，也可拖入文件或粘贴图片。支持PDF、Office文档、表格、文字和图片，每个文件最多20MB。生成文件会以下载卡片展示。</p><a class="button" href="/classic/" target="_blank" rel="noopener">打开原版界面 ${icon('arrow')}</a>`);
  }

  async function command(name,target) {
    if(name==='settings') {settings();return true;}
    if(name==='toggle-learning-access') {const input=document.querySelector('#setting-access-code');if(input){const show=input.type==='password';input.type=show?'text':'password';target.textContent=show?'隐藏访问码':'显示访问码';target.setAttribute('aria-pressed',String(show));}return true;}
    if(name==='copy-learning-access') {await copyLearningAccess();return true;}
    if(name==='catalog') {await exercises();return true;}
    if(name==='catalog-clear-search'){catalogQuery='';document.querySelector('#catalog-search').value='';renderCatalogResults();document.querySelector('#catalog-search').focus();return true;}
    if(name==='catalog-clear-refinements'){
      const topic=filters.learning_topic;
      catalogQuery='';filters={...emptyCatalogFilters(),learning_topic:topic};
      await loadCatalog('learning_topic');
      return true;
    }
    if(name==='catalog-reset'){catalogQuery='';filters=emptyCatalogFilters();await loadCatalog('search');return true;}
    if(name==='research-export') {await researchExport(target.dataset.scope||'session');return true;}
    if(name==='learning-profile') {await learning(true);return true;}
    if(name==='learning-progress') {await learning(false);return true;}
    if(name==='preferences') {await preferences();return true;}
    if(name==='evidence') {evidence();return true;}
    if(name==='copy-reference') {const row=evidenceRows[Number(target.dataset.referenceIndex)];if(row)await C.copy(row.reference||'');return true;}
    if(name==='copy-references') {if(evidenceRows.length)await C.copy(bibliographyText(evidenceRows));return true;}
    if(name==='exercise-record') {exerciseRecord(target.dataset.exercise);return true;}
    if(name==='step-record') {stepRecord(target.dataset.step);return true;}
    if(name==='show-summary') {showSummary();return true;}
    if(name==='recommendations') {await recommendations();return true;}
    if(name==='recommended-case') {await recommendedCase(target);return true;}
    if(name==='clear-learning') {clearLearning();return true;}
    if(name==='confirm-clear-learning'){await C.api(`/sessions/${C.S.view.state.session_id}/learning-profile`,undefined,'DELETE');await C.refresh();await learning(true);C.toast('学习档案已清空，原始会话保留');return true;}
    if(name==='delete-preference'){await C.api(`/sessions/${panelSid}/preferences/${encodeURIComponent(target.dataset.key)}`,undefined,'DELETE');await preferences();return true;}
    if(name==='clear-preferences'){await C.api(`/sessions/${panelSid}/preferences`,undefined,'DELETE');await preferences();return true;}
    if(name==='help'){help();return true;}
    return false;
  }
  function chooseCategory(value) {if(!modules[value]||value===category)return;category=value;catalogQuery='';filters=emptyCatalogFilters();loadCatalog('module-'+value);}
  return {command,rename,settings,chooseCategory,caseSetup};
}
