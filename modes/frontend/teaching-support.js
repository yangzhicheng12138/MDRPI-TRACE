import {escape,markdown} from './markdown.js?v=20261116-i18n';

const placements=new Set(['hint','explanation','reference']);
const actions={hint:'hint',stuck:'hint',hint_detail:'hint',explain:'explanation',reference:'reference'};
const shown=value=>typeof value==='string'?value:value?.zh||value?.en||'';

/** Read-only support cards. Messages and answer evidence keep their original indices. */
export function createTeachingSupport(C) {
  const {S,busy,currentJob,isObjectiveQuestion,teachingReferenceAccess}=C;
  const $=selector=>document.querySelector(selector);
  const openStates=new Map();
  const lastOpened=new Map();
  let scope='',wide=null,single=null,observer,observerReady=false,completedCard=null;
  const data=()=>S.mode==='teaching'&&S.view?.teaching_support?.enabled?S.view.teaching_support:null;
  const enabled=()=>Boolean(data()&&$('#teaching-layout')&&$('#teaching-hints-body'));
  const objective=()=>Boolean(data()?.objective||isObjectiveQuestion(S.view?.state));
  const identity=()=>[S.view?.state.session_id,data()?.exercise_id,data()?.step_id].join(':');
  const side=placement=>placement==='hint'?'hints':'explanations';
  const panel=name=>$(`#teaching-${name} .teaching-support-panel`);
  // Desktop scrolls the whole panel; compact layouts scroll only the body.
  const scroller=body=>getComputedStyle(body).overflowY==='visible'?body.closest('.teaching-support'):body;
  const sidePlacement=(job=currentJob(),active=S.view?.active)=>{
    if(job?.done)return null;
    const live=active?.teaching_support;
    const value=job&&Object.hasOwn(job,'supportPlacement')?job.supportPlacement:live&&Object.hasOwn(live,'placement')?live.placement:data()?.pending?.placement;
    return enabled()&&placements.has(value)?value:null;
  };
  const errorPlacement=()=>{
    if(!enabled())return null;
    const last=S.supportOutcome;
    if(last?.scope===identity()&&placements.has(last.placement))return last.placement;
    const value=data()?.failure?.placement??data()?.pending?.placement;
    return placements.has(value)?value:null;
  };

  function syncLayout() {
    const shell=$('#teaching-layout'),main=$('.main');
    if(!shell||!main)return;
    const active=enabled();
    shell.classList.toggle('is-teaching',active);
    shell.classList.toggle('support-enabled',active);
    shell.classList.toggle('is-objective',active&&objective());
    main.classList.toggle('teaching-panels-active',active);
    $('#teaching-hints').hidden=!active;
    $('#teaching-explanations').hidden=!active||objective();
    if(!active){scope='';return;}
    const nextScope=identity(),width=main.getBoundingClientRect().width,nextWide=width>=1100,nextSingle=width<=560;
    if(scope!==nextScope||wide!==nextWide||single!==nextSingle){
      scope=nextScope;wide=nextWide;single=nextSingle;
      const desired={hints:wide||Boolean(openStates.get(scope+':hints')),explanations:!objective()&&(wide||Boolean(openStates.get(scope+':explanations')))};
      if(single&&desired.hints&&desired.explanations){desired[lastOpened.get(scope)==='explanations'?'hints':'explanations']=false;}
      for(const name of ['hints','explanations']){
        const node=panel(name);if(node)node.open=desired[name];
      }
    }
  }

  function ensureObserver() {
    const main=$('.main');if(observerReady||!main)return;
    observerReady=true;
    if(typeof ResizeObserver==='function'){
      observer=new ResizeObserver(syncLayout);observer.observe(main);
    }else window.addEventListener('resize',syncLayout);
    for(const name of ['hints','explanations']){
      const node=panel(name);
      node?.addEventListener('toggle',()=>{
        if(!scope)return;
        if(!wide)openStates.set(scope+':'+name,node.open);
        if(node.open){
          lastOpened.set(scope,name);
          if(single){
            const other=name==='hints'?'explanations':'hints',otherPanel=panel(other);
            if(otherPanel)otherPanel.open=false;
            openStates.set(scope+':'+other,false);
          }
        }
      });
    }
  }

  function locateCard(placement,intent='',completed=false) {
    const targetScope=identity(),name=side(placement);
    requestAnimationFrame(()=>{
      if(!enabled()||identity()!==targetScope)return;
      const body=$(`#teaching-${name}-body`);
      const selector=completed&&intent==='hint_detail'?'.hint-details':placement==='hint'?'.teaching-hint-card:not(.is-unseen)':placement==='reference'?'.teaching-reference-card':'.teaching-explanation-card';
      const cards=body?.querySelectorAll(selector);
      const latest=(!completed&&body?.querySelector('[data-support-live]'))||cards?.[cards.length-1];
      if(latest&&body){const scroll=scroller(body);scroll.scrollTop+=latest.getBoundingClientRect().top-scroll.getBoundingClientRect().top;}
    });
  }

  function disclose(placement,locate=false) {
    if(!enabled()||!placements.has(placement))return;
    const name=side(placement),node=panel(name);
    if(!node)return;
    if(single){const other=name==='hints'?'explanations':'hints',otherPanel=panel(other);if(otherPanel)otherPanel.open=false;openStates.set(identity()+':'+other,false);}
    node.open=true;openStates.set(identity()+':'+name,true);
    lastOpened.set(identity(),name);
    if(locate)locateCard(placement);
  }

  function updateBody(node,html) {
    if(!node||node._supportHTML===html)return;
    const expanded=new Set([...node.querySelectorAll('details[data-support-card][open]')].map(row=>row.dataset.supportCard));
    const scroll=scroller(node),savedTop=scroll.scrollTop;
    node.innerHTML=html;node._supportHTML=html;
    for(const row of node.querySelectorAll('details[data-support-card]'))if(expanded.has(row.dataset.supportCard))row.open=true;
    scroll.scrollTop=savedTop;
  }

  function button(label,action,primary=false) {
    const state=S.view?.state,blocked=busy()||Boolean(action!=='retry'&&state?.pending&&!state.cancelled);
    return `<button type="button" class="button ${primary?'primary':''}" data-command="support-turn" data-value="${action}" ${blocked?'disabled':''}>${label}</button>`;
  }

  function requestText(value) {
    const request=shown(value).trim();if(!request)return '';
    // A mixed submission already has its complete original text in the center.
    if((S.view?.state.messages||[]).some(row=>row.role==='user'&&row.kind==='student'&&row.content===request))return '';
    return `<p class="panel-card-request">${escape(request)}</p>`;
  }

  function liveCard(placement) {
    const job=currentJob(),active=S.view?.active,current=sidePlacement(job,active);
    if((!job&&!active)||job?.done||current!==placement)return '';
    const title={hint:'正在准备提示',explanation:'正在补充讲解',reference:'正在整理参考答案'}[placement];
    const status=job?.stopping?'正在停止，已保存的内容会保留':job?.status||active?.phase||title;
    return `<section class="teaching-support-card panel-turn-status" data-support-live="${placement}" aria-label="${title}" aria-live="polite"><div class="panel-card-heading support-card-meta"><span class="pulse"></span><strong>${escape(title)}</strong></div><p>${escape(status)}</p>${job?.text?`<div class="markdown panel-card-body support-card-body">${markdown(job.text)}</div>`:''}<div class="panel-actions"><button type="button" class="button" data-command="support-stop" ${job?.stopping||job?.uploading?'disabled':''}>停止生成</button></div></section>`;
  }

  function errorCard(placement) {
    const state=S.view?.state;
    if(currentJob()||S.view?.active||errorPlacement()!==placement||!(state?.error||S.view?.connection_error||state?.cancelled))return '';
    const error=state.error||S.view.connection_error||'这次请求已停止，已看提示和作答均保留。';
    return `<section class="teaching-support-card panel-turn-status panel-turn-error" role="status"><strong>${state.cancelled?'本次帮助已停止':'本次帮助尚未完成'}</strong><p>${escape(error)}</p>${state.pending?`<div class="panel-actions">${button('重试本回合','retry',true)}</div>`:''}</section>`;
  }

  function hintCards(support) {
    const hints=Array.isArray(support.hints)?support.hints:[],state=S.view.state,isObjective=objective(),access=teachingReferenceAccess(state);
    const latest=hints.at(-1),levels=isObjective?hints.map((hint,index)=>({hint,level:hint.level||index+1})):[1,2,3].map(level=>({hint:hints.find(row=>Number(row.level)===level)||(!hints[level-1]?.level?hints[level-1]:null),level}));
    const cards=levels.map(({hint,level})=>{
      if(!hint)return '';
      const details=Array.isArray(hint.details)?hint.details:[];
      const detailBody=details.map(row=>`<section class="hint-detail">${requestText(row.request)}<div class="markdown panel-card-body support-card-body">${markdown(shown(row.content))}</div></section>`).join('');
      return `<section class="teaching-support-card teaching-hint-card" data-hint-id="${escape(hint.id||'')}" data-hint-level="${level}"><div class="panel-card-heading support-card-meta"><strong>提示 ${level}</strong><span>已查看</span></div><div class="markdown panel-card-body support-card-body">${markdown(shown(hint.text))}</div>${detailBody?`<details class="hint-details" data-support-card="hint-${escape(hint.id||level)}"><summary>更详细的讲解<span>${details.length} 条</span></summary>${detailBody}</details>`:''}${!isObjective&&hint===latest?`<div class="panel-actions">${button('把提示讲详细','hint_detail')}</div>`:''}</section>`;
    }).join('');
    const hintCount=isObjective?Number.isInteger(support.hint_seen)&&support.hint_seen>=0?support.hint_seen:hints.length:access.hints_seen;
    const hintTotal=isObjective&&Number.isInteger(support.hint_total)&&support.hint_total>=0?support.hint_total:null;
    const hintRemaining=hintTotal===null?null:Math.max(0,hintTotal-hintCount);
    $('#teaching-hints-count').textContent=isObjective?hintTotal===null?`${hintCount} 条`:`${hintCount}/${hintTotal}`:`${hintCount}/3`;
    const canHint=isObjective?state.stage!=='complete'&&(hintRemaining===null?!(state.hint_status?.length&&state.hint_status.every(Boolean)):hintRemaining>0):access.remaining>0;
    const note=isObjective?state.stage==='complete'?'本题已完成，可在作答记录中复盘。':hintTotal===0?'请按题面选择并提交。':hintTotal!==null&&hintRemaining>0?`本题共 ${hintTotal} 条提示，还可查看 ${hintRemaining} 条；按需展开。`:hintTotal!==null?'本题提示已全部查看，可继续选择并提交。':'选择选项即可提交，提示按需查看。':access.allowed?'参考答案已解锁，也可以继续独立补充。':`还可查看 ${access.remaining} 条提示；细化已有提示不增加次数。`;
    const progress=isObjective?'':`<div class="teaching-hint-progress" aria-label="已看 ${hintCount} 条提示">${[1,2,3].map(level=>`<span class="hint-progress-dot ${level<=hintCount?'is-seen':level===hintCount+1?'is-current':''}" aria-hidden="true"></span>`).join('')}</div>`;
    const unseen=!isObjective&&access.remaining?'<p class="panel-empty teaching-panel-empty">下一条提示尚未查看。</p>':isObjective&&!cards?'<p class="panel-empty teaching-panel-empty">先独立选择，需要时再查看提示。</p>':'';
    return `${progress}<p class="panel-note">${escape(note)}</p><div class="panel-actions">${canHint?button(hintCount?'下一条提示':'查看第一条提示','hint',true):''}${!isObjective&&access.allowed?button('查看参考答案','reference',!canHint):''}</div>${liveCard('hint')}${errorCard('hint')}<div class="teaching-hint-list">${cards}${unseen}</div>`;
  }

  function explanationCards(support) {
    const rows=Array.isArray(support.explanations)?support.explanations:[],refs=Array.isArray(support.references)?support.references:[];
    $('#teaching-explanations-count').textContent=rows.length?`${rows.length} 条`:'';
    const explanations=rows.map((row,index)=>`<section class="teaching-support-card teaching-explanation-card" data-explanation-id="${escape(row.id||'')}"><div class="panel-card-heading support-card-meta"><strong>讲解 ${index+1}</strong></div>${requestText(row.request)}<div class="markdown panel-card-body support-card-body">${markdown(shown(row.content))}</div></section>`).join('');
    const references=refs.map((row,index)=>`<details class="teaching-support-card teaching-reference-card" data-support-card="reference-${escape(row.id||index)}"><summary class="panel-card-heading support-card-meta"><strong>参考答案${refs.length>1?' '+(index+1):''}</strong><span>已查看 · 展开复盘</span></summary><div class="markdown panel-card-body support-card-body">${markdown(shown(row.content))}</div></details>`).join('');
    return `<p class="panel-note">想理解某一步，可以在输入框直接追问。</p><div class="panel-actions">${button('解释这道题','explain')}</div>${liveCard('explanation')}${errorCard('explanation')}${explanations||'<p class="panel-empty teaching-panel-empty">还没有补充讲解，按自己的节奏作答。</p>'}${references}${liveCard('reference')}${errorCard('reference')}`;
  }

  function render() {
    ensureObserver();syncLayout();if(!enabled())return;
    const support=data();
    updateBody($('#teaching-hints-body'),hintCards(support));
    if(!objective())updateBody($('#teaching-explanations-body'),explanationCards(support));
    if(completedCard?.scope===identity()){
      const completed=completedCard;
      const selectors=completedCard.intent==='hint_detail'?'.hint-details':completedCard.placement==='reference'?'.teaching-reference-card':null;
      const cards=selectors?document.querySelectorAll(`#teaching-${side(completedCard.placement)}-body ${selectors}`):[];
      if(cards.length)cards[cards.length-1].open=true;
      completedCard=null;
      locateCard(completed.placement,completed.intent,true);
    }
  }

  function beginTurn(job,action,text) {
    if(!enabled())return;
    const retryPlacement=action==='retry'?errorPlacement()||data()?.pending?.placement:null;
    S.supportOutcome=null;
    job.supportScope=identity();
    job.supportPlacement=actions[action]||(placements.has(retryPlacement)?retryPlacement:null);
    if(objective()&&job.supportPlacement!=='hint')job.supportPlacement=null;
    job.requestPending=!objective()&&action==='send';
    job.hideRequest=Boolean(job.supportPlacement);
    job.requestText=text;
    job.requestMessageFloor=S.view?.messages?.length||0;
    job.supportIntent=action;
    if(job.supportPlacement)disclose(job.supportPlacement);
  }

  function streamEvent(job,value) {
    if(!value||typeof value!=='object')return;
    const previous=job.supportPlacement,next=placements.has(value.placement)?value.placement:null;
    if(value.reset||previous!==next){
      if(!previous&&next&&job.text)job.centerText=job.text;
      if(previous&&!next)job.centerText='';
      job.text='';
    }
    job.supportPlacement=next;
    job.supportIntent=value.intent||job.supportIntent;
    if(typeof value.hide_request==='boolean'){job.hideRequest=value.hide_request;job.requestPending=false;}
    if(next)disclose(next,true);
  }

  function finishTurn(job) {
    if(job?.supportPlacement&&job.supportScope===identity()){
      S.supportOutcome={scope:job.supportScope,placement:job.supportPlacement};
      if(job.done)completedCard={scope:job.supportScope,placement:job.supportPlacement,intent:job.supportIntent};
      disclose(job.supportPlacement);
    }
  }

  function hiddenMessage(message,index,job=currentJob()) {
    if(!enabled())return false;
    const support=data();
    if(message.role==='user'&&message.kind==='student')return false;
    if(message.role==='assistant'&&placements.has(support.message_placements?.[index]))return true;
    if(message.role==='user'&&placements.has(support.request_placements?.[index]))return true;
    const resumed=!job&&(S.view?.active?.teaching_support||support.pending||support.failure);
    if(resumed?.hide_request&&message.role==='user'&&message.kind==='help'&&!message.answer_text&&message.turn_id===S.view?.state.turn_id)return true;
    const isCurrentRequest=job?.requestText&&message.role==='user'&&message.content===job.requestText&&index>=job.requestMessageFloor;
    return Boolean(isCurrentRequest&&(job.requestPending||job.hideRequest));
  }

  return {enabled,objective,render,beginTurn,streamEvent,finishTurn,hiddenMessage,sidePlacement,errorPlacement};
}
