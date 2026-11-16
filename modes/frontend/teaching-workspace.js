import {escape,markdown} from './markdown.js?v=20261116-i18n';
import {bilingualTitle} from './i18n.js';

const stepTitles=['识别线索','分析线索','确定优先问题','制定解决方案','采取行动','评价结果'];
const types={single_choice:'单选题',multiple_choice:'多选题',true_false:'判断题',ordering:'排序题',short_answer:'简答题',concept_card:'概念卡',six_step_case:'六步案例'};
const objectiveTypes=['single_choice','multiple_choice','true_false','ordering'];
const text=(value,language='zh')=>typeof value==='string'?value:value?.[language]||value?.zh||value?.en||'';

export function teachingClinicalFlow(state) {
  const flow=state?.clinical_flow;
  return flow?.version==='guided_v1'&&Array.isArray(flow.steps)&&Number.isInteger(flow.current_step)&&flow.steps[flow.current_step]?flow:null;
}

export function teachingReferenceAccess(state) {
  const flow=teachingClinicalFlow(state),step=flow?.steps[flow.current_step],access=state?.reference_access;
  const actual=flow?step.hints?.length||0:state?.hint_events?.length||0;
  const seen=Math.min(3,Math.max(0,Number.isInteger(access?.hints_seen)?access.hints_seen:actual));
  const remaining=Math.min(3,Math.max(0,Number.isInteger(access?.remaining)?access.remaining:3-seen));
  const checks=flow?step.criterion_checks:state?.reviews?.at(-1)?.criterion_checks,passed=Boolean(checks?.length&&checks.every(row=>row.status==='met')&&(flow?['passed','awaiting_next'].includes(step.status):state.reviews.at(-1).ready_for_summary));
  const viewed=flow?step.reference_views?.length:state?.reference_views?.length;
  return {allowed:typeof access?.allowed==='boolean'?access.allowed:seen>=3||Boolean(viewed)||passed,hints_seen:seen,remaining,reason:access?.reason||''};
}

function compactFeedback(view,context={}) {
  const rows=Array.isArray(view.criteria)?view.criteria.filter(row=>row&&['met','partial','missing','incorrect','uncertain'].includes(row.status)):[];
  const covered=Number.isInteger(view.covered)&&view.covered>=0?view.covered:rows.filter(row=>row.status==='met').length,total=Number.isInteger(view.total)&&view.total>=0?view.total:rows.length;
  const labels={met:'已覆盖',partial:'待补充',missing:'待补充',uncertain:'待确认',incorrect:'需修正'};
  const messageTarget=ref=>{
    if(!ref||typeof ref.message_at!=='string'||!Array.isArray(context.messages)||!Number.isInteger(context.index))return null;
    let start=0;for(let i=context.index-1;i>=0;i--)if(context.messages[i]?.kind==='learning_section'){start=i+1;break;}
    let found=[];for(let i=start;i<context.index;i++){const message=context.messages[i];if(message?.role==='user'&&message.kind==='student'&&message.at===ref.message_at)found.push(i);}
    if(view.step_id){const scoped=found.filter(i=>context.messages[i].step_id===view.step_id||context.messages[i].annotations?.some(row=>row.step_id===view.step_id));if(scoped.length)found=scoped;}
    return found.length===1?found[0]:null;
  };
  const criterion=row=>{
    const refs=Array.isArray(row.evidence_refs)?row.evidence_refs:[],seen=new Set(),links=[];
    for(const ref of refs){const index=messageTarget(ref);if(index===null||seen.has(index))continue;seen.add(index);links.push(`<button type="button" class="criterion-evidence-link" data-command="evidence-message" data-index="${index}" data-criterion="${escape(row.id||'')}">${row.status==='met'?'定位已覆盖原话':'查看原作答'}</button>`);}
    const help={after_hint:'提示后补充',after_reference:'参考后补充',after_explanation:'讲解后补充'}[row.answer_context];
    return `<li class="criterion-row status-${row.status}"><div class="criterion-heading"><span class="criterion-status">${labels[row.status]}</span><strong>${escape(text(row.label))}</strong></div>${row.note?`<p>${escape(text(row.note))}</p>`:''}${help||row.judgment_revised?`<small class="criterion-context">${escape([help,row.judgment_revised?'本次重新核对调整了判定':''].filter(Boolean).join(' · '))}</small>`:''}${links.length?`<div class="criterion-links">${links.join('')}</div>`:''}</li>`;
  };
  const groups=[{name:'已覆盖要点',states:['met'],tone:'covered',open:false},{name:'待补充与待确认',states:['partial','missing','uncertain'],tone:'pending',open:true},{name:'需要修正',states:['incorrect'],tone:'correction',open:true}];
  const details=groups.map(group=>{const selected=rows.filter(row=>group.states.includes(row.status));return selected.length?`<details class="feedback-section feedback-${group.tone}" ${group.open?'open':''}><summary>${group.name}<span>${selected.length} 项</span></summary><ul class="criterion-list">${selected.map(criterion).join('')}</ul></details>`:'';}).join('');
  return `<section class="compact-feedback" aria-label="本次作答核对"><div class="feedback-overview"><strong>已覆盖 ${covered}/${total} 个要点</strong><span>${view.passed?'当前要求已覆盖':'可继续补充'}</span></div>${details}${view.next_action?.text?`<p class="feedback-next">${escape(text(view.next_action.text))}</p>`:''}</section>`;
}

// The backend provides quote-checked evidence positions and compact criteria.
// Historical records fall back to their projected display text.
export function teachingFeedback(content,view,context) {
  if(view&&typeof view==='object'&&Array.isArray(view.criteria))return compactFeedback(view,context);
  const headings=['已经达到的内容','可能遗漏的内容','需要修正的内容','继续思考','What you have covered','What may be missing','What needs correction or clarification','Continue thinking'];
  const matches=[...content.matchAll(/^\*\*([^\n]+)\*\*\s*$/gm)].filter(row=>headings.includes(row[1]));
  if(matches.length!==4)return markdown(content);
  const prefix=content.slice(0,matches[0].index);
  return (prefix?markdown(prefix):'')+matches.map((row,index)=>{
    const body=content.slice(row.index+row[0].length,matches[index+1]?.index??content.length).trim();
    const empty=/^(本次未发现此类内容。|尚无当前步骤的评价。|None identified in this assessment\.|This step has not been assessed yet\.)$/.test(body);
    const open=index===3||index>0&&!empty;
    return `<details class="feedback-section" ${open?'open':''}><summary>${escape(row[1])}<span>${empty?'暂无':index===0?'查看已覆盖要点':'展开详情'}</span></summary><div class="markdown">${markdown(body)}</div></details>`;
  }).join('');
}

export function createTeachingWorkspace(C) {
  const {S,read,write,busy,openDialog,toast}=C;
  const submissions=new Set();
  const feedbackEvents=new Map();
  const current=()=>S.mode==='teaching'?S.view?.state:null;
  const identity=state=>state.session_id+':'+(state.exercise_id||state.exercise_started_at||state.bank_item?.id||'');
  const objective=state=>Boolean(state&&!teachingClinicalFlow(state)&&objectiveTypes.includes(state.bank_item?.question_type));
  const task=state=>{
    const flow=teachingClinicalFlow(state);
    return S.view?.teaching_task?.split('\n\n').slice(1).join('\n\n')||text(flow?.rubric_snapshot?.steps?.find(row=>row.id===flow.steps[flow.current_step].id)?.question,state.language);
  };
  const choices=state=>Object.entries(state.bank_item?.options||{}).filter(([key,label])=>typeof label==='string'&&/^[A-Z]$/.test(key));
  const attemptCount=state=>S.view?.answer_count??state.attempts?.length??0;
  function submittedSelection(state) {
    const kind=state.bank_item.question_type,count=attemptCount(state),review=state.reviews?.at(-1);
    const check=review?.answer_count===count?review.criterion_checks?.[0]:null;
    const answer=state.attempts?.at(-1)||'';
    if(!count)return null;
    if(kind==='true_false'){
      const match=answer.match(/^(?:我的判断\s*[:：]\s*)?(正确|错误|true|false)(?:[。.!！\s]|$)/i);
      const truth=typeof check?.selected_truth==='boolean'?check.selected_truth:match?['正确','true'].includes(match[1].toLowerCase()):null;
      return truth===null?null:{selected:[truth?'正确':'错误'],order:[]};
    }
    if(kind==='ordering'){
      const match=answer.match(/^我的顺序\s*[:：]\s*([A-Z](?:\s*(?:→|>|、|,|，|-)\s*[A-Z])*)/);
      const order=Array.isArray(check?.selected_order)?check.selected_order:match?match[1].match(/[A-Z]/g):null;
      return order?{selected:[],order}:null;
    }
    const match=answer.match(/^(?:我的选择\s*[:：]\s*)?([A-Z](?:\s*[、,，]\s*[A-Z])*)[。.!！\s]*$/);
    const selected=Array.isArray(check?.selected_options)?check.selected_options:match?match[1].match(/[A-Z]/g):null;
    return selected?{selected,order:[]}:null;
  }
  function saved(state) {
    const all=read('answer-tools',{}),key=identity(state),count=attemptCount(state);
    const row=all[key]||{selected:[],order:[],attempts:count};
    row.selected=Array.isArray(row.selected)?row.selected:[];row.order=Array.isArray(row.order)?row.order:[];
    const submitted=submittedSelection(state);
    const restoreComplete=state.stage==='complete'&&submitted&&
      (JSON.stringify(row.selected)!==JSON.stringify(submitted.selected)||JSON.stringify(row.order)!==JSON.stringify(submitted.order));
    if(!all[key]||row.attempts!==count||count>0&&!Object.hasOwn(row,'submitted')||restoreComplete){
      row.selected=submitted?.selected.slice()||[];row.order=submitted?.order.slice()||[];
      row.submitted=submitted;row.error='';row.attempts=count;delete row.prefix;
      store(state,row);
    }else if(submitted&&JSON.stringify(row.submitted)!==JSON.stringify(submitted)){
      const untouched=row.submitted?
        JSON.stringify(row.selected)===JSON.stringify(row.submitted.selected)&&JSON.stringify(row.order)===JSON.stringify(row.submitted.order):
        !row.selected.length&&!row.order.length;
      if(untouched){row.selected=submitted.selected.slice();row.order=submitted.order.slice();}
      row.submitted=submitted;store(state,row);
    }
    return row;
  }
  function store(state,row) {
    const all=read('answer-tools',{});all[identity(state)]={...row,at:Date.now()};
    write('answer-tools',Object.fromEntries(Object.entries(all).sort((a,b)=>(b[1].at||0)-(a[1].at||0)).slice(0,50)));
  }
  function orderFor(state,row) {
    const keys=choices(state).map(([key])=>key);
    return row.order.length===keys.length&&new Set(row.order).size===keys.length&&row.order.every(key=>keys.includes(key))?row.order:keys;
  }
  function answerFor(state,row) {
    const kind=state.bank_item.question_type;
    if(kind==='ordering')return '我的顺序：'+orderFor(state,row).join(' → ')+'。';
    return row.selected.length?(kind==='true_false'?'我的判断：':'我的选择：')+row.selected.join('、')+'。':'';
  }
  function result(state,row) {
    const kind=state.bank_item.question_type,review=state.reviews?.at(-1),count=attemptCount(state);
    // Old semantic reviews cannot establish a correct objective answer.
    const valid=count>0&&review?.objective_question_type===kind&&review.answer_count===count;
    const checks=valid?review.criterion_checks||[]:[];
    const same=Boolean(row.submitted&&(kind==='ordering'?
      JSON.stringify(orderFor(state,row))===JSON.stringify(row.submitted.order):
      JSON.stringify(row.selected.slice().sort())===JSON.stringify(row.submitted.selected.slice().sort())));
    const correct=checks.length>0&&checks.every(check=>check.status==='met');
    const uncertain=!checks.length||checks.some(check=>check.status==='uncertain');
    const status=correct?'correct':uncertain?'neutral':'incorrect';
    let message='';
    if(row.error)message=row.error;
    else if(count&&!valid)message='已有作答记录，请重新核对后查看结果。';
    else if(checks.length&&!same)message=kind==='ordering'?'已调整顺序，请提交答案。':'已修改选择，请提交答案。';
    else if(checks.length){
      message=correct?'回答正确。':uncertain?'答案待确认，请重新选择并提交。':
        kind==='ordering'?'顺序不正确，请调整后重新提交。':kind==='true_false'?'判断不正确，请重新选择。':
        kind==='multiple_choice'&&checks.some(check=>check.status==='partial')?'回答错误，选项不完整，请重新选择。':
        kind==='multiple_choice'?'回答错误，存在错选，请重新选择。':'回答错误。';
    }
    return {correct:correct&&same,status:same?status:'neutral',message,
      event:valid&&same?`${count}:${review.assessment_method||'objective'}:${status}:${answerFor(state,row)}`:''};
  }
  function focus() {
    const state=current(),block=document.querySelector('#teaching-focus');
    block.hidden=!state||!state.case||['setup','case','topic'].includes(state.stage);
    if(block.hidden){block.innerHTML='';block._renderedHTML='';return;}
    const flow=teachingClinicalFlow(state),step=flow?.steps[flow.current_step],language=state.language||'zh';
    const checks=step?.criterion_checks||state.reviews?.at(-1)?.criterion_checks||[];
    const met=checks.filter(row=>row.status==='met').length;
    const done=state.stage==='complete',passed=step?.status==='awaiting_next';
    const choice=objective(state);
    block.classList.toggle('choice-focus',choice);
    const choiceCorrect=choice&&result(state,saved(state)).correct;
    const access=teachingReferenceAccess(state),remaining=access.remaining,viewed=flow?step.reference_views?.length:state.reference_views?.length;
    const status=choice?choiceCorrect?'答案正确':attemptCount(state)?state.bank_item.question_type==='ordering'?'重新排序':'重新选择':state.bank_item.question_type==='ordering'?'确认顺序':'请选择':done?flow?.status==='interrupted'?'本轮已结束 · 可查看未完成记录':'本题已总结':passed?'本步已通过 · 进入下一步':viewed?'参考答案已展开 · 可继续作答':checks.length?`已覆盖 ${met}/${checks.length} 个要点 · 可直接补充`:remaining<3?'先试着整理一两句，需要时继续求助':'先独立作答，需要时再获取提示';
    const question=flow?task(state):state.case.split(/\n\n[A-Z]\. /)[0];
    const support=!choice&&(!done||!access.allowed)?`<p class="teaching-support-note"><span class="hint-progress">提示 ${access.hints_seen}/3 · 还可 ${remaining} 次</span>${viewed?'已查看参考答案，可用自己的话继续补充。':access.allowed?'需要时可查看参考答案。':done?'本轮已总结，仍可查看剩余提示；三条提示后解锁参考答案。':'可追问或把已有提示讲详细；看完三条提示后解锁参考答案。'}</p>`:'';
    const html=`<div class="teaching-focus-top"><strong>${flow?`第 ${flow.current_step+1}/6 步 · ${stepTitles[flow.current_step]}`:escape(types[state.bank_item?.question_type]||'当前练习')}</strong><span class="focus-status ${(choice?choiceCorrect:passed||done)?'covered':''}">${escape(status)}</span></div><p class="focus-question">${escape(done?'回看总结和参考要点，或选择下一道练习继续。':question)}</p>${support}<div class="focus-links"><button type="button" data-command="teaching-case">${flow?'病例原文':'完整题目'}</button>${flow&&!done?'<button type="button" data-command="teaching-task">本步任务</button>':''}<button type="button" data-command="catalog">换一道练习</button></div>`;
    if(block._renderedHTML!==html){block.innerHTML=html;block._renderedHTML=html;}
  }
  function answerTools() {
    const state=current(),block=document.querySelector('#answer-tools');
    block.hidden=!objective(state)||['setup','case','topic'].includes(state.stage);
    if(block.hidden){block.innerHTML='';block._renderedHTML='';block._feedbackToken='';block.dataset.result='neutral';block.setAttribute('aria-busy','false');block.classList.remove('objective-feedback-fresh');return;}
    const kind=state.bank_item.question_type,rows=choices(state),row=saved(state);
    if(kind!=='true_false'&&!rows.length){block.hidden=true;block.innerHTML='';return;}
    const feedback=result(state,row),waiting=submissions.has(identity(state))||busy()||Boolean(state.pending);
    const locked=waiting||state.stage==='complete'||feedback.correct;
    const valid=kind==='ordering'?rows.length>0:kind==='multiple_choice'?row.selected.length>0:row.selected.length===1;
    const help=kind==='multiple_choice'?'选择所有正确选项，然后提交答案。':kind==='ordering'?'用上下按钮调整顺序，然后提交答案。':kind==='true_false'?'选择正确或错误，然后提交答案。':'选择一个选项，然后提交答案。';
    const stem=state.case.replace(/\n\s*A[.．、):：]\s*[\s\S]*$/,'').replace(/[；;]\s*理由可补充[。.]/g,'。');
    let body;
    if(kind==='ordering'){
      const order=orderFor(state,row);
      body=`<ol class="answer-order" aria-label="当前答案顺序">${order.map((key,index)=>`<li><span><b>${escape(key)}</b><span data-no-i18n>${escape(state.bank_item.options[key])}</span></span><div><button type="button" data-command="answer-move" data-key="${key}" data-direction="-1" aria-label="上移选项${key}" ${locked||index===0?'disabled':''}>↑</button><button type="button" data-command="answer-move" data-key="${key}" data-direction="1" aria-label="下移选项${key}" ${locked||index===order.length-1?'disabled':''}>↓</button></div></li>`).join('')}</ol>`;
    }else{
      const values=kind==='true_false'?[['正确','该表述正确'],['错误','该表述错误']]:rows;
      body=`<fieldset class="answer-choices" ${locked?'disabled':''}><legend class="sr-only">${escape(types[kind])}选项</legend>${values.map(([key,label])=>`<label class="answer-choice ${row.selected.includes(key)?'selected':''}"><input type="${kind==='multiple_choice'?'checkbox':'radio'}" name="teaching-answer" value="${escape(key)}" ${row.selected.includes(key)?'checked':''}><span><b>${escape(key)}</b><span ${kind==='true_false'?'':'data-no-i18n'}>${escape(label)}</span></span></label>`).join('')}</fieldset>`;
    }
    const html=`<div class="answer-tools-heading"><strong>${escape(types[kind])}作答</strong><button type="button" data-command="answer-clear" ${locked?'disabled':''}>${kind==='ordering'?'重置顺序':'清除选择'}</button></div><div class="objective-stem">${markdown(stem)}</div><p class="objective-help">${help}</p>${body}<div class="objective-submit-row"><button type="button" class="button primary objective-submit" data-command="objective-submit" aria-describedby="objective-feedback" ${locked||!valid?'disabled':''}>${waiting?'正在提交…':feedback.correct?'已答对':state.stage==='complete'?'本轮已结束':'提交答案'}</button><span class="objective-keyboard-hint">也可按 Enter 提交</span></div>`;
    if(!block.querySelector('.objective-controls')){
      block.innerHTML='<div class="objective-controls"></div><div id="objective-feedback" class="objective-feedback" role="status" aria-live="polite" aria-atomic="true"></div>';
      block._renderedHTML='';
    }
    if(block._renderedHTML!==html){block.querySelector('.objective-controls').innerHTML=html;block._renderedHTML=html;}
    block.tabIndex=0;block.setAttribute('aria-label',types[kind]+'作答');block.setAttribute('aria-busy',String(waiting));
    block.dataset.result=waiting?'pending':feedback.status;
    if(waiting||feedback.status==='neutral')block.classList.remove('objective-feedback-fresh');
    const live=block.querySelector('#objective-feedback'),message=waiting?'正在提交答案…':feedback.message;
    if(live.textContent!==message)live.textContent=message;
    live.hidden=false;
    if(!waiting&&feedback.event&&row.feedbackEvent!==feedback.event&&feedbackEvents.get(identity(state))!==feedback.event&&feedback.status!=='neutral'){
      feedbackEvents.set(identity(state),feedback.event);
      row.feedbackEvent=feedback.event;store(state,row);block._feedbackToken=feedback.event;
      block.classList.remove('objective-feedback-fresh');
      requestAnimationFrame(()=>{if(!block.hidden&&block._feedbackToken===feedback.event)block.classList.add('objective-feedback-fresh');});
    }
    if(!block._objectiveListeners){
      block._objectiveListeners=true;
      block.addEventListener('change',event=>{
        const state=current(),node=event.target;
        if(!objective(state)||!node.matches('input[name="teaching-answer"]')||busy()||submissions.has(identity(state))||state.pending||state.stage==='complete')return;
        const next=saved(state);next.selected=[...block.querySelectorAll('input:checked')].map(input=>input.value);next.error='';
        store(state,next);answerTools();block.querySelector(`input[value="${CSS.escape(node.value)}"]`)?.focus();
      });
      block.addEventListener('keydown',event=>{
        if(event.key!=='Enter'||event.isComposing||event.repeat||event.shiftKey||event.ctrlKey||event.metaKey||event.altKey||event.target.closest('[data-command="answer-move"],[data-command="answer-clear"],a'))return;
        event.preventDefault();event.stopPropagation();void submit();
      });
      block.addEventListener('animationend',event=>{
        if(event.animationName==='objective-success-glow'||event.animationName==='objective-error-flash')block.classList.remove('objective-feedback-fresh');
      });
    }
  }
  // Objective controls own their drafts; typing in the chat composer has no effect.
  function syncDraft() {}
  async function submit() {
    const state=current();if(!objective(state))return;
    const key=identity(state),row=saved(state);
    if(submissions.has(key)||busy()||state.pending||state.stage==='complete'||result(state,row).correct)return;
    const answer=answerFor(state,row);
    if(!answer||state.bank_item.question_type!=='multiple_choice'&&state.bank_item.question_type!=='ordering'&&row.selected.length!==1)return;
    if(typeof C.submitObjective!=='function'){toast('提交暂时不可用，请刷新页面后重试',true);return;}
    if(state.bank_item.question_type==='ordering')row.order=orderFor(state,row).slice();
    row.error='';store(state,row);submissions.add(key);answerTools();
    try{
      const submitted=await C.submitObjective(answer);
      if(submitted===false&&identity(current()||{})===key){const next=saved(current());next.error='答案尚未提交，请重试。';store(current(),next);}
    }catch(error){
      if(current()&&identity(current())===key){const next=saved(current());next.error='提交失败，已保留选择，请重试。';store(current(),next);}
      toast(error?.message||'提交失败，请重试',true);
    }finally{submissions.delete(key);render();}
  }
  function render(){focus();answerTools();}
  function command(name,target) {
    const state=current();
    if(!name.startsWith('teaching-')&&!name.startsWith('answer-')&&name!=='objective-submit')return false;
    if(!state)return true;
    if(name==='teaching-case'||name==='teaching-task'){
      openDialog(name==='teaching-task'?'本步任务':teachingClinicalFlow(state)?'病例原文':'完整题目',`<p class="panel-note">${escape(bilingualTitle(state.bank_item?.title)||'当前教学练习')}</p><div class="markdown">${markdown(name==='teaching-task'?task(state):state.case)}</div>${name==='teaching-task'?'<p class="panel-note">只需回答当前步骤。已经答对的内容会继续保留，可以分多次补充。</p>':''}`);
      return true;
    }
    if(name==='objective-submit'){void submit();return true;}
    if(busy()||submissions.has(identity(state))||state.pending||state.stage==='complete')return true;
    const row=saved(state);
    if(name==='answer-move'){
      const order=orderFor(state,row).slice();
      const index=order.indexOf(target.dataset.key),next=index+Number(target.dataset.direction);
      if(index>=0&&next>=0&&next<order.length){[order[index],order[next]]=[order[next],order[index]];row.order=order;row.error='';store(state,row);answerTools();(document.querySelector(`[data-command="answer-move"][data-key="${target.dataset.key}"][data-direction="${target.dataset.direction}"]:not(:disabled)`)||document.querySelector(`[data-command="answer-move"][data-key="${target.dataset.key}"]:not(:disabled)`))?.focus();}
      return true;
    }
    if(name==='answer-clear'){row.selected=[];row.order=[];row.error='';store(state,row);answerTools();return true;}
    return false;
  }
  return {render,command,syncDraft};
}
