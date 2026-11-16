import {getChatTasks} from './chat-tasks.js?v=20261116-i18n';
import {startLocalization,localize,t,interfaceLanguage,interfaceLocale,bilingualTitle} from './i18n.js';
import {icon,hydrateIcons} from './icons.js';
import {escape,markdown,renderMessage} from './markdown.js?v=20261116-i18n';
import {patchHTML,moveElement,highlightedExcerpt,createReadingNavigation,locateSearchResult} from './reading.js?v=20261116-brand';
import {api,stream} from './api.js';
import {createPanels} from './panels.js?v=20261116-i18n';
import {createTeachingWorkspace,teachingFeedback,teachingClinicalFlow,teachingReferenceAccess} from './teaching-workspace.js?v=20261116-i18n';
import {createTeachingSupport} from './teaching-support.js?v=20261116-teaching-help';
import {createWorkspaceFeatures} from './workspace-features.js?v=20261012-issue-repairs';

const $=selector=>document.querySelector(selector);
function stableHTML(node,html){if(node._renderedHTML!==html){node.innerHTML=html;node._renderedHTML=html;}}
const modes={chat:'普通问答',teaching:'教学练习'};
const publicMode=mode=>Object.prototype.hasOwnProperty.call(modes,mode);
const S={mode:'chat',view:null,history:[],total:0,archived:false,search:'',jobs:new Map(),files:new Map(),follow:true,navigating:false,preparing:false,request:0,historyRequest:0};
const reading=createReadingNavigation({onNavigate:()=>{S.follow=false;$('#jump-latest').hidden=false;}});
const read=(key,fallback=null)=>{try{return JSON.parse(localStorage.getItem('mdrpi.'+key))??fallback;}catch(_){return fallback;}};
const write=(key,value)=>{try{localStorage.setItem('mdrpi.'+key,JSON.stringify(value));return true;}catch(_){return false;}};
const sid=()=>S.view?.state.session_id;
const draftKey=()=>sid()||'new-'+S.mode;
const currentFiles=()=>S.files.get(draftKey())||[];
const currentJob=()=>S.jobs.get(sid());
const fileQueues=new Map();
const inputLimit=()=>S.mode==='performance'?1000:12000;
const inputLength=value=>Array.from(value).length;
const busy=()=>Boolean(currentJob()||S.view?.active||S.navigating||S.preparing);
function defaults(mode=S.mode){return {mode,policy:mode==='performance'?'none':'web',web_mode:'live',timezone:'Asia/Shanghai',response_language:interfaceLanguage(),memory_profile:'',learning_profile:'',...read('defaults.'+mode,{}),...(mode==='performance'?{}:{policy:'web'})};}
function saveDefaults(mode,value){write('defaults.'+mode,value);}
function saveDraft(){const values=read('drafts',{});values[draftKey()]={text:$('#composer-input').value,at:Date.now()};const trimmed=Object.fromEntries(Object.entries(values).sort((a,b)=>b[1].at-a[1].at).slice(0,50));const saved=write('drafts',trimmed);$('#draft-note').textContent=saved&&$('#composer-input').value?'草稿已保存':'';}
function loadDraft(){const value=read('drafts',{})[draftKey()]?.text||'';$('#composer-input').value=value;$('#draft-note').textContent=value?'草稿已保存':'';resizeInput();}
const nativeFieldSizing=CSS.supports('field-sizing','content');
let inputResizeFrame;
function resizeInput(){
  updateSend();
  if(nativeFieldSizing)return;
  // Keep auto-growing drafts on older browsers; coalesce repeated updates.
  cancelAnimationFrame(inputResizeFrame);
  inputResizeFrame=requestAnimationFrame(()=>{
    const field=$('#composer-input');
    field.style.height='auto';
    field.style.height=Math.min(field.scrollHeight,parseFloat(getComputedStyle(field).maxHeight)||210)+'px';
  });
}
if(!nativeFieldSizing){
  let previousWidth=0;
  new ResizeObserver(([entry])=>{
    if(entry.contentRect.width!==previousWidth){previousWidth=entry.contentRect.width;resizeInput();}
  }).observe($('#composer-input'));
  window.addEventListener('resize',resizeInput);
}
function toast(message,error=false){const node=document.createElement('div');node.className='toast'+(error?' error':'');node.textContent=message;$('#toasts').append(node);setTimeout(()=>node.remove(),error?6500:3200);}
function closeDialog(){$('#dialog').close();}
function openDialog(title,body,wide=false){closeMobileSidebar();closePopover();const node=$('#dialog');node.dataset.panelTitle=title;node.classList.toggle('wide',wide);$('#dialog-title').textContent=title;$('#dialog-body').innerHTML=body;if(!node.open)node.showModal();}
function closePopover(){$('#popover').hidden=true;}
function appearance(){try{return localStorage.getItem('mdrpi.appearance')||'system';}catch(_){return 'system';}}
function setTheme(value){try{localStorage.setItem('mdrpi.appearance',value);}catch(_){}document.documentElement.dataset.theme=value==='system'?(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'):value;document.querySelectorAll('[data-theme-value]').forEach(node=>node.classList.toggle('selected',node.dataset.themeValue===value));const themeButton=$('[data-command="theme"]');themeButton.innerHTML=icon(document.documentElement.dataset.theme==='dark'?'sun':'moon');}
const sidebarQuery=matchMedia('(max-width: 760px)');
const sidebarInertState=new Map();
function syncSidebarState(){
  const open=sidebarQuery.matches?$('#app').classList.contains('sidebar-open'):!$('#app').classList.contains('sidebar-collapsed');
  document.querySelectorAll('[data-command="sidebar"]').forEach(button=>button.setAttribute('aria-expanded',String(open)));
}
function openSidebar(){
  if($('#dialog').open)return;
  if(sidebarQuery.matches){
    if(!$('#app').classList.contains('sidebar-open')){
      closePopover();
      $('#app').classList.add('sidebar-open');
      $('#sidebar').setAttribute('role','dialog');
      $('#sidebar').setAttribute('aria-modal','true');
      // This navigation is also a desktop sidebar. Keep its existing layout,
      // while making the mobile overlay modal for keyboard and assistive tech.
      document.querySelectorAll('.main,#artifact-panel,.skip-link').forEach(node=>{
        sidebarInertState.set(node,node.inert);node.inert=true;
      });
      $('.sidebar-close').focus({preventScroll:true});
    }
  }else{
    $('#app').classList.remove('sidebar-collapsed');write('collapsed',false);
  }
  syncSidebarState();
}
function toggleSidebar(){
  if(sidebarQuery.matches){
    if($('#app').classList.contains('sidebar-open'))closeMobileSidebar();else openSidebar();
  }else{
    const collapsed=$('#app').classList.toggle('sidebar-collapsed');
    write('collapsed',collapsed);syncSidebarState();
  }
}
function closeMobileSidebar(){
  const wasOpen=$('#app').classList.contains('sidebar-open');
  $('#app').classList.remove('sidebar-open');
  $('#sidebar').removeAttribute('role');$('#sidebar').removeAttribute('aria-modal');
  sidebarInertState.forEach((value,node)=>{node.inert=value;});sidebarInertState.clear();
  syncSidebarState();
  if(wasOpen){closePopover();$('.topbar [data-command="sidebar"]').focus({preventScroll:true});}
}
sidebarQuery.addEventListener('change',closeMobileSidebar);
function updateURL(push=false){const url=new URL(location.href);if(sid())url.searchParams.set('session',sid());else url.searchParams.delete('session');url.searchParams.delete('mode');const method=push?'pushState':'replaceState';if(url.href!==location.href)history[method]({},'',url);write('last',{mode:S.mode,sid:sid()||''});if(sid())write('last.'+S.mode,sid());}
function remember(push=false){const state=S.view?.state;if(state){saveDefaults(S.mode,{policy:state.policy,web_mode:state.web_mode,replay_run_id:state.replay_run_id,timezone:state.timezone,response_language:state.response_language,memory_profile:state.memory_profile,learning_profile:state.learning_profile});}updateURL(push);}
async function applyView(value){S.view=value;S.mode=value.state.mode;remember();render();}
async function refresh(){if(!sid())return;const target=sid(),view=await api(`/sessions/${target}/view`);if(sid()===target)await applyView(view);}
async function ensureSession(){const waitingKey=draftKey();while(fileQueues.has(waitingKey))await fileQueues.get(waitingKey);if(S.view)return S.view.state;if(S.creation)return S.creation;const key=draftKey(),files=currentFiles(),draft=$('#composer-input').value,mode=S.mode,request=S.request;S.creation=(async()=>{const state=await api('/sessions',defaults(mode)),view=await api(`/sessions/${state.session_id}/view`);S.files.set(state.session_id,files);S.files.delete(key);const drafts=read('drafts',{});drafts[state.session_id]={text:draft,at:Date.now()};delete drafts[key];write('drafts',drafts);write('last.'+mode,state.session_id);if(request===S.request&&mode===S.mode&&!S.view){S.view=view;remember();render();}return state;})();try{return await S.creation;}finally{S.creation=null;}}
async function navigate(target,push=true){if(sid()===target){closeDialog();closeMobileSidebar();return;}saveDraft();S.navigating=true;updateSend();const request=++S.request;try{const view=await api(`/sessions/${encodeURIComponent(target)}/view`);if(request!==S.request)return;
  // Research sessions remain available to internal scripts/APIs, not this UI.
  if(!publicMode(view.state?.mode)){
    S.view=null;S.mode='chat';S.search='';S.archived=false;S.follow=true;S.navigating=false;
    $('#history-search').value='';updateURL(false);closeDialog();closeMobileSidebar();render();await loadHistory();return;
  }
  S.view=view;S.mode=view.state.mode;S.search='';S.archived=false;$('#history-search').value='';S.follow=true;S.navigating=false;remember(push);closeDialog();closeMobileSidebar();render();await loadHistory();scrollLatest();}finally{if(request===S.request){S.navigating=false;updateSend();}}}
async function newConversation(){saveDraft();S.request++;S.navigating=false;S.view=null;S.follow=true;write('last.'+S.mode,'');updateURL(true);closeDialog();closePopover();closeMobileSidebar();render();$('#composer-input').focus();await loadHistory();}
async function selectMode(mode){if(!publicMode(mode)||S.mode===mode)return;saveDraft();S.request++;S.mode=mode;S.view=null;S.search='';S.archived=false;$('#history-search').value='';S.follow=true;loadDraft();const target=read('last.'+mode);closeDialog();closeMobileSidebar();if(target){try{await navigate(target);return;}catch(_){write('last.'+mode,'');}}updateURL(true);render();await loadHistory();}

const panels=createPanels({S,api,defaults,saveDefaults,appearance,openDialog,closeDialog,ensureSession,refresh,render,toast,loadHistory,startTurn,copy});
const teachingWorkspace=createTeachingWorkspace({S,read,write,busy,openDialog,saveDraft,resizeInput,toast,submitObjective});
const features=createWorkspaceFeatures({S,api,sid,busy,read,write,toast,navigate,startTurn,learningAction,renderMessages,saveDraft,resizeInput});
const teachingSupport=createTeachingSupport({S,busy,currentJob,isObjectiveQuestion:state=>isObjectiveQuestion(state),teachingReferenceAccess});

function groupLabel(row){if(row.pinned)return '置顶';const now=new Date(),date=new Date(row.updated_at);const a=new Date(now.getFullYear(),now.getMonth(),now.getDate()),b=new Date(date.getFullYear(),date.getMonth(),date.getDate());const days=Math.round((a-b)/86400000);return days===0?'今天':days===1?'昨天':days<7?'最近 7 天':'更早';}
async function loadHistory(more=false){const request=++S.historyRequest,mode=S.mode,offset=more?S.history.length:0;const data=await api(`/conversations?mode=${mode}&q=${encodeURIComponent(S.search)}&archived=${S.archived}&offset=${offset}&limit=30`);if(request!==S.historyRequest||mode!==S.mode)return;S.history=more?S.history.concat(data.items):data.items;S.total=data.total;renderHistory();}
function renderHistory(){let previous='';$('#history-label').textContent=S.search?'搜索结果':S.archived?'已归档会话':'最近对话';const html=S.history.map(row=>{const label=groupLabel(row),group=label===previous?'':`<div class="history-group">${label}</div>`;previous=label;return `${group}<div class="history-row ${row.session_id===sid()?'current':''}"><button class="history-item" data-command="open-session" data-sid="${row.session_id}" title="${escape(row.title)}"><span class="history-item-text"><span class="history-item-title">${escape(row.title)}</span>${row.search_match?`<span class="history-excerpt">${highlightedExcerpt(row.search_match)}</span>`:''}</span>${row.status==='running'?'<i class="running-dot"></i>':''}</button><button class="icon-button row-menu" data-command="row-menu" data-sid="${row.session_id}" aria-label="${escape(row.title)}的选项">${icon('more')}</button></div>`;}).join('');$('#history-list').innerHTML=html||`<div class="history-empty">${S.search?'没有找到相关对话。':S.archived?'这里暂时没有归档会话。':'从新对话开始。<br>聊过的内容会保存在这里。'}</div>`;if(S.history.length<S.total)$('#history-list').insertAdjacentHTML('beforeend','<button class="history-more" data-command="history-more">加载更多对话</button>');}
function renderQuickTasks(){
  const block=$('#chat-shortcuts'),hasMessages=(S.view?.messages||[]).some(message=>message.kind!=='welcome');
  block.hidden=S.mode!=='chat'||(!hasMessages&&!currentJob());
  if(!block.hidden)block.innerHTML=getChatTasks().map(task=>`<button type="button" data-command="prompt" data-value="${escape(task.prompt)}">${escape(task.title)}</button>`).join('');
}

function renderWelcome(){const block=$('#welcome'),state=S.view?.state;const empty=!currentJob()&&(!state||!(S.view.messages||[]).some(message=>message.kind!=='welcome'));block.hidden=!empty;if(!empty)return;
  const content={chat:{kicker:'YOUR EVERYDAY ASSISTANT',title:'把问题交给我，<br><em>一起找到答案。</em>',intro:'从日常问答到专业护理，从一份资料到一个想法。<br>随时提问，也可以带上文件一起聊。',cards:getChatTasks().map(task=>[task.icon,task.title,task.note,'prompt',task.prompt])},teaching:{kicker:'MDRPI · PBL · 六步临床推理',title:'从一个MDRPI案例开始，<br><em>完成六步临床推理。</em>',intro:'先独立判断，再获得分层提示与针对性反馈；<br>案例按六步推理，知识题按题型作答；完成后复盘。',cards:[['book','标准化案例训练','从题库选择案例与知识练习','catalog',''],['edit','导入自有案例','带入脱敏案例，练习分析与判断','turn','choose_case'],['spark','定制案例训练','选择主题，准备适合你的案例','turn','choose_topic'],['clock','复盘与学习档案','查看训练记录，继续练习与复习','learning-profile','']]},performance:{kicker:'REVIEW & EVALUATE',title:'让一次测试，<br><em>留下完整记录。</em>',intro:'提交病例，运行六步回答与反思流程。<br>完成后可查看结果并导出测试记录。',cards:[['file','输入测试病例','将病例正文粘贴到下方输入框','focus',''],['settings','设置证据方式','实时联网、关闭联网或回放证据','settings','']]}}[S.mode];
  block.innerHTML=`<div class="welcome-kicker">${content.kicker}</div><h1>${content.title}</h1><p class="welcome-intro">${content.intro}</p><div class="starter-grid">${content.cards.map(([shape,title,note,command,value])=>`<button class="starter" data-command="${command}" data-value="${escape(value)}"><span class="starter-icon">${icon(shape)}</span><span><strong>${title}</strong><small>${note}</small></span><span class="starter-arrow">${icon('arrow')}</span></button>`).join('')}</div><div class="welcome-footer">${icon('shield')}对话会保存，可从侧栏随时继续。</div>`;}

function renderStage(){const block=$('#stage-strip'),state=S.view?.state;block.hidden=S.mode!=='teaching'||!state||['setup','case','topic'].includes(state.stage);if(block.hidden)return;
  const flow=teachingClinicalFlow(state);
  if(flow?.version==='guided_v1'){
    const titles=['识别线索','分析线索','确定优先问题','制定解决方案','采取行动','评价结果'];
    stableHTML(block,`<span class="stage-description">${state.case_origin==='generated_case'?'AI生成·仅供练习':state.case_origin==='user_deidentified_case'?'自有脱敏案例':state.bank_item?.kind==='standardized_case'?'标准化案例·形成性练习':'六步练习'} · ${flow.status==='interrupted'?'已中断':flow.status==='completed'?'已完成':'当前 '+(flow.current_step+1)+'/6'}<button data-command="learning-progress">学习进度</button></span><div class="clinical-steps">${flow.steps.map((row,index)=>`<button data-command="step-record" data-step="${index}" class="stage-step ${index===flow.current_step?'current':''}"><i>${['passed','awaiting_next'].includes(row.status)?'✓':index+1}</i>${titles[index]}</button>`).join('')}</div>`);
    return;
  }
  const step=state.stage==='complete'?3:['probe','hint','reflect'].includes(state.stage)?2:1;stableHTML(block,['选题','作答','反馈','总结'].map((title,index)=>`<span class="stage-step ${index===step?'current':''}"><i>${index<step?'✓':index+1}</i>${title}</span>`).join('')+`<span class="stage-description">${state.bank_item?.title?escape(bilingualTitle(state.bank_item.title)):'当前练习'}<button data-command="learning-progress">学习进度</button></span>`);}

const isObjectiveQuestion=state=>!teachingClinicalFlow(state)&&['single_choice','multiple_choice','true_false','ordering'].includes(state?.bank_item?.question_type);
function objectiveLayout(state){
  const active=S.mode==='teaching'&&isObjectiveQuestion(state)&&state.case&&!['setup','case','topic'].includes(state.stage);
  $('.composer-zone').hidden=Boolean(active);
  const area=$('#scroll-area'),messages=$('#messages'),panel=$('#answer-tools');
  let records=$('#objective-records');
  if(active){
    if(!records){records=document.createElement('details');records.id='objective-records';records.className='feedback-section objective-records';const summary=document.createElement('summary');summary.textContent='作答记录与参考答案';records.append(summary);area.insertBefore(records,messages);}
    if(panel.parentElement!==area||panel.nextElementSibling!==records)area.insertBefore(panel,records);
    if(messages.parentElement!==records)moveElement(records,messages);
  }else{
    if(records){moveElement(area,messages,records);records.remove();}
    const actions=$('#action-area');
    if(panel.parentElement!==area||panel.nextElementSibling!==actions)area.insertBefore(panel,actions);
  }
}
async function submitObjective(answer){
  const state=S.view?.state;
  if(!isObjectiveQuestion(state)||busy()||state.pending||state.stage==='complete')return false;
  const target=sid(),before=S.view.answer_count??state.attempts.length;
  await startTurn(answer,'send',{},true);
  if(sid()!==target)return false;
  if(S.view?.state.error||S.view?.connection_error)throw new Error(S.view.state.error||S.view.connection_error);
  return (S.view.answer_count??S.view.state.attempts.length)>before;
}
function render(){const state=S.view?.state;renderQuickTasks();document.title=`${S.view?.title||t(modes[S.mode])} · MDRPI`;$('#header-title').textContent=S.view?.title||t({chat:'新对话',teaching:'新的教学练习',performance:'新的性能测试'}[S.mode]);$('#mode-pill').textContent=modes[S.mode];$('#header-subtitle').textContent=state?.learning_profile?'学习档案 · '+state.learning_profile:state?.memory_profile?'偏好档案 · '+state.memory_profile:{chat:'从一个问题开始',teaching:'尝试作答，按自己的节奏学习',performance:'六步回答与反思评测'}[S.mode];document.querySelectorAll('[data-mode]').forEach(node=>node.classList.toggle('active',node.dataset.mode===S.mode));$('#composer-input').removeAttribute('maxlength');$('#composer-input').placeholder=S.mode==='teaching'?state?.stage==='complete'?isObjectiveQuestion(state)?'本题已总结，选择下一道练习继续':'可以继续追问，或选择下一道练习…':state&&['answer','probe','hint','reflect'].includes(state.stage)?isObjectiveQuestion(state)?'请选择选项或输入字母…':'可以作答，也可以求助、追问或让我讲详细一些…':'输入练习主题，或选择上方的练习方式…':S.mode==='performance'?'在这里粘贴病例正文…':'有问题，尽管问…';objectiveLayout(state);features.sync();renderWelcome();renderStage();renderMessages();renderAttachments();renderActions();renderError();renderFiles();renderHistory();loadDraft();teachingWorkspace.render();renderLive();setTheme(appearance());}

let messagesSession;
function renderMessages(){
  teachingSupport.render();
  let messages=S.view?.messages||[];const job=currentJob();
  const accepted=job?.user&&messages.some((message,index)=>message.role==='user'&&message.content===job.user&&(Number.isInteger(job.requestMessageFloor)?index>=job.requestMessageFloor:new Date(message.at).getTime()>=job.started-1000));
  if(job?.user&&!accepted)messages=messages.concat({role:'user',content:job.user,at:new Date(job.started).toISOString(),kind:'user'});
  S.displayedMessages=messages;
  // Keep the full array and its original indices for copying, editing and evidence links.
  const container=$('#messages');
  if(messagesSession!==sid()){container.replaceChildren();delete container._patchedHTML;messagesSession=sid();}
  patchHTML(container,messages.map((m,i)=>teachingSupport.hiddenMessage(m,i,job)?'':renderMessage(m,i,{mode:S.mode,messages,teachingFeedback:S.mode==='teaching'?teachingFeedback:null,editor:features.editor(m),canEdit:!busy()&&m.role==='user'&&(S.mode==='chat'&&m.kind==='user'||S.mode==='teaching'&&m.kind==='student'&&S.view.state.messages.some(row=>row.at===m.at))})).join(''));
  reading.refresh(S.mode);
  if(S.follow)requestAnimationFrame(()=>{if(S.follow)scrollLatest();});
}
function scrollLatest(){const node=$('#scroll-area'),latest=$('#messages').lastElementChild;
  if(isObjectiveQuestion(S.view?.state)&&!$('#answer-tools').hidden){const panel=$('#answer-tools');node.scrollTop+=panel.getBoundingClientRect().top-node.getBoundingClientRect().top-12;}
  else if(isObjectiveQuestion(S.view?.state)&&latest){node.scrollTop+=latest.getBoundingClientRect().top-node.getBoundingClientRect().top-12;}
  else node.scrollTop=node.scrollHeight;
  S.follow=true;$('#jump-latest').hidden=true;}
function updateSend(){
  const state=S.view?.state,button=$('#send-button'),working=busy(),preparingFiles=fileQueues.has(draftKey());
  const length=inputLength($('#composer-input').value),limit=inputLimit(),tooLong=length>limit,note=$('#composer-limit-note');
  note.hidden=!length&&!preparingFiles;
  note.classList.toggle('over-limit',tooLong);
  note.textContent=tooLong?`已保留全文 ${length} 字，超过 ${limit} 字上限；请修改后发送。`:preparingFiles?'正在准备附件，请稍候…':`${length} / ${limit} 字`;
  $('#composer-input').setAttribute('aria-invalid',String(tooLong));
  $('#artifact-revision-form button[type=submit]').disabled=working||S.mode!=='chat';
  const blocked=state?.pending&&!state.cancelled&&!working||state?.stage==='complete'&&(state.mode!=='teaching'||isObjectiveQuestion(state));
  button.innerHTML=icon(working?'stop':'arrow-up');button.setAttribute('aria-label',working?'停止生成':'发送消息');button.classList.toggle('stopping',working);
  button.disabled=S.navigating||S.preparing||(working?state?.mode==='performance'||Boolean(currentJob()?.stopping||currentJob()?.uploading):preparingFiles||tooLong||blocked||(!$('#composer-input').value.trim()&&!currentFiles().length));
}

function renderAttachments(){const files=currentFiles(),saved=S.view?.state.attachments?.filter(file=>file.active!==false)||[];const block=$('#attachment-list');block.hidden=!files.length&&!saved.length;block.innerHTML=saved.map(file=>`<div class="attachment-card">${icon('file')}<button type="button" class="attachment-details" data-command="file-details" data-file="${file.file_id}" aria-label="查看${escape(file.name)}的读取情况"><strong>${escape(file.name)}</strong><small>${file.warnings?.length?'有读取提示，请查看':'已加入参考资料'}</small></button><button type="button" data-command="remove-file" data-file="${file.file_id}" aria-label="移除${escape(file.name)}">${icon('x')}</button></div>`).join('')+files.map((file,index)=>`<div class="attachment-card ${file.error?'failed':''}">${icon('file')}<span><strong>${escape(file.file.name)}</strong><small>${escape(file.error||(file.uploading?'正在读取…':formatBytes(file.file.size)+' · 待发送'))}</small></span><button type="button" data-command="remove-local-file" data-index="${index}" aria-label="移除${escape(file.file.name)}">${icon('x')}</button></div>`).join('');}
function formatBytes(value){return value>=1048576?(value/1048576).toFixed(1)+' MB':Math.max(1,Math.round(value/1024))+' KB';}
function addFiles(list){
  if(S.creation||S.preparing||currentJob()?.uploading){toast('会话或附件正在提交，请完成后再添加文件');return;}
  if(S.mode==='performance'){toast('性能测试使用病例文本；文件可在问答或教学模式添加',true);return;}
  const key=draftKey(),allowed=/\.(pdf|docx|pptx|txt|md|csv|xlsx|png|jpe?g|webp)$/i;
  const candidates=Array.from(list).filter(file=>{if(!allowed.test(file.name)){toast(file.name+'：请使用支持的文档或图片格式',true);return false;}if(file.size>20*1048576){toast(file.name+' 超过20MB',true);return false;}return true;});
  if(!candidates.length)return;
  const pending=(fileQueues.get(key)||Promise.resolve()).then(async()=>{
    for(const file of candidates){
      let sha256='';
      try{if(globalThis.crypto?.subtle)sha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',await file.arrayBuffer())),value=>value.toString(16).padStart(2,'0')).join('');}catch(_){/* Preserve distinct files if hashing is unavailable. */}
      const files=(S.files.get(key)||[]).slice();
      if(files.some(item=>item.file===file||(sha256&&item.file.name===file.name&&item.sha256===sha256))){toast(file.name+'：相同内容的附件已添加');continue;}
      if(files.length>=5){toast('一次最多添加5个文件',true);break;}
      files.push({file,sha256});S.files.set(key,files);
      if(key===draftKey())renderAttachments();
    }
  }).catch(error=>toast('附件准备失败：'+error.message,true));
  fileQueues.set(key,pending);updateSend();
  pending.finally(()=>{if(fileQueues.get(key)===pending)fileQueues.delete(key);if(key===draftKey()){renderAttachments();updateSend();}});
}

function renderFiles(){const files=S.view?.state.generated_files||[],block=$('#generated-files');block.hidden=!files.length;block.innerHTML=features.cards(files);}
function actionButton(label,command,attrs='',primary=false){return `<button class="button ${primary?'primary':''}" data-command="${command}" ${attrs}>${label}</button>`;}
function renderActions(){const state=S.view?.state,flow=teachingClinicalFlow(state),block=$('#action-area');block.hidden=!state||busy();if(block.hidden)return;const buttons=[],auxiliary=[];let heading='';const subjective=state.mode==='teaching'&&!isObjectiveQuestion(state),support=teachingSupport.enabled(),addMore=button=>(subjective?auxiliary:buttons).push(button);if(state.mode==='teaching'){
  if(['setup','case','topic'].includes(state.stage)){block.hidden=true;return;}
  heading=state.stage==='complete'?'接下来，按自己的节奏继续':subjective?'继续补充，或获取下一条提示':'需要时，选择一种帮助';
  if(!state.pending){
    if(state.stage==='complete'){for(const [operation,label] of state.practice_ended?[['resume','继续学习']]:[['next','下一题'],['similar','同类练习'],['review','复习遗漏'],['finish','结束学习']]){const button=actionButton(label,'learning-action',`data-operation="${operation}"`,operation==='next'||operation==='resume');if(subjective&&!['next','resume'].includes(operation))auxiliary.push(button);else buttons.push(button);}}
    else if(!subjective){if(!support)buttons.push(actionButton('给我提示','turn','data-value="hint"',true));buttons.push(actionButton('重新选择','continue-answer'));if(state.attempts.length)buttons.push(actionButton('评价我的回答','turn','data-value="feedback"'),actionButton('结束并总结','turn','data-value="summary"'));}
    if(subjective){
      const access=teachingReferenceAccess(state),passed=flow?.steps[flow.current_step].status==='awaiting_next',done=state.stage==='complete';
      if(!done){
        heading=passed?'当前步骤已覆盖，可以进入下一步':support?'继续整理你的回答':access.remaining?'继续补充，或获取下一条提示':'需要时查看参考答案，再试着补充';
        if(passed)buttons.push(actionButton('进入下一步','next-step','',true));
        else if(!support&&access.remaining)buttons.push(actionButton('下一条提示','turn','data-value="hint"',true));
        if(!support&&access.allowed)buttons.push(actionButton('查看参考答案','turn','data-value="reference"',!passed));
        if(!support&&access.hints_seen)buttons.push(actionButton('把提示讲详细','turn','data-value="hint_detail"'));
        if(!passed)buttons.push(actionButton((S.view.answer_count??state.attempts.length)?'继续补充':'开始作答','continue-answer'));
        auxiliary.push(actionButton('结束并总结','turn','data-value="summary"'));
      }else if(!support&&access.allowed)buttons.push(actionButton('复盘参考答案','turn','data-value="reference"'));
      else if(!support&&access.remaining){buttons.push(actionButton('下一条提示','turn','data-value="hint"'));if(access.hints_seen)auxiliary.push(actionButton('把提示讲详细','turn','data-value="hint_detail"'));}
      if(!support)auxiliary.push(actionButton('解释这道题','turn','data-value="explain"'));
      const route=state.input_route;
      if(state.stage!=='complete'&&route?.confidence==='uncertain'&&['uncertain_intent','router_unavailable'].includes(route.reason)&&route.raw_text?.trim()){
        auxiliary.unshift(...buttons.splice(0));buttons.push(actionButton('这是我的回答','confirm-answer','',true),actionButton('我是在求助','confirm-help'));heading='请确认这条消息的用途';
      }
    }
    if(state.reviews.length&&state.recheck_for_answer!==(S.view.answer_count??state.attempts.length)&&(!flow||flow.status==='in_progress'&&flow.steps[flow.current_step].reviews.length))addMore(actionButton('重新核对','turn','data-value="recheck"'));
    if(state.stage==='complete'){addMore(actionButton('查看总结','show-summary'));addMore(actionButton('推荐练习','recommendations'));}
  }
  addMore(actionButton('学习进度','learning-progress'));addMore(actionButton('学习档案','learning-profile'));
}
if(state.sources?.length||S.view.notices?.length)addMore(actionButton('本轮参考资料','evidence'));
block.hidden=!buttons.length&&!auxiliary.length;if(block.hidden)return;
const rating=state.mode==='teaching'&&state.stage==='complete'&&(!flow||flow.status==='completed')&&state.learning_profile&&state.learning_archive?.saved&&state.exercise_learning_profile===state.learning_profile&&!state.learning_self_rating?.rating;
stableHTML(block,`${heading?`<div class="action-heading">${heading}</div>`:''}<div class="action-row">${buttons.join('')}</div>${auxiliary.length?`<details class="teaching-more"><summary>更多学习操作</summary><div class="action-row">${auxiliary.join('')}</div></details>`:''}${rating?`<div class="rating-area"><div class="action-heading">这道题的回忆感觉如何？根据你的自评安排复习。</div><div class="rating-row">${[['还需再学','Again'],['回忆困难','Hard'],['能够回忆','Good'],['回忆轻松','Easy']].map(([zh,en],index)=>`<button class="rating-button" data-rating="${index+1}">${zh}<span>${en}</span></button>`).join('')}</div></div>`:''}`);}
function renderError(){
  const state=S.view?.state,block=$('#turn-error'),error=state?.error||S.view?.connection_error,parts=[];
  if((error||state?.cancelled)&&!teachingSupport.errorPlacement())parts.push(`<strong>${state.cancelled?'本轮已停止':'这轮还没有完成'}</strong><p>${escape(error||'输入和已生成的片段已保存，可以重试或继续输入。')}</p>${state.pending?actionButton('重试本回合','turn','data-value="retry"',true):''}`);
  const archive=state?.learning_archive;
  if(archive&&archive.saved===false&&archive.error)parts.push(`<strong>学习档案保存失败</strong><p>本轮会话仍保留，档案尚未更新。${escape(archive.error)}</p>${actionButton('重试保存档案','retry-learning-save','',true)}`);
  const performance=state?.performance_status,result=state?.performance_result;
  if(performance?.is_partial&&result){
    const reason={revision_limit:'已达到修订次数上限',budget_exceeded:'已达到运行预算',time_limit:'已达到时间上限',source_changed_before_publication:'发布前来源校验失败',not_recorded:'记录未提供'}[performance.stop_reason]||performance.stop_reason;
    const issues=(result.unresolved_issues||[]).filter(row=>row&&typeof row==='object');
    parts.push(`<strong>${S.view?.active||state.pending?'最近已保存的性能结果':'本轮性能分析'}尚未完全通过自检</strong><p>结果状态：${escape(performance.status)}；停止原因：${escape(reason)}。自动自检仍有 ${issues.length} 项未解决提示，${performance.error_count||0} 项运行错误；这些提示数量不等于已确认的临床错误数量。</p>${issues.length?`<details><summary>查看未解决提示</summary>${issues.map(row=>`<p><b>${escape(row.severity||'未分类')}</b> · ${escape(row.message||row.description||row.reason||row.issue||JSON.stringify(row))}${row.step?`（${escape(row.step)}）`:''}</p>`).join('')}</details>`:''}`);
  }
  block.hidden=!parts.length;if(parts.length)block.innerHTML=parts.join('<hr>');
}

function renderLive(){
  const job=currentJob(),active=S.view?.active,block=$('#turn-status'),output=$('#stream-message'),placement=teachingSupport.sidePlacement(job,active);
  teachingSupport.render();
  const centerText=job?.done?'':placement?job?.centerText||'':job?.text||job?.centerText||'';
  block.hidden=(!job&&!active)||Boolean(placement)||Boolean(job?.done);output.hidden=!centerText;
  patchHTML(output,centerText?`<article class="message assistant"><div class="message-meta"><img src="/ui/assets/brand-care-t.svg" alt="" width="26" height="26"><strong>MDRPI</strong><span class="message-kind">正在输出</span></div><div class="message-body markdown">${markdown(centerText)}</div></article>`:'');
  if(block.hidden){renderActions();updateSend();return;}teachingWorkspace.render();
  const label=job?.status||active?.phase||'请求正在后台继续，等待保存结果';
  const teachingWait=S.mode==='teaching'?'<p class="teaching-wait">正在核对已提交的内容，通常需要几十秒。可以先写下一段草稿；当前回合完成后再发送。</p>':'';
  const started=job?.started||(active?.elapsed_seconds!==undefined?Date.now()-active.elapsed_seconds*1000:read('turn.'+sid()));const elapsed=started?Math.max(0,Math.floor((Date.now()-started)/1000)):null;
  const oldOpen=block.querySelector('details')?.open;
  block.innerHTML=`<details ${oldOpen?'open':''}><summary><span class="pulse"></span><span>${escape(job?.stopping?'正在停止，已生成内容会保留':label)}</span>${elapsed!==null?`<span class="elapsed">${elapsed<60?elapsed+'秒':Math.floor(elapsed/60)+'分'+elapsed%60+'秒'}</span>`:''}</summary>${job?.phases?.length?`<div class="phase-history">${job.phases.map(value=>`<span>${escape(value)}</span>`).join('')}</div>`:''}</details>${teachingWait}`;
  renderActions();updateSend();if(S.follow)requestAnimationFrame(()=>{if(S.follow)scrollLatest();});
}

async function startTurn(text='',action='send',options={},fromControls=false) {
  if(busy())return;
  if(fileQueues.has(draftKey())){toast('附件正在准备，请稍候再发送');return;}
  if(action==='send'&&inputLength(text)>inputLimit()){toast(`输入为 ${inputLength(text)} 字，超过本模式 ${inputLimit()} 字的上限；全文已保留，请修改后发送。`,true);updateSend();return;}
  if(action==='send'&&!text.trim()&&!currentFiles().length)return;
  const original=text,artifact_id=action==='send'?(S.pendingRevision?.artifact_id||''):'';S.preparing=true;updateSend();let target,job;
  try {
    const state=await ensureSession();target=state.session_id;
    job={started:Date.now(),text:'',status:'正在处理本轮请求',phases:[],user:action==='send'?text:'',mode:state.mode};
    teachingSupport.beginTurn(job,action,text);
    S.jobs.set(target,job);S.preparing=false;write('turn.'+target,job.started);write('submission.'+target,{text,action,artifact_id,from_controls:fromControls,at:job.started});
    if(sid()===target){if(action==='send'&&!fromControls){$('#composer-input').value='';saveDraft();}render();}
    const files=S.files.get(target)||[];
    if(files.length){
      job.uploading=true;
      for(const item of files.slice()){
        item.error='';item.uploading=true;job.status='正在读取 '+item.file.name;if(sid()===target){renderAttachments();renderLive();}
        try{const body=new FormData();body.append('file',item.file);await api(`/sessions/${target}/files`,body,'POST',240000);files.splice(files.indexOf(item),1);}catch(error){item.error=error.message;throw new Error(item.file.name+'：'+error.message);}finally{item.uploading=false;}
      }
      job.uploading=false;
      if(sid()===target)await refresh();
      if(!text.trim()){
        if(state.mode==='teaching'&&!['setup','case','topic'].includes(state.stage)){toast('文件已加入参考资料，请继续作答');return;}
        text=state.mode==='teaching'?'请根据刚上传的文件准备一个练习案例，先让我作答。':'请概括刚上传文件的主要内容。';
        job.user=text;if(sid()===target)renderMessages();
      }
    }
    let renderScheduled=false,received=false;
    const loadStreamingView=()=>{
      const request=(job.viewRequest||0)+1;job.viewRequest=request;
      api(`/sessions/${target}/view`).then(view=>{if(sid()===target&&!job.done&&S.jobs.get(target)===job&&job.viewRequest===request){S.view=view;if(job.centerText&&view.messages.some(message=>message.role==='assistant'&&message.content===job.centerText))job.centerText='';remember();renderWelcome();renderStage();renderMessages();}}).catch(()=>{});
    };
    await stream(target,{text,action,artifact_id,...options},(kind,value)=>{
      if(!received){received=true;loadHistory().catch(()=>{});loadStreamingView();}
      if(kind==='token'){job.text+=value;job.status='正在输出回答';}
      else if(kind==='answer_text'){job.text=value;job.status='正在输出回答';}
      else if(kind==='status'){job.status=String(value).replace(/[.…]+$/,'');if(job.phases.at(-1)!==job.status)job.phases.push(job.status);}
      else if(kind==='teaching_support'){
        teachingSupport.streamEvent(job,value);
        if(sid()===target)renderMessages();
        if(value?.phase==='responding')loadStreamingView();
      }
      else if(kind==='done'){job.done=true;job.view=value;write('submission.'+target,null);if(sid()===target){S.view=value;if(isObjectiveQuestion(value.state))S.follow=true;remember();features.revisionCompleted(value).catch(error=>toast(error.message,true));}}
      if(sid()===target&&!renderScheduled){renderScheduled=true;setTimeout(()=>{renderScheduled=false;if(sid()===target){if(job.done)render();else renderLive();}},70);}
    });
  } catch(error) {
    if(job){job.uploading=false;job.error=error.message;}
    if(target){
      try {const view=await api(`/sessions/${target}/view`);if(sid()===target){S.view=view;if(!view.active&&!view.state.pending){S.view.connection_error=error.message;if(!fromControls&&!$('#composer-input').value){$('#composer-input').value=original;saveDraft();}}else{toast('连接中断；请求仍会保存，可等待或停止',true);}}}
      catch(_){if(sid()===target&&S.view)S.view.connection_error=error.message;toast(error.message,true);}
    }else toast(error.message,true);
  } finally {
    S.preparing=false;
    if(target){teachingSupport.finishTurn(job);S.jobs.delete(target);if(sid()===target){try{const fresh=await api(`/sessions/${target}/view`);if(S.view?.connection_error)fresh.connection_error=S.view.connection_error;S.view=fresh;}catch(_){}render();}}
    else updateSend();
    loadHistory().catch(()=>{});
  }
}

async function stopTurn(){if(!sid()||currentJob()?.uploading)return;const target=sid(),job=currentJob();if(job){job.stopping=true;renderLive();}try{const result=await api(`/sessions/${target}/cancel`,{});if(result.status==='unsupported')toast('研究测试仍在后台运行，可从历史中查看结果');else if(result.status==='idle'){await refresh();if(job)job.stopping=false;}}catch(error){if(job)job.stopping=false;throw error;}}
async function learningAction(operation,exercise_id=''){if(busy())return;S.preparing=true;updateSend();try{const state=await ensureSession(),target=state.session_id;await api(`/sessions/${target}/learning`,{action:operation,exercise_id});if(sid()===target){S.follow=true;closeDialog();await refresh();scrollLatest();}await loadHistory();}finally{S.preparing=false;render();}}
function showPopover(anchor,html){const node=$('#popover');node.innerHTML=html;node.hidden=false;const rect=anchor.getBoundingClientRect(),height=node.offsetHeight;node.style.left=Math.max(10,Math.min(rect.left,innerWidth-node.offsetWidth-10))+'px';node.style.top=(rect.bottom+height+15<innerHeight?rect.bottom+7:Math.max(10,rect.top-height-7))+'px';}
function menuButton(label,command,shape,attrs=''){return `<button class="menu-button" role="menuitem" data-command="${command}" ${attrs}>${icon(shape)}<span>${label}</span></button>`;}
function conversationMenu(anchor,target=sid()){
  if(!target){showPopover(anchor,menuButton('偏好与设置','settings','settings')+menuButton('使用说明','help','help'));return;}
  const row=S.history.find(row=>row.session_id===target)||{session_id:target,title:S.view.title,pinned:S.view.metadata?.pinned,archived:S.view.metadata?.archived};
  showPopover(anchor,`<div role="menu">${menuButton('重命名','rename','edit',`data-sid="${target}" data-title="${escape(row.title)}"`)}${menuButton(row.pinned?'取消置顶':'置顶对话','pin','pin',`data-sid="${target}" data-pinned="${row.pinned?'false':'true'}"`)}${menuButton('复制会话链接','copy-link','link',`data-sid="${target}"`)}<a class="menu-button" href="/api/sessions/${target}/export" download role="menuitem">${icon('download')}<span>导出完整记录</span></a>${target===sid()?menuButton('导出科研数据','research-export','download'):''}<div class="menu-divider"></div>${menuButton(row.archived?'取消归档':'归档对话','archive','archive',`data-sid="${target}" data-archived="${row.archived?'false':'true'}"`)}${S.mode==='chat'?menuButton('偏好记忆','preferences','memory'):''}</div>`);
}
async function copy(text){if(navigator.clipboard?.writeText){try{await navigator.clipboard.writeText(text);toast('已复制');return;}catch(_){}}const field=document.createElement('textarea');field.value=text;document.body.append(field);field.select();const succeeded=document.execCommand('copy');field.remove();toast(succeeded?'已复制':'复制未成功，请选择文字复制',!succeeded);}

async function command(name,target){
  if(teachingWorkspace.command(name,target))return;
  if(await features.command(name,target))return;
  if(name==='close-dialog'){closeDialog();return;}
  if(name==='new'){await newConversation();return;}
  if(name==='sidebar'){toggleSidebar();return;}
  if(name==='latest'){scrollLatest();return;}
  if(name==='theme'){setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');return;}
  if(name==='check-service'){await health();return;}
  if(name==='open-session'){
    const match=S.history.find(row=>row.session_id===target.dataset.sid)?.search_match,query=S.search;
    await navigate(target.dataset.sid);
    if(sid()===target.dataset.sid&&match?.message_at){
      S.follow=false;$('#jump-latest').hidden=false;
      if(!locateSearchResult(S.displayedMessages||[],match,query))toast('已打开会话，匹配内容可在该轮教学记录中查看');
    }
    return;
  }
  if(name==='history-more'){await loadHistory(true);return;}
  if(name==='archive-list'){S.archived=!S.archived;await loadHistory();return;}
  if(name==='conversation-menu'){conversationMenu(target);return;}
  if(name==='row-menu'){conversationMenu(target,target.dataset.sid);return;}
  if(name==='rename'){closePopover();panels.rename(target.dataset.sid,target.dataset.title);return;}
  if(name==='pin'||name==='archive'){const id=target.dataset.sid,field=name==='pin'?'pinned':'archived',value=target.dataset[field]==='true';closePopover();await api(`/sessions/${id}/metadata`,{[field]:value},'PATCH');await loadHistory();if(sid()===id)await refresh();toast(name==='pin'?(value?'对话已置顶':'已取消置顶'):(value?'已归档，可从侧栏找回':'已恢复到最近对话'));return;}
  if(name==='copy-link'){const url=new URL('/ui/',location.origin);url.searchParams.set('session',target.dataset.sid);closePopover();await copy(url.href);return;}
  if(name==='attach'){if(S.mode==='performance'){toast('此模式使用病例文本输入');return;}$('#file-picker').click();return;}
  if(name==='focus'){$('#composer-input').focus();return;}
  if(name==='prompt'){
    const field=$('#composer-input'),prompt=target.dataset.value||'',current=field.value;
    if(!prompt)return;
    if(!current.includes(prompt)){
      const next=current?current+'\n\n'+prompt:prompt;
      field.value=next;
      if(current)toast('已追加专业引导，原草稿保留');
    }
    saveDraft();resizeInput();field.focus();field.setSelectionRange(field.value.length,field.value.length);return;
  }
  if(name==='turn'){if(['choose_case','choose_topic'].includes(target.dataset.value)){await panels.caseSetup(target.dataset.value);return;}closeDialog();closePopover();await startTurn('',target.dataset.value);return;}
  if(name==='support-turn'){if(!teachingSupport.enabled())return;await startTurn('',target.dataset.value);return;}
  if(name==='support-stop'){await stopTurn();return;}
  if(name==='confirm-answer'||name==='confirm-help'){
    const state=S.view?.state,route=state?.input_route;
    if(!state||isObjectiveQuestion(state)||state.stage==='complete'||route?.confidence!=='uncertain'||!['uncertain_intent','router_unavailable'].includes(route.reason)||!route.raw_text?.trim())return;
    await startTurn(route.raw_text,name==='confirm-answer'?'submit_answer':'explain',{},true);return;
  }
  if(name==='next-step'){const flow=teachingClinicalFlow(S.view?.state);if(flow)await startTurn('','next_step',{expected_step:flow.current_step});return;}
  if(name==='continue-answer'){if(isObjectiveQuestion(S.view?.state)){const panel=$('#answer-tools');panel.scrollIntoView({block:'start'});(panel.querySelector('input:not(:disabled),button:not(:disabled)')||panel).focus();}else{const field=$('#composer-input');field.focus();field.scrollIntoView({block:'center'});}return;}
  if(name==='evidence-message'){
    const index=Number(target.dataset.index);if(!Number.isInteger(index)||index<0)return;
    const area=target.closest('#dialog-body')||$('#messages'),message=area.querySelector(`.message[data-message-index="${index}"]`);if(!message)return;
    const mark=[...message.querySelectorAll('.answer-evidence')].find(node=>{try{return JSON.parse(node.dataset.criterionIds||'[]').includes(target.dataset.criterion);}catch(_){return false;}});
    if(area.id==='messages')S.follow=false;message.scrollIntoView({block:'center',behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});message.tabIndex=-1;(mark||message).focus({preventScroll:true});return;
  }
  if(name==='retry-learning-save'){if(busy())return;target.disabled=true;try{await api(`/sessions/${sid()}/learning-profile/save`,{});await refresh();if(S.view.state.learning_archive?.saved)toast('学习档案已保存');else toast('档案尚未保存，请查看提示',true);}finally{if(target.isConnected)target.disabled=false;}return;}
  if(name==='learning-action'){await learningAction(target.dataset.operation,target.dataset.exercise||'');return;}
  if(name==='remove-local-file'){if(busy())return;const files=currentFiles().slice();files.splice(Number(target.dataset.index),1);S.files.set(draftKey(),files);renderAttachments();updateSend();return;}
  if(name==='remove-file'){if(busy()){toast('当前正在读取资料，请在本轮结束后移除');return;}await api(`/sessions/${sid()}/files/${target.dataset.file}`,undefined,'DELETE');await refresh();return;}
  if(name==='file-details'){const file=S.view.state.attachments.find(file=>file.file_id===target.dataset.file);if(!file)return;openDialog(file.name,`<p class="panel-note">已加入本会话参考资料。移除后不再用于后续回答，原始文件仍保留。</p>${file.warnings?.length?`<div class="record-row"><strong>读取时需要留意</strong>${file.warnings.map(note=>`<p>${escape(note)}</p>`).join('')}</div>`:'<p class="panel-note">读取完成，没有额外的解析提示。</p>'}<a class="button" href="/api/sessions/${sid()}/files/${file.file_id}" download>${icon('download')}下载原文件</a>`);return;}
  if(name==='copy-message'){const messages=target.closest('#dialog-body')?S.panelMessages:S.displayedMessages,message=messages?.[Number(target.dataset.index)];await copy(message?.role==='user'?message.content||'':message?.display_content??message?.content??'');return;}
  if(name==='reuse-message'){const messages=target.closest('#dialog-body')?S.panelMessages:S.displayedMessages;$('#composer-input').value=messages?.[Number(target.dataset.index)]?.content||'';saveDraft();resizeInput();closeDialog();$('#composer-input').focus();return;}
  if(name==='copy-code'){await copy(target.closest('.code-block').querySelector('code').textContent);return;}
  await panels.command(name,target);
}

let searchTimer,dragDepth=0,pollBusy=false;
document.addEventListener('click',async event=>{
  const target=event.target.closest('[data-command],[data-mode],[data-theme-value],[data-category],[data-case],[data-rating]');
  if(!event.target.closest('#popover')&&!target?.matches('[data-command="conversation-menu"],[data-command="row-menu"]'))closePopover();
  if(!target||target.disabled)return;
  try{
    if(target.dataset.mode){await selectMode(target.dataset.mode);return;}
    if(target.dataset.themeValue){setTheme(target.dataset.themeValue);return;}
    if(target.dataset.category){panels.chooseCategory(target.dataset.category);return;}
    if(target.dataset.case){if(S.selectingCase||busy())return;S.selectingCase=true;try{closeDialog();if(S.mode==='teaching'&&S.view?.state.case&&!['setup','case','topic'].includes(S.view.state.stage))await newConversation();await startTurn(target.dataset.case,'library_case');}finally{S.selectingCase=false;}return;}
    if(target.dataset.rating){target.disabled=true;try{const result=await api(`/sessions/${sid()}/learning-profile/rating`,{rating:Number(target.dataset.rating)});await refresh();toast('自评已保存，下次复习：'+new Date(result.next_due).toLocaleString(interfaceLocale()));}finally{target.disabled=false;}return;}
    await command(target.dataset.command,target);
  }catch(error){toast(error.message,true);}
});
$('#composer').addEventListener('submit',event=>{event.preventDefault();if(busy()){stopTurn().catch(error=>toast(error.message,true));return;}const text=$('#composer-input').value.trim();if(text.startsWith('/resume ')){const target=text.slice(8).trim();$('#composer-input').value='';saveDraft();navigate(target).catch(error=>toast(error.message,true));}else if(text==='/new'){newConversation().catch(error=>toast(error.message,true));}else if(text==='/export'&&sid()){const link=document.createElement('a');link.href=`/api/sessions/${sid()}/export`;link.download='';link.click();$('#composer-input').value='';saveDraft();resizeInput();}else startTurn(text);});
$('#composer-input').addEventListener('input',()=>{teachingWorkspace.syncDraft();saveDraft();resizeInput();});
$('#composer-input').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing&&event.keyCode!==229){event.preventDefault();if(!busy())$('#composer').requestSubmit();}});
$('#composer-input').addEventListener('paste',event=>{const files=[...event.clipboardData.items].filter(item=>item.kind==='file').map(item=>item.getAsFile()).filter(Boolean);if(files.length){event.preventDefault();addFiles(files);}});
$('#file-picker').addEventListener('change',event=>{addFiles(event.target.files);event.target.value='';});
$('#history-search').addEventListener('input',event=>{S.search=event.target.value;clearTimeout(searchTimer);searchTimer=setTimeout(()=>loadHistory().catch(error=>toast(error.message,true)),250);});
$('#sidebar-scrim').addEventListener('click',closeMobileSidebar);
$('#scroll-area').addEventListener('scroll',()=>{const node=$('#scroll-area'),latest=$('#messages').lastElementChild;
  const offset=latest?latest.getBoundingClientRect().top-node.getBoundingClientRect().top:Infinity;
  const panel=$('#answer-tools'),panelOffset=panel.getBoundingClientRect().top-node.getBoundingClientRect().top;
  const atChoiceFeedback=isObjectiveQuestion(S.view?.state)&&!panel.hidden&&panelOffset>=-1&&panelOffset<45;
  S.follow=atChoiceFeedback||node.scrollHeight-node.scrollTop-node.clientHeight<90;$('#jump-latest').hidden=S.follow;
},{passive:true});
document.addEventListener('dragenter',event=>{if(event.dataTransfer.types.includes('Files')){event.preventDefault();dragDepth++;$('#drop-overlay').hidden=false;}});
document.addEventListener('dragover',event=>{if(event.dataTransfer.types.includes('Files'))event.preventDefault();});
document.addEventListener('dragleave',event=>{if(event.dataTransfer.types.includes('Files')){dragDepth--;if(dragDepth<=0)$('#drop-overlay').hidden=true;}});
document.addEventListener('drop',event=>{if(event.dataTransfer.types.includes('Files')){event.preventDefault();dragDepth=0;$('#drop-overlay').hidden=true;addFiles(event.dataTransfer.files);}});
document.addEventListener('keydown',event=>{
  if(event.key==='Enter'&&!event.defaultPrevented&&!event.repeat&&!event.isComposing&&!event.shiftKey&&!event.ctrlKey&&!event.metaKey&&!event.altKey&&isObjectiveQuestion(S.view?.state)&&!$('#dialog').open&&!['INPUT','TEXTAREA','SELECT','BUTTON','A','SUMMARY'].includes(event.target.tagName)&&!event.target.isContentEditable){event.preventDefault();teachingWorkspace.command('objective-submit',event.target);return;}
  if(!$('#dialog').open&&(event.metaKey||event.ctrlKey)&&event.key.toLowerCase()==='k'){event.preventDefault();openSidebar();$('#history-search').focus();}
  if(!$('#dialog').open&&event.key==='/'&&!event.metaKey&&!event.ctrlKey&&!['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)&&!event.target.isContentEditable){event.preventDefault();openSidebar();$('#history-search').focus();}
  if((event.metaKey||event.ctrlKey)&&event.shiftKey&&event.key.toLowerCase()==='o'){event.preventDefault();newConversation().catch(error=>toast(error.message,true));}
  if(event.key==='Escape'&&!$('#dialog').open){closePopover();closeMobileSidebar();}
});
// Native light-dismiss where supported. The fallback requires both ends of
// the pointer gesture outside, so selecting text cannot accidentally close it.
if(!('closedBy' in HTMLDialogElement.prototype)){
  const dialog=$('#dialog');let pressedOutside=false;
  const outside=event=>{const rect=dialog.getBoundingClientRect();return event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom;};
  dialog.addEventListener('pointerdown',event=>{pressedOutside=event.isPrimary&&event.button===0&&outside(event);});
  dialog.addEventListener('pointerup',event=>{if(event.isPrimary&&event.button===0&&pressedOutside&&outside(event))closeDialog();pressedOutside=false;});
  dialog.addEventListener('pointercancel',()=>{pressedOutside=false;});
  dialog.addEventListener('close',()=>{pressedOutside=false;});
}
window.addEventListener('popstate',()=>{const value=new URL(location.href).searchParams.get('session');if(value)navigate(value,false).catch(error=>toast(error.message,true));else newConversation().catch(()=>{});});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{if(appearance()==='system')setTheme('system');});

let healthBusy=false,healthTimer,lastReadinessAt=0;
function showConnection(label,tone,notice=''){
  $('#connection-dot').className=tone;
  if($('#connection-label').textContent!==label)$('#connection-label').textContent=label;
  if($('#service-notice').hidden!==!notice)$('#service-notice').hidden=!notice;
  if($('#service-notice-text').textContent!==notice)$('#service-notice-text').textContent=notice;
}
async function health(){
  if(healthBusy)return;healthBusy=true;clearTimeout(healthTimer);let next=20000;
  try{
    const result=await api('/health',undefined,'GET',4000);
    let readiness;
    try{readiness=await api('/readiness',undefined,'GET',4000);}catch(_){readiness={};}
    // A refresh in progress is not a service failure. Preserve the last
    // confirmed notice (including real outages), and check again quietly.
    const confirmedFailure=readiness.model==='unavailable'||(readiness.model==='ready'&&readiness.knowledge==='unavailable');
    if(!confirmedFailure&&(readiness.checking||readiness.model==='checking'||readiness.knowledge==='checking')){
      next=2000;
      if(!lastReadinessAt||Date.now()-lastReadinessAt>45000){
        $('#connection-dot').className='checking';
        if($('#connection-label').textContent!=='正在检查服务')$('#connection-label').textContent='正在检查服务';
      }
      return result;
    }
    lastReadinessAt=Date.now();
    if(readiness.model==='ready'&&readiness.knowledge==='ready')showConnection('服务就绪，可以提问','online');
    else if(readiness.model==='unavailable')showConnection('模型尚未就绪','offline','模型尚未就绪，可以先编辑问题，恢复后再发送。');
    else if(readiness.model==='ready'&&readiness.knowledge==='unavailable')showConnection('知识库暂不可用','degraded','知识库暂不可用，依赖资料的回答可能无法完成。');
    else showConnection('模型状态待确认','degraded','网页已连接，模型或知识库的状态暂时无法确认。');
    return result;
  }catch(_){lastReadinessAt=0;showConnection('网页连接中断','offline','连接已中断，正在重连。当前草稿仍会保留。');next=5000;return null;}
  finally{healthBusy=false;healthTimer=setTimeout(health,next);}
}
async function poll(){if(pollBusy||!sid()||currentJob())return;pollBusy=true;try{if(S.view?.active){const target=sid(),view=await api(`/sessions/${target}/view`);if(sid()===target){await applyView(view);if(!view.active){await loadHistory();write('submission.'+target,null);}}}}catch(_){}finally{pollBusy=false;}}
async function boot(){hydrateIcons();if(read('collapsed',false)&&!matchMedia('(max-width:760px)').matches)$('#app').classList.add('sidebar-collapsed');const query=new URL(location.href).searchParams,remembered=read('last',{});S.mode=query.has('mode')?(publicMode(query.get('mode'))?query.get('mode'):'chat'):(publicMode(remembered.mode)?remembered.mode:'chat');render();await health();const target=query.get('session')||(publicMode(remembered.mode)&&remembered.mode===S.mode?remembered.sid:null);if(target){try{await navigate(target,false);}catch(error){toast('未能恢复该会话：'+error.message,true);S.view=null;updateURL();render();}}else updateURL();
  if(sid()){const pending=read('submission.'+sid());if(pending&&!S.view.active&&!S.view.state.pending){const accepted=S.view.state.messages.some(message=>message.role==='user'&&message.content===pending.text&&new Date(message.at).getTime()>=pending.at-1000);if(!accepted&&!pending.from_controls&&!$('#composer-input').value&&pending.text){$('#composer-input').value=pending.text;saveDraft();resizeInput();toast('上次待发送内容已恢复到输入框');}write('submission.'+sid(),null);}}
  await loadHistory();syncSidebarState();setInterval(()=>{if(currentJob()||S.view?.active)renderLive();},1000);setInterval(poll,2000);$('#composer-input').focus();}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>boot().catch(error=>toast(error.message,true)),{once:true});else boot().catch(error=>toast(error.message,true));

startLocalization();
window.addEventListener('mdrpi:languagechange',()=>{
  const field=document.activeElement,selection=field instanceof HTMLTextAreaElement?[field.selectionStart,field.selectionEnd]:null;
  const draft=$('#composer-input').value;
  saveDraft();render();$('#composer-input').value=draft;resizeInput();localize(document.body);
  if(selection&&field.isConnected)field.setSelectionRange(...selection);
});
