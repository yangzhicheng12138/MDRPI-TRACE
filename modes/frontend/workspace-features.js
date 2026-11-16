import {icon} from './icons.js';
import {escape,markdown} from './markdown.js';

export function createWorkspaceFeatures({S,api,sid,busy,read,write,toast,navigate,startTurn,learningAction,renderMessages,saveDraft,resizeInput}) {
  const $=selector=>document.querySelector(selector);
  let editing=null,previewing=null,previewRequest=0,seenSession=null,savingEdit=false;
  const currentExercise=()=>S.view?.state.exercise_id||(sid()?sid()+':first':'');
  const saveEdit=()=>{if(editing)write('edit.'+editing.sid,editing);};
  function cancelEdit(){if(editing)write('edit.'+editing.sid,null);editing=null;renderMessages();}
  function sync(){
    if(seenSession!==sid()){
      closePreview(false);seenSession=sid();editing=sid()?read('edit.'+sid()):null;
      const saved=read('preview.'+sid());if(saved&&S.view?.state.generated_files?.some(f=>f.artifact_id===saved))queueMicrotask(()=>openPreview(saved));
    }
    if(editing&&!S.view?.messages.some(m=>m.at===editing.at&&m.role==='user'))editing=null;
    if(editing&&S.mode==='teaching'&&(!S.view.state.messages.some(m=>m.at===editing.at)||
        editing.exerciseId&&editing.exerciseId!==currentExercise())){write('edit.'+editing.sid,null);editing=null;}
    const selection=read('revision.'+sid());
    S.pendingRevision=S.view?.state.generated_files?.find(f=>f.artifact_id===selection)||null;
    renderRevision();renderBranches();
  }
  function renderBranches(){
    const block=$('#branch-context'),family=S.view?.branches?.items||[];block.hidden=!family.length;
    if(block.hidden)return;
    block.innerHTML=`<span>${icon('branch')}对话版本</span><select id="branch-select" aria-label="切换对话版本">${family.map((item,i)=>`<option value="${item.session_id}" ${item.session_id===sid()?'selected':''}>${escape(i?`修订 ${i} · ${item.title}`:item.title)}</option>`).join('')}</select>${S.view.state.branch_origin?`<button data-command="open-session" data-sid="${S.view.state.branch_origin.source_session_id}">查看修改前的对话</button>`:''}`;
  }
  function editor(message){
    if(!editing||editing.at!==message.at||editing.sid!==sid())return '';
    const teaching=S.mode==='teaching';
    return `<div class="message-editor"><label for="message-edit-text">${teaching?'补充或更正这次作答':'修改这条消息'}</label><textarea id="message-edit-text" maxlength="12000" rows="5">${escape(editing.text)}</textarea><p>${teaching?(S.view.state.stage==='complete'?'将开始同题的新一轮练习，保留原作答和评价。':'提交为新的学生作答，保留此前的回答和评价。'):'保存后从这里生成新回答，原对话和后续消息保留在原版本中。'}</p><div><button class="button" data-command="cancel-message-edit">取消</button><button class="button primary" data-command="save-message-edit" ${busy()?'disabled':''}>${teaching?'提交更正':'保存并生成'}</button></div></div>`;
  }
  function beginEdit(index){
    if(busy())return;const message=S.displayedMessages?.[index];if(!message)return;
    if(S.mode==='teaching'&&!S.view.state.messages.some(m=>m.at===message.at))return;
    editing={sid:sid(),at:message.at,text:message.content,exerciseId:S.mode==='teaching'?currentExercise():null};saveEdit();renderMessages();
    const field=$('#message-edit-text');field?.focus();field?.scrollIntoView({block:'center',behavior:'smooth'});
  }
  async function saveMessage(){
    if(!editing||busy()||savingEdit)return;const text=$('#message-edit-text').value.trim();if(!text){toast('请输入修改后的内容',true);return;}
    const edit={...editing,text};saveEdit();
    savingEdit=true;
    try {
    if(S.mode==='chat'){
      const created=await api(`/sessions/${sid()}/branches`,{message_at:edit.at,text});
      cancelEdit();await navigate(created.session_id);await startTurn(text);
    }else{
      cancelEdit();if(S.view.state.stage==='complete')await learningAction('retry_exercise',currentExercise());
      await startTurn(text);
    }
    } finally {savingEdit=false;}
  }
  function renderRevision(){
    const block=$('#revision-chip');block.hidden=!S.pendingRevision;
    block.innerHTML=S.pendingRevision?`${icon('file')}<span>继续修改 <b>${escape(S.pendingRevision.name)}</b></span><button class="icon-button" data-command="clear-revision" aria-label="取消文件修改选择" ${busy()?'disabled':''}>${icon('x')}</button>`:'';
  }
  function setRevision(artifact){S.pendingRevision=artifact;write('revision.'+sid(),artifact?.artifact_id||null);renderRevision();}
  function cards(files){
    return files.map(file=>`<div class="artifact-card"><button class="artifact-open" data-command="preview-artifact" data-artifact="${file.artifact_id}" aria-label="预览${escape(file.name)}">${icon(file.mime_type.startsWith('image/')?'chart':'file')}<span><strong>${escape(file.name)}</strong><small>${file.validation?.valid===false?'未通过文件检查 · ':''}${escape(file.mime_type.startsWith('image/')?'图表 / 图片':file.name.endsWith('.docx')?'Word 文档':file.name.endsWith('.xlsx')?'Excel 表格':file.name.endsWith('.csv')?'数据表格':'生成文件')} · ${file.version>1?'版本 '+file.version+' · ':''}${file.size>=1048576?(file.size/1048576).toFixed(1)+' MB':Math.max(1,Math.round(file.size/1024))+' KB'}</small></span></button><a class="download-card artifact-download" href="${file.url}" download aria-label="下载${escape(file.name)}">${icon('download')}</a></div>`).join('');
  }
  function closePreview(clearSaved=true){
    previewRequest++;previewing=null;const panel=$('#artifact-panel');if(!panel)return;
    panel.hidden=true;$('#app').classList.remove('artifact-open');panel.removeAttribute('aria-modal');
    if(clearSaved)write('preview.'+seenSession,null);
  }
  function tableMarkup(sheets){
    return sheets.map(sheet=>`<section class="artifact-sheet"><h3>${escape(sheet.name)}</h3><div class="table-wrap"><table>${sheet.rows.map((row,i)=>`<tr>${row.map(cell=>`<${i?'td':'th'}>${escape(cell)}</${i?'td':'th'}>`).join('')}</tr>`).join('')}</table></div></section>`).join('');
  }
  async function openPreview(artifactId){
    if(!sid())return;const target=sid(),request=++previewRequest,panel=$('#artifact-panel');
    previewing=null;$('#artifact-preview-title').textContent=S.view.state.generated_files.find(f=>f.artifact_id===artifactId)?.name||'文件预览';
    $('#artifact-preview-version').innerHTML='';$('#artifact-preview-download').hidden=true;$('#artifact-preview-note').textContent='';
    panel.hidden=false;$('#app').classList.add('artifact-open');panel.setAttribute('role',matchMedia('(max-width:1100px)').matches?'dialog':'complementary');
    if(panel.getAttribute('role')==='dialog')panel.setAttribute('aria-modal','true');else panel.removeAttribute('aria-modal');
    $('#artifact-preview-body').innerHTML='<div class="preview-loading">正在读取文件…</div>';
    try{
      const result=await api(`/sessions/${target}/artifacts/${artifactId}/preview`);
      if(request!==previewRequest||target!==sid())return;
      previewing=result;write('preview.'+target,artifactId);
      const file=result.artifact,files=S.view.state.generated_files||[],root=file.root_artifact_id||file.artifact_id;
      const versions=files.filter(f=>(f.root_artifact_id||f.artifact_id)===root);
      $('#artifact-preview-title').textContent=file.name;
      $('#artifact-preview-version').innerHTML=versions.length>1?`<select id="artifact-version-select" aria-label="选择文件版本">${versions.map(f=>`<option value="${f.artifact_id}" ${f.artifact_id===artifactId?'selected':''}>版本 ${f.version||1} · ${escape(f.name)}</option>`).join('')}</select>`:`<span>版本 ${file.version||1}</span>`;
      $('#artifact-preview-download').href=file.url;$('#artifact-preview-download').setAttribute('download',file.name);
      $('#artifact-preview-download').hidden=false;
      $('#artifact-preview-note').textContent=(file.validation?.valid===false?'文件未通过检查：'+(file.validation.error||'内容或格式待确认')+'。':'')+(result.truncated?'当前显示部分内容。':'')+result.note;
      const body=$('#artifact-preview-body');
      body.innerHTML=result.kind==='image'?`<img class="artifact-image" src="${file.url}" alt="${escape(file.name)}">`:result.kind==='table'?tableMarkup(result.sheets):result.kind==='markdown'?`<div class="markdown">${markdown(result.text)}</div>`:result.kind==='text'?`<pre class="artifact-text">${escape(result.text)}</pre>`:'<div class="preview-loading">此格式请下载原文件查看。</div>';
      // A document preview does not load remote tracking images.
      body.querySelectorAll('img').forEach(img=>{if(!img.getAttribute('src').startsWith(`/api/sessions/${target}/artifacts/`))img.remove();});
      $('#artifact-revision-input').value='';$('#artifact-revision-error').textContent='';
      if(panel.getAttribute('role')==='dialog')$('#artifact-close').focus();
    }catch(error){if(request===previewRequest)$('#artifact-preview-body').innerHTML=`<p class="inline-error">${escape(error.message)}</p>`;}
  }
  async function revise(){
    if(!previewing||busy())return;const instruction=$('#artifact-revision-input').value.trim();if(!instruction){$('#artifact-revision-error').textContent='写下希望修改的地方。';$('#artifact-revision-input').focus();return;}
    const file=previewing.artifact;setRevision(file);if(matchMedia('(max-width:1100px)').matches)closePreview();
    await startTurn(`请修改《${file.name}》：${instruction}。请保留原文件，生成可下载的新版本。`);
  }
  async function revisionCompleted(view){
    const parent=S.pendingRevision;if(!parent)return;setRevision(null);
    const created=(view.state.artifacts||[]).filter(f=>f.revision_of===parent.artifact_id);
    if(created.length&&!$('#artifact-panel').hidden)await openPreview(created.at(-1).artifact_id);
  }
  async function command(name,target){
    if(name==='edit-message'){beginEdit(Number(target.dataset.index));return true;}
    if(name==='cancel-message-edit'){cancelEdit();return true;}
    if(name==='save-message-edit'){await saveMessage();return true;}
    if(name==='preview-artifact'){await openPreview(target.dataset.artifact);return true;}
    if(name==='close-artifact'){closePreview();return true;}
    if(name==='clear-revision'){setRevision(null);return true;}
    return false;
  }
  document.addEventListener('input',event=>{if(event.target.id==='message-edit-text'&&editing){editing.text=event.target.value;saveEdit();}});
  document.addEventListener('change',event=>{if(event.target.id==='branch-select')navigate(event.target.value).catch(e=>toast(e.message,true));if(event.target.id==='artifact-version-select')openPreview(event.target.value);});
  $('#artifact-revision-form').addEventListener('submit',event=>{event.preventDefault();revise().catch(e=>toast(e.message,true));});
  document.addEventListener('keydown',event=>{
    if(event.key==='Escape'&&!$('#artifact-panel').hidden){event.preventDefault();closePreview();}
    if(event.key==='Tab'&&!$('#artifact-panel').hidden&&$('#artifact-panel').getAttribute('aria-modal')==='true'){
      const nodes=[...$('#artifact-panel').querySelectorAll('button,a,textarea,select')].filter(n=>!n.disabled&&!n.hidden),first=nodes[0],last=nodes.at(-1);
      if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
    }
  });
  matchMedia('(max-width:1100px)').addEventListener('change',event=>{const panel=$('#artifact-panel');if(panel.hidden)return;panel.setAttribute('role',event.matches?'dialog':'complementary');if(event.matches)panel.setAttribute('aria-modal','true');else panel.removeAttribute('aria-modal');});
  return {sync,editor,cards,command,revisionCompleted,openPreview};
}
