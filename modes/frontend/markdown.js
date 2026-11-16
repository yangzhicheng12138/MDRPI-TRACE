import {icon} from './icons.js';
import {interfaceLanguage,interfaceLocale} from './i18n.js';
export function escape(value='') { return String(value).replace(/[&<>"']/g, value=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[value])); }
const markdownCache=new Map();let cachedCharacters=0;
export function markdown(text='') {
  if(markdownCache.has(text))return markdownCache.get(text);
  const wrapper=document.createElement('div');
  if (!globalThis.marked || !globalThis.DOMPurify) { wrapper.textContent=text; return wrapper.innerHTML; }
  wrapper.innerHTML=DOMPurify.sanitize(marked.parse(text,{gfm:true,breaks:false}),{
    USE_PROFILES:{html:true}, ALLOW_DATA_ATTR:false,
    FORBID_TAGS:['style','script','iframe','form','button','input','textarea','select'],
    FORBID_ATTR:['style','id','name']
  });
  wrapper.querySelectorAll('a').forEach(link=>{ link.target='_blank'; link.rel='noopener noreferrer'; });
  wrapper.querySelectorAll('img').forEach(image=>{image.loading='lazy';image.referrerPolicy='no-referrer';});
  wrapper.querySelectorAll('table').forEach(table=>{const box=document.createElement('div');box.className='table-wrap';table.replaceWith(box);box.append(table);});
  wrapper.querySelectorAll('pre').forEach(pre=>{
    const code=pre.querySelector('code'); const language=code?.className.replace('language-','') || 'text';
    if(code && globalThis.hljs && hljs.getLanguage(language)) code.innerHTML=hljs.highlight(code.textContent,{language}).value;
    const box=document.createElement('div');box.className='code-block';
    const head=document.createElement('div');head.className='code-head';head.innerHTML=`<span>${escape(language)}</span><button type="button" data-command="copy-code" aria-label="复制代码">${icon('copy')}复制代码</button>`;
    pre.replaceWith(box);box.append(head,pre);
  });
  const html=wrapper.innerHTML;
  // Bound both entry count and text/HTML size for long conversations.
  const size=text.length+html.length;
  if(size<=150000){
    while(markdownCache.size&&(markdownCache.size>=32||cachedCharacters+size>600000)){
      const key=markdownCache.keys().next().value;cachedCharacters-=key.length+markdownCache.get(key).length;markdownCache.delete(key);
    }
    markdownCache.set(text,html);cachedCharacters+=size;
  }
  return html;
}

function annotatedAnswer(content,annotations) {
  if(!Array.isArray(annotations)||!annotations.length)return escape(content);
  const boundary=index=>!(index>0&&index<content.length&&/[\uD800-\uDBFF]/.test(content[index-1])&&/[\uDC00-\uDFFF]/.test(content[index]));
  const ranges=annotations.filter(row=>row&&Number.isInteger(row.start)&&Number.isInteger(row.end)&&row.start>=0&&row.end>row.start&&row.end<=content.length&&boundary(row.start)&&boundary(row.end)&&Array.isArray(row.criteria)&&row.criteria.length)
    .map(row=>({start:row.start,end:row.end,criteria:row.criteria.filter(point=>point&&typeof point.label==='string'&&point.label.trim()).map(point=>({id:String(point.id||''),label:point.label.trim()}))}))
    .filter(row=>row.criteria.length).sort((a,b)=>a.start-b.start||a.end-b.end);
  const merged=[];
  for(const row of ranges){
    const previous=merged.at(-1);
    if(previous&&row.start<previous.end){previous.end=Math.max(previous.end,row.end);previous.criteria.push(...row.criteria);}
    else merged.push({...row,criteria:row.criteria.slice()});
  }
  let cursor=0,html='';
  for(const row of merged){
    const labels=[...new Set(row.criteria.map(point=>point.label))].join('；'),description='已覆盖要点：'+labels,ids=[...new Set(row.criteria.map(point=>point.id))];
    html+=escape(content.slice(cursor,row.start))+`<mark class="answer-evidence" tabindex="0" data-criterion-ids="${escape(JSON.stringify(ids))}" title="${escape(description)}"><span class="answer-evidence-text">${escape(content.slice(row.start,row.end))}</span><span class="answer-evidence-tag" aria-hidden="true">已覆盖</span><span class="sr-only">（${escape(description)}）</span></mark>`;
    cursor=row.end;
  }
  return html+escape(content.slice(cursor));
}

export function renderMessage(message,index,options={}) {
  const content=message.display_content??message.content??'';
  // Translate only service-generated setup labels, never a learner's answer.
  let userContent=message.content||'';
  if(message.kind==='setup'&&interfaceLanguage()==='en')userContent=userContent.replace(/^选择练习：/,'Selected exercise: ').replace(/^我提供案例$/,'Use my own case').replace(/^按主题出题$/,'Create a case by topic');
  if(message.kind==='welcome') return '';
  if(message.kind==='learning_section') return `<div class="exercise-separator">${escape(content.replace(/^###\s*/,''))}</div>`;
  const user=message.role==='user';
  const labels={feedback:'作答反馈',hint:'提示',summary:'练习总结',recheck:'重新核对',case:'练习题',cancelled:'已停止',teaching_dialogue:'教学讲解',clarification:'确认与说明',social:'交流'};
  let time='';try{time=new Date(message.at).toLocaleTimeString(interfaceLocale(),{hour:'2-digit',minute:'2-digit'});}catch(_){}
  return `<article class="message ${user?'user':'assistant'}" data-message-index="${index}">
    ${user?'':`<div class="message-meta"><img src="/ui/assets/brand-care-t.svg" alt="" width="26" height="26"><strong>MDRPI</strong>${labels[message.kind]?`<span class="message-kind">${labels[message.kind]}</span>`:''}<time class="message-time">${escape(time)}</time></div>`}
    <div class="message-body ${user?'':'markdown'}">${user?options.mode==='teaching'&&message.kind==='student'?annotatedAnswer(message.content||'',message.annotations):escape(userContent):options.teachingFeedback&&message.kind!=='summary'&&(message.feedback_view||['feedback','recheck'].includes(message.kind))?options.teachingFeedback(content,message.feedback_view,{messages:options.messages,index}):markdown(content)}</div>
    ${options.editor||''}
    <div class="message-actions"><button type="button" data-command="copy-message" data-index="${index}" aria-label="复制${user?'我的消息':'回答'}">${icon('copy')}复制</button>${options.canEdit?`<button type="button" data-command="edit-message" data-index="${index}" aria-label="${options.mode==='teaching'?'补充更正作答':'编辑消息'}">${icon('edit')}${options.mode==='teaching'?'补充更正':'编辑'}</button>`:user?`<button type="button" data-command="reuse-message" data-index="${index}" aria-label="将这条消息放回输入框">${icon('edit')}放入输入框</button>`:''}</div>
  </article>`;
}
