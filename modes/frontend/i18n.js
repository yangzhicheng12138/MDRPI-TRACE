import {messages} from './messages-en.js';

const KEY='mdrpi.interface-language';
const supported=value=>value==='en'||value==='zh';
function preference(){
  try{const value=globalThis.localStorage?.getItem(KEY);if(supported(value))return value;}catch(_){}
  return globalThis.navigator?.language?.toLowerCase().startsWith('zh')?'zh':'en';
}
let language=preference();
export const interfaceLanguage=()=>language;
export const interfaceLocale=()=>language==='zh'?'zh-CN':'en-US';

// Whole UI messages only. Captured data (names, filenames, numbers) is retained.
const patterns=[
  [/^(\d+)\s*条$/,(_,n)=>`${n} item${n==='1'?'':'s'}`],
  [/^(\d+)\s*项$/,(_,n)=>`${n} point${n==='1'?'':'s'}`],
  [/^(\d+)\s*题$/,(_,n)=>`${n} question${n==='1'?'':'s'}`],
  [/^(\d+)\s*道$/,(_,n)=>`${n} question${n==='1'?'':'s'}`],
  [/^(\d+)\s*次$/,(_,n)=>`${n} time${n==='1'?'':'s'}`],
  [/^(\d+)\s*\/\s*(\d+) 字$/,(_,n,max)=>`${n} / ${max} characters`],
  [/^提示 (\d+)$/,(_,n)=>`Hint ${n}`],
  [/^讲解 (\d+)$/,(_,n)=>`Explanation ${n}`],
  [/^提示 (\d+)\/3 · 还可 (\d+) 次$/,(_,n,left)=>`Hints ${n}/3 · ${left} remaining`],
  [/^还可查看 (\d+) 条提示；细化已有提示不增加次数。$/,(_,left)=>`${left} hints remaining. Exploring an existing hint does not use another hint.`],
  [/^本题共 (\d+) 条提示，还可查看 (\d+) 条；按需展开。$/,(_,total,left)=>`${total} hints in total; ${left} remaining. Open them as needed.`],
  [/^已看 (\d+) 条提示$/,(_,n)=>`${n} hints viewed`],
  [/^已覆盖 (\d+)\/(\d+) 个要点( · 可直接补充)?$/,(_,n,total,extra)=>`${n}/${total} points covered${extra?' · You can add more':''}`],
  [/^第 (\d+)\/6 步 · (.+)$/,(_,n,title)=>`Step ${n}/6 · ${messages[title]||title}`],
  [/^当前 (\d+)\/6$/,(_,n)=>`Current step ${n}/6`],
  [/^符合条件：(\d+) 道练习( · 已按关键词搜索)?$/,(_,n,searched)=>`${n} matching exercises${searched?' · Keyword filter applied':''}`],
  [/^已选 (\d+) 项$/,(_,n)=>`${n} selected`],
  [/^原始难度：(.+)$/,(_,label)=>`Source difficulty: ${messages[label]||label}`],
  [/^(单选题|多选题|判断题|排序题|简答题)(作答|选项)$/,(_,kind,part)=>`${messages[kind]||kind} ${part==='作答'?'answer':'options'}`],
  [/^已保留全文 (\d+) 字，超过 (\d+) 字上限；请修改后发送。$/,(_,n,limit)=>`All ${n} characters are saved. Shorten the message to ${limit} characters before sending.`],
  [/^输入为 (\d+) 字，超过本模式 (\d+) 字的上限；全文已保留，请修改后发送。$/,(_,n,limit)=>`Your ${n}-character draft is saved. This mode allows ${limit} characters; shorten it before sending.`],
  [/^学习档案 · (.+)$/,(_,name)=>`Learning profile · ${name}`],
  [/^偏好档案 · (.+)$/,(_,name)=>`Preference profile · ${name}`],
  [/^(.+)的选项$/,(_,name)=>`Options for ${name}`],
  [/^查看(.+)的读取情况$/,(_,name)=>`View reading details for ${name}`],
  [/^移除(.+)$/,(_,name)=>`Remove ${name}`],
  [/^上移选项([A-Z])$/,(_,key)=>`Move option ${key} up`],
  [/^下移选项([A-Z])$/,(_,key)=>`Move option ${key} down`],
  [/^(.+) · 待发送$/,(_,size)=>`${size} · Ready to send`],
  [/^请求未完成（(\d+)），请稍后重试。$/,(_,code)=>`Request failed (${code}). Please retry.`],
];

export function translate(source,locale=language){
  if(typeof source!=='string'||locale!=='en')return source;
  const trimmed=source.trim();
  let target=messages[trimmed];
  if(target===undefined){
    for(const [pattern,render] of patterns){
      const match=trimmed.match(pattern);
      if(match){target=render(...match);break;}
    }
  }
  if(target===undefined&&/[·、]/.test(trimmed)){
    // Composite catalog labels only when every component is a known UI label.
    const parts=trimmed.split(/(\s*·\s*|、)/);
    if(parts.every((part,index)=>index%2||messages[part]!==undefined||/^\d+$/.test(part))){
      target=parts.map((part,index)=>index%2?(part==='、'?', ':part):(messages[part]||part)).join('');
    }
  }
  if(target===undefined)return source;
  return source.slice(0,source.indexOf(trimmed))+target+source.slice(source.indexOf(trimmed)+trimmed.length);
}
export const t=translate;
export function bilingualTitle(value){
  const parts=String(value??'').split(' / ');
  return language==='en'&&parts.length>1?parts.slice(1).join(' / '):parts[0];
}

// UI labels are projected onto the existing vanilla DOM. No model output,
// student answer, editable value or request payload is rewritten.
const content='[data-no-i18n],[translate="no"],script,style,textarea,pre,code,kbd,.markdown,.message-body,.focus-question,.history-item-title,.history-excerpt,.objective-stem,.option-text,.source-quote,.source-excerpt,.content-text,.quality-excerpt,#artifact-preview-body,#artifact-preview-title,#header-title';
const uiInsideContent='[data-i18n],button[data-command],.code-head,.message-meta,.message-actions,.feedback-section>summary,.feedback-overview,.criterion-status,.answer-evidence-tag,.feedback-next';
const immutable='[data-no-i18n],[translate="no"],textarea,pre,code,kbd';
const attrs=['aria-label','aria-description','placeholder','title'];
const sources=new WeakMap(),attributeSources=new WeakMap();
let started=false,observer,scheduled=false;
const pending=new Set();

function mayTranslate(element){
  if(!element||element.closest(immutable))return false;
  if(element.closest(uiInsideContent))return true;
  return !element.closest(content);
}
function textNode(node){
  const parent=node.parentElement;
  if(!mayTranslate(parent))return;
  const remembered=sources.get(node);
  const source=remembered&&node.data===remembered.rendered?remembered.source:node.data;
  const rendered=translate(source);
  if(parent.tagName==='OPTION'&&!parent.hasAttribute('value')){
    // Option labels are localized; server-side enum values remain unchanged.
    parent.setAttribute('value',source);
  }
  sources.set(node,{source,rendered});
  if(node.data!==rendered)node.data=rendered;
}
function elementAttributes(element){
  const field=element.matches('input,textarea');
  if((!field&&!mayTranslate(element))||element.closest('[data-no-i18n],[translate="no"]')||element.matches('.history-item'))return;
  const saved=attributeSources.get(element)||{};
  for(const name of attrs){
    const value=element.getAttribute(name);if(value===null)continue;
    const previous=saved[name];
    const source=previous&&value===previous.rendered?previous.source:value;
    const rendered=translate(source);saved[name]={source,rendered};
    if(value!==rendered)element.setAttribute(name,rendered);
  }
  attributeSources.set(element,saved);
}
export function localize(root=document.body){
  if(!root)return;
  if(root.nodeType===3){textNode(root);return;}
  if(root.nodeType!==1)return;
  elementAttributes(root);
  const walker=document.createTreeWalker(root,NodeFilter.SHOW_ELEMENT|NodeFilter.SHOW_TEXT);
  while(walker.nextNode()){
    const node=walker.currentNode;
    if(node.nodeType===3)textNode(node);else elementAttributes(node);
  }
}
function syncSelectors(){
  document.documentElement.lang=language==='en'?'en':'zh-CN';
  document.querySelectorAll('[data-interface-language]').forEach(select=>{select.value=language;});
}
export function setInterfaceLanguage(value,{persist=true}={}){
  if(!supported(value))return false;
  language=value;
  if(persist)try{localStorage.setItem(KEY,value);}catch(_){}
  if(typeof document!=='undefined'){
    localize(document.body);syncSelectors();
    window.dispatchEvent(new CustomEvent('mdrpi:languagechange',{detail:{language}}));
  }
  return true;
}
export function startLocalization(){
  if(started)return;started=true;
  const title=document.title;
  const updateTitle=()=>{if(document.title===title||document.title===translate(title,'en'))document.title=translate(title);};
  localize(document.body);syncSelectors();updateTitle();
  document.addEventListener('change',event=>{
    if(event.target.matches('[data-interface-language]'))setInterfaceLanguage(event.target.value);
  });
  window.addEventListener('mdrpi:languagechange',updateTitle);
  window.addEventListener('storage',event=>{
    if(event.key===KEY&&supported(event.newValue))setInterfaceLanguage(event.newValue,{persist:false});
  });
  observer=new MutationObserver(records=>{
    for(const record of records){
      if(record.type==='childList')record.addedNodes.forEach(node=>pending.add(node));
      else pending.add(record.target);
    }
    if(scheduled)return;scheduled=true;
    queueMicrotask(()=>{
      scheduled=false;
      const nodes=[...pending];pending.clear();
      for(const node of nodes)if(node.isConnected)localize(node);
    });
  });
  observer.observe(document.body,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:attrs});
}
