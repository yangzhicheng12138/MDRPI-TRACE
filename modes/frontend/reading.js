import {escape} from './markdown.js?v=20261116-i18n';

// Only pass HTML produced by the existing sanitized Markdown renderer or our
// escaped UI templates. Preserve unchanged nodes and text selection on append.
export function patchHTML(target, html) {
  if(target._patchedHTML===html)return;
  const template=document.createElement('template');template.innerHTML=html;
  syncChildren(target,template.content);
  target._patchedHTML=html;
}
function syncChildren(target, source) {
  let current=target.firstChild;
  for(const fresh of [...source.childNodes]){
    if(!current){target.append(fresh);continue;}
    const next=current.nextSibling;
    if(current.nodeType!==fresh.nodeType||current.nodeName!==fresh.nodeName){current.replaceWith(fresh);}
    else if(current.nodeType===Node.TEXT_NODE){
      if(fresh.data!==current.data){
        if(fresh.data.startsWith(current.data))current.appendData(fresh.data.slice(current.length));
        else current.replaceData(0,current.length,fresh.data);
      }
    }else if(current.nodeType===Node.ELEMENT_NODE&&!current.isEqualNode(fresh)){
      for(const attr of [...current.attributes]){
        const readingAttribute=current.hasAttribute('data-reading-target')&&['id','tabindex','data-reading-target'].includes(attr.name);
        if(!fresh.hasAttribute(attr.name)&&attr.name!=='open'&&!readingAttribute)current.removeAttribute(attr.name);
      }
      for(const attr of fresh.attributes)if(current.getAttribute(attr.name)!==attr.value)current.setAttribute(attr.name,attr.value);
      syncChildren(current,fresh);
    }
    current=next;
  }
  while(current){const next=current.nextSibling;current.remove();current=next;}
}

export function moveElement(parent, element, before=null) {
  if('moveBefore' in Element.prototype&&element.isConnected&&parent.isConnected)parent.moveBefore(element,before);
  else parent.insertBefore(element,before);
}

export function highlightedExcerpt(match) {
  const text=Array.from(match?.text||''),start=Number(match?.start),end=Number(match?.end);
  if(!Number.isInteger(start)||!Number.isInteger(end)||start<0||end<=start||end>text.length)return escape(text.join(''));
  return escape(text.slice(0,start).join(''))+'<mark>'+escape(text.slice(start,end).join(''))+'</mark>'+escape(text.slice(end).join(''));
}

const steps=[
  ['识别线索',/识别线索|recognize\s+cues/i],
  ['分析线索',/分析线索|analy[sz]e\s+cues/i],
  ['问题排序',/问题排序|确定优先问题|优先假设|prioriti[sz]e\s+hypothes/i],
  ['生成方案',/生成方案|制定解决方案|generate\s+solutions/i],
  ['采取行动',/采取行动|take\s+action/i],
  ['评价结果',/评价结果|评估结果|evaluate\s+outcomes/i],
];

export function createReadingNavigation({onNavigate}) {
  const area=document.querySelector('#scroll-area'),nav=document.querySelector('#answer-outline');
  const native=CSS.supports('scroll-target-group: auto')&&CSS.supports('selector(:target-current)');
  let targets=[],observer=null,scrollTimer,lastCurrent=null;
  function setCurrent(current){
    nav.querySelectorAll('a').forEach(link=>{
      const active=link===current;link.setAttribute('aria-current',active?'location':'false');
      link.classList.toggle('target-current',!native&&active);
    });
    if(current&&current!==lastCurrent){
      const bounds=nav.getBoundingClientRect(),link=current.getBoundingClientRect();
      if(link.left<bounds.left||link.right>bounds.right)nav.scrollLeft+=link.left-bounds.left-(bounds.width-link.width)/2;
    }
    lastCurrent=current;
  }
  function sync(){
    if(!targets.length)return;
    let current=native?nav.querySelector('a:target-current'):null;
    if(!current){
      const edge=area.getBoundingClientRect().top+80;
      const target=targets.filter(item=>item.getBoundingClientRect().top<=edge).at(-1)||targets[0];
      current=nav.querySelector(`a[href="#${target.id}"]`);
    }
    setCurrent(current);
  }
  area.addEventListener('scroll',()=>{clearTimeout(scrollTimer);scrollTimer=setTimeout(sync,100);},{passive:true});
  area.addEventListener('scrollend',sync);
  nav.addEventListener('click',event=>{
    const link=event.target.closest('a');if(!link)return;
    const target=targets.find(item=>'#'+item.id===link.hash);if(!target)return;
    event.preventDefault();onNavigate();
    target.focus({preventScroll:true});
    area.scrollTo({top:area.scrollTop+target.getBoundingClientRect().top-area.getBoundingClientRect().top-16,
      behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'instant':'smooth'});
    setCurrent(link);
  });
  function refresh(mode){
    const articles=mode==='performance'?[...document.querySelectorAll('#messages .message.assistant')].reverse():[];
    let found=[];
    for(const article of articles){
      const headings=[...article.querySelectorAll('.message-body h1,.message-body h2,.message-body h3,.message-body h4')];
      found=steps.map(([label,pattern],index)=>({label,index,node:headings.find(h=>pattern.test(h.textContent))})).filter(row=>row.node);
      if(found.length>=3)break;
      found=[];
    }
    const next=found.map(row=>row.node);
    if(next.length===targets.length&&next.every((node,index)=>node===targets[index]))return;
    observer?.disconnect();targets=next;lastCurrent=null;nav.hidden=!found.length;
    if(!found.length){nav.replaceChildren();return;}
    for(const row of found){
      row.node.id=`answer-step-${row.node.closest('[data-message-index]').dataset.messageIndex}-${row.index}`;
      row.node.tabIndex=-1;row.node.dataset.readingTarget='true';
    }
    nav.innerHTML='<span class="outline-label">六步定位</span>'+found.map(row=>`<a href="#${row.node.id}"><span>${row.index+1}</span>${row.label}</a>`).join('');
    if(!native){
      observer=new IntersectionObserver(sync,{root:area,rootMargin:'-10% 0px -65% 0px'});
      targets.forEach(node=>observer.observe(node));
    }
    requestAnimationFrame(sync);
  }
  return {refresh};
}

export function locateSearchResult(messages, match, query) {
  if(!match?.message_at)return false;
  const index=messages.findIndex(message=>message.at===match.message_at&&message.role===match.role&&message.kind===match.kind);
  const article=document.querySelector(`#messages [data-message-index="${index}"]`);
  if(!article)return false;
  for(let node=article.parentElement;node;node=node.parentElement)if(node.tagName==='DETAILS')node.open=true;
  document.querySelectorAll('.search-result-focus').forEach(node=>node.classList.remove('search-result-focus'));
  article.classList.add('search-result-focus');article.tabIndex=-1;
  article.focus({preventScroll:true});
  // Mark only text nodes, so a search term can never inject HTML or break links.
  const fold=value=>value.toLowerCase().replace(/ß/g,'ss').replace(/ς/g,'σ');
  const body=article.querySelector('.message-body'),needle=fold(query.trim());
  const walker=document.createTreeWalker(body,NodeFilter.SHOW_TEXT);
  let node,marked;
  while(needle&&(node=walker.nextNode())){
    if(node.parentElement.closest('button,mark,script,style'))continue;
    let folded='',offset=0;const starts=[],ends=[];
    for(const char of node.data){const value=fold(char);folded+=value;for(let n=0;n<value.length;n++){starts.push(offset);ends.push(offset+char.length);}offset+=char.length;}
    const at=folded.indexOf(needle);
    if(at<0)continue;
    const range=document.createRange();range.setStart(node,starts[at]);range.setEnd(node,ends[at+needle.length-1]);
    marked=document.createElement('mark');marked.className='search-hit';range.surroundContents(marked);break;
  }
  (marked||article).scrollIntoView({block:'center',behavior:'instant'});
  return true;
}
