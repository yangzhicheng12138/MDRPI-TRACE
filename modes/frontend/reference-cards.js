import {escape} from './markdown.js?v=20261011-teaching-ux';
import {icon} from './icons.js';

// Structured audit fields remain in the API. This is the reader's bibliography:
// supplied citation facts and working links, with no metadata/checking notices.
export function originalReferenceLink(link={}) {
  if(link?.status!=='verified'||!/^https:\/\//i.test(link.url||''))return '';
  return `<a class="button" href="${escape(link.url)}" target="_blank" rel="noopener noreferrer">查看原文 ${icon('arrow')}</a>`;
}

export function bibliographyText(rows=[]) {
  return rows.map((row,index)=>`[${index+1}] ${row.reference||''}`).join('\n');
}

export function teachingReferenceCards(data,notices=[]) {
  const rows=data.references||[];
  const head=`<div class="reference-panel-heading"><p class="panel-note">参考文献 · ${escape(data.standard||'GB/T 7714-2025')}</p>${rows.length?'<button type="button" class="button" data-command="copy-references">复制全部题录</button>':''}</div>`;
  return head+notices.map(note=>`<p class="panel-note">${escape(note)}</p>`).join('')+rows.map((row,index)=>{
    const webpage=row.reference_type==='EB'||row.resource_type==='网页'||/\[EB(?:\/OL)?\]/.test(row.reference||'');
    const locations=webpage?[]:row.pdf_locations||(row.locations||[]).filter(value=>/^PDF\s*(?:第\s*\d|pages?\s+\d)/i.test(value));
    const standard=row.standard_number;
    const link=originalReferenceLink(row.link);
    return `<article class="record-row reference-card teaching-reference-card"><strong><span class="reference-number">[${index+1}]</span> ${escape(row.title)}</strong><p class="reference-citation">${escape(row.reference)}</p>
      ${row.resource_type?`<p class="reference-kind">资料类型：${escape(row.resource_type)}</p>`:''}
      ${standard?`<p>标准编号：${escape(standard)}</p>${row.source_name?`<p>责任机构：${escape(row.source_name)}</p>`:''}`:''}
      ${standard&&(row.publication_date||row.effective_date)?`<p>${row.publication_date?'发布日期：'+escape(row.publication_date):''}${row.effective_date?(row.publication_date?' · ':'')+'实施日期：'+escape(row.effective_date):''}</p>`:''}
      ${locations.length?`<p class="reference-location">原文位置：${escape(locations.join('；'))}</p>`:''}
      <div class="action-row reference-actions"><button type="button" class="button" data-command="copy-reference" data-reference-index="${index}">复制题录</button><span id="reference-original-${index}" class="reference-original" ${link?'':'hidden'}>${link}</span></div>
    </article>`;
  }).join('')+(rows.length?'':'<p class="panel-note">这轮没有保存可展示的外部参考资料。</p>');
}

export function updateReferenceLinks(data,root=document) {
  (data.references||[]).forEach((row,index)=>{
    const node=root.querySelector(`#reference-original-${index}`);
    if(!node)return;
    const html=originalReferenceLink(row.link);
    node.hidden=!html;
    if(node.innerHTML!==html)node.innerHTML=html;
  });
}
