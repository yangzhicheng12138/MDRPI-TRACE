function learnerKey() {
  let key=localStorage.getItem('mdrpi.learner-key');
  if(!/^[a-f0-9]{64}$/.test(key||'')){key=Array.from(crypto.getRandomValues(new Uint8Array(32)),x=>x.toString(16).padStart(2,'0')).join('');localStorage.setItem('mdrpi.learner-key',key);}
  return key;
}
export async function api(path,body,method,timeout=20000) {
  const response=await fetch('/api'+path,{method:method || (body===undefined?'GET':'POST'),
    headers:{'X-Learner-Key':learnerKey(),...(body instanceof FormData?{}:{'Content-Type':'application/json'})},
    body:body===undefined?undefined:body instanceof FormData?body:JSON.stringify(body),signal:AbortSignal.timeout(timeout)});
  const data=await response.json().catch(()=>({detail:'服务未返回可读取的结果。'}));
  if(!response.ok) { const detail=data.detail || data.error || `请求失败（${response.status}）`; throw new Error(Array.isArray(detail)?detail.map(item=>item.msg).join('；'):detail); }
  return data;
}

export async function stream(sid,body,onEvent) {
  const response=await fetch(`/api/sessions/${encodeURIComponent(sid)}/turn/stream`,{method:'POST',headers:{'Content-Type':'application/json','X-Learner-Key':learnerKey()},body:JSON.stringify(body)});
  if(!response.ok) { const data=await response.json();throw new Error(data.detail || '本次请求未成功。'); }
  const reader=response.body.getReader(),decoder=new TextDecoder();let buffered='',completed=false;
  const frame=value=>{
    const lines=value.split('\n');const name=lines.find(line=>line.startsWith('event:'))?.slice(6).trim();
    const raw=lines.filter(line=>line.startsWith('data:')).map(line=>line.slice(5).trimStart()).join('\n');
    if(!name || !raw) return;
    const data=JSON.parse(raw); if(name==='done') completed=true;
    if(name==='error') throw new Error(data.message);
    onEvent(name,data);
  };
  while(true) { const {value,done}=await reader.read();if(done)break;buffered+=decoder.decode(value,{stream:true});let at;while((at=buffered.indexOf('\n\n'))!==-1){frame(buffered.slice(0,at));buffered=buffered.slice(at+2);} }
  buffered+=decoder.decode();if(buffered.trim())frame(buffered);
  if(!completed) throw new Error('连接已中断，正在从已保存的会话恢复。');
}
