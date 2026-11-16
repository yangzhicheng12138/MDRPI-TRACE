import {escape} from './markdown.js';

const labels={recognize:'识别线索',analyze:'分析线索',prioritize:'确定优先问题',solutions:'制定解决方案',action:'采取行动',evaluate:'评价结果'};
const populationLabels={adult:'成人',pediatric:'儿童',neonatal:'新生儿'};
const deviceLabels={niv_mask:'无创通气面罩',nasogastric_tube:'鼻胃管',endotracheal_tube:'气管插管',tracheostomy_tube:'气管切开套管',pulse_oximeter:'血氧探头',cervical_collar:'颈托',cast:'石膏/支具',nasal_cannula:'鼻导管',nasojejunal_tube:'鼻空肠管',nasal_prong_cpap:'鼻塞式CPAP',bite_block:'口咬器',general_mdrpi:'其他器械'};
const validValue=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=100;
const number=value=>Number.isFinite(Number(value))?Number(value):0;
const point=(index,radius)=>{const angle=-Math.PI/2+index*Math.PI/3;return [180+Math.cos(angle)*radius,150+Math.sin(angle)*radius];};
const pair=values=>values.map(value=>value.toFixed(2)).join(',');

export function renderRadar(sixSteps=[]) {
  const axes=Object.entries(labels).map(([id,label])=>({id,label,...sixSteps.find(row=>row.id===id)}));
  const measured=axes.filter(axis=>validValue(axis.value));
  const description=axes.map(axis=>`${axis.label}：${validValue(axis.value)?axis.value+'%':'未评估'}`).join('；');
  const grid=[25,50,75,100].map(level=>`<polygon points="${axes.map((axis,index)=>pair(point(index,level*.93))).join(' ')}" class="radar-grid"/>`).join('');
  const rays=axes.map((axis,index)=>`<line x1="180" y1="150" x2="${point(index,93)[0].toFixed(2)}" y2="${point(index,93)[1].toFixed(2)}" class="radar-grid"/>`).join('');
  const polygon=measured.length===6?`<polygon class="radar-area" points="${axes.map((axis,index)=>pair(point(index,axis.value*.93))).join(' ')}"/>`:'';
  const markers=axes.map((axis,index)=>{
    if(!validValue(axis.value))return '';
    const [x,y]=point(index,axis.value*.93);
    return `<line class="radar-value-ray" x1="180" y1="150" x2="${x.toFixed(2)}" y2="${y.toFixed(2)}"/><circle class="radar-value" data-axis="${escape(axis.id)}" cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="4"/>`;
  }).join('');
  const captions=axes.map((axis,index)=>{
    const [x,y]=point(index,115),anchor=index===0||index===3?'middle':index<3?'start':'end';
    return `<text class="radar-label" x="${x.toFixed(2)}" y="${y.toFixed(2)}" text-anchor="${anchor}">${escape(axis.label)}</text><text class="radar-label-value" x="${x.toFixed(2)}" y="${(y+15).toFixed(2)}" text-anchor="${anchor}">${validValue(axis.value)?axis.value+'%':'未评估'}</text>`;
  }).join('');
  return `<figure class="learning-radar" data-measured-axes="${measured.length}"><svg viewBox="0 0 360 300" role="img" aria-label="${escape(description)}">${grid}${rays}${polygon}${markers}${captions}</svg><figcaption>${measured.length?'显示实际作答评价中的必答点覆盖率。':'暂无六步作答评价，尚未形成能力测量。'}${measured.length>0&&measured.length<6?'未评估维度留空，不参与连接。':''}</figcaption></figure>`;
}

export function renderLearningAnalytics(data) {
  if(!data)return '<p class="panel-note">这批历史记录未保存六步明细，暂无雷达数据。</p>';
  const counts=data.counts||{},axes=data.six_steps||[],trend=data.trend||{};
  const stats=[['训练轮次',counts.training_rounds],['完成轮次',counts.completed_rounds],['中断轮次',counts.interrupted_rounds],['进行中',counts.in_progress_rounds],['案例/题目覆盖',counts.unique_cases],['提示使用',counts.hint_count]].map(([label,value])=>`<div class="stat"><b>${number(value)}</b><span>${label}</span></div>`).join('');
  const rows=axes.map(axis=>`<tr data-axis="${escape(axis.id)}"><th scope="row">${escape(axis.label)}</th><td>${validValue(axis.value)?axis.value+'%':'未评估'}</td><td>${number(axis.met)}/${number(axis.required)}</td><td>${number(axis.sample_count)}</td><td>${number(axis.hint_count)}</td></tr>`).join('');
  const table=`<div class="analytics-table-wrap"><table class="analytics-table"><caption>六步作答覆盖记录</caption><thead><tr><th scope="col">步骤</th><th scope="col">覆盖率</th><th scope="col">已覆盖/必答点</th><th scope="col">评价轮次</th><th scope="col">提示</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  const coverage=[['difficulty','难度',{}],['populations','人群',populationLabels],['device_types','医疗器械',deviceLabels]].map(([key,label,mapping])=>`<div class="coverage-row"><strong>${label}</strong><span>${(data.coverage?.[key]||[]).map(row=>`${escape(mapping[row.id]||row.id)} ${number(row.completed_rounds)}/${number(row.rounds)}轮完成`).join(' · ')||'暂无记录'}</span></div>`).join('');
  const weak=(data.weak_points||[]).map(row=>`<li>${escape(labels[row.step_id]||row.step_id)}：${escape(row.label)} <span>记录 ${number(row.count)} 次</span></li>`).join('');
  const change=trend.available?`<p class="panel-note">同一评价版本，前组 ${number(trend.previous)}% → 近期 ${number(trend.recent)}%（${trend.delta>0?'+':''}${number(trend.delta)} 个百分点）。${escape(trend.reason||'')}</p>`:`<p class="panel-note">${escape(trend.reason||'同一评价版本下的已完成训练不足，暂无可比较变化。')}</p>`;
  return `<section class="learning-analytics" aria-label="真实学习记录"><div class="stat-grid analytics-stats">${stats}</div><h3 class="panel-heading">六步表现</h3><p class="panel-note">必答点覆盖率来自学生真实作答和保存的评价，属于形成性练习记录；不是经专家验证的临床能力分数。未评估步骤不计零分。</p><div class="learning-chart">${renderRadar(axes)}${table}</div><h3 class="panel-heading">练习覆盖</h3>${coverage}<h3 class="panel-heading">常见遗漏与待巩固内容</h3>${weak?`<ul class="weak-point-list">${weak}</ul>`:'<p class="panel-note">暂未记录明确的遗漏或错误；这不代表全部能力已经评估。</p>'}<h3 class="panel-heading">近期变化</h3>${change}</section>`;
}
