// Deliberately use textContent for all model- and material-supplied text.
const $ = selector => document.querySelector(selector);
const API = '/api/question-generation';
const statusNames = {
  queued: '等待执行', retrieving: '检索与取证中', generating: '生成题包中',
  reviewing: '独立审核中', revising: '根据审核修订中', completed: '本轮完成',
  partial: '部分题目通过', needs_evidence: '证据仍需补充', failed: '任务未完成',
  approved: 'AI 审核通过', accepted: 'AI 审核通过', accept: 'AI 审核通过',
  rejected: '未通过审核', reject: '未通过审核', needs_revision: '需要修订',
  revision_needed: '需要修订', revise: '需要修订',
  needs_review: '需重新审核',
  draft: '待审核草稿', pending: '等待审核', pass: 'AI 审核通过',
};
const terminal = new Set(['completed', 'partial', 'needs_evidence', 'failed']);
const populationNames = {adult: '成人', pediatric: '儿童', neonatal: '新生儿'};
const issueNames = {medical_error: '医学错误', evidence_mismatch: '证据适用性问题', material_gap: '材料不足', ambiguity: '题意不明确', assessment_mismatch: '评分要求不一致', format: '题目格式问题'};
const stepNames = {recognize: '识别线索', recognize_cues: '识别线索', analyze: '分析线索', analyze_cues: '分析线索', prioritize: '问题排序', prioritize_hypotheses: '问题排序', solutions: '生成方案', generate_solutions: '生成方案', action: '采取行动', take_action: '采取行动', evaluate: '评价结果', evaluate_outcomes: '评价结果'};
const auditStatusNames = {consistent: '一致', aligned: '匹配', supported: '支持', limited: '支持范围有限', unsupported: '未获支持', misaligned: '不匹配', uncertain: '待核查', distinct: '考点有区别', overlap: '考点重叠', not_applicable: '本项不适用', passed: '已核对', pass: '已核对', failed: '需复核', fail: '需复核', warning: '需复核'};
let currentId = '';
let pollTimer = null;
let materialRows = [];
let deviceLabels = {};
let suggestedTopic = '无创通气面罩相关压力性损伤';
let questionSignature = '';
let pollFailures = 0;
let posting = false;
let batchPlanSignature = '';
let batchPlanJobId = '';

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}

function text(value) {
  if (value === undefined || value === null) return '';
  if (typeof value === 'string') return value;
  return JSON.stringify(value, null, 2);
}

function label(value) { return statusNames[value] || value || '等待更新'; }

function safeUrl(value) {
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : '';
  } catch (_) { return ''; }
}

function showError(target, message) {
  target.textContent = message || '';
  target.hidden = !message;
}

async function request(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  const headers = {'Content-Type': 'application/json'};
  try {
    const key = localStorage.getItem('mdrpi.learner-key');
    if (/^[a-f0-9]{64}$/.test(key || '')) headers['X-Learner-Key'] = key;
  } catch (_) { /* The existing same-origin cookie remains usable. */ }
  try {
    const response = await fetch(API + path, {...options, headers, credentials: 'same-origin', signal: controller.signal});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? '出题要求不完整或超出限制，请检查表单。' : data.detail;
      throw new Error(detail || `请求未完成（${response.status}），请稍后重试。`);
    }
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('连接超时，任务可能仍在执行。请刷新任务列表确认，避免重复提交。');
    if (error instanceof TypeError) throw new Error('暂时无法连接服务，请确认网络后重试。');
    throw error;
  } finally { clearTimeout(timeout); }
}

async function loadMaterials() {
  const container = $('#materials');
  try {
    const data = await request('/materials');
    materialRows = Array.isArray(data) ? data : data.materials || [];
    deviceLabels = data.device_labels || {};
    populateDevices();
  } catch (error) {
    container.replaceChildren(node('p', 'error', error.message));
    $('#generate').disabled = true;
  }
}

function populationsOf(material) {
  return Array.isArray(material.populations) ? material.populations : material.population ? [material.population] : [];
}

function populateDevices() {
  const population = $('[name=population]').value;
  const select = $('[name=device_type]');
  const previous = select.value;
  const available = [...new Set(materialRows.filter(material => populationsOf(material).includes(population)).flatMap(material => material.device_types || []))];
  select.replaceChildren(...available.map(value => {
    const option = node('option', '', deviceLabels[value] || value); option.value = value; return option;
  }));
  if (available.includes(previous)) select.value = previous;
  else if (available.includes('niv_mask')) select.value = 'niv_mask';
  select.disabled = !available.length;
  renderMaterials();
}

function renderMaterials() {
  const container = $('#materials');
  const selected = new Set(Array.from(container.querySelectorAll('input:checked')).map(input => input.value));
  const population = $('[name=population]').value;
  const device = $('[name=device_type]').value;
  const compatible = materialRows.filter(material => populationsOf(material).includes(population) && (material.device_types || []).includes(device));
  container.replaceChildren();
  compatible.forEach((material, index) => {
      const id = material.material_id || material.id;
      if (!id) return;
      const option = node('label', 'material-option');
      const checkbox = node('input');
      checkbox.type = 'checkbox'; checkbox.name = 'material_ids'; checkbox.value = String(id);
      checkbox.checked = selected.has(String(id));
      checkbox.id = `material-${index}`;
      const description = node('span');
      description.append(node('strong', '', material.title || material.name || id));
      const populations = Array.isArray(material.populations) ? material.populations.map(value => populationNames[value] || value).join(' / ') : populationNames[material.population] || material.population;
      const devices = Array.isArray(material.device_types) ? material.device_types.map(value => deviceLabels[value] || value).join(' / ') : material.device_type;
      const meta = [populations, devices,
        material.description || material.summary].filter(Boolean).join(' · ');
      if (meta) description.append(node('small', '', meta));
      option.append(checkbox, description); container.append(option);
  });
  if (!container.childElementCount) container.append(node('p', 'muted', '该人群与器械暂无匹配材料，请更换选择或先登记开发材料。'));
  $('#generate').disabled = posting || !container.querySelector('input');
  const topic = $('[name=topic]');
  if (device && topic.value === suggestedTopic) {
    suggestedTopic = `${deviceLabels[device] || device}相关压力性损伤`;
    topic.value = suggestedTopic;
  }
}

function localDate(value) {
  if (!value) return '';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '' : date.toLocaleString('zh-CN', {month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit'});
}

async function loadJobs() {
  try {
    const data = await request('/jobs');
    const rows = Array.isArray(data) ? data : data.jobs || [];
    const container = $('#jobs');
    container.replaceChildren();
    rows.slice(0, 20).forEach(job => {
      const id = job.job_id;
      if (!id) return;
      const button = node('button', `job-row${id === currentId ? ' active' : ''}${!terminal.has(job.status) ? ' running' : ''}`);
      button.type = 'button'; button.dataset.jobId = id;
      button.append(node('i', `job-dot ${job.status || ''}`));
      const content = node('span');
      content.append(node('strong', '', job.blueprint?.topic || job.topic || '出题任务'));
      content.append(node('small', '', [label(job.status), localDate(job.created_at || job.created)].filter(Boolean).join(' · ')));
      button.append(content); button.addEventListener('click', () => selectJob(id)); container.append(button);
    });
    if (!rows.length) container.append(node('p', 'muted', '还没有出题任务。'));
  } catch (error) { $('#jobs').replaceChildren(node('p', 'error', error.message)); }
}

function addSection(container, title, value) {
  if (value === undefined || value === null || value === '' || (Array.isArray(value) && !value.length)) return;
  container.append(node('h3', '', title));
  if (Array.isArray(value)) {
    const list = node('ul');
    value.forEach(item => {
      if (!item || typeof item !== 'object') { list.append(node('li', 'content-text', item)); return; }
      const issueTitle = item.kind ? [issueNames[item.kind] || item.kind, item.location].filter(Boolean).join(' · ') : '';
      const entry = node('li', 'content-text', issueTitle || item.criterion || item.text || item.description || item.answer || item.error || text(item));
      if (issueTitle && item.text) entry.append(node('div', '', `对应内容：${item.text}`));
      if (Array.isArray(item.acceptable_answers) && item.acceptable_answers.length) entry.append(node('div', 'muted', `等价表述：${item.acceptable_answers.join('；')}`));
      if (item.explanation) entry.append(node('div', '', item.explanation));
      if (item.requested_change) entry.append(node('div', '', `修改建议：${item.requested_change}`));
      if (Array.isArray(item.evidence_ids) && item.evidence_ids.length) entry.append(node('div', 'muted', `证据编号：${item.evidence_ids.join('、')}`));
      list.append(entry);
    });
    container.append(list);
  } else container.append(node('div', 'content-text', text(value)));
}

function details(summary, className = '') {
  const element = node('details', `question-details ${className}`);
  element.append(node('summary', '', summary));
  const body = node('div', 'details-body');
  element.append(body);
  return {element, body};
}

function readableEvidenceText(value) {
  // PDF extraction can retain table formatting. Convert only known formatting
  // tags; clinical text, numbers and comparisons remain literal textContent.
  return text(value)
    .replace(/<br\b[^>]*>/gi, '\n')
    .replace(/<\/(?:td|th)\s*>/gi, '\t')
    .replace(/<\/(?:table|thead|tbody|tfoot|tr|p|div)\s*>/gi, '\n')
    .replace(/<(?:table|thead|tbody|tfoot|tr|td|th|p|div)\b[^>]*>/gi, '');
}

function evidenceBlock(container, evidence) {
  const rows = Array.isArray(evidence) ? evidence : evidence && typeof evidence === 'object' ? Object.values(evidence) : [];
  if (!rows.length) { container.append(node('p', 'muted', '此题的证据详情尚未返回，可下载题包查看完整记录。')); return; }
  rows.forEach(source => {
    if (typeof source !== 'object' || !source) { container.append(node('p', 'content-text', text(source))); return; }
    const item = node('div', 'evidence-item');
    const originalTitle = source.title || source.source_title || source.evidence_id || source.source_id || '证据片段';
    const title = /<\/?(?:table|tr|td|th)\b/i.test(originalTitle) ? '网页证据（PDF）' : originalTitle;
    item.append(node('strong', '', title));
    const evidenceId = source.evidence_id || source.source_id;
    if (evidenceId) item.append(node('p', 'muted', `证据编号：${evidenceId}`));
    const url = safeUrl(source.url || source.source_url);
    if (url) { const link = node('a', '', url); link.href = url; link.target = '_blank'; link.rel = 'noopener noreferrer'; item.append(link); }
    const location = [source.location, source.page && `第 ${source.page} 页`, source.population && `人群：${populationNames[source.population] || source.population}`, source.device_type].filter(Boolean).join(' · ');
    if (location) item.append(node('p', 'muted', location));
    const excerpt = source.quote || source.excerpt || source.text || source.content;
    if (excerpt) item.append(node('p', '', readableEvidenceText(excerpt)));
    if (source.supports || source.supported_claim) item.append(node('p', '', `支持内容：${text(source.supports || source.supported_claim)}`));
    container.append(item);
  });
}

function displayText(value) {
  if (typeof value === 'string') return value.trim();
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return [value.zh, value.text, value.label, value.title].find(item => typeof item === 'string' && item.trim())?.trim() || '';
  }
  return '';
}

function recordRows(value) {
  return Array.isArray(value) ? value.filter(item => item !== null && item !== undefined) : typeof value === 'string' && value.trim() ? [value] : [];
}

function plannedItems(job) {
  return Array.isArray(job.batch_plan?.items) ? job.batch_plan.items.filter(item => item && typeof item === 'object' && !Array.isArray(item)) : [];
}

function planForItem(item, index, job) {
  if (item.plan && typeof item.plan === 'object' && !Array.isArray(item.plan) && Object.keys(item.plan).length) return item.plan;
  const rows = plannedItems(job);
  return rows.find(row => row.id && (row.id === item.plan_id || row.id === item.item_id)) || rows[index] || null;
}

function planSteps(plan) {
  return [...new Set(recordRows(plan?.steps).map(value => displayText(value)).filter(Boolean).map(value => stepNames[value] || value))];
}

function materialLabel(plan, job) {
  const id = displayText(plan?.material_id);
  const rows = [...(Array.isArray(job.materials) ? job.materials : []), ...materialRows];
  const material = rows.find(row => row && (row.id === id || row.material_id === id));
  return {id, title: displayText(plan?.material_title) || displayText(material?.title) || displayText(material?.name) || id || '来源材料待关联'};
}

function renderBatchPlan(job) {
  const panel = $('#batch-plan');
  const body = $('#batch-plan-body');
  const plan = job.batch_plan;
  if (!plan || typeof plan !== 'object' || Array.isArray(plan)) {
    panel.hidden = true;
    batchPlanSignature = '';
    return;
  }
  const rows = plannedItems(job);
  const warnings = [...new Set([...recordRows(plan.warnings), ...recordRows(plan.coverage?.warnings)].map(displayText).filter(Boolean))];
  if (!rows.length && !warnings.length) { panel.hidden = true; return; }
  const signature = JSON.stringify([job.job_id, plan, job.blueprint?.learning_objective, job.blueprint?.count, job.materials?.map?.(row => [row.id, row.title])]);
  panel.hidden = false;
  if (batchPlanJobId !== job.job_id) { panel.open = true; batchPlanJobId = job.job_id; }
  if (signature === batchPlanSignature) return;
  batchPlanSignature = signature;
  body.replaceChildren();
  const materialIds = new Set(rows.map(row => displayText(row.material_id)).filter(Boolean));
  const steps = new Set(rows.flatMap(planSteps));
  const requested = Number(job.blueprint?.count) || rows.length;
  $('#batch-plan-caption').textContent = `已安排 ${rows.length} 道 · ${materialIds.size} 份来源材料${steps.size ? ` · ${steps.size} 项能力` : ''}`;
  const goal = displayText(job.blueprint?.learning_objective);
  if (goal) { const goalBlock = node('div', 'batch-goal'); goalBlock.append(node('span', '', '整套学习目标'), node('p', '', goal)); body.append(goalBlock); }
  const stats = node('div', 'batch-stats');
  [[`${rows.length}${requested > rows.length ? ` / ${requested}` : ''}`, '计划题目'], [String(materialIds.size), '来源材料'], [steps.size ? String(steps.size) : '待标注', '涉及能力']].forEach(([value, title]) => {
    const stat = node('div'); stat.append(node('strong', '', value), node('span', '', title)); stats.append(stat);
  });
  body.append(stats);
  const selectedIds = Array.isArray(job.blueprint?.material_ids) ? job.blueprint.material_ids : [];
  const unassigned = selectedIds.filter(id => !materialIds.has(id));
  if (requested > rows.length) warnings.push(`已返回 ${rows.length} / ${requested} 道命题安排，其余安排尚待补齐。`);
  if (rows.length && unassigned.length) warnings.push(`已选材料中有 ${unassigned.length} 份尚未分配题目。`);
  if (warnings.length) {
    const notice = node('div', 'coverage-notice'); notice.append(node('strong', '', '覆盖提醒'));
    const list = node('ul'); [...new Set(warnings)].forEach(warning => list.append(node('li', '', warning))); notice.append(list); body.append(notice);
  }
  const list = node('ol', 'batch-plan-list');
  rows.forEach((row, index) => {
    const item = node('li', 'batch-plan-row');
    item.append(node('span', 'batch-item-number', String(index + 1).padStart(2, '0')));
    const content = node('div', 'batch-item-content');
    content.append(node('h3', '', displayText(row.focus) || '考查目标待补充'));
    const material = materialLabel(row, job);
    content.append(node('p', 'plan-material', `来源模板：${material.title}`));
    if (material.id && material.title !== material.id) content.append(node('p', 'plan-source-id', `材料编号：${material.id}`));
    const steps = planSteps(row);
    if (steps.length) { const chips = node('div', 'plan-chips'); steps.forEach(step => chips.append(node('span', 'plan-chip', step))); content.append(chips); }
    const guidance = displayText(row.task_guidance);
    if (guidance) content.append(node('p', 'plan-guidance', guidance));
    item.append(content); list.append(item);
  });
  if (rows.length) body.append(list);
}

function appendQuestionPlan(card, plan, job) {
  if (!plan) return;
  const focus = displayText(plan.focus);
  const material = materialLabel(plan, job);
  if (!focus && !material.id) return;
  const block = node('div', 'question-plan');
  if (focus) block.append(node('span', 'plan-chip focus-chip', `考查：${focus}`));
  if (material.id) block.append(node('span', 'plan-chip material-chip', `来源：${material.title}`));
  card.append(block);
}

function qualityRows(item) {
  const quality = item.quality_checks;
  if (!quality || typeof quality !== 'object') return {duplicates: [], checks: [], summary: ''};
  return {duplicates: recordRows(quality.duplicate_candidates), checks: recordRows(quality.review_checks), summary: displayText(quality.summary)};
}

function qualityText(value, fallback) {
  if (typeof value === 'string') return value;
  return [value?.message, value?.summary, value?.reason, value?.description, value?.note, value?.criterion, value?.title, value?.label].map(displayText).find(Boolean) || fallback;
}

function auditBadge(record) {
  const status = typeof record?.status === 'string' ? record.status : record?.passed === true ? 'passed' : record?.passed === false ? 'failed' : 'uncertain';
  const text = auditStatusNames[status] || '待核查';
  const positive = ['consistent', 'aligned', 'supported', 'distinct', 'passed', 'pass'].includes(status);
  return node('span', `audit-badge${positive ? ' positive' : status === 'not_applicable' ? ' neutral' : ' attention'}`, text);
}

function appendQualityChecks(body, item) {
  const {duplicates, checks, summary} = qualityRows(item);
  if (!duplicates.length && !checks.length && !summary) return;
  const section = details('质量复核记录', 'quality-checks');
  if (summary) section.body.append(node('p', 'content-text', summary));
  if (duplicates.length) {
    section.body.append(node('h3', '', `相似候选 ${duplicates.length} 条`), node('p', 'muted', '相似候选需结合题干与考查目标复核。'));
    const list = node('ul', 'quality-check-list');
    duplicates.forEach((candidate, index) => list.append(node('li', 'content-text', qualityText(candidate, `相似候选 ${index + 1}`))));
    section.body.append(list);
  }
  if (checks.length) {
    section.body.append(node('h3', '', '逐项质量复核'));
    const list = node('ul', 'quality-check-list');
    checks.forEach((check, index) => {
      const row = node('li'); row.append(node('span', 'content-text', qualityText(check, `质量复核 ${index + 1}`)));
      if (check && typeof check === 'object') row.append(auditBadge(check));
      if (displayText(check?.text)) row.append(node('div', 'quality-excerpt', `涉及内容：${check.text}`));
      list.append(row);
    });
    section.body.append(list);
  }
  body.append(section.element);
}

function appendQualityAudit(body, audit, question) {
  if (!audit || typeof audit !== 'object' || Array.isArray(audit)) return;
  const supports = recordRows(audit.evidence_support).filter(row => row && typeof row === 'object');
  if (!audit.case_facts && !audit.assessment_alignment && !audit.distinctness && !supports.length) return;
  const section = details('逐项核对依据', 'quality-audit');
  section.body.append(node('p', 'muted', '审核智能体的逐项核对记录，可结合来源正文与依据说明复核。'));
  [[audit.case_facts, '病例事实一致性'], [audit.assessment_alignment, '题面与评分对齐'], [audit.distinctness, '相似题考点复核']].forEach(([record, title]) => {
    if (!record || typeof record !== 'object') return;
    const row = node('div', 'audit-check');
    const heading = node('div', 'audit-check-heading'); heading.append(node('strong', '', title), auditBadge(record)); row.append(heading);
    const note = displayText(record.note);
    if (note) row.append(node('p', 'content-text', note));
    section.body.append(row);
  });
  supports.forEach(record => {
    const row = node('div', 'audit-check');
    const id = displayText(record.criterion_id);
    const heading = node('div', 'audit-check-heading'); heading.append(node('strong', '', id ? `评分点 ${id}` : '评分点证据支持'), auditBadge(record)); row.append(heading);
    const criterion = Array.isArray(question.scoring_points) ? question.scoring_points.find(point => point.id === id) : null;
    if (criterion?.criterion) row.append(node('p', 'content-text', criterion.criterion));
    const rationale = displayText(record.rationale);
    if (rationale) row.append(node('p', 'content-text', rationale));
    recordRows(record.sources).forEach(source => {
      if (!source || typeof source !== 'object') return;
      const isSummary = typeof source.summary === 'string';
      const quote = displayText(isSummary ? source.summary : source.quote);
      const evidenceId = displayText(source.evidence_id);
      const block = node(isSummary ? 'div' : 'blockquote', 'audit-quote');
      if (evidenceId) block.append(node('span', 'audit-source-id', `证据编号：${evidenceId}`));
      if (source.status) block.append(auditBadge(source));
      if (isSummary) block.append(node('span', 'muted', '依据概述'));
      if (quote) block.append(node('div', 'content-text', readableEvidenceText(quote)));
      else if (source.status && source.status !== 'supported') block.append(node('div', 'muted', isSummary ? '该来源未提供可支持此评分点的依据说明。' : '该来源未提供可支持此评分点的原文片段。'));
      if (evidenceId || quote) row.append(block);
    });
    section.body.append(row);
  });
  body.append(section.element);
}

function renderQuestion(item, index, job) {
  const q = item.question && typeof item.question === 'object' ? item.question : item;
  const review = item.review || item.latest_review || {};
  const status = item.status || review.verdict || review.decision || review.status || 'draft';
  const approved = ['accepted', 'approved', 'pass', 'accept'].includes(status);
  const card = node('article', 'question-card');
  card.dataset.itemId = item.item_id || item.id || index;
  const top = node('div', 'question-top');
  top.append(node('strong', '', `题目 ${index + 1} · ${q.question_type === 'single_choice' || job.blueprint?.question_type === 'single_choice' ? '单项选择' : '简答'}`));
  top.append(node('span', `status-pill${approved ? '' : ' warning'}`, label(status))); card.append(top);
  const plan = planForItem(item, index, job);
  appendQuestionPlan(card, plan, job);
  const quality = qualityRows(item);
  if (quality.duplicates.length || quality.checks.length || quality.summary) {
    const message = quality.duplicates.length ? `质量复核：记录了 ${quality.duplicates.length} 条相似候选，需结合考点复核。` : quality.checks.length ? `质量复核：已记录 ${quality.checks.length} 项核对提示，可在教师详情查看。` : '教师详情中有本题的质量复核记录。';
    card.append(node('div', `quality-preview${quality.duplicates.length ? ' attention' : ''}`, message));
  }
  const body = node('div', 'question-body');
  if ((Object.hasOwn(item, 'question') && !item.question) || !Object.keys(q).length) {
    const message = item.error ? text(item.error) : status === 'needs_evidence' ? '证据尚不足，暂未生成题目。请查看本轮检索记录。' : terminal.has(job.status) ? '本轮尚未形成可审核的题目，请查看任务记录。' : '正在依据选定材料准备题目，生成后将在这里展示。';
    body.append(node('p', item.error ? 'error' : 'muted', message));
    card.append(body);
    return card;
  }
  const stem = q.stem || q.case || q.case_text || q.scenario || (typeof item.question === 'string' ? item.question : '');
  if (stem) body.append(node('div', 'content-text', stem));
  const prompt = q.prompt || q.task || q.question_text || q.question;
  if (typeof prompt === 'string' && prompt !== stem) addSection(body, '作答任务', prompt);
  if (q.title && !stem && !prompt) body.append(node('div', 'content-text', q.title));
  const options = q.options || q.choices;
  if (options) {
    const list = node('ol', 'options');
    const entries = Array.isArray(options) ? options.map((option, i) => [String.fromCharCode(65 + i), option]) : Object.entries(options);
    entries.forEach(([key, value]) => {
      const option = typeof value === 'object' && value ? value.text || value.content || value.option || text(value) : text(value);
      const optionKey = typeof value === 'object' && value ? value.label || value.id || key : key;
      list.append(node('li', '', `${optionKey}. ${option}`));
    });
    body.append(list);
  }
  if (!body.childElementCount) body.append(node('div', 'content-text', text(q)));
  if (item.error) body.append(node('p', 'error', text(item.error)));
  card.append(body);
  if (!approved) card.append(node('div', 'item-notice', '该题暂未通过独立审核，请先查看审核意见，勿直接作为正式教学题使用。'));
  const answer = details('参考答案与评分点');
  addSection(answer.body, '正确选项', q.correct_options);
  addSection(answer.body, '参考答案', q.reference_answer || q.answer || item.reference_answer || item.answer);
  addSection(answer.body, '得分点', q.scoring_points || q.rubric || q.scoring_criteria || item.scoring_points);
  addSection(answer.body, '可接受的等价答案', q.equivalent_answers || q.acceptable_answers);
  addSection(answer.body, '常见误解', q.common_misconceptions || q.misconceptions);
  addSection(answer.body, '干扰项解释', q.distractor_explanations || q.option_explanations);
  addSection(answer.body, '分层提示', q.hints || q.layered_hints);
  if (!answer.body.childElementCount) answer.body.append(node('p', 'muted', '参考答案尚未生成。'));
  card.append(answer.element);
  const audit = details('教师详情 · 审核意见与证据');
  addSection(audit.body, '审核结论', review.summary || review.reason || review.verdict);
  addSection(audit.body, '审核发现', review.issues || review.findings);
  if (Array.isArray(review.evidence_gaps)) {
    addSection(audit.body, '待补充证据', review.evidence_gaps.map(gap => `查询：${gap.query || ''}\n原因：${gap.reason || ''}`));
  }
  if (plan) {
    addSection(audit.body, '本题考查目标', displayText(plan.focus));
    const material = materialLabel(plan, job);
    addSection(audit.body, '来源模板', material.id && material.title !== material.id ? `${material.title}（${material.id}）` : material.title);
    addSection(audit.body, '命题要求', displayText(plan.task_guidance));
  }
  appendQualityChecks(audit.body, item);
  appendQualityAudit(audit.body, review.quality_audit, q);
  addSection(audit.body, '独立作答', item.blind_review?.attempted_answer || item.blind_review?.answer || item.blind_review?.independent_answer || item.blind_review?.response || item.independent_answer || review.independent_answer || review.blind_answer);
  addSection(audit.body, '题干歧义检查', item.blind_review?.ambiguity_notes);
  addSection(audit.body, '教学目标', q.learning_objective || job.blueprint?.learning_objective);
  addSection(audit.body, '预期难度', `${q.difficulty || job.blueprint?.difficulty || '未标注'}（尚需学生试做验证）`);
  audit.body.append(node('h3', '', '可核查证据'));
  const snapshot = job.evidence_snapshots?.[item.evidence_snapshot_id];
  evidenceBlock(audit.body, item.evidence || q.evidence || item.sources || q.sources || snapshot?.sources || snapshot || job.evidence?.sources || job.evidence);
  const raw = node('details'); raw.append(node('summary', '', '查看此题完整记录'));
  raw.append(node('pre', '', JSON.stringify(item, null, 2))); audit.body.append(raw);
  card.append(audit.element);
  return card;
}

function renderJob(job) {
  $('#empty').hidden = true; $('#job-detail').hidden = false;
  const blueprint = job.blueprint || {};
  $('#job-title').textContent = blueprint.topic || job.topic || '出题任务';
  $('#job-meta').textContent = [populationNames[blueprint.population], deviceLabels[blueprint.device_type] || blueprint.device_type, blueprint.difficulty && `预期${blueprint.difficulty}`, localDate(job.created_at || job.created)].filter(Boolean).join(' · ');
  const items = job.questions || job.items || [];
  const hasJobErrors = !!job.error || (Array.isArray(job.errors) ? job.errors.length > 0 : !!job.errors);
  const reviewNotPassed = job.status === 'failed' && items.length > 0 && !hasJobErrors
    && items.every(item => ['revision_needed', 'rejected', 'needs_review'].includes(item.status));
  const status = $('#job-status'); status.textContent = reviewNotPassed ? '本轮审核未通过' : label(job.status);
  status.className = `status-pill${reviewNotPassed ? ' warning' : job.status === 'failed' ? ' error' : ['needs_evidence', 'partial'].includes(job.status) ? ' warning' : ''}`;
  const progress = job.progress || {};
  const total = progress.total || blueprint.count || items.length || 1;
  const done = progress.completed ?? progress.done ?? items.filter(item => ['accepted', 'approved', 'rejected', 'needs_evidence', 'failed'].includes(item.status)).length;
  const percent = typeof progress === 'number' ? progress : progress.percent ?? (terminal.has(job.status) ? 100 : Math.min(95, done / total * 100));
  $('#progress-meter').value = Math.max(0, Math.min(100, percent));
  $('#progress-label').textContent = progress.message || progress.stage || label(job.status);
  $('#progress-count').textContent = `${done} / ${total} 道`;
  showError($('#job-error'), job.error ? text(typeof job.error === 'object' ? job.error.message || job.error : job.error) : '');
  const summary = job.summary;
  const summaryBox = $('#job-summary');
  if (summary && (typeof summary !== 'object' || Object.keys(summary).length)) {
    if (typeof summary === 'string') summaryBox.textContent = summary;
    else if (summary.message) summaryBox.textContent = summary.message;
    else {
      const approved = summary.approved ?? summary.approved_count;
      const reviewNeeded = summary.needs_revision ?? summary.rejected ?? summary.rejected_count;
      summaryBox.textContent = approved !== undefined ? `AI 审核通过 ${approved} 道${reviewNeeded !== undefined ? `，仍待处理 ${reviewNeeded} 道` : ''}。完整记录可在各题详情中查看。` : text(summary);
    }
    summaryBox.hidden = false;
  } else if (items.length) {
    const approved = items.filter(item => ['accepted', 'approved', 'pass', 'accept'].includes(item.status)).length;
    summaryBox.textContent = `AI 审核通过 ${approved} 道，待处理 ${items.length - approved} 道。完整记录可在各题详情中查看。`;
    summaryBox.hidden = false;
  } else summaryBox.hidden = true;
  renderBatchPlan(job);
  const evidencePanel = $('#job-evidence');
  const evidenceBody = $('#job-evidence-body');
  const sources = job.evidence?.sources || [];
  evidencePanel.hidden = !sources.length && !job.errors?.length && !job.evidence?.events?.length;
  evidenceBody.replaceChildren();
  if (!evidencePanel.hidden) {
    const retrieval = job.retrieval_state || job.evidence || {};
    if (typeof retrieval.local_success === 'boolean' || typeof retrieval.web_success === 'boolean') {
      addSection(evidenceBody, '取证结果', `本地证据：${retrieval.local_success ? '已取得' : '尚不足'}；网页证据：${retrieval.web_success ? '已取得' : '尚不足'}。`);
    }
    addSection(evidenceBody, '任务记录', job.errors);
    if (sources.length) evidenceBlock(evidenceBody, sources);
    if (job.evidence?.events?.length) {
      const record = node('details'); record.append(node('summary', '', '查看检索过程记录'));
      record.append(node('pre', '', text(job.evidence.events))); evidenceBody.append(record);
    }
  }
  const signature = JSON.stringify([items, job.evidence, job.evidence_snapshots, blueprint, job.batch_plan]);
  if (signature !== questionSignature) {
    const opened = new Set(Array.from($('#questions').querySelectorAll('article')).flatMap(card => Array.from(card.querySelectorAll(':scope > details')).map((detail, i) => detail.open ? `${card.dataset.itemId}:${i}` : null)).filter(Boolean));
    $('#questions').replaceChildren(...items.map((item, index) => renderQuestion(item, index, job)));
    $('#questions').querySelectorAll('article').forEach(card => card.querySelectorAll(':scope > details').forEach((detail, i) => { detail.open = opened.has(`${card.dataset.itemId}:${i}`); }));
    questionSignature = signature;
  }
  $('#export').href = `${API}/jobs/${encodeURIComponent(job.job_id)}/export`;
  document.querySelectorAll('.job-row').forEach(button => button.classList.toggle('active', button.dataset.jobId === job.job_id));
}

async function pollJob(id) {
  clearTimeout(pollTimer);
  if (id !== currentId) return;
  try {
    const job = await request(`/jobs/${encodeURIComponent(id)}`);
    if (id !== currentId) return;
    pollFailures = 0; renderJob(job);
    if (terminal.has(job.status)) { await loadJobs(); return; }
    pollTimer = setTimeout(() => pollJob(id), document.hidden ? 6000 : 2500);
  } catch (error) {
    if (id !== currentId) return;
    pollFailures += 1;
    showError($('#job-error'), `${error.message} 页面会继续尝试读取任务进度。`);
    pollTimer = setTimeout(() => pollJob(id), Math.min(20000, pollFailures * 4000));
  }
}

async function selectJob(id) {
  currentId = id; questionSignature = ''; pollFailures = 0;
  try { sessionStorage.setItem('mdrpi.question-studio.current', id); } catch (_) { /* Optional navigation state. */ }
  $('#empty').hidden = true; $('#job-detail').hidden = false;
  $('#job-title').textContent = '正在读取任务…';
  $('#questions').replaceChildren();
  $('#batch-plan').hidden = true;
  batchPlanSignature = '';
  showError($('#job-error'), '');
  await pollJob(id);
}

$('#blueprint').addEventListener('submit', async event => {
  event.preventDefault();
  if (posting) return;
  showError($('#form-error'), '');
  const form = event.currentTarget;
  if (!form.reportValidity()) return;
  const data = new FormData(form);
  const materials = data.getAll('material_ids');
  if (!materials.length) { showError($('#form-error'), '请至少选择一份已有材料。'); $('#materials').scrollIntoView({behavior: 'auto', block: 'center'}); return; }
  if (materials.length > 5) { showError($('#form-error'), '每次最多选择 5 份材料，请保留与当前目标最相关的材料。'); return; }
  const body = Object.fromEntries(data.entries()); body.material_ids = materials; body.count = Number(body.count);
  posting = true;
  const button = $('#generate'); button.disabled = true;
  button.firstElementChild.textContent = '正在提交任务…';
  try {
    const job = await request('/jobs', {method: 'POST', body: JSON.stringify(body)});
    await loadJobs(); await selectJob(job.job_id);
    if (matchMedia('(max-width: 760px)').matches) $('#job-detail').scrollIntoView({behavior: 'auto', block: 'start'});
  } catch (error) { showError($('#form-error'), error.message); }
  finally { posting = false; button.disabled = !$('#materials input'); button.firstElementChild.textContent = '生成并审核题目'; }
});

$('#refresh-jobs').addEventListener('click', loadJobs);
$('[name=population]').addEventListener('change', populateDevices);
$('[name=device_type]').addEventListener('change', renderMaterials);
window.addEventListener('pagehide', () => clearTimeout(pollTimer));

await Promise.allSettled([loadMaterials(), loadJobs()]);
try {
  const remembered = sessionStorage.getItem('mdrpi.question-studio.current');
  if (!currentId && remembered && /^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(remembered)) await selectJob(remembered);
} catch (_) { /* State restoration is optional. */ }
