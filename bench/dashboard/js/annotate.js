// ============================================================
// Annotation Questionnaire — Compact + Bilingual
// ============================================================

const annState = {
  annotator: null,
  profiles: [],
  dimensions: {},
  progress: {},
  groundTruth: {},   // pid → {A:'L', B:'M', ...}
  currentProfile: null,
  traceCache: {},
  lang: 'en',        // 'en' | 'zh'
};

// ── i18n ────────────────────────────────────────────────────
const I18N = {
  en: {
    annotator_id: 'Annotator ID:', enter: 'Enter', profiles: 'Profiles',
    select_profile: 'Select a profile from the sidebar',
    annotation: 'Annotation', select_lmr: 'Select L / M / R for each dimension',
    submit_next: 'Submit & Next →', notes_placeholder: 'Notes (optional)',
    results: 'Results', annotation_results: 'Annotation Results', close: 'Close',
    loading: 'Loading...', no_traces: 'No trajectories found.',
    confirm_partial: 'Not all dimensions filled. Save anyway?',
    all_done: 'All profiles annotated! View Results for summary.',
    confidence: 'Conf', events: 'events',
    feedback_title: 'Result', correct: 'Correct', incorrect: 'Incorrect',
    your_answer: 'You', truth: 'Truth', score: 'Score', next_profile: 'Next →',
    per_annotator: 'Per-Annotator Accuracy', confusion: 'Confusion Matrices',
    inter_annotator: 'Inter-Annotator Agreement',
    // task types
    understand: 'understand', create: 'create', organize: 'organize',
    iterate: 'iterate', synthesize: 'synthesize', maintain: 'maintain',
    // dimensions
    dim_A: 'Consumption', dim_B: 'Production', dim_C: 'Organization',
    dim_D: 'Iteration', dim_E: 'Rhythm', dim_F: 'Cross-Modal',
    q_A: 'How does this user explore/read files?',
    q_B: 'How does this user produce output?',
    q_C: 'How does this user organize files?',
    q_D: 'How does this user refine existing work?',
    q_E: "What is this user's work rhythm?",
    q_F: 'Does this user use visual materials?',
    // L/M/R labels
    A_L: 'Sequential', A_M: 'Targeted', A_R: 'Breadth-first',
    B_L: 'Comprehensive', B_M: 'Balanced', B_R: 'Minimal',
    C_L: 'Deep Nested', C_M: 'Adaptive', C_R: 'Flat',
    D_L: 'Incremental', D_M: 'Balanced', D_R: 'Rewrite',
    E_L: 'Phased', E_M: 'Steady', E_R: 'Bursty',
    F_L: 'Visual-heavy', F_M: 'Balanced', F_R: 'Text-only',
    // L/M/R short descriptions
    Ad_L: 'Reads one-by-one, full content, revisits', Ad_M: 'Searches first, reads matched sections', Ad_R: 'Scans broadly, reads first lines only',
    Bd_L: 'Multi-heading, tables, 200+ lines, 3+ files', Bd_M: '2 headings, 80-150 lines, 1-2 files', Bd_R: 'Flat bullets, one file, <60 lines',
    Cd_L: '3+ level dirs, prefixed names, backups', Cd_M: '1-2 levels when needed, mixed style', Cd_R: 'All in root, no dirs, overwrites',
    Dd_L: 'Many small edits, reviews, backups', Dd_M: 'Moderate edits, occasional review', Dd_R: 'Overwrites entire files, one pass',
    Ed_L: 'Read→plan→write→review phases', Ed_M: 'Interleaves read/write naturally', Ed_R: 'Rapid bursts, frequent switches',
    Fd_L: 'Charts/diagrams, figures/, captions', Fd_M: 'Markdown tables, structured format', Fd_R: 'Pure text only, no tables/images',
  },
  zh: {
    annotator_id: '标注者 ID:', enter: '进入', profiles: '用户档案',
    select_profile: '从左侧选择一个 Profile 开始标注',
    annotation: '维度标注', select_lmr: '根据行为轨迹为每个维度选择 L / M / R',
    submit_next: '提交并下一个 →', notes_placeholder: '备注（可选）',
    results: '结果', annotation_results: '标注结果', close: '关闭',
    loading: '加载中...', no_traces: '未找到该 Profile 的轨迹数据。',
    confirm_partial: '有维度未选择，仍然保存？',
    all_done: '全部标注完成！点击 Results 查看汇总。',
    confidence: '置信', events: '事件',
    feedback_title: '结果', correct: '正确', incorrect: '错误',
    your_answer: '你的答案', truth: '正确答案', score: '得分', next_profile: '下一个 →',
    per_annotator: '标注者准确率', confusion: '混淆矩阵',
    inter_annotator: '标注者间一致性',
    understand: '理解', create: '创作', organize: '组织',
    iterate: '迭代', synthesize: '综合', maintain: '维护',
    dim_A: '消费模式', dim_B: '生产风格', dim_C: '组织偏好',
    dim_D: '迭代策略', dim_E: '工作节奏', dim_F: '跨模态',
    q_A: '用户如何探索和阅读文件？',
    q_B: '用户如何产出内容？',
    q_C: '用户如何组织文件和目录？',
    q_D: '用户如何修改和完善已有工作？',
    q_E: '用户的工作节奏是怎样的？',
    q_F: '用户是否使用视觉材料？',
    A_L: '顺序深读', A_M: '搜索定位', A_R: '广度浏览',
    B_L: '详尽全面', B_M: '均衡适度', B_R: '极简精炼',
    C_L: '深层嵌套', C_M: '按需调整', C_R: '完全扁平',
    D_L: '增量打磨', D_M: '适度迭代', D_R: '大幅重写',
    E_L: '分阶段', E_M: '稳定节奏', E_R: '突发式',
    F_L: '重视视觉', F_M: '均衡使用', F_R: '纯文本',
    Ad_L: '逐个读、全文、反复回看', Ad_M: '先搜索再读匹配段落', Ad_R: '广泛扫描、只读开头',
    Bd_L: '多级标题、表格、200+行、3+文件', Bd_M: '2级标题、80-150行、1-2文件', Bd_R: '扁平列表、单文件、<60行',
    Cd_L: '3+层目录、前缀命名、备份', Cd_M: '需要时建1-2层、混合风格', Cd_R: '全部放根目录、直接覆盖',
    Dd_L: '多次小改、逐次审查、备份', Dd_M: '适度编辑、偶尔审查', Dd_R: '整文件覆盖、一次到位',
    Ed_L: '先全读→计划→写→审查', Ed_M: '读写自然交替', Ed_R: '快速爆发、频繁切换',
    Fd_L: '生成图表、figures/目录、图注', Fd_M: 'Markdown 表格、结构化格式', Fd_R: '纯文字、无表格/图片',
  },
};

function t(key) { return I18N[annState.lang][key] || I18N.en[key] || key; }

function annToggleLang() {
  annState.lang = annState.lang === 'en' ? 'zh' : 'en';
  document.getElementById('annLangBtn').textContent = annState.lang === 'en' ? '中文' : 'EN';
  applyI18n();
  if (annState.dimensions && Object.keys(annState.dimensions).length) renderDimensionPanel();
  if (annState.currentProfile && annState.traceCache[annState.currentProfile]) {
    renderTraces(annState.traceCache[annState.currentProfile]);
  }
}

function applyI18n() {
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    if (I18N[annState.lang][key]) el.textContent = I18N[annState.lang][key];
  });
  document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
    const key = el.dataset['i18nPlaceholder'];
    if (I18N[annState.lang][key]) el.placeholder = I18N[annState.lang][key];
  });
}

// ── Event rendering helpers ─────────────────────────────────
const EVT_COLORS = {
  file_read: '#22d3ee', file_write: '#4ade80', file_edit: '#fb923c',
  dir_create: '#fbbf24', file_rename: '#a78bfa', file_move: '#a78bfa',
  file_delete: '#f87171', file_copy: '#a78bfa', file_browse: '#6c8aff',
  file_search: '#6c8aff', context_switch: '#8b8fa3', cross_file_reference: '#8b8fa3',
};
const EVT_SHORT = {
  file_read: 'READ', file_write: 'WRITE', file_edit: 'EDIT',
  dir_create: 'MKDIR', file_move: 'MOVE', file_rename: 'REN',
  file_delete: 'DEL', file_copy: 'COPY', file_browse: 'BROWSE',
  file_search: 'SEARCH', context_switch: 'CTX', cross_file_reference: 'XREF',
};

// ── Start ───────────────────────────────────────────────────
function annStart() {
  const id = document.getElementById('annotatorInput').value.trim();
  if (!id) return;
  annState.annotator = id;
  fetch(`/api/annotate/init?annotator=${encodeURIComponent(id)}`)
    .then(r => r.json())
    .then(data => {
      annState.profiles = data.profiles;
      annState.dimensions = data.dimensions;
      annState.progress = data.progress || {};
      annState.groundTruth = data.ground_truth || {};
      // Hide input, show user label
      document.getElementById('annotatorInput').style.display = 'none';
      document.querySelector('#annLoginBar .ann-btn-primary').style.display = 'none';
      const uLabel = document.getElementById('annUserLabel');
      uLabel.textContent = id;
      uLabel.style.display = '';
      const pLabel = document.getElementById('annProgressLabel');
      pLabel.style.display = '';
      document.getElementById('annWorkspace').style.display = 'block';
      renderProfileList();
      renderDimensionPanel();
      updateProgressLabel();
      applyI18n();
    })
    .catch(err => alert('Failed: ' + err.message));
}

// ── Profile sidebar ─────────────────────────────────────────
function renderProfileList() {
  const el = document.getElementById('annProfileList');
  el.innerHTML = '';
  annState.profiles.forEach(p => {
    const done = annState.progress[p.profile_id];
    const active = annState.currentProfile === p.profile_id;
    const div = document.createElement('div');
    div.className = 'ann-profile-item' + (active ? ' active' : '') + (done ? ' done' : '');
    div.innerHTML = `<span class="ann-profile-num">#${p.index}</span><span class="ann-profile-status">${done ? '✓' : active ? '●' : ''}</span>`;
    div.onclick = () => selectProfile(p.profile_id);
    el.appendChild(div);
  });
}

function updateProgressLabel() {
  const done = Object.keys(annState.progress).length;
  document.getElementById('annProgressLabel').textContent = `${done}/${annState.profiles.length}`;
}

// ── Select profile ──────────────────────────────────────────
function selectProfile(pid) {
  annState.currentProfile = pid;
  renderProfileList();
  document.getElementById('annFormWrap').style.display = '';
  annState.progress[pid] ? loadSavedAnnotation(annState.progress[pid]) : clearAnnotationPanel();

  const grid = document.getElementById('annTraces');
  if (annState.traceCache[pid]) {
    renderTraces(annState.traceCache[pid]);
  } else {
    grid.innerHTML = `<div class="card" style="grid-column:1/-1;text-align:center;padding:30px;color:var(--text2)">${t('loading')}</div>`;
    fetch(`/api/annotate/traces/${encodeURIComponent(pid)}`)
      .then(r => r.json())
      .then(data => { annState.traceCache[pid] = data; if (annState.currentProfile === pid) renderTraces(data); })
      .catch(() => { grid.innerHTML = '<div class="card" style="grid-column:1/-1;color:var(--red)">Error loading traces</div>'; });
  }
}

// ── Render trace grid (3 cols, compact cards) ───────────────
function renderTraces(data) {
  const grid = document.getElementById('annTraces');
  if (!data.traces || !data.traces.length) {
    grid.innerHTML = `<div class="card" style="grid-column:1/-1;text-align:center;padding:30px;color:var(--text2)">${t('no_traces')}</div>`;
    return;
  }
  grid.innerHTML = data.traces.map(tr => renderTraceCard(tr)).join('');
}

function renderTraceCard(tr) {
  const dimTags = (tr.activated_dimensions || []).map(d => `<span class="ann-dim-tag">${d}</span>`).join('');
  const typeName = t(tr.task_type);
  const s = tr.stats;

  // Stats badges — only non-zero
  const badges = [];
  if (s.reads)        badges.push(`${s.reads} read`);
  if (s.writes)       badges.push(`${s.writes} write`);
  if (s.edits)        badges.push(`${s.edits} edit`);
  if (s.dirs_created) badges.push(`${s.dirs_created} dir`);
  if (s.searches)     badges.push(`${s.searches} search`);
  if (s.browses)      badges.push(`${s.browses} browse`);
  if (s.cross_refs)   badges.push(`${s.cross_refs} xref`);

  // Event-type distribution bar
  const evtBar = buildEvtBar(tr.events);

  // All events — scrollable inside card
  const keyEvts = renderAllEvents(tr.events);

  // Hints for activated dimensions
  const hints = (tr.activated_dimensions || [])
    .filter(d => tr.hints[d])
    .map(d => `<b>${d}</b> ${esc(tr.hints[d])}`)
    .join(' · ');

  return `<div class="ann-trace-card">
    <div class="ann-trace-header">
      <span class="ann-trace-id">${tr.task_id}</span>
      <span class="ann-type-badge ann-type-${tr.task_type}">${typeName}</span>
      ${dimTags}
    </div>
    <div class="ann-trace-name" title="${esc(tr.task_name)}">${esc(tr.task_name)}</div>
    <div class="ann-stats">${badges.map(b => `<span class="ann-stat-badge">${b}</span>`).join('')}<span class="ann-stat-badge" style="color:var(--text)">${s.total_events} ${t('events')}</span></div>
    <div class="ann-evt-bar">${evtBar}</div>
    <div class="ann-key-evts">${keyEvts}</div>
    ${hints ? `<div class="ann-hint-line">${hints}</div>` : ''}
  </div>`;
}

function buildEvtBar(events) {
  if (!events || !events.length) return '';
  const counts = {};
  events.forEach(e => { const et = e.event_type; counts[et] = (counts[et] || 0) + 1; });
  const total = events.length;
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([et, n]) => {
      const pct = Math.max(n / total * 100, 2);
      const color = EVT_COLORS[et] || '#8b8fa3';
      return `<div class="ann-evt-bar-seg" style="width:${pct}%;background:${color}" title="${EVT_SHORT[et] || et}: ${n}"></div>`;
    }).join('');
}

function renderAllEvents(events) {
  if (!events || !events.length) return '';
  return events.map(e => {
    const et = e.event_type;
    const color = EVT_COLORS[et] || '#8b8fa3';
    const label = EVT_SHORT[et] || et;
    return `<div class="ann-key-evt"><span class="ann-key-evt-type" style="color:${color}">${label}</span> <span>${esc(fmtEvt(e))}</span></div>`;
  }).join('');
}

function fmtEvt(e) {
  const fp = shortPath(e.file_path || e.dir_path || '');
  switch (e.event_type) {
    case 'file_read': return `${fp}${e.view_range ? ' ['+e.view_range[0]+'-'+e.view_range[1]+']':''}${e.content_length ? ' '+e.content_length+'ch':''}`;
    case 'file_write': return `${e.operation==='overwrite'?'Overwrite':'Create'}: ${fp}${e.content_length?' ('+e.content_length+'ch)':''}`;
    case 'file_edit': return `${fp} (+${e.lines_added||0}/-${e.lines_deleted||0})`;
    case 'dir_create': return fp + '/';
    case 'file_move': case 'file_rename': return `${shortPath(e.old_path||'')} → ${shortPath(e.new_path||'')}`;
    case 'file_delete': return fp;
    case 'file_copy': return `${shortPath(e.source_path||'')} → ${shortPath(e.dest_path||'')}`;
    case 'file_browse': return `${fp} (${Array.isArray(e.files_listed)?e.files_listed.length:'?'} files)`;
    case 'file_search': return `${e.search_type||'search'}: "${e.query||''}"${e.files_matched!=null?', '+e.files_matched+' matches':''}`;
    case 'context_switch': return `${shortPath(e.from_file||'')} → ${shortPath(e.to_file||'')}`;
    case 'cross_file_reference': return `${shortPath(e.source_file||'')} → ${shortPath(e.target_file||'')}`;
    default: return fp;
  }
}
function shortPath(p) { return (p || '').split('/').filter(Boolean).slice(-2).join('/'); }

// ── Dimension panel (3x2 grid with pill radios) ─────────────
function renderDimensionPanel() {
  const el = document.getElementById('annDimensions');
  const DIMS = ['A','B','C','D','E','F'];
  el.innerHTML = DIMS.map(d => {
    const name = t('dim_' + d);
    const question = t('q_' + d);
    return `<div class="ann-dim-card" id="dim-card-${d}">
      <div class="ann-dim-top"><span class="ann-dim-key">${d}</span> <span class="ann-dim-name">${name}</span></div>
      <div class="ann-dim-q">${question}</div>
      <div class="ann-pills">
        ${['L','M','R'].map(v => `<label class="ann-pill" id="pill-${d}-${v}" onclick="onPill('${d}','${v}')">
          <input type="radio" name="dim-${d}" value="${v}">
          <span class="ann-pill-val">${v}</span>
          <span class="ann-pill-label">${t(d+'_'+v)}</span>
          <span class="ann-pill-desc">${t(d+'d_'+v)}</span>
        </label>`).join('')}
      </div>
      <div class="ann-conf">
        <span class="ann-conf-lbl">${t('confidence')}:</span>
        ${[1,2,3].map(c => `<label class="ann-conf-dot" id="conf-${d}-${c}" onclick="onConf('${d}',${c})"><input type="radio" name="conf-${d}" value="${c}">${c}</label>`).join('')}
      </div>
    </div>`;
  }).join('');
}

function onPill(dim, val) {
  document.querySelectorAll(`#dim-card-${dim} .ann-pill`).forEach(p => p.classList.remove('selected'));
  const pill = document.getElementById(`pill-${dim}-${val}`);
  if (pill) { pill.classList.add('selected'); pill.querySelector('input').checked = true; }
}

function onConf(dim, val) {
  document.querySelectorAll(`#dim-card-${dim} .ann-conf-dot`).forEach(d => d.classList.remove('selected'));
  const dot = document.getElementById(`conf-${dim}-${val}`);
  if (dot) { dot.classList.add('selected'); dot.querySelector('input').checked = true; }
}

function clearAnnotationPanel() {
  document.querySelectorAll('#annDimensions input[type="radio"]').forEach(r => r.checked = false);
  document.querySelectorAll('#annDimensions .ann-pill, #annDimensions .ann-conf-dot').forEach(el => el.classList.remove('selected'));
  document.getElementById('annNotes').value = '';
}

function loadSavedAnnotation(saved) {
  clearAnnotationPanel();
  const dims = saved.dimensions || {};
  Object.entries(dims).forEach(([d, v]) => {
    if (v.value) onPill(d, v.value);
    if (v.confidence) onConf(d, v.confidence);
  });
  document.getElementById('annNotes').value = saved.notes || '';
}

// ── Submit ──────────────────────────────────────────────────
function annSubmit() {
  if (!annState.currentProfile) return;
  const dims = {};
  let allFilled = true;
  ['A','B','C','D','E','F'].forEach(d => {
    const v = document.querySelector(`input[name="dim-${d}"]:checked`);
    const c = document.querySelector(`input[name="conf-${d}"]:checked`);
    if (!v) allFilled = false;
    dims[d] = { value: v ? v.value : null, confidence: c ? parseInt(c.value) : null };
  });
  if (!allFilled && !confirm(t('confirm_partial'))) return;

  const payload = {
    annotator: annState.annotator,
    profile_id: annState.currentProfile,
    dimensions: dims,
    notes: document.getElementById('annNotes').value,
    timestamp: new Date().toISOString(),
  };
  fetch('/api/annotate/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  }).then(r => r.json()).then(() => {
    annState.progress[annState.currentProfile] = payload;
    updateProgressLabel();
    renderProfileList();
    // Fetch ground truth on-demand to avoid caching issues
    fetch(`/api/annotate/truth/${encodeURIComponent(annState.currentProfile)}`)
      .then(r => r.json())
      .then(gt => showFeedback(dims, gt))
      .catch(() => showFeedback(dims, annState.groundTruth[annState.currentProfile] || {}));
  }).catch(err => alert('Save failed: ' + err.message));
}

// ── Instant feedback overlay ────────────────────────────────
function showFeedback(submitted, gt) {
  const DIMS = ['A','B','C','D','E','F'];
  let correct = 0;
  const rows = DIMS.map(d => {
    const yours = submitted[d]?.value || '—';
    const truth = gt[d] || '?';
    const match = yours === truth;
    if (match) correct++;
    const dimName = t('dim_' + d);
    const yoursLabel = yours !== '—' ? t(d + '_' + yours) : '—';
    const truthLabel = truth !== '?' ? t(d + '_' + truth) : '?';
    return `<tr>
      <td style="font-weight:700;color:var(--accent);width:28px;">${d}</td>
      <td style="font-size:12px;color:var(--text2);width:80px;">${dimName}</td>
      <td style="text-align:center;font-weight:700;">${yours}</td>
      <td style="text-align:center;font-size:11px;">${yoursLabel}</td>
      <td style="text-align:center;font-weight:700;">${truth}</td>
      <td style="text-align:center;font-size:11px;">${truthLabel}</td>
      <td style="text-align:center;font-size:16px;">${match ? '<span style="color:var(--green)">&#10003;</span>' : '<span style="color:var(--red)">&#10007;</span>'}</td>
    </tr>`;
  }).join('');

  const pct = Math.round(correct / DIMS.length * 100);
  const color = pct >= 80 ? 'var(--green)' : pct >= 50 ? 'var(--yellow)' : 'var(--red)';

  // Find profile display index
  const prof = annState.profiles.find(p => p.profile_id === annState.currentProfile);
  const label = prof ? `#${prof.index}` : annState.currentProfile;

  const overlay = document.createElement('div');
  overlay.className = 'ann-modal';
  overlay.id = 'feedbackOverlay';
  overlay.innerHTML = `<div class="ann-modal-content" style="max-width:520px;padding:20px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
      <h2 style="font-size:16px;margin:0;">${t('feedback_title')} — ${label}</h2>
      <span style="font-size:22px;font-weight:800;color:${color}">${correct}/${DIMS.length}</span>
    </div>
    <table style="width:100%;font-size:13px;border-collapse:collapse;">
      <thead><tr>
        <th></th><th></th>
        <th colspan="2" style="text-align:center;font-size:11px;color:var(--text2);padding-bottom:6px;">${t('your_answer')}</th>
        <th colspan="2" style="text-align:center;font-size:11px;color:var(--text2);padding-bottom:6px;">${t('truth')}</th>
        <th></th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div style="text-align:right;margin-top:16px;">
      <button class="ann-btn ann-btn-primary" onclick="closeFeedbackAndNext()" style="padding:8px 24px;">${t('next_profile')}</button>
    </div>
  </div>`;
  document.body.appendChild(overlay);
}

function closeFeedbackAndNext() {
  const el = document.getElementById('feedbackOverlay');
  if (el) el.remove();
  const next = annState.profiles.find(p => !annState.progress[p.profile_id]);
  if (next) selectProfile(next.profile_id);
  else alert(t('all_done'));
}

// ── Results ─────────────────────────────────────────────────
function showResults() {
  const modal = document.getElementById('resultsModal');
  const content = document.getElementById('resultsContent');
  modal.style.display = 'flex';
  content.innerHTML = `<p style="color:var(--text2)">${t('loading')}</p>`;
  fetch('/api/annotate/results').then(r => r.json()).then(data => {
    if (data.error) { content.innerHTML = `<p style="color:var(--text2)">${data.error}</p>`; return; }
    renderResults(data);
  }).catch(err => { content.innerHTML = `<p style="color:var(--red)">${err.message}</p>`; });
}

function renderResults(data) {
  const content = document.getElementById('resultsContent');
  let h = '';
  // Accuracy table
  h += `<h3 style="margin-bottom:8px;">${t('per_annotator')}</h3>`;
  h += '<table class="result-table"><thead><tr><th>Annotator</th><th>#</th><th>A</th><th>B</th><th>C</th><th>D</th><th>E</th><th>F</th><th>Avg</th></tr></thead><tbody>';
  Object.entries(data.annotators).forEach(([name, res]) => {
    h += `<tr><td style="font-weight:600">${esc(name)}</td><td>${res.profiles_annotated}</td>`;
    ['A','B','C','D','E','F'].forEach(d => {
      const pct = Math.round(res.per_dimension[d].accuracy * 100);
      h += `<td style="text-align:center;color:${pct>=60?'var(--green)':pct>=40?'var(--yellow)':'var(--red)'};font-weight:600">${pct}%</td>`;
    });
    const oPct = Math.round(res.overall_accuracy * 100);
    h += `<td style="text-align:center;font-weight:700;color:${oPct>=60?'var(--green)':oPct>=40?'var(--yellow)':'var(--red)'}">${oPct}%</td></tr>`;
  });
  h += '</tbody></table>';
  // Confusion
  h += `<h3 style="margin:20px 0 8px;">${t('confusion')}</h3><div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">`;
  ['A','B','C','D','E','F'].forEach(d => {
    const m = data.confusion[d]; if (!m) return;
    h += `<div class="card" style="padding:10px;"><h4 style="font-size:12px;margin-bottom:6px;">${d}: ${t('dim_'+d)}</h4>`;
    h += '<table style="font-size:11px;width:100%;"><thead><tr><th></th><th>→L</th><th>→M</th><th>→R</th></tr></thead><tbody>';
    ['L','M','R'].forEach(gt => {
      h += `<tr><td style="font-weight:600">${gt}</td>`;
      ['L','M','R'].forEach(pr => {
        const v = m[gt][pr]; const bg = gt===pr ? 'rgba(74,222,128,.15)' : v>0 ? 'rgba(248,113,113,.1)' : '';
        h += `<td style="text-align:center;background:${bg}">${v}</td>`;
      });
      h += '</tr>';
    });
    h += '</tbody></table></div>';
  });
  h += '</div>';
  // Inter-annotator
  if (data.inter_annotator && data.inter_annotator.length) {
    h += `<h3 style="margin:20px 0 8px;">${t('inter_annotator')}</h3>`;
    h += '<table class="result-table"><thead><tr><th>Pair</th><th>Agree</th><th>κ</th><th>N</th></tr></thead><tbody>';
    data.inter_annotator.forEach(p => {
      const kc = p.kappa>=0.4?'var(--green)':p.kappa>=0.2?'var(--yellow)':'var(--red)';
      h += `<tr><td>${esc(p.annotators.join(' vs '))}</td><td style="text-align:center">${Math.round(p.agreement*100)}%</td><td style="text-align:center;color:${kc};font-weight:600">${p.kappa.toFixed(3)}</td><td style="text-align:center">${p.n}</td></tr>`;
    });
    h += '</tbody></table>';
  }
  content.innerHTML = h;
}

// ── Utilities ───────────────────────────────────────────────
function esc(s) { return s ? String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') : ''; }

// Enter key to start
document.getElementById('annotatorInput').addEventListener('keydown', e => { if (e.key === 'Enter') annStart(); });
