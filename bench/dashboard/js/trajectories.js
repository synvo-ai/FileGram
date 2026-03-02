// ============================================================
// Trajectory Viewer
// ============================================================
const trajState = {
  index: null, // {trajectories: [...], tasks: {...}}
  selectedProfile: null,
  compareProfile: null,
  selectedTask: null,
  data: null,      // current trajectory data
  compareData: null,
  currentStep: 0,
  playing: false,
  playTimer: null,
  cache: {},       // cache[profile_task] = data
  fileContentCache: {},
  mediaBlobCache: {},
  profileCache: {},  // cache[profile_name] = profile data
};

const FILE_ICONS = {
  md: '\u{1F4C4}', txt: '\u{1F4C4}', eml: '\u{1F4E7}', csv: '\u{1F4CA}',
  json: '\u{1F4C4}', yaml: '\u{1F4C4}', yml: '\u{1F4C4}',
  pdf: '\u{1F4D5}', docx: '\u{1F4D1}', doc: '\u{1F4D1}',
  jpg: '\u{1F5BC}', jpeg: '\u{1F5BC}', png: '\u{1F5BC}', gif: '\u{1F5BC}',
  ics: '\u{1F4C5}', log: '\u{1F4C4}',
};
const DIR_ICON = '\u{1F4C1}';
const TEXT_EXTS = new Set(['md','txt','eml','csv','ics','json','yaml','yml','log']);
const IMAGE_EXTS = new Set(['png','jpg','jpeg','gif','webp','svg','bmp']);
const PREVIEW_EXTS = new Set(['pdf']);
const EVENT_LABELS = {
  file_read: 'READ', file_write: 'WRITE', file_edit: 'EDIT',
  dir_create: 'MKDIR', file_rename: 'RENAME', file_move: 'MOVE',
  file_delete: 'DELETE', file_copy: 'COPY', file_browse: 'BROWSE',
  file_search: 'SEARCH', iteration_start: 'TURN START', iteration_end: 'TURN END',
  context_switch: 'CTX SWITCH', error_encounter: 'ERROR', error_response: 'RECOVERY',
  fs_snapshot: 'SNAPSHOT', session_start: 'SESSION START', session_end: 'SESSION END',
  cross_file_reference: 'FILE REF',
};

// --- Memory Model Input: narrative builder (mirrors base.py events_to_narrative) ---
const NARRATIVE_SKIP = new Set([
  'tool_call','llm_response','compaction_triggered',
  'session_start','session_end','iteration_start','iteration_end',
  'error_encounter','error_response'
]);
function formatNarrativeLine(evt, idx) {
  const et = evt.event_type;
  const fp = (evt.file_path || evt.dir_path || '').split('/').filter(Boolean);
  const short = fp.length > 0 ? fp.join('/') : '';
  let cls = 'mem-type-other';
  let line = '';
  switch (et) {
    case 'file_read': {
      cls = 'mem-type-read';
      const vr = evt.view_range ? ` (lines [${evt.view_range[0]}, ${evt.view_range[1]}]` : ' (';
      const vc = evt.view_count ? `, view #${evt.view_count}` : '';
      const cl = evt.content_length ? `, ${evt.content_length} chars` : '';
      line = `Read file: ${short}${vr}${vc}${cl})`;
      break;
    }
    case 'file_write': {
      cls = 'mem-type-write';
      const op = evt.operation || 'create';
      const cl = evt.content_length ? ` (${evt.content_length} chars)` : '';
      line = `${op === 'create' ? 'Create' : 'Overwrite'} file: ${short}${cl}`;
      break;
    }
    case 'file_edit': {
      cls = 'mem-type-edit';
      const la = evt.lines_added || 0, ld = evt.lines_deleted || 0;
      line = `Edit file: ${short} (+${la}/-${ld} lines)`;
      break;
    }
    case 'dir_create': {
      cls = 'mem-type-dir';
      const depth = evt.directory_depth != null ? `, depth=${evt.directory_depth}` : '';
      line = `Create directory: ${short}/${depth}`;
      break;
    }
    case 'file_rename': {
      cls = 'mem-type-move';
      const op = (evt.old_path||'').split('/').filter(Boolean).join('/');
      const np = (evt.new_path||'').split('/').filter(Boolean).join('/');
      line = `Rename: ${op} -> ${np}`;
      break;
    }
    case 'file_move': {
      cls = 'mem-type-move';
      const op = (evt.old_path||'').split('/').filter(Boolean).join('/');
      const np = (evt.new_path||'').split('/').filter(Boolean).join('/');
      line = `Move: ${op} -> ${np}`;
      break;
    }
    case 'file_delete': {
      cls = 'mem-type-del';
      line = `Delete file: ${short}`;
      break;
    }
    case 'file_copy': {
      cls = 'mem-type-move';
      const sp = (evt.source_path||'').split('/').pop();
      const dp = (evt.dest_path||'').split('/').pop();
      const bk = evt.is_backup ? ' (backup)' : '';
      line = `Copy: ${sp} -> ${dp}${bk}`;
      break;
    }
    case 'file_browse': {
      cls = 'mem-type-search';
      const cnt = Array.isArray(evt.files_listed) ? evt.files_listed.length : '?';
      const dp = evt.depth != null ? `, depth=${evt.depth}` : '';
      line = `Browse directory: ${short} (${cnt} files${dp})`;
      break;
    }
    case 'file_search': {
      cls = 'mem-type-search';
      const st = evt.search_type || 'search';
      const q = evt.query || '';
      const m = evt.files_matched != null ? `, ${evt.files_matched} matches` : '';
      line = `Search (${st}): query='${q}'${m}`;
      break;
    }
    case 'fs_snapshot': {
      cls = 'mem-type-other';
      const fc = evt.file_count || '?';
      const md = evt.max_depth != null ? `, max_depth=${evt.max_depth}` : '';
      line = `FS Snapshot: ${fc} files${md}`;
      break;
    }
    case 'context_switch': {
      cls = 'mem-type-other';
      const ff = (evt.from_file||'').split('/').pop();
      const tf = (evt.to_file||'').split('/').pop();
      line = `Context switch: ${ff} -> ${tf}`;
      break;
    }
    case 'cross_file_reference': {
      cls = 'mem-type-other';
      const sf = (evt.source_file||'').split('/').pop();
      const tf = (evt.target_file||'').split('/').pop();
      const rt = evt.reference_type ? ` (${evt.reference_type})` : '';
      line = `Cross-file ref: ${sf} -> ${tf}${rt}`;
      break;
    }
    default:
      line = `${et}: ${short}`;
  }
  return {cls, line};
}

function renderMemoryInput() {
  const d = trajState.data;
  const contentEl = document.getElementById('trajMemoryContent');
  const metaEl = document.getElementById('trajMemoryMeta');
  if (!d || !contentEl) return;
  const events = d.events || [];
  const step = trajState.currentStep;

  // Filter and format events up to current step
  let narrativeIdx = 0;
  let html = '';
  for (let i = 0; i < step && i < events.length; i++) {
    const evt = events[i];
    if (NARRATIVE_SKIP.has(evt.event_type)) continue;
    narrativeIdx++;
    const isCurrent = (i === step - 1);
    const {cls, line} = formatNarrativeLine(evt, narrativeIdx);
    const rowCls = isCurrent ? ' mem-current' : '';
    html += `<div class="${rowCls}"><span class="mem-idx">[${narrativeIdx}]</span> <span class="mem-type ${cls}">${line}</span></div>`;
    // Show inline content for file_write (created) with media_ref
    if ((evt.event_type === 'file_write' && evt.operation === 'create') || evt.event_type === 'file_edit') {
      const ref = evt.media_ref;
      if (ref && ref.hash && trajState.mediaBlobCache && trajState.mediaBlobCache[ref.hash]) {
        const content = trajState.mediaBlobCache[ref.hash];
        const truncated = content.length > 800 ? content.substring(0, 800) + '\n... (truncated)' : content;
        const label = evt.event_type === 'file_edit' ? 'diff' : 'content';
        html += `<span class="mem-content">--- ${label} ---\n${escapeHtml(truncated)}\n--- end ---</span>`;
      }
    }
  }
  if (narrativeIdx === 0) {
    html = '<span style="color:var(--text2)">No behavioral events yet (step 0)</span>';
  }
  metaEl.textContent = `${narrativeIdx} behavioral events at step ${step} / ${events.length} total (excludes iteration, session, error metadata)`;
  contentEl.innerHTML = html;
  // Auto-scroll to bottom
  contentEl.scrollTop = contentEl.scrollHeight;
}

// Pre-fetch all media blobs for current trajectory
async function prefetchMediaBlobs(d) {
  if (!d) return;
  trajState.mediaBlobCache = trajState.mediaBlobCache || {};
  const events = d.events || [];
  const hashes = new Set();
  for (const evt of events) {
    if (evt.media_ref && evt.media_ref.hash) hashes.add(evt.media_ref.hash);
  }
  const toFetch = [...hashes].filter(h => !trajState.mediaBlobCache[h]);
  if (toFetch.length === 0) return;
  // Fetch in parallel (batch of up to 20)
  const tasks = toFetch.slice(0, 30).map(async h => {
    try {
      const resp = await fetch(`/api/media/${d.profile}/${d.task}/${h}`);
      if (resp.ok) trajState.mediaBlobCache[h] = await resp.text();
    } catch(e) { /* skip */ }
  });
  await Promise.all(tasks);
}

async function initTrajectories() {
  try {
    const resp = await fetch('/api/trajectories');
    trajState.index = await resp.json();
    buildTrajSelectors();
  } catch(e) {
    console.error('Failed to load trajectories:', e);
  }
}

function buildTrajSelectors() {
  const {trajectories, tasks} = trajState.index;
  // Unique tasks
  const taskIds = [...new Set(trajectories.map(t => t.task))].sort();
  const sel = document.getElementById('trajTaskSelect');
  sel.innerHTML = taskIds.map(t => {
    const td = tasks[t] || {};
    const label = `${t} — ${td.name_en || td.name || ''}`;
    const typeBadge = td.type ? ` [${td.type}]` : '';
    return `<option value="${t}">${label}${typeBadge}</option>`;
  }).join('');

  trajState.selectedTask = taskIds[0] || null;
  sel.addEventListener('change', () => {
    trajState.selectedTask = sel.value;
    trajState.compareProfile = null;
    loadTrajectory();
  });

  // Unique profiles
  const profiles = [...new Set(trajectories.map(t => t.profile))].sort();
  const btnContainer = document.getElementById('trajProfileBtns');
  btnContainer.innerHTML = profiles.map(p =>
    `<button class="profile-btn" data-profile="${p}" onclick="selectTrajProfile('${p}')">${p.replace(/_/g,' ')}</button>`
  ).join('');
  const cmpContainer = document.getElementById('trajCompareBtns');
  cmpContainer.innerHTML = '<button class="profile-btn compare-none" onclick="selectTrajCompare(null)" style="font-style:italic;">None</button>' +
    profiles.map(p =>
      `<button class="profile-btn" data-cmp="${p}" onclick="selectTrajCompare('${p}')">${p.replace(/_/g,' ')}</button>`
    ).join('');

  trajState.selectedProfile = profiles[0] || null;
  if (trajState.selectedProfile) loadTrajectory();
}

window.selectTrajProfile = function(p) {
  trajState.selectedProfile = p;
  document.querySelectorAll('#trajProfileBtns .profile-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.profile === p)
  );
  loadTrajectory();
};

window.selectTrajCompare = function(p) {
  trajState.compareProfile = p;
  document.querySelectorAll('#trajCompareBtns .profile-btn').forEach(b => {
    b.classList.toggle('compare-active', b.dataset.cmp === p);
    if (!b.dataset.cmp) b.classList.toggle('active', p === null);
  });
  if (p && p !== trajState.selectedProfile) {
    loadCompareTrajectory();
  } else {
    trajState.compareProfile = null;
    trajState.compareData = null;
    renderTrajView();
  }
};

async function loadTrajectory() {
  const {selectedProfile: p, selectedTask: t} = trajState;
  if (!p || !t) return;
  const key = `${p}_${t}`;
  if (!trajState.cache[key]) {
    try {
      const resp = await fetch(`/api/trajectory/${p}/${t}`);
      if (!resp.ok) { console.error('Trajectory not found'); return; }
      trajState.cache[key] = await resp.json();
    } catch(e) { console.error(e); return; }
  }
  trajState.data = trajState.cache[key];
  trajState.currentStep = 0;
  stopPlay();
  // Highlight active profile btn
  document.querySelectorAll('#trajProfileBtns .profile-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.profile === p)
  );
  renderTrajView();
  // Pre-fetch media blobs for memory input panel (async, non-blocking)
  prefetchMediaBlobs(trajState.data).then(() => renderMemoryInput());
}

async function loadCompareTrajectory() {
  const {compareProfile: p, selectedTask: t} = trajState;
  if (!p || !t) return;
  const key = `${p}_${t}`;
  if (!trajState.cache[key]) {
    try {
      const resp = await fetch(`/api/trajectory/${p}/${t}`);
      if (!resp.ok) return;
      trajState.cache[key] = await resp.json();
    } catch(e) { return; }
  }
  trajState.compareData = trajState.cache[key];
  renderTrajView();
}

// --- File tree reconstruction ---
function reconstructState(data, step) {
  // Start with workspace files
  const files = {}; // path -> {path, size, type, source, media_ref}
  const dirs = new Set();
  (data.workspace_files || []).forEach(f => {
    files[f.path] = {...f, source: 'workspace', media_ref: null};
    // Extract dirs
    const parts = f.path.split('/');
    for (let i = 1; i < parts.length; i++) {
      dirs.add(parts.slice(0, i).join('/'));
    }
  });
  const agentCreated = new Set();
  const events = data.events || [];
  const badges = []; // last N events that affected files

  for (let i = 0; i < step; i++) {
    const evt = events[i];
    const et = evt.event_type;
    if (et === 'file_write') {
      const fp = evt.file_path || '';
      if (evt.operation === 'create') {
        files[fp] = {path: fp, size: evt.content_length || 0, type: (fp.split('.').pop() || 'unknown'), source: 'agent', media_ref: evt.media_ref || null};
        agentCreated.add(fp);
      } else {
        if (files[fp]) { files[fp].size = evt.content_length || files[fp].size; files[fp].media_ref = evt.media_ref || files[fp].media_ref; files[fp].source = 'edited'; }
      }
      // Ensure parent dirs exist
      const parts = fp.split('/');
      for (let j = 1; j < parts.length; j++) dirs.add(parts.slice(0, j).join('/'));
    } else if (et === 'dir_create') {
      const dp = evt.dir_path || evt.file_path || '';
      dirs.add(dp);
    } else if (et === 'file_move' || et === 'file_rename') {
      const old_p = evt.old_path || '';
      let new_p = evt.new_path || '';
      // If new_path is an existing directory, move file inside it (mv file dir/)
      if (dirs.has(new_p) && files[old_p]) {
        const filename = old_p.split('/').pop();
        new_p = new_p + '/' + filename;
      }
      if (files[old_p]) {
        const f = files[old_p];
        delete files[old_p];
        f.path = new_p;
        files[new_p] = f;
        if (agentCreated.has(old_p)) { agentCreated.delete(old_p); agentCreated.add(new_p); }
      }
      // Ensure parent dirs
      const parts = new_p.split('/');
      for (let j = 1; j < parts.length; j++) dirs.add(parts.slice(0, j).join('/'));
    } else if (et === 'file_delete') {
      const fp = evt.file_path || '';
      delete files[fp];
      agentCreated.delete(fp);
    } else if (et === 'file_edit') {
      const fp = evt.file_path || '';
      if (files[fp]) { files[fp].media_ref = evt.media_ref || files[fp].media_ref; files[fp].source = 'edited'; }
    } else if (et === 'file_copy') {
      const sp = evt.source_path || '';
      const dp = evt.dest_path || '';
      if (files[sp]) {
        files[dp] = {...files[sp], path: dp, source: 'agent', media_ref: evt.media_ref || null};
        agentCreated.add(dp);
      }
      const parts = dp.split('/');
      for (let j = 1; j < parts.length; j++) dirs.add(parts.slice(0, j).join('/'));
    }
  }

  // Collect recent badges (current event + previous 2)
  const badgeMap = {};
  for (let i = Math.max(0, step - 3); i < step; i++) {
    const evt = events[i];
    const et = evt.event_type;
    const age = step - 1 - i; // 0 = current, 1 = prev, etc.
    if (et === 'file_read') badgeMap[evt.file_path] = {type: 'read', age};
    else if (et === 'file_write') badgeMap[evt.file_path] = {type: evt.operation === 'create' ? 'written' : 'edited', age};
    else if (et === 'file_edit') badgeMap[evt.file_path] = {type: 'edited', age};
    else if (et === 'dir_create') badgeMap[evt.dir_path || evt.file_path] = {type: 'newdir', age};
    else if (et === 'file_move' || et === 'file_rename') {
      let np = evt.new_path;
      // If new_path is a directory, compute actual dest path
      if (dirs.has(np)) np = np + '/' + (evt.old_path || '').split('/').pop();
      badgeMap[np] = {type: 'moved', age, from: evt.old_path};
    }
    else if (et === 'file_delete') badgeMap[evt.file_path] = {type: 'deleted', age};
  }

  return {files, dirs, agentCreated, badgeMap};
}

function buildTreeHtml(state, selectedFile) {
  const {files, dirs, agentCreated, badgeMap} = state;
  // Build tree structure
  const tree = {};
  // Add dirs
  dirs.forEach(d => {
    const parts = d.split('/');
    let node = tree;
    parts.forEach(p => {
      if (!node[p]) node[p] = {__isDir: true};
      node = node[p];
    });
  });
  // Add files
  Object.values(files).forEach(f => {
    const parts = f.path.split('/');
    let node = tree;
    for (let i = 0; i < parts.length - 1; i++) {
      if (!node[parts[i]]) node[parts[i]] = {__isDir: true};
      node = node[parts[i]];
    }
    node[parts[parts.length - 1]] = {__isFile: true, ...f};
  });

  let html = '';
  function renderNode(obj, depth, parentPath) {
    const keys = Object.keys(obj).filter(k => k !== '__isDir' && k !== '__isFile').sort((a, b) => {
      const aDir = obj[a].__isDir && !obj[a].__isFile;
      const bDir = obj[b].__isDir && !obj[b].__isFile;
      if (aDir !== bDir) return aDir ? -1 : 1;
      return a.localeCompare(b);
    });
    keys.forEach(k => {
      const child = obj[k];
      const fullPath = parentPath ? `${parentPath}/${k}` : k;
      const indent = '<span class="indent"></span>'.repeat(depth);
      if (child.__isFile) {
        const ext = (child.type || '').toLowerCase();
        const icon = FILE_ICONS[ext] || '\u{1F4C4}';
        const isAgent = agentCreated.has(fullPath);
        const badge = badgeMap[fullPath];
        let badgeHtml = '';
        if (badge) {
          const cls = `badge badge-${badge.type}`;
          const labels = {written:'WRITTEN', read:'READ', newdir:'NEW DIR', moved:'MOVED', deleted:'DELETED', edited:'EDITED'};
          badgeHtml = `<span class="${cls}" style="opacity:${1 - badge.age * 0.25}">${labels[badge.type] || badge.type}${badge.from ? ' \u2190 '+badge.from.split('/').pop() : ''}</span>`;
        }
        const sel = selectedFile === fullPath ? ' selected' : '';
        const agentCls = isAgent ? ' agent-created' : '';
        html += `<div class="traj-tree-item${sel}${agentCls}" onclick="trajSelectFile('${fullPath.replace(/'/g,"\\'")}')" title="${fullPath}">${indent}<span class="icon">${icon}</span>${k}${badgeHtml}</div>`;
      } else if (child.__isDir) {
        const badge = badgeMap[fullPath];
        let badgeHtml = '';
        if (badge) {
          const cls = `badge badge-${badge.type}`;
          badgeHtml = `<span class="${cls}" style="opacity:${1 - badge.age * 0.25}">${badge.type === 'newdir' ? 'NEW DIR' : badge.type.toUpperCase()}</span>`;
        }
        html += `<div class="traj-tree-item" title="${fullPath}">${indent}<span class="icon">${DIR_ICON}</span>${k}/${badgeHtml}</div>`;
        renderNode(child, depth + 1, fullPath);
      }
    });
  }
  renderNode(tree, 0, '');
  return html || '<span style="color:var(--text2)">Empty workspace</span>';
}

// --- Profile rendering ---
const DIM_LABELS = {
  A: 'Consumption', B: 'Production', C: 'Organization',
  D: 'Iteration', E: 'Rhythm', F: 'Cross-Modal',
};
const DIM_VALUES = {
  L: {A:'sequential', B:'comprehensive', C:'deeply_nested', D:'incremental', E:'phased', F:'visual-heavy'},
  M: {A:'targeted', B:'balanced', C:'adaptive', D:'balanced', E:'steady', F:'balanced'},
  R: {A:'breadth-first', B:'minimal', C:'flat', D:'rewrite', E:'bursty', F:'text-only'},
};

async function loadProfileData(profileName) {
  if (trajState.profileCache[profileName]) return trajState.profileCache[profileName];
  try {
    const resp = await fetch(`/api/profile/${profileName}`);
    if (!resp.ok) return null;
    const contentType = resp.headers.get('Content-Type') || '';
    if (contentType.includes('application/json')) {
      const data = await resp.json();
      trajState.profileCache[profileName] = data;
      return data;
    }
    // Fallback: raw text (no yaml module on server)
    return null;
  } catch(e) { return null; }
}

function renderProfileInfo(profile, profileName) {
  const metaEl = document.getElementById('trajProfileMeta');
  const contentEl = document.getElementById('trajProfileContent');
  if (!profile) {
    metaEl.textContent = profileName || '';
    contentEl.innerHTML = '<span style="color:var(--text2)">Profile data not available</span>';
    return;
  }
  const basic = profile.basic || {};
  const wh = profile.work_habits || {};
  const personality = profile.personality || {};
  const vec = profile._vector || {};

  // Meta line
  const vecStr = Object.entries(vec).map(([d,v]) => `${d}:${v}`).join(' ');
  metaEl.textContent = `${basic.name || profileName} — ${basic.role || ''} (${basic.language || ''}) ${vecStr ? '| ' + vecStr : ''}`;

  // Build side-by-side layout: Dimensions (left) | Attributes (right)
  let html = '<div style="display:flex;gap:24px;">';

  // Left column: L/M/R dimension vector
  html += '<div style="flex:1;min-width:0;">';
  html += '<table style="width:100%;border-collapse:collapse;font-size:12px;">';
  html += '<tr><td colspan="2" style="padding:6px 0 4px;font-weight:600;color:var(--accent);border-bottom:1px solid var(--border);">Dimension Vector (L/M/R)</td></tr>';
  if (Object.keys(vec).length > 0) {
    for (const [dim, val] of Object.entries(vec)) {
      const label = DIM_LABELS[dim] || dim;
      const valLabel = (DIM_VALUES[val] && DIM_VALUES[val][dim]) || val;
      const color = val === 'L' ? '#22d3ee' : val === 'R' ? '#fb923c' : '#a78bfa';
      html += `<tr><td style="padding:3px 8px 3px 0;color:var(--text2);white-space:nowrap;">${dim}: ${label}</td><td style="padding:3px 0;"><span style="background:${color};color:#000;padding:1px 6px;border-radius:3px;font-weight:600;font-size:11px;">${val}</span> <span style="margin-left:4px;">${valLabel}</span></td></tr>`;
    }
  }
  html += '</table></div>';

  // Right column: Profile Attributes
  html += '<div style="flex:1;min-width:0;">';
  html += '<table style="width:100%;border-collapse:collapse;font-size:12px;">';
  html += '<tr><td colspan="2" style="padding:6px 0 4px;font-weight:600;color:var(--accent);border-bottom:1px solid var(--border);">Profile Attributes</td></tr>';
  const attrOrder = [
    ['working_style', 'Working Style'], ['thoroughness', 'Thoroughness'],
    ['reading_strategy', 'Reading Strategy'], ['edit_strategy', 'Edit Strategy'],
    ['directory_style', 'Directory Style'], ['naming', 'Naming'],
    ['version_strategy', 'Version Strategy'], ['error_handling', 'Error Handling'],
    ['output_structure', 'Output Structure'], ['documentation', 'Documentation'],
    ['cross_modal_behavior', 'Cross-Modal'],
  ];
  html += `<tr><td style="padding:3px 8px 3px 0;color:var(--text2);">Name</td><td style="padding:3px 0;">${escapeHtml(basic.name || '')}</td></tr>`;
  html += `<tr><td style="padding:3px 8px 3px 0;color:var(--text2);">Role</td><td style="padding:3px 0;">${escapeHtml(basic.role || '')}</td></tr>`;
  html += `<tr><td style="padding:3px 8px 3px 0;color:var(--text2);">Language</td><td style="padding:3px 0;">${escapeHtml(basic.language || '')}</td></tr>`;
  html += `<tr><td style="padding:3px 8px 3px 0;color:var(--text2);">Tone</td><td style="padding:3px 0;">${escapeHtml(personality.tone || '')}</td></tr>`;
  html += `<tr><td style="padding:3px 8px 3px 0;color:var(--text2);">Verbosity</td><td style="padding:3px 0;">${escapeHtml(personality.verbosity || '')}</td></tr>`;
  for (const [key, label] of attrOrder) {
    const val = wh[key];
    if (val) {
      html += `<tr><td style="padding:3px 8px 3px 0;color:var(--text2);">${label}</td><td style="padding:3px 0;">${escapeHtml(String(val))}</td></tr>`;
    }
  }
  html += '</table></div>';

  html += '</div>'; // close flex container
  contentEl.innerHTML = html;
}

// --- Rendering ---
let trajSelectedFile = null;

window.trajSelectFile = function(path) {
  trajSelectedFile = path;
  renderTrajFileTree();
  loadTrajFileContent(path);
};

function renderTrajView() {
  const d = trajState.data;
  if (!d) return;
  const events = d.events || [];
  const totalSteps = events.length;

  // Stats
  const s = d.summary || {};
  document.getElementById('trajStats').innerHTML = `
    <div><b>Duration:</b> ${((s.total_duration_ms||0)/1000).toFixed(1)}s</div>
    <div><b>Steps:</b> ${totalSteps}</div>
    <div><b>Iterations:</b> ${s.total_iterations||0}</div>
    <div><b>Files read:</b> ${s.unique_files_read||0}</div>
    <div><b>Files created:</b> ${(s.files_created||[]).length}</div>
    <div><b>Dirs created:</b> ${(s.dirs_created||[]).length}</div>
  `;

  // Scrubber
  const scrubber = document.getElementById('trajScrubber');
  scrubber.max = totalSteps;
  scrubber.value = trajState.currentStep;

  // Iteration markers + tick marks + legend
  renderIterMarkers(events);

  // Task prompt
  const td = d.task_def || {};
  const promptZh = td.prompt_zh || '';
  const promptEn = td.prompt_en || '';
  const promptText = promptZh ? (promptZh + (promptEn ? '\n\n--- English ---\n\n' + promptEn : '')) : promptEn || '(No prompt available)';
  document.getElementById('trajPromptText').textContent = promptText;
  const taskType = td.type ? `[${td.type}]` : '';
  const taskDims = (td.dimensions || []).join(', ');
  document.getElementById('trajPromptMeta').textContent = `${td.name || ''} ${taskType}${taskDims ? ' — Dims: ' + taskDims : ''}`;

  // Profile info (async, non-blocking)
  const profileName = d.profile || trajState.selectedProfile;
  if (profileName) {
    loadProfileData(profileName).then(profile => renderProfileInfo(profile, profileName));
  }

  // Compare mode
  const isCompare = trajState.compareProfile && trajState.compareData;
  document.getElementById('trajMainRow').style.display = isCompare ? 'none' : '';
  document.getElementById('trajCompareRow').style.display = isCompare ? '' : 'none';

  renderTrajStepLabel();
  renderTrajFileTree();
  renderTrajEventLog();
  renderMemoryInput();
}

// Event type -> color category for tick marks and legend
const TICK_CATEGORIES = {
  file_read:    {color: '#22d3ee', label: 'Read'},
  file_write:   {color: '#4ade80', label: 'Write'},
  file_edit:    {color: '#fb923c', label: 'Edit'},
  dir_create:   {color: '#fbbf24', label: 'Mkdir'},
  file_rename:  {color: '#a78bfa', label: 'Rename/Move'},
  file_move:    {color: '#a78bfa', label: 'Rename/Move'},
  file_delete:  {color: '#f87171', label: 'Delete'},
  file_browse:  {color: '#6c8aff', label: 'Browse/Search'},
  file_search:  {color: '#6c8aff', label: 'Browse/Search'},
  file_copy:    {color: '#4ade80', label: 'Copy'},
  context_switch:   {color: '#555b6e', label: 'Context Switch'},
  error_encounter:  {color: '#ef4444', label: 'Error'},
  error_response:   {color: '#ef4444', label: 'Error'},
  iteration_start:  {color: 'rgba(108,138,255,.4)', label: 'Agent Turn'},
  iteration_end:    {color: 'rgba(108,138,255,.4)', label: 'Agent Turn'},
};

function renderIterMarkers(events) {
  const iterContainer = document.getElementById('trajIterMarkers');
  const tickContainer = document.getElementById('trajTickMarks');
  const legendContainer = document.getElementById('trajLegend');
  const total = events.length;
  if (total === 0) { iterContainer.innerHTML = ''; tickContainer.innerHTML = ''; legendContainer.innerHTML = ''; return; }

  // Tick marks
  let tickHtml = '';
  const seenLabels = new Set();
  events.forEach((evt, i) => {
    const cat = TICK_CATEGORIES[evt.event_type];
    if (!cat) return;
    const pct = (i / total * 100).toFixed(2);
    tickHtml += `<span class="traj-tick" style="left:${pct}%;background:${cat.color};" title="#${i+1} ${evt.event_type}"></span>`;
    seenLabels.add(cat.label);
  });
  tickContainer.innerHTML = tickHtml;

  // Iteration markers (text labels)
  let iterHtml = '';
  events.forEach((evt, i) => {
    if (evt.event_type === 'iteration_start') {
      const pct = (i / total * 100).toFixed(1);
      const iterNum = evt.iteration_number || '?';
      iterHtml += `<span class="traj-iter-marker" style="left:${pct}%">Iter ${iterNum}</span>`;
    }
  });
  iterContainer.innerHTML = iterHtml;

  // Legend -- only show categories present in this trajectory
  const legendItems = [];
  const added = new Set();
  for (const [type, cat] of Object.entries(TICK_CATEGORIES)) {
    if (seenLabels.has(cat.label) && !added.has(cat.label)) {
      added.add(cat.label);
      legendItems.push(`<span class="traj-legend-item"><span class="traj-legend-swatch" style="background:${cat.color}"></span>${cat.label}</span>`);
    }
  }
  legendContainer.innerHTML = legendItems.join('');
}

function renderTrajStepLabel() {
  const d = trajState.data;
  if (!d) return;
  const events = d.events || [];
  const step = trajState.currentStep;
  const total = events.length;
  let detail = '';
  if (step > 0 && step <= total) {
    const evt = events[step - 1];
    const et = evt.event_type;
    const fp = evt.file_path || evt.dir_path || evt.old_path || '';
    const shortName = fp.split('/').pop() || '';
    detail = ` \u2014 ${EVENT_LABELS[et] || et} ${shortName}`;
  }
  document.getElementById('trajStepLabel').textContent = `Step ${step} / ${total}${detail}`;
}

function renderTrajFileTree() {
  const d = trajState.data;
  if (!d) return;
  const isCompare = trajState.compareProfile && trajState.compareData;
  if (isCompare) {
    const stateA = reconstructState(d, trajState.currentStep);
    const stateB = reconstructState(trajState.compareData, trajState.currentStep);
    document.getElementById('trajTreeLabelA').textContent = trajState.selectedProfile.replace(/_/g,' ');
    document.getElementById('trajTreeLabelB').textContent = trajState.compareProfile.replace(/_/g,' ');
    document.getElementById('trajTreeA').innerHTML = buildTreeHtml(stateA, null);
    document.getElementById('trajTreeB').innerHTML = buildTreeHtml(stateB, null);
  } else {
    const state = reconstructState(d, trajState.currentStep);
    document.getElementById('trajTreeLabel').textContent = `(step ${trajState.currentStep})`;
    document.getElementById('trajTree').innerHTML = buildTreeHtml(state, trajSelectedFile);
  }
}

// Fields to skip in event detail (already shown in summary row or uninteresting)
const EVT_SKIP_FIELDS = new Set(['event_type','profile_id','timestamp']);
// Fields that deserve special formatting
const EVT_DETAIL_FIELDS = {
  // error_encounter
  error_type:    {label: 'Error Type'},
  context:       {label: 'Context', wide: true, errorStyle: true},
  severity:      {label: 'Severity'},
  tool_name:     {label: 'Tool'},
  // error_response
  strategy:      {label: 'Strategy'},
  latency_ms:    {label: 'Latency', fmt: v => v + ' ms'},
  resolution_successful: {label: 'Resolved', bool: true},
  // file_read
  view_count:    {label: 'Views'},
  view_range:    {label: 'Lines', fmt: v => Array.isArray(v) ? v[0]+'-'+v[1] : v},
  content_length:{label: 'Size', fmt: v => v >= 1000 ? (v/1024).toFixed(1)+' KB' : v+' B'},
  revisit_interval_ms: {label: 'Revisit', fmt: v => v > 0 ? (v/1000).toFixed(1)+'s ago' : 'first'},
  // file_write
  operation:     {label: 'Operation'},
  // dir_create
  depth:         {label: 'Depth'},
  sibling_count: {label: 'Siblings'},
  // file_rename
  naming_pattern_change: {label: 'Pattern'},
  // file_search
  search_type:   {label: 'Search Type'},
  query:         {label: 'Query'},
  files_matched: {label: 'Matches', fmt: v => Array.isArray(v) ? v.length+' files' : v},
  // file_browse
  files_listed:  {label: 'Files Listed', fmt: v => Array.isArray(v) ? v.length+' files' : v},
  // iteration_end
  duration_ms:   {label: 'Duration', fmt: v => (v/1000).toFixed(1)+'s'},
  tools_called:  {label: 'Tools Called'},
  has_tool_error:{label: 'Had Error', bool: true},
  // context_switch
  trigger:       {label: 'Trigger'},
  switch_count:  {label: 'Switch #'},
  // file_edit
  lines_added:   {label: 'Lines +'},
  lines_deleted: {label: 'Lines -'},
  lines_modified:{label: 'Lines ~'},
  diff_summary:  {label: 'Diff', wide: true},
  edit_tool:     {label: 'Tool'},
  // cross_file_reference
  reference_type:{label: 'Ref Type'},
  interval_ms:   {label: 'Interval', fmt: v => (v/1000).toFixed(1)+'s'},
  // fs_snapshot
  total_files:   {label: 'Total Files'},
  max_depth:     {label: 'Max Depth'},
  file_count_by_type: {label: 'By Type', fmt: v => typeof v === 'object' ? Object.entries(v).map(([k,c])=>k+':'+c).join(', ') : v},
};

let trajExpandedEvent = null; // index of expanded event, or null

window.trajToggleEvent = function(idx, e) {
  e.stopPropagation();
  trajExpandedEvent = (trajExpandedEvent === idx) ? null : idx;
  renderTrajEventLog();
};

function buildEventDetailHtml(evt) {
  let cells = '';
  for (const [key, val] of Object.entries(evt)) {
    if (EVT_SKIP_FIELDS.has(key)) continue;
    // Skip path fields already shown in row summary
    if (key === 'file_path' || key === 'dir_path' || key === 'old_path' || key === 'new_path') continue;
    if (key === 'from_file' || key === 'to_file' || key === 'source_file' || key === 'target_file') continue;
    if (key === 'iteration_number') continue;
    // Skip media refs (hashes not useful to display)
    if (key.startsWith('media_ref') || key.endsWith('_hash')) continue;
    if (val === null || val === undefined || val === '') continue;

    const spec = EVT_DETAIL_FIELDS[key] || {label: key.replace(/_/g,' ')};
    let display;
    if (spec.bool) {
      display = `<span class="evt-val ${val ? 'evt-success' : 'evt-fail'}">${val ? 'Yes' : 'No'}</span>`;
    } else if (spec.fmt) {
      display = `<span class="evt-val">${escapeHtml(String(spec.fmt(val)))}</span>`;
    } else if (spec.errorStyle) {
      display = `<span class="evt-val evt-error-ctx">${escapeHtml(String(val))}</span>`;
    } else {
      display = `<span class="evt-val">${escapeHtml(String(val))}</span>`;
    }

    if (spec.wide || spec.errorStyle) {
      cells += `<div style="grid-column:1/-1"><span class="evt-key">${spec.label}:</span> ${display}</div>`;
    } else {
      cells += `<div><span class="evt-key">${spec.label}:</span> ${display}</div>`;
    }
  }
  return cells || '<div style="color:var(--text2)">(no additional details)</div>';
}

function renderTrajEventLog() {
  const d = trajState.data;
  if (!d) return;
  const events = d.events || [];
  const step = trajState.currentStep;
  // Show window of ~15 events around current step
  const start = Math.max(0, step - 7);
  const end = Math.min(events.length, step + 8);
  let html = '<table><tbody>';
  for (let i = start; i < end; i++) {
    const evt = events[i];
    const et = evt.event_type;
    const isCurrent = (i === step - 1);
    const isExpanded = (trajExpandedEvent === i);
    const cls = isCurrent ? ' class="current-event"' : '';
    const fp = evt.file_path || evt.dir_path || evt.old_path || '';
    const shortPath = fp.split('/').slice(-2).join('/') || '';
    let detail = shortPath;
    if (et === 'file_move' || et === 'file_rename') {
      const np = (evt.new_path || '').split('/').pop();
      detail = `${(evt.old_path||'').split('/').pop()} \u2192 ${np}`;
    } else if (et === 'iteration_start') {
      detail = `Iteration ${evt.iteration_number || '?'}`;
    } else if (et === 'iteration_end') {
      const dur = evt.duration_ms ? (evt.duration_ms/1000).toFixed(1)+'s' : '';
      detail = `Iteration ${evt.iteration_number || '?'} ${dur}`;
    } else if (et === 'file_read') {
      const vr = evt.view_range ? `[${evt.view_range[0]}-${evt.view_range[1]}]` : '';
      detail = `${shortPath} ${vr}`;
    } else if (et === 'context_switch') {
      detail = `${(evt.from_file||'').split('/').pop()} \u2192 ${(evt.to_file||'').split('/').pop()}`;
    } else if (et === 'error_encounter') {
      detail = `${evt.error_type || 'error'} (${evt.severity || ''})`;
    } else if (et === 'error_response') {
      const ok = evt.resolution_successful;
      detail = `${evt.strategy || ''} ${ok ? '\u2714' : '\u2718'}`;
    } else if (et === 'file_search') {
      detail = `${evt.search_type||''}: "${evt.query||''}"`;
    } else if (et === 'file_browse') {
      const cnt = Array.isArray(evt.files_listed) ? evt.files_listed.length : '?';
      detail = `${shortPath || '.'} (${cnt} files)`;
    } else if (et === 'cross_file_reference') {
      detail = `${(evt.source_file||'').split('/').pop()} \u2192 ${(evt.target_file||'').split('/').pop()}`;
    }
    const expandIcon = isExpanded ? '\u25BC' : '\u25B6';
    html += `<tr${cls} onclick="trajToggleEvent(${i},event)"><td style="color:var(--text2);width:30px;text-align:right">#${i+1}</td><td><span class="traj-event-type evt-${et}">${EVENT_LABELS[et]||et}</span></td><td style="max-width:220px;overflow:hidden;text-overflow:ellipsis">${detail}</td><td style="width:16px;color:var(--text2);font-size:9px">${expandIcon}</td></tr>`;
    if (isExpanded) {
      html += `<tr class="traj-evt-detail"><td colspan="4"><div class="traj-evt-detail-grid">${buildEventDetailHtml(evt)}</div></td></tr>`;
    }
  }
  html += '</tbody></table>';
  const logEl = document.getElementById('trajEventLog');
  logEl.innerHTML = html;
  // Scroll current event into view within the log container only (don't move the page)
  const cur = logEl.querySelector('tr.current-event');
  if (cur) {
    const logTop = logEl.scrollTop;
    const logH = logEl.clientHeight;
    const rowTop = cur.offsetTop - logEl.offsetTop;
    if (rowTop < logTop || rowTop + cur.offsetHeight > logTop + logH) {
      logEl.scrollTop = rowTop - logH / 2;
    }
  }
}

window.trajJumpTo = function(step) {
  trajState.currentStep = step;
  syncScrubber();
  renderTrajStepLabel();
  renderTrajFileTree();
  renderTrajEventLog();
  renderMemoryInput();
};

function syncScrubber() {
  document.getElementById('trajScrubber').value = trajState.currentStep;
}

// File viewer
async function loadTrajFileContent(path) {
  const d = trajState.data;
  if (!d) return;
  const header = document.getElementById('trajViewerHeader');
  const content = document.getElementById('trajViewerContent');
  const ext = (path.split('.').pop() || '').toLowerCase();

  // Determine file source at current step
  const state = reconstructState(d, trajState.currentStep);
  const fileInfo = state.files[path];

  if (!fileInfo) {
    header.innerHTML = `<span>${path}</span>`;
    content.innerHTML = '<span style="color:var(--text2)">File not found at current step</span>';
    return;
  }

  const source = fileInfo.source;
  const sourceBadgeCls = source === 'workspace' ? 'source-workspace' : source === 'agent' ? 'source-agent' : 'source-edited';
  const sourceLabel = source === 'workspace' ? 'WORKSPACE' : source === 'agent' ? 'AGENT-CREATED' : 'EDITED';
  header.innerHTML = `<span>${path.split('/').pop()}</span><span class="source-badge ${sourceBadgeCls}">${sourceLabel}</span><span style="color:var(--text2);font-size:11px">${fileInfo.size ? fileInfo.size + ' bytes' : ''}</span>`;

  // Workspace images: render inline
  if (source === 'workspace' && IMAGE_EXTS.has(ext)) {
    const url = `/api/workspace-file/${d.task}/${encodeURIComponent(path)}`;
    content.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;padding:8px;"><img src="${url}" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:4px;" alt="${path.split('/').pop()}"></div>`;
    return;
  }

  // Workspace PDFs: render inline with iframe
  if (source === 'workspace' && PREVIEW_EXTS.has(ext)) {
    const url = `/api/workspace-file/${d.task}/${encodeURIComponent(path)}`;
    content.innerHTML = `<iframe src="${url}" style="width:100%;height:100%;min-height:320px;border:none;border-radius:4px;background:#fff;"></iframe>`;
    return;
  }

  // Unsupported binary files (docx, xlsx, etc.)
  if (!TEXT_EXTS.has(ext) && !IMAGE_EXTS.has(ext) && !PREVIEW_EXTS.has(ext) && !fileInfo.media_ref) {
    content.innerHTML = `<div class="binary-placeholder"><div class="big-icon">${FILE_ICONS[ext] || '\u{1F4C4}'}</div><div>${path.split('/').pop()}</div><div style="font-size:12px;margin-top:4px">${ext.toUpperCase()} \u2022 ${fileInfo.size} bytes</div></div>`;
    return;
  }

  // Agent-created or edited: load from media blob
  if (fileInfo.media_ref && fileInfo.media_ref.hash) {
    const cacheKey = `media_${d.profile}_${d.task}_${fileInfo.media_ref.hash}`;
    if (!trajState.fileContentCache[cacheKey]) {
      try {
        const resp = await fetch(`/api/media/${d.profile}/${d.task}/${fileInfo.media_ref.hash}`);
        if (resp.ok) trajState.fileContentCache[cacheKey] = await resp.text();
        else trajState.fileContentCache[cacheKey] = `(Failed to load: ${resp.status})`;
      } catch(e) { trajState.fileContentCache[cacheKey] = '(Network error)'; }
    }
    const text = trajState.fileContentCache[cacheKey];
    if (fileInfo.media_ref.type === 'diff') {
      renderDiff(content, text);
    } else {
      renderTextContent(content, text);
    }
    return;
  }

  // Workspace text file: load from workspace API
  if (source === 'workspace' && TEXT_EXTS.has(ext)) {
    const cacheKey = `ws_${d.task}_${path}`;
    if (!trajState.fileContentCache[cacheKey]) {
      try {
        const resp = await fetch(`/api/workspace-file/${d.task}/${encodeURIComponent(path)}`);
        if (resp.ok) trajState.fileContentCache[cacheKey] = await resp.text();
        else trajState.fileContentCache[cacheKey] = `(Failed: ${resp.status})`;
      } catch(e) { trajState.fileContentCache[cacheKey] = '(Network error)'; }
    }
    renderTextContent(content, trajState.fileContentCache[cacheKey]);
    return;
  }

  content.innerHTML = `<div class="binary-placeholder"><div class="big-icon">${FILE_ICONS[ext] || '\u{1F4C4}'}</div><div>${path.split('/').pop()}</div><div style="font-size:12px;margin-top:4px">${ext.toUpperCase()} \u2022 ${fileInfo.size} bytes</div></div>`;
}

function renderTextContent(container, text) {
  const lines = text.split('\n');
  let html = '';
  lines.forEach((line, i) => {
    const escaped = escapeHtml(line);
    html += `<span class="line-num">${i + 1}</span>${escaped}\n`;
  });
  container.innerHTML = html;
}

function renderDiff(container, text) {
  const lines = text.split('\n');
  let html = '';
  lines.forEach((line, i) => {
    const escaped = escapeHtml(line);
    let cls = '';
    if (line.startsWith('+') && !line.startsWith('+++')) cls = 'diff-add';
    else if (line.startsWith('-') && !line.startsWith('---')) cls = 'diff-del';
    html += `<span class="line-num">${i + 1}</span><span class="${cls}">${escaped}</span>\n`;
  });
  container.innerHTML = html;
}

// --- Controls ---
document.getElementById('trajScrubber').addEventListener('input', function() {
  trajState.currentStep = parseInt(this.value);
  renderTrajStepLabel();
  renderTrajFileTree();
  renderTrajEventLog();
  renderMemoryInput();
});

document.getElementById('trajPrev').addEventListener('click', () => {
  if (trajState.currentStep > 0) {
    trajState.currentStep--;
    syncScrubber();
    renderTrajStepLabel();
    renderTrajFileTree();
    renderTrajEventLog();
  }
});

document.getElementById('trajNext').addEventListener('click', () => {
  const max = (trajState.data?.events || []).length;
  if (trajState.currentStep < max) {
    trajState.currentStep++;
    syncScrubber();
    renderTrajStepLabel();
    renderTrajFileTree();
    renderTrajEventLog();
  }
});

document.getElementById('trajPlay').addEventListener('click', () => {
  if (trajState.playing) { stopPlay(); } else { startPlay(); }
});

function startPlay() {
  trajState.playing = true;
  const btn = document.getElementById('trajPlay');
  btn.textContent = '\u23F8 Pause';
  btn.classList.add('playing');
  playStep();
}

function stopPlay() {
  trajState.playing = false;
  const btn = document.getElementById('trajPlay');
  btn.textContent = '\u25B6 Play';
  btn.classList.remove('playing');
  if (trajState.playTimer) { clearTimeout(trajState.playTimer); trajState.playTimer = null; }
}

function playStep() {
  if (!trajState.playing) return;
  const max = (trajState.data?.events || []).length;
  if (trajState.currentStep >= max) { stopPlay(); return; }
  trajState.currentStep++;
  syncScrubber();
  renderTrajStepLabel();
  renderTrajFileTree();
  renderTrajEventLog();
  renderMemoryInput();
  const speed = parseInt(document.getElementById('trajSpeed').value) || 1000;
  trajState.playTimer = setTimeout(playStep, speed);
}

// Keyboard shortcuts when trajectory tab is active
document.addEventListener('keydown', (e) => {
  const trajPanel = document.getElementById('panel-trajectories');
  if (!trajPanel.classList.contains('active') || !trajState.data) return;
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft' || e.key === 'a') {
    e.preventDefault();
    document.getElementById('trajPrev').click();
  } else if (e.key === 'ArrowRight' || e.key === 'd') {
    e.preventDefault();
    document.getElementById('trajNext').click();
  } else if (e.key === ' ') {
    e.preventDefault();
    document.getElementById('trajPlay').click();
  }
});

// Drag-to-resize handles
(function() {
  let activeHandle = null, startX = 0, startLeftW = 0, startRightW = 0, leftEl = null, rightEl = null;
  document.querySelectorAll('.traj-resize-handle').forEach(handle => {
    handle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      const row = handle.closest('.traj-main-row');
      leftEl = handle.previousElementSibling;
      rightEl = handle.nextElementSibling;
      if (!leftEl || !rightEl) return;
      activeHandle = handle;
      startX = e.clientX;
      startLeftW = leftEl.getBoundingClientRect().width;
      startRightW = rightEl.getBoundingClientRect().width;
      handle.classList.add('active');
      document.body.classList.add('traj-resizing');
    });
  });
  document.addEventListener('mousemove', (e) => {
    if (!activeHandle) return;
    const dx = e.clientX - startX;
    const newLeft = Math.max(120, startLeftW + dx);
    const newRight = Math.max(120, startRightW - dx);
    // Only apply if both panels stay above minimum
    if (newLeft >= 120 && newRight >= 120) {
      leftEl.style.width = newLeft + 'px';
      leftEl.style.flex = 'none';
      // If right is the flex viewer, let it stay flex; otherwise set width
      if (rightEl.classList.contains('traj-viewer-panel')) {
        // viewer stays flex:1, no width override needed
        rightEl.style.width = '';
        rightEl.style.flex = '1';
      } else {
        rightEl.style.width = newRight + 'px';
        rightEl.style.flex = 'none';
      }
      // If left is the viewer
      if (leftEl.classList.contains('traj-viewer-panel')) {
        leftEl.style.width = '';
        leftEl.style.flex = '1';
      }
    }
  });
  document.addEventListener('mouseup', () => {
    if (activeHandle) {
      activeHandle.classList.remove('active');
      document.body.classList.remove('traj-resizing');
      activeHandle = null; leftEl = null; rightEl = null;
    }
  });
})();

// Initialize when trajectories tab is first clicked
let trajInitialized = false;
document.querySelector('.tab[data-tab="trajectories"]').addEventListener('click', () => {
  if (!trajInitialized) {
    trajInitialized = true;
    initTrajectories();
  }
});
