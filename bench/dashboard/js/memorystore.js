// ============================================================
// Memory Store Viewer
// ============================================================

let msIndex = null;
let msCurrentData = null;
let msSources = [];
let msCurrentSource = null;
let msFullMode = true;  // default: show full untruncated data

const MS_METHOD_LABELS = {
  filegramos_simple: 'FileGramOS Simple',
  filegramos: 'FileGramOS',
  full_context: 'Full Context',
  eager_summarization: 'Eager Summ.',
  naive_rag: 'Naive RAG',
  mem0: 'Mem0',
  zep: 'Zep',
  memos: 'MemOS',
  memu: 'MemU',
  evermemos: 'EverMemOS',
};

const MS_METHOD_ORDER = [
  'filegramos', 'filegramos_simple', 'full_context', 'eager_summarization',
  'naive_rag', 'mem0', 'zep', 'memos', 'memu', 'evermemos',
];

// Load sources on tab activation
document.querySelector('[data-tab="memorystore"]').addEventListener('click', async () => {
  if (msSources.length === 0) await loadMSSources();
});

async function loadMSSources() {
  try {
    const resp = await fetch('/api/memory-store/sources');
    msSources = await resp.json();
    const sel = document.getElementById('msSource');
    if (msSources.length === 0) {
      sel.innerHTML = '<option value="">No caches found</option>';
      return;
    }
    sel.innerHTML = msSources.map(s =>
      `<option value="${s.slug}">${s.slug} (${s.profiles} profiles)</option>`
    ).join('');
    sel.addEventListener('change', () => {
      msCurrentSource = sel.value;
      msIndex = null;
      loadMSIndex();
    });
    msCurrentSource = msSources[0].slug;
    await loadMSIndex();
  } catch (e) {
    document.getElementById('msContent').innerHTML =
      '<div class="card"><p style="color:var(--red)">Failed to load memory store sources</p></div>';
  }
}

async function loadMSIndex() {
  try {
    const sourceParam = msCurrentSource ? `?source=${msCurrentSource}` : '';
    const resp = await fetch(`/api/memory-store/index${sourceParam}`);
    msIndex = await resp.json();
    populateMSSelectors();
  } catch (e) {
    document.getElementById('msContent').innerHTML =
      '<div class="card"><p style="color:var(--red)">Failed to load memory store index</p></div>';
  }
}

function populateMSSelectors() {
  const profileSel = document.getElementById('msProfile');
  const methodSel = document.getElementById('msMethod');

  const profiles = Object.keys(msIndex).sort();
  profileSel.innerHTML = profiles.map(p =>
    `<option value="${p}">${p}</option>`
  ).join('');

  const updateMethods = () => {
    const profile = profileSel.value;
    const methods = Object.keys(msIndex[profile] || {});
    methods.sort((a, b) => MS_METHOD_ORDER.indexOf(a) - MS_METHOD_ORDER.indexOf(b));
    methodSel.innerHTML = methods.map(m =>
      `<option value="${m}">${MS_METHOD_LABELS[m] || m}</option>`
    ).join('');
    loadMSDetail();
  };

  profileSel.addEventListener('change', updateMethods);
  methodSel.addEventListener('change', loadMSDetail);
  updateMethods();
}

// Full/Compact toggle
document.getElementById('msFullToggle').addEventListener('click', () => {
  msFullMode = !msFullMode;
  const btn = document.getElementById('msFullToggle');
  btn.textContent = msFullMode ? 'Compact' : 'Full';
  btn.classList.toggle('active', msFullMode);
  loadMSDetail();
});
// Set initial state
document.getElementById('msFullToggle').classList.add('active');

async function loadMSDetail() {
  const profile = document.getElementById('msProfile').value;
  const method = document.getElementById('msMethod').value;
  if (!profile || !method) return;

  const info = document.getElementById('msInfo');
  info.textContent = 'Loading...';

  try {
    const params = new URLSearchParams();
    if (msCurrentSource) params.set('source', msCurrentSource);
    if (msFullMode) params.set('full', '1');
    const qs = params.toString();
    const resp = await fetch(`/api/memory-store/${profile}/${method}${qs ? '?' + qs : ''}`);
    msCurrentData = await resp.json();
    const mode = msCurrentData.full_mode ? 'Full' : 'Compact';
    info.textContent = `${msCurrentData.size_kb} KB | ${mode}`;
    renderMSDetail();
  } catch (e) {
    info.textContent = 'Error loading';
    document.getElementById('msContent').innerHTML =
      `<div class="card"><p style="color:var(--red)">Failed to load: ${e.message}</p></div>`;
  }
}

function renderMSDetail() {
  const container = document.getElementById('msContent');
  const d = msCurrentData;

  // Check if this is a FileGramOS MemoryStore (has memory_store field)
  if (d.memory_store) {
    renderMemoryStore(container, d.memory_store);
    return;
  }

  // Fall back to generic rendering
  const data = d.data;
  const perTrajKeys = d.per_trajectory_keys;

  let html = '';

  // Summary card
  html += '<div class="card"><h3>Structure Overview</h3><table class="ms-table"><thead><tr><th>Field</th><th>Type</th><th>Size</th><th>Per-Trajectory?</th></tr></thead><tbody>';
  for (const [key, val] of Object.entries(data)) {
    const isPT = perTrajKeys.includes(key);
    const size = val.type === 'list' ? val.count : val.type === 'ndarray' ? val.shape.join('\u00d7') : '\u2014';
    html += `<tr>
      <td><code>${key}</code></td>
      <td>${val.type}</td>
      <td>${size}</td>
      <td>${isPT ? '<span class="ms-badge ms-badge-green">Yes (20)</span>' : val.type === 'list' && val.count > 0 ? '<span class="ms-badge ms-badge-orange">Consolidated</span>' : '\u2014'}</td>
    </tr>`;
  }
  html += '</tbody></table></div>';

  // Per-trajectory data
  if (perTrajKeys.length > 0) {
    html += renderPerTrajectory(data, perTrajKeys);
  }

  // Consolidated data
  const consolidatedKeys = Object.keys(data).filter(k =>
    !perTrajKeys.includes(k) && data[k].type === 'list' && data[k].count > 0
  );
  if (consolidatedKeys.length > 0) {
    html += renderConsolidated(data, consolidatedKeys);
  }

  // Scalar/boolean fields
  const scalarKeys = Object.keys(data).filter(k =>
    data[k].type !== 'list' && data[k].type !== 'ndarray'
  );
  if (scalarKeys.length > 0) {
    html += '<div class="card"><h3>Scalars</h3>';
    for (const k of scalarKeys) {
      html += `<div style="margin-bottom:4px;"><code>${k}</code>: <strong>${data[k].value}</strong></div>`;
    }
    html += '</div>';
  }

  // ndarray info
  const arrKeys = Object.keys(data).filter(k => data[k].type === 'ndarray');
  if (arrKeys.length > 0) {
    html += '<div class="card"><h3>Embeddings</h3>';
    for (const k of arrKeys) {
      html += `<div><code>${k}</code>: shape=${data[k].shape.join('\u00d7')}, dtype=${data[k].dtype}</div>`;
    }
    html += '</div>';
  }

  container.innerHTML = html;
  attachToggleHandlers(container);
}

// ============================================================
// FileGramOS MemoryStore Renderer
// ============================================================

function renderMemoryStore(container, ms) {
  let html = '';

  // Header
  html += `<div class="card" style="border-left:4px solid var(--accent);">
    <h3 style="margin:0;">FileGramOS Memory Profile</h3>
    <div style="color:var(--text2);margin-top:4px;">
      ${ms.n_sessions} sessions analyzed \u00b7 ${ms.n_deviant} behavioral anomalies detected
    </div>
  </div>`;

  // Rendered Profile (main view)
  html += `<div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h3>Rendered Profile</h3>
      <button class="ms-toggle" data-target="ms-rendered-detail">\u25bc Collapse</button>
    </div>
    <div id="ms-rendered-detail" class="ms-rendered-profile">
      ${renderMarkdown(ms.rendered_profile)}
    </div>
  </div>`;

  // Channel 1: Procedural
  const ch1 = ms.channel_1_procedural;
  html += `<div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h3>Channel 1: Procedural Patterns</h3>
      <button class="ms-toggle" data-target="ms-ch1-detail">\u25b6 Expand</button>
    </div>
    <div id="ms-ch1-detail" style="display:none;margin-top:12px;">`;

  if (ch1.dimension_classifications && ch1.dimension_classifications.length > 0) {
    html += '<h4 style="margin:8px 0 4px;font-size:13px;">Dimension Classifications</h4>';
    for (const cls of ch1.dimension_classifications) {
      html += `<div class="ms-classification">${escapeHtml(cls)}</div>`;
    }
  }
  if (ch1.behavioral_patterns && ch1.behavioral_patterns.length > 0) {
    html += '<h4 style="margin:12px 0 4px;font-size:13px;">Behavioral Patterns</h4>';
    for (const pat of ch1.behavioral_patterns) {
      html += `<div class="ms-pattern">${escapeHtml(pat)}</div>`;
    }
  }
  if (ch1.aggregate && Object.keys(ch1.aggregate).length > 0) {
    html += '<h4 style="margin:12px 0 4px;font-size:13px;">Aggregate Statistics</h4>';
    html += '<div class="ms-obj">';
    for (const [group, stats] of Object.entries(ch1.aggregate)) {
      if (typeof stats === 'object' && stats !== null) {
        html += `<div class="ms-obj-section">
          <div class="ms-obj-key">${group}</div>
          <div class="ms-obj-nested">`;
        for (const [k, v] of Object.entries(stats)) {
          const dv = typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v;
          html += `<span class="ms-kv"><code>${k}</code>: <strong>${dv}</strong></span>`;
        }
        html += '</div></div>';
      }
    }
    html += '</div>';
  }
  html += '</div></div>';

  // Channel 2: Semantic
  const ch2 = ms.channel_2_semantic;
  html += `<div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h3>Channel 2: Semantic Content</h3>
      <button class="ms-toggle" data-target="ms-ch2-detail">\u25b6 Expand</button>
    </div>
    <div id="ms-ch2-detail" style="display:none;margin-top:12px;">`;

  html += `<div style="margin-bottom:8px;"><strong>Files:</strong> ${ch2.total_files} total</div>`;
  if (ch2.filenames && ch2.filenames.length > 0) {
    html += '<div class="ms-filename-grid">';
    for (const fn of ch2.filenames) {
      const base = fn.includes('/') ? fn.split('/').pop() : fn;
      html += `<span class="ms-filename">${escapeHtml(base)}</span>`;
    }
    html += '</div>';
  }
  if (ch2.directories && ch2.directories.length > 0) {
    html += `<div style="margin:8px 0;"><strong>Directories:</strong> ${ch2.directories.map(d => `<code>${escapeHtml(d)}</code>`).join(', ')}</div>`;
  }
  if (ch2.content_samples && ch2.content_samples.length > 0) {
    html += `<h4 style="margin:12px 0 4px;font-size:13px;">Representative Content Samples (${ch2.content_samples.length})</h4>`;
    for (const s of ch2.content_samples) {
      const sizeK = (s.size / 1000).toFixed(0);
      const tid = s.trajectory_id ? ` \u00b7 ${s.trajectory_id.split('_').pop()}` : '';
      const imp = s.importance ? ` \u00b7 importance: ${s.importance.toFixed(2)}` : '';
      const id = 'ms-sample-' + Math.random().toString(36).slice(2, 8);
      html += `<div class="ms-sample">
        <div><code>${escapeHtml(s.path)}</code> <span style="color:var(--text2);">(${sizeK}K, .${s.type}${tid}${imp})</span></div>`;
      if (s.preview && s.preview.length > 200) {
        html += `<div class="ms-preview">${escapeHtml(s.preview.slice(0, 200))}...</div>
          <button class="ms-toggle" data-target="${id}" style="font-size:11px;margin-top:3px;">\u25b6 Full preview (${s.preview.length} chars)</button>
          <pre id="${id}" style="display:none;" class="ms-pre">${escapeHtml(s.preview)}</pre>`;
      } else {
        html += `<div class="ms-preview">${escapeHtml(s.preview || '')}</div>`;
      }
      html += '</div>';
    }
  }

  // LLM Narratives
  if (ch2.llm_narratives && Object.keys(ch2.llm_narratives).length > 0) {
    html += `<h4 style="margin:16px 0 4px;font-size:13px;">LLM Behavioral Narratives (${Object.keys(ch2.llm_narratives).length} sessions)</h4>`;
    for (const [tid, narrative] of Object.entries(ch2.llm_narratives)) {
      const shortTid = tid.split('_').pop();
      html += `<div class="ms-narrative-card">
        <div style="font-weight:600;font-size:12px;margin-bottom:6px;color:var(--accent);">${escapeHtml(shortTid)}</div>`;
      if (typeof narrative === 'object' && narrative !== null) {
        for (const [k, v] of Object.entries(narrative)) {
          const val = Array.isArray(v) ? v.join('; ') : String(v);
          html += `<div style="margin:3px 0;font-size:12px;"><span style="color:var(--text2);">${escapeHtml(k)}:</span> ${escapeHtml(val)}</div>`;
        }
      } else {
        html += `<div style="font-size:12px;">${escapeHtml(String(narrative))}</div>`;
      }
      html += '</div>';
    }
  }
  html += '</div></div>';

  // Channel 3: Episodic
  const ch3 = ms.channel_3_episodic;
  html += `<div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <h3>Channel 3: Episodic Consistency</h3>
      <button class="ms-toggle" data-target="ms-ch3-detail">\u25b6 Expand</button>
    </div>
    <div id="ms-ch3-detail" style="display:none;margin-top:12px;">`;

  // Centroid fingerprint visualization
  if (ch3.centroid && ch3.centroid.length > 0) {
    html += `<h4 style="margin:0 0 6px;font-size:13px;">Fingerprint Centroid (${ch3.centroid.length} dims)</h4>`;
    html += '<div class="ms-fingerprint-bar">';
    const maxVal = Math.max(...ch3.centroid.map(Math.abs), 0.001);
    for (let i = 0; i < ch3.centroid.length; i++) {
      const v = ch3.centroid[i];
      const h = Math.abs(v / maxVal) * 100;
      const color = v >= 0 ? 'var(--accent)' : 'var(--red)';
      html += `<div class="ms-fp-bar" style="height:${Math.max(h, 2)}%;background:${color};" title="dim ${i}: ${v.toFixed(3)}"></div>`;
    }
    html += '</div>';
  }

  // Per-session distances
  if (ch3.per_session_distances && Object.keys(ch3.per_session_distances).length > 0) {
    html += '<h4 style="margin:12px 0 6px;font-size:13px;">Per-Session Distance from Centroid</h4>';
    html += '<div style="display:flex;flex-wrap:wrap;gap:6px;">';
    for (const [tid, dist] of Object.entries(ch3.per_session_distances)) {
      const shortTid = tid.split('_').pop();
      const isDeviant = ch3.deviation_flags && ch3.deviation_flags[tid];
      const bgColor = isDeviant ? 'rgba(248,113,113,.15)' : 'rgba(108,138,255,.08)';
      html += `<span style="font-size:11px;padding:3px 8px;border-radius:4px;background:${bgColor};">
        ${escapeHtml(shortTid)}: ${dist.toFixed(3)}${isDeviant ? ' *' : ''}
      </span>`;
    }
    html += '</div>';
  }

  if (ch3.absence_flags && ch3.absence_flags.length > 0) {
    html += '<div style="margin:12px 0 8px;"><strong>Never observed:</strong></div>';
    for (const a of ch3.absence_flags) {
      html += `<div style="margin-left:12px;color:var(--text2);">\u2022 ${escapeHtml(a)}</div>`;
    }
  }

  // Deviations
  const devFlags = ch3.deviation_flags || {};
  const devTids = Object.keys(devFlags).filter(k => devFlags[k]);
  if (devTids.length > 0) {
    html += `<div style="margin:8px 0;"><strong>Anomalous sessions (${devTids.length}):</strong></div>`;
    for (const tid of devTids) {
      const details = (ch3.deviation_details || {})[tid];
      let desc = '';
      if (Array.isArray(details) && details.length > 0) {
        desc = details.map(d => `${d.dimension || ''} (${d.feature || ''}, \u03b4=${(d.delta || 0).toFixed(1)})`).join(', ');
      }
      html += `<div style="margin-left:12px;"><span class="ms-badge ms-badge-orange">${tid.split('_').pop()}</span> ${escapeHtml(desc)}</div>`;
    }
  }

  // Consistency flags
  if (ch3.consistency_flags && typeof ch3.consistency_flags === 'object') {
    const cfKeys = Object.keys(ch3.consistency_flags).filter(k => k !== 'n_trajectories' && k !== 'note');
    if (cfKeys.length > 0) {
      html += '<h4 style="margin:12px 0 6px;font-size:13px;">Consistency Analysis</h4>';
      for (const attr of cfKeys) {
        const metrics = ch3.consistency_flags[attr];
        if (typeof metrics !== 'object' || metrics === null) continue;
        html += `<div class="ms-obj-section"><div class="ms-obj-key">${attr}</div><div class="ms-obj-nested">`;
        for (const [key, info] of Object.entries(metrics)) {
          if (typeof info !== 'object' || info === null) continue;
          const stable = info.stable ? '\u2705' : '\u26a0\ufe0f';
          const cv = typeof info.cv === 'number' ? ` cv=${info.cv.toFixed(2)}` : '';
          html += `<span class="ms-kv">${stable} <code>${key}</code>${cv}</span>`;
        }
        html += '</div></div>';
      }
    }
  }
  html += '</div></div>';

  // Per-Engram Detail (expandable)
  if (ms.engrams && ms.engrams.length > 0) {
    html += `<div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h3>Per-Engram Detail (${ms.engrams.length} trajectories)</h3>
        <button class="ms-toggle" data-target="ms-engrams-detail">\u25b6 Expand</button>
      </div>
      <div id="ms-engrams-detail" style="display:none;margin-top:12px;">`;

    // Engram tabs
    html += '<div class="ms-tabs" id="msEngramTabs">';
    for (let i = 0; i < ms.engrams.length; i++) {
      const eng = ms.engrams[i];
      const shortId = eng.task_id;
      const devMark = eng.is_deviant ? '*' : '';
      html += `<button class="ms-tab-btn ${i === 0 ? 'active' : ''}" data-idx="${i}">${shortId}${devMark}</button>`;
    }
    html += '</div>';

    for (let i = 0; i < ms.engrams.length; i++) {
      const eng = ms.engrams[i];
      html += `<div class="ms-engram-content" id="msEngram${i}" style="display:${i === 0 ? 'block' : 'none'};">`;

      // Engram header
      html += `<div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px;font-size:12px;">
        <span><strong>Trajectory:</strong> ${escapeHtml(eng.trajectory_id)}</span>
        <span><strong>Events:</strong> ${eng.behavioral_event_count}/${eng.event_count}</span>
        <span><strong>Importance:</strong> ${eng.importance_score.toFixed(4)}</span>
        <span><strong>Distance:</strong> ${eng.distance_from_centroid}</span>
        ${eng.is_perturbed ? '<span class="ms-badge ms-badge-orange">Perturbed</span>' : ''}
        ${eng.is_deviant ? '<span class="ms-badge ms-badge-orange">Deviant</span>' : ''}
      </div>`;

      // Fingerprint mini-bar
      if (eng.fingerprint && eng.fingerprint.length > 0) {
        const maxFp = Math.max(...eng.fingerprint.map(Math.abs), 0.001);
        html += '<div style="margin-bottom:8px;"><strong style="font-size:12px;">Fingerprint</strong>';
        html += '<div class="ms-fingerprint-bar" style="height:30px;">';
        for (let j = 0; j < eng.fingerprint.length; j++) {
          const v = eng.fingerprint[j];
          const h = Math.abs(v / maxFp) * 100;
          const color = v >= 0 ? 'var(--accent)' : 'var(--red)';
          html += `<div class="ms-fp-bar" style="height:${Math.max(h, 2)}%;background:${color};" title="dim ${j}: ${v.toFixed(3)}"></div>`;
        }
        html += '</div></div>';
      }

      // Procedural features
      if (eng.procedural && Object.keys(eng.procedural).length > 0) {
        const procId = `ms-eng-proc-${i}`;
        html += `<div style="margin-bottom:8px;">
          <button class="ms-toggle" data-target="${procId}" style="font-size:12px;">\u25b6 Procedural Features (${Object.keys(eng.procedural).length} dimensions)</button>
          <div id="${procId}" style="display:none;margin-top:6px;">`;
        html += '<div class="ms-obj">';
        for (const [group, stats] of Object.entries(eng.procedural)) {
          if (typeof stats === 'object' && stats !== null) {
            html += `<div class="ms-obj-section"><div class="ms-obj-key">${group}</div><div class="ms-obj-nested">`;
            for (const [k, v] of Object.entries(stats)) {
              const dv = typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v;
              html += `<span class="ms-kv"><code>${k}</code>: <strong>${dv}</strong></span>`;
            }
            html += '</div></div>';
          }
        }
        html += '</div></div></div>';
      }

      // Semantic unit
      const sem = eng.semantic;
      if (sem) {
        const semId = `ms-eng-sem-${i}`;
        const semCount = (sem.created_files || []).length + (sem.edit_chains || []).length + (sem.cross_file_refs || []).length;
        html += `<div style="margin-bottom:8px;">
          <button class="ms-toggle" data-target="${semId}" style="font-size:12px;">\u25b6 Semantic Unit (${semCount} items)</button>
          <div id="${semId}" style="display:none;margin-top:6px;">`;

        if (sem.created_files && sem.created_files.length > 0) {
          html += '<div style="margin-bottom:6px;font-size:12px;font-weight:600;color:var(--green);">Created Files</div>';
          for (const cf of sem.created_files) {
            html += `<div class="ms-sample" style="margin:3px 0;">
              <div><code>${escapeHtml(cf.path)}</code> <span style="color:var(--text2);">(${(cf.content_length/1000).toFixed(0)}K, .${cf.file_type})</span></div>
              ${cf.preview ? `<div class="ms-preview">${escapeHtml(cf.preview.slice(0, 200))}</div>` : ''}
            </div>`;
          }
        }
        if (sem.edit_chains && sem.edit_chains.length > 0) {
          html += '<div style="margin:8px 0 6px;font-size:12px;font-weight:600;color:#fb923c;">Edit Chains</div>';
          for (const ec of sem.edit_chains) {
            html += `<div style="font-size:12px;margin:3px 0;"><code>${escapeHtml(ec.path)}</code> +${ec.lines_added}/-${ec.lines_deleted}</div>`;
            if (ec.diff_preview) {
              html += `<pre class="ms-pre" style="font-size:10px;max-height:100px;overflow:hidden;">${escapeHtml(ec.diff_preview)}</pre>`;
            }
          }
        }
        if (sem.cross_file_refs && sem.cross_file_refs.length > 0) {
          html += '<div style="margin:8px 0 6px;font-size:12px;font-weight:600;color:var(--text2);">Cross-File References</div>';
          for (const cr of sem.cross_file_refs) {
            html += `<div style="font-size:11px;margin:2px 0;">${escapeHtml(cr.source_file)} \u2192 ${escapeHtml(cr.target_file)} (${escapeHtml(cr.reference_type)})</div>`;
          }
        }
        if (sem.created_filenames && sem.created_filenames.length > 0) {
          html += `<div style="margin:8px 0 4px;font-size:12px;"><strong>Filenames:</strong> ${sem.created_filenames.map(f => `<code>${escapeHtml(f.split('/').pop())}</code>`).join(', ')}</div>`;
        }
        if (sem.llm_encoding) {
          html += '<div style="margin:8px 0 4px;font-size:12px;font-weight:600;color:var(--accent);">LLM Encoding</div>';
          for (const [k, v] of Object.entries(sem.llm_encoding)) {
            const val = Array.isArray(v) ? v.join('; ') : String(v);
            html += `<div style="font-size:11px;margin:2px 0;"><span style="color:var(--text2);">${escapeHtml(k)}:</span> ${escapeHtml(val)}</div>`;
          }
        }
        html += '</div></div>';
      }

      html += '</div>';
    }

    html += '</div></div>';
  }

  // Session Evidence Table (legacy — kept)
  if (ms.sessions && ms.sessions.length > 0) {
    html += `<div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h3>Session Evidence (${ms.sessions.length} sessions)</h3>
        <button class="ms-toggle" data-target="ms-sessions-table">\u25b6 Expand</button>
      </div>
      <div id="ms-sessions-table" style="display:none;margin-top:12px;">
      <table class="ms-table" style="font-size:12px;">
        <thead><tr>
          <th>Task</th><th>Reads</th><th>Search</th><th>Output</th>
          <th>Files</th><th>Dirs</th><th>Edits</th><th>Key Outputs</th>
        </tr></thead><tbody>`;
    for (const s of ms.sessions) {
      const tid = s.is_deviant ? `${s.task_id}<span style="color:var(--red);">*</span>` : s.task_id;
      const outK = s.output_chars >= 1000 ? `${(s.output_chars / 1000).toFixed(1)}K` : s.output_chars;
      const dirs = s.dirs_created > 0 ? `${s.dirs_created}/${s.max_depth}` : '\u2014';
      const edits = s.edits > 0 ? s.edits : '\u2014';
      const fnames = (s.filenames || []).map(f => escapeHtml(f.length > 25 ? f.slice(0, 22) + '\u2026' : f)).join(', ');
      html += `<tr${s.is_deviant ? ' style="background:rgba(255,100,100,0.08);"' : ''}>
        <td>${tid}</td><td>${s.reads}</td><td>${s.search_pct}%</td>
        <td>${outK}</td><td>${s.files_created}</td><td>${dirs}</td>
        <td>${edits}</td><td style="font-size:11px;">${fnames}</td>
      </tr>`;
    }
    html += '</tbody></table></div></div>';
  }

  container.innerHTML = html;
  attachToggleHandlers(container);

  // Engram tab switching
  setTimeout(() => {
    document.querySelectorAll('#msEngramTabs .ms-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#msEngramTabs .ms-tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.ms-engram-content').forEach(c => c.style.display = 'none');
        btn.classList.add('active');
        document.getElementById('msEngram' + btn.dataset.idx).style.display = 'block';
      });
    });
  }, 0);
}

function renderMarkdown(text) {
  if (!text) return '';
  // Simple markdown to HTML conversion
  let html = escapeHtml(text);
  // Headers
  html = html.replace(/^## (.+)$/gm, '<h2 style="font-size:16px;margin:12px 0 6px;">$1</h2>');
  html = html.replace(/^### (.+)$/gm, '<h3 style="font-size:14px;margin:10px 0 4px;color:var(--accent);">$1</h3>');
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Table header
  html = html.replace(/^\|(.+)\|$/gm, (match) => {
    const cells = match.split('|').filter(c => c.trim());
    return '<tr>' + cells.map(c => `<td style="padding:2px 6px;border-bottom:1px solid var(--border);">${c.trim()}</td>`).join('') + '</tr>';
  });
  // Remove separator rows (|---|---|)
  html = html.replace(/<tr>(<td[^>]*>-+<\/td>)+<\/tr>/g, '');
  // Wrap tables
  html = html.replace(/(<tr>[\s\S]*?<\/tr>(?:\s*<tr>[\s\S]*?<\/tr>)*)/g,
    '<table class="ms-table" style="font-size:12px;margin:4px 0;">$1</table>');
  // List items
  html = html.replace(/^  - (.+)$/gm, '<div style="margin-left:16px;">\u2022 $1</div>');
  // Line breaks
  html = html.replace(/\n\n/g, '<div style="height:8px;"></div>');
  html = html.replace(/\n/g, '<br>');
  return html;
}

function attachToggleHandlers(container) {
  container.querySelectorAll('.ms-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = document.getElementById(btn.dataset.target);
      if (!target) return;
      const isHidden = target.style.display === 'none';
      target.style.display = isHidden ? 'block' : 'none';
      btn.textContent = isHidden ? '\u25bc Collapse' : '\u25b6 Expand';
    });
  });
}

// ============================================================
// Generic renderers (for non-FileGramOS methods)
// ============================================================

function renderPerTrajectory(data, keys) {
  // Show task-by-task timeline
  const mainKey = keys[0]; // e.g., 'features' or 'narratives'
  const items = data[mainKey].items;
  const count = data[mainKey].count;

  let html = `<div class="card"><h3>Per-Trajectory Data (${count} trajectories)</h3>`;
  html += '<div class="ms-tabs" id="msTrajTabs">';

  for (let i = 0; i < Math.min(count, 20); i++) {
    html += `<button class="ms-tab-btn ${i === 0 ? 'active' : ''}" data-idx="${i}">T-${String(i + 1).padStart(2, '0')}</button>`;
  }
  html += '</div>';

  // Content for each trajectory
  for (let i = 0; i < Math.min(count, 20); i++) {
    html += `<div class="ms-traj-content" id="msTraj${i}" style="display:${i === 0 ? 'block' : 'none'};">`;
    for (const key of keys) {
      const item = data[key].items[i];
      if (item === undefined) continue;
      html += `<h4 style="margin:12px 0 6px;font-size:13px;color:var(--text2);">${key}</h4>`;
      html += renderValue(item, key);
    }
    html += '</div>';
  }

  html += '</div>';

  // Add tab switching after render
  setTimeout(() => {
    document.querySelectorAll('#msTrajTabs .ms-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#msTrajTabs .ms-tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.ms-traj-content').forEach(c => c.style.display = 'none');
        btn.classList.add('active');
        document.getElementById('msTraj' + btn.dataset.idx).style.display = 'block';
      });
    });
  }, 0);

  return html;
}

function renderConsolidated(data, keys) {
  let html = '';
  for (const key of keys) {
    const val = data[key];
    const id = 'ms-cons-' + key;
    html += `<div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h3>${key} <span style="font-weight:normal;color:var(--text2);font-size:12px;">(${val.count} items)</span></h3>
        <button class="ms-toggle" data-target="${id}">\u25b6 Expand</button>
      </div>
      <div id="${id}" style="display:none;margin-top:12px;">`;

    // Show items as a scrollable list
    html += '<div class="ms-items-list">';
    for (let i = 0; i < val.items.length; i++) {
      const item = val.items[i];
      if (typeof item === 'string') {
        html += `<div class="ms-item"><span class="ms-item-idx">${i + 1}</span><span class="ms-item-text">${escapeHtml(item)}</span></div>`;
      } else {
        html += `<div class="ms-item"><span class="ms-item-idx">${i + 1}</span>`;
        html += renderValue(item, key);
        html += '</div>';
      }
    }
    html += '</div></div></div>';
  }
  return html;
}

function renderValue(val, context) {
  if (val === null || val === undefined) return '<span style="color:var(--text2)">null</span>';
  if (typeof val === 'string') {
    // In full mode, always show everything (with expand for very long)
    if (val.length > 200) {
      const id = 'ms-val-' + Math.random().toString(36).slice(2, 8);
      if (msFullMode) {
        // Full mode: show all, but collapsible for readability
        return `<div class="ms-narrative">
          <div class="ms-narrative-preview">${escapeHtml(val.slice(0, 300))}...</div>
          <button class="ms-toggle" data-target="${id}" style="font-size:11px;margin-top:4px;">\u25b6 Show all (${val.length} chars)</button>
          <pre id="${id}" style="display:none;" class="ms-pre">${escapeHtml(val)}</pre>
        </div>`;
      }
      return `<div class="ms-narrative">
        <div class="ms-narrative-preview">${escapeHtml(val.slice(0, 200))}...</div>
        <button class="ms-toggle" data-target="${id}" style="font-size:11px;margin-top:4px;">\u25b6 Expand</button>
        <pre id="${id}" style="display:none;" class="ms-pre">${escapeHtml(val)}</pre>
      </div>`;
    }
    return `<span class="ms-str">${escapeHtml(val)}</span>`;
  }
  if (typeof val === 'number' || typeof val === 'boolean') {
    return `<span class="ms-num">${val}</span>`;
  }
  if (Array.isArray(val)) {
    if (val.length === 0) return '<span style="color:var(--text2)">[]</span>';
    let html = '<div class="ms-array">';
    for (const item of val) {
      html += `<div class="ms-array-item">${renderValue(item, context)}</div>`;
    }
    html += '</div>';
    return html;
  }
  if (typeof val === 'object') {
    return renderObjectAsTable(val, context);
  }
  return escapeHtml(String(val));
}

function renderObjectAsTable(obj, context) {
  const keys = Object.keys(obj);
  if (keys.length === 0) return '<span style="color:var(--text2)">{}</span>';

  // For feature objects with nested stats, render as compact key-value
  let html = '<div class="ms-obj">';
  for (const [k, v] of Object.entries(obj)) {
    if (typeof v === 'object' && v !== null && !Array.isArray(v)) {
      // Nested object (like reading_strategy stats)
      html += `<div class="ms-obj-section">
        <div class="ms-obj-key">${k}</div>
        <div class="ms-obj-nested">`;
      for (const [kk, vv] of Object.entries(v)) {
        const displayVal = typeof vv === 'number' ? (Number.isInteger(vv) ? vv : vv.toFixed(3)) : vv;
        html += `<span class="ms-kv"><code>${kk}</code>: <strong>${displayVal}</strong></span>`;
      }
      html += '</div></div>';
    } else if (typeof v === 'string' && v.length > 100) {
      html += `<div class="ms-obj-section">
        <div class="ms-obj-key">${k}</div>
        <div class="ms-obj-val ms-str-long">${escapeHtml(v)}</div>
      </div>`;
    } else {
      const displayVal = typeof v === 'number' && !Number.isInteger(v) ? v.toFixed(3) : v;
      html += `<span class="ms-kv"><code>${k}</code>: <strong>${displayVal === null ? 'null' : displayVal}</strong></span>`;
    }
  }
  html += '</div>';
  return html;
}

function escapeHtml(str) {
  if (typeof str !== 'string') return String(str);
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
