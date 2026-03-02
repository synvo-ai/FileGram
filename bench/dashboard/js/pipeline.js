// ============================================================
// FileGramOS Pipeline Walkthrough
// ============================================================

let plSources = [];
let plCurrentSource = null;
let plProfiles = [];
let plData = null;

// Load sources on tab activation
document.querySelector('[data-tab="pipeline"]').addEventListener('click', async () => {
  if (plSources.length === 0) await loadPLSources();
});

async function loadPLSources() {
  try {
    const resp = await fetch('/api/memory-store/sources');
    plSources = await resp.json();
    const sel = document.getElementById('plSource');
    if (plSources.length === 0) {
      sel.innerHTML = '<option value="">No caches found</option>';
      return;
    }
    sel.innerHTML = plSources.map(s =>
      `<option value="${s.slug}">${s.slug} (${s.profiles} profiles)</option>`
    ).join('');
    sel.addEventListener('change', () => {
      plCurrentSource = sel.value;
      loadPLProfiles();
    });
    plCurrentSource = plSources[0].slug;
    await loadPLProfiles();
  } catch (e) {
    document.getElementById('plContent').innerHTML =
      '<div class="card"><p style="color:var(--red)">Failed to load sources</p></div>';
  }
}

async function loadPLProfiles() {
  try {
    const sourceParam = plCurrentSource ? `?source=${plCurrentSource}` : '';
    const resp = await fetch(`/api/memory-store/index${sourceParam}`);
    const index = await resp.json();
    // Filter profiles that have filegramos or filegramos_simple
    plProfiles = Object.keys(index).filter(p => {
      const methods = Object.keys(index[p] || {});
      return methods.includes('filegramos') || methods.includes('filegramos_simple');
    }).sort();

    const sel = document.getElementById('plProfile');
    if (plProfiles.length === 0) {
      sel.innerHTML = '<option value="">No FileGramOS profiles found</option>';
      return;
    }
    sel.innerHTML = plProfiles.map(p => `<option value="${p}">${p}</option>`).join('');
    sel.addEventListener('change', loadPLData);
    await loadPLData();
  } catch (e) {
    document.getElementById('plContent').innerHTML =
      '<div class="card"><p style="color:var(--red)">Failed to load profiles</p></div>';
  }
}

async function loadPLData() {
  const profile = document.getElementById('plProfile').value;
  if (!profile) return;

  const info = document.getElementById('plInfo');
  info.textContent = 'Loading pipeline data...';

  try {
    const params = new URLSearchParams();
    if (plCurrentSource) params.set('source', plCurrentSource);
    const qs = params.toString();
    const resp = await fetch(`/api/pipeline/${profile}${qs ? '?' + qs : ''}`);
    plData = await resp.json();
    info.textContent = `Method: ${plData.method} | Example: ${plData.example_task_id}`;
    renderPipeline();
  } catch (e) {
    info.textContent = 'Error';
    document.getElementById('plContent').innerHTML =
      `<div class="card"><p style="color:var(--red)">Failed to load: ${e.message}</p></div>`;
  }
}

function renderPipeline() {
  const container = document.getElementById('plContent');
  const d = plData;
  const steps = d.steps;
  let html = '';

  // Pipeline overview diagram
  html += `<div class="card pl-overview">
    <h3 style="margin-bottom:12px;">Pipeline Architecture</h3>
    <div class="pl-flow">
      <div class="pl-flow-step pl-step-1" data-step="1">
        <div class="pl-flow-icon">1</div>
        <div class="pl-flow-label">Raw Events</div>
        <div class="pl-flow-sub">events.json</div>
      </div>
      <div class="pl-flow-arrow">\u2192</div>
      <div class="pl-flow-step pl-step-2" data-step="2">
        <div class="pl-flow-icon">2</div>
        <div class="pl-flow-label">Normalize</div>
        <div class="pl-flow-sub">EventNormalizer</div>
      </div>
      <div class="pl-flow-arrow">\u2192</div>
      <div class="pl-flow-step pl-step-3" data-step="3">
        <div class="pl-flow-icon">3</div>
        <div class="pl-flow-label">Extract</div>
        <div class="pl-flow-sub">FeatureExtractor</div>
      </div>
      <div class="pl-flow-arrow">\u2192</div>
      <div class="pl-flow-step pl-step-4" data-step="4">
        <div class="pl-flow-icon">4</div>
        <div class="pl-flow-label">Semantic</div>
        <div class="pl-flow-sub">Content + Diffs</div>
      </div>
      <div class="pl-flow-arrow">\u2192</div>
      <div class="pl-flow-step pl-step-5" data-step="5">
        <div class="pl-flow-icon">5</div>
        <div class="pl-flow-label">Engram</div>
        <div class="pl-flow-sub">Per-trajectory unit</div>
      </div>
      <div class="pl-flow-arrow">\u2192</div>
      <div class="pl-flow-step pl-step-6" data-step="6">
        <div class="pl-flow-icon">6</div>
        <div class="pl-flow-label">Consolidate</div>
        <div class="pl-flow-sub">MemoryStore</div>
      </div>
      <div class="pl-flow-arrow">\u2192</div>
      <div class="pl-flow-step pl-step-7" data-step="7">
        <div class="pl-flow-icon">7</div>
        <div class="pl-flow-label">Retrieve</div>
        <div class="pl-flow-sub">Rendered Profile</div>
      </div>
    </div>
  </div>`;

  // Step 1: Raw Events
  html += renderStep(1, 'Raw Events (events.json)',
    'Each trajectory session produces a raw JSON event log. These are the atomic behavioral signals captured during the agent\'s execution.',
    () => {
      if (!steps.step1_raw_events || steps.step1_raw_events.length === 0) {
        return '<p style="color:var(--text2)">No raw events available for this trajectory</p>';
      }
      let h = `<div style="font-size:12px;color:var(--text2);margin-bottom:8px;">Showing first ${steps.step1_raw_events.length} behavioral events from trajectory ${escapeHtml(d.example_task_id)}</div>`;
      for (let i = 0; i < steps.step1_raw_events.length; i++) {
        const evt = steps.step1_raw_events[i];
        h += `<div class="pl-event-card">
          <span class="pl-event-idx">${i + 1}</span>
          <span class="traj-event-type evt-${evt.event_type}">${evt.event_type}</span>
          <code style="font-size:11px;margin-left:8px;">${escapeHtml(evt.file_path || evt.dir_path || evt.source_file || '')}</code>
        </div>`;
        // Show key fields
        const skipKeys = new Set(['event_type', 'file_path', 'timestamp', 'profile_id', 'iteration_number']);
        const fields = Object.entries(evt).filter(([k]) => !skipKeys.has(k) && evt[k] !== '' && evt[k] !== 0 && evt[k] !== false && evt[k] !== null);
        if (fields.length > 0) {
          h += '<div class="pl-event-fields">';
          for (const [k, v] of fields.slice(0, 6)) {
            const sv = typeof v === 'string' && v.length > 60 ? v.slice(0, 57) + '...' : v;
            h += `<span class="ms-kv"><code>${k}</code>: <strong>${typeof sv === 'object' ? JSON.stringify(sv).slice(0, 50) : sv}</strong></span>`;
          }
          h += '</div>';
        }
      }
      return h;
    });

  // Step 2: Normalized Events
  html += renderStep(2, 'EventNormalizer \u2192 NormalizedEvent',
    'The EventNormalizer converts raw dicts into typed NormalizedEvent objects. It validates event types, unifies field names (e.g. source_path \u2192 source_file), resolves media references, and skips non-behavioral events.',
    () => {
      if (!steps.step2_normalized || steps.step2_normalized.length === 0) {
        return '<p style="color:var(--text2)">No normalized events available</p>';
      }
      let h = '<div class="pl-compare-grid">';
      h += '<div class="pl-compare-col"><h4 style="font-size:13px;color:var(--text2);margin-bottom:8px;">Raw Dict (before)</h4>';
      for (let i = 0; i < Math.min(steps.step1_raw_events.length, steps.step2_normalized.length, 4); i++) {
        const raw = steps.step1_raw_events[i];
        h += `<pre class="ms-pre" style="font-size:10px;max-height:120px;">${escapeHtml(JSON.stringify(raw, null, 2))}</pre>`;
      }
      h += '</div>';
      h += '<div class="pl-compare-col"><h4 style="font-size:13px;color:var(--accent);margin-bottom:8px;">NormalizedEvent (after)</h4>';
      for (let i = 0; i < Math.min(steps.step2_normalized.length, 4); i++) {
        const ne = steps.step2_normalized[i];
        // Only show non-default fields
        const filtered = {};
        for (const [k, v] of Object.entries(ne)) {
          if (v !== '' && v !== 0 && v !== false && v !== null && v !== 1) filtered[k] = v;
          else if (k === 'event_type') filtered[k] = v;
        }
        h += `<pre class="ms-pre" style="font-size:10px;max-height:120px;">${escapeHtml(JSON.stringify(filtered, null, 2))}</pre>`;
      }
      h += '</div></div>';

      h += `<div class="pl-note">
        <strong>Key transformations:</strong>
        <ul style="margin:4px 0 0 16px;font-size:12px;line-height:1.6;">
          <li><code>event_type</code> string \u2192 <code>ConsumerEventType</code> enum (14 values)</li>
          <li><code>source_path</code> / <code>dest_path</code> \u2192 unified <code>source_file</code> / <code>target_file</code></li>
          <li><code>_resolved_content</code> \u2192 <code>resolved_content</code> (media refs resolved)</li>
          <li>Non-behavioral events (tool_call, llm_response, etc.) silently filtered out</li>
        </ul>
      </div>`;
      return h;
    });

  // Step 3: Feature Extraction
  html += renderStep(3, 'FeatureExtractor \u2192 Procedural Features',
    'The FeatureExtractor deterministically extracts 11 feature groups from NormalizedEvents. Each group corresponds to profile dimensions (A\u2013F). No LLM involved \u2014 pure computation.',
    () => {
      const feats = steps.step3_features;
      if (!feats || Object.keys(feats).length === 0) {
        return '<p style="color:var(--text2)">No features available</p>';
      }
      let h = `<div style="font-size:12px;color:var(--text2);margin-bottom:8px;">Extracted from trajectory ${escapeHtml(d.example_task_id)} (${Object.keys(feats).length} feature groups)</div>`;
      h += '<div class="pl-features-grid">';
      for (const [group, stats] of Object.entries(feats)) {
        if (typeof stats !== 'object' || stats === null) continue;
        h += `<div class="pl-feature-group">
          <div class="pl-feature-group-name">${escapeHtml(group)}</div>`;
        for (const [k, v] of Object.entries(stats)) {
          const dv = typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v;
          const isNonZero = (typeof v === 'number' && v > 0) || (typeof v === 'boolean' && v);
          h += `<div class="pl-feature-item${isNonZero ? ' pl-feature-active' : ''}">
            <span class="pl-feature-key">${k}</span>
            <span class="pl-feature-val">${dv}</span>
          </div>`;
        }
        h += '</div>';
      }
      h += '</div>';
      return h;
    });

  // Step 4: Semantic Channel
  html += renderStep(4, 'Semantic Channel Extraction',
    'The semantic channel captures content produced by the user: created files with previews, edit diffs, cross-file references, and filenames. This is what the user CREATED, not just how they navigated.',
    () => {
      const sem = steps.step4_semantic;
      if (!sem) return '<p style="color:var(--text2)">No semantic data available</p>';
      let h = '';

      if (sem.created_files && sem.created_files.length > 0) {
        h += `<h4 style="font-size:13px;color:var(--green);margin-bottom:6px;">Created Files (${sem.created_files.length})</h4>`;
        for (const cf of sem.created_files) {
          h += `<div class="ms-sample"><div><code>${escapeHtml(cf.path)}</code> <span style="color:var(--text2);">(${(cf.content_length/1000).toFixed(0)}K, .${cf.file_type})</span></div>`;
          if (cf.preview) h += `<pre class="ms-pre" style="font-size:10px;max-height:80px;">${escapeHtml(cf.preview)}</pre>`;
          h += '</div>';
        }
      }
      if (sem.edit_chains && sem.edit_chains.length > 0) {
        h += `<h4 style="font-size:13px;color:#fb923c;margin:12px 0 6px;">Edit Chains (${sem.edit_chains.length})</h4>`;
        for (const ec of sem.edit_chains) {
          h += `<div style="font-size:12px;margin:4px 0;"><code>${escapeHtml(ec.path)}</code> +${ec.lines_added}/-${ec.lines_deleted}</div>`;
          if (ec.diff_preview) h += `<pre class="ms-pre" style="font-size:10px;max-height:60px;">${escapeHtml(ec.diff_preview)}</pre>`;
        }
      }
      if (sem.cross_file_refs && sem.cross_file_refs.length > 0) {
        h += `<h4 style="font-size:13px;color:var(--text2);margin:12px 0 6px;">Cross-File References (${sem.cross_file_refs.length})</h4>`;
        for (const cr of sem.cross_file_refs) {
          h += `<div style="font-size:11px;margin:2px 0;">${escapeHtml(cr.source_file)} \u2192 ${escapeHtml(cr.target_file)} (${escapeHtml(cr.reference_type)})</div>`;
        }
      }
      if (sem.created_filenames && sem.created_filenames.length > 0) {
        h += `<h4 style="font-size:13px;margin:12px 0 6px;">All Created Filenames (${sem.created_filenames.length})</h4>`;
        h += '<div class="ms-filename-grid">';
        for (const fn of sem.created_filenames) {
          h += `<span class="ms-filename">${escapeHtml(fn.split('/').pop())}</span>`;
        }
        h += '</div>';
      }
      if (sem.llm_encoding) {
        h += '<h4 style="font-size:13px;color:var(--accent);margin:12px 0 6px;">LLM Behavioral Encoding</h4>';
        h += '<div class="ms-narrative-card">';
        for (const [k, v] of Object.entries(sem.llm_encoding)) {
          const val = Array.isArray(v) ? v.join('; ') : String(v);
          h += `<div style="font-size:12px;margin:3px 0;"><span style="color:var(--text2);">${escapeHtml(k)}:</span> ${escapeHtml(val)}</div>`;
        }
        h += '</div>';
      }
      return h;
    });

  // Step 5: Engram Assembly
  html += renderStep(5, 'EngramEncoder \u2192 Complete Engram',
    'The EngramEncoder combines procedural features, semantic unit, fingerprint vector, and importance score into a single Engram \u2014 the atomic memory unit for one trajectory. Below shows the ACTUAL data stored in the pkl cache.',
    () => {
      const eng = steps.step5_engram;
      if (!eng || !eng.trajectory_id) return '<p style="color:var(--text2)">No engram data available</p>';

      // Header summary
      let h = `<div class="pl-engram-overview">
        <div class="pl-engram-field"><span class="pl-engram-label">trajectory_id</span><span class="pl-engram-value">${escapeHtml(eng.trajectory_id)}</span></div>
        <div class="pl-engram-field"><span class="pl-engram-label">task_id</span><span class="pl-engram-value">${escapeHtml(eng.task_id)}</span></div>
        <div class="pl-engram-field"><span class="pl-engram-label">event_count</span><span class="pl-engram-value">${eng.event_count} total, ${eng.behavioral_event_count} behavioral</span></div>
        <div class="pl-engram-field"><span class="pl-engram-label">importance_score</span><span class="pl-engram-value">${eng.importance_score.toFixed(4)}</span></div>
        <div class="pl-engram-field"><span class="pl-engram-label">is_perturbed</span><span class="pl-engram-value">${eng.is_perturbed}</span></div>
      </div>`;

      // Tab bar for sub-views
      const tabId = 'engram-tab';
      h += `<div class="pl-engram-tabs" style="display:flex;gap:0;margin:12px 0 0;border-bottom:2px solid #333;">
        <button class="pl-engram-tab active" onclick="switchEngramTab(this,'${tabId}','procedural')" style="padding:6px 14px;font-size:12px;cursor:pointer;background:none;border:none;border-bottom:2px solid var(--accent);color:var(--accent);margin-bottom:-2px;">Procedural</button>
        <button class="pl-engram-tab" onclick="switchEngramTab(this,'${tabId}','semantic')" style="padding:6px 14px;font-size:12px;cursor:pointer;background:none;border:none;border-bottom:2px solid transparent;color:var(--text2);margin-bottom:-2px;">Semantic</button>
        <button class="pl-engram-tab" onclick="switchEngramTab(this,'${tabId}','fingerprint')" style="padding:6px 14px;font-size:12px;cursor:pointer;background:none;border:none;border-bottom:2px solid transparent;color:var(--text2);margin-bottom:-2px;">Fingerprint</button>
        <button class="pl-engram-tab" onclick="switchEngramTab(this,'${tabId}','raw')" style="padding:6px 14px;font-size:12px;cursor:pointer;background:none;border:none;border-bottom:2px solid transparent;color:var(--text2);margin-bottom:-2px;">Raw JSON</button>
      </div>`;

      // --- Procedural tab (actual stored dict) ---
      const feats = steps.step3_features || {};
      let procHtml = '<div style="font-size:11px;color:var(--text2);margin-bottom:8px;">Actual <code>engram.procedural</code> dict stored in pkl — deterministic stats extracted from events</div>';
      procHtml += '<div class="pl-features-grid">';
      for (const [group, stats] of Object.entries(feats)) {
        if (typeof stats !== 'object' || stats === null) continue;
        procHtml += `<div class="pl-feature-group"><div class="pl-feature-group-name">${escapeHtml(group)}</div>`;
        for (const [k, v] of Object.entries(stats)) {
          const dv = typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v;
          const isNonZero = (typeof v === 'number' && v > 0) || (typeof v === 'boolean' && v);
          procHtml += `<div class="pl-feature-item${isNonZero ? ' pl-feature-active' : ''}"><span class="pl-feature-key">${k}</span><span class="pl-feature-val">${dv}</span></div>`;
        }
        procHtml += '</div>';
      }
      procHtml += '</div>';

      // --- Semantic tab (actual stored content) ---
      const sem = steps.step4_semantic || {};
      let semHtml = '<div style="font-size:11px;color:var(--text2);margin-bottom:8px;">Actual <code>engram.semantic</code> unit — created files, edit diffs, cross-refs stored in pkl</div>';
      if (sem.created_files && sem.created_files.length > 0) {
        semHtml += `<h4 style="font-size:13px;color:var(--green);margin-bottom:6px;">Created Files (${sem.created_files.length})</h4>`;
        for (const cf of sem.created_files) {
          semHtml += `<div class="ms-sample"><div><code>${escapeHtml(cf.path)}</code> <span style="color:var(--text2);">(${(cf.content_length/1000).toFixed(1)}K chars, .${cf.file_type})</span></div>`;
          if (cf.preview) semHtml += `<pre class="ms-pre" style="font-size:10px;max-height:100px;">${escapeHtml(cf.preview)}</pre>`;
          semHtml += '</div>';
        }
      }
      if (sem.edit_chains && sem.edit_chains.length > 0) {
        semHtml += `<h4 style="font-size:13px;color:#fb923c;margin:12px 0 6px;">Edit Chains (${sem.edit_chains.length})</h4>`;
        for (const ec of sem.edit_chains) {
          semHtml += `<div style="font-size:12px;margin:4px 0;"><code>${escapeHtml(ec.path)}</code> <span style="color:var(--green);">+${ec.lines_added}</span>/<span style="color:var(--red);">-${ec.lines_deleted}</span></div>`;
          if (ec.diff_preview) semHtml += `<pre class="ms-pre" style="font-size:10px;max-height:80px;">${escapeHtml(ec.diff_preview)}</pre>`;
        }
      }
      if (sem.cross_file_refs && sem.cross_file_refs.length > 0) {
        semHtml += `<h4 style="font-size:13px;color:var(--text2);margin:12px 0 6px;">Cross-File References (${sem.cross_file_refs.length})</h4>`;
        for (const cr of sem.cross_file_refs) {
          semHtml += `<div style="font-size:11px;margin:2px 0;">${escapeHtml(cr.source_file)} \u2192 ${escapeHtml(cr.target_file)} <span style="color:var(--text2);">(${escapeHtml(cr.reference_type)})</span></div>`;
        }
      }
      if (sem.created_filenames && sem.created_filenames.length > 0) {
        semHtml += `<h4 style="font-size:13px;margin:12px 0 6px;">All Created Filenames (${sem.created_filenames.length})</h4>`;
        semHtml += '<div class="ms-filename-grid">';
        for (const fn of sem.created_filenames) {
          semHtml += `<span class="ms-filename">${escapeHtml(fn.split('/').pop())}</span>`;
        }
        semHtml += '</div>';
      }
      if (sem.dir_structure_diff && sem.dir_structure_diff.length > 0) {
        semHtml += `<h4 style="font-size:13px;margin:12px 0 6px;">Directory Structure Diff</h4>`;
        semHtml += '<div style="font-size:11px;font-family:monospace;">' + sem.dir_structure_diff.map(d => escapeHtml(d)).join('<br>') + '</div>';
      }
      if (sem.llm_encoding) {
        semHtml += '<h4 style="font-size:13px;color:var(--accent);margin:12px 0 6px;">LLM Behavioral Encoding</h4>';
        semHtml += `<pre class="ms-pre" style="font-size:10px;max-height:200px;">${escapeHtml(JSON.stringify(sem.llm_encoding, null, 2))}</pre>`;
      }
      if (Object.keys(sem).length === 0) semHtml += '<p style="color:var(--text2);">No semantic data for this trajectory</p>';

      // --- Fingerprint tab ---
      let fpHtml = '<div style="font-size:11px;color:var(--text2);margin-bottom:8px;">Actual <code>engram.fingerprint</code> — fixed-size vector computed from procedural features, used for episodic distance/deviation analysis</div>';
      if (eng.fingerprint && eng.fingerprint.length > 0) {
        const maxVal = Math.max(...eng.fingerprint.map(Math.abs), 0.001);
        fpHtml += '<div class="ms-fingerprint-bar" style="height:50px;margin-bottom:8px;">';
        for (let i = 0; i < eng.fingerprint.length; i++) {
          const v = eng.fingerprint[i];
          const height = Math.abs(v / maxVal) * 100;
          const color = v >= 0 ? 'var(--accent)' : 'var(--red)';
          fpHtml += `<div class="ms-fp-bar" style="height:${Math.max(height, 2)}%;background:${color};" title="dim ${i}: ${v.toFixed(4)}"></div>`;
        }
        fpHtml += '</div>';
        fpHtml += `<div style="font-size:11px;color:var(--text2);margin-bottom:4px;">${eng.fingerprint_dims} dimensions</div>`;
        fpHtml += `<pre class="ms-pre" style="font-size:10px;max-height:120px;">[${eng.fingerprint.map(v => v.toFixed(4)).join(', ')}]</pre>`;
      } else {
        fpHtml += '<p style="color:var(--text2);">No fingerprint available</p>';
      }

      // --- Raw JSON tab ---
      const rawObj = {
        trajectory_id: eng.trajectory_id,
        task_id: eng.task_id,
        event_count: eng.event_count,
        behavioral_event_count: eng.behavioral_event_count,
        importance_score: eng.importance_score,
        is_perturbed: eng.is_perturbed,
        fingerprint: eng.fingerprint ? `[${eng.fingerprint_dims} floats]` : null,
        procedural: feats,
        semantic: {
          created_files: (sem.created_files || []).map(cf => ({ path: cf.path, content_length: cf.content_length, file_type: cf.file_type, preview: cf.preview ? cf.preview.slice(0, 100) + '...' : null })),
          edit_chains: (sem.edit_chains || []).map(ec => ({ path: ec.path, lines_added: ec.lines_added, lines_deleted: ec.lines_deleted })),
          cross_file_refs: sem.cross_file_refs || [],
          created_filenames: sem.created_filenames || [],
          dir_structure_diff: sem.dir_structure_diff || [],
          llm_encoding: sem.llm_encoding || null,
        },
      };
      let rawHtml = '<div style="font-size:11px;color:var(--text2);margin-bottom:8px;">Full Engram structure as stored in <code>filegramos.pkl</code> (content previews truncated)</div>';
      rawHtml += `<pre class="ms-pre" style="font-size:10px;max-height:500px;overflow:auto;">${escapeHtml(JSON.stringify(rawObj, null, 2))}</pre>`;

      h += `<div id="${tabId}-procedural" class="pl-engram-panel" style="margin-top:12px;">${procHtml}</div>`;
      h += `<div id="${tabId}-semantic" class="pl-engram-panel" style="margin-top:12px;display:none;">${semHtml}</div>`;
      h += `<div id="${tabId}-fingerprint" class="pl-engram-panel" style="margin-top:12px;display:none;">${fpHtml}</div>`;
      h += `<div id="${tabId}-raw" class="pl-engram-panel" style="margin-top:12px;display:none;">${rawHtml}</div>`;

      h += `<div class="pl-note" style="margin-top:12px;">
        <strong>Engram = Procedural + Semantic + Fingerprint</strong><br>
        One Engram per trajectory. The tabs above show the actual data stored in the pkl cache. Multiple Engrams are then consolidated into a MemoryStore (Step 6).
      </div>`;
      return h;
    });

  // Step 6: Consolidation
  html += renderStep(6, 'EngramConsolidator \u2192 MemoryStore (3 Channels)',
    `The EngramConsolidator merges ${(d.all_engrams || []).length} Engrams into a unified MemoryStore with three channels. Deviant trajectories are detected via fingerprint distance and down-weighted.`,
    () => {
      const cons = steps.step6_consolidation;
      if (!cons) return '<p style="color:var(--text2)">No consolidation data available</p>';

      let h = `<div style="font-size:13px;margin-bottom:12px;"><strong>${cons.n_engrams} Engrams consolidated</strong></div>`;

      // Channel 1
      h += `<div class="pl-channel-card">
        <h4 style="color:var(--accent);margin:0 0 8px;">Channel 1: Procedural</h4>
        <p style="font-size:12px;color:var(--text2);margin-bottom:8px;">Aggregated statistics across all trajectories + LLM-generated dimension classifications and behavioral patterns.</p>
        <div style="font-size:12px;margin-bottom:6px;"><strong>Dimensions:</strong> ${cons.channel_1_procedural.aggregate_dimensions.join(', ')}</div>
        <div style="font-size:12px;margin-bottom:6px;"><strong>Classifications (${cons.channel_1_procedural.n_classifications}):</strong></div>`;
      for (const cls of (cons.channel_1_procedural.classifications || []).slice(0, 6)) {
        h += `<div class="ms-classification">${escapeHtml(cls)}</div>`;
      }
      if (cons.channel_1_procedural.patterns && cons.channel_1_procedural.patterns.length > 0) {
        h += `<div style="font-size:12px;margin:8px 0 4px;"><strong>Patterns (${cons.channel_1_procedural.n_patterns}):</strong></div>`;
        for (const pat of cons.channel_1_procedural.patterns.slice(0, 4)) {
          h += `<div class="ms-pattern">${escapeHtml(pat)}</div>`;
        }
      }
      h += '</div>';

      // Channel 2
      h += `<div class="pl-channel-card">
        <h4 style="color:var(--green);margin:0 0 8px;">Channel 2: Semantic</h4>
        <p style="font-size:12px;color:var(--text2);margin-bottom:8px;">Stratified content samples, merged filenames, directory structures, and LLM behavioral narratives.</p>
        <div style="font-size:12px;">
          <strong>${cons.channel_2_semantic.n_representative_samples}</strong> representative samples |
          <strong>${cons.channel_2_semantic.n_filenames}</strong> unique filenames |
          <strong>${cons.channel_2_semantic.n_directories}</strong> directories |
          <strong>${cons.channel_2_semantic.n_llm_narratives}</strong> LLM narratives
        </div>`;
      if (cons.channel_2_semantic.filenames_sample && cons.channel_2_semantic.filenames_sample.length > 0) {
        h += '<div class="ms-filename-grid" style="margin-top:6px;">';
        for (const fn of cons.channel_2_semantic.filenames_sample) {
          h += `<span class="ms-filename">${escapeHtml(fn.split('/').pop())}</span>`;
        }
        h += '</div>';
      }
      h += '</div>';

      // Channel 3
      h += `<div class="pl-channel-card">
        <h4 style="color:#fbbf24;margin:0 0 8px;">Channel 3: Episodic</h4>
        <p style="font-size:12px;color:var(--text2);margin-bottom:8px;">Fingerprint-based consistency analysis: centroid computation, per-session deviation scoring, and absence detection.</p>
        <div style="font-size:12px;">
          <strong>${cons.channel_3_episodic.centroid_dims}</strong>-dim centroid |
          <strong>${cons.channel_3_episodic.n_deviant}</strong> deviant sessions |
          <strong>${cons.channel_3_episodic.n_absence_flags}</strong> absence flags
        </div>`;

      // Per-session distances bar chart
      const dists = cons.channel_3_episodic.per_session_distances || {};
      const devFlags = cons.channel_3_episodic.deviation_flags || {};
      if (Object.keys(dists).length > 0) {
        const maxDist = Math.max(...Object.values(dists), 0.001);
        h += '<div style="margin-top:8px;"><strong style="font-size:12px;">Per-Session Distance</strong>';
        h += '<div class="pl-dist-bars">';
        for (const [tid, dist] of Object.entries(dists)) {
          const shortTid = tid.split('_').pop();
          const pct = (dist / maxDist) * 100;
          const isDev = devFlags[tid];
          const color = isDev ? 'var(--red)' : 'var(--accent)';
          h += `<div class="pl-dist-bar-wrap" title="${tid}: ${dist.toFixed(3)}">
            <div class="pl-dist-bar" style="width:${pct}%;background:${color};"></div>
            <span class="pl-dist-label">${shortTid} (${dist.toFixed(2)})${isDev ? ' *' : ''}</span>
          </div>`;
        }
        h += '</div></div>';
      }

      if (cons.channel_3_episodic.absence_flags && cons.channel_3_episodic.absence_flags.length > 0) {
        h += '<div style="margin-top:8px;font-size:12px;"><strong>Never observed:</strong> ' +
          cons.channel_3_episodic.absence_flags.join('; ') + '</div>';
      }
      h += '</div>';

      return h;
    });

  // Step 7: Rendered Profile
  html += renderStep(7, 'QueryAdaptiveRetriever \u2192 Rendered Profile',
    'The final step: QueryAdaptiveRetriever composes the three channels into a clean natural-language behavioral profile. This is what gets fed to the LLM for profile inference.',
    () => {
      const text = steps.step7_rendered;
      if (!text) return '<p style="color:var(--text2)">No rendered profile available</p>';
      return `<div class="ms-rendered-profile">${renderMarkdown(text)}</div>`;
    });

  // All engrams overview
  if (d.all_engrams && d.all_engrams.length > 0) {
    html += `<div class="card">
      <h3>All Engrams Overview (${d.all_engrams.length} trajectories)</h3>
      <table class="ms-table" style="font-size:12px;margin-top:8px;">
        <thead><tr>
          <th>Task</th><th>Events</th><th>Behavioral</th><th>Importance</th><th>Perturbed</th><th>Deviant</th>
        </tr></thead><tbody>`;
    for (const eng of d.all_engrams) {
      const devStyle = eng.is_deviant ? ' style="background:rgba(255,100,100,0.08);"' : '';
      html += `<tr${devStyle}>
        <td>${escapeHtml(eng.task_id)}</td>
        <td>${eng.event_count}</td>
        <td>${eng.behavioral_event_count}</td>
        <td>${eng.importance_score.toFixed(4)}</td>
        <td>${eng.is_perturbed ? '<span class="ms-badge ms-badge-orange">Yes</span>' : '\u2014'}</td>
        <td>${eng.is_deviant ? '<span class="ms-badge ms-badge-orange">Yes</span>' : '\u2014'}</td>
      </tr>`;
    }
    html += '</tbody></table></div>';
  }

  container.innerHTML = html;
  attachToggleHandlers(container);
}

function renderStep(num, title, description, contentFn) {
  const id = `pl-step-${num}-content`;
  return `<div class="card pl-step-card" data-step="${num}">
    <div class="pl-step-header" onclick="toggleStep(this, '${id}')">
      <div class="pl-step-num">${num}</div>
      <div>
        <h3 style="margin:0;font-size:14px;">${title}</h3>
        <p style="margin:4px 0 0;font-size:12px;color:var(--text2);line-height:1.5;">${description}</p>
      </div>
      <span class="pl-step-arrow">\u25b6</span>
    </div>
    <div id="${id}" class="pl-step-body" style="display:none;">
      ${contentFn()}
    </div>
  </div>`;
}

function toggleStep(header, targetId) {
  const body = document.getElementById(targetId);
  if (!body) return;
  const isHidden = body.style.display === 'none';
  body.style.display = isHidden ? 'block' : 'none';
  const arrow = header.querySelector('.pl-step-arrow');
  if (arrow) arrow.textContent = isHidden ? '\u25bc' : '\u25b6';
}

function switchEngramTab(btn, prefix, tabName) {
  // Hide all panels
  const panels = document.querySelectorAll(`[id^="${prefix}-"]`);
  panels.forEach(p => { if (p.classList.contains('pl-engram-panel')) p.style.display = 'none'; });
  // Show target panel
  const target = document.getElementById(`${prefix}-${tabName}`);
  if (target) target.style.display = 'block';
  // Update tab styles
  const tabs = btn.parentElement.querySelectorAll('.pl-engram-tab');
  tabs.forEach(t => {
    t.style.borderBottomColor = 'transparent';
    t.style.color = 'var(--text2)';
    t.classList.remove('active');
  });
  btn.style.borderBottomColor = 'var(--accent)';
  btn.style.color = 'var(--accent)';
  btn.classList.add('active');
}
