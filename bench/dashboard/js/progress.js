// ============================================================
// Progress Monitor
// ============================================================
let progressData = null;
let progressTimer = null;

async function fetchProgress() {
  try {
    const resp = await fetch('/api/progress');
    if (!resp.ok) return;
    progressData = await resp.json();
    renderProgress();
  } catch(e) {
    console.error('Progress fetch error:', e);
  }
}

function renderProgress() {
  const container = document.getElementById('progressContainer');
  if (!container || !progressData) return;

  const { done, profiles, profile_ids, tasks, total_done, total_target, active_processes, perturbed } = progressData;
  const pct = (total_done / total_target * 100).toFixed(1);

  let html = '';

  // Task type legend + perturbation legend
  const typeColors = {understand:'#22d3ee', create:'#4ade80', organize:'#fbbf24', synthesize:'#a78bfa', iterate:'#fb923c', maintain:'#f87171'};
  html += `<div style="display:flex;gap:16px;margin-bottom:12px;align-items:center;flex-wrap:wrap;">
    <span style="font-size:12px;color:var(--text2);font-weight:600;">Task Types:</span>`;
  for (const [type, color] of Object.entries(typeColors)) {
    html += `<span style="font-size:12px;display:flex;align-items:center;gap:4px;">
      <span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${color};"></span>
      <span style="color:${color};font-weight:600;">${type}</span>
    </span>`;
  }
  html += `<span style="margin-left:12px;font-size:12px;color:var(--text2);font-weight:600;">|</span>
    <span style="font-size:12px;display:flex;align-items:center;gap:4px;">
      <span style="color:var(--green);font-weight:700;">\u2713</span>
      <span style="color:var(--text2);">standard</span>
    </span>
    <span style="font-size:12px;display:flex;align-items:center;gap:4px;">
      <span style="color:#f87171;font-weight:700;">\u2713</span>
      <span style="color:var(--text2);">perturbed</span>
    </span>
    <span style="font-size:12px;display:flex;align-items:center;gap:4px;">
      <span style="color:rgba(248,113,113,.4);font-weight:700;">\u25C6</span>
      <span style="color:var(--text2);">perturbed (pending)</span>
    </span>`;
  html += '</div>';

  // Overall progress bar
  html += `<div class="progress-bar-wrap">
    <div class="progress-bar-outer">
      <div class="progress-bar-inner" style="width:${pct}%"></div>
      <div class="progress-bar-label">${total_done} / ${total_target} (${pct}%)</div>
    </div>
  </div>`;

  // Summary cards: p1-p10 vs p11-p20 + standard vs perturbed
  const group1 = profiles.filter(p => parseInt(p.replace('p','')) <= 10);
  const group2 = profiles.filter(p => parseInt(p.replace('p','')) > 10);
  const g1Done = group1.reduce((s, p) => s + (done[p] || []).length, 0);
  const g2Done = group2.reduce((s, p) => s + (done[p] || []).length, 0);
  const g1Total = group1.length * tasks.length;
  const g2Total = group2.length * tasks.length;

  // Count standard (T-01..T-20) vs multimodal (T-21..T-32)
  const stdTasks = tasks.filter(t => parseInt(t.replace('T-','')) <= 20);
  const mmTasks = tasks.filter(t => parseInt(t.replace('T-','')) > 20);
  const stdDone = profiles.reduce((s, p) => s + (done[p] || []).filter(t => parseInt(t.replace('T-','')) <= 20).length, 0);
  const mmDone = profiles.reduce((s, p) => s + (done[p] || []).filter(t => parseInt(t.replace('T-','')) > 20).length, 0);

  // Count perturbed totals
  let totalPerturbed = 0, perturbedDone = 0;
  if (perturbed) {
    profiles.forEach(p => {
      const pt = perturbed[p] || [];
      totalPerturbed += pt.length;
      perturbedDone += pt.filter(t => (done[p] || []).includes(t)).length;
    });
  }

  html += `<div style="display:flex;gap:12px;margin-bottom:16px;flex-wrap:wrap;">
    <div style="flex:1;min-width:140px;padding:8px 14px;border-radius:8px;background:rgba(108,138,255,.06);font-size:13px;">
      <b>p1\u2013p10</b>: ${g1Done}/${g1Total} (${g1Total>0?(g1Done/g1Total*100).toFixed(0):'0'}%)
    </div>
    <div style="flex:1;min-width:140px;padding:8px 14px;border-radius:8px;background:rgba(74,222,128,.06);font-size:13px;">
      <b>p11\u2013p20</b>: ${g2Done}/${g2Total} (${g2Total>0?(g2Done/g2Total*100).toFixed(0):'0'}%)
    </div>
    <div style="flex:1;min-width:140px;padding:8px 14px;border-radius:8px;background:rgba(108,138,255,.12);font-size:13px;border:1px solid rgba(108,138,255,.25);">
      <b>T-01\u2013T-20</b>: ${stdDone}/${profiles.length * stdTasks.length} (${stdTasks.length > 0 ? (stdDone/(profiles.length * stdTasks.length)*100).toFixed(0):'0'}%)
    </div>
    <div style="flex:1;min-width:140px;padding:8px 14px;border-radius:8px;background:rgba(251,191,36,.12);font-size:13px;border:1px solid rgba(251,191,36,.25);">
      <b>T-21\u2013T-32</b>: ${mmDone}/${profiles.length * mmTasks.length} (${mmTasks.length > 0 ? (mmDone/(profiles.length * mmTasks.length)*100).toFixed(0):'0'}%)
    </div>
    <div style="flex:1;min-width:140px;padding:8px 14px;border-radius:8px;background:rgba(248,113,113,.12);font-size:13px;border:1px solid rgba(248,113,113,.25);">
      <b style="color:#f87171;">Perturbed</b>: ${perturbedDone}/${totalPerturbed} (${totalPerturbed > 0 ? (perturbedDone/totalPerturbed*100).toFixed(0):'0'}%)
    </div>
  </div>`;

  // Matrix table with type-colored headers
  html += '<div style="overflow-x:auto"><table class="progress-matrix"><thead><tr><th></th>';
  tasks.forEach(t => {
    const info = TASK_INFO[t];
    const typeColor = info ? (typeColors[info.type] || 'var(--text2)') : 'var(--text2)';
    const num = t.replace('T-','');
    const title = info ? `${info.type} — Dims: ${info.dims}` : '';
    html += `<th style="color:${typeColor};" title="${title}">${num}</th>`;
  });
  html += '<th></th><th style="min-width:60px"></th></tr></thead><tbody>';

  profiles.forEach((p, idx) => {
    // Visual separator between p10 and p11
    if (idx > 0 && parseInt(p.replace('p','')) === 11) {
      html += `<tr class="progress-group-sep"><td colspan="${tasks.length + 3}" style="height:2px;background:var(--border);padding:0;"></td></tr>`;
    }
    const pid = profile_ids[p];
    const doneTasks = done[p] || [];
    const perturbedTasks = (perturbed && perturbed[p]) || [];
    let pDone = 0;
    html += `<tr><td class="profile-label">${p}</td>`;
    tasks.forEach(t => {
      const isDone = doneTasks.includes(t);
      const isPerturbed = perturbedTasks.includes(t);
      if (isDone) pDone++;
      if (isDone && isPerturbed) {
        html += `<td class="done-cell" style="color:#f87171;" title="${p} × ${t} (perturbed)">\u2713</td>`;
      } else if (isDone) {
        html += `<td class="done-cell" title="${p} × ${t}">\u2713</td>`;
      } else if (isPerturbed) {
        html += `<td class="pending-cell" style="color:rgba(248,113,113,.4);" title="${p} × ${t} (perturbed, pending)">\u25C6</td>`;
      } else {
        html += `<td class="pending-cell">\u00b7</td>`;
      }
    });
    const pPct = (pDone / tasks.length * 100).toFixed(0);
    html += `<td class="profile-count">${pDone}/${tasks.length}</td>`;
    // Mini progress bar per profile
    const barColor = pDone === tasks.length ? 'var(--green)' : 'var(--accent)';
    html += `<td style="padding:0 4px;"><div style="width:50px;height:6px;background:var(--border);border-radius:3px;overflow:hidden;"><div style="width:${pPct}%;height:100%;background:${barColor};border-radius:3px;"></div></div></td>`;
    html += '</tr>';
  });

  // Type row at bottom showing task type per column
  html += `<tr style="border-top:2px solid var(--border);"><td style="font-size:11px;color:var(--text2);font-weight:600;">Type</td>`;
  tasks.forEach(t => {
    const info = TASK_INFO[t];
    const typeColor = info ? (typeColors[info.type] || 'var(--text2)') : 'var(--text2)';
    const abbrev = info ? info.type.substring(0, 3).toUpperCase() : '?';
    html += `<td style="font-size:10px;font-weight:600;color:${typeColor};" title="${info ? info.type : ''}">${abbrev}</td>`;
  });
  html += '<td></td><td></td></tr>';

  html += '</tbody></table></div>';

  // Meta info
  html += `<div class="progress-meta">
    <span>Active processes: <b>${active_processes}</b></span>
    <span>Remaining: <b>${total_target - total_done}</b></span>
    <span>Tasks: <b>${tasks.length}</b> (${stdTasks.length} standard + ${mmTasks.length} multimodal)</span>
    <span>Last updated: <b>${new Date().toLocaleTimeString()}</b></span>
  </div>`;

  // Task detail table
  html += `<div style="margin-top:20px;">
    <h3 style="font-size:14px;margin-bottom:10px;color:var(--text);">Task Details</h3>
    <table style="font-size:12px;">
      <thead><tr>
        <th>Task</th><th>Type</th><th>Dims</th><th>Completion</th>
      </tr></thead><tbody>`;
  tasks.forEach(t => {
    const info = TASK_INFO[t];
    if (!info) return;
    const typeColor = typeColors[info.type] || 'var(--text2)';
    const taskDone = profiles.filter(p => (done[p] || []).includes(t)).length;
    const taskPct = (taskDone / profiles.length * 100).toFixed(0);
    html += `<tr>
      <td style="font-weight:600;">${t}</td>
      <td style="color:${typeColor};font-weight:600;">${info.type}</td>
      <td>${info.dims}</td>
      <td><span style="font-weight:600;">${taskDone}/${profiles.length}</span> <span style="color:var(--text2);">(${taskPct}%)</span></td>
    </tr>`;
  });
  html += '</tbody></table></div>';

  container.innerHTML = html;
}

function startProgressPolling() {
  if (progressTimer) return;
  fetchProgress();
  progressTimer = setInterval(fetchProgress, 5000);
}

function stopProgressPolling() {
  if (progressTimer) {
    clearInterval(progressTimer);
    progressTimer = null;
  }
}

// Auto-start polling when trajectories tab is shown
const origTabClick = document.querySelectorAll('.tab');
origTabClick.forEach(t => {
  t.addEventListener('click', () => {
    if (t.dataset.tab === 'trajectories') {
      startProgressPolling();
    } else {
      stopProgressPolling();
    }
  });
});
