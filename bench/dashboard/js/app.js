// ============================================================
// State
// ============================================================
let DATA = null;
let charts = {};
let currentProfile = null;

// ============================================================
// Tabs
// ============================================================
function switchTab(tabName) {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
  const tab = document.querySelector(`.tab[data-tab="${tabName}"]`);
  const panel = document.getElementById('panel-' + tabName);
  if (tab) tab.classList.add('active');
  if (panel) panel.classList.add('active');
}
document.querySelectorAll('.tab').forEach(t => {
  t.addEventListener('click', () => switchTab(t.dataset.tab));
});
// Support hash-based tab linking (e.g., /#annotate)
if (window.location.hash) {
  const tab = window.location.hash.replace('#', '');
  if (document.getElementById('panel-' + tab)) switchTab(tab);
}

// ============================================================
// Data loading
// ============================================================
let lastDataHash = '';

async function fetchData() {
  try {
    const resp = await fetch('/api/data');
    const text = await resp.text();
    const hash = simpleHash(text);
    if (hash === lastDataHash) return;
    lastDataHash = hash;
    DATA = JSON.parse(text);
    currentProfile = currentProfile || Object.keys(DATA.profiles)[0];
    document.getElementById('statusText').textContent = `Live \u00b7 ${Object.keys(DATA.profiles).length} profiles \u00b7 ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    document.getElementById('statusText').textContent = 'Connection error';
  }
}

function simpleHash(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) { h = ((h << 5) - h + str.charCodeAt(i)) | 0; }
  return String(h);
}

fetchData();
setInterval(fetchData, 5000);
