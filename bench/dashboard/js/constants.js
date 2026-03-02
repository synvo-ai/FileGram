// ============================================================
// Constants
// ============================================================
const METHODS = ['full_context','naive_rag','eager_summarization','mem0','zep','memos','memu','evermemos','filegramos_simple'];
const METHOD_DISPLAY = {
  full_context:'Full Context', naive_rag:'Naive RAG', eager_summarization:'Eager Summ.',
  mem0:'Mem0', zep:'Zep', memos:'MemOS', memu:'MemU', evermemos:'EverMemOS', filegramos_simple:'FileGramOS'
};
const ATTRIBUTES = ['name','role','language','tone','output_detail','working_style','thoroughness','documentation','error_handling','reading_strategy','output_structure','directory_style','naming','edit_strategy','version_strategy','cross_modal_behavior'];
const DIMS = ['A','B','C','D','E','F'];
const DIM_NAMES = {A:'Consumption',B:'Production',C:'Organization',D:'Iteration',E:'Rhythm',F:'Cross-Modal'};
const ATTR_TO_DIM = {
  name:'General', role:'General', language:'General', error_handling:'General',
  reading_strategy:'A', tone:'B', output_detail:'B', output_structure:'B',
  working_style:'B', thoroughness:'B', documentation:'B',
  directory_style:'C', naming:'C', edit_strategy:'D', version_strategy:'D',
  cross_modal_behavior:'F'
};
const DIM_GROUPS = {General:['name','role','language','error_handling'], A:['reading_strategy'], B:['tone','output_detail','output_structure','working_style','thoroughness','documentation'], C:['directory_style','naming'], D:['edit_strategy','version_strategy'], F:['cross_modal_behavior']};
const CH_PROCEDURAL = ['working_style','thoroughness','error_handling','reading_strategy','directory_style','edit_strategy','version_strategy','output_detail'];
const CH_SEMANTIC = ['name','role','language','tone','output_structure','documentation'];
const CH_MIXED = ['naming','cross_modal_behavior'];

const COVERED_ATTRIBUTES = ['name','role','language','tone','output_detail','output_structure','working_style','thoroughness','documentation','error_handling','reading_strategy','directory_style','naming','version_strategy'];
const NOT_COVERED = ATTRIBUTES.filter(a => !COVERED_ATTRIBUTES.includes(a));
const COVERED_PROC = COVERED_ATTRIBUTES.filter(a => CH_PROCEDURAL.includes(a));
const COVERED_SEM = COVERED_ATTRIBUTES.filter(a => CH_SEMANTIC.includes(a));

function attrChannel(a) { if (CH_PROCEDURAL.includes(a)) return 'Procedural'; if (CH_SEMANTIC.includes(a)) return 'Semantic'; return 'Mixed'; }

const PROFILE_LMR = {
  p1_methodical:{A:'L',B:'L',C:'L',D:'L',E:'L',F:'M'},
  p2_thorough_reviser:{A:'L',B:'L',C:'R',D:'R',E:'L',F:'M'},
  p3_efficient_executor:{A:'M',B:'R',C:'R',D:'R',E:'R',F:'R'},
  p4_structured_analyst:{A:'M',B:'L',C:'M',D:'L',E:'M',F:'L'},
  p5_balanced_organizer:{A:'R',B:'M',C:'M',D:'M',E:'M',F:'M'},
  p6_quick_curator:{A:'R',B:'M',C:'L',D:'R',E:'R',F:'R'},
  p7_visual_reader:{A:'L',B:'M',C:'M',D:'M',E:'R',F:'L'},
  p8_minimal_editor:{A:'R',B:'R',C:'M',D:'L',E:'M',F:'R'},
  p9_visual_organizer:{A:'M',B:'M',C:'L',D:'M',E:'L',F:'L'},
  p10_silent_auditor:{A:'L',B:'R',C:'R',D:'R',E:'M',F:'M'},
  p11_meticulous_planner:{A:'M',B:'L',C:'L',D:'L',E:'L',F:'R'},
  p12_prolific_scanner:{A:'R',B:'L',C:'R',D:'L',E:'R',F:'M'},
  p13_visual_architect:{A:'L',B:'M',C:'L',D:'M',E:'L',F:'L'},
  p14_concise_organizer:{A:'M',B:'R',C:'L',D:'L',E:'R',F:'R'},
  p15_thorough_surveyor:{A:'R',B:'L',C:'M',D:'M',E:'M',F:'M'},
  p16_phased_minimalist:{A:'M',B:'M',C:'R',D:'M',E:'L',F:'R'},
  p17_creative_archivist:{A:'L',B:'L',C:'L',D:'L',E:'R',F:'L'},
  p18_decisive_scanner:{A:'R',B:'R',C:'R',D:'R',E:'L',F:'M'},
  p19_agile_pragmatist:{A:'M',B:'M',C:'R',D:'M',E:'R',F:'R'},
  p20_visual_auditor:{A:'L',B:'R',C:'M',D:'R',E:'M',F:'L'},
};
const COLORS = ['#6c8aff','#f87171','#4ade80','#fbbf24','#a78bfa','#fb923c','#22d3ee','#f472b6','#84cc16'];

// ============================================================
// Task Info (type and dimensions for all 32 tasks)
// ============================================================
const TASK_INFO = {
  'T-01': { type: 'understand',  dims: 'A,B' },
  'T-02': { type: 'understand',  dims: 'A,B' },
  'T-03': { type: 'create',      dims: 'B,C' },
  'T-04': { type: 'create',      dims: 'B,C,F' },
  'T-05': { type: 'organize',    dims: 'C' },
  'T-06': { type: 'synthesize',  dims: 'A,B,F' },
  'T-07': { type: 'synthesize',  dims: 'A,B' },
  'T-08': { type: 'create',      dims: 'B,C,F' },
  'T-09': { type: 'iterate',     dims: 'D,B' },
  'T-10': { type: 'maintain',    dims: 'D,C,E' },
  'T-11': { type: 'iterate',     dims: 'D,A' },
  'T-12': { type: 'iterate',     dims: 'D,C' },
  'T-13': { type: 'iterate',     dims: 'D,A,E' },
  'T-14': { type: 'organize',    dims: 'C,D' },
  'T-15': { type: 'synthesize',  dims: 'A,D,B' },
  'T-16': { type: 'understand',  dims: 'A,E,B' },
  'T-17': { type: 'understand',  dims: 'A,C' },
  'T-18': { type: 'maintain',    dims: 'D,E,C' },
  'T-19': { type: 'iterate',     dims: 'D,B' },
  'T-20': { type: 'create',      dims: 'B,C,E' },
  'T-21': { type: 'organize',    dims: 'C,A,D' },
  'T-22': { type: 'understand',  dims: 'A,F,B' },
  'T-23': { type: 'organize',    dims: 'C,F,A' },
  'T-24': { type: 'synthesize',  dims: 'A,B,F' },
  'T-25': { type: 'understand',  dims: 'A,F,B' },
  'T-26': { type: 'organize',    dims: 'C,A,F' },
  'T-27': { type: 'create',      dims: 'B,F,C' },
  'T-28': { type: 'synthesize',  dims: 'A,B,F' },
  'T-29': { type: 'organize',    dims: 'C,D,A' },
  'T-30': { type: 'iterate',     dims: 'D,F,A' },
  'T-31': { type: 'create',      dims: 'B,F,C' },
  'T-32': { type: 'maintain',    dims: 'D,A,F' },
};
// Backward compatibility alias
const TASK_TIERS = Object.fromEntries(Object.entries(TASK_INFO).map(([k,v]) => [k, {...v, tier: 1, diff: 0, note: ''}]));

// ============================================================
// Color helpers
// ============================================================
function scoreColor(s) {
  if (s >= 4.5) return 'var(--score5)';
  if (s >= 3.5) return 'var(--score4)';
  if (s >= 2.5) return 'var(--score3)';
  if (s >= 1.5) return 'var(--score2)';
  return 'var(--score1)';
}
function scoreBg(s) {
  if (s >= 4.5) return 'rgba(34,197,94,.25)';
  if (s >= 3.5) return 'rgba(134,239,172,.2)';
  if (s >= 2.5) return 'rgba(251,191,36,.2)';
  if (s >= 1.5) return 'rgba(251,146,60,.2)';
  return 'rgba(239,68,68,.25)';
}
function lmrColor(v) {
  if (v === 'L') return '#d5e8d4';
  if (v === 'M') return '#fff2cc';
  return '#f8cecc';
}
function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ============================================================
// Data access helpers (depend on global DATA)
// ============================================================
function getScore(profile, method, attr) {
  const d = DATA.profiles[profile]?.[method]?.judge_scores?.scores?.[attr];
  return d ? d.score : null;
}
function getJustification(profile, method, attr) {
  const d = DATA.profiles[profile]?.[method]?.judge_scores?.scores?.[attr];
  return d ? d.justification : '';
}
function getInferred(profile, method, attr) {
  const d = DATA.profiles[profile]?.[method]?.judge_scores?.scores?.[attr];
  return d ? d.inferred : '';
}
function getGroundTruth(profile, method, attr) {
  const d = DATA.profiles[profile]?.[method]?.judge_scores?.scores?.[attr];
  return d ? d.ground_truth : '';
}
function getAvgScore(profile, method) {
  const scores = COVERED_ATTRIBUTES.map(a => getScore(profile, method, a)).filter(s => s !== null);
  return scores.length ? scores.reduce((a,b) => a+b, 0) / scores.length : 0;
}
function getAvgScoreAll16(profile, method) {
  return DATA.profiles[profile]?.[method]?.avg_score || 0;
}
function allProfiles() { return Object.keys(DATA.profiles); }

function methodOverall(method) {
  const ps = allProfiles();
  return ps.reduce((s, p) => s + getAvgScore(p, method), 0) / ps.length;
}

function dimAvg(profile, method, dimKey) {
  const attrs = DIM_GROUPS[dimKey];
  if (!attrs) return 0;
  const scores = attrs.map(a => getScore(profile, method, a)).filter(s => s !== null);
  return scores.length ? scores.reduce((a,b) => a+b, 0) / scores.length : 0;
}

function channelAvg(profile, method, attrs) {
  const scores = attrs.map(a => getScore(profile, method, a)).filter(s => s !== null);
  return scores.length ? scores.reduce((a,b) => a+b, 0) / scores.length : 0;
}
function crossProfileChannelAvg(method, attrs) {
  const ps = allProfiles();
  return ps.reduce((s, p) => s + channelAvg(p, method, attrs), 0) / ps.length;
}
