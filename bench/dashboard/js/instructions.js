// ============================================================
// Instructions tab — comprehensive project documentation
// ============================================================

function buildInstructionsMarkdown() {
  // -- Task descriptions --
  const TASK_NAMES = {
    'T-01': 'Investment analyst work overview summary',
    'T-02': 'Legal case materials review and timeline',
    'T-03': 'Personal knowledge base creation',
    'T-04': 'Meeting minutes and follow-up document creation',
    'T-05': 'Messy folder cleanup and reorganization',
    'T-06': 'Multi-source synthesis research report',
    'T-07': 'Diary and notes synthesis into personal profile',
    'T-08': 'Quarterly work summary report creation',
    'T-09': 'Report revision and condensation',
    'T-10': 'Knowledge base content update and maintenance',
    'T-11': 'Multi-file error detection and correction',
    'T-12': 'Document format standardization',
    'T-13': 'Review feedback integration and revision',
    'T-14': 'Version management and archiving',
    'T-15': 'Conflicting reports analysis and reconciliation',
    'T-16': 'Time-constrained priority triage',
    'T-17': 'File system health check and diagnostics',
    'T-18': 'Knowledge base three-round incremental update',
    'T-19': 'Document audience adaptation',
    'T-20': 'Weekly report management system setup',
    'T-21': 'File system cleanup and deduplication',
    'T-22': 'Film and animation collection catalog',
    'T-23': 'Travel photo organization and album creation',
    'T-24': 'Investment earnings call analysis report',
    'T-25': 'Legal case multimedia evidence review',
    'T-26': 'Personal digital asset archiving',
    'T-27': 'Student portfolio compilation',
    'T-28': 'Pet care documentation synthesis',
    'T-29': 'Company registration database construction',
    'T-30': 'Voice memo organization and archiving',
    'T-31': 'Nature scenery video collection curation',
    'T-32': 'Cross-modal archive consistency check',
  };

  // -- Build task table rows --
  let taskRows = '';
  for (const [tid, info] of Object.entries(TASK_INFO)) {
    const name = TASK_NAMES[tid] || '';
    taskRows += `| ${tid} | ${info.type} | ${info.dims} | ${name} |\n`;
  }

  // -- Build profile matrix rows --
  let profileRows = '';
  for (const [pid, lmr] of Object.entries(PROFILE_LMR)) {
    const displayName = pid.replace(/_/g, '\\_');
    profileRows += `| ${displayName} | ${lmr.A} | ${lmr.B} | ${lmr.C} | ${lmr.D} | ${lmr.E} | ${lmr.F} |\n`;
  }

  // -- Build attribute table --
  const ATTR_DETAILS = {
    working_style:    { dim: 'E', L: 'methodical', M: 'exploratory', R: 'pragmatic', channel: 'Procedural' },
    thoroughness:     { dim: 'A', L: 'exhaustive', M: 'balanced', R: 'minimal', channel: 'Procedural' },
    documentation:    { dim: 'B', L: 'comprehensive', M: 'moderate', R: 'minimal', channel: 'Semantic' },
    organization_style: { dim: 'C', L: 'deeply\\_nested', M: 'adaptive', R: 'flat', channel: 'Procedural' },
    naming_convention:  { dim: 'C', L: 'descriptive\\_long', M: 'mixed', R: 'short\\_abbreviation', channel: 'Mixed' },
    version_strategy:   { dim: 'D', L: 'keep\\_history', M: 'archive\\_old', R: 'overwrite', channel: 'Procedural' },
    error_handling:     { dim: 'D', L: 'defensive', M: 'balanced', R: 'optimistic', channel: 'Procedural' },
    reading_strategy:   { dim: 'A', L: 'sequential\\_deep', M: 'targeted\\_search', R: 'breadth\\_first', channel: 'Procedural' },
    output_structure:   { dim: 'B', L: 'hierarchical', M: 'sectioned', R: 'flat\\_list', channel: 'Semantic' },
    directory_style:    { dim: 'C', L: 'nested\\_by\\_topic', M: 'adaptive', R: 'flat', channel: 'Procedural' },
    edit_strategy:      { dim: 'D', L: 'incremental\\_small', M: 'balanced', R: 'bulk\\_rewrite', channel: 'Procedural' },
    cross_modal_behavior: { dim: 'F', L: 'visual\\_heavy', M: 'tables\\_and\\_references', R: 'text\\_only', channel: 'Mixed' },
    tone:               { dim: 'B', L: 'professional', M: 'friendly', R: 'casual', channel: 'Semantic' },
    output_detail:      { dim: 'B', L: 'detailed', M: 'balanced', R: 'concise', channel: 'Procedural' },
  };

  let attrRows = '';
  for (const [attr, info] of Object.entries(ATTR_DETAILS)) {
    attrRows += `| ${attr} | ${info.dim} | ${info.L} | ${info.M} | ${info.R} | ${info.channel} |\n`;
  }

  return `# FileGram

## Overview

FileGram is a framework that grounds agent memory in **file-system behavioral traces** for personalized OS-level intelligence. It targets knowledge workers across all professions, not just programmers. The project consists of three components:

| Component | Role | Description |
|-----------|------|-------------|
| **FileGramEngine** | Data Generation | Persona-driven behavioral data generation pipeline. Code agents execute tasks under controlled user profiles, producing fine-grained file-level behavioral traces. |
| **FileGramBench** | Evaluation | Multimodal file-system memory benchmark (includes FileGramQA). Evaluates whether memory systems can infer user preferences, predict behavior, and build accurate profiles. |
| **FileGramOS** | Memory Method | Bottom-up memory method. Decomposes behavioral signals into procedural, semantic, and episodic channels. Defers abstraction to query time. |

### Core Thesis

> User preferences and work habits are best inferred from **what users do with files** (behavioral traces / "engrams"), not from what they say (dialogue / self-reports).

---

## FileGramEngine: Data Generation Pipeline

\`\`\`
INPUT:
  Task Prompt  +  Profile (YAML)  +  Workspace (sandbox)
       |              |                    |
       v              v                    v
  +-------------------------------------------------+
  |          FileGramEngine Agent Runtime            |
  |                                                  |
  |  LLM Loop  -->  Tools (read/write/edit/bash/     |
  |  (GPT-4.1,      grep/glob)                      |
  |   Claude,        |                               |
  |   Gemini)   BehaviorCollector                    |
  |              - Real-time event capture            |
  |              - File hash tracking                 |
  |              - Context switch detection           |
  |              - FS snapshot                        |
  +-------------------------------------------------+
       |
       v
OUTPUT (per session):
  signal/{profile}_{task}/
  +-- events.json      # Raw behavioral event log
  +-- summary.json     # Aggregated session statistics
  +-- media/
      +-- blobs/       # Content-addressable file snapshots
      +-- diffs/       # Unified diffs
      +-- manifest.json
\`\`\`

**Key design**: The same workspace (fixed files) is given to different profiles (different "users"). Behavioral differences come purely from profile variation, not from data differences.

### Agent Action Space

FileGramEngine runs in **behavioral mode**, which restricts the agent to a fixed set of 8 file-system tools. All non-behavioral tools (web search, sub-agent spawning, code navigation, etc.) are disabled to ensure a clean, controlled action space.

| Tool | Type | Description |
|------|------|-------------|
| \`read\` | Dedicated | Read file contents (with optional line range) |
| \`write\` | Dedicated | Create or overwrite a file |
| \`edit\` | Dedicated | Targeted string replacement within a file |
| \`multiedit\` | Dedicated | Multiple edits to a single file in one call |
| \`grep\` | Dedicated | Search file contents by regex pattern |
| \`glob\` | Dedicated | Find files by name/path pattern |
| \`list\` | Dedicated | List directory contents |
| \`bash\` | Shell | Execute arbitrary shell commands (mkdir, mv, cp, rm, ls, cat, etc.) |

**Disabled in behavioral mode**: task, todo, plan, skill, webfetch, websearch, question, mcp, batch, lsp, codesearch, apply\\_patch — these are agent infrastructure tools that do not produce file-system behavioral signals.

The 7 dedicated tools (read/write/edit/multiedit/grep/glob/list) each map directly to a behavioral event type. The bash tool is the wildcard — it can execute any shell command, but only filesystem-modifying commands (\`mkdir\`, \`mv\`, \`cp\`, \`rm\`, \`ls\`) are parsed into behavioral events via regex matching (see below).

### Behavioral Signal Schema

#### Event Categories

| Category | Event Types | Dimension |
|----------|------------|-----------|
| File Read | file\\_read, file\\_search, file\\_browse | A (Consumption) |
| File Write | file\\_write, file\\_edit | B, C, D |
| File Organization | file\\_rename, file\\_move, dir\\_create, file\\_delete, file\\_copy, fs\\_snapshot | C (Organization) |
| Workflow | iteration\\_start, iteration\\_end, context\\_switch | D, E |
| Cross-File | cross\\_file\\_reference | A, D, F |

#### Key Observable Indicators

| Dimension | Key Metrics |
|-----------|-------------|
| A | file\\_read count, search\\_ratio, browse\\_ratio, revisit\\_count |
| B | avg\\_content\\_length, files\\_created, heading\\_depth |
| C | dir\\_create count, max\\_depth, file\\_move count, file\\_delete count |
| D | edit\\_count per file, avg\\_lines\\_modified, backup\\_copies |
| E | context\\_switch rate, phase separation, same-file streak length |
| F | image/media files created, table syntax presence, figure references |

### Tool-to-Signal Mapping

FileGramEngine intercepts every tool call the LLM agent makes and converts it into behavioral signals. The agent has access to 6 tools: **read**, **write**, **edit**, **grep**, **glob**, and **bash**. The first 5 tools map directly to behavioral events, while **bash** requires command parsing to extract filesystem operations.

#### Direct Tool Mapping (1:1)

| Agent Tool | Behavioral Event | Key Fields Captured |
|------------|-----------------|---------------------|
| \`read(file, range)\` | \`file_read\` | file\\_path, view\\_range, content\\_length, view\\_count, revisit\\_interval\\_ms |
| \`write(file, content)\` | \`file_write\` | file\\_path, operation (create/overwrite), content\\_length, before\\_hash, after\\_hash |
| \`edit(file, old, new)\` | \`file_edit\` | file\\_path, lines\\_added, lines\\_deleted, lines\\_modified, diff\\_summary |
| \`grep(pattern)\` | \`file_search\` | search\\_type="grep", query, files\\_matched |
| \`glob(pattern)\` | \`file_search\` | search\\_type="glob", query, files\\_matched |

#### Bash Command Parsing (regex-based extraction)

Bash commands are first split on \`&&\`, \`||\`, \`;\` into individual commands, then each is matched against regex patterns:

| Bash Command | Regex Pattern | Behavioral Event | Routing Logic |
|-------------|--------------|-----------------|---------------|
| \`ls\` / \`dir\` | \`^\\s*(?:ls\|dir)\\b(.*)$\` | \`file_browse\` | Extract directory path from args, count output lines as files\\_listed |
| \`mkdir [-p]\` | \`^\\s*mkdir\\s+(.+)$\` | \`dir_create\` | One event per directory created; compute depth and sibling\\_count |
| \`mv src dest\` (same dir) | \`^\\s*mv\\s+(.+)$\` | \`file_rename\` | If src and dest share parent directory -> rename; detects naming\\_pattern\\_change |
| \`mv src dest\` (diff dir) | (same pattern) | \`file_move\` | If parent directories differ -> move; records destination\\_directory\\_depth |
| \`rm file\` | \`^\\s*rm\\s+(.+)$\` | \`file_delete\` | Computes file\\_age\\_ms from first access; checks was\\_temporary by extension |
| \`cp src dest\` | \`^\\s*cp\\s+(.+)$\` | \`file_copy\` | Checks is\\_backup by dest extension (.bak/.backup/.orig/.old/.save) |
| \`cat\`, \`grep\`, \`head\`, etc. | *(no match)* | **No event** | Non-filesystem bash commands are NOT captured as behavioral events |

> **Design note**: Commands like \`cat\`, \`grep\`, \`head\` executed via bash do NOT produce behavioral events. This is intentional — the agent is guided (via profile system\\_prompt) to use dedicated tools (read, grep) instead of bash equivalents. If an agent bypasses dedicated tools and uses bash grep, the search behavior is lost from the behavioral trace.

#### Implicit Events (cross-cutting)

These events are emitted automatically by the BehaviorCollector, triggered by the operations above:

| Implicit Event | Trigger Condition | What It Captures |
|---------------|-------------------|-----------------|
| \`context_switch\` | Any read/edit where the target file differs from the previously accessed file | from\\_file, to\\_file, trigger source, switch\\_count |
| \`cross_file_reference\` | A read of file A followed by an edit of file B (causal chain) | source\\_file, target\\_file, reference\\_type, interval\\_ms |

### Behavioral vs. Simulation Events

Events in \`events.json\` are classified into two categories. Only **behavioral events** represent user-observable file-system actions and are used by FileGramOS and FileGramBench. **Simulation events** are agent/LLM internals logged for debugging and cost tracking only.

#### Behavioral Events (used by FileGramOS)

These are things a real OS could observe — file opens, writes, renames, directory changes:

| Event Type | Category | What a Real OS Would See |
|-----------|----------|-------------------------|
| \`file_read\` | Consumption | User opened and viewed a file |
| \`file_search\` | Consumption | User searched for files by content or name |
| \`file_browse\` | Consumption | User listed a directory's contents |
| \`file_write\` | Production | User created or overwrote a file |
| \`file_edit\` | Production | User made targeted changes to a file |
| \`file_rename\` | Organization | User renamed a file |
| \`file_move\` | Organization | User moved a file to a different directory |
| \`dir_create\` | Organization | User created a new directory |
| \`file_delete\` | Organization | User deleted a file |
| \`file_copy\` | Organization | User duplicated a file (possibly as backup) |
| \`context_switch\` | Workflow | User switched from working on one file to another |
| \`cross_file_reference\` | Workflow | User read file A then modified file B (causal link) |

#### Simulation Events (NOT used by FileGramOS — debugging/cost only)

These are LLM agent internals that a real OS would never see:

| Event Type | Purpose | Why Excluded |
|-----------|---------|-------------|
| \`fs_snapshot\` | Directory tree snapshot at session start/end | System-triggered checkpoint, not a user action |
| \`iteration_start/end\` | LLM turn boundaries (rhythm segmentation) | Agent loop structure, not a user-initiated operation |
| \`error_encounter\` | Tool execution failure | Agent-internal error, not a user-visible filesystem action |
| \`error_response\` | How the agent reacted to failure (retry/skip/rethink) | Agent decision-making logic, not observable by OS |
| \`tool_call\` | Log every tool invocation with parameters and timing | Agent implementation detail, not user behavior |
| \`llm_response\` | Log response time, input/output tokens, reasoning | LLM cost tracking, invisible to a real OS |
| \`compaction_triggered\` | Context window compression events | Agent memory management, not user action |
| \`session_start/end\` | Session bookkeeping with profile\\_id, task\\_id | Metadata, not behavioral signal |

> **Key principle**: The boundary between behavioral and simulation events maps to the question "could a real operating system observe this without access to the LLM's internals?" File operations = yes (behavioral). Token counts and tool routing = no (simulation).

---

## Profile Dimensions (6 Dimensions x 3 Tiers)

Each profile is defined by a **6-dimensional L/M/R vector**. Dimensions are derived top-down from OS agent use cases.

### Dimension A: Consumption Pattern
*How does the user explore, locate, and understand information?*

| Tier | Value | Behavioral Spec |
|------|-------|----------------|
| **L** | sequential | Read files one by one in logical order. Full file ranges. Read 70%+ before conclusions. Revisit 3+ files. Never use grep/glob. |
| **M** | targeted | Search first, read second. Use grep for keywords, glob for file types. Read only matched files, targeted sections. |
| **R** | breadth-first | Scan with ls and glob broadly. Read first 10-20 lines only. Browse widely, rarely revisit. Coverage over depth. |

### Dimension B: Production Style
*What format, detail level, and structure does the user prefer for output?*

| Tier | Value | Behavioral Spec |
|------|-------|----------------|
| **L** | comprehensive | Multi-level headings (##/###/####). Data tables, appendices, reference lists. 200+ lines. 3+ output files. |
| **M** | balanced | 2 heading levels. Occasional tables. Single main file. 80-150 lines. 1-2 files. |
| **R** | minimal | Flat bullet-point lists. No headings deeper than ##. One file only. Under 60 lines. |

### Dimension C: Organization Preference
*How does the user manage file systems: directories, naming, versions?*

| Tier | Value | Behavioral Spec |
|------|-------|----------------|
| **L** | deeply\\_nested | 3+ level dirs via mkdir -p. Long descriptive names with prefixes. README at each level. Backup before modify. |
| **M** | adaptive | 1-2 level dirs when needed. Practical names, mixed styles. Keep old versions only when important. |
| **R** | flat | All files in root. No subdirectories. Short abbreviated names. Overwrite directly. Delete temp files. |

### Dimension D: Iteration Strategy
*How does the user modify and refine existing work?*

| Tier | Value | Behavioral Spec |
|------|-------|----------------|
| **L** | incremental | Many small edit calls (few lines each). Review after each edit. Create backups. Multiple passes. Never delete content. |
| **M** | balanced | Moderate edit calls of reasonable scope. Occasional review. Balance refinement with efficiency. |
| **R** | rewrite | Overwrite entire file with write. Delete-recreate pattern. One decisive pass, no backups. |

### Dimension E: Work Rhythm
*What temporal pattern does the user follow?*

| Tier | Value | Behavioral Spec |
|------|-------|----------------|
| **L** | phased | Distinct phases: (1) read all, (2) plan, (3) write, (4) review. Minimize context switches. |
| **M** | steady | Interleave reading and writing naturally. Moderate context switching. Logical but not rigid order. |
| **R** | bursty | Rapid reactive bursts. Frequent file switches. Start writing before finishing reading. Jump by urgency. |

### Dimension F: Cross-Modal Behavior
*Does the user create or use visual materials?*

| Tier | Value | Behavioral Spec |
|------|-------|----------------|
| **L** | visual-heavy | Generate charts/diagrams. Reference figures with numbered captions. Maintain figures/ directory. |
| **M** | balanced | Include markdown tables where data warrants. No standalone image files. Structured formatting. |
| **R** | text-only | Pure text only. No tables, no images, no charts. Prose and bullet lists. Describe data inline. |

---

## Profile Attributes (16 Total)

Each profile has 16 observable attributes mapped to dimensions and memory channels.

| Attribute | Dim | L Value | M Value | R Value | Channel |
|-----------|-----|---------|---------|---------|---------|
${attrRows}

### Channel Classification

- **Procedural** (8): working\\_style, thoroughness, error\\_handling, reading\\_strategy, directory\\_style, edit\\_strategy, version\\_strategy, output\\_detail
- **Semantic** (6): name, role, language, tone, output\\_structure, documentation
- **Mixed** (2): naming, cross\\_modal\\_behavior

---

## Profile Matrix (20 Profiles x 6 Dimensions)

| Profile | A | B | C | D | E | F |
|---------|---|---|---|---|---|---|
${profileRows}

**Design constraints**: 6-7 profiles per L/M/R per dimension. 10 Chinese / 10 English. No duplicate vectors. Fine-grained pairs (1-2 dim difference) and coarse pairs (5+ dim difference) included.

---

## Tasks (32 Total)

| Task | Type | Dimensions | Description |
|------|------|-----------|-------------|
${taskRows}

### Task Types

| Type | Target Dimensions | Description |
|------|:-:|-------------|
| **understand** | A, B | Read files and produce summary/judgment |
| **create** | B, C | Open-ended content creation |
| **organize** | C | File system restructuring |
| **synthesize** | A, B | Read multiple files, produce integrated output |
| **iterate** | D | Revise existing work with feedback |
| **maintain** | D, E | Update/extend previous work |

---

## FileGramOS: Memory Method

FileGramOS decomposes behavioral traces into three memory channels — **Procedural** (how the user works), **Semantic** (what changed and why), and **Episodic** (behavioral consistency over time). Two key design principles: (1) **Delta-only input** — memory is built from operation sequences and file diffs, not pre-existing file content; (2) **Deferred abstraction** — structured representations are stored raw and interpreted only at query time.

<div style="margin:16px 0;text-align:center;">
<img src="/img/architecture.png" alt="FileGramOS Three-Channel Behavioral Memory Architecture" style="max-width:50%;border:1px solid #444;border-radius:8px;" />
<p style="color:#aaa;font-size:12px;margin-top:6px;">FileGramOS Three-Channel Behavioral Memory Architecture</p>
</div>

<a href="#" onclick="document.querySelector('[data-tab=pipeline]').click();return false;" style="color:#4fc3f7;text-decoration:underline;cursor:pointer;">→ See the **FileGramOS Pipeline** tab for a full interactive walkthrough with real data</a>

---

## FileGramBench: Evaluation Benchmark

FileGramBench evaluates memory systems through **FileGramQA** — a programmatically generated MCQ benchmark with 1,058 questions across 3 tracks (+ 1 multimodal track).

### Track Overview

| Track | Name | Channel | Sub-types | Questions | Metric |
|:-----:|------|---------|:---------:|:---------:|--------|
| 1 | **Understanding** | P + S | 2 + freeform | 149 MCQ | MCQ + Likert |
| 2 | **Reasoning** | P | 5 | 465 | MCQ accuracy |
| 3 | **Detection** | E | 5 | 444 | MCQ accuracy |
| 4 | **Multimodal** | P + S + E | — | *(shared)* | MCQ accuracy |

### Evaluation Pipeline

Each memory system (baseline or FileGramOS) first **ingests** all behavioral trajectories for a profile into its internal representation (cached as pickle). At evaluation time, the system's stored memory is extracted and provided as **context** alongside the question. The LLM judge sees: *memory context + question + choices → single-letter answer*.

\`\`\`
Phase 1: Memory Ingestion (offline, per profile)
  For each trajectory (events.json + media/):
    baseline.ingest(trajectory)  →  cached memory state (.pkl)

Phase 2: QA Evaluation (per question)
  memory_context = method.extract(profile, relevant_task_ids)
  prompt = memory_context + question_text + choices
  answer = LLM(prompt)  →  compare to ground truth
\`\`\`

### Input Context Summary

What each memory system receives as **context** varies by question type:

| Sub-Type | # Traj | Profiles | Input Composition | Key Challenge |
|----------|:------:|:--------:|-------------------|---------------|
| 1.1 Dim Classification | All | 1 | All events from 1 profile | Classify L/M/R from aggregated behavior |
| 1.2 Semantic Attribute | All | 1 | All events + created file contents | Infer identity from output content |
| 2.1 Fingerprint | 1 | 1+3 distractors | 1 anonymous trajectory + 4 candidate portraits | Match single trajectory to portrait |
| 2.2 Counterfactual | N-1 | 1 | All events except held-out task | Predict behavior on unseen task |
| 2.3 Anomaly Detection | 5 | 2 | 4 real + 1 impostor (labeled A-E) | Find the outlier trajectory |
| 2.4 Multi-User Attr | 2 | 2 | 2 profiles × same task, **interleaved** | Disentangle 2 users' behaviors |
| 2.5 Task Separation | 2 | 1 | 1 profile × 2 tasks, **interleaved** | Identify task boundaries |
| 3.1 Impersonation | 6 | 2 | 5 real + 1 similar impostor (A-F) | Detect fine-grained impostor |
| 3.2 Perturbation Det | 5 | 1 | 4 standard + 1 perturbed (A-E) | Flag deviant session |
| 3.3 Dim Identification | 6 | 1 | 5 standard + 1 perturbed | Identify shifted dimension |
| 3.4 Shift Direction | 6 | 1 | 5 standard + 1 perturbed | Identify L/M/R transition |
| 3.5 Change Description | 6 | 1 | 5 standard + 1 perturbed | Describe before→after change |

---

### Track 1: Understanding — Attribute Recognition (149 MCQ + freeform)

Given N behavioral trajectories from one user, recognize the user's behavioral attributes. Two MCQ sub-types plus a freeform profile reconstruction task.

#### 1.1 Dimension Classification (89 Qs, 3-choice)

Select which L/M/R semantic description matches the user's behavior on a given dimension.

**Input context**: All trajectories (events.json) from the target profile. Only dimensions with ≥30% activation rate across the trajectory set generate questions.

\`\`\`json
"input_trajectories": [
  "p4_structured_analyst_T-01",  "p4_structured_analyst_T-03",
  "p4_structured_analyst_T-05",  "p4_structured_analyst_T-07",
  "p4_structured_analyst_T-12",  "p4_structured_analyst_T-16",
  ...  // all available trajectories for this profile
]
\`\`\`

> **Q**: Based on the behavioral trajectories provided, which description best matches this user's **Consumption Pattern** (Dimension A)?
>
> - **(A)** Sequential deep reader: reads files one by one in logical order, going through each document completely. Rarely uses search tools, preferring to read all available content systematically. Frequently revisits files to cross-reference details.
> - **(B)** Targeted searcher: searches for specific keywords or file types first, then reads only the matched files and relevant sections. Efficient and selective in information gathering.
> - **(C)** Breadth-first scanner: quickly scans directory listings and file headers to get an overview. Prioritizes coverage over depth, browsing widely but rarely revisiting.

#### 1.2 Semantic Attribute (60 Qs, 3-4 choice)

Infer user identity attributes (language, role, documentation habits) from output content. Three sub-categories: language (20 Qs, 4-choice), role (20 Qs, 4-choice), documentation (20 Qs, 3-choice).

**Input context**: All trajectories from target profile, including created file contents (blobs). Language/role are inferred from content semantics; documentation habits from file creation patterns.

\`\`\`json
"input_trajectories": [
  "p12_prolific_scanner_T-01",  "p12_prolific_scanner_T-04",
  "p12_prolific_scanner_T-06",  "p12_prolific_scanner_T-09",
  ...  // all available trajectories for this profile
]
\`\`\`

> **Q**: Based on the user's output files and behavioral traces, what language does this user primarily write in?
>
> - **(A)** Chinese
> - **(B)** English
> - **(C)** Both equally
> - **(D)** Unable to determine

#### 1.3 Profile Reconstruction (freeform, Likert 1-5)

Open-ended task: reconstruct a complete 16-attribute behavioral profile. Scored by per-attribute LLM judge (Likert 1-5 with strict rubrics that penalize vague or generic answers). Data efficiency measured at N ∈ {1, 3, 5, 10, 20} trajectories.

**Input context**: N trajectories from target profile (variable N for data efficiency curve).

---

### Track 2: Reasoning — Behavioral Inference & Trace Disentanglement (465 MCQ)

Tests whether systems can infer behavioral patterns and disentangle interleaved signals. Five sub-types spanning two sub-tracks.

#### Sub-track: Behavioral Inference (235 MCQ)

#### 2.1 Behavioral Fingerprint (100 Qs, 4-choice)

Match an anonymous trajectory to a multi-dimension behavioral portrait (fine-grained distractors).

**Input context**: **1 single anonymous trajectory** (events.json from one task). Choices are 4 candidate 6-dimensional behavioral portraits — 1 correct + 3 distractors (preferring fine-grained neighbors with 1-3 dim differences).

\`\`\`json
"input_trajectories": [
  "p1_methodical_T-05"    // single anonymous trajectory
]
\`\`\`

> **Q**: Based on the behavioral trajectory provided, which user profile description best matches the observed behavior?
>
> - **(A)** This user reads files thoroughly and sequentially, produces detailed multi-level documents with auxiliary files, organizes into deep nested directories, makes many small incremental edits, and follows a structured phased workflow.
> - **(B)** This user searches for specific information before reading, writes minimal flat bullet-point output, keeps all files in root directory, overwrites files in one pass, and works in rapid reactive bursts.
> - **(C)** This user scans broadly but produces comprehensive output, uses adaptive directory structures, balances incremental and bulk editing, and maintains a steady work rhythm.
> - **(D)** This user reads sequentially but writes minimal output, uses flat organization, rewrites entire files, and follows a phased workflow with visual evidence.

#### 2.2 Counterfactual Reasoning (75 Qs, 4-choice)

Predict behavior on a held-out task from N-1 observed trajectories.

**Input context**: **N-1 trajectories** from target profile (one task held out). The system must predict how this user would behave on the unseen task type. Minimum 3 remaining trajectories required.

\`\`\`json
"input_trajectories": [
  "p6_quick_curator_T-01",  "p6_quick_curator_T-03",
  "p6_quick_curator_T-05",  "p6_quick_curator_T-12",
  "p6_quick_curator_T-16"
  // T-09 held out — this is the task to predict
]
\`\`\`

> **Q**: Based on this user's behavior across 5 other tasks, predict how they would approach an **iterate** task (T-09: Report revision and condensation). Which behavioral prediction is most consistent?
>
> - **(A)** Would make many small targeted edits across multiple passes, creating backups before each modification, and producing a detailed revision log.
> - **(B)** Would rewrite the entire report from scratch in one decisive pass, overwriting the original without backups.
> - **(C)** Would search for key sections to revise, make moderate edits, and produce a balanced condensed version.
> - **(D)** Would scan the report broadly, delete most content, and produce a minimal bullet-point summary.

#### 2.3 Anomaly Detection (60 Qs, 5-choice)

Identify an impostor trajectory planted among four genuine ones from the same user.

**Input context**: **5 trajectories** labeled Session A-E with task names — 4 real from target profile + 1 impostor from a different profile (preferring fine-grained pairs). No profile IDs exposed.

\`\`\`json
"input_trajectories": [
  "p1_methodical_T-01",     // Session A — real
  "p1_methodical_T-03",     // Session B — real
  "p2_thorough_reviser_T-05", // Session C — IMPOSTOR (fine-grained pair)
  "p1_methodical_T-06",     // Session D — real
  "p1_methodical_T-09"      // Session E — real
]
\`\`\`

> **Q**: Five behavioral trajectories are provided. Four belong to the same user and one was produced by a different user. Which session is the outlier?
>
> - Session A: Investment Analyst Work Summary (T-01, understand)
> - Session B: Personal Knowledge Base Creation (T-03, create)
> - Session C: Messy Folder Cleanup (T-05, organize)
> - Session D: Multi-source Research Report (T-06, synthesize)
> - Session E: Report Revision and Condensation (T-09, iterate)

#### Sub-track: Trace Disentanglement (230 MCQ)

#### 2.4 Multi-User Attribution (128 Qs, 4-choice, grouped)

Two users' event streams interleaved on the same task — identify behavioral differences and which dimensions differ.

**Input context**: **2 trajectories from 2 different profiles**, both performing the **same task**. Events are **interleaved** into a single mixed stream. The system must disentangle which events belong to which user.

\`\`\`json
"input_trajectories": [
  "p1_methodical_T-01",         // User A
  "p3_efficient_executor_T-01"  // User B (same task, different profile)
]
// Events from both trajectories are INTERLEAVED in temporal order
\`\`\`

> **Q1**: Two users (User A and User B) performed the same task T-01: Investment Analyst Work Summary. Their event streams have been interleaved. What is the primary behavioral difference between them?
>
> - **(A)** User A reads files sequentially and thoroughly while User B quickly scans files broadly without deep reading
> - **(B)** Both users have nearly identical reading and writing patterns
> - **(C)** User A and User B differ primarily in the language they write in
> - **(D)** The users show opposite documentation habits

> **Q2**: On which behavioral dimension(s) do User A and User B primarily differ?
>
> - **(A)** A (Consumption Pattern), B (Production Style)
> - **(B)** C (Organization), D (Iteration)
> - **(C)** E (Work Rhythm), F (Cross-Modal)
> - **(D)** B (Production Style) only

#### 2.5 Single-User Task Separation (102 Qs, 3-4 choice, grouped)

Traces from two task types mixed for a single user — identify task boundaries and behavioral patterns.

**Input context**: **2 trajectories from the same profile** on **different task types** (e.g., "understand" + "organize"). Events are **interleaved** into a single mixed stream. The system must identify that two distinct tasks are present and describe their behavioral signatures.

\`\`\`json
"input_trajectories": [
  "p5_balanced_organizer_T-01",  // understand task
  "p5_balanced_organizer_T-05"   // organize task (same user, different type)
]
// Events from both tasks are INTERLEAVED in temporal order
\`\`\`

> **Q1**: This event stream is from a single user but may contain multiple tasks interleaved. How many distinct tasks are represented?
>
> - **(A)** 1 task
> - **(B)** 2 tasks
> - **(C)** 3 tasks
> - **(D)** 4 tasks

> **Q2**: Two tasks from the same user are interleaved. Which description best characterizes the behavioral patterns that distinguish the two task groups?
>
> - **(A)** One group involves extensive file reading with summary output; the other involves directory restructuring and file movement
> - **(B)** One group focuses on content creation with rich formatting; the other on iterative revision of existing files
> - **(C)** Both groups show identical behavioral patterns with no clear distinction
> - **(D)** One group involves iterative editing with cross-file referencing; the other focuses on broad scanning and minimal output

---

### Track 3: Detection — Anomaly & Shift Analysis (444 MCQ)

Tests episodic memory: maintaining per-session awareness and detecting behavioral deviations. Leverages **behavioral perturbation**: 5 of 32 trajectories per profile have one dimension shifted by one tier (from real \`perturbation.json\` metadata, not hard-stitched). Five sub-types spanning two sub-tracks.

#### Sub-track: Anomaly Detection (156 MCQ)

#### 3.1 Impersonation Detection (60 Qs, 6-choice)

Five real sessions plus one impostor from a similar profile — identify the fake.

**Input context**: **6 trajectories** labeled Session A-F — 5 real from target profile + 1 impostor from a **fine-grained neighbor** (1-3 dim differences). Harder than 2.3 Anomaly Detection because the impostor is behaviorally similar.

\`\`\`json
"input_trajectories": [
  "p1_methodical_T-01",         // Session A — real
  "p1_methodical_T-02",         // Session B — real
  "p1_methodical_T-03",         // Session C — real
  "p1_methodical_T-05",         // Session D — real
  "p1_methodical_T-06",         // Session E — real
  "p2_thorough_reviser_T-09"    // Session F — IMPOSTOR (fine-grained, 2-dim diff)
]
\`\`\`

> **Q**: Six behavioral trajectories are provided. Five belong to the same user and one was produced by a different user with a **similar** behavioral profile. Which session is the impostor?
>
> - Session A: Investment Analyst Work Summary (T-01, understand)
> - Session B: Legal Case Review (T-02, understand)
> - Session C: Knowledge Base Creation (T-03, create)
> - Session D: Folder Cleanup (T-05, organize)
> - Session E: Research Report (T-06, synthesize)
> - Session F: Report Revision (T-09, iterate)

#### 3.2 Perturbation Detection (96 Qs, 5-choice)

Four standard sessions plus one perturbed — flag the deviant.

**Input context**: **5 trajectories** from the same profile labeled A-E — 4 standard + 1 with a **real dimensional perturbation** (1 dimension shifted by 1 tier). Unlike impersonation, the deviant trajectory is from the *same* profile with controlled behavioral manipulation.

\`\`\`json
"input_trajectories": [
  "p10_silent_auditor_T-28",    // Session A — standard
  "p10_silent_auditor_T-16",    // Session B — standard
  "p10_silent_auditor_T-24",    // Session C — PERTURBED (dim C: R→M)
  "p10_silent_auditor_T-04",    // Session D — standard
  "p10_silent_auditor_T-01"     // Session E — standard
]
\`\`\`

> **Q**: Five behavioral trajectories from the same user are provided. Four follow the user's typical patterns, while one shows a deviation. Which session deviates from the user's normal behavior?
>
> - Session A: Pet Care Documentation Synthesis (T-28, synthesize)
> - Session B: Time-constrained Triage (T-16, understand)
> - Session C: Investment Earnings Call Analysis (T-24, synthesize)
> - Session D: Meeting Minutes & Follow-up Docs (T-04, create)
> - Session E: Investment Analyst Work Summary (T-01, understand)

#### Sub-track: Shift Analysis (288 MCQ)

For 3.3–3.5 below, the input context is the same: **6 trajectories** — 5 standard baseline sessions + 1 perturbed session with a known dimensional shift.

\`\`\`json
"input_trajectories": [
  "p8_minimal_editor_T-01",     // baseline
  "p8_minimal_editor_T-05",     // baseline
  "p8_minimal_editor_T-07",     // baseline
  "p8_minimal_editor_T-16",     // baseline
  "p8_minimal_editor_T-22",     // baseline
  "p8_minimal_editor_T-12"      // PERTURBED (1 dim shifted)
]
\`\`\`

#### 3.3 Shifted Dimension Identification (96 Qs, 6-choice)

Identify which dimension changed in the deviant session.

> **Q**: A behavioral deviation was detected in this user's session on T-12: Format Standardization. Which behavioral dimension shifted in the deviant session?
>
> - **(A)** A (Consumption Pattern)
> - **(B)** B (Production Style)
> - **(C)** C (Organization Preference)
> - **(D)** D (Iteration Strategy)
> - **(E)** E (Work Rhythm)
> - **(F)** F (Cross-Modal Behavior)

#### 3.4 Shift Direction (96 Qs, 3-choice)

Identify the direction of change (L/M/R transition). The shifted dimension is given.

> **Q**: The shifted dimension is **C (Organization Preference)**. In which direction did the behavior change?
>
> - **(A)** deeply\\_nested (L) -> adaptive (M)
> - **(B)** deeply\\_nested (L) -> flat (R)
> - **(C)** flat (R) -> adaptive (M)

#### 3.5 Behavioral Change Description (96 Qs, 4-choice)

Select the correct natural-language before→after behavioral change description — **no dimension labels provided**. Combines dimension identification and direction judgment into a single harder question. Distractors describe plausible changes on *other* dimensions.

> **Q**: A user's typical behavioral patterns are shown across 5 baseline sessions. One additional session (Format Standardization, T-12) shows a behavioral deviation. Which description best characterizes the behavioral change observed in the deviant session?
>
> - **(A)** Changed from creating deep multi-level directory hierarchies with descriptive naming to keeping all files in a flat root structure with short names
> - **(B)** Changed from making many small incremental edits with backups to rewriting entire files in single decisive passes
> - **(C)** Changed from reading files sequentially and thoroughly to quickly scanning headers and browsing broadly
> - **(D)** Changed from producing comprehensive multi-section documents to writing minimal bullet-point summaries

---

### Track 4: Multimodal Grounding

The same Markdown outputs from Tracks 1-3 are rendered as PDFs/images before memory ingestion; behavioral event logs remain unchanged. Two conditions: *without caption* (metadata only) and *with caption* (one-sentence file description). Systems relying on structured traces (e.g., FileGramOS) should be modality-invariant; content-dependent systems will degrade.

**Input context**: Same trajectory events as Tracks 1-3, but created file contents are provided as rendered PDF/image instead of raw Markdown text.

---

### Baselines Compared Against FileGramOS

| Method | Type | Description |
|--------|------|-------------|
| **filegramos\\_simple** | Procedural extraction | Deterministic feature extraction from events.json |
| **full\\_context** | Upper bound proxy | All trajectories dumped into context window |
| **memu** | Hierarchical | Hierarchical memory with real LLM ingest |
| **eager\\_sum** | Eager abstraction | LLM summarizes each session into NL |
| **naive\\_rag** | Retrieval | Chunk + Cohere embed-v4.0 + nearest retrieval |
| **evermemos** | 3-stage encoding | Encoding -> consolidation -> retrieval pipeline |
| **memos** | Memory extraction | Memory extraction + consolidation (real LLM ingest) |
| **zep** | Knowledge graph | Triple extraction -> graph, temporal edges |
| **mem0** | Interaction-driven | Azure LLM + HuggingFace embeddings + Qdrant |

**Key hypothesis**: FileGramOS's advantage is **Procedural** — deterministic feature extraction from events.json gives precise behavioral stats that LLM-based baselines cannot match. LLM baselines are already strong on Semantic (content analysis).

### Main Results

<div style="overflow-x:auto;margin:16px 0;">
<table style="border-collapse:collapse;font-size:13px;width:100%;min-width:900px;">
<thead>
<tr style="border-bottom:2px solid #555;">
  <th rowspan="2" style="text-align:left;padding:6px;">Method</th>
  <th colspan="2" style="text-align:center;padding:6px;border-left:1px solid #444;">Tokens</th>
  <th colspan="2" style="text-align:center;padding:6px;border-left:1px solid #444;">T1: Understanding</th>
  <th colspan="2" style="text-align:center;padding:6px;border-left:1px solid #444;">T2: Reasoning</th>
  <th colspan="2" style="text-align:center;padding:6px;border-left:1px solid #444;">T3: Detection</th>
  <th colspan="3" style="text-align:center;padding:6px;border-left:1px solid #444;">Channel</th>
  <th rowspan="2" style="text-align:center;padding:6px;border-left:1px solid #444;">Avg</th>
</tr>
<tr style="border-bottom:1px solid #555;font-size:12px;">
  <th style="text-align:right;padding:4px 6px;border-left:1px solid #444;">In.</th>
  <th style="text-align:right;padding:4px 6px;">Out.</th>
  <th style="text-align:center;padding:4px 6px;border-left:1px solid #444;">AttrRec</th>
  <th style="text-align:center;padding:4px 6px;">ProfRec</th>
  <th style="text-align:center;padding:4px 6px;border-left:1px solid #444;">BehavInf</th>
  <th style="text-align:center;padding:4px 6px;">TraceDis</th>
  <th style="text-align:center;padding:4px 6px;border-left:1px solid #444;">AnomDet</th>
  <th style="text-align:center;padding:4px 6px;">ShiftAna</th>
  <th style="text-align:center;padding:4px 6px;border-left:1px solid #444;">Proc</th>
  <th style="text-align:center;padding:4px 6px;">Sem</th>
  <th style="text-align:center;padding:4px 6px;">Epi</th>
</tr>
</thead>
<tbody>
<tr><td style="padding:4px 6px;">Full Context</td><td style="text-align:right;padding:4px 6px;border-left:1px solid #444;">625.2K</td><td style="text-align:right;padding:4px 6px;">45.9K</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;text-decoration:underline;">48.3</td><td style="text-align:center;padding:4px 6px;">50.0</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">31.9</td><td style="text-align:center;padding:4px 6px;text-decoration:underline;">75.2</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">42.5</td><td style="text-align:center;padding:4px 6px;">33.3</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;text-decoration:underline;">50.9</td><td style="text-align:center;padding:4px 6px;text-decoration:underline;">57.5</td><td style="text-align:center;padding:4px 6px;">37.5</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">46.9</td></tr>
<tr><td style="padding:4px 6px;">Naive RAG</td><td style="text-align:right;padding:4px 6px;border-left:1px solid #444;">625.2K</td><td style="text-align:right;padding:4px 6px;">3.0K</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">47.7</td><td style="text-align:center;padding:4px 6px;">46.8</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">24.7</td><td style="text-align:center;padding:4px 6px;">54.7</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">37.5</td><td style="text-align:center;padding:4px 6px;">22.4</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">40.1</td><td style="text-align:center;padding:4px 6px;">50.9</td><td style="text-align:center;padding:4px 6px;">29.3</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">37.1</td></tr>
<tr><td style="padding:4px 6px;">Eager Summ.</td><td style="text-align:right;padding:4px 6px;border-left:1px solid #444;">625.2K</td><td style="text-align:right;padding:4px 6px;">3.7K</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">43.6</td><td style="text-align:center;padding:4px 6px;text-decoration:underline;">55.6</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;text-decoration:underline;">41.3</td><td style="text-align:center;padding:4px 6px;">56.4</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">60.0</td><td style="text-align:center;padding:4px 6px;font-weight:bold;">42.2</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">47.3</td><td style="text-align:center;padding:4px 6px;">52.8</td><td style="text-align:center;padding:4px 6px;font-weight:bold;">50.3</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;text-decoration:underline;">48.6</td></tr>
<tr><td style="padding:4px 6px;">Mem0</td><td style="text-align:right;padding:4px 6px;border-left:1px solid #444;">119.9K</td><td style="text-align:right;padding:4px 6px;">3.0K</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">44.3</td><td style="text-align:center;padding:4px 6px;">48.1</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">23.8</td><td style="text-align:center;padding:4px 6px;">45.7</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">35.6</td><td style="text-align:center;padding:4px 6px;">29.7</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">35.3</td><td style="text-align:center;padding:4px 6px;">50.7</td><td style="text-align:center;padding:4px 6px;">32.4</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">35.4</td></tr>
<tr><td style="padding:4px 6px;">Zep</td><td style="text-align:right;padding:4px 6px;border-left:1px solid #444;">219.1K</td><td style="text-align:right;padding:4px 6px;">3.8K</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">46.3</td><td style="text-align:center;padding:4px 6px;">50.4</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">26.8</td><td style="text-align:center;padding:4px 6px;">55.1</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">48.1</td><td style="text-align:center;padding:4px 6px;">29.2</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">41.6</td><td style="text-align:center;padding:4px 6px;">49.4</td><td style="text-align:center;padding:4px 6px;">37.8</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">40.6</td></tr>
<tr><td style="padding:4px 6px;">MemOS</td><td style="text-align:right;padding:4px 6px;border-left:1px solid #444;">302.3K</td><td style="text-align:right;padding:4px 6px;">4.2K</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">44.3</td><td style="text-align:center;padding:4px 6px;">52.0</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">26.0</td><td style="text-align:center;padding:4px 6px;">53.4</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">36.9</td><td style="text-align:center;padding:4px 6px;">29.7</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">40.3</td><td style="text-align:center;padding:4px 6px;">48.5</td><td style="text-align:center;padding:4px 6px;">33.0</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">37.9</td></tr>
<tr><td style="padding:4px 6px;">MemU</td><td style="text-align:right;padding:4px 6px;border-left:1px solid #444;">293.6K</td><td style="text-align:right;padding:4px 6px;">7.9K</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;font-weight:bold;">49.0</td><td style="text-align:center;padding:4px 6px;">50.4</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">29.4</td><td style="text-align:center;padding:4px 6px;">62.0</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">54.4</td><td style="text-align:center;padding:4px 6px;text-decoration:underline;">40.6</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">45.7</td><td style="text-align:center;padding:4px 6px;">51.9</td><td style="text-align:center;padding:4px 6px;">46.9</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">46.6</td></tr>
<tr><td style="padding:4px 6px;">EverMemOS</td><td style="text-align:right;padding:4px 6px;border-left:1px solid #444;">1098.9K</td><td style="text-align:right;padding:4px 6px;">8.4K</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">47.0</td><td style="text-align:center;padding:4px 6px;font-weight:bold;">57.7</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">36.6</td><td style="text-align:center;padding:4px 6px;">59.4</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;font-weight:bold;">63.1</td><td style="text-align:center;padding:4px 6px;">35.9</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">47.5</td><td style="text-align:center;padding:4px 6px;">53.8</td><td style="text-align:center;padding:4px 6px;text-decoration:underline;">48.3</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">47.9</td></tr>
<tr style="border-top:2px solid #555;background:rgba(79,195,247,0.1);"><td style="padding:4px 6px;font-weight:bold;">FileGramOS</td><td style="text-align:right;padding:4px 6px;border-left:1px solid #444;">109.7K</td><td style="text-align:right;padding:4px 6px;">4.3K</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;text-decoration:underline;">48.3</td><td style="text-align:center;padding:4px 6px;">53.8</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;font-weight:bold;">51.9</td><td style="text-align:center;padding:4px 6px;font-weight:bold;">75.6</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;text-decoration:underline;">60.6</td><td style="text-align:center;padding:4px 6px;">34.9</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;font-weight:bold;">59.5</td><td style="text-align:center;padding:4px 6px;font-weight:bold;">59.4</td><td style="text-align:center;padding:4px 6px;">46.6</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;font-weight:bold;">55.2</td></tr>
<tr style="border-top:1px solid #555;color:#888;"><td style="padding:4px 6px;">No Context</td><td style="text-align:right;padding:4px 6px;border-left:1px solid #444;">—</td><td style="text-align:right;padding:4px 6px;">—</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">30.2</td><td style="text-align:center;padding:4px 6px;">—</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">17.0</td><td style="text-align:center;padding:4px 6px;">36.8</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">35.6</td><td style="text-align:center;padding:4px 6px;">28.1</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">27.2</td><td style="text-align:center;padding:4px 6px;">31.7</td><td style="text-align:center;padding:4px 6px;">31.5</td><td style="text-align:center;padding:4px 6px;border-left:1px solid #444;">29.1</td></tr>
</tbody>
</table>
<p style="color:#aaa;font-size:11px;margin-top:6px;">
All scores are accuracy (%), scaled 0–100. <b>Bold</b> = best, <u>underline</u> = second best.<br/>
<em>T1</em> — <b>AttrRec</b>: dim classification + semantic attribute (149 MCQ); <b>ProfRec</b>: open-ended profile reconstruction (Likert 1–5, rescaled).<br/>
<em>T2</em> — <b>BehavInf</b>: fingerprint + counterfactual + anomaly detection (235 MCQ); <b>TraceDis</b>: multi-user attribution + task separation (230 MCQ).<br/>
<em>T3</em> — <b>AnomDet</b>: impersonation + perturbation detection (156 MCQ); <b>ShiftAna</b>: shifted dim ID + direction + behavioral change description (288 MCQ).<br/>
<em>Channel</em> — <b>Proc</b>: procedural sub-types; <b>Sem</b>: semantic (sem_attr + ProfRec); <b>Epi</b>: episodic (AnomDet + ShiftAna).<br/>
<em>Tokens</em> — <b>In.</b>: total stored memory per profile; <b>Out.</b>: retrieved context per query (avg).
</p>
</div>

#### Key Findings

**(1) Structured extraction outperforms all memory baselines by preserving information structure.**
FileGramOS achieves the highest overall accuracy of 55.2%, surpassing the strongest baseline (Eager Summ., 48.6%) by +6.6pp and the Full Context upper bound (46.9%) by +8.3pp. The root cause is a fundamental difference in abstraction direction: all baselines adopt a *top-down* approach — an LLM interprets raw events through pre-defined semantic categories, imposing linguistic biases during ingestion — while FileGramOS follows a *bottom-up* path (events → typed features → aggregated statistics → query-time classification) with zero LLM calls during ingestion. The Full Context baseline scores *below* FileGramOS despite having access to strictly more information, confirming that raw recall is insufficient.

**(2) Behavioral inference demands quantitative precision that NL-based methods lack.**
FileGramOS leads Track 2 BehavInf by a wide margin (51.9% vs. 41.3% for Eager Summ.). Quantitative features make cross-profile comparisons unambiguous: \`reads=18, dirs=2, output=40K chars\` vs. \`reads=6, dirs=0, output=3.6K chars\`. NL-based baselines describe both profiles with similarly positive language ("systematic" and "structured"), obscuring genuine differences. On TraceDis, FileGramOS achieves 75.6%, matching Full Context (75.2%) while using ~10× fewer tokens.

**(3) Content–behavior confusion undermines interaction-driven and graph-based methods.**
Mem0 (35.4%) and Zep (40.6%) rank near the bottom, only marginally above No Context (29.1%). Mem0 forces file-system events into a pseudo-conversational format designed for dialogue preferences, stripping temporal and structural information. Zep extracts entity-relationship triples that conflate workspace *content* with user *behavior* — storing domain facts shared identically across all profiles as behavioral memories.

**(4) Eager abstraction destroys quantitative signal through linguistic homogenization.**
Eager Summ. achieves competitive overall accuracy (48.6%) but LLM summarizers describe *what* happened (task narratives) rather than *how* it happened (behavioral metrics). Statistics like \`search_ratio=0.35\` are reduced to "occasionally searches for information," and both a methodical deep-reader and a quick scanner are described as "systematic" and "thorough."

**(5) The procedural channel drives the advantage; semantic parity already exists.**
FileGramOS's lead is concentrated in procedural attributes (59.5% vs. 50.9% best baseline). On semantic attributes (name, role, language, tone) — recoverable from file content — FileGramOS leads by a smaller margin (59.4%). The episodic channel (46.6%) is slightly below Eager Summ. (50.3%), suggesting current CV-based consistency metrics don't yet capture temporal dynamics as effectively as narrative-level pattern matching.

**(6) Extreme token efficiency: structured features as behavioral compression.**
FileGramOS achieves 55.2% with ~4.3K tokens of retrieved context per query, compared to 45.9K for Full Context (46.9%) and 3.0K for Naive RAG (37.1%) — an 18pp accuracy gain over comparably-sized RAG. The stored memory is ~110K tokens per profile with maximum discrimination, while EverMemOS stores ~1M tokens with *less* discrimination — a 10× compression ratio with better quality.

**(7) Track 3 reveals a two-tier difficulty structure in episodic detection.**
On AnomDet (impersonation + perturbation detection), FileGramOS achieves 60.6%, competitive with EverMemOS (63.1%). However, ShiftAna proves universally difficult: the best method reaches only 42.2% (Eager Summ.), with FileGramOS at 34.9%. The hardest sub-type is *Behavioral Change Description*, which requires identifying a concrete before→after change without dimension labels — combining dimension identification and direction judgment into a single question.

### Question Design Principles

- All choices are **semantic natural-language descriptions** — no raw statistics in choices
- Task references include concrete names (e.g., "Investment Analyst Work Summary (T-01, understand)")
- Distractors preferentially use **fine-grained profile pairs** to maximize difficulty
- Deduplication: skip questions where profiles produce identical semantic predictions
`;
}

function renderInstructions() {
  const md = buildInstructionsMarkdown();
  const html = marked.parse(md);
  document.getElementById('instructionsContent').innerHTML = html;
  buildToc();
}

function buildToc() {
  const container = document.getElementById('instructionsContent');
  const headings = container.querySelectorAll('h1, h2, h3');
  const toc = document.getElementById('instructionsToc');
  let tocHtml = '';
  headings.forEach((h, i) => {
    const id = 'ins-heading-' + i;
    h.id = id;
    const level = h.tagName.toLowerCase();
    if (level === 'h1' || level === 'h2' || level === 'h3') {
      tocHtml += `<a href="#${id}" class="toc-${level}" onclick="document.getElementById('${id}').scrollIntoView({behavior:'smooth',block:'start'});return false;">${h.textContent}</a>`;
    }
  });
  toc.innerHTML = tocHtml;
}

renderInstructions();
