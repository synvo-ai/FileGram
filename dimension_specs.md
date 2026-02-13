# L/M/R Dimension Specifications

This document defines the **Left / Middle / Right (L/M/R) three-tier framework** for all 6 FileGram profile dimensions. Each dimension captures a distinct axis of behavioral variation observable from file-system traces.

## Framework Overview

Each dimension is specified across three layers:

| Layer | Purpose | Consumed By |
|-------|---------|-------------|
| **Layer 1: Behavioral Spec** | Concrete instructions for what the agent should DO. Injected into `system_prompt_addition` to drive agent behavior. | FileGram agent runtime |
| **Layer 2: Observable Indicators** | Computable metrics from `events.json` that distinguish L from M from R. Serve as ground truth for FileGramBench evaluation. | FileGramBench evaluation pipeline |
| **Layer 3: Task-Type Weights** | Per-task-type importance (Essential 5 / Important 3 / Optional 1) indicating how strongly each dimension should manifest. | Task selection and evaluation weighting |

**Design constraint**: Every Layer 1 instruction must produce at least one Layer 2 indicator. If a behavioral spec cannot be verified from events.json, it is not actionable and must be rewritten.

---

## Dimension A: Consumption Pattern (信息消费模式)

How users explore, locate, and understand information when facing a new set of files.

**Derivation**: Supports UC1 (Proactive Assistance) and UC4 (Context Recovery). An OS agent needs to know: what does this user read first? How do they find information? How deep do they read? Do they revisit?

|  | Left: sequential | Middle: targeted | Right: breadth-first |
|---|---|---|---|
| **Layer 1: Behavioral Spec** | Read files one by one in logical order (e.g., README first, then data files, then reports). Use `read` with full file ranges (line 1 to EOF). Read every available file before forming conclusions. Revisit files to cross-check facts. Do not use `grep` or `glob` to search for specific content — navigate by reading sequentially. | Search first, read second. Use `grep` to locate keywords and `glob` to find relevant file types before opening anything. Use `read` only on files matching search results. Read targeted sections (specific line ranges), not entire files. Skip files that search results indicate are irrelevant. | Scan the landscape before diving in. Use `bash ls` and `glob` to survey directory structure broadly. Use `read` with small line ranges (first 10-20 lines) to sample many files. Browse widely rather than reading deeply. Rarely revisit a file once scanned. Prioritize coverage over depth. |
| **Layer 2: Observable Indicators** | `count(file_read) / count(unique_files_in_workspace) >= 0.7` | `count(file_search) >= 3` | `count(file_browse) >= 3` |
|  | `mean(file_read.view_range[1] - file_read.view_range[0]) >= 0.8 * mean(file_read.content_length)` (reads most of each file) | `count(file_search) / count(file_read) >= 0.3` (searches relative to reads) | `mean(file_read.view_range[1] - file_read.view_range[0]) <= 30` (small read windows) |
|  | `count(file_read where view_count >= 2) >= 3` (revisits at least 3 files) | `mean(file_read.view_range[1] - file_read.view_range[0]) < 0.6 * mean(file_read.content_length)` (partial reads) | `count(unique_files_read) / count(file_read) >= 0.8` (few revisits, many distinct files) |
|  | `count(file_search) <= 1` (almost never searches) | `count(file_read where view_count >= 2) <= 2` (few revisits) | `count(file_search) <= 1` (browses, does not keyword-search) |
|  | Sequential ordering: timestamps of `file_read` events follow directory listing or logical order | Files opened after search: `file_search.files_opened_after` is non-empty for most searches | High file_browse to file_read ratio: `count(file_browse) / count(file_read) >= 0.3` |

### Layer 3: Task-Type Weights

| Task Type | Weight | Rationale |
|-----------|--------|-----------|
| Understand | Essential (5) | Core consumption task — reading strategy is the primary differentiator |
| Synthesize | Essential (5) | Must read multiple sources — strategy determines reading order and depth |
| Organize | Important (3) | Must browse/scan to understand current file state before reorganizing |
| Create | Optional (1) | Minimal reading needed for pure creation tasks |
| Iterate | Important (3) | Must read existing work + feedback before editing |
| Maintain | Important (3) | Must read prior output and new information |
| Multimodal | Important (3) | Must scan for image/data files across the workspace |

---

## Dimension B: Production Style (生产风格)

How users produce content — format choices, level of detail, structural complexity, and use of auxiliary materials.

**Derivation**: Supports UC2 (Personalized Defaults) and UC7 (Delegation Quality). An OS agent needs to know: what format does this user prefer? How detailed are their outputs? What structure do they use?

|  | Left: comprehensive | Middle: balanced | Right: minimal |
|---|---|---|---|
| **Layer 1: Behavioral Spec** | Produce thorough, detailed documents with hierarchical structure. Use multi-level headings (##, ###, ####). Include data tables, reference lists, appendices, and glossaries as separate files or sections. Create auxiliary files (e.g., README index, appendix, data summary) alongside main output. Write long-form prose with supporting evidence for every claim. Target 200+ lines per main output file. | Produce well-organized content with moderate detail. Use 2 levels of headings (##, ###). Include occasional tables where data warrants it. Write a single main output file with clear sections. Balance prose with bullet points. Target 80-150 lines per output file. | Produce concise, actionable outputs. Use flat bullet-point lists. No headings deeper than ##. One output file only — no auxiliary materials. No tables, no appendices, no reference lists. Every sentence must earn its place. Target under 60 lines per output file. |
| **Layer 2: Observable Indicators** | `mean(file_write.content_length where operation="create") >= 3000` chars | `mean(file_write.content_length where operation="create") in [1000, 3000]` chars | `mean(file_write.content_length where operation="create") <= 1000` chars |
|  | `count(file_write where operation="create") >= 3` per task (multiple output files) | `count(file_write where operation="create") in [1, 3]` per task | `count(file_write where operation="create") == 1` per task |
|  | `max(file_write.directory_depth for created files) >= 2` (creates files in subdirs) | Created files may be in root or 1 level deep | All created files at `directory_depth == 0` |
|  | Output files contain markdown heading patterns: `####` present (4th-level headings) | Output files contain `###` but not `####` | Output files contain at most `##` headings |
|  | Auxiliary file types created: README, index, appendix, glossary, or data summary | At most 1 auxiliary file | Zero auxiliary files |

### Layer 3: Task-Type Weights

| Task Type | Weight | Rationale |
|-----------|--------|-----------|
| Create | Essential (5) | Core production task — output style is the primary differentiator |
| Synthesize | Essential (5) | Produces summary documents — style directly observable |
| Multimodal | Important (3) | Creates documents with varying detail and visual integration |
| Understand | Important (3) | Produces a summary/briefing — format and detail vary |
| Iterate | Important (3) | Revision output reflects production style |
| Maintain | Important (3) | Update output reflects production habits |
| Organize | Optional (1) | Primarily about file operations, not content production |

---

## Dimension C: Organization Preference (组织偏好)

How users manage file-system structure: directory hierarchy, naming conventions, version handling, and cleanup behavior.

**Derivation**: Supports UC3 (Smart Organization), UC5 (Behavioral Continuity), and UC6 (Conflict Detection). An OS agent needs to know: how does this user structure directories? How do they name files? How do they handle old versions?

|  | Left: deeply_nested | Middle: adaptive | Right: flat |
|---|---|---|---|
| **Layer 1: Behavioral Spec** | Create deep directory trees (3+ levels) organized by topic or category. Use `bash mkdir -p` to build hierarchical structures. Move files into appropriate subdirectories with `bash mv`. Use long, descriptive file names with prefixes (e.g., `01_fundamental_analysis_framework.md`). Create README or index files at each directory level. Before modifying any file, create a backup copy (`bash cp file file.bak` or move to `_archive/`). Never delete old versions. | Create 1-2 levels of directories when the file count warrants it. Use practical file names that balance clarity with brevity. Create directories only when there are 3+ files of a similar type. Use a mix of naming styles depending on context. Keep old versions only when explicitly important. | Keep all files in the root working directory. Do not create subdirectories — use `write` to place files directly in the workspace root. Use short, abbreviated file names (e.g., `summary.md`, `notes.txt`, `rpt.md`). Overwrite files directly rather than creating new versions. Delete temporary or intermediate files with `bash rm` when done. |
| **Layer 2: Observable Indicators** | `count(dir_create) >= 3` per task | `count(dir_create) in [1, 2]` per task | `count(dir_create) == 0` per task |
|  | `max(dir_create.depth) >= 3` | `max(dir_create.depth) in [1, 2]` | No dir_create events |
|  | `count(file_move) >= 2` (actively files documents into dirs) | `count(file_move) in [0, 2]` | `count(file_move) == 0` |
|  | `count(file_copy where is_backup=true) >= 1` (creates backups) | Backup behavior inconsistent | `count(file_copy where is_backup=true) == 0` |
|  | `count(file_rename) >= 1` with descriptive naming patterns | Mixed naming patterns | `count(file_delete) >= 1` (cleans up temp files) |
|  | `count(file_delete) == 0` (never deletes) | Occasional deletes | `file_write.operation == "overwrite"` events present |
|  | Mean filename length (from file_write/file_rename new_path) >= 25 chars | Mean filename length in [12, 25] chars | Mean filename length <= 12 chars |

### Layer 3: Task-Type Weights

| Task Type | Weight | Rationale |
|-----------|--------|-----------|
| Organize | Essential (5) | Core organization task — directory and naming choices are the primary output |
| Create | Essential (5) | Must decide where to place created files and how to name them |
| Maintain | Important (3) | Must integrate new content into existing structure |
| Multimodal | Important (3) | Must organize mixed media files |
| Iterate | Important (3) | Version handling behavior (backup vs overwrite) is directly tested |
| Understand | Optional (1) | Primarily about reading, minimal file organization |
| Synthesize | Optional (1) | Focus is on content synthesis, not file management |

---

## Dimension D: Iteration Strategy (迭代策略)

How users refine and modify existing work: edit granularity, frequency, backup behavior before editing, and willingness to rewrite vs. patch.

**Derivation**: Supports UC5 (Behavioral Continuity) and UC6 (Conflict Detection). An OS agent needs to know: does this user make many small edits or few large ones? Do they preserve history? Do they rewrite or refine?

|  | Left: incremental | Middle: balanced | Right: rewrite |
|---|---|---|---|
| **Layer 1: Behavioral Spec** | Make many small, targeted edits. Use `edit` (not `write`) to modify specific sections — change a few lines at a time. Review the result after each edit before proceeding. Create a backup copy before any modification (`bash cp`). Make multiple passes over the same file, refining progressively. Never delete content — rephrase or restructure instead. High iteration count with small changes per iteration. | Make moderate edits of reasonable scope. Use `edit` for focused changes and `write` only when creating new content. Review occasionally between edits. Balance between incremental refinement and efficient completion. | Prefer wholesale replacement over patching. When revisions are needed, use `write` to overwrite the entire file with a new version rather than making multiple small `edit` calls. If an edit exceeds ~30% of the file, rewrite the whole file. Delete and recreate files rather than extensively modifying them. Make one decisive pass, not multiple refinement rounds. No backup copies needed — the new version replaces the old. |
| **Layer 2: Observable Indicators** | `count(file_edit) >= 5` per modified file | `count(file_edit) in [2, 4]` per modified file | `count(file_edit) <= 1` per modified file |
|  | `mean(file_edit.lines_modified) <= 10` (small edits) | `mean(file_edit.lines_modified) in [10, 30]` | `mean(file_edit.lines_modified) >= 30` OR file rewritten via `file_write` overwrite |
|  | `count(file_copy where is_backup=true) >= 1` before editing | Backup behavior optional | `count(file_copy where is_backup=true) == 0` |
|  | `count(iteration_end) / count(unique_files_modified) >= 3` (multiple passes per file) | Ratio in [1.5, 3] | Ratio <= 1.5 |
|  | `count(file_write where operation="overwrite") == 0` (never overwrites) | May occasionally overwrite | `count(file_write where operation="overwrite") >= 1` |
|  | `file_edit.lines_deleted` is low relative to `lines_added` (reshapes, doesn't delete) | Balanced add/delete ratio | `file_delete` followed by `file_write` on same path (delete-recreate pattern) |

### Layer 3: Task-Type Weights

| Task Type | Weight | Rationale |
|-----------|--------|-----------|
| Iterate | Essential (5) | Core iteration task — edit strategy is the primary differentiator |
| Maintain | Essential (5) | Updating existing work directly tests iteration approach |
| Create | Important (3) | Self-revision during creation reveals iteration habits |
| Synthesize | Important (3) | May revise drafts during synthesis |
| Organize | Optional (1) | File operations, not content editing |
| Understand | Optional (1) | Primarily reading, minimal editing |
| Multimodal | Optional (1) | Focus is on cross-modal content, not edit patterns |

---

## Dimension E: Work Rhythm (工作节奏)

Temporal patterns of work: whether the user works in distinct phases, maintains a steady pace, or operates in rapid bursts with frequent context switches.

**Derivation**: Supports UC4 (Context Recovery) and UC5 (Behavioral Continuity). An OS agent needs to know: does this user work in phases? How often do they switch between files? Is there a clear planning stage before execution?

|  | Left: phased | Middle: steady | Right: bursty |
|---|---|---|---|
| **Layer 1: Behavioral Spec** | Work in clear, distinct phases. Phase 1: read/browse all materials without writing anything. Phase 2: plan and outline (create a plan file or use structured thinking). Phase 3: execute — write all outputs. Phase 4: review and verify by re-reading outputs and source files. Complete each phase fully before moving to the next. Minimize context switches — stay focused on one file at a time for extended stretches. | Work at a consistent pace. Interleave reading and writing naturally — read a file, write some notes, read another file, write more. Moderate context switching. No extreme clustering of reads or writes. Process files in a logical but not rigidly sequential order. | Work in rapid, reactive bursts. Frequently switch between files. Interleave reads and writes quickly — read a snippet, immediately write, jump to another file, come back. High context-switch rate. May start writing before finishing reading. Jump between tasks/files based on what seems most urgent or interesting in the moment. |
| **Layer 2: Observable Indicators** | Phase detection: first N% of events are `file_read`/`file_browse`/`file_search` only, last M% are `file_write`/`file_edit` only, with N >= 40 and M >= 30 | Read and write events are interspersed throughout the timeline | Read and write events are highly intermixed with no phase separation |
|  | `count(context_switch) / count(iteration_end) <= 1.5` (few switches per iteration) | `count(context_switch) / count(iteration_end) in [1.5, 3.0]` | `count(context_switch) / count(iteration_end) >= 3.0` (many switches per iteration) |
|  | Longest consecutive same-file streak >= 5 tool calls | Longest consecutive same-file streak in [3, 5] | Longest consecutive same-file streak <= 2 |
|  | `context_switch.trigger` values are orderly (e.g., "sequential_access", "planned_read") | Mixed trigger types | `context_switch.trigger` values include "reactive", "interrupt", or rapid back-and-forth |
|  | Low variance in inter-event timing (consistent pace within phases) | Moderate variance | High variance in inter-event timing (bursts of rapid activity with pauses) |

### Layer 3: Task-Type Weights

| Task Type | Weight | Rationale |
|-----------|--------|-----------|
| Synthesize | Essential (5) | Multi-step task with reading + writing — phased vs bursty is most visible |
| Maintain | Essential (5) | Continuing prior work — rhythm is directly observable |
| Understand | Important (3) | Reading-heavy task — phase structure (read-then-summarize vs interleaved) is visible |
| Create | Important (3) | Planning-then-executing vs diving in is observable |
| Iterate | Important (3) | Review-then-edit vs rapid-fire editing patterns |
| Organize | Important (3) | Browse-then-organize vs interleaved scanning and moving |
| Multimodal | Optional (1) | Rhythm is secondary to cross-modal behavior |

---

## Dimension F: Cross-Modal Behavior (跨模态行为)

Whether and how users create, reference, and integrate visual materials (images, charts, diagrams, data tables) alongside text.

**Derivation**: Supports UC2 (Personalized Defaults) and UC7 (Delegation Quality). An OS agent needs to know: does this user create visual materials? Do their documents reference images? Do they maintain text-image consistency?

|  | Left: visual-heavy | Middle: balanced | Right: text-only |
|---|---|---|---|
| **Layer 1: Behavioral Spec** | Actively create visual materials to support text content. Use `bash` to generate charts, diagrams, or structured data visualizations (e.g., Mermaid diagrams in markdown, ASCII art, CSV data tables for charting). Reference images and figures explicitly in documents with numbered figure captions (e.g., "See Figure 1"). Create cross-references between text files and data/image files. Maintain a visual assets directory for generated materials. When summarizing data, always include a visual representation alongside text. | Include structured data tables in markdown documents where data warrants it. Use markdown table syntax for comparisons, summaries, and data presentation. Occasionally reference external data files. Do not create standalone image files, but format text to be visually structured (tables, aligned columns, clear formatting). | Produce pure text output only. No tables, no images, no diagrams, no charts. Use prose paragraphs and bullet-point lists exclusively. Never create image files or reference visual materials. If data must be presented, use inline text descriptions rather than tables. Keep all content in plain narrative or list format. |
| **Layer 2: Observable Indicators** | `count(file_write where file_type in ["png","jpg","svg","csv","mermaid"]) >= 1` (creates visual files) | Output files contain markdown table syntax (`\|---\|`) in at least 1 created file | `count(file_write where file_type in ["png","jpg","svg","csv"]) == 0` |
|  | `count(cross_file_reference where target_file matches image/data extensions) >= 2` | `count(cross_file_reference where reference_type involves data files) in [1, 2]` | `count(cross_file_reference where target_file matches image/data extensions) == 0` |
|  | Created text files contain figure reference patterns ("Figure \d", "Fig\.", "see chart", "see diagram") | Created text files contain table headers but no figure references | Created text files contain no table syntax and no figure references |
|  | `file_write` events include files in a dedicated media/assets/figures directory | No dedicated media directory | No media-related file operations at all |
|  | `bash` commands include image generation or data visualization tools | `bash` commands do not generate images | No visualization-related bash commands |

### Layer 3: Task-Type Weights

| Task Type | Weight | Rationale |
|-----------|--------|-----------|
| Multimodal | Essential (5) | Core cross-modal task — visual behavior is the primary differentiator |
| Create | Important (3) | Whether user adds visual materials to created content |
| Synthesize | Important (3) | Whether user creates comparison tables or visualizations during synthesis |
| Understand | Important (3) | Whether summary includes visual representations of findings |
| Iterate | Optional (1) | Focus is on edit strategy, not modality |
| Organize | Optional (1) | Focus is on file structure, not content modality |
| Maintain | Optional (1) | Focus is on update strategy, not modality |

---

## Cross-Dimension Signal Map

Some events serve multiple dimensions. This is by design — FileGramOS's deferred abstraction means the same signal can be interpreted differently depending on the query.

| Event Type | Dimensions Served | Why It Crosses Dimensions |
|---|---|---|
| `file_read` | **A** + E | Reading depth/order (A) and temporal pattern of reads (E) |
| `file_write` | **B** + C + F | Content length/structure (B), file placement/naming (C), file type created (F) |
| `file_edit` | **B** + **D** | Content style in edits (B) and edit granularity/frequency (D) |
| `file_search` | **A** | Primary indicator for targeted consumption (A:M) |
| `file_browse` | **A** + E | Browsing behavior (A) and phase detection (E) |
| `dir_create` | **C** | Primary indicator for organization depth (C) |
| `file_move` | **C** | Filing behavior and organization preference (C) |
| `file_rename` | **C** | Naming convention indicator (C) |
| `file_copy` | **C** + **D** | Backup behavior serves both organization (C) and iteration safety (D) |
| `file_delete` | **C** + D | Cleanup behavior (C) and willingness to discard (D) |
| `context_switch` | A + **E** | Navigation pattern (A) and work rhythm (E) |
| `cross_file_reference` | A + D + **F** | Reading flow (A), edit triggers (D), text-image linking (F) |
| `iteration_start/end` | D + **E** | Iteration count (D) and pacing (E) |
| `fs_snapshot` | C + E | Organization state (C) and when structure changes (E) |
| `error_encounter/response` | D | Error handling approach reveals iteration resilience |

**Bold** indicates the primary dimension served.

---

## Pilot Profile Vectors

The 6 pilot profiles are assigned L/M/R values to maximize diagnostic coverage with deliberate overlap and differentiation patterns.

| Profile | A (Consumption) | B (Production) | C (Organization) | D (Iteration) | E (Rhythm) | F (Cross-Modal) |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| **p1_methodical** | L (sequential) | L (comprehensive) | L (deeply_nested) | L (incremental) | L (phased) | M (balanced) |
| **p2_thorough_reviser** | L (sequential) | L (comprehensive) | R (flat) | R (rewrite) | L (phased) | M (balanced) |
| **p3_efficient_executor** | M (targeted) | R (minimal) | R (flat) | R (rewrite) | R (bursty) | R (text-only) |
| **p4_structured_analyst** | M (targeted) | L (comprehensive) | M (adaptive) | L (incremental) | M (steady) | L (visual-heavy) |
| **p5_balanced_organizer** | R (breadth-first) | M (balanced) | M (adaptive) | M (balanced) | M (steady) | M (balanced) |
| **p6_quick_curator** | R (breadth-first) | M (balanced) | L (deeply_nested) | R (rewrite) | R (bursty) | R (text-only) |

### Design Validation

**Fine-grained pair**: p1 vs p2 share A:L, B:L, E:L, F:M but differ on C (L vs R) and D (L vs R). Tests whether the evaluation can detect organization and iteration differences when consumption and production style are identical.

**Coarse-grained pair**: p1 vs p3 differ on 5 of 6 dimensions (A:L/M, B:L/R, C:L/R, D:L/R, E:L/R). Only F is close (M vs R). Tests whether broad behavioral differences are detectable.

**Interesting mixes**:
- p4: Targeted reader (A:M) who writes comprehensively (B:L) with visual materials (F:L) — searches like p3 but produces like p1
- p6: Breadth-first browser (A:R) who uses deeply nested dirs (C:L) — rapid scanning paired with meticulous organization

**Per-dimension coverage** (each tier value appears in at least 2 profiles):

| Dim | Left | Middle | Right |
|-----|------|--------|-------|
| A | p1, p2 | p3, p4 | p5, p6 |
| B | p1, p2, p4 | p5, p6 | p3 |
| C | p1, p6 | p4, p5 | p2, p3 |
| D | p1, p4 | p5 | p2, p3, p6 |
| E | p1, p2 | p4, p5 | p3, p6 |
| F | p4 | p1, p2, p5 | p3, p6 |
