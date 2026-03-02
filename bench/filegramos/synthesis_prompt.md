# FileGramOS Simple: LLM Synthesis Prompt (Step 3)

## System Prompt

You are an expert at inferring user work habit profiles from file-system behavioral statistics. You will be given structured feature statistics extracted from a user's behavioral trajectories across multiple tasks. Your job is to synthesize these statistics into a coherent profile.

## User Prompt Template

```
Below are aggregated behavioral statistics from {n_trajectories} file-system task trajectories performed by a single user. These statistics were deterministically extracted from the user's file operations (reads, writes, edits, directory operations, etc.).

{aggregated_features}

Based on these statistics, infer this user's work habit profile for the following attributes. For each attribute, use the statistics to justify your inference.

Attributes to infer:
{attributes_list}

Attribute value options (choose the best fit):
- reading_strategy: sequential_deep | breadth_first | targeted_search
- output_detail: detailed | moderate | minimal
- output_structure: hierarchical | flat | freeform
- directory_style: nested_by_topic | flat | by_type
- naming: date_prefix_descriptive | long_descriptive | short_abbrev
- edit_strategy: incremental_small | bulk_rewrite
- version_strategy: keep_history | archive_old | overwrite
- tone: professional | friendly | casual
- verbosity: detailed | balanced | concise

For each attribute, explain which specific statistics support your inference.

Respond in JSON format:
{
  "inferred_profile": {
    "<attribute_name>": {
      "value": "<chosen value from options>",
      "justification": "<which statistics support this, and why>"
    },
    ...
  }
}
```

## Why This Prompt is Different from Baselines

1. **Structured input**: The LLM receives pre-computed statistics (ratios, means, counts), NOT raw event narratives. This reduces information overload.

2. **Attribute-aligned features**: Each statistic section corresponds to a specific profile attribute. The LLM doesn't need to figure out WHAT to look at — only HOW to interpret it.

3. **Cross-trajectory aggregation**: Statistics are already aggregated across 10 trajectories. Baselines typically see individual event sequences without cross-trajectory perspective.

4. **File-system-specific features**: Statistics like `avg_dir_depth`, `naming_pattern`, `small_edit_ratio` are domain-specific features that dialogue-memory systems don't extract.

## Expected Advantage

FileGramOS Simple should excel especially on:
- **directory_style**: explicit depth and nesting statistics
- **naming**: pattern detection (date prefix, length, separator style)
- **edit_strategy**: quantitative small_edit_ratio and lines_per_edit
- **version_strategy**: explicit backup/delete/overwrite counts

And may have similar performance to baselines on:
- **tone**: requires content analysis, not just structural statistics
- **output_structure**: partially structural, partially content-dependent
