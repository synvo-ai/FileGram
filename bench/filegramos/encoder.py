"""Stage 1: Per-trajectory Engram encoding.

Wraps FeatureExtractor to produce typed Engram objects with procedural,
semantic, and fingerprint data. Optionally uses LLM for semantic encoding.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

from .engram import (
    ContentSample,
    CrossRef,
    EditChainSample,
    Engram,
    SemanticUnit,
)
from .feature_extraction import FeatureExtractor
from .fingerprint import compute_fingerprint
from .parsers import ParserRegistry
from .schema import (
    MAX_DIVERSITY_ATTRIBUTES,
    SIMULATION_TYPES,
    ConsumerEventType,
    NormalizedEvent,
)
from .tuning import (
    LLM_ENCODE_EDIT_CHARS,
    LLM_ENCODE_FILE_CHARS,
    LLM_ENCODE_MAX_EDITS,
    LLM_ENCODE_MAX_FILES,
)


class EngramEncoder:
    """Encode a single trajectory into an Engram.

    Reuses FeatureExtractor for procedural + semantic extraction,
    then layers on fingerprint computation and importance scoring.
    llm_fn is required for semantic encoding.
    """

    def __init__(self, llm_fn: Callable | None = None):
        """
        Args:
            llm_fn: Callback for LLM-based semantic encoding (required).
                Signature: (prompt: str, system_prompt: str | None) -> str
        """
        if llm_fn is None:
            raise ValueError("llm_fn is required for EngramEncoder")
        self._llm_fn = llm_fn
        self._parser = ParserRegistry()

    def encode(
        self,
        events: list[dict[str, Any]] | list[NormalizedEvent],
        task_id: str,
        trajectory_id: str = "",
        is_perturbed: bool = False,
    ) -> Engram:
        """Encode a trajectory's events into an Engram.

        Args:
            events: Event list — either raw dicts (with _resolved_content)
                or pre-normalised ``NormalizedEvent`` instances.
            task_id: The task identifier (e.g., "T-01").
            trajectory_id: Unique ID for this trajectory (e.g., "p1_methodical_T-01").
            is_perturbed: Whether this trajectory used perturbed profile settings.

        Returns:
            A fully populated Engram.
        """
        _is_typed = bool(events) and isinstance(events[0], NormalizedEvent)

        # Filter behavioral events
        if _is_typed:
            behavioral_count = sum(
                1
                for e in events
                if e.event_type.value not in SIMULATION_TYPES  # type: ignore[union-attr]
            )
        else:
            behavioral_count = sum(
                1
                for e in events
                if e.get("event_type") not in SIMULATION_TYPES  # type: ignore[union-attr]
            )

        # 1. Procedural: reuse FeatureExtractor entirely (handles both types)
        extractor = FeatureExtractor(events)
        procedural = extractor.extract_all()
        auxiliary = extractor.extract_auxiliary()

        # 2. Semantic: build typed SemanticUnit from raw extraction
        raw_semantic = extractor.extract_semantic_channel()
        semantic = self._build_semantic_unit(events, raw_semantic)

        # 3. LLM semantic encoding (per-trajectory behavioral narrative)
        if self._llm_fn and (semantic.created_files or semantic.edit_chains):
            semantic.llm_encoding = self._llm_encode_semantic(
                semantic,
                procedural,
                task_id,
            )

        # 4. Fingerprint
        fingerprint = compute_fingerprint(procedural)

        # 5. Importance scoring
        if _is_typed:
            behavioral_list: list[dict] = []  # not used in typed path
        else:
            behavioral_list = [
                e
                for e in events  # type: ignore[union-attr]
                if e.get("event_type") not in SIMULATION_TYPES
            ]
        importance = self._compute_importance(
            procedural,
            behavioral_list,
            events if not _is_typed else [],  # type: ignore[arg-type]
            behavioral_count=behavioral_count,
            total_count=len(events),
        )

        return Engram(
            trajectory_id=trajectory_id,
            task_id=task_id,
            procedural=procedural,
            auxiliary=auxiliary,
            semantic=semantic,
            fingerprint=fingerprint,
            event_count=len(events),
            behavioral_event_count=behavioral_count,
            importance_score=importance,
            is_perturbed=is_perturbed,
        )

    def _build_semantic_unit(
        self,
        events: list[dict[str, Any]] | list[NormalizedEvent],
        raw_semantic: dict[str, Any],
    ) -> SemanticUnit:
        """Convert raw semantic extraction to typed SemanticUnit."""
        # Created files -> ContentSample
        created_files = []
        for cf in raw_semantic.get("created_files", []):
            path = cf.get("path", "")
            ext = PurePosixPath(path).suffix.lstrip(".") if path else ""
            preview = cf.get("preview", "")
            parsed = self._parser.parse(preview, ext) if preview else ""
            created_files.append(
                ContentSample(
                    path=path,
                    content_length=cf.get("content_length", 0),
                    sample_type="create",
                    content_preview=parsed,
                    file_type=ext or "unknown",
                    full_content=cf.get("content", ""),
                )
            )

        # Edit chains -> EditChainSample
        edit_chains = []
        for ed in raw_semantic.get("edit_diffs", []):
            edit_chains.append(
                EditChainSample(
                    path=ed.get("path", ""),
                    lines_added=ed.get("lines_added", 0),
                    lines_deleted=ed.get("lines_deleted", 0),
                    diff_preview=ed.get("diff_preview", ""),
                    full_diff=ed.get("diff_content", ""),
                )
            )

        # Cross-file references — handle both typed and legacy events
        cross_refs = []
        _is_typed = bool(events) and isinstance(events[0], NormalizedEvent)
        if _is_typed:
            for e in events:
                ne: NormalizedEvent = e  # type: ignore[assignment]
                if ne.event_type == ConsumerEventType.CROSS_FILE_REFERENCE:
                    cross_refs.append(
                        CrossRef(
                            source_file=self._anonymize_path(ne.source_file),
                            target_file=self._anonymize_path(ne.target_file),
                            reference_type=ne.reference_type,
                        )
                    )
        else:
            for e in events:
                if e.get("event_type") == "cross_file_reference":  # type: ignore[union-attr]
                    cross_refs.append(
                        CrossRef(
                            source_file=self._anonymize_path(e.get("source_file", "")),  # type: ignore[union-attr]
                            target_file=self._anonymize_path(e.get("target_file", "")),  # type: ignore[union-attr]
                            reference_type=e.get("reference_type", ""),  # type: ignore[union-attr]
                        )
                    )

        return SemanticUnit(
            created_files=created_files,
            edit_chains=edit_chains,
            cross_file_refs=cross_refs,
            created_filenames=raw_semantic.get("created_filenames", []),
            dir_structure_diff=raw_semantic.get("dir_structure_diff", []),
        )

    def _llm_encode_semantic(
        self,
        semantic: SemanticUnit,
        procedural: dict[str, Any],
        task_id: str,
    ) -> dict[str, Any] | None:
        """Use LLM to generate a structured behavioral narrative for this trajectory.

        Analyzes both the content produced AND the operation patterns to create
        a rich per-trajectory behavioral encoding that deterministic extraction
        cannot capture (writing style nuances, organizational rationale, etc.).

        Returns a dict with behavioral narrative, or None on failure.
        """
        # Build content evidence
        content_parts = []
        for cf in semantic.created_files[:LLM_ENCODE_MAX_FILES]:
            if cf.content_preview:
                content_parts.append(
                    f"[Created] {cf.path} ({cf.content_length} chars, {cf.file_type})\n"
                    f"{cf.content_preview[:LLM_ENCODE_FILE_CHARS]}"
                )
        for ec in semantic.edit_chains[:LLM_ENCODE_MAX_EDITS]:
            if ec.diff_preview:
                content_parts.append(
                    f"[Edited] {ec.path} (+{ec.lines_added}/-{ec.lines_deleted} lines)\n"
                    f"{ec.diff_preview[:LLM_ENCODE_EDIT_CHARS]}"
                )

        if not content_parts:
            return None

        # Build procedural evidence summary
        proc_summary_parts = []
        for attr, feats in procedural.items():
            notable = []
            for k, v in feats.items():
                if isinstance(v, (int, float)) and v > 0:
                    notable.append(f"{k}={v}")
                elif isinstance(v, bool) and v:
                    notable.append(k)
            if notable:
                proc_summary_parts.append(f"{attr}: {', '.join(notable[:4])}")

        proc_evidence = "\n".join(proc_summary_parts[:8]) if proc_summary_parts else "(no procedural data)"

        prompt = (
            f"You are analyzing behavioral data from task '{task_id}'. "
            "A user performed file operations and created/edited content. "
            "Based on BOTH the content samples and operation statistics below, "
            "write a concise behavioral profile for THIS specific task session.\n\n"
            "## Content Evidence\n"
            + "\n\n".join(content_parts)
            + "\n\n## Operation Statistics\n"
            + proc_evidence
            + "\n\n## Instructions\n"
            "Respond in JSON with these fields:\n"
            "{\n"
            '  "writing_style": "1-2 sentences describing tone, formality, language choice",\n'
            '  "content_structure": "1-2 sentences on document organization (headings, tables, lists)",\n'
            '  "detail_level": "comprehensive | moderate | minimal — with brief justification",\n'
            '  "work_approach": "1-2 sentences on how they approached the task (methodical/reactive/etc)",\n'
            '  "notable_habits": ["list", "of", "2-4", "distinctive behavioral observations"]\n'
            "}\n"
            "Be SPECIFIC — cite concrete evidence from the content/stats. "
            "Avoid generic descriptions."
        )

        try:
            result = self._llm_fn(prompt, system_prompt=None)
            import json

            # Handle trailing commas
            import re as _re

            text = result.strip()
            if "```" in text:
                lines = text.split("\n")
                text = "\n".join(l for l in lines if not l.strip().startswith("```"))
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start : end + 1]
            text = _re.sub(r",\s*([}\]])", r"\1", text)
            return json.loads(text)
        except Exception:
            return None

    def _compute_importance(
        self,
        procedural: dict[str, Any],
        behavioral: list[dict],
        all_events: list[dict],
        *,
        behavioral_count: int | None = None,
        total_count: int | None = None,
    ) -> float:
        """Score trajectory importance based on behavioral richness.

        Higher scores for trajectories with:
        - More behavioral events (more signal)
        - Greater feature diversity (more dimensions activated)
        - More unique file operations

        When *behavioral_count* / *total_count* are given (typed path),
        they are used directly instead of computing from the raw lists.
        """
        n_behavioral = behavioral_count if behavioral_count is not None else len(behavioral)
        n_total = total_count if total_count is not None else len(all_events)
        n_total = max(n_total, 1)
        density = n_behavioral / n_total

        # Feature diversity: how many attributes have non-zero signals
        diversity = 0
        for attr_name, attr_dict in procedural.items():
            for key, val in attr_dict.items():
                if isinstance(val, (int, float)) and val > 0:
                    diversity += 1
                    break
                if isinstance(val, bool) and val:
                    diversity += 1
                    break

        importance = density * (1 + diversity / MAX_DIVERSITY_ATTRIBUTES)
        return round(importance, 4)

    @staticmethod
    def _anonymize_path(path: str) -> str:
        """Strip sandbox prefix from path."""
        if not path or path == "?":
            return path
        m = re.search(r"/sandbox/[^/]+/(.*)", path)
        if m:
            return m.group(1) or "."
        return path.rsplit("/", 1)[-1] if "/" in path else path
