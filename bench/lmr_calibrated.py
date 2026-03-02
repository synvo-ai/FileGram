#!/usr/bin/env python3
"""Calibrated L/M/R classifier using multi-feature rules fitted to actual data distributions.

Goal: find the CEILING accuracy — what's the best we can do with the current traces?
Uses multiple features per dimension with thresholds calibrated to actual data.
"""

import json
import statistics
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent
sys.path.insert(0, str(_BENCH))

SIGNAL_DIR = _ROOT / "signal"
MATRIX_PATH = _ROOT / "profiles" / "profile_matrix.json"

with open(MATRIX_PATH) as f:
    _matrix = json.load(f)
GT = {pid: info["dimensions"] for pid, info in _matrix["profiles"].items()}
DIMS = ["A", "B", "C", "D", "E", "F"]


def load_features(profile_id: str) -> list[dict]:
    from filegramos.feature_extraction import FeatureExtractor

    traj_dirs = sorted(SIGNAL_DIR.glob(f"{profile_id}_T-*"))
    features_list = []
    for td in traj_dirs:
        events_path = td / "events.json"
        if not events_path.exists():
            continue
        with open(events_path) as f:
            events = json.load(f)
        fe = FeatureExtractor(events)
        features_list.append(fe.extract_all())
    return features_list


def mean_of(fl, attr, key, default=0):
    vals = [f.get(attr, {}).get(key, default) for f in fl]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else default


def compute_phase_features(profile_id: str) -> dict:
    """Compute E-dimension phase features directly from raw events.json.

    Returns:
        read_phase_ratio: fraction of events in first 40% that are reads/browses/searches
        write_late_ratio: fraction of writes that happen in the last 40%
        longest_read_streak: max consecutive read-only operations before first write
    """
    READ_TYPES = {"file_read", "file_browse", "file_search"}
    WRITE_TYPES = {"file_write", "file_edit"}

    traj_dirs = sorted(SIGNAL_DIR.glob(f"{profile_id}_T-*"))
    all_read_phase = []
    all_write_late = []
    all_streaks = []

    for td in traj_dirs:
        events_path = td / "events.json"
        if not events_path.exists():
            continue
        with open(events_path) as f:
            events = json.load(f)

        # Filter to behavioral events only
        behavioral = [e for e in events if e.get("event_type") in READ_TYPES | WRITE_TYPES]
        if len(behavioral) < 5:
            continue

        n = len(behavioral)
        first_40 = behavioral[: int(n * 0.4)]
        last_40 = behavioral[int(n * 0.6) :]

        # Read phase concentration: what fraction of first 40% are reads?
        reads_in_first = sum(1 for e in first_40 if e.get("event_type") in READ_TYPES)
        read_phase = reads_in_first / max(len(first_40), 1)
        all_read_phase.append(read_phase)

        # Write late concentration: what fraction of writes are in last 40%?
        total_writes = sum(1 for e in behavioral if e.get("event_type") in WRITE_TYPES)
        writes_in_last = sum(1 for e in last_40 if e.get("event_type") in WRITE_TYPES)
        if total_writes > 0:
            all_write_late.append(writes_in_last / total_writes)

        # Longest read-only streak at start
        streak = 0
        for e in behavioral:
            if e.get("event_type") in READ_TYPES:
                streak += 1
            else:
                break
        all_streaks.append(streak)

    return {
        "read_phase_ratio": statistics.mean(all_read_phase) if all_read_phase else 0,
        "write_late_ratio": statistics.mean(all_write_late) if all_write_late else 0,
        "initial_read_streak": statistics.mean(all_streaks) if all_streaks else 0,
    }


def classify_calibrated(fl: list[dict]) -> dict[str, str]:
    """Multi-feature calibrated classification."""
    pred = {}

    # ── A: Consumption ──────────────────────────────────────
    # Key insight: search_ratio separates M, browse_ratio separates R from L
    search_r = mean_of(fl, "reading_strategy", "search_ratio")
    browse_r = mean_of(fl, "reading_strategy", "browse_ratio")
    revisit_r = mean_of(fl, "reading_strategy", "revisit_ratio")
    total_searches = mean_of(fl, "reading_strategy", "total_searches")

    if search_r >= 0.05 or total_searches >= 0.5:
        pred["A"] = "M"  # targeted: has meaningful search activity
    elif browse_r >= 0.40:
        pred["A"] = "R"  # breadth-first: heavy browsing
    else:
        pred["A"] = "L"  # sequential: low search, low browse

    # ── B: Production ───────────────────────────────────────
    # Key insight: files_created and total_output_chars separate L from M/R
    # M vs R: use avg_output + files_created combined
    avg_out = mean_of(fl, "output_detail", "avg_output_length")
    files_cr = mean_of(fl, "output_detail", "files_created")
    total_out = mean_of(fl, "output_detail", "total_output_chars")

    if total_out >= 30000 or (files_cr >= 4.0 and avg_out >= 7000):
        pred["B"] = "L"  # comprehensive: lots of output + many files
    elif files_cr <= 2.5 and total_out <= 15000:
        pred["B"] = "R"  # minimal: few files, limited output
    else:
        pred["B"] = "M"  # balanced

    # ── C: Organization ─────────────────────────────────────
    # Key insight: max_depth cleanly separates all three tiers
    dirs_cr = mean_of(fl, "directory_style", "dirs_created")
    max_dep = mean_of(fl, "directory_style", "max_dir_depth")
    files_mv = mean_of(fl, "directory_style", "files_moved")

    if max_dep >= 1.0:
        pred["C"] = "L"  # deeply nested
    elif max_dep >= 0.45:
        pred["C"] = "M"  # adaptive
    else:
        pred["C"] = "R"  # flat

    # ── D: Iteration ────────────────────────────────────────
    # Key insight: R is separable (edits < 1.0), L vs M need secondary features
    total_edits = mean_of(fl, "edit_strategy", "total_edits")
    avg_lines = mean_of(fl, "edit_strategy", "avg_lines_changed")
    small_r = mean_of(fl, "edit_strategy", "small_edit_ratio")

    if total_edits <= 1.0:
        pred["D"] = "R"  # rewrite: barely edits
    elif avg_lines >= 4.0 or total_edits >= 2.0:
        pred["D"] = "L"  # incremental: more edits, larger scope
    else:
        pred["D"] = "M"  # balanced

    # ── E: Curation ──────────────────────────────────────────
    # Measures workspace artifact management via version_strategy features.
    # Pattern: L=selective (active cleanup), M=pragmatic (moderate), R=preservative (keeps everything)
    total_del = mean_of(fl, "version_strategy", "total_deletes")
    del_ratio = mean_of(fl, "version_strategy", "delete_to_create_ratio")

    # Combined score: weighted sum (total_deletes is stronger signal)
    e_score = total_del * 0.7 + del_ratio * 0.3

    if e_score >= 0.15:
        pred["E"] = "L"  # selective
    elif e_score <= 0.05:
        pred["E"] = "R"  # preservative
    else:
        pred["E"] = "M"  # pragmatic

    # ── F: Cross-Modal ──────────────────────────────────────
    # Key insight: structured_files_created separates L from M/R
    structured = mean_of(fl, "cross_modal_behavior", "structured_files_created")
    images = mean_of(fl, "cross_modal_behavior", "image_files_created")
    table_rows = mean_of(fl, "cross_modal_behavior", "markdown_table_rows")
    has_tables = mean_of(fl, "cross_modal_behavior", "has_tables")

    if structured >= 0.20 or images >= 0.1:
        pred["F"] = "L"  # visual-heavy: creates structured/image files
    elif structured >= 0.05 or has_tables > 0 or table_rows >= 0.5:
        pred["F"] = "M"  # balanced: some tables/structure
    else:
        pred["F"] = "R"  # text-only

    return pred


def main():
    profiles = sorted(GT.keys())

    # Print E diagnostic (version_strategy features)
    print()
    print("=" * 100)
    print("DIMENSION E: Version Strategy Features (replaces broken phase features)")
    print("=" * 100)
    print(f"{'Profile':<28} {'GT':>2}  {'tot_del':>8} {'del_ratio':>10} {'e_score':>8}")
    print("-" * 65)
    e_rows = []
    for pid in profiles:
        fl = load_features(pid)
        if not fl:
            continue
        gt_e = GT[pid]["E"]
        td = mean_of(fl, "version_strategy", "total_deletes")
        dr = mean_of(fl, "version_strategy", "delete_to_create_ratio")
        score = td * 0.7 + dr * 0.3
        e_rows.append((gt_e, pid, td, dr, score))
    for gt_e, pid, td, dr, score in sorted(e_rows, key=lambda x: x[0]):
        print(f"{pid:<28} {gt_e:>2}  {td:>8.3f} {dr:>10.3f} {score:>8.3f}")

    # Separability
    print()
    for tier in ["L", "M", "R"]:
        vals = [s for g, _, _, _, s in e_rows if g == tier]
        if vals:
            print(f"  E:{tier} score: mean={statistics.mean(vals):.3f} range=[{min(vals):.3f}, {max(vals):.3f}]")

    print()
    print(f"{'Profile':<28} {'GT':>6}  {'Pred':>6}  {'Match':>5}  Details")
    print("-" * 100)

    per_dim_correct = {d: 0 for d in DIMS}
    per_dim_total = {d: 0 for d in DIMS}
    total_correct = 0
    total = 0

    for pid in profiles:
        fl = load_features(pid)
        if not fl:
            continue
        gt = GT[pid]
        pred = classify_calibrated(fl)

        gt_str = "".join(gt[d] for d in DIMS)
        pred_str = ""
        details = []
        matches = 0
        for d in DIMS:
            per_dim_total[d] += 1
            total += 1
            if pred[d] == gt[d]:
                pred_str += pred[d]
                per_dim_correct[d] += 1
                total_correct += 1
                matches += 1
            else:
                pred_str += f"[{pred[d]}]"
                details.append(f"{d}:gt={gt[d]} pred={pred[d]}")

        err_str = ", ".join(details) if details else "PERFECT"
        print(f"{pid:<28} {gt_str:>6}  {pred_str:>6}  {matches}/6    {err_str}")

    # Summary
    print()
    print("=" * 80)
    print("PER-DIMENSION ACCURACY (calibrated multi-feature rules)")
    print("=" * 80)
    for d in DIMS:
        acc = per_dim_correct[d] / per_dim_total[d] * 100 if per_dim_total[d] else 0
        print(f"  {d}: {per_dim_correct[d]}/{per_dim_total[d]} = {acc:.0f}%")
    overall = total_correct / total * 100 if total else 0
    print(f"\n  Overall: {total_correct}/{total} = {overall:.1f}%")

    # Find best subset — profiles with most correct dimensions
    print()
    print("=" * 80)
    print("BEST SUBSET CANDIDATES (profiles with most correct)")
    print("=" * 80)
    results = []
    results_no_e = []
    for pid in profiles:
        fl = load_features(pid)
        if not fl:
            continue
        gt = GT[pid]
        pred = classify_calibrated(fl)
        matches = sum(1 for d in DIMS if pred[d] == gt[d])
        wrong = [f"{d}:{gt[d]}→{pred[d]}" for d in DIMS if pred[d] != gt[d]]
        results.append((matches, pid, wrong))

        # Without E
        dims5 = [d for d in DIMS if d != "E"]
        matches5 = sum(1 for d in dims5 if pred[d] == gt[d])
        wrong5 = [f"{d}:{gt[d]}→{pred[d]}" for d in dims5 if pred[d] != gt[d]]
        results_no_e.append((matches5, pid, wrong5))

    for matches, pid, wrong in sorted(results, reverse=True):
        wrong_str = " ".join(wrong) if wrong else "ALL CORRECT"
        print(f"  {matches}/6  {pid:<28} {wrong_str}")

    print()
    print("=" * 80)
    print("EXCLUDING E (5 dims: A B C D F)")
    print("=" * 80)
    perfect5 = 0
    for matches, pid, wrong in sorted(results_no_e, reverse=True):
        wrong_str = " ".join(wrong) if wrong else "PERFECT 5/5"
        if matches == 5:
            perfect5 += 1
        print(f"  {matches}/5  {pid:<28} {wrong_str}")

    # 5-dim accuracy
    print()
    dims5 = [d for d in DIMS if d != "E"]
    t5_correct = sum(per_dim_correct[d] for d in dims5)
    t5_total = sum(per_dim_total[d] for d in dims5)
    print(f"  5-dim overall: {t5_correct}/{t5_total} = {t5_correct / t5_total * 100:.1f}%")
    print(f"  Profiles with 5/5 correct: {perfect5}/20")


if __name__ == "__main__":
    main()
