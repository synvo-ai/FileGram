#!/usr/bin/env python3
"""Diagnostic: print actual feature values per profile for all 6 L/M/R dimensions.

Shows exact numbers vs thresholds so we can see where traces differentiate and where they don't.
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


def mean_of(features_list, attr, key, default=0):
    vals = [f.get(attr, {}).get(key, default) for f in features_list]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else default


def main():
    profiles = sorted(GT.keys())

    # Build feature table
    rows = []
    for pid in profiles:
        fl = load_features(pid)
        if not fl:
            continue
        gt = GT[pid]
        n = len(fl)

        row = {
            "profile": pid,
            "n": n,
            "gt": "".join(gt[d] for d in DIMS),
            # A: Consumption
            "A_gt": gt["A"],
            "search_ratio": mean_of(fl, "reading_strategy", "search_ratio"),
            "browse_ratio": mean_of(fl, "reading_strategy", "browse_ratio"),
            "revisit_ratio": mean_of(fl, "reading_strategy", "revisit_ratio"),
            "total_reads": mean_of(fl, "reading_strategy", "total_reads"),
            "total_searches": mean_of(fl, "reading_strategy", "total_searches"),
            "total_browses": mean_of(fl, "reading_strategy", "total_browses"),
            # B: Production
            "B_gt": gt["B"],
            "avg_output_len": mean_of(fl, "output_detail", "avg_output_length"),
            "files_created": mean_of(fl, "output_detail", "files_created"),
            "total_output": mean_of(fl, "output_detail", "total_output_chars"),
            # C: Organization
            "C_gt": gt["C"],
            "dirs_created": mean_of(fl, "directory_style", "dirs_created"),
            "max_depth": mean_of(fl, "directory_style", "max_dir_depth"),
            "files_moved": mean_of(fl, "directory_style", "files_moved"),
            # D: Iteration
            "D_gt": gt["D"],
            "total_edits": mean_of(fl, "edit_strategy", "total_edits"),
            "avg_lines": mean_of(fl, "edit_strategy", "avg_lines_changed"),
            "small_edit_r": mean_of(fl, "edit_strategy", "small_edit_ratio"),
            # E: Curation
            "E_gt": gt["E"],
            "total_deletes": mean_of(fl, "version_strategy", "total_deletes"),
            "del_ratio": mean_of(fl, "version_strategy", "delete_to_create_ratio"),
            # F: Cross-Modal
            "F_gt": gt["F"],
            "images": mean_of(fl, "cross_modal_behavior", "image_files_created"),
            "structured": mean_of(fl, "cross_modal_behavior", "structured_files_created"),
            "table_rows": mean_of(fl, "cross_modal_behavior", "markdown_table_rows"),
            "has_tables": mean_of(fl, "cross_modal_behavior", "has_tables"),
        }
        rows.append(row)

    # Print per-dimension tables
    print("=" * 120)
    print("DIMENSION A: Consumption (L=sequential, M=targeted, R=breadth-first)")
    print("Thresholds: L: search≈0, revisit≥0.10 | M: search_ratio≥0.15 | R: browse≥0.15, revisit<0.05")
    print("=" * 120)
    print(
        f"{'Profile':<28} {'GT':>2}  {'search_r':>9} {'browse_r':>9} {'revisit_r':>9} {'searches':>9} {'browses':>9} {'reads':>9}"
    )
    print("-" * 110)
    for r in sorted(rows, key=lambda x: x["A_gt"]):
        print(
            f"{r['profile']:<28} {r['A_gt']:>2}  {r['search_ratio']:>9.3f} {r['browse_ratio']:>9.3f} "
            f"{r['revisit_ratio']:>9.3f} {r['total_searches']:>9.1f} {r['total_browses']:>9.1f} {r['total_reads']:>9.1f}"
        )

    print()
    print("=" * 120)
    print("DIMENSION B: Production (L=comprehensive, M=balanced, R=minimal)")
    print("Thresholds: L: avg≥3000, files≥3 | M: avg∈[1000,3000] | R: avg≤1000, files=1")
    print("=" * 120)
    print(f"{'Profile':<28} {'GT':>2}  {'avg_out':>9} {'files_cr':>9} {'total_out':>10}")
    print("-" * 70)
    for r in sorted(rows, key=lambda x: x["B_gt"]):
        print(
            f"{r['profile']:<28} {r['B_gt']:>2}  {r['avg_output_len']:>9.0f} {r['files_created']:>9.1f} {r['total_output']:>10.0f}"
        )

    print()
    print("=" * 120)
    print("DIMENSION C: Organization (L=deeply_nested, M=adaptive, R=flat)")
    print("Thresholds: L: dirs≥3, depth≥3 | M: dirs∈[1,2] | R: dirs=0")
    print("=" * 120)
    print(f"{'Profile':<28} {'GT':>2}  {'dirs_cr':>9} {'max_dep':>9} {'files_mv':>9}")
    print("-" * 70)
    for r in sorted(rows, key=lambda x: x["C_gt"]):
        print(
            f"{r['profile']:<28} {r['C_gt']:>2}  {r['dirs_created']:>9.1f} {r['max_depth']:>9.1f} {r['files_moved']:>9.1f}"
        )

    print()
    print("=" * 120)
    print("DIMENSION D: Iteration (L=incremental, M=balanced, R=rewrite)")
    print("Thresholds: L: edits≥5, avg_lines≤10 | M: edits∈[2,4] | R: edits≤1")
    print("=" * 120)
    print(f"{'Profile':<28} {'GT':>2}  {'tot_edits':>10} {'avg_lines':>10} {'small_r':>9}")
    print("-" * 70)
    for r in sorted(rows, key=lambda x: x["D_gt"]):
        print(
            f"{r['profile']:<28} {r['D_gt']:>2}  {r['total_edits']:>10.1f} {r['avg_lines']:>10.1f} {r['small_edit_r']:>9.3f}"
        )

    print()
    print("=" * 120)
    print("DIMENSION E: Curation (L=selective, M=pragmatic, R=preservative)")
    print("Thresholds: L: e_score≥0.15 | R: e_score≤0.05 | M: in between")
    print("=" * 120)
    print(f"{'Profile':<28} {'GT':>2}  {'tot_del':>8} {'del_ratio':>10} {'e_score':>8}")
    print("-" * 65)
    for r in sorted(rows, key=lambda x: x["E_gt"]):
        e_score = r["total_deletes"] * 0.7 + r["del_ratio"] * 0.3
        print(f"{r['profile']:<28} {r['E_gt']:>2}  {r['total_deletes']:>8.3f} {r['del_ratio']:>10.3f} {e_score:>8.3f}")

    print()
    print("=" * 120)
    print("DIMENSION F: Cross-Modal (L=visual-heavy, M=balanced, R=text-only)")
    print("Thresholds: L: images≥1 or structured≥1 | M: has_tables>0 | R: all zero")
    print("=" * 120)
    print(f"{'Profile':<28} {'GT':>2}  {'images':>9} {'structur':>9} {'tbl_rows':>9} {'has_tbl':>9}")
    print("-" * 80)
    for r in sorted(rows, key=lambda x: x["F_gt"]):
        print(
            f"{r['profile']:<28} {r['F_gt']:>2}  {r['images']:>9.2f} {r['structured']:>9.2f} "
            f"{r['table_rows']:>9.1f} {r['has_tables']:>9.2f}"
        )

    # Summary: per-dimension separability
    print()
    print("=" * 120)
    print("SEPARABILITY ANALYSIS")
    print("=" * 120)
    for dim, key_metric, groups in [
        ("A", "search_ratio", {"L": [], "M": [], "R": []}),
        ("B", "avg_output_len", {"L": [], "M": [], "R": []}),
        ("C", "dirs_created", {"L": [], "M": [], "R": []}),
        ("D", "total_edits", {"L": [], "M": [], "R": []}),
        ("E", "switch_rate", {"L": [], "M": [], "R": []}),
        ("F", "has_tables", {"L": [], "M": [], "R": []}),
    ]:
        for r in rows:
            groups[r[f"{dim}_gt"]].append(r[key_metric])

        print(f"\n  Dim {dim} ({key_metric}):")
        for tier in ["L", "M", "R"]:
            vals = groups[tier]
            if vals:
                mu = statistics.mean(vals)
                mn, mx = min(vals), max(vals)
                print(f"    {tier}: n={len(vals):>2}  mean={mu:>8.2f}  range=[{mn:.2f}, {mx:.2f}]")
            else:
                print(f"    {tier}: n= 0")

        # Check overlap
        all_tiers = [(tier, v) for tier in ["L", "M", "R"] for v in groups[tier]]
        if all(groups[t] for t in ["L", "M", "R"]):
            l_max, m_min, m_max, r_min = max(groups["L"]), min(groups["M"]), max(groups["M"]), min(groups["R"])
            overlap = "OVERLAPPING" if l_max >= m_min or m_max >= r_min else "SEPARABLE"
            print(f"    → {overlap} (L_max={l_max:.2f} vs M_min={m_min:.2f}, M_max={m_max:.2f} vs R_min={r_min:.2f})")


if __name__ == "__main__":
    main()
