#!/usr/bin/env python3
"""L/M/R Cross-Validation: Extract behavioral stats per profile, classify blind, compare to ground truth.

Loads filegramos ingest caches, extracts per-profile aggregate stats
across all trajectories, and outputs a CSV + per-dimension confusion analysis.
"""

import json
import statistics
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_ROOT = _BENCH.parent
sys.path.insert(0, str(_BENCH))

CACHE_DIR = _BENCH / "ingest_cache_gemini_2.5_flash"
MATRIX_PATH = _ROOT / "profiles" / "profile_matrix.json"
SIGNAL_DIR = _ROOT / "signal"

# Load ground truth
with open(MATRIX_PATH) as f:
    matrix = json.load(f)
GT = {pid: info["dimensions"] for pid, info in matrix["profiles"].items()}
DIM_LABELS = matrix["dimensions"]


def load_trajectory_features(profile_id: str) -> list[dict]:
    """Load per-trajectory procedural features from events.json via FeatureExtractor."""
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
        features = fe.extract_all()
        features["_trajectory"] = td.name
        features_list.append(features)
    return features_list


def aggregate_stats(features_list: list[dict]) -> dict:
    """Compute per-attribute aggregate statistics across trajectories."""
    if not features_list:
        return {}

    # Collect all numeric values per attribute.key
    collectors: dict[str, list[float]] = {}
    for feat in features_list:
        for attr_name, attr_dict in feat.items():
            if attr_name.startswith("_") or not isinstance(attr_dict, dict):
                continue
            for key, val in attr_dict.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    full_key = f"{attr_name}.{key}"
                    collectors.setdefault(full_key, []).append(float(val))
                elif isinstance(val, bool):
                    full_key = f"{attr_name}.{key}"
                    collectors.setdefault(full_key, []).append(1.0 if val else 0.0)

    agg = {}
    for key, vals in sorted(collectors.items()):
        if not vals:
            continue
        mu = statistics.mean(vals)
        agg[key] = {
            "mean": round(mu, 3),
            "std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0,
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }
    return agg


def classify_from_stats(agg: dict) -> dict[str, str]:
    """Rule-based L/M/R classification from aggregate stats."""
    pred = {}

    # --- A: Consumption Pattern ---
    search_r = agg.get("reading_strategy.search_ratio", {}).get("mean", 0)
    browse_r = agg.get("reading_strategy.browse_ratio", {}).get("mean", 0)
    revisit_r = agg.get("reading_strategy.revisit_ratio", {}).get("mean", 0)
    total_reads = agg.get("reading_strategy.total_reads", {}).get("mean", 0)

    if search_r >= 0.15:
        pred["A"] = "M"  # targeted search
    elif browse_r >= 0.15 or (revisit_r < 0.05 and total_reads < 15):
        pred["A"] = "R"  # breadth-first / skim
    else:
        pred["A"] = "L"  # sequential deep

    # --- B: Production Style ---
    avg_output = agg.get("output_detail.avg_output_length", {}).get("mean", 0)
    files_created = agg.get("output_detail.files_created", {}).get("mean", 0)
    total_chars = agg.get("output_detail.total_output_chars", {}).get("mean", 0)

    if total_chars >= 8000 or (avg_output >= 3000 and files_created >= 3):
        pred["B"] = "L"  # comprehensive
    elif total_chars <= 3000 or (avg_output <= 1500 and files_created <= 1.5):
        pred["B"] = "R"  # minimal
    else:
        pred["B"] = "M"  # balanced

    # --- C: Organization Preference ---
    dirs_created = agg.get("directory_style.dirs_created", {}).get("mean", 0)
    max_depth = agg.get("directory_style.max_dir_depth", {}).get("mean", 0)
    files_moved = agg.get("directory_style.files_moved", {}).get("mean", 0)

    if dirs_created >= 2 and max_depth >= 2:
        pred["C"] = "L"  # deeply nested
    elif dirs_created <= 0.5 and max_depth <= 0.5:
        pred["C"] = "R"  # flat
    else:
        pred["C"] = "M"  # adaptive

    # --- D: Iteration Strategy ---
    total_edits = agg.get("edit_strategy.total_edits", {}).get("mean", 0)
    avg_lines = agg.get("edit_strategy.avg_lines_changed", {}).get("mean", 0)
    small_edit_r = agg.get("edit_strategy.small_edit_ratio", {}).get("mean", 0)

    if total_edits >= 3 and (small_edit_r >= 0.5 or avg_lines <= 15):
        pred["D"] = "L"  # incremental
    elif total_edits <= 1 or avg_lines >= 30:
        pred["D"] = "R"  # rewrite
    else:
        pred["D"] = "M"  # balanced

    # --- E: Curation ---
    total_del = agg.get("version_strategy.total_deletes", {}).get("mean", 0)
    del_ratio = agg.get("version_strategy.delete_to_create_ratio", {}).get("mean", 0)
    e_score = total_del * 0.7 + del_ratio * 0.3

    if e_score >= 0.15:
        pred["E"] = "L"  # selective (active cleanup)
    elif e_score <= 0.05:
        pred["E"] = "R"  # preservative (keeps everything)
    else:
        pred["E"] = "M"  # pragmatic (moderate)

    # --- F: Cross-Modal Behavior ---
    has_tables = agg.get("cross_modal_behavior.has_tables", {}).get("mean", 0)
    images = agg.get("cross_modal_behavior.image_files_created", {}).get("mean", 0)
    structured = agg.get("cross_modal_behavior.structured_files_created", {}).get("mean", 0)
    table_rows = agg.get("cross_modal_behavior.markdown_table_rows", {}).get("mean", 0)

    visual_score = images + structured + (1 if table_rows > 5 else 0)
    if visual_score >= 1.5 or images >= 0.5:
        pred["F"] = "L"  # visual-heavy
    elif has_tables < 0.2 and images < 0.1 and table_rows <= 1:
        pred["F"] = "R"  # text-only
    else:
        pred["F"] = "M"  # balanced

    return pred


def main():
    profiles = sorted(GT.keys())
    dims = ["A", "B", "C", "D", "E", "F"]

    # Per-dimension confusion tracking
    confusion = {d: {"correct": 0, "total": 0, "errors": []} for d in dims}

    all_results = []

    print(f"{'Profile':<28} {'Dim':>3}  {'GT':>5} {'Pred':>5} {'Match':>5}")
    print("-" * 75)

    for pid in profiles:
        gt = GT[pid]
        features_list = load_trajectory_features(pid)
        n_traj = len(features_list)
        agg = aggregate_stats(features_list)
        pred = classify_from_stats(agg)

        profile_results = {"profile": pid, "n_trajectories": n_traj, "dims": {}}

        for d in dims:
            g = gt[d]
            p = pred.get(d, "?")
            match = "✓" if g == p else "✗"
            confusion[d]["total"] += 1
            if g == p:
                confusion[d]["correct"] += 1
            else:
                confusion[d]["errors"].append(
                    {
                        "profile": pid,
                        "gt": g,
                        "pred": p,
                    }
                )

            profile_results["dims"][d] = {"gt": g, "pred": p, "match": g == p}

        # Print per-profile
        for d in dims:
            r = profile_results["dims"][d]
            mark = "✓" if r["match"] else "✗"
            if d == "A":
                print(f"{pid:<28}  {d:>2}   {r['gt']:>3}   {r['pred']:>3}   {mark}")
            else:
                print(f"{'':28}  {d:>2}   {r['gt']:>3}   {r['pred']:>3}   {mark}")

        # Also print key stats for debugging
        rs = agg.get("reading_strategy.search_ratio", {}).get("mean", 0)
        br = agg.get("reading_strategy.browse_ratio", {}).get("mean", 0)
        rv = agg.get("reading_strategy.revisit_ratio", {}).get("mean", 0)
        tr = agg.get("reading_strategy.total_reads", {}).get("mean", 0)
        oc = agg.get("output_detail.total_output_chars", {}).get("mean", 0)
        fc = agg.get("output_detail.files_created", {}).get("mean", 0)
        dc = agg.get("directory_style.dirs_created", {}).get("mean", 0)
        md = agg.get("directory_style.max_dir_depth", {}).get("mean", 0)
        te = agg.get("edit_strategy.total_edits", {}).get("mean", 0)
        al = agg.get("edit_strategy.avg_lines_changed", {}).get("mean", 0)
        sr = agg.get("edit_strategy.small_edit_ratio", {}).get("mean", 0)
        cs = agg.get("reading_strategy.context_switch_rate", {}).get("mean", 0)
        tb = agg.get("cross_modal_behavior.has_tables", {}).get("mean", 0)
        im = agg.get("cross_modal_behavior.image_files_created", {}).get("mean", 0)
        mtr = agg.get("cross_modal_behavior.markdown_table_rows", {}).get("mean", 0)
        print(
            f"  Stats: reads={tr:.1f} search={rs:.2f} browse={br:.2f} revisit={rv:.2f} "
            f"out={oc:.0f} files={fc:.1f} dirs={dc:.1f} depth={md:.1f} "
            f"edits={te:.1f} lines={al:.0f} small_r={sr:.2f} switch={cs:.2f} "
            f"tables={tb:.2f} images={im:.1f} tbl_rows={mtr:.1f}"
        )
        print()

        all_results.append(profile_results)

    # Summary
    print("=" * 75)
    print("PER-DIMENSION ACCURACY")
    print("=" * 75)
    for d in dims:
        c = confusion[d]
        acc = c["correct"] / c["total"] * 100 if c["total"] > 0 else 0
        label = DIM_LABELS.get(
            f"{'ABCDEF'[dims.index(d)]}_{'consumption production organization iteration rhythm crossmodal'.split()[dims.index(d)]}",
            {},
        )
        print(f"  {d}: {c['correct']}/{c['total']} = {acc:.0f}%")
        if c["errors"]:
            for err in c["errors"]:
                print(f"     ✗ {err['profile']}: GT={err['gt']} Pred={err['pred']}")

    total_correct = sum(c["correct"] for c in confusion.values())
    total = sum(c["total"] for c in confusion.values())
    print(f"\n  Overall: {total_correct}/{total} = {total_correct / total * 100:.1f}%")

    # Save detailed results
    out_path = _BENCH / "lmr_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "results": all_results,
                "per_dimension": {
                    d: {
                        "accuracy": confusion[d]["correct"] / confusion[d]["total"] if confusion[d]["total"] else 0,
                        "correct": confusion[d]["correct"],
                        "total": confusion[d]["total"],
                        "errors": confusion[d]["errors"],
                    }
                    for d in dims
                },
                "overall_accuracy": total_correct / total if total else 0,
            },
            f,
            indent=2,
        )
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
