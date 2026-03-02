#!/usr/bin/env python3
"""Search for features that could replace E dimension.

Requirements: varies across profiles, orthogonal to A/B/C/D/F.
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

# Features already used by A/B/C/D/F
USED_FEATURES = {
    # A
    "reading_strategy.search_ratio",
    "reading_strategy.browse_ratio",
    "reading_strategy.revisit_ratio",
    "reading_strategy.total_searches",
    "reading_strategy.total_browses",
    "reading_strategy.total_reads",
    # B
    "output_detail.avg_output_length",
    "output_detail.files_created",
    "output_detail.total_output_chars",
    # C
    "directory_style.dirs_created",
    "directory_style.max_dir_depth",
    "directory_style.files_moved",
    # D
    "edit_strategy.total_edits",
    "edit_strategy.avg_lines_changed",
    "edit_strategy.small_edit_ratio",
    # F
    "cross_modal_behavior.image_files_created",
    "cross_modal_behavior.structured_files_created",
    "cross_modal_behavior.markdown_table_rows",
    "cross_modal_behavior.has_tables",
}


def load_all_features(pid):
    from filegramos.feature_extraction import FeatureExtractor

    traj_dirs = sorted(SIGNAL_DIR.glob(f"{pid}_T-*"))
    fl = []
    for td in traj_dirs:
        ep = td / "events.json"
        if not ep.exists():
            continue
        with open(ep) as f:
            events = json.load(f)
        fe = FeatureExtractor(events)
        fl.append(fe.extract_all())
    return fl


def main():
    profiles = sorted(GT.keys())

    # Collect ALL features per profile
    profile_features = {}
    for pid in profiles:
        fl = load_all_features(pid)
        if not fl:
            continue
        # Aggregate: mean of each numeric feature
        agg = {}
        for feat in fl:
            for attr_name, attr_dict in feat.items():
                if not isinstance(attr_dict, dict):
                    continue
                for key, val in attr_dict.items():
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        full_key = f"{attr_name}.{key}"
                        agg.setdefault(full_key, []).append(float(val))
        profile_features[pid] = {k: statistics.mean(v) for k, v in agg.items()}

    # Get all feature keys
    all_keys = set()
    for pf in profile_features.values():
        all_keys.update(pf.keys())

    # Filter to unused features
    unused = sorted(all_keys - USED_FEATURES)

    print(f"Total features: {len(all_keys)}")
    print(f"Used by A-F: {len(USED_FEATURES)}")
    print(f"Unused: {len(unused)}")
    print()

    # For each unused feature, check:
    # 1. Coefficient of variation across profiles (higher = more variation)
    # 2. Separation by E ground truth (L/M/R)
    # 3. Correlation with A/B/C/D/F GT labels (low = orthogonal)

    results = []
    for fkey in unused:
        vals = {pid: profile_features[pid].get(fkey, 0) for pid in profiles}
        all_vals = list(vals.values())

        if max(all_vals) == min(all_vals) == 0:
            continue  # skip all-zero
        if max(all_vals) == min(all_vals):
            continue  # skip constant

        cv = statistics.stdev(all_vals) / max(abs(statistics.mean(all_vals)), 0.001)

        # Group by E tier
        e_groups = {"L": [], "M": [], "R": []}
        for pid in profiles:
            e_groups[GT[pid]["E"]].append(vals[pid])

        e_means = {t: statistics.mean(v) if v else 0 for t, v in e_groups.items()}
        e_ranges = {t: (min(v), max(v)) if v else (0, 0) for t, v in e_groups.items()}

        # E separation: range of means / overall std
        e_mean_spread = max(e_means.values()) - min(e_means.values())
        e_sep = e_mean_spread / max(statistics.stdev(all_vals), 0.001)

        # Check correlation with other dims (want LOW correlation)
        max_other_sep = 0
        max_other_dim = ""
        for dim in ["A", "B", "C", "D", "F"]:
            g = {"L": [], "M": [], "R": []}
            for pid in profiles:
                g[GT[pid][dim]].append(vals[pid])
            means = [statistics.mean(v) if v else 0 for v in g.values()]
            spread = max(means) - min(means)
            sep = spread / max(statistics.stdev(all_vals), 0.001)
            if sep > max_other_sep:
                max_other_sep = sep
                max_other_dim = dim

        # Orthogonality score: E_sep / max_other_sep (higher = more orthogonal to others)
        ortho = e_sep / max(max_other_sep, 0.001) if e_sep > 0 else 0

        results.append(
            {
                "feature": fkey,
                "cv": cv,
                "e_sep": e_sep,
                "e_means": e_means,
                "e_ranges": e_ranges,
                "max_other_dim": max_other_dim,
                "max_other_sep": max_other_sep,
                "ortho": ortho,
            }
        )

    # Sort by E separation score
    results.sort(key=lambda x: x["e_sep"], reverse=True)

    print("=" * 130)
    print("TOP FEATURES BY E-TIER SEPARATION (higher e_sep = better)")
    print("=" * 130)
    print(f"{'Feature':<45} {'CV':>5} {'E_sep':>6} {'E:L':>8} {'E:M':>8} {'E:R':>8} {'Best_other':>10} {'Ortho':>6}")
    print("-" * 130)
    for r in results[:30]:
        em = r["e_means"]
        print(
            f"{r['feature']:<45} {r['cv']:>5.2f} {r['e_sep']:>6.2f} "
            f"{em['L']:>8.2f} {em['M']:>8.2f} {em['R']:>8.2f} "
            f"{r['max_other_dim']}={r['max_other_sep']:>5.2f}  {r['ortho']:>6.2f}"
        )

    # Show detailed distribution for top 5 E-separating features
    print()
    print("=" * 130)
    print("DETAILED: Top 5 E-separating features — per-profile values")
    print("=" * 130)
    for r in results[:5]:
        fkey = r["feature"]
        em = r["e_means"]
        print(f"\n  {fkey}  (E_sep={r['e_sep']:.2f}, Ortho={r['ortho']:.2f})")
        print(f"    E:L mean={em['L']:.3f}  E:M mean={em['M']:.3f}  E:R mean={em['R']:.3f}")
        for tier in ["L", "M", "R"]:
            mn, mx = r["e_ranges"][tier]
            print(f"    E:{tier} range=[{mn:.3f}, {mx:.3f}]")

        # Per-profile
        for tier in ["L", "M", "R"]:
            pids = [pid for pid in profiles if GT[pid]["E"] == tier]
            vals = [(pid, profile_features[pid].get(fkey, 0)) for pid in pids]
            vals.sort(key=lambda x: x[1])
            val_str = ", ".join(f"{p[:8]}={v:.2f}" for p, v in vals)
            print(f"    E:{tier}: {val_str}")


if __name__ == "__main__":
    main()
