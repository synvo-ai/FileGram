#!/usr/bin/env python3
"""Batch experiment runner for FileGram.

Runs task x profile combinations in parallel using asyncio.

Usage:
    python experiments/run.py                    # Run all experiments
    python experiments/run.py --dry-run          # Show what would run
    python experiments/run.py --profile alex     # Only run with alex profile
    python experiments/run.py --task task1       # Only run task1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Task:
    id: str
    name: str
    prompt: str


@dataclass
class Run:
    id: str
    profile: str
    task: Task
    work_dir: Path
    output_dir: Path


def load_config(config_path: Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def build_runs(
    config: dict,
    base_dir: Path,
    profile_filter: str | None,
    task_filter: str | None,
) -> list[Run]:
    timestamp = int(time.time() * 1000)

    profiles = [profile_filter] if profile_filter else config["profiles"]
    tasks_raw = config["tasks"]
    if task_filter:
        tasks_raw = [t for t in tasks_raw if t["id"] == task_filter]

    tasks = [Task(id=t["id"], name=t["name"], prompt=t["prompt"]) for t in tasks_raw]

    output_base = Path(config["behavior"]["output_dir"])
    if not output_base.is_absolute():
        output_base = base_dir / output_base

    runs: list[Run] = []
    for profile in profiles:
        for task in tasks:
            run_id = f"{task.id}_{profile}_{timestamp}"
            work_dir = base_dir / "experiments" / "workspaces" / f"{task.id}_{profile}"
            output_dir = output_base / run_id
            runs.append(
                Run(
                    id=run_id,
                    profile=profile,
                    task=task,
                    work_dir=work_dir,
                    output_dir=output_dir,
                )
            )

    return runs


async def setup(run: Run) -> None:
    """Set up workspace directory for a run."""
    run.work_dir.mkdir(parents=True, exist_ok=True)
    run.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize git repo for sandbox detection
    proc = await asyncio.create_subprocess_exec(
        "git",
        "init",
        cwd=str(run.work_dir),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()

    # Create README in workspace
    readme = run.work_dir / "README.md"
    readme.write_text(
        f"# Experiment Workspace\n\n"
        f"- **Task**: {run.task.name}\n"
        f"- **Profile**: {run.profile}\n"
        f"- **ID**: {run.id}\n\n"
        f"## Task Prompt\n{run.task.prompt}\n",
        encoding="utf-8",
    )


async def execute(run: Run, timeout_s: int) -> dict:
    """Execute a single experiment run."""
    start = time.time()
    print(f"[{run.id}] Starting: {run.task.name} with profile {run.profile}")

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "filegram.main",
            "-d",
            str(run.work_dir),
            "-p",
            run.profile,
            "-1",
            run.task.prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "SYNVOCOWORK_BEHAVIOR_OUTPUT": str(run.output_dir),
            },
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            duration = time.time() - start
            print(f"[{run.id}] Timed out after {duration:.1f}s")
            return {"success": False, "error": "timeout"}

        duration = time.time() - start
        exit_code = proc.returncode

        # Save outputs
        (run.output_dir / "stdout.txt").write_bytes(stdout)
        (run.output_dir / "stderr.txt").write_bytes(stderr)

        meta = {
            "id": run.id,
            "profile": run.profile,
            "task": {
                "id": run.task.id,
                "name": run.task.name,
                "prompt": run.task.prompt,
            },
            "exit_code": exit_code,
            "duration_s": round(duration, 2),
            "start_time": start,
            "end_time": time.time(),
        }
        (run.output_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"[{run.id}] Completed in {duration:.1f}s (exit: {exit_code})")
        return {"success": exit_code == 0}

    except Exception as e:
        error = str(e)
        print(f"[{run.id}] Failed: {error}")
        return {"success": False, "error": error}


async def main() -> int:
    parser = argparse.ArgumentParser(description="FileGram batch experiment runner")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--profile", type=str, help="Only run this profile")
    parser.add_argument("--task", type=str, help="Only run this task")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent

    config_path = Path(args.config) if args.config else script_dir / "config.json"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return 1

    config = load_config(config_path)
    all_runs = build_runs(config, base_dir, args.profile, args.task)

    print("=" * 60)
    print("FileGram Batch Experiment Runner")
    print("=" * 60)
    print(f"\nTotal experiments: {len(all_runs)}")
    print(f"Profiles: {', '.join(sorted(set(r.profile for r in all_runs)))}")
    print(f"Tasks: {', '.join(sorted(set(r.task.id for r in all_runs)))}")
    print(f"Output: {config['behavior']['output_dir']}")
    print()

    if args.dry_run:
        print("DRY RUN - Would execute:\n")
        for run in all_runs:
            print(f'  - {run.task.name} ({run.task.id}) with profile "{run.profile}"')
            print(f"    Workspace: {run.work_dir}")
            print(f"    Output: {run.output_dir}")
            print()
        return 0

    print("Setting up workspaces...")
    await asyncio.gather(*[setup(r) for r in all_runs])
    print("Workspaces ready.\n")

    timeout_s = config.get("options", {}).get("timeout_s", 300)

    print("Starting experiments in parallel...\n")
    results = await asyncio.gather(*[execute(r, timeout_s) for r in all_runs])

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    ok = sum(1 for r in results if r["success"])
    fail = len(results) - ok

    print(f"\nSuccessful: {ok}/{len(all_runs)}")
    print(f"Failed: {fail}/{len(all_runs)}")

    if fail > 0:
        print("\nFailed experiments:")
        for run, result in zip(all_runs, results):
            if not result["success"]:
                print(f"  - {run.id}: {result.get('error', 'Unknown error')}")

    print(f"\nResults saved to: {config['behavior']['output_dir']}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
