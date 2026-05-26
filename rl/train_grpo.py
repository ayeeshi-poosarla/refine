#!/usr/bin/env python3
"""
GRPO training loop for REFINE — local helpers and policy inspection.

The Modal entry point lives in exhaustive_rollout.py (same app as rollout_one):

    modal run rl/exhaustive_rollout.py::train_grpo --task guo_readmission
    modal run rl/exhaustive_rollout.py::train_grpo --task guo_readmission --steps 5 --greedy
    modal run rl/exhaustive_rollout.py::train_grpo --task lab_hyperkalemia --lr 0.05 --seed 7

This file provides local CLI utilities for inspecting saved policies and logs
without needing Modal.

Usage (local, no GPU):
    python rl/train_grpo.py --show-policy guo_readmission
    python rl/train_grpo.py --show-log    guo_readmission
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rl.policy import ActionPolicy

POLICY_DIR = REPO_ROOT / "data" / "rl" / "policies"
LOG_DIR    = REPO_ROOT / "data" / "rl" / "training_log"


def show_policy(task: str) -> None:
    """Print all saved policy snapshots for a task."""
    task_dir = POLICY_DIR / task
    if not task_dir.exists():
        print(f"No policy snapshots found for {task!r} in {task_dir}")
        return

    snapshots = sorted(task_dir.glob("policy_step_*.json"))
    for snap in snapshots:
        policy = ActionPolicy.load(snap)
        step   = snap.stem.replace("policy_step_", "")
        print(f"\n── Step {step} ({snap.name}) ──────────────────────")
        print(f"  {'ACTION':<35} {'PROB':>6} {'LOGIT':>8}")
        print(f"  {'-'*35} {'-'*6} {'-'*8}")
        probs = policy.probabilities()
        for i, name in enumerate(policy.action_names):
            marker = " ◀" if i == int(np.argmax(probs)) else ""
            print(f"  {name:<35} {probs[i]:>6.3f} {policy.logits[i]:>8.4f}{marker}")


def show_log(task: str) -> None:
    """Print the AUROC progression and chosen actions from the training log."""
    log_path = LOG_DIR / f"{task}.json"
    if not log_path.exists():
        print(f"No training log found at {log_path}")
        return

    log = json.loads(log_path.read_text())
    print(f"\nGRPO training log — task={log['task']}  steps={log['steps']}  lr={log['lr']}")
    print(f"\n  {'STEP':>4}  {'ACTION SELECTED':<35}  {'STATE AUROC':>11}")
    print(f"  {'-'*4}  {'-'*35}  {'-'*11}")
    for entry in log["history"]:
        s   = entry["step"]
        act = entry["action_selected"] or "— (init)"
        auc = entry["state_auroc"]
        print(f"  {s:>4}  {act:<35}  {auc:>11.4f}")

    print(f"\n  Reward table per step:")
    for entry in log["history"]:
        if not entry["rewards"]:
            continue
        print(f"\n  Step {entry['step']} rewards:")
        for aname, r in sorted(entry["rewards"].items(), key=lambda x: -x[1]):
            adv = entry["advantages"].get(aname, 0.0)
            print(f"    {aname:<35}  auroc={r:.4f}  adv={adv:+.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--show-policy", metavar="TASK", help="Print saved policy snapshots")
    p.add_argument("--show-log",    metavar="TASK", help="Print training log")
    args = p.parse_args()

    if args.show_policy:
        show_policy(args.show_policy)
    elif args.show_log:
        show_log(args.show_log)
    else:
        p.print_help()
        print("\nTo run training:  modal run rl/exhaustive_rollout.py::train_grpo --task TASK")


if __name__ == "__main__":
    main()
