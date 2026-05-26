#!/usr/bin/env python3
"""
GRPO training loop for REFINE rubric policy optimisation.

Imports `app` and `rollout_one` from exhaustive_rollout.py so that both live
in the same Modal app ("refine-exhaustive-rollout") — this is what prevents the
"Function has not been hydrated" error that occurs when calling a Modal function
from a different app.

Two branching modes
-------------------
chain (default)
    Each step applies the policy-selected action to the current state and
    branches from the result.  State always advances regardless of reward.

best-state  (--best-state-rollout)
    Tracks the highest-AUROC rubric seen so far.  Every exhaustive rollout
    branches from that best state.  The best state only advances when a new
    action beats the current best AUROC; otherwise the state stays frozen and
    the policy still gets a GRPO update from the new rewards.

Run (Modal)
-----------
    modal run rl/train_grpo.py --task guo_readmission
    modal run rl/train_grpo.py --task guo_readmission --steps 5 --greedy
    modal run rl/train_grpo.py --tasks all --steps 3 --best-state-rollout
    modal run rl/train_grpo.py --task guo_readmission --n-trajectories 3 --lr 0.1 --beta 0.05

Inspect saved artefacts (local, no GPU)
-----------------------------------------
    python rl/train_grpo.py --show-policy guo_readmission
    python rl/train_grpo.py --show-log    guo_readmission
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Import Modal app + rollout_one from exhaustive_rollout so they share the
#    same "refine-exhaustive-rollout" app.  rollout_one.starmap() then works
#    without a hydration error.
from rl.exhaustive_rollout import app, rollout_one, REPO_ROOT, TASKS

sys.path.insert(0, str(REPO_ROOT))

from rl.actions import ALL_ACTIONS
from rl.policy import ActionPolicy
from rl.state import RubricState

# ── Directory layout ──────────────────────────────────────────────────────────
TRAJ_DIR   = REPO_ROOT / "data" / "rl" / "trajectories"
POLICY_DIR = REPO_ROOT / "data" / "rl" / "policies"
LOG_DIR    = REPO_ROOT / "data" / "rl" / "training_log"

ACTION_MAP   = {a.name: a for a in ALL_ACTIONS}
ACTION_NAMES = [a.name for a in ALL_ACTIONS]


# ── Rollout helpers ───────────────────────────────────────────────────────────

def _load_iter0_rewards(task: str) -> dict[str, float]:
    rewards: dict[str, float] = {}
    for a in ALL_ACTIONS:
        p = TRAJ_DIR / task / a.name / "trajectory.json"
        if not p.exists():
            continue
        traj = json.loads(p.read_text())
        if not traj.get("skipped"):
            rewards[a.name] = traj["new_test_auroc"]
    return rewards


def _load_base_auroc(task: str) -> float:
    p = REPO_ROOT / "data" / "results" / "global-rubric" / task / "metrics.json"
    return json.loads(p.read_text()).get("test_auroc", float("nan")) if p.exists() else float("nan")


def _run_rollout_step(
    task: str,
    task_query: str,
    state: RubricState,
    base_auroc: float,
) -> dict[str, float]:
    """Dispatch all K actions on `state` to Modal; return {action: auroc}."""
    state_data   = {"rubric": state.rubric, "records": state.records}
    base_metrics = {"test_auroc": base_auroc}
    inputs = [(task, a.name, task_query, state_data, base_metrics) for a in ALL_ACTIONS]
    print(f"  Dispatching {len(inputs)} Modal rollout jobs…")
    results = list(rollout_one.starmap(inputs))
    rewards: dict[str, float] = {}
    for traj in results:
        aname = traj["action"]
        if not traj.get("skipped"):
            rewards[aname] = traj["new_test_auroc"]
        else:
            print(f"    {aname}: SKIPPED — {traj.get('skip_reason', '')}")
    return rewards


def _apply_with_fallback(
    chosen: str,
    state: RubricState,
    policy: ActionPolicy,
) -> tuple[str, RubricState]:
    """Try chosen; fall back by probability rank on ValueError."""
    ranked  = [ACTION_NAMES[i] for i in np.argsort(-policy.probabilities())]
    ordered = [chosen] + [a for a in ranked if a != chosen]
    for name in ordered:
        try:
            new_state = ACTION_MAP[name].apply(state)
            if name != chosen:
                print(f"    Fell back: {chosen} → {name}")
            return name, new_state
        except ValueError as e:
            print(f"    {name} not applicable ({e}), trying next…")
    raise RuntimeError("No applicable action found for current state.")


def _print_table(
    policy: ActionPolicy,
    rewards: dict[str, float],
    advantages: dict[str, float],
    step: int,
) -> None:
    probs = policy.probabilities()
    print(f"\n  ── Step {step} policy {'─'*38}")
    print(f"  {'ACTION':<35} {'REWARD':>7} {'ADV':>8} {'PROB':>6} {'LOGIT':>8}")
    print(f"  {'-'*35} {'-'*7} {'-'*8} {'-'*6} {'-'*8}")
    for i, name in enumerate(policy.action_names):
        r   = rewards.get(name, float("nan"))
        adv = advantages.get(name, 0.0)
        p   = float(probs[i])
        l   = float(policy.logits[i])
        r_s   = f"{r:.4f}"    if not np.isnan(r) else "  skip"
        adv_s = f"{adv:+.4f}" if advantages       else "       "
        print(f"  {name:<35} {r_s:>7} {adv_s:>8} {p:>6.3f} {l:>8.4f}")
    print()


# ── Per-task training loop ────────────────────────────────────────────────────

def _run_one_task(
    task: str,
    task_query: str,
    steps: int,
    lr: float,
    beta: float,
    greedy: bool,
    seed: int,
    traj_idx: int,
    n_trajectories: int,
    best_state_rollout: bool,
) -> dict:
    multi   = n_trajectories > 1
    mode    = "best_state" if best_state_rollout else "chain"
    run_tag = f"lr{lr}_beta{beta}_{'bs' if best_state_rollout else 'chain'}_s{steps}"
    tag     = f"{run_tag}/traj{traj_idx:02d}/" if multi else f"{run_tag}/"

    print(f"\n{'='*62}")
    print(f" GRPO [{mode}] — task={task}  steps={steps}  lr={lr}  beta={beta}  "
          f"greedy={greedy}  seed={seed}" + (f"  traj={traj_idx}" if multi else ""))
    print(f"{'='*62}")

    # ── Step 0: initial policy from iter-0 exhaustive rollout ─────────────
    print("\n[Step 0]  Loading iteration-0 rewards…")
    rewards_0 = _load_iter0_rewards(task)
    if len(rewards_0) < 2:
        raise RuntimeError(
            f"Need ≥ 2 iter-0 rewards for {task!r}, found {len(rewards_0)}.\n"
            f"Run `modal run rl/exhaustive_rollout.py --tasks {task}` first."
        )

    rng    = np.random.RandomState(seed)
    policy = ActionPolicy(ACTION_NAMES)
    adv_0  = policy.grpo_update(rewards_0, lr=lr, beta=beta)
    _print_table(policy, rewards_0, adv_0, step=0)
    policy.save(POLICY_DIR / task / f"{tag}policy_step_0.json")
    print(f"  Best action after step 0: {policy.best_action()}")

    state       = RubricState.from_disk(
        task,
        rubric_dir=REPO_ROOT / "data" / "rubric",
        rubricified_dir=REPO_ROOT / "data" / "rubric" / "rubricified",
    )
    state_auroc = _load_base_auroc(task)

    log: dict = {
        "task": task, "mode": mode,
        "steps": steps, "lr": lr, "beta": beta,
        "greedy": greedy, "seed": seed, "traj_idx": traj_idx,
        "history": [{
            "step": 0, "action_selected": None,
            "state_auroc": state_auroc,
            "rewards": rewards_0, "advantages": adv_0,
            "policy": policy.to_dict(),
        }],
    }

    # Tracking state — chain and best-state modes share initialisation
    prev_rewards = rewards_0   # chain: look up AUROC of chosen action
    best_state   = state       # best-state: highest-AUROC rubric seen
    best_auroc   = state_auroc # best-state: high-water mark

    # ── Training loop ──────────────────────────────────────────────────────
    for t in range(1, steps + 1):
        print(f"\n[Step {t}/{steps}]")

        if best_state_rollout:
            # ── Best-state: always branch from the highest-AUROC rubric ───
            print(f"  Branching from best state (AUROC={best_auroc:.4f})")
            rewards_t = _run_rollout_step(task, task_query, best_state, best_auroc)

            adv_t = policy.grpo_update(rewards_t, lr=lr, beta=beta)
            _print_table(policy, rewards_t, adv_t, step=t)
            policy.save(POLICY_DIR / task / f"{tag}policy_step_{t}.json")

            # Best action from this rollout (by raw reward, not policy)
            valid_r = {a: r for a, r in rewards_t.items() if not np.isnan(r)}
            best_action_step = max(valid_r, key=valid_r.get) if valid_r else None
            best_reward_step = valid_r[best_action_step] if best_action_step else float("nan")

            # Policy selection logged for reference; does not drive advancement
            chosen = policy.best_action() if greedy else policy.sample(rng)
            print(f"  Policy selected  : {chosen}")

            # Advance best_state only when AUROC improves
            advanced = False
            if best_action_step and best_reward_step > best_auroc:
                best_state = ACTION_MAP[best_action_step].apply(best_state)
                best_auroc = best_reward_step
                advanced   = True
                print(f"  Best state → {best_action_step}  AUROC={best_auroc:.4f}")
            else:
                print(f"  Best state unchanged  (AUROC={best_auroc:.4f})")

            log["history"].append({
                "step": t,
                "action_selected": chosen,
                "best_action_step": best_action_step,
                "best_state_advanced": advanced,
                "state_auroc": best_auroc,
                "rewards": rewards_t, "advantages": adv_t,
                "policy": policy.to_dict(),
            })

        else:
            # ── Chain: apply policy action, then branch from result ────────
            chosen = policy.best_action() if greedy else policy.sample(rng)
            probs  = policy.probabilities()
            print(f"  Selected : {chosen}")
            print(f"  Probs    : " + "  ".join(
                f"{n[:8]}={probs[i]:.3f}" for i, n in enumerate(ACTION_NAMES)
            ))

            chosen, state = _apply_with_fallback(chosen, state, policy)
            state_auroc   = prev_rewards.get(chosen, state_auroc)
            print(f"  State AUROC after action: {state_auroc:.4f}")

            rewards_t = _run_rollout_step(task, task_query, state, state_auroc)

            adv_t = policy.grpo_update(rewards_t, lr=lr, beta=beta)
            _print_table(policy, rewards_t, adv_t, step=t)
            policy.save(POLICY_DIR / task / f"{tag}policy_step_{t}.json")

            log["history"].append({
                "step": t, "action_selected": chosen,
                "state_auroc": state_auroc,
                "rewards": rewards_t, "advantages": adv_t,
                "policy": policy.to_dict(),
            })
            prev_rewards = rewards_t

    # ── Save log ───────────────────────────────────────────────────────────
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    traj_suffix = f"_traj{traj_idx:02d}" if multi else ""
    log_path    = LOG_DIR / f"{task}_{run_tag}{traj_suffix}.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\nTraining log → {log_path}")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f" Final policy [{mode}]" + (f"  [traj {traj_idx}]" if multi else ""))
    _print_table(policy, {}, {}, step=steps)
    print("  AUROC progression:")
    for entry in log["history"]:
        s   = entry["step"]
        act = entry["action_selected"] or "—"
        auc = entry["state_auroc"]
        note = ""
        if best_state_rollout and entry.get("best_action_step"):
            note = f"  best={entry['best_action_step']}"
            if entry.get("best_state_advanced"):
                note += " ✓"
        print(f"    step {s:2d}  {act:<35}  {auc:.4f}{note}")
    print(f"{'='*62}\n")
    return log


# ── Modal entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def train_grpo(
    task:               str   = "guo_readmission",
    tasks:              str   = "",
    steps:              int   = 5,
    lr:                 float = 0.1,
    beta:               float = 0.0,
    greedy:             bool  = False,
    seed:               int   = 42,
    n_trajectories:     int   = 1,
    best_state_rollout: bool  = False,
) -> None:
    """
    GRPO training loop across one or more tasks.

    --task               : single task name  (default: guo_readmission)
    --tasks              : comma-separated list, or "all" for all 4 tasks
    --steps              : RL steps after the step-0 policy init  (default: 5)
    --lr                 : GRPO learning rate  (default: 0.1)
    --beta               : KL-penalty toward uniform reference  (default: 0.0)
    --greedy             : argmax action selection; default is stochastic
    --seed               : base RNG seed  (default: 42)
    --n-trajectories     : independent runs per task (seed+i each)  (default: 1)
    --best-state-rollout : always branch from the highest-AUROC rubric seen;
                           state advances only when reward improves  (default: False)
    """
    from config.tasks import TASKS as TASK_QUERIES

    if tasks == "all" or task == "all":
        task_list = list(TASKS)
    elif tasks:
        task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    else:
        task_list = [task]

    print(f"Tasks ({len(task_list)}): {task_list}")
    print(f"Trajectories per task: {n_trajectories}")
    print(f"Mode: {'best_state' if best_state_rollout else 'chain'}")

    all_logs: list[dict] = []
    for t_name in task_list:
        task_query = TASK_QUERIES.get(t_name, "")
        for traj_idx in range(n_trajectories):
            log = _run_one_task(
                task=t_name,
                task_query=task_query,
                steps=steps,
                lr=lr,
                beta=beta,
                greedy=greedy,
                seed=seed + traj_idx,
                traj_idx=traj_idx,
                n_trajectories=n_trajectories,
                best_state_rollout=best_state_rollout,
            )
            all_logs.append(log)

    # ── Cross-task / cross-trajectory summary ─────────────────────────────────
    if len(all_logs) > 1:
        print(f"\n{'='*70}")
        print(" Summary")
        print(f"{'='*70}")
        print(f"  {'TASK':<22} {'TRAJ':>4}  {'INIT AUROC':>10}  {'FINAL AUROC':>11}")
        print(f"  {'-'*22} {'-'*4}  {'-'*10}  {'-'*11}")
        for lg in all_logs:
            init_auc  = lg["history"][0]["state_auroc"]
            final_auc = lg["history"][-1]["state_auroc"]
            print(f"  {lg['task']:<22} {lg['traj_idx']:>4}  {init_auc:>10.4f}  {final_auc:>11.4f}")
        print(f"{'='*70}\n")


# ── Local inspection utilities (no Modal needed) ──────────────────────────────

def show_policy(task: str) -> None:
    task_dir = POLICY_DIR / task
    if not task_dir.exists():
        print(f"No snapshots found for {task!r} in {task_dir}")
        return
    for snap in sorted(task_dir.rglob("policy_step_*.json")):
        policy = ActionPolicy.load(snap)
        label  = snap.relative_to(POLICY_DIR / task)
        print(f"\n── {label} ──────────────────────────────────────")
        print(f"  {'ACTION':<35} {'PROB':>6} {'LOGIT':>8}")
        print(f"  {'-'*35} {'-'*6} {'-'*8}")
        probs = policy.probabilities()
        for i, name in enumerate(policy.action_names):
            marker = " ◀" if i == int(np.argmax(probs)) else ""
            print(f"  {name:<35} {probs[i]:>6.3f} {policy.logits[i]:>8.4f}{marker}")


def show_log(task: str) -> None:
    # Collect all log files for this task (single + multi-trajectory)
    log_files = sorted(LOG_DIR.glob(f"{task}*.json"))
    if not log_files:
        print(f"No training log found for {task!r} in {LOG_DIR}")
        return
    for log_path in log_files:
        log = json.loads(log_path.read_text())
        mode = log.get("mode", "chain")
        print(f"\n── {log_path.name}  [{mode}]  lr={log['lr']}  beta={log['beta']} ──")
        print(f"  {'STEP':>4}  {'ACTION SELECTED':<35}  {'AUROC':>7}"
              + ("  BEST ACTION / ADV?" if mode == "best_state" else ""))
        print(f"  {'-'*4}  {'-'*35}  {'-'*7}"
              + ("  " + "-"*25 if mode == "best_state" else ""))
        for entry in log["history"]:
            s   = entry["step"]
            act = entry["action_selected"] or "— (init)"
            auc = entry["state_auroc"]
            extra = ""
            if mode == "best_state" and entry.get("best_action_step"):
                adv = "✓" if entry.get("best_state_advanced") else " "
                extra = f"  {adv} {entry['best_action_step']}"
            print(f"  {s:>4}  {act:<35}  {auc:>7.4f}{extra}")
        print()
        if any(entry.get("rewards") for entry in log["history"]):
            print("  Per-step reward tables:")
            for entry in log["history"]:
                if not entry.get("rewards"):
                    continue
                print(f"\n  Step {entry['step']} rewards (from {'best state' if mode == 'best_state' else 'current state'}):")
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
        print("\nTo run training:  modal run rl/train_grpo.py --task TASK")


if __name__ == "__main__":
    main()
