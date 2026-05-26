#!/usr/bin/env python3
"""
Exhaustive RL rollout — apply every action to every completed task in parallel
on Modal GPU instances.

Each (task × action) pair gets its own A100 container which:
  1. Applies the action programmatically to the task's current rubric + records.
  2. Wraps each modified rubricified_text into the standard SFT prompt format
     (matches create_globalrubric_sft.py).
  3. Generates Qwen3-Embedding-8B embeddings with last-token pooling + L2 norm
     (matches generate_embeddings.py, batch_size=2).
  4. Evaluates via val-set LogReg C-selection → test AUROC + bootstrap CIs
     (matches eval_embeddings.py).
  5. Saves a full trajectory (s0, action, s1, reward) to the Modal Volume and
     to local disk.

Trajectory files written to:
  Modal Volume  : refine-rollout-results  /{task}/{action_name}/
  Local disk    : data/rl/trajectories/{task}/{action_name}/

Run from REFINE repo root:
    modal run rl/exhaustive_rollout.py
    modal run rl/exhaustive_rollout.py --tasks guo_readmission guo_los
    modal run rl/exhaustive_rollout.py --actions remove_lowest_variance_field
"""

import json
import sys
from pathlib import Path

import modal

# ── Repo root (works from both GCP VM and local machine) ──────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Evaluation constants (mirrors eval_embeddings.py) ─────────────────────────
C_VALUES = [
    1e-6, 2.61e-6, 6.81e-6, 1.78e-5, 4.64e-5,
    1.21e-4, 3.16e-4, 8.25e-4, 2.15e-3, 5.62e-3,
    1.47e-2, 3.83e-2, 0.1,
]
SEED = 42
BOOTSTRAP_N = 1000

# SFT prompt template (mirrors create_globalrubric_sft.py)
SYSTEM_MESSAGE = "You are a medical expert specializing in clinical risk prediction."


def _build_prompt(task_query: str, rubricified_text: str) -> str:
    user = (
        f"Based on the patient's EHR below, predict: {task_query}\n\n"
        f"--- Patient EHR ---\n{rubricified_text}\n--- End of EHR ---\n\n"
        f"Based on the above EHR, predict: {task_query}\n"
        "Respond with exactly one word: Yes or No."
    )
    return f"{SYSTEM_MESSAGE}\n\n{user}"


# ── Modal image: bake rl/ + config/ code in at build time ─────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.51.0",
        "numpy",
        "scikit-learn",
        "scipy",
        "tqdm",
    )
    .add_local_dir(str(REPO_ROOT / "rl"),     remote_path="/REFINE/rl")
    .add_local_dir(str(REPO_ROOT / "config"), remote_path="/REFINE/config")
)

# ── Modal Volumes ──────────────────────────────────────────────────────────────
model_cache = modal.Volume.from_name("refine-qwen3-weights",   create_if_missing=True)
results_vol = modal.Volume.from_name("refine-rollout-results", create_if_missing=True)

app = modal.App("refine-exhaustive-rollout")

TASKS = ["guo_readmission", "guo_los", "new_lupus", "lab_hyperkalemia"]


# ── Remote function ────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    max_containers=4,
    volumes={
        "/cache":   model_cache,
        "/results": results_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    memory=16384,
)
def rollout_one(
    task: str,
    action_name: str,
    task_query: str,
    state_data: dict,
    base_metrics: dict,
) -> dict:
    """Apply one action, embed with Qwen3, evaluate LogReg, save trajectory."""
    import os
    import json
    import sys

    import numpy as np
    import torch
    import torch.nn.functional as F
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm

    sys.path.insert(0, "/REFINE")
    from rl.actions import ALL_ACTIONS
    from rl.state import RubricState

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    tag = f"[{task}|{action_name}]"
    print(f"{tag} Starting")

    # ── Apply action ──────────────────────────────────────────────────────────
    action_map = {a.name: a for a in ALL_ACTIONS}
    if action_name not in action_map:
        raise ValueError(f"Unknown action: {action_name!r}")

    state = RubricState(
        task=task,
        rubric=state_data["rubric"],
        records=state_data["records"],
    )
    try:
        new_state   = action_map[action_name].apply(state)
    except ValueError as e:
        print(f"{tag} Action not applicable — skipping: {e}")
        return {
            "task": task, "action": action_name, "skipped": True,
            "skip_reason": str(e),
            "base_test_auroc": base_metrics.get("test_auroc", float("nan")),
            "new_test_auroc": float("nan"),
            "delta_auroc": float("nan"),
        }
    action_meta = new_state.rubric.get("_last_action", {})
    print(f"{tag} Action applied — meta: {json.dumps(action_meta, default=str)}")

    # ── Load / cache Qwen3-Embedding-8B ──────────────────────────────────────
    MODEL_NAME  = "Qwen/Qwen3-Embedding-8B"
    MODEL_CACHE = "/cache/qwen3-embedding-8b"
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if Path(MODEL_CACHE).exists():
        print(f"{tag} Loading model from volume cache")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CACHE, trust_remote_code=True, padding_side="left"
        )
        model = AutoModel.from_pretrained(
            MODEL_CACHE, trust_remote_code=True, dtype=torch.float16
        )
    else:
        print(f"{tag} Downloading {MODEL_NAME} (first run only)")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, padding_side="left"
        )
        model = AutoModel.from_pretrained(
            MODEL_NAME, trust_remote_code=True, dtype=torch.float16
        )
        tokenizer.save_pretrained(MODEL_CACHE)
        model.save_pretrained(MODEL_CACHE)
        model_cache.commit()
        print(f"{tag} Model cached to volume")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device).eval()

    # ── Embed all splits ──────────────────────────────────────────────────────
    def _last_token_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Last non-padding token (matches generate_embeddings.py EmbeddingEncoder._pool)."""
        if mask[:, -1].sum() == mask.shape[0]:
            return hidden[:, -1]
        seq_len = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_len]

    @torch.no_grad()
    def embed(texts: list, batch_size: int = 2) -> np.ndarray:
        # Instruction prefix matches EmbeddingEncoder.encode(instruction=task_query)
        prefixed = [f"Instruct: {task_query}\nQuery:\n{t}" for t in texts]
        parts = []
        for i in tqdm(range(0, len(prefixed), batch_size), desc=f"{tag} embed"):
            batch = prefixed[i : i + batch_size]
            tok = tokenizer(
                batch,
                max_length=8192,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            out = model(**tok)
            h   = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            emb = _last_token_pool(h, tok["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1).cpu().numpy()
            parts.append(emb)
        return np.concatenate(parts, axis=0)

    split_embeddings: dict = {}
    split_labels:     dict = {}
    for split, recs in new_state.records.items():
        prompts = [_build_prompt(task_query, r["rubricified_text"]) for r in recs]
        labels  = np.array([int(r["label"]) for r in recs])
        print(f"{tag} Embedding {split} ({len(prompts)} records)")
        split_embeddings[split] = embed(prompts)
        split_labels[split]     = labels

    # ── Evaluate: val-based C selection → test AUROC (mirrors eval_embeddings.py) ──
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(split_embeddings["train"])
    y_train = split_labels["train"]
    X_test  = scaler.transform(split_embeddings["test"])
    y_test  = split_labels["test"]

    best_c       = C_VALUES[0]
    best_val_auc = -1.0
    if "val" in split_embeddings:
        X_val = scaler.transform(split_embeddings["val"])
        y_val = split_labels["val"]
        for c in C_VALUES:
            clf = LogisticRegression(
                C=c, penalty="l2", solver="lbfgs",
                max_iter=1000, class_weight="balanced", random_state=SEED,
            )
            clf.fit(X_train, y_train)
            if len(np.unique(y_val)) >= 2:
                score = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
                if score > best_val_auc:
                    best_val_auc, best_c = score, c

    clf_final = LogisticRegression(
        C=best_c, penalty="l2", solver="lbfgs",
        max_iter=1000, class_weight="balanced", random_state=SEED,
    )
    clf_final.fit(X_train, y_train)
    y_score = clf_final.predict_proba(X_test)[:, 1]

    def _bootstrap_ci(y_true, y_sc, metric_fn, n: int = BOOTSTRAP_N) -> tuple:
        rng    = np.random.RandomState(SEED)
        scores = []
        for _ in range(n):
            idx = rng.choice(len(y_true), len(y_true), replace=True)
            if len(np.unique(y_true[idx])) >= 2:
                scores.append(metric_fn(y_true[idx], y_sc[idx]))
        arr = np.array(scores) if scores else np.array([0.0])
        return float(np.mean(arr)), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    auroc, auroc_lo, auroc_hi = _bootstrap_ci(y_test, y_score, roc_auc_score)
    auprc, auprc_lo, auprc_hi = _bootstrap_ci(y_test, y_score, average_precision_score)

    eval_metrics = {
        "best_c":        best_c,
        "val_auroc":     best_val_auc,
        "test_auroc":    auroc,
        "test_auroc_ci": [auroc_lo, auroc_hi],
        "test_auprc":    auprc,
        "test_auprc_ci": [auprc_lo, auprc_hi],
        "n_train":       int(len(y_train)),
        "n_test":        int(len(y_test)),
    }

    base_auroc = base_metrics.get("test_auroc", float("nan"))
    print(
        f"{tag} base={base_auroc:.4f} → new={auroc:.4f}  "
        f"(Δ={auroc - base_auroc:+.4f})"
    )

    # ── Assemble trajectory ───────────────────────────────────────────────────
    trajectory = {
        "task":                       task,
        "action":                     action_name,
        "action_meta":                action_meta,
        "base_test_auroc":            base_auroc,
        "new_test_auroc":             auroc,
        "delta_auroc":                auroc - base_auroc,
        "before_rubric_instructions": state.rubric["rubric_instructions"],
        "after_rubric_instructions":  new_state.rubric["rubric_instructions"],
        "n_records":                  {s: len(r) for s, r in new_state.records.items()},
        "eval_metrics":               eval_metrics,
    }

    # ── Persist to Modal Volume ───────────────────────────────────────────────
    out = Path(f"/results/{task}/{action_name}")
    out.mkdir(parents=True, exist_ok=True)

    (out / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
    (out / "rubric.json").write_text(json.dumps(new_state.rubric, indent=2))
    for split, recs in new_state.records.items():
        (out / f"{split}.json").write_text(json.dumps(recs, indent=2))
    for split, emb in split_embeddings.items():
        np.savez(
            str(out / f"embeddings_{split}.npz"),
            embeddings=emb,
            labels=split_labels[split],
        )

    results_vol.commit()
    print(f"{tag} Saved to /results/{task}/{action_name}/")
    return trajectory


# ── Local helpers ──────────────────────────────────────────────────────────────
def _load_state_data(task: str) -> dict:
    rubric_dir      = REPO_ROOT / "data" / "rubric"
    rubricified_dir = rubric_dir / "rubricified"

    rubric  = json.loads((rubric_dir / task / "rubric.json").read_text())
    records: dict = {}
    for split in ("train", "val", "test"):
        p = rubricified_dir / task / f"{split}.json"
        if p.exists():
            records[split] = json.loads(p.read_text())
    if not records:
        raise FileNotFoundError(f"No rubricified splits found for task: {task}")
    return {"rubric": rubric, "records": records}


def _load_base_metrics(task: str) -> dict:
    p = REPO_ROOT / "data" / "results" / "global-rubric" / task / "metrics.json"
    if not p.exists():
        print(f"  WARNING: no base metrics found for {task}, delta_auroc will be NaN")
        return {}
    return json.loads(p.read_text())


def _save_locally(task: str, action_name: str, trajectory: dict) -> None:
    out = REPO_ROOT / "data" / "rl" / "trajectories" / task / action_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectory.json").write_text(json.dumps(trajectory, indent=2))


# ── Local entrypoint ───────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(
    tasks:   str = "",
    actions: str = "",
):
    """
    Launch exhaustive rollout.

    --tasks   : comma-separated task names (default: all 4 completed tasks)
    --actions : comma-separated action names (default: all 6 actions)
    """
    sys.path.insert(0, str(REPO_ROOT))
    from rl.actions import ALL_ACTIONS
    from config.tasks import TASKS as TASK_QUERIES

    task_list    = [t.strip() for t in tasks.split(",") if t.strip()] if tasks else TASKS
    action_names = [a.strip() for a in actions.split(",") if a.strip()] if actions else [a.name for a in ALL_ACTIONS]

    print(f"Tasks   ({len(task_list)}):   {task_list}")
    print(f"Actions ({len(action_names)}): {action_names}")
    print(f"Total jobs: {len(task_list) * len(action_names)}")

    # Validate tasks have data before submitting
    inputs = []
    for task in task_list:
        state_data   = _load_state_data(task)
        base_metrics = _load_base_metrics(task)
        task_query   = TASK_QUERIES.get(task, "")
        for action_name in action_names:
            inputs.append((task, action_name, task_query, state_data, base_metrics))

    # Run all jobs in parallel on Modal
    print(f"\nSubmitting {len(inputs)} jobs to Modal...")
    results = list(rollout_one.starmap(inputs))

    # Save locally and print summary table
    print("\n" + "=" * 80)
    print(f"{'TASK':<22} {'ACTION':<35} {'BASE':>6} {'NEW':>6} {'DELTA':>7}")
    print("=" * 80)
    for traj in sorted(results, key=lambda t: (t["task"], t["action"])):
        task   = traj["task"]
        action = traj["action"]
        if traj.get("skipped"):
            print(f"{task:<22} {action:<35} {'SKIP':>6} {'':>6} {traj['skip_reason'][:30]}")
            continue
        base   = traj["base_test_auroc"]
        new    = traj["new_test_auroc"]
        delta  = traj["delta_auroc"]
        print(f"{task:<22} {action:<35} {base:>6.4f} {new:>6.4f} {delta:>+7.4f}")
        _save_locally(task, action, traj)

    print("=" * 80)
    print(f"\nDone. {len(results)} trajectories saved to data/rl/trajectories/")


# ── GRPO training loop ─────────────────────────────────────────────────────────
# Run with:
#   modal run rl/exhaustive_rollout.py::train_grpo --task guo_readmission
#   modal run rl/exhaustive_rollout.py::train_grpo --task guo_readmission --steps 5 --greedy

def _grpo_load_iter0_rewards(task: str, traj_root: Path, all_action_names: list) -> dict:
    rewards = {}
    for name in all_action_names:
        p = traj_root / task / name / "trajectory.json"
        if not p.exists():
            continue
        traj = json.loads(p.read_text())
        if not traj.get("skipped"):
            rewards[name] = traj["new_test_auroc"]
    return rewards


def _grpo_run_rollout_step(
    task: str,
    task_query: str,
    state,
    base_auroc: float,
    all_actions: list,
) -> dict:
    """Dispatch exhaustive rollout on `state` via Modal; return {action: auroc}."""
    state_data   = {"rubric": state.rubric, "records": state.records}
    base_metrics = {"test_auroc": base_auroc}
    inputs = [
        (task, a.name, task_query, state_data, base_metrics)
        for a in all_actions
    ]
    print(f"  Dispatching {len(inputs)} Modal rollout jobs…")
    results = list(rollout_one.starmap(inputs))
    rewards = {}
    for traj in results:
        aname = traj["action"]
        if not traj.get("skipped"):
            rewards[aname] = traj["new_test_auroc"]
        else:
            print(f"    {aname}: SKIPPED — {traj.get('skip_reason', '')}")
    return rewards


def _grpo_apply_with_fallback(chosen: str, state, policy, action_map: dict) -> tuple:
    """Apply chosen action; fall back by probability rank on ValueError."""
    import numpy as np
    probs   = policy.probabilities()
    ranked  = [policy.action_names[i] for i in np.argsort(-probs)]
    ordered = [chosen] + [a for a in ranked if a != chosen]
    for name in ordered:
        try:
            new_state = action_map[name].apply(state)
            if name != chosen:
                print(f"    Fell back: {chosen} → {name}")
            return name, new_state
        except ValueError as e:
            print(f"    {name} not applicable ({e}), trying next…")
    raise RuntimeError("No applicable action found for current state.")


def _grpo_print_table(policy, rewards: dict, advantages: dict, step: int) -> None:
    import numpy as np
    probs = policy.probabilities()
    print(f"\n  ── Step {step} policy {'─'*38}")
    print(f"  {'ACTION':<35} {'REWARD':>7} {'ADV':>8} {'PROB':>6} {'LOGIT':>8}")
    print(f"  {'-'*35} {'-'*7} {'-'*8} {'-'*6} {'-'*8}")
    for i, name in enumerate(policy.action_names):
        r   = rewards.get(name, float("nan"))
        adv = advantages.get(name, 0.0)
        p   = float(probs[i])
        l   = float(policy.logits[i])
        r_s   = f"{r:.4f}"    if not np.isnan(r)  else "  skip"
        adv_s = f"{adv:+.4f}" if advantages        else "       "
        print(f"  {name:<35} {r_s:>7} {adv_s:>8} {p:>6.3f} {l:>8.4f}")
    print()


@app.local_entrypoint()
def train_grpo(
    task:   str   = "guo_readmission",
    steps:  int   = 5,
    lr:     float = 0.1,
    greedy: bool  = False,
    seed:   int   = 42,
) -> None:
    """
    GRPO training loop (T steps) for a single task.

    --task   : EHRSHOT task name  (default: guo_readmission)
    --steps  : RL steps after the initial policy update  (default: 5)
    --lr     : GRPO learning rate  (default: 0.1)
    --greedy : deterministic action selection (argmax); default is stochastic
    --seed   : RNG seed for stochastic sampling  (default: 42)
    """
    import numpy as np
    sys.path.insert(0, str(REPO_ROOT))

    from config.tasks import TASKS as TASK_QUERIES
    from rl.actions import ALL_ACTIONS
    from rl.policy import ActionPolicy
    from rl.state import RubricState

    task_query   = TASK_QUERIES.get(task, "")
    action_map   = {a.name: a for a in ALL_ACTIONS}
    action_names = [a.name for a in ALL_ACTIONS]
    traj_root    = REPO_ROOT / "data" / "rl" / "trajectories"
    policy_dir   = REPO_ROOT / "data" / "rl" / "policies"
    log_dir      = REPO_ROOT / "data" / "rl" / "training_log"
    rng          = np.random.RandomState(seed)

    print(f"\n{'='*62}")
    print(f" GRPO Training — task={task}  steps={steps}  lr={lr}  greedy={greedy}")
    print(f"{'='*62}")

    # ── Step 0: policy from iteration-0 rewards ────────────────────────────────
    print("\n[Step 0]  Loading iteration-0 exhaustive rollout rewards…")
    rewards_0 = _grpo_load_iter0_rewards(task, traj_root, action_names)

    if len(rewards_0) < 2:
        raise RuntimeError(
            f"Need ≥ 2 iter-0 rewards for {task!r}, found {len(rewards_0)}.\n"
            f"Run `modal run rl/exhaustive_rollout.py --tasks {task}` first."
        )

    policy = ActionPolicy(action_names)
    adv_0  = policy.grpo_update(rewards_0, lr=lr)
    _grpo_print_table(policy, rewards_0, adv_0, step=0)
    policy.save(policy_dir / task / "policy_step_0.json")
    print(f"  Best action after step 0: {policy.best_action()}")

    # Load initial state and base AUROC
    state = RubricState.from_disk(
        task,
        rubric_dir=REPO_ROOT / "data" / "rubric",
        rubricified_dir=REPO_ROOT / "data" / "rubric" / "rubricified",
    )
    base_metrics_path = REPO_ROOT / "data" / "results" / "global-rubric" / task / "metrics.json"
    state_auroc = (
        json.loads(base_metrics_path.read_text()).get("test_auroc", float("nan"))
        if base_metrics_path.exists() else float("nan")
    )

    log = {
        "task": task, "steps": steps, "lr": lr, "greedy": greedy, "seed": seed,
        "history": [{
            "step": 0, "action_selected": None,
            "state_auroc": state_auroc,
            "rewards": rewards_0, "advantages": adv_0,
            "policy": policy.to_dict(),
        }],
    }

    prev_rewards = rewards_0   # rewards for actions on the current state

    # ── Training loop ──────────────────────────────────────────────────────────
    for t in range(1, steps + 1):
        print(f"\n[Step {t}/{steps}]")

        # a. Select action
        chosen = policy.best_action() if greedy else policy.sample(rng)
        probs  = policy.probabilities()
        print(f"  Selected : {chosen}")
        print(f"  Probs    : " + "  ".join(
            f"{n[:8]}={probs[i]:.3f}" for i, n in enumerate(action_names)
        ))

        # b. Apply action (pure Python, no GPU)
        chosen, state = _grpo_apply_with_fallback(chosen, state, policy, action_map)
        state_auroc = prev_rewards.get(chosen, state_auroc)
        print(f"  State AUROC after action: {state_auroc:.4f}")

        # c. Exhaustive mini-rollout on new state via Modal (rollout_one is in this app)
        rewards_t = _grpo_run_rollout_step(task, task_query, state, state_auroc, ALL_ACTIONS)

        # d. GRPO update
        adv_t = policy.grpo_update(rewards_t, lr=lr)
        _grpo_print_table(policy, rewards_t, adv_t, step=t)
        policy.save(policy_dir / task / f"policy_step_{t}.json")

        log["history"].append({
            "step": t, "action_selected": chosen,
            "state_auroc": state_auroc,
            "rewards": rewards_t, "advantages": adv_t,
            "policy": policy.to_dict(),
        })

        prev_rewards = rewards_t

    # ── Save log ───────────────────────────────────────────────────────────────
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task}.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\nTraining log → {log_path}")

    # ── Final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f" Final policy — {steps} RL steps")
    _grpo_print_table(policy, {}, {}, step=steps)
    print("  AUROC progression:")
    for entry in log["history"]:
        s   = entry["step"]
        act = entry["action_selected"] or "—"
        auc = entry["state_auroc"]
        print(f"    step {s:2d}  {act:<35}  {auc:.4f}")
    print(f"{'='*62}\n")
