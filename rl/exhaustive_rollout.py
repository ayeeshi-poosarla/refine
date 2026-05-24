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
        "torch==2.3.0",
        "transformers>=4.40.0",
        "numpy",
        "scikit-learn",
        "scipy",
        "tqdm",
    )
    .copy_local_dir(str(REPO_ROOT / "rl"),     "/REFINE/rl")
    .copy_local_dir(str(REPO_ROOT / "config"), "/REFINE/config")
)

# ── Modal Volumes ──────────────────────────────────────────────────────────────
model_cache = modal.Volume.from_name("refine-qwen3-weights",   create_if_missing=True)
results_vol = modal.Volume.from_name("refine-rollout-results", create_if_missing=True)

app = modal.App("refine-exhaustive-rollout")

TASKS = ["guo_readmission", "guo_los", "new_lupus", "lab_hyperkalemia"]


# ── Remote function ────────────────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/cache":   model_cache,
        "/results": results_vol,
    },
    timeout=3600,
    memory=32768,
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
    new_state   = action_map[action_name].apply(state)
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
            MODEL_CACHE, trust_remote_code=True, torch_dtype=torch.float16
        )
    else:
        print(f"{tag} Downloading {MODEL_NAME} (first run only)")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, padding_side="left"
        )
        model = AutoModel.from_pretrained(
            MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float16
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
    tasks:   list[str] = TASKS,
    actions: list[str] = [],
):
    """
    Launch exhaustive rollout.

    --tasks   : subset of tasks to run (default: all 4 completed tasks)
    --actions : subset of action names to run (default: all 6 actions)
    """
    sys.path.insert(0, str(REPO_ROOT))
    from rl.actions import ALL_ACTIONS
    from config.tasks import TASKS as TASK_QUERIES

    action_names = actions if actions else [a.name for a in ALL_ACTIONS]

    print(f"Tasks   ({len(tasks)}):   {tasks}")
    print(f"Actions ({len(action_names)}): {action_names}")
    print(f"Total jobs: {len(tasks) * len(action_names)}")

    # Validate tasks have data before submitting
    inputs = []
    for task in tasks:
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
        base   = traj["base_test_auroc"]
        new    = traj["new_test_auroc"]
        delta  = traj["delta_auroc"]
        print(f"{task:<22} {action:<35} {base:>6.4f} {new:>6.4f} {delta:>+7.4f}")
        _save_locally(task, action, traj)

    print("=" * 80)
    print(f"\nDone. {len(results)} trajectories saved to data/rl/trajectories/")
