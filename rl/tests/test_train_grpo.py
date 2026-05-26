#!/usr/bin/env python3
"""Tests for rl/train_grpo.py — GRPO training helpers.

Modal-dependent functions (_run_rollout_step, _run_one_task) are tested with
rollout_one.starmap patched out so no GPU or network calls are made.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rl.policy import ActionPolicy
from rl.state import RubricState
import rl.train_grpo as tg


# ── Fixtures ─────────────────────────────────────────────────────────────────

ACTIONS = ["action_a", "action_b", "action_c", "action_d"]


def _make_record(patient_id, split_label, label, fields=None):
    fields = fields or {"AGE": "65", "SEX": "MALE"}
    text = "\n".join(f"*   **{k}:** {v}" for k, v in fields.items())
    return {
        "patient_id": patient_id,
        "prediction_time": "2020-01-01",
        "label": label,
        "rubricified_text": text,
    }


def _make_state(n_train=4, n_val=2, n_test=2):
    rubric = {
        "task": "test_task",
        "rubric_instructions": "*   **AGE:** [age]\n*   **SEX:** [sex]\n",
        "task_query": "Will the patient be readmitted?",
        "num_examples": 0,
        "usage": {},
    }
    records = {
        "train": [_make_record(i, "train", i % 2 == 0) for i in range(n_train)],
        "val":   [_make_record(i, "val",   i % 2 == 0) for i in range(n_val)],
        "test":  [_make_record(i, "test",  i % 2 == 0) for i in range(n_test)],
    }
    return RubricState(task="test_task", rubric=rubric, records=records)


def _write_trajectory(traj_dir: Path, task: str, action: str, auroc: float,
                      skipped: bool = False, skip_reason: str = ""):
    p = traj_dir / task / action
    p.mkdir(parents=True, exist_ok=True)
    traj = {
        "task": task, "action": action,
        "new_test_auroc": auroc,
        "base_test_auroc": 0.70,
        "delta_auroc": auroc - 0.70,
    }
    if skipped:
        traj["skipped"] = True
        traj["skip_reason"] = skip_reason
        traj["new_test_auroc"] = float("nan")
    (p / "trajectory.json").write_text(json.dumps(traj))


def _mock_rollout_results(rewards: dict[str, float]) -> list[dict]:
    """Build the list rollout_one.starmap would return."""
    results = []
    for action, auroc in rewards.items():
        if np.isnan(auroc):
            results.append({
                "action": action, "skipped": True,
                "skip_reason": "not applicable",
                "new_test_auroc": float("nan"),
            })
        else:
            results.append({
                "action": action, "skipped": False,
                "new_test_auroc": auroc,
            })
    return results


# ── _load_iter0_rewards ───────────────────────────────────────────────────────

def test_load_iter0_rewards_reads_all_valid():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task = "test_task"
        _write_trajectory(tmp, task, "action_a", 0.75)
        _write_trajectory(tmp, task, "action_b", 0.80)

        a1, a2 = MagicMock(), MagicMock()
        a1.name, a2.name = "action_a", "action_b"
        with patch.object(tg, "TRAJ_DIR", tmp), \
             patch.object(tg, "ALL_ACTIONS", [a1, a2]):
            rewards = tg._load_iter0_rewards(task)

        assert rewards["action_a"] == 0.75
        assert rewards["action_b"] == 0.80
    print("PASS test_load_iter0_rewards_reads_all_valid")


def test_load_iter0_rewards_skips_skipped_actions():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task = "test_task"
        _write_trajectory(tmp, task, "action_a", 0.75)
        _write_trajectory(tmp, task, "action_b", float("nan"), skipped=True, skip_reason="no fields")

        a1, a2 = MagicMock(), MagicMock()
        a1.name, a2.name = "action_a", "action_b"
        with patch.object(tg, "TRAJ_DIR", tmp), \
             patch.object(tg, "ALL_ACTIONS", [a1, a2]):
            rewards = tg._load_iter0_rewards(task)

        assert "action_a" in rewards
        assert "action_b" not in rewards
    print("PASS test_load_iter0_rewards_skips_skipped_actions")


def test_load_iter0_rewards_missing_file_not_included():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        task = "test_task"
        _write_trajectory(tmp, task, "action_a", 0.75)
        # action_b file does not exist

        a1, a2 = MagicMock(), MagicMock()
        a1.name, a2.name = "action_a", "action_b"
        with patch.object(tg, "TRAJ_DIR", tmp), \
             patch.object(tg, "ALL_ACTIONS", [a1, a2]):
            rewards = tg._load_iter0_rewards(task)

        assert "action_a" in rewards
        assert "action_b" not in rewards
    print("PASS test_load_iter0_rewards_missing_file_not_included")


# ── _load_base_auroc ──────────────────────────────────────────────────────────

def test_load_base_auroc_reads_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        metrics_path = tmp / "data" / "results" / "global-rubric" / "test_task" / "metrics.json"
        metrics_path.parent.mkdir(parents=True)
        metrics_path.write_text(json.dumps({"test_auroc": 0.713}))

        with patch.object(tg, "REPO_ROOT", tmp):
            auroc = tg._load_base_auroc("test_task")

        assert abs(auroc - 0.713) < 1e-9
    print("PASS test_load_base_auroc_reads_metrics")


def test_load_base_auroc_returns_nan_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(tg, "REPO_ROOT", Path(tmp)):
            auroc = tg._load_base_auroc("nonexistent_task")
    assert np.isnan(auroc)
    print("PASS test_load_base_auroc_returns_nan_when_missing")


# ── _apply_with_fallback ──────────────────────────────────────────────────────

def _make_mock_action(name: str, raises: bool = False) -> MagicMock:
    action = MagicMock()
    action.name = name
    if raises:
        action.apply.side_effect = ValueError(f"{name} not applicable")
    else:
        action.apply.return_value = _make_state()
    return action


def test_apply_with_fallback_succeeds_first_try():
    state  = _make_state()
    policy = ActionPolicy(ACTIONS, logits=np.zeros(len(ACTIONS)))
    action_map  = {a: _make_mock_action(a) for a in ACTIONS}

    with patch.object(tg, "ACTION_MAP", action_map), \
         patch.object(tg, "ACTION_NAMES", ACTIONS):
        chosen, new_state = tg._apply_with_fallback("action_a", state, policy)

    assert chosen == "action_a"
    action_map["action_a"].apply.assert_called_once_with(state)
    print("PASS test_apply_with_fallback_succeeds_first_try")


def test_apply_with_fallback_falls_back_when_chosen_raises():
    state  = _make_state()
    policy = ActionPolicy(ACTIONS, logits=np.array([5.0, -1.0, -1.0, -1.0]))
    # action_a raises, action_b succeeds
    action_map = {
        "action_a": _make_mock_action("action_a", raises=True),
        "action_b": _make_mock_action("action_b"),
        "action_c": _make_mock_action("action_c"),
        "action_d": _make_mock_action("action_d"),
    }

    with patch.object(tg, "ACTION_MAP", action_map), \
         patch.object(tg, "ACTION_NAMES", ACTIONS):
        chosen, new_state = tg._apply_with_fallback("action_a", state, policy)

    assert chosen != "action_a"
    action_map["action_a"].apply.assert_called_once()
    print("PASS test_apply_with_fallback_falls_back_when_chosen_raises")


def test_apply_with_fallback_raises_when_all_fail():
    state  = _make_state()
    policy = ActionPolicy(ACTIONS)
    action_map = {a: _make_mock_action(a, raises=True) for a in ACTIONS}

    with patch.object(tg, "ACTION_MAP", action_map), \
         patch.object(tg, "ACTION_NAMES", ACTIONS):
        try:
            tg._apply_with_fallback("action_a", state, policy)
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass
    print("PASS test_apply_with_fallback_raises_when_all_fail")


# ── _run_rollout_step ─────────────────────────────────────────────────────────

def test_run_rollout_step_returns_non_skipped_rewards():
    state   = _make_state()
    rewards = {"action_a": 0.75, "action_b": 0.80, "action_c": float("nan")}

    mock_results = _mock_rollout_results(rewards)

    a1, a2, a3 = MagicMock(), MagicMock(), MagicMock()
    a1.name, a2.name, a3.name = "action_a", "action_b", "action_c"
    mock_rollout = MagicMock()
    mock_rollout.starmap.return_value = mock_results

    with patch.object(tg, "rollout_one", mock_rollout), \
         patch.object(tg, "ALL_ACTIONS", [a1, a2, a3]):
        result = tg._run_rollout_step("test_task", "query", state, 0.70)

    assert "action_a" in result and abs(result["action_a"] - 0.75) < 1e-9
    assert "action_b" in result and abs(result["action_b"] - 0.80) < 1e-9
    assert "action_c" not in result   # skipped
    print("PASS test_run_rollout_step_returns_non_skipped_rewards")


def test_run_rollout_step_dispatches_all_actions():
    state   = _make_state()
    actions = [MagicMock(), MagicMock()]
    actions[0].name, actions[1].name = "action_a", "action_b"
    mock_rollout = MagicMock()
    mock_rollout.starmap.return_value = [
        {"action": "action_a", "new_test_auroc": 0.75, "skipped": False},
        {"action": "action_b", "new_test_auroc": 0.70, "skipped": False},
    ]

    with patch.object(tg, "rollout_one", mock_rollout), \
         patch.object(tg, "ALL_ACTIONS", actions):
        tg._run_rollout_step("test_task", "query", state, 0.70)

    call_args = mock_rollout.starmap.call_args[0][0]
    dispatched_actions = [item[1] for item in call_args]
    assert dispatched_actions == ["action_a", "action_b"]
    print("PASS test_run_rollout_step_dispatches_all_actions")


# ── _run_one_task (chain mode) ────────────────────────────────────────────────

def _setup_task_dirs(tmp: Path, task: str, rewards_0: dict[str, float],
                     base_auroc: float = 0.70) -> None:
    """Write the minimal on-disk fixtures _run_one_task needs."""
    # iter-0 trajectories
    for action, auroc in rewards_0.items():
        _write_trajectory(tmp / "traj", task, action, auroc)

    # rubric + rubricified records
    rubric_dir      = tmp / "data" / "rubric" / task
    rubricified_dir = tmp / "data" / "rubric" / "rubricified" / task
    rubric_dir.mkdir(parents=True)
    rubricified_dir.mkdir(parents=True)
    rubric = {
        "task": task, "rubric_instructions": "*   **AGE:** [age]\n",
        "task_query": "test?", "num_examples": 0, "usage": {},
    }
    rubric_dir.joinpath("rubric.json").write_text(json.dumps(rubric))
    for split in ("train", "val", "test"):
        recs = [_make_record(i, split, i % 2 == 0) for i in range(4)]
        rubricified_dir.joinpath(f"{split}.json").write_text(json.dumps(recs))

    # base metrics
    metrics_dir = tmp / "data" / "results" / "global-rubric" / task
    metrics_dir.mkdir(parents=True)
    metrics_dir.joinpath("metrics.json").write_text(json.dumps({"test_auroc": base_auroc}))


def test_run_one_task_chain_log_structure():
    """Chain mode: log history has one entry per step + step 0."""
    task   = "test_task"
    actions = ["action_a", "action_b", "action_c"]
    rewards_0 = {"action_a": 0.71, "action_b": 0.75, "action_c": 0.68}
    step1_rewards = {"action_a": 0.72, "action_b": 0.76, "action_c": 0.69}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _setup_task_dirs(tmp, task, rewards_0)
        policy_dir = tmp / "policies"
        log_dir    = tmp / "logs"

        mock_actions = [MagicMock(), MagicMock(), MagicMock()]
        for ma, name in zip(mock_actions, actions):
            ma.name = name
            new_state = _make_state()
            ma.apply.return_value = new_state

        with patch.object(tg, "TRAJ_DIR",    tmp / "traj"), \
             patch.object(tg, "REPO_ROOT",   tmp), \
             patch.object(tg, "ALL_ACTIONS", mock_actions), \
             patch.object(tg, "ACTION_MAP",  {a: ma for a, ma in zip(actions, mock_actions)}), \
             patch.object(tg, "ACTION_NAMES", actions), \
             patch.object(tg, "_run_rollout_step", return_value=step1_rewards):

            log = tg._run_one_task(
                task=task, task_query="test?", steps=2,
                lr=0.1, beta=0.0, greedy=True, seed=42,
                traj_idx=0, n_trajectories=1,
                best_state_rollout=False,
            )

    assert log["mode"] == "chain"
    assert len(log["history"]) == 3          # step 0 + 2 RL steps
    assert log["history"][0]["step"] == 0
    assert log["history"][1]["step"] == 1
    assert log["history"][2]["step"] == 2
    assert all("policy" in e for e in log["history"])
    assert all("rewards" in e for e in log["history"])
    print("PASS test_run_one_task_chain_log_structure")


def test_run_one_task_chain_greedy_selects_best():
    """With greedy=True the policy always picks the highest-prob action."""
    task   = "test_task"
    actions = ["action_a", "action_b", "action_c"]
    # action_b has highest reward → highest advantage → selected at every step
    rewards_0 = {"action_a": 0.68, "action_b": 0.80, "action_c": 0.69}
    step_rewards = rewards_0.copy()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _setup_task_dirs(tmp, task, rewards_0)

        mock_actions = [MagicMock(), MagicMock(), MagicMock()]
        for ma, name in zip(mock_actions, actions):
            ma.name = name
            ma.apply.return_value = _make_state()

        with patch.object(tg, "TRAJ_DIR",    tmp / "traj"), \
             patch.object(tg, "REPO_ROOT",   tmp), \
             patch.object(tg, "ALL_ACTIONS", mock_actions), \
             patch.object(tg, "ACTION_MAP",  {a: ma for a, ma in zip(actions, mock_actions)}), \
             patch.object(tg, "ACTION_NAMES", actions), \
             patch.object(tg, "_run_rollout_step", return_value=step_rewards):

            log = tg._run_one_task(
                task=task, task_query="test?", steps=3,
                lr=0.1, beta=0.0, greedy=True, seed=42,
                traj_idx=0, n_trajectories=1,
                best_state_rollout=False,
            )

    # All RL steps should choose action_b (greedy + highest reward)
    for entry in log["history"][1:]:
        assert entry["action_selected"] == "action_b"
    print("PASS test_run_one_task_chain_greedy_selects_best")


# ── _run_one_task (best-state mode) ──────────────────────────────────────────

def test_run_one_task_best_state_log_has_extra_fields():
    """best_state mode: history entries include best_action_step, best_state_advanced."""
    task   = "test_task"
    actions = ["action_a", "action_b", "action_c"]
    rewards_0 = {"action_a": 0.71, "action_b": 0.75, "action_c": 0.68}
    # Step 1 improves over best (0.75) → best_state_advanced = True
    step1_rewards = {"action_a": 0.72, "action_b": 0.78, "action_c": 0.69}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _setup_task_dirs(tmp, task, rewards_0)

        mock_actions = [MagicMock(), MagicMock(), MagicMock()]
        for ma, name in zip(mock_actions, actions):
            ma.name = name
            ma.apply.return_value = _make_state()

        with patch.object(tg, "TRAJ_DIR",    tmp / "traj"), \
             patch.object(tg, "REPO_ROOT",   tmp), \
             patch.object(tg, "ALL_ACTIONS", mock_actions), \
             patch.object(tg, "ACTION_MAP",  {a: ma for a, ma in zip(actions, mock_actions)}), \
             patch.object(tg, "ACTION_NAMES", actions), \
             patch.object(tg, "_run_rollout_step", return_value=step1_rewards):

            log = tg._run_one_task(
                task=task, task_query="test?", steps=1,
                lr=0.1, beta=0.0, greedy=True, seed=42,
                traj_idx=0, n_trajectories=1,
                best_state_rollout=True,
            )

    assert log["mode"] == "best_state"
    step1 = log["history"][1]
    assert "best_action_step" in step1
    assert "best_state_advanced" in step1
    print("PASS test_run_one_task_best_state_log_has_extra_fields")


def test_run_one_task_best_state_advances_on_improvement():
    """best_state: best_state_advanced=True when step rewards beat best_auroc."""
    task   = "test_task"
    actions = ["action_a", "action_b"]
    # iter-0 best is action_b=0.75; step 1 has action_b=0.80 (beats 0.75)
    rewards_0 = {"action_a": 0.70, "action_b": 0.75}
    step1_rewards = {"action_a": 0.74, "action_b": 0.80}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _setup_task_dirs(tmp, task, rewards_0)

        mock_actions = [MagicMock(), MagicMock()]
        for ma, name in zip(mock_actions, actions):
            ma.name = name
            ma.apply.return_value = _make_state()

        with patch.object(tg, "TRAJ_DIR",    tmp / "traj"), \
             patch.object(tg, "REPO_ROOT",   tmp), \
             patch.object(tg, "ALL_ACTIONS", mock_actions), \
             patch.object(tg, "ACTION_MAP",  {a: ma for a, ma in zip(actions, mock_actions)}), \
             patch.object(tg, "ACTION_NAMES", actions), \
             patch.object(tg, "_run_rollout_step", return_value=step1_rewards):

            log = tg._run_one_task(
                task=task, task_query="test?", steps=1,
                lr=0.1, beta=0.0, greedy=True, seed=42,
                traj_idx=0, n_trajectories=1,
                best_state_rollout=True,
            )

    step1 = log["history"][1]
    assert step1["best_action_step"] == "action_b"
    assert step1["best_state_advanced"] is True
    assert abs(step1["state_auroc"] - 0.80) < 1e-9
    print("PASS test_run_one_task_best_state_advances_on_improvement")


def test_run_one_task_best_state_frozen_when_no_improvement():
    """best_state: best_state_advanced=False when step rewards are all below best_auroc."""
    task   = "test_task"
    actions = ["action_a", "action_b"]
    rewards_0 = {"action_a": 0.70, "action_b": 0.80}
    # Step 1 rewards are all worse than 0.80 (the base auroc)
    step1_rewards = {"action_a": 0.72, "action_b": 0.78}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # base_auroc=0.80 so best_auroc starts at 0.80; step1 max=0.78 < 0.80 → frozen
        _setup_task_dirs(tmp, task, rewards_0, base_auroc=0.80)

        mock_actions = [MagicMock(), MagicMock()]
        for ma, name in zip(mock_actions, actions):
            ma.name = name
            ma.apply.return_value = _make_state()

        with patch.object(tg, "TRAJ_DIR",    tmp / "traj"), \
             patch.object(tg, "REPO_ROOT",   tmp), \
             patch.object(tg, "ALL_ACTIONS", mock_actions), \
             patch.object(tg, "ACTION_MAP",  {a: ma for a, ma in zip(actions, mock_actions)}), \
             patch.object(tg, "ACTION_NAMES", actions), \
             patch.object(tg, "_run_rollout_step", return_value=step1_rewards):

            log = tg._run_one_task(
                task=task, task_query="test?", steps=1,
                lr=0.1, beta=0.0, greedy=True, seed=42,
                traj_idx=0, n_trajectories=1,
                best_state_rollout=True,
            )

    step1 = log["history"][1]
    assert step1["best_state_advanced"] is False
    # state_auroc should still be the previous best (0.80), not 0.78
    assert abs(step1["state_auroc"] - 0.80) < 1e-9
    print("PASS test_run_one_task_best_state_frozen_when_no_improvement")


def test_run_one_task_best_state_policy_updated_even_when_frozen():
    """Policy logits must shift even when the best_state doesn't advance."""
    task   = "test_task"
    actions = ["action_a", "action_b"]
    rewards_0 = {"action_a": 0.70, "action_b": 0.80}
    step1_rewards = {"action_a": 0.72, "action_b": 0.78}   # below base_auroc=0.80

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _setup_task_dirs(tmp, task, rewards_0, base_auroc=0.80)

        mock_actions = [MagicMock(), MagicMock()]
        for ma, name in zip(mock_actions, actions):
            ma.name = name
            ma.apply.return_value = _make_state()

        with patch.object(tg, "TRAJ_DIR",    tmp / "traj"), \
             patch.object(tg, "REPO_ROOT",   tmp), \
             patch.object(tg, "ALL_ACTIONS", mock_actions), \
             patch.object(tg, "ACTION_MAP",  {a: ma for a, ma in zip(actions, mock_actions)}), \
             patch.object(tg, "ACTION_NAMES", actions), \
             patch.object(tg, "_run_rollout_step", return_value=step1_rewards):

            log = tg._run_one_task(
                task=task, task_query="test?", steps=1,
                lr=0.1, beta=0.0, greedy=True, seed=42,
                traj_idx=0, n_trajectories=1,
                best_state_rollout=True,
            )

    step0_logits = log["history"][0]["policy"]["logits"]
    step1_logits = log["history"][1]["policy"]["logits"]
    # Logits must have changed even though state didn't advance
    assert step0_logits != step1_logits
    print("PASS test_run_one_task_best_state_policy_updated_even_when_frozen")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_load_iter0_rewards_reads_all_valid()
    test_load_iter0_rewards_skips_skipped_actions()
    test_load_iter0_rewards_missing_file_not_included()
    test_load_base_auroc_reads_metrics()
    test_load_base_auroc_returns_nan_when_missing()
    test_apply_with_fallback_succeeds_first_try()
    test_apply_with_fallback_falls_back_when_chosen_raises()
    test_apply_with_fallback_raises_when_all_fail()
    test_run_rollout_step_returns_non_skipped_rewards()
    test_run_rollout_step_dispatches_all_actions()
    test_run_one_task_chain_log_structure()
    test_run_one_task_chain_greedy_selects_best()
    test_run_one_task_best_state_log_has_extra_fields()
    test_run_one_task_best_state_advances_on_improvement()
    test_run_one_task_best_state_frozen_when_no_improvement()
    test_run_one_task_best_state_policy_updated_even_when_frozen()
    print("\nAll train_grpo tests passed.")
