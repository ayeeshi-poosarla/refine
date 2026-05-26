#!/usr/bin/env python3
"""Tests for rl/policy.py — ActionPolicy (GRPO discrete policy)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl.policy import ActionPolicy

ACTIONS = ["action_a", "action_b", "action_c", "action_d"]


# ── probabilities() ───────────────────────────────────────────────────────────

def test_uniform_init_gives_equal_probs():
    policy = ActionPolicy(ACTIONS)
    probs = policy.probabilities()
    assert len(probs) == len(ACTIONS)
    assert np.allclose(probs, 1 / len(ACTIONS), atol=1e-9)
    print("PASS test_uniform_init_gives_equal_probs")


def test_probabilities_sum_to_one():
    policy = ActionPolicy(ACTIONS, logits=np.array([1.0, -2.0, 0.5, 3.0]))
    assert abs(policy.probabilities().sum() - 1.0) < 1e-9
    print("PASS test_probabilities_sum_to_one")


def test_probabilities_largest_logit_has_largest_prob():
    policy = ActionPolicy(ACTIONS, logits=np.array([0.0, 0.0, 0.0, 5.0]))
    probs = policy.probabilities()
    assert np.argmax(probs) == 3
    print("PASS test_probabilities_largest_logit_has_largest_prob")


def test_probabilities_stable_with_large_logits():
    policy = ActionPolicy(ACTIONS, logits=np.array([1000.0, 999.0, 998.0, 997.0]))
    probs = policy.probabilities()
    assert np.all(np.isfinite(probs))
    assert abs(probs.sum() - 1.0) < 1e-9
    print("PASS test_probabilities_stable_with_large_logits")


# ── grpo_update() ─────────────────────────────────────────────────────────────

def test_grpo_update_returns_advantages():
    policy = ActionPolicy(ACTIONS)
    rewards = {a: float(i) for i, a in enumerate(ACTIONS)}
    adv = policy.grpo_update(rewards, lr=0.1)
    assert set(adv.keys()) == set(ACTIONS)
    assert all(isinstance(v, float) for v in adv.values())
    print("PASS test_grpo_update_returns_advantages")


def test_grpo_update_advantages_mean_zero():
    policy = ActionPolicy(ACTIONS)
    rewards = {"action_a": 0.7, "action_b": 0.8, "action_c": 0.6, "action_d": 0.75}
    adv = policy.grpo_update(rewards, lr=0.1)
    adv_values = list(adv.values())
    assert abs(np.mean(adv_values)) < 1e-9
    print("PASS test_grpo_update_advantages_mean_zero")


def test_grpo_update_advantages_std_one():
    policy = ActionPolicy(ACTIONS)
    rewards = {"action_a": 0.7, "action_b": 0.8, "action_c": 0.6, "action_d": 0.75}
    adv = policy.grpo_update(rewards, lr=0.1)
    adv_values = list(adv.values())
    assert abs(np.std(adv_values) - 1.0) < 1e-6
    print("PASS test_grpo_update_advantages_std_one")


def test_grpo_update_shifts_logits_in_advantage_direction():
    policy = ActionPolicy(ACTIONS)
    # Best reward → positive advantage → logit increases
    rewards = {"action_a": 0.60, "action_b": 0.90, "action_c": 0.65, "action_d": 0.70}
    before = policy.logits.copy()
    adv = policy.grpo_update(rewards, lr=0.1)
    # action_b has highest reward → advantage > 0 → logit increased
    assert policy.logits[1] > before[1]
    # action_a has lowest reward → advantage < 0 → logit decreased
    assert policy.logits[0] < before[0]
    print("PASS test_grpo_update_shifts_logits_in_advantage_direction")


def test_grpo_update_missing_action_gets_zero_advantage():
    policy = ActionPolicy(ACTIONS)
    # action_d is not in rewards (e.g. skipped)
    rewards = {"action_a": 0.7, "action_b": 0.8, "action_c": 0.6}
    before_d = policy.logits[3]
    adv = policy.grpo_update(rewards, lr=0.1)
    assert adv["action_d"] == 0.0
    assert policy.logits[3] == before_d   # no logit change for missing action
    print("PASS test_grpo_update_missing_action_gets_zero_advantage")


def test_grpo_update_nan_reward_excluded_from_stats():
    policy = ActionPolicy(ACTIONS)
    rewards = {"action_a": 0.7, "action_b": float("nan"), "action_c": 0.8, "action_d": 0.75}
    adv = policy.grpo_update(rewards, lr=0.1)
    # NaN action gets zero advantage, others get non-zero
    assert adv["action_b"] == 0.0
    assert adv["action_a"] != 0.0
    print("PASS test_grpo_update_nan_reward_excluded_from_stats")


def test_grpo_update_fewer_than_two_valid_is_noop():
    policy = ActionPolicy(ACTIONS)
    before = policy.logits.copy()
    rewards = {"action_a": 0.7}   # only one valid reward
    adv = policy.grpo_update(rewards, lr=0.1)
    assert np.all(policy.logits == before)
    assert all(v == 0.0 for v in adv.values())
    print("PASS test_grpo_update_fewer_than_two_valid_is_noop")


def test_grpo_update_lr_scales_logit_change():
    rewards = {"action_a": 0.6, "action_b": 0.9, "action_c": 0.65, "action_d": 0.70}

    pol_small = ActionPolicy(ACTIONS)
    pol_large = ActionPolicy(ACTIONS)
    pol_small.grpo_update(rewards, lr=0.01)
    pol_large.grpo_update(rewards, lr=1.0)

    # Higher lr → larger shift in best and worst actions
    delta_small = abs(pol_small.logits[1] - 0.0)
    delta_large = abs(pol_large.logits[1] - 0.0)
    assert delta_large > delta_small
    print("PASS test_grpo_update_lr_scales_logit_change")


# ── grpo_update() with beta ───────────────────────────────────────────────────

def test_beta_zero_same_as_no_beta():
    rewards = {"action_a": 0.6, "action_b": 0.9, "action_c": 0.65, "action_d": 0.70}
    pol_no_beta = ActionPolicy(ACTIONS)
    pol_beta0   = ActionPolicy(ACTIONS)
    pol_no_beta.grpo_update(rewards, lr=0.1)
    pol_beta0.grpo_update(rewards, lr=0.1, beta=0.0)
    assert np.allclose(pol_no_beta.logits, pol_beta0.logits)
    print("PASS test_beta_zero_same_as_no_beta")


def test_beta_reduces_prob_spread_after_multiple_updates():
    # beta only kicks in after the policy has shifted from uniform;
    # after two updates the spread should be smaller with beta > 0
    rewards = {"action_a": 0.60, "action_b": 0.90, "action_c": 0.65, "action_d": 0.70}
    pol0 = ActionPolicy(ACTIONS)
    polb = ActionPolicy(ACTIONS)
    for _ in range(2):
        pol0.grpo_update(rewards, lr=0.1, beta=0.0)
        polb.grpo_update(rewards, lr=0.1, beta=0.5)
    spread0 = max(pol0.probabilities()) - min(pol0.probabilities())
    spreadb = max(polb.probabilities()) - min(polb.probabilities())
    assert spreadb < spread0, (
        f"beta should reduce prob spread but got beta={spreadb:.4f} >= no-beta={spread0:.4f}"
    )
    print("PASS test_beta_reduces_prob_spread_after_multiple_updates")


def test_beta_returned_advantages_unaffected():
    # The returned dict reflects the group-normalised advantages (before KL subtraction)
    rewards = {"action_a": 0.60, "action_b": 0.90, "action_c": 0.65, "action_d": 0.70}
    pol0 = ActionPolicy(ACTIONS)
    polb = ActionPolicy(ACTIONS)
    adv0 = pol0.grpo_update(rewards, lr=0.1, beta=0.0)
    advb = polb.grpo_update(rewards, lr=0.1, beta=0.5)
    # Returned advantages should be the same raw values
    for a in ACTIONS:
        assert abs(adv0[a] - advb[a]) < 1e-9, f"Returned advantage differs for {a}"
    print("PASS test_beta_returned_advantages_unaffected")


# ── best_action() / sample() ──────────────────────────────────────────────────

def test_best_action_returns_argmax():
    policy = ActionPolicy(ACTIONS, logits=np.array([0.1, 0.5, 0.2, 0.3]))
    assert policy.best_action() == "action_b"
    print("PASS test_best_action_returns_argmax")


def test_best_action_consistent_with_probabilities():
    policy = ActionPolicy(ACTIONS, logits=np.array([1.0, 3.0, 0.5, 2.0]))
    best = policy.best_action()
    probs = policy.probabilities()
    assert best == ACTIONS[np.argmax(probs)]
    print("PASS test_best_action_consistent_with_probabilities")


def test_sample_returns_valid_action():
    policy = ActionPolicy(ACTIONS)
    rng = np.random.RandomState(0)
    for _ in range(20):
        assert policy.sample(rng) in ACTIONS
    print("PASS test_sample_returns_valid_action")


def test_sample_distribution_approximately_matches_probs():
    # Skewed policy: action_b should be sampled most often
    policy = ActionPolicy(ACTIONS, logits=np.array([0.0, 5.0, 0.0, 0.0]))
    rng = np.random.RandomState(42)
    counts = {a: 0 for a in ACTIONS}
    N = 2000
    for _ in range(N):
        counts[policy.sample(rng)] += 1
    # action_b should have >> 50% of samples given logit=5
    assert counts["action_b"] / N > 0.8
    print("PASS test_sample_distribution_approximately_matches_probs")


def test_sample_without_rng_does_not_crash():
    policy = ActionPolicy(ACTIONS)
    result = policy.sample()   # no rng arg
    assert result in ACTIONS
    print("PASS test_sample_without_rng_does_not_crash")


# ── save() / load() ───────────────────────────────────────────────────────────

def test_save_load_roundtrip_logits():
    logits = np.array([0.3, -1.2, 0.8, 0.0])
    policy = ActionPolicy(ACTIONS, logits=logits)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        policy.save(path)
        loaded = ActionPolicy.load(path)
        assert loaded.action_names == ACTIONS
        assert np.allclose(loaded.logits, logits)
    finally:
        path.unlink(missing_ok=True)
    print("PASS test_save_load_roundtrip_logits")


def test_save_creates_parent_directories():
    policy = ActionPolicy(ACTIONS)
    with tempfile.TemporaryDirectory() as tmp:
        nested = Path(tmp) / "deep" / "nested" / "policy.json"
        policy.save(nested)
        assert nested.exists()
    print("PASS test_save_creates_parent_directories")


def test_load_json_contains_probabilities():
    policy = ActionPolicy(ACTIONS, logits=np.array([1.0, 2.0, 0.5, 1.5]))
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)
    try:
        policy.save(path)
        data = json.loads(path.read_text())
        assert "action_names" in data
        assert "logits" in data
        assert "probabilities" in data
        assert len(data["probabilities"]) == len(ACTIONS)
        assert abs(sum(data["probabilities"]) - 1.0) < 1e-9
    finally:
        path.unlink(missing_ok=True)
    print("PASS test_load_json_contains_probabilities")


def test_repr_contains_action_names():
    policy = ActionPolicy(["foo", "bar"])
    r = repr(policy)
    assert "foo" in r and "bar" in r
    print("PASS test_repr_contains_action_names")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_uniform_init_gives_equal_probs()
    test_probabilities_sum_to_one()
    test_probabilities_largest_logit_has_largest_prob()
    test_probabilities_stable_with_large_logits()
    test_grpo_update_returns_advantages()
    test_grpo_update_advantages_mean_zero()
    test_grpo_update_advantages_std_one()
    test_grpo_update_shifts_logits_in_advantage_direction()
    test_grpo_update_missing_action_gets_zero_advantage()
    test_grpo_update_nan_reward_excluded_from_stats()
    test_grpo_update_fewer_than_two_valid_is_noop()
    test_grpo_update_lr_scales_logit_change()
    test_beta_zero_same_as_no_beta()
    test_beta_reduces_prob_spread_after_multiple_updates()
    test_beta_returned_advantages_unaffected()
    test_best_action_returns_argmax()
    test_best_action_consistent_with_probabilities()
    test_sample_returns_valid_action()
    test_sample_distribution_approximately_matches_probs()
    test_sample_without_rng_does_not_crash()
    test_save_load_roundtrip_logits()
    test_save_creates_parent_directories()
    test_load_json_contains_probabilities()
    test_repr_contains_action_names()
    print("\nAll policy tests passed.")
