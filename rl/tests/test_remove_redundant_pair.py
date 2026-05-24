#!/usr/bin/env python3
"""Tests for remove_redundant_pair action."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from rl.actions.remove_redundant_pair import (
    compute_individual_aurocs,
    compute_pairwise_correlations,
    find_most_correlated_pair,
    parse_records,
    remove_field_from_rubric,
    remove_field_from_text,
    RemoveRedundantPair,
)
from rl.state import RubricState

FIELD_RE = re.compile(r'\*\*([A-Z_]+):\*\*\s*(.+)')


def make_record(patient_id, prediction_time, label, fields: dict) -> dict:
    text = "\n".join(f"*   **{k}:** {v}" for k, v in fields.items())
    return {
        "patient_id": patient_id,
        "prediction_time": prediction_time,
        "label": label,
        "rubricified_text": text,
    }


def make_state(records_by_split: dict, instructions: str) -> RubricState:
    rubric = {
        "task": "test_task",
        "rubric_instructions": instructions,
        "task_query": "test",
        "num_examples": 0,
        "usage": {},
    }
    return RubricState(task="test_task", rubric=rubric, records=records_by_split)


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_compute_pairwise_correlations_perfect():
    # COPY is identical to ORIGINAL → |r| = 1.0
    records = {
        "train": [
            make_record(i, f"2020-01-{i+1:02d}", i % 2,
                        {"ORIGINAL": str(i), "COPY": str(i), "UNRELATED": str(i % 3)})
            for i in range(6)
        ]
    }
    parsed = parse_records(records)
    corrs  = compute_pairwise_correlations(parsed)
    assert ("COPY", "ORIGINAL") in corrs or ("ORIGINAL", "COPY") in corrs
    pair = ("COPY", "ORIGINAL") if ("COPY", "ORIGINAL") in corrs else ("ORIGINAL", "COPY")
    assert abs(corrs[pair] - 1.0) < 1e-6, f"Expected 1.0, got {corrs[pair]}"
    print("PASS test_compute_pairwise_correlations_perfect")


def test_compute_pairwise_correlations_independent():
    # Two independent fields should have low correlation
    rng = np.random.default_rng(0)
    vals_a = rng.integers(0, 5, 50).tolist()
    vals_b = rng.integers(0, 5, 50).tolist()
    records = {
        "train": [
            make_record(i, f"2020-01-01", i % 2,
                        {"FIELD_A": str(vals_a[i]), "FIELD_B": str(vals_b[i])})
            for i in range(50)
        ]
    }
    parsed = parse_records(records)
    corrs  = compute_pairwise_correlations(parsed)
    pair   = ("FIELD_A", "FIELD_B")
    assert corrs[pair] < 0.5, f"Expected low correlation, got {corrs[pair]:.3f}"
    print("PASS test_compute_pairwise_correlations_independent")


def test_find_most_correlated_pair():
    corrs = {
        ("A", "B"): 0.9,
        ("A", "C"): 0.3,
        ("B", "C"): 0.6,
    }
    pair = find_most_correlated_pair(corrs)
    assert pair == ("A", "B"), f"Expected ('A','B'), got {pair}"
    print("PASS test_find_most_correlated_pair")


def test_compute_individual_aurocs_removes_lower():
    # GOOD predicts label perfectly; BAD is random noise
    n = 40
    train_records = [
        make_record(i, f"2020-01-{i+1:02d}", i % 2,
                    {"GOOD": "Pos" if i % 2 else "Neg",
                     "BAD":  "X"   if i % 3 else "Y"})
        for i in range(n)
    ]
    test_records = [
        make_record(i + n, f"2020-02-{i+1:02d}", i % 2,
                    {"GOOD": "Pos" if i % 2 else "Neg",
                     "BAD":  "X"   if i % 3 else "Y"})
        for i in range(10)
    ]
    parsed = parse_records({"train": train_records, "test": test_records})
    aurocs = compute_individual_aurocs(parsed, ("GOOD", "BAD"))
    assert aurocs["GOOD"] > aurocs["BAD"], (
        f"GOOD AUROC ({aurocs['GOOD']:.3f}) should exceed BAD ({aurocs['BAD']:.3f})"
    )
    print("PASS test_compute_individual_aurocs_removes_lower")


def test_remove_field_from_rubric():
    instructions = (
        "*   **KEEP:** [This stays]\n"
        "*   **REMOVE:** [This goes]\n"
        "*   **ALSO_KEEP:** [This stays too]\n"
    )
    result = remove_field_from_rubric(instructions, "REMOVE")
    assert "REMOVE" not in result
    assert "KEEP" in result
    assert "ALSO_KEEP" in result
    print("PASS test_remove_field_from_rubric")


def test_remove_field_from_text():
    text = (
        "*   **KEEP:** present\n"
        "*   **REMOVE:** gone\n"
        "*   **ALSO_KEEP:** here\n"
    )
    result = remove_field_from_text(text, "REMOVE")
    fields = dict(FIELD_RE.findall(result))
    assert "REMOVE" not in fields
    assert "KEEP" in fields
    assert "ALSO_KEEP" in fields
    print("PASS test_remove_field_from_text")


def test_apply_removes_lower_auroc_from_correlated_pair():
    # SHADOW mirrors SIGNAL exactly (high correlation); SIGNAL predicts label
    n_train, n_test = 30, 10
    def make_rows(start, count, split):
        return [
            make_record(
                start + i, f"2020-0{1 + (split=='test')}-{i+1:02d}", i % 2,
                {
                    "SIGNAL": "Pos" if i % 2 else "Neg",
                    "SHADOW": "Pos" if i % 2 else "Neg",  # identical to SIGNAL
                    "NOISE":  str(i % 7),
                }
            )
            for i in range(count)
        ]

    state = make_state(
        {"train": make_rows(0, n_train, "train"), "test": make_rows(n_train, n_test, "test")},
        "*   **SIGNAL:** [The real signal]\n*   **SHADOW:** [Mirror]\n*   **NOISE:** [Random]\n",
    )
    new_state = RemoveRedundantPair().apply(state)

    meta = new_state.rubric["_last_action"]
    # The most correlated pair must be (SIGNAL, SHADOW)
    assert set(meta["pair"]) == {"SIGNAL", "SHADOW"}, f"Unexpected pair: {meta['pair']}"
    # One of them is removed; since AUROC is equal (they're identical), either is fine
    assert meta["removed"] in ("SIGNAL", "SHADOW")
    assert meta["kept"]    in ("SIGNAL", "SHADOW")
    assert meta["removed"] != meta["kept"]

    # Removed field must be gone from rubric and all records
    assert meta["removed"] not in new_state.rubric["rubric_instructions"]
    for recs in new_state.records.values():
        for r in recs:
            fields = dict(FIELD_RE.findall(r["rubricified_text"]))
            assert meta["removed"] not in fields
            assert meta["kept"]    in fields
    print("PASS test_apply_removes_lower_auroc_from_correlated_pair")


def test_apply_does_not_mutate_input():
    records = {
        "train": [make_record(i, f"2020-01-{i+1:02d}", i % 2,
                              {"FIELD_A": str(i), "FIELD_B": str(i)})
                  for i in range(6)],
        "test":  [make_record(i + 10, f"2020-02-{i+1:02d}", i % 2,
                              {"FIELD_A": str(i), "FIELD_B": str(i)})
                  for i in range(4)],
    }
    state = make_state(records, "*   **FIELD_A:** [A]\n*   **FIELD_B:** [B]\n")
    original_instr = state.rubric["rubric_instructions"]
    original_texts = [r["rubricified_text"] for r in state.records["train"]]

    RemoveRedundantPair().apply(state)

    assert state.rubric["rubric_instructions"] == original_instr
    for orig, r in zip(original_texts, state.records["train"]):
        assert r["rubricified_text"] == orig
    print("PASS test_apply_does_not_mutate_input")


def test_end_to_end_via_subprocess():
    with tempfile.TemporaryDirectory() as rub_tmp, \
         tempfile.TemporaryDirectory() as rubricified_tmp:
        rub_tmp         = Path(rub_tmp)
        rubricified_tmp = Path(rubricified_tmp)
        task            = "test_task"

        train_recs = [
            make_record(i, f"2020-01-{i+1:02d}", i % 2,
                        {"ALPHA": "High" if i % 2 else "Low",
                         "BETA":  "High" if i % 2 else "Low",   # mirrors ALPHA
                         "GAMMA": str(i % 5)})
            for i in range(20)
        ]
        test_recs = [
            make_record(i + 20, f"2020-02-{i+1:02d}", i % 2,
                        {"ALPHA": "High" if i % 2 else "Low",
                         "BETA":  "High" if i % 2 else "Low",
                         "GAMMA": str(i % 5)})
            for i in range(10)
        ]
        for split, recs in [("train", train_recs), ("test", test_recs)]:
            d = rubricified_tmp / task
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{split}.json").write_text(json.dumps(recs))

        rubric = {
            "task": task,
            "rubric_instructions": (
                "*   **ALPHA:** [High/Low]\n"
                "*   **BETA:** [High/Low]\n"
                "*   **GAMMA:** [0-4]\n"
            ),
            "task_query": "test", "num_examples": 0, "usage": {},
        }
        d = rub_tmp / task
        d.mkdir(parents=True, exist_ok=True)
        (d / "rubric.json").write_text(json.dumps(rubric))

        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        result = subprocess.run(
            [sys.executable,
             str(Path(__file__).resolve().parents[1] / "actions" / "remove_redundant_pair.py"),
             "--task", task,
             "--rubric_dir", str(rub_tmp),
             "--rubricified_dir", str(rubricified_tmp)],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        assert "|r|=" in result.stdout

        updated_rubric = json.loads((rub_tmp / task / "rubric.json").read_text())
        instr = updated_rubric["rubric_instructions"]
        # One of ALPHA/BETA should be gone, GAMMA should remain
        removed_count = sum(1 for f in ("ALPHA", "BETA") if f"**{f}:**" not in instr)
        assert removed_count == 1, f"Expected exactly one of ALPHA/BETA removed; got: {instr}"
        assert "GAMMA" in instr

        for split in ("train", "test"):
            updated = json.loads((rubricified_tmp / task / f"{split}.json").read_text())
            for r in updated:
                fields = dict(FIELD_RE.findall(r["rubricified_text"]))
                present = [f for f in ("ALPHA", "BETA") if f in fields]
                assert len(present) == 1, f"Expected exactly one of ALPHA/BETA; got {fields}"
                assert "GAMMA" in fields

    print("PASS test_end_to_end_via_subprocess")


if __name__ == "__main__":
    test_compute_pairwise_correlations_perfect()
    test_compute_pairwise_correlations_independent()
    test_find_most_correlated_pair()
    test_compute_individual_aurocs_removes_lower()
    test_remove_field_from_rubric()
    test_remove_field_from_text()
    test_apply_removes_lower_auroc_from_correlated_pair()
    test_apply_does_not_mutate_input()
    test_end_to_end_via_subprocess()
    print("\nAll tests passed.")
