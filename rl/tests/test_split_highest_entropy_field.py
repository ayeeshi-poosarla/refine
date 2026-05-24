#!/usr/bin/env python3
"""Tests for split_highest_entropy_field action."""

import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from rl.actions.split_highest_entropy_field import (
    compute_bin_mapping,
    compute_entropy,
    collect_field_values,
    find_highest_entropy_field,
    replace_field_in_rubric,
    replace_field_in_text,
    SplitHighestEntropyField,
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


def make_state(records_by_split: dict[str, list[dict]], instructions: str) -> RubricState:
    rubric = {
        "task": "test_task",
        "rubric_instructions": instructions,
        "task_query": "test",
        "num_examples": 0,
        "usage": {},
    }
    return RubricState(task="test_task", rubric=rubric, records=records_by_split)


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_compute_entropy_uniform():
    vals = ["A", "B", "C", "D"]  # perfectly uniform → max entropy = log2(4) = 2.0
    e = compute_entropy(vals)
    assert abs(e - 2.0) < 1e-9, f"Expected 2.0, got {e}"
    print("PASS test_compute_entropy_uniform")


def test_compute_entropy_constant():
    vals = ["A", "A", "A"]
    e = compute_entropy(vals)
    assert e == 0.0, f"Expected 0.0, got {e}"
    print("PASS test_compute_entropy_constant")


def test_find_highest_entropy_field():
    field_vals = {
        "CONSTANT": ["X", "X", "X"],           # entropy = 0
        "BINARY":   ["Yes", "No", "Yes", "No"],  # entropy = 1.0
        "UNIFORM":  ["A", "B", "C", "D"],        # entropy = 2.0
    }
    result = find_highest_entropy_field(field_vals)
    assert result == "UNIFORM", f"Expected UNIFORM, got {result}"
    print("PASS test_find_highest_entropy_field")


def test_compute_bin_mapping_numeric():
    # values: 1, 2, 3, 4, 5 — median is 3 (index 2 of sorted list)
    values = ["1", "2", "3", "4", "5"]
    bmap = compute_bin_mapping(values)
    assert bmap["1"] == "LOW"
    assert bmap["2"] == "LOW"
    assert bmap["3"] == "LOW"
    assert bmap["4"] == "HIGH"
    assert bmap["5"] == "HIGH"
    print("PASS test_compute_bin_mapping_numeric")


def test_compute_bin_mapping_categorical():
    # 4 values sorted alphabetically: A, B, C, D → A,B=LOW, C,D=HIGH
    values = ["C", "A", "D", "B", "A", "C"]
    bmap = compute_bin_mapping(values)
    assert bmap["A"] == "LOW"
    assert bmap["B"] == "LOW"
    assert bmap["C"] == "HIGH"
    assert bmap["D"] == "HIGH"
    print("PASS test_compute_bin_mapping_categorical")


def test_compute_bin_mapping_single_value():
    values = ["Only"]
    bmap = compute_bin_mapping(values)
    assert set(bmap.values()) <= {"LOW", "HIGH"}
    print("PASS test_compute_bin_mapping_single_value")


def test_replace_field_in_rubric():
    instructions = (
        "*   **AGE:** [Patient age]\n"
        "*   **RISK:** [Risk level]\n"
        "*   **SEX:** [Male/Female]\n"
    )
    bin_map = {"Low": "LOW", "Medium": "LOW", "High": "HIGH"}
    result = replace_field_in_rubric(instructions, "RISK", "RISK_BIN", bin_map)
    assert "RISK_BIN" in result, "New field not in rubric"
    assert "**RISK:**" not in result, "Old field still present"
    assert "LOW" in result and "HIGH" in result
    assert "AGE" in result and "SEX" in result
    print("PASS test_replace_field_in_rubric")


def test_replace_field_in_text():
    text = (
        "*   **AGE:** 65\n"
        "*   **RISK:** High\n"
        "*   **SEX:** MALE\n"
    )
    result = replace_field_in_text(text, "RISK", "RISK_BIN", "HIGH")
    fields = dict(FIELD_RE.findall(result))
    assert "RISK_BIN" in fields, f"RISK_BIN missing; fields={fields}"
    assert fields["RISK_BIN"] == "HIGH"
    assert "RISK" not in fields or "RISK_BIN" in fields
    assert "AGE" in fields and "SEX" in fields
    print("PASS test_replace_field_in_text")


def test_apply_replaces_highest_entropy_field():
    records = {
        "train": [
            make_record(1, "2020-01-01", 1, {"CONSTANT": "X", "DIVERSE": "A"}),
            make_record(2, "2020-01-02", 0, {"CONSTANT": "X", "DIVERSE": "B"}),
            make_record(3, "2020-01-03", 1, {"CONSTANT": "X", "DIVERSE": "C"}),
            make_record(4, "2020-01-04", 0, {"CONSTANT": "X", "DIVERSE": "D"}),
        ],
        "test": [
            make_record(5, "2020-02-01", 1, {"CONSTANT": "X", "DIVERSE": "A"}),
            make_record(6, "2020-02-02", 0, {"CONSTANT": "X", "DIVERSE": "C"}),
        ],
    }
    instructions = (
        "*   **CONSTANT:** [Always X]\n"
        "*   **DIVERSE:** [Many values]\n"
    )
    state = make_state(records, instructions)
    new_state = SplitHighestEntropyField().apply(state)

    # DIVERSE has higher entropy — should be replaced by DIVERSE_BIN
    assert "DIVERSE_BIN" in new_state.rubric["rubric_instructions"]
    assert "**DIVERSE:**" not in new_state.rubric["rubric_instructions"]
    assert "CONSTANT" in new_state.rubric["rubric_instructions"]

    for recs in new_state.records.values():
        for r in recs:
            fields = dict(FIELD_RE.findall(r["rubricified_text"]))
            assert "DIVERSE_BIN" in fields, f"DIVERSE_BIN missing: {fields}"
            assert fields["DIVERSE_BIN"] in ("LOW", "HIGH")
            assert "CONSTANT" in fields

    meta = new_state.rubric["_last_action"]
    assert meta["source_field"] == "DIVERSE"
    assert meta["new_field"] == "DIVERSE_BIN"
    assert meta["entropy"] > 0
    print("PASS test_apply_replaces_highest_entropy_field")


def test_apply_does_not_mutate_input():
    records = {
        "train": [make_record(1, "2020-01-01", 1, {"FIELD_A": "X", "FIELD_B": "Y"})],
    }
    state = make_state(records, "*   **FIELD_A:** [A]\n*   **FIELD_B:** [B]\n")
    original_text = state.records["train"][0]["rubricified_text"]

    SplitHighestEntropyField().apply(state)

    assert state.records["train"][0]["rubricified_text"] == original_text
    print("PASS test_apply_does_not_mutate_input")


def test_end_to_end_via_subprocess():
    with tempfile.TemporaryDirectory() as rub_tmp, \
         tempfile.TemporaryDirectory() as rubricified_tmp:
        rub_tmp         = Path(rub_tmp)
        rubricified_tmp = Path(rubricified_tmp)
        task            = "test_task"

        records_train = [
            make_record(i, f"2020-01-{i+1:02d}", i % 2,
                        {"AGE": str(20 + i * 5), "RISK": ["Low", "Medium", "High", "Low", "High"][i]})
            for i in range(5)
        ]
        records_test = [
            make_record(i + 10, f"2020-02-{i+1:02d}", i % 2,
                        {"AGE": str(60 + i), "RISK": ["Low", "High", "Medium"][i]})
            for i in range(3)
        ]

        for split, recs in [("train", records_train), ("test", records_test)]:
            d = rubricified_tmp / task
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{split}.json").write_text(json.dumps(recs))

        rubric = {
            "task": task,
            "rubric_instructions": "*   **AGE:** [Age in years]\n*   **RISK:** [Low/Medium/High]\n",
            "task_query": "test", "num_examples": 0, "usage": {},
        }
        d = rub_tmp / task
        d.mkdir(parents=True, exist_ok=True)
        (d / "rubric.json").write_text(json.dumps(rubric))

        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        result = subprocess.run(
            [sys.executable,
             str(Path(__file__).resolve().parents[1] / "actions" / "split_highest_entropy_field.py"),
             "--task", task,
             "--rubric_dir", str(rub_tmp),
             "--rubricified_dir", str(rubricified_tmp)],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        assert "entropy" in result.stdout.lower()

        updated_rubric = json.loads((rub_tmp / task / "rubric.json").read_text())
        instr = updated_rubric["rubric_instructions"]
        assert "_BIN" in instr, "No _BIN field in updated rubric"

        for split in ("train", "test"):
            updated = json.loads((rubricified_tmp / task / f"{split}.json").read_text())
            for r in updated:
                fields = dict(FIELD_RE.findall(r["rubricified_text"]))
                assert any("_BIN" in k for k in fields), f"No _BIN field in record: {fields}"
                bin_vals = [v for k, v in fields.items() if "_BIN" in k]
                assert all(v in ("LOW", "HIGH") for v in bin_vals), f"Unexpected bin value: {bin_vals}"

    print("PASS test_end_to_end_via_subprocess")


if __name__ == "__main__":
    test_compute_entropy_uniform()
    test_compute_entropy_constant()
    test_find_highest_entropy_field()
    test_compute_bin_mapping_numeric()
    test_compute_bin_mapping_categorical()
    test_compute_bin_mapping_single_value()
    test_replace_field_in_rubric()
    test_replace_field_in_text()
    test_apply_replaces_highest_entropy_field()
    test_apply_does_not_mutate_input()
    test_end_to_end_via_subprocess()
    print("\nAll tests passed.")
