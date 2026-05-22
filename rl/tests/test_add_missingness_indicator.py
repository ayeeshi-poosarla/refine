#!/usr/bin/env python3
"""Tests for add_missingness_indicator action."""

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from actions.add_missingness_indicator import (
    add_field_to_rubric,
    add_field_to_text,
    compute_missing_rates,
    find_highest_missing_field,
)

FIELD_RE = re.compile(r'\*\*([A-Z_]+):\*\*\s*(.+)')


def make_record(patient_id, prediction_time, label, fields: dict) -> dict:
    text = "\n".join(f"*   **{k}:** {v}" for k, v in fields.items())
    return {
        "patient_id": patient_id,
        "prediction_time": prediction_time,
        "label": bool(label),
        "rubricified_text": text,
    }


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_compute_missing_rates():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        records = [
            make_record(1, "2020-01-01", True,  {"LAB": "N/A",    "AGE": "65"}),
            make_record(2, "2020-01-02", False, {"LAB": "N/A",    "AGE": "45"}),
            make_record(3, "2020-01-03", True,  {"LAB": "3.5 Normal", "AGE": "70"}),
        ]
        d = tmp / "test_task"
        d.mkdir()
        with open(d / "train.json", "w") as f:
            json.dump(records, f)

        rates = compute_missing_rates(tmp, "test_task")
        assert rates["LAB"] == (2, 3), f"Expected (2,3), got {rates['LAB']}"
        assert rates["AGE"] == (0, 3), f"Expected (0,3), got {rates['AGE']}"
    print("PASS test_compute_missing_rates")


def test_find_highest_missing_field():
    rates = {"AGE": (1, 100), "LAB": (95, 100), "SEX": (10, 100)}
    result = find_highest_missing_field(rates)
    assert result == "LAB", f"Expected LAB, got {result}"
    print("PASS test_find_highest_missing_field")


def test_add_field_to_rubric_inserts_after_source():
    instructions = (
        "*   **LAB:** [latest lab value]\n"
        "*   **AGE:** [patient age]\n"
    )
    result = add_field_to_rubric(instructions, "LAB", "LAB_IS_MISSING")
    lines = result.strip().split("\n")
    lab_idx = next(i for i, l in enumerate(lines) if "**LAB:**" in l and "IS_MISSING" not in l)
    missing_idx = next(i for i, l in enumerate(lines) if "**LAB_IS_MISSING:**" in l)
    assert missing_idx == lab_idx + 1
    assert "AGE" in result
    print("PASS test_add_field_to_rubric_inserts_after_source")


def test_add_field_to_text_missing_value_gets_1():
    text = "*   **LAB:** N/A\n*   **AGE:** 65\n"
    result = add_field_to_text(text, "LAB", "LAB_IS_MISSING")
    fields = dict(FIELD_RE.findall(result))
    assert fields["LAB_IS_MISSING"] == "1", f"Expected 1, got {fields['LAB_IS_MISSING']}"
    print("PASS test_add_field_to_text_missing_value_gets_1")


def test_add_field_to_text_none_value_gets_1():
    text = "*   **LAB:** None\n"
    result = add_field_to_text(text, "LAB", "LAB_IS_MISSING")
    fields = dict(FIELD_RE.findall(result))
    assert fields["LAB_IS_MISSING"] == "1"
    print("PASS test_add_field_to_text_none_value_gets_1")


def test_add_field_to_text_present_value_gets_0():
    text = "*   **LAB:** 3.5 Normal\n*   **AGE:** 65\n"
    result = add_field_to_text(text, "LAB", "LAB_IS_MISSING")
    fields = dict(FIELD_RE.findall(result))
    assert fields["LAB_IS_MISSING"] == "0", f"Expected 0, got {fields['LAB_IS_MISSING']}"
    print("PASS test_add_field_to_text_present_value_gets_0")


def test_add_field_to_text_no_data_gets_1():
    text = "*   **LAB:** No data\n"
    result = add_field_to_text(text, "LAB", "LAB_IS_MISSING")
    fields = dict(FIELD_RE.findall(result))
    assert fields["LAB_IS_MISSING"] == "1"
    print("PASS test_add_field_to_text_no_data_gets_1")


def test_end_to_end_adds_missingness_field():
    with tempfile.TemporaryDirectory() as rub_tmp, \
         tempfile.TemporaryDirectory() as rubricified_tmp:
        rub_tmp = Path(rub_tmp)
        rubricified_tmp = Path(rubricified_tmp)
        task = "test_task"

        records_train = [
            make_record(1, "2020-01-01", True,  {"LAB": "N/A",        "AGE": "65"}),
            make_record(2, "2020-01-02", False, {"LAB": "N/A",        "AGE": "45"}),
            make_record(3, "2020-01-03", True,  {"LAB": "3.5 Normal", "AGE": "70"}),
        ]
        records_test = [
            make_record(4, "2020-02-01", False, {"LAB": "None", "AGE": "55"}),
        ]

        for split, recs in [("train", records_train), ("test", records_test)]:
            d = rubricified_tmp / task
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{split}.json", "w") as f:
                json.dump(recs, f)

        instructions = "*   **LAB:** [lab value]\n*   **AGE:** [age]\n"
        d = rub_tmp / task
        d.mkdir(parents=True, exist_ok=True)
        rubric = {"task": task, "rubric_instructions": instructions,
                   "task_query": "test", "num_examples": 0, "usage": {}}
        with open(d / "rubric.json", "w") as f:
            json.dump(rubric, f)

        import subprocess
        result = subprocess.run(
            [sys.executable,
             str(Path(__file__).resolve().parents[1] / "actions" / "add_missingness_indicator.py"),
             "--task", task,
             "--rubric_dir", str(rub_tmp),
             "--rubricified_dir", str(rubricified_tmp)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

        # Check rubric template contains IS_MISSING
        rubric = json.load(open(rub_tmp / task / "rubric.json"))
        assert "IS_MISSING" in rubric["rubric_instructions"]

        # Check train records
        updated_train = json.load(open(rubricified_tmp / task / "train.json"))
        for r in updated_train:
            fields = dict(FIELD_RE.findall(r["rubricified_text"]))
            assert "LAB_IS_MISSING" in fields, "Missing indicator not added"
            expected = "1" if fields["LAB"] in ("N/A", "None", "No data") else "0"
            assert fields["LAB_IS_MISSING"] == expected, \
                f"Expected {expected}, got {fields['LAB_IS_MISSING']} for LAB={fields['LAB']}"

        # Check test records
        updated_test = json.load(open(rubricified_tmp / task / "test.json"))
        for r in updated_test:
            fields = dict(FIELD_RE.findall(r["rubricified_text"]))
            assert fields.get("LAB_IS_MISSING") == "1"

    print("PASS test_end_to_end_adds_missingness_field")


if __name__ == "__main__":
    test_compute_missing_rates()
    test_find_highest_missing_field()
    test_add_field_to_rubric_inserts_after_source()
    test_add_field_to_text_missing_value_gets_1()
    test_add_field_to_text_none_value_gets_1()
    test_add_field_to_text_present_value_gets_0()
    test_add_field_to_text_no_data_gets_1()
    test_end_to_end_adds_missingness_field()
    print("\nAll tests passed.")
