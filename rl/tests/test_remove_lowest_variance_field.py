#!/usr/bin/env python3
"""Tests for remove_lowest_variance_field action."""

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from rl.actions.remove_lowest_variance_field import (
    compute_unique_counts,
    find_lowest_variance_field,
    remove_field_from_rubric,
    remove_field_from_text,
)

FIELD_RE = re.compile(r'\*\*([A-Z_]+):\*\*\s*(.+)')


def make_record(patient_id, prediction_time, label, fields: dict) -> dict:
    text = "\n".join(f"*   **{k}:** {v}" for k, v in fields.items())
    return {
        "patient_id": patient_id,
        "prediction_time": prediction_time,
        "label": label,
        "rubricified_text": text,
    }


def write_rubricified(tmp: Path, task: str, split: str, records: list):
    d = tmp / task
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{split}.json", "w") as f:
        json.dump(records, f)


def write_rubric(tmp: Path, task: str, instructions: str):
    d = tmp / task
    d.mkdir(parents=True, exist_ok=True)
    rubric = {"task": task, "rubric_instructions": instructions,
               "task_query": "test", "num_examples": 0, "usage": {}}
    with open(d / "rubric.json", "w") as f:
        json.dump(rubric, f)


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_compute_unique_counts_identifies_constant_field():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        records = [
            make_record(1, "2020-01-01", True,  {"AGE": "65", "SEX": "MALE",   "CONSTANT": "None"}),
            make_record(2, "2020-01-02", False, {"AGE": "45", "SEX": "FEMALE", "CONSTANT": "None"}),
            make_record(3, "2020-01-03", True,  {"AGE": "70", "SEX": "MALE",   "CONSTANT": "None"}),
        ]
        write_rubricified(tmp, "test_task", "train", records)
        counts = compute_unique_counts(tmp, "test_task")
        assert counts["CONSTANT"] == 1, f"Expected 1 unique value, got {counts['CONSTANT']}"
        assert counts["AGE"] == 3
        assert counts["SEX"] == 2
    print("PASS test_compute_unique_counts_identifies_constant_field")


def test_find_lowest_variance_field():
    counts = {"AGE": 50, "SEX": 2, "CONSTANT_FIELD": 1, "LAB": 30}
    result = find_lowest_variance_field(counts)
    assert result == "CONSTANT_FIELD", f"Expected CONSTANT_FIELD, got {result}"
    print("PASS test_find_lowest_variance_field")


def test_remove_field_from_rubric():
    instructions = (
        "**SECTION:**\n"
        "*   **AGE:** [Patient age]\n"
        "*   **REMOVE_ME:** [This should be removed]\n"
        "*   **SEX:** [Male/Female]\n"
    )
    result = remove_field_from_rubric(instructions, "REMOVE_ME")
    assert "REMOVE_ME" not in result, "Field still present after removal"
    assert "AGE" in result, "AGE should remain"
    assert "SEX" in result, "SEX should remain"
    print("PASS test_remove_field_from_rubric")


def test_remove_field_from_text():
    text = (
        "*   **AGE:** 65\n"
        "*   **REMOVE_ME:** None\n"
        "*   **SEX:** MALE\n"
    )
    result = remove_field_from_text(text, "REMOVE_ME")
    assert "REMOVE_ME" not in result
    parsed = dict(FIELD_RE.findall(result))
    assert "AGE" in parsed
    assert "SEX" in parsed
    print("PASS test_remove_field_from_text")


def test_end_to_end_removes_field_from_all_records():
    with tempfile.TemporaryDirectory() as rub_tmp, \
         tempfile.TemporaryDirectory() as rubricified_tmp:
        rub_tmp = Path(rub_tmp)
        rubricified_tmp = Path(rubricified_tmp)
        task = "test_task"

        records_train = [
            make_record(i, f"2020-01-0{i+1}", i % 2 == 0,
                        {"AGE": str(30 + i), "CONSTANT": "None", "SEX": "MALE" if i % 2 else "FEMALE"})
            for i in range(5)
        ]
        records_test = [
            make_record(i + 10, f"2020-02-0{i+1}", i % 2 == 0,
                        {"AGE": str(50 + i), "CONSTANT": "None", "SEX": "MALE"})
            for i in range(3)
        ]
        write_rubricified(rubricified_tmp, task, "train", records_train)
        write_rubricified(rubricified_tmp, task, "test", records_test)

        instructions = "*   **AGE:** [age]\n*   **CONSTANT:** [always None]\n*   **SEX:** [sex]\n"
        write_rubric(rub_tmp, task, instructions)

        # Run via subprocess to use the same arg parsing as real usage
        import subprocess, os
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        result = subprocess.run(
            [sys.executable,
             str(Path(__file__).resolve().parents[1] / "actions" / "remove_lowest_variance_field.py"),
             "--task", task,
             "--rubric_dir", str(rub_tmp),
             "--rubricified_dir", str(rubricified_tmp)],
            capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        assert "CONSTANT" in result.stdout

        # Verify rubric template
        rubric = json.load(open(rub_tmp / task / "rubric.json"))
        assert "CONSTANT" not in rubric["rubric_instructions"]
        assert "AGE" in rubric["rubric_instructions"]

        # Verify all records
        for split, records in [("train", records_train), ("test", records_test)]:
            updated = json.load(open(rubricified_tmp / task / f"{split}.json"))
            for r in updated:
                fields = dict(FIELD_RE.findall(r["rubricified_text"]))
                assert "CONSTANT" not in fields, f"CONSTANT still in {split} record"
                assert "AGE" in fields

    print("PASS test_end_to_end_removes_field_from_all_records")


if __name__ == "__main__":
    test_compute_unique_counts_identifies_constant_field()
    test_find_lowest_variance_field()
    test_remove_field_from_rubric()
    test_remove_field_from_text()
    test_end_to_end_removes_field_from_all_records()
    print("\nAll tests passed.")
