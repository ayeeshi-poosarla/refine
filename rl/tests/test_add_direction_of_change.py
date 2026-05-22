#!/usr/bin/env python3
"""Tests for add_direction_of_change action."""

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from rl.actions.add_direction_of_change import (
    add_field_to_rubric,
    add_field_to_text,
    compute_directions,
    compute_individual_aurocs,
    parse_fields,
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

def test_compute_directions_single_event():
    records = {"train": [make_record(1, "2020-01-01", True, {"LAB": "100 Normal"})]}
    directions = compute_directions(records, "LAB")
    assert directions[(1, "2020-01-01")] == "N/A"
    print("PASS test_compute_directions_single_event")


def test_compute_directions_increasing():
    records = {"train": [
        make_record(1, "2020-01-01", True,  {"LAB": "100 Normal"}),
        make_record(1, "2020-02-01", False, {"LAB": "150 High"}),
    ]}
    directions = compute_directions(records, "LAB")
    assert directions[(1, "2020-01-01")] == "N/A"
    assert directions[(1, "2020-02-01")] == "Increasing"
    print("PASS test_compute_directions_increasing")


def test_compute_directions_decreasing():
    records = {"train": [
        make_record(1, "2020-01-01", True,  {"LAB": "200 High"}),
        make_record(1, "2020-02-01", False, {"LAB": "80 Low"}),
    ]}
    directions = compute_directions(records, "LAB")
    assert directions[(1, "2020-02-01")] == "Decreasing"
    print("PASS test_compute_directions_decreasing")


def test_compute_directions_stable():
    records = {"train": [
        make_record(1, "2020-01-01", True,  {"LAB": "100 Normal"}),
        make_record(1, "2020-02-01", False, {"LAB": "100 Normal"}),
    ]}
    directions = compute_directions(records, "LAB")
    assert directions[(1, "2020-02-01")] == "Stable"
    print("PASS test_compute_directions_stable")


def test_compute_directions_non_numeric():
    records = {"train": [
        make_record(1, "2020-01-01", True,  {"SEX": "MALE"}),
        make_record(1, "2020-02-01", False, {"SEX": "MALE"}),
    ]}
    directions = compute_directions(records, "SEX")
    assert directions[(1, "2020-02-01")] == "N/A"
    print("PASS test_compute_directions_non_numeric")


def test_add_field_to_rubric_inserts_after_source():
    instructions = (
        "*   **SOURCE_FIELD:** [some instruction]\n"
        "*   **OTHER_FIELD:** [other]\n"
    )
    result = add_field_to_rubric(instructions, "SOURCE_FIELD", "SOURCE_FIELD_DIRECTION")
    lines = result.strip().split("\n")
    source_idx = next(i for i, l in enumerate(lines) if "SOURCE_FIELD:**" in l and "DIRECTION" not in l)
    direction_idx = next(i for i, l in enumerate(lines) if "SOURCE_FIELD_DIRECTION:**" in l)
    assert direction_idx == source_idx + 1, "DIRECTION field not immediately after source"
    assert "OTHER_FIELD" in result
    print("PASS test_add_field_to_rubric_inserts_after_source")


def test_add_field_to_text_inserts_after_source():
    text = (
        "*   **LAB:** 100 Normal\n"
        "*   **SEX:** MALE\n"
    )
    result = add_field_to_text(text, "LAB", "LAB_DIRECTION", "Increasing")
    fields = dict(FIELD_RE.findall(result))
    assert "LAB_DIRECTION" in fields
    assert fields["LAB_DIRECTION"] == "Increasing"
    assert "SEX" in fields
    # Verify ordering: LAB_DIRECTION appears right after LAB
    lines = result.strip().split("\n")
    lab_idx = next(i for i, l in enumerate(lines) if "**LAB:**" in l and "DIRECTION" not in l)
    dir_idx = next(i for i, l in enumerate(lines) if "**LAB_DIRECTION:**" in l)
    assert dir_idx == lab_idx + 1
    print("PASS test_add_field_to_text_inserts_after_source")


def test_compute_individual_aurocs_returns_numeric():
    train_rows = [
        {"label": i % 2, "patient_id": i, "prediction_time": f"2020-01-0{i+1}",
         "fields": {"SCORE": str(i * 10)}}
        for i in range(6)
    ]
    test_rows = [
        {"label": i % 2, "patient_id": i + 10, "prediction_time": f"2020-02-0{i+1}",
         "fields": {"SCORE": str(i * 10)}}
        for i in range(4)
    ]
    aurocs = compute_individual_aurocs(train_rows, test_rows)
    assert "SCORE" in aurocs
    assert 0.0 <= aurocs["SCORE"] <= 1.0
    print("PASS test_compute_individual_aurocs_returns_numeric")


def test_end_to_end_adds_direction_field():
    with tempfile.TemporaryDirectory() as rub_tmp, \
         tempfile.TemporaryDirectory() as rubricified_tmp:
        rub_tmp = Path(rub_tmp)
        rubricified_tmp = Path(rubricified_tmp)
        task = "test_task"

        # Patient 1 has two events (direction computable), patient 2 has one
        records_train = [
            make_record(1, "2020-01-01", True,  {"SCORE": "100 Normal", "SEX": "MALE"}),
            make_record(1, "2020-02-01", False, {"SCORE": "150 High",   "SEX": "MALE"}),
            make_record(2, "2020-01-15", True,  {"SCORE": "80 Low",     "SEX": "FEMALE"}),
        ]
        records_test = [
            make_record(3, "2020-03-01", False, {"SCORE": "120 Normal", "SEX": "MALE"}),
        ]

        for split, recs in [("train", records_train), ("test", records_test)]:
            d = rubricified_tmp / task
            d.mkdir(parents=True, exist_ok=True)
            with open(d / f"{split}.json", "w") as f:
                json.dump(recs, f)

        instructions = "*   **SCORE:** [score]\n*   **SEX:** [sex]\n"
        d = rub_tmp / task
        d.mkdir(parents=True, exist_ok=True)
        rubric = {"task": task, "rubric_instructions": instructions,
                   "task_query": "test", "num_examples": 0, "usage": {}}
        with open(d / "rubric.json", "w") as f:
            json.dump(rubric, f)

        import subprocess, os
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        result = subprocess.run(
            [sys.executable,
             str(Path(__file__).resolve().parents[1] / "actions" / "add_direction_of_change.py"),
             "--task", task,
             "--rubric_dir", str(rub_tmp),
             "--rubricified_dir", str(rubricified_tmp)],
            capture_output=True, text=True, env=env
        )
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

        # Check rubric template
        rubric = json.load(open(rub_tmp / task / "rubric.json"))
        assert "_DIRECTION" in rubric["rubric_instructions"]

        # Check records have direction field
        updated = json.load(open(rubricified_tmp / task / "train.json"))
        for r in updated:
            fields = dict(FIELD_RE.findall(r["rubricified_text"]))
            direction_fields = [k for k in fields if k.endswith("_DIRECTION")]
            assert len(direction_fields) == 1, f"Expected 1 direction field, got {direction_fields}"
            assert fields[direction_fields[0]] in ("Increasing", "Decreasing", "Stable", "N/A")

    print("PASS test_end_to_end_adds_direction_field")


if __name__ == "__main__":
    test_compute_directions_single_event()
    test_compute_directions_increasing()
    test_compute_directions_decreasing()
    test_compute_directions_stable()
    test_compute_directions_non_numeric()
    test_add_field_to_rubric_inserts_after_source()
    test_add_field_to_text_inserts_after_source()
    test_compute_individual_aurocs_returns_numeric()
    test_end_to_end_adds_direction_field()
    print("\nAll tests passed.")
